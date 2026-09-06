"""Building the scannable universe from the exchange's own asset list.

Alpaca publishes every US equity it knows about — roughly eleven thousand
symbols — in a single request. That list is the honest starting point for
"scan the whole market": it is what is actually listed, rather than an index
membership or a hand-maintained watchlist that quietly goes stale.

The list changes slowly (listings, delistings, ticker changes), so it is cached
on disk for a day. That matters more than it sounds: without it, every scan
starts by re-downloading eleven thousand records that changed by maybe four.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from trading_bot.config.settings import AlpacaSettings
from trading_bot.data.market_data import MarketDataProvider
from trading_bot.universe.filters import (
    FilterReport,
    LiquidityProfile,
    UniverseFilter,
    passes_liquidity_filters,
    passes_static_filters,
    profile_liquidity,
)

logger = logging.getLogger(__name__)

#: How long a cached asset list stays fresh. Listings change daily at most, and
#: a stale entry costs one failed fetch rather than a wrong answer.
ASSET_CACHE_TTL = timedelta(hours=20)


#: Alpaca issues paper keys with a ``PK`` prefix and live keys with ``AK``.
#: The prefix is the one thing about a rejected credential that can be checked
#: without a working connection, so it is the only cause below stated as fact
#: rather than as a possibility.
PAPER_KEY_PREFIX = "PK"


def explain_auth_failure(api_key: str, error: object) -> str:
    """Turn Alpaca's 401 into something a person can act on.

    Alpaca answers every rejected credential with the same body — literally
    ``{"message": "unauthorized."}`` — for at least four different causes. A
    newly created account whose compliance review has not cleared yet reads
    exactly like a mistyped secret, which sends people to check the wrong
    thing. This names the causes in the order they actually occur.
    """
    key = (api_key or "").strip()
    lines = [
        "Alpaca rejected these credentials. It answers every rejected "
        "credential the same way, so this cannot tell the causes apart — "
        "they are listed in the order they usually occur:",
        "",
        "  1. The account is still under review. A new Alpaca account cannot "
        "use its keys until the compliance check clears, usually a few hours "
        "to a couple of days. The dashboard shows a banner while it is "
        "pending, and key regeneration fails too.",
        "  2. The key and secret are from different pairs. Regenerating "
        "issues a new pair and invalidates the old one at once, so a new key "
        "kept with an old secret fails exactly like a wrong key. Replace both "
        "together, from the same screen.",
        "  3. A copy-paste error — a trailing space, or a secret truncated "
        "at the end. One missing character fails identically to a wrong key.",
    ]
    if key and not key.upper().startswith(PAPER_KEY_PREFIX):
        lines.append(
            f"  4. **These look like live-account keys** — this key starts "
            f"{key[:2]!r}, and paper keys start 'PK'. The asset list is read "
            "from the paper endpoint, and Alpaca keeps separate pairs for "
            "paper and live. Generate keys from the Paper dashboard."
        )
    else:
        lines.append(
            "  4. Live-account keys used against the paper endpoint. Ruled "
            "out here: this key has the 'PK' prefix of a paper key."
        )
    lines += ["", f"Alpaca said: {error}"]
    return "\n".join(lines)


class UniverseError(RuntimeError):
    """The universe could not be built."""


@dataclass(slots=True)
class Universe:
    """The symbols a scan will consider, and how they were chosen."""

    symbols: tuple[str, ...]
    profiles: dict[str, LiquidityProfile] = field(default_factory=dict)
    #: Daily bars fetched for the liquidity screen, kept for whatever runs next.
    #: The screen already downloads a year of history per symbol; discarding it
    #: would make a market-wide scan fetch the entire market twice.
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    static_report: FilterReport = field(default_factory=FilterReport)
    liquidity_report: FilterReport = field(default_factory=FilterReport)
    built_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __len__(self) -> int:
        return len(self.symbols)

    @property
    def total_dollar_volume(self) -> float:
        return sum(profile.median_dollar_volume for profile in self.profiles.values())

    def summary_lines(self) -> list[str]:
        lines = ["Static filters (no price data needed):"]
        lines += [f"  {line}" for line in self.static_report.summary_lines()]
        if self.liquidity_report.considered:
            lines.append("Liquidity filters (needs daily bars):")
            lines += [f"  {line}" for line in self.liquidity_report.summary_lines()]
        lines.append(f"Universe: {len(self.symbols):,} symbols")
        return lines

    def as_frame(self) -> pd.DataFrame:
        """The universe as a table, sorted by turnover."""
        if not self.profiles:
            return pd.DataFrame({"symbol": list(self.symbols)})
        rows = [
            {
                "symbol": profile.symbol,
                "last_close": profile.last_close,
                "median_dollar_volume": profile.median_dollar_volume,
                "median_volume": profile.median_volume,
                "atr_pct": profile.atr_pct,
                "bars": profile.bars,
            }
            for profile in self.profiles.values()
        ]
        frame = pd.DataFrame(rows)
        return frame.sort_values("median_dollar_volume", ascending=False, ignore_index=True)


class AssetCatalogue:
    """The exchange's asset list, cached on disk.

    Parameters
    ----------
    settings:
        Supplies credentials.
    cache_dir:
        Where the cached list is written. ``None`` disables caching.
    client:
        Pre-built Alpaca ``TradingClient``, primarily for tests.
    """

    def __init__(
        self,
        settings: AlpacaSettings,
        *,
        cache_dir: Path | None = None,
        client: Any | None = None,
    ) -> None:
        self._settings = settings
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._client = client

    @property
    def _cache_path(self) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / "assets.json"

    def _build_client(self):
        if self._client is not None:
            return self._client
        if not self._settings.has_credentials:
            raise UniverseError(
                "Alpaca credentials are missing. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY in your .env file. A free data-only account "
                "is enough — no funding, and no orders are ever placed."
            )
        from alpaca.trading.client import TradingClient

        # Always the paper endpoint: this reads the asset list and nothing else,
        # and the asset list is identical on both. Reaching for the live
        # endpoint here would put a credential with trading rights on a code
        # path that has no reason to hold one.
        self._client = TradingClient(
            api_key=self._settings.api_key,
            secret_key=self._settings.secret_key,
            paper=True,
        )
        return self._client

    def fetch(self, *, use_cache: bool = True) -> list[dict[str, Any]]:
        """Every active US equity Alpaca lists, as plain dicts."""
        cached = self._read_cache() if use_cache else None
        if cached is not None:
            logger.info("Using cached asset list (%d symbols)", len(cached))
            return cached

        from alpaca.trading.enums import AssetClass, AssetStatus
        from alpaca.trading.requests import GetAssetsRequest

        client = self._build_client()
        started = time.perf_counter()
        request = GetAssetsRequest(
            status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY
        )
        try:
            assets = client.get_all_assets(request)
        except Exception as error:  # noqa: BLE001 - surfaced with context
            if _is_auth_failure(error):
                raise UniverseError(
                    explain_auth_failure(self._settings.api_key, error)
                ) from error
            raise UniverseError(f"Could not fetch the asset list: {error}") from error

        records = [_asset_to_dict(asset) for asset in assets]
        logger.info(
            "Fetched %d assets in %.1fs", len(records), time.perf_counter() - started
        )
        self._write_cache(records)
        return records

    def _read_cache(self) -> list[dict[str, Any]] | None:
        path = self._cache_path
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
            stamp = datetime.fromisoformat(payload["fetched_at"])
            if datetime.now(timezone.utc) - stamp > ASSET_CACHE_TTL:
                return None
            return payload["assets"]
        except Exception as error:  # noqa: BLE001 - a bad cache is not fatal
            logger.warning("Ignoring unreadable asset cache: %s", error)
            return None

    def _write_cache(self, records: list[dict[str, Any]]) -> None:
        path = self._cache_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {"fetched_at": datetime.now(timezone.utc).isoformat(), "assets": records}
                )
            )
        except OSError as error:
            logger.warning("Could not write the asset cache: %s", error)


def _is_auth_failure(error: object) -> bool:
    """Whether an exception is Alpaca refusing the credentials.

    Matched on the message rather than a status attribute: the SDK raises
    several exception types here, and reading ``.code`` on one of them
    re-parses the response body and can raise again on a non-JSON error page.
    """
    text = str(error).lower()
    return "unauthorized" in text or "401" in text or "forbidden" in text


def _asset_to_dict(asset: Any) -> dict[str, Any]:
    """Flatten an Alpaca asset to the fields the filters read."""
    return {
        "symbol": str(getattr(asset, "symbol", "")),
        "name": str(getattr(asset, "name", "") or ""),
        "exchange": str(getattr(getattr(asset, "exchange", ""), "value", "")
                        or getattr(asset, "exchange", "")),
        "asset_class": str(getattr(getattr(asset, "asset_class", ""), "value", "")
                           or getattr(asset, "asset_class", "")),
        "status": str(getattr(getattr(asset, "status", ""), "value", "")
                      or getattr(asset, "status", "")),
        "tradable": bool(getattr(asset, "tradable", False)),
        "shortable": bool(getattr(asset, "shortable", False)),
        "fractionable": bool(getattr(asset, "fractionable", False)),
        "marginable": bool(getattr(asset, "marginable", False)),
    }


class _Record:
    """Attribute access over an asset dict, for the filter functions."""

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        return self._data.get(name)


def screen_static(
    records: list[dict[str, Any]], filters: UniverseFilter
) -> tuple[list[str], FilterReport]:
    """Apply the metadata-only filters. Cheap, and run first."""
    report = FilterReport(considered=len(records))
    kept = [
        str(record["symbol"]).upper()
        for record in records
        if passes_static_filters(_Record(record), filters, report)
    ]
    report.kept = len(kept)
    logger.info("Static filters kept %d of %d symbols", len(kept), len(records))
    return kept, report


def screen_liquidity(
    frames: dict[str, pd.DataFrame], filters: UniverseFilter
) -> tuple[list[str], dict[str, LiquidityProfile], FilterReport]:
    """Apply the filters that need daily bars."""
    report = FilterReport(considered=len(frames))
    profiles: dict[str, LiquidityProfile] = {}

    for symbol, frame in frames.items():
        profile = profile_liquidity(symbol, frame)
        if profile is None:
            report.drop("no usable bars")
            continue
        if not passes_liquidity_filters(profile, filters, report):
            continue
        profiles[symbol] = profile

    ranked = sorted(
        profiles.values(), key=lambda item: -item.median_dollar_volume
    )
    if filters.max_symbols is not None and len(ranked) > filters.max_symbols:
        report.drop(
            f"beyond the {filters.max_symbols:,}-symbol cap",
            len(ranked) - filters.max_symbols,
        )
        ranked = ranked[: filters.max_symbols]

    symbols = [profile.symbol for profile in ranked]
    report.kept = len(symbols)
    return symbols, {symbol: profiles[symbol] for symbol in symbols}, report


def build_universe(
    catalogue: AssetCatalogue,
    provider: MarketDataProvider,
    filters: UniverseFilter | None = None,
    *,
    screen_bars: int = 250,
    use_cache: bool = True,
    progress: Any | None = None,
) -> Universe:
    """Discover, filter and rank the symbols worth scanning.

    Runs the two stages in order: metadata filters over the full listing, then
    liquidity filters over daily bars for whatever survived. The second stage is
    where nearly all the time goes, which is exactly why the first one exists.

    Parameters
    ----------
    catalogue:
        Source of the asset list.
    provider:
        Where daily bars come from.
    filters:
        Defaults to :class:`UniverseFilter`.
    screen_bars:
        Daily bars fetched per symbol for the liquidity screen. 250 is about a
        trading year — enough for a 200-day average to exist.
    progress:
        Optional callable taking ``(done, total)``, for a progress bar.

    Returns
    -------
    Universe
    """
    settings = filters or UniverseFilter()
    records = catalogue.fetch(use_cache=use_cache)
    candidates, static_report = screen_static(records, settings)
    if not candidates:
        raise UniverseError(
            "No symbols survived the metadata filters. Check the `exchanges` "
            "setting — an empty or misspelled exchange list rejects everything."
        )

    logger.info("Fetching daily bars for %d candidates", len(candidates))
    frames, fetch_report = provider.fetch_watchlist(
        candidates, "1Day", lookback_bars=screen_bars, progress=progress
    )
    if fetch_report.failed:
        static_report.drop("no data returned", len(fetch_report.failed))

    symbols, profiles, liquidity_report = screen_liquidity(frames, settings)
    if not symbols:
        raise UniverseError(
            "No symbols passed the liquidity filters. The most likely cause is "
            f"min_dollar_volume (${settings.min_dollar_volume:,.0f}/day) being "
            "set higher than anything in the candidate list."
        )

    return Universe(
        symbols=tuple(symbols),
        profiles=profiles,
        frames={symbol: frames[symbol] for symbol in symbols if symbol in frames},
        static_report=static_report,
        liquidity_report=liquidity_report,
    )
