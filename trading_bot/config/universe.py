"""The symbol universe.

Why a curated catalogue rather than "every US equity"
-----------------------------------------------------
The streaming API is not a firehose you can point at the whole market. Alpaca's
free (Basic) plan caps a websocket subscription at **30 symbols**, and even on a
paid plan a scanner that subscribes to thousands of tickers spends its CPU on
message parsing rather than analysis. So the universe is *chosen*, not
discovered, and the choice lives in configuration.

Three layers, in increasing order of specificity:

1. **Built-in categories** below — liquid, widely-traded names grouped by what
   they are. Stable enough to be checked into source control.
2. **A JSON file** — :func:`load_catalogue` merges it over the built-ins, so a
   user can add, replace or drop whole categories without editing code.
3. **Explicit includes/excludes** — passed to :func:`resolve_universe` by the
   caller, usually from the command line.

Membership here is a statement about *liquidity and interest*, not about
quality. Nothing in this module decides that a symbol is worth trading; it only
decides which symbols are worth watching. Screening for tradability happens in
:mod:`trading_bot.universe.filters`.

Note on full-market scanning
----------------------------
:mod:`trading_bot.universe.discovery` builds a universe from Alpaca's asset list
instead, for the periodic full-market sweep. That path uses REST polling, where
the 30-symbol websocket cap does not apply. The two coexist on purpose: stream a
focused list continuously, sweep the whole market occasionally.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: Maximum symbols a single websocket subscription may carry on Alpaca's free
#: (Basic) plan. Documented at https://docs.alpaca.markets/docs/about-market-data-api
#: — "WebSocket Subscriptions: 30 symbols". The server enforces this itself; we
#: check first so the failure is a clear message instead of a rejected socket.
FREE_STREAM_SYMBOL_LIMIT = 30

MEGA_CAPS: tuple[str, ...] = (
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "BRK.B",
    "JPM", "LLY", "V", "XOM", "UNH", "MA", "COST", "JNJ", "PG", "HD", "WMT",
)

TECH: tuple[str, ...] = (
    "AAPL", "MSFT", "GOOGL", "META", "ORCL", "CRM", "ADBE", "NOW", "INTU",
    "IBM", "CSCO", "ACN", "PANW", "SNOW", "UBER", "SHOP", "NFLX", "PLTR",
)

SEMICONDUCTORS: tuple[str, ...] = (
    "NVDA", "AMD", "AVGO", "TSM", "INTC", "MU", "QCOM", "TXN", "ADI",
    "LRCX", "AMAT", "KLAC", "ARM", "MRVL", "ON", "SMCI",
)

FINANCIALS: tuple[str, ...] = (
    "JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "BLK",
    "AXP", "SPGI", "COF", "PNC", "USB", "BX",
)

ENERGY: tuple[str, ...] = (
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "VLO", "OXY", "HAL", "DVN", "FANG",
)

INDEX_ETFS: tuple[str, ...] = ("SPY", "QQQ", "IWM", "DIA", "VTI", "MDY", "RSP")

SECTOR_ETFS: tuple[str, ...] = (
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC", "SMH",
)

#: The built-in catalogue. Category names are upper-case by convention so that
#: ``--categories tech,index_etfs`` on the command line resolves the same way.
DEFAULT_CATEGORIES: Mapping[str, tuple[str, ...]] = {
    "MEGA_CAPS": MEGA_CAPS,
    "TECH": TECH,
    "SEMICONDUCTORS": SEMICONDUCTORS,
    "FINANCIALS": FINANCIALS,
    "ENERGY": ENERGY,
    "INDEX_ETFS": INDEX_ETFS,
    "SECTOR_ETFS": SECTOR_ETFS,
}

#: What the streamer watches when nothing is configured: the index ETFs (which
#: give the market's own state) plus the mega caps. 27 symbols — deliberately
#: under the free plan's 30, leaving room for a few user additions.
DEFAULT_STREAM_CATEGORIES: tuple[str, ...] = ("INDEX_ETFS", "MEGA_CAPS")


class UniverseConfigError(ValueError):
    """Raised when a universe cannot be resolved from the given configuration.

    Named for *configuration* rather than the universe itself so it cannot be
    confused with :class:`trading_bot.universe.UniverseError`, which is about
    discovering tradable symbols from the broker's asset list.
    """


def normalize_symbol(symbol: str) -> str:
    """Upper-case and strip a ticker.

    Alpaca is case-sensitive and expects upper case; users are not. Class shares
    keep their dot (``BRK.B``) because that is the form Alpaca's API accepts.
    """
    return str(symbol).strip().upper()


def normalize_symbols(symbols: Iterable[str]) -> tuple[str, ...]:
    """Normalise and de-duplicate, preserving first-seen order.

    Order is preserved rather than sorted so that a user's stated priority
    survives — the first 30 symbols of an over-long list are the ones they
    listed first, not the ones that sort earliest.
    """
    seen: dict[str, None] = {}
    for symbol in symbols:
        cleaned = normalize_symbol(symbol)
        if cleaned:
            seen.setdefault(cleaned, None)
    return tuple(seen)


@dataclass(frozen=True, slots=True)
class UniverseCatalogue:
    """Named groups of symbols, resolvable into a flat watchlist."""

    categories: Mapping[str, tuple[str, ...]]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.categories)

    def category(self, name: str) -> tuple[str, ...]:
        """Symbols in one category.

        Raises :class:`UniverseConfigError` naming the available categories, because a
        typo here otherwise silently produces an empty watchlist.
        """
        key = str(name).strip().upper()
        try:
            return self.categories[key]
        except KeyError:
            available = ", ".join(sorted(self.categories)) or "(none)"
            raise UniverseConfigError(
                f"Unknown symbol category {name!r}. Available: {available}"
            ) from None

    def resolve(
        self,
        categories: Iterable[str] | None = None,
        *,
        include: Iterable[str] = (),
        exclude: Iterable[str] = (),
    ) -> tuple[str, ...]:
        """Flatten ``categories`` plus ``include``, minus ``exclude``.

        Categories are expanded in the order given, so the resulting watchlist
        reads in a predictable order rather than an arbitrary set order.
        """
        collected: list[str] = []
        for name in categories or ():
            collected.extend(self.category(name))
        collected.extend(include)
        removed = set(normalize_symbols(exclude))
        return tuple(s for s in normalize_symbols(collected) if s not in removed)

    def describe(self) -> str:
        rows = [
            f"{name:<16} {len(symbols):>3} symbols"
            for name, symbols in sorted(self.categories.items())
        ]
        return "\n".join(rows)


def default_catalogue() -> UniverseCatalogue:
    """The built-in catalogue, with symbols normalised."""
    return UniverseCatalogue(
        {name: normalize_symbols(symbols) for name, symbols in DEFAULT_CATEGORIES.items()}
    )


def load_catalogue(path: Path | str | None = None) -> UniverseCatalogue:
    """Built-in categories, overridden by a JSON file when one is given.

    The file is a flat mapping of category name to a list of symbols::

        {
          "MY_WATCHLIST": ["AAPL", "NVDA"],
          "ENERGY": ["XOM", "CVX"]
        }

    A category present in the file **replaces** the built-in of the same name
    rather than extending it — that is what makes it possible to shrink a
    category, which merging could never do. An empty list removes the category.

    A missing file is not an error: the built-ins are the answer. A malformed
    file *is* an error, because silently falling back would give the user a
    universe they did not ask for.
    """
    catalogue = dict(default_catalogue().categories)
    if path is None:
        return UniverseCatalogue(catalogue)

    file_path = Path(path)
    if not file_path.exists():
        logger.debug("No universe file at %s; using built-in categories", file_path)
        return UniverseCatalogue(catalogue)

    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UniverseConfigError(f"Could not read universe file {file_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise UniverseConfigError(
            f"Universe file {file_path} must contain a JSON object mapping "
            "category names to symbol lists"
        )

    for name, symbols in payload.items():
        key = str(name).strip().upper()
        if not key:
            continue
        if not isinstance(symbols, (list, tuple)):
            raise UniverseConfigError(
                f"Category {name!r} in {file_path} must be a list of symbols, "
                f"got {type(symbols).__name__}"
            )
        cleaned = normalize_symbols(symbols)
        if cleaned:
            catalogue[key] = cleaned
        else:
            catalogue.pop(key, None)

    logger.info("Loaded universe overrides from %s", file_path)
    return UniverseCatalogue(catalogue)


def stream_capacity_error(
    symbols: Iterable[str], limit: int = FREE_STREAM_SYMBOL_LIMIT
) -> str | None:
    """Explain why ``symbols`` will not fit one websocket subscription, or None.

    Returned rather than raised so the caller decides whether an over-long list
    is fatal (streaming) or merely worth mentioning (a REST scan, where the cap
    does not apply).
    """
    watched = normalize_symbols(symbols)
    if limit <= 0 or len(watched) <= limit:
        return None
    return (
        f"{len(watched)} symbols requested but the data plan allows {limit} on a "
        f"single websocket subscription. Drop {len(watched) - limit} symbols, or "
        "scan the rest over REST with `python main.py scan`."
    )
