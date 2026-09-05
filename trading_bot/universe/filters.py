"""Deciding which symbols are worth scanning at all.

A US equity scan starts from roughly eleven thousand listed symbols, and most of
them are not things a retail swing trader can or should trade. Filtering is not
an optimisation here — it is the difference between a ranked list of real
opportunities and a list topped by an illiquid shell company whose indicators
happen to look tidy.

Two kinds of filter, applied in two stages because they cost different amounts:

**Static filters** read the exchange's own metadata — asset class, exchange,
tradability. They need no price data, so they run first and cheaply, cutting the
universe by roughly half before a single bar is fetched.

**Liquidity filters** need daily bars. They are what actually removes the
untradable: a symbol whose average turnover is $80,000 a day cannot absorb a
retail position without moving, and its spread will eat the edge the scan found.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

#: Exchanges a Robinhood account can actually trade. Robinhood lists NYSE,
#: NASDAQ, AMEX/NYSE American, ARCA and BATS/Cboe issues. It does **not**
#: support OTC or pink-sheet securities, so scanning them would produce
#: opportunities that cannot be acted on.
ROBINHOOD_EXCHANGES: frozenset[str] = frozenset(
    {"NYSE", "NASDAQ", "AMEX", "ARCA", "NYSEARCA", "BATS"}
)

#: Substrings that mark a leveraged or inverse product. These decay against the
#: holder over multi-day holds — the daily-reset maths means a 3x fund does not
#: return 3x over a week — so they are excluded from a swing scan by default.
#: Matched against the security's *name*, not its symbol, because ticker letters
#: are not a reliable signal.
LEVERAGED_MARKERS: tuple[str, ...] = (
    "2X", "3X", "-1X", "1.5X", "ULTRA", "ULTRASHORT", "LEVERAGED",
    "INVERSE", "BEAR ", "BULL ", "DAILY SHORT", "DOUBLE", "TRIPLE",
)

#: Names that mark a structure rather than a company: closed-end funds, trusts
#: and blank-cheque vehicles whose price action reflects flows and deal news
#: rather than the technical behaviour these strategies assume.
STRUCTURE_MARKERS: tuple[str, ...] = (
    "ACQUISITION CORP", "ACQUISITION CO", "BLANK CHECK", "SPAC",
    "CLOSED END", "CLOSED-END", "ROYALTY TRUST", "LIQUIDATING TRUST",
)

#: Suffixes on a symbol that indicate a warrant, right, unit or preferred
#: share. These are thin, structurally different instruments, and several are
#: not available on Robinhood at all.
DERIVATIVE_SUFFIX = re.compile(r"[.\-+](W|WS|R|RT|U|UN|P[A-Z]?)$", re.IGNORECASE)


#: Roughly what share of a US stock's consolidated volume prints on IEX. IEX is
#: one venue among many, and the free Alpaca feed reports only what crossed it —
#: so turnover computed from those bars is a small fraction of the real figure.
#: Deliberately a range, not a precise number: it varies by symbol and by day,
#: and pretending otherwise would invite a "correction factor" that is really a
#: guess.
IEX_VOLUME_SHARE = (0.015, 0.04)


def feed_liquidity_warning(feed: str, min_dollar_volume: float) -> str | None:
    """Warn when the turnover floor is measured against a single-venue feed.

    The ``iex`` feed reports only trades that crossed IEX, so a symbol turning
    over $20M a day across all venues may show well under $1M here. A filter
    written as "$10M a day" then behaves like a far stricter one, and the
    result is a scan that returns almost nothing and looks broken rather than
    strict.

    Returns ``None`` when there is nothing to warn about.
    """
    if feed.strip().lower() != "iex" or min_dollar_volume <= 0:
        return None
    low, high = IEX_VOLUME_SHARE
    return (
        f"Turnover is measured from the 'iex' feed, which sees only trades that "
        f"crossed IEX — very roughly {low:.1%}-{high:.1%} of a symbol's real "
        f"volume. Your ${min_dollar_volume:,.0f}/day floor therefore behaves "
        f"more like ${min_dollar_volume / high:,.0f}-${min_dollar_volume / low:,.0f} "
        f"of consolidated turnover. If the scan returns almost nothing, lower "
        f"--min-dollar-volume before concluding the market is quiet; a paid "
        f"'sip' feed reports the consolidated tape and needs no adjustment."
    )


@dataclass(frozen=True, slots=True)
class UniverseFilter:
    """What counts as a scannable symbol.

    Every default here is a judgement about what a retail swing trader can
    realistically hold, and every one is meant to be overridden. The thresholds
    matter more than they look: raising ``min_dollar_volume`` is the single most
    effective way to make scan results actionable, and lowering it is the
    fastest way to fill the rankings with names you cannot get filled in.

    Attributes
    ----------
    exchanges:
        Venues to include. Defaults to those a Robinhood account can trade.
    min_price:
        Floor on the last close. Below about $5 spreads widen sharply as a
        share of price, and many institutions cannot hold the name at all,
        which thins the book further.
    max_price:
        Ceiling on the last close. With a small account a $2,000 share price
        makes position sizing impossible — one share may already exceed the
        risk budget. ``None`` disables the check.
    min_dollar_volume:
        Average daily turnover — price times volume, not volume alone. A
        million shares of a $0.40 stock is $400,000 of liquidity, not a
        million dollars of it, and volume alone hides that.
    min_history_bars:
        Bars required before a symbol can be analysed. A recent IPO has no
        200-day average and no established structure; indicators computed on
        it are arithmetic without meaning.
    exclude_leveraged:
        Drop leveraged and inverse funds — see :data:`LEVERAGED_MARKERS`.
    exclude_structures:
        Drop SPACs, closed-end funds and similar — see :data:`STRUCTURE_MARKERS`.
    exclude_derivatives:
        Drop warrants, rights, units and preferred shares by symbol suffix.
    require_fractionable:
        Only keep symbols the broker will sell in fractions. Off by default:
        it is an Alpaca property and does not describe Robinhood.
    max_symbols:
        Hard cap on the universe, applied after ranking by turnover. Guards
        against a filter mistake turning into an enormous scan.
    """

    exchanges: frozenset[str] = ROBINHOOD_EXCHANGES
    min_price: float = 5.0
    max_price: float | None = 1_000.0
    min_dollar_volume: float = 10_000_000.0
    min_history_bars: int = 200
    exclude_leveraged: bool = True
    exclude_structures: bool = True
    exclude_derivatives: bool = True
    require_fractionable: bool = False
    max_symbols: int | None = 4_000
    #: Symbols always kept regardless of the filters above.
    always_include: frozenset[str] = field(default_factory=frozenset)
    #: Symbols always dropped, whatever else says.
    never_include: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.min_price < 0:
            raise ValueError(f"min_price must be >= 0, got {self.min_price}")
        if self.max_price is not None and self.max_price <= self.min_price:
            raise ValueError(
                f"max_price ({self.max_price}) must exceed min_price ({self.min_price})"
            )
        if self.min_dollar_volume < 0:
            raise ValueError(
                f"min_dollar_volume must be >= 0, got {self.min_dollar_volume}"
            )
        if self.max_symbols is not None and self.max_symbols < 1:
            raise ValueError(f"max_symbols must be >= 1, got {self.max_symbols}")


@dataclass(slots=True)
class FilterReport:
    """Why the universe ended up the size it did.

    A scan that silently returns forty symbols out of eleven thousand is
    indistinguishable from a broken one. This records each stage's toll so the
    shape of the funnel is visible.
    """

    considered: int = 0
    kept: int = 0
    dropped: dict[str, int] = field(default_factory=dict)

    def drop(self, reason: str, count: int = 1) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + count

    def summary_lines(self) -> list[str]:
        lines = [f"{self.considered:,} symbols considered, {self.kept:,} kept"]
        for reason, count in sorted(self.dropped.items(), key=lambda item: -item[1]):
            lines.append(f"  {count:>6,}  dropped: {reason}")
        return lines


def passes_static_filters(
    asset, filters: UniverseFilter, report: FilterReport | None = None
) -> bool:
    """Whether an asset survives the checks that need no price data.

    Parameters
    ----------
    asset:
        Anything exposing ``symbol``, ``name``, ``exchange``, ``tradable``,
        ``status`` and ``fractionable`` — the Alpaca ``Asset`` model, this
        project's :class:`AssetInfo`, or a stub in a test.
    """
    def note(reason: str) -> bool:
        if report is not None:
            report.drop(reason)
        return False

    symbol = str(getattr(asset, "symbol", "")).upper()
    if not symbol:
        return note("no symbol")
    if symbol in filters.never_include:
        return note("explicitly excluded")
    if symbol in filters.always_include:
        return True

    if not bool(getattr(asset, "tradable", False)):
        return note("not tradable")
    status = str(getattr(asset, "status", "active"))
    if status.lower().removeprefix("assetstatus.") != "active":
        return note("not active")

    exchange = _enum_value(getattr(asset, "exchange", "")).upper()
    if exchange not in filters.exchanges:
        return note(f"exchange {exchange or 'unknown'}")

    asset_class = _enum_value(getattr(asset, "asset_class", "us_equity")).lower()
    if asset_class and asset_class != "us_equity":
        return note(f"asset class {asset_class}")

    if filters.require_fractionable and not bool(getattr(asset, "fractionable", False)):
        return note("not fractionable")

    if filters.exclude_derivatives and DERIVATIVE_SUFFIX.search(symbol):
        return note("warrant/right/unit/preferred")

    name = str(getattr(asset, "name", "") or "").upper()
    if filters.exclude_leveraged and any(mark in name for mark in LEVERAGED_MARKERS):
        return note("leveraged or inverse")
    if filters.exclude_structures and any(mark in name for mark in STRUCTURE_MARKERS):
        return note("SPAC/closed-end/trust")

    return True


def _enum_value(value) -> str:
    """Enum members stringify as ``AssetExchange.NASDAQ``; take the value."""
    return str(getattr(value, "value", value) or "")


@dataclass(frozen=True, slots=True)
class LiquidityProfile:
    """What a symbol's recent daily bars say about tradability."""

    symbol: str
    last_close: float
    avg_dollar_volume: float
    avg_volume: float
    bars: int
    #: Average true range as a percentage of price — how much room a swing
    #: trade has to work with. A stock that moves 0.3% a day cannot pay for a
    #: stop plus costs, whatever its chart looks like.
    atr_pct: float | None = None

    @property
    def is_liquid_enough(self) -> bool:
        return self.avg_dollar_volume > 0


def profile_liquidity(
    symbol: str, bars: pd.DataFrame, *, window: int = 20
) -> LiquidityProfile | None:
    """Summarise a symbol's tradability from its daily bars.

    Returns ``None`` when the frame is unusable, rather than fabricating a
    profile from nothing.
    """
    if bars is None or bars.empty:
        return None
    required = {"close", "volume", "high", "low"}
    if not required.issubset(bars.columns):
        return None

    recent = bars.tail(window)
    close = recent["close"].astype("float64")
    volume = recent["volume"].astype("float64")
    if close.empty or not float(close.iloc[-1]) > 0:
        return None

    turnover = (close * volume).mean()
    ranges = (recent["high"].astype("float64") - recent["low"].astype("float64")).mean()
    last = float(close.iloc[-1])

    return LiquidityProfile(
        symbol=symbol,
        last_close=last,
        avg_dollar_volume=float(turnover) if pd.notna(turnover) else 0.0,
        avg_volume=float(volume.mean()) if pd.notna(volume.mean()) else 0.0,
        bars=len(bars),
        atr_pct=float(ranges / last * 100) if pd.notna(ranges) and last > 0 else None,
    )


def passes_liquidity_filters(
    profile: LiquidityProfile, filters: UniverseFilter, report: FilterReport | None = None
) -> bool:
    """Whether a symbol's price and turnover make it worth analysing."""
    def note(reason: str) -> bool:
        if report is not None:
            report.drop(reason)
        return False

    if profile.symbol in filters.always_include:
        return True
    if profile.bars < filters.min_history_bars:
        return note(f"under {filters.min_history_bars} bars of history")
    if profile.last_close < filters.min_price:
        return note(f"price under ${filters.min_price:,.2f}")
    if filters.max_price is not None and profile.last_close > filters.max_price:
        return note(f"price over ${filters.max_price:,.2f}")
    if profile.avg_dollar_volume < filters.min_dollar_volume:
        return note(f"turnover under ${filters.min_dollar_volume:,.0f}/day")
    return True
