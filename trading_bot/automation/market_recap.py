"""What the market did today, in the terms a swing trader acts on.

The end-of-day report already covered what *the bot* did — symbols scanned,
setups found, orders placed. That answers "is it working", not "what happened
out there", and the second question is the one that decides whether tomorrow's
setups are worth taking at all. A breakout list means something different on a
day the whole tape rallied than on a day only your three names did.

Everything here is computed from **daily bars the scanner already fetches** —
index and sector ETFs, priced by the same provider as everything else. No extra
subscription, no second vendor, and the recap works on a data-only key.

What it deliberately does not do
--------------------------------
It does not comment on the economy. CPI prints, Fed decisions and payroll
numbers are not in a price feed, and inferring them from index moves would be
narration dressed as analysis — "stocks fell on rate fears" written by something
that cannot see rates. Where macro context genuinely matters,
:func:`describe_regime` says what the *price behaviour* implies and stops there.
A news or macro provider would slot in behind :class:`MarketRecap.headlines`;
until one exists, the field stays empty rather than invented.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from trading_bot.data.market_data import MarketDataProvider
from trading_bot.utils.timeframes import Timeframe

logger = logging.getLogger(__name__)

#: The market's own state. SPY is the tape; the others say who is carrying it.
INDEX_ETFS: tuple[tuple[str, str], ...] = (
    ("SPY", "S&P 500"),
    ("QQQ", "Nasdaq 100"),
    ("IWM", "Russell 2000"),
    ("DIA", "Dow 30"),
)

#: Sector ETFs, named so the recap reads like English rather than tickers.
SECTOR_ETFS: tuple[tuple[str, str], ...] = (
    ("XLK", "Technology"),
    ("XLF", "Financials"),
    ("XLE", "Energy"),
    ("XLV", "Health Care"),
    ("XLI", "Industrials"),
    ("XLY", "Consumer Disc."),
    ("XLP", "Consumer Staples"),
    ("XLU", "Utilities"),
    ("XLB", "Materials"),
    ("XLRE", "Real Estate"),
    ("XLC", "Communications"),
)

#: Bars pulled per symbol. Enough for a 50-day trend read and an ATR baseline.
LOOKBACK_BARS = 120

#: SPY's 14-day ATR as a percentage of price, in ordinary conditions. Measured
#: rather than assumed: the long-run average sits near 1%, and the bands below
#: are where the character of the tape visibly changes.
CALM_ATR_PCT = 0.8
ELEVATED_ATR_PCT = 1.5
STRESSED_ATR_PCT = 2.5


@dataclass(frozen=True, slots=True)
class InstrumentMove:
    """One symbol's day, with just enough context to interpret it."""

    symbol: str
    name: str
    close: float
    change_pct: float
    change_5d_pct: float
    above_50d: bool
    atr_pct: float | None = None

    @property
    def direction(self) -> str:
        return "up" if self.change_pct > 0 else "down" if self.change_pct < 0 else "flat"


@dataclass(frozen=True, slots=True)
class MarketRecap:
    """The close, summarised."""

    as_of: datetime
    indices: tuple[InstrumentMove, ...] = ()
    sectors: tuple[InstrumentMove, ...] = ()
    #: Share of sectors that closed higher, 0-1. The cheapest honest breadth
    #: measure available without a full market scan.
    breadth: float | None = None
    regime: str = "unknown"
    benchmark_atr_pct: float | None = None
    #: Reserved for a news or macro provider. Empty until one exists; the recap
    #: says nothing about the economy rather than guessing at it.
    headlines: tuple[str, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def benchmark(self) -> InstrumentMove | None:
        return self.indices[0] if self.indices else None

    @property
    def leaders(self) -> tuple[InstrumentMove, ...]:
        return tuple(sorted(self.sectors, key=lambda s: s.change_pct, reverse=True)[:3])

    @property
    def laggards(self) -> tuple[InstrumentMove, ...]:
        return tuple(sorted(self.sectors, key=lambda s: s.change_pct)[:3])

    @property
    def usable(self) -> bool:
        return bool(self.indices or self.sectors)


def _move(symbol: str, name: str, frame: pd.DataFrame) -> InstrumentMove | None:
    """Summarise one symbol's daily frame, or None when it is too short."""
    if frame is None or len(frame) < 2:
        return None
    closes = frame["close"].astype("float64")
    close = float(closes.iloc[-1])
    previous = float(closes.iloc[-2])
    if not np.isfinite(close) or previous <= 0:
        return None

    change = (close - previous) / previous * 100.0
    week_ago = float(closes.iloc[-6]) if len(closes) >= 6 else previous
    change_5d = (close - week_ago) / week_ago * 100.0 if week_ago > 0 else 0.0

    above_50d = False
    if len(closes) >= 50:
        above_50d = close > float(closes.tail(50).mean())

    atr_pct = None
    if len(frame) >= 15:
        highs = frame["high"].astype("float64")
        lows = frame["low"].astype("float64")
        previous_close = closes.shift(1)
        true_range = pd.concat(
            [highs - lows, (highs - previous_close).abs(), (lows - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        average = float(true_range.tail(14).mean())
        if np.isfinite(average) and close > 0:
            atr_pct = average / close * 100.0

    return InstrumentMove(
        symbol=symbol, name=name, close=close, change_pct=change,
        change_5d_pct=change_5d, above_50d=above_50d, atr_pct=atr_pct,
    )


def describe_regime(atr_pct: float | None) -> str:
    """Name the volatility regime from the benchmark's own range.

    A label, not a forecast. It exists because the same setup deserves a
    different position size in a 0.7% tape and a 2.5% one, and a reader glancing
    at a report should not have to work that out from a decimal.
    """
    if atr_pct is None:
        return "unknown"
    if atr_pct < CALM_ATR_PCT:
        return "calm"
    if atr_pct < ELEVATED_ATR_PCT:
        return "normal"
    if atr_pct < STRESSED_ATR_PCT:
        return "elevated"
    return "stressed"


def build_recap(
    provider: MarketDataProvider,
    *,
    now: datetime | None = None,
    timeframe: str | Timeframe = "1Day",
) -> MarketRecap:
    """Fetch the index and sector tape and summarise it.

    One batched request for eighteen liquid ETFs. A symbol that fails to return
    is dropped and noted rather than raising: a recap missing Real Estate is
    still worth reading, and an end-of-day report that crashes tells you less
    than one with a gap in it.
    """
    moment = now or datetime.now(timezone.utc)
    wanted = [*INDEX_ETFS, *SECTOR_ETFS]
    symbols = [symbol for symbol, _ in wanted]

    errors: list[str] = []
    frames: dict[str, pd.DataFrame] = {}
    try:
        frames, report = provider.fetch_watchlist(
            symbols, timeframe, lookback_bars=LOOKBACK_BARS, end=moment
        )
        if report.failed:
            errors.append(f"{len(report.failed)} symbol(s) returned no data")
    except Exception as error:  # noqa: BLE001 - a recap is never worth a crash
        logger.exception("Market recap could not fetch bars")
        return MarketRecap(as_of=moment, errors=(f"market data unavailable: {error}",))

    indices: list[InstrumentMove] = []
    sectors: list[InstrumentMove] = []
    for symbol, name in wanted:
        move = _move(symbol, name, frames.get(symbol))
        if move is None:
            continue
        (indices if (symbol, name) in INDEX_ETFS else sectors).append(move)

    breadth = None
    if sectors:
        breadth = sum(1 for s in sectors if s.change_pct > 0) / len(sectors)

    benchmark_atr = indices[0].atr_pct if indices else None
    return MarketRecap(
        as_of=moment,
        indices=tuple(indices),
        sectors=tuple(sectors),
        breadth=breadth,
        regime=describe_regime(benchmark_atr),
        benchmark_atr_pct=benchmark_atr,
        errors=tuple(errors),
    )


def _breadth_sentence(recap: MarketRecap) -> str:
    """Say what the spread of sectors implies, without overclaiming."""
    if recap.breadth is None:
        return ""
    share = recap.breadth
    up = round(share * len(recap.sectors))
    total = len(recap.sectors)
    if share >= 0.8:
        read = "broad — the whole tape participated"
    elif share >= 0.6:
        read = "positive but uneven"
    elif share >= 0.4:
        read = "mixed; this was a stock-picker's day rather than a market move"
    elif share >= 0.2:
        read = "negative and broad"
    else:
        read = "broadly negative — almost nothing was spared"
    return f"{up}/{total} sectors higher — {read}."


def render_recap(recap: MarketRecap) -> str:
    """The report a person reads at the close."""
    stamp = recap.as_of.astimezone(timezone.utc).strftime("%Y-%m-%d")
    lines = [f"MARKET RECAP — {stamp}"]

    if not recap.usable:
        lines.append("")
        lines.append("No market data was available for this session.")
        lines.extend(f"  ! {error}" for error in recap.errors)
        return "\n".join(lines)

    lines.append("")
    for move in recap.indices:
        trend = "above" if move.above_50d else "below"
        lines.append(
            f"  {move.name:<14} {move.close:>9,.2f}  {move.change_pct:>+6.2f}%"
            f"   (5d {move.change_5d_pct:+.2f}%, {trend} its 50-day)"
        )

    if recap.sectors:
        lines.append("")
        leaders = ", ".join(f"{m.name} {m.change_pct:+.2f}%" for m in recap.leaders)
        laggards = ", ".join(f"{m.name} {m.change_pct:+.2f}%" for m in recap.laggards)
        lines.append(f"  Leading:  {leaders}")
        lines.append(f"  Lagging:  {laggards}")

    breadth = _breadth_sentence(recap)
    if breadth:
        lines.append("")
        lines.append(f"  {breadth}")

    if recap.benchmark_atr_pct is not None:
        lines.append(
            f"  Volatility {recap.regime} — the S&P's average daily range is "
            f"{recap.benchmark_atr_pct:.2f}% of price."
        )
        lines.append(f"  {_regime_advice(recap.regime)}")

    if recap.headlines:
        lines.append("")
        lines.extend(f"  • {headline}" for headline in recap.headlines)

    if recap.errors:
        lines.append("")
        lines.extend(f"  ! {error}" for error in recap.errors)

    lines.append("")
    lines.append(
        "  This describes what already happened. It is not a forecast, and "
        "nothing here\n  adjusts your position sizes on its own."
    )
    return "\n".join(lines)


def _regime_advice(regime: str) -> str:
    """One line on what the regime means for a swing book."""
    return {
        "calm": (
            "Stops sized in ATR will sit close in absolute terms; breakouts tend "
            "to carry further than they look."
        ),
        "normal": "Ordinary conditions for a multi-week swing hold.",
        "elevated": (
            "ATR-sized stops are wide, so the same risk budget buys fewer shares. "
            "That is the system working, not a reason to override it."
        ),
        "stressed": (
            "Correlations rise in tape like this and single-name setups stop being "
            "independent bets. Fewer positions, not smaller ones."
        ),
    }.get(regime, "Not enough history to judge the volatility regime.")


__all__ = [
    "INDEX_ETFS",
    "SECTOR_ETFS",
    "InstrumentMove",
    "MarketRecap",
    "build_recap",
    "describe_regime",
    "render_recap",
]
