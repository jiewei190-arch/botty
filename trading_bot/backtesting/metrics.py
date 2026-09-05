"""Performance statistics.

Every figure here is computed from the simulated equity curve and the trade log,
with the degenerate cases handled explicitly rather than left to produce a
plausible-looking number from nothing.

Two things this module refuses to do quietly:

* **Annualise wrongly.** A Sharpe ratio from 15-minute bars must be scaled by the
  square root of ~6,552 periods per year, not ~252. Using the daily figure on
  intraday data inflates it about fivefold, and that mistake is common enough
  that the timeframe is a required argument rather than an optional one.
* **Report a statistic the sample cannot support.** A Sharpe ratio or a win rate
  from nine trades is noise wearing a decimal point. Results carry a
  ``sample_warning`` when the trade count is too small to mean anything, and the
  report prints it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from trading_bot.utils.timeframes import Timeframe

logger = logging.getLogger(__name__)

#: Below this many trades, ratios are dominated by luck. Widely cited rule of
#: thumb; the exact number matters less than refusing to imply false precision.
MIN_MEANINGFUL_TRADES = 30

#: Ceiling on the annualised-return display, in percent. Extrapolating a handful
#: of bars to a year produces a number, not a forecast.
_ANNUALISED_CAP_PCT = 10_000.0

#: The cap expressed as a log growth factor, so the extrapolation can be bounded
#: before it is exponentiated rather than after.
_MAX_LOG_GROWTH = math.log(1 + _ANNUALISED_CAP_PCT / 100)


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Everything measured from one backtest."""

    starting_equity: float
    ending_equity: float
    total_return_pct: float
    annualised_return_pct: float

    total_trades: int
    wins: int
    losses: int
    win_rate: float

    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float
    expectancy_r: float

    max_drawdown_pct: float
    max_drawdown_value: float
    max_drawdown_bars: int

    sharpe_ratio: float
    sortino_ratio: float

    total_commission: float
    total_slippage: float
    gapped_stops: int

    bars: int
    exposure_pct: float
    sample_warning: str | None = None
    exit_breakdown: dict[str, int] = field(default_factory=dict)

    @property
    def net_profit(self) -> float:
        return self.ending_equity - self.starting_equity

    @property
    def is_meaningful(self) -> bool:
        """Whether the sample supports the ratios reported."""
        return self.sample_warning is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "starting_equity": round(self.starting_equity, 2),
            "ending_equity": round(self.ending_equity, 2),
            "net_profit": round(self.net_profit, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "annualised_return_pct": round(self.annualised_return_pct, 2),
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 2),
            "average_win": round(self.average_win, 2),
            "average_loss": round(self.average_loss, 2),
            "largest_win": round(self.largest_win, 2),
            "largest_loss": round(self.largest_loss, 2),
            "profit_factor": round(self.profit_factor, 3),
            "expectancy_r": round(self.expectancy_r, 3),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "max_drawdown_value": round(self.max_drawdown_value, 2),
            "max_drawdown_bars": self.max_drawdown_bars,
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "sortino_ratio": round(self.sortino_ratio, 3),
            "total_commission": round(self.total_commission, 2),
            "total_slippage": round(self.total_slippage, 2),
            "gapped_stops": self.gapped_stops,
            "bars": self.bars,
            "exposure_pct": round(self.exposure_pct, 2),
            "sample_warning": self.sample_warning,
            "exit_breakdown": dict(self.exit_breakdown),
        }

    def summary_lines(self) -> list[str]:
        """Human-readable report body."""
        lines = [
            f"Starting equity   : ${self.starting_equity:,.2f}",
            f"Ending equity     : ${self.ending_equity:,.2f}",
            f"Net profit        : ${self.net_profit:,.2f}  ({self.total_return_pct:+.2f}%)",
            f"Annualised return : {self.annualised_return_pct:+.2f}%",
            "",
            f"Trades            : {self.total_trades}"
            f"  ({self.wins} win / {self.losses} loss)",
            f"Win rate          : {self.win_rate:.1f}%",
            f"Average win       : ${self.average_win:,.2f}",
            f"Average loss      : ${self.average_loss:,.2f}",
            f"Largest win       : ${self.largest_win:,.2f}",
            f"Largest loss      : ${self.largest_loss:,.2f}",
            f"Profit factor     : {self.profit_factor:.2f}",
            f"Expectancy        : {self.expectancy_r:+.3f} R per trade",
            "",
            f"Max drawdown      : {self.max_drawdown_pct:.2f}%"
            f"  (${self.max_drawdown_value:,.2f} over {self.max_drawdown_bars} bars)",
            f"Sharpe ratio      : {self.sharpe_ratio:.2f}",
            f"Sortino ratio     : {self.sortino_ratio:.2f}",
            f"Time in market    : {self.exposure_pct:.1f}%",
            "",
            f"Commission paid   : ${self.total_commission:,.2f}",
            f"Slippage cost     : ${self.total_slippage:,.2f}",
        ]
        if self.gapped_stops:
            lines.append(
                f"Stops gapped      : {self.gapped_stops}"
                "  (filled worse than the stop price)"
            )
        if self.exit_breakdown:
            lines.append("")
            lines.append("Exits by reason:")
            for reason, count in sorted(
                self.exit_breakdown.items(), key=lambda item: -item[1]
            ):
                lines.append(f"  {count:>4}  {reason}")
        if self.sample_warning:
            lines.extend(["", f"!! {self.sample_warning}"])
        return lines


def _drawdown_series(equity: pd.Series) -> tuple[pd.Series, float, float, int]:
    """Drawdown curve plus the worst drawdown and how long it lasted."""
    running_peak = equity.cummax()
    drawdown = (equity - running_peak) / running_peak.replace(0, np.nan) * 100
    drawdown = drawdown.fillna(0.0)

    if drawdown.empty:
        return drawdown, 0.0, 0.0, 0

    worst_pct = float(drawdown.min())
    trough = drawdown.idxmin()
    peak_value = float(running_peak.loc[trough])
    worst_value = peak_value - float(equity.loc[trough])

    # Longest stretch spent below a previous peak.
    under_water = equity < running_peak
    longest = current = 0
    for flag in under_water:
        current = current + 1 if flag else 0
        longest = max(longest, current)

    return drawdown, abs(worst_pct), worst_value, longest


def _ratio(returns: pd.Series, periods_per_year: float, *, downside_only: bool) -> float:
    """Sharpe or Sortino, annualised. Assumes a zero risk-free rate."""
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0

    mean = float(clean.mean())
    if downside_only:
        negative = clean[clean < 0]
        deviation = float(negative.std(ddof=1)) if len(negative) >= 2 else 0.0
    else:
        deviation = float(clean.std(ddof=1))

    if deviation <= 0:
        # No variability at all: a ratio would be infinite, which is not a
        # measurement. Report zero and let the sample warning speak.
        return 0.0
    return mean / deviation * math.sqrt(periods_per_year)


def calculate_metrics(
    equity_curve: pd.Series,
    trades: list[dict[str, Any]],
    *,
    timeframe: Timeframe | str,
    starting_equity: float,
    bars_in_market: int = 0,
) -> PerformanceMetrics:
    """Compute performance statistics.

    Parameters
    ----------
    equity_curve:
        Account equity marked to market on every bar, indexed by timestamp.
    trades:
        Closed trades, each with at least ``pnl`` and optionally ``r_multiple``,
        ``commission``, ``slippage``, ``exit_reason`` and ``gapped``.
    timeframe:
        Bar size, used to annualise correctly.
    starting_equity:
        Equity the run began with.
    bars_in_market:
        Bars spent holding a position, for the exposure figure.

    Returns
    -------
    PerformanceMetrics
    """
    parsed = Timeframe.parse(timeframe)
    bars = len(equity_curve)
    ending = float(equity_curve.iloc[-1]) if bars else starting_equity

    total_return_pct = (
        (ending / starting_equity - 1) * 100 if starting_equity > 0 else 0.0
    )

    years = bars / parsed.periods_per_year if parsed.periods_per_year > 0 else 0.0
    if years > 0 and starting_equity > 0 and ending > 0:
        # Computed in log space. A short sample makes 1/years enormous, and the
        # direct power overflows a float before the clip below can bound it —
        # two 15-minute bars of a 3x gain annualise to 3**3276.
        log_growth = math.log(ending / starting_equity) / years
        annualised = (
            _ANNUALISED_CAP_PCT
            if log_growth >= _MAX_LOG_GROWTH
            else (math.exp(log_growth) - 1) * 100
        )
    else:
        annualised = 0.0
    # A three-day sample extrapolated to a year is not a forecast; cap the
    # display so an absurd figure does not read as a result.
    annualised = float(np.clip(annualised, -100.0, _ANNUALISED_CAP_PCT))

    pnls = [float(trade.get("pnl", 0.0)) for trade in trades]
    wins = [value for value in pnls if value > 0]
    losses = [value for value in pnls if value <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))

    r_multiples = [
        float(trade["r_multiple"])
        for trade in trades
        if trade.get("r_multiple") is not None
    ]

    returns = equity_curve.pct_change() if bars > 1 else pd.Series(dtype="float64")
    drawdown, worst_pct, worst_value, longest = _drawdown_series(equity_curve)

    warning: str | None = None
    if not trades:
        warning = "No trades were taken — every ratio below is undefined, not zero."
    elif len(trades) < MIN_MEANINGFUL_TRADES:
        warning = (
            f"Only {len(trades)} trade(s). Win rate, profit factor and Sharpe are "
            f"dominated by luck below about {MIN_MEANINGFUL_TRADES}; treat them as "
            "anecdotes rather than measurements."
        )

    exit_breakdown: dict[str, int] = {}
    for trade in trades:
        reason = str(trade.get("exit_reason") or "unknown")
        exit_breakdown[reason] = exit_breakdown.get(reason, 0) + 1

    return PerformanceMetrics(
        starting_equity=starting_equity,
        ending_equity=ending,
        total_return_pct=total_return_pct,
        annualised_return_pct=annualised,
        total_trades=len(trades),
        wins=len(wins),
        losses=len(losses),
        win_rate=(len(wins) / len(trades) * 100) if trades else 0.0,
        average_win=(gross_profit / len(wins)) if wins else 0.0,
        average_loss=(sum(losses) / len(losses)) if losses else 0.0,
        largest_win=max(wins) if wins else 0.0,
        largest_loss=min(losses) if losses else 0.0,
        profit_factor=(gross_profit / gross_loss) if gross_loss > 0 else (
            float("inf") if gross_profit > 0 else 0.0
        ),
        expectancy_r=float(np.mean(r_multiples)) if r_multiples else 0.0,
        max_drawdown_pct=worst_pct,
        max_drawdown_value=worst_value,
        max_drawdown_bars=longest,
        sharpe_ratio=_ratio(returns, parsed.periods_per_year, downside_only=False),
        sortino_ratio=_ratio(returns, parsed.periods_per_year, downside_only=True),
        total_commission=sum(float(trade.get("commission", 0.0)) for trade in trades),
        total_slippage=sum(float(trade.get("slippage", 0.0)) for trade in trades),
        gapped_stops=sum(1 for trade in trades if trade.get("gapped")),
        bars=bars,
        exposure_pct=(bars_in_market / bars * 100) if bars else 0.0,
        sample_warning=warning,
        exit_breakdown=exit_breakdown,
    )


def drawdown_curve(equity_curve: pd.Series) -> pd.Series:
    """Percentage drawdown from the running peak, for charting."""
    return _drawdown_series(equity_curve)[0]
