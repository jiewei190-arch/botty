"""Does the score predict anything?

The scanner ranks setups 0-100. That number is the share of a strategy's
supporting evidence that was present — it is *not* a probability of profit, and
this module exists because the difference matters enormously to anyone reading
the list.

The honest way to attach odds to a score is to measure them. Run the backtester
over a long window and many symbols, bucket the closed trades by the score they
entered on, and count how each bucket actually resolved. That yields a real
statement: "setups scoring 85-90 resolved as winners 47% of the time across 312
trades." Historical frequency, with its sample size attached.

What this cannot become
-----------------------
A per-trade probability. "This setup has a 73% chance" is not a thing this or
any tool can honestly say: the sample is a different market, the future is not
drawn from the same distribution, and a frequency measured over 300 trades has
a wide interval around it even on its own terms.

What it is good for is comparison and falsification. If the 85+ band wins no
more often than the 55-60 band, the ranking is decoration and should be ignored
or rebuilt — and finding that out is worth more than any number here.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: Trades a band needs before its win rate is worth reading. Below this the
#: figure moves several points on a single trade, which is noise wearing a
#: decimal point.
MIN_BAND_SAMPLE = 30

#: Default score boundaries. Coarse on purpose: fine buckets look precise and
#: are mostly sampling error.
DEFAULT_BANDS: tuple[float, ...] = (0.0, 60.0, 70.0, 80.0, 90.0, 100.01)


@dataclass(frozen=True, slots=True)
class ScoreBand:
    """How setups in one score range actually resolved."""

    low: float
    high: float
    trades: int
    wins: int
    total_r: float
    #: Trades whose R multiple was recorded, the denominator for ``average_r``.
    measured: int = 0

    @property
    def win_rate(self) -> float:
        return (self.wins / self.trades * 100) if self.trades else 0.0

    @property
    def average_r(self) -> float:
        return (self.total_r / self.measured) if self.measured else 0.0

    @property
    def is_meaningful(self) -> bool:
        return self.trades >= MIN_BAND_SAMPLE

    @property
    def label(self) -> str:
        return f"{self.low:.0f}-{min(self.high, 100):.0f}"

    def standard_error(self) -> float:
        """Rough standard error of the win rate, in percentage points.

        A band at 45% over 40 trades carries an error of about 8 points, which
        is the difference between an edge and nothing. Reporting the rate
        without it invites reading noise as signal.
        """
        if self.trades < 2:
            return 0.0
        rate = self.wins / self.trades
        return math.sqrt(rate * (1 - rate) / self.trades) * 100


@dataclass(slots=True)
class Calibration:
    """What a backtest says about the score's predictive value."""

    bands: list[ScoreBand] = field(default_factory=list)
    total_trades: int = 0
    total_wins: int = 0
    total_r: float = 0.0
    measured: int = 0

    @property
    def base_rate(self) -> float:
        """Win rate across every trade, whatever it scored.

        The number every band must beat to justify the ranking's existence.
        """
        return (self.total_wins / self.total_trades * 100) if self.total_trades else 0.0

    @property
    def expectancy_r(self) -> float:
        return (self.total_r / self.measured) if self.measured else 0.0

    @property
    def usable_bands(self) -> list[ScoreBand]:
        return [band for band in self.bands if band.is_meaningful]

    def lift(self, band: ScoreBand) -> float:
        """Percentage points by which a band beats taking every setup."""
        return band.win_rate - self.base_rate

    @property
    def is_monotonic(self) -> bool:
        """Whether win rate rises with score across the bands worth reading.

        The property a ranking claims by existing. If it does not hold, a
        higher score does not mean a better setup and the order on screen is
        not information.
        """
        rates = [band.win_rate for band in self.usable_bands]
        return all(later >= earlier for earlier, later in zip(rates, rates[1:], strict=False))

    @property
    def verdict(self) -> str:
        """A plain reading of whether the score earned its place."""
        usable = self.usable_bands
        if self.total_trades < MIN_BAND_SAMPLE:
            return (
                f"Only {self.total_trades} trades — too few to say anything. "
                "Widen the window or the symbol list."
            )
        if len(usable) < 2:
            return (
                "Only one score band has enough trades to read, so the bands "
                "cannot be compared. The score is unvalidated."
            )
        best, worst = max(usable, key=lambda b: b.win_rate), min(usable, key=lambda b: b.win_rate)
        spread = best.win_rate - worst.win_rate
        error = best.standard_error() + worst.standard_error()
        if spread <= error:
            return (
                f"The best and worst bands differ by {spread:.1f} points, within "
                f"the {error:.1f}-point sampling error. On this sample the score "
                "does not separate winners from losers — treat the ranking as "
                "unproven rather than as information."
            )
        direction = "rises with" if self.is_monotonic else "does not rise cleanly with"
        return (
            f"Win rate {direction} score: {worst.label} wins {worst.win_rate:.1f}%, "
            f"{best.label} wins {best.win_rate:.1f}% — a {spread:.1f}-point spread "
            f"against {error:.1f} points of sampling error."
        )

    def summary_lines(self) -> list[str]:
        lines = [
            f"Base rate: {self.base_rate:.1f}% of {self.total_trades:,} trades won, "
            f"expectancy {self.expectancy_r:+.3f}R",
            "",
            f"  {'score':<10}{'trades':>8}{'win rate':>12}{'+/-':>7}{'avg R':>9}{'lift':>9}",
            "  " + "-" * 55,
        ]
        for band in self.bands:
            if not band.trades:
                continue
            note = "" if band.is_meaningful else "  (too few)"
            lines.append(
                f"  {band.label:<10}{band.trades:>8,}{band.win_rate:>11.1f}%"
                f"{band.standard_error():>6.1f}{band.average_r:>9.2f}"
                f"{self.lift(band):>+8.1f}{note}"
            )
        lines += ["", self.verdict]
        return lines


def calibrate(
    trades: list[dict[str, Any]], bands: tuple[float, ...] = DEFAULT_BANDS
) -> Calibration:
    """Measure how trades resolved, bucketed by the score they entered on.

    Parameters
    ----------
    trades:
        Closed trades from one or more backtests. Each needs ``pnl`` and
        ``confidence``; ``r_multiple`` is used when present.
    bands:
        Score boundaries, ascending. The last value is exclusive.

    Returns
    -------
    Calibration
    """
    edges = sorted(bands)
    if len(edges) < 2:
        raise ValueError("bands needs at least a lower and an upper boundary")

    buckets: list[dict[str, float]] = [
        {"wins": 0, "trades": 0, "total_r": 0.0, "measured": 0}
        for _ in range(len(edges) - 1)
    ]
    result = Calibration()

    for trade in trades:
        score = trade.get("confidence")
        if score is None:
            continue
        index = _band_index(float(score), edges)
        if index is None:
            continue

        won = float(trade.get("pnl", 0.0)) > 0
        bucket = buckets[index]
        bucket["trades"] += 1
        result.total_trades += 1
        if won:
            bucket["wins"] += 1
            result.total_wins += 1

        r_multiple = trade.get("r_multiple")
        if r_multiple is not None:
            bucket["total_r"] += float(r_multiple)
            bucket["measured"] += 1
            result.total_r += float(r_multiple)
            result.measured += 1

    result.bands = [
        ScoreBand(
            low=edges[index],
            high=edges[index + 1],
            trades=int(bucket["trades"]),
            wins=int(bucket["wins"]),
            total_r=bucket["total_r"],
            measured=int(bucket["measured"]),
        )
        for index, bucket in enumerate(buckets)
    ]
    return result


def _band_index(score: float, edges: list[float]) -> int | None:
    """Which band a score falls in, or None when outside every band."""
    for index in range(len(edges) - 1):
        if edges[index] <= score < edges[index + 1]:
            return index
    return None
