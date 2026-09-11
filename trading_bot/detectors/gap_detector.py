"""Overnight gaps.

A gap is the difference between where a symbol stopped trading and where it
starts again. It is the cleanest evidence a scanner gets that something happened
while the market was shut — earnings, a guidance change, a sector-wide move —
because price cannot travel there continuously; it simply reopens elsewhere.

Two measurement bases, chosen by what is actually available:

* **open** — once the regular session has printed, the gap is the official
  distance from yesterday's close to today's open. This is the number every
  other participant is looking at.
* **premarket** — before 09:30 there is no open yet, so the last pre-market
  print stands in. Less reliable (pre-market prices move on small size) but it
  is the only read available during the hours when the decision is being made.

The metrics also record how much of the gap has already been given back. A gap
that held all morning and a gap that closed within ten minutes are different
events, and the second is worth less attention.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

from trading_bot.detectors.base import Detector, GapDetectorConfig
from trading_bot.detectors.context import DetectionContext
from trading_bot.detectors.models import Bias, DetectorSignal, SignalType, scale_strength

logger = logging.getLogger(__name__)

#: A fully retraced gap keeps this share of its strength. Not zero: the gap
#: still happened, and the retracement is itself worth seeing. Not one: a gap
#: that has been given back is a smaller event than one that held.
FILLED_GAP_RETENTION = 0.5


class GapDetector(Detector):
    """Detects a session opening away from the previous close."""

    name: ClassVar[str] = "gap"
    signal_type: ClassVar[SignalType] = SignalType.GAP

    def __init__(self, config: GapDetectorConfig | None = None) -> None:
        self.config = config or GapDetectorConfig()

    def evaluate(self, context: DetectionContext) -> DetectorSignal | None:
        today = context.today
        reference = context.previous_close
        if today is None or reference is None or reference <= 0:
            return None

        if today.regular is not None:
            basis, gap_price = "open", today.regular.open
        elif today.premarket is not None:
            basis, gap_price = "premarket", today.premarket.close
        else:
            return None

        if not np.isfinite(gap_price) or gap_price <= 0:
            return None

        gap = gap_price - reference
        gap_pct = gap / reference * 100.0
        if abs(gap_pct) < self.config.threshold_pct:
            return None

        direction = Bias.BULLISH if gap > 0 else Bias.BEARISH
        raw_strength = scale_strength(
            abs(gap_pct),
            floor=self.config.threshold_pct,
            ceiling=self.config.saturation_pct,
        )

        last = context.last_price if context.last_price is not None else gap_price
        fill_fraction = self._fill_fraction(gap_price, reference, last)
        retention = 1.0 - (1.0 - FILLED_GAP_RETENTION) * fill_fraction
        strength = raw_strength * retention

        metrics = {
            "gap_pct": gap_pct,
            "gap_points": gap,
            "previous_close": reference,
            "gap_price": gap_price,
            "last_price": last,
            "gap_fill_pct": fill_fraction * 100.0,
            "raw_strength": raw_strength,
            "premarket_basis": 1.0 if basis == "premarket" else 0.0,
        }

        where = "Pre-market" if basis == "premarket" else "Opened"
        fill_note = (
            f", {fill_fraction * 100:.0f}% filled" if fill_fraction > 0.05 else ", holding"
        )
        reason = (
            f"{where} {abs(gap_pct):.2f}% {'above' if gap > 0 else 'below'} the previous "
            f"close of {reference:.2f}{fill_note}"
        )

        return self._signal(
            context,
            direction=direction,
            strength=strength,
            reason=reason,
            metrics=metrics,
        )

    @staticmethod
    def _fill_fraction(gap_price: float, reference: float, last: float) -> float:
        """How much of the gap price has been retraced, 0 to 1.

        1.0 means price has traded all the way back to the previous close (or
        through it). Measured from the gap price rather than from the extreme of
        the day, because the question is how much of the gap survives *now*.
        """
        span = gap_price - reference
        if span == 0 or not np.isfinite(last):
            return 0.0
        retraced = (gap_price - last) / span
        return float(min(max(retraced, 0.0), 1.0))
