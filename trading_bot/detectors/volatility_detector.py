"""Volatility expansion.

Markets alternate between compression and expansion. A symbol whose bars have
suddenly grown much larger than its own recent norm is being repriced, and that
is worth knowing regardless of which way it is going — which is why this is the
one detector that routinely reports NEUTRAL. Expansion is a statement about
*how much* price is moving, not about where it is headed.

The measurement is a ratio of two true-range averages over the same series: a
short window over a long one. Using the symbol against itself means no
cross-sectional assumption is needed — a 3% daily range is calm for one ticker
and extraordinary for another, and neither has to be configured.

A second, coarser reading compares today's session range with the median of
recent sessions. It is reported alongside because the bar-level ratio can spike
on a single wide bar; agreement between the two is what distinguishes a regime
change from one loud print.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

from trading_bot.detectors.base import Detector, VolatilityDetectorConfig
from trading_bot.detectors.context import DetectionContext
from trading_bot.detectors.models import Bias, DetectorSignal, SignalType, scale_strength
from trading_bot.indicators import calculate_true_range

logger = logging.getLogger(__name__)


class VolatilityDetector(Detector):
    """Detects bar ranges well above the symbol's own recent norm."""

    name: ClassVar[str] = "volatility_expansion"
    signal_type: ClassVar[SignalType] = SignalType.VOLATILITY_EXPANSION

    def __init__(self, config: VolatilityDetectorConfig | None = None) -> None:
        self.config = config or VolatilityDetectorConfig()

    def evaluate(self, context: DetectionContext) -> DetectorSignal | None:
        frame = context.bars
        slow = self.config.slow_period
        fast = self.config.fast_period
        if frame.empty or len(frame) < slow + 1 or fast >= slow:
            return None

        true_range = calculate_true_range(frame["high"], frame["low"], frame["close"])
        values = true_range.to_numpy(dtype="float64")
        fast_window = values[-fast:]
        slow_window = values[-slow:]
        if np.all(np.isnan(fast_window)) or np.all(np.isnan(slow_window)):
            return None

        fast_range = float(np.nanmean(fast_window))
        slow_range = float(np.nanmean(slow_window))
        if not np.isfinite(fast_range) or not np.isfinite(slow_range) or slow_range <= 0:
            return None

        ratio = fast_range / slow_range
        if ratio < self.config.threshold:
            return None

        last = context.last_price or 0.0
        metrics = {
            "expansion_ratio": ratio,
            "fast_true_range": fast_range,
            "slow_true_range": slow_range,
            "fast_period": float(fast),
            "slow_period": float(slow),
        }
        if last > 0:
            metrics["current_range_pct"] = fast_range / last * 100.0
            metrics["baseline_range_pct"] = slow_range / last * 100.0

        session_ratio = self._session_range_ratio(context)
        if session_ratio is not None:
            metrics["session_range_ratio"] = session_ratio

        change_pct = self._recent_change_pct(context)
        metrics["change_pct"] = change_pct
        direction = Bias.from_change(change_pct, deadband=self.config.direction_deadband_pct)

        lean = {
            Bias.BULLISH: "expanding upward",
            Bias.BEARISH: "expanding downward",
            Bias.NEUTRAL: "two-sided",
        }[direction]
        agreement = (
            f", session range {session_ratio:.1f}x normal" if session_ratio is not None else ""
        )
        reason = (
            f"Bar ranges {ratio:.1f}x the {slow}-bar norm, {lean}{agreement}"
        )

        return self._signal(
            context,
            direction=direction,
            strength=scale_strength(
                ratio, floor=self.config.threshold, ceiling=self.config.saturation
            ),
            reason=reason,
            metrics=metrics,
        )

    def _session_range_ratio(self, context: DetectionContext) -> float | None:
        """Today's high-low range against the median of recent sessions."""
        today = context.today
        if today is None or today.regular is None:
            return None
        current = today.regular.range_pct
        history = [
            session.regular.range_pct
            for session in context.history
            if session.regular is not None and session.regular.range_pct > 0
        ]
        if current <= 0 or len(history) < 3:
            return None
        baseline = float(np.median(history))
        if baseline <= 0:
            return None
        return current / baseline

    def _recent_change_pct(self, context: DetectionContext) -> float:
        """Price change over the fast window — which way the expansion leans."""
        closes = context.bars["close"].to_numpy(dtype="float64")
        span = min(self.config.fast_period, closes.size - 1)
        if span < 1:
            return 0.0
        start, last = closes[-(span + 1)], closes[-1]
        if not np.isfinite(start) or start <= 0 or not np.isfinite(last):
            return 0.0
        return (last - start) / start * 100.0
