"""Unusual volume.

Relative volume is the ratio of what has traded to what normally trades. The
subtlety is in the word "normally": comparing a half-finished session against a
full-day average makes every morning look quiet and every afternoon look busy,
which is a property of the clock rather than of the stock.

So the comparison is **time-of-day matched**. Volume accumulated through the
first ``k`` bars of today is compared with volume accumulated through the first
``k`` bars of each recent session. At 10:15 today is measured against 10:15
before, and the ratio means the same thing at every hour of the day.

Volume has no direction of its own — a surge on a collapsing price is not
bullish. The direction reported here comes from the price change that
accompanied the surge, and is NEUTRAL when the price barely moved, which is
itself informative: heavy volume going nowhere is distribution or accumulation,
not a trend.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

from trading_bot.detectors.base import Detector, VolumeDetectorConfig
from trading_bot.detectors.context import DetectionContext
from trading_bot.detectors.models import Bias, DetectorSignal, SignalType, scale_strength

logger = logging.getLogger(__name__)


class VolumeDetector(Detector):
    """Detects participation well above the symbol's own normal."""

    name: ClassVar[str] = "unusual_volume"
    signal_type: ClassVar[SignalType] = SignalType.UNUSUAL_VOLUME

    def __init__(self, config: VolumeDetectorConfig | None = None) -> None:
        self.config = config or VolumeDetectorConfig()

    def evaluate(self, context: DetectionContext) -> DetectorSignal | None:
        today = context.today
        if today is None or len(context.history) < self.config.min_sessions:
            return None

        matched = self._time_matched_ratio(context)
        if matched is None:
            matched = self._premarket_ratio(context)
        if matched is None:
            return None

        ratio, traded, baseline, sessions_used, basis = matched
        if ratio < self.config.threshold:
            return None

        extent = today.regular if basis == "session" else today.premarket
        change_pct = extent.change_pct if extent else 0.0
        direction = Bias.from_change(change_pct, deadband=self.config.direction_deadband_pct)

        metrics = {
            "relative_volume": ratio,
            "volume": traded,
            "baseline_volume": baseline,
            "sessions_used": float(sessions_used),
            "change_pct": change_pct,
        }
        bar_ratio = self._bar_ratio(context)
        if bar_ratio is not None:
            metrics["bar_relative_volume"] = bar_ratio

        where = "pre-market " if basis == "premarket" else ""
        move = (
            f"price {change_pct:+.2f}%"
            if direction is not Bias.NEUTRAL
            else f"price flat ({change_pct:+.2f}%)"
        )
        reason = (
            f"{where}volume {ratio:.1f}x the {sessions_used}-session norm for this "
            f"time of day, {move}"
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

    def _time_matched_ratio(
        self, context: DetectionContext
    ) -> tuple[float, float, float, int, str] | None:
        """Today's cumulative regular volume against prior days at the same point."""
        today = context.today
        if today is None:
            return None
        curve = context.session_volume_curve(today.day)
        if curve.size == 0:
            return None

        position = curve.size - 1
        traded = float(curve[position])
        baselines: list[float] = []
        for session in context.history:
            prior = context.session_volume_curve(session.day)
            if prior.size == 0:
                continue
            # A short prior session (a half day, or a gap in the data) is
            # compared at its own last bar rather than being skipped: its total
            # is the best available answer to "how much had traded by now".
            baselines.append(float(prior[min(position, prior.size - 1)]))

        if len(baselines) < self.config.min_sessions:
            return None
        baseline = float(np.median(baselines))
        if baseline <= 0 or traded <= 0:
            return None
        return traded / baseline, traded, baseline, len(baselines), "session"

    def _premarket_ratio(
        self, context: DetectionContext
    ) -> tuple[float, float, float, int, str] | None:
        """Before the bell, compare pre-market volume with prior pre-markets.

        Pre-market volume is a fraction of regular volume, so it can only be
        judged against other pre-markets. This is the earliest honest read
        available on whether a symbol is drawing unusual interest today.
        """
        today = context.today
        if today is None or today.premarket is None:
            return None
        traded = today.premarket.volume
        baselines = [
            session.premarket.volume
            for session in context.history
            if session.premarket is not None and session.premarket.volume > 0
        ]
        if traded <= 0 or len(baselines) < self.config.min_sessions:
            return None
        baseline = float(np.median(baselines))
        if baseline <= 0:
            return None
        return traded / baseline, traded, baseline, len(baselines), "premarket"

    def _bar_ratio(self, context: DetectionContext) -> float | None:
        """The last bar's volume against recent bars of the same session type.

        Kept separate from the session ratio because they answer different
        questions: the session ratio says the day is busy, the bar ratio says
        *right now* is busy. A quiet day with one enormous bar is a different
        event from a uniformly heavy day.
        """
        frame = context.bars
        if frame.empty or len(frame) < 2:
            return None
        try:
            last_session = context.bar_sessions.iloc[-1]
            same = frame[context.bar_sessions == last_session]
        except (IndexError, KeyError):  # pragma: no cover - defensive
            same = frame
        if len(same) < 2:
            return None
        window = same["volume"].iloc[-(self.config.bar_lookback + 1) : -1]
        if window.empty:
            return None
        baseline = float(window.mean())
        current = float(same["volume"].iloc[-1])
        if not np.isfinite(baseline) or baseline <= 0 or not np.isfinite(current):
            return None
        return current / baseline


