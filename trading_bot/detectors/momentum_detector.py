"""Price momentum.

Three pieces of evidence about the same thing:

* **Size** — how far price moved over the lookback, in ATR units. ATR is the
  denominator because a 2% move means something different in a utility than in a
  small-cap semiconductor; dividing by the symbol's own typical range puts every
  ticker on one scale, which is exactly what a cross-market scanner needs.
* **Speed** — the same move expressed in percent, kept for readability and used
  as the fallback when history is too short for an ATR.
* **Persistence** — how many consecutive bars pushed the same way. A 2-ATR move
  made of one spike and four bars of chop is a different animal from four bars
  each closing higher.

Size decides whether there is a signal at all. Persistence can only strengthen
one, never create one: a long run of tiny bars is drift, not momentum.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

from trading_bot.detectors.base import Detector, MomentumDetectorConfig
from trading_bot.detectors.context import DetectionContext
from trading_bot.detectors.models import (
    Bias,
    DetectorSignal,
    SignalType,
    clamp_strength,
    scale_strength,
)
from trading_bot.indicators import calculate_atr

logger = logging.getLogger(__name__)


def consecutive_run(closes: np.ndarray) -> int:
    """Signed count of trailing bars that closed in the same direction.

    ``+3`` means the last three bars each closed above the one before. An
    unchanged close ends the run: flat is not agreement.
    """
    if closes.size < 2:
        return 0
    diffs = np.diff(closes)
    if diffs.size == 0 or not np.isfinite(diffs[-1]) or diffs[-1] == 0:
        return 0
    sign = 1 if diffs[-1] > 0 else -1
    run = 0
    for value in diffs[::-1]:
        if not np.isfinite(value) or value == 0:
            break
        if (value > 0) != (sign > 0):
            break
        run += 1
    return run * sign


class MomentumDetector(Detector):
    """Detects fast, sustained directional movement."""

    name: ClassVar[str] = "momentum"
    signal_type: ClassVar[SignalType] = SignalType.MOMENTUM

    def __init__(self, config: MomentumDetectorConfig | None = None) -> None:
        self.config = config or MomentumDetectorConfig()

    def evaluate(self, context: DetectionContext) -> DetectorSignal | None:
        frame = context.bars
        lookback = self.config.lookback_bars
        if frame.empty or len(frame) <= lookback:
            return None

        closes = frame["close"].to_numpy(dtype="float64")
        start = closes[-(lookback + 1)]
        last = closes[-1]
        if not np.isfinite(start) or not np.isfinite(last) or start <= 0:
            return None

        move = last - start
        change_pct = move / start * 100.0
        atr = self._atr(context)
        atr_multiple = move / atr if atr and atr > 0 else None

        qualifies_on_atr = (
            atr_multiple is not None and abs(atr_multiple) >= self.config.atr_threshold
        )
        qualifies_on_pct = abs(change_pct) >= self.config.percent_threshold
        if not (qualifies_on_atr or qualifies_on_pct):
            return None

        if atr_multiple is not None:
            base = scale_strength(
                abs(atr_multiple),
                floor=self.config.atr_threshold,
                ceiling=self.config.atr_saturation,
            )
        else:
            # No ATR yet (short history): fall back to percent, saturating at
            # three times the qualifying move so the scale stays comparable.
            base = scale_strength(
                abs(change_pct),
                floor=self.config.percent_threshold,
                ceiling=self.config.percent_threshold * 3,
            )

        run = consecutive_run(closes)
        direction = Bias.from_change(change_pct)
        if direction is Bias.NEUTRAL:
            return None

        # A run only counts when it agrees with the move it is supposed to
        # corroborate; a bullish move ending in three red bars is weakening.
        agreeing_run = abs(run) if (run > 0) == (direction is Bias.BULLISH) else 0
        run_score = scale_strength(
            float(agreeing_run),
            floor=float(self.config.run_threshold),
            ceiling=float(self.config.run_threshold + 4),
        )
        strength = clamp_strength(base + self.config.run_weight * run_score)

        metrics = {
            "change_pct": change_pct,
            "lookback_bars": float(lookback),
            "consecutive_bars": float(run),
            "run_score": run_score,
            "base_strength": base,
        }
        if atr_multiple is not None:
            metrics["atr_multiple"] = atr_multiple
            metrics["atr"] = float(atr)
        session_extent = context.today.regular if context.today else None
        if session_extent is not None:
            metrics["session_change_pct"] = session_extent.change_pct

        atr_note = f" ({abs(atr_multiple):.1f} ATR)" if atr_multiple is not None else ""
        run_note = (
            f", {agreeing_run} bars in a row"
            if agreeing_run >= self.config.run_threshold
            else ""
        )
        reason = (
            f"{'Up' if direction is Bias.BULLISH else 'Down'} {abs(change_pct):.2f}% "
            f"over {lookback} bars{atr_note}{run_note}"
        )

        return self._signal(
            context,
            direction=direction,
            strength=strength,
            reason=reason,
            metrics=metrics,
        )

    def _atr(self, context: DetectionContext) -> float | None:
        """Latest ATR, reusing a precomputed column when the frame has one.

        The scanner enriches frames once with ``calculate_all_indicators``; when
        it has, recomputing here would be pure waste, and — more importantly —
        could disagree with the value every other component is reading.
        """
        period = self.config.atr_period
        column = f"ATR_{period}"
        frame = context.bars
        if column in frame.columns:
            value = frame[column].iloc[-1]
            return float(value) if np.isfinite(value) else None
        if len(frame) < period + 1:
            return None
        series = calculate_atr(frame["high"], frame["low"], frame["close"], period=period)
        if series.empty:
            return None
        value = series.iloc[-1]
        return float(value) if np.isfinite(value) else None
