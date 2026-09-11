"""Level breakouts.

Traders watch a small number of prices in every symbol: yesterday's high and
low, the pre-market extremes, and the range the session has carved out so far.
Those levels matter because everyone can see them, which makes a decisive move
through one a genuine event rather than a coincidence of arithmetic.

Two rules keep this from firing constantly:

* **A buffer.** Price must close *past* the level by a margin, not merely touch
  it. Levels are magnets; a tick through one and back is the most common thing
  that happens at a level, and calling that a breakout would fill the scanner
  with noise.
* **Freshness.** The level must have been intact on the previous bar. Without
  this, a symbol that broke out at 09:45 keeps reporting a breakout until the
  close, and the one alert that mattered is buried under forty that did not.

Volume is deliberately *not* checked here even though breakouts are usually
judged with it. Confirmation is the scoring engine's job — keeping the
detectors independent is what lets each be tested and tuned on its own, and what
stops one weak reading from being laundered through another detector's logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from trading_bot.detectors.base import BreakoutDetectorConfig, Detector
from trading_bot.detectors.context import DetectionContext
from trading_bot.detectors.models import Bias, DetectorSignal, SignalType, scale_strength
from trading_bot.utils.market_hours import MarketSession

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Level:
    """A price other people are watching."""

    name: str
    price: float
    direction: Bias
    #: How much the market cares, 0-1. Yesterday's high is on every chart;
    #: the high of the last forty minutes is on far fewer.
    significance: float


class BreakoutDetector(Detector):
    """Detects a decisive close through a watched level."""

    name: ClassVar[str] = "breakout"
    signal_type: ClassVar[SignalType] = SignalType.BREAKOUT

    def __init__(self, config: BreakoutDetectorConfig | None = None) -> None:
        self.config = config or BreakoutDetectorConfig()

    def evaluate(self, context: DetectionContext) -> DetectorSignal | None:
        frame = context.bars
        if len(frame) < 2:
            return None

        closes = frame["close"].to_numpy(dtype="float64")
        last, previous = closes[-1], closes[-2]
        if not np.isfinite(last) or not np.isfinite(previous) or last <= 0:
            return None

        levels = self._levels(context)
        if not levels:
            return None

        buffer = self.config.buffer_pct / 100.0
        best: tuple[float, Level, float] | None = None
        broken: dict[str, float] = {}

        for level in levels:
            if level.price <= 0 or not np.isfinite(level.price):
                continue
            if level.direction is Bias.BULLISH:
                cleared = last > level.price * (1 + buffer)
                was_intact = previous <= level.price
                distance_pct = (last - level.price) / level.price * 100.0
            else:
                cleared = last < level.price * (1 - buffer)
                was_intact = previous >= level.price
                distance_pct = (level.price - last) / level.price * 100.0
            if not cleared:
                continue
            if self.config.require_fresh and not was_intact:
                continue

            broken[level.name] = level.price
            strength = level.significance * scale_strength(
                distance_pct,
                floor=self.config.buffer_pct,
                ceiling=self.config.saturation_pct,
            )
            # The headline is the *most watched* level cleared, not the one
            # cleared by the widest margin. Blowing 2% through the high of the
            # last half hour is a smaller event than edging past yesterday's,
            # and a trader reading the alert expects the bigger level named
            # first. Distance only breaks ties within a significance tier.
            candidate = (level.significance, strength)
            if best is None or candidate > (best[1].significance, best[0]):
                best = (strength, level, distance_pct)

        if best is None:
            return None

        strength, level, distance_pct = best
        metrics = {
            "level": level.price,
            "level_significance": level.significance,
            "distance_pct": distance_pct,
            "close": last,
            "previous_close_bar": previous,
            "levels_broken": float(len(broken)),
        }
        metrics.update({f"level_{name}": price for name, price in broken.items()})

        others = [name for name in broken if name != level.name]
        also = f" (also cleared {', '.join(others)})" if others else ""
        verb = "above" if level.direction is Bias.BULLISH else "below"
        reason = (
            f"Closed {distance_pct:.2f}% {verb} the {level.name.replace('_', ' ')} "
            f"at {level.price:.2f}{also}"
        )

        return self._signal(
            context,
            direction=level.direction,
            strength=strength,
            reason=reason,
            metrics=metrics,
        )

    def _levels(self, context: DetectionContext) -> list[Level]:
        """Assemble the levels worth testing for this symbol right now."""
        levels: list[Level] = []

        previous = context.previous
        if previous is not None:
            levels.append(Level("previous_day_high", previous.high, Bias.BULLISH, 1.0))
            levels.append(Level("previous_day_low", previous.low, Bias.BEARISH, 1.0))

        today = context.today
        if today is not None and today.premarket is not None:
            levels.append(Level("premarket_high", today.premarket.high, Bias.BULLISH, 0.85))
            levels.append(Level("premarket_low", today.premarket.low, Bias.BEARISH, 0.85))

        levels.extend(self._intraday_levels(context))
        return levels

    def _intraday_levels(self, context: DetectionContext) -> list[Level]:
        """Today's range so far, excluding the bar being judged.

        Excluding the last bar is essential: a bar is always inside a range that
        includes it, so a range computed over all of today can never be broken.
        """
        today = context.today
        if today is None or context.bars.empty:
            return []
        if context.intraday and context.bar_sessions.iloc[-1] is not MarketSession.REGULAR:
            # Before the bell or after it there is no intraday range to break.
            return []

        session_bars = context.regular_bars(today.day)
        if len(session_bars) < self.config.min_session_bars + 1:
            return []
        if session_bars.index[-1] != context.bars.index[-1]:
            return []

        prior = session_bars.iloc[:-1]
        high = float(prior["high"].max())
        low = float(prior["low"].min())
        if not np.isfinite(high) or not np.isfinite(low):
            return []
        return [
            Level("intraday_high", high, Bias.BULLISH, 0.7),
            Level("intraday_low", low, Bias.BEARISH, 0.7),
        ]
