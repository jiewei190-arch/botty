"""The detector contract and its thresholds.

A detector reads a :class:`~trading_bot.detectors.context.DetectionContext` and
returns at most one :class:`~trading_bot.detectors.models.DetectorSignal`. It
does not fetch data, does not store anything, does not know what a portfolio is,
and cannot cause an order to exist. That isolation is what makes each one
testable from a hand-built frame in a few lines.

Thresholds live in the config objects below rather than in the detector bodies,
so that tuning a scanner never means editing logic. Every default is stated with
the reasoning behind it; a number in a trading system that nobody can justify is
a number nobody can safely change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

from trading_bot.detectors.context import DetectionContext
from trading_bot.detectors.models import Bias, DetectorSignal, SignalType


@dataclass(frozen=True, slots=True)
class VolumeDetectorConfig:
    """Thresholds for unusual volume."""

    #: Relative volume at which participation stops being ordinary. 1.5x is the
    #: conventional line: below it, normal day-to-day variation explains the
    #: reading without needing a catalyst.
    threshold: float = 1.5
    #: Relative volume mapped to full strength. Past ~5x the distinction between
    #: "huge" and "huger" carries no extra information for ranking.
    saturation: float = 5.0
    #: Sessions of history needed before a baseline means anything.
    min_sessions: int = 3
    #: Bars averaged for the single-bar relative volume reading.
    bar_lookback: int = 20
    #: Price move (percent) below which a volume spike is called directionless.
    direction_deadband_pct: float = 0.15


@dataclass(frozen=True, slots=True)
class MomentumDetectorConfig:
    """Thresholds for price momentum."""

    #: Bars the move is measured over.
    lookback_bars: int = 5
    #: Move size in ATR units that counts as strong. Expressing the threshold in
    #: ATR rather than percent is what makes one setting work for both a $9 stock
    #: and a $900 one.
    atr_threshold: float = 1.5
    #: ATR multiple mapped to full strength.
    atr_saturation: float = 4.0
    #: A move this large in percent qualifies even when ATR is unavailable.
    percent_threshold: float = 2.0
    #: Consecutive same-direction bars that count as a run worth noting.
    run_threshold: int = 3
    #: Weight of the consecutive-run evidence in the final strength (0-1).
    run_weight: float = 0.25
    atr_period: int = 14


@dataclass(frozen=True, slots=True)
class BreakoutDetectorConfig:
    """Thresholds for level breaks."""

    #: How far past a level price must close before the break is real, as a
    #: percentage. Without a buffer, every tick through a level is a "breakout"
    #: and the scanner reports noise all day.
    buffer_pct: float = 0.05
    #: Distance past the level mapped to full strength.
    saturation_pct: float = 1.5
    #: Minimum bars in today's session before an intraday-range break counts;
    #: the first few bars of a day have no meaningful range to break out of.
    min_session_bars: int = 6
    #: Require the level to have been intact on the previous bar. Keeps the
    #: signal to the moment of the break rather than repeating for hours.
    require_fresh: bool = True


@dataclass(frozen=True, slots=True)
class GapDetectorConfig:
    """Thresholds for overnight gaps."""

    #: Percentage gap that counts. Under 1% is ordinary overnight drift.
    threshold_pct: float = 1.0
    #: Gap size mapped to full strength.
    saturation_pct: float = 6.0


@dataclass(frozen=True, slots=True)
class VolatilityDetectorConfig:
    """Thresholds for volatility expansion."""

    #: Bars in the "now" volatility window.
    fast_period: int = 5
    #: Bars in the baseline window it is compared against.
    slow_period: int = 30
    #: Expansion ratio that counts as abnormal.
    threshold: float = 1.6
    #: Ratio mapped to full strength.
    saturation: float = 4.0
    #: Price move (percent) below which the expansion is called directionless.
    direction_deadband_pct: float = 0.25


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    """Every detector's thresholds in one object."""

    volume: VolumeDetectorConfig = field(default_factory=VolumeDetectorConfig)
    momentum: MomentumDetectorConfig = field(default_factory=MomentumDetectorConfig)
    breakout: BreakoutDetectorConfig = field(default_factory=BreakoutDetectorConfig)
    gap: GapDetectorConfig = field(default_factory=GapDetectorConfig)
    volatility: VolatilityDetectorConfig = field(default_factory=VolatilityDetectorConfig)


DEFAULT_DETECTOR_CONFIG = DetectorConfig()


class Detector(ABC):
    """One independent observation about a symbol."""

    #: Stable identifier used in logs, database rows and alert payloads.
    name: ClassVar[str] = "detector"
    #: The kind of signal this detector emits.
    signal_type: ClassVar[SignalType]

    @abstractmethod
    def evaluate(self, context: DetectionContext) -> DetectorSignal | None:
        """Return a signal, or None when nothing unusual is happening.

        Returning None is the common case and is not a failure. A detector that
        fires on every bar has a threshold problem, not a sensitivity advantage.
        """

    def _signal(
        self,
        context: DetectionContext,
        *,
        direction: Bias,
        strength: float,
        reason: str,
        metrics: dict[str, float],
    ) -> DetectorSignal:
        """Build a signal stamped with this detector's identity and the context."""
        return DetectorSignal(
            symbol=context.symbol,
            timestamp=context.last_timestamp or context.now,
            signal_type=self.signal_type,
            direction=direction,
            strength=strength,
            reason=reason,
            metrics=metrics,
            session=context.session,
            detector=self.name,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(name={self.name!r})"
