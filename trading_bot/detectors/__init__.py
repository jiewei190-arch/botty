"""Signal detection (Phase 1).

Five independent detectors, one shared signal shape, one scoring engine::

    from trading_bot.detectors import DetectorEngine

    engine = DetectorEngine()
    analysis = engine.analyze("NVDA", bars)
    print(analysis.score.describe())

Each detector reads a :class:`~trading_bot.detectors.context.DetectionContext`
and returns at most one :class:`~trading_bot.detectors.models.DetectorSignal`.
None of them fetches data, stores anything, or can cause an order to exist.

Why ``detectors`` and not ``scanners``
--------------------------------------
:mod:`trading_bot.scanner` already exists and does something different: it takes
strategy signals, sizes them against a portfolio and ranks tradable candidates.
These are the raw observations *underneath* that — naming the package
``scanners`` would have put two packages one character apart in the same
codebase, each doing a different job. The distinction is worth the deviation from
the original layout sketch.
"""

from trading_bot.detectors.base import (
    DEFAULT_DETECTOR_CONFIG,
    BreakoutDetectorConfig,
    Detector,
    DetectorConfig,
    GapDetectorConfig,
    MomentumDetectorConfig,
    VolatilityDetectorConfig,
    VolumeDetectorConfig,
)
from trading_bot.detectors.breakout_detector import BreakoutDetector, Level
from trading_bot.detectors.context import (
    DEFAULT_HISTORY_SESSIONS,
    DetectionContext,
    Extent,
    SessionStats,
    build_context,
)
from trading_bot.detectors.engine import (
    DetectorEngine,
    SymbolAnalysis,
    build_detectors,
)
from trading_bot.detectors.gap_detector import GapDetector
from trading_bot.detectors.models import (
    MAX_STRENGTH,
    MIN_STRENGTH,
    Bias,
    DetectorSignal,
    SignalType,
    clamp_strength,
    scale_strength,
)
from trading_bot.detectors.momentum_detector import MomentumDetector, consecutive_run
from trading_bot.detectors.scoring import (
    DEFAULT_WEIGHTS,
    OpportunityScore,
    ScoringWeights,
    rank,
    score_signals,
)
from trading_bot.detectors.volatility_detector import VolatilityDetector
from trading_bot.detectors.volume_detector import VolumeDetector

__all__ = [
    "DEFAULT_DETECTOR_CONFIG",
    "DEFAULT_HISTORY_SESSIONS",
    "DEFAULT_WEIGHTS",
    "MAX_STRENGTH",
    "MIN_STRENGTH",
    "Bias",
    "BreakoutDetector",
    "BreakoutDetectorConfig",
    "DetectionContext",
    "Detector",
    "DetectorConfig",
    "DetectorEngine",
    "DetectorSignal",
    "Extent",
    "GapDetector",
    "GapDetectorConfig",
    "Level",
    "MomentumDetector",
    "MomentumDetectorConfig",
    "OpportunityScore",
    "ScoringWeights",
    "SessionStats",
    "SignalType",
    "SymbolAnalysis",
    "VolatilityDetector",
    "VolatilityDetectorConfig",
    "VolumeDetector",
    "VolumeDetectorConfig",
    "build_context",
    "build_detectors",
    "clamp_strength",
    "consecutive_run",
    "rank",
    "scale_strength",
    "score_signals",
]
