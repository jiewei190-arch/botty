"""Running the detectors over symbols.

The engine is deliberately thin: build the context once per symbol, hand it to
each detector, collect what comes back, score it. Everything interesting lives
in the detectors and the scoring rules; this file's only real responsibility is
that **one broken detector cannot take down a scan**.

That matters more than it sounds. A scanner is a long-running process watching
dozens of symbols, and a single malformed frame — one symbol with a zero price,
one day with no bars — must not end the session for the other twenty-nine. Every
detector call is therefore isolated, and a failure is logged loudly, counted, and
skipped rather than raised.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from trading_bot.detectors.base import DEFAULT_DETECTOR_CONFIG, Detector, DetectorConfig
from trading_bot.detectors.breakout_detector import BreakoutDetector
from trading_bot.detectors.context import (
    DEFAULT_HISTORY_SESSIONS,
    DetectionContext,
    build_context,
)
from trading_bot.detectors.gap_detector import GapDetector
from trading_bot.detectors.models import DetectorSignal
from trading_bot.detectors.momentum_detector import MomentumDetector
from trading_bot.detectors.scoring import (
    DEFAULT_WEIGHTS,
    OpportunityScore,
    ScoringWeights,
    rank,
    score_signals,
)
from trading_bot.detectors.volatility_detector import VolatilityDetector
from trading_bot.detectors.volume_detector import VolumeDetector

logger = logging.getLogger(__name__)


def build_detectors(config: DetectorConfig | None = None) -> tuple[Detector, ...]:
    """The standard suite, in the order their results are reported."""
    settings = config or DEFAULT_DETECTOR_CONFIG
    return (
        VolumeDetector(settings.volume),
        MomentumDetector(settings.momentum),
        BreakoutDetector(settings.breakout),
        GapDetector(settings.gap),
        VolatilityDetector(settings.volatility),
    )


@dataclass(frozen=True, slots=True)
class SymbolAnalysis:
    """What the engine learned about one symbol."""

    symbol: str
    context: DetectionContext
    signals: tuple[DetectorSignal, ...]
    score: OpportunityScore
    #: Detector names that raised. Empty in normal operation.
    failures: tuple[str, ...] = field(default_factory=tuple)

    @property
    def overall(self) -> float:
        return self.score.overall

    @property
    def fired(self) -> bool:
        return bool(self.signals)


class DetectorEngine:
    """Owns the detector suite and the scoring weights."""

    def __init__(
        self,
        config: DetectorConfig | None = None,
        *,
        weights: ScoringWeights | None = None,
        detectors: Sequence[Detector] | None = None,
        history_sessions: int = DEFAULT_HISTORY_SESSIONS,
    ) -> None:
        self.config = config or DEFAULT_DETECTOR_CONFIG
        self.weights = weights or DEFAULT_WEIGHTS
        self.detectors: tuple[Detector, ...] = tuple(detectors or build_detectors(self.config))
        self.history_sessions = history_sessions

    @property
    def detector_names(self) -> tuple[str, ...]:
        return tuple(detector.name for detector in self.detectors)

    def analyze(
        self,
        symbol: str,
        bars: pd.DataFrame,
        *,
        now: datetime | None = None,
    ) -> SymbolAnalysis:
        """Run every detector over one symbol's bars and score the result."""
        context = build_context(
            symbol, bars, now=now, history_sessions=self.history_sessions
        )
        signals: list[DetectorSignal] = []
        failures: list[str] = []

        for detector in self.detectors:
            try:
                signal = detector.evaluate(context)
            except Exception:  # noqa: BLE001 - one detector must not end the scan
                logger.exception(
                    "Detector %s failed on %s; skipping it for this bar",
                    detector.name,
                    context.symbol,
                )
                failures.append(detector.name)
                continue
            if signal is not None:
                signals.append(signal)

        score = score_signals(
            context.symbol,
            signals,
            weights=self.weights,
            timestamp=context.last_timestamp or context.now,
            session=context.session,
        )
        return SymbolAnalysis(
            symbol=context.symbol,
            context=context,
            signals=tuple(signals),
            score=score,
            failures=tuple(failures),
        )

    def analyze_many(
        self,
        frames: Mapping[str, pd.DataFrame],
        *,
        now: datetime | None = None,
    ) -> tuple[SymbolAnalysis, ...]:
        """Analyse several symbols, ranked best first.

        A symbol whose frame is missing or empty still produces an analysis with
        a zero score, so callers can tell "watched and quiet" from "not watched".
        """
        results = [
            self.analyze(symbol, frame if frame is not None else pd.DataFrame(), now=now)
            for symbol, frame in frames.items()
        ]
        order = {score.symbol: index for index, score in enumerate(rank(r.score for r in results))}
        return tuple(sorted(results, key=lambda r: order.get(r.symbol, len(order))))

    def signals_from(self, analyses: Iterable[SymbolAnalysis]) -> tuple[DetectorSignal, ...]:
        """Flatten analyses into a single strongest-first signal list."""
        collected = [signal for analysis in analyses for signal in analysis.signals]
        return tuple(sorted(collected, key=lambda s: s.strength, reverse=True))
