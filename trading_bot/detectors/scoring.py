"""Ranking detector signals.

The scoring engine answers one question — *of everything the detectors saw, what
deserves a human's attention first?* — and refuses to answer any other. It does
not decide whether to trade, what size, or where the stop goes. Those decisions
need an account, a risk budget and a strategy; a rank needs none of them, and
keeping the two apart is what stops a good-looking score from turning itself
into an order.

How the number is built
-----------------------
Each detector contributes its 0-10 strength. A silent detector contributes
**zero**, not "no opinion" — this is the one place where the project's usual
renormalise-around-missing-inputs rule is deliberately inverted. A detector
staying quiet is a measurement: nothing unusual happened. Dropping it and
rescaling would let a symbol with one loud reading outrank a symbol with four
independent confirmations, which is precisely backwards for a scanner whose
entire value is confluence.

Direction is a weighted vote, and disagreement is reported rather than hidden.
A symbol whose volume leans bullish while its breakout leans bearish is not a
0-strength opportunity — it is a *confused* one, and the caller should be able
to see that instead of receiving an average that quietly cancels it out.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from trading_bot.detectors.models import (
    MAX_STRENGTH,
    Bias,
    DetectorSignal,
    SignalType,
)
from trading_bot.utils.market_hours import MarketSession

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ScoringWeights:
    """How much each kind of evidence counts toward the overall score.

    The defaults lean on volume, momentum and breakouts because those three are
    what a discretionary trader actually scans for. Gaps rank lower because the
    information is usually already in the price by the time anyone can act on
    it; volatility lowest because it says something is happening without saying
    what, which is context rather than a reason.
    """

    unusual_volume: float = 0.25
    momentum: float = 0.25
    breakout: float = 0.25
    gap: float = 0.15
    volatility_expansion: float = 0.10

    def as_mapping(self) -> dict[SignalType, float]:
        return {
            SignalType.UNUSUAL_VOLUME: self.unusual_volume,
            SignalType.MOMENTUM: self.momentum,
            SignalType.BREAKOUT: self.breakout,
            SignalType.GAP: self.gap,
            SignalType.VOLATILITY_EXPANSION: self.volatility_expansion,
        }

    def weight(self, signal_type: SignalType) -> float:
        return self.as_mapping().get(signal_type, 0.0)

    @property
    def total(self) -> float:
        return sum(self.as_mapping().values())

    def __post_init__(self) -> None:
        for name, value in self.as_mapping().items():
            if value < 0:
                raise ValueError(f"Weight for {name.value} cannot be negative: {value}")
        if self.total <= 0:
            raise ValueError("At least one scoring weight must be greater than zero")


DEFAULT_WEIGHTS = ScoringWeights()


@dataclass(frozen=True, slots=True)
class OpportunityScore:
    """A ranked symbol and the evidence behind its rank."""

    symbol: str
    timestamp: datetime
    overall: float
    direction: Bias
    #: Strength per signal type, zero where the detector stayed quiet.
    components: Mapping[SignalType, float]
    signals: tuple[DetectorSignal, ...]
    #: 1.0 when every directional signal agrees, 0.0 when they exactly cancel.
    agreement: float
    session: MarketSession = MarketSession.CLOSED
    weights: ScoringWeights = field(default_factory=lambda: DEFAULT_WEIGHTS)

    @property
    def fired(self) -> tuple[DetectorSignal, ...]:
        """Signals that actually fired, strongest first."""
        return tuple(sorted(self.signals, key=lambda s: s.strength, reverse=True))

    @property
    def conflicted(self) -> bool:
        """True when directional signals materially disagree."""
        return self.agreement < 0.6 and len(self.signals) > 1

    def component(self, signal_type: SignalType) -> float:
        return float(self.components.get(signal_type, 0.0))

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "overall_score": round(self.overall, 2),
            "direction": self.direction.value,
            "agreement": round(self.agreement, 3),
            "conflicted": self.conflicted,
            "session": self.session.value,
            "components": {
                signal_type.value: round(float(value), 2)
                for signal_type, value in self.components.items()
            },
            "signals": [signal.as_dict() for signal in self.fired],
        }

    def describe(self) -> str:
        """The alert-style block from the Phase 1 specification."""
        lines = [
            f"SYMBOL: {self.symbol}",
            f"OVERALL SCORE: {self.overall:.1f}",
            f"DIRECTION: {self.direction.value}",
            "SIGNALS:",
        ]
        for signal in self.fired:
            lines.append(f"  * {signal.signal_type.value}: {signal.strength:.1f} — {signal.reason}")
        if not self.signals:
            lines.append("  (none)")
        if self.conflicted:
            lines.append(f"  ! signals disagree on direction (agreement {self.agreement:.0%})")
        return "\n".join(lines)


def score_signals(
    symbol: str,
    signals: Iterable[DetectorSignal],
    *,
    weights: ScoringWeights | None = None,
    timestamp: datetime | None = None,
    session: MarketSession | None = None,
) -> OpportunityScore:
    """Combine one symbol's detector signals into a ranked score.

    Duplicate signal types are resolved by keeping the strongest, so a caller
    that runs two configurations of the same detector cannot double-count.
    """
    active = weights or DEFAULT_WEIGHTS
    collected: dict[SignalType, DetectorSignal] = {}
    for signal in signals:
        existing = collected.get(signal.signal_type)
        if existing is None or signal.strength > existing.strength:
            collected[signal.signal_type] = signal

    components = {
        signal_type: collected[signal_type].strength if signal_type in collected else 0.0
        for signal_type in active.as_mapping()
    }
    for signal_type, signal in collected.items():
        components.setdefault(signal_type, signal.strength)

    weighted = sum(active.weight(t) * value for t, value in components.items())
    overall = weighted / active.total if active.total > 0 else 0.0

    directed = [
        (active.weight(signal.signal_type) * signal.strength, signal.direction.sign)
        for signal in collected.values()
        if signal.direction.sign != 0
    ]
    gross = sum(abs(magnitude) for magnitude, _ in directed)
    net = sum(magnitude * sign for magnitude, sign in directed)
    # Nothing directional to disagree about counts as full agreement rather than
    # none: a lone volatility-expansion reading is neutral, not contradictory.
    agreement = abs(net) / gross if gross > 0 else 1.0

    stamps = [signal.timestamp for signal in collected.values()]
    when = timestamp or (max(stamps) if stamps else datetime.now(timezone.utc))
    where = session or next(
        (signal.session for signal in collected.values()), MarketSession.CLOSED
    )

    return OpportunityScore(
        symbol=str(symbol).strip().upper(),
        timestamp=when,
        overall=min(max(overall, 0.0), MAX_STRENGTH),
        direction=Bias.from_change(net),
        components=components,
        signals=tuple(collected.values()),
        agreement=float(min(max(agreement, 0.0), 1.0)),
        session=where,
        weights=active,
    )


def rank(scores: Iterable[OpportunityScore]) -> tuple[OpportunityScore, ...]:
    """Highest score first; ties broken by how many detectors fired.

    The tiebreak matters more than it looks: two symbols at 4.2 where one got
    there on a single reading and the other on three is not a tie in any sense a
    trader cares about.
    """
    return tuple(
        sorted(scores, key=lambda s: (s.overall, len(s.signals), s.agreement), reverse=True)
    )
