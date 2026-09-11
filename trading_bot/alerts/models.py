"""What an alert is, and how it reads.

An alert is a *decision to interrupt someone*. That framing drives everything
here: it carries the price and the session because an alert read twenty minutes
later without them is useless, it names the individual signals rather than only
the score because "8.7" tells a trader nothing actionable, and it has a priority
because a scanner that shouts equally about everything trains its reader to
ignore it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from trading_bot.detectors.models import Bias, DetectorSignal
from trading_bot.detectors.scoring import OpportunityScore
from trading_bot.utils.market_hours import MarketSession


class AlertPriority(str, Enum):
    """How loudly to say it."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

    @classmethod
    def from_score(cls, score: float, *, threshold: float) -> AlertPriority:
        """Priority relative to the threshold that let the alert through.

        Anchored to the threshold rather than to fixed score bands, because a
        user who lowered the threshold to 3 wants their 3s labelled relative to
        what they asked for, not permanently branded LOW.
        """
        if score >= threshold * 1.5:
            return cls.HIGH
        if score >= threshold * 1.2:
            return cls.MEDIUM
        return cls.LOW


#: Human-readable names for signal types, used in the alert body.
SIGNAL_LABELS: dict[str, str] = {
    "UNUSUAL_VOLUME": "Relative Volume Spike",
    "MOMENTUM": "Strong Momentum",
    "BREAKOUT": "Level Breakout",
    "GAP": "Overnight Gap",
    "VOLATILITY_EXPANSION": "Volatility Expansion",
}


@dataclass(frozen=True, slots=True)
class Alert:
    """One notification about one symbol."""

    symbol: str
    timestamp: datetime
    direction: Bias
    score: float
    priority: AlertPriority
    session: MarketSession
    signals: tuple[DetectorSignal, ...]
    price: float | None = None
    #: The de-duplication identity. Repeats within the cooldown share this.
    key: str = ""
    #: Why this alert was allowed through when a similar one was suppressed.
    trigger: str = "new"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", str(self.symbol).strip().upper())
        if self.timestamp.tzinfo is None:
            object.__setattr__(self, "timestamp", self.timestamp.replace(tzinfo=timezone.utc))
        if not self.key:
            object.__setattr__(self, "key", f"{self.symbol}:{self.direction.value}")

    @property
    def signal_types(self) -> tuple[str, ...]:
        return tuple(signal.signal_type.value for signal in self.signals)

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "direction": self.direction.value,
            "score": round(float(self.score), 2),
            "priority": self.priority.value,
            "session": self.session.value,
            "price": self.price,
            "key": self.key,
            "trigger": self.trigger,
            "signals": [signal.as_dict() for signal in self.signals],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_score(
        cls,
        score: OpportunityScore,
        *,
        threshold: float,
        price: float | None = None,
        trigger: str = "new",
        metadata: dict[str, Any] | None = None,
    ) -> Alert:
        return cls(
            symbol=score.symbol,
            timestamp=score.timestamp,
            direction=score.direction,
            score=score.overall,
            priority=AlertPriority.from_score(score.overall, threshold=threshold),
            session=score.session,
            signals=score.fired,
            price=price,
            trigger=trigger,
            metadata=metadata or {},
        )


def render_alert(alert: Alert) -> str:
    """The console block from the Phase 1 specification."""
    price = f"${alert.price:,.2f}" if alert.price else "n/a"
    lines = [
        f"🚨 {alert.priority.value} PRIORITY MARKET SETUP",
        "",
        f"Symbol: {alert.symbol}",
        f"Price: {price}",
        f"Direction: {alert.direction.value}",
        f"Overall Score: {alert.score:.1f}",
        "",
        "Signals:",
    ]
    for signal in alert.signals:
        label = SIGNAL_LABELS.get(signal.signal_type.value, signal.signal_type.value)
        lines.append(f"✓ {label} ({signal.strength:.1f}) — {signal.reason}")
    if not alert.signals:
        lines.append("(no individual signals recorded)")
    lines += [
        "",
        f"Timestamp: {alert.timestamp.isoformat()}",
        f"Market Session: {alert.session.value}",
    ]
    if alert.trigger != "new":
        lines.append(f"Trigger: {alert.trigger}")
    return "\n".join(lines)
