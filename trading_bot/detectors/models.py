"""The standardized detector signal.

Every detector answers the same shape of question — "is something unusual
happening in this symbol, which way does it lean, and how strongly?" — so every
detector returns the same object. That uniformity is what lets the scoring
engine combine five unrelated pieces of evidence without knowing anything about
how any of them was computed.

Why 0-10 for strength
---------------------
The scale is fixed and shared so that a 9 from the volume detector means the
same amount of "unusual" as a 9 from the gap detector. A detector converts its
own natural units (a volume ratio, a percentage gap, an ATR multiple) into that
shared scale through an explicit, documented mapping. Anything the detector
measured is preserved verbatim in ``metrics``, so nothing is lost to the
squeeze — the raw numbers stay auditable.

Strength is *not* a probability and *not* a recommendation. It says how far from
normal the observation is, nothing more.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from trading_bot.utils.market_hours import MarketSession

#: Strength runs 0-10 inclusive.
MIN_STRENGTH = 0.0
MAX_STRENGTH = 10.0


class SignalType(str, Enum):
    """What kind of unusual behaviour was observed."""

    UNUSUAL_VOLUME = "UNUSUAL_VOLUME"
    MOMENTUM = "MOMENTUM"
    BREAKOUT = "BREAKOUT"
    GAP = "GAP"
    VOLATILITY_EXPANSION = "VOLATILITY_EXPANSION"


class Bias(str, Enum):
    """Which way an observation leans.

    Deliberately *not* :class:`trading_bot.strategies.SignalDirection`, which is
    LONG/SHORT — an instruction about a position. A detector never instructs. It
    reports that the evidence points up, down, or neither; whether that becomes
    a long, a short or nothing at all is decided much later.
    """

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"

    @property
    def sign(self) -> int:
        if self is Bias.BULLISH:
            return 1
        if self is Bias.BEARISH:
            return -1
        return 0

    @property
    def opposite(self) -> Bias:
        if self is Bias.BULLISH:
            return Bias.BEARISH
        if self is Bias.BEARISH:
            return Bias.BULLISH
        return Bias.NEUTRAL

    @classmethod
    def from_change(cls, change: float, *, deadband: float = 0.0) -> Bias:
        """Bias implied by a signed number, with a neutral band around zero."""
        if not math.isfinite(change) or abs(change) <= deadband:
            return cls.NEUTRAL
        return cls.BULLISH if change > 0 else cls.BEARISH


def clamp_strength(value: float) -> float:
    """Force a raw score into the 0-10 band.

    A non-finite score becomes 0 rather than propagating a NaN into the scoring
    engine, where it would poison a weighted average silently.
    """
    if not math.isfinite(value):
        return MIN_STRENGTH
    return max(MIN_STRENGTH, min(MAX_STRENGTH, float(value)))


def scale_strength(value: float, *, floor: float, ceiling: float) -> float:
    """Map ``value`` linearly from ``[floor, ceiling]`` onto 0-10.

    ``floor`` is the threshold at which an observation stops being ordinary, so
    it maps to 0, not to some arbitrary minimum interest level. ``ceiling`` is
    the point past which more is not meaningfully more — a volume ratio of 40x
    and one of 400x are both simply "enormous", and letting the second dominate
    a weighted score would rank a data error above a real setup.
    """
    if not math.isfinite(value) or ceiling <= floor:
        return MIN_STRENGTH
    return clamp_strength((value - floor) / (ceiling - floor) * MAX_STRENGTH)


@dataclass(frozen=True, slots=True)
class DetectorSignal:
    """One detector's observation about one symbol at one moment."""

    symbol: str
    timestamp: datetime
    signal_type: SignalType
    direction: Bias
    strength: float
    reason: str
    metrics: dict[str, float] = field(default_factory=dict)
    session: MarketSession = MarketSession.CLOSED
    detector: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", str(self.symbol).strip().upper())
        object.__setattr__(self, "strength", clamp_strength(self.strength))
        stamp = self.timestamp
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        object.__setattr__(self, "timestamp", stamp.astimezone(timezone.utc))
        # Coerce numpy scalars to plain floats. They compare and print the same,
        # but json.dumps cannot serialise them, so a numpy value surviving to
        # here would be silently stringified on its way into the database.
        object.__setattr__(
            self,
            "metrics",
            {str(key): float(value) for key, value in dict(self.metrics).items()},
        )

    @property
    def key(self) -> str:
        """Identity for de-duplication.

        Symbol, kind and direction — deliberately *not* the timestamp or the
        strength. Two consecutive bars both reporting a bullish volume spike are
        the same event being observed twice, and alerting on both is noise. The
        alert manager pairs this key with a cooldown to decide what is new.
        """
        return f"{self.symbol}:{self.signal_type.value}:{self.direction.value}"

    def metric(self, name: str, default: float | None = None) -> float | None:
        value = self.metrics.get(name, default)
        return None if value is None else float(value)

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form. Matches the documented signal schema."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "signal_type": self.signal_type.value,
            "direction": self.direction.value,
            "strength": round(self.strength, 2),
            "metrics": {k: round(float(v), 6) for k, v in self.metrics.items()},
            "reason": self.reason,
            "session": self.session.value,
            "detector": self.detector,
        }

    def describe(self) -> str:
        return (
            f"{self.symbol} {self.signal_type.value} {self.direction.value} "
            f"{self.strength:.1f}/10 — {self.reason}"
        )
