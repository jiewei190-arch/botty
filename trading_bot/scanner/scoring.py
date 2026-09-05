"""Opportunity scoring.

Why this exists
---------------
A strategy's own confidence is **not comparable across strategies**. Momentum
reporting 80 means "80% of momentum's evidence is present"; mean reversion
reporting 80 means 80% of a completely different checklist. Sorting a mixed list
by those numbers ranks the checklists, not the opportunities.

So the scanner scores every candidate again on a **common yardstick**, measured
from the market rather than from the strategy that found it. Seven factors, each
0-100, each answering the same question: *how much does this support a trade in
this direction?*

===============  ======  ==================================================
Factor           Weight  Reads
===============  ======  ==================================================
Trend             0.20   Trend direction and strength, damped by agreement
Risk / reward     0.17   What the setup pays for what it risks
Momentum          0.15   MACD state and whether it is building
Conviction        0.15   The strategy's own confidence
Volume            0.13   Participation, and whether it confirms the move
Structure         0.10   Room to the nearest level in the way
RSI headroom      0.10   How far from exhaustion the move is
===============  ======  ==================================================

Every factor is **direction-aware**: a strongly bullish trend scores near 100 for
a long and near 0 for a short. Factors without data are dropped and the remaining
weights renormalised, so a short history lowers the number of inputs rather than
silently scoring a missing factor as zero.

The result is a 0-100 **trade confidence score**. It ranks opportunities against
each other. It is not a probability of profit, and nothing here should be read as
one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from trading_bot.indicators import (
    IndicatorConfig,
    SupportResistance,
    TrendAnalysis,
    VolumeAnalysis,
    analyze_trend,
    analyze_volume,
    atr_column,
    detect_macd_momentum,
    detect_macd_signal,
    find_support_resistance,
    rsi_column,
)
from trading_bot.strategies import Signal, SignalDirection

logger = logging.getLogger(__name__)

#: Factor weights. They need not sum to 1 — scores are normalised by the weight
#: of the factors that actually had data.
DEFAULT_WEIGHTS: dict[str, float] = {
    "trend": 0.20,
    "risk_reward": 0.17,
    "momentum": 0.15,
    "conviction": 0.15,
    "volume": 0.13,
    "structure": 0.10,
    "rsi_headroom": 0.10,
}

#: Reward:risk that scores 100. Anything beyond is equally good for ranking.
_RR_SATURATION = 4.0

#: Room to the opposing level, in ATRs, that scores 100.
_STRUCTURE_SATURATION_ATR = 3.0


@dataclass(frozen=True, slots=True)
class FactorScore:
    """One scored input to a trade confidence score."""

    name: str
    score: float          # 0-100
    weight: float
    detail: str

    @property
    def contribution(self) -> float:
        return self.score * self.weight

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 1),
            "weight": self.weight,
            "detail": self.detail,
        }


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _latest(data: pd.DataFrame, column: str) -> float | None:
    if column not in data.columns:
        return None
    value = data[column].iloc[-1]
    return None if pd.isna(value) else float(value)


# -- individual factors --------------------------------------------------------


def score_trend(
    trend: TrendAnalysis, direction: SignalDirection
) -> FactorScore:
    """How strongly the trend backs this direction.

    ``strength`` is already a 0-100 directional scale, so a long reads it
    directly and a short reads its mirror. The result is then damped toward
    neutral by the trend model's own agreement: components that disagreed should
    not produce a confident score in either direction.
    """
    raw = trend.strength if direction is SignalDirection.LONG else 100 - trend.strength
    damped = 50 + (raw - 50) * (trend.confidence / 100)
    return FactorScore(
        "trend",
        _clamp(damped),
        DEFAULT_WEIGHTS["trend"],
        f"{trend.direction.value} at {trend.strength}/100 (agreement {trend.confidence})",
    )


def score_momentum(
    data: pd.DataFrame, direction: SignalDirection, config: IndicatorConfig
) -> FactorScore | None:
    """MACD state, refined by whether momentum is building or fading."""
    state = detect_macd_signal(data, config)
    if state == "NEUTRAL":
        return None

    building = detect_macd_momentum(data, config)
    with_trade = state == ("BULLISH" if direction is SignalDirection.LONG else "BEARISH")
    expanding = building == ("INCREASING" if direction is SignalDirection.LONG else "DECREASING")

    # (agrees with the trade, momentum expanding) -> score
    score = {
        (True, True): 100.0,    # with the trade and building
        (True, False): 70.0,    # with the trade but fading
        (False, True): 30.0,    # against the trade but easing
        (False, False): 5.0,    # against the trade and accelerating
    }[(with_trade, expanding)]

    return FactorScore(
        "momentum",
        score,
        DEFAULT_WEIGHTS["momentum"],
        f"MACD {state.lower()}, histogram {building.lower()}",
    )


def score_volume(volume: VolumeAnalysis) -> FactorScore | None:
    """Participation behind the move.

    Direction-neutral — heavy volume supports whatever is happening — but a bar
    whose volume confirms its own price move earns a bonus.
    """
    relative = volume.relative_volume
    if relative is None:
        return None

    if relative >= 2.5:
        score = 100.0
    elif relative >= 1.5:
        score = 80.0 + (relative - 1.5) * 20
    elif relative >= 1.0:
        score = 55.0 + (relative - 1.0) * 50
    elif relative >= 0.7:
        score = 30.0 + (relative - 0.7) * 83
    else:
        score = _clamp(relative / 0.7 * 30)

    if volume.confirms_price:
        score = _clamp(score + 8)

    return FactorScore(
        "volume",
        _clamp(score),
        DEFAULT_WEIGHTS["volume"],
        f"{relative:.2f}x average ({volume.condition.value.lower()})"
        + (", confirming the move" if volume.confirms_price else ""),
    )


def score_rsi_headroom(
    data: pd.DataFrame, direction: SignalDirection, config: IndicatorConfig
) -> FactorScore | None:
    """How much room is left before the move is exhausted.

    This measures headroom, not whether RSI is "good". Being oversold is not
    penalised for a long — the strategy already decided that was the entry. What
    is penalised is buying into an already-extended move, or shorting one that
    has already collapsed.
    """
    rsi = _latest(data, rsi_column(config.rsi_period))
    if rsi is None:
        return None

    if direction is SignalDirection.LONG:
        # Full marks up to 65, fading to zero by 90.
        score = 100.0 if rsi <= 65 else _clamp(100 - (rsi - 65) / 25 * 100)
        detail = f"RSI {rsi:.1f}" + (" — extended" if rsi > 75 else " — room to run")
    else:
        score = 100.0 if rsi >= 35 else _clamp(100 - (35 - rsi) / 25 * 100)
        detail = f"RSI {rsi:.1f}" + (" — extended" if rsi < 25 else " — room to fall")

    return FactorScore("rsi_headroom", score, DEFAULT_WEIGHTS["rsi_headroom"], detail)


def score_risk_reward(signal: Signal) -> FactorScore:
    """What the setup pays for what it risks.

    Saturates at 4:1 — beyond that the difference stops being meaningful for
    ranking, and rewarding it further would favour setups whose targets are
    simply unrealistic.
    """
    ratio = signal.risk_reward_ratio
    score = _clamp((ratio - 1.0) / (_RR_SATURATION - 1.0) * 100)
    return FactorScore(
        "risk_reward",
        score,
        DEFAULT_WEIGHTS["risk_reward"],
        f"1:{ratio:.2f} reward to risk",
    )


def score_structure(
    levels: SupportResistance,
    signal: Signal,
    atr: float | None,
) -> FactorScore | None:
    """Room to the nearest level standing in the trade's way.

    A long into resistance a few cents overhead has nowhere to go, however good
    everything else looks.
    """
    if atr is None or atr <= 0:
        return None

    is_long = signal.direction is SignalDirection.LONG
    blocking = levels.nearest_resistance if is_long else levels.nearest_support
    if blocking is None:
        return FactorScore(
            "structure",
            85.0,
            DEFAULT_WEIGHTS["structure"],
            "No level in the way within the lookback",
        )

    room_atr = abs(blocking.price - signal.entry_price) / atr
    score = _clamp(room_atr / _STRUCTURE_SATURATION_ATR * 100)
    label = "resistance" if is_long else "support"
    return FactorScore(
        "structure",
        score,
        DEFAULT_WEIGHTS["structure"],
        f"{room_atr:.1f} ATR of room to {label} at {blocking.price:,.2f}",
    )


def score_conviction(signal: Signal) -> FactorScore:
    """The strategy's own confidence.

    Weighted modestly on purpose: it says how complete *that strategy's*
    evidence was, which is exactly the number that does not compare across
    strategies. It still carries information — a strategy firing at 95 found more
    of what it looks for than one firing at 60.
    """
    return FactorScore(
        "conviction",
        _clamp(signal.confidence),
        DEFAULT_WEIGHTS["conviction"],
        f"{signal.strategy} fired at {signal.confidence:.0f}/100",
    )


# -- composite -----------------------------------------------------------------


def score_opportunity(
    signal: Signal,
    data: pd.DataFrame,
    config: IndicatorConfig | None = None,
    *,
    trend: TrendAnalysis | None = None,
    volume: VolumeAnalysis | None = None,
    levels: SupportResistance | None = None,
) -> tuple[float, tuple[FactorScore, ...]]:
    """Score a signal on the common yardstick.

    Parameters
    ----------
    signal:
        The candidate.
    data:
        The indicator-enriched frame it was found in.
    config:
        Indicator configuration.
    trend, volume, levels:
        Precomputed analyses. Supplied by the scanner so they are not recomputed
        once per strategy for the same symbol.

    Returns
    -------
    tuple
        ``(confidence, factors)`` — the 0-100 score and every factor behind it.

    Example
    -------
    >>> confidence, factors = score_opportunity(signal, enriched)
    >>> round(confidence)
    78
    """
    settings = config or IndicatorConfig()
    trend = trend if trend is not None else analyze_trend(data, settings)
    volume = volume if volume is not None else analyze_volume(data, settings)
    levels = levels if levels is not None else find_support_resistance(data, settings)
    atr = _latest(data, atr_column(settings.atr_period))

    candidates = [
        score_trend(trend, signal.direction),
        score_risk_reward(signal),
        score_momentum(data, signal.direction, settings),
        score_conviction(signal),
        score_volume(volume),
        score_structure(levels, signal, atr),
        score_rsi_headroom(data, signal.direction, settings),
    ]
    factors = tuple(factor for factor in candidates if factor is not None)

    if not factors:
        return 0.0, ()

    # Renormalise over the factors that had data, so a missing input reduces the
    # evidence rather than scoring zero and dragging the total down.
    total_weight = sum(factor.weight for factor in factors)
    confidence = sum(factor.contribution for factor in factors) / total_weight
    return _clamp(confidence), factors
