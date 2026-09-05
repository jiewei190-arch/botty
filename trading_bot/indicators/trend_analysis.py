"""Trend classification.

Turns indicator columns into a single answer: which way is this market leaning,
how strongly, and *why*. The "why" is not decoration — a signal the bot cannot
explain is a signal you cannot debug, and Phase 5's confidence score is built
from these components.

Scoring model
-------------
Five independent components each produce a score in ``[-1, +1]`` (negative is
bearish) and carry a weight:

===========================  ======  ===================================
Component                    Weight  Reads
===========================  ======  ===================================
EMA alignment                 0.25   Ordering of the configured EMAs
Price vs moving averages      0.20   Where price sits relative to them
Market structure              0.20   Higher highs / lower lows
MACD momentum                 0.20   Histogram sign and direction
Trend slope                   0.15   EMA slope, normalised by ATR
===========================  ======  ===================================

Components with insufficient data are dropped and the remaining weights are
renormalised, so a short history yields a weaker *confidence* rather than a
silently wrong answer.

The weighted sum maps to ``strength`` on a **directional 0-100 scale**: 0 is
maximally bearish, 50 neutral, 100 maximally bullish. ``confidence`` is separate
and reports how much the components agreed and how many had data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from trading_bot.indicators.price_action import detect_market_structure
from trading_bot.indicators.technical_indicators import (
    DEFAULT_CONFIG,
    MACD_HISTOGRAM_COL,
    IndicatorConfig,
    atr_column,
    calculate_atr,
    calculate_ema,
    calculate_macd,
    ema_column,
    validate_ohlcv,
)

logger = logging.getLogger(__name__)


class TrendDirection(str, Enum):
    """Trend labels, ordered from most bearish to most bullish."""

    STRONG_BEARISH = "STRONG_BEARISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    BULLISH = "BULLISH"
    STRONG_BULLISH = "STRONG_BULLISH"

    @property
    def is_bullish(self) -> bool:
        return self in (TrendDirection.BULLISH, TrendDirection.STRONG_BULLISH)

    @property
    def is_bearish(self) -> bool:
        return self in (TrendDirection.BEARISH, TrendDirection.STRONG_BEARISH)


#: Strength thresholds, checked from the top down.
_DIRECTION_THRESHOLDS: tuple[tuple[float, TrendDirection], ...] = (
    (75.0, TrendDirection.STRONG_BULLISH),
    (60.0, TrendDirection.BULLISH),
    (40.0, TrendDirection.NEUTRAL),
    (25.0, TrendDirection.BEARISH),
)


@dataclass(frozen=True, slots=True)
class TrendComponent:
    """One scored input to the trend verdict."""

    name: str
    score: float          # [-1, +1]
    weight: float
    detail: str

    @property
    def contribution(self) -> float:
        return self.score * self.weight


@dataclass(frozen=True, slots=True)
class TrendAnalysis:
    """Result of :func:`analyze_trend`."""

    direction: TrendDirection
    #: Directional score: 0 maximally bearish, 50 neutral, 100 maximally bullish.
    strength: int
    #: 0-100. How much the components agreed and how many had data.
    confidence: int
    reasons: tuple[str, ...] = ()
    components: tuple[TrendComponent, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialisable summary.

        Example
        -------
        >>> analyze_trend(enriched).as_dict()
        {'direction': 'BULLISH', 'strength': 78, 'confidence': 82, 'reasons': [...]}
        """
        return {
            "direction": self.direction.value,
            "strength": self.strength,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "components": {
                component.name: round(component.score, 3) for component in self.components
            },
        }


def _series_for(data: pd.DataFrame, column: str, fallback) -> pd.Series:
    """Reuse a precomputed column when present, otherwise compute it."""
    if column in data.columns:
        return data[column]
    return fallback()


def _score_ema_alignment(
    data: pd.DataFrame, config: IndicatorConfig
) -> tuple[TrendComponent | None, list[str]]:
    """Score how cleanly the EMAs are stacked shortest-to-longest."""
    periods = sorted(config.ema_periods)
    if len(periods) < 2:
        return None, []

    values: list[tuple[int, float]] = []
    for period in periods:
        series = _series_for(
            data, ema_column(period), lambda p=period: calculate_ema(data["close"], p)
        )
        value = series.iloc[-1] if len(series) else np.nan
        if not pd.isna(value):
            values.append((period, float(value)))

    if len(values) < 2:
        return None, []

    pairs = list(zip(values, values[1:], strict=False))
    bullish = sum(1 for (_, fast), (_, slow) in pairs if fast > slow)
    bearish = sum(1 for (_, fast), (_, slow) in pairs if fast < slow)
    score = (bullish - bearish) / len(pairs)

    reasons: list[str] = []
    if bullish == len(pairs):
        reasons.append(
            "Bullish EMA alignment (" + " > ".join(f"EMA {p}" for p, _ in values) + ")"
        )
    elif bearish == len(pairs):
        reasons.append(
            "Bearish EMA alignment (" + " < ".join(f"EMA {p}" for p, _ in values) + ")"
        )
    else:
        for (fast_period, fast), (slow_period, slow) in pairs:
            if fast > slow:
                reasons.append(f"EMA {fast_period} above EMA {slow_period}")

    detail = f"{bullish}/{len(pairs)} EMA pairs in bullish order"
    return TrendComponent("ema_alignment", score, 0.25, detail), reasons


def _score_price_vs_averages(
    data: pd.DataFrame, config: IndicatorConfig
) -> tuple[TrendComponent | None, list[str]]:
    """Score price position relative to each configured EMA."""
    close = float(data["close"].iloc[-1])
    above = 0
    total = 0
    reasons: list[str] = []

    for period in sorted(config.ema_periods):
        series = _series_for(
            data, ema_column(period), lambda p=period: calculate_ema(data["close"], p)
        )
        value = series.iloc[-1] if len(series) else np.nan
        if pd.isna(value):
            continue
        total += 1
        if close > float(value):
            above += 1
            if period >= 50:  # only the meaningful ones are worth reporting
                reasons.append(f"Price above EMA {period}")
        elif period >= 50:
            reasons.append(f"Price below EMA {period}")

    if total == 0:
        return None, []

    score = (above / total) * 2 - 1
    return (
        TrendComponent("price_vs_averages", score, 0.20, f"price above {above}/{total} EMAs"),
        reasons,
    )


def _score_structure(
    data: pd.DataFrame, config: IndicatorConfig
) -> tuple[TrendComponent | None, list[str]]:
    """Score swing structure (higher highs / lower lows)."""
    structure = detect_market_structure(data, config)
    if structure == "UNDETERMINED":
        return None, []

    mapping = {
        "HIGHER_HIGHS_HIGHER_LOWS": (1.0, "Higher highs and higher lows"),
        "LOWER_HIGHS_LOWER_LOWS": (-1.0, "Lower highs and lower lows"),
        "RANGING": (0.0, "Swing structure is ranging"),
    }
    score, reason = mapping[structure]
    return TrendComponent("structure", score, 0.20, structure), [reason]


def _score_macd(
    data: pd.DataFrame, config: IndicatorConfig
) -> tuple[TrendComponent | None, list[str]]:
    """Score MACD histogram level and direction."""
    if MACD_HISTOGRAM_COL in data.columns:
        histogram = data[MACD_HISTOGRAM_COL]
    else:
        histogram = calculate_macd(
            data["close"], config.macd_fast, config.macd_slow, config.macd_signal
        )[MACD_HISTOGRAM_COL]

    clean = histogram.dropna()
    if clean.empty:
        return None, []

    current = float(clean.iloc[-1])
    previous = float(clean.iloc[-2]) if len(clean) >= 2 else current
    rising = current > previous

    # Sign carries most of the weight; direction refines it. A positive but
    # shrinking histogram is bullish with fading conviction.
    if current > 0:
        score = 1.0 if rising else 0.5
        reason = "Positive MACD momentum" + (" and rising" if rising else " but fading")
    elif current < 0:
        score = -1.0 if not rising else -0.5
        reason = "Negative MACD momentum" + (" and falling" if not rising else " but recovering")
    else:
        score = 0.0
        reason = "MACD momentum flat"

    return TrendComponent("macd", score, 0.20, f"histogram={current:.4f}"), [reason]


def _score_slope(
    data: pd.DataFrame, config: IndicatorConfig
) -> tuple[TrendComponent | None, list[str]]:
    """Score the slope of a medium EMA, normalised by ATR.

    Dividing by ATR makes the score comparable across symbols and timeframes: a
    $1 move per bar is decisive in a quiet stock and noise in a volatile one.
    """
    periods = sorted(config.ema_periods)
    if not periods:
        return None, []
    # Middle-length EMA: responsive enough to turn, slow enough to mean something.
    period = periods[len(periods) // 2]

    series = _series_for(
        data, ema_column(period), lambda p=period: calculate_ema(data["close"], p)
    ).dropna()
    lookback = max(3, period // 4)
    if len(series) < lookback + 1:
        return None, []

    change = float(series.iloc[-1]) - float(series.iloc[-1 - lookback])

    atr_series = _series_for(
        data,
        atr_column(config.atr_period),
        lambda: calculate_atr(data["high"], data["low"], data["close"], config.atr_period),
    ).dropna()
    if atr_series.empty or float(atr_series.iloc[-1]) <= 0:
        return None, []

    # Normalised slope of 1.0 means the EMA moved one ATR over the window.
    normalised = change / float(atr_series.iloc[-1])
    score = float(np.clip(normalised, -1.0, 1.0))

    reasons: list[str] = []
    if score > 0.25:
        reasons.append(f"EMA {period} rising")
    elif score < -0.25:
        reasons.append(f"EMA {period} falling")

    return (
        TrendComponent("slope", score, 0.15, f"EMA{period} moved {normalised:.2f} ATR"),
        reasons,
    )


def analyze_trend(
    data: pd.DataFrame,
    config: IndicatorConfig | None = None,
) -> TrendAnalysis:
    """Classify the trend on the most recent bar.

    Parameters
    ----------
    data:
        OHLCV frame. Pass the output of
        :func:`~trading_bot.indicators.technical_indicators.calculate_all_indicators`
        to avoid recomputing indicators; any missing ones are computed on demand.
    config:
        Periods and thresholds.

    Returns
    -------
    TrendAnalysis
        ``direction``, ``strength`` (0-100 directional), ``confidence`` and the
        human-readable ``reasons`` behind the verdict.

    Raises
    ------
    InvalidDataError
        The frame is empty or malformed.

    Example
    -------
    >>> trend = analyze_trend(enriched)
    >>> trend.direction, trend.strength
    (<TrendDirection.BULLISH: 'BULLISH'>, 78)
    """
    settings = config or DEFAULT_CONFIG
    validate_ohlcv(data, name="data")

    components: list[TrendComponent] = []
    reasons: list[str] = []
    for scorer in (
        _score_ema_alignment,
        _score_price_vs_averages,
        _score_structure,
        _score_macd,
        _score_slope,
    ):
        component, component_reasons = scorer(data, settings)
        if component is not None:
            components.append(component)
            reasons.extend(component_reasons)

    if not components:
        logger.debug("No trend components could be scored (%d bars)", len(data))
        return TrendAnalysis(
            direction=TrendDirection.NEUTRAL,
            strength=50,
            confidence=0,
            reasons=("Insufficient data for trend analysis",),
        )

    # Renormalise over the components that actually had data.
    total_weight = sum(component.weight for component in components)
    net = sum(component.contribution for component in components) / total_weight
    net = float(np.clip(net, -1.0, 1.0))
    strength = int(round(50 + net * 50))

    direction = TrendDirection.STRONG_BEARISH
    for threshold, label in _DIRECTION_THRESHOLDS:
        if strength >= threshold:
            direction = label
            break

    # Confidence blends how much of the model had data with how much it agreed.
    coverage = total_weight / 1.0
    if net == 0:
        agreement = 0.0
    else:
        aligned = sum(
            component.weight
            for component in components
            if np.sign(component.score) == np.sign(net) and component.score != 0
        )
        agreement = aligned / total_weight
    confidence = int(round(100 * coverage * (0.4 + 0.6 * agreement)))
    confidence = int(np.clip(confidence, 0, 100))

    return TrendAnalysis(
        direction=direction,
        strength=strength,
        confidence=confidence,
        reasons=tuple(reasons),
        components=tuple(components),
    )
