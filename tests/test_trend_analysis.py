"""Trend classification: direction, strength scaling, and confidence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_bars
from trading_bot.indicators import (
    IndicatorConfig,
    InvalidDataError,
    TrendDirection,
    analyze_trend,
    calculate_all_indicators,
)


def trending(periods: int, drift: float, seed: int = 5, noise: float = 0.004) -> pd.DataFrame:
    """A price series with a controlled drift per bar."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, noise, periods)))
    index = pd.date_range("2024-01-02", periods=periods, freq="15min", tz="UTC")
    index.name = "timestamp"
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.004,
            "low": close * 0.996,
            "close": close,
            "volume": rng.integers(100_000, 500_000, periods).astype("float64"),
        },
        index=index,
    )


def test_strong_uptrend_is_classified_bullish():
    trend = analyze_trend(calculate_all_indicators(trending(300, 0.004)))
    assert trend.direction.is_bullish
    assert trend.strength > 60


def test_strong_downtrend_is_classified_bearish():
    trend = analyze_trend(calculate_all_indicators(trending(300, -0.004)))
    assert trend.direction.is_bearish
    assert trend.strength < 40


def test_strength_is_a_directional_scale_centred_on_50():
    up = analyze_trend(calculate_all_indicators(trending(300, 0.005)))
    down = analyze_trend(calculate_all_indicators(trending(300, -0.005)))
    assert up.strength > 50 > down.strength
    assert 0 <= up.strength <= 100
    assert 0 <= down.strength <= 100


def test_stronger_drift_scores_higher():
    weak = analyze_trend(calculate_all_indicators(trending(300, 0.0005, noise=0.006)))
    strong = analyze_trend(calculate_all_indicators(trending(300, 0.006, noise=0.002)))
    assert strong.strength >= weak.strength


def test_direction_label_always_agrees_with_the_reported_strength():
    """The label is derived from the strength, so the two can never disagree."""
    for drift in (0.006, 0.002, 0.0, -0.002, -0.006):
        trend = analyze_trend(calculate_all_indicators(trending(300, drift, noise=0.002)))
        strength = trend.strength
        if strength >= 75:
            expected = TrendDirection.STRONG_BULLISH
        elif strength >= 60:
            expected = TrendDirection.BULLISH
        elif strength >= 40:
            expected = TrendDirection.NEUTRAL
        elif strength >= 25:
            expected = TrendDirection.BEARISH
        else:
            expected = TrendDirection.STRONG_BEARISH
        assert trend.direction is expected, f"drift={drift} strength={strength}"


def test_a_sustained_decline_lands_in_the_bearish_family():
    trend = analyze_trend(calculate_all_indicators(trending(300, -0.006, noise=0.002)))
    assert trend.direction.is_bearish


def test_reasons_explain_the_verdict():
    trend = analyze_trend(calculate_all_indicators(trending(300, 0.004)))
    assert trend.reasons
    assert any("EMA" in reason for reason in trend.reasons)


def test_components_are_scored_within_bounds():
    trend = analyze_trend(calculate_all_indicators(trending(300, 0.003)))
    assert trend.components
    for component in trend.components:
        assert -1.0 <= component.score <= 1.0
        assert component.weight > 0


def test_confidence_grows_with_available_history():
    """A short frame can score fewer components, so it should be less confident."""
    short = analyze_trend(calculate_all_indicators(trending(30, 0.004)))
    long = analyze_trend(calculate_all_indicators(trending(300, 0.004)))
    assert long.confidence > short.confidence


def test_confidence_falls_when_components_disagree():
    """A choppy oscillator has conflicting signals; a clean trend does not."""
    periods = 300
    x = np.arange(periods)
    oscillating = 100 + 4 * np.sin(x / 25)
    index = pd.date_range("2024-01-02", periods=periods, freq="15min", tz="UTC")
    index.name = "timestamp"
    choppy = pd.DataFrame(
        {
            "open": oscillating,
            "high": oscillating * 1.004,
            "low": oscillating * 0.996,
            "close": oscillating,
            "volume": np.full(periods, 200_000.0),
        },
        index=index,
    )
    clean = analyze_trend(calculate_all_indicators(trending(300, 0.006, noise=0.001)))
    conflicted = analyze_trend(calculate_all_indicators(choppy))
    assert conflicted.confidence < clean.confidence


def test_insufficient_data_is_neutral_with_zero_confidence():
    trend = analyze_trend(calculate_all_indicators(make_bars(1, seed=80)))
    assert trend.direction is TrendDirection.NEUTRAL
    assert trend.strength == 50
    assert trend.confidence == 0


def test_analysis_serializes_to_the_documented_shape():
    payload = analyze_trend(calculate_all_indicators(trending(300, 0.004))).as_dict()
    assert set(payload) >= {"direction", "strength", "confidence", "reasons"}
    assert isinstance(payload["direction"], str)
    assert isinstance(payload["strength"], int)
    assert isinstance(payload["reasons"], list)


def test_direction_helpers_agree_with_the_labels():
    assert TrendDirection.STRONG_BULLISH.is_bullish
    assert TrendDirection.BULLISH.is_bullish
    assert not TrendDirection.NEUTRAL.is_bullish
    assert not TrendDirection.NEUTRAL.is_bearish
    assert TrendDirection.BEARISH.is_bearish
    assert TrendDirection.STRONG_BEARISH.is_bearish


def test_works_without_precomputed_indicator_columns():
    """Raw OHLCV is acceptable; missing indicators are computed on demand."""
    raw = trending(300, 0.004)
    assert analyze_trend(raw).direction is analyze_trend(calculate_all_indicators(raw)).direction


def test_honours_a_custom_config():
    config = IndicatorConfig(ema_periods=(5, 13), sma_periods=(10,))
    trend = analyze_trend(calculate_all_indicators(trending(200, 0.004), config), config)
    assert trend.components


def test_validates_its_input():
    with pytest.raises(InvalidDataError):
        analyze_trend(pd.DataFrame())


def test_works_across_timeframes():
    for freq in ("5min", "1h", "1D"):
        bars = make_bars(260, freq=freq, seed=81)
        assert analyze_trend(calculate_all_indicators(bars)).components, freq
