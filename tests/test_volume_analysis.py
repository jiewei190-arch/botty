"""Volume classification, participation trend, and price confirmation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_bars
from trading_bot.indicators import (
    IndicatorConfig,
    InvalidDataError,
    VolumeCondition,
    analyze_volume,
    calculate_all_indicators,
    classify_relative_volume,
    detect_volume_condition,
    detect_volume_trend,
    volume_confirms_price,
)


def frame_with_volume(volumes: list[float], closes: list[float] | None = None) -> pd.DataFrame:
    """Build a frame with an explicit volume profile."""
    count = len(volumes)
    prices = np.array(closes if closes else [100.0] * count, dtype="float64")
    index = pd.date_range("2024-01-02", periods=count, freq="15min", tz="UTC")
    index.name = "timestamp"
    return pd.DataFrame(
        {
            "open": np.concatenate([[prices[0]], prices[:-1]]),
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.array(volumes, dtype="float64"),
        },
        index=index,
    )


# ============================================================================
# Classification
# ============================================================================


@pytest.mark.parametrize(
    ("relative", "expected"),
    [
        (0.3, VolumeCondition.LOW),
        (0.69, VolumeCondition.LOW),
        (0.7, VolumeCondition.NORMAL),
        (1.0, VolumeCondition.NORMAL),
        (1.49, VolumeCondition.NORMAL),
        (1.5, VolumeCondition.HIGH),
        (2.4, VolumeCondition.HIGH),
        (2.5, VolumeCondition.SPIKE),
        (10.0, VolumeCondition.SPIKE),
    ],
)
def test_default_thresholds_bucket_relative_volume(relative, expected):
    assert classify_relative_volume(relative) is expected


def test_unknown_relative_volume_is_treated_as_normal():
    assert classify_relative_volume(None) is VolumeCondition.NORMAL
    assert classify_relative_volume(float("nan")) is VolumeCondition.NORMAL


def test_thresholds_are_configurable():
    loose = IndicatorConfig(
        volume_low_threshold=0.3, volume_high_threshold=3.0, volume_spike_threshold=5.0
    )
    assert classify_relative_volume(2.0, loose) is VolumeCondition.NORMAL
    assert classify_relative_volume(2.0) is VolumeCondition.HIGH


def test_elevated_covers_high_and_spike():
    assert VolumeCondition.HIGH.is_elevated
    assert VolumeCondition.SPIKE.is_elevated
    assert not VolumeCondition.NORMAL.is_elevated
    assert not VolumeCondition.LOW.is_elevated


# ============================================================================
# Detection on real frames
# ============================================================================


def test_detects_a_volume_spike():
    data = frame_with_volume([100_000.0] * 20 + [500_000.0])
    assert detect_volume_condition(calculate_all_indicators(data)) is VolumeCondition.SPIKE


def test_detects_low_volume():
    data = frame_with_volume([100_000.0] * 20 + [20_000.0])
    assert detect_volume_condition(calculate_all_indicators(data)) is VolumeCondition.LOW


def test_steady_volume_is_normal():
    data = frame_with_volume([100_000.0] * 25)
    assert detect_volume_condition(calculate_all_indicators(data)) is VolumeCondition.NORMAL


def test_condition_is_normal_during_warmup():
    assert detect_volume_condition(frame_with_volume([100_000.0] * 5)) is VolumeCondition.NORMAL


# ============================================================================
# Participation trend
# ============================================================================


def test_detects_rising_participation():
    data = frame_with_volume([100_000.0] * 20 + [300_000.0] * 5)
    assert detect_volume_trend(calculate_all_indicators(data)) == "RISING"


def test_detects_falling_participation():
    data = frame_with_volume([300_000.0] * 20 + [50_000.0] * 5)
    assert detect_volume_trend(calculate_all_indicators(data)) == "FALLING"


def test_steady_participation_is_reported_as_steady():
    data = frame_with_volume([100_000.0] * 30)
    assert detect_volume_trend(calculate_all_indicators(data)) == "STEADY"


def test_trend_is_unknown_without_enough_bars():
    assert detect_volume_trend(frame_with_volume([100_000.0] * 3)) == "UNKNOWN"


# ============================================================================
# Price confirmation
# ============================================================================


def test_a_move_on_heavy_volume_is_confirmed():
    closes = [100.0] * 20 + [105.0]
    data = frame_with_volume([100_000.0] * 20 + [400_000.0], closes)
    assert volume_confirms_price(calculate_all_indicators(data))


def test_a_move_on_thin_volume_is_not_confirmed():
    closes = [100.0] * 20 + [105.0]
    data = frame_with_volume([100_000.0] * 20 + [10_000.0], closes)
    assert not volume_confirms_price(calculate_all_indicators(data))


def test_an_unchanged_bar_confirms_nothing():
    data = frame_with_volume([100_000.0] * 20 + [400_000.0])
    assert not volume_confirms_price(calculate_all_indicators(data))


# ============================================================================
# Full analysis
# ============================================================================


def test_analysis_reports_the_relative_volume_multiple():
    data = frame_with_volume([100_000.0] * 20 + [200_000.0])
    analysis = analyze_volume(calculate_all_indicators(data))
    assert analysis.relative_volume == pytest.approx(200_000 / 105_000, rel=1e-6)
    assert analysis.current_volume == 200_000.0
    assert analysis.condition is VolumeCondition.HIGH


def test_analysis_explains_itself():
    data = frame_with_volume([100_000.0] * 20 + [500_000.0])
    analysis = analyze_volume(calculate_all_indicators(data))
    assert analysis.reasons
    assert any("spike" in reason.lower() for reason in analysis.reasons)


def test_analysis_reports_unavailable_relative_volume_during_warmup():
    analysis = analyze_volume(frame_with_volume([100_000.0] * 5))
    assert analysis.relative_volume is None
    assert any("unavailable" in reason for reason in analysis.reasons)


def test_analysis_serializes():
    payload = analyze_volume(calculate_all_indicators(make_bars(100, seed=90))).as_dict()
    assert set(payload) >= {"condition", "relative_volume", "trend", "confirms_price"}


def test_works_without_precomputed_columns():
    raw = frame_with_volume([100_000.0] * 20 + [500_000.0])
    assert analyze_volume(raw).condition is VolumeCondition.SPIKE


def test_validates_its_input():
    with pytest.raises(InvalidDataError):
        analyze_volume(pd.DataFrame())


def test_zero_volume_history_does_not_divide_by_zero():
    analysis = analyze_volume(frame_with_volume([0.0] * 25))
    assert analysis.relative_volume is None
    assert analysis.condition is VolumeCondition.NORMAL
