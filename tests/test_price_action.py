"""Swing points, support/resistance, and the pivot confirmation lag.

The confirmation-lag tests matter most: a pivot identified before its
confirming bars exist is lookahead bias, and it makes a backtest lie.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_bars
from trading_bot.indicators import (
    IndicatorConfig,
    InvalidDataError,
    detect_market_structure,
    find_support_resistance,
    find_swing_points,
    swing_point_columns,
)


def wave(periods: int = 60, amplitude: float = 10.0, wavelength: float = 6.0) -> pd.DataFrame:
    """A clean sine wave, which has unambiguous swing highs and lows."""
    x = np.arange(periods)
    price = 100 + amplitude * np.sin(x / wavelength)
    index = pd.date_range("2024-01-02", periods=periods, freq="15min", tz="UTC")
    index.name = "timestamp"
    return pd.DataFrame(
        {
            "open": price,
            "high": price + 1.0,
            "low": price - 1.0,
            "close": price,
            "volume": np.full(periods, 100_000.0),
        },
        index=index,
    )


def staircase(steps: list[float]) -> pd.DataFrame:
    """Build a frame from explicit price levels."""
    values = np.array(steps, dtype="float64")
    index = pd.date_range("2024-01-02", periods=len(values), freq="15min", tz="UTC")
    index.name = "timestamp"
    return pd.DataFrame(
        {
            "open": values,
            "high": values + 0.5,
            "low": values - 0.5,
            "close": values,
            "volume": np.full(len(values), 100_000.0),
        },
        index=index,
    )


# ============================================================================
# Swing points
# ============================================================================


def test_finds_swing_highs_and_lows_in_a_wave():
    points = find_swing_points(wave(), strength=3)
    assert any(point.kind == "high" for point in points)
    assert any(point.kind == "low" for point in points)


def test_swing_high_is_the_maximum_of_its_window():
    data = wave()
    for point in find_swing_points(data, strength=3):
        if point.kind == "high":
            window = data["high"].iloc[max(0, point.index - 3) : point.index + 4]
            assert point.price == pytest.approx(window.max())


def test_swing_low_is_the_minimum_of_its_window():
    data = wave()
    for point in find_swing_points(data, strength=3):
        if point.kind == "low":
            window = data["low"].iloc[max(0, point.index - 3) : point.index + 4]
            assert point.price == pytest.approx(window.min())


def test_confirmation_lag_equals_the_pivot_strength():
    """A pivot needs `strength` bars after it before it can be identified."""
    for strength in (2, 3, 5):
        points = find_swing_points(wave(80), strength=strength)
        assert points, f"no pivots at strength {strength}"
        assert all(point.confirmed_index - point.index == strength for point in points)


def test_no_pivot_is_confirmed_within_the_final_bars():
    """The tail of the series cannot contain a confirmed pivot — that is the point."""
    data = wave(80)
    strength = 3
    last_index = len(data) - 1
    confirmed = [
        point
        for point in find_swing_points(data, strength=strength)
        if point.confirmed_index <= last_index
    ]
    assert all(point.index <= last_index - strength for point in confirmed)


def test_higher_strength_finds_fewer_more_significant_pivots():
    data = wave(120, wavelength=8.0)
    assert len(find_swing_points(data, strength=2)) >= len(
        find_swing_points(data, strength=8)
    )


def test_no_pivots_when_the_series_is_shorter_than_the_window():
    assert find_swing_points(wave(5), strength=3) == []


def test_monotonic_series_has_no_interior_pivots():
    """A straight line up has no swing highs before its final bars."""
    data = staircase([100.0 + index for index in range(40)])
    highs = [point for point in find_swing_points(data, strength=3) if point.kind == "high"]
    assert all(point.index >= len(data) - 4 for point in highs)


def test_swing_strength_must_be_positive():
    with pytest.raises(ValueError, match="strength must be >= 1"):
        find_swing_points(wave(), strength=0)


def test_swing_point_columns_align_with_the_frame():
    data = wave()
    flags = swing_point_columns(data, strength=3)
    assert list(flags.index) == list(data.index)
    assert flags["SWING_HIGH"].sum() > 0
    assert flags["SWING_LOW"].sum() > 0


def test_swing_points_are_returned_in_chronological_order():
    points = find_swing_points(wave(120), strength=3)
    assert [point.index for point in points] == sorted(point.index for point in points)


# ============================================================================
# Support and resistance
# ============================================================================


def test_levels_split_around_the_reference_price():
    levels = find_support_resistance(wave(120))
    assert all(level.price < levels.price for level in levels.support)
    assert all(level.price > levels.price for level in levels.resistance)


def test_support_is_ordered_nearest_first():
    levels = find_support_resistance(wave(150))
    prices = [level.price for level in levels.support]
    assert prices == sorted(prices, reverse=True)


def test_resistance_is_ordered_nearest_first():
    levels = find_support_resistance(wave(150))
    prices = [level.price for level in levels.resistance]
    assert prices == sorted(prices)


def test_repeated_touches_merge_into_one_level():
    """A price revisited several times is one level, not several."""
    levels = find_support_resistance(wave(200, wavelength=6.0))
    all_levels = list(levels.support) + list(levels.resistance)
    assert all_levels
    assert any(level.touches > 1 for level in all_levels)


def test_levels_further_apart_than_the_tolerance_stay_separate():
    config = IndicatorConfig(level_tolerance_pct=0.1, swing_strength=2)
    data = staircase(
        [100, 105, 100, 95, 100, 120, 100, 80, 100, 130, 100, 70, 100, 105, 100]
    )
    levels = find_support_resistance(data, config)
    prices = [level.price for level in levels.support + levels.resistance]
    assert len(set(round(price, 2) for price in prices)) == len(prices)


def test_as_of_excludes_pivots_not_yet_confirmed():
    """The lookahead guarantee: asking about bar N sees only what bar N knew."""
    data = wave(120)
    early = find_support_resistance(data, as_of=40)
    assert all(point.confirmed_index <= 40 for point in early.swing_points)
    assert all(point.index <= 40 for point in early.swing_points)


def test_as_of_yields_a_subset_of_the_full_history():
    data = wave(120)
    early = set(point.index for point in find_support_resistance(data, as_of=50).swing_points)
    late = set(point.index for point in find_support_resistance(data, as_of=110).swing_points)
    assert early <= late or len(early) <= len(late)


def test_as_of_is_bounds_checked():
    with pytest.raises(InvalidDataError, match="as_of"):
        find_support_resistance(wave(60), as_of=999)


def test_max_levels_is_respected():
    levels = find_support_resistance(wave(300, wavelength=4.0), max_levels=2)
    assert len(levels.support) <= 2
    assert len(levels.resistance) <= 2


def test_nearest_levels_are_the_closest_ones():
    levels = find_support_resistance(wave(150))
    if levels.support:
        assert levels.nearest_support is levels.support[0]
    if levels.resistance:
        assert levels.nearest_resistance is levels.resistance[0]


def test_nearest_levels_are_none_when_no_levels_exist():
    levels = find_support_resistance(staircase([100.0] * 40))
    assert levels.nearest_support is None
    assert levels.nearest_resistance is None


def test_level_distance_is_signed():
    levels = find_support_resistance(wave(150))
    if levels.nearest_support:
        assert levels.nearest_support.distance_pct(levels.price) < 0
    if levels.nearest_resistance:
        assert levels.nearest_resistance.distance_pct(levels.price) > 0


def test_support_resistance_serializes():
    payload = find_support_resistance(wave(150)).as_dict()
    assert set(payload) >= {"price", "support", "resistance"}


def test_explicit_reference_price_is_honoured():
    levels = find_support_resistance(wave(150), price=1000.0)
    assert levels.price == 1000.0
    assert not levels.resistance   # nothing trades above 1000 in this series


# ============================================================================
# Market structure
# ============================================================================


def test_detects_higher_highs_and_higher_lows():
    """Zigzag with peaks 110/120/130/140 and troughs 105/110/115."""
    data = staircase([100, 110, 105, 120, 110, 130, 115, 140, 130])
    assert detect_market_structure(data, IndicatorConfig(swing_strength=1)) == (
        "HIGHER_HIGHS_HIGHER_LOWS"
    )


def test_detects_lower_highs_and_lower_lows():
    """The mirror image: peaks 135/125/120 and troughs 130/120/110/100."""
    data = staircase([140, 130, 135, 120, 125, 110, 120, 100, 110])
    assert detect_market_structure(data, IndicatorConfig(swing_strength=1)) == (
        "LOWER_HIGHS_LOWER_LOWS"
    )


def test_flat_market_is_not_reported_as_a_downtrend():
    """A constant series has no pivots at all, so structure is undetermined.

    Before pivots required a strict inequality, every bar of a flat series was
    both a swing high and a swing low, and equal highs read as "lower highs".
    """
    data = staircase([100.0] * 30)
    assert find_swing_points(data, strength=3) == []
    assert detect_market_structure(data) == "UNDETERMINED"


def test_a_flat_double_top_registers_exactly_one_pivot():
    """Strict-then-inclusive comparison keeps plateaus from double-counting."""
    data = staircase([100, 101, 102, 103, 105, 105, 103, 102, 101, 100])
    highs = [point for point in find_swing_points(data, strength=2) if point.kind == "high"]
    assert len(highs) == 1


def test_structure_is_undetermined_without_enough_pivots():
    assert detect_market_structure(staircase([100.0] * 10)) == "UNDETERMINED"


def test_structure_respects_the_as_of_bar():
    data = wave(150)
    assert detect_market_structure(data, as_of=15) == "UNDETERMINED"


# ============================================================================
# Validation
# ============================================================================


def test_swing_points_validate_their_input():
    with pytest.raises(InvalidDataError):
        find_swing_points(pd.DataFrame())


def test_support_resistance_validates_its_input():
    with pytest.raises(InvalidDataError):
        find_support_resistance(make_bars(50, seed=70).drop(columns=["high"]))


# ----------------------------------------------------------------------------
# Vectorised pivot detection
#
# The neighbour comparison is done with shifted numpy views rather than a
# per-bar DataFrame, because the original was 64% of a backtest's runtime. The
# tie-breaking rule is what makes it subtle: strict on the earlier side and
# inclusive on the later one, so a flat double top is a pivot but a flat series
# is not every pivot at once.
# ----------------------------------------------------------------------------


def _reference_pivots(data: pd.DataFrame, span: int) -> tuple[np.ndarray, np.ndarray]:
    """The pivot rule stated plainly, one bar at a time.

    Deliberately naive and O(n * span): it exists to disagree with the fast
    implementation if the fast one is ever wrong.
    """
    highs = data["high"].to_numpy(dtype="float64")
    lows = data["low"].to_numpy(dtype="float64")
    count = len(highs)
    is_high = np.zeros(count, dtype=bool)
    is_low = np.zeros(count, dtype=bool)
    for i in range(span, count - span):
        before = slice(i - span, i)
        after = slice(i + 1, i + span + 1)
        is_high[i] = (highs[i] > highs[before]).all() and (highs[i] >= highs[after]).all()
        is_low[i] = (lows[i] < lows[before]).all() and (lows[i] <= lows[after]).all()
    return is_high, is_low


def _pivot_masks(data: pd.DataFrame, span: int) -> tuple[np.ndarray, np.ndarray]:
    is_high = np.zeros(len(data), dtype=bool)
    is_low = np.zeros(len(data), dtype=bool)
    for point in find_swing_points(data, span):
        (is_high if point.kind == "high" else is_low)[point.index] = True
    return is_high, is_low


@pytest.mark.parametrize("span", [1, 2, 3, 5, 8])
def test_pivots_match_a_naive_implementation(span):
    for seed in range(6):
        data = make_bars(120, seed=seed)
        fast_high, fast_low = _pivot_masks(data, span)
        slow_high, slow_low = _reference_pivots(data, span)
        assert (fast_high == slow_high).all()
        assert (fast_low == slow_low).all()


@pytest.mark.parametrize("span", [1, 2, 3])
def test_a_flat_series_has_no_pivots(span):
    """Every bar ties with every other; none of them is a turning point."""
    data = staircase([100.0] * 30)
    assert find_swing_points(data, span) == []


def test_a_flat_double_top_is_still_a_pivot():
    """Inclusive on the later side, so an exact retest does not erase the high."""
    closes = [100.0, 101, 102, 103, 110, 103, 102, 110, 102, 101, 100]
    points = find_swing_points(staircase(closes), 2)
    assert any(point.kind == "high" for point in points)


@pytest.mark.parametrize("length", [0, 1, 2, 3])
def test_short_frames_yield_no_pivots_rather_than_failing(length):
    if length == 0:
        pytest.skip("an empty frame is rejected by validation, not by the pivot rule")
    data = staircase([100.0 + i for i in range(length)])
    assert find_swing_points(data, 2) == []


def test_pivot_prices_come_from_the_right_column():
    data = make_bars(80, seed=9)
    for point in find_swing_points(data, 3):
        column = "high" if point.kind == "high" else "low"
        assert point.price == pytest.approx(float(data[column].iloc[point.index]))
        assert point.timestamp == data.index[point.index]


def test_validation_can_be_skipped_by_internal_callers():
    """The flag is an internal optimisation; it must not change what is found."""
    data = make_bars(100, seed=13)
    assert find_swing_points(data, 3, validate=False) == find_swing_points(data, 3)


def test_validation_still_runs_by_default():
    data = make_bars(50, seed=14)
    data.loc[data.index[10], "high"] = np.nan
    with pytest.raises(InvalidDataError):
        find_swing_points(data, 3)
