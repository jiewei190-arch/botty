"""Technical indicator engine: calculation correctness and input validation.

Reference values are derived independently — by hand from the indicator's
definition, or by a plain Python loop written separately from the vectorised
implementation. Asserting a function against itself proves nothing.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_bars
from trading_bot.indicators import (
    BB_LOWER_COL,
    BB_MIDDLE_COL,
    BB_PERCENT_B_COL,
    BB_UPPER_COL,
    BB_WIDTH_COL,
    MACD_COL,
    MACD_HISTOGRAM_COL,
    MACD_SIGNAL_COL,
    RELATIVE_VOLUME_COL,
    IndicatorConfig,
    InsufficientDataError,
    InvalidDataError,
    calculate_all_indicators,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_relative_volume,
    calculate_rsi,
    calculate_sma,
    calculate_true_range,
    calculate_volume_sma,
    detect_bollinger_condition,
    detect_ema_crossover,
    detect_macd_momentum,
    detect_macd_signal,
    detect_rsi_condition,
    latest_values,
    validate_ohlcv,
)

# Wilder's published example series from "New Concepts in Technical Trading Systems".
WILDER_CLOSES = [
    44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
    45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64,
    46.21, 46.25, 45.71, 46.45, 45.78, 45.35, 44.03, 44.18, 44.22, 44.57,
]


def ohlcv(closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    """Build a minimal valid OHLCV frame from a close series."""
    values = pd.Series(closes, dtype="float64")
    index = pd.date_range("2024-01-02 14:30", periods=len(values), freq="15min", tz="UTC")
    index.name = "timestamp"
    return pd.DataFrame(
        {
            "open": values.shift(1).fillna(values.iloc[0]).to_numpy(),
            "high": (values * 1.005).to_numpy(),
            "low": (values * 0.995).to_numpy(),
            "close": values.to_numpy(),
            "volume": np.array(volumes if volumes else [100_000.0] * len(values)),
        },
        index=index,
    )


# ============================================================================
# Simple moving average
# ============================================================================


def test_sma_matches_hand_computed_mean():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = calculate_sma(series, 3)
    assert result.iloc[2] == pytest.approx(2.0)   # (1+2+3)/3
    assert result.iloc[4] == pytest.approx(4.0)   # (3+4+5)/3


def test_sma_is_nan_during_warmup():
    result = calculate_sma(pd.Series([1.0, 2.0, 3.0, 4.0]), 3)
    assert result.iloc[:2].isna().all()
    assert result.notna().sum() == 2


def test_sma_of_a_constant_series_is_that_constant():
    result = calculate_sma(pd.Series([7.0] * 10), 5)
    assert result.dropna().eq(7.0).all()


def test_sma_rejects_invalid_period():
    with pytest.raises(ValueError, match="period must be >= 1"):
        calculate_sma(pd.Series([1.0, 2.0]), 0)


# ============================================================================
# Exponential moving average
# ============================================================================


def reference_ema(values: list[float], period: int) -> list[float | None]:
    """EMA computed by an explicit loop, seeded with the SMA (TA-Lib convention)."""
    alpha = 2 / (period + 1)
    out: list[float | None] = [None] * len(values)
    previous = sum(values[:period]) / period
    out[period - 1] = previous
    for index in range(period, len(values)):
        previous = values[index] * alpha + previous * (1 - alpha)
        out[index] = previous
    return out


def test_ema_matches_an_independent_recursive_implementation():
    expected = reference_ema(WILDER_CLOSES, 10)
    result = calculate_ema(pd.Series(WILDER_CLOSES), 10)
    for index, want in enumerate(expected):
        if want is None:
            assert pd.isna(result.iloc[index])
        else:
            assert result.iloc[index] == pytest.approx(want, abs=1e-10)


def test_ema_is_seeded_with_sma_not_the_first_value():
    """The seeding choice is what makes values match charting platforms."""
    series = pd.Series(WILDER_CLOSES)
    ours = calculate_ema(series, 10)
    naive = series.ewm(span=10, adjust=False).mean()
    assert ours.iloc[9] == pytest.approx(series.iloc[:10].mean())
    assert abs(ours.iloc[9] - naive.iloc[9]) > 0.01


def test_ema_first_value_appears_at_index_period_minus_one():
    result = calculate_ema(pd.Series(WILDER_CLOSES), 10)
    assert result.first_valid_index() == 9


def test_ema_reacts_faster_than_sma():
    """A step change should move the EMA further than the SMA."""
    values = pd.Series([10.0] * 20 + [20.0] * 5)
    assert calculate_ema(values, 10).iloc[-1] > calculate_sma(values, 10).iloc[-1]


def test_ema_of_a_constant_series_is_that_constant():
    assert calculate_ema(pd.Series([5.0] * 30), 10).dropna().eq(5.0).all()


def test_ema_returns_all_nan_when_series_is_shorter_than_period():
    assert calculate_ema(pd.Series([1.0, 2.0, 3.0]), 10).isna().all()


# ============================================================================
# RSI
# ============================================================================


def test_rsi_matches_wilders_definition_computed_by_hand():
    """First RSI(14) of the Wilder series, derived from the raw prices.

    Over the first 14 changes the gains total 3.34 and the losses 1.40, so
    avg_gain = 0.2385714, avg_loss = 0.1, RS = 2.3857143 and
    RSI = 100 - 100 / (1 + RS) = 70.4641.
    """
    result = calculate_rsi(pd.Series(WILDER_CLOSES), 14)
    assert result.first_valid_index() == 14
    assert result.iloc[14] == pytest.approx(70.4641, abs=1e-3)


def test_rsi_stays_within_bounds():
    bars = make_bars(300, seed=17)
    values = calculate_rsi(bars["close"], 14).dropna()
    assert not values.empty
    assert values.between(0, 100).all()


def test_rsi_is_100_when_every_change_is_a_gain():
    rising = pd.Series([100.0 + index for index in range(30)])
    assert calculate_rsi(rising, 14).dropna().eq(100.0).all()


def test_rsi_is_zero_when_every_change_is_a_loss():
    falling = pd.Series([100.0 - index for index in range(30)])
    assert calculate_rsi(falling, 14).dropna().eq(0.0).all()


def test_rsi_is_50_for_a_flat_series():
    """A flat market is neither overbought nor oversold — not 0/0."""
    result = calculate_rsi(pd.Series([50.0] * 30), 14).dropna()
    assert not result.empty
    assert result.eq(50.0).all()


def test_rsi_period_is_configurable():
    """A longer period simply delays the first value."""
    series = pd.Series(WILDER_CLOSES)
    assert calculate_rsi(series, 7).first_valid_index() == 7
    assert calculate_rsi(series, 21).first_valid_index() == 21
    assert calculate_rsi(series, 40).isna().all()   # longer than the series


def test_rsi_returns_all_nan_when_data_is_too_short():
    assert calculate_rsi(pd.Series([1.0, 2.0, 3.0]), 14).isna().all()


# ============================================================================
# MACD
# ============================================================================


def test_macd_line_is_the_difference_of_its_emas():
    close = make_bars(200, seed=3)["close"]
    macd = calculate_macd(close, 12, 26, 9)
    expected = calculate_ema(close, 12) - calculate_ema(close, 26)
    pd.testing.assert_series_equal(
        macd[MACD_COL].dropna(), expected.dropna(), check_names=False
    )


def test_macd_histogram_is_line_minus_signal():
    macd = calculate_macd(make_bars(200, seed=4)["close"])
    computed = macd[MACD_COL] - macd[MACD_SIGNAL_COL]
    pd.testing.assert_series_equal(
        macd[MACD_HISTOGRAM_COL].dropna(), computed.dropna(), check_names=False
    )


def test_macd_returns_the_three_expected_columns():
    macd = calculate_macd(make_bars(120, seed=5)["close"])
    assert list(macd.columns) == [MACD_COL, MACD_SIGNAL_COL, MACD_HISTOGRAM_COL]


def test_macd_is_positive_in_an_uptrend():
    rising = pd.Series([100.0 * (1.01**index) for index in range(120)])
    assert calculate_macd(rising)[MACD_COL].dropna().iloc[-1] > 0


def test_macd_rejects_fast_period_not_below_slow():
    with pytest.raises(ValueError, match="must be less than"):
        calculate_macd(pd.Series([1.0] * 50), fast_period=26, slow_period=12)


def test_macd_periods_are_configurable():
    close = make_bars(200, seed=6)["close"]
    default = calculate_macd(close)
    faster = calculate_macd(close, 5, 13, 4)
    assert not np.allclose(
        default[MACD_COL].dropna().tail(10), faster[MACD_COL].dropna().tail(10)
    )


# ============================================================================
# Bollinger Bands
# ============================================================================


def test_bollinger_middle_band_is_the_sma():
    close = make_bars(100, seed=8)["close"]
    bands = calculate_bollinger_bands(close, 20, 2.0)
    pd.testing.assert_series_equal(
        bands[BB_MIDDLE_COL].dropna(), calculate_sma(close, 20).dropna(), check_names=False
    )


def test_bollinger_uses_the_population_standard_deviation():
    """pandas defaults to ddof=1, which would widen the bands."""
    close = pd.Series(WILDER_CLOSES)
    bands = calculate_bollinger_bands(close, 20, 2.0)
    window = close.iloc[:20]
    assert bands[BB_UPPER_COL].iloc[19] == pytest.approx(
        window.mean() + 2 * window.std(ddof=0)
    )
    assert bands[BB_UPPER_COL].iloc[19] != pytest.approx(
        window.mean() + 2 * window.std(ddof=1)
    )


def test_bollinger_bands_are_ordered():
    bands = calculate_bollinger_bands(make_bars(100, seed=9)["close"]).dropna()
    assert (bands[BB_UPPER_COL] >= bands[BB_MIDDLE_COL]).all()
    assert (bands[BB_MIDDLE_COL] >= bands[BB_LOWER_COL]).all()


def test_bollinger_bands_collapse_on_a_constant_series():
    bands = calculate_bollinger_bands(pd.Series([100.0] * 40), 20, 2.0).dropna()
    assert bands[BB_UPPER_COL].eq(100.0).all()
    assert bands[BB_WIDTH_COL].eq(0.0).all()


def test_bollinger_percent_b_locates_price_within_the_bands():
    close = make_bars(100, seed=10)["close"]
    bands = calculate_bollinger_bands(close, 20, 2.0).dropna()
    recomputed = (close.loc[bands.index] - bands[BB_LOWER_COL]) / (
        bands[BB_UPPER_COL] - bands[BB_LOWER_COL]
    )
    pd.testing.assert_series_equal(
        bands[BB_PERCENT_B_COL], recomputed, check_names=False
    )


def test_wider_std_multiplier_widens_the_bands():
    close = make_bars(100, seed=11)["close"]
    narrow = calculate_bollinger_bands(close, 20, 1.0).dropna()
    wide = calculate_bollinger_bands(close, 20, 3.0).dropna()
    assert (wide[BB_UPPER_COL] >= narrow[BB_UPPER_COL]).all()


def test_bollinger_rejects_non_positive_std():
    with pytest.raises(ValueError, match="num_std must be > 0"):
        calculate_bollinger_bands(pd.Series([1.0] * 30), 20, 0.0)


# ============================================================================
# ATR
# ============================================================================


def reference_atr(highs, lows, closes, period):
    """ATR computed by an explicit Wilder loop."""
    true_ranges = [None]
    for index in range(1, len(highs)):
        true_ranges.append(
            max(
                highs[index] - lows[index],
                abs(highs[index] - closes[index - 1]),
                abs(lows[index] - closes[index - 1]),
            )
        )
    out = [None] * len(highs)
    previous = sum(true_ranges[1 : period + 1]) / period
    out[period] = previous
    for index in range(period + 1, len(highs)):
        previous = (previous * (period - 1) + true_ranges[index]) / period
        out[index] = previous
    return out


def test_atr_matches_an_independent_wilder_implementation():
    bars = make_bars(80, seed=21)
    expected = reference_atr(
        bars["high"].tolist(), bars["low"].tolist(), bars["close"].tolist(), 14
    )
    result = calculate_atr(bars["high"], bars["low"], bars["close"], 14)
    for index, want in enumerate(expected):
        if want is None:
            assert pd.isna(result.iloc[index])
        else:
            assert result.iloc[index] == pytest.approx(want, abs=1e-10)


def test_true_range_is_undefined_on_the_first_bar():
    """A true range needs a previous close, which the first bar does not have."""
    bars = make_bars(10, seed=22)
    true_range = calculate_true_range(bars["high"], bars["low"], bars["close"])
    assert pd.isna(true_range.iloc[0])
    assert true_range.iloc[1:].notna().all()


def test_true_range_captures_an_overnight_gap():
    """A gap beyond the bar's own span must dominate the high-low range."""
    frame = pd.DataFrame(
        {"high": [10.0, 20.0], "low": [9.0, 19.0], "close": [9.5, 19.5]}
    )
    true_range = calculate_true_range(frame["high"], frame["low"], frame["close"])
    assert true_range.iloc[1] == pytest.approx(10.5)  # |20 - 9.5| beats 20 - 19


def test_atr_first_value_appears_at_index_period():
    """One bar is lost to the undefined first true range, then 14 are smoothed."""
    bars = make_bars(60, seed=23)
    atr = calculate_atr(bars["high"], bars["low"], bars["close"], 14)
    assert atr.iloc[:14].isna().all()
    assert not pd.isna(atr.iloc[14])


def test_atr_is_never_negative():
    bars = make_bars(200, seed=24)
    assert (calculate_atr(bars["high"], bars["low"], bars["close"], 14).dropna() >= 0).all()


def test_atr_rises_with_volatility():
    calm = ohlcv([100.0 + 0.05 * index for index in range(60)])
    wild = calm.copy()
    wild["high"] = wild["close"] * 1.10
    wild["low"] = wild["close"] * 0.90
    calm_atr = calculate_atr(calm["high"], calm["low"], calm["close"], 14).iloc[-1]
    wild_atr = calculate_atr(wild["high"], wild["low"], wild["close"], 14).iloc[-1]
    assert wild_atr > calm_atr


# ============================================================================
# Volume
# ============================================================================


def test_volume_sma_matches_a_hand_computed_mean():
    volume = pd.Series([100.0, 200.0, 300.0, 400.0])
    assert calculate_volume_sma(volume, 2).iloc[1] == pytest.approx(150.0)


def test_relative_volume_is_one_for_constant_volume():
    result = calculate_relative_volume(pd.Series([1000.0] * 30), 20).dropna()
    assert result.eq(1.0).all()


def test_relative_volume_reports_a_doubling():
    volume = pd.Series([1000.0] * 20 + [2000.0])
    assert calculate_relative_volume(volume, 20).iloc[-1] == pytest.approx(2000 / 1050)


def test_relative_volume_is_nan_when_average_volume_is_zero():
    assert calculate_relative_volume(pd.Series([0.0] * 25), 20).dropna().empty


# ============================================================================
# calculate_all_indicators
# ============================================================================


def test_all_indicators_produces_the_documented_columns():
    result = calculate_all_indicators(make_bars(300, seed=30))
    for column in (
        "SMA_20", "SMA_50", "SMA_100", "SMA_200",
        "EMA_9", "EMA_20", "EMA_50", "EMA_200",
        "RSI_14", MACD_COL, MACD_SIGNAL_COL, MACD_HISTOGRAM_COL,
        BB_UPPER_COL, BB_MIDDLE_COL, BB_LOWER_COL, BB_WIDTH_COL, BB_PERCENT_B_COL,
        "ATR_14", "ATR_14_PCT", "VOLUME_SMA_20", RELATIVE_VOLUME_COL,
    ):
        assert column in result.columns, f"missing {column}"


def test_all_indicators_preserves_the_input_columns():
    bars = make_bars(100, seed=31)
    result = calculate_all_indicators(bars)
    for column in bars.columns:
        pd.testing.assert_series_equal(result[column], bars[column])


def test_all_indicators_does_not_mutate_the_input():
    bars = make_bars(100, seed=32)
    before = bars.copy()
    calculate_all_indicators(bars)
    pd.testing.assert_frame_equal(bars, before)


def test_all_indicators_preserves_the_index():
    bars = make_bars(120, seed=33)
    result = calculate_all_indicators(bars)
    pd.testing.assert_index_equal(result.index, bars.index)


def test_all_indicators_accepts_uppercase_column_names():
    bars = make_bars(60, seed=34).rename(columns=str.upper)
    result = calculate_all_indicators(bars)
    assert "RSI_14" in result.columns
    assert "close" in result.columns


def test_all_indicators_honours_a_custom_config():
    config = IndicatorConfig(ema_periods=(5, 13), sma_periods=(10,), rsi_period=7)
    result = calculate_all_indicators(make_bars(100, seed=35), config)
    assert {"EMA_5", "EMA_13", "SMA_10", "RSI_7"} <= set(result.columns)
    assert "EMA_200" not in result.columns


def test_short_history_yields_nan_columns_rather_than_wrong_numbers():
    """EMA-200 from 60 bars is not an approximation, it is a wrong number."""
    result = calculate_all_indicators(make_bars(60, seed=36))
    assert result["EMA_200"].isna().all()
    assert result["RSI_14"].notna().any()   # short indicators still work


def test_strict_mode_rejects_insufficient_history():
    with pytest.raises(InsufficientDataError, match="needs"):
        calculate_all_indicators(make_bars(60, seed=37), strict=True)


def test_strict_mode_accepts_sufficient_history():
    assert not calculate_all_indicators(make_bars(250, seed=38), strict=True).empty


def test_all_indicators_works_across_timeframes():
    """Nothing in the engine assumes a particular bar size."""
    for freq in ("1min", "5min", "15min", "1h", "1D"):
        bars = make_bars(260, freq=freq, seed=39)
        result = calculate_all_indicators(bars)
        assert result["RSI_14"].notna().any(), freq
        assert result["EMA_200"].notna().any(), freq


def test_indicators_use_only_past_and_present_data():
    """The anti-lookahead guarantee.

    Computing on a truncated history must reproduce the values computed on the
    full history. If any indicator peeked ahead, these would differ.
    """
    bars = make_bars(300, seed=40)
    cutoff = 250
    full = calculate_all_indicators(bars)
    partial = calculate_all_indicators(bars.iloc[:cutoff])

    for column in ("EMA_20", "SMA_50", "RSI_14", "ATR_14", MACD_COL, BB_UPPER_COL):
        expected = full[column].iloc[:cutoff]
        pd.testing.assert_series_equal(partial[column], expected, check_names=False)


def test_latest_values_converts_nan_to_none():
    result = calculate_all_indicators(make_bars(60, seed=41))
    values = latest_values(result, ["close", "RSI_14", "EMA_200"])
    assert isinstance(values["close"], float)
    assert values["EMA_200"] is None


def test_latest_values_rejects_an_empty_frame():
    with pytest.raises(InvalidDataError):
        latest_values(pd.DataFrame())


# ============================================================================
# Validation
# ============================================================================


def test_validation_rejects_a_non_dataframe():
    with pytest.raises(InvalidDataError, match="must be a pandas DataFrame"):
        validate_ohlcv([1, 2, 3])


def test_validation_rejects_an_empty_frame():
    with pytest.raises(InvalidDataError, match="is empty"):
        validate_ohlcv(pd.DataFrame())


def test_validation_names_the_missing_columns():
    bars = make_bars(50, seed=42).drop(columns=["volume"])
    with pytest.raises(InvalidDataError, match="missing required column"):
        validate_ohlcv(bars)


def test_validation_rejects_unsorted_data():
    bars = make_bars(50, seed=43).iloc[::-1]
    with pytest.raises(InvalidDataError, match="not sorted"):
        validate_ohlcv(bars)


def test_validation_rejects_duplicate_timestamps():
    bars = make_bars(50, seed=44)
    duplicated = pd.concat([bars, bars.iloc[[10]]]).sort_index()
    with pytest.raises(InvalidDataError, match="duplicate timestamps"):
        validate_ohlcv(duplicated)


def test_validation_rejects_missing_values():
    bars = make_bars(50, seed=45)
    bars.iloc[5, bars.columns.get_loc("close")] = np.nan
    with pytest.raises(InvalidDataError, match="missing value"):
        validate_ohlcv(bars)


def test_validation_rejects_infinite_values():
    bars = make_bars(50, seed=46)
    bars.iloc[5, bars.columns.get_loc("high")] = np.inf
    with pytest.raises(InvalidDataError, match="infinite"):
        validate_ohlcv(bars)


def test_validation_rejects_non_positive_prices():
    bars = make_bars(50, seed=47)
    bars.iloc[5, bars.columns.get_loc("low")] = -1.0
    with pytest.raises(InvalidDataError, match="non-positive"):
        validate_ohlcv(bars)


def test_validation_rejects_negative_volume():
    bars = make_bars(50, seed=48)
    bars.iloc[5, bars.columns.get_loc("volume")] = -100.0
    with pytest.raises(InvalidDataError, match="negative"):
        validate_ohlcv(bars)


def test_validation_rejects_a_non_numeric_column():
    bars = make_bars(50, seed=49)
    bars["close"] = bars["close"].astype(str)
    with pytest.raises(InvalidDataError, match="must be numeric"):
        validate_ohlcv(bars)


def test_validation_enforces_a_minimum_row_count():
    with pytest.raises(InsufficientDataError, match="30 are required"):
        validate_ohlcv(make_bars(10, seed=50), min_rows=30)


def test_validation_accepts_a_well_formed_frame():
    validate_ohlcv(make_bars(50, seed=51), min_rows=50)


# ============================================================================
# Configuration
# ============================================================================


def test_config_rejects_inverted_macd_periods():
    with pytest.raises(ValueError, match="macd_fast"):
        IndicatorConfig(macd_fast=26, macd_slow=12)


def test_config_rejects_inverted_rsi_thresholds():
    with pytest.raises(ValueError, match="rsi_oversold"):
        IndicatorConfig(rsi_oversold=80.0, rsi_overbought=20.0)


def test_config_rejects_non_positive_periods():
    with pytest.raises(ValueError, match="rsi_period"):
        IndicatorConfig(rsi_period=0)


def test_config_rejects_inverted_volume_thresholds():
    with pytest.raises(ValueError, match="volume_low_threshold"):
        IndicatorConfig(volume_low_threshold=2.0, volume_high_threshold=1.0)


def test_config_rejects_a_spike_threshold_below_the_high_threshold():
    with pytest.raises(ValueError, match="volume_spike_threshold"):
        IndicatorConfig(volume_high_threshold=2.0, volume_spike_threshold=1.5)


def test_config_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        IndicatorConfig().rsi_period = 7


def test_config_with_overrides_returns_a_validated_copy():
    config = IndicatorConfig().with_overrides(rsi_period=7)
    assert config.rsi_period == 7
    assert IndicatorConfig().rsi_period == 14
    with pytest.raises(ValueError):
        IndicatorConfig().with_overrides(rsi_period=-1)


def test_max_lookback_covers_the_longest_indicator():
    assert IndicatorConfig().max_lookback == 200
    assert IndicatorConfig(ema_periods=(9,), sma_periods=(20,)).max_lookback == 35


# ============================================================================
# Signal helpers
# ============================================================================


def test_ema_crossover_detects_a_bullish_cross():
    """After a flat stretch both EMAs sit at the same value; one up bar separates them.

    The cross must land on the *final* bar, since that is the only bar the
    detector inspects.
    """
    closes = [100.0] * 40 + [200.0]
    result = detect_ema_crossover(calculate_all_indicators(ohlcv(closes)), 9, 20)
    assert result.signal == "BULLISH_CROSSOVER"
    assert result.occurred
    assert result.fast_ema > result.slow_ema


def test_ema_crossover_reports_none_for_an_established_trend():
    """A long-standing trend is not a crossover; reporting it would fire daily."""
    closes = [100.0 + 2 * index for index in range(80)]
    result = detect_ema_crossover(calculate_all_indicators(ohlcv(closes)), 9, 20)
    assert result.signal == "NONE"
    assert not result.occurred


def test_ema_crossover_detects_a_bearish_cross():
    closes = [100.0] * 40 + [50.0]
    result = detect_ema_crossover(calculate_all_indicators(ohlcv(closes)), 9, 20)
    assert result.signal == "BEARISH_CROSSOVER"
    assert result.fast_ema < result.slow_ema


def test_ema_crossover_is_safe_during_warmup():
    result = detect_ema_crossover(calculate_all_indicators(make_bars(15, seed=60)), 9, 20)
    assert result.signal == "NONE"
    assert result.fast_ema is None


def test_ema_crossover_rejects_inverted_periods():
    with pytest.raises(ValueError, match="must be less than"):
        detect_ema_crossover(make_bars(50, seed=61), 20, 9)


def test_ema_crossover_as_dict_matches_the_documented_shape():
    payload = detect_ema_crossover(calculate_all_indicators(make_bars(120, seed=62))).as_dict()
    assert set(payload) >= {"signal", "occurred", "fast_ema", "slow_ema"}


def test_rsi_condition_classifies_the_three_states():
    assert detect_rsi_condition(15.0) == "OVERSOLD"
    assert detect_rsi_condition(50.0) == "NEUTRAL"
    assert detect_rsi_condition(85.0) == "OVERBOUGHT"


def test_rsi_condition_thresholds_are_configurable():
    tight = IndicatorConfig(rsi_oversold=45.0, rsi_overbought=55.0)
    assert detect_rsi_condition(40.0, tight) == "OVERSOLD"
    assert detect_rsi_condition(40.0) == "NEUTRAL"


def test_rsi_condition_is_neutral_when_unavailable():
    assert detect_rsi_condition(float("nan")) == "NEUTRAL"
    assert detect_rsi_condition(calculate_all_indicators(make_bars(10, seed=63))) == "NEUTRAL"


def test_rsi_condition_reads_the_frame_column():
    rising = ohlcv([100.0 + index for index in range(40)])
    assert detect_rsi_condition(calculate_all_indicators(rising)) == "OVERBOUGHT"


def test_macd_signal_follows_momentum_turns():
    """MACD compares the line to its own average, so it tracks momentum shifts."""
    rally = ohlcv([100.0] * 60 + [100.0 * (1.02**index) for index in range(1, 40)])
    crash = ohlcv(
        [100.0 + index for index in range(100)]
        + [200.0 - 6 * index for index in range(1, 20)]
    )
    assert detect_macd_signal(calculate_all_indicators(rally)) == "BULLISH"
    assert detect_macd_signal(calculate_all_indicators(crash)) == "BEARISH"


def test_macd_histogram_turns_positive_as_an_exponential_decay_flattens():
    """A property worth pinning down, because it looks like a bug.

    On a constant-rate decay the gap between the fast and slow EMAs shrinks in
    absolute terms as price approaches zero, so the MACD line rises toward zero
    from below and crosses above its lagging signal line. MACD reports the
    *change* in momentum, not the direction of price.
    """
    decay = calculate_all_indicators(ohlcv([100.0 * (0.99**index) for index in range(120)]))
    assert decay[MACD_COL].dropna().iloc[-1] < 0        # still below zero: downtrend
    assert detect_macd_signal(decay) == "BULLISH"       # but decelerating


def test_macd_signal_is_neutral_during_warmup():
    assert detect_macd_signal(calculate_all_indicators(make_bars(10, seed=64))) == "NEUTRAL"


def test_macd_momentum_detects_acceleration():
    accelerating = ohlcv([100.0 * (1.015**index) for index in range(120)])
    assert detect_macd_momentum(calculate_all_indicators(accelerating)) == "INCREASING"


def test_macd_momentum_is_flat_without_enough_history():
    assert detect_macd_momentum(calculate_all_indicators(make_bars(20, seed=65))) == "FLAT"


def test_bollinger_condition_detects_a_break_above_the_upper_band():
    closes = [100.0] * 30 + [140.0]
    assert detect_bollinger_condition(calculate_all_indicators(ohlcv(closes))) == "ABOVE_UPPER"


def test_bollinger_condition_detects_a_break_below_the_lower_band():
    closes = [100.0] * 30 + [60.0]
    assert detect_bollinger_condition(calculate_all_indicators(ohlcv(closes))) == "BELOW_LOWER"


def test_bollinger_condition_is_normal_inside_the_bands():
    bars = calculate_all_indicators(make_bars(200, seed=66))
    assert detect_bollinger_condition(bars) in ("NORMAL", "SQUEEZE")
