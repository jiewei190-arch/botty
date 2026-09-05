"""Technical indicator calculations.

This module is the single source of indicator math for the whole bot. Strategies
never recompute an indicator themselves — they call
:func:`calculate_all_indicators` once and read the resulting columns, so a
backtest and the live scanner cannot silently disagree about what "RSI 14" means.

Conventions
-----------
These choices decide whether values match a charting platform. They are fixed
here and documented so nothing downstream has to guess:

* **EMA** is seeded with the SMA of the first ``period`` values, then smoothed
  recursively with ``alpha = 2 / (period + 1)`` — the TA-Lib and TradingView
  convention. A plain ``ewm(adjust=False)`` seeds with the *first value*, which
  produces visibly different results for the first few dozen bars.
* **RSI and ATR** use Wilder's smoothing (``alpha = 1 / period``), not a simple
  moving average. This is what "RSI 14" means everywhere it is quoted.
* **True Range** requires a previous close, so it is undefined on the first bar.
  ATR therefore produces its first value at index ``period``, matching TA-Lib.
* **Bollinger Bands** use the *population* standard deviation (``ddof=0``).
  pandas defaults to the sample deviation (``ddof=1``), which would make the
  bands slightly too wide.

Warm-up
-------
Every indicator returns ``NaN`` until it has enough history to be correct. An
EMA-200 computed from 30 bars is not a rough EMA-200, it is a wrong number, and
a strategy acting on it would be trading noise. Callers must handle ``NaN``;
:func:`latest_values` and the analysis modules do.

No lookahead
------------
Every calculation here uses only the current bar and earlier ones. Rolling
windows are trailing and never centred. Combined with the Phase 1 guarantee that
forming bars are dropped before analysis, a value at bar ``t`` is knowable at
bar ``t``.

Example
-------
>>> from trading_bot.indicators import calculate_all_indicators
>>> enriched = calculate_all_indicators(bars)
>>> enriched[["close", "EMA_20", "RSI_14", "ATR_14"]].tail(1)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Columns an input frame must provide (the Phase 1 normalized bar schema).
REQUIRED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

# Column names for indicators that do not vary with a period.
MACD_COL = "MACD"
MACD_SIGNAL_COL = "MACD_SIGNAL"
MACD_HISTOGRAM_COL = "MACD_HISTOGRAM"
BB_UPPER_COL = "BB_UPPER"
BB_MIDDLE_COL = "BB_MIDDLE"
BB_LOWER_COL = "BB_LOWER"
BB_WIDTH_COL = "BB_WIDTH"
BB_PERCENT_B_COL = "BB_PERCENT_B"
RELATIVE_VOLUME_COL = "RELATIVE_VOLUME"


class IndicatorError(Exception):
    """Base class for indicator-engine failures."""


class InvalidDataError(IndicatorError):
    """Input data is missing, malformed, or violates the expected schema."""


class InsufficientDataError(IndicatorError):
    """Not enough history to compute the requested indicator correctly."""


# -- naming --------------------------------------------------------------------


def sma_column(period: int) -> str:
    """Column name for a simple moving average, e.g. ``SMA_20``."""
    return f"SMA_{period}"


def ema_column(period: int) -> str:
    """Column name for an exponential moving average, e.g. ``EMA_9``."""
    return f"EMA_{period}"


def rsi_column(period: int) -> str:
    """Column name for RSI, e.g. ``RSI_14``."""
    return f"RSI_{period}"


def atr_column(period: int) -> str:
    """Column name for ATR, e.g. ``ATR_14``."""
    return f"ATR_{period}"


def atr_percent_column(period: int) -> str:
    """Column name for ATR expressed as a percentage of price, e.g. ``ATR_14_PCT``."""
    return f"ATR_{period}_PCT"


def volume_sma_column(period: int) -> str:
    """Column name for a volume moving average, e.g. ``VOLUME_SMA_20``."""
    return f"VOLUME_SMA_{period}"


# -- configuration -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IndicatorConfig:
    """Parameters for the indicator engine.

    Every period and threshold the bot uses lives here, so a strategy, a
    backtest parameter sweep or the dashboard can vary them without touching
    calculation code. Instances are frozen; use :meth:`with_overrides` to derive
    a variant.

    Example
    -------
    >>> fast = IndicatorConfig().with_overrides(rsi_period=7, rsi_oversold=25.0)
    >>> fast.rsi_period
    7
    """

    # Trend
    sma_periods: tuple[int, ...] = (20, 50, 100, 200)
    ema_periods: tuple[int, ...] = (9, 20, 50, 200)

    # Momentum
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Volatility
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    #: Bandwidth below this percentile of its recent range counts as a squeeze.
    squeeze_percentile: float = 25.0
    squeeze_lookback: int = 120
    atr_period: int = 14

    # Volume — multiples of the volume moving average.
    volume_sma_period: int = 20
    volume_low_threshold: float = 0.7
    volume_high_threshold: float = 1.5
    volume_spike_threshold: float = 2.5

    # Price action
    #: Bars either side of a pivot required to confirm a swing point.
    swing_strength: int = 3
    #: Bars scanned when collecting support and resistance levels.
    level_lookback: int = 120
    #: Pivots within this percentage of each other merge into one level.
    level_tolerance_pct: float = 0.75

    def __post_init__(self) -> None:
        for name, value in (
            ("rsi_period", self.rsi_period),
            ("macd_fast", self.macd_fast),
            ("macd_slow", self.macd_slow),
            ("macd_signal", self.macd_signal),
            ("bollinger_period", self.bollinger_period),
            ("atr_period", self.atr_period),
            ("volume_sma_period", self.volume_sma_period),
            ("swing_strength", self.swing_strength),
            ("level_lookback", self.level_lookback),
        ):
            if value < 1:
                raise ValueError(f"IndicatorConfig.{name} must be >= 1, got {value}")
        if not self.sma_periods and not self.ema_periods:
            raise ValueError("IndicatorConfig needs at least one moving-average period")
        for period in (*self.sma_periods, *self.ema_periods):
            if period < 1:
                raise ValueError(f"Moving-average periods must be >= 1, got {period}")
        if self.macd_fast >= self.macd_slow:
            raise ValueError(
                f"macd_fast ({self.macd_fast}) must be less than macd_slow ({self.macd_slow})"
            )
        if not 0 <= self.rsi_oversold < self.rsi_overbought <= 100:
            raise ValueError(
                "Require 0 <= rsi_oversold < rsi_overbought <= 100, got "
                f"{self.rsi_oversold} and {self.rsi_overbought}"
            )
        if self.bollinger_std <= 0:
            raise ValueError(f"bollinger_std must be > 0, got {self.bollinger_std}")
        if not 0 < self.volume_low_threshold < self.volume_high_threshold:
            raise ValueError(
                "Require 0 < volume_low_threshold < volume_high_threshold, got "
                f"{self.volume_low_threshold} and {self.volume_high_threshold}"
            )
        if self.volume_spike_threshold <= self.volume_high_threshold:
            raise ValueError(
                "volume_spike_threshold must exceed volume_high_threshold, got "
                f"{self.volume_spike_threshold} and {self.volume_high_threshold}"
            )
        if not 0 < self.squeeze_percentile < 100:
            raise ValueError(
                f"squeeze_percentile must be between 0 and 100, got {self.squeeze_percentile}"
            )
        if self.level_tolerance_pct <= 0:
            raise ValueError(
                f"level_tolerance_pct must be > 0, got {self.level_tolerance_pct}"
            )

    def with_overrides(self, **overrides: Any) -> IndicatorConfig:
        """Return a copy with the given fields replaced (validated on construction)."""
        return replace(self, **overrides)

    @property
    def max_lookback(self) -> int:
        """Bars needed before every configured indicator has a value.

        Useful for sizing a warm-up window: request at least this many extra bars
        before the period a strategy actually wants to trade.
        """
        return max(
            (
                *self.sma_periods,
                *self.ema_periods,
                self.rsi_period + 1,
                self.macd_slow + self.macd_signal,
                self.bollinger_period,
                self.atr_period + 1,
                self.volume_sma_period,
                2 * self.swing_strength + 1,
            )
        )


DEFAULT_CONFIG = IndicatorConfig()


# -- validation ----------------------------------------------------------------


def validate_ohlcv(
    data: pd.DataFrame,
    *,
    min_rows: int | None = None,
    require_index: bool = True,
    name: str = "data",
) -> None:
    """Validate an OHLCV frame, raising a specific error on the first problem.

    Checks, in order: the object is a non-empty DataFrame; required columns are
    present; the index is a sorted, duplicate-free ``DatetimeIndex``; price and
    volume columns are numeric, finite and non-negative; and enough rows exist.

    Parameters
    ----------
    data:
        Frame to check.
    min_rows:
        Minimum number of rows required. Raises :class:`InsufficientDataError`
        when there are fewer.
    require_index:
        Enforce the ``DatetimeIndex`` contract. Set False for frames indexed by
        position (rare — Phase 1 providers always supply timestamps).
    name:
        Label used in error messages.

    Raises
    ------
    InvalidDataError
        The frame is unusable.
    InsufficientDataError
        The frame is well-formed but too short.

    Example
    -------
    >>> validate_ohlcv(bars, min_rows=200)
    """
    if not isinstance(data, pd.DataFrame):
        raise InvalidDataError(f"{name} must be a pandas DataFrame, got {type(data).__name__}")
    if data.empty:
        raise InvalidDataError(f"{name} is empty — no bars to analyse")

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise InvalidDataError(
            f"{name} is missing required column(s): {missing}. "
            f"Present columns: {list(data.columns)}"
        )

    if require_index:
        if not isinstance(data.index, pd.DatetimeIndex):
            raise InvalidDataError(
                f"{name} must be indexed by timestamp, got {type(data.index).__name__}"
            )
        if not data.index.is_monotonic_increasing:
            raise InvalidDataError(
                f"{name} is not sorted by timestamp. Indicators assume chronological "
                "order; sort with data.sort_index() first."
            )
        if data.index.has_duplicates:
            duplicates = data.index[data.index.duplicated()].unique()[:3].tolist()
            raise InvalidDataError(
                f"{name} contains duplicate timestamps (e.g. {duplicates}). "
                "Deduplicate before calculating indicators."
            )

    for column in REQUIRED_COLUMNS:
        series = data[column]
        if not pd.api.types.is_numeric_dtype(series):
            raise InvalidDataError(
                f"{name}['{column}'] must be numeric, got dtype {series.dtype}"
            )
        if series.isna().any():
            count = int(series.isna().sum())
            raise InvalidDataError(
                f"{name}['{column}'] contains {count} missing value(s). "
                "Indicators would propagate them silently; clean the data first."
            )
        if np.isinf(series.to_numpy(dtype="float64")).any():
            raise InvalidDataError(f"{name}['{column}'] contains infinite values")

    for column in ("open", "high", "low", "close"):
        if (data[column] <= 0).any():
            raise InvalidDataError(f"{name}['{column}'] contains non-positive prices")
    if (data["volume"] < 0).any():
        raise InvalidDataError(f"{name}['volume'] contains negative values")

    if min_rows is not None and len(data) < min_rows:
        raise InsufficientDataError(
            f"{name} has {len(data)} bar(s) but {min_rows} are required. "
            "Request a longer history or reduce the indicator period."
        )


def _as_float_series(values: pd.Series | pd.DataFrame, name: str) -> pd.Series:
    """Coerce input to a float Series, with a clear error for bad input."""
    if isinstance(values, pd.DataFrame):
        raise InvalidDataError(f"{name} must be a Series, got a DataFrame")
    if not isinstance(values, pd.Series):
        raise InvalidDataError(f"{name} must be a pandas Series, got {type(values).__name__}")
    if values.empty:
        raise InvalidDataError(f"{name} is empty")
    return values.astype("float64")


def _empty_like(series: pd.Series) -> pd.Series:
    """An all-NaN float Series sharing ``series``'s index."""
    return pd.Series(np.nan, index=series.index, dtype="float64")


# -- smoothing primitives ------------------------------------------------------


def _seeded_recursive_mean(series: pd.Series, period: int, alpha: float) -> pd.Series:
    """Recursive smoothing seeded with the SMA of the first ``period`` values.

    Values before the seed are ``NaN``. Implemented by seeding the first element
    of the tail and delegating to ``ewm``, so it stays vectorised rather than
    looping bar by bar.
    """
    values = series.dropna()
    if len(values) < period:
        return _empty_like(series)

    tail = values.iloc[period - 1 :].copy()
    tail.iloc[0] = values.iloc[:period].mean()
    smoothed = tail.ewm(alpha=alpha, adjust=False).mean()
    return smoothed.reindex(series.index)


def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (``alpha = 1 / period``), the basis of RSI and ATR.

    Parameters
    ----------
    series:
        Values to smooth.
    period:
        Averaging period; must be >= 1.

    Returns
    -------
    pandas.Series
        Smoothed values, ``NaN`` until ``period`` inputs are available.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    values = _as_float_series(series, "series")
    return _seeded_recursive_mean(values, period, alpha=1.0 / period)


# -- trend indicators ----------------------------------------------------------


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average.

    Parameters
    ----------
    series:
        Values to average, typically ``data["close"]``.
    period:
        Window length in bars.

    Returns
    -------
    pandas.Series
        Rolling mean, ``NaN`` for the first ``period - 1`` bars.

    Example
    -------
    >>> data["SMA_20"] = calculate_sma(data["close"], 20)
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    values = _as_float_series(series, "series")
    return values.rolling(window=period, min_periods=period).mean()


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average, seeded with an SMA (TA-Lib convention).

    Parameters
    ----------
    series:
        Values to average, typically ``data["close"]``.
    period:
        Smoothing period; ``alpha = 2 / (period + 1)``.

    Returns
    -------
    pandas.Series
        EMA values, ``NaN`` for the first ``period - 1`` bars.

    Notes
    -----
    Seeding with the SMA rather than the first observation is what makes these
    values line up with TradingView and TA-Lib. The difference decays but is
    material over the first few multiples of ``period``.

    Example
    -------
    >>> data["EMA_20"] = calculate_ema(data["close"], 20)
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    values = _as_float_series(series, "series")
    return _seeded_recursive_mean(values, period, alpha=2.0 / (period + 1))


# -- momentum indicators -------------------------------------------------------


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing.

    Parameters
    ----------
    close:
        Closing prices.
    period:
        Averaging period (default 14).

    Returns
    -------
    pandas.Series
        RSI in ``[0, 100]``. The first value appears at index ``period``, since
        ``period`` price *changes* are needed.

    Notes
    -----
    Edge cases are resolved explicitly rather than left as division by zero:
    an unbroken run of gains gives 100, an unbroken run of losses gives 0, and a
    perfectly flat window gives 50.

    Example
    -------
    >>> data["RSI_14"] = calculate_rsi(data["close"], 14)
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    values = _as_float_series(close, "close")
    if len(values) <= period:
        return _empty_like(values)

    delta = values.diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)

    avg_gain = wilder_smooth(gains.iloc[1:], period).reindex(values.index)
    avg_loss = wilder_smooth(losses.iloc[1:], period).reindex(values.index)

    rsi = pd.Series(np.nan, index=values.index, dtype="float64")
    computed = avg_gain.notna() & avg_loss.notna()

    # Normal case: both averages positive.
    normal = computed & (avg_loss > 0)
    rs = avg_gain[normal] / avg_loss[normal]
    rsi[normal] = 100.0 - (100.0 / (1.0 + rs))

    # Degenerate cases, resolved by definition rather than by 0/0.
    all_gains = computed & (avg_loss == 0) & (avg_gain > 0)
    all_losses = computed & (avg_gain == 0) & (avg_loss > 0)
    flat = computed & (avg_gain == 0) & (avg_loss == 0)
    rsi[all_gains] = 100.0
    rsi[all_losses] = 0.0
    rsi[flat] = 50.0
    return rsi


def calculate_macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """Moving Average Convergence Divergence.

    Parameters
    ----------
    close:
        Closing prices.
    fast_period, slow_period:
        EMA periods whose difference forms the MACD line (defaults 12 and 26).
    signal_period:
        EMA period of the MACD line (default 9).

    Returns
    -------
    pandas.DataFrame
        Columns ``MACD``, ``MACD_SIGNAL`` and ``MACD_HISTOGRAM``
        (``MACD - MACD_SIGNAL``).

    Example
    -------
    >>> macd = calculate_macd(data["close"])
    >>> data = data.join(macd)
    """
    if fast_period >= slow_period:
        raise ValueError(
            f"fast_period ({fast_period}) must be less than slow_period ({slow_period})"
        )
    values = _as_float_series(close, "close")

    macd_line = calculate_ema(values, fast_period) - calculate_ema(values, slow_period)

    # The signal line is an EMA *of the MACD line*, so it must be computed on the
    # MACD's own warm-up-trimmed values. With too little history that leaves
    # nothing to average, which is a valid warm-up state rather than an error.
    defined = macd_line.dropna()
    if defined.empty:
        signal_line = pd.Series(np.nan, index=values.index, dtype="float64")
    else:
        signal_line = calculate_ema(defined, signal_period).reindex(values.index)
    return pd.DataFrame(
        {
            MACD_COL: macd_line,
            MACD_SIGNAL_COL: signal_line,
            MACD_HISTOGRAM_COL: macd_line - signal_line,
        },
        index=values.index,
    )


# -- volatility indicators -----------------------------------------------------


def calculate_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands with the population standard deviation.

    Parameters
    ----------
    close:
        Closing prices.
    period:
        Moving-average window (default 20).
    num_std:
        Band distance in standard deviations (default 2.0).

    Returns
    -------
    pandas.DataFrame
        ``BB_MIDDLE``, ``BB_UPPER``, ``BB_LOWER``, ``BB_WIDTH`` (band span as a
        fraction of the middle band) and ``BB_PERCENT_B`` (position of price
        within the bands: 0 at the lower band, 1 at the upper).

    Notes
    -----
    ``ddof=0`` is deliberate — charting platforms use the population deviation,
    and pandas defaults to the sample deviation, which widens the bands.

    Example
    -------
    >>> data = data.join(calculate_bollinger_bands(data["close"]))
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    if num_std <= 0:
        raise ValueError(f"num_std must be > 0, got {num_std}")
    values = _as_float_series(close, "close")

    middle = values.rolling(window=period, min_periods=period).mean()
    deviation = values.rolling(window=period, min_periods=period).std(ddof=0)
    upper = middle + num_std * deviation
    lower = middle - num_std * deviation

    span = upper - lower
    width = (span / middle).where(middle != 0)
    percent_b = ((values - lower) / span).where(span > 0)

    return pd.DataFrame(
        {
            BB_MIDDLE_COL: middle,
            BB_UPPER_COL: upper,
            BB_LOWER_COL: lower,
            BB_WIDTH_COL: width,
            BB_PERCENT_B_COL: percent_b,
        },
        index=values.index,
    )


def calculate_true_range(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """True Range: the greater of the bar's span and its gap from the prior close.

    Returns
    -------
    pandas.Series
        ``NaN`` on the first bar, which has no previous close.

    Example
    -------
    >>> tr = calculate_true_range(data["high"], data["low"], data["close"])
    """
    highs = _as_float_series(high, "high")
    lows = _as_float_series(low, "low")
    closes = _as_float_series(close, "close")

    previous_close = closes.shift(1)
    ranges = pd.concat(
        [
            highs - lows,
            (highs - previous_close).abs(),
            (lows - previous_close).abs(),
        ],
        axis=1,
    )
    true_range = ranges.max(axis=1)
    true_range.iloc[0] = np.nan  # undefined without a previous close
    return true_range


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range (Wilder).

    ATR measures volatility in price units, which is what makes it the right
    input for volatility-scaled stops and position sizing in Phase 4 — unlike a
    percentage stop, it adapts to how much the instrument actually moves.

    Parameters
    ----------
    high, low, close:
        Bar series.
    period:
        Averaging period (default 14).

    Returns
    -------
    pandas.Series
        ATR values, first available at index ``period``.

    Example
    -------
    >>> data["ATR_14"] = calculate_atr(data["high"], data["low"], data["close"])
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    true_range = calculate_true_range(high, low, close)

    # True range is undefined on the first bar, so smoothing runs from the second.
    # A single-bar frame leaves nothing to smooth — a warm-up state, not an error.
    defined = true_range.iloc[1:]
    if defined.empty:
        return _empty_like(true_range)
    return wilder_smooth(defined, period).reindex(true_range.index)


# -- volume indicators ---------------------------------------------------------


def calculate_volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    """Moving average of volume.

    Example
    -------
    >>> data["VOLUME_SMA_20"] = calculate_volume_sma(data["volume"], 20)
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    values = _as_float_series(volume, "volume")
    return values.rolling(window=period, min_periods=period).mean()


def calculate_relative_volume(volume: pd.Series, period: int = 20) -> pd.Series:
    """Current volume as a multiple of its moving average.

    ``1.0`` means average participation, ``2.0`` means twice the usual.

    Returns
    -------
    pandas.Series
        Ratio, ``NaN`` during warm-up or where the average volume is zero.

    Example
    -------
    >>> data["RELATIVE_VOLUME"] = calculate_relative_volume(data["volume"])
    """
    values = _as_float_series(volume, "volume")
    average = calculate_volume_sma(values, period)
    return (values / average).where(average > 0)


# -- master function -----------------------------------------------------------


def calculate_all_indicators(
    data: pd.DataFrame,
    config: IndicatorConfig | None = None,
    *,
    strict: bool = False,
) -> pd.DataFrame:
    """Append every configured indicator to an OHLCV frame.

    This is the engine's main entry point. Strategies call it once per symbol and
    then read columns; they never recompute indicator maths themselves.

    Parameters
    ----------
    data:
        OHLCV frame in the Phase 1 format: lowercase ``open, high, low, close,
        volume`` columns on a sorted, duplicate-free ``DatetimeIndex``. Column
        names differing only in case are accepted and normalised.
    config:
        Periods and thresholds. Defaults to :data:`DEFAULT_CONFIG`.
    strict:
        When True, raise :class:`InsufficientDataError` if the frame is too short
        for any configured indicator. When False (the default), short indicators
        are emitted as all-``NaN`` columns and a warning is logged — a 300-bar
        frame should still yield a usable RSI even though EMA-200 cannot be
        computed.

    Returns
    -------
    pandas.DataFrame
        A **new** frame: the input columns plus indicator columns
        (``SMA_*``, ``EMA_*``, ``RSI_*``, ``MACD*``, ``BB_*``, ``ATR_*``,
        ``VOLUME_SMA_*``, ``RELATIVE_VOLUME``). The input is never mutated.

    Raises
    ------
    InvalidDataError
        The frame is empty, malformed, or missing required columns.
    InsufficientDataError
        ``strict=True`` and the frame is shorter than an indicator needs.

    Example
    -------
    >>> enriched = calculate_all_indicators(bars)
    >>> enriched[["close", "RSI_14", "MACD_HISTOGRAM"]].tail(3)
    """
    settings = config or DEFAULT_CONFIG
    frame = _normalize_columns(data)
    validate_ohlcv(frame, name="data")

    rows = len(frame)
    if strict and rows < settings.max_lookback:
        raise InsufficientDataError(
            f"data has {rows} bar(s) but the configuration needs "
            f"{settings.max_lookback} for every indicator. Fetch more history or "
            "pass strict=False to accept NaN columns for the long-period indicators."
        )

    result = frame.copy()
    close = result["close"]
    skipped: list[str] = []

    def _note_if_short(label: str, required: int) -> bool:
        if rows < required:
            skipped.append(f"{label} (needs {required})")
            return True
        return False

    for period in settings.sma_periods:
        _note_if_short(sma_column(period), period)
        result[sma_column(period)] = calculate_sma(close, period)

    for period in settings.ema_periods:
        _note_if_short(ema_column(period), period)
        result[ema_column(period)] = calculate_ema(close, period)

    _note_if_short(rsi_column(settings.rsi_period), settings.rsi_period + 1)
    result[rsi_column(settings.rsi_period)] = calculate_rsi(close, settings.rsi_period)

    _note_if_short(MACD_COL, settings.macd_slow + settings.macd_signal)
    macd = calculate_macd(close, settings.macd_fast, settings.macd_slow, settings.macd_signal)
    for column in macd.columns:
        result[column] = macd[column]

    _note_if_short(BB_MIDDLE_COL, settings.bollinger_period)
    bands = calculate_bollinger_bands(close, settings.bollinger_period, settings.bollinger_std)
    for column in bands.columns:
        result[column] = bands[column]

    _note_if_short(atr_column(settings.atr_period), settings.atr_period + 1)
    atr = calculate_atr(result["high"], result["low"], close, settings.atr_period)
    result[atr_column(settings.atr_period)] = atr
    # ATR as a percentage of price makes volatility comparable across symbols.
    result[atr_percent_column(settings.atr_period)] = (atr / close * 100).where(close > 0)

    _note_if_short(volume_sma_column(settings.volume_sma_period), settings.volume_sma_period)
    result[volume_sma_column(settings.volume_sma_period)] = calculate_volume_sma(
        result["volume"], settings.volume_sma_period
    )
    result[RELATIVE_VOLUME_COL] = calculate_relative_volume(
        result["volume"], settings.volume_sma_period
    )

    if skipped:
        logger.warning(
            "Only %d bar(s) supplied; these indicators are all-NaN: %s",
            rows,
            ", ".join(skipped),
        )
    return result


def _normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Lower-case OHLCV column names when they differ only by case."""
    if not isinstance(data, pd.DataFrame):
        raise InvalidDataError(f"data must be a pandas DataFrame, got {type(data).__name__}")
    renames = {
        column: column.lower()
        for column in data.columns
        if isinstance(column, str)
        and column.lower() in REQUIRED_COLUMNS
        and column != column.lower()
    }
    return data.rename(columns=renames) if renames else data


def latest_values(data: pd.DataFrame, columns: list[str] | None = None) -> dict[str, Any]:
    """Indicator values on the most recent bar, as a plain dict.

    ``NaN`` becomes ``None`` so the result is JSON-serialisable and safe to log
    or store.

    Example
    -------
    >>> latest_values(enriched, ["close", "RSI_14"])
    {'close': 210.5, 'RSI_14': 62.4}
    """
    if data.empty:
        raise InvalidDataError("cannot read the latest values of an empty frame")
    row = data.iloc[-1]
    selected = columns if columns is not None else list(data.columns)
    output: dict[str, Any] = {}
    for column in selected:
        if column not in data.columns:
            continue
        value = row[column]
        output[column] = None if pd.isna(value) else float(value)
    return output


# -- signal helpers ------------------------------------------------------------
#
# These interpret indicator values. They never place, size or recommend a trade —
# strategies (Phase 3) combine them, and the risk manager (Phase 4) decides.

CrossoverSignal = Literal["BULLISH_CROSSOVER", "BEARISH_CROSSOVER", "NONE"]
RsiCondition = Literal["OVERSOLD", "NEUTRAL", "OVERBOUGHT"]
MacdSignal = Literal["BULLISH", "BEARISH", "NEUTRAL"]
BollingerCondition = Literal["ABOVE_UPPER", "BELOW_LOWER", "SQUEEZE", "NORMAL"]


@dataclass(frozen=True, slots=True)
class CrossoverResult:
    """Outcome of an EMA crossover check on the latest bar."""

    signal: CrossoverSignal
    occurred: bool
    fast_ema: float | None
    slow_ema: float | None
    fast_period: int
    slow_period: int
    #: Separation between the averages as a percentage of the slow EMA.
    separation_pct: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """Plain-dict view, convenient for logging and persistence."""
        return {
            "signal": self.signal,
            "occurred": self.occurred,
            "fast_ema": self.fast_ema,
            "slow_ema": self.slow_ema,
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "separation_pct": self.separation_pct,
        }


def detect_ema_crossover(
    data: pd.DataFrame,
    fast_period: int = 9,
    slow_period: int = 20,
) -> CrossoverResult:
    """Detect an EMA crossover on the most recent bar.

    A crossover is reported only when the ordering actually *changed* on the last
    bar. A fast EMA that has been above the slow one for weeks is a trend, not a
    crossover, and reporting it as one would fire an entry on every bar.

    Parameters
    ----------
    data:
        Frame containing the two EMA columns (run
        :func:`calculate_all_indicators` first, or the EMAs are computed here).
    fast_period, slow_period:
        EMA periods; ``fast_period`` must be the shorter one.

    Returns
    -------
    CrossoverResult
        ``signal`` is ``BULLISH_CROSSOVER``, ``BEARISH_CROSSOVER`` or ``NONE``.

    Example
    -------
    >>> detect_ema_crossover(enriched, 9, 20).as_dict()
    {'signal': 'BULLISH_CROSSOVER', 'occurred': True, ...}
    """
    if fast_period >= slow_period:
        raise ValueError(
            f"fast_period ({fast_period}) must be less than slow_period ({slow_period})"
        )
    fast = _column_or_compute(data, ema_column(fast_period), fast_period)
    slow = _column_or_compute(data, ema_column(slow_period), slow_period)

    none_result = CrossoverResult(
        signal="NONE",
        occurred=False,
        fast_ema=None,
        slow_ema=None,
        fast_period=fast_period,
        slow_period=slow_period,
    )
    if len(data) < 2:
        return none_result

    fast_now, fast_prev = fast.iloc[-1], fast.iloc[-2]
    slow_now, slow_prev = slow.iloc[-1], slow.iloc[-2]
    if any(pd.isna(value) for value in (fast_now, fast_prev, slow_now, slow_prev)):
        return none_result

    separation = ((fast_now - slow_now) / slow_now * 100) if slow_now else None
    crossed_up = fast_prev <= slow_prev and fast_now > slow_now
    crossed_down = fast_prev >= slow_prev and fast_now < slow_now

    signal: CrossoverSignal = "NONE"
    if crossed_up:
        signal = "BULLISH_CROSSOVER"
    elif crossed_down:
        signal = "BEARISH_CROSSOVER"

    return CrossoverResult(
        signal=signal,
        occurred=signal != "NONE",
        fast_ema=float(fast_now),
        slow_ema=float(slow_now),
        fast_period=fast_period,
        slow_period=slow_period,
        separation_pct=float(separation) if separation is not None else None,
    )


def _column_or_compute(data: pd.DataFrame, column: str, period: int) -> pd.Series:
    """Reuse a precomputed EMA column, or compute it if absent."""
    if column in data.columns:
        return data[column]
    if "close" not in data.columns:
        raise InvalidDataError(
            f"data has neither a '{column}' column nor 'close' to compute it from"
        )
    return calculate_ema(data["close"], period)


def detect_rsi_condition(
    data: pd.DataFrame | float,
    config: IndicatorConfig | None = None,
    *,
    period: int | None = None,
) -> RsiCondition:
    """Classify the latest RSI as ``OVERSOLD``, ``NEUTRAL`` or ``OVERBOUGHT``.

    Thresholds come from ``config`` rather than being hardcoded, so a strategy
    can tighten or loosen them.

    Parameters
    ----------
    data:
        Frame containing an RSI column, or a bare RSI value.
    config:
        Supplies ``rsi_period``, ``rsi_oversold`` and ``rsi_overbought``.
    period:
        Override the RSI period to read from ``data``.

    Returns
    -------
    str
        One of ``OVERSOLD``, ``NEUTRAL``, ``OVERBOUGHT``. An unavailable RSI
        (warm-up) reports ``NEUTRAL``, the non-committal answer.

    Example
    -------
    >>> detect_rsi_condition(enriched)
    'NEUTRAL'
    """
    settings = config or DEFAULT_CONFIG
    rsi_period = period if period is not None else settings.rsi_period

    if isinstance(data, pd.DataFrame):
        column = rsi_column(rsi_period)
        series = data[column] if column in data.columns else calculate_rsi(
            data["close"], rsi_period
        )
        value = series.iloc[-1] if len(series) else np.nan
    else:
        value = data

    if value is None or pd.isna(value):
        return "NEUTRAL"
    if value < settings.rsi_oversold:
        return "OVERSOLD"
    if value > settings.rsi_overbought:
        return "OVERBOUGHT"
    return "NEUTRAL"


def detect_macd_signal(data: pd.DataFrame, config: IndicatorConfig | None = None) -> MacdSignal:
    """Classify MACD state on the latest bar.

    ``BULLISH`` when the MACD line is above its signal line, ``BEARISH`` when
    below, ``NEUTRAL`` while the values are still warming up.

    Example
    -------
    >>> detect_macd_signal(enriched)
    'BULLISH'
    """
    macd, signal, _ = _macd_columns(data, config)
    if macd is None or signal is None:
        return "NEUTRAL"
    if macd > signal:
        return "BULLISH"
    if macd < signal:
        return "BEARISH"
    return "NEUTRAL"


def detect_macd_crossover(
    data: pd.DataFrame, config: IndicatorConfig | None = None
) -> CrossoverSignal:
    """Detect a MACD/signal-line crossover that happened on the latest bar.

    Example
    -------
    >>> detect_macd_crossover(enriched)
    'BULLISH_CROSSOVER'
    """
    settings = config or DEFAULT_CONFIG
    frame = _ensure_macd(data, settings)
    if len(frame) < 2:
        return "NONE"

    macd_now, macd_prev = frame[MACD_COL].iloc[-1], frame[MACD_COL].iloc[-2]
    signal_now, signal_prev = frame[MACD_SIGNAL_COL].iloc[-1], frame[MACD_SIGNAL_COL].iloc[-2]
    if any(pd.isna(value) for value in (macd_now, macd_prev, signal_now, signal_prev)):
        return "NONE"

    if macd_prev <= signal_prev and macd_now > signal_now:
        return "BULLISH_CROSSOVER"
    if macd_prev >= signal_prev and macd_now < signal_now:
        return "BEARISH_CROSSOVER"
    return "NONE"


def detect_macd_momentum(
    data: pd.DataFrame,
    config: IndicatorConfig | None = None,
    *,
    lookback: int = 3,
) -> Literal["INCREASING", "DECREASING", "FLAT"]:
    """Report whether MACD momentum is building or fading.

    Compares the histogram against its value ``lookback`` bars ago. A rising
    histogram means the spread between the MACD and its signal is widening —
    momentum is accelerating, regardless of which side of zero it sits on.

    Example
    -------
    >>> detect_macd_momentum(enriched)
    'INCREASING'
    """
    settings = config or DEFAULT_CONFIG
    frame = _ensure_macd(data, settings)
    histogram = frame[MACD_HISTOGRAM_COL].dropna()
    if len(histogram) < lookback + 1:
        return "FLAT"

    current = float(histogram.iloc[-1])
    previous = float(histogram.iloc[-1 - lookback])
    if current > previous:
        return "INCREASING"
    if current < previous:
        return "DECREASING"
    return "FLAT"


def _ensure_macd(data: pd.DataFrame, config: IndicatorConfig) -> pd.DataFrame:
    """Return a frame that definitely carries MACD columns."""
    if {MACD_COL, MACD_SIGNAL_COL, MACD_HISTOGRAM_COL} <= set(data.columns):
        return data
    if "close" not in data.columns:
        raise InvalidDataError("data has neither MACD columns nor 'close' to compute them")
    return calculate_macd(data["close"], config.macd_fast, config.macd_slow, config.macd_signal)


def _macd_columns(
    data: pd.DataFrame, config: IndicatorConfig | None
) -> tuple[float | None, float | None, float | None]:
    """Latest MACD, signal and histogram values, or None during warm-up."""
    frame = _ensure_macd(data, config or DEFAULT_CONFIG)
    if frame.empty:
        return None, None, None
    row = frame.iloc[-1]
    values = []
    for column in (MACD_COL, MACD_SIGNAL_COL, MACD_HISTOGRAM_COL):
        value = row.get(column)
        values.append(None if value is None or pd.isna(value) else float(value))
    return values[0], values[1], values[2]


def detect_bollinger_condition(
    data: pd.DataFrame, config: IndicatorConfig | None = None
) -> BollingerCondition:
    """Classify where price sits relative to its Bollinger Bands.

    Returns ``ABOVE_UPPER``, ``BELOW_LOWER``, ``SQUEEZE`` when bandwidth is in the
    bottom ``squeeze_percentile`` of its recent range, or ``NORMAL``.

    A squeeze signals compressed volatility, which often precedes an expansion —
    useful context for the Phase 3 breakout strategy, though it says nothing
    about direction.

    Example
    -------
    >>> detect_bollinger_condition(enriched)
    'NORMAL'
    """
    settings = config or DEFAULT_CONFIG
    frame = data
    if BB_UPPER_COL not in frame.columns:
        if "close" not in frame.columns:
            raise InvalidDataError("data has neither Bollinger columns nor 'close'")
        bands = calculate_bollinger_bands(
            frame["close"], settings.bollinger_period, settings.bollinger_std
        )
        frame = frame.join(bands) if isinstance(frame, pd.DataFrame) else bands

    if frame.empty:
        return "NORMAL"
    row = frame.iloc[-1]
    close = row.get("close")
    upper, lower = row.get(BB_UPPER_COL), row.get(BB_LOWER_COL)

    if close is not None and not pd.isna(close):
        if upper is not None and not pd.isna(upper) and close > upper:
            return "ABOVE_UPPER"
        if lower is not None and not pd.isna(lower) and close < lower:
            return "BELOW_LOWER"

    if is_bollinger_squeeze(frame, settings):
        return "SQUEEZE"
    return "NORMAL"


def is_bollinger_squeeze(data: pd.DataFrame, config: IndicatorConfig | None = None) -> bool:
    """True when Bollinger bandwidth is unusually narrow.

    "Unusually" is relative to the symbol's own recent bandwidth
    (``squeeze_lookback`` bars), not an absolute number — a fixed threshold would
    label every low-volatility instrument as permanently squeezed.

    Example
    -------
    >>> is_bollinger_squeeze(enriched)
    False
    """
    settings = config or DEFAULT_CONFIG
    if BB_WIDTH_COL not in data.columns:
        if "close" not in data.columns:
            return False
        width = calculate_bollinger_bands(
            data["close"], settings.bollinger_period, settings.bollinger_std
        )[BB_WIDTH_COL]
    else:
        width = data[BB_WIDTH_COL]

    recent = width.dropna().tail(settings.squeeze_lookback)
    # Too few observations to say what "narrow" means for this symbol.
    if len(recent) < max(10, settings.bollinger_period // 2):
        return False

    threshold = float(np.percentile(recent.to_numpy(), settings.squeeze_percentile))
    return bool(recent.iloc[-1] <= threshold)
