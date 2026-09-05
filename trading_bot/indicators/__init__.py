"""Technical indicator engine (Phase 2).

The analysis layer. Everything here reads market data and reports what it sees;
nothing here sizes, places or recommends a trade — that is the job of the
strategies (Phase 3), the risk manager (Phase 4) and the execution layer
(Phase 7).

Typical use::

    from trading_bot.indicators import (
        calculate_all_indicators,
        analyze_trend,
        analyze_volume,
        find_support_resistance,
    )

    enriched = calculate_all_indicators(bars)
    trend = analyze_trend(enriched)
    volume = analyze_volume(enriched)
    levels = find_support_resistance(enriched)

Every indicator is computed once by :func:`calculate_all_indicators` and read
from columns thereafter, so a strategy and a backtest can never disagree about
what an indicator value was.
"""

from trading_bot.indicators.price_action import (
    SWING_HIGH_COL,
    SWING_LOW_COL,
    Level,
    SupportResistance,
    SwingPoint,
    detect_market_structure,
    find_support_resistance,
    find_swing_points,
    swing_point_columns,
)
from trading_bot.indicators.technical_indicators import (
    BB_LOWER_COL,
    BB_MIDDLE_COL,
    BB_PERCENT_B_COL,
    BB_UPPER_COL,
    BB_WIDTH_COL,
    DEFAULT_CONFIG,
    MACD_COL,
    MACD_HISTOGRAM_COL,
    MACD_SIGNAL_COL,
    RELATIVE_VOLUME_COL,
    REQUIRED_COLUMNS,
    CrossoverResult,
    IndicatorConfig,
    IndicatorError,
    InsufficientDataError,
    InvalidDataError,
    atr_column,
    atr_percent_column,
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
    detect_macd_crossover,
    detect_macd_momentum,
    detect_macd_signal,
    detect_rsi_condition,
    ema_column,
    is_bollinger_squeeze,
    latest_values,
    rsi_column,
    sma_column,
    validate_ohlcv,
    volume_sma_column,
    wilder_smooth,
)
from trading_bot.indicators.trend_analysis import (
    TrendAnalysis,
    TrendComponent,
    TrendDirection,
    analyze_trend,
)
from trading_bot.indicators.volume_analysis import (
    VolumeAnalysis,
    VolumeCondition,
    analyze_volume,
    classify_relative_volume,
    detect_volume_condition,
    detect_volume_trend,
    volume_confirms_price,
)

__all__ = [
    # Configuration and errors
    "IndicatorConfig",
    "DEFAULT_CONFIG",
    "IndicatorError",
    "InvalidDataError",
    "InsufficientDataError",
    "REQUIRED_COLUMNS",
    "validate_ohlcv",
    # Core calculations
    "calculate_all_indicators",
    "calculate_sma",
    "calculate_ema",
    "calculate_rsi",
    "calculate_macd",
    "calculate_bollinger_bands",
    "calculate_atr",
    "calculate_true_range",
    "calculate_volume_sma",
    "calculate_relative_volume",
    "wilder_smooth",
    "latest_values",
    # Column naming
    "sma_column",
    "ema_column",
    "rsi_column",
    "atr_column",
    "atr_percent_column",
    "volume_sma_column",
    "MACD_COL",
    "MACD_SIGNAL_COL",
    "MACD_HISTOGRAM_COL",
    "BB_UPPER_COL",
    "BB_MIDDLE_COL",
    "BB_LOWER_COL",
    "BB_WIDTH_COL",
    "BB_PERCENT_B_COL",
    "RELATIVE_VOLUME_COL",
    "SWING_HIGH_COL",
    "SWING_LOW_COL",
    # Signal helpers
    "CrossoverResult",
    "detect_ema_crossover",
    "detect_rsi_condition",
    "detect_macd_signal",
    "detect_macd_crossover",
    "detect_macd_momentum",
    "detect_bollinger_condition",
    "is_bollinger_squeeze",
    "detect_volume_condition",
    # Trend analysis
    "TrendDirection",
    "TrendAnalysis",
    "TrendComponent",
    "analyze_trend",
    # Volume analysis
    "VolumeCondition",
    "VolumeAnalysis",
    "analyze_volume",
    "classify_relative_volume",
    "detect_volume_trend",
    "volume_confirms_price",
    # Price action
    "SwingPoint",
    "Level",
    "SupportResistance",
    "find_swing_points",
    "swing_point_columns",
    "find_support_resistance",
    "detect_market_structure",
]
