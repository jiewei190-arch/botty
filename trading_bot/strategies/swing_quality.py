"""High-quality swing strategy for selective multi-day setups.

This strategy is intentionally sparse. It is designed for daily bars and a
roughly one-to-four week holding window. A valid signal needs three things at
once:

* durable trend structure (20 EMA > 50 EMA > 200 EMA),
* a fresh entry event (pullback/reclaim or 20-bar breakout), and
* enough momentum room to avoid chasing an exhausted move.

Supporting evidence such as rising trend strength and healthy relative volume
raises confidence, but cannot rescue a structurally invalid trade.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_bot.indicators import analyze_trend, analyze_volume, find_support_resistance, rsi_column
from trading_bot.strategies.base_strategy import (
    BaseStrategy,
    Condition,
    Signal,
    SignalDirection,
    StrategyConfig,
)


@dataclass(frozen=True, slots=True)
class SwingQualityConfig(StrategyConfig):
    """Parameters for :class:`SwingQualityStrategy`.

    Defaults deliberately favour quality over frequency. With daily bars,
    ``max_holding_bars=20`` is approximately one trading month.
    """

    min_confidence: float = 70.0
    max_holding_bars: int | None = 20

    fast_ema: int = 20
    mid_ema: int = 50
    long_ema: int = 200
    breakout_lookback: int = 20
    pullback_window: int = 6

    rsi_floor: float = 45.0
    rsi_ceiling: float = 70.0
    min_relative_volume: float = 1.0
    min_trend_strength: float = 60.0

    allow_long: bool = True
    allow_short: bool = False


class SwingQualityStrategy(BaseStrategy):
    """Selective trend-following swing strategy.

    The strategy does not try to predict every move. It waits for an established
    primary trend, then requires either a constructive pullback/reclaim or a
    fresh breakout. This makes "no trade" a normal and desirable outcome.
    """

    name = "swing_quality"

    def __init__(self, config: SwingQualityConfig | None = None, indicators=None) -> None:
        super().__init__(config or SwingQualityConfig(), indicators)
        self.config: SwingQualityConfig

    @property
    def min_bars(self) -> int:
        return max(
            self.indicators.max_lookback,
            self.config.long_ema + self.config.pullback_window + 5,
        )

    def evaluate(self, symbol: str, data: pd.DataFrame) -> Signal | None:
        if not self.config.allow_long:
            return None

        close_series = data["close"].astype("float64")
        if len(close_series) < self.min_bars:
            return None

        ema_fast_series = close_series.ewm(span=self.config.fast_ema, adjust=False).mean()
        ema_mid_series = close_series.ewm(span=self.config.mid_ema, adjust=False).mean()
        ema_long_series = close_series.ewm(span=self.config.long_ema, adjust=False).mean()

        close = float(close_series.iloc[-1])
        fast = float(ema_fast_series.iloc[-1])
        mid = float(ema_mid_series.iloc[-1])
        long_term = float(ema_long_series.iloc[-1])

        trend_stack = fast > mid > long_term and close > fast
        fast_rising = fast > float(ema_fast_series.iloc[-6])
        mid_rising = mid > float(ema_mid_series.iloc[-11])

        rsi_col = rsi_column(self.indicators.rsi_period)
        rsi_value = data[rsi_col].iloc[-1] if rsi_col in data.columns else None
        rsi = None if rsi_value is None or pd.isna(rsi_value) else float(rsi_value)
        rsi_room = rsi is not None and self.config.rsi_floor <= rsi <= self.config.rsi_ceiling

        pullback_reclaim = self._pullback_reclaim(close_series, ema_fast_series)
        breakout = self._fresh_breakout(close_series)
        trigger = pullback_reclaim or breakout

        trend = analyze_trend(data, self.indicators)
        trend_quality = trend.direction.is_bullish and trend.strength >= self.config.min_trend_strength

        volume = analyze_volume(data, self.indicators)
        relative_volume = volume.relative_volume
        healthy_volume = (
            relative_volume is not None
            and relative_volume >= self.config.min_relative_volume
        )

        conditions = [
            Condition(
                "primary_trend",
                trend_stack,
                weight=2.5,
                required=True,
                detail=(
                    f"Bullish EMA stack: {self.config.fast_ema} > "
                    f"{self.config.mid_ema} > {self.config.long_ema}"
                ),
            ),
            Condition(
                "entry_trigger",
                trigger,
                weight=2.5,
                required=True,
                detail=(
                    "Constructive pullback reclaimed the 20 EMA"
                    if pullback_reclaim
                    else f"Fresh {self.config.breakout_lookback}-bar breakout"
                ),
            ),
            Condition(
                "momentum_room",
                rsi_room,
                weight=1.5,
                required=True,
                detail=(
                    f"RSI {rsi:.1f} is in the swing entry zone"
                    if rsi is not None
                    else "RSI unavailable"
                ),
            ),
            Condition(
                "ema_slopes",
                fast_rising and mid_rising,
                weight=1.25,
                detail="20 EMA and 50 EMA are both rising",
            ),
            Condition(
                "trend_quality",
                trend_quality,
                weight=1.25,
                detail=f"Trend strength {trend.strength}/100",
            ),
            Condition(
                "volume_confirmation",
                healthy_volume,
                weight=1.0,
                detail=(
                    f"Relative volume {relative_volume:.2f}x"
                    if relative_volume is not None
                    else "Relative volume unavailable"
                ),
            ),
        ]

        levels = find_support_resistance(data, self.indicators)
        return self._build_signal(
            symbol=symbol,
            direction=SignalDirection.LONG,
            data=data,
            conditions=conditions,
            levels=levels,
            metadata={
                "intended_timeframe": "1Day",
                "expected_hold_bars": self.config.max_holding_bars,
                "ema20": fast,
                "ema50": mid,
                "ema200": long_term,
                "rsi": rsi,
                "relative_volume": relative_volume,
                "trend_strength": trend.strength,
                "trigger": "pullback_reclaim" if pullback_reclaim else "breakout",
            },
        )

    def _pullback_reclaim(self, closes: pd.Series, ema_fast: pd.Series) -> bool:
        """Price recently tested/breached the fast EMA and has reclaimed it."""
        window = self.config.pullback_window
        recent_close = closes.tail(window + 1)
        recent_ema = ema_fast.tail(window + 1)
        if len(recent_close) < window + 1:
            return False

        now_above = float(recent_close.iloc[-1]) > float(recent_ema.iloc[-1])
        improving = float(recent_close.iloc[-1]) > float(recent_close.iloc[-2])
        prior_test = bool((recent_close.iloc[:-1] <= recent_ema.iloc[:-1] * 1.01).any())
        return now_above and improving and prior_test

    def _fresh_breakout(self, closes: pd.Series) -> bool:
        """Close above the previous N-bar closing high, excluding the current bar."""
        lookback = self.config.breakout_lookback
        if len(closes) <= lookback:
            return False
        prior_high = float(closes.iloc[-(lookback + 1):-1].max())
        current = float(closes.iloc[-1])
        previous = float(closes.iloc[-2])
        return current > prior_high and current > previous
