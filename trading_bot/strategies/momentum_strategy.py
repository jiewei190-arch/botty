"""Momentum strategy: join an established trend on a fresh acceleration.

The thesis is continuation — a market already trending, that has just given a
timing trigger, tends to keep going. The hard part is not finding trends but
avoiding entries after the move is spent, which is what the RSI ceiling and the
volume requirement are for.

Entry (long)
------------
Required, all of which must hold:

* A bullish EMA crossover on this bar, or price above a rising fast EMA while the
  averages are already stacked bullishly.
* The wider trend is bullish — the Phase 2 trend classifier, not just one EMA.
* RSI below the entry ceiling. Buying a market at RSI 85 is buying the top of a
  move; the ceiling is deliberately lower than the classic 70 "overbought" line
  because this strategy is *entering*, not holding.

Optional, contributing confidence:

* Above-average volume, confirming real participation.
* MACD above its signal line and the histogram expanding.
* Price above the long-term EMA.
* A strong trend-strength score.

Short entries mirror this and are disabled by default.

Exit
----
Beyond the protective stop and target handled by the base class: momentum fading
(MACD crossing down while held long), or the trend classifier flipping against
the position. Both are discretionary — the stop is what caps the loss.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from trading_bot.indicators import (
    MACD_HISTOGRAM_COL,
    analyze_trend,
    analyze_volume,
    detect_ema_crossover,
    detect_macd_momentum,
    detect_macd_signal,
    ema_column,
    find_support_resistance,
    rsi_column,
)
from trading_bot.strategies.base_strategy import (
    BaseStrategy,
    Condition,
    ExitReason,
    ExitSignal,
    Position,
    Signal,
    SignalDirection,
    StrategyConfig,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MomentumConfig(StrategyConfig):
    """Parameters for :class:`MomentumStrategy`."""

    #: EMA pair whose crossover provides the timing trigger.
    fast_ema: int = 9
    slow_ema: int = 20
    #: Longer EMA used as a directional filter.
    trend_ema: int = 50
    #: Do not buy above this RSI — the move is already extended.
    rsi_entry_ceiling: float = 68.0
    #: Do not short below this RSI.
    rsi_entry_floor: float = 32.0
    #: Volume multiple required for the participation bonus.
    min_relative_volume: float = 1.1
    #: Minimum trend strength (0-100, 50 neutral) for a long.
    min_trend_strength: float = 58.0
    #: Bars over which the MACD histogram must expand to count as accelerating.
    momentum_lookback: int = 3
    #: Accept a pullback-and-resume entry, not just a fresh crossover.
    allow_trend_continuation: bool = True
    #: Bars within which price must have reclaimed the fast EMA to count as a
    #: resumed pullback. Without this window, "price is above the EMA" is true on
    #: most bars of a trend and the strategy signals almost continuously.
    continuation_window: int = 5
    #: Exit when MACD crosses against an open position.
    exit_on_momentum_fade: bool = True
    #: Exit when the trend classifier flips against an open position.
    exit_on_trend_reversal: bool = True


class MomentumStrategy(BaseStrategy):
    """Trend-continuation strategy driven by EMA, MACD, RSI and volume.

    Example
    -------
    >>> strategy = MomentumStrategy(MomentumConfig(min_confidence=65))
    >>> signal = strategy.generate_signal("AAPL", strategy.prepare(bars))
    """

    name = "momentum"

    def __init__(self, config: MomentumConfig | None = None, indicators=None) -> None:
        super().__init__(config or MomentumConfig(), indicators)
        self.config: MomentumConfig

    @property
    def min_bars(self) -> int:
        return max(
            self.indicators.max_lookback,
            self.config.trend_ema + self.config.momentum_lookback + 5,
        )

    def evaluate(self, symbol: str, data: pd.DataFrame) -> Signal | None:
        trend = analyze_trend(data, self.indicators)
        long_signal = self._evaluate_direction(symbol, data, SignalDirection.LONG, trend)
        if long_signal is not None:
            return long_signal
        return self._evaluate_direction(symbol, data, SignalDirection.SHORT, trend)

    def _evaluate_direction(
        self,
        symbol: str,
        data: pd.DataFrame,
        direction: SignalDirection,
        trend,
    ) -> Signal | None:
        if not self.config.allows(direction):
            return None

        bar = data.iloc[-1]
        close = float(bar["close"])
        is_long = direction is SignalDirection.LONG

        fast = _latest(data, ema_column(self.config.fast_ema))
        slow = _latest(data, ema_column(self.config.slow_ema))
        long_term = _latest(data, ema_column(self.config.trend_ema))
        rsi = _latest(data, rsi_column(self.indicators.rsi_period))

        crossover = detect_ema_crossover(data, self.config.fast_ema, self.config.slow_ema)
        wanted_cross = "BULLISH_CROSSOVER" if is_long else "BEARISH_CROSSOVER"
        fresh_cross = crossover.signal == wanted_cross

        # A fresh crossover is one trigger. The other is a pullback that has just
        # resumed: price dipped through the fast EMA within the last few bars and
        # has now reclaimed it, with the averages still stacked. Requiring the
        # *reclaim* rather than merely "price is above the EMA" is what keeps this
        # an event — the latter is true on most bars of a trend and would signal
        # almost continuously.
        stacked = (
            fast is not None
            and slow is not None
            and ((fast > slow) if is_long else (fast < slow))
        )
        continuation = (
            self.config.allow_trend_continuation
            and stacked
            and self._pullback_resumed(data, direction)
        )
        has_trigger = fresh_cross or continuation

        if is_long:
            trend_ok = (
                trend.direction.is_bullish
                and trend.strength >= self.config.min_trend_strength
            )
            rsi_ok = rsi is None or rsi <= self.config.rsi_entry_ceiling
            rsi_detail = (
                f"RSI {rsi:.1f} leaves room to run" if rsi is not None else "RSI unavailable"
            )
        else:
            trend_ok = trend.direction.is_bearish and trend.strength <= (
                100 - self.config.min_trend_strength
            )
            rsi_ok = rsi is None or rsi >= self.config.rsi_entry_floor
            rsi_detail = (
                f"RSI {rsi:.1f} leaves room to fall" if rsi is not None else "RSI unavailable"
            )

        trigger_detail = (
            f"Bullish EMA {self.config.fast_ema}/{self.config.slow_ema} crossover"
            if fresh_cross and is_long
            else f"Bearish EMA {self.config.fast_ema}/{self.config.slow_ema} crossover"
            if fresh_cross
            else f"Pullback to EMA {self.config.fast_ema} resumed higher"
            if is_long
            else f"Rally to EMA {self.config.fast_ema} resumed lower"
        )

        volume = analyze_volume(data, self.indicators)
        relative_volume = volume.relative_volume
        volume_ok = (
            relative_volume is not None and relative_volume >= self.config.min_relative_volume
        )

        macd_state = detect_macd_signal(data, self.indicators)
        macd_ok = macd_state == ("BULLISH" if is_long else "BEARISH")
        momentum = detect_macd_momentum(
            data, self.indicators, lookback=self.config.momentum_lookback
        )
        histogram = _latest(data, MACD_HISTOGRAM_COL)
        accelerating = _is_accelerating(histogram, momentum, is_long)

        beyond_long_term = long_term is not None and (
            (close > long_term) if is_long else (close < long_term)
        )

        conditions = [
            Condition(
                "trigger", has_trigger, weight=2.0, required=True, detail=trigger_detail
            ),
            Condition(
                "trend",
                trend_ok,
                weight=2.0,
                required=True,
                detail=f"{trend.direction.value} trend at {trend.strength}/100",
            ),
            Condition(
                "rsi_room", rsi_ok, weight=1.0, required=True, detail=rsi_detail
            ),
            Condition(
                "volume",
                volume_ok,
                weight=1.5,
                detail=(
                    f"Volume {relative_volume:.2f}x average confirms the move"
                    if relative_volume is not None
                    else "Volume unavailable"
                ),
            ),
            Condition(
                "macd", macd_ok, weight=1.5, detail=f"MACD {macd_state.lower()}"
            ),
            Condition(
                "acceleration",
                accelerating,
                weight=1.0,
                detail=f"MACD histogram {momentum.lower()}",
            ),
            Condition(
                "long_term_ema",
                beyond_long_term,
                weight=1.0,
                detail=(
                    f"Price above EMA {self.config.trend_ema}"
                    if is_long
                    else f"Price below EMA {self.config.trend_ema}"
                ),
            ),
        ]

        levels = find_support_resistance(data, self.indicators)
        return self._build_signal(
            symbol=symbol,
            direction=direction,
            data=data,
            conditions=conditions,
            levels=levels,
            metadata={
                "trend_direction": trend.direction.value,
                "trend_strength": trend.strength,
                "rsi": rsi,
                "relative_volume": relative_volume,
                "macd_state": macd_state,
                "fresh_crossover": fresh_cross,
            },
        )

    def _pullback_resumed(self, data: pd.DataFrame, direction: SignalDirection) -> bool:
        """True when price has just reclaimed the fast EMA after dipping through it.

        Looks for a close on the wrong side of the fast EMA within the last
        ``continuation_window`` bars, followed by a close back on the right side
        now. That is a pullback that has resumed, rather than an arbitrary bar in
        the middle of a trend.
        """
        column = ema_column(self.config.fast_ema)
        if column not in data.columns:
            return False

        window = self.config.continuation_window
        recent = data.tail(window + 1)
        if len(recent) < 2:
            return False

        closes = recent["close"].to_numpy(dtype="float64")
        emas = recent[column].to_numpy(dtype="float64")
        if pd.isna(emas).any():
            return False

        is_long = direction is SignalDirection.LONG
        now_onside = closes[-1] > emas[-1] if is_long else closes[-1] < emas[-1]
        if not now_onside:
            return False

        prior = closes[:-1] <= emas[:-1] if is_long else closes[:-1] >= emas[:-1]
        return bool(prior.any())

    def evaluate_exit(self, position: Position, data: pd.DataFrame) -> ExitSignal | None:
        close = float(data.iloc[-1]["close"])
        is_long = position.direction is SignalDirection.LONG

        if self.config.exit_on_momentum_fade:
            state = detect_macd_signal(data, self.indicators)
            if state == ("BEARISH" if is_long else "BULLISH"):
                return ExitSignal(
                    reason=ExitReason.MOMENTUM_FADE,
                    price=close,
                    detail=f"MACD turned {state.lower()} against the position",
                )

        if self.config.exit_on_trend_reversal:
            trend = analyze_trend(data, self.indicators)
            reversed_now = trend.direction.is_bearish if is_long else trend.direction.is_bullish
            if reversed_now:
                return ExitSignal(
                    reason=ExitReason.TREND_REVERSAL,
                    price=close,
                    detail=f"Trend flipped to {trend.direction.value}",
                    metadata={"trend_strength": trend.strength},
                )
        return None


def _latest(data: pd.DataFrame, column: str) -> float | None:
    """Latest value of ``column``, or ``None`` when missing or warming up."""
    if column not in data.columns:
        return None
    value = data[column].iloc[-1]
    return None if pd.isna(value) else float(value)


def _is_accelerating(histogram: float | None, momentum: str, is_long: bool) -> bool:
    """Whether MACD momentum is building in the trade's direction."""
    if histogram is None:
        return False
    if is_long:
        return histogram > 0 and momentum == "INCREASING"
    return histogram < 0 and momentum == "DECREASING"
