"""Breakout strategy: trade the expansion out of a consolidation.

The thesis is that volatility is cyclical. A market that has coiled into a tight
range is storing energy, and the release tends to run. The failure mode is the
false breakout — price pokes through a level on no volume and immediately
reverses — so the three required conditions all exist to filter those out:
a genuine prior consolidation, a decisive close beyond the level, and volume
confirming that someone actually showed up.

Entry (long)
------------
Required:

* Price closes above the nearest resistance level by more than
  ``breakout_buffer_pct``. A close *beyond* the level, not an intrabar poke —
  wicks through resistance are what stop-hunts look like.
* The bars before the break formed a real consolidation: their whole range fits
  inside ``max_consolidation_width_atr`` ATRs.
* Volume at or above ``min_volume_spike``. A breakout nobody participated in is
  the definition of a false one.

Optional:

* The level had several touches — a level tested repeatedly matters more.
* The bar closed in the top of its own range, showing it held the gain.
* The wider trend agrees.
* Bollinger bandwidth was squeezed going in.

Target
------
Either the measured move (the consolidation's own height projected from the
break) or an ATR multiple, whichever the configuration selects. The stop sits
back inside the range, below the level that just broke — if price returns there,
the breakout failed and the reason for the trade is gone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from trading_bot.indicators import (
    SupportResistance,
    analyze_trend,
    analyze_volume,
    atr_column,
    find_support_resistance,
    is_bollinger_squeeze,
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
class BreakoutConfig(StrategyConfig):
    """Parameters for :class:`BreakoutStrategy`."""

    #: Bars examined for the consolidation that precedes the break.
    consolidation_bars: int = 20
    #: The consolidation's full range must fit inside this many ATRs.
    max_consolidation_width_atr: float = 3.0
    #: Price must clear the level by this percentage to count as a break.
    breakout_buffer_pct: float = 0.15
    #: Volume multiple required to confirm the break.
    min_volume_spike: float = 1.4
    #: A level needs at least this many touches to be worth trading.
    min_level_touches: int = 1
    #: Project the consolidation's height from the broken level as the target.
    use_measured_move: bool = True
    #: Refuse to enter more than this many ATRs beyond the broken level. Entering
    #: far above the range is chasing: the risk is measured from the level, so the
    #: further price has already run, the worse the trade pays.
    max_entry_extension_atr: float = 1.5
    #: Close must sit in the top (or bottom) of this fraction of the bar's range.
    strong_close_fraction: float = 0.6
    #: Exit if price falls back inside the range that broke.
    exit_on_failed_breakout: bool = True

    def __post_init__(self) -> None:
        # Explicit parent call: @dataclass(slots=True) rebuilds the class, which
        # breaks the __class__ cell that zero-argument super() depends on.
        StrategyConfig.__post_init__(self)
        if self.consolidation_bars < 2:
            raise ValueError(
                f"consolidation_bars must be >= 2, got {self.consolidation_bars}"
            )
        if self.max_consolidation_width_atr <= 0:
            raise ValueError(
                "max_consolidation_width_atr must be > 0, got "
                f"{self.max_consolidation_width_atr}"
            )
        if self.breakout_buffer_pct < 0:
            raise ValueError(
                f"breakout_buffer_pct must be >= 0, got {self.breakout_buffer_pct}"
            )
        if not 0 < self.strong_close_fraction <= 1:
            raise ValueError(
                f"strong_close_fraction must be within (0, 1], got "
                f"{self.strong_close_fraction}"
            )


class BreakoutStrategy(BaseStrategy):
    """Trade decisive, confirmed breaks out of a consolidation.

    Example
    -------
    >>> strategy = BreakoutStrategy(BreakoutConfig(min_volume_spike=1.8))
    >>> signal = strategy.generate_signal("NVDA", strategy.prepare(bars))
    """

    name = "breakout"

    def __init__(self, config: BreakoutConfig | None = None, indicators=None) -> None:
        super().__init__(config or BreakoutConfig(), indicators)
        self.config: BreakoutConfig

    @property
    def min_bars(self) -> int:
        return max(self.indicators.max_lookback, self.config.consolidation_bars + 10)

    def evaluate(self, symbol: str, data: pd.DataFrame) -> Signal | None:
        levels = find_support_resistance(data, self.indicators)
        signal = self._evaluate_direction(symbol, data, SignalDirection.LONG, levels)
        if signal is not None:
            return signal
        return self._evaluate_direction(symbol, data, SignalDirection.SHORT, levels)

    def _evaluate_direction(
        self,
        symbol: str,
        data: pd.DataFrame,
        direction: SignalDirection,
        levels: SupportResistance,
    ) -> Signal | None:
        if not self.config.allows(direction):
            return None

        is_long = direction is SignalDirection.LONG
        bar = data.iloc[-1]
        close = float(bar["close"])
        high, low = float(bar["high"]), float(bar["low"])

        level_price, level_touches = self._breakout_level(data, direction)
        buffer_fraction = self.config.breakout_buffer_pct / 100

        if level_price is None:
            broke = False
            level_detail = "No consolidation range to break"
        else:
            threshold = (
                level_price * (1 + buffer_fraction)
                if is_long
                else level_price * (1 - buffer_fraction)
            )
            broke = close > threshold if is_long else close < threshold
            level_detail = (
                f"Closed {'above the range high' if is_long else 'below the range low'} "
                f"at {level_price:.2f} ({level_touches} touch"
                f"{'es' if level_touches != 1 else ''})"
            )

        consolidated, width_atr = self._consolidation(data)
        volume = analyze_volume(data, self.indicators)
        relative_volume = volume.relative_volume
        volume_ok = (
            relative_volume is not None and relative_volume >= self.config.min_volume_spike
        )

        atr = self._consolidation_atr(data)
        if level_price is not None and atr:
            extension = abs(close - level_price) / atr
            not_chasing = extension <= self.config.max_entry_extension_atr
            extension_detail = f"Entry sits {extension:.1f} ATR beyond the level"
        else:
            not_chasing = level_price is None
            extension_detail = "Extension unavailable"

        bar_range = high - low
        if bar_range > 0:
            position_in_bar = (close - low) / bar_range
            strong_close = (
                position_in_bar >= self.config.strong_close_fraction
                if is_long
                else position_in_bar <= 1 - self.config.strong_close_fraction
            )
        else:
            strong_close = False

        well_tested = level_price is not None and level_touches >= self.config.min_level_touches
        trend = analyze_trend(data, self.indicators)
        trend_agrees = trend.direction.is_bullish if is_long else trend.direction.is_bearish
        squeezed = is_bollinger_squeeze(data.iloc[:-1], self.indicators)

        conditions = [
            Condition("break", broke, weight=2.5, required=True, detail=level_detail),
            Condition(
                "consolidation",
                consolidated,
                weight=2.0,
                required=True,
                detail=(
                    f"Consolidated within {width_atr:.1f} ATR over "
                    f"{self.config.consolidation_bars} bars"
                    if width_atr is not None
                    else "Consolidation unavailable"
                ),
            ),
            Condition(
                "volume",
                volume_ok,
                weight=2.0,
                required=True,
                detail=(
                    f"Volume {relative_volume:.2f}x average confirms the break"
                    if relative_volume is not None
                    else "Volume unavailable"
                ),
            ),
            Condition(
                "not_chasing",
                not_chasing,
                weight=1.5,
                required=True,
                detail=extension_detail,
            ),
            Condition(
                "level_quality",
                well_tested,
                weight=1.0,
                detail=(
                    f"Level tested {level_touches} time"
                    f"{'s' if level_touches != 1 else ''}"
                    if level_price is not None
                    else "No level"
                ),
            ),
            Condition(
                "strong_close",
                strong_close,
                weight=1.5,
                detail=f"Closed in the {'top' if is_long else 'bottom'} of the bar's range",
            ),
            Condition(
                "trend",
                trend_agrees,
                weight=1.0,
                detail=f"{trend.direction.value} trend supports the break",
            ),
            Condition(
                "squeeze",
                squeezed,
                weight=1.0,
                detail="Volatility was compressed before the break",
            ),
        ]

        target = (
            self._measured_move_target(data, direction, close, level_price)
            if self.config.use_measured_move
            else None
        )

        return self._build_signal(
            symbol=symbol,
            direction=direction,
            data=data,
            conditions=conditions,
            levels=levels,
            target_override=target,
            # The measured move is a preference, not the reason for the trade.
            target_is_mandatory=False,
            metadata={
                "level_price": level_price,
                "level_touches": level_touches,
                "consolidation_width_atr": width_atr,
                "relative_volume": relative_volume,
                "trend_direction": trend.direction.value,
                "squeeze_before_break": squeezed,
            },
        )

    def _breakout_level(
        self, data: pd.DataFrame, direction: SignalDirection
    ) -> tuple[float | None, int]:
        """The ceiling (or floor) of the consolidation that is being broken.

        Derived from the consolidation window itself rather than from the pivot
        levels. Pivot levels are classified relative to the current price, so
        once price clears the range every pivot has already been reclassified as
        support and the resistance that just broke no longer exists to be found.
        The range's own high is unambiguous and uses the same window that defines
        the consolidation.

        Returns
        -------
        tuple
            ``(level_price, touches)`` — how many bars in the window reached
            within a tenth of an ATR of that level, which is how many times it
            was actually tested.
        """
        bars = self.config.consolidation_bars
        window = data.iloc[-(bars + 1) : -1]
        if len(window) < bars:
            return None, 0

        is_long = direction is SignalDirection.LONG
        price = float(window["high"].max()) if is_long else float(window["low"].min())

        atr = self._consolidation_atr(data)
        tolerance = (atr * 0.1) if atr else price * 0.001
        if is_long:
            touches = int((window["high"] >= price - tolerance).sum())
        else:
            touches = int((window["low"] <= price + tolerance).sum())
        return price, touches

    def _consolidation_atr(self, data: pd.DataFrame) -> float | None:
        """ATR as of the end of the consolidation, before the break widened it.

        Using the current ATR would be circular: the breakout bar is itself a
        large true range, so it inflates ATR and makes the quiet period that
        preceded it look wider than it was — often enough to fail the very test
        it should pass.
        """
        column = atr_column(self.indicators.atr_period)
        if column not in data.columns or len(data) < 2:
            return None
        value = data[column].iloc[-2]
        if pd.isna(value) or float(value) <= 0:
            return None
        return float(value)

    def _consolidation(self, data: pd.DataFrame) -> tuple[bool, float | None]:
        """Whether the bars before this one formed a tight range.

        The breakout bar itself is excluded — its range is wide by definition and
        including it would mask the very compression being looked for.
        """
        bars = self.config.consolidation_bars
        window = data.iloc[-(bars + 1) : -1]
        if len(window) < bars:
            return False, None

        atr = self._consolidation_atr(data)
        if atr is None:
            return False, None

        span = float(window["high"].max()) - float(window["low"].min())
        width_atr = span / atr
        return width_atr <= self.config.max_consolidation_width_atr, width_atr

    def _measured_move_target(
        self,
        data: pd.DataFrame,
        direction: SignalDirection,
        close: float,
        level_price: float | None,
    ) -> float | None:
        """Project the consolidation's height from the **broken level**.

        Projecting from the entry instead would silently inflate the target by
        however far price had already run — the later the entry, the more
        generous the target, which is backwards.

        Returns ``None`` when the projection no longer sits beyond the entry: the
        move has already been made, so there is no measured move left to collect
        and the base class falls back to a volatility-based target.
        """
        bars = self.config.consolidation_bars
        window = data.iloc[-(bars + 1) : -1]
        if len(window) < bars or level_price is None:
            return None

        span = float(window["high"].max()) - float(window["low"].min())
        if span <= 0:
            return None

        target = level_price + span * direction.sign
        reached = target <= close if direction is SignalDirection.LONG else target >= close
        return None if reached else target

    def evaluate_exit(self, position: Position, data: pd.DataFrame) -> ExitSignal | None:
        """Exit when the breakout fails and price falls back into the range."""
        if not self.config.exit_on_failed_breakout:
            return None

        level_price = position.metadata.get("level_price")
        if level_price is None:
            return None

        close = float(data.iloc[-1]["close"])
        failed = (
            close < float(level_price)
            if position.direction is SignalDirection.LONG
            else close > float(level_price)
        )
        if failed:
            return ExitSignal(
                reason=ExitReason.SIGNAL_FLIP,
                price=close,
                detail=f"Price fell back through {float(level_price):.2f} — the break failed",
            )
        return None
