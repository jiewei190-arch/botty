"""Mean reversion: fade a stretched move back toward its average.

The thesis is the opposite of momentum — price that has moved far from its mean
in a short time tends to snap back. That works in ranging markets and fails
badly in trending ones, which is why the strongest filter here is a *veto*: this
strategy refuses to trade against a strong trend. Buying oversold readings in a
collapse is the classic way to lose money with an indicator that looks right.

Entry (long)
------------
Required:

* RSI at or below the oversold threshold.
* Price stretched to or beyond the lower Bollinger Band.
* **Confirmation** — the setup alone is not enough. Price must have closed back
  inside the band, or printed a reversal bar, or turned RSI up. Without this the
  strategy catches falling knives.
* The trend is not strongly bearish. A stretched market in a strong downtrend is
  not oversold, it is trending.

Optional:

* Price a long way below its moving average, measured in standard deviations.
* Elevated volume, suggesting capitulation rather than a drift.
* A support level nearby.

Target
------
The mean itself — the middle Bollinger Band — not an ATR multiple. If reverting
to the mean does not pay at least ``min_risk_reward`` against the stop, the
signal is dropped rather than have its target stretched past the thesis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from trading_bot.indicators import (
    BB_LOWER_COL,
    BB_MIDDLE_COL,
    BB_PERCENT_B_COL,
    BB_UPPER_COL,
    analyze_trend,
    analyze_volume,
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
class MeanReversionConfig(StrategyConfig):
    """Parameters for :class:`MeanReversionStrategy`.

    Two inherited defaults are deliberately overridden, because mean reversion
    has a different risk shape from trend following:

    * ``atr_stop_multiplier`` is tighter (1.0 rather than 1.5). The thesis is
      "price bounces from here"; if it keeps going, the thesis is wrong
      immediately and there is nothing to be gained by giving it room.
    * ``min_risk_reward`` is lower (1.5 rather than 2.0). A reversion to the mean
      pays roughly one standard deviation, so demanding a trend-following 2:1
      would reject essentially every valid setup. Mean reversion earns its
      expectancy from a **high win rate at modest reward**, the mirror image of
      momentum. Comparing the two on reward:risk alone is a category error —
      compare them on expectancy, which Phase 6's backtester measures.
    """

    atr_stop_multiplier: float = 1.0
    min_risk_reward: float = 1.5

    #: RSI at or below this is a candidate long.
    rsi_oversold: float = 30.0
    #: RSI at or above this is a candidate short.
    rsi_overbought: float = 70.0
    #: Bollinger %B at or below this counts as stretched to the downside.
    percent_b_low: float = 0.05
    #: Bollinger %B at or above this counts as stretched to the upside.
    percent_b_high: float = 0.95
    #: Minimum deviation from the moving average, in standard deviations.
    min_deviation_sigma: float = 1.8
    #: Require a confirmation bar before entering. Disable at your peril.
    require_confirmation: bool = True
    #: Bars within which the stretch must have occurred for it to still be fresh.
    confirmation_window: int = 3
    #: Bars over which the regime filter measures the slow EMA's drift.
    regime_lookback: int = 20
    #: Maximum adverse drift of the slow EMA, in ATR units over ``regime_lookback``
    #: bars, before a counter-trend entry is refused. Calibrated against measured
    #: behaviour: a range-bound market keeps this within roughly +/-1, a steady
    #: trend runs 1-4, and a sustained collapse reaches -10 and beyond.
    max_adverse_regime_drift: float = 3.0
    #: Volume multiple that suggests capitulation rather than drift.
    capitulation_volume: float = 1.3
    #: Exit once price has reverted this far toward the mean (1.0 = fully).
    reversion_exit_fraction: float = 1.0

    def __post_init__(self) -> None:
        # Explicit parent call, not super(): @dataclass(slots=True) builds a *new*
        # class object, which invalidates the __class__ cell that zero-argument
        # super() relies on.
        StrategyConfig.__post_init__(self)
        if not 0 <= self.rsi_oversold < self.rsi_overbought <= 100:
            raise ValueError(
                "Require 0 <= rsi_oversold < rsi_overbought <= 100, got "
                f"{self.rsi_oversold} and {self.rsi_overbought}"
            )
        if not 0 <= self.percent_b_low < self.percent_b_high <= 1:
            raise ValueError(
                "Require 0 <= percent_b_low < percent_b_high <= 1, got "
                f"{self.percent_b_low} and {self.percent_b_high}"
            )
        if self.confirmation_window < 1:
            raise ValueError(
                f"confirmation_window must be >= 1, got {self.confirmation_window}"
            )
        if not 0 < self.reversion_exit_fraction <= 1:
            raise ValueError(
                f"reversion_exit_fraction must be within (0, 1], got "
                f"{self.reversion_exit_fraction}"
            )


class MeanReversionStrategy(BaseStrategy):
    """Fade stretched moves back to the mean, with a strong-trend veto.

    Example
    -------
    >>> strategy = MeanReversionStrategy(MeanReversionConfig(rsi_oversold=25))
    >>> signal = strategy.generate_signal("SPY", strategy.prepare(bars))
    """

    name = "mean_reversion"

    def __init__(self, config: MeanReversionConfig | None = None, indicators=None) -> None:
        super().__init__(config or MeanReversionConfig(), indicators)
        self.config: MeanReversionConfig

    @property
    def min_bars(self) -> int:
        """Enough history for the slow EMA *and* the regime window behind it.

        Without the extra window the regime filter has nothing to measure on its
        first bars, and the strategy would trade blind exactly where it is most
        dangerous.
        """
        periods = sorted(self.indicators.ema_periods)
        slowest = periods[-1] if periods else 0
        return max(self.indicators.max_lookback, slowest + self.config.regime_lookback + 1)

    def evaluate(self, symbol: str, data: pd.DataFrame) -> Signal | None:
        trend = analyze_trend(data, self.indicators)
        signal = self._evaluate_direction(symbol, data, SignalDirection.LONG, trend)
        if signal is not None:
            return signal
        return self._evaluate_direction(symbol, data, SignalDirection.SHORT, trend)

    def _evaluate_direction(
        self, symbol: str, data: pd.DataFrame, direction: SignalDirection, trend
    ) -> Signal | None:
        if not self.config.allows(direction):
            return None

        is_long = direction is SignalDirection.LONG
        bar = data.iloc[-1]
        close = float(bar["close"])

        rsi = _latest(data, rsi_column(self.indicators.rsi_period))
        percent_b = _latest(data, BB_PERCENT_B_COL)
        middle = _latest(data, BB_MIDDLE_COL)
        band = _latest(data, BB_LOWER_COL if is_long else BB_UPPER_COL)

        # Without a mean to revert to there is no trade.
        if middle is None or band is None:
            return None

        rsi_stretched = rsi is not None and (
            rsi <= self.config.rsi_oversold if is_long else rsi >= self.config.rsi_overbought
        )
        band_stretched = self._recently_stretched(data, direction)

        confirmed, confirmation_detail = self._confirmation(data, direction)
        regime_ok, regime_detail = self._regime_permits(data, is_long)

        deviation = self._deviation_sigma(data, close, middle)
        deviation_ok = deviation is not None and deviation >= self.config.min_deviation_sigma

        volume = analyze_volume(data, self.indicators)
        relative_volume = volume.relative_volume
        capitulation = (
            relative_volume is not None and relative_volume >= self.config.capitulation_volume
        )

        levels = find_support_resistance(data, self.indicators)
        level = levels.nearest_support if is_long else levels.nearest_resistance
        near_level = level is not None and abs(level.distance_pct(close)) <= 2.0

        conditions = [
            Condition(
                "rsi_stretched",
                rsi_stretched,
                weight=2.0,
                required=True,
                detail=(
                    f"RSI {rsi:.1f} is {'oversold' if is_long else 'overbought'}"
                    if rsi is not None
                    else "RSI unavailable"
                ),
            ),
            Condition(
                "band_stretched",
                band_stretched,
                weight=2.0,
                required=True,
                detail=(
                    f"Price stretched to the {'lower' if is_long else 'upper'} Bollinger Band"
                ),
            ),
            Condition(
                "confirmation",
                confirmed,
                weight=2.0,
                required=self.config.require_confirmation,
                detail=confirmation_detail,
            ),
            Condition(
                "regime_permits",
                regime_ok,
                weight=1.5,
                required=True,
                detail=regime_detail,
            ),
            Condition(
                "deviation",
                deviation_ok,
                weight=1.0,
                detail=(
                    f"Price {deviation:.1f}σ from its mean"
                    if deviation is not None
                    else "Deviation unavailable"
                ),
            ),
            Condition(
                "capitulation",
                capitulation,
                weight=1.0,
                detail=(
                    f"Volume {relative_volume:.2f}x average suggests capitulation"
                    if relative_volume is not None
                    else "Volume unavailable"
                ),
            ),
            Condition(
                "structure",
                near_level,
                weight=1.0,
                detail=(
                    f"{'Support' if is_long else 'Resistance'} within "
                    f"{abs(level.distance_pct(close)):.2f}%"
                    if level is not None
                    else "No nearby level"
                ),
            ),
        ]

        # Revert to the mean; partial targets exit before the band for realism.
        target = close + (middle - close) * self.config.reversion_exit_fraction

        return self._build_signal(
            symbol=symbol,
            direction=direction,
            data=data,
            conditions=conditions,
            levels=levels,
            target_override=target,
            metadata={
                "rsi": rsi,
                "percent_b": percent_b,
                "middle_band": middle,
                "deviation_sigma": deviation,
                "relative_volume": relative_volume,
                "trend_direction": trend.direction.value,
                "trend_strength": trend.strength,
            },
        )

    def _regime_permits(self, data: pd.DataFrame, is_long: bool) -> tuple[bool, str]:
        """Refuse to fade a sustained move; permit fading a swing inside a range.

        The short-term trend classifier cannot make this call. At the trough of a
        range every one of its components genuinely reads bearish — that *is* the
        setup — so using it as the veto would block every valid mean-reversion
        long.

        The regime is measured instead from the drift of the slowest configured
        EMA over ``regime_lookback`` bars, scaled by ATR. That average is slow
        enough to ignore individual swings: it stays near flat while price
        oscillates inside a range, and only moves decisively once the range
        itself starts migrating.
        """
        periods = sorted(self.indicators.ema_periods)
        column = ema_column(periods[-1]) if periods else None
        atr = self._atr_value(data)
        lookback = self.config.regime_lookback

        # Fail *closed*. This strategy's characteristic failure is catching a
        # falling knife, and an unassessable regime is exactly when that happens.
        # Refusing to trade costs an opportunity; permitting blindly costs money.
        if column is None or column not in data.columns or atr is None:
            return False, "Regime cannot be assessed — refusing a counter-trend entry"

        series = data[column].dropna()
        if len(series) < lookback + 1:
            return False, "Too little history to judge the regime — refusing"

        drift = (float(series.iloc[-1]) - float(series.iloc[-1 - lookback])) / atr
        limit = self.config.max_adverse_regime_drift
        permitted = drift >= -limit if is_long else drift <= limit

        if permitted:
            return True, f"No sustained opposing move ({drift:+.1f} ATR over {lookback} bars)"
        return False, f"Refusing to fade a sustained move ({drift:+.1f} ATR over {lookback} bars)"

    def _recently_stretched(self, data: pd.DataFrame, direction: SignalDirection) -> bool:
        """Whether %B breached its threshold within the confirmation window.

        Checking a window rather than only the current bar is what lets the
        strategy enter *after* price closes back inside the band — the stretch is
        the setup, the re-entry is the trigger.
        """
        if BB_PERCENT_B_COL not in data.columns:
            return False
        window = data[BB_PERCENT_B_COL].tail(self.config.confirmation_window + 1).dropna()
        if window.empty:
            return False
        if direction is SignalDirection.LONG:
            return bool((window <= self.config.percent_b_low).any())
        return bool((window >= self.config.percent_b_high).any())

    def _confirmation(self, data: pd.DataFrame, direction: SignalDirection) -> tuple[bool, str]:
        """Look for evidence the fall has actually stopped."""
        if not self.config.require_confirmation:
            return True, "Confirmation not required"
        if len(data) < 2:
            return False, "Too little history to confirm"

        is_long = direction is SignalDirection.LONG
        bar = data.iloc[-1]
        close, open_ = float(bar["close"]), float(bar["open"])

        # 1. A reversal bar: closed against the prevailing pressure.
        if (close > open_) if is_long else (close < open_):
            return True, f"Reversal bar closed {'up' if is_long else 'down'}"

        # 2. Price closed back inside the band after being outside it.
        percent_b = _latest(data, BB_PERCENT_B_COL)
        if percent_b is not None:
            inside = (
                percent_b > self.config.percent_b_low
                if is_long
                else percent_b < self.config.percent_b_high
            )
            if inside:
                return True, "Price closed back inside the Bollinger Band"

        # 3. RSI has turned up from its low.
        column = rsi_column(self.indicators.rsi_period)
        if column in data.columns:
            recent = data[column].tail(3).dropna()
            if len(recent) >= 2:
                turning = (
                    recent.iloc[-1] > recent.iloc[-2]
                    if is_long
                    else recent.iloc[-1] < recent.iloc[-2]
                )
                if turning:
                    return True, f"RSI turning {'up' if is_long else 'down'}"

        return False, "Awaiting confirmation"

    def _deviation_sigma(
        self, data: pd.DataFrame, close: float, middle: float
    ) -> float | None:
        """How far price sits from its mean, in standard deviations.

        Derived from the band width so it uses the same σ the bands do.
        """
        upper = _latest(data, BB_UPPER_COL)
        if upper is None or middle is None:
            return None
        sigma = (upper - middle) / self.indicators.bollinger_std
        if sigma <= 0:
            return None
        return abs(close - middle) / sigma

    def evaluate_exit(self, position: Position, data: pd.DataFrame) -> ExitSignal | None:
        """Exit once price has reverted to the mean, or RSI has normalised."""
        close = float(data.iloc[-1]["close"])
        is_long = position.direction is SignalDirection.LONG

        middle = _latest(data, BB_MIDDLE_COL)
        if middle is not None:
            reverted = close >= middle if is_long else close <= middle
            if reverted:
                return ExitSignal(
                    reason=ExitReason.TARGET_REACHED,
                    price=close,
                    detail=f"Price reverted to its mean at {middle:.2f}",
                )

        rsi = _latest(data, rsi_column(self.indicators.rsi_period))
        if rsi is not None:
            normalised = (
                rsi >= self.config.rsi_overbought
                if is_long
                else rsi <= self.config.rsi_oversold
            )
            if normalised:
                return ExitSignal(
                    reason=ExitReason.TARGET_REACHED,
                    price=close,
                    detail=f"RSI reached {rsi:.1f} — the stretch has unwound",
                )
        return None


def _latest(data: pd.DataFrame, column: str) -> float | None:
    """Latest value of ``column``, or ``None`` when missing or warming up."""
    if column not in data.columns:
        return None
    value = data[column].iloc[-1]
    return None if pd.isna(value) else float(value)
