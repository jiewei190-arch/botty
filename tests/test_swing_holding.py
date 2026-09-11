"""The holding period a swing strategy actually delivers.

These tests exist because the strategy's documentation and its behaviour had
drifted apart. ``swing_quality`` declared a one-to-four week hold and a 20-bar
cap; measured on real AAPL and MSFT daily bars it held for a median of two
trading days and ran two trades past the cap to 22 and 30 bars.

Each test below pins one of the three mechanisms that caused that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_bot.backtesting.engine import BacktestConfig, Backtester
from trading_bot.data.market_data import normalize_bars
from trading_bot.strategies import build_strategy
from trading_bot.strategies.base_strategy import (
    BaseStrategy,
    Condition,
    ExitReason,
    Position,
    Signal,
    SignalDirection,
    StrategyConfig,
    entry_drift_fraction,
    entry_still_valid,
)
from trading_bot.strategies.swing_quality import SwingQualityConfig


def flat_frame(periods: int = 260, *, price: float = 100.0, wick: float = 1.0) -> pd.DataFrame:
    """Daily bars that go nowhere, so only the mechanism under test moves."""
    index = pd.date_range("2024-01-02", periods=periods, freq="B", tz="UTC")
    closes = np.full(periods, price)
    return normalize_bars(
        pd.DataFrame(
            {
                "open": closes,
                "high": closes + wick,
                "low": closes - wick,
                "close": closes,
                "volume": np.full(periods, 1e6),
            },
            index=index,
        ),
        symbol="T",
    )


class FiresOnce(BaseStrategy):
    """Signals on one nominated bar and never exits of its own accord."""

    name = "fires_once"

    def __init__(self, config: StrategyConfig, *, fire_at: int) -> None:
        super().__init__(config)
        self.fire_at = fire_at

    def evaluate(self, symbol: str, data: pd.DataFrame) -> Signal | None:
        if len(data) != self.fire_at:
            return None
        return self._build_signal(
            symbol=symbol,
            direction=SignalDirection.LONG,
            data=data,
            conditions=[Condition("always", True, weight=1.0, required=True, detail="test")],
        )


def position_held(bars: int) -> Position:
    return Position(
        symbol="T",
        direction=SignalDirection.LONG,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        quantity=10,
        entry_timestamp=pd.Timestamp("2024-01-02", tz="UTC").to_pydatetime(),
        strategy="fires_once",
        bars_held=bars,
    )


class TestTimeStop:
    def test_returns_nothing_before_the_limit(self):
        strategy = FiresOnce(StrategyConfig(max_holding_bars=20), fire_at=220)
        assert strategy.check_time_stop(position_held(19), price=100.0) is None

    def test_fires_at_the_limit(self):
        strategy = FiresOnce(StrategyConfig(max_holding_bars=20), fire_at=220)
        signal = strategy.check_time_stop(position_held(20), price=100.0)
        assert signal is not None
        assert signal.reason is ExitReason.TIME_STOP
        assert "20" in signal.detail

    def test_no_limit_never_fires(self):
        strategy = FiresOnce(StrategyConfig(max_holding_bars=None), fire_at=220)
        assert strategy.check_time_stop(position_held(5_000), price=100.0) is None

    def test_a_backtest_honours_the_cap(self):
        """The regression: the cap lived in should_exit(), which nothing called.

        Without an enforced cap this position never closes — the frame is flat,
        so neither the stop nor the target is ever touched.
        """
        strategy = FiresOnce(
            StrategyConfig(min_confidence=0.0, max_holding_bars=10), fire_at=220
        )
        result = Backtester(
            [strategy],
            BacktestConfig(starting_equity=100_000.0, timeframe="1Day", warmup_bars=210,
                           close_at_end=False),
        ).run({"T": flat_frame()})

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade["exit_reason"] == ExitReason.TIME_STOP.value
        # One bar of latency is by design: the decision reads a bar's close and
        # fills at the next open, exactly as an entry does.
        assert trade["bars_held"] <= 11

    def test_an_uncapped_strategy_still_runs_to_the_end(self):
        strategy = FiresOnce(
            StrategyConfig(min_confidence=0.0, max_holding_bars=None), fire_at=220
        )
        result = Backtester(
            [strategy],
            BacktestConfig(starting_equity=100_000.0, timeframe="1Day", warmup_bars=210,
                           close_at_end=False),
        ).run({"T": flat_frame()})
        assert result.trades == []  # still open at the end, and not force-closed


def signal_at(entry: float, stop: float) -> Signal:
    return Signal(
        symbol="T",
        direction=SignalDirection.LONG,
        timestamp=pd.Timestamp("2024-01-02", tz="UTC").to_pydatetime(),
        entry_price=entry,
        stop_loss=stop,
        take_profit=entry + (entry - stop) * 2,
        confidence=80.0,
        strategy="t",
    )


class TestEntryDrift:
    @pytest.mark.parametrize(
        "fill,expected",
        [
            (100.0, 0.0),    # filled where the signal said
            (101.0, 0.0),    # filled better; never counts against the setup
            (98.5, 0.5),     # half the planned risk gone
            (97.0, 1.0),     # opens already at the stop
            (95.0, 1.667),   # gapped clean through it
        ],
    )
    def test_fraction_of_planned_risk_consumed(self, fill, expected):
        assert entry_drift_fraction(signal_at(100.0, 97.0), fill) == pytest.approx(
            expected, abs=1e-3
        )

    def test_shorts_measure_drift_upward(self):
        short = Signal(
            symbol="T",
            direction=SignalDirection.SHORT,
            timestamp=pd.Timestamp("2024-01-02", tz="UTC").to_pydatetime(),
            entry_price=100.0,
            stop_loss=103.0,
            take_profit=94.0,
            confidence=80.0,
            strategy="t",
        )
        assert entry_drift_fraction(short, 101.5) == pytest.approx(0.5)
        assert entry_drift_fraction(short, 98.0) == 0.0

    def test_a_stop_at_the_entry_is_refused_upstream(self):
        """The zero-risk guard in the helper is belt and braces, not the wall.

        A Signal cannot carry a stop at its own entry price, so the degenerate
        case the helper guards against never reaches it from real code.
        """
        with pytest.raises(ValueError, match="must be below entry"):
            signal_at(100.0, 100.0)

    def test_limit_is_inclusive(self):
        assert entry_still_valid(signal_at(100.0, 97.0), 98.5, limit=0.5)
        assert not entry_still_valid(signal_at(100.0, 97.0), 98.4, limit=0.5)

    def test_a_backtest_skips_a_setup_that_gapped_away(self):
        """The measured case: a 1.4% overnight gap left 0.035% of intended room."""
        frame = flat_frame()
        values = frame.copy()
        # The fill bar opens most of the way to a stop that sits ~3% below.
        values.iloc[220, values.columns.get_loc("open")] = 97.2
        values.iloc[220, values.columns.get_loc("close")] = 97.4
        values.iloc[220, values.columns.get_loc("low")] = 96.2

        config = {"min_confidence": 0.0, "max_holding_bars": None}
        strict = Backtester(
            [FiresOnce(StrategyConfig(**config, max_entry_drift=0.5), fire_at=220)],
            BacktestConfig(starting_equity=100_000.0, timeframe="1Day", warmup_bars=210),
        ).run({"T": values})
        assert strict.signals_generated == 1
        assert strict.trades == []
        assert strict.rejection_reasons["gapped past the entry"] == 1

        lenient = Backtester(
            [FiresOnce(StrategyConfig(**config, max_entry_drift=1.0), fire_at=220)],
            BacktestConfig(starting_equity=100_000.0, timeframe="1Day", warmup_bars=210),
        ).run({"T": values})
        assert len(lenient.trades) == 1

    def test_the_limit_is_validated(self):
        with pytest.raises(ValueError, match="max_entry_drift"):
            StrategyConfig(max_entry_drift=0.0)
        with pytest.raises(ValueError, match="max_entry_drift"):
            StrategyConfig(max_entry_drift=1.5)


class TestSwingQualityGeometry:
    """The exit distances have to agree with the holding window they claim."""

    def test_target_is_reachable_before_the_holding_cap(self):
        config = build_strategy("swing_quality").config
        assert config.max_holding_bars is not None
        # A random walk covers k ATRs in roughly k^2 bars. A target further out
        # than the cap can reach means the clock closes working trades.
        bars_to_target = config.atr_target_multiplier**2
        assert bars_to_target <= config.max_holding_bars, (
            f"target at {config.atr_target_multiplier} ATR needs about "
            f"{bars_to_target:.0f} bars but the cap is {config.max_holding_bars}"
        )

    def test_the_stop_survives_the_first_few_sessions(self):
        config = build_strategy("swing_quality").config
        # Under ~4 bars of expected survival the strategy is day trading.
        assert config.atr_stop_multiplier**2 >= 4.0
        # And structure must never pull the stop back inside that.
        assert config.min_stop_atr >= 2.0

    def test_reward_to_risk_still_clears_the_floor(self):
        config = build_strategy("swing_quality").config
        ratio = config.atr_target_multiplier / config.atr_stop_multiplier
        assert ratio >= config.min_risk_reward

    def test_the_cap_is_inside_a_thirty_calendar_day_window(self):
        config = build_strategy("swing_quality").config
        # ~21 trading days is 30 calendar days.
        assert config.max_holding_bars <= 21

    def test_the_alert_advertises_the_configured_cap(self):
        """Signal metadata is what an alert shows; it must not drift from config."""
        strategy = build_strategy("swing_quality")
        assert strategy.config.max_holding_bars == SwingQualityConfig().max_holding_bars
