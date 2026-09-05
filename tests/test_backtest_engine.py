"""Backtest engine tests.

The engine's whole value is that it does not cheat, so most of these tests are
about what it refuses to do: fill at a price it has only just seen, resolve an
ambiguous bar in its own favour, or let an open loss escape the results.

A scripted strategy is used throughout. Testing the engine through a real
strategy would conflate "did the simulation execute correctly" with "did the
strategy fire", and only the first is under test here.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from trading_bot.backtesting import (
    FRICTIONLESS,
    BacktestConfig,
    Backtester,
    CostModel,
)
from trading_bot.config.settings import RiskSettings
from trading_bot.strategies.base_strategy import (
    BaseStrategy,
    ExitReason,
    ExitSignal,
    Position,
    Signal,
    SignalDirection,
    StrategyConfig,
)

START = pd.Timestamp("2025-03-03 14:30", tz="UTC")


def frame(rows: list[tuple[float, float, float, float]], start=START) -> pd.DataFrame:
    """Build an OHLCV frame from (open, high, low, close) tuples."""
    index = pd.date_range(start, periods=len(rows), freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [1_000_000.0] * len(rows),
        },
        index=index,
    )


def flat(bars: int, price: float = 100.0) -> pd.DataFrame:
    return frame([(price, price, price, price)] * bars)


@dataclass(frozen=True, slots=True)
class ScriptedConfig(StrategyConfig):
    pass


class ScriptedStrategy(BaseStrategy):
    """Emits signals at named bar positions, so the engine can be tested alone."""

    name = "scripted"

    def __init__(
        self,
        entries: dict[int, tuple[float, float, float]] | None = None,
        *,
        exit_at: set[int] | None = None,
        direction: SignalDirection = SignalDirection.LONG,
        warmup: int = 0,
    ) -> None:
        super().__init__(ScriptedConfig(min_confidence=0.0))
        self.entries = entries or {}
        self.exit_at = exit_at or set()
        self.direction = direction
        self._warmup = warmup

    @property
    def min_bars(self) -> int:
        return self._warmup

    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        return data  # no indicators needed

    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Signal | None:
        return self.evaluate(symbol, data)

    def evaluate(self, symbol: str, data: pd.DataFrame) -> Signal | None:
        position = len(data) - 1
        prices = self.entries.get(position)
        if prices is None:
            return None
        entry, stop, target = prices
        return Signal(
            symbol=symbol,
            direction=self.direction,
            strategy=self.name,
            confidence=90.0,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            timestamp=data.index[-1].to_pydatetime(),
        )

    def evaluate_exit(self, position: Position, data: pd.DataFrame) -> ExitSignal | None:
        if len(data) - 1 in self.exit_at:
            return ExitSignal(
                reason=ExitReason.SIGNAL_FLIP,
                price=float(data.iloc[-1]["close"]),
                detail="scripted",
            )
        return None


def run(strategy, frames, **config):
    settings = dict(costs=FRICTIONLESS, starting_equity=100_000.0)
    settings.update(config)
    return Backtester([strategy], BacktestConfig(**settings)).run(frames)


class TestConstruction:
    def test_a_backtest_needs_a_strategy(self):
        with pytest.raises(ValueError, match="at least one strategy"):
            Backtester([])

    def test_a_backtest_needs_data(self):
        with pytest.raises(ValueError, match="No data"):
            run(ScriptedStrategy(), {})

    def test_starting_equity_must_be_positive(self):
        with pytest.raises(ValueError, match="starting_equity"):
            BacktestConfig(starting_equity=0)

    def test_timeframe_is_validated_at_construction(self):
        with pytest.raises(ValueError):
            BacktestConfig(timeframe="nonsense")


class TestNoLookahead:
    def test_entry_fills_at_the_next_bar_open_not_the_signal_bar(self):
        """The single most common backtest lie, pinned."""
        bars = frame(
            [(100, 100, 100, 100)] * 5
            + [(100, 100, 100, 100)]  # bar 5: signal generated here
            + [(120, 121, 119, 120)]  # bar 6: fills at 120, the open we could reach
            + [(120, 121, 119, 120)] * 3
        )
        result = run(ScriptedStrategy({5: (100.0, 90.0, 200.0)}), {"AAA": bars})
        assert len(result.trades) == 1
        assert result.trades[0]["entry_price"] == 120.0
        assert result.trades[0]["entry_time"] == bars.index[6]

    def test_a_signal_on_the_final_bar_never_fills(self):
        """There is no next bar to fill at, so it must expire, not fill at the close."""
        bars = flat(8)
        result = run(ScriptedStrategy({7: (100.0, 90.0, 120.0)}), {"AAA": bars})
        assert result.trades == []
        assert result.signals_generated == 1

    def test_the_strategy_only_ever_sees_history(self):
        """Recorded from inside the strategy: no frame may extend past its own bar."""
        seen: list[tuple[int, pd.Timestamp]] = []
        bars = flat(12)

        class Watcher(ScriptedStrategy):
            def evaluate(self, symbol, data):
                seen.append((len(data), data.index[-1]))
                return None

        run(Watcher(), {"AAA": bars})
        assert seen, "strategy was never called"
        for length, last in seen:
            assert last == bars.index[length - 1]
            assert length <= len(bars)

    def test_warmup_bars_are_not_traded(self):
        bars = flat(20)
        strategy = ScriptedStrategy({i: (100.0, 90.0, 120.0) for i in range(20)})
        result = run(strategy, {"AAA": bars}, warmup_bars=15)
        # Signals start at bar 15 and fill on 16, so no earlier entry exists.
        assert result.trades[0]["entry_time"] == bars.index[16]


class TestExits:
    def test_stop_closes_the_position_at_the_stop(self):
        bars = frame(
            [(100, 100, 100, 100)] * 3
            + [(100, 101, 99, 100)]  # 3: signal
            + [(100, 101, 99, 100)]  # 4: entry at 100
            + [(100, 101, 94, 96)]  # 5: low pierces the 95 stop
            + [(96, 97, 95, 96)]
        )
        result = run(ScriptedStrategy({3: (100.0, 95.0, 130.0)}), {"AAA": bars})
        assert len(result.trades) == 1
        assert result.trades[0]["exit_reason"] == "STOP_LOSS"
        assert result.trades[0]["exit_price"] == 95.0
        assert result.trades[0]["gapped"] is False

    def test_a_gap_through_the_stop_fills_at_the_open(self):
        bars = frame(
            [(100, 101, 99, 100)] * 3
            + [(100, 101, 99, 100)]  # 3: signal
            + [(100, 101, 99, 100)]  # 4: entry at 100
            + [(88, 89, 87, 88)]  # 5: opens far below the 95 stop
            + [(88, 89, 87, 88)]
        )
        result = run(ScriptedStrategy({3: (100.0, 95.0, 130.0)}), {"AAA": bars})
        assert result.trades[0]["exit_price"] == 88.0
        assert result.trades[0]["gapped"] is True
        assert result.metrics.gapped_stops == 1

    def test_target_closes_at_the_limit(self):
        bars = frame(
            [(100, 101, 99, 100)] * 3
            + [(100, 101, 99, 100)]
            + [(100, 101, 99, 100)]  # 4: entry at 100
            + [(100, 140, 99, 135)]  # 5: trades through the 110 target
            + [(135, 136, 134, 135)]
        )
        result = run(ScriptedStrategy({3: (100.0, 95.0, 110.0)}), {"AAA": bars})
        assert result.trades[0]["exit_reason"] == "TAKE_PROFIT"
        assert result.trades[0]["exit_price"] == 110.0  # never the 140 high

    def test_the_stop_wins_a_bar_that_touches_both(self):
        """The intrabar path is unknowable; assuming the good outcome is cheating."""
        bars = frame(
            [(100, 101, 99, 100)] * 3
            + [(100, 101, 99, 100)]
            + [(100, 101, 99, 100)]  # 4: entry at 100
            + [(100, 115, 89, 100)]  # 5: hits both the 110 target and the 95 stop
            + [(100, 101, 99, 100)]
        )
        result = run(ScriptedStrategy({3: (100.0, 95.0, 110.0)}), {"AAA": bars})
        assert result.trades[0]["exit_reason"] == "STOP_LOSS"

    def test_a_discretionary_exit_fills_at_the_open(self):
        bars = frame(
            [(100, 101, 99, 100)] * 3
            + [(100, 101, 99, 100)]
            + [(100, 101, 99, 100)]  # 4: entry at 100
            + [(104, 105, 103, 104)]  # 5: scripted exit, fills at this open
            + [(104, 105, 103, 104)]
        )
        strategy = ScriptedStrategy({3: (100.0, 90.0, 200.0)}, exit_at={5})
        result = run(strategy, {"AAA": bars})
        assert result.trades[0]["exit_reason"] == "SIGNAL_FLIP"
        assert result.trades[0]["exit_price"] == 104.0

    def test_open_positions_are_closed_at_the_end(self):
        """An unrealised loss must not escape by simply never being closed."""
        bars = frame(
            [(100, 101, 99, 100)] * 4
            + [(100, 101, 99, 100)]
            + [(97, 98, 96, 97)]
        )
        result = run(ScriptedStrategy({3: (100.0, 80.0, 200.0)}), {"AAA": bars})
        assert len(result.trades) == 1
        assert result.trades[0]["exit_reason"] == "end_of_data"
        assert result.trades[0]["pnl"] < 0

    def test_close_at_end_can_be_disabled(self):
        bars = frame([(100, 101, 99, 100)] * 4 + [(100, 101, 99, 100)] * 2)
        result = run(
            ScriptedStrategy({3: (100.0, 80.0, 200.0)}),
            {"AAA": bars},
            close_at_end=False,
        )
        assert result.trades == []


class TestAccounting:
    def test_profit_reaches_the_equity_curve(self):
        bars = frame(
            [(100, 101, 99, 100)] * 3
            + [(100, 101, 99, 100)]
            + [(100, 101, 99, 100)]  # 4: entry at 100
            + [(100, 120, 99, 119)]  # 5: target 110 hit
            + [(119, 120, 118, 119)]
        )
        result = run(ScriptedStrategy({3: (100.0, 95.0, 110.0)}), {"AAA": bars})
        trade = result.trades[0]
        assert trade["pnl"] == pytest.approx(trade["quantity"] * 10.0)
        assert result.equity_curve.iloc[-1] == pytest.approx(
            100_000.0 + trade["pnl"]
        )
        assert result.metrics.ending_equity == pytest.approx(
            result.equity_curve.iloc[-1]
        )

    def test_costs_are_charged_on_both_sides(self):
        bars = frame(
            [(100, 101, 99, 100)] * 3
            + [(100, 101, 99, 100)]
            + [(100, 101, 99, 100)]
            + [(100, 120, 99, 119)]
            + [(119, 120, 118, 119)]
        )
        result = run(
            ScriptedStrategy({3: (100.0, 95.0, 110.0)}),
            {"AAA": bars},
            costs=CostModel(commission_per_trade=1.0, slippage_pct=0.0),
        )
        trade = result.trades[0]
        assert trade["commission"] == pytest.approx(2.0)
        assert trade["pnl"] == pytest.approx(trade["gross_pnl"] - 2.0)

    def test_r_multiple_is_measured_against_the_planned_risk(self):
        bars = frame(
            [(100, 101, 99, 100)] * 3
            + [(100, 101, 99, 100)]
            + [(100, 101, 99, 100)]  # entry 100, stop 95 -> 5.00 of risk
            + [(100, 120, 99, 119)]  # target 110 -> +10.00 = 2R
            + [(119, 120, 118, 119)]
        )
        result = run(ScriptedStrategy({3: (100.0, 95.0, 110.0)}), {"AAA": bars})
        assert result.trades[0]["r_multiple"] == pytest.approx(2.0)

    def test_equity_curve_has_one_point_per_bar(self):
        bars = flat(30)
        result = run(ScriptedStrategy(), {"AAA": bars})
        assert len(result.equity_curve) == len(bars)
        assert result.metrics.bars == len(bars)
        assert list(result.equity_curve.index) == list(bars.index)

    def test_a_run_with_no_signals_leaves_equity_untouched(self):
        result = run(ScriptedStrategy(), {"AAA": flat(30)})
        assert result.trades == []
        assert (result.equity_curve == 100_000.0).all()
        assert result.metrics.total_return_pct == 0.0
        assert result.metrics.exposure_pct == 0.0


class TestRiskIntegration:
    def test_a_signal_failing_risk_never_becomes_a_trade(self):
        """The engine may not open a position the risk manager rejected."""
        bars = frame([(100, 101, 99, 100)] * 4 + [(100, 101, 99, 100)] * 3)
        # 1:1 reward:risk fails the 2:1 minimum.
        result = run(
            ScriptedStrategy({3: (100.0, 95.0, 105.0)}),
            {"AAA": bars},
            risk=RiskSettings(min_risk_reward_ratio=2.0),
        )
        assert result.trades == []
        assert result.signals_rejected == 1
        assert result.rejection_reasons

    def test_rejections_are_grouped_by_check_not_by_message(self):
        """Messages carry prices, so grouping on them yields a key per rejection."""
        entries = {i: (100.0, 95.0, 105.0 + i * 0.01) for i in range(4, 25)}
        result = run(
            ScriptedStrategy(entries),
            {"AAA": flat(30)},
            risk=RiskSettings(min_risk_reward_ratio=2.0),
        )
        assert result.signals_rejected > 5
        assert len(result.rejection_reasons) == 1

    def test_max_open_positions_is_respected(self):
        frames = {name: flat(12) for name in ("AAA", "BBB", "CCC")}
        strategy = ScriptedStrategy({i: (100.0, 90.0, 130.0) for i in range(3, 10)})
        result = run(
            strategy, frames, risk=RiskSettings(max_open_positions=2)
        )
        # Never more than two positions were open at once.
        opens = sorted((t["entry_time"], t["exit_time"]) for t in result.trades)
        for entry, _ in opens:
            concurrent = sum(1 for e, x in opens if e <= entry < x)
            assert concurrent <= 2

    def test_position_size_respects_the_risk_budget(self):
        bars = frame(
            [(100, 101, 99, 100)] * 4
            + [(100, 101, 99, 100)]
            + [(100, 101, 94, 96)]
            + [(96, 97, 95, 96)]
        )
        result = run(
            ScriptedStrategy({3: (100.0, 95.0, 130.0)}),
            {"AAA": bars},
            risk=RiskSettings(max_risk_per_trade_pct=1.0),
        )
        trade = result.trades[0]
        # 1% of 100,000 = 1,000 of risk, over 5.00 per share = 200 shares.
        assert trade["quantity"] == pytest.approx(200.0)


class TestMultiSymbol:
    def test_symbols_are_simulated_on_a_shared_timeline(self):
        frames = {"AAA": flat(15), "BBB": flat(15, price=50.0)}
        strategy = ScriptedStrategy({5: (100.0, 90.0, 130.0)})
        result = run(strategy, frames)
        assert result.symbols == ("AAA", "BBB")
        assert len(result.equity_curve) == 15

    def test_a_symbol_with_a_shorter_history_is_handled(self):
        frames = {"AAA": flat(20), "BBB": flat(8)}
        result = run(ScriptedStrategy(), frames)
        assert len(result.equity_curve) == 20

    def test_a_failing_strategy_does_not_stop_the_run(self):
        class Exploding(ScriptedStrategy):
            def evaluate(self, symbol, data):
                if symbol == "BBB":
                    raise RuntimeError("boom")
                return super().evaluate(symbol, data)

        frames = {"AAA": flat(12), "BBB": flat(12)}
        result = run(Exploding({5: (100.0, 90.0, 130.0)}), frames)
        assert len(result.equity_curve) == 12
        assert any(t["symbol"] == "AAA" for t in result.trades)


class TestReporting:
    def test_summary_names_the_symbols_and_strategy(self):
        result = run(ScriptedStrategy(), {"AAA": flat(20)})
        text = result.summary()
        assert "AAA" in text and "scripted" in text

    def test_trade_frame_round_trips_the_trades(self):
        bars = frame(
            [(100, 101, 99, 100)] * 4
            + [(100, 101, 99, 100)]
            + [(100, 120, 99, 119)]
            + [(119, 120, 118, 119)]
        )
        result = run(ScriptedStrategy({3: (100.0, 95.0, 110.0)}), {"AAA": bars})
        table = result.trade_frame
        assert len(table) == 1
        assert {"symbol", "entry_price", "exit_price", "pnl"} <= set(table.columns)

    def test_as_dict_is_serialisable(self):
        import json

        result = run(ScriptedStrategy(), {"AAA": flat(20)})
        json.dumps(result.as_dict(), default=str)

    def test_empty_trade_frame_has_no_rows(self):
        result = run(ScriptedStrategy(), {"AAA": flat(20)})
        assert result.trade_frame.empty


class TestDeterminism:
    def test_the_same_inputs_produce_the_same_result(self):
        bars = frame(
            [(100, 101, 99, 100)] * 4
            + [(100, 101, 99, 100)]
            + [(100, 120, 99, 119)]
            + [(119, 120, 118, 119)]
        )
        first = run(ScriptedStrategy({3: (100.0, 95.0, 110.0)}), {"AAA": bars})
        second = run(ScriptedStrategy({3: (100.0, 95.0, 110.0)}), {"AAA": bars})
        assert first.trades == second.trades
        pd.testing.assert_series_equal(first.equity_curve, second.equity_curve)
