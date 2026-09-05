"""Strategy engine: the contract, the registry, and shared behaviour.

Per-strategy logic lives in ``test_strategy_signals.py``. This file covers the
guarantees every strategy must hold regardless of its edge.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from tests.conftest import make_bars
from trading_bot.indicators import InvalidDataError
from trading_bot.strategies import (
    BaseStrategy,
    BreakoutStrategy,
    Condition,
    ExitReason,
    MeanReversionStrategy,
    MomentumConfig,
    MomentumStrategy,
    Position,
    Signal,
    SignalDirection,
    StrategyConfig,
    StrategyError,
    available_strategies,
    build_strategy,
    register_strategy,
    score_conditions,
)

NOW = datetime(2024, 1, 2, 15, 0, tzinfo=timezone.utc)


def a_signal(**overrides) -> Signal:
    """A valid long signal, with fields overridable."""
    payload = {
        "symbol": "AAPL",
        "direction": SignalDirection.LONG,
        "strategy": "test",
        "confidence": 75.0,
        "entry_price": 100.0,
        "stop_loss": 98.0,
        "take_profit": 106.0,
        "timestamp": NOW,
    }
    payload.update(overrides)
    return Signal(**payload)


# ============================================================================
# Signal
# ============================================================================


def test_signal_computes_risk_reward():
    signal = a_signal()
    assert signal.risk_per_share == pytest.approx(2.0)
    assert signal.reward_per_share == pytest.approx(6.0)
    assert signal.risk_reward_ratio == pytest.approx(3.0)
    assert signal.stop_distance_pct == pytest.approx(2.0)


def test_short_signal_risk_reward():
    signal = a_signal(
        direction=SignalDirection.SHORT, entry_price=100.0, stop_loss=102.0, take_profit=94.0
    )
    assert signal.risk_per_share == pytest.approx(2.0)
    assert signal.risk_reward_ratio == pytest.approx(3.0)


def test_long_signal_rejects_a_stop_above_entry():
    """Sizing divides by this distance; a negative one must never reach Phase 4."""
    with pytest.raises(ValueError, match="stop_loss.*below entry"):
        a_signal(stop_loss=105.0)


def test_long_signal_rejects_a_target_below_entry():
    with pytest.raises(ValueError, match="take_profit.*above entry"):
        a_signal(take_profit=95.0)


def test_short_signal_rejects_a_stop_below_entry():
    with pytest.raises(ValueError, match="stop_loss.*above entry"):
        a_signal(direction=SignalDirection.SHORT, stop_loss=98.0, take_profit=94.0)


def test_signal_rejects_non_positive_prices():
    with pytest.raises(ValueError, match="must be positive"):
        a_signal(entry_price=0.0)


def test_signal_rejects_out_of_range_confidence():
    with pytest.raises(ValueError, match="confidence"):
        a_signal(confidence=140.0)


def test_signal_serializes_for_the_database():
    payload = a_signal(reasons=("Bullish crossover",)).as_dict()
    assert payload["symbol"] == "AAPL"
    assert payload["direction"] == "LONG"
    assert payload["risk_reward"] == pytest.approx(3.0)
    assert payload["reasons"] == ["Bullish crossover"]


def test_direction_sign_and_opposite():
    assert SignalDirection.LONG.sign == 1
    assert SignalDirection.SHORT.sign == -1
    assert SignalDirection.LONG.opposite is SignalDirection.SHORT


# ============================================================================
# Conditions
# ============================================================================


def test_confidence_is_the_share_of_weight_that_passed():
    conditions = [
        Condition("a", True, weight=3.0),
        Condition("b", False, weight=1.0),
    ]
    valid, confidence, _ = score_conditions(conditions)
    assert valid
    assert confidence == pytest.approx(75.0)


def test_a_failed_required_condition_vetoes_the_signal():
    conditions = [
        Condition("must", False, weight=1.0, required=True),
        Condition("nice", True, weight=9.0),
    ]
    valid, confidence, _ = score_conditions(conditions)
    assert not valid
    assert confidence == pytest.approx(90.0)   # still reported, but vetoed


def test_reasons_describe_only_the_passing_conditions():
    conditions = [
        Condition("a", True, detail="Trend is up"),
        Condition("b", False, detail="Volume is thin"),
    ]
    _, _, reasons = score_conditions(conditions)
    assert reasons == ("Trend is up",)


def test_empty_conditions_are_not_a_signal():
    assert score_conditions([]) == (False, 0.0, ())


def test_condition_rejects_a_negative_weight():
    with pytest.raises(ValueError, match="weight"):
        Condition("a", True, weight=-1.0)


# ============================================================================
# StrategyConfig
# ============================================================================


def test_config_rejects_disabling_both_directions():
    with pytest.raises(ValueError, match="never trade"):
        StrategyConfig(allow_long=False, allow_short=False)


def test_config_rejects_invalid_multipliers():
    with pytest.raises(ValueError, match="atr_stop_multiplier"):
        StrategyConfig(atr_stop_multiplier=0)
    with pytest.raises(ValueError, match="min_risk_reward"):
        StrategyConfig(min_risk_reward=-1)


def test_config_allows_checks_direction_permission():
    long_only = StrategyConfig(allow_long=True, allow_short=False)
    assert long_only.allows(SignalDirection.LONG)
    assert not long_only.allows(SignalDirection.SHORT)


def test_config_with_overrides_is_validated():
    assert StrategyConfig().with_overrides(min_confidence=80).min_confidence == 80
    with pytest.raises(ValueError):
        StrategyConfig().with_overrides(min_confidence=200)


def test_subclass_config_runs_both_validators():
    """@dataclass(slots=True) breaks zero-arg super(); the parent call must be explicit."""
    with pytest.raises(ValueError, match="min_confidence"):
        MomentumConfig(min_confidence=150)
    assert MomentumConfig(rsi_entry_ceiling=60).rsi_entry_ceiling == 60


# ============================================================================
# Exits shared by every strategy
# ============================================================================


class _NeverSignals(BaseStrategy):
    """Minimal concrete strategy, for exercising base-class behaviour."""

    name = "never"

    def evaluate(self, symbol, data):
        return None


@pytest.fixture
def strategy() -> _NeverSignals:
    return _NeverSignals()


def a_position(**overrides) -> Position:
    payload = {
        "symbol": "AAPL",
        "direction": SignalDirection.LONG,
        "entry_price": 100.0,
        "stop_loss": 98.0,
        "take_profit": 106.0,
    }
    payload.update(overrides)
    return Position(**payload)


def bars_with_range(high: float, low: float, close: float) -> pd.DataFrame:
    """A frame whose final bar spans the given range."""
    base = make_bars(60, seed=5)
    base.iloc[-1, base.columns.get_loc("high")] = high
    base.iloc[-1, base.columns.get_loc("low")] = low
    base.iloc[-1, base.columns.get_loc("close")] = close
    base.iloc[-1, base.columns.get_loc("open")] = close
    return base


def test_stop_hit_closes_the_position(strategy):
    exit_signal = strategy.should_exit(a_position(), bars_with_range(101, 97, 99))
    assert exit_signal.reason is ExitReason.STOP_LOSS
    assert exit_signal.price == pytest.approx(98.0)


def test_target_hit_closes_the_position(strategy):
    exit_signal = strategy.should_exit(a_position(), bars_with_range(107, 101, 106.5))
    assert exit_signal.reason is ExitReason.TAKE_PROFIT
    assert exit_signal.price == pytest.approx(106.0)


def test_a_bar_touching_both_levels_resolves_as_a_stop(strategy):
    """Intrabar order is unknowable; assuming the good outcome inflates backtests."""
    exit_signal = strategy.should_exit(a_position(), bars_with_range(107, 97, 103))
    assert exit_signal.reason is ExitReason.STOP_LOSS
    assert exit_signal.metadata["ambiguous_bar"] is True


def test_short_position_stops_out_on_a_rally(strategy):
    position = a_position(
        direction=SignalDirection.SHORT, stop_loss=102.0, take_profit=94.0
    )
    exit_signal = strategy.should_exit(position, bars_with_range(103, 99, 102.5))
    assert exit_signal.reason is ExitReason.STOP_LOSS


def test_no_exit_while_price_stays_between_the_levels(strategy):
    assert strategy.should_exit(a_position(), bars_with_range(104, 99, 101)) is None


def test_time_stop_closes_a_stale_position():
    strategy = _NeverSignals(StrategyConfig(max_holding_bars=10))
    exit_signal = strategy.should_exit(
        a_position(bars_held=10), bars_with_range(104, 99, 101)
    )
    assert exit_signal.reason is ExitReason.TIME_STOP


def test_protective_exits_take_priority_over_the_time_stop():
    strategy = _NeverSignals(StrategyConfig(max_holding_bars=1))
    exit_signal = strategy.should_exit(
        a_position(bars_held=99), bars_with_range(101, 97, 99)
    )
    assert exit_signal.reason is ExitReason.STOP_LOSS


def test_position_reports_progress_in_r_multiples():
    position = a_position()
    assert position.r_multiple(104.0) == pytest.approx(2.0)
    assert position.r_multiple(98.0) == pytest.approx(-1.0)
    assert position.unrealized_pnl_per_share(103.0) == pytest.approx(3.0)


def test_short_position_r_multiple_is_signed_correctly():
    position = a_position(direction=SignalDirection.SHORT, stop_loss=102.0, take_profit=94.0)
    assert position.r_multiple(96.0) == pytest.approx(2.0)


# ============================================================================
# Exit placement
# ============================================================================


def test_atr_stop_scales_with_volatility(strategy):
    tight, _, _ = strategy.build_exits(SignalDirection.LONG, 100.0, atr=0.5)
    wide, _, _ = strategy.build_exits(SignalDirection.LONG, 100.0, atr=3.0)
    assert 100 - wide > 100 - tight


def test_target_always_meets_the_minimum_risk_reward(strategy):
    for atr in (0.2, 1.0, 5.0):
        stop, target, _ = strategy.build_exits(SignalDirection.LONG, 100.0, atr=atr)
        risk, reward = 100 - stop, target - 100
        assert reward / risk >= strategy.config.min_risk_reward - 1e-9


def test_short_exits_are_placed_on_the_correct_sides(strategy):
    stop, target, _ = strategy.build_exits(SignalDirection.SHORT, 100.0, atr=1.0)
    assert stop > 100 > target


def test_percentage_stop_is_used_when_atr_is_unavailable(strategy):
    stop, _, notes = strategy.build_exits(SignalDirection.LONG, 100.0, atr=None)
    assert stop == pytest.approx(98.0)   # 2% default
    assert any("percentage stop" in note for note in notes)


def test_a_long_stop_never_crosses_zero(strategy):
    stop, target, _ = strategy.build_exits(SignalDirection.LONG, 1.0, atr=50.0)
    assert stop > 0
    assert target > 1.0


def test_build_exits_rejects_a_non_positive_entry(strategy):
    with pytest.raises(ValueError, match="entry_price"):
        strategy.build_exits(SignalDirection.LONG, 0.0, atr=1.0)


# ============================================================================
# generate_signal contract
# ============================================================================


def test_short_history_yields_no_signal(strategy):
    assert strategy.generate_signal("AAPL", make_bars(30, seed=6)) is None


def test_malformed_input_is_rejected(strategy):
    with pytest.raises(InvalidDataError):
        strategy.generate_signal("AAPL", pd.DataFrame())


def test_prepare_is_idempotent(strategy):
    once = strategy.prepare(make_bars(250, seed=7))
    twice = strategy.prepare(once)
    pd.testing.assert_series_equal(once["RSI_14"], twice["RSI_14"])


def test_confidence_floor_suppresses_weak_signals():
    class _AlwaysWeak(BaseStrategy):
        name = "weak"

        def evaluate(self, symbol, data):
            return self._build_signal(
                symbol=symbol,
                direction=SignalDirection.LONG,
                data=data,
                conditions=[
                    Condition("a", True, weight=1.0),
                    Condition("b", False, weight=9.0),
                ],
            )

    data = make_bars(250, seed=8)
    assert _AlwaysWeak(StrategyConfig(min_confidence=50)).generate_signal("AAPL", data) is None
    assert _AlwaysWeak(StrategyConfig(min_confidence=5)).generate_signal("AAPL", data) is not None


def test_disallowed_direction_produces_no_signal():
    class _AlwaysShort(BaseStrategy):
        name = "shorty"

        def evaluate(self, symbol, data):
            return self._build_signal(
                symbol=symbol,
                direction=SignalDirection.SHORT,
                data=data,
                conditions=[Condition("a", True, weight=1.0)],
            )

    strategy = _AlwaysShort(StrategyConfig(allow_short=False))
    assert strategy.generate_signal("AAPL", make_bars(250, seed=9)) is None


# ============================================================================
# Registry
# ============================================================================


def test_registry_lists_the_three_strategies():
    assert available_strategies() == ["breakout", "mean_reversion", "momentum"]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("momentum", MomentumStrategy),
        ("mean_reversion", MeanReversionStrategy),
        ("breakout", BreakoutStrategy),
        ("MOMENTUM", MomentumStrategy),
        ("  Breakout  ", BreakoutStrategy),
    ],
)
def test_build_strategy_by_name(name, expected):
    assert isinstance(build_strategy(name), expected)


def test_build_strategy_applies_typed_overrides():
    strategy = build_strategy("momentum", min_confidence=70, rsi_entry_ceiling=60)
    assert strategy.config.min_confidence == 70
    assert strategy.config.rsi_entry_ceiling == 60


def test_build_strategy_rejects_an_unknown_name():
    with pytest.raises(StrategyError, match="Unknown strategy"):
        build_strategy("does_not_exist")


def test_build_strategy_rejects_an_unknown_parameter():
    with pytest.raises(StrategyError, match="no parameter"):
        build_strategy("momentum", nonsense=1)


def test_build_strategy_rejects_config_and_overrides_together():
    with pytest.raises(StrategyError, match="not both"):
        build_strategy("momentum", config=MomentumConfig(), min_confidence=70)


def test_registering_a_duplicate_name_is_refused():
    class _Clash(BaseStrategy):
        name = "momentum"

        def evaluate(self, symbol, data):
            return None

    with pytest.raises(StrategyError, match="already registered"):
        register_strategy(_Clash)


def test_registering_an_unnamed_strategy_is_refused():
    class _Unnamed(BaseStrategy):
        def evaluate(self, symbol, data):
            return None

    with pytest.raises(StrategyError, match="unique class-level"):
        register_strategy(_Unnamed)


def test_a_registered_strategy_becomes_buildable():
    class _Custom(BaseStrategy):
        name = "custom_test_strategy"

        def evaluate(self, symbol, data):
            return None

    try:
        register_strategy(_Custom)
        assert isinstance(build_strategy("custom_test_strategy"), _Custom)
    finally:
        from trading_bot.strategies import STRATEGY_REGISTRY

        STRATEGY_REGISTRY.pop("custom_test_strategy", None)


# ============================================================================
# Stop distance floor
# ============================================================================


def test_a_structural_stop_is_floored_at_a_minimum_atr(strategy):
    """A stop inside the noise is not a stop.

    A structural level can sit a few cents from the entry. Left alone, that
    produces a stop an ordinary bar would take out, while inflating both
    reward:risk and — because sizing divides by the stop distance — position
    size. Found by reading a scan that showed 11.6:1 on a stop 0.26 ATR wide,
    where 98% of recent bars had a range wider than the whole stop.
    """
    from trading_bot.indicators import Level, SupportResistance

    atr = 2.0
    # Support one cent below the entry: a naive structural stop would be ~1 cent.
    levels = SupportResistance(
        price=100.0,
        support=(Level(price=99.99, touches=3, last_touch_index=10, kind="support"),),
        resistance=(),
        swing_points=(),
    )
    stop, _, notes = strategy.build_exits(
        SignalDirection.LONG, 100.0, atr=atr, levels=levels
    )
    distance = 100.0 - stop
    assert distance >= atr * strategy.config.min_stop_atr - 1e-9
    assert any("inside the noise" in note for note in notes)


def test_the_floor_does_not_widen_an_already_sensible_stop(strategy):
    stop, _, notes = strategy.build_exits(SignalDirection.LONG, 100.0, atr=1.0)
    assert 100.0 - stop == pytest.approx(strategy.config.atr_stop_multiplier)
    assert not any("inside the noise" in note for note in notes)


def test_the_floor_applies_to_shorts_too(strategy):
    from trading_bot.indicators import Level, SupportResistance

    levels = SupportResistance(
        price=100.0,
        support=(),
        resistance=(Level(price=100.01, touches=2, last_touch_index=5, kind="resistance"),),
        swing_points=(),
    )
    stop, _, _ = strategy.build_exits(
        SignalDirection.SHORT, 100.0, atr=2.0, levels=levels
    )
    assert stop - 100.0 >= 2.0 * strategy.config.min_stop_atr - 1e-9


def test_the_floor_caps_runaway_reward_to_risk(strategy):
    """The inflated ratio was the visible symptom of the real problem."""
    from trading_bot.indicators import Level, SupportResistance

    levels = SupportResistance(
        price=100.0,
        support=(Level(price=99.99, touches=3, last_touch_index=10, kind="support"),),
        resistance=(),
        swing_points=(),
    )
    stop, target, _ = strategy.build_exits(
        SignalDirection.LONG, 100.0, atr=2.0, levels=levels
    )
    assert (target - 100.0) / (100.0 - stop) < 10


def test_min_stop_atr_is_configurable_and_validated():
    assert StrategyConfig(min_stop_atr=0.0).min_stop_atr == 0.0
    with pytest.raises(ValueError, match="min_stop_atr"):
        StrategyConfig(min_stop_atr=-1.0)
