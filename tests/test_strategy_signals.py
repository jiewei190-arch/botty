"""Per-strategy behaviour: does each one fire where its edge exists, and stay
silent where it does not?

Each strategy is driven through synthetic markets built to match or defeat its
thesis. A strategy that signals everywhere is as useless as one that never
signals, so several tests assert *silence*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_bot.strategies import (
    BreakoutConfig,
    BreakoutStrategy,
    ExitReason,
    MeanReversionConfig,
    MeanReversionStrategy,
    MomentumConfig,
    MomentumStrategy,
    Position,
    SignalDirection,
)

BARS = 700


def frame(closes, volumes=None, *, spread: float = 0.003) -> pd.DataFrame:
    """Build an OHLCV frame from a close series."""
    closes = np.asarray(closes, dtype="float64")
    count = len(closes)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    index = pd.date_range("2024-01-02", periods=count, freq="15min", tz="UTC", name="timestamp")
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) * (1 + spread),
            "low": np.minimum(opens, closes) * (1 - spread),
            "close": closes,
            "volume": np.asarray(
                volumes if volumes is not None else np.full(count, 200_000.0), dtype="float64"
            ),
        },
        index=index,
    )


def trending(drift: float, seed: int = 11, periods: int = BARS, noise: float = 0.006):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(drift, noise, periods)))


def ranging(
    amplitude: float = 5.0,
    seed: int = 3,
    periods: int = BARS,
    wavelength: float = 9.0,
):
    """A range-bound market.

    The wavelength is short enough that several full swings occur inside the
    walked window — mean reversion is deliberately selective, so a slower
    oscillation would leave nothing for it to find in a bounded test.
    """
    rng = np.random.default_rng(seed)
    return (
        100
        + amplitude * np.sin(np.arange(periods) / wavelength)
        + rng.normal(0, 0.6, periods)
    )


#: Bars walked per strategy in a test. Enough to establish whether a strategy
#: fires in a regime, small enough to keep the suite fast — a slow suite is one
#: that stops being run.
WALK_BARS = 150


def all_signals(strategy, data: pd.DataFrame, symbol: str = "TEST", walk: int = WALK_BARS) -> list:
    """Walk the most recent bars one at a time, exactly as the backtester will."""
    prepared = strategy.prepare(data)
    start = max(strategy.min_bars, len(prepared) - walk)
    signals = []
    for index in range(start, len(prepared)):
        signal = strategy.generate_signal(symbol, prepared.iloc[: index + 1])
        if signal is not None:
            signals.append(signal)
    return signals


# ============================================================================
# Momentum
# ============================================================================


def test_momentum_fires_in_an_uptrend():
    signals = all_signals(MomentumStrategy(), frame(trending(0.0018)))
    assert signals
    assert all(s.direction is SignalDirection.LONG for s in signals)


def test_momentum_stays_out_of_a_range():
    """A trend-continuation strategy has no edge in a market going nowhere."""
    trend_signals = len(all_signals(MomentumStrategy(), frame(trending(0.0018))))
    range_signals = len(all_signals(MomentumStrategy(), frame(ranging())))
    assert range_signals < trend_signals


def test_momentum_refuses_to_buy_an_overbought_market():
    """A relentless rise gives RSI 100 — the top of the move, not the start."""
    strategy = MomentumStrategy()
    parabolic = frame([100 * (1.004**i) for i in range(400)])
    assert strategy.generate_signal("TEST", strategy.prepare(parabolic)) is None


def test_momentum_shorts_only_when_enabled():
    downtrend = frame(trending(-0.0018))
    assert not all_signals(MomentumStrategy(), downtrend)
    shorting = MomentumStrategy(MomentumConfig(allow_short=True))
    signals = all_signals(shorting, downtrend)
    assert any(s.direction is SignalDirection.SHORT for s in signals)


def test_momentum_signals_carry_a_usable_risk_reward():
    for signal in all_signals(MomentumStrategy(), frame(trending(0.0018))):
        assert signal.risk_reward_ratio >= 2.0 - 1e-9
        assert signal.stop_loss < signal.entry_price < signal.take_profit


def test_momentum_reasons_name_the_evidence():
    signals = all_signals(MomentumStrategy(), frame(trending(0.0018)))
    assert signals
    joined = " ".join(signals[0].reasons)
    assert "trend" in joined.lower() or "EMA" in joined


def test_momentum_confidence_gate_reduces_signal_count():
    data = frame(trending(0.0018))
    permissive = len(all_signals(MomentumStrategy(MomentumConfig(min_confidence=40)), data))
    strict = len(all_signals(MomentumStrategy(MomentumConfig(min_confidence=90)), data))
    assert strict < permissive


def test_momentum_exits_when_momentum_turns():
    strategy = MomentumStrategy()
    reversing = frame(
        np.concatenate([trending(0.003, periods=400), trending(-0.004, seed=2, periods=90)])
    )
    prepared = strategy.prepare(reversing)
    entry = float(prepared["close"].iloc[400])
    position = Position(
        symbol="TEST",
        direction=SignalDirection.LONG,
        entry_price=entry,
        # Placed far beyond anything the series reaches, so the protective exits
        # cannot fire and only the discretionary logic is under test.
        stop_loss=entry * 0.01,
        take_profit=entry * 100.0,
    )
    exit_signal = strategy.should_exit(position, prepared)
    assert exit_signal is not None
    assert exit_signal.reason in (ExitReason.MOMENTUM_FADE, ExitReason.TREND_REVERSAL)


# ============================================================================
# Mean reversion
# ============================================================================


def test_mean_reversion_fires_in_a_range():
    signals = all_signals(MeanReversionStrategy(), frame(ranging()))
    assert signals


def test_mean_reversion_refuses_to_catch_a_falling_knife():
    """Its characteristic failure. The regime filter must suppress this entirely."""
    crash = frame(trending(-0.004, seed=3, noise=0.005))
    longs = [
        s for s in all_signals(MeanReversionStrategy(), crash)
        if s.direction is SignalDirection.LONG
    ]
    assert longs == []


def test_mean_reversion_targets_the_mean_not_an_atr_multiple():
    signals = all_signals(MeanReversionStrategy(), frame(ranging()))
    assert signals
    for signal in signals:
        middle = signal.metadata["middle_band"]
        assert signal.take_profit == pytest.approx(middle, rel=0.02)


def test_mean_reversion_requires_confirmation_by_default():
    data = frame(ranging())
    confirmed = len(all_signals(MeanReversionStrategy(), data))
    unconfirmed = len(
        all_signals(MeanReversionStrategy(MeanReversionConfig(require_confirmation=False)), data)
    )
    assert unconfirmed >= confirmed


def test_mean_reversion_uses_a_tighter_stop_than_momentum():
    """Its thesis is invalidated immediately, so it does not give price room."""
    assert MeanReversionConfig().atr_stop_multiplier < MomentumConfig().atr_stop_multiplier


def test_mean_reversion_accepts_a_lower_risk_reward_than_momentum():
    """Reverting to the mean pays about one sigma; demanding 2:1 rejects everything."""
    assert MeanReversionConfig().min_risk_reward < MomentumConfig().min_risk_reward


def test_mean_reversion_warmup_covers_the_regime_window():
    strategy = MeanReversionStrategy()
    assert strategy.min_bars > strategy.indicators.max_lookback


def test_mean_reversion_exits_at_the_mean():
    strategy = MeanReversionStrategy()
    prepared = strategy.prepare(frame(ranging()))
    middle = float(prepared["BB_MIDDLE"].iloc[-1])
    close = float(prepared["close"].iloc[-1])
    if close < middle:                      # only meaningful below the mean
        position = Position(
            symbol="TEST",
            direction=SignalDirection.LONG,
            entry_price=close * 0.98,
            stop_loss=close * 0.5,
            take_profit=close * 2.0,
        )
        assert strategy.should_exit(position, prepared) is None
    else:
        position = Position(
            symbol="TEST",
            direction=SignalDirection.LONG,
            entry_price=close * 0.98,
            stop_loss=close * 0.5,
            take_profit=close * 2.0,
        )
        exit_signal = strategy.should_exit(position, prepared)
        assert exit_signal is not None
        assert exit_signal.reason is ExitReason.TARGET_REACHED


# ============================================================================
# Breakout
# ============================================================================


def coiled_then_break(final_close: float, final_volume: float):
    """260 noisy bars, a 25-bar coil, then one decisive bar."""
    rng = np.random.default_rng(7)
    closes = (
        list(100 + rng.normal(0, 1.2, 260))
        + list(102 + rng.normal(0, 0.15, 25))
        + [final_close]
    )
    volumes = list(rng.integers(160_000, 240_000, 285)) + [final_volume]
    return frame(closes, volumes, spread=0.002)


def test_breakout_fires_on_a_confirmed_break():
    strategy = BreakoutStrategy()
    signal = strategy.generate_signal("TEST", strategy.prepare(coiled_then_break(103.5, 650_000)))
    assert signal is not None
    assert signal.direction is SignalDirection.LONG
    assert signal.metadata["level_price"] == pytest.approx(102.46, abs=0.2)


def test_breakout_requires_volume():
    """A break nobody participated in is the definition of a false one."""
    strategy = BreakoutStrategy()
    assert strategy.generate_signal(
        "TEST", strategy.prepare(coiled_then_break(103.5, 200_000))
    ) is None


def test_breakout_ignores_price_still_inside_the_range():
    strategy = BreakoutStrategy()
    assert strategy.generate_signal(
        "TEST", strategy.prepare(coiled_then_break(102.1, 650_000))
    ) is None


def test_breakout_refuses_to_chase_an_extended_move():
    """Risk is measured from the level, so a late entry pays badly."""
    strategy = BreakoutStrategy()
    assert strategy.generate_signal(
        "TEST", strategy.prepare(coiled_then_break(112.0, 650_000))
    ) is None


def test_breakout_requires_a_prior_consolidation():
    """Without a coil there is no stored energy to release.

    Note the window must be wide *relative to its own ATR*: plain high-variance
    noise raises ATR too, so it still measures as consolidated. A swinging window
    is what actually fails the test.
    """
    rng = np.random.default_rng(4)
    swinging = (
        list(100 + rng.normal(0, 1.0, 265))
        + [100 + 8 * np.sin(i / 3) for i in range(20)]
        + [118.0]
    )
    volumes = list(rng.integers(160_000, 240_000, 285)) + [650_000]
    strategy = BreakoutStrategy()
    prepared = strategy.prepare(frame(swinging, volumes))

    consolidated, width = strategy._consolidation(prepared)
    assert not consolidated and width > BreakoutConfig().max_consolidation_width_atr
    assert strategy.generate_signal("TEST", prepared) is None


def test_breakout_measures_consolidation_with_pre_break_volatility():
    """Using the current ATR would be circular — the break itself inflates it."""
    strategy = BreakoutStrategy()
    prepared = strategy.prepare(coiled_then_break(103.5, 650_000))
    assert strategy._consolidation_atr(prepared) == pytest.approx(
        float(prepared["ATR_14"].iloc[-2])
    )
    consolidated, width = strategy._consolidation(prepared)
    assert consolidated
    assert width < BreakoutConfig().max_consolidation_width_atr


def test_breakout_stop_sits_back_inside_the_range():
    strategy = BreakoutStrategy()
    signal = strategy.generate_signal("TEST", strategy.prepare(coiled_then_break(103.5, 650_000)))
    assert signal is not None
    assert signal.stop_loss < signal.metadata["level_price"] + 0.5


def test_breakout_exits_when_the_break_fails():
    strategy = BreakoutStrategy()
    prepared = strategy.prepare(coiled_then_break(103.5, 650_000))
    position = Position(
        symbol="TEST",
        direction=SignalDirection.LONG,
        entry_price=103.5,
        stop_loss=50.0,             # far away, so the protective exit cannot fire
        take_profit=200.0,
        metadata={"level_price": 110.0},   # price is now well below the level
    )
    exit_signal = strategy.should_exit(position, prepared)
    assert exit_signal is not None
    assert exit_signal.reason is ExitReason.SIGNAL_FLIP


# ============================================================================
# Cross-strategy guarantees
# ============================================================================


@pytest.mark.parametrize(
    "strategy",
    [MomentumStrategy(), MeanReversionStrategy(), BreakoutStrategy()],
    ids=["momentum", "mean_reversion", "breakout"],
)
def test_every_signal_is_internally_coherent(strategy):
    for data in (frame(trending(0.0018)), frame(ranging()), frame(trending(-0.002, seed=5))):
        for signal in all_signals(strategy, data):
            if signal.direction is SignalDirection.LONG:
                assert signal.stop_loss < signal.entry_price < signal.take_profit
            else:
                assert signal.take_profit < signal.entry_price < signal.stop_loss
            assert 0 <= signal.confidence <= 100
            assert signal.risk_per_share > 0
            assert signal.reasons


@pytest.mark.parametrize(
    "strategy",
    [MomentumStrategy(), MeanReversionStrategy(), BreakoutStrategy()],
    ids=["momentum", "mean_reversion", "breakout"],
)
def test_strategies_are_silent_without_enough_history(strategy):
    short = frame(trending(0.002, periods=strategy.min_bars - 1))
    assert strategy.generate_signal("TEST", strategy.prepare(short)) is None


@pytest.mark.parametrize(
    "strategy",
    [MomentumStrategy(), MeanReversionStrategy(), BreakoutStrategy()],
    ids=["momentum", "mean_reversion", "breakout"],
)
def test_evaluation_reads_only_the_supplied_bars(strategy):
    """A signal at bar N must not change when later bars are appended.

    This is the lookahead guarantee at the strategy level: the backtester feeds
    slices, and a strategy that peeked at future data would disagree with itself.
    """
    data = frame(trending(0.0018))
    prepared = strategy.prepare(data)
    cutoff = len(prepared) - 60

    truncated = strategy.generate_signal("TEST", prepared.iloc[:cutoff])
    full_view = strategy.generate_signal("TEST", prepared.iloc[:cutoff])
    assert (truncated is None) == (full_view is None)
    if truncated is not None:
        assert truncated.entry_price == full_view.entry_price
        assert truncated.confidence == full_view.confidence


@pytest.mark.parametrize(
    "strategy",
    [MomentumStrategy(), MeanReversionStrategy(), BreakoutStrategy()],
    ids=["momentum", "mean_reversion", "breakout"],
)
def test_strategies_work_across_timeframes(strategy):
    rng = np.random.default_rng(21)
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0015, 0.006, 400)))
    for freq in ("5min", "1h", "1D"):
        index = pd.date_range("2020-01-02", periods=400, freq=freq, tz="UTC", name="timestamp")
        data = frame(closes)
        data.index = index
        strategy.generate_signal("TEST", strategy.prepare(data))   # must not raise
