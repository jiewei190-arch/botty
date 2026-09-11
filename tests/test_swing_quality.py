"""Tests for the selective swing-quality strategy."""

from __future__ import annotations

import pandas as pd

from trading_bot.strategies import (
    SwingQualityConfig,
    SwingQualityStrategy,
    available_strategies,
    build_strategy,
)


def test_swing_quality_is_registered() -> None:
    assert "swing_quality" in available_strategies()
    strategy = build_strategy("swing_quality")
    assert isinstance(strategy, SwingQualityStrategy)


def test_swing_quality_defaults_are_selective_and_month_sized() -> None:
    config = SwingQualityConfig()
    assert config.min_confidence == 70.0
    # 21 trading days, not 20: the target sits at sqrt(21) ATRs so it stays
    # reachable before the cap fires. See TestSwingQualityGeometry.
    assert config.max_holding_bars == 21
    assert config.fast_ema == 20
    assert config.mid_ema == 50
    assert config.long_ema == 200
    assert config.min_risk_reward >= 2.0


def test_fresh_breakout_excludes_current_bar_from_lookback() -> None:
    strategy = SwingQualityStrategy(SwingQualityConfig(breakout_lookback=5))
    closes = pd.Series([100.0, 101.0, 102.0, 101.5, 102.5, 103.0])
    assert strategy._fresh_breakout(closes) is True


def test_no_breakout_when_latest_close_is_below_prior_high() -> None:
    strategy = SwingQualityStrategy(SwingQualityConfig(breakout_lookback=5))
    closes = pd.Series([100.0, 101.0, 104.0, 102.0, 103.0, 103.5])
    assert strategy._fresh_breakout(closes) is False


def test_pullback_reclaim_requires_recent_test_and_recovery() -> None:
    strategy = SwingQualityStrategy(SwingQualityConfig(pullback_window=4))
    closes = pd.Series([101.0, 100.0, 99.5, 100.2, 101.2])
    ema = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0])
    assert strategy._pullback_reclaim(closes, ema) is True


def test_pullback_reclaim_rejects_extended_price_with_no_test() -> None:
    strategy = SwingQualityStrategy(SwingQualityConfig(pullback_window=4))
    closes = pd.Series([105.0, 105.5, 106.0, 106.5, 107.0])
    ema = pd.Series([100.0, 100.2, 100.4, 100.6, 100.8])
    assert strategy._pullback_reclaim(closes, ema) is False
