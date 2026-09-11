"""Regression tests for swing-oriented backtest request plumbing."""

from trading_bot.backtesting.runner import BacktestRequest, build_strategies
from trading_bot.config.settings import RiskSettings


def test_backtest_defaults_to_daily_bars() -> None:
    request = BacktestRequest(symbols=("AAPL",), strategies=("swing_quality",))
    assert request.timeframe == "1Day"


def test_backtest_inherits_risk_confidence_floor() -> None:
    risk = RiskSettings(min_confidence=77.0)
    request = BacktestRequest(
        symbols=("AAPL",),
        strategies=("swing_quality",),
        risk=risk,
    )
    strategy = build_strategies(request)[0]
    assert strategy.config.min_confidence == 77.0


def test_explicit_backtest_confidence_override_wins() -> None:
    risk = RiskSettings(min_confidence=77.0)
    request = BacktestRequest(
        symbols=("AAPL",),
        strategies=("swing_quality",),
        risk=risk,
        min_confidence=82.0,
    )
    strategy = build_strategies(request)[0]
    assert strategy.config.min_confidence == 82.0
