"""Backtesting engine (Phase 6).

Candle-by-candle simulation through the same strategy and risk code the live bot
uses, with explicit assumptions about fills, costs and slippage.

    from trading_bot.backtesting import Backtester, BacktestConfig, CostModel

    tester = Backtester([build_strategy("momentum")], BacktestConfig(
        starting_equity=10_000, costs=CostModel(slippage_pct=0.05),
    ))
    result = tester.run({"AAPL": bars})
    print(result.summary())

A backtest is a lower bound on how wrong you can be, not a forecast. The model
here errs pessimistic on purpose — see :mod:`trading_bot.backtesting.execution`
for the three assumptions that decide whether it resembles reality.
"""

from trading_bot.backtesting.engine import (
    BacktestConfig,
    Backtester,
    BacktestResult,
    SimulatedPosition,
)
from trading_bot.backtesting.execution import (
    FRICTIONLESS,
    CostModel,
    Fill,
    FillModel,
    FillReason,
)
from trading_bot.backtesting.metrics import (
    MIN_MEANINGFUL_TRADES,
    PerformanceMetrics,
    calculate_metrics,
    drawdown_curve,
)

__all__ = [
    "Backtester",
    "BacktestConfig",
    "BacktestResult",
    "SimulatedPosition",
    "CostModel",
    "FillModel",
    "Fill",
    "FillReason",
    "FRICTIONLESS",
    "PerformanceMetrics",
    "calculate_metrics",
    "drawdown_curve",
    "MIN_MEANINGFUL_TRADES",
]
