"""Risk management (Phase 4).

The gate between a proposed trade and a real one. :class:`RiskManager` is the
only thing in the system that turns a signal into a quantity, and the execution
layer takes a :class:`RiskDecision` rather than a signal — so a trade cannot
reach a broker without having passed these limits.

    from trading_bot.risk import RiskManager, build_portfolio_state

    portfolio = build_portfolio_state(account=account, database=db)
    decision = RiskManager(settings.risk).evaluate(signal, portfolio)
    if decision.approved:
        ...  # Phase 7 places decision.shares
    else:
        print(decision.rejection_reason)
"""

from trading_bot.risk.portfolio import (
    OpenPosition,
    PortfolioState,
    build_portfolio_state,
    session_start,
)
from trading_bot.risk.position_sizing import (
    PositionSize,
    SizingConstraint,
    calculate_position_size,
    to_decimal,
)
from trading_bot.risk.risk_manager import RiskCheck, RiskDecision, RiskManager

__all__ = [
    "RiskManager",
    "RiskDecision",
    "RiskCheck",
    "PortfolioState",
    "OpenPosition",
    "build_portfolio_state",
    "session_start",
    "PositionSize",
    "SizingConstraint",
    "calculate_position_size",
    "to_decimal",
]
