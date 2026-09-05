"""Market scanner (Phase 5).

Ranks opportunities across a watchlist on a single 0-100 trade confidence score,
so candidates found by different strategies can be compared to each other.

    from trading_bot.scanner import MarketScanner

    scanner = MarketScanner(strategies, risk_manager=risk)
    result = scanner.scan(frames, portfolio=portfolio)
    for opportunity in result.opportunities:
        print(opportunity.rank, opportunity.symbol, opportunity.confidence)

The score ranks opportunities against each other. It is **not** a probability of
profit, and nothing here should be read as one.
"""

from trading_bot.scanner.scanner import (
    MarketScanner,
    Opportunity,
    ScannerConfig,
    ScanResult,
)
from trading_bot.scanner.scoring import (
    DEFAULT_WEIGHTS,
    FactorScore,
    score_conviction,
    score_momentum,
    score_opportunity,
    score_risk_reward,
    score_rsi_headroom,
    score_structure,
    score_trend,
    score_volume,
)

__all__ = [
    "MarketScanner",
    "ScannerConfig",
    "ScanResult",
    "Opportunity",
    "FactorScore",
    "DEFAULT_WEIGHTS",
    "score_opportunity",
    "score_trend",
    "score_momentum",
    "score_volume",
    "score_rsi_headroom",
    "score_risk_reward",
    "score_structure",
    "score_conviction",
]
