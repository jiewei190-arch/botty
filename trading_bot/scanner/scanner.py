"""The market scanner.

Runs every configured strategy across a watchlist, scores what they find on one
common yardstick, checks each candidate against the risk limits, and returns a
ranked list.

The ordering matters and is deliberate:

1. **Filter** — drop symbols too illiquid to trade before spending any analysis
   on them. A perfect setup in something that trades 4,000 shares a day is not
   an opportunity.
2. **Find** — every strategy evaluates every surviving symbol.
3. **Score** — each candidate is re-scored on the common yardstick, because
   strategy confidences are not comparable to each other.
4. **Size** — the risk manager approves or rejects, and attaches a quantity.
5. **Rank** — by the common score, best first.

Risk runs *after* scoring rather than before, so a rejected opportunity still
appears with its score and the reason it was refused. A scanner that silently
dropped everything the limits blocked would leave you unable to tell a quiet
market from a mis-set limit.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from trading_bot.indicators import (
    IndicatorConfig,
    analyze_trend,
    analyze_volume,
    calculate_all_indicators,
    find_support_resistance,
    volume_sma_column,
)
from trading_bot.risk import PortfolioState, RiskDecision, RiskManager
from trading_bot.scanner.scoring import FactorScore, score_opportunity
from trading_bot.strategies import BaseStrategy, Signal, explain_blockers

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ScannerConfig:
    """Scanner filters and limits."""

    #: Skip symbols whose average daily turnover is below this. A perfect setup
    #: in an illiquid name cannot be entered or exited at the modelled price.
    min_avg_dollar_volume: float = 1_000_000.0
    #: Skip symbols priced below this — spreads and slippage dominate.
    min_price: float = 1.0
    #: Only opportunities at or above this score are returned.
    min_confidence: float = 0.0
    #: Cap on returned opportunities. ``None`` for all of them.
    max_results: int | None = None

    def __post_init__(self) -> None:
        if self.min_avg_dollar_volume < 0:
            raise ValueError("min_avg_dollar_volume must be >= 0")
        if self.min_price < 0:
            raise ValueError("min_price must be >= 0")
        if not 0 <= self.min_confidence <= 100:
            raise ValueError("min_confidence must be within 0-100")
        if self.max_results is not None and self.max_results < 1:
            raise ValueError("max_results must be >= 1 or None")


@dataclass(frozen=True, slots=True)
class Opportunity:
    """One ranked candidate."""

    signal: Signal
    confidence: float
    factors: tuple[FactorScore, ...]
    decision: RiskDecision | None = None
    rank: int = 0

    @property
    def symbol(self) -> str:
        return self.signal.symbol

    @property
    def tradable(self) -> bool:
        """Whether risk approved it. ``None`` decision means risk was not run."""
        return self.decision is not None and self.decision.approved

    @property
    def quantity(self) -> int:
        return self.decision.shares if self.decision else 0

    @property
    def rejection_reason(self) -> str | None:
        if self.decision is None or self.decision.approved:
            return None
        return self.decision.rejection_reason

    @property
    def reasons(self) -> tuple[str, ...]:
        """Why this setup exists, from the strategy that found it."""
        return self.signal.reasons

    def top_factors(self, count: int = 3) -> tuple[FactorScore, ...]:
        """The strongest contributors, for a compact summary."""
        return tuple(sorted(self.factors, key=lambda f: -f.contribution)[:count])

    def weakest_factor(self) -> FactorScore | None:
        """The lowest-scoring factor — the caveat worth reading."""
        return min(self.factors, key=lambda f: f.score) if self.factors else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "symbol": self.symbol,
            "direction": self.signal.direction.value,
            "strategy": self.signal.strategy,
            "confidence": round(self.confidence, 1),
            "entry": self.signal.entry_price,
            "stop_loss": self.signal.stop_loss,
            "take_profit": self.signal.take_profit,
            "risk_reward": round(self.signal.risk_reward_ratio, 2),
            "quantity": self.quantity,
            "tradable": self.tradable,
            "rejection_reason": self.rejection_reason,
            "factors": [factor.as_dict() for factor in self.factors],
            "reasons": list(self.reasons),
        }


@dataclass(slots=True)
class ScanResult:
    """Everything one scan produced, including what it did not find."""

    opportunities: tuple[Opportunity, ...] = ()
    scanned: int = 0
    #: Conditions that blocked entries, counted across the watchlist.
    blockers: dict[str, int] = field(default_factory=dict)
    #: Symbols that could not be analysed, with the reason.
    failures: dict[str, str] = field(default_factory=dict)
    #: Symbols filtered out before analysis, with the reason.
    skipped: dict[str, str] = field(default_factory=dict)
    halt_reason: str | None = None
    duration_seconds: float = 0.0
    timestamp: datetime | None = None

    @property
    def tradable(self) -> tuple[Opportunity, ...]:
        return tuple(item for item in self.opportunities if item.tradable)

    @property
    def best(self) -> Opportunity | None:
        return self.opportunities[0] if self.opportunities else None

    def summary(self) -> str:
        return (
            f"{len(self.opportunities)} opportunit"
            f"{'y' if len(self.opportunities) == 1 else 'ies'} from {self.scanned} "
            f"symbol(s) in {self.duration_seconds:.1f}s "
            f"({len(self.tradable)} cleared risk)"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "scanned": self.scanned,
            "opportunities": [item.as_dict() for item in self.opportunities],
            "blockers": dict(self.blockers),
            "failures": dict(self.failures),
            "skipped": dict(self.skipped),
            "halt_reason": self.halt_reason,
            "duration_seconds": round(self.duration_seconds, 2),
        }


class MarketScanner:
    """Scores and ranks opportunities across a watchlist.

    Example
    -------
    >>> scanner = MarketScanner([build_strategy(name) for name in available_strategies()])
    >>> result = scanner.scan(frames, portfolio=portfolio)
    >>> for opportunity in result.opportunities[:5]:
    ...     print(opportunity.rank, opportunity.symbol, opportunity.confidence)
    """

    def __init__(
        self,
        strategies: list[BaseStrategy],
        *,
        indicators: IndicatorConfig | None = None,
        risk_manager: RiskManager | None = None,
        config: ScannerConfig | None = None,
    ) -> None:
        if not strategies:
            raise ValueError("A scanner needs at least one strategy")
        self.strategies = strategies
        self.indicators = indicators or IndicatorConfig()
        self.risk_manager = risk_manager
        self.config = config or ScannerConfig()

    # -- filtering ---------------------------------------------------------------

    def _liquidity_reason(self, frame: pd.DataFrame) -> str | None:
        """Why this symbol is not worth analysing, or ``None`` if it is."""
        if frame.empty:
            return "no bars"

        close = float(frame["close"].iloc[-1])
        if close < self.config.min_price:
            return f"price {close:,.2f} below the {self.config.min_price:,.2f} minimum"

        column = volume_sma_column(self.indicators.volume_sma_period)
        average_volume = (
            frame[column].iloc[-1] if column in frame.columns else frame["volume"].mean()
        )
        if pd.isna(average_volume):
            average_volume = float(frame["volume"].mean())

        turnover = float(average_volume) * close
        if turnover < self.config.min_avg_dollar_volume:
            return (
                f"turnover ${turnover:,.0f} below the "
                f"${self.config.min_avg_dollar_volume:,.0f} minimum"
            )
        return None

    # -- scanning ----------------------------------------------------------------

    def scan(
        self,
        frames: dict[str, pd.DataFrame],
        *,
        portfolio: PortfolioState | None = None,
        now: datetime | None = None,
    ) -> ScanResult:
        """Scan pre-fetched bars and return a ranked result.

        Taking frames rather than fetching them keeps the scanner testable
        offline and lets a backtest replay history through the same code.

        Parameters
        ----------
        frames:
            Symbol to OHLCV frame. Indicators are computed if absent.
        portfolio:
            Portfolio state for risk validation. Without it, opportunities are
            scored and ranked but not sized.
        now:
            Evaluation time, for cooldown arithmetic.
        """
        started = time.perf_counter()
        result = ScanResult(timestamp=now or datetime.now(tz=None).astimezone())

        if self.risk_manager is not None and portfolio is not None:
            result.halt_reason = self.risk_manager.trading_halted(portfolio, now=now)

        candidates: list[tuple[Signal, pd.DataFrame, dict]] = []

        for symbol in sorted(frames):
            frame = frames[symbol]
            try:
                enriched = self._prepare(frame)
            except Exception as error:  # noqa: BLE001 - one symbol must not stop the scan
                logger.exception("Could not prepare %s", symbol)
                result.failures[symbol] = str(error)
                continue

            reason = self._liquidity_reason(enriched)
            if reason is not None:
                result.skipped[symbol] = reason
                continue

            result.scanned += 1

            # Computed once per symbol and shared with every strategy and the
            # scorer, rather than recomputed per strategy.
            context = {
                "trend": analyze_trend(enriched, self.indicators),
                "volume": analyze_volume(enriched, self.indicators),
                "levels": find_support_resistance(enriched, self.indicators),
            }

            for strategy in self.strategies:
                try:
                    signal = strategy.generate_signal(symbol, enriched)
                except Exception as error:  # noqa: BLE001
                    logger.exception("%s failed on %s", strategy.name, symbol)
                    result.failures[f"{symbol}/{strategy.name}"] = str(error)
                    continue

                if signal is None:
                    for name in explain_blockers(strategy):
                        key = f"{strategy.name}.{name}"
                        result.blockers[key] = result.blockers.get(key, 0) + 1
                    continue

                candidates.append((signal, enriched, context))

        opportunities = self._score_and_rank(candidates, portfolio, now)
        result.opportunities = opportunities
        result.duration_seconds = time.perf_counter() - started
        logger.info("Scan complete: %s", result.summary())
        return result

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Enrich a frame once, reusing indicator columns if already present."""
        from trading_bot.indicators import atr_column

        if atr_column(self.indicators.atr_period) in frame.columns:
            return frame
        return calculate_all_indicators(frame, self.indicators)

    def _score_and_rank(
        self,
        candidates: list[tuple[Signal, pd.DataFrame, dict]],
        portfolio: PortfolioState | None,
        now: datetime | None,
    ) -> tuple[Opportunity, ...]:
        """Score every candidate, run risk, and order the survivors."""
        scored: list[tuple[Signal, float, tuple[FactorScore, ...]]] = []
        for signal, enriched, context in candidates:
            confidence, factors = score_opportunity(
                signal,
                enriched,
                self.indicators,
                trend=context["trend"],
                volume=context["volume"],
                levels=context["levels"],
            )
            if confidence >= self.config.min_confidence:
                scored.append((signal, confidence, factors))

        scored.sort(key=lambda item: item[1], reverse=True)

        # Risk runs on the ranked order, so when position slots are scarce they
        # go to the highest-scoring opportunities rather than to whichever symbol
        # happens to sort first alphabetically.
        decisions: dict[int, RiskDecision] = {}
        if self.risk_manager is not None and portfolio is not None:
            ordered = [signal for signal, _, _ in scored]
            working = portfolio
            for signal in ordered:
                decision = self.risk_manager.evaluate(signal, working, now=now)
                decisions[id(signal)] = decision
                if decision.approved:
                    working = _with_position(working, signal, decision)

        opportunities = [
            Opportunity(
                signal=signal,
                confidence=confidence,
                factors=factors,
                decision=decisions.get(id(signal)),
                rank=index,
            )
            for index, (signal, confidence, factors) in enumerate(scored, start=1)
        ]
        if self.config.max_results is not None:
            opportunities = opportunities[: self.config.max_results]
        return tuple(opportunities)


def _with_position(
    portfolio: PortfolioState, signal: Signal, decision: RiskDecision
) -> PortfolioState:
    """Fold an approval into the portfolio so the next check sees it."""
    from trading_bot.risk import OpenPosition
    from trading_bot.risk.position_sizing import ZERO, to_decimal

    entry = to_decimal(signal.entry_price)
    return PortfolioState(
        equity=portfolio.equity,
        cash=portfolio.cash,
        buying_power=max(portfolio.buying_power - decision.position_value, ZERO),
        positions=(
            *portfolio.positions,
            OpenPosition(
                symbol=signal.symbol,
                direction=signal.direction.value,
                quantity=decision.quantity,
                entry_price=entry,
                current_price=entry,
                stop_loss=to_decimal(signal.stop_loss),
                take_profit=to_decimal(signal.take_profit),
                strategy=signal.strategy,
            ),
        ),
        realized_pnl_today=portfolio.realized_pnl_today,
        session_start_equity=portfolio.session_start_equity,
        consecutive_losses=portfolio.consecutive_losses,
        last_loss_at=portfolio.last_loss_at,
        trading_blocked=portfolio.trading_blocked,
        halted=portfolio.halted,
        halt_reason=portfolio.halt_reason,
    )
