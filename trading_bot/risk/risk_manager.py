"""The risk gate.

Every trade passes through :meth:`RiskManager.evaluate`, which is the only thing
in the system that turns a signal into a quantity. If it does not approve, there
is no size, and with no size there is nothing to send to a broker. That is the
enforcement mechanism: the execution layer (Phase 7) takes a
:class:`RiskDecision`, not a :class:`~trading_bot.strategies.Signal`, so
"forgetting" to check risk is not an available mistake.

What it checks, in order
------------------------

=====================  ==================================================
Check                  Blocks a trade when
=====================  ==================================================
``account_tradable``   The broker has blocked the account, or the operator
                       pulled the kill switch
``daily_loss``         Today's loss has reached the daily limit
``cooldown``           A run of losses is still cooling off
``open_positions``     Every position slot is already used
``duplicate``          A position in this symbol is already open
``confidence``         The signal is below the confidence floor
``risk_reward``        The setup pays too little for what it risks
``position_size``      The limits leave less than one share
``exposure``           The new position would breach total exposure
=====================  ==================================================

Every check is reported whether it passed or failed, so a rejection explains
itself and an approval shows how much headroom was left.

Design
------
The manager is a **pure function** of a signal and a
:class:`~trading_bot.risk.portfolio.PortfolioState`. It performs no I/O: no
database reads, no broker calls, no clock lookups it was not given. Every limit
is therefore testable in isolation, and a backtest exercises exactly the same
code the live bot does rather than a parallel implementation that drifts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from trading_bot.config.settings import RiskSettings
from trading_bot.risk.portfolio import PortfolioState
from trading_bot.risk.position_sizing import (
    ZERO,
    PositionSize,
    SizingConstraint,
    calculate_position_size,
    to_decimal,
)
from trading_bot.strategies import Signal

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RiskCheck:
    """One limit's verdict.

    ``limit`` and ``actual`` are recorded even on a pass, so the decision shows
    how close the trade came rather than only whether it cleared.
    """

    name: str
    passed: bool
    detail: str
    limit: float | None = None
    actual: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "limit": self.limit,
            "actual": self.actual,
        }


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """The verdict on one signal.

    A decision with ``approved=False`` carries ``quantity=0``. There is no way
    to obtain a size without an approval, which is the point.
    """

    approved: bool
    signal: Signal
    quantity: Decimal = ZERO
    sizing: PositionSize | None = None
    checks: tuple[RiskCheck, ...] = ()
    rejection_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def shares(self) -> int:
        """Whole shares to trade. Zero unless approved."""
        return int(self.quantity)

    @property
    def risk_amount(self) -> Decimal:
        return self.sizing.risk_amount if self.sizing else ZERO

    @property
    def position_value(self) -> Decimal:
        return self.sizing.position_value if self.sizing else ZERO

    @property
    def failed_checks(self) -> tuple[RiskCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def summary(self) -> str:
        """One line, suitable for a log or a dashboard row."""
        if not self.approved:
            return f"REJECTED {self.signal.symbol}: {self.rejection_reason}"
        return (
            f"APPROVED {self.signal.symbol} {self.signal.direction.value} "
            f"{self.quantity} share(s) risking {self.risk_amount:.2f} "
            f"({self.sizing.binding_constraint.description} was binding)"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "symbol": self.signal.symbol,
            "direction": self.signal.direction.value,
            "strategy": self.signal.strategy,
            "quantity": float(self.quantity),
            "risk_amount": float(self.risk_amount),
            "position_value": float(self.position_value),
            "rejection_reason": self.rejection_reason,
            "checks": [check.as_dict() for check in self.checks],
            "sizing": self.sizing.as_dict() if self.sizing else None,
        }


class RiskManager:
    """Applies the configured limits to a proposed trade.

    Example
    -------
    >>> manager = RiskManager(settings.risk)
    >>> decision = manager.evaluate(signal, portfolio)
    >>> if decision.approved:
    ...     print(decision.shares, decision.risk_amount)
    ... else:
    ...     print(decision.rejection_reason)
    """

    def __init__(
        self,
        settings: RiskSettings | None = None,
        *,
        allow_fractional: bool = False,
        slippage_pct: float = 0.0,
    ) -> None:
        self.settings = settings or RiskSettings()
        self.allow_fractional = allow_fractional
        #: Assume stops fill this much worse than their price when sizing.
        #: Zero by default: the size is exactly what the stated stop implies.
        self.slippage_pct = slippage_pct

    # -- portfolio-level gate ----------------------------------------------------

    def trading_halted(
        self, portfolio: PortfolioState, *, now: datetime | None = None
    ) -> str | None:
        """Why trading is stopped right now, or ``None`` if it is not.

        Cheap enough to call before scanning, so a halted bot does not spend
        time and rate limit looking for trades it cannot take.
        """
        if portfolio.halted:
            return portfolio.halt_reason or "Trading halted by the operator"
        if portfolio.trading_blocked:
            return "The broker has blocked this account"

        loss_pct = portfolio.daily_pnl_pct
        if loss_pct <= -self.settings.max_daily_loss_pct:
            return (
                f"Daily loss limit reached: {loss_pct:.2f}% against a "
                f"{self.settings.max_daily_loss_pct:.2f}% limit"
            )

        remaining = self._cooldown_remaining(portfolio, now)
        if remaining is not None:
            return (
                f"Cooling off after {portfolio.consecutive_losses} consecutive losses — "
                f"{int(remaining.total_seconds() // 60)} minute(s) remaining"
            )
        return None

    def _cooldown_remaining(
        self, portfolio: PortfolioState, now: datetime | None = None
    ) -> timedelta | None:
        """Time left in a losing-streak cooldown, or ``None`` if not cooling off."""
        if portfolio.consecutive_losses < self.settings.consecutive_loss_limit:
            return None
        if self.settings.cooldown_minutes <= 0:
            return None
        if portfolio.last_loss_at is None:
            # Streak reached but no timestamp: refuse until one exists rather
            # than treat an unknown cooldown as an expired one.
            return timedelta(minutes=self.settings.cooldown_minutes)

        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        elapsed = current - portfolio.last_loss_at
        window = timedelta(minutes=self.settings.cooldown_minutes)
        return window - elapsed if elapsed < window else None

    # -- per-signal evaluation ---------------------------------------------------

    def evaluate(
        self,
        signal: Signal,
        portfolio: PortfolioState,
        *,
        now: datetime | None = None,
    ) -> RiskDecision:
        """Approve or reject a signal, and size it if approved.

        Returns
        -------
        RiskDecision
            ``approved`` is True only when every check passed and the limits
            leave at least one share.
        """
        checks: list[RiskCheck] = []

        halt = self.trading_halted(portfolio, now=now)
        checks.append(
            RiskCheck(
                "account_tradable",
                halt is None,
                halt or "Account is able to trade",
            )
        )
        checks.append(self._check_daily_loss(portfolio))
        checks.append(self._check_cooldown(portfolio, now))
        checks.append(self._check_open_positions(portfolio))
        checks.append(self._check_duplicate(signal, portfolio))
        checks.append(self._check_confidence(signal))
        checks.append(self._check_risk_reward(signal))

        # Size even when a check has already failed: a rejected signal's
        # would-be size is exactly what you need to tune a limit.
        sizing = self._size(signal, portfolio)
        checks.append(self._check_size(sizing))
        checks.append(self._check_exposure(sizing, portfolio))

        failed = [check for check in checks if not check.passed]
        if failed:
            reason = failed[0].detail
            logger.info(
                "Risk rejected %s %s from %s: %s",
                signal.symbol, signal.direction.value, signal.strategy, reason,
            )
            return RiskDecision(
                approved=False,
                signal=signal,
                quantity=ZERO,
                sizing=sizing,
                checks=tuple(checks),
                rejection_reason=reason,
                metadata={"failed_checks": [check.name for check in failed]},
            )

        decision = RiskDecision(
            approved=True,
            signal=signal,
            quantity=sizing.quantity,
            sizing=sizing,
            checks=tuple(checks),
        )
        logger.info("%s", decision.summary())
        return decision

    # -- individual checks -------------------------------------------------------

    def _check_daily_loss(self, portfolio: PortfolioState) -> RiskCheck:
        loss_pct = portfolio.daily_pnl_pct
        limit = self.settings.max_daily_loss_pct
        passed = loss_pct > -limit
        return RiskCheck(
            "daily_loss",
            passed,
            (
                f"Daily P&L {loss_pct:+.2f}% is within the {limit:.2f}% loss limit"
                if passed
                else f"Daily loss limit reached: {loss_pct:.2f}% against {limit:.2f}%"
            ),
            limit=-limit,
            actual=loss_pct,
        )

    def _check_cooldown(
        self, portfolio: PortfolioState, now: datetime | None
    ) -> RiskCheck:
        remaining = self._cooldown_remaining(portfolio, now)
        passed = remaining is None
        return RiskCheck(
            "cooldown",
            passed,
            (
                f"{portfolio.consecutive_losses} consecutive loss(es), below the "
                f"{self.settings.consecutive_loss_limit} that trigger a cooldown"
                if passed
                else f"Cooling off after {portfolio.consecutive_losses} consecutive "
                f"losses — {int(remaining.total_seconds() // 60)} minute(s) remaining"
            ),
            limit=float(self.settings.consecutive_loss_limit),
            actual=float(portfolio.consecutive_losses),
        )

    def _check_open_positions(self, portfolio: PortfolioState) -> RiskCheck:
        limit = self.settings.max_open_positions
        passed = portfolio.open_count < limit
        return RiskCheck(
            "open_positions",
            passed,
            (
                f"{portfolio.open_count} of {limit} position slots used"
                if passed
                else f"All {limit} position slots are in use"
            ),
            limit=float(limit),
            actual=float(portfolio.open_count),
        )

    def _check_duplicate(self, signal: Signal, portfolio: PortfolioState) -> RiskCheck:
        held = portfolio.has_position(signal.symbol)
        return RiskCheck(
            "duplicate",
            not held,
            (
                f"Already holding {signal.symbol} — refusing to add to it"
                if held
                else f"No open position in {signal.symbol}"
            ),
        )

    def _check_confidence(self, signal: Signal) -> RiskCheck:
        limit = self.settings.min_confidence
        passed = signal.confidence >= limit
        return RiskCheck(
            "confidence",
            passed,
            (
                f"Confidence {signal.confidence:.0f} meets the {limit:.0f} floor"
                if passed
                else f"Confidence {signal.confidence:.0f} is below the {limit:.0f} floor"
            ),
            limit=limit,
            actual=signal.confidence,
        )

    def _check_risk_reward(self, signal: Signal) -> RiskCheck:
        limit = self.settings.min_risk_reward
        ratio = signal.risk_reward_ratio
        passed = ratio >= limit
        return RiskCheck(
            "risk_reward",
            passed,
            (
                f"Reward:risk of 1:{ratio:.2f} meets the 1:{limit:.2f} minimum"
                if passed
                else f"Reward:risk of 1:{ratio:.2f} is below the 1:{limit:.2f} minimum"
            ),
            limit=limit,
            actual=ratio,
        )

    def _check_size(self, sizing: PositionSize) -> RiskCheck:
        passed = sizing.is_tradable
        return RiskCheck(
            "position_size",
            passed,
            (
                sizing.explain()
                if passed
                else "The limits allow less than one share — "
                + (sizing.notes[0] if sizing.notes else "position rounds to zero")
            ),
            limit=float(sizing.risk_budget),
            actual=float(sizing.risk_amount),
        )

    def _check_exposure(
        self, sizing: PositionSize, portfolio: PortfolioState
    ) -> RiskCheck:
        limit = self.settings.max_portfolio_exposure_pct
        if portfolio.equity <= 0:
            return RiskCheck("exposure", False, "Account equity is zero or negative")

        projected = portfolio.total_exposure + sizing.position_value
        projected_pct = float(projected / portfolio.equity * 100)
        # Sizing already caps to this limit; a breach here means the account
        # was over-exposed before this trade was considered.
        passed = projected_pct <= limit + 1e-9
        return RiskCheck(
            "exposure",
            passed,
            (
                f"Exposure would be {projected_pct:.1f}% of equity, within {limit:.0f}%"
                if passed
                else f"Exposure would reach {projected_pct:.1f}%, above the {limit:.0f}% limit"
            ),
            limit=limit,
            actual=projected_pct,
        )

    # -- sizing ------------------------------------------------------------------

    def _size(self, signal: Signal, portfolio: PortfolioState) -> PositionSize:
        """Size the signal, returning a zero size if it cannot be sized at all."""
        try:
            return calculate_position_size(
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                equity=portfolio.equity,
                max_risk_per_trade_pct=self.settings.max_risk_per_trade_pct,
                max_position_size_pct=self.settings.max_position_size_pct,
                max_portfolio_exposure_pct=self.settings.max_portfolio_exposure_pct,
                current_exposure=portfolio.total_exposure,
                buying_power=portfolio.buying_power or None,
                allow_fractional=self.allow_fractional,
                slippage_pct=self.slippage_pct,
            )
        except ValueError as error:
            logger.warning("Could not size %s: %s", signal.symbol, error)
            entry = to_decimal(signal.entry_price)
            return PositionSize(
                quantity=ZERO,
                risk_amount=ZERO,
                position_value=ZERO,
                risk_per_share=abs(entry - to_decimal(signal.stop_loss)),
                risk_budget=ZERO,
                binding_constraint=SizingConstraint.NONE,
                notes=(str(error),),
            )

    def evaluate_many(
        self,
        signals: list[Signal],
        portfolio: PortfolioState,
        *,
        now: datetime | None = None,
    ) -> list[RiskDecision]:
        """Evaluate several signals against a portfolio that fills as it goes.

        Approvals accumulate into the portfolio before the next signal is judged,
        so a batch cannot approve six trades into five slots — which is exactly
        what evaluating each one against the original snapshot would do.
        Highest confidence is considered first, so when slots are scarce they go
        to the strongest setups.
        """
        from trading_bot.risk.portfolio import OpenPosition

        decisions: list[RiskDecision] = []
        working = portfolio
        for signal in sorted(signals, key=lambda item: item.confidence, reverse=True):
            decision = self.evaluate(signal, working, now=now)
            decisions.append(decision)
            if decision.approved:
                entry = to_decimal(signal.entry_price)
                working = PortfolioState(
                    equity=working.equity,
                    cash=working.cash,
                    buying_power=max(working.buying_power - decision.position_value, ZERO),
                    positions=(
                        *working.positions,
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
                    realized_pnl_today=working.realized_pnl_today,
                    session_start_equity=working.session_start_equity,
                    consecutive_losses=working.consecutive_losses,
                    last_loss_at=working.last_loss_at,
                    trading_blocked=working.trading_blocked,
                    halted=working.halted,
                    halt_reason=working.halt_reason,
                )
        return decisions
