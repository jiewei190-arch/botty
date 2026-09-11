"""Guarded conversion of approved scan opportunities into paper bracket orders."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from trading_bot.execution.broker import BrokerError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    placed: int = 0
    failed: int = 0
    skipped_held: int = 0
    stale_blocked: int = 0
    circuit_opened: bool = False


class PaperExecutor:
    """Submit risk-approved opportunities; incapable of opting into live orders."""

    def __init__(self, broker, database, notifier, settings, *, now=None) -> None:
        if not broker.is_paper:
            raise BrokerError("automated execution is paper-only in this release")
        self.broker = broker
        self.db = database
        self.notifier = notifier
        self.settings = settings
        self.now = now or (lambda: datetime.now(timezone.utc))

    def _notify(self, message: str) -> None:
        try:
            self.notifier.send(message)
        except Exception:
            logger.exception("Execution notification failed")

    def execute(self, opportunities, capacity: int) -> ExecutionReport:
        held = {item["symbol"] for item in self.broker.get_positions()}
        placed = failed = skipped = stale = 0
        circuit_opened = False
        for opportunity in opportunities[:capacity]:
            signal = opportunity.signal
            decision = opportunity.decision
            if signal.symbol in held:
                skipped += 1
                continue
            if decision is None or not decision.approved or int(decision.shares) < 1:
                continue

            signal_time = signal.timestamp
            if signal_time.tzinfo is None:
                signal_time = signal_time.replace(tzinfo=timezone.utc)
            age_hours = (
                self.now() - signal_time.astimezone(timezone.utc)
            ).total_seconds() / 3600
            if age_hours > self.settings.automation.max_signal_age_hours:
                stale += 1
                message = f"{signal.symbol} signal is stale ({age_hours:.1f} hours); order blocked"
                self.db.events.record(
                    category="stale_signal", level="ERROR", symbol=signal.symbol,
                    message=message,
                )
                self._notify(f"BOTTY NEEDS ATTENTION: {message}.")
                continue

            stamp = signal.timestamp.strftime("%Y%m%d")
            client_id = f"botty-{stamp}-{signal.symbol}-{signal.strategy}".lower()
            signal_id = self.db.signals.record(
                symbol=signal.symbol, strategy=signal.strategy,
                direction=signal.direction.value, confidence=opportunity.confidence,
                ts=signal.timestamp, entry_price=signal.entry_price,
                stop_loss=signal.stop_loss, take_profit=signal.take_profit,
                risk_reward=signal.risk_reward_ratio, reasons=signal.reasons,
                accepted=True,
            )
            try:
                order = self.broker.submit_bracket_order(
                    symbol=signal.symbol, qty=int(decision.shares),
                    side="buy" if signal.direction.value == "LONG" else "sell",
                    take_profit=signal.take_profit, stop_loss=signal.stop_loss,
                    client_order_id=client_id,
                )
            except BrokerError as error:
                failed += 1
                self.db.events.record(
                    category="order_submission", level="ERROR", symbol=signal.symbol,
                    message=str(error), payload={"client_order_id": client_id},
                )
                self._notify(f"BOTTY NEEDS ATTENTION: {signal.symbol} paper order failed: {error}")
                if failed >= self.settings.automation.max_order_failures_per_scan:
                    circuit_opened = True
                    message = "Order rejection limit reached; remaining entries blocked"
                    self.db.events.record(
                        category="execution_circuit_breaker", level="CRITICAL",
                        message=message, payload={"failures": failed},
                    )
                    self._notify(
                        "BOTTY NEEDS ATTENTION: execution circuit breaker opened "
                        f"after {failed} order failures."
                    )
                    break
                continue
            self.db.orders.record(
                signal_id=signal_id, broker_order_id=order["id"],
                client_order_id=order["client_order_id"], symbol=signal.symbol,
                side="buy" if signal.direction.value == "LONG" else "sell",
                qty=int(decision.shares), order_type="market", time_in_force="gtc",
                status=order["status"],
                raw={"stop_loss": signal.stop_loss, "take_profit": signal.take_profit},
            )
            placed += 1
            held.add(signal.symbol)
            self._notify(
                f"Botty paper order submitted: {signal.symbol} {signal.direction.value} "
                f"x{int(decision.shares)}; stop ${signal.stop_loss:,.2f}, "
                f"target ${signal.take_profit:,.2f}."
            )
        return ExecutionReport(placed, failed, skipped, stale, circuit_opened)
