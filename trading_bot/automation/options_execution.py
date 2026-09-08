"""Paper execution and supervision for long swing options."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone

from trading_bot.options import SwingOptionSelector

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OptionExecutionReport:
    placed: int = 0
    skipped: int = 0
    failed: int = 0


class OptionPaperExecutor:
    """Turn approved underlying setups into capped-loss long option positions."""

    def __init__(self, broker, chain, database, notifier, settings) -> None:
        if not broker.is_paper:
            raise ValueError("automated option execution is paper-only")
        self.broker, self.chain, self.db = broker, chain, database
        self.notifier, self.settings = notifier, settings
        self.selector = SwingOptionSelector(settings.options)

    def _notify(self, message: str) -> None:
        try:
            self.notifier.send(message)
        except Exception:
            logger.exception("Option notification failed")

    def execute(self, opportunities, capacity: int) -> OptionExecutionReport:
        active = self.db.option_selections.active()
        active_underlyings = {row["underlying_symbol"] for row in active}
        if len(active_underlyings) >= self.settings.options.max_open_positions:
            return OptionExecutionReport(skipped=len(opportunities[:capacity]))
        held_underlyings = {row["underlying_symbol"] for row in active}
        remaining_premium = max(
            0.0, self.settings.options.max_total_premium
            - self.db.option_selections.premium_committed()
        )
        remaining_contracts = max(
            0, self.settings.options.max_total_contracts
            - self.db.option_selections.contracts_committed()
        )
        placed = skipped = failed = 0
        for opportunity in opportunities[:capacity]:
            signal = opportunity.signal
            decision = opportunity.decision
            if (
                signal.symbol in held_underlyings
                or len(held_underlyings) >= self.settings.options.max_open_positions
                or remaining_contracts < 1
                or decision is None
                or not decision.approved
            ):
                skipped += 1
                continue
            selection_id = None
            try:
                quotes = self.chain.quotes(
                    signal.symbol, signal.direction.value, signal.entry_price
                )
                selection = self.selector.select(
                    direction=signal.direction.value, quotes=quotes, as_of=date.today(),
                    confidence=opportunity.confidence,
                    remaining_premium=remaining_premium,
                    remaining_contracts=remaining_contracts,
                )
                if selection is None:
                    skipped += 1
                    self.db.events.record(
                        category="option_selection", symbol=signal.symbol,
                        message="No contract passed DTE, delta, liquidity, and premium rules",
                    )
                    continue
                quote = selection.quote
                signal_id = self.db.signals.record(
                    symbol=signal.symbol, strategy=signal.strategy,
                    direction=signal.direction.value, confidence=opportunity.confidence,
                    ts=signal.timestamp, entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss, take_profit=signal.take_profit,
                    risk_reward=signal.risk_reward_ratio, reasons=signal.reasons, accepted=True,
                    metadata={"instrument": "option"},
                )
                selection_id = self.db.option_selections.record(
                    signal_id=signal_id, underlying_symbol=quote.underlying,
                    contract_symbol=quote.symbol, contract_type=quote.contract_type,
                    expiration=quote.expiration, strike=quote.strike, bid=quote.bid,
                    ask=quote.ask, delta=quote.delta,
                    implied_volatility=quote.implied_volatility,
                    daily_volume=quote.daily_volume, open_interest=quote.open_interest,
                    quantity=selection.quantity, estimated_cost=selection.estimated_cost,
                    metadata={
                        "planned_hold_days": [self.settings.options.planned_min_hold_days,
                                              self.settings.options.planned_max_hold_days],
                        "dte_at_entry": selection.days_to_expiry,
                    },
                )
                client_id = (
                    f"botty-opt-{date.today():%Y%m%d}-{signal.symbol}-"
                    f"{signal.strategy}"
                ).lower()
                order = self.broker.submit_option_order(
                    contract_symbol=quote.symbol, qty=selection.quantity,
                    limit_price=quote.ask, client_order_id=client_id, side="buy",
                )
                self.db.orders.record(
                    signal_id=signal_id, broker_order_id=order["id"],
                    client_order_id=order.get("client_order_id"), symbol=quote.symbol,
                    side="buy", qty=selection.quantity, order_type="limit",
                    time_in_force="day", limit_price=quote.ask,
                    status=order.get("status", "accepted"),
                    raw={"underlying": signal.symbol, "instrument": "option"},
                )
                self.db.option_selections.set_status(selection_id, "submitted")
                remaining_premium -= selection.estimated_cost
                remaining_contracts -= selection.quantity
                held_underlyings.add(signal.symbol)
                placed += 1
                self._notify(
                    f"BOTTY SWING FOUND: {signal.symbol} {signal.direction.value} "
                    f"({opportunity.confidence:.0f}/100) | {quote.symbol} "
                    f"x{selection.quantity} @ limit ${quote.ask:.2f} | "
                    f"estimated premium risk ${selection.estimated_cost:,.0f} | "
                    f"{selection.days_to_expiry} DTE | PAPER ORDER SUBMITTED."
                )
            except Exception as error:  # noqa: BLE001 - isolate one candidate
                failed += 1
                if selection_id is not None:
                    self.db.option_selections.set_status(selection_id, "failed")
                logger.exception("Option execution failed for %s", signal.symbol)
                self.db.events.record(
                    category="option_execution", level="ERROR", symbol=signal.symbol,
                    message=str(error),
                )
                self._notify(f"BOTTY NEEDS ATTENTION: {signal.symbol} option entry failed: {error}")
                if failed >= self.settings.automation.max_order_failures_per_scan:
                    break
        return OptionExecutionReport(placed, skipped, failed)


class OptionPositionSupervisor:
    """Exit long options by premium P&L, time held, or expiry buffer."""

    def __init__(self, broker, chain, database, notifier, settings, *, now=None) -> None:
        self.broker, self.chain, self.db = broker, chain, database
        self.notifier, self.settings = notifier, settings
        self.now = now or (lambda: datetime.now(timezone.utc))

    def _notify(self, message: str) -> None:
        try:
            self.notifier.send(message)
        except Exception:
            logger.exception("Option exit notification failed")

    def supervise(self) -> int:
        active = self.db.option_selections.active()
        if not active:
            return 0
        positions = {row["symbol"]: row for row in self.broker.get_positions()}
        open_sells = {
            row["symbol"] for row in self.broker.get_orders(status="open", nested=False)
            if str(row.get("side")).lower() == "sell"
        }
        quote_map = self.chain.latest_mid([
            row["contract_symbol"] for row in active if row["contract_symbol"] in positions
        ])
        submitted = 0
        today = self.now().date()
        for selected in active:
            symbol = selected["contract_symbol"]
            position = positions.get(symbol)
            if position is None or symbol in open_sells or symbol not in quote_map:
                continue
            local_position = self.db.positions.get(symbol)
            opened_at = local_position.get("opened_at") if local_position else None
            opened_day = (
                datetime.fromisoformat(opened_at).date()
                if opened_at else datetime.fromisoformat(selected["selected_at"]).date()
            )
            if opened_day == today:
                continue  # structural day-trade block
            mid, bid, _ask = quote_map[symbol]
            entry = float(position["avg_entry_price"])
            change_pct = (mid / entry - 1) * 100 if entry else 0.0
            expiry = date.fromisoformat(str(selected["expiration"]))
            held_days = (today - opened_day).days
            dte = (expiry - today).days
            reason = None
            if change_pct >= self.settings.options.profit_target_pct:
                reason = "profit_target"
            elif change_pct <= -self.settings.options.stop_loss_pct:
                reason = "premium_stop"
            elif held_days >= self.settings.options.planned_max_hold_days:
                reason = "maximum_hold"
            elif dte <= self.settings.options.exit_before_expiry_days:
                reason = "expiry_buffer"
            if reason is None or bid <= 0:
                continue
            order = self.broker.submit_option_order(
                contract_symbol=symbol, qty=int(abs(float(position["qty"]))),
                limit_price=bid, side="sell",
                client_order_id=f"botty-opt-exit-{today:%Y%m%d}-{selected['id']}",
            )
            self.db.orders.record(
                broker_order_id=order["id"], client_order_id=order.get("client_order_id"),
                symbol=symbol, side="sell", qty=abs(float(position["qty"])),
                order_type="limit", time_in_force="day", limit_price=bid,
                status=order.get("status", "accepted"), raw={"exit_reason": reason},
            )
            self.db.option_selections.set_status(int(selected["id"]), "exit_submitted")
            submitted += 1
            self._notify(
                f"BOTTY SWING EXIT: {selected['underlying_symbol']} | {symbol} "
                f"x{int(abs(float(position['qty'])))} @ limit ${bid:.2f} | "
                f"reason {reason.replace('_', ' ')} | PAPER EXIT SUBMITTED."
            )
        return submitted
