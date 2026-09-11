"""Paper execution and supervision for long swing options."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from trading_bot.options import SwingOptionSelector

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OptionExecutionReport:
    placed: int = 0
    alerted: int = 0
    skipped: int = 0
    failed: int = 0
    decisions: tuple[OptionExecutionDecision, ...] = ()


@dataclass(frozen=True, slots=True)
class OptionExecutionDecision:
    """Final option-level disposition for one underlying candidate."""

    symbol: str
    approved: bool
    reason: str


class OptionPaperExecutor:
    """Select contracts and either alert or place capped-loss paper positions."""

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
        """Alert on every ranked setup; submit only what the caps allow.

        Alerting and ordering are separated deliberately. The premium budget,
        the position count and the per-scan capacity exist to protect capital,
        so they gate **orders**. Letting them gate **alerts** too meant a
        manually-traded account went quiet after the second name in a scan and
        the reader simply never saw the third-ranked setup — the caps were
        silently rationing information rather than money. Every candidate that
        resolves to a real contract is now named, with what Botty did about it.

        The one candidate still passed over in silence is an underlying Botty
        already holds, where a second alert is a duplicate rather than news.
        """
        considered = list(opportunities)
        decisions: list[OptionExecutionDecision] = []
        active = self.db.option_selections.active()
        held_underlyings = {row["underlying_symbol"] for row in active}
        positions_full = (
            len(held_underlyings) >= self.settings.options.max_open_positions
        )
        remaining_premium = max(
            0.0, self.settings.options.max_total_premium
            - self.db.option_selections.premium_committed()
        )
        remaining_contracts = max(
            0, self.settings.options.max_total_contracts
            - self.db.option_selections.contracts_committed()
        )
        placed = alerted = skipped = failed = 0
        order_capacity = max(0, capacity)
        lookups = 0
        for opportunity in considered:
            signal = opportunity.signal
            decision = opportunity.decision
            if signal.symbol in held_underlyings:
                skipped += 1
                decisions.append(OptionExecutionDecision(
                    signal.symbol, False,
                    "not approved: Botty already tracks this underlying",
                ))
                continue
            if lookups >= self.settings.options.max_alerts_per_scan:
                skipped += 1
                decisions.append(OptionExecutionDecision(
                    signal.symbol, False,
                    "not approved: per-scan alert budget spent on higher-ranked setups",
                ))
                continue
            # Why no order goes out. This no longer suppresses the alert.
            order_block = None
            if positions_full:
                order_block = "maximum open Botty positions reached"
            elif placed >= order_capacity:
                order_block = "account capacity filled by higher-ranked contracts"
            elif remaining_contracts < 1:
                order_block = "total contract limit reached"
            elif decision is None:
                order_block = "no risk decision was available"
            elif not decision.approved:
                order_block = (
                    getattr(decision, "rejection_reason", None) or "risk checks failed"
                )
            selection_id = None
            try:
                lookups += 1
                quotes = self.chain.quotes(
                    signal.symbol, signal.direction.value, signal.entry_price
                )
                # A blocked candidate is sized against the full per-trade
                # budget rather than what is left of the account's: no order is
                # going out, and an alert should name the contract worth buying
                # rather than one shrunk to fit capacity the reader does not share.
                budget = (
                    self.settings.options.max_premium_per_trade
                    if order_block else remaining_premium
                )
                contract_room = (
                    self.settings.options.max_contracts_per_trade
                    if order_block else remaining_contracts
                )
                selection = self.selector.select(
                    direction=signal.direction.value, quotes=quotes, as_of=date.today(),
                    confidence=opportunity.confidence,
                    remaining_premium=budget,
                    remaining_contracts=contract_room,
                )
                if selection is None:
                    skipped += 1
                    reason = (
                        "not approved: no option contract passed DTE, delta, liquidity, "
                        "spread, and $500–$1,000 premium rules"
                    )
                    decisions.append(OptionExecutionDecision(signal.symbol, False, reason))
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
                contract_summary = (
                    f"{quote.contract_type.upper()} ${quote.strike:,.2f} "
                    f"exp {quote.expiration.isoformat()}, x{selection.quantity} "
                    f"at ${quote.ask:.2f}; premium ${selection.estimated_cost:,.0f}, "
                    f"{selection.days_to_expiry} DTE ({quote.symbol})"
                )
                # Written for a phone. The contract is the headline because it
                # is the thing actually bought; everything else is support.
                strike = (
                    f"{quote.strike:,.0f}" if float(quote.strike).is_integer()
                    else f"{quote.strike:,.2f}"
                )
                exit_by = quote.expiration - timedelta(
                    days=self.settings.options.exit_before_expiry_days
                )
                alert_message = (
                    f"BOTTY SWING ALERT — {signal.symbol} "
                    f"{signal.direction.value} ({opportunity.confidence:.0f}/100)\n"
                    f"  {signal.symbol} ${strike} {quote.contract_type.upper()}"
                    f"  {quote.expiration.isoformat()}  ({selection.days_to_expiry} DTE)\n"
                    f"  Buy x{selection.quantity} @ limit ${quote.ask:.2f}"
                    f"   —   premium ${selection.estimated_cost:,.0f}\n"
                    f"  Stock ${signal.entry_price:,.2f} · "
                    f"stop ${signal.stop_loss:,.2f} · "
                    f"target ${signal.take_profit:,.2f}\n"
                    f"  Hold {self.settings.options.planned_min_hold_days}-"
                    f"{self.settings.options.planned_max_hold_days} days; "
                    f"close by {exit_by.isoformat()}\n"
                    f"  {quote.symbol}"
                )
                if order_block:
                    self.db.option_selections.set_status(selection_id, "alerted")
                    alerted += 1
                    skipped += 1
                    decisions.append(OptionExecutionDecision(
                        signal.symbol, False,
                        f"alerted, no order ({order_block}): {contract_summary}",
                    ))
                    self._notify(f"{alert_message}\n  NO ORDER: {order_block}.")
                    continue
                if self.settings.options.alert_only:
                    self.db.option_selections.set_status(selection_id, "alerted")
                    alerted += 1
                    decisions.append(OptionExecutionDecision(
                        signal.symbol, True, f"approved alert: {contract_summary}",
                    ))
                    self._notify(f"{alert_message}\n  ALERT ONLY — NO ORDER SUBMITTED.")
                    continue
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
                # Only a real order consumes the account's capacity. An alert
                # spends nothing, so it must not decrement these.
                remaining_premium -= selection.estimated_cost
                remaining_contracts -= selection.quantity
                held_underlyings.add(signal.symbol)
                placed += 1
                alerted += 1
                decisions.append(OptionExecutionDecision(
                    signal.symbol, True, f"approved: {contract_summary}",
                ))
                self._notify(f"{alert_message}\n  PAPER ORDER SUBMITTED.")
            except Exception as error:  # noqa: BLE001 - isolate one candidate
                failed += 1
                decisions.append(OptionExecutionDecision(
                    signal.symbol, False, f"not approved: option execution failed ({error})"
                ))
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
        return OptionExecutionReport(
            placed=placed, alerted=alerted, skipped=skipped, failed=failed,
            decisions=tuple(decisions),
        )


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
