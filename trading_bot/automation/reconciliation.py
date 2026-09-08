"""Make the local audit database agree with Alpaca's broker truth."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from trading_bot.automation.notifications import Notifier
from trading_bot.data.database import Database

logger = logging.getLogger(__name__)
TERMINAL_FAILURES = {"canceled", "expired", "rejected", "suspended"}


def _flatten(orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for order in orders:
        flat.append(order)
        flat.extend(_flatten(list(order.get("legs") or [])))
    return flat


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    orders_seen: int = 0
    positions_seen: int = 0
    trades_opened: int = 0
    trades_closed: int = 0
    warnings: int = 0
    unprotected_positions: int = 0


class BrokerReconciler:
    """Idempotently import orders/fills and mirror currently open positions."""

    def __init__(self, broker, database: Database, notifier: Notifier) -> None:
        self.broker = broker
        self.db = database
        self.notifier = notifier

    def _notify(self, message: str) -> None:
        try:
            self.notifier.send(message)
        except Exception:
            logger.exception("Reconciliation notification failed")

    def _alert_once(self, key: str, message: str, *, symbol: str | None = None) -> None:
        state_key = f"active_alert:{key}"
        if self.db.state.get(state_key) == "true":
            return
        self.db.state.set(state_key, "true")
        self.db.events.record(
            category="automation_guard", level="ERROR", symbol=symbol,
            message=message,
        )
        self._notify(f"BOTTY NEEDS ATTENTION: {message}")

    def _resolve_alert(self, key: str) -> None:
        self.db.state.set(f"active_alert:{key}", "false")

    def reconcile(self) -> ReconciliationReport:
        self.db.initialize()
        account = self.broker.get_account()
        orders = _flatten(self.broker.get_orders(status="all", nested=True))
        positions = self.broker.get_positions()
        broker_symbols = {position["symbol"] for position in positions}
        opened = closed = warnings = 0

        for order in orders:
            self.db.orders.upsert_broker_order(order)
            client_id = str(order.get("client_order_id") or "")
            status = str(order.get("status") or "").lower()
            if client_id.startswith("botty-") and status in TERMINAL_FAILURES:
                message = f"Botty order {client_id} is {status}."
                self.db.events.record(
                    category="order_failure", level="ERROR", symbol=order.get("symbol"),
                    message=message, payload=order,
                )
                self._alert_once(
                    f"order:{order['id']}:{status}", message,
                    symbol=order.get("symbol"),
                )

            if (
                client_id.startswith("botty-") and status == "filled"
                and order.get("filled_avg_price") is not None
                and self.db.trades.by_broker_entry_order(str(order["id"])) is None
            ):
                legs = list(order.get("legs") or [])
                stop = next((leg.get("stop_price") for leg in legs if leg.get("stop_price")), None)
                target = next(
                    (leg.get("limit_price") for leg in legs if leg.get("limit_price")), None
                )
                direction = "LONG" if str(order["side"]).lower() == "buy" else "SHORT"
                trade_id = self.db.trades.open_trade(
                    symbol=str(order["symbol"]), direction=direction,
                    qty=float(order.get("filled_qty") or order.get("qty") or 0),
                    entry_price=float(order["filled_avg_price"]),
                    entry_ts=order.get("filled_at"), stop_loss=stop, take_profit=target,
                    metadata={"broker_entry_order_id": order["id"], "client_order_id": client_id},
                )
                opened += 1
                if str(order["symbol"]) in broker_symbols:
                    position = next(p for p in positions if p["symbol"] == order["symbol"])
                    self.db.positions.upsert(
                        symbol=str(order["symbol"]), direction=direction,
                        qty=abs(float(position["qty"])),
                        avg_entry_price=float(position["avg_entry_price"]), trade_id=trade_id,
                        stop_loss=stop, take_profit=target,
                        opened_at=order.get("filled_at"),
                    )
                else:
                    closing = next(
                        (
                            leg for leg in legs
                            if str(leg.get("status")).lower() == "filled"
                            and leg.get("filled_avg_price") is not None
                        ),
                        None,
                    )
                    if closing:
                        result = self.db.trades.close_trade(
                            trade_id, exit_price=float(closing["filled_avg_price"]),
                            exit_ts=closing.get("filled_at"),
                            exit_reason="stop_or_target_recovered",
                        )
                        closed += 1
                        pnl = float(result["pnl"]) if result else 0.0
                        self._notify(
                            f"Botty recovered closed paper trade: {order['symbol']} "
                            f"P&L ${pnl:,.2f}."
                        )

        unprotected = 0
        # Refresh every broker-held position, including manual ones. Unknown/manual
        # positions remain visible but are deliberately not invented as Botty trades.
        for position in positions:
            existing = self.db.positions.get(position["symbol"])
            trade = self.db.trades.open_for_symbol(position["symbol"])
            self.db.positions.upsert(
                symbol=position["symbol"],
                direction="LONG" if str(position["side"]).lower() == "long" else "SHORT",
                qty=abs(float(position["qty"])),
                avg_entry_price=float(position["avg_entry_price"]),
                trade_id=int(trade["id"]) if trade else None,
                strategy=existing.get("strategy") if existing else None,
                stop_loss=existing.get("stop_loss") if existing else None,
                take_profit=existing.get("take_profit") if existing else None,
                opened_at=existing.get("opened_at") if existing else None,
                metadata={"broker_managed": True, "unrealized_pl": position.get("unrealized_pl")},
            )
            botty_entries = [
                order for order in orders
                if order.get("symbol") == position["symbol"]
                and str(order.get("client_order_id") or "").startswith("botty-")
                and str(order.get("status")).lower() == "filled"
                and order.get("parent_order_id") is None
            ]
            if trade and botty_entries:
                latest = max(
                    botty_entries, key=lambda item: str(item.get("filled_at") or "")
                )
                active_legs = [
                    leg for leg in latest.get("legs") or []
                    if str(leg.get("status")).lower() not in TERMINAL_FAILURES | {"filled"}
                ]
                alert_key = f"unprotected:{position['symbol']}"
                if len(active_legs) < 2:
                    unprotected += 1
                    self._alert_once(
                        alert_key,
                        f"{position['symbol']} has no complete stop/target protection",
                        symbol=position["symbol"],
                    )
                else:
                    self._resolve_alert(alert_key)

        for local in self.db.positions.all():
            if local["symbol"] in broker_symbols:
                continue
            trade = self.db.trades.open_for_symbol(local["symbol"])
            if trade:
                closing = self._closing_fill(orders, trade)
                if closing:
                    reason = "stop_or_target" if closing.get("parent_order_id") else "broker_exit"
                    result = self.db.trades.close_trade(
                        int(trade["id"]), exit_price=float(closing["filled_avg_price"]),
                        exit_ts=closing.get("filled_at"), exit_reason=reason,
                    )
                    closed += 1
                    pnl = float(result["pnl"]) if result else 0.0
                    self._notify(
                        f"Botty paper trade closed: {local['symbol']} P&L ${pnl:,.2f}."
                    )
                else:
                    warnings += 1
                    message = f"{local['symbol']} disappeared without a matching exit fill"
                    self.db.events.record(
                        category="reconciliation", level="ERROR", symbol=local["symbol"],
                        message=message,
                    )
                    self._alert_once(
                        f"missing_exit:{local['symbol']}", message, symbol=local["symbol"]
                    )
                    continue
            self.db.positions.remove(local["symbol"])

        self.db.equity.record(
            equity=float(account.equity), cash=float(account.cash),
            unrealized_pnl=sum(float(p.get("unrealized_pl") or 0) for p in positions),
            open_positions=len(positions), ts=datetime.now(timezone.utc),
        )
        return ReconciliationReport(
            orders_seen=len(orders), positions_seen=len(positions), trades_opened=opened,
            trades_closed=closed, warnings=warnings,
            unprotected_positions=unprotected,
        )

    @staticmethod
    def _closing_fill(
        orders: list[dict[str, Any]], trade: dict[str, Any]
    ) -> dict[str, Any] | None:
        closing_side = "sell" if str(trade["direction"]).upper() == "LONG" else "buy"
        candidates = [
            order for order in orders
            if order.get("symbol") == trade["symbol"]
            and str(order.get("side")).lower() == closing_side
            and str(order.get("status")).lower() == "filled"
            and order.get("filled_avg_price") is not None
            and str(order.get("filled_at") or "") >= str(trade["entry_ts"])
        ]
        return max(candidates, key=lambda item: str(item.get("filled_at") or ""), default=None)
