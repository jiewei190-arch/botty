from datetime import datetime, timezone
from decimal import Decimal

from trading_bot.automation.reconciliation import BrokerReconciler
from trading_bot.data.database import Database
from trading_bot.data.models import AccountSnapshot


class Recorder:
    def __init__(self):
        self.messages = []

    def send(self, message):
        self.messages.append(message)


def account():
    return AccountSnapshot(
        account_id="paper", status="ACTIVE", currency="USD",
        equity=Decimal("10000"), cash=Decimal("8000"),
        buying_power=Decimal("16000"), portfolio_value=Decimal("10000"),
        last_equity=Decimal("9900"), is_paper=True,
    )


def entry(status="filled"):
    return {
        "id": "entry-1", "parent_order_id": None, "client_order_id": "botty-aapl-momentum",
        "symbol": "AAPL", "side": "buy", "type": "market", "time_in_force": "gtc",
        "status": status, "qty": 10, "filled_qty": 10,
        "filled_avg_price": 100.0 if status == "filled" else None,
        "created_at": datetime(2026, 9, 8, 13, 30, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 9, 8, 13, 31, tzinfo=timezone.utc),
        "filled_at": datetime(2026, 9, 8, 13, 31, tzinfo=timezone.utc),
        "limit_price": None, "stop_price": None,
        "legs": [
            {"id": "target-1", "parent_order_id": "entry-1", "client_order_id": "leg-t",
             "symbol": "AAPL", "side": "sell", "type": "limit", "time_in_force": "gtc",
             "status": "new", "qty": 10, "filled_qty": 0, "filled_avg_price": None,
             "limit_price": 120.0, "stop_price": None, "created_at": None,
             "updated_at": None, "filled_at": None, "legs": []},
            {"id": "stop-1", "parent_order_id": "entry-1", "client_order_id": "leg-s",
             "symbol": "AAPL", "side": "sell", "type": "stop", "time_in_force": "gtc",
             "status": "new", "qty": 10, "filled_qty": 0, "filled_avg_price": None,
             "limit_price": None, "stop_price": 95.0, "created_at": None,
             "updated_at": None, "filled_at": None, "legs": []},
        ],
    }


class FakeBroker:
    def __init__(self, orders, positions):
        self.orders = orders
        self.positions = positions

    def get_account(self):
        return account()

    def get_orders(self, **kwargs):
        return self.orders

    def get_positions(self):
        return self.positions


def position():
    return {"symbol": "AAPL", "qty": 10, "side": "long", "avg_entry_price": 100,
            "unrealized_pl": 25}


def test_filled_botty_entry_opens_trade_and_position():
    db = Database(":memory:")
    report = BrokerReconciler(FakeBroker([entry()], [position()]), db, Recorder()).reconcile()
    assert report.trades_opened == 1
    assert db.trades.open_for_symbol("AAPL")["entry_price"] == 100
    assert db.positions.get("AAPL")["stop_loss"] == 95
    assert db.equity.latest()["open_positions"] == 1


def test_reconciliation_is_idempotent():
    db = Database(":memory:")
    reconciler = BrokerReconciler(FakeBroker([entry()], [position()]), db, Recorder())
    reconciler.reconcile()
    reconciler.reconcile()
    assert len(db.trades.open_trades()) == 1
    assert len(db.query("SELECT * FROM orders")) == 3


def test_filled_bracket_leg_closes_trade_when_position_disappears():
    db = Database(":memory:")
    broker = FakeBroker([entry()], [position()])
    reconciler = BrokerReconciler(broker, db, Recorder())
    reconciler.reconcile()
    closed = entry()
    closed["legs"][0].update({
        "status": "filled", "filled_qty": 10, "filled_avg_price": 120,
        "filled_at": datetime(2026, 9, 9, 14, 0, tzinfo=timezone.utc),
    })
    broker.orders, broker.positions = [closed], []
    report = reconciler.reconcile()
    assert report.trades_closed == 1
    assert db.trades.history()[0]["pnl"] == 200
    assert db.positions.get("AAPL") is None


def test_rejected_botty_order_sends_attention_alert():
    notices = Recorder()
    rejected = entry("rejected")
    rejected["filled_at"] = None
    BrokerReconciler(FakeBroker([rejected], []), Database(":memory:"), notices).reconcile()
    assert any("NEEDS ATTENTION" in message for message in notices.messages)


def test_missing_protective_leg_alert_is_deduplicated():
    notices = Recorder()
    incomplete = entry()
    incomplete["legs"] = incomplete["legs"][:1]
    reconciler = BrokerReconciler(
        FakeBroker([incomplete], [position()]), Database(":memory:"), notices
    )
    first = reconciler.reconcile()
    reconciler.reconcile()
    assert first.unprotected_positions == 1
    attention = [message for message in notices.messages if "protection" in message]
    assert len(attention) == 1


def test_round_trip_completed_while_offline_is_recovered_exactly_once():
    db = Database(":memory:")
    completed = entry()
    completed["legs"][0].update({
        "status": "filled", "filled_qty": 10, "filled_avg_price": 120,
        "filled_at": datetime(2026, 9, 9, 14, 0, tzinfo=timezone.utc),
    })
    reconciler = BrokerReconciler(FakeBroker([completed], []), db, Recorder())
    first = reconciler.reconcile()
    second = reconciler.reconcile()
    assert first.trades_opened == 1
    assert first.trades_closed == 1
    assert second.trades_opened == 0
    assert second.trades_closed == 0
    assert len(db.trades.history()) == 1
    assert db.trades.history()[0]["pnl"] == 200


def test_option_round_trip_completed_while_offline_uses_contract_multiplier():
    contract = "AAPL261120C00250000"
    option_entry = entry()
    option_entry.update({
        "id": "option-entry", "client_order_id": "botty-opt-20260908-aapl-momentum",
        "symbol": contract, "qty": 1, "filled_qty": 1, "filled_avg_price": 5.0,
        "legs": [],
    })
    option_exit = {
        **option_entry,
        "id": "option-exit", "client_order_id": "botty-opt-exit-20260920-1",
        "side": "sell", "filled_avg_price": 7.0,
        "created_at": datetime(2026, 9, 20, 14, 0, tzinfo=timezone.utc),
        "filled_at": datetime(2026, 9, 20, 14, 1, tzinfo=timezone.utc),
    }
    db = Database(":memory:")
    reconciler = BrokerReconciler(
        FakeBroker([option_entry, option_exit], []), db, Recorder()
    )
    report = reconciler.reconcile()
    assert report.trades_opened == 1
    assert report.trades_closed == 1
    assert db.trades.history()[0]["pnl"] == 200
