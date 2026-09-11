from datetime import datetime, timedelta, timezone
from decimal import Decimal

from trading_bot.automation.execution import PaperExecutor
from trading_bot.config.settings import Settings
from trading_bot.data.database import Database
from trading_bot.execution.broker import BrokerError
from trading_bot.risk import RiskDecision
from trading_bot.scanner.scanner import Opportunity
from trading_bot.strategies import Signal, SignalDirection

NOW = datetime(2026, 9, 8, 14, 0, tzinfo=timezone.utc)


class Recorder:
    def __init__(self):
        self.messages = []

    def send(self, message):
        self.messages.append(message)


class FakeBroker:
    is_paper = True

    def __init__(self, *, held=(), fail=False):
        self.held = set(held)
        self.fail = fail
        self.submitted = []

    def get_positions(self):
        return [{"symbol": symbol} for symbol in self.held]

    def submit_bracket_order(self, **order):
        if self.fail:
            raise BrokerError("simulated rejection")
        self.submitted.append(order)
        return {
            "id": f"order-{len(self.submitted)}",
            "client_order_id": order["client_order_id"],
            "status": "accepted",
        }


def opportunity(symbol="AAPL", *, timestamp=NOW):
    signal = Signal(
        symbol=symbol, direction=SignalDirection.LONG, strategy="momentum",
        confidence=80, entry_price=100, stop_loss=95, take_profit=110,
        timestamp=timestamp,
    )
    decision = RiskDecision(approved=True, signal=signal, quantity=Decimal("5"))
    return Opportunity(signal=signal, confidence=80, factors=(), decision=decision, rank=1)


def test_approved_fresh_opportunity_becomes_audited_bracket_order():
    db, broker, notices = Database(":memory:"), FakeBroker(), Recorder()
    db.initialize()
    report = PaperExecutor(
        broker, db, notices, Settings(), now=lambda: NOW
    ).execute((opportunity(),), 1)
    assert report.placed == 1
    assert broker.submitted[0]["stop_loss"] == 95
    assert len(db.signals.recent()) == 1
    assert len(db.orders.open_orders()) == 1
    assert any("submitted" in message for message in notices.messages)


def test_stale_signal_is_blocked_before_broker_call():
    db, broker = Database(":memory:"), FakeBroker()
    db.initialize()
    stale = opportunity(timestamp=NOW - timedelta(hours=121))
    report = PaperExecutor(broker, db, Recorder(), Settings(), now=lambda: NOW).execute(
        (stale,), 1
    )
    assert report.stale_blocked == 1
    assert not broker.submitted


def test_existing_position_is_not_duplicated():
    db, broker = Database(":memory:"), FakeBroker(held={"AAPL"})
    db.initialize()
    report = PaperExecutor(broker, db, Recorder(), Settings(), now=lambda: NOW).execute(
        (opportunity(),), 1
    )
    assert report.skipped_held == 1
    assert not broker.submitted


def test_rejection_burst_opens_circuit_and_stops_remaining_orders():
    db, broker = Database(":memory:"), FakeBroker(fail=True)
    db.initialize()
    report = PaperExecutor(broker, db, Recorder(), Settings(), now=lambda: NOW).execute(
        (opportunity("AAA"), opportunity("BBB"), opportunity("CCC")), 3
    )
    assert report.failed == 2
    assert report.circuit_opened
    assert len(db.events.recent(category="execution_circuit_breaker")) == 1


def test_live_broker_is_structurally_refused():
    broker = FakeBroker()
    broker.is_paper = False
    db = Database(":memory:")
    db.initialize()
    try:
        PaperExecutor(broker, db, Recorder(), Settings())
    except BrokerError as error:
        assert "paper-only" in str(error)
    else:
        raise AssertionError("live executor was accepted")
