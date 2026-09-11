from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

from trading_bot.automation.options_execution import OptionPaperExecutor, OptionPositionSupervisor
from trading_bot.automation.tracking import PriceTracker
from trading_bot.options import OptionQuote


class Notifier:
    def __init__(self):
        self.messages = []

    def send(self, message):
        self.messages.append(message)


class Chain:
    def __init__(self, *, mid=5.2, bid=5.1, ask=5.3):
        self.mid, self.bid, self.ask = mid, bid, ask

    def quotes(self, underlying, direction, underlying_price):
        kind = "call" if direction == "LONG" else "put"
        return [OptionQuote(
            symbol=f"{underlying}261120C00250000", underlying=underlying,
            contract_type=kind, expiration=date.today() + timedelta(days=75),
            strike=underlying_price, bid=5.0, ask=5.2, delta=0.60,
            daily_volume=100, open_interest=500,
        )]

    def latest_mid(self, symbols):
        return {symbol: (self.mid, self.bid, self.ask) for symbol in symbols}


class Broker:
    is_paper = True

    def __init__(self):
        self.submissions = []

    def submit_option_order(self, **kwargs):
        self.submissions.append(kwargs)
        return {
            "id": f"order-{len(self.submissions)}", "client_order_id": kwargs["client_order_id"],
            "status": "accepted",
        }

    def get_positions(self):
        return []

    def get_orders(self, **kwargs):
        return []


def opportunity(*, approved=True, confidence=95):
    signal = SimpleNamespace(
        symbol="AAPL", strategy="momentum", direction=SimpleNamespace(value="LONG"),
        entry_price=250.0, stop_loss=240.0, take_profit=270.0,
        risk_reward_ratio=2.0, reasons=("strong trend",),
        timestamp=datetime.now(timezone.utc),
    )
    return SimpleNamespace(
        signal=signal, confidence=confidence,
        decision=SimpleNamespace(approved=approved),
    )


def test_option_executor_only_places_risk_approved_paper_swing(database, settings):
    broker = Broker()
    notifier = Notifier()
    settings = settings.model_copy(update={
        "options": settings.options.model_copy(update={"alert_only": False})
    })
    report = OptionPaperExecutor(
        broker, Chain(), database, notifier, settings
    ).execute([opportunity()], capacity=1)
    assert report.placed == 1
    assert broker.submissions[0]["side"] == "buy"
    selected = database.option_selections.active()[0]
    assert 500 <= selected["estimated_cost"] <= 1000
    assert "CALL $250.00" in notifier.messages[0]
    assert str(date.today() + timedelta(days=75)) in notifier.messages[0]


def test_option_executor_alert_only_sends_contract_without_order(database, settings):
    broker = Broker()
    notifier = Notifier()
    report = OptionPaperExecutor(
        broker, Chain(), database, notifier, settings
    ).execute([opportunity()], capacity=1)

    assert report.alerted == 1
    assert report.placed == 0
    assert broker.submissions == []
    assert "BOTTY SWING ALERT: AAPL LONG" in notifier.messages[0]
    assert "CALL $250.00" in notifier.messages[0]
    assert "ALERT ONLY — NO ORDER SUBMITTED" in notifier.messages[0]
    assert database.option_selections.recent()[0]["status"] == "alerted"


def test_option_executor_rejects_unapproved_setup(database, settings):
    broker = Broker()
    report = OptionPaperExecutor(
        broker, Chain(), database, Notifier(), settings
    ).execute([opportunity(approved=False)], capacity=1)
    assert report.skipped == 1
    assert broker.submissions == []


def _selected_position(database, *, opened_at):
    symbol = "AAPL261120C00250000"
    selection_id = database.option_selections.record(
        selected_at=opened_at - timedelta(days=1), underlying_symbol="AAPL",
        contract_symbol=symbol, contract_type="call",
        expiration=(opened_at.date() + timedelta(days=75)), strike=250,
        bid=5.0, ask=5.2, delta=0.6, daily_volume=100, open_interest=500,
        quantity=1, estimated_cost=520, status="open",
    )
    database.positions.upsert(
        symbol=symbol, direction="LONG", qty=1, avg_entry_price=5.2,
        opened_at=opened_at,
    )
    return selection_id, symbol


def test_option_supervisor_structurally_blocks_same_day_exit(database, settings):
    now = datetime(2026, 9, 8, 18, tzinfo=timezone.utc)
    _selection_id, symbol = _selected_position(database, opened_at=now)
    broker = Broker()
    broker.get_positions = lambda: [{"symbol": symbol, "qty": 1, "avg_entry_price": 5.2}]
    supervisor = OptionPositionSupervisor(
        broker, Chain(mid=2.0, bid=1.9, ask=2.1), database, Notifier(), settings,
        now=lambda: now,
    )
    assert supervisor.supervise() == 0
    assert broker.submissions == []


def test_option_supervisor_exits_after_entry_day_when_stop_hits(database, settings):
    opened = datetime(2026, 9, 7, 18, tzinfo=timezone.utc)
    selection_id, symbol = _selected_position(database, opened_at=opened)
    broker = Broker()
    broker.get_positions = lambda: [{"symbol": symbol, "qty": 1, "avg_entry_price": 5.2}]
    supervisor = OptionPositionSupervisor(
        broker, Chain(mid=2.0, bid=1.9, ask=2.1), database, Notifier(), settings,
        now=lambda: opened + timedelta(days=1),
    )
    assert supervisor.supervise() == 1
    assert broker.submissions[0]["side"] == "sell"
    assert next(row for row in database.option_selections.active() if row["id"] == selection_id)


def test_tracker_records_underlying_and_contract(database):
    now = datetime.now(timezone.utc)
    _selection_id, symbol = _selected_position(database, opened_at=now)

    class Stocks:
        def get_latest_price(self, underlying):
            assert underlying == "AAPL"
            return 251.0

    assert PriceTracker(Stocks(), Chain(), database).capture() == 2
    assert database.prices.history("AAPL")[-1]["price"] == 251.0
    assert database.prices.history(symbol)[-1]["asset_kind"] == "option"
