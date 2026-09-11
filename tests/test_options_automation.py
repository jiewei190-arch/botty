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
    def __init__(self, *, mid=5.2, bid=5.1, ask=5.3, empty_for=()):
        self.mid, self.bid, self.ask = mid, bid, ask
        self.empty_for = set(empty_for)

    def quotes(self, underlying, direction, underlying_price):
        if underlying in self.empty_for:
            return []
        kind = "call" if direction == "LONG" else "put"
        return [OptionQuote(
            symbol=f"{underlying}261120C00250000", underlying=underlying,
            contract_type=kind, expiration=date.today() + timedelta(days=60),
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


def opportunity(symbol="AAPL", *, approved=True, confidence=95):
    signal = SimpleNamespace(
        symbol=symbol, strategy="momentum", direction=SimpleNamespace(value="LONG"),
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
    # Pin the contract's content, not the sentence it sits in: ticker,
    # strike, right, expiry and DTE are what the reader acts on.
    message = notifier.messages[0]
    assert "AAPL $250 CALL" in message
    assert "60 DTE" in message
    assert str(date.today() + timedelta(days=60)) in message


def test_option_executor_alert_only_sends_contract_without_order(database, settings):
    broker = Broker()
    notifier = Notifier()
    report = OptionPaperExecutor(
        broker, Chain(), database, notifier, settings
    ).execute([opportunity()], capacity=1)

    assert report.alerted == 1
    assert report.placed == 0
    assert broker.submissions == []
    message = notifier.messages[0]
    assert "BOTTY SWING ALERT" in message
    assert "AAPL LONG" in message
    assert "AAPL $250 CALL" in message
    assert "60 DTE" in message
    assert str(date.today() + timedelta(days=60)) in message
    assert "ALERT ONLY — NO ORDER SUBMITTED" in message
    assert database.option_selections.recent()[0]["status"] == "alerted"


def test_option_executor_keeps_searching_until_capacity_is_filled(database, settings):
    broker = Broker()
    notifier = Notifier()
    report = OptionPaperExecutor(
        broker, Chain(empty_for={"VG"}), database, notifier, settings
    ).execute([opportunity("VG"), opportunity("AAPL")], capacity=1)

    assert report.alerted == 1
    assert report.skipped == 1
    assert broker.submissions == []
    message = notifier.messages[0]
    assert "BOTTY SWING ALERT" in message
    assert "AAPL LONG" in message
    assert "AAPL $250 CALL" in message
    assert "60 DTE" in message
    assert database.option_selections.recent()[0]["underlying_symbol"] == "AAPL"


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


# ---------------------------------------------------------------------------
# Every ranked setup gets named
#
# The caps below exist to protect capital, so they gate orders. They used to
# gate alerts too, which meant a manually-traded account saw the first couple
# of names in a scan and never learned the rest existed. These pin the split.
# ---------------------------------------------------------------------------


def _chain_for(*symbols):
    """A chain that answers for every symbol, not just AAPL."""
    class MultiChain(Chain):
        def quotes(self, underlying, direction, underlying_price):
            kind = "call" if direction == "LONG" else "put"
            return [OptionQuote(
                symbol=f"{underlying}261120C00250000", underlying=underlying,
                contract_type=kind, expiration=date.today() + timedelta(days=60),
                strike=underlying_price, bid=5.0, ask=5.2, delta=0.60,
                daily_volume=100, open_interest=500,
            )]
    return MultiChain()


def test_setups_beyond_order_capacity_are_still_alerted(database, settings):
    """Capacity of one used to mean one alert and silence for the rest."""
    notifier = Notifier()
    report = OptionPaperExecutor(
        Broker(), _chain_for(), database, notifier, settings
    ).execute(
        [opportunity("AAPL"), opportunity("NVDA"), opportunity("NET")],
        capacity=1,
    )

    assert report.alerted == 3
    assert len(notifier.messages) == 3
    named = " ".join(notifier.messages)
    for symbol in ("AAPL", "NVDA", "NET"):
        assert f"{symbol} $250 CALL" in named


def test_a_risk_declined_setup_is_alerted_with_the_reason(database, settings):
    """Botty declining to buy it is not a reason to keep it from the reader."""
    notifier = Notifier()
    report = OptionPaperExecutor(
        Broker(), _chain_for(), database, notifier, settings
    ).execute([opportunity("NET", approved=False)], capacity=1)

    assert report.alerted == 1
    message = notifier.messages[0]
    assert "NET $250 CALL" in message
    assert "60 DTE" in message
    assert "NO ORDER" in message
    assert report.placed == 0


def test_a_full_account_still_alerts_every_candidate(database, settings):
    """The early return sent zero alerts for the whole scan."""
    database.option_selections.record(
        signal_id=None, underlying_symbol="TSLA", contract_symbol="TSLA1",
        contract_type="call", expiration=date.today() + timedelta(days=60),
        strike=250.0, bid=5.0, ask=5.2, delta=0.6, implied_volatility=0.3,
        daily_volume=10, open_interest=500, quantity=1, estimated_cost=520.0,
    )
    database.option_selections.record(
        signal_id=None, underlying_symbol="MSFT", contract_symbol="MSFT1",
        contract_type="call", expiration=date.today() + timedelta(days=60),
        strike=250.0, bid=5.0, ask=5.2, delta=0.6, implied_volatility=0.3,
        daily_volume=10, open_interest=500, quantity=1, estimated_cost=520.0,
    )
    notifier = Notifier()

    report = OptionPaperExecutor(
        Broker(), _chain_for(), database, notifier, settings
    ).execute([opportunity("AAPL"), opportunity("NVDA")], capacity=2)

    assert report.alerted == 2
    assert report.placed == 0
    assert all("NO ORDER" in m for m in notifier.messages)


def test_an_underlying_already_held_is_not_alerted_twice(database, settings):
    """The one silence kept on purpose: a duplicate is not news."""
    database.option_selections.record(
        signal_id=None, underlying_symbol="AAPL", contract_symbol="AAPL1",
        contract_type="call", expiration=date.today() + timedelta(days=60),
        strike=250.0, bid=5.0, ask=5.2, delta=0.6, implied_volatility=0.3,
        daily_volume=10, open_interest=500, quantity=1, estimated_cost=520.0,
    )
    notifier = Notifier()

    report = OptionPaperExecutor(
        Broker(), _chain_for(), database, notifier, settings
    ).execute([opportunity("AAPL")], capacity=2)

    assert notifier.messages == []
    assert report.alerted == 0
    assert report.skipped == 1


def test_the_per_scan_alert_budget_bounds_api_lookups(database, settings):
    """Alerts are bounded by a budget, not by the money caps."""
    settings = settings.model_copy(update={
        "options": settings.options.model_copy(update={"max_alerts_per_scan": 2})
    })
    notifier = Notifier()

    report = OptionPaperExecutor(
        Broker(), _chain_for(), database, notifier, settings
    ).execute(
        [opportunity("AAPL"), opportunity("NVDA"), opportunity("NET")],
        capacity=5,
    )

    assert len(notifier.messages) == 2
    assert report.alerted == 2


def test_an_alert_names_strike_dte_and_right(database, settings):
    """Exactly what was asked for: ticker, strike price, DTE, call or put."""
    notifier = Notifier()
    OptionPaperExecutor(
        Broker(), _chain_for(), database, notifier, settings
    ).execute([opportunity("NET")], capacity=1)

    message = notifier.messages[0]
    assert "NET" in message                                   # ticker
    assert "$250" in message                                  # strike
    assert "CALL" in message                                  # right
    assert "60 DTE" in message                                # DTE
    assert str(date.today() + timedelta(days=60)) in message  # expiry


def test_a_put_setup_says_put(database, settings):
    notifier = Notifier()
    bearish = opportunity("NET")
    bearish.signal.direction = SimpleNamespace(value="SHORT")
    OptionPaperExecutor(
        Broker(), _chain_for(), database, notifier, settings
    ).execute([bearish], capacity=1)

    assert "NET $250 PUT" in notifier.messages[0]


def test_an_alert_only_selection_does_not_consume_account_capacity(database, settings):
    """An alert spends no money, so it must not shrink the next one's budget."""
    notifier = Notifier()
    OptionPaperExecutor(
        Broker(), _chain_for(), database, notifier, settings
    ).execute(
        [opportunity("AAPL"), opportunity("NVDA"), opportunity("NET")],
        capacity=1,
    )

    costs = [row["estimated_cost"] for row in database.option_selections.recent()]
    assert len(costs) == 3
    assert all(500 <= cost <= 1000 for cost in costs)
