"""Broker adapter, with emphasis on the paper/live safety guarantee."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from trading_bot.config.settings import LIVE_CONFIRMATION_PHRASE, Settings, TradingMode
from trading_bot.execution.broker import AlpacaBroker, BrokerError, build_broker


class FakeTradingClient:
    def __init__(self, *, equity="10000.00", last_equity="9800.00", blocked=False):
        self._account = SimpleNamespace(
            id="abcd1234-5678", status="ACTIVE", currency="USD",
            equity=equity, cash="5000.00", buying_power="20000.00",
            portfolio_value=equity, last_equity=last_equity,
            pattern_day_trader=False, trading_blocked=blocked,
            transfers_blocked=False, account_blocked=False, daytrade_count=0,
        )
        self.calls: list[str] = []

    def get_account(self):
        self.calls.append("get_account")
        return self._account

    def get_clock(self):
        self.calls.append("get_clock")
        return SimpleNamespace(timestamp="2024-01-02T15:00:00Z", is_open=True,
                               next_open=None, next_close="2024-01-02T21:00:00Z")

    def get_asset(self, symbol):
        self.calls.append(f"get_asset:{symbol}")
        if symbol == "BADSYM":
            raise ValueError("asset not found")
        return SimpleNamespace(
            symbol=symbol, name=f"{symbol} Inc", exchange="NASDAQ",
            tradable=symbol != "NOTRADE", shortable=True, fractionable=True,
            marginable=True, status="active",
        )

    def get_all_positions(self):
        self.calls.append("get_all_positions")
        return [
            SimpleNamespace(
                symbol="AAPL", qty="10", side=SimpleNamespace(value="long"),
                avg_entry_price="100.0", current_price="110.0", market_value="1100.0",
                cost_basis="1000.0", unrealized_pl="100.0", unrealized_plpc="0.10",
            )
        ]


@pytest.fixture
def paper_settings() -> Settings:
    base = Settings()
    return base.with_overrides(
        alpaca=base.alpaca.model_copy(update={"api_key": "k", "secret_key": "s"})
    )


def test_account_snapshot_parses_money_as_decimal(paper_settings):
    broker = AlpacaBroker(paper_settings, client=FakeTradingClient())
    account = broker.get_account()
    assert account.equity == Decimal("10000.00")
    assert account.daily_pnl == Decimal("200.00")
    assert account.daily_pnl_pct == pytest.approx(2.0408, rel=1e-3)
    assert account.is_paper
    assert account.can_trade


def test_blocked_account_cannot_trade(paper_settings):
    broker = AlpacaBroker(paper_settings, client=FakeTradingClient(blocked=True))
    assert not broker.get_account().can_trade


def test_unparseable_money_defaults_to_zero(paper_settings):
    broker = AlpacaBroker(paper_settings, client=FakeTradingClient(equity="n/a"))
    assert broker.get_account().equity == Decimal("0")


def test_clock_describes_the_session(paper_settings):
    clock = AlpacaBroker(paper_settings, client=FakeTradingClient()).get_clock()
    assert clock.is_open
    assert "OPEN" in clock.describe()


def test_validate_symbols_splits_tradable_from_rejected(paper_settings):
    broker = AlpacaBroker(paper_settings, client=FakeTradingClient())
    tradable, rejected = broker.validate_symbols(["AAPL", "NOTRADE", "BADSYM"])
    assert tradable == ["AAPL"]
    assert set(rejected) == {"NOTRADE", "BADSYM"}


def test_positions_are_returned_as_plain_dicts(paper_settings):
    positions = AlpacaBroker(paper_settings, client=FakeTradingClient()).get_positions()
    assert positions[0]["symbol"] == "AAPL"
    assert positions[0]["unrealized_plpc"] == pytest.approx(10.0)


def test_ping_reports_failure_without_raising(paper_settings):
    class Broken(FakeTradingClient):
        def get_account(self):
            raise RuntimeError("401 unauthorized")

    assert not AlpacaBroker(paper_settings, client=Broken()).ping()


def test_missing_credentials_produce_an_actionable_error():
    with pytest.raises(BrokerError, match="ALPACA_API_KEY"):
        AlpacaBroker(Settings())


def test_backtest_mode_refuses_to_build_a_broker():
    settings = Settings(trading_mode=TradingMode.BACKTEST)
    with pytest.raises(BrokerError, match="BACKTEST mode"):
        build_broker(settings)


# -- the safety guarantee --------------------------------------------------------


def test_paper_mode_constructs_a_paper_client(monkeypatch, paper_settings):
    """Default configuration must never reach the live endpoint."""
    captured: dict[str, object] = {}

    def spy(**kwargs):
        captured.update(kwargs)
        return FakeTradingClient()

    monkeypatch.setattr("trading_bot.execution.broker.TradingClient", spy)
    broker = AlpacaBroker(paper_settings)
    assert captured["paper"] is True
    assert broker.is_paper


def test_live_client_only_built_when_both_locks_pass(monkeypatch):
    captured: dict[str, object] = {}

    def spy(**kwargs):
        captured.update(kwargs)
        return FakeTradingClient()

    monkeypatch.setattr("trading_bot.execution.broker.TradingClient", spy)
    live = Settings(
        trading_mode=TradingMode.LIVE,
        enable_live_trading=True,
        live_trading_confirmation=LIVE_CONFIRMATION_PHRASE,
        alpaca=Settings().alpaca.model_copy(update={"api_key": "k", "secret_key": "s"}),
    )
    broker = AlpacaBroker(live)
    assert captured["paper"] is False
    assert not broker.is_paper
