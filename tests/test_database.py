"""Database schema, trade lifecycle arithmetic and risk-relevant queries."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from trading_bot.data.database import SCHEMA_VERSION, Database


def test_initialize_creates_every_table(database):
    tables = {
        row["name"]
        for row in database.query("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {
        "runs", "signals", "orders", "trades", "positions", "equity_snapshots", "bot_events"
    } <= tables


def test_initialize_is_idempotent(database):
    assert database.initialize() == SCHEMA_VERSION
    assert database.initialize() == SCHEMA_VERSION


def test_migration_from_a_fresh_file(tmp_path):
    db = Database(tmp_path / "nested" / "bot.db")
    assert db.initialize() == SCHEMA_VERSION
    assert (tmp_path / "nested" / "bot.db").exists()
    db.close()


# -- signals ---------------------------------------------------------------------


def test_rejected_signals_are_recorded_with_their_reason(database):
    database.signals.record(
        symbol="aapl", strategy="Momentum", direction="long", confidence=55,
        accepted=False, rejection_reason="confidence below threshold",
    )
    signal = database.signals.recent()[0]
    assert signal["symbol"] == "AAPL"
    assert signal["direction"] == "LONG"
    assert signal["accepted"] == 0
    assert signal["rejection_reason"] == "confidence below threshold"


def test_signal_reasons_round_trip_as_a_list(database):
    database.signals.record(
        symbol="NVDA", strategy="Momentum", direction="LONG", confidence=82,
        reasons=["EMA crossover", "MACD bullish"], accepted=True,
    )
    assert database.signals.recent()[0]["reasons"] == ["EMA crossover", "MACD bullish"]


# -- trades ----------------------------------------------------------------------


def test_long_trade_pnl_is_net_of_fees(database):
    trade_id = database.trades.open_trade(
        symbol="AAPL", direction="LONG", qty=10, entry_price=100.0, stop_loss=98.0, fees=1.0
    )
    closed = database.trades.close_trade(trade_id, exit_price=110.0, fees=1.0)
    assert closed["gross_pnl"] == pytest.approx(100.0)
    assert closed["pnl"] == pytest.approx(98.0)          # 100 gross - 2 total fees
    assert closed["pnl_pct"] == pytest.approx(9.8)       # 98 / 1000
    assert closed["status"] == "closed"


def test_short_trade_profits_when_price_falls(database):
    trade_id = database.trades.open_trade(
        symbol="TSLA", direction="SHORT", qty=5, entry_price=200.0, stop_loss=210.0
    )
    closed = database.trades.close_trade(trade_id, exit_price=180.0)
    assert closed["pnl"] == pytest.approx(100.0)
    assert closed["r_multiple"] == pytest.approx(2.0)    # 20 gained / 10 risked


def test_r_multiple_is_negative_on_a_stop_out(database):
    trade_id = database.trades.open_trade(
        symbol="AMD", direction="LONG", qty=10, entry_price=100.0, stop_loss=98.0
    )
    closed = database.trades.close_trade(trade_id, exit_price=98.0, exit_reason="stop_loss")
    assert closed["r_multiple"] == pytest.approx(-1.0)
    assert closed["exit_reason"] == "stop_loss"


def test_closing_an_unknown_trade_returns_none(database):
    assert database.trades.close_trade(999, exit_price=10.0) is None


def test_open_trades_excludes_closed_ones(database):
    open_id = database.trades.open_trade(
        symbol="AAPL", direction="LONG", qty=1, entry_price=100.0
    )
    closed_id = database.trades.open_trade(
        symbol="MSFT", direction="LONG", qty=1, entry_price=100.0
    )
    database.trades.close_trade(closed_id, exit_price=105.0)
    assert [t["id"] for t in database.trades.open_trades()] == [open_id]


# -- statistics ------------------------------------------------------------------


def _closed_trade(database, pnl_target: float, symbol: str = "AAPL") -> None:
    trade_id = database.trades.open_trade(
        symbol=symbol, direction="LONG", qty=1, entry_price=100.0, stop_loss=99.0
    )
    database.trades.close_trade(trade_id, exit_price=100.0 + pnl_target)


def test_statistics_aggregate_wins_losses_and_profit_factor(database):
    for pnl in (10.0, 20.0, -5.0, -5.0):
        _closed_trade(database, pnl)
    stats = database.trades.statistics()
    assert stats["total_trades"] == 4
    assert stats["wins"] == 2
    assert stats["losses"] == 2
    assert stats["win_rate"] == pytest.approx(50.0)
    assert stats["total_pnl"] == pytest.approx(20.0)
    assert stats["avg_win"] == pytest.approx(15.0)
    assert stats["avg_loss"] == pytest.approx(-5.0)
    assert stats["profit_factor"] == pytest.approx(3.0)  # 30 profit / 10 loss


def test_statistics_on_an_empty_database(database):
    stats = database.trades.statistics()
    assert stats["total_trades"] == 0
    assert stats["win_rate"] == 0.0


def test_consecutive_losses_counts_the_current_streak(database):
    for pnl in (10.0, -5.0, -5.0, -5.0):
        _closed_trade(database, pnl)
    assert database.trades.consecutive_losses() == 3


def test_consecutive_losses_resets_after_a_win(database):
    for pnl in (-5.0, -5.0, 10.0):
        _closed_trade(database, pnl)
    assert database.trades.consecutive_losses() == 0


def test_realized_pnl_since_ignores_older_trades(database):
    _closed_trade(database, 50.0)
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    assert database.trades.realized_pnl_since(future) == 0.0
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    assert database.trades.realized_pnl_since(past) == pytest.approx(50.0)


# -- positions -------------------------------------------------------------------


def test_position_upsert_updates_in_place(database):
    database.positions.upsert(
        symbol="AAPL", direction="LONG", qty=10, avg_entry_price=100.0, stop_loss=98.0
    )
    database.positions.upsert(
        symbol="AAPL", direction="LONG", qty=15, avg_entry_price=101.0, stop_loss=99.0
    )
    assert database.positions.count() == 1
    position = database.positions.get("AAPL")
    assert position["qty"] == 15
    assert position["stop_loss"] == 99.0


def test_position_removal(database):
    database.positions.upsert(symbol="AAPL", direction="LONG", qty=1, avg_entry_price=1.0)
    database.positions.remove("aapl")
    assert database.positions.count() == 0


# -- equity and events -----------------------------------------------------------


def test_equity_curve_is_returned_in_chronological_order(database):
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for offset, equity in enumerate([10_000, 10_500, 10_200]):
        database.equity.record(equity=equity, ts=base + timedelta(days=offset))
    curve = database.equity.curve()
    assert [row["equity"] for row in curve] == [10_000, 10_500, 10_200]
    assert database.equity.latest()["equity"] == 10_200


def test_events_filter_by_category(database):
    database.events.record(category="risk", message="rejected", symbol="AAPL")
    database.events.record(category="order", message="submitted", symbol="MSFT")
    events = database.events.recent(category="risk")
    assert len(events) == 1
    assert events[0]["symbol"] == "AAPL"


def test_event_payload_round_trips(database):
    database.events.record(category="risk", message="sized", payload={"qty": 25, "risk": 100.0})
    assert database.events.recent()[0]["payload"] == {"qty": 25, "risk": 100.0}


def test_run_lifecycle(database):
    run_id = database.runs.start(
        mode="paper", strategy="Momentum", symbols=["AAPL"], starting_equity=10_000
    )
    database.runs.finish(run_id, ending_equity=10_450)
    run = database.runs.latest()
    assert run["id"] == run_id
    assert run["ended_at"] is not None
    assert run["ending_equity"] == 10_450


def test_transaction_rolls_back_on_error(database):
    database.positions.upsert(symbol="AAPL", direction="LONG", qty=1, avg_entry_price=1.0)
    with pytest.raises(RuntimeError), database.transaction() as connection:
        connection.execute("DELETE FROM positions")
        raise RuntimeError("boom")
    assert database.positions.count() == 1
