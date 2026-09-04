"""End-to-end wiring: provider -> cache -> normalization -> database.

These exercise the layers together, which is where interface mismatches show up
that per-module tests miss.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tests.conftest import make_bars
from tests.test_market_data import FakeStockClient
from trading_bot.data.cache import BarCache
from trading_bot.data.database import Database
from trading_bot.data.market_data import AlpacaMarketData, drop_incomplete_bars
from trading_bot.utils.timeframes import Timeframe


def test_second_fetch_is_served_from_cache(settings, tmp_path):
    """The cache must actually prevent a second network call."""
    frame = make_bars(200)
    client = FakeStockClient({"AAPL": frame})
    cache = BarCache(tmp_path / "cache")
    provider = AlpacaMarketData(settings.alpaca, settings.data, client=client, cache=cache)

    start, end = frame.index[0], frame.index[-1]
    first = provider.get_bars("AAPL", "15Min", start=start, end=end)
    assert client.call_count == 1

    second = provider.get_bars("AAPL", "15Min", start=start, end=end)
    assert client.call_count == 1          # no second request
    assert len(second) == len(first)
    assert second["close"].iloc[-1] == pytest.approx(first["close"].iloc[-1])


def test_cache_survives_a_new_provider_instance(settings, tmp_path):
    """Cached bars persist across process restarts."""
    frame = make_bars(100)
    cache_dir = tmp_path / "cache"
    start, end = frame.index[0], frame.index[-1]

    first_client = FakeStockClient({"AAPL": frame})
    AlpacaMarketData(
        settings.alpaca, settings.data, client=first_client, cache=BarCache(cache_dir)
    ).get_bars("AAPL", "15Min", start=start, end=end)

    second_client = FakeStockClient({"AAPL": frame})
    result = AlpacaMarketData(
        settings.alpaca, settings.data, client=second_client, cache=BarCache(cache_dir)
    ).get_bars("AAPL", "15Min", start=start, end=end)

    assert second_client.call_count == 0
    assert len(result) == 100


def test_watchlist_fetch_feeds_the_lookahead_guard(settings):
    """The realistic pipeline: fetch a watchlist, then strip forming bars."""
    now = datetime(2024, 1, 3, 12, 7, tzinfo=timezone.utc)
    frames = {
        "AAPL": make_bars(200, start=datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)),
        "MSFT": make_bars(200, start=datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc), seed=11),
    }
    provider = AlpacaMarketData(
        settings.alpaca, settings.data, client=FakeStockClient(frames)
    )
    timeframe = Timeframe.parse("15Min")

    fetched, report = provider.fetch_watchlist(
        ["AAPL", "MSFT"], timeframe, lookback_bars=50, end=now
    )
    assert report.success_count == 2

    for symbol, frame in fetched.items():
        safe = drop_incomplete_bars(frame, timeframe, now=now)
        assert safe.index[-1] <= now - timeframe.duration, f"{symbol} exposed a forming bar"


def test_signal_to_closed_trade_round_trip(tmp_path):
    """A signal becomes a trade, and the trade's P&L reaches the statistics."""
    database = Database(tmp_path / "bot.db")
    database.initialize()

    run_id = database.runs.start(mode="paper", strategy="Momentum", starting_equity=10_000)
    signal_id = database.signals.record(
        run_id=run_id, symbol="AAPL", strategy="Momentum", direction="LONG",
        confidence=82, entry_price=210.50, stop_loss=207.00, take_profit=218.00,
        risk_reward=2.14, reasons=["Bullish EMA crossover"], accepted=True,
    )
    trade_id = database.trades.open_trade(
        run_id=run_id, signal_id=signal_id, symbol="AAPL", strategy="Momentum",
        direction="LONG", qty=25, entry_price=210.50, stop_loss=207.00,
        take_profit=218.00,
    )
    database.positions.upsert(
        symbol="AAPL", run_id=run_id, trade_id=trade_id, direction="LONG",
        qty=25, avg_entry_price=210.50, stop_loss=207.00, take_profit=218.00,
    )
    assert database.positions.count() == 1

    closed = database.trades.close_trade(
        trade_id, exit_price=218.00, exit_reason="take_profit"
    )
    database.positions.remove("AAPL")
    database.runs.finish(run_id, ending_equity=10_187.50)

    assert closed["pnl"] == pytest.approx(187.50)     # 25 shares * $7.50
    assert closed["r_multiple"] == pytest.approx(2.142857, rel=1e-4)

    stats = database.trades.statistics(run_id=run_id)
    assert stats["total_trades"] == 1
    assert stats["win_rate"] == 100.0
    assert database.positions.count() == 0
    database.close()


def test_risk_guards_read_from_recorded_trades(tmp_path):
    """The data the Phase 4 risk manager will depend on is queryable today."""
    database = Database(tmp_path / "bot.db")
    database.initialize()

    for exit_price in (98.0, 98.0, 98.0):      # three consecutive stop-outs
        trade_id = database.trades.open_trade(
            symbol="AAPL", direction="LONG", qty=10, entry_price=100.0, stop_loss=98.0
        )
        database.trades.close_trade(trade_id, exit_price=exit_price, exit_reason="stop_loss")

    assert database.trades.consecutive_losses() == 3    # would trigger a cooldown
    today = datetime.now(timezone.utc) - timedelta(hours=1)
    assert database.trades.realized_pnl_since(today) == pytest.approx(-60.0)
    database.close()
