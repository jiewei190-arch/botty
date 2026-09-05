"""Market data normalization, the lookahead guard, and the Alpaca provider."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from tests.conftest import make_alpaca_frame, make_bars
from trading_bot.data.market_data import (
    AlpacaMarketData,
    MarketDataError,
    StaticMarketData,
    drop_incomplete_bars,
    ensure_utc,
    normalize_bars,
)
from trading_bot.utils.timeframes import Timeframe


def _utc(moment) -> pd.Timestamp | None:
    """Interpret a request timestamp as UTC.

    ``StockBarsRequest`` strips tzinfo and stores naive UTC internally, so a
    naive value here means UTC — not local time.
    """
    if moment is None:
        return None
    stamp = pd.Timestamp(moment)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


class FakeBarSet:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.df = frame


class FakeStockClient:
    """Stand-in for ``StockHistoricalDataClient`` that records its requests."""

    def __init__(self, frames: dict[str, pd.DataFrame], *, fail_times: int = 0) -> None:
        self.frames = frames
        self.requests: list[object] = []
        self.fail_times = fail_times
        self.call_count = 0

    def get_stock_bars(self, request):
        self.call_count += 1
        self.requests.append(request)
        if self.call_count <= self.fail_times:
            raise ConnectionError("simulated transient failure")
        symbols = request.symbol_or_symbols
        if isinstance(symbols, str):
            symbols = [symbols]
        available = {}
        for symbol in symbols:
            if symbol not in self.frames:
                continue
            frame = self.frames[symbol]
            # The real API honours the requested window; so must the fake.
            start, end = _utc(request.start), _utc(request.end)
            if start is not None:
                frame = frame[frame.index >= start]
            if end is not None:
                frame = frame[frame.index <= end]
            if not frame.empty:
                available[symbol] = frame
        if not available:
            return FakeBarSet(pd.DataFrame())
        return FakeBarSet(make_alpaca_frame(available))


# -- ensure_utc ------------------------------------------------------------------


def test_ensure_utc_treats_naive_input_as_utc():
    assert ensure_utc(datetime(2024, 1, 2, 15, 0)).tzinfo is timezone.utc


def test_ensure_utc_converts_other_zones():
    eastern = datetime(2024, 1, 2, 10, 30, tzinfo=timezone(timedelta(hours=-5)))
    assert ensure_utc(eastern).hour == 15


# -- normalize_bars --------------------------------------------------------------


def test_normalize_extracts_a_single_symbol_from_multiindex():
    frame = make_alpaca_frame({"AAPL": make_bars(10), "MSFT": make_bars(10, seed=2)})
    result = normalize_bars(frame, symbol="AAPL")
    assert not isinstance(result.index, pd.MultiIndex)
    assert len(result) == 10
    assert result.index.name == "timestamp"


def test_normalize_sorts_and_deduplicates():
    frame = make_bars(20)
    scrambled = pd.concat([frame.iloc[10:], frame.iloc[:12]])  # unsorted, 2 duplicates
    result = normalize_bars(scrambled)
    assert result.index.is_monotonic_increasing
    assert not result.index.has_duplicates
    assert len(result) == 20


def test_normalize_localizes_naive_index_to_utc():
    frame = make_bars(5)
    frame.index = frame.index.tz_localize(None)
    assert normalize_bars(frame).index.tz is not None


def test_normalize_drops_rows_violating_ohlc_invariants():
    frame = make_bars(10)
    frame.iloc[3, frame.columns.get_loc("high")] = frame["low"].iloc[3] - 1  # high < low
    frame.iloc[6, frame.columns.get_loc("close")] = -5.0  # negative price
    result = normalize_bars(frame)
    assert len(result) == 8


def test_normalize_raises_when_required_columns_missing():
    frame = make_bars(5).drop(columns=["volume"])
    with pytest.raises(MarketDataError, match="missing required columns"):
        normalize_bars(frame, symbol="AAPL")


def test_normalize_returns_canonical_empty_frame():
    result = normalize_bars(pd.DataFrame())
    assert result.empty
    assert list(result.columns)[:5] == ["open", "high", "low", "close", "volume"]
    assert result.index.tz is not None


def test_normalize_returns_float_dtypes():
    result = normalize_bars(make_bars(5))
    assert all(str(dtype) == "float64" for dtype in result.dtypes)


# -- lookahead guard -------------------------------------------------------------


def test_drop_incomplete_bars_removes_the_still_forming_bar():
    """A 15-min bar stamped 14:00 is not usable until 14:15."""
    frame = make_bars(4, start=datetime(2024, 1, 2, 13, 30, tzinfo=timezone.utc))
    # bars at 13:30, 13:45, 14:00, 14:15
    now = datetime(2024, 1, 2, 14, 20, tzinfo=timezone.utc)
    result = drop_incomplete_bars(frame, Timeframe.parse("15Min"), now=now)
    assert result.index[-1] == pd.Timestamp("2024-01-02 14:00", tz="UTC")
    assert len(result) == 3


def test_drop_incomplete_bars_keeps_a_bar_that_just_closed():
    frame = make_bars(3, start=datetime(2024, 1, 2, 13, 30, tzinfo=timezone.utc))
    now = datetime(2024, 1, 2, 14, 15, tzinfo=timezone.utc)  # 14:00 bar closed exactly now
    result = drop_incomplete_bars(frame, Timeframe.parse("15Min"), now=now)
    assert result.index[-1] == pd.Timestamp("2024-01-02 14:00", tz="UTC")


def test_drop_incomplete_bars_handles_empty_input():
    empty = normalize_bars(pd.DataFrame())
    assert drop_incomplete_bars(empty, Timeframe.parse("1Day")).empty


# -- AlpacaMarketData ------------------------------------------------------------


def test_provider_requires_credentials(settings):
    stripped = settings.alpaca.model_copy(update={"api_key": None, "secret_key": None})
    with pytest.raises(MarketDataError, match="credentials are missing"):
        AlpacaMarketData(stripped)


def test_get_bars_returns_normalized_frame(settings):
    client = FakeStockClient({"AAPL": make_bars(50)})
    provider = AlpacaMarketData(settings.alpaca, settings.data, client=client)
    result = provider.get_bars("aapl", "15Min", limit=20)
    assert len(result) == 20
    assert result.index.is_monotonic_increasing
    assert client.call_count == 1


def test_get_bars_retries_transient_failures(settings, monkeypatch):
    monkeypatch.setattr("trading_bot.utils.retry.time.sleep", lambda _: None)
    client = FakeStockClient({"AAPL": make_bars(10)}, fail_times=2)
    provider = AlpacaMarketData(settings.alpaca, settings.data, client=client)
    assert len(provider.get_bars("AAPL", "15Min")) == 10
    assert client.call_count == 3


def test_get_bars_multi_splits_results_per_symbol(settings):
    client = FakeStockClient({"AAPL": make_bars(30), "MSFT": make_bars(30, seed=3)})
    provider = AlpacaMarketData(settings.alpaca, settings.data, client=client)
    result = provider.get_bars_multi(["AAPL", "MSFT", "MISSING"], "15Min")
    assert set(result) == {"AAPL", "MSFT"}
    assert len(result["AAPL"]) == 30


def test_get_bars_multi_deduplicates_symbols(settings):
    client = FakeStockClient({"AAPL": make_bars(10)})
    provider = AlpacaMarketData(settings.alpaca, settings.data, client=client)
    provider.get_bars_multi(["AAPL", "aapl", " AAPL "], "15Min")
    assert client.requests[0].symbol_or_symbols == ["AAPL"]


def test_missing_symbol_returns_empty_not_error(settings):
    client = FakeStockClient({})
    provider = AlpacaMarketData(settings.alpaca, settings.data, client=client)
    assert provider.get_bars("NOPE", "1Day").empty


def test_fetch_watchlist_reports_failures_without_raising(settings):
    client = FakeStockClient({"AAPL": make_bars(120)})
    provider = AlpacaMarketData(settings.alpaca, settings.data, client=client)
    frames, report = provider.fetch_watchlist(["AAPL", "BROKEN"], "15Min", lookback_bars=60)
    assert set(frames) == {"AAPL"}
    assert len(frames["AAPL"]) == 60
    assert "BROKEN" in report.failed
    assert report.success_count == 1
    assert "1/2 symbols" in report.summary()


def test_provider_passes_configured_feed_and_adjustment(settings):
    client = FakeStockClient({"AAPL": make_bars(5)})
    alpaca = settings.alpaca.model_copy(update={"data_feed": "sip", "adjustment": "split"})
    AlpacaMarketData(alpaca, settings.data, client=client).get_bars("AAPL", "1Day")
    request = client.requests[0]
    assert request.feed.value == "sip"
    assert request.adjustment.value == "split"


# -- StaticMarketData ------------------------------------------------------------


def test_static_provider_slices_by_date_range():
    frame = make_bars(100)
    provider = StaticMarketData({"AAPL": frame})
    midpoint = frame.index[50]
    result = provider.get_bars("AAPL", "15Min", start=midpoint)
    assert result.index[0] == midpoint
    assert len(result) == 50


def test_static_provider_returns_empty_for_unknown_symbol():
    assert StaticMarketData({"AAPL": make_bars(5)}).get_bars("TSLA", "15Min").empty


def test_get_latest_price_falls_back_to_last_close():
    frame = make_bars(10)
    provider = StaticMarketData({"AAPL": frame})
    assert provider.get_latest_price("AAPL") == pytest.approx(float(frame["close"].iloc[-1]))
