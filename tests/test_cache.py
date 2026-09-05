"""Bar cache round-trips, coverage rules and failure tolerance."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from tests.conftest import make_bars
from trading_bot.data.cache import BarCache
from trading_bot.utils.timeframes import Timeframe

TF = Timeframe.parse("15Min")


@pytest.fixture
def cache(tmp_path) -> BarCache:
    return BarCache(tmp_path / "cache")


def test_round_trip(cache):
    frame = make_bars(50)
    cache.put("AAPL", TF, frame)
    result = cache.get("AAPL", TF, frame.index[0], frame.index[-1])
    assert result is not None
    assert len(result) == 50
    assert result.index.tz is not None


def test_miss_when_nothing_cached(cache):
    now = datetime.now(timezone.utc)
    assert cache.get("AAPL", TF, now - timedelta(days=1), now) is None


def test_miss_when_request_starts_before_cached_history(cache):
    frame = make_bars(50)
    cache.put("AAPL", TF, frame)
    earlier = frame.index[0] - timedelta(days=10)
    assert cache.get("AAPL", TF, earlier, frame.index[-1]) is None


def test_miss_when_request_ends_after_cached_history(cache):
    frame = make_bars(50)
    cache.put("AAPL", TF, frame)
    later = frame.index[-1] + timedelta(days=5)
    assert cache.get("AAPL", TF, frame.index[0], later) is None


def test_hit_returns_only_the_requested_window(cache):
    frame = make_bars(100)
    cache.put("AAPL", TF, frame)
    result = cache.get("AAPL", TF, frame.index[20], frame.index[40])
    assert len(result) == 21
    assert result.index[0] == frame.index[20]


def test_put_merges_and_deduplicates_overlapping_writes(cache):
    frame = make_bars(100)
    cache.put("AAPL", TF, frame.iloc[:60])
    cache.put("AAPL", TF, frame.iloc[40:])
    result = cache.get("AAPL", TF, frame.index[0], frame.index[-1])
    assert len(result) == 100
    assert not result.index.has_duplicates


def test_cache_key_separates_feeds(tmp_path):
    frame = make_bars(10)
    iex = BarCache(tmp_path / "c", feed="iex")
    sip = BarCache(tmp_path / "c", feed="sip")
    iex.put("AAPL", TF, frame)
    assert sip.get("AAPL", TF, frame.index[0], frame.index[-1]) is None


def test_cache_key_separates_adjustments(tmp_path):
    frame = make_bars(10)
    raw = BarCache(tmp_path / "c", adjustment="raw")
    adjusted = BarCache(tmp_path / "c", adjustment="all")
    raw.put("AAPL", TF, frame)
    assert adjusted.get("AAPL", TF, frame.index[0], frame.index[-1]) is None


def test_cache_key_separates_timeframes(cache):
    frame = make_bars(10)
    cache.put("AAPL", TF, frame)
    daily = Timeframe.parse("1Day")
    assert cache.get("AAPL", daily, frame.index[0], frame.index[-1]) is None


def test_expired_entries_are_ignored(tmp_path):
    frame = make_bars(10)
    writer = BarCache(tmp_path / "c")
    writer.put("AAPL", TF, frame)
    expiring = BarCache(tmp_path / "c", max_age=timedelta(seconds=-1))
    assert expiring.get("AAPL", TF, frame.index[0], frame.index[-1]) is None


def test_corrupt_cache_file_is_ignored_not_fatal(cache):
    frame = make_bars(10)
    cache.put("AAPL", TF, frame)
    cache.path_for("AAPL", TF).write_bytes(b"not a parquet file")
    assert cache.get("AAPL", TF, frame.index[0], frame.index[-1]) is None


def test_empty_writes_are_no_ops(cache):
    cache.put("AAPL", TF, pd.DataFrame())
    assert cache.stats()["files"] == 0


def test_clear_targets_one_symbol(cache):
    cache.put("AAPL", TF, make_bars(5))
    cache.put("MSFT", TF, make_bars(5, seed=3))
    assert cache.clear("AAPL") == 1
    assert cache.stats()["symbols"] == ["MSFT"]


def test_stats_summarize_contents(cache):
    cache.put("AAPL", TF, make_bars(5))
    stats = cache.stats()
    assert stats["files"] == 1
    assert stats["symbols"] == ["AAPL"]
    assert stats["size_mb"] >= 0
