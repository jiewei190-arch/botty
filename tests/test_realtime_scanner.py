"""The continuously running scanner: aggregation, alerting and the REST cycle."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from tests.conftest import session_bars
from trading_bot.alerts import Alert, AlertConfig, AlertManager, CallbackChannel
from trading_bot.config.settings import load_settings
from trading_bot.data.market_data import MarketDataProvider, StaticMarketData
from trading_bot.data.market_stream import StreamEvent, StreamEventType
from trading_bot.scanner.factory import (
    build_alert_manager,
    build_detector_config,
    build_stream_config,
    build_weights,
    resolve_symbols,
)
from trading_bot.scanner.realtime import (
    BarAggregator,
    RealtimeScanner,
    RealtimeScannerConfig,
    next_poll_delay,
)
from trading_bot.utils.timeframes import Timeframe


def bar(**overrides) -> dict[str, float]:
    payload = {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0}
    payload.update(overrides)
    return payload


class TestBarAggregator:
    def test_emits_only_when_the_bucket_rolls_over(self):
        aggregator = BarAggregator("5Min")
        base = datetime(2026, 9, 8, 14, 0, tzinfo=timezone.utc)
        emitted = [
            aggregator.add("AAPL", bar(close=100.0 + i), base + timedelta(minutes=i))
            for i in range(7)
        ]
        completed = [item for item in emitted if item is not None]
        assert len(completed) == 1
        start, row = completed[0]
        assert start == base
        assert row["volume"] == 50.0  # five minute bars folded together

    def test_the_completed_bar_spans_the_whole_bucket(self):
        aggregator = BarAggregator("5Min")
        base = datetime(2026, 9, 8, 14, 0, tzinfo=timezone.utc)
        aggregator.add("AAPL", bar(open=100.0, high=101.0, low=99.5, close=100.5), base)
        aggregator.add(
            "AAPL",
            bar(open=100.5, high=103.0, low=98.0, close=102.0),
            base + timedelta(minutes=1),
        )
        completed = aggregator.add("AAPL", bar(), base + timedelta(minutes=5))
        assert completed is not None
        _, row = completed
        assert row["open"] == 100.0
        assert row["high"] == 103.0
        assert row["low"] == 98.0
        assert row["close"] == 102.0

    def test_a_partial_bucket_is_never_published(self):
        """A forming bar has a low that has not finished falling."""
        aggregator = BarAggregator("15Min")
        base = datetime(2026, 9, 8, 14, 0, tzinfo=timezone.utc)
        for minute in range(14):
            assert aggregator.add("AAPL", bar(), base + timedelta(minutes=minute)) is None
        assert aggregator.pending("AAPL") is not None

    def test_a_late_bar_does_not_reopen_a_closed_bucket(self):
        aggregator = BarAggregator("5Min")
        base = datetime(2026, 9, 8, 14, 0, tzinfo=timezone.utc)
        aggregator.add("AAPL", bar(), base)
        aggregator.add("AAPL", bar(), base + timedelta(minutes=5))
        assert aggregator.add("AAPL", bar(close=999.0), base) is None
        assert aggregator.pending("AAPL")["close"] != 999.0

    def test_symbols_are_bucketed_independently(self):
        aggregator = BarAggregator("5Min")
        base = datetime(2026, 9, 8, 14, 0, tzinfo=timezone.utc)
        aggregator.add("AAPL", bar(close=1.0), base)
        aggregator.add("NVDA", bar(close=2.0), base)
        assert aggregator.pending("AAPL")["close"] == 1.0
        assert aggregator.pending("NVDA")["close"] == 2.0


class FailingProvider(MarketDataProvider):
    def get_bars(self, *args, **kwargs):  # pragma: no cover - never reached
        raise AssertionError("should not be called")

    def get_bars_multi(self, *args, **kwargs):  # pragma: no cover - never reached
        raise AssertionError("should not be called")

    def fetch_watchlist(self, *args, **kwargs):
        raise RuntimeError("market data is down")


def scanner_for(frames: dict[str, pd.DataFrame], *, alerts=None, **config_overrides):
    defaults = {
        "symbols": tuple(frames),
        "timeframe": "5Min",
        "lookback_bars": 900,
    }
    return RealtimeScanner(
        StaticMarketData(frames),
        config=RealtimeScannerConfig(**(defaults | config_overrides)),
        alerts=alerts,
    )


def busy_and_quiet() -> dict[str, pd.DataFrame]:
    """One symbol with a volume surge and a gap, one without."""
    quiet = session_bars(sessions=11, end_day="2026-09-03", volume=(2_000, 2_100))
    busy = pd.concat(
        [
            quiet,
            session_bars(
                sessions=1,
                end_day="2026-09-04",
                volume=(9_000, 9_100),
                start_price=float(quiet["close"].iloc[-1]) * 1.03,
            ),
        ]
    )
    busy.index.name = "timestamp"
    calm = pd.concat(
        [
            quiet,
            session_bars(
                sessions=1,
                end_day="2026-09-04",
                volume=(2_000, 2_100),
                start_price=float(quiet["close"].iloc[-1]),
            ),
        ]
    )
    calm.index.name = "timestamp"
    return {"BUSY": busy, "CALM": calm}


class TestScanCycle:
    def test_a_cycle_ranks_and_alerts(self):
        received: list[Alert] = []
        manager = AlertManager([CallbackChannel(received.append)], AlertConfig(min_score=1.0))
        frames = busy_and_quiet()
        scanner = scanner_for(frames, alerts=manager)
        cycle = scanner.scan_once(now=frames["BUSY"].index[-1].to_pydatetime())

        assert len(cycle.analyses) == 2
        assert cycle.analyses[0].symbol == "BUSY"
        assert cycle.signal_count >= 1
        assert [alert.symbol for alert in received] == ["BUSY"]

    def test_a_data_failure_is_a_cycle_error_not_a_crash(self):
        scanner = RealtimeScanner(
            FailingProvider(),
            config=RealtimeScannerConfig(symbols=("AAPL",)),
        )
        cycle = scanner.scan_once()
        assert cycle.errors and "market data is down" in cycle.errors[0]
        assert cycle.analyses == ()

    def test_regular_hours_only_suppresses_analysis_outside_the_session(self):
        frames = busy_and_quiet()
        scanner = scanner_for(frames, regular_hours_only=True)
        scanner.seed()
        overnight = datetime(2026, 9, 5, 3, 0, tzinfo=timezone.utc)
        assert scanner.analyze("BUSY", now=overnight) is None

    def test_status_reports_what_is_loaded(self):
        frames = busy_and_quiet()
        scanner = scanner_for(frames)
        scanner.seed()
        status = scanner.status()
        assert status["frames"] == 2
        assert status["timeframe"] == "5Min"
        assert status["seeded"] is True


class TestStreamPath:
    def test_a_completed_bucket_appends_a_bar_and_analyses(self):
        frames = busy_and_quiet()
        scanner = scanner_for(frames)
        scanner.seed()
        before = len(scanner.frames["BUSY"])
        last = scanner.frames["BUSY"].index[-1]

        analysis = None
        for minute in range(6):
            event = StreamEvent(
                StreamEventType.BAR,
                "BUSY",
                (last + pd.Timedelta(minutes=5 + minute)).to_pydatetime(),
                bar(close=200.0 + minute, volume=50_000.0),
            )
            analysis = scanner.handle_event(event) or analysis

        assert analysis is not None
        assert len(scanner.frames["BUSY"]) == before + 1

    def test_partial_buckets_do_not_touch_the_frame(self):
        frames = busy_and_quiet()
        scanner = scanner_for(frames)
        scanner.seed()
        before = len(scanner.frames["BUSY"])
        last = scanner.frames["BUSY"].index[-1]
        for minute in range(4):
            scanner.handle_event(
                StreamEvent(
                    StreamEventType.BAR,
                    "BUSY",
                    (last + pd.Timedelta(minutes=5 + minute)).to_pydatetime(),
                    bar(),
                )
            )
        assert len(scanner.frames["BUSY"]) == before

    def test_trades_and_quotes_are_ignored(self):
        scanner = scanner_for(busy_and_quiet())
        scanner.seed()
        event = StreamEvent(
            StreamEventType.TRADE, "BUSY", datetime.now(timezone.utc), {"price": 1.0}
        )
        assert scanner.handle_event(event) is None

    def test_appending_a_duplicate_timestamp_replaces_it(self):
        scanner = scanner_for(busy_and_quiet())
        scanner.seed()
        stamp = scanner.frames["BUSY"].index[-1].to_pydatetime()
        before = len(scanner.frames["BUSY"])
        scanner.append_bar("BUSY", stamp, bar(close=555.0))
        assert len(scanner.frames["BUSY"]) == before
        assert scanner.frames["BUSY"]["close"].iloc[-1] == 555.0

    def test_frames_are_bounded(self):
        scanner = scanner_for(busy_and_quiet(), max_bars=200)
        scanner.seed()
        assert all(len(frame) <= 200 for frame in scanner.frames.values())


class TestPollDelay:
    @pytest.mark.parametrize(
        "timeframe,expected",
        [("1Min", 60.0), ("5Min", 300.0), ("15Min", 900.0)],
    )
    def test_delay_tracks_the_timeframe(self, timeframe, expected):
        assert next_poll_delay(Timeframe.parse(timeframe)) == expected

    def test_there_is_a_floor(self):
        assert next_poll_delay(Timeframe.parse("1Min"), floor_seconds=90) == 90.0


class TestFactory:
    def test_resolves_the_configured_universe(self, settings):
        symbols = resolve_symbols(settings)
        assert "SPY" in symbols and "AAPL" in symbols

    def test_an_explicit_override_wins(self, settings):
        assert resolve_symbols(settings, override=["nvda", "amd"]) == ("NVDA", "AMD")

    def test_environment_thresholds_reach_the_detectors(self, monkeypatch):
        monkeypatch.setenv("DETECTOR_VOLUME_THRESHOLD", "3.0")
        monkeypatch.setenv("DETECTOR_GAP_THRESHOLD_PCT", "2.5")
        config = build_detector_config(load_settings())
        assert config.volume.threshold == 3.0
        assert config.gap.threshold_pct == 2.5
        # Untouched defaults survive.
        assert config.volume.saturation == 5.0

    def test_environment_weights_reach_the_scorer(self, monkeypatch):
        monkeypatch.setenv("SCANNER_WEIGHT_GAP", "0.5")
        assert build_weights(load_settings()).gap == 0.5

    def test_channels_follow_the_alert_settings(self, settings, database, monkeypatch):
        monkeypatch.setenv("ALERT_CONSOLE_ENABLED", "false")
        manager = build_alert_manager(load_settings(), database=database)
        assert "console" not in {channel.name for channel in manager.channels}
        assert "database" in {channel.name for channel in manager.channels}

    def test_cooldowns_are_primed_from_stored_alerts(self, settings, database):
        from trading_bot.detectors import Bias, DetectorSignal, SignalType, score_signals

        now = datetime.now(timezone.utc)
        score = score_signals(
            "NVDA",
            [DetectorSignal("NVDA", now, SignalType.MOMENTUM, Bias.BULLISH, 9.0, "x")],
        )
        database.alerts.record(Alert.from_score(score, threshold=1.0))
        manager = build_alert_manager(settings, database=database)
        assert "NVDA:BULLISH" in manager._state

    def test_stream_config_uses_the_configured_feed(self, settings):
        config = build_stream_config(settings, ("AAPL", "NVDA"))
        assert config.feed == settings.alpaca.data_feed
        assert config.subscribe_bars


def test_scanner_seeds_from_history(monkeypatch):
    frames = busy_and_quiet()
    scanner = scanner_for(frames)
    counts = scanner.seed()
    assert set(counts) == {"BUSY", "CALM"}
    assert all(count > 100 for count in counts.values())
    assert not np.isnan(scanner.frames["BUSY"]["close"].iloc[-1])
