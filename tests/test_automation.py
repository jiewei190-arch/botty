from datetime import datetime, timezone
from types import SimpleNamespace

from trading_bot.automation.runner import MarketOpenRunner


class Recorder:
    def __init__(self):
        self.messages = []

    def send(self, message):
        self.messages.append(message)


class ClockBroker:
    def __init__(self, *clocks):
        self.clocks = iter(clocks)

    def get_clock(self):
        return next(self.clocks)


def clock(day, is_open=True):
    return SimpleNamespace(
        timestamp=datetime(2026, 9, day, 14, 0, tzinfo=timezone.utc),
        is_open=is_open,
    )


def test_runner_scans_once_per_open_session():
    scans = []
    notices = Recorder()
    runner = MarketOpenRunner(
        ClockBroker(clock(8), clock(8), clock(9)),
        lambda: scans.append("scan") or 0,
        notices,
    )
    assert runner.step()
    assert not runner.step()
    assert runner.step()
    assert scans == ["scan", "scan"]


def test_closed_market_does_not_scan():
    scans = []
    runner = MarketOpenRunner(
        ClockBroker(clock(8, False)), lambda: scans.append("scan") or 0, Recorder()
    )
    assert not runner.step()
    assert not scans


def test_scan_failure_requests_attention_without_killing_runner():
    notices = Recorder()
    runner = MarketOpenRunner(
        ClockBroker(clock(8)), lambda: (_ for _ in ()).throw(RuntimeError("boom")), notices
    )
    assert runner.step()
    assert any("NEEDS ATTENTION" in message for message in notices.messages)


def test_broken_notification_channel_does_not_stop_scan():
    class BrokenNotifier:
        def send(self, message):
            raise RuntimeError("webhook offline")

    scans = []
    runner = MarketOpenRunner(
        ClockBroker(clock(8)), lambda: scans.append("scan") or 0, BrokenNotifier()
    )
    assert runner.step()
    assert scans == ["scan"]
