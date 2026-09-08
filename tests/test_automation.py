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


def test_open_market_supervises_on_every_poll_but_scans_once():
    scans, supervision = [], []
    runner = MarketOpenRunner(
        ClockBroker(clock(8), clock(8)), lambda: scans.append("scan") or 0, Recorder(),
        supervise_once=lambda: supervision.append("reconcile"),
    )
    runner.step()
    runner.step()
    assert scans == ["scan"]
    assert supervision == ["reconcile", "reconcile"]


def test_completed_session_survives_runner_restart():
    saved = []
    first = MarketOpenRunner(
        ClockBroker(clock(8)), lambda: 0, Recorder(), save_last_session=saved.append
    )
    assert first.step()
    restarted = MarketOpenRunner(
        ClockBroker(clock(8)), lambda: (_ for _ in ()).throw(AssertionError("rescanned")),
        Recorder(), load_last_session=lambda: saved[-1],
    )
    assert not restarted.step()


def test_failed_scan_is_retried_in_same_session():
    outcomes = iter([1, 0])
    runner = MarketOpenRunner(
        ClockBroker(clock(8), clock(8)), lambda: next(outcomes), Recorder()
    )
    assert runner.step()
    assert runner.step()


def test_heartbeat_records_closed_and_open_polls():
    states = []
    runner = MarketOpenRunner(
        ClockBroker(clock(8, False), clock(8, True)), lambda: 0, Recorder(),
        heartbeat=states.append,
    )
    runner.step()
    runner.step()
    assert states == [False, True]


def test_temporary_clock_failure_is_retried_instead_of_crashing():
    notices = Recorder()

    class BrokenClock:
        def get_clock(self):
            raise RuntimeError("temporary outage")

    runner = MarketOpenRunner(BrokenClock(), lambda: 0, notices)
    assert not runner.step()
    assert any("clock unavailable" in message for message in notices.messages)
