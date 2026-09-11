from datetime import date, datetime, timezone
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


def clock(day, is_open=True, hour=14, minute=0):
    return SimpleNamespace(
        timestamp=datetime(2026, 9, day, hour, minute, tzinfo=timezone.utc),
        is_open=is_open,
    )


def test_runner_scans_once_per_open_session_when_interval_disabled():
    scans = []
    notices = Recorder()
    runner = MarketOpenRunner(
        ClockBroker(clock(8), clock(8), clock(9)),
        lambda: scans.append("scan") or 0,
        notices,
        open_scan_interval_seconds=None,
    )
    assert runner.step()
    assert not runner.step()
    assert runner.step()
    assert scans == ["scan", "scan"]


def test_runner_rescans_hourly_by_default():
    scans = []
    runner = MarketOpenRunner(
        ClockBroker(
            clock(8, hour=14, minute=0),
            clock(8, hour=14, minute=30),
            clock(8, hour=15, minute=0),
        ),
        lambda: scans.append("scan") or 0,
        Recorder(),
    )
    assert runner.step()
    assert not runner.step()
    assert runner.step()
    assert scans == ["scan", "scan"]


def test_closed_market_does_not_execute_scan():
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


def test_close_summary_is_sent_once_after_completed_session():
    notices = Recorder()
    saved = []
    runner = MarketOpenRunner(
        ClockBroker(clock(8, True), clock(8, False), clock(8, False)),
        lambda: 0,
        notices,
        close_summary=lambda session: f"closing report {session}",
        save_last_close_summary=saved.append,
    )

    assert runner.step()
    assert not runner.step()
    assert not runner.step()
    assert notices.messages.count("closing report 2026-09-08") == 1
    assert saved == [date(2026, 9, 8)]


def test_close_summary_survives_restart_without_duplicate():
    notices = Recorder()
    runner = MarketOpenRunner(
        ClockBroker(clock(8, False)),
        lambda: 0,
        notices,
        load_last_session=lambda: date(2026, 9, 8),
        load_last_close_summary=lambda: date(2026, 9, 8),
        close_summary=lambda session: f"closing report {session}",
    )

    assert not runner.step()
    assert not notices.messages


# ---------------------------------------------------------------------------
# A restart must not cost the rest of the trading session
#
# `_last_scan_at` lived only in memory. After a mid-session restart it was None
# while the session was already recorded complete, so `_scan_due` took neither
# branch that can return True: not a new session, and no timestamp to measure
# the interval from. The runner announced "scans run every 60 minute(s) while
# open" and then scanned zero more times that day.
# ---------------------------------------------------------------------------


def test_a_restart_mid_session_resumes_scanning_on_the_interval():
    """The regression: an hour had passed, so the next poll is due."""
    scans = []
    scanned_at = datetime(2026, 9, 11, 13, 33, tzinfo=timezone.utc)
    runner = MarketOpenRunner(
        ClockBroker(clock(11, hour=15, minute=41)),
        lambda: scans.append("scan") or 0,
        Recorder(),
        open_scan_interval_seconds=3600,
        load_last_session=lambda: date(2026, 9, 11),
        load_last_scan_at=lambda: scanned_at,
    )

    assert runner.step()
    assert scans == ["scan"]


def test_a_restart_soon_after_a_scan_does_not_rescan():
    """A crash loop must not turn into a scan loop."""
    scans = []
    scanned_at = datetime(2026, 9, 11, 15, 39, tzinfo=timezone.utc)
    runner = MarketOpenRunner(
        ClockBroker(clock(11, hour=15, minute=41)),
        lambda: scans.append("scan") or 0,
        Recorder(),
        open_scan_interval_seconds=3600,
        load_last_session=lambda: date(2026, 9, 11),
        load_last_scan_at=lambda: scanned_at,
    )

    assert not runner.step()
    assert scans == []


def test_the_scan_time_is_persisted_when_a_scan_runs():
    saved = []
    runner = MarketOpenRunner(
        ClockBroker(clock(11, hour=14, minute=30)),
        lambda: 0,
        Recorder(),
        open_scan_interval_seconds=3600,
        save_last_scan_at=saved.append,
    )

    assert runner.step()
    assert saved == [datetime(2026, 9, 11, 14, 30, tzinfo=timezone.utc)]


def test_a_first_ever_start_still_scans_the_open():
    """Nothing persisted yet: a new session is always scanned."""
    scans = []
    runner = MarketOpenRunner(
        ClockBroker(clock(11)),
        lambda: scans.append("scan") or 0,
        Recorder(),
        open_scan_interval_seconds=3600,
        load_last_scan_at=lambda: None,
    )

    assert runner.step()
    assert scans == ["scan"]


def test_a_failed_scan_is_still_recorded_as_attempted():
    """A crashing scan must not retry every poll for the rest of the day."""
    saved = []
    runner = MarketOpenRunner(
        ClockBroker(clock(11, hour=14, minute=30)),
        lambda: 1,  # non-zero exit: the scan failed
        Recorder(),
        open_scan_interval_seconds=3600,
        save_last_scan_at=saved.append,
    )

    assert runner.step()
    assert saved == [datetime(2026, 9, 11, 14, 30, tzinfo=timezone.utc)]


def test_an_upgrade_with_no_recorded_scan_time_scans_once():
    """Every deployment upgrading into the persisted timestamp has no key yet.

    Its session is already marked complete, so without this the upgrade sits
    idle until the next morning's bell -- the exact failure the timestamp was
    added to prevent, reintroduced by the migration.
    """
    scans = []
    saved = []
    runner = MarketOpenRunner(
        ClockBroker(clock(11, hour=17, minute=40), clock(11, hour=17, minute=45)),
        lambda: scans.append("scan") or 0,
        Recorder(),
        open_scan_interval_seconds=3600,
        load_last_session=lambda: date(2026, 9, 11),
        load_last_scan_at=lambda: None,       # the key has never been written
        save_last_scan_at=saved.append,
    )

    assert runner.step()
    assert scans == ["scan"]
    # And the very next poll is quiet, because the scan recorded its time.
    assert not runner.step()
    assert scans == ["scan"]
    assert saved == [datetime(2026, 9, 11, 17, 40, tzinfo=timezone.utc)]
