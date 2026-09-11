"""Always-on market supervision for swing scanning.

One scan at the regular-session open, then a rescan every
``open_scan_interval_seconds`` while the market stays open — a swing setup that
forms at 11:00 should not wait until tomorrow to be seen. Set the interval to
``None`` for strict once-per-session behaviour.

The runner stays online around the clock but only scans while Alpaca reports the
regular session open, so nothing can place an extended-hours order by accident.
State is restart-safe: the session already scanned and the session already
summarised are both persisted, so a redeploy mid-afternoon does not re-scan or
re-send the closing report.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import date, datetime

from trading_bot.automation.notifications import Notifier

logger = logging.getLogger(__name__)


class MarketOpenRunner:
    """Clock-driven loop with restart-safe, once-per-session scan semantics."""

    def __init__(
        self,
        broker,
        scan_once: Callable[[], int],
        notifier: Notifier,
        *,
        supervise_once: Callable[[], object] | None = None,
        closed_poll_seconds: int = 300,
        open_poll_seconds: int = 60,
        open_scan_interval_seconds: int | None = 3600,
        load_last_session: Callable[[], date | None] | None = None,
        save_last_session: Callable[[date], None] | None = None,
        load_last_scan_at: Callable[[], datetime | None] | None = None,
        save_last_scan_at: Callable[[datetime], None] | None = None,
        heartbeat: Callable[[bool], None] | None = None,
        close_summary: Callable[[date], str] | None = None,
        load_last_close_summary: Callable[[], date | None] | None = None,
        save_last_close_summary: Callable[[date], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.broker = broker
        self.scan_once = scan_once
        self.notifier = notifier
        self.supervise_once = supervise_once
        if closed_poll_seconds < 15:
            raise ValueError("closed_poll_seconds must be at least 15")
        if open_scan_interval_seconds is not None and open_scan_interval_seconds < 60:
            raise ValueError("open_scan_interval_seconds must be at least 60 seconds")
        self.closed_poll_seconds = closed_poll_seconds
        self.open_poll_seconds = open_poll_seconds
        self.open_scan_interval_seconds = open_scan_interval_seconds
        self.sleep = sleep
        self.save_last_session = save_last_session
        self.heartbeat = heartbeat
        self.close_summary = close_summary
        self.save_last_close_summary = save_last_close_summary
        self._last_session = load_last_session() if load_last_session else None
        self._last_close_summary = (
            load_last_close_summary() if load_last_close_summary else None
        )
        self._market_is_open = False
        self._scan_failure_notified = False
        self._clock_failure_notified = False
        self.save_last_scan_at = save_last_scan_at
        # Restored, not reset. The interval means "an hour since the last scan",
        # which is a fact about the market rather than about this process: a
        # restart that zeroed it left the runner unable to rescan for the rest
        # of the session, because _scan_due treats a missing timestamp as "not
        # due" and the session had already been marked complete. Restoring it
        # also keeps a crash loop from rescanning on every boot, since the
        # timestamp it reads back is genuinely recent.
        self._last_scan_at: datetime | None = (
            load_last_scan_at() if load_last_scan_at else None
        )

    def _notify(self, message: str) -> None:
        """A broken alert channel must never stop market supervision."""
        try:
            self.notifier.send(message)
        except Exception:
            logger.exception("Notification delivery failed")

    def step(self) -> bool:
        """Poll once; return True only when a scan was run."""
        try:
            clock = self.broker.get_clock()
        except Exception as error:
            logger.exception("Market clock poll failed; Botty will retry")
            if not self._clock_failure_notified:
                self._notify(f"BOTTY NEEDS ATTENTION: market clock unavailable: {error}")
                self._clock_failure_notified = True
            self._market_is_open = False
            return False
        self._clock_failure_notified = False
        self._market_is_open = bool(clock.is_open)
        if self.heartbeat is not None:
            self.heartbeat(self._market_is_open)
        if not clock.is_open:
            session = self._last_session
            if (
                session is not None
                and session != self._last_close_summary
                and self.close_summary is not None
            ):
                try:
                    message = self.close_summary(session)
                    self._notify(message)
                    self._last_close_summary = session
                    if self.save_last_close_summary is not None:
                        self.save_last_close_summary(session)
                except Exception:
                    logger.exception("End-of-day summary failed; Botty will retry")
            return False
        if self.supervise_once is not None:
            try:
                self.supervise_once()
            except Exception as error:
                logger.exception("Position supervision failed")
                self._notify(f"BOTTY NEEDS ATTENTION: position supervision failed: {error}")
        session = clock.timestamp.date()
        new_session = session != self._last_session
        if not self._scan_due(clock.timestamp, new_session):
            return False

        self._last_scan_at = clock.timestamp
        if self.save_last_scan_at is not None:
            self.save_last_scan_at(clock.timestamp)
        label = "market-open" if new_session else "scheduled swing"
        self._notify(f"Botty {label} scan started ({clock.timestamp.isoformat()}).")
        try:
            code = self.scan_once()
        except Exception as error:
            logger.exception("Automated market-open scan crashed")
            if not self._scan_failure_notified:
                self._notify(f"BOTTY NEEDS ATTENTION: scan crashed: {error}")
                self._scan_failure_notified = True
            return True
        if code:
            if not self._scan_failure_notified:
                self._notify(f"BOTTY NEEDS ATTENTION: scan exited with code {code}.")
                self._scan_failure_notified = True
        else:
            self._scan_failure_notified = False
            self._last_session = session
            if self.save_last_session is not None:
                self.save_last_session(session)
            self._notify(f"Botty {label} scan completed.")
        return True

    def _scan_due(self, timestamp, new_session: bool) -> bool:
        """Whether a scan should run now.

        The open is always scanned. After that the interval decides, so a setup
        that forms at 11:00 is seen the same day rather than at tomorrow's bell.
        """
        if new_session:
            return True
        if self.open_scan_interval_seconds is None:
            return False
        if self._last_scan_at is None:
            # The session is recorded as scanned but nothing says when: a
            # database written before this timestamp was persisted, which is
            # every deployment upgrading into it. Scanning once is what keeps
            # such an upgrade from sitting idle for the rest of its first
            # session -- the failure this mechanism exists to prevent.
            #
            # Only when the result can be recorded, though. Persistence is what
            # makes this safe rather than a loop: the timestamp is written
            # before the scan runs, so the next poll measures a real interval.
            # A caller that cannot record one gets the stricter old reading,
            # where a restart never repeats a session already marked complete.
            return self.save_last_scan_at is not None
        elapsed = (timestamp - self._last_scan_at).total_seconds()
        return elapsed >= self.open_scan_interval_seconds

    def run_forever(self) -> None:
        cadence = (
            "once per market session"
            if self.open_scan_interval_seconds is None
            else f"every {self.open_scan_interval_seconds // 60} minute(s) while open"
        )
        self._notify(f"Botty automation is online; scans run {cadence}.")
        while True:
            self.step()
            self.sleep(
                self.open_poll_seconds if self._market_is_open
                else self.closed_poll_seconds
            )
