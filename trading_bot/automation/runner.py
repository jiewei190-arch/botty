"""Run exactly one scan when each regular US market session opens."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import date

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
        load_last_session: Callable[[], date | None] | None = None,
        save_last_session: Callable[[date], None] | None = None,
        heartbeat: Callable[[bool], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.broker = broker
        self.scan_once = scan_once
        self.notifier = notifier
        self.supervise_once = supervise_once
        self.closed_poll_seconds = closed_poll_seconds
        self.open_poll_seconds = open_poll_seconds
        self.sleep = sleep
        self.save_last_session = save_last_session
        self.heartbeat = heartbeat
        self._last_session = load_last_session() if load_last_session else None
        self._market_is_open = False
        self._scan_failure_notified = False
        self._clock_failure_notified = False

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
            return False
        if self.supervise_once is not None:
            try:
                self.supervise_once()
            except Exception as error:
                logger.exception("Position supervision failed")
                self._notify(f"BOTTY NEEDS ATTENTION: position supervision failed: {error}")
        session = clock.timestamp.date()
        if session == self._last_session:
            return False

        self._notify(f"Botty market-open scan started ({session.isoformat()}).")
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
            self._notify("Botty market-open scan completed.")
        return True

    def run_forever(self) -> None:
        self._notify("Botty automation is online and waiting for the market.")
        while True:
            self.step()
            self.sleep(
                self.open_poll_seconds if self._market_is_open
                else self.closed_poll_seconds
            )
