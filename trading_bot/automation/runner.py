"""Always-on market supervision for swing scanning.

The runner is intentionally conservative about execution: it can stay online
24/7, but paper/live order-producing scans are only invoked while Alpaca reports
the regular market open. That avoids accidental extended-hours bracket/order
behaviour while still giving Botty a persistent process that wakes, checks the
clock, and resumes automatically after restarts.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime

from trading_bot.automation.notifications import Notifier

logger = logging.getLogger(__name__)


class MarketOpenRunner:
    """Clock-driven, restart-safe swing scanner.

    By default this preserves the original once-per-session behaviour. Set
    ``open_scan_interval_seconds`` to a positive value to re-run the swing scan
    periodically while the regular market is open. The first scan of every new
    session still runs immediately.
    """

    def __init__(
        self,
        broker,
        scan_once: Callable[[], int],
        notifier: Notifier,
        *,
        closed_poll_seconds: int = 300,
        open_scan_interval_seconds: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if closed_poll_seconds < 15:
            raise ValueError("closed_poll_seconds must be at least 15")
        if open_scan_interval_seconds is not None and open_scan_interval_seconds < 60:
            raise ValueError("open_scan_interval_seconds must be at least 60 seconds")

        self.broker = broker
        self.scan_once = scan_once
        self.notifier = notifier
        self.closed_poll_seconds = closed_poll_seconds
        self.open_scan_interval_seconds = open_scan_interval_seconds
        self.sleep = sleep
        self._last_session = None
        self._last_scan_at: datetime | None = None

    def _notify(self, message: str) -> None:
        """A broken alert channel must never stop market supervision."""
        try:
            self.notifier.send(message)
        except Exception:
            logger.exception("Notification delivery failed")

    def _scan_due(self, timestamp: datetime, session) -> bool:
        """Whether a new open-session scan should run now."""
        if session != self._last_session:
            return True
        if self.open_scan_interval_seconds is None or self._last_scan_at is None:
            return False
        elapsed = (timestamp - self._last_scan_at).total_seconds()
        return elapsed >= self.open_scan_interval_seconds

    def step(self) -> bool:
        """Poll once; return True only when a scan was run."""
        clock = self.broker.get_clock()
        if not clock.is_open:
            return False

        timestamp = clock.timestamp
        session = timestamp.date()
        if not self._scan_due(timestamp, session):
            return False

        new_session = session != self._last_session
        self._last_session = session
        self._last_scan_at = timestamp
        label = "market-open" if new_session else "scheduled swing"
        self._notify(f"Botty {label} scan started ({timestamp.isoformat()}).")
        try:
            code = self.scan_once()
        except Exception as error:
            logger.exception("Automated swing scan crashed")
            self._notify(f"BOTTY NEEDS ATTENTION: scan crashed: {error}")
            return True

        if code:
            self._notify(f"BOTTY NEEDS ATTENTION: scan exited with code {code}.")
        else:
            self._notify(f"Botty {label} scan completed.")
        return True

    def run_forever(self) -> None:
        cadence = (
            "once per market session"
            if self.open_scan_interval_seconds is None
            else f"every {self.open_scan_interval_seconds // 60} minute(s) while open"
        )
        self._notify(
            "Botty automation is online 24/7; execution scans run " + cadence + "."
        )
        while True:
            ran = self.step()
            if ran and self.open_scan_interval_seconds is not None:
                pause = min(60, self.open_scan_interval_seconds)
            elif ran:
                pause = 60
            else:
                pause = self.closed_poll_seconds
            self.sleep(pause)
