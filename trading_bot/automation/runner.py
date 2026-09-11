"""Run exactly one scan when each regular US market session opens."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

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
        closed_poll_seconds: int = 300,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.broker = broker
        self.scan_once = scan_once
        self.notifier = notifier
        self.closed_poll_seconds = closed_poll_seconds
        self.sleep = sleep
        self._last_session = None

    def _notify(self, message: str) -> None:
        """A broken alert channel must never stop market supervision."""
        try:
            self.notifier.send(message)
        except Exception:
            logger.exception("Notification delivery failed")

    def step(self) -> bool:
        """Poll once; return True only when a scan was run."""
        clock = self.broker.get_clock()
        if not clock.is_open:
            return False
        session = clock.timestamp.date()
        if session == self._last_session:
            return False

        self._last_session = session
        self._notify(f"Botty market-open scan started ({session.isoformat()}).")
        try:
            code = self.scan_once()
        except Exception as error:
            logger.exception("Automated market-open scan crashed")
            self._notify(f"BOTTY NEEDS ATTENTION: scan crashed: {error}")
            return True
        if code:
            self._notify(f"BOTTY NEEDS ATTENTION: scan exited with code {code}.")
        else:
            self._notify("Botty market-open scan completed.")
        return True

    def run_forever(self) -> None:
        self._notify("Botty automation is online and waiting for the market.")
        while True:
            ran = self.step()
            self.sleep(60 if ran else self.closed_poll_seconds)
