"""Where alerts go.

Each channel does one thing with an :class:`~trading_bot.alerts.models.Alert`
and knows nothing about why it fired. Adding Discord, Telegram, SMS or email
later means writing one ``deliver`` method — no change to the manager, the
detectors or the scoring engine.

A channel that raises must never end a scan. The manager isolates every
delivery, so a Discord outage costs you Discord alerts, not the scanner.
"""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, TextIO

from trading_bot.alerts.models import Alert, render_alert

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from trading_bot.data.database import AlertRepository

logger = logging.getLogger(__name__)


class AlertChannel(ABC):
    """A destination for alerts."""

    name: ClassVar[str] = "channel"

    @abstractmethod
    def deliver(self, alert: Alert) -> None:
        """Send one alert. Raising is tolerated; the manager logs and continues."""

    def close(self) -> None:  # noqa: B027 - optional hook, not part of the contract
        """Release anything the channel holds.

        Deliberately concrete and empty: most channels hold nothing, and forcing
        every one to write ``pass`` would be noise. A channel with a socket or a
        file overrides it.
        """
        return None


class ConsoleChannel(AlertChannel):
    """Prints the human-readable alert block."""

    name: ClassVar[str] = "console"

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stdout

    def deliver(self, alert: Alert) -> None:
        print(render_alert(alert), file=self._stream)
        print("", file=self._stream)
        self._stream.flush()


class LogChannel(AlertChannel):
    """Emits the alert as a structured log record.

    The payload rides on ``extra``, so the JSON-lines handler writes the whole
    alert as one machine-readable object while the console handler prints a
    readable single line. One call, two representations, no duplication.
    """

    name: ClassVar[str] = "log"

    def __init__(self, target: logging.Logger | None = None) -> None:
        self._logger = target or logging.getLogger("trading_bot.alerts")

    def deliver(self, alert: Alert) -> None:
        self._logger.warning(
            "ALERT %s %s score=%.1f (%s)",
            alert.symbol,
            alert.direction.value,
            alert.score,
            ", ".join(alert.signal_types) or "no signals",
            extra={"alert": alert.as_dict(), "event": "alert"},
        )


class DatabaseChannel(AlertChannel):
    """Persists the alert so the dashboard and later sessions can see it."""

    name: ClassVar[str] = "database"

    def __init__(self, repository: AlertRepository) -> None:
        self._repository = repository

    def deliver(self, alert: Alert) -> None:
        self._repository.record(alert)


class CallbackChannel(AlertChannel):
    """Calls a function. The seam for Discord, Telegram, SMS or email.

    A webhook integration is ``CallbackChannel(post_to_discord)`` — the manager
    needs no knowledge of it, and tests can capture alerts with a list append.
    """

    name: ClassVar[str] = "callback"

    def __init__(self, callback: Callable[[Alert], Any], *, name: str | None = None) -> None:
        self._callback = callback
        if name:
            self.name = name

    def deliver(self, alert: Alert) -> None:
        self._callback(alert)
