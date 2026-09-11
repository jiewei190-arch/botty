"""Small, dependency-free notification boundary for unattended runs."""

from __future__ import annotations

import json
import logging
from typing import Protocol
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class Notifier(Protocol):
    def send(self, message: str) -> None: ...


class NullNotifier:
    """Log notifications when no external destination has been configured."""

    def send(self, message: str) -> None:
        logger.info("NOTIFICATION: %s", message)


class WebhookNotifier:
    """Send plain-text alerts to a Discord or Slack incoming webhook."""

    def __init__(self, url: str, *, kind: str = "discord", timeout: float = 10.0) -> None:
        if not url.startswith("https://"):
            raise ValueError("Notification webhook must use HTTPS")
        if kind not in {"discord", "slack"}:
            raise ValueError("Webhook kind must be discord or slack")
        self.url = url
        self.kind = kind
        self.timeout = timeout

    def send(self, message: str) -> None:
        key = "content" if self.kind == "discord" else "text"
        body = json.dumps({key: message[:1900]}).encode("utf-8")
        request = Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json", "User-Agent": "Botty/0.1"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                if response.status >= 300:
                    raise RuntimeError(f"notification webhook returned HTTP {response.status}")
        except Exception as error:
            logger.error("Could not send notification: %s", error)
            raise


def build_notifier(settings) -> Notifier:
    options = settings.automation
    if not options.webhook_url:
        return NullNotifier()
    return WebhookNotifier(
        options.webhook_url,
        kind=options.webhook_kind,
        timeout=options.webhook_timeout_seconds,
    )
