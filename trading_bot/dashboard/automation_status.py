"""Read automation status from the actual cloud worker.

When AUTO_STATUS_URL is configured, the dashboard calls the worker's authenticated
/status endpoint.  Local SQLite remains a fallback for single-host development.
"""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from trading_bot.data.database import Database


def load_automation_status(settings) -> tuple[dict, str | None]:
    url = (settings.automation.status_url or "").strip()
    token = (settings.automation.status_token or "").strip()
    if url:
        endpoint = f"{url.rstrip('/')}/status"
        request = Request(endpoint, headers={"Authorization": f"Bearer {token}"})
        try:
            with urlopen(request, timeout=8) as response:  # noqa: S310 - configured endpoint
                return json.loads(response.read().decode("utf-8")), None
        except (HTTPError, URLError, TimeoutError, ValueError) as error:
            return {}, f"Worker status endpoint unavailable: {error}"

    database = Database(settings.data.database_path)
    database.initialize()
    try:
        return {
            "automation_heartbeat": database.state.get("automation_heartbeat"),
            "market_open": database.state.get("market_open") or "unknown",
            "last_completed_session": database.state.get("last_completed_session") or "never",
            "equity": database.equity.latest(),
            "positions": database.positions.all(),
            "orders": database.orders.open_orders(),
            "open_trades": database.trades.open_trades(),
            "errors": database.events.recent(limit=20, level="ERROR"),
            "closed_stats": database.trades.statistics(),
        }, None
    finally:
        database.close()
