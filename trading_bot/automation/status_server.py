"""Small authenticated HTTP status endpoint for the cloud automation worker.

The Streamlit dashboard and the automation worker can run on different hosts,
so they cannot safely communicate through the worker's local SQLite file.  This
module exposes a read-only snapshot over HTTP.  `/health` contains no account
information and is suitable for a platform health check; `/status` requires the
shared AUTO_STATUS_TOKEN.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import date, datetime, timezone
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from trading_bot.data.database import Database


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def automation_snapshot(database_path: Path) -> dict[str, Any]:
    database = Database(database_path)
    database.initialize()
    try:
        heartbeat = database.state.get("automation_heartbeat")
        market_open = database.state.get("market_open") or "unknown"
        last_session = database.state.get("last_completed_session") or "never"
        equity = database.equity.latest()
        positions = database.positions.all()
        orders = database.orders.open_orders()
        trades = database.trades.open_trades()
        errors = database.events.recent(limit=20, level="ERROR")
        closed_stats = database.trades.statistics()
        return _jsonable(
            {
                "server_time": datetime.now(timezone.utc),
                "automation_heartbeat": heartbeat,
                "market_open": market_open,
                "last_completed_session": last_session,
                "equity": equity,
                "positions": positions,
                "orders": orders,
                "open_trades": trades,
                "errors": errors,
                "closed_stats": closed_stats,
            }
        )
    finally:
        database.close()


class AutomationStatusServer:
    """Daemon HTTP server exposing the worker's real local state."""

    def __init__(self, database_path: Path, token: str, port: int) -> None:
        self.database_path = database_path
        self.token = token
        self.port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: object) -> None:
                return

            def _send(self, code: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                if self.path.rstrip("/") == "/health":
                    self._send(200, {"ok": True})
                    return
                if self.path.rstrip("/") != "/status":
                    self._send(404, {"error": "not found"})
                    return
                if not owner.token:
                    self._send(503, {"error": "status token is not configured"})
                    return
                expected = f"Bearer {owner.token}"
                if self.headers.get("Authorization") != expected:
                    self._send(401, {"error": "unauthorized"})
                    return
                try:
                    self._send(200, automation_snapshot(owner.database_path))
                except Exception as error:  # noqa: BLE001 - status must report failure
                    self._send(500, {"error": str(error)})

        self._server = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="botty-status-server",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


def build_status_server(settings) -> AutomationStatusServer | None:
    """Build a server when the hosting platform supplies a web port."""
    raw_port = os.getenv("PORT", "").strip()
    if not raw_port:
        return None
    token = (settings.automation.status_token or "").strip()
    return AutomationStatusServer(settings.data.database_path, token, int(raw_port))
