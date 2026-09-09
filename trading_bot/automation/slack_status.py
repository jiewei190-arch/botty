"""Answer Botty health checks from Slack over Socket Mode."""

from __future__ import annotations

import contextlib
import re
from datetime import datetime, timezone
from pathlib import Path

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from trading_bot.config.settings import get_settings
from trading_bot.data.database import Database


def status_message(
    database_path: Path,
    *,
    max_heartbeat_age_minutes: float = 15.0,
    now: datetime | None = None,
) -> str:
    """Build a compact, read-only status response suitable for Slack."""
    database = Database(database_path)
    database.initialize()
    try:
        heartbeat = database.state.get("automation_heartbeat")
        market_open = database.state.get("market_open")
        last_session = database.state.get("last_completed_session")
        latest_equity = database.equity.latest()
        positions = database.positions.all()
        open_orders = database.orders.open_orders()
        open_trades = database.trades.open_trades()
        recent_errors = database.events.recent(limit=5, level="ERROR")
    finally:
        database.close()

    current = now or datetime.now(timezone.utc)
    age_minutes: float | None = None
    if heartbeat:
        with contextlib.suppress(ValueError):
            stamp = datetime.fromisoformat(heartbeat)
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
            age_minutes = (current - stamp).total_seconds() / 60

    healthy = age_minutes is not None and age_minutes <= max_heartbeat_age_minutes
    health = "✅ ONLINE & HEALTHY" if healthy else "🚨 OFFLINE OR STALE"
    heartbeat_text = f"{age_minutes:.1f} min ago" if age_minutes is not None else "never"
    market = "OPEN" if market_open == "true" else "CLOSED" if market_open == "false" else "unknown"
    equity = f"${float(latest_equity['equity']):,.2f}" if latest_equity else "unavailable"
    return "\n".join(
        (
            f"*BOTTY STATUS — {health}*",
            f"• Heartbeat: {heartbeat_text}",
            f"• Market: {market}",
            f"• Last full scan: {last_session or 'never'}",
            f"• Open positions/orders: {len(positions)}/{len(open_orders)}",
            f"• Open Botty trades: {len(open_trades)}",
            f"• Paper equity: {equity}",
            f"• Recent errors: {len(recent_errors)}",
        )
    )


def main() -> None:
    """Connect to Slack and serve status requests until the process stops."""
    settings = get_settings()
    bot_token = settings.automation.slack_bot_token
    app_token = settings.automation.slack_app_token
    if not bot_token or not app_token:
        raise SystemExit("Slack status requires both Socket Mode tokens")

    app = App(token=bot_token)

    def current_status() -> str:
        return status_message(settings.data.database_path)

    @app.command("/botty-status")
    def botty_status_command(ack, respond) -> None:
        ack()
        respond(current_status(), response_type="ephemeral")

    @app.message(re.compile(r"^\s*status\s*$", re.IGNORECASE))
    def botty_status_message(message, say) -> None:
        if message.get("bot_id") or message.get("subtype"):
            return
        say(current_status())

    SocketModeHandler(app, app_token).start()


if __name__ == "__main__":
    main()
