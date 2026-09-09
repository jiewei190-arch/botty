from datetime import datetime, timedelta, timezone

from trading_bot.automation.slack_status import status_message
from trading_bot.data.database import Database


def test_status_message_reports_healthy_runtime(tmp_path):
    path = tmp_path / "botty.db"
    now = datetime.now(timezone.utc)
    database = Database(path)
    database.initialize()
    database.state.set("automation_heartbeat", (now - timedelta(minutes=2)).isoformat())
    database.state.set("market_open", "false")
    database.state.set("last_completed_session", "2026-09-09")
    database.close()

    message = status_message(path, now=now)

    assert "ONLINE & HEALTHY" in message
    assert "Heartbeat: 2.0 min ago" in message
    assert "Market: CLOSED" in message
    assert "Last full scan: 2026-09-09" in message


def test_status_message_reports_missing_heartbeat(tmp_path):
    message = status_message(tmp_path / "botty.db")

    assert "OFFLINE OR STALE" in message
    assert "Heartbeat: never" in message
