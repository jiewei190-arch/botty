from trading_bot.automation.status_server import automation_snapshot, build_status_server
from trading_bot.config.settings import AutomationSettings, DataSettings, Settings
from trading_bot.dashboard.automation_status import load_automation_status
from trading_bot.data.database import Database


def test_status_token_is_redacted_from_config():
    settings = Settings(automation=AutomationSettings(status_token="super-secret-token"))
    assert settings.redacted_dict()["automation"]["status_token"] == "***set***"


def test_worker_snapshot_contains_live_state(tmp_path):
    db_path = tmp_path / "worker.db"
    database = Database(db_path)
    database.initialize()
    database.state.set("automation_heartbeat", "2026-09-08T15:30:00+00:00")
    database.state.set("market_open", "true")
    database.state.set("last_completed_session", "2026-09-08")
    database.close()

    snapshot = automation_snapshot(db_path)
    assert snapshot["market_open"] == "true"
    assert snapshot["last_completed_session"] == "2026-09-08"
    assert snapshot["automation_heartbeat"] == "2026-09-08T15:30:00+00:00"


def test_dashboard_falls_back_to_local_database(tmp_path):
    db_path = tmp_path / "dashboard.db"
    database = Database(db_path)
    database.initialize()
    database.state.set("market_open", "true")
    database.close()

    settings = Settings(data=DataSettings(database_path=db_path))
    status, error = load_automation_status(settings)
    assert error is None
    assert status["market_open"] == "true"


def test_status_server_is_disabled_without_platform_port(monkeypatch):
    monkeypatch.delenv("PORT", raising=False)
    assert build_status_server(Settings()) is None
