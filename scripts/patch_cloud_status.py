"""One-time patch to wire cloud worker status into settings, CLI and dashboard."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Settings -----------------------------------------------------------------
path = ROOT / "trading_bot/config/settings.py"
text = path.read_text()
old = '''    max_order_failures_per_scan: int = Field(default=2, ge=1, le=20)\n    #: Optional Discord or Slack incoming-webhook URL. Kept out of logs/config dumps.\n'''
new = '''    max_order_failures_per_scan: int = Field(default=2, ge=1, le=20)\n    #: Public base URL of the deployed automation worker, used only by the dashboard.\n    status_url: str | None = None\n    #: Shared bearer token protecting the worker's read-only /status endpoint.\n    status_token: str | None = None\n    #: Optional Discord or Slack incoming-webhook URL. Kept out of logs/config dumps.\n'''
if old not in text:
    raise RuntimeError("AutomationSettings insertion point not found")
text = text.replace(old, new, 1)
old = '''        if automation.get("webhook_url"):\n            automation["webhook_url"] = "***set***"\n        return payload\n'''
new = '''        if automation.get("webhook_url"):\n            automation["webhook_url"] = "***set***"\n        if automation.get("status_token"):\n            automation["status_token"] = "***set***"\n        return payload\n'''
if old not in text:
    raise RuntimeError("settings redaction point not found")
path.write_text(text)

# CLI worker ---------------------------------------------------------------
path = ROOT / "trading_bot/main.py"
text = path.read_text()
old = '''        from trading_bot.automation.runner import MarketOpenRunner\n        from trading_bot.execution.broker import BrokerError, build_broker\n'''
new = '''        from trading_bot.automation.runner import MarketOpenRunner\n        from trading_bot.automation.status_server import build_status_server\n        from trading_bot.execution.broker import BrokerError, build_broker\n'''
if old not in text:
    raise RuntimeError("worker import point not found")
text = text.replace(old, new, 1)
old = '''        database = Database(settings.data.database_path)\n        database.initialize()\n        reconciler = BrokerReconciler(broker, database, build_notifier(settings))\n'''
new = '''        database = Database(settings.data.database_path)\n        database.initialize()\n        status_server = build_status_server(settings)\n        if status_server is not None:\n            status_server.start()\n        reconciler = BrokerReconciler(broker, database, build_notifier(settings))\n'''
if old not in text:
    raise RuntimeError("worker status-server start point not found")
text = text.replace(old, new, 1)
old = '''        finally:\n            database.close()\n        return EXIT_OK\n'''
new = '''        finally:\n            if status_server is not None:\n                status_server.close()\n            database.close()\n        return EXIT_OK\n'''
if old not in text:
    raise RuntimeError("worker status-server close point not found")
text = text.replace(old, new, 1)
path.write_text(text)

# Dashboard ---------------------------------------------------------------
path = ROOT / "trading_bot/dashboard/app.py"
text = path.read_text()
old = '''def _automation(settings) -> None:\n    """Read-only operating view of the unattended paper trader."""\n    from trading_bot.data.database import Database\n\n    database = Database(settings.data.database_path)\n    database.initialize()\n    heartbeat = database.state.get("automation_heartbeat")\n    market_open = database.state.get("market_open") or "unknown"\n    last_session = database.state.get("last_completed_session") or "never"\n    equity = database.equity.latest()\n    positions = database.positions.all()\n    orders = database.orders.open_orders()\n    trades = database.trades.open_trades()\n    errors = database.events.recent(limit=20, level="ERROR")\n    closed_stats = database.trades.statistics()\n    database.close()\n'''
new = '''def _automation(settings) -> None:\n    """Read-only operating view of the unattended paper trader."""\n    from trading_bot.dashboard.automation_status import load_automation_status\n\n    status, status_error = load_automation_status(settings)\n    heartbeat = status.get("automation_heartbeat")\n    market_open = status.get("market_open", "unknown")\n    last_session = status.get("last_completed_session", "never")\n    equity = status.get("equity")\n    positions = status.get("positions") or []\n    orders = status.get("orders") or []\n    trades = status.get("open_trades") or []\n    errors = status.get("errors") or []\n    closed_stats = status.get("closed_stats") or {"total_trades": 0, "total_pnl": 0.0}\n'''
if old not in text:
    raise RuntimeError("dashboard automation block not found")
text = text.replace(old, new, 1)
old = '''    st.subheader("Automation")\n    st.caption("Read-only health, broker reconciliation, and paper performance.")\n    if healthy:\n'''
new = '''    st.subheader("Automation")\n    st.caption("Read-only health, broker reconciliation, and paper performance.")\n    if status_error:\n        st.warning(status_error)\n    if healthy:\n'''
if old not in text:
    raise RuntimeError("dashboard status warning point not found")
text = text.replace(old, new, 1)
path.write_text(text)

print("Cloud automation status integration patched")
