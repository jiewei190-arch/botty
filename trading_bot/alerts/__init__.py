"""Alerting (Phase 1).

::

    from trading_bot.alerts import AlertManager, ConsoleChannel, LogChannel

    alerts = AlertManager([ConsoleChannel(), LogChannel()])
    alerts.consider(analysis.score, price=analysis.context.last_price)

The manager decides *whether* to interrupt someone; the channels decide *where*
that lands. Discord, Telegram, SMS and email are a ``CallbackChannel`` away.
"""

from trading_bot.alerts.alert_manager import AlertConfig, AlertManager
from trading_bot.alerts.channels import (
    AlertChannel,
    CallbackChannel,
    ConsoleChannel,
    DatabaseChannel,
    LogChannel,
)
from trading_bot.alerts.models import (
    SIGNAL_LABELS,
    Alert,
    AlertPriority,
    render_alert,
)

__all__ = [
    "SIGNAL_LABELS",
    "Alert",
    "AlertChannel",
    "AlertConfig",
    "AlertManager",
    "AlertPriority",
    "CallbackChannel",
    "ConsoleChannel",
    "DatabaseChannel",
    "LogChannel",
    "render_alert",
]
