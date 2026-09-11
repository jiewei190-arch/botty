"""Persist and format one restart-safe Slack summary per market session."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date
from typing import Any

REPORT_STATE_KEY = "daily_scan_report"
LAST_SENT_STATE_KEY = "last_close_summary_session"


def save_scan_report(database, session: date, universe, sweep, execution_report=None) -> None:
    """Save the compact facts needed for the closing report."""
    opportunities = []
    for item in sweep.opportunities:
        opportunities.append({
            "symbol": item.signal.symbol,
            "direction": item.signal.direction.value,
            "strategy": item.signal.strategy,
            "confidence": round(float(item.confidence), 1),
            "risk_reward": round(float(item.signal.risk_reward_ratio), 2),
            "reasons": list(item.signal.reasons[:3]),
        })
    decisions = [
        {
            "symbol": item.symbol,
            "approved": item.approved,
            "reason": item.reason,
        }
        for item in getattr(execution_report, "decisions", ())
    ]
    payload = {
        "session": session.isoformat(),
        "catalogue_considered": universe.static_report.considered,
        "universe_size": sweep.universe_size,
        "scanned": sweep.scanned,
        "elapsed_seconds": round(float(sweep.elapsed_seconds), 1),
        "opportunities": opportunities,
        "decisions": decisions,
        "placed": int(getattr(execution_report, "placed", 0)),
        "alerted": int(getattr(execution_report, "alerted", 0)),
        "skipped": int(getattr(execution_report, "skipped", 0)),
        "failed": int(getattr(execution_report, "failed", 0)),
        "blockers": dict(sweep.blockers),
        "filtered_out": dict(sweep.filtered_out),
        "stale": dict(sweep.stale),
        "halt_reason": sweep.halt_reason,
    }
    database.state.set(REPORT_STATE_KEY, json.dumps(payload, separators=(",", ":")))
    database.events.record(
        category="daily_scan_report",
        message=f"Saved closing report for {session.isoformat()}",
        payload=payload,
    )


def load_scan_report(database, session: date) -> dict[str, Any] | None:
    raw = database.state.get(REPORT_STATE_KEY)
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    return payload if payload.get("session") == session.isoformat() else None


def _reason_family(reason: str) -> str:
    """Group a decision reason so the report counts causes, not symbols.

    The per-symbol verdicts belong on the dashboard, which already lists them.
    Repeating them in Slack made the closing message a second copy of a screen
    the reader already has, and buried the market summary above it. What does
    not appear anywhere else is *why* a day produced no alerts, so that is what
    survives here -- as a tally of causes.
    """
    text = re.sub(r"^(not approved|approved(?: alert)?|alerted)[:,]?\s*", "", reason.strip())
    # Execution errors carry a raw payload that is unique per failure; the
    # cause is the same one regardless of which broker message came back.
    text = re.sub(r"\s*\(.*", "", text)
    return text.rstrip(".") or "no reason recorded"


def format_close_summary(database, session: date) -> str:
    """Render the bot's side of the closing report.

    Deliberately short. It follows the market recap, and its job is to say
    whether Botty worked and why it stayed quiet -- not to re-list the
    candidates, which the dashboard shows in full.
    """
    payload = load_scan_report(database, session)
    if payload is None:
        return (
            f"BOTTY — {session.isoformat()}\n"
            "No completed scan report was available for this session."
        )

    lines = [
        f"*BOTTY — {session.isoformat()}*",
        (
            f"• Scanned {payload['scanned']:,} liquid symbols of "
            f"{payload['catalogue_considered']:,} listed in "
            f"{payload['elapsed_seconds']:,.1f}s — "
            f"{len(payload['opportunities'])} chart setup(s)."
        ),
        (
            f"• Option alerts: {payload.get('alerted', 0)} | "
            f"Paper orders: {payload['placed']} | Errors: {payload['failed']}"
        ),
    ]
    if payload.get("halt_reason"):
        lines.append(f"• Trading halt: {payload['halt_reason']}")

    blocked = [d for d in payload.get("decisions", []) if not d["approved"]]
    if blocked and not payload.get("alerted", 0):
        tally = Counter(_reason_family(d["reason"]) for d in blocked)
        lines.append("• Nothing was alerted because:")
        lines.extend(f"    {count}× {reason}" for reason, count in tally.most_common(3))
    elif not payload["opportunities"]:
        lines.append("• No setup cleared the chart and risk funnel today.")

    lines.append(
        f"• Open now: {len(database.positions.all())} position(s), "
        f"{len(database.orders.open_orders())} order(s). Paper trading only."
    )
    lines.append("• Per-symbol detail is on the dashboard.")
    return "\n".join(lines)[:3900]
