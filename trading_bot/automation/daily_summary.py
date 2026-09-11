"""Persist and format one restart-safe Slack summary per market session."""

from __future__ import annotations

import json
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


def format_close_summary(database, session: date) -> str:
    """Render a concise report that stays below Slack's practical message limit."""
    payload = load_scan_report(database, session)
    if payload is None:
        return (
            f"BOTTY END-OF-DAY SUMMARY — {session.isoformat()}\n"
            "No completed scan report was available for this session."
        )

    lines = [
        f"*BOTTY END-OF-DAY SUMMARY — {session.isoformat()}*",
        (
            f"• Started with {payload['catalogue_considered']:,} listed assets; "
            f"analysed {payload['scanned']:,} liquid symbols "
            f"in {payload['elapsed_seconds']:,.1f}s"
        ),
        (
            f"• Final chart setups: {len(payload['opportunities'])} | "
            f"Option alerts: {payload.get('alerted', 0)} | "
            f"Paper orders: {payload['placed']} | "
            f"Skipped: {payload['skipped']} | Errors: {payload['failed']}"
        ),
    ]
    if payload.get("halt_reason"):
        lines.append(f"• Trading halt: {payload['halt_reason']}")

    candidates = {item["symbol"]: item for item in payload["opportunities"]}
    decisions = payload.get("decisions", [])
    if decisions:
        lines.append("\n*Final candidate decisions* ")
        for decision in decisions[:10]:
            candidate = candidates.get(decision["symbol"], {})
            score = candidate.get("confidence")
            label = "✅" if decision["approved"] else "❌"
            score_text = f" ({score:.0f}/100)" if score is not None else ""
            lines.append(
                f"{label} *{decision['symbol']}*{score_text} — {decision['reason']}"
            )
            if decision["approved"] and candidate.get("reasons"):
                lines.append(f"   Why: {'; '.join(candidate['reasons'][:2])}")
    elif payload["opportunities"]:
        lines.append("\n*Final chart setups* ")
        for candidate in payload["opportunities"][:10]:
            lines.append(
                f"• *{candidate['symbol']}* {candidate['direction']} "
                f"({candidate['confidence']:.0f}/100) — option decision unavailable"
            )
    else:
        lines.append("\nNo setup cleared the full chart and risk funnel today.")

    combined = dict(payload.get("blockers", {}))
    for reason, count in payload.get("filtered_out", {}).items():
        combined[reason] = combined.get(reason, 0) + count
    if payload.get("stale"):
        combined["signal was stale"] = len(payload["stale"])
    if combined:
        lines.append("\n*Most common rejection reasons* ")
        for reason, count in sorted(combined.items(), key=lambda item: -item[1])[:5]:
            lines.append(f"• {count:,}× {reason}")

    lines.append(
        f"\nOpen now: {len(database.positions.all())} position(s), "
        f"{len(database.orders.open_orders())} order(s). Paper trading only."
    )
    return "\n".join(lines)[:3900]
