from datetime import date, datetime, timezone
from types import SimpleNamespace

from trading_bot.automation.daily_summary import format_close_summary, save_scan_report


def test_daily_summary_explains_approved_and_rejected_candidates(database):
    signal = lambda symbol: SimpleNamespace(  # noqa: E731
        symbol=symbol, direction=SimpleNamespace(value="LONG"), strategy="momentum",
        risk_reward_ratio=2.5, reasons=("trend aligned", "volume confirmed"),
    )
    opportunities = (
        SimpleNamespace(signal=signal("META"), confidence=93),
        SimpleNamespace(signal=signal("NVDA"), confidence=88),
    )
    universe = SimpleNamespace(static_report=SimpleNamespace(considered=11_000))
    sweep = SimpleNamespace(
        opportunities=opportunities, universe_size=2_400, scanned=2_400,
        elapsed_seconds=42.5, blockers={"momentum.volume": 70},
        filtered_out={"reward:risk below 2:1": 3}, stale={}, halt_reason=None,
    )
    execution = SimpleNamespace(
        placed=1, skipped=1, failed=0,
        decisions=(
            SimpleNamespace(symbol="META", approved=True, reason="approved: contract submitted"),
            SimpleNamespace(symbol="NVDA", approved=False, reason="not approved: spread too wide"),
        ),
    )

    session = date(2026, 9, 9)
    save_scan_report(database, session, universe, sweep, execution)
    message = format_close_summary(database, session)

    assert "BOTTY END-OF-DAY SUMMARY" in message
    assert "META" in message and "approved: contract submitted" in message
    assert "NVDA" in message and "not approved: spread too wide" in message
    assert "momentum.volume" in message


def test_daily_summary_handles_no_setups(database):
    universe = SimpleNamespace(static_report=SimpleNamespace(considered=9_000))
    sweep = SimpleNamespace(
        opportunities=(), universe_size=2_000, scanned=2_000, elapsed_seconds=10,
        blockers={"breakout.trigger": 1_950}, filtered_out={}, stale={},
        halt_reason=None,
    )
    session = datetime.now(timezone.utc).date()
    save_scan_report(database, session, universe, sweep)

    message = format_close_summary(database, session)

    assert "No setup cleared" in message
    assert "breakout.trigger" in message
