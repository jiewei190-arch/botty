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

    # The per-symbol verdicts and the detector histogram are dashboard
    # material; repeating them here made the closing message a second copy of
    # a screen the reader already has, and pushed the market recap out of view.
    assert "BOTTY" in message
    assert "META" not in message
    assert "momentum.volume" not in message
    assert "Per-symbol detail is on the dashboard" in message
    # What it must still carry: did Botty work, and what did it do.
    assert "Paper orders: 1" in message


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
    assert "breakout.trigger" not in message


def test_the_summary_counts_causes_rather_than_listing_symbols(database):
    """A quiet day must say why, without re-listing what the dashboard shows."""
    universe = SimpleNamespace(static_report=SimpleNamespace(considered=14_315))
    sweep = SimpleNamespace(
        opportunities=(), universe_size=357, scanned=357, elapsed_seconds=30.4,
        blockers={}, filtered_out={}, stale={}, halt_reason=None,
    )
    rules = "not approved: no option contract passed DTE, delta and premium rules"
    execution = SimpleNamespace(
        placed=0, alerted=0, skipped=3, failed=2,
        decisions=(
            SimpleNamespace(symbol="VG", approved=False, reason=rules),
            SimpleNamespace(symbol="KEYS", approved=False, reason=rules),
            SimpleNamespace(symbol="ELV", approved=False, reason=rules),
            SimpleNamespace(symbol="NET", approved=False, reason=(
                'not approved: option execution failed ({"message":"symbol limit is 100"})'
            )),
            SimpleNamespace(symbol="CI", approved=False, reason=(
                'not approved: option execution failed ({"message":"symbol limit is 100"})'
            )),
        ),
    )
    session = date(2026, 9, 11)
    save_scan_report(database, session, universe, sweep, execution)

    message = format_close_summary(database, session)

    assert "3× no option contract passed DTE, delta and premium rules" in message
    assert "2× option execution failed" in message
    # The two execution failures carried different raw payloads but one cause.
    assert "symbol limit is 100" not in message
    for symbol in ("VG", "KEYS", "ELV", "NET", "CI"):
        assert symbol not in message


def test_a_day_that_alerted_does_not_explain_itself(database):
    """The 'why nothing fired' line is for quiet days only."""
    universe = SimpleNamespace(static_report=SimpleNamespace(considered=14_315))
    sweep = SimpleNamespace(
        opportunities=(), universe_size=357, scanned=357, elapsed_seconds=30.4,
        blockers={}, filtered_out={}, stale={}, halt_reason=None,
    )
    execution = SimpleNamespace(
        placed=0, alerted=2, skipped=1, failed=0,
        decisions=(
            SimpleNamespace(
                symbol="NET", approved=False, reason="not approved: risk checks failed"
            ),
        ),
    )
    session = date(2026, 9, 11)
    save_scan_report(database, session, universe, sweep, execution)

    message = format_close_summary(database, session)

    assert "Option alerts: 2" in message
    assert "Nothing was alerted because" not in message


def test_the_summary_stays_short_enough_to_read_on_a_phone(database):
    universe = SimpleNamespace(static_report=SimpleNamespace(considered=14_315))
    sweep = SimpleNamespace(
        opportunities=(), universe_size=357, scanned=357, elapsed_seconds=30.4,
        blockers={f"detector.{i}": 900 - i for i in range(40)},
        filtered_out={}, stale={}, halt_reason=None,
    )
    execution = SimpleNamespace(
        placed=0, alerted=0, skipped=30, failed=0,
        decisions=tuple(
            SimpleNamespace(symbol=f"S{i}", approved=False, reason=f"not approved: reason {i % 4}")
            for i in range(30)
        ),
    )
    session = date(2026, 9, 11)
    save_scan_report(database, session, universe, sweep, execution)

    message = format_close_summary(database, session)

    # Thirty rejected candidates and forty detector reasons still collapse to
    # a handful of lines: at most three causes are named.
    assert len(message.splitlines()) <= 12
