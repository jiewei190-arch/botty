"""The risk gate: every limit, and the guarantee that nothing bypasses it."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from trading_bot.config.settings import RiskSettings
from trading_bot.data.database import Database
from trading_bot.risk import (
    OpenPosition,
    PortfolioState,
    RiskManager,
    build_portfolio_state,
    session_start,
)
from trading_bot.strategies import Signal, SignalDirection

NOW = datetime(2026, 9, 5, 15, 0, tzinfo=timezone.utc)


def a_signal(**overrides) -> Signal:
    payload = {
        "symbol": "AAPL",
        "direction": SignalDirection.LONG,
        "strategy": "momentum",
        "confidence": 82.0,
        "entry_price": 210.50,
        "stop_loss": 207.00,
        "take_profit": 218.00,
        "timestamp": NOW,
    }
    payload.update(overrides)
    return Signal(**payload)


def a_portfolio(**overrides) -> PortfolioState:
    payload = {"equity": Decimal("10000"), "buying_power": Decimal("20000")}
    payload.update(overrides)
    return PortfolioState(**payload)


def a_position(symbol: str, price: float = 100.0, qty: int = 5) -> OpenPosition:
    return OpenPosition(
        symbol, "LONG", Decimal(qty), Decimal(str(price)), Decimal(str(price))
    )


@pytest.fixture
def manager() -> RiskManager:
    return RiskManager()


# ============================================================================
# Approval
# ============================================================================


def test_a_sound_signal_is_approved_and_sized(manager):
    decision = manager.evaluate(a_signal(), a_portfolio(), now=NOW)
    assert decision.approved
    assert decision.shares > 0
    assert decision.risk_amount > 0
    assert decision.rejection_reason is None


def test_an_approval_reports_every_check(manager):
    decision = manager.evaluate(a_signal(), a_portfolio(), now=NOW)
    names = {check.name for check in decision.checks}
    assert names == {
        "account_tradable", "daily_loss", "cooldown", "open_positions",
        "duplicate", "confidence", "risk_reward", "position_size", "exposure",
    }
    assert all(check.passed for check in decision.checks)


def test_risked_amount_respects_the_configured_percentage(manager):
    decision = manager.evaluate(
        a_signal(entry_price=100.0, stop_loss=98.0), a_portfolio(), now=NOW
    )
    assert decision.risk_amount <= Decimal("100")     # 1% of $10,000


# ============================================================================
# Each limit
# ============================================================================


def test_operator_kill_switch_blocks_everything(manager):
    portfolio = a_portfolio(halted=True, halt_reason="Kill switch engaged")
    decision = manager.evaluate(a_signal(), portfolio, now=NOW)
    assert not decision.approved
    assert "Kill switch" in decision.rejection_reason


def test_a_broker_block_stops_trading(manager):
    decision = manager.evaluate(a_signal(), a_portfolio(trading_blocked=True), now=NOW)
    assert not decision.approved
    assert "blocked" in decision.rejection_reason


def test_daily_loss_limit_stops_trading(manager):
    portfolio = a_portfolio(
        equity=Decimal("9600"),
        session_start_equity=Decimal("10000"),
        realized_pnl_today=Decimal("-400"),
    )
    decision = manager.evaluate(a_signal(), portfolio, now=NOW)
    assert not decision.approved
    assert "Daily loss limit" in decision.rejection_reason


def test_daily_loss_counts_unrealised_losses(manager):
    """A limit ignoring open positions could pass while the account bled."""
    losing = OpenPosition("TSLA", "LONG", Decimal("100"), Decimal("100"), Decimal("96"))
    portfolio = a_portfolio(
        session_start_equity=Decimal("10000"), positions=(losing,)
    )
    assert portfolio.daily_pnl == Decimal("-400")
    assert not manager.evaluate(a_signal(), portfolio, now=NOW).approved


def test_a_loss_within_the_limit_still_trades(manager):
    portfolio = a_portfolio(
        equity=Decimal("9800"),
        session_start_equity=Decimal("10000"),
        realized_pnl_today=Decimal("-200"),
    )
    assert manager.evaluate(a_signal(), portfolio, now=NOW).approved


def test_cooldown_blocks_after_consecutive_losses(manager):
    portfolio = a_portfolio(
        consecutive_losses=3, last_loss_at=NOW - timedelta(minutes=10)
    )
    decision = manager.evaluate(a_signal(), portfolio, now=NOW)
    assert not decision.approved
    assert "Cooling off" in decision.rejection_reason


def test_cooldown_expires(manager):
    portfolio = a_portfolio(
        consecutive_losses=3, last_loss_at=NOW - timedelta(minutes=90)
    )
    assert manager.evaluate(a_signal(), portfolio, now=NOW).approved


def test_losses_below_the_streak_limit_do_not_cool_off(manager):
    portfolio = a_portfolio(consecutive_losses=2, last_loss_at=NOW)
    assert manager.evaluate(a_signal(), portfolio, now=NOW).approved


def test_an_unknown_last_loss_time_fails_closed(manager):
    """Streak reached but no timestamp: refuse rather than assume it expired."""
    portfolio = a_portfolio(consecutive_losses=5, last_loss_at=None)
    assert not manager.evaluate(a_signal(), portfolio, now=NOW).approved


def test_position_slots_are_finite(manager):
    portfolio = a_portfolio(positions=tuple(a_position(s) for s in "ABCDE"))
    decision = manager.evaluate(a_signal(), portfolio, now=NOW)
    assert not decision.approved
    assert "slots are in use" in decision.rejection_reason


def test_adding_to_an_existing_position_is_refused(manager):
    portfolio = a_portfolio(positions=(a_position("AAPL"),))
    decision = manager.evaluate(a_signal(), portfolio, now=NOW)
    assert not decision.approved
    assert "Already holding" in decision.rejection_reason


def test_duplicate_detection_ignores_case(manager):
    portfolio = a_portfolio(positions=(a_position("aapl"),))
    assert not manager.evaluate(a_signal(symbol="AAPL"), portfolio, now=NOW).approved


def test_low_confidence_is_rejected(manager):
    decision = manager.evaluate(a_signal(confidence=45), a_portfolio(), now=NOW)
    assert not decision.approved
    assert "below the 60 floor" in decision.rejection_reason


def test_poor_reward_to_risk_is_rejected(manager):
    decision = manager.evaluate(a_signal(take_profit=213.00), a_portfolio(), now=NOW)
    assert not decision.approved
    assert "Reward:risk" in decision.rejection_reason


def test_a_position_too_small_to_trade_is_rejected(manager):
    tiny = a_portfolio(equity=Decimal("300"), buying_power=Decimal("300"))
    decision = manager.evaluate(a_signal(), tiny, now=NOW)
    assert not decision.approved
    assert decision.shares == 0


def test_exposure_limit_is_enforced(manager):
    """Already over-exposed: sizing yields nothing and the check reports why."""
    heavy = tuple(
        OpenPosition(s, "LONG", Decimal("70"), Decimal("100"), Decimal("100"))
        for s in ("A", "B")
    )
    portfolio = a_portfolio(positions=heavy)          # $14,000 on $10,000 equity
    decision = manager.evaluate(a_signal(), portfolio, now=NOW)
    assert not decision.approved


# ============================================================================
# The gate itself
# ============================================================================


def test_a_rejected_decision_carries_no_quantity(manager):
    """No size means nothing for the execution layer to send."""
    decision = manager.evaluate(a_signal(confidence=10), a_portfolio(), now=NOW)
    assert not decision.approved
    assert decision.quantity == Decimal("0")
    assert decision.shares == 0


def test_rejection_reason_names_the_first_failing_check(manager):
    portfolio = a_portfolio(halted=True, halt_reason="Maintenance")
    decision = manager.evaluate(a_signal(confidence=5), portfolio, now=NOW)
    assert decision.rejection_reason == "Maintenance"
    assert {"account_tradable", "confidence"} <= {c.name for c in decision.failed_checks}


def test_a_rejected_signal_still_reports_its_would_be_size(manager):
    """Exactly what you need to decide which limit to tune."""
    decision = manager.evaluate(a_signal(confidence=10), a_portfolio(), now=NOW)
    assert decision.sizing is not None
    assert decision.sizing.quantity > 0        # sizing was fine; confidence was not


def test_decision_serializes(manager):
    payload = manager.evaluate(a_signal(), a_portfolio(), now=NOW).as_dict()
    assert set(payload) >= {"approved", "symbol", "quantity", "checks", "sizing"}


def test_summary_reads_as_a_log_line(manager):
    approved = manager.evaluate(a_signal(), a_portfolio(), now=NOW).summary()
    rejected = manager.evaluate(a_signal(confidence=1), a_portfolio(), now=NOW).summary()
    assert approved.startswith("APPROVED AAPL")
    assert rejected.startswith("REJECTED AAPL")


# ============================================================================
# Halt gate
# ============================================================================


def test_trading_halted_reports_none_when_clear(manager):
    assert manager.trading_halted(a_portfolio()) is None


@pytest.mark.parametrize(
    "portfolio_kwargs",
    [
        {"halted": True, "halt_reason": "Manual stop"},
        {"trading_blocked": True},
        {
            "equity": Decimal("9000"),
            "session_start_equity": Decimal("10000"),
            "realized_pnl_today": Decimal("-1000"),
        },
    ],
)
def test_trading_halted_detects_each_stop_condition(manager, portfolio_kwargs):
    assert manager.trading_halted(a_portfolio(**portfolio_kwargs)) is not None


# ============================================================================
# Batch evaluation
# ============================================================================


def test_a_batch_cannot_overfill_the_position_slots():
    """Judged against the original snapshot, six signals would all be approved."""
    manager = RiskManager(RiskSettings(max_open_positions=2))
    signals = [
        a_signal(symbol=symbol, entry_price=50.0, stop_loss=49.0, take_profit=53.0)
        for symbol in ("AAA", "BBB", "CCC", "DDD")
    ]
    decisions = manager.evaluate_many(signals, a_portfolio(), now=NOW)
    assert sum(1 for decision in decisions if decision.approved) == 2


def test_a_batch_spends_slots_on_the_strongest_signals():
    manager = RiskManager(RiskSettings(max_open_positions=1))
    weak = a_signal(symbol="WEAK", confidence=65, entry_price=50.0,
                    stop_loss=49.0, take_profit=53.0)
    strong = a_signal(symbol="STRONG", confidence=95, entry_price=50.0,
                      stop_loss=49.0, take_profit=53.0)
    decisions = manager.evaluate_many([weak, strong], a_portfolio(), now=NOW)
    approved = [d.signal.symbol for d in decisions if d.approved]
    assert approved == ["STRONG"]


def test_a_batch_accumulates_exposure():
    """Each approval must reduce the headroom the next one sees."""
    manager = RiskManager(RiskSettings(max_portfolio_exposure_pct=30.0))
    signals = [
        a_signal(symbol=symbol, entry_price=100.0, stop_loss=99.0, take_profit=103.0)
        for symbol in ("AAA", "BBB", "CCC", "DDD", "EEE")
    ]
    decisions = manager.evaluate_many(signals, a_portfolio(), now=NOW)
    total = sum((d.position_value for d in decisions if d.approved), Decimal("0"))
    assert total <= Decimal("10000") * Decimal("0.30")


def test_a_batch_returns_a_decision_for_every_signal():
    manager = RiskManager(RiskSettings(max_open_positions=1))
    signals = [a_signal(symbol=s) for s in ("AAA", "BBB", "CCC")]
    assert len(manager.evaluate_many(signals, a_portfolio(), now=NOW)) == 3


# ============================================================================
# Portfolio state assembly
# ============================================================================


def test_state_builds_from_broker_positions():
    portfolio = build_portfolio_state(
        equity=10_000,
        broker_positions=[
            {"symbol": "aapl", "qty": 10, "avg_entry_price": 200, "current_price": 210,
             "side": "long"},
        ],
    )
    assert portfolio.open_count == 1
    assert portfolio.has_position("AAPL")
    assert portfolio.total_exposure == Decimal("2100")
    assert portfolio.unrealized_pnl == Decimal("100")


def test_state_skips_unreadable_positions_rather_than_failing():
    """One bad row must not blind every limit."""
    portfolio = build_portfolio_state(
        equity=10_000,
        broker_positions=[{"nonsense": True}, {"symbol": "OK", "qty": 1,
                                               "avg_entry_price": 10, "current_price": 10}],
    )
    assert portfolio.open_count == 1


def test_state_reads_history_from_the_database(tmp_path):
    database = Database(tmp_path / "risk.db")
    database.initialize()
    for exit_price in (98.0, 98.0):
        trade_id = database.trades.open_trade(
            symbol="AAPL", direction="LONG", qty=10, entry_price=100.0, stop_loss=98.0
        )
        database.trades.close_trade(trade_id, exit_price=exit_price)

    portfolio = build_portfolio_state(equity=10_000, database=database)
    assert portfolio.consecutive_losses == 2
    assert portfolio.realized_pnl_today == Decimal("-40")
    assert portfolio.last_loss_at is not None
    database.close()


def test_state_survives_a_broken_database():
    class Broken:
        class trades:  # noqa: N801 - mirrors the Database attribute path
            @staticmethod
            def realized_pnl_since(_):
                raise RuntimeError("database is gone")

    portfolio = build_portfolio_state(equity=10_000, database=Broken())
    assert portfolio.equity == Decimal("10000")
    assert portfolio.consecutive_losses == 0


def test_session_start_precedes_the_us_open():
    boundary = session_start(datetime(2026, 9, 5, 15, 0, tzinfo=timezone.utc))
    assert boundary.hour == 8
    assert boundary.date() == datetime(2026, 9, 5).date()


def test_session_start_rolls_back_before_the_boundary():
    boundary = session_start(datetime(2026, 9, 5, 3, 0, tzinfo=timezone.utc))
    assert boundary.date() == datetime(2026, 9, 4).date()


def test_risk_remaining_is_zero_once_the_stop_is_beyond_price():
    """A stop moved to breakeven or better carries no further downside."""
    protected = OpenPosition(
        "AAPL", "LONG", Decimal("10"), Decimal("100"), Decimal("110"),
        stop_loss=Decimal("112"),
    )
    assert protected.risk_remaining == Decimal("0")


def test_portfolio_serializes():
    payload = a_portfolio().as_dict()
    assert set(payload) >= {"equity", "open_positions", "daily_pnl", "exposure_pct"}
