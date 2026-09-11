from datetime import date, timedelta

from trading_bot.config.settings import OptionsSettings
from trading_bot.options import OptionQuote, SwingOptionSelector

TODAY = date(2026, 9, 8)


def quote(*, dte=30, ask=5.2, bid=5.0, delta=0.6, kind="call"):
    return OptionQuote(
        symbol=f"TEST-{dte}", underlying="TEST", contract_type=kind,
        expiration=TODAY + timedelta(days=dte), strike=100, bid=bid, ask=ask,
        delta=delta, daily_volume=100, open_interest=500,
    )


def test_contracts_over_60_dte_are_rejected():
    selector = SwingOptionSelector(OptionsSettings())
    assert selector.select(
        direction="LONG", quotes=[quote(dte=61)], as_of=TODAY, confidence=89
    ) is None


def test_high_confidence_does_not_extend_past_60_dte():
    assert SwingOptionSelector(OptionsSettings()).select(
        direction="LONG", quotes=[quote(dte=61)], as_of=TODAY, confidence=92
    ) is None


def test_seven_dte_boundary_is_allowed():
    selection = SwingOptionSelector(OptionsSettings()).select(
        direction="LONG", quotes=[quote(dte=7)], as_of=TODAY, confidence=80
    )
    assert selection is not None
    assert selection.days_to_expiry == 7


def test_contracts_below_seven_dte_are_rejected():
    assert SwingOptionSelector(OptionsSettings()).select(
        direction="LONG", quotes=[quote(dte=6)], as_of=TODAY, confidence=80
    ) is None


def test_budget_is_hard_bounded_between_500_and_1000():
    selector = SwingOptionSelector(OptionsSettings())
    selection = selector.select(
        direction="LONG", quotes=[quote(ask=3.0, bid=2.9)], as_of=TODAY, confidence=80
    )
    assert selection is not None
    assert selection.quantity == 2
    assert selection.estimated_cost == 600


def test_contract_is_rejected_when_four_cannot_reach_minimum_budget():
    selector = SwingOptionSelector(OptionsSettings())
    assert selector.select(
        direction="LONG", quotes=[quote(ask=1.0, bid=0.98)], as_of=TODAY, confidence=80
    ) is None


def test_wide_spread_and_wrong_direction_are_rejected():
    selector = SwingOptionSelector(OptionsSettings())
    assert selector.select(
        direction="LONG", quotes=[quote(bid=3, ask=5), quote(kind="put")],
        as_of=TODAY, confidence=95,
    ) is None
