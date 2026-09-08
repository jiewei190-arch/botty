from datetime import date, timedelta

from trading_bot.config.settings import OptionsSettings
from trading_bot.options import OptionQuote, SwingOptionSelector

TODAY = date(2026, 9, 8)


def quote(*, dte=75, ask=5.2, bid=5.0, delta=0.6, kind="call"):
    return OptionQuote(
        symbol=f"TEST-{dte}", underlying="TEST", contract_type=kind,
        expiration=TODAY + timedelta(days=dte), strike=100, bid=bid, ask=ask,
        delta=delta, daily_volume=100, open_interest=500,
    )


def test_standard_setup_is_limited_to_90_dte():
    selector = SwingOptionSelector(OptionsSettings())
    assert selector.select(
        direction="LONG", quotes=[quote(dte=100)], as_of=TODAY, confidence=89
    ) is None


def test_a_plus_setup_can_select_up_to_120_dte():
    selection = SwingOptionSelector(OptionsSettings()).select(
        direction="LONG", quotes=[quote(dte=110)], as_of=TODAY, confidence=92
    )
    assert selection is not None
    assert selection.days_to_expiry == 110


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
