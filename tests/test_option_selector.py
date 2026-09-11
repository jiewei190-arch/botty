from datetime import date, timedelta

from trading_bot.config.settings import OptionsSettings
from trading_bot.options import OptionQuote, SwingOptionSelector

TODAY = date(2026, 9, 8)


SETTINGS = OptionsSettings()


def quote(*, dte=SETTINGS.target_dte, ask=5.2, bid=5.0, delta=0.6, kind="call"):
    return OptionQuote(
        symbol=f"TEST-{dte}", underlying="TEST", contract_type=kind,
        expiration=TODAY + timedelta(days=dte), strike=100, bid=bid, ask=ask,
        delta=delta, daily_volume=100, open_interest=500,
    )


def test_contracts_past_the_ordinary_window_are_rejected():
    selector = SwingOptionSelector(SETTINGS)
    assert selector.select(
        direction="LONG", quotes=[quote(dte=SETTINGS.max_dte + 1)],
        as_of=TODAY, confidence=SETTINGS.exceptional_min_confidence - 1,
    ) is None


def test_high_confidence_extends_the_window_but_not_past_its_own_limit():
    selector = SwingOptionSelector(SETTINGS)
    assert selector.select(
        direction="LONG", quotes=[quote(dte=SETTINGS.max_dte + 1)],
        as_of=TODAY, confidence=SETTINGS.exceptional_min_confidence + 1,
    ) is not None
    assert selector.select(
        direction="LONG", quotes=[quote(dte=SETTINGS.exceptional_max_dte + 1)],
        as_of=TODAY, confidence=SETTINGS.exceptional_min_confidence + 1,
    ) is None


def test_the_shortest_contract_still_outlives_the_holding_window():
    """The floor is derived, not chosen: hold + exit buffer.

    A contract that expires inside the planned hold is a losing trade decided
    on the day it is opened, whatever the stock does.
    """
    assert SETTINGS.min_dte >= (
        SETTINGS.planned_max_hold_days + SETTINGS.exit_before_expiry_days
    )
    selection = SwingOptionSelector(SETTINGS).select(
        direction="LONG", quotes=[quote(dte=SETTINGS.min_dte)], as_of=TODAY, confidence=80
    )
    assert selection is not None
    assert selection.days_to_expiry == SETTINGS.min_dte


def test_contracts_below_the_floor_are_rejected():
    assert SwingOptionSelector(SETTINGS).select(
        direction="LONG", quotes=[quote(dte=SETTINGS.min_dte - 1)],
        as_of=TODAY, confidence=80
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
