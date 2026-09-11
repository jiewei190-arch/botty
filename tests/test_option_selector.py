from datetime import date, timedelta

import pytest

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


# ---------------------------------------------------------------------------
# The premium ceiling priced large caps out entirely
#
# $1,000 with at most 4 contracts means a contract must cost $125-$1,000. At
# 0.50-0.70 delta and 60 DTE that is roughly 5-7% of spot, so every underlying
# much above $200 was excluded before any judgement was applied -- silently,
# and exactly where the liquid movers are. The relaxed floor buys further out
# of the money instead, and says that it did.
# ---------------------------------------------------------------------------


def _graded_chain(*, prices, as_of, deltas):
    """One expiry, contracts at the given asks and deltas."""
    expiry = as_of + timedelta(days=60)
    return [
        OptionQuote(
            symbol=f"X{i}", underlying="X", contract_type="call", expiration=expiry,
            strike=100.0 + i, bid=round(ask * 0.98, 2), ask=ask, delta=delta,
            open_interest=500, daily_volume=100, underlying_price=400.0,
        )
        for i, (ask, delta) in enumerate(zip(prices, deltas, strict=True))
    ]


def test_the_normal_band_wins_when_it_is_affordable():
    as_of = date.today()
    selector = SwingOptionSelector(OptionsSettings())
    # A 0.60-delta contract at $7.00 fits the budget; a 0.38 one also would.
    quotes = _graded_chain(prices=[7.00, 4.00], deltas=[0.60, 0.38], as_of=as_of)

    selection = selector.select(direction="LONG", quotes=quotes, as_of=as_of)

    assert selection is not None
    assert selection.quote.delta == 0.60
    assert not selection.relaxed


def test_a_cheaper_contract_is_taken_when_the_band_prices_out():
    """The ELV case: every 0.50+ contract costs more than the cap."""
    as_of = date.today()
    selector = SwingOptionSelector(OptionsSettings())
    # 0.60 delta at $25.00 is $2,500 — over the $1,000 ceiling. 0.38 at $8 fits.
    quotes = _graded_chain(prices=[25.00, 8.00], deltas=[0.60, 0.38], as_of=as_of)

    selection = selector.select(direction="LONG", quotes=quotes, as_of=as_of)

    assert selection is not None
    assert selection.quote.delta == 0.38
    assert selection.relaxed
    assert selection.estimated_cost == pytest.approx(800.0)


def test_a_relaxed_selection_says_so_in_its_description():
    as_of = date.today()
    selector = SwingOptionSelector(OptionsSettings())
    quotes = _graded_chain(prices=[25.00, 8.00], deltas=[0.60, 0.38], as_of=as_of)

    selection = selector.select(direction="LONG", quotes=quotes, as_of=as_of)

    assert "[reduced delta]" in selection.describe()


def test_the_relaxed_floor_still_has_a_floor():
    """Cheap is not the goal; below the relaxed floor nothing is bought."""
    as_of = date.today()
    selector = SwingOptionSelector(OptionsSettings())
    # Only a 0.20-delta contract is affordable — under the 0.35 relaxed floor.
    quotes = _graded_chain(prices=[25.00, 6.00], deltas=[0.60, 0.20], as_of=as_of)

    assert selector.select(direction="LONG", quotes=quotes, as_of=as_of) is None


def test_relaxing_never_loosens_liquidity_or_expiry():
    """Only delta relaxes. A wide spread stays rejected in both passes."""
    as_of = date.today()
    expiry = as_of + timedelta(days=60)
    quotes = [OptionQuote(
        symbol="X", underlying="X", contract_type="call", expiration=expiry,
        strike=100.0, bid=4.00, ask=8.00, delta=0.38,   # 66% spread
        open_interest=500, daily_volume=100, underlying_price=400.0,
    )]

    selector = SwingOptionSelector(OptionsSettings())

    assert selector.select(direction="LONG", quotes=quotes, as_of=as_of) is None


def test_the_relaxed_floor_must_sit_below_the_normal_one():
    with pytest.raises(ValueError, match="relaxed_min_abs_delta"):
        OptionsSettings(relaxed_min_abs_delta=0.55)
