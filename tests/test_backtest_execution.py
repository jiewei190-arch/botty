"""Fill-model tests — the assumptions that decide whether a backtest is honest.

Each test here pins one of the three claims in ``execution``'s module docstring:
entries fill at the *next* bar's open, stops are triggers rather than guaranteed
prices, and costs are paid on both sides and always against the trade.
"""

from __future__ import annotations

import pandas as pd
import pytest

from trading_bot.backtesting import (
    FRICTIONLESS,
    CostModel,
    Fill,
    FillModel,
    FillReason,
)
from trading_bot.strategies import SignalDirection

LONG = SignalDirection.LONG
SHORT = SignalDirection.SHORT


def bar(open_=100.0, high=101.0, low=99.0, close=100.5, volume=1_000_000.0):
    return pd.Series(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


class TestCostModel:
    def test_rejects_negative_components(self):
        for field in (
            "commission_per_trade",
            "commission_per_share",
            "commission_pct",
            "slippage_pct",
            "min_commission",
        ):
            with pytest.raises(ValueError, match=field):
                CostModel(**{field: -1.0})

    def test_commission_sums_its_three_components(self):
        costs = CostModel(
            commission_per_trade=1.0, commission_per_share=0.005, commission_pct=0.01
        )
        # 1.00 flat + 100 x 0.005 + 0.01% of 100 x 50
        assert costs.commission(100, 50.0) == pytest.approx(1.0 + 0.5 + 0.5)

    def test_zero_commission_ignores_the_minimum(self):
        """A broker charging nothing charges nothing, floor or no floor."""
        assert CostModel(min_commission=1.0).commission(100, 50.0) == 0.0

    def test_minimum_applies_once_any_component_is_charged(self):
        costs = CostModel(commission_per_share=0.001, min_commission=1.0)
        assert costs.commission(10, 50.0) == 1.0  # 0.01 raised to the floor
        assert costs.commission(10_000, 50.0) == 10.0  # above the floor, untouched

    @pytest.mark.parametrize(
        ("direction", "is_entry", "expect_higher"),
        [
            (LONG, True, True),  # buying to open  -> pay up
            (LONG, False, False),  # selling to close -> receive less
            (SHORT, True, False),  # selling to open  -> receive less
            (SHORT, False, True),  # buying to close  -> pay up
        ],
    )
    def test_slippage_always_moves_against_the_trade(
        self, direction, is_entry, expect_higher
    ):
        costs = CostModel(slippage_pct=0.1)
        slipped = costs.slip(100.0, direction, is_entry=is_entry)
        assert (slipped > 100.0) is expect_higher
        assert slipped == pytest.approx(100.1 if expect_higher else 99.9)

    def test_frictionless_is_free(self):
        assert FRICTIONLESS.commission(1000, 100.0) == 0.0
        assert FRICTIONLESS.slip(100.0, LONG, is_entry=True) == 100.0


class TestEntryFills:
    def test_entry_fills_at_the_open_not_the_close(self):
        """The signal bar's close is unavailable; the next bar's open is not."""
        fill = FillModel(FRICTIONLESS).entry_fill(bar(open_=100.0, close=105.0), LONG, 10)
        assert fill.price == 100.0
        assert fill.reason is FillReason.ENTRY

    def test_long_entry_pays_up(self):
        fill = FillModel(CostModel(slippage_pct=0.05)).entry_fill(bar(), LONG, 100)
        assert fill.price == pytest.approx(100.05)
        assert fill.slippage_cost == pytest.approx(0.05 * 100)

    def test_short_entry_receives_less(self):
        fill = FillModel(CostModel(slippage_pct=0.05)).entry_fill(bar(), SHORT, 100)
        assert fill.price == pytest.approx(99.95)

    def test_notional_uses_absolute_quantity(self):
        fill = Fill(price=10.0, quantity=-5, commission=0.0, reason=FillReason.ENTRY)
        assert fill.notional == 50.0


class TestStopFills:
    def test_ordinary_stop_fills_at_the_stop(self):
        """Bar trades through the stop without gapping — filled at the trigger."""
        fill = FillModel(FRICTIONLESS).stop_fill(
            bar(open_=100.0, low=97.0), LONG, 100, stop_price=98.0
        )
        assert fill.price == 98.0
        assert fill.gapped is False

    def test_gap_through_a_long_stop_fills_at_the_open(self):
        """A stop is a trigger, not a promise. Opening below it costs the gap."""
        fill = FillModel(FRICTIONLESS).stop_fill(
            bar(open_=94.0, high=95.0, low=93.0), LONG, 100, stop_price=98.0
        )
        assert fill.price == 94.0
        assert fill.gapped is True

    def test_gap_through_a_short_stop_fills_at_the_open(self):
        fill = FillModel(FRICTIONLESS).stop_fill(
            bar(open_=104.0, high=105.0, low=103.0), SHORT, 100, stop_price=102.0
        )
        assert fill.price == 104.0
        assert fill.gapped is True

    def test_gapped_fill_is_never_better_than_the_stop(self):
        """The whole point: modelling the gap must not flatter the result."""
        model = FillModel(FRICTIONLESS)
        gapped = model.stop_fill(bar(open_=90.0, low=89.0), LONG, 100, stop_price=98.0)
        clean = model.stop_fill(bar(open_=100.0, low=97.0), LONG, 100, stop_price=98.0)
        assert gapped.price < clean.price

    def test_stop_exit_slippage_moves_against_the_trade(self):
        fill = FillModel(CostModel(slippage_pct=0.1)).stop_fill(
            bar(open_=100.0, low=97.0), LONG, 100, stop_price=98.0
        )
        assert fill.price == pytest.approx(98.0 * 0.999)  # sold lower than the stop


class TestTargetFills:
    def test_target_fills_exactly_at_the_limit(self):
        """Conservative on purpose: a resting limit does not capture the overshoot."""
        fill = FillModel(FRICTIONLESS).target_fill(
            bar(open_=100.0, high=120.0), LONG, 100, target_price=105.0
        )
        assert fill.price == 105.0
        assert fill.slippage_cost == 0.0

    def test_target_is_not_improved_by_slippage(self):
        fill = FillModel(CostModel(slippage_pct=0.5)).target_fill(
            bar(high=120.0), LONG, 100, target_price=105.0
        )
        assert fill.price == 105.0

    def test_target_still_pays_commission(self):
        fill = FillModel(CostModel(commission_per_trade=1.0)).target_fill(
            bar(), LONG, 100, target_price=105.0
        )
        assert fill.commission == 1.0


class TestExitFills:
    def test_discretionary_exit_uses_the_open(self):
        fill = FillModel(FRICTIONLESS).exit_fill(bar(open_=100.0, close=90.0), LONG, 100)
        assert fill.price == 100.0
        assert fill.reason is FillReason.SIGNAL_EXIT

    def test_reason_is_carried_through(self):
        fill = FillModel(FRICTIONLESS).exit_fill(
            bar(), LONG, 100, FillReason.END_OF_DATA
        )
        assert fill.reason is FillReason.END_OF_DATA

    def test_every_reason_but_entry_is_an_exit(self):
        assert not FillReason.ENTRY.is_exit
        assert all(
            reason.is_exit for reason in FillReason if reason is not FillReason.ENTRY
        )
