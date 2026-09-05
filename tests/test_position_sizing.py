"""Position sizing arithmetic.

Expected values are computed by hand from the formula, not by calling the
function a second way.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from trading_bot.risk import SizingConstraint, calculate_position_size, to_decimal


def size(**overrides):
    """Size a position with sensible defaults and no caps but the risk budget."""
    payload = {
        "entry_price": 100.0,
        "stop_loss": 98.0,
        "equity": 10_000,
        "max_risk_per_trade_pct": 1.0,
        "max_position_size_pct": 100.0,
        "max_portfolio_exposure_pct": 100.0,
    }
    payload.update(overrides)
    return calculate_position_size(**payload)


# ============================================================================
# The core formula
# ============================================================================


def test_matches_the_hand_computed_example():
    """$10,000 at 1% is a $100 budget; a $3.50 stop distance buys 28 shares.

    100 / 3.50 = 28.571…, floored to 28, risking 28 × 3.50 = $98.
    """
    result = size(entry_price=210.50, stop_loss=207.00)
    assert result.risk_budget == Decimal("100.0")
    assert result.risk_per_share == Decimal("3.5")
    assert result.quantity == Decimal("28")
    assert result.risk_amount == Decimal("98.0")


def test_a_wider_stop_buys_fewer_shares():
    """The whole point: risk stays constant as the stop moves."""
    tight = size(stop_loss=99.0)     # $1 risk  -> 100 shares
    wide = size(stop_loss=95.0)      # $5 risk  -> 20 shares
    assert tight.quantity == Decimal("100")
    assert wide.quantity == Decimal("20")
    assert tight.risk_amount == wide.risk_amount == Decimal("100")


def test_risk_scales_with_the_percentage():
    assert size(max_risk_per_trade_pct=2.0).quantity == Decimal("100")
    assert size(max_risk_per_trade_pct=0.5).quantity == Decimal("25")


def test_risk_scales_with_equity():
    assert size(equity=20_000).quantity == Decimal("100")
    assert size(equity=5_000).quantity == Decimal("25")


def test_short_trades_size_identically():
    """Direction is irrelevant; only the distance to the stop matters."""
    long_side = size(entry_price=100.0, stop_loss=98.0)
    short_side = size(entry_price=100.0, stop_loss=102.0)
    assert long_side.quantity == short_side.quantity


def test_actual_risk_never_exceeds_the_budget():
    for stop in (99.3, 97.7, 94.1, 88.8):
        result = size(stop_loss=stop)
        assert result.risk_amount <= result.risk_budget


# ============================================================================
# Caps
# ============================================================================


def test_position_size_cap_can_bind_before_the_risk_budget():
    """On a small account a large-priced stock hits the position cap first."""
    result = size(entry_price=210.50, stop_loss=207.00, max_position_size_pct=20.0)
    assert result.quantity == Decimal("9")            # $2,000 / $210.50
    assert result.binding_constraint is SizingConstraint.POSITION_SIZE


def test_exposure_cap_accounts_for_positions_already_held():
    result = size(max_portfolio_exposure_pct=60.0, current_exposure=5_500)
    # $6,000 allowed - $5,500 held = $500 of headroom at $100 = 5 shares
    assert result.quantity == Decimal("5")
    assert result.binding_constraint is SizingConstraint.PORTFOLIO_EXPOSURE


def test_no_shares_when_exposure_is_already_full():
    result = size(max_portfolio_exposure_pct=60.0, current_exposure=6_000)
    assert result.quantity == Decimal("0")
    assert not result.is_tradable


def test_buying_power_caps_the_size():
    result = size(buying_power=450)
    assert result.quantity == Decimal("4")
    assert result.binding_constraint is SizingConstraint.BUYING_POWER


def test_the_smallest_cap_always_wins():
    result = size(
        entry_price=100.0, stop_loss=98.0,
        max_position_size_pct=30.0,        # 30 shares
        max_portfolio_exposure_pct=50.0,   # 50 shares
        buying_power=1_200,                # 12 shares
    )
    assert result.quantity == Decimal("12")
    assert result.binding_constraint is SizingConstraint.BUYING_POWER


def test_every_cap_is_reported_for_diagnosis():
    """"Why is my position so small?" must be answerable."""
    result = size(max_position_size_pct=20.0, buying_power=5_000)
    assert set(result.caps) >= {"risk_budget", "position_size", "buying_power"}
    assert "limited by" in result.explain()


def test_uncapped_sizing_reports_the_risk_budget_as_binding():
    assert size().binding_constraint is SizingConstraint.RISK_BUDGET


# ============================================================================
# Rounding and fractions
# ============================================================================


def test_quantity_rounds_down_never_up():
    """Rounding up would risk more than the budget allows."""
    result = size(entry_price=100.0, stop_loss=96.5)   # 100 / 3.5 = 28.57
    assert result.quantity == Decimal("28")


def test_sub_one_share_is_not_tradable_in_whole_shares():
    result = size(equity=100, entry_price=5_000.0, stop_loss=4_900.0)
    assert result.quantity == Decimal("0")
    assert not result.is_tradable
    assert result.notes


def test_fractional_shares_are_allowed_when_enabled():
    result = size(equity=100, entry_price=5_000.0, stop_loss=4_900.0, allow_fractional=True)
    assert 0 < result.quantity < 1


def test_shares_property_is_a_whole_number():
    assert isinstance(size().shares, int)


# ============================================================================
# Slippage assumption
# ============================================================================


def test_slippage_assumption_reduces_the_size():
    """Assuming a worse stop fill must never produce a larger position."""
    plain = size(entry_price=100.0, stop_loss=98.0)
    buffered = size(entry_price=100.0, stop_loss=98.0, slippage_pct=0.5)
    assert buffered.quantity < plain.quantity
    assert buffered.risk_per_share > plain.risk_per_share
    assert any("slippage" in note for note in buffered.notes)


def test_no_slippage_assumption_by_default():
    """The size is exactly what the stated stop implies — no hidden adjustment."""
    result = size(entry_price=100.0, stop_loss=98.0)
    assert result.risk_per_share == Decimal("2")
    assert not result.notes


# ============================================================================
# Validation
# ============================================================================


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"entry_price": 0}, "entry_price"),
        ({"entry_price": -5}, "entry_price"),
        ({"stop_loss": 0}, "stop_loss"),
        ({"equity": 0}, "equity"),
        ({"equity": -100}, "equity"),
        ({"max_risk_per_trade_pct": 0}, "max_risk_per_trade_pct"),
        ({"max_risk_per_trade_pct": 150}, "max_risk_per_trade_pct"),
        ({"slippage_pct": -1}, "slippage_pct"),
    ],
)
def test_invalid_inputs_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        size(**overrides)


def test_a_zero_width_stop_is_rejected():
    """It would divide by zero and imply an infinite position."""
    with pytest.raises(ValueError, match="cannot be equal"):
        size(entry_price=100.0, stop_loss=100.0)


# ============================================================================
# Decimal handling
# ============================================================================


def test_prices_avoid_binary_float_artefacts():
    assert to_decimal(0.1) == Decimal("0.1")
    assert to_decimal("2.30") == Decimal("2.30")
    assert to_decimal(None) == Decimal("0")


def test_unparseable_values_fall_back_rather_than_raise():
    assert to_decimal("not a number") == Decimal("0")
    assert to_decimal(object(), Decimal("7")) == Decimal("7")


def test_money_stays_exact_through_sizing():
    """Float arithmetic here would drift by fractions of a cent per trade."""
    result = size(entry_price=33.33, stop_loss=33.03, equity=10_000)
    assert result.risk_per_share == Decimal("0.30")
    assert isinstance(result.risk_amount, Decimal)


def test_result_serializes():
    payload = size().as_dict()
    assert set(payload) >= {"quantity", "risk_amount", "binding_constraint", "caps"}
