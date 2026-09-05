"""Position sizing.

How many shares to buy is not a preference — it is arithmetic, derived from how
much you are willing to lose and how far away the stop sits:

    risk budget   = equity × max_risk_per_trade_pct / 100
    risk per share = |entry − stop|
    quantity      = risk budget ÷ risk per share

That is the whole idea, and it is why a strategy must supply a stop before it
can supply a size. A wide stop buys fewer shares and a tight stop buys more, so
every trade risks the same amount regardless of how volatile the instrument is.
Sizing by a fixed dollar amount or a fixed share count instead makes each trade
risk a different, unknown quantity — which is how accounts die from a run of
"small" trades.

Four caps then apply, and the smallest wins:

============================  =============================================
Cap                           Guards against
============================  =============================================
Risk budget                   Losing more than intended on one trade
Position size                 One name dominating the account
Portfolio exposure            Being fully invested across many names
Buying power                  Ordering more than the broker will allow
============================  =============================================

The result reports **which cap bound**, because "why is my position so small?"
is otherwise unanswerable. On a $10,000 account the 20% position cap allows
about $2,000 — nine shares of a $210 stock — while the 1% risk budget with a
$3.50 stop would allow twenty-eight. The tighter cap is not a bug; knowing which
one it was is the difference between tuning the right number and the wrong one.

Money is handled as :class:`~decimal.Decimal`. Prices arriving as floats are
converted through ``str`` so a value like ``0.1`` stays ``0.1`` rather than
becoming ``0.1000000000000000055511151231257827``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

#: Quantities are rounded down to this precision for fractional trading.
FRACTIONAL_PRECISION = Decimal("0.0001")

ZERO = Decimal("0")


class SizingConstraint(str, Enum):
    """Which limit determined the final quantity."""

    RISK_BUDGET = "risk_budget"
    POSITION_SIZE = "position_size"
    PORTFOLIO_EXPOSURE = "portfolio_exposure"
    BUYING_POWER = "buying_power"
    #: Nothing bound — the requested size was allowed in full.
    NONE = "none"

    @property
    def description(self) -> str:
        return {
            SizingConstraint.RISK_BUDGET: "risk per trade",
            SizingConstraint.POSITION_SIZE: "maximum position size",
            SizingConstraint.PORTFOLIO_EXPOSURE: "portfolio exposure",
            SizingConstraint.BUYING_POWER: "buying power",
            SizingConstraint.NONE: "no binding limit",
        }[self]


def to_decimal(value: Any, default: Decimal = ZERO) -> Decimal:
    """Convert to Decimal without inheriting binary-float artefacts."""
    if isinstance(value, Decimal):
        return value
    if value is None:
        return default
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        logger.warning("Could not convert %r to Decimal; using %s", value, default)
        return default


@dataclass(frozen=True, slots=True)
class PositionSize:
    """The outcome of a sizing calculation.

    Attributes
    ----------
    quantity:
        Shares to trade. ``0`` means the trade cannot be taken at any size the
        limits allow.
    risk_amount:
        Money actually at risk — ``quantity × risk per share``. Always at or
        below the risk budget, because quantity is rounded down.
    position_value:
        ``quantity × entry price``.
    risk_per_share:
        Distance from entry to stop, including any slippage assumption.
    binding_constraint:
        Which cap produced the final quantity.
    caps:
        Quantity each cap would have allowed on its own, for diagnosis.
    """

    quantity: Decimal
    risk_amount: Decimal
    position_value: Decimal
    risk_per_share: Decimal
    risk_budget: Decimal
    binding_constraint: SizingConstraint
    caps: dict[str, Decimal] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def is_tradable(self) -> bool:
        """False when the limits leave nothing to trade."""
        return self.quantity > 0

    @property
    def shares(self) -> int:
        """Whole-share view, for brokers that do not accept fractions."""
        return int(self.quantity)

    def risk_pct_of(self, equity: Decimal) -> float:
        """Money at risk as a percentage of equity."""
        if equity <= 0:
            return 0.0
        return float(self.risk_amount / equity * 100)

    def explain(self) -> str:
        """One line saying what set the size."""
        if not self.is_tradable:
            return "No tradable size: the limits allow fewer than one share"
        return (
            f"{self.quantity} share(s) — limited by {self.binding_constraint.description}"
            f" (risking {self.risk_amount:.2f} of a {self.risk_budget:.2f} budget)"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "quantity": float(self.quantity),
            "risk_amount": float(self.risk_amount),
            "position_value": float(self.position_value),
            "risk_per_share": float(self.risk_per_share),
            "risk_budget": float(self.risk_budget),
            "binding_constraint": self.binding_constraint.value,
            "caps": {name: float(value) for name, value in self.caps.items()},
            "notes": list(self.notes),
        }


def calculate_position_size(
    *,
    entry_price: Any,
    stop_loss: Any,
    equity: Any,
    max_risk_per_trade_pct: float,
    max_position_size_pct: float = 100.0,
    max_portfolio_exposure_pct: float = 100.0,
    current_exposure: Any = ZERO,
    buying_power: Any = None,
    allow_fractional: bool = False,
    slippage_pct: float = 0.0,
) -> PositionSize:
    """Size a position from its stop distance, then apply every cap.

    Parameters
    ----------
    entry_price:
        Intended entry.
    stop_loss:
        Where the trade is wrong. Must differ from ``entry_price``.
    equity:
        Account equity the percentages are measured against.
    max_risk_per_trade_pct:
        Percentage of equity risked if the stop is hit.
    max_position_size_pct:
        Cap on one position's value as a percentage of equity.
    max_portfolio_exposure_pct:
        Cap on the value of all positions combined.
    current_exposure:
        Market value of positions already held.
    buying_power:
        Broker buying power. Ignored when ``None``.
    allow_fractional:
        Permit fractional shares. Whole shares otherwise.
    slippage_pct:
        Assume the stop fills this much worse than its price. Widens the
        assumed risk per share, which *reduces* size. Default 0.0 — the size is
        exactly what the stated stop implies, with no hidden adjustment.

    Returns
    -------
    PositionSize

    Raises
    ------
    ValueError
        Prices or percentages are unusable.

    Example
    -------
    >>> size = calculate_position_size(
    ...     entry_price=210.50, stop_loss=207.00, equity=10_000,
    ...     max_risk_per_trade_pct=1.0, max_position_size_pct=100.0,
    ... )
    >>> size.quantity, float(size.risk_amount)
    (Decimal('28'), 98.0)
    """
    entry = to_decimal(entry_price)
    stop = to_decimal(stop_loss)
    account_equity = to_decimal(equity)

    if entry <= 0:
        raise ValueError(f"entry_price must be positive, got {entry}")
    if stop <= 0:
        raise ValueError(f"stop_loss must be positive, got {stop}")
    if entry == stop:
        raise ValueError("entry_price and stop_loss cannot be equal — risk would be zero")
    if account_equity <= 0:
        raise ValueError(f"equity must be positive, got {account_equity}")
    if not 0 < max_risk_per_trade_pct <= 100:
        raise ValueError(
            f"max_risk_per_trade_pct must be within (0, 100], got {max_risk_per_trade_pct}"
        )
    if slippage_pct < 0:
        raise ValueError(f"slippage_pct must be >= 0, got {slippage_pct}")

    notes: list[str] = []

    risk_per_share = abs(entry - stop)
    if slippage_pct > 0:
        # Assume the stop fills worse than its price, so the position is smaller
        # than a perfect fill would justify.
        widened = risk_per_share + entry * to_decimal(slippage_pct) / 100
        notes.append(
            f"Risk widened from {risk_per_share:.4f} to {widened:.4f} for assumed slippage"
        )
        risk_per_share = widened

    risk_budget = account_equity * to_decimal(max_risk_per_trade_pct) / 100

    # Each cap expressed as a quantity, so they are directly comparable.
    caps: dict[str, Decimal] = {
        SizingConstraint.RISK_BUDGET.value: risk_budget / risk_per_share,
    }
    if max_position_size_pct < 100:
        caps[SizingConstraint.POSITION_SIZE.value] = (
            account_equity * to_decimal(max_position_size_pct) / 100 / entry
        )
    if max_portfolio_exposure_pct < 100:
        headroom = (
            account_equity * to_decimal(max_portfolio_exposure_pct) / 100
            - to_decimal(current_exposure)
        )
        caps[SizingConstraint.PORTFOLIO_EXPOSURE.value] = max(headroom, ZERO) / entry
    if buying_power is not None:
        caps[SizingConstraint.BUYING_POWER.value] = max(to_decimal(buying_power), ZERO) / entry

    binding_name = min(caps, key=lambda name: caps[name])
    raw_quantity = caps[binding_name]

    quantity = (
        raw_quantity.quantize(FRACTIONAL_PRECISION, rounding=ROUND_DOWN)
        if allow_fractional
        else raw_quantity.to_integral_value(rounding=ROUND_DOWN)
    )
    quantity = max(quantity, ZERO)

    if quantity <= 0:
        notes.append(
            f"The {SizingConstraint(binding_name).description} limit allows "
            f"{raw_quantity:.4f} shares, which rounds down to zero"
        )

    return PositionSize(
        quantity=quantity,
        risk_amount=quantity * risk_per_share,
        position_value=quantity * entry,
        risk_per_share=risk_per_share,
        risk_budget=risk_budget,
        binding_constraint=SizingConstraint(binding_name),
        caps=caps,
        notes=tuple(notes),
    )
