"""Fill modelling — where backtests usually start lying.

Three assumptions decide whether a backtest resembles reality. All three are made
explicit here rather than buried in the engine.

**1. You cannot trade at a price you have only just observed.**
A signal produced from bar ``i``'s close is filled at bar ``i + 1``'s *open*.
Filling at the signal bar's own close is the single most common backtesting
error: it assumes you saw the close and traded at it, which is not a thing that
can happen. On a trending instrument it quietly awards every trade a free bar of
profit.

**2. Gaps blow through stops.**
A stop is not a guaranteed price, it is a trigger. When a bar opens beyond the
stop, the fill is the *open*, not the stop — that is what a gap does to a real
account. A backtester that always fills stops exactly at the stop price
understates the tail of the loss distribution, which is the part that matters.

**3. Costs are paid on both sides.**
Commission, and slippage that always moves against the trade. Slippage that
sometimes helped would be modelling a different, friendlier market.

Limit exits are the one place this model is conservative rather than pessimistic:
a target is assumed to fill exactly at its limit, never better.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from trading_bot.strategies import SignalDirection

logger = logging.getLogger(__name__)


class FillReason(str, Enum):
    """Why a fill happened, which determines how its price is derived."""

    ENTRY = "entry"
    STOP = "stop"
    TARGET = "target"
    SIGNAL_EXIT = "signal_exit"
    #: Forced liquidation at the end of the backtest.
    END_OF_DATA = "end_of_data"

    @property
    def is_exit(self) -> bool:
        return self is not FillReason.ENTRY


@dataclass(frozen=True, slots=True)
class CostModel:
    """Trading costs.

    Attributes
    ----------
    commission_per_trade:
        Flat fee per fill. Most US retail brokers charge zero.
    commission_per_share:
        Per-share fee, as some brokers and most non-US venues charge.
    commission_pct:
        Percentage of notional, common outside US equities.
    slippage_pct:
        Fraction of price lost to slippage on every fill, **always against the
        trade**. 0.05% is a reasonable default for liquid US equities on
        marketable orders; illiquid names are far worse.
    min_commission:
        Floor applied once the components are summed, when any are non-zero.
    """

    commission_per_trade: float = 0.0
    commission_per_share: float = 0.0
    commission_pct: float = 0.0
    slippage_pct: float = 0.05
    min_commission: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("commission_per_trade", self.commission_per_trade),
            ("commission_per_share", self.commission_per_share),
            ("commission_pct", self.commission_pct),
            ("slippage_pct", self.slippage_pct),
            ("min_commission", self.min_commission),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")

    def commission(self, quantity: float, price: float) -> float:
        """Commission for one fill."""
        total = (
            self.commission_per_trade
            + self.commission_per_share * abs(quantity)
            + self.commission_pct / 100 * abs(quantity) * price
        )
        if total <= 0:
            return 0.0
        return max(total, self.min_commission)

    def slip(self, price: float, direction: SignalDirection, *, is_entry: bool) -> float:
        """Move a price against the trade by the slippage assumption.

        Buying pays more and selling receives less, whichever side of the trade
        is being opened or closed.
        """
        if self.slippage_pct <= 0:
            return price
        fraction = self.slippage_pct / 100
        buying = (direction is SignalDirection.LONG) == is_entry
        return price * (1 + fraction) if buying else price * (1 - fraction)


#: Zero costs. Useful for isolating strategy behaviour in a test, never for
#: judging whether a strategy is worth trading.
FRICTIONLESS = CostModel(commission_per_trade=0.0, slippage_pct=0.0)


@dataclass(frozen=True, slots=True)
class Fill:
    """A completed fill."""

    price: float
    quantity: float
    commission: float
    reason: FillReason
    #: Difference between the modelled fill and the reference price it came from.
    slippage_cost: float = 0.0
    #: True when the bar opened beyond the stop, so the fill is worse than it.
    gapped: bool = False

    @property
    def notional(self) -> float:
        return self.price * abs(self.quantity)

    def as_dict(self) -> dict[str, Any]:
        return {
            "price": self.price,
            "quantity": self.quantity,
            "commission": self.commission,
            "reason": self.reason.value,
            "slippage_cost": self.slippage_cost,
            "gapped": self.gapped,
        }


class FillModel:
    """Turns an intent and a bar into a fill price.

    Example
    -------
    >>> model = FillModel(CostModel(slippage_pct=0.05))
    >>> fill = model.entry_fill(bar, SignalDirection.LONG, quantity=100)
    >>> fill.price > bar["open"]      # a buy pays up
    True
    """

    def __init__(self, costs: CostModel | None = None) -> None:
        self.costs = costs or CostModel()

    def entry_fill(
        self, bar: Any, direction: SignalDirection, quantity: float
    ) -> Fill:
        """Fill an entry at the bar's open.

        The bar passed here is the one *after* the signal was generated. The
        engine enforces that; this method assumes it.
        """
        reference = float(bar["open"])
        price = self.costs.slip(reference, direction, is_entry=True)
        return Fill(
            price=price,
            quantity=quantity,
            commission=self.costs.commission(quantity, price),
            reason=FillReason.ENTRY,
            slippage_cost=abs(price - reference) * abs(quantity),
        )

    def stop_fill(
        self, bar: Any, direction: SignalDirection, quantity: float, stop_price: float
    ) -> Fill:
        """Fill a stop, honouring gaps.

        A stop is a trigger, not a guaranteed price. If the bar opened beyond it,
        the fill is the open — which is how a real account experiences a gap.
        """
        open_price = float(bar["open"])
        is_long = direction is SignalDirection.LONG
        gapped = open_price <= stop_price if is_long else open_price >= stop_price

        reference = open_price if gapped else stop_price
        price = self.costs.slip(reference, direction, is_entry=False)
        if gapped:
            logger.debug(
                "Stop gapped: opened at %.4f through a %.4f stop", open_price, stop_price
            )
        return Fill(
            price=price,
            quantity=quantity,
            commission=self.costs.commission(quantity, price),
            reason=FillReason.STOP,
            slippage_cost=abs(price - reference) * abs(quantity),
            gapped=gapped,
        )

    def target_fill(
        self, bar: Any, direction: SignalDirection, quantity: float, target_price: float
    ) -> Fill:
        """Fill a target.

        Assumed to fill exactly at the limit — never better, even when the bar
        traded well through it. This is the model's one conservative assumption;
        being optimistic here would manufacture profit that a resting limit order
        would not have captured.
        """
        return Fill(
            price=target_price,
            quantity=quantity,
            commission=self.costs.commission(quantity, target_price),
            reason=FillReason.TARGET,
            slippage_cost=0.0,
        )

    def exit_fill(
        self,
        bar: Any,
        direction: SignalDirection,
        quantity: float,
        reason: FillReason = FillReason.SIGNAL_EXIT,
    ) -> Fill:
        """Fill a discretionary exit at the bar's open, like an entry."""
        reference = float(bar["open"])
        price = self.costs.slip(reference, direction, is_entry=False)
        return Fill(
            price=price,
            quantity=quantity,
            commission=self.costs.commission(quantity, price),
            reason=reason,
            slippage_cost=abs(price - reference) * abs(quantity),
        )
