"""Select liquid long calls/puts with 7-to-60 days until expiration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import ceil

from trading_bot.config.settings import OptionsSettings


@dataclass(frozen=True, slots=True)
class OptionQuote:
    symbol: str
    underlying: str
    contract_type: str
    expiration: date
    strike: float
    bid: float
    ask: float
    delta: float | None
    daily_volume: int = 0
    open_interest: int = 0
    implied_volatility: float | None = None
    #: Spot price of the underlying when this quote was taken. Used to judge
    #: moneyness when the feed supplies no greeks.
    underlying_price: float | None = None

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2 if self.bid > 0 else self.ask

    @property
    def spread_pct(self) -> float:
        return ((self.ask - self.bid) / self.mid * 100) if self.mid > 0 else float("inf")

    @property
    def moneyness_pct(self) -> float | None:
        """Strike relative to spot, as a percentage.

        Negative means in the money for a call and out of the money for a put;
        the sign is flipped for puts so one band reads the same for both.
        """
        if not self.underlying_price or self.underlying_price <= 0:
            return None
        raw = (self.strike - self.underlying_price) / self.underlying_price * 100
        return raw if self.contract_type.lower() == "call" else -raw


@dataclass(frozen=True, slots=True)
class OptionSelection:
    quote: OptionQuote
    quantity: int
    estimated_cost: float
    days_to_expiry: int
    score: float
    #: ``"greeks"`` when the feed supplied a delta, ``"moneyness"`` when the
    #: strike's distance from spot stood in for one. Surfaced in alerts so a
    #: reader can tell a measured selection from an approximated one.
    delta_source: str = "greeks"

    @property
    def approximated(self) -> bool:
        return self.delta_source != "greeks"

    def describe(self) -> str:
        """The one line a trader actually reads: ticker, strike, type, expiry."""
        quote = self.quote
        return (
            f"{quote.underlying} ${quote.strike:g} "
            f"{quote.contract_type.upper()} "
            f"{quote.expiration.isoformat()} ({self.days_to_expiry}DTE)"
        )


class SwingOptionSelector:
    """Deterministic liquidity-first contract selection; no price prediction here."""

    def __init__(self, settings: OptionsSettings) -> None:
        self.settings = settings

    def select(
        self, *, direction: str, quotes: list[OptionQuote], as_of: date,
        confidence: float = 0, remaining_premium: float | None = None,
        remaining_contracts: int | None = None,
    ) -> OptionSelection | None:
        desired = "call" if direction.upper() == "LONG" else "put"
        budget = min(
            self.settings.max_premium_per_trade,
            self.settings.max_total_premium if remaining_premium is None else remaining_premium,
        )
        contract_cap = min(
            self.settings.max_contracts_per_trade,
            self.settings.max_total_contracts
            if remaining_contracts is None
            else remaining_contracts,
        )
        max_dte = (
            self.settings.exceptional_max_dte
            if confidence >= self.settings.exceptional_min_confidence
            else self.settings.max_dte
        )
        if budget < self.settings.min_premium_per_trade:
            return None
        candidates: list[OptionSelection] = []
        for quote in quotes:
            dte = (quote.expiration - as_of).days
            delta = abs(quote.delta) if quote.delta is not None else None
            if quote.contract_type.lower() != desired:
                continue
            if not self.settings.min_dte <= dte <= max_dte:
                continue
            if quote.ask <= 0 or quote.bid < 0 or quote.spread_pct > self.settings.max_spread_pct:
                continue
            # Delta is the right filter and the first choice. Alpaca's free
            # indicative feed may return no greeks at all, and rejecting on a
            # missing delta silently emptied the whole chain, so the strike's
            # distance from spot stands in — a coarser proxy for the same thing,
            # flagged so nothing downstream mistakes it for a measured delta.
            if delta is not None:
                if not self.settings.min_abs_delta <= delta <= self.settings.max_abs_delta:
                    continue
                delta_source = "greeks"
            else:
                moneyness = quote.moneyness_pct
                if moneyness is None:
                    continue
                if not (
                    self.settings.fallback_min_moneyness_pct
                    <= moneyness
                    <= self.settings.fallback_max_moneyness_pct
                ):
                    continue
                delta_source = "moneyness"
            if quote.daily_volume < self.settings.min_daily_volume:
                continue
            if quote.open_interest < self.settings.min_open_interest:
                continue
            per_contract = quote.ask * 100
            minimum_qty = ceil(self.settings.min_premium_per_trade / per_contract)
            affordable_qty = min(contract_cap, int(budget // per_contract))
            if affordable_qty < 1 or affordable_qty < minimum_qty:
                continue
            # Use the smallest size that satisfies the user's range. A budget is
            # a ceiling, not a target to spend merely because buying power exists.
            quantity = minimum_qty
            # Without a delta there is nothing to measure the target against, so
            # distance from the middle of the moneyness band plays the same role.
            if delta is not None:
                delta_penalty = abs(delta - self.settings.target_delta) * 100
            else:
                midpoint = (
                    self.settings.fallback_min_moneyness_pct
                    + self.settings.fallback_max_moneyness_pct
                ) / 2
                delta_penalty = abs((quote.moneyness_pct or 0.0) - midpoint) * 10
            score = (
                delta_penalty
                + abs(dte - self.settings.target_dte) / 10
                + quote.spread_pct
            )
            candidates.append(OptionSelection(
                quote=quote, quantity=quantity, estimated_cost=per_contract * quantity,
                days_to_expiry=dte, score=score, delta_source=delta_source,
            ))
        return min(candidates, key=lambda item: item.score) if candidates else None
