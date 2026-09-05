"""Value objects shared across the data layer.

Bar series travel as pandas DataFrames (that is what indicators and the
backtester want); point-in-time facts such as quotes and account snapshots
travel as frozen dataclasses so callers get attribute access and type checking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

#: Canonical OHLCV column order produced by every :class:`MarketDataProvider`.
OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

#: Optional enrichment columns Alpaca supplies; kept when present.
EXTRA_BAR_COLUMNS: tuple[str, ...] = ("trade_count", "vwap")

#: Full column set of a normalized bar frame.
BAR_COLUMNS: tuple[str, ...] = OHLCV_COLUMNS + EXTRA_BAR_COLUMNS


@dataclass(frozen=True, slots=True)
class Quote:
    """Top-of-book snapshot for a symbol."""

    symbol: str
    timestamp: datetime
    bid_price: float
    ask_price: float
    bid_size: float = 0.0
    ask_size: float = 0.0

    @property
    def mid_price(self) -> float:
        """Midpoint, falling back to whichever side is populated."""
        if self.bid_price > 0 and self.ask_price > 0:
            return (self.bid_price + self.ask_price) / 2
        return self.ask_price or self.bid_price

    @property
    def spread(self) -> float:
        if self.bid_price > 0 and self.ask_price > 0:
            return self.ask_price - self.bid_price
        return 0.0

    @property
    def spread_pct(self) -> float:
        """Spread as a percentage of mid — a liquidity filter for the scanner."""
        mid = self.mid_price
        return (self.spread / mid * 100) if mid > 0 else 0.0


@dataclass(frozen=True, slots=True)
class MarketClock:
    """Current market session state as reported by the broker."""

    timestamp: datetime
    is_open: bool
    next_open: datetime | None = None
    next_close: datetime | None = None

    def describe(self) -> str:
        if self.is_open:
            closes = self.next_close.isoformat() if self.next_close else "unknown"
            return f"OPEN (closes {closes})"
        opens = self.next_open.isoformat() if self.next_open else "unknown"
        return f"CLOSED (opens {opens})"


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """Broker account state.

    Monetary values are ``Decimal`` because they come from the broker as exact
    decimal strings; converting to float happens only at the point of display or
    of position-size arithmetic.
    """

    account_id: str
    status: str
    currency: str
    equity: Decimal
    cash: Decimal
    buying_power: Decimal
    portfolio_value: Decimal
    last_equity: Decimal
    pattern_day_trader: bool = False
    trading_blocked: bool = False
    transfers_blocked: bool = False
    account_blocked: bool = False
    daytrade_count: int = 0
    is_paper: bool = True

    @property
    def daily_pnl(self) -> Decimal:
        """Change in equity since the previous session close."""
        return self.equity - self.last_equity

    @property
    def daily_pnl_pct(self) -> float:
        if self.last_equity == 0:
            return 0.0
        return float(self.daily_pnl / self.last_equity * 100)

    @property
    def can_trade(self) -> bool:
        """False when the broker has blocked the account for any reason."""
        return not (self.trading_blocked or self.account_blocked)


@dataclass(frozen=True, slots=True)
class AssetInfo:
    """Tradability metadata for a symbol."""

    symbol: str
    name: str
    exchange: str
    tradable: bool
    shortable: bool
    fractionable: bool
    marginable: bool = False
    status: str = "active"

    @property
    def is_active(self) -> bool:
        return self.status.lower() == "active" and self.tradable


@dataclass(slots=True)
class DataFetchReport:
    """Outcome of a multi-symbol fetch — surfaced by the CLI and the scanner."""

    requested: list[str] = field(default_factory=list)
    succeeded: dict[str, int] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    from_cache: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def success_count(self) -> int:
        return len(self.succeeded)

    @property
    def total_bars(self) -> int:
        return sum(self.succeeded.values())

    def summary(self) -> str:
        return (
            f"{self.success_count}/{len(self.requested)} symbols, "
            f"{self.total_bars:,} bars "
            f"({len(self.from_cache)} served from cache, {len(self.failed)} failed)"
        )
