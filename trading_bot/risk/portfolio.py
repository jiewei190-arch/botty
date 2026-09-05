"""Portfolio state — the facts the risk manager needs to decide.

:class:`RiskManager` is a pure function of a signal and a
:class:`PortfolioState`. It never reads the database or calls the broker itself,
which is what makes every limit testable without a network or a fixture
database, and lets a backtest feed it a simulated portfolio through exactly the
same path the live bot uses.

Assembling that state from the real sources is the job of
:func:`build_portfolio_state`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal
from typing import Any

from trading_bot.risk.position_sizing import ZERO, to_decimal

logger = logging.getLogger(__name__)

#: US equities open at 09:30 America/New_York, which is 13:30 UTC outside DST
#: and 14:30 during it. Used only to bound "today's" realised P&L.
_SESSION_RESET_UTC = time(hour=8)


@dataclass(frozen=True, slots=True)
class OpenPosition:
    """A position the account currently holds."""

    symbol: str
    direction: str
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    strategy: str = ""

    @property
    def market_value(self) -> Decimal:
        """Absolute exposure, regardless of side."""
        return abs(self.quantity) * self.current_price

    @property
    def unrealized_pnl(self) -> Decimal:
        sign = Decimal(1) if self.direction.upper() in ("LONG", "BUY") else Decimal(-1)
        return (self.current_price - self.entry_price) * self.quantity * sign

    @property
    def risk_remaining(self) -> Decimal:
        """Money still exposed between the current price and the stop.

        Zero once the stop is beyond the current price — a position whose stop
        has been moved to breakeven or better carries no further downside.
        """
        if self.stop_loss is None:
            return self.market_value
        is_long = self.direction.upper() in ("LONG", "BUY")
        distance = (
            self.current_price - self.stop_loss
            if is_long
            else self.stop_loss - self.current_price
        )
        return max(distance, ZERO) * abs(self.quantity)


@dataclass(frozen=True, slots=True)
class PortfolioState:
    """Everything the risk manager consults, and nothing else."""

    equity: Decimal
    cash: Decimal = ZERO
    buying_power: Decimal = ZERO
    positions: tuple[OpenPosition, ...] = ()
    #: Realised profit and loss since the session began.
    realized_pnl_today: Decimal = ZERO
    #: Equity at the start of the session, for the daily-loss percentage.
    session_start_equity: Decimal | None = None
    consecutive_losses: int = 0
    last_loss_at: datetime | None = None
    #: The broker has blocked the account.
    trading_blocked: bool = False
    #: Operator kill switch, independent of anything the broker says.
    halted: bool = False
    halt_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def open_count(self) -> int:
        return len(self.positions)

    @property
    def total_exposure(self) -> Decimal:
        """Combined market value of open positions."""
        return sum((position.market_value for position in self.positions), ZERO)

    @property
    def exposure_pct(self) -> float:
        if self.equity <= 0:
            return 0.0
        return float(self.total_exposure / self.equity * 100)

    @property
    def unrealized_pnl(self) -> Decimal:
        return sum((position.unrealized_pnl for position in self.positions), ZERO)

    @property
    def daily_pnl(self) -> Decimal:
        """Realised plus unrealised.

        Unrealised losses count toward the daily limit deliberately. A limit that
        ignored open positions could be satisfied while the account bled, simply
        because nothing had been closed yet.
        """
        return self.realized_pnl_today + self.unrealized_pnl

    @property
    def daily_pnl_pct(self) -> float:
        base = self.session_start_equity or self.equity
        if base <= 0:
            return 0.0
        return float(self.daily_pnl / base * 100)

    def has_position(self, symbol: str) -> bool:
        target = symbol.strip().upper()
        return any(position.symbol.upper() == target for position in self.positions)

    def position_for(self, symbol: str) -> OpenPosition | None:
        target = symbol.strip().upper()
        for position in self.positions:
            if position.symbol.upper() == target:
                return position
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "equity": float(self.equity),
            "open_positions": self.open_count,
            "total_exposure": float(self.total_exposure),
            "exposure_pct": round(self.exposure_pct, 2),
            "realized_pnl_today": float(self.realized_pnl_today),
            "unrealized_pnl": float(self.unrealized_pnl),
            "daily_pnl": float(self.daily_pnl),
            "daily_pnl_pct": round(self.daily_pnl_pct, 2),
            "consecutive_losses": self.consecutive_losses,
            "trading_blocked": self.trading_blocked,
            "halted": self.halted,
        }


def session_start(now: datetime | None = None) -> datetime:
    """Start of the current trading session, in UTC.

    Used to bound "today's" realised P&L. The boundary sits before the US open
    in either DST offset, so a session is never split across the reset.
    """
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)

    boundary = current.replace(
        hour=_SESSION_RESET_UTC.hour, minute=0, second=0, microsecond=0
    )
    if current < boundary:
        boundary -= timedelta(days=1)
    return boundary


def build_portfolio_state(
    *,
    account: Any = None,
    broker_positions: list[dict[str, Any]] | None = None,
    database: Any = None,
    equity: Any = None,
    now: datetime | None = None,
    halted: bool = False,
    halt_reason: str = "",
) -> PortfolioState:
    """Assemble portfolio state from the broker and the trade database.

    Every source is optional so the caller can supply whatever it has: a
    backtest passes ``equity`` and positions directly, the live bot passes an
    account snapshot and a database.

    Parameters
    ----------
    account:
        An :class:`~trading_bot.data.models.AccountSnapshot`, or anything with
        ``equity``, ``cash``, ``buying_power`` and ``can_trade``.
    broker_positions:
        Position dicts as returned by ``AlpacaBroker.get_positions()``.
    database:
        A :class:`~trading_bot.data.database.Database` for realised P&L and the
        losing streak.
    equity:
        Overrides the account's equity, for simulations.
    now:
        Current time, for session bounds and cooldowns.
    """
    current = now or datetime.now(timezone.utc)

    resolved_equity = to_decimal(
        equity if equity is not None else getattr(account, "equity", ZERO)
    )
    cash = to_decimal(getattr(account, "cash", ZERO))
    buying_power = to_decimal(getattr(account, "buying_power", ZERO))
    blocked = bool(account is not None and not getattr(account, "can_trade", True))

    positions: list[OpenPosition] = []
    for raw in broker_positions or []:
        try:
            positions.append(
                OpenPosition(
                    symbol=str(raw["symbol"]).upper(),
                    direction=str(raw.get("side", "long")).upper(),
                    quantity=to_decimal(raw.get("qty", 0)),
                    entry_price=to_decimal(raw.get("avg_entry_price", 0)),
                    current_price=to_decimal(
                        raw.get("current_price") or raw.get("avg_entry_price", 0)
                    ),
                    stop_loss=to_decimal(raw["stop_loss"]) if raw.get("stop_loss") else None,
                    take_profit=(
                        to_decimal(raw["take_profit"]) if raw.get("take_profit") else None
                    ),
                    strategy=str(raw.get("strategy", "")),
                )
            )
        except Exception as error:  # noqa: BLE001 - one bad row must not blind the limits
            logger.warning("Skipping unreadable position %r: %s", raw, error)

    realized = ZERO
    streak = 0
    last_loss: datetime | None = None
    if database is not None:
        boundary = session_start(current)
        try:
            realized = to_decimal(database.trades.realized_pnl_since(boundary))
            streak = int(database.trades.consecutive_losses())
            last_loss = _last_loss_timestamp(database)
        except Exception as error:  # noqa: BLE001 - degrade to the safe default
            logger.warning("Could not read risk history from the database: %s", error)

    # Session-start equity, so the daily-loss percentage measures the drawdown
    # from where the day began rather than from where it stands now.
    start_equity = resolved_equity - realized
    if account is not None and getattr(account, "last_equity", None):
        start_equity = to_decimal(account.last_equity)

    return PortfolioState(
        equity=resolved_equity,
        cash=cash,
        buying_power=buying_power,
        positions=tuple(positions),
        realized_pnl_today=realized,
        session_start_equity=start_equity if start_equity > 0 else resolved_equity,
        consecutive_losses=streak,
        last_loss_at=last_loss,
        trading_blocked=blocked,
        halted=halted,
        halt_reason=halt_reason,
    )


def _last_loss_timestamp(database: Any) -> datetime | None:
    """When the most recent losing trade closed, for the cooldown clock."""
    rows = database.query(
        "SELECT exit_ts FROM trades WHERE status = 'closed' AND pnl < 0 "
        "ORDER BY exit_ts DESC LIMIT 1"
    )
    if not rows or not rows[0].get("exit_ts"):
        return None
    try:
        stamp = datetime.fromisoformat(str(rows[0]["exit_ts"]))
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)
