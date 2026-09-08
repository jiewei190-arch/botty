"""Broker connectivity.

Phase 1 exposes **read-only** account, clock and asset access. No method here
places, modifies or cancels an order; order routing lands in Phase 7 behind the
same safety checks.

Safety design
-------------
``TradingClient`` is constructed with ``paper=True`` unless
:attr:`Settings.is_live` is True, and that property is only True once *both*
live-trading locks in :mod:`trading_bot.config.settings` are satisfied. There is
no code path that reaches the live endpoint from default configuration.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

from trading_bot.config.settings import Settings, TradingMode
from trading_bot.data.market_data import ensure_utc
from trading_bot.data.models import AccountSnapshot, AssetInfo, MarketClock
from trading_bot.utils.retry import retry_call

logger = logging.getLogger(__name__)


class BrokerError(RuntimeError):
    """Raised when the broker cannot be reached or rejects a request."""


def _optional_utc(value: Any) -> datetime | None:
    """Coerce a broker timestamp to UTC, tolerating None and ISO strings."""
    if value is None:
        return None
    try:
        return ensure_utc(value)
    except (ValueError, TypeError):
        logger.warning("Could not parse broker timestamp %r", value)
        return None


def _decimal(value: Any, default: str = "0") -> Decimal:
    """Convert a broker-supplied numeric string to Decimal without losing precision."""
    if value is None:
        return Decimal(default)
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        logger.warning("Could not parse %r as a decimal; defaulting to %s", value, default)
        return Decimal(default)


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _order_dict(order: Any, *, parent_order_id: str | None = None) -> dict[str, Any]:
    """Normalize an Alpaca Order model without retaining credential-bearing clients."""
    order_id = str(getattr(order, "id", ""))
    return {
        "id": order_id,
        "parent_order_id": parent_order_id,
        "client_order_id": str(getattr(order, "client_order_id", "") or ""),
        "symbol": str(getattr(order, "symbol", "")),
        "side": _enum_value(getattr(order, "side", "")),
        "type": _enum_value(getattr(order, "type", getattr(order, "order_type", ""))),
        "time_in_force": _enum_value(getattr(order, "time_in_force", "")),
        "status": _enum_value(getattr(order, "status", "unknown")),
        "qty": float(getattr(order, "qty", 0) or 0),
        "filled_qty": float(getattr(order, "filled_qty", 0) or 0),
        "filled_avg_price": _optional_float(getattr(order, "filled_avg_price", None)),
        "limit_price": _optional_float(getattr(order, "limit_price", None)),
        "stop_price": _optional_float(getattr(order, "stop_price", None)),
        "created_at": _optional_utc(getattr(order, "created_at", None)),
        "updated_at": _optional_utc(getattr(order, "updated_at", None)),
        "filled_at": _optional_utc(getattr(order, "filled_at", None)),
        "legs": [
            _order_dict(leg, parent_order_id=order_id)
            for leg in (getattr(order, "legs", None) or [])
        ],
    }
class AlpacaBroker:
    """Read-only Alpaca trading-API client.

    Parameters
    ----------
    settings:
        Full application settings; the live/paper endpoint is derived from it.
    client:
        Optional pre-built client, primarily for tests.
    """

    def __init__(self, settings: Settings, *, client: TradingClient | None = None) -> None:
        self._settings = settings
        alpaca = settings.alpaca

        if client is None:
            if not alpaca.has_credentials:
                raise BrokerError(
                    "Alpaca credentials are missing. Set ALPACA_API_KEY and "
                    "ALPACA_SECRET_KEY in your .env file."
                )
            # `paper` is False only when both live-trading locks passed.
            client = TradingClient(
                api_key=alpaca.api_key,
                secret_key=alpaca.secret_key,
                paper=not settings.is_live,
            )
            if settings.is_live:
                logger.critical(
                    "LIVE TRADING ENDPOINT SELECTED (%s). Real money is at risk.",
                    alpaca.live_base_url,
                )
            else:
                logger.info("Connected to Alpaca PAPER endpoint (%s)", alpaca.paper_base_url)

        self._client = client

    @property
    def is_paper(self) -> bool:
        return not self._settings.is_live

    @property
    def mode(self) -> TradingMode:
        return self._settings.trading_mode

    def _call(self, func, description: str):
        return retry_call(
            func,
            max_attempts=self._settings.alpaca.max_retries,
            base_delay=self._settings.alpaca.retry_base_delay_seconds,
            description=description,
        )

    # -- read-only queries -------------------------------------------------------

    def get_account(self) -> AccountSnapshot:
        """Current account state."""
        account = self._call(self._client.get_account, "get_account")
        return AccountSnapshot(
            account_id=str(getattr(account, "id", "")),
            status=str(getattr(account, "status", "unknown")),
            currency=str(getattr(account, "currency", "USD")),
            equity=_decimal(getattr(account, "equity", None)),
            cash=_decimal(getattr(account, "cash", None)),
            buying_power=_decimal(getattr(account, "buying_power", None)),
            portfolio_value=_decimal(getattr(account, "portfolio_value", None)),
            last_equity=_decimal(getattr(account, "last_equity", None)),
            pattern_day_trader=bool(getattr(account, "pattern_day_trader", False)),
            trading_blocked=bool(getattr(account, "trading_blocked", False)),
            transfers_blocked=bool(getattr(account, "transfers_blocked", False)),
            account_blocked=bool(getattr(account, "account_blocked", False)),
            daytrade_count=int(getattr(account, "daytrade_count", 0) or 0),
            is_paper=self.is_paper,
        )

    def get_clock(self) -> MarketClock:
        """Market session state."""
        clock = self._call(self._client.get_clock, "get_clock")
        return MarketClock(
            timestamp=_optional_utc(clock.timestamp) or datetime.now(timezone.utc),
            is_open=bool(clock.is_open),
            next_open=_optional_utc(getattr(clock, "next_open", None)),
            next_close=_optional_utc(getattr(clock, "next_close", None)),
        )

    def is_market_open(self) -> bool:
        try:
            return self.get_clock().is_open
        except Exception as error:  # noqa: BLE001 - treat unknown state as closed
            logger.error("Could not determine market state: %s", error)
            return False

    def get_asset(self, symbol: str) -> AssetInfo | None:
        """Tradability metadata, or None when the symbol is unknown."""
        symbol = symbol.strip().upper()
        try:
            asset = self._call(lambda: self._client.get_asset(symbol), f"get_asset({symbol})")
        except Exception as error:  # noqa: BLE001 - unknown symbols are expected
            logger.warning("Could not resolve asset %s: %s", symbol, error)
            return None
        return AssetInfo(
            symbol=str(asset.symbol),
            name=str(getattr(asset, "name", "") or ""),
            exchange=str(getattr(asset, "exchange", "")),
            tradable=bool(getattr(asset, "tradable", False)),
            shortable=bool(getattr(asset, "shortable", False)),
            fractionable=bool(getattr(asset, "fractionable", False)),
            marginable=bool(getattr(asset, "marginable", False)),
            status=str(getattr(asset, "status", "active")),
        )

    def validate_symbols(self, symbols: list[str]) -> tuple[list[str], dict[str, str]]:
        """Split a watchlist into tradable symbols and rejected ones with reasons."""
        tradable: list[str] = []
        rejected: dict[str, str] = {}
        for symbol in symbols:
            asset = self.get_asset(symbol)
            if asset is None:
                rejected[symbol] = "unknown symbol"
            elif not asset.is_active:
                rejected[symbol] = f"not tradable (status={asset.status})"
            else:
                tradable.append(asset.symbol)
        return tradable, rejected

    def get_positions(self) -> list[dict[str, Any]]:
        """Open broker positions as plain dicts."""
        positions = self._call(self._client.get_all_positions, "get_all_positions")
        return [
            {
                "symbol": str(position.symbol),
                "qty": float(position.qty),
                "side": str(getattr(position.side, "value", position.side)),
                "avg_entry_price": float(position.avg_entry_price),
                "current_price": float(position.current_price or 0),
                "market_value": float(position.market_value or 0),
                "cost_basis": float(position.cost_basis or 0),
                "unrealized_pl": float(position.unrealized_pl or 0),
                "unrealized_plpc": float(position.unrealized_plpc or 0) * 100,
            }
            for position in positions
        ]

    def get_orders(self, *, status: str = "all", nested: bool = True) -> list[dict[str, Any]]:
        """Recent broker orders, normalized with bracket legs included."""
        try:
            query_status = QueryOrderStatus(status.lower())
        except ValueError as error:
            raise BrokerError(f"Unknown order query status {status!r}") from error
        request = GetOrdersRequest(status=query_status, limit=500, nested=nested)
        orders = self._call(
            lambda: self._client.get_orders(filter=request), f"get_orders({status})"
        )
        return [_order_dict(order) for order in orders]

    def ping(self) -> bool:
        """True when the broker API is reachable and authenticated."""
        try:
            self.get_account()
            return True
        except Exception as error:  # noqa: BLE001
            logger.error("Broker connectivity check failed: %s", error)
            return False

    # -- guarded execution ------------------------------------------------------

    def submit_bracket_order(
        self,
        *,
        symbol: str,
        qty: int,
        side: str,
        take_profit: float,
        stop_loss: float,
        client_order_id: str,
        allow_live: bool = False,
    ) -> dict[str, Any]:
        """Submit an entry with broker-hosted stop and target protection.

        Live submission requires both the global live locks *and* an explicit
        per-call opt-in. The automated runner deliberately never opts in.
        """
        if qty < 1:
            raise BrokerError(f"Order quantity must be positive, got {qty}")
        if not self.is_paper and not allow_live:
            raise BrokerError("Live order refused: this call is paper-only")
        normalized_side = side.strip().lower()
        if normalized_side not in {"buy", "sell"}:
            raise BrokerError(f"Order side must be buy or sell, got {side!r}")
        request = MarketOrderRequest(
            symbol=symbol.strip().upper(),
            qty=qty,
            side=OrderSide.BUY if normalized_side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(float(take_profit), 2)),
            stop_loss=StopLossRequest(stop_price=round(float(stop_loss), 2)),
            client_order_id=client_order_id[:48],
        )
        try:
            order = self._call(
                lambda: self._client.submit_order(order_data=request),
                f"submit_bracket_order({symbol})",
            )
        except Exception as error:
            raise BrokerError(f"Broker rejected {symbol} bracket order: {error}") from error
        return {
            "id": str(getattr(order, "id", "")),
            "client_order_id": str(getattr(order, "client_order_id", client_order_id)),
            "symbol": str(getattr(order, "symbol", symbol)),
            "status": _enum_value(getattr(order, "status", "accepted")),
            "qty": float(getattr(order, "qty", qty)),
        }


def build_broker(settings: Settings) -> AlpacaBroker:
    """Construct the broker client for the configured mode.

    Raises in BACKTEST mode: backtests must not depend on a broker connection.
    """
    if settings.trading_mode is TradingMode.BACKTEST:
        raise BrokerError(
            "BACKTEST mode does not use a broker connection. "
            "Set TRADING_MODE=paper to connect to Alpaca."
        )
    return AlpacaBroker(settings)
