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

    def ping(self) -> bool:
        """True when the broker API is reachable and authenticated."""
        try:
            self.get_account()
            return True
        except Exception as error:  # noqa: BLE001
            logger.error("Broker connectivity check failed: %s", error)
            return False


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
