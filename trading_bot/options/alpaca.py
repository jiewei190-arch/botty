"""Alpaca option-chain adapter with normalized, testable output."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from trading_bot.config.settings import Settings
from trading_bot.options.selector import OptionQuote


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def _value(value: Any) -> str:
    return str(getattr(value, "value", value)).lower()


#: Alpaca rejects a market-data request naming more than this many symbols with
#: ``{"message":"symbol limit is 100"}``. A single underlying easily exceeds it:
#: a liquid name inside a +/-20% strike band across 45-120 DTE carries several
#: hundred contracts once weekly expirations are included. Requesting the whole
#: chain at once therefore failed for exactly the busiest, most tradeable names
#: and left the thin ones working, which is the worst possible failure shape.
SYMBOL_REQUEST_LIMIT = 100


def _batched(symbols: list[str], size: int = SYMBOL_REQUEST_LIMIT) -> list[list[str]]:
    """Split ``symbols`` into request-sized chunks, preserving order."""
    return [symbols[i : i + size] for i in range(0, len(symbols), size)]


class AlpacaOptionChain:
    """Fetch contracts and snapshots only inside the configured swing window."""

    def __init__(self, settings: Settings, *, trading_client=None, data_client=None) -> None:
        self.settings = settings
        if trading_client is None or data_client is None:
            if not settings.alpaca.has_credentials:
                raise ValueError("Alpaca credentials are required for option chains")
            from alpaca.data.historical.option import OptionHistoricalDataClient
            from alpaca.trading.client import TradingClient

            trading_client = trading_client or TradingClient(
                settings.alpaca.api_key, settings.alpaca.secret_key,
                paper=not settings.is_live,
            )
            data_client = data_client or OptionHistoricalDataClient(
                settings.alpaca.api_key, settings.alpaca.secret_key,
            )
        self.trading_client = trading_client
        self.data_client = data_client

    def quotes(self, underlying: str, direction: str, underlying_price: float) -> list[OptionQuote]:
        from alpaca.data.requests import OptionSnapshotRequest
        from alpaca.trading.enums import ContractType
        from alpaca.trading.requests import GetOptionContractsRequest

        today = date.today()
        kind = ContractType.CALL if direction.upper() == "LONG" else ContractType.PUT
        width = 0.20
        request = GetOptionContractsRequest(
            underlying_symbols=[underlying.upper()], type=kind,
            expiration_date_gte=today + timedelta(days=self.settings.options.min_dte),
            expiration_date_lte=today + timedelta(days=self.settings.options.exceptional_max_dte),
            strike_price_gte=str(round(underlying_price * (1 - width), 2)),
            strike_price_lte=str(round(underlying_price * (1 + width), 2)),
            limit=1000,
        )
        response = self.trading_client.get_option_contracts(request)
        contracts = list(getattr(response, "option_contracts", None) or [])
        if not contracts:
            return []
        symbols = [str(contract.symbol) for contract in contracts]
        # The feed is stated rather than defaulted. Alpaca serves OPRA to paid
        # plans and an indicative feed to the free one; leaving it unset asks for
        # whatever the account happens to have, which makes a failure here depend
        # on the subscription instead of on the code.
        snapshots: dict[str, Any] = {}
        for batch in _batched(symbols):
            page = self.data_client.get_option_snapshot(
                OptionSnapshotRequest(
                    symbol_or_symbols=batch, feed=self.settings.alpaca.options_feed
                )
            )
            snapshots.update(page if isinstance(page, dict) else dict(page))
        output: list[OptionQuote] = []
        for contract in contracts:
            symbol = str(contract.symbol)
            snap = snapshots.get(symbol)
            quote = getattr(snap, "latest_quote", None)
            greeks = getattr(snap, "greeks", None)
            if quote is None:
                continue
            # No daily volume here on purpose: an option snapshot carries the
            # latest trade, latest quote, implied volatility and greeks, and
            # nothing else. Reading a `daily_bar` off it returned None every
            # time, so every contract scored zero volume and a non-zero
            # `min_daily_volume` rejected the entire chain in silence. Open
            # interest below comes from the contract record, which does have it.
            output.append(OptionQuote(
                symbol=symbol, underlying=underlying.upper(),
                contract_type=_value(contract.type),
                expiration=contract.expiration_date,
                strike=_number(contract.strike_price), bid=_number(quote.bid_price),
                ask=_number(quote.ask_price), delta=getattr(greeks, "delta", None),
                open_interest=int(_number(getattr(contract, "open_interest", 0))),
                implied_volatility=getattr(snap, "implied_volatility", None),
                underlying_price=underlying_price,
            ))
        return output

    def latest_mid(self, symbols: list[str]) -> dict[str, tuple[float, float, float]]:
        """Return contract -> (mid, bid, ask), omitting invalid quotes."""
        if not symbols:
            return {}
        from alpaca.data.requests import OptionLatestQuoteRequest

        quotes: dict[str, Any] = {}
        for batch in _batched(symbols):
            page = self.data_client.get_option_latest_quote(
                OptionLatestQuoteRequest(symbol_or_symbols=batch)
            )
            quotes.update(page if isinstance(page, dict) else dict(page))
        result = {}
        for symbol, quote in quotes.items():
            bid, ask = _number(quote.bid_price), _number(quote.ask_price)
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else ask or bid
            if mid > 0:
                result[str(symbol)] = (mid, bid, ask)
        return result
