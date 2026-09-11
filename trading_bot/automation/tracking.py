"""Persist movement of selected underlyings and their exact option contracts."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class PriceTracker:
    def __init__(self, stock_data, option_chain, database) -> None:
        self.stock_data, self.option_chain, self.db = stock_data, option_chain, database

    def capture(self) -> int:
        active = self.db.option_selections.active()
        captured = 0
        underlyings = list(dict.fromkeys(row["underlying_symbol"] for row in active))
        for symbol in underlyings:
            try:
                price = self.stock_data.get_latest_price(symbol)
                if price is not None and price > 0:
                    self.db.prices.record(
                        symbol=symbol, underlying_symbol=symbol,
                        asset_kind="underlying", price=price,
                    )
                    captured += 1
            except Exception as error:  # noqa: BLE001
                logger.warning("Underlying snapshot failed for %s: %s", symbol, error)
        contract_underlying = {
            row["contract_symbol"]: row["underlying_symbol"] for row in active
        }
        try:
            quotes = self.option_chain.latest_mid(list(contract_underlying))
        except Exception as error:  # noqa: BLE001
            logger.warning("Option snapshot batch failed: %s", error)
            quotes = {}
        for symbol, (mid, bid, ask) in quotes.items():
            self.db.prices.record(
                symbol=symbol, underlying_symbol=contract_underlying[symbol],
                asset_kind="option", price=mid, bid=bid, ask=ask,
            )
            captured += 1
        return captured
