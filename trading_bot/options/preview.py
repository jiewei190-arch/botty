"""Choosing the exact option contract behind a scanned play.

A scanned play is a thesis about a stock. This turns it into the thing a person
actually buys: a ticker, a strike, an expiration and a call or a put. Nothing
here places an order, and nothing here predicts anything — the contract is
chosen by liquidity and by the holding window, from a chain the market is
quoting right now.

It lives in :mod:`trading_bot.options` rather than in the dashboard because the
CLI and the automated runner need the same answer. Two implementations of
"which contract" would drift, and the one a person reads on screen would stop
matching the one the bot would buy.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from trading_bot.config.settings import Settings
from trading_bot.data.database import Database


def preview_option_trades(
    settings: Settings,
    opportunities,
    *,
    capacity: int,
    chain=None,
    as_of: date | None = None,
) -> list[dict[str, Any]]:
    """Choose exact option contracts without submitting any orders."""
    from trading_bot.options import SwingOptionSelector
    from trading_bot.options.alpaca import AlpacaOptionChain

    database = Database(settings.data.database_path)
    database.initialize()
    try:
        active = database.option_selections.active()
        held = {row["underlying_symbol"] for row in active}
        remaining_premium = max(
            0.0,
            settings.options.max_total_premium
            - database.option_selections.premium_committed(),
        )
        remaining_contracts = max(
            0,
            settings.options.max_total_contracts
            - database.option_selections.contracts_committed(),
        )
    finally:
        database.close()

    selector = SwingOptionSelector(settings.options)
    option_chain = chain or AlpacaOptionChain(settings)
    rows: list[dict[str, Any]] = []
    available_slots = max(0, settings.options.max_open_positions - len(held))
    preview_limit = min(max(0, capacity), available_slots)
    qualifying_contracts = 0

    for opportunity in opportunities:
        signal = opportunity.signal
        base = {
            "Underlying": signal.symbol,
            "Direction": signal.direction.value,
            "Score": round(float(opportunity.confidence), 1),
        }
        if signal.symbol in held:
            rows.append({**base, "Status": "Already tracked; no duplicate entry"})
            continue
        if qualifying_contracts >= preview_limit:
            rows.append({**base, "Status": "Outside current account/position capacity"})
            continue
        try:
            quotes = option_chain.quotes(
                signal.symbol, signal.direction.value, signal.entry_price
            )
            selection = selector.select(
                direction=signal.direction.value,
                quotes=quotes,
                as_of=as_of or date.today(),
                confidence=opportunity.confidence,
                remaining_premium=remaining_premium,
                remaining_contracts=remaining_contracts,
            )
        except Exception as error:  # noqa: BLE001 - explain one preview failure
            rows.append({**base, "Status": f"Option data unavailable: {error}"})
            continue
        if selection is None:
            rows.append({
                **base,
                "Status": "No contract passed DTE, delta, liquidity, spread and premium rules",
            })
            continue

        quote = selection.quote
        rows.append({
            **base,
            "Type": quote.contract_type.upper(),
            "Strike": float(quote.strike),
            "Expiration": quote.expiration.isoformat(),
            "DTE": selection.days_to_expiry,
            "Contracts": selection.quantity,
            "Limit": float(quote.ask),
            "Estimated premium": float(selection.estimated_cost),
            "Contract": quote.symbol,
            "Status": "QUALIFIES — preview only",
        })
        remaining_premium -= selection.estimated_cost
        remaining_contracts -= selection.quantity
        qualifying_contracts += 1

    return rows
