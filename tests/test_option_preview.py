from datetime import date
from types import SimpleNamespace

from trading_bot.dashboard.data import preview_option_trades
from trading_bot.options import OptionQuote


class Chain:
    def quotes(self, underlying, direction, underlying_price):
        return [OptionQuote(
            symbol=f"{underlying}261120C00250000",
            underlying=underlying,
            contract_type="call" if direction == "LONG" else "put",
            expiration=date(2026, 10, 16),
            strike=underlying_price,
            bid=5.0,
            ask=5.2,
            delta=0.60,
            daily_volume=100,
            open_interest=500,
        )]


class FirstSymbolHasNoContract(Chain):
    def quotes(self, underlying, direction, underlying_price):
        if underlying == "VG":
            return []
        return super().quotes(underlying, direction, underlying_price)


def option_opportunity(symbol, confidence):
    signal = SimpleNamespace(
        symbol=symbol, direction=SimpleNamespace(value="LONG"), entry_price=250.0,
    )
    return SimpleNamespace(signal=signal, confidence=confidence)


def test_option_preview_returns_human_readable_contract(database, settings):
    opportunity = option_opportunity("META", 95)

    rows = preview_option_trades(
        settings,
        [opportunity],
        capacity=1,
        chain=Chain(),
        as_of=date(2026, 9, 9),
    )

    assert rows[0]["Type"] == "CALL"
    assert rows[0]["Strike"] == 250.0
    assert rows[0]["Expiration"] == "2026-10-16"
    assert rows[0]["DTE"] == (date(2026, 10, 16) - date(2026, 9, 9)).days
    assert rows[0]["Status"] == "QUALIFIES — preview only"


def test_option_preview_keeps_searching_after_higher_ranked_symbol_has_no_contract(
    database, settings,
):
    rows = preview_option_trades(
        settings,
        [option_opportunity("VG", 95), option_opportunity("AAPL", 90)],
        capacity=1,
        chain=FirstSymbolHasNoContract(),
        as_of=date(2026, 9, 9),
    )

    assert rows[0]["Underlying"] == "VG"
    assert rows[0]["Status"].startswith("No contract passed")
    assert rows[1]["Underlying"] == "AAPL"
    assert rows[1]["Status"] == "QUALIFIES — preview only"
