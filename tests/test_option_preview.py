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
            expiration=date(2026, 11, 20),
            strike=underlying_price,
            bid=5.0,
            ask=5.2,
            delta=0.60,
            daily_volume=100,
            open_interest=500,
        )]


def test_option_preview_returns_human_readable_contract(database, settings):
    signal = SimpleNamespace(
        symbol="META", direction=SimpleNamespace(value="LONG"), entry_price=250.0,
    )
    opportunity = SimpleNamespace(signal=signal, confidence=95)

    rows = preview_option_trades(
        settings,
        [opportunity],
        capacity=1,
        chain=Chain(),
        as_of=date(2026, 9, 9),
    )

    assert rows[0]["Type"] == "CALL"
    assert rows[0]["Strike"] == 250.0
    assert rows[0]["Expiration"] == "2026-11-20"
    assert rows[0]["DTE"] == (date(2026, 11, 20) - date(2026, 9, 9)).days
    assert rows[0]["Status"] == "QUALIFIES — preview only"
