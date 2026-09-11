"""A scanned play has to name the contract it would be traded through.

The bot can find a good chart and still leave the reader guessing what to
actually buy. These tests pin the four things a trader needs on screen —
ticker, strike, call or put, and how long it has to work — everywhere a play is
shown.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from trading_bot.config.settings import OptionsSettings, load_settings
from trading_bot.main import _option_contracts, _render_contract

QUALIFYING = {
    "Underlying": "NVDA",
    "Direction": "LONG",
    "Type": "CALL",
    "Strike": 185.0,
    "Expiration": "2026-11-20",
    "DTE": 70,
    "Contracts": 1,
    "Limit": 8.35,
    "Estimated premium": 835.0,
    "Contract": "NVDA261120C00185000",
    "Status": "QUALIFIES — preview only",
}


class TestContractLine:
    def test_names_ticker_strike_type_and_dte(self, capsys):
        _render_contract(QUALIFYING)
        out = capsys.readouterr().out
        assert "NVDA" in out
        assert "$185" in out
        assert "CALL" in out
        assert "2026-11-20" in out
        assert "70DTE" in out

    def test_shows_the_premium_a_reader_would_pay(self, capsys):
        _render_contract(QUALIFYING)
        out = capsys.readouterr().out
        assert "$8.35" in out
        assert "835" in out

    def test_puts_read_as_puts(self, capsys):
        _render_contract({**QUALIFYING, "Type": "PUT", "Direction": "SHORT"})
        out = capsys.readouterr().out
        assert "PUT" in out
        assert "CALL" not in out

    def test_explains_itself_when_nothing_qualified(self, capsys):
        _render_contract({"Underlying": "AMD", "Status": "No contract passed the rules"})
        assert "No contract passed the rules" in capsys.readouterr().out

    def test_prints_nothing_when_there_is_no_contract(self, capsys):
        _render_contract(None)
        assert capsys.readouterr().out == ""


class TestContractLookup:
    def sweep(self):
        signal = SimpleNamespace(symbol="NVDA", direction=SimpleNamespace(value="LONG"))
        return SimpleNamespace(
            opportunities=[SimpleNamespace(signal=signal, confidence=88.0)],
            concurrent_capacity=1,
        )

    def test_skipped_on_request(self, settings):
        assert _option_contracts(settings, self.sweep(), skip=True) == {}

    def test_skipped_without_credentials(self, settings):
        """No key means no chain; the stock plan still has to print."""
        bare = settings.with_overrides(
            alpaca=settings.alpaca.model_copy(update={"api_key": None, "secret_key": None})
        )
        assert not bare.alpaca.has_credentials
        assert _option_contracts(bare, self.sweep(), skip=False) == {}

    def test_skipped_when_options_are_disabled(self, monkeypatch):
        monkeypatch.setenv("OPTIONS_ENABLED", "false")
        monkeypatch.setenv("ALPACA_API_KEY", "PKTEST")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
        assert _option_contracts(load_settings(), self.sweep(), skip=False) == {}

    def test_a_broken_chain_does_not_end_the_scan(self, monkeypatch, capsys):
        """An option chain is a second API. Losing it costs the contract, not the scan."""
        monkeypatch.setenv("ALPACA_API_KEY", "PKTEST")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
        import trading_bot.options.preview as preview

        def explode(*args, **kwargs):
            raise RuntimeError("option chain endpoint unavailable")

        monkeypatch.setattr(preview, "preview_option_trades", explode)
        assert _option_contracts(load_settings(), self.sweep(), skip=False) == {}
        assert "option contracts unavailable" in capsys.readouterr().out


class TestHoldingWindowAgreement:
    """The contract has to outlive the trade it is bought for."""

    def test_the_shortest_contract_survives_the_longest_hold(self):
        options = OptionsSettings()
        assert options.min_dte >= (
            options.planned_max_hold_days + options.exit_before_expiry_days
        )

    def test_the_option_hold_matches_the_equity_hold(self):
        from trading_bot.strategies.swing_quality import SwingQualityConfig

        options = OptionsSettings()
        equity_cap_days = SwingQualityConfig().max_holding_bars  # trading days
        # 21 trading days is about 30 calendar days; the option plan must not
        # promise to hold for less than the strategy intends to.
        assert options.planned_max_hold_days >= equity_cap_days

    def test_an_incoherent_window_is_refused(self):
        with pytest.raises(ValueError, match="OPTIONS_MIN_DTE"):
            OptionsSettings(min_dte=7, planned_max_hold_days=30, exit_before_expiry_days=14)
