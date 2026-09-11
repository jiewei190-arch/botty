"""Configuration behaviour, with particular attention to the live-trading locks."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trading_bot.config.settings import (
    LIVE_CONFIRMATION_PHRASE,
    AlpacaSettings,
    DataSettings,
    OptionsSettings,
    RiskSettings,
    Settings,
    TradingMode,
    load_settings,
)


def test_defaults_to_paper_mode():
    assert load_settings().trading_mode is TradingMode.PAPER


def test_options_default_to_alert_only_over_the_swing_holding_window():
    options = OptionsSettings()
    assert options.alert_only is True
    # The hold matches the equity swing strategy: 7-30 calendar days.
    assert options.planned_min_hold_days == 7
    assert options.planned_max_hold_days == 30
    assert options.min_dte <= options.target_dte <= options.max_dte


def test_the_shortest_contract_outlives_the_planned_hold():
    """The floor is derived from the window, not picked.

    A 7-DTE contract held for three weeks expires worthless whatever the stock
    does, and the old default allowed exactly that.
    """
    options = OptionsSettings()
    assert options.min_dte >= options.planned_max_hold_days + options.exit_before_expiry_days


def test_a_contract_that_expires_inside_the_hold_is_refused():
    with pytest.raises(ValidationError, match="OPTIONS_MIN_DTE"):
        OptionsSettings(min_dte=7, planned_max_hold_days=30, exit_before_expiry_days=14)


def test_live_mode_rejected_without_any_lock(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    with pytest.raises(ValidationError, match="ENABLE_LIVE_TRADING"):
        load_settings()


def test_live_mode_rejected_with_only_first_lock(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    with pytest.raises(ValidationError, match="LIVE_TRADING_CONFIRMATION"):
        load_settings()


def test_live_mode_rejected_with_wrong_confirmation_phrase(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("LIVE_TRADING_CONFIRMATION", "yes please")
    with pytest.raises(ValidationError, match="LIVE_TRADING_CONFIRMATION"):
        load_settings()


def test_live_mode_accepted_with_both_locks(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("LIVE_TRADING_CONFIRMATION", LIVE_CONFIRMATION_PHRASE)
    settings = load_settings()
    assert settings.is_live
    assert settings.broker_base_url == settings.alpaca.live_base_url


def test_enabling_live_flag_alone_does_not_arm_live_trading(monkeypatch):
    """The flag without the mode must stay on the paper endpoint."""
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("LIVE_TRADING_CONFIRMATION", LIVE_CONFIRMATION_PHRASE)
    settings = load_settings()
    assert not settings.is_live
    assert settings.broker_base_url == settings.alpaca.paper_base_url


def test_watchlist_parses_comma_separated_env(monkeypatch):
    monkeypatch.setenv("DATA_WATCHLIST", " aapl , msft,nvda ")
    assert load_settings().data.watchlist == ["AAPL", "MSFT", "NVDA"]


def test_watchlist_deduplicates_preserving_order(monkeypatch):
    monkeypatch.setenv("DATA_WATCHLIST", "AAPL,MSFT,AAPL,SPY")
    assert load_settings().data.watchlist == ["AAPL", "MSFT", "SPY"]


def test_empty_watchlist_rejected(monkeypatch):
    monkeypatch.setenv("DATA_WATCHLIST", " , ")
    with pytest.raises(ValidationError):
        load_settings()


def test_risk_per_trade_cannot_exceed_daily_loss_limit():
    with pytest.raises(ValidationError, match="MAX_DAILY_LOSS_PCT"):
        RiskSettings(max_risk_per_trade_pct=5.0, max_daily_loss_pct=3.0)


def test_position_size_cannot_exceed_portfolio_exposure():
    with pytest.raises(ValidationError, match="PORTFOLIO_EXPOSURE"):
        RiskSettings(max_position_size_pct=80.0, max_portfolio_exposure_pct=60.0)


def test_risk_percentages_must_be_positive():
    with pytest.raises(ValidationError):
        RiskSettings(max_risk_per_trade_pct=0)


def test_invalid_data_feed_rejected():
    with pytest.raises(ValidationError, match="data_feed"):
        AlpacaSettings(data_feed="nasdaq")


def test_invalid_adjustment_rejected():
    with pytest.raises(ValidationError, match="adjustment"):
        AlpacaSettings(adjustment="sideways")


def test_secrets_are_masked_in_redacted_dump():
    settings = Settings(
        alpaca=AlpacaSettings(api_key="PKABCDEFGH1234", secret_key="verysecretvalue9876")
    )
    dump = settings.redacted_dict()
    assert dump["alpaca"]["api_key"].endswith("1234")
    assert "PKABCDEFGH" not in dump["alpaca"]["api_key"]
    assert "verysecret" not in dump["alpaca"]["secret_key"]


def test_settings_are_immutable():
    settings = Settings()
    with pytest.raises(ValidationError):
        settings.trading_mode = TradingMode.LIVE


def test_with_overrides_returns_a_new_object():
    original = Settings()
    updated = original.with_overrides(trading_mode=TradingMode.BACKTEST)
    assert original.trading_mode is TradingMode.PAPER
    assert updated.trading_mode is TradingMode.BACKTEST


def test_has_credentials_reflects_both_keys():
    assert not AlpacaSettings(api_key="only-key").has_credentials
    assert AlpacaSettings(api_key="k", secret_key="s").has_credentials


def test_backtest_mode_does_not_use_broker():
    assert not TradingMode.BACKTEST.uses_broker
    assert TradingMode.PAPER.uses_broker


def test_lookback_bars_bounded():
    with pytest.raises(ValidationError):
        DataSettings(lookback_bars=10)
