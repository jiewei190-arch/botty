"""CLI wiring: argument parsing, exit codes and offline command behaviour."""

from __future__ import annotations

import json

import pytest

from trading_bot.main import EXIT_CONFIG_ERROR, EXIT_FAILURE, EXIT_OK, build_parser, main


@pytest.fixture(autouse=True)
def _sandbox(tmp_path, monkeypatch):
    """Point every runtime path at a temporary directory."""
    monkeypatch.setenv("DATA_DATABASE_PATH", str(tmp_path / "bot.db"))
    monkeypatch.setenv("DATA_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LOG_DIRECTORY", str(tmp_path / "logs"))


def test_parser_requires_a_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_config_command_masks_secrets(capsys, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "PKSECRETKEY1234")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "supersecret5678")
    assert main(["config"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert "PKSECRETKEY" not in payload["alpaca"]["api_key"]
    assert payload["alpaca"]["api_key"].endswith("1234")


def test_db_init_creates_the_schema(capsys, tmp_path):
    assert main(["db-init"]) == EXIT_OK
    assert "schema v1" in capsys.readouterr().out
    assert (tmp_path / "bot.db").exists()


def test_db_init_is_repeatable():
    assert main(["db-init"]) == EXIT_OK
    assert main(["db-init"]) == EXIT_OK


def test_check_fails_cleanly_without_credentials(capsys):
    assert main(["check"]) == EXIT_FAILURE
    output = capsys.readouterr().out
    assert "[FAIL] credentials" in output
    assert "[PASS] database" in output


def test_check_skips_the_broker_in_backtest_mode(capsys):
    main(["--mode", "backtest", "check"])
    output = capsys.readouterr().out
    assert "backtest mode needs no broker" in output


def test_invalid_live_configuration_reports_a_config_error(capsys, monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    assert main(["check"]) == EXIT_CONFIG_ERROR
    assert "ENABLE_LIVE_TRADING" in capsys.readouterr().err


def test_cache_command_reports_an_empty_cache(capsys):
    assert main(["cache"]) == EXIT_OK
    assert "Files           : 0" in capsys.readouterr().out


def test_fetch_without_credentials_fails_gracefully(capsys):
    assert main(["fetch", "--symbols", "AAPL"]) == EXIT_FAILURE
    assert "credentials are missing" in capsys.readouterr().err


def test_mode_override_is_applied(capsys):
    main(["--mode", "backtest", "config"])
    assert json.loads(capsys.readouterr().out)["trading_mode"] == "backtest"


# -- Phase 2: analyze -------------------------------------------------------------


def test_analyze_demo_runs_without_credentials(capsys):
    """The demo path must work with no API keys, and say so loudly."""
    assert main(["analyze", "--demo", "--symbols", "AAPL"]) == EXIT_OK
    output = capsys.readouterr().out
    assert "NOT REAL MARKET DATA" in output
    assert "MARKET ANALYSIS" in output
    for section in ("TREND", "MOVING AVERAGES", "MOMENTUM", "VOLATILITY", "VOLUME", "SIGNALS"):
        assert section in output, section


def test_analyze_demo_handles_multiple_symbols(capsys):
    assert main(["analyze", "--demo", "--symbols", "AAPL,NVDA"]) == EXIT_OK
    output = capsys.readouterr().out
    assert output.count("MARKET ANALYSIS") == 2


def test_analyze_reports_a_trend_direction(capsys):
    main(["analyze", "--demo", "--symbols", "SPY"])
    output = capsys.readouterr().out
    assert any(
        label in output
        for label in ("STRONG_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG_BEARISH")
    )


def test_analyze_without_credentials_fails_cleanly(capsys):
    assert main(["analyze", "--symbols", "AAPL"]) == EXIT_FAILURE
    assert "credentials are missing" in capsys.readouterr().err


def test_analyze_respects_a_bar_count(capsys):
    main(["analyze", "--demo", "--symbols", "AAPL", "--bars", "300"])
    assert "Bars analysed : 300" in capsys.readouterr().out


# -- Phase 3: signals -------------------------------------------------------------


def test_signals_demo_runs_without_credentials(capsys):
    assert main(["signals", "--demo", "--symbols", "AAPL,NVDA"]) == EXIT_OK
    output = capsys.readouterr().out
    assert "NOT REAL MARKET DATA" in output
    assert "Scanned 2 symbol(s)" in output


def test_signals_explains_why_no_setup_was_found(capsys):
    """An idle bot with no explanation is indistinguishable from a broken one."""
    main(["signals", "--demo", "--symbols", "AAPL,NVDA,SPY,QQQ"])
    output = capsys.readouterr().out
    if "No setups met the entry criteria" in output:
        assert "Most common blockers" in output


def test_signals_accepts_a_single_strategy(capsys):
    assert main(["signals", "--demo", "--symbols", "AAPL", "--strategy", "momentum"]) == EXIT_OK
    assert "1 strategy" in capsys.readouterr().out


def test_signals_accepts_a_strategy_list(capsys):
    main(["signals", "--demo", "--symbols", "AAPL", "--strategy", "momentum,breakout"])
    assert "2 strategies" in capsys.readouterr().out


def test_signals_rejects_an_unknown_strategy(capsys):
    assert main(["signals", "--demo", "--strategy", "nonsense"]) == EXIT_FAILURE
    assert "Unknown strategy" in capsys.readouterr().err


def test_signals_confidence_override_is_accepted(capsys):
    assert main(
        ["signals", "--demo", "--symbols", "AAPL", "--min-confidence", "95"]
    ) == EXIT_OK


def test_signals_without_credentials_fails_cleanly(capsys):
    assert main(["signals", "--symbols", "AAPL"]) == EXIT_FAILURE
    assert "credentials are missing" in capsys.readouterr().err


# -- Phase 5: scan ----------------------------------------------------------------


def test_scan_demo_runs_without_credentials(capsys):
    assert main(["scan", "--demo", "--symbols", "AAPL", "--min-dollar-volume", "0"]) == EXIT_OK
    output = capsys.readouterr().out
    assert "MARKET SCAN" in output
    assert "NOT REAL MARKET DATA" in output


def test_scan_ranks_and_scores(capsys):
    main(["scan", "--demo", "--symbols", "AAPL,AMZN", "--bars", "405",
          "--min-dollar-volume", "0"])
    output = capsys.readouterr().out
    if "Confidence:" in output:
        assert "Score breakdown" in output
        assert "Suggested Entry" in output
        assert "Risk/Reward" in output


def test_scan_reports_liquidity_filtering(capsys):
    """A high turnover floor should filter the demo symbols out."""
    main(["scan", "--demo", "--symbols", "AAPL", "--min-dollar-volume", "1e12"])
    assert "Filtered out before analysis" in capsys.readouterr().out


def test_scan_rejects_an_unknown_strategy(capsys):
    assert main(["scan", "--demo", "--strategy", "nonsense"]) == EXIT_FAILURE
    assert "Unknown strategy" in capsys.readouterr().err


def test_scan_without_credentials_fails_cleanly(capsys):
    assert main(["scan", "--symbols", "AAPL"]) == EXIT_FAILURE
    assert "credentials are missing" in capsys.readouterr().err


def test_scan_caps_results(capsys):
    main(["scan", "--demo", "--bars", "405", "--top", "1", "--min-dollar-volume", "0"])
    output = capsys.readouterr().out
    assert output.count("Direction:") <= 1


# ----------------------------------------------------------------------------
# backtest
# ----------------------------------------------------------------------------


def test_backtest_demo_runs_without_credentials(capsys):
    assert main(["backtest", "--demo", "--symbols", "AAPL", "--strategy", "momentum"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "BACKTEST" in out
    assert "Sharpe ratio" in out


def test_backtest_warns_that_demo_data_is_synthetic(capsys):
    main(["backtest", "--demo", "--symbols", "AAPL"])
    assert "NOT REAL MARKET DATA" in capsys.readouterr().out


def test_backtest_says_results_are_not_a_forecast(capsys):
    main(["backtest", "--demo", "--symbols", "AAPL"])
    assert "not a forecast" in capsys.readouterr().out


def test_backtest_honours_starting_capital(capsys):
    main(["backtest", "--demo", "--symbols", "AAPL", "--capital", "50000"])
    assert "$50,000.00" in capsys.readouterr().out


def test_backtest_rejects_an_unknown_strategy(capsys):
    assert main(["backtest", "--demo", "--strategy", "nonsense"]) == EXIT_FAILURE
    assert "Unknown strategy" in capsys.readouterr().err


def test_backtest_rejects_an_unparseable_date():
    with pytest.raises(SystemExit):
        main(["backtest", "--demo", "--start", "last tuesday"])


def test_backtest_rejects_a_backwards_date_range(capsys):
    result = main(
        ["backtest", "--demo", "--start", "2025-06-01", "--end", "2025-05-01"]
    )
    assert result == EXIT_FAILURE
    assert "must be before" in capsys.readouterr().err


def test_backtest_writes_a_trade_csv(tmp_path, capsys):
    target = tmp_path / "trades.csv"
    main(["backtest", "--demo", "--symbols", "AAPL", "--csv", str(target)])
    assert target.exists()
    assert "symbol,strategy,direction" in target.read_text().splitlines()[0]


def test_backtest_writes_a_result_json(tmp_path):
    target = tmp_path / "result.json"
    main(["backtest", "--demo", "--symbols", "AAPL", "--json", str(target)])
    payload = json.loads(target.read_text())
    assert {"metrics", "trades", "symbols"} <= set(payload)


def test_backtest_can_list_every_trade(capsys):
    main(["backtest", "--demo", "--symbols", "AAPL", "--trades"])
    out = capsys.readouterr().out
    assert "REASON" in out


def test_backtest_no_costs_removes_slippage(capsys):
    main(["backtest", "--demo", "--symbols", "AAPL", "--no-costs"])
    assert "Slippage cost     : $0.00" in capsys.readouterr().out


def test_backtest_accepts_multiple_strategies(capsys):
    result = main(
        ["backtest", "--demo", "--symbols", "AAPL", "--strategy", "momentum,breakout"]
    )
    assert result == EXIT_OK
    assert "momentum" in capsys.readouterr().out
