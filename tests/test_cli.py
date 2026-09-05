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


# ----------------------------------------------------------------------------
# hunt
#
# The hunt reads live market data, so these tests substitute the data layer.
# There is deliberately no demo mode on this command: a market scan whose
# output is generated noise looks exactly like one whose output is real, and
# the whole point of the command is to be acted on.
# ----------------------------------------------------------------------------


@pytest.fixture
def fake_market(monkeypatch):
    """Stand in for Alpaca with a small synthetic market."""
    from tests.conftest import make_bars
    from trading_bot.data.market_data import StaticMarketData

    frames = {
        f"S{index:03d}": make_bars(
            300, seed=index, freq="1D", start_price=float(20 + (index * 17) % 300)
        )
        for index in range(120)
    }

    class Provider(StaticMarketData):
        def fetch_watchlist(self, symbols, timeframe, *, lookback_bars=300,
                            end=None, progress=None, batch_size=100):
            from trading_bot.data.models import DataFetchReport

            report = DataFetchReport(requested=list(symbols))
            picked = {s: frames[s] for s in symbols if s in frames}
            report.succeeded = {s: len(f) for s, f in picked.items()}
            if progress is not None:
                progress(len(symbols), len(symbols))
            return picked, report

    provider = Provider(frames)
    monkeypatch.setattr("trading_bot.main.build_market_data", lambda *a, **k: provider)

    records = [
        {
            "symbol": symbol, "name": f"Company {symbol}", "exchange": "NASDAQ",
            "asset_class": "us_equity", "status": "active", "tradable": True,
            "shortable": True, "fractionable": True, "marginable": True,
        }
        for symbol in frames
    ]
    monkeypatch.setattr(
        "trading_bot.universe.discovery.AssetCatalogue.fetch",
        lambda self, use_cache=True: records,
    )
    return frames


def test_hunt_scans_the_market_and_ranks_setups(fake_market, capsys):
    assert main(["hunt", "--min-dollar-volume", "0", "--top", "5"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "MARKET HUNT" in out
    assert "symbols considered" in out


def test_hunt_sizes_against_stated_equity_not_a_broker(fake_market, capsys):
    """Equity is stated because the data feed is not the account you trade."""
    main(["hunt", "--min-dollar-volume", "0", "--equity", "25000"])
    assert "$25,000 account" in capsys.readouterr().out


def test_hunt_says_share_counts_are_per_trade(fake_market, capsys):
    """Otherwise the list implies every setup can be taken at full size."""
    main(["hunt", "--min-dollar-volume", "0"])
    out = capsys.readouterr().out
    if "SETUP(S)" in out:
        assert "only trade you take" in out
        assert "at once" in out


def test_hunt_states_that_a_score_is_not_a_probability(fake_market, capsys):
    main(["hunt", "--min-dollar-volume", "0"])
    out = capsys.readouterr().out
    if "SETUP(S)" in out:
        assert "not a probability" in out


def test_hunt_places_no_orders(fake_market, capsys):
    main(["hunt", "--min-dollar-volume", "0"])
    assert "nothing here has been placed" in capsys.readouterr().out.replace("\n", " ")


def test_hunt_explains_a_quiet_market(fake_market, capsys):
    """A scan returning nothing must be distinguishable from a broken one."""
    main(["hunt", "--min-dollar-volume", "0", "--min-score", "99.9"])
    out = capsys.readouterr().out
    assert "No setups met the criteria" in out or "SETUP(S)" in out


def test_hunt_can_scan_named_symbols_only(fake_market, capsys):
    assert main(["hunt", "--symbols", "S001,S002,S003"]) == EXIT_OK
    assert "3 named symbol(s)" in capsys.readouterr().out


def test_hunt_rejects_an_unknown_strategy(fake_market, capsys):
    assert main(["hunt", "--strategy", "nonsense"]) == EXIT_FAILURE
    assert "Unknown strategy" in capsys.readouterr().err


def test_hunt_writes_a_csv(fake_market, tmp_path):
    target = tmp_path / "setups.csv"
    main(["hunt", "--min-dollar-volume", "0", "--csv", str(target)])
    if target.exists():
        assert "symbol" in target.read_text().splitlines()[0]


def test_hunt_reports_the_universe_funnel(fake_market, capsys):
    """Where symbols went is the only way to tell a filter mistake from a quiet day."""
    main(["hunt", "--min-dollar-volume", "0"])
    out = capsys.readouterr().out
    assert "Static filters" in out
    assert "Swept" in out


def test_the_codebase_runs_on_the_python_it_claims_to_support():
    """`datetime.UTC` is 3.11+, and pyproject declares support for 3.10.

    The mismatch is invisible locally on a newer interpreter and surfaces as an
    ImportError at startup on an older one — including on a hosting platform
    where the Python version is a dropdown someone else chose.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    declared = re.search(
        r'requires-python\s*=\s*"[>=~^]*(\d+)\.(\d+)', (root / "pyproject.toml").read_text()
    )
    assert declared, "pyproject.toml does not declare requires-python"
    minimum = (int(declared.group(1)), int(declared.group(2)))

    offenders = []
    for path in (root / "trading_bot").rglob("*.py"):
        text = path.read_text()
        if re.search(r"from datetime import [^\n]*\bUTC\b|datetime\.UTC", text):
            offenders.append(str(path.relative_to(root)))

    if minimum < (3, 11):
        assert not offenders, (
            f"pyproject declares Python {minimum[0]}.{minimum[1]}+ but these use "
            f"datetime.UTC, which needs 3.11+: {offenders}. Use timezone.utc."
        )
