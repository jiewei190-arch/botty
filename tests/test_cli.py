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
