"""Logging setup: sinks, structured payloads and the signal block format."""

from __future__ import annotations

import json
import logging

import pytest

from trading_bot.utils.logging_setup import (
    configure_logging,
    log_banner,
    log_signal_block,
)


@pytest.fixture
def log_dir(tmp_path):
    configure_logging(level="DEBUG", log_dir=tmp_path, json_enabled=True, force=True)
    yield tmp_path
    for handler in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(handler)
        handler.close()


def test_creates_all_expected_sinks(log_dir):
    logging.getLogger("test").info("hello")
    logging.getLogger().handlers[0].flush()
    assert (log_dir / "trading_bot.log").exists()
    assert (log_dir / "errors.log").exists()
    assert (log_dir / "events.jsonl").exists()


def test_error_log_only_captures_errors(log_dir):
    logger = logging.getLogger("test")
    logger.info("routine")
    logger.error("broken")
    for handler in logging.getLogger().handlers:
        handler.flush()
    errors = (log_dir / "errors.log").read_text()
    assert "broken" in errors
    assert "routine" not in errors


def test_signal_block_contains_the_key_fields(log_dir, caplog):
    with caplog.at_level(logging.INFO):
        log_signal_block(
            logging.getLogger("signals"),
            symbol="AAPL", strategy="Momentum", direction="LONG", confidence=82,
            entry=210.50, stop_loss=207.00, take_profit=218.00,
            risk_validation="PASSED", position_size=25,
            reasons=["Bullish EMA crossover", "MACD bullish"],
        )
    text = caplog.text
    assert "AAPL SIGNAL DETECTED" in text
    assert "Confidence      : 82/100" in text
    assert "$210.50" in text
    assert "Risk/Reward     : 1:2.14" in text
    assert "✓ Bullish EMA crossover" in text
    assert "Position Size   : 25 shares" in text


def test_signal_block_emits_machine_readable_json(log_dir):
    log_signal_block(
        logging.getLogger("signals"),
        symbol="NVDA", strategy="Breakout", direction="LONG", confidence=71,
        entry=100.0, stop_loss=98.0, take_profit=106.0, reasons=["Volume spike"],
    )
    for handler in logging.getLogger().handlers:
        handler.flush()
    lines = (log_dir / "events.jsonl").read_text().strip().splitlines()
    record = json.loads(lines[-1])
    assert record["event"] == "signal"
    assert record["symbol"] == "NVDA"
    assert record["confidence"] == 71
    assert record["reasons"] == ["Volume spike"]


def test_signal_block_tolerates_missing_prices(log_dir, caplog):
    with caplog.at_level(logging.INFO):
        log_signal_block(
            logging.getLogger("signals"),
            symbol="SPY", strategy="MeanReversion", direction="SHORT", confidence=64,
        )
    assert "SPY SIGNAL DETECTED" in caplog.text
    assert "Risk/Reward" not in caplog.text


def test_banner_renders_key_values(log_dir, caplog):
    with caplog.at_level(logging.INFO):
        log_banner(logging.getLogger("boot"), "STARTUP", {"Mode": "PAPER"})
    assert "STARTUP" in caplog.text
    assert "Mode" in caplog.text and "PAPER" in caplog.text


def test_configure_logging_is_idempotent(tmp_path):
    configure_logging(log_dir=tmp_path, force=True)
    count = len(logging.getLogger().handlers)
    configure_logging(log_dir=tmp_path)
    assert len(logging.getLogger().handlers) == count
