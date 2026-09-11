"""Alert gating, de-duplication and delivery."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pytest

from trading_bot.alerts import (
    Alert,
    AlertChannel,
    AlertConfig,
    AlertManager,
    AlertPriority,
    CallbackChannel,
    ConsoleChannel,
    DatabaseChannel,
    LogChannel,
    render_alert,
)
from trading_bot.detectors import Bias, DetectorSignal, SignalType, score_signals
from trading_bot.utils.market_hours import MarketSession

START = datetime(2026, 9, 8, 15, 0, tzinfo=timezone.utc)


def scored(
    symbol: str = "NVDA",
    *,
    strengths=(9.0, 9.0, 9.0),
    direction: Bias = Bias.BULLISH,
    at: datetime | None = None,
):
    """A score built from as many bullish signals as strengths given."""
    kinds = [
        SignalType.UNUSUAL_VOLUME,
        SignalType.MOMENTUM,
        SignalType.BREAKOUT,
        SignalType.GAP,
        SignalType.VOLATILITY_EXPANSION,
    ]
    signals = [
        DetectorSignal(
            symbol=symbol,
            timestamp=at or START,
            signal_type=kind,
            direction=direction,
            strength=strength,
            reason=f"{kind.value} fired",
            session=MarketSession.REGULAR,
        )
        # strict=False on purpose: fewer strengths than kinds means fewer signals.
        for kind, strength in zip(kinds, strengths, strict=False)
    ]
    return score_signals(symbol, signals, timestamp=at or START, session=MarketSession.REGULAR)


class Capture(AlertChannel):
    name = "capture"

    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    def deliver(self, alert: Alert) -> None:
        self.alerts.append(alert)


class Broken(AlertChannel):
    name = "broken"

    def deliver(self, alert: Alert) -> None:
        raise RuntimeError("channel down")


class TestGating:
    def test_below_the_threshold_is_silent(self):
        manager = AlertManager([Capture()], AlertConfig(min_score=8.0))
        assert manager.consider(scored(strengths=(4.0,))) is None
        assert manager.stats["suppressed_below_threshold"] == 1

    def test_above_the_threshold_alerts(self):
        capture = Capture()
        manager = AlertManager([capture], AlertConfig(min_score=3.0))
        alert = manager.consider(scored(), price=182.4)
        assert alert is not None
        assert capture.alerts == [alert]
        assert alert.price == 182.4
        assert alert.trigger == "new"

    def test_a_directionless_score_is_suppressed_by_default(self):
        manager = AlertManager([Capture()], AlertConfig(min_score=1.0))
        signals = [
            DetectorSignal("X", START, SignalType.MOMENTUM, Bias.BULLISH, 9.0, "up"),
            DetectorSignal("X", START, SignalType.BREAKOUT, Bias.BEARISH, 9.0, "down"),
        ]
        assert manager.consider(score_signals("X", signals)) is None
        assert manager.stats["suppressed_no_direction"] == 1

    def test_conflicted_scores_can_be_filtered(self):
        manager = AlertManager([Capture()], AlertConfig(min_score=1.0, min_agreement=0.9))
        signals = [
            DetectorSignal("X", START, SignalType.MOMENTUM, Bias.BULLISH, 9.0, "up"),
            DetectorSignal("X", START, SignalType.BREAKOUT, Bias.BEARISH, 3.0, "down"),
        ]
        assert manager.consider(score_signals("X", signals)) is None
        assert manager.stats["suppressed_conflicted"] == 1


class TestCooldown:
    def manager(self, **overrides):
        defaults = {"min_score": 1.0, "cooldown_seconds": 900}
        return AlertManager([Capture()], AlertConfig(**(defaults | overrides)))

    def test_the_same_setup_is_reported_once_per_window(self):
        manager = self.manager()
        assert manager.consider(scored(at=START)) is not None
        assert manager.consider(scored(at=START + timedelta(minutes=5))) is None
        assert manager.stats["suppressed_cooldown"] == 1

    def test_the_window_reopens_once_it_elapses(self):
        manager = self.manager()
        assert manager.consider(scored(at=START)) is not None
        later = manager.consider(scored(at=START + timedelta(minutes=20)))
        assert later is not None
        assert later.trigger == "repeat"

    def test_a_materially_stronger_setup_escalates_early(self):
        manager = self.manager(escalation_delta=1.5)
        assert manager.consider(scored(strengths=(6.0,))) is not None
        escalated = manager.consider(
            scored(strengths=(9.0, 9.0, 9.0, 9.0), at=START + timedelta(minutes=2))
        )
        assert escalated is not None
        assert escalated.trigger == "escalation"

    def test_a_marginally_stronger_setup_does_not(self):
        manager = self.manager(escalation_delta=1.5)
        assert manager.consider(scored(strengths=(6.0,))) is not None
        assert manager.consider(
            scored(strengths=(6.4,), at=START + timedelta(minutes=2))
        ) is None

    def test_the_opposite_direction_is_a_different_alert(self):
        manager = self.manager()
        assert manager.consider(scored(at=START)) is not None
        flipped = manager.consider(
            scored(direction=Bias.BEARISH, at=START + timedelta(minutes=1))
        )
        assert flipped is not None

    def test_cooldown_uses_the_score_timestamp_not_the_clock(self):
        """Replaying a day of bars must produce that day's alerts, deterministically."""
        manager = self.manager()
        assert manager.consider(scored(at=START)) is not None
        assert manager.consider(scored(at=START + timedelta(seconds=60))) is None
        assert manager.consider(scored(at=START + timedelta(seconds=1800))) is not None

    def test_the_session_budget_is_a_hard_cap(self):
        manager = self.manager(max_per_symbol_per_session=2, cooldown_seconds=0)
        for minute in range(6):
            manager.consider(scored(at=START + timedelta(minutes=minute)))
        assert manager.stats["sent"] == 2
        assert manager.stats["suppressed_session_budget"] >= 1

    def test_a_new_trading_day_resets_the_budget(self):
        manager = self.manager(max_per_symbol_per_session=1)
        assert manager.consider(scored(at=START)) is not None
        tomorrow = START + timedelta(days=1)
        assert manager.consider(scored(at=tomorrow)) is not None

    def test_priming_restores_the_cooldown_after_a_restart(self):
        manager = self.manager()
        manager.prime("NVDA:BULLISH", timestamp=START, score=9.0)
        assert manager.consider(scored(at=START + timedelta(minutes=1))) is None


class TestDelivery:
    def test_a_broken_channel_does_not_stop_the_others(self, caplog):
        capture = Capture()
        manager = AlertManager([Broken(), capture], AlertConfig(min_score=1.0))
        with caplog.at_level(logging.ERROR):
            assert manager.consider(scored()) is not None
        assert len(capture.alerts) == 1
        assert manager.stats["channel_error_broken"] == 1

    def test_console_channel_prints_the_block(self, capsys):
        manager = AlertManager([ConsoleChannel()], AlertConfig(min_score=1.0))
        manager.consider(scored(), price=100.0)
        out = capsys.readouterr().out
        assert "PRIORITY MARKET SETUP" in out
        assert "Symbol: NVDA" in out

    def test_log_channel_carries_the_payload_for_the_json_handler(self, caplog):
        manager = AlertManager([LogChannel()], AlertConfig(min_score=1.0))
        with caplog.at_level(logging.WARNING):
            manager.consider(scored())
        record = next(r for r in caplog.records if getattr(r, "event", None) == "alert")
        assert record.alert["symbol"] == "NVDA"
        assert record.alert["signals"]

    def test_database_channel_persists_the_alert(self, database):
        manager = AlertManager([DatabaseChannel(database.alerts)], AlertConfig(min_score=1.0))
        alert = manager.consider(scored(), price=100.0)
        assert alert is not None
        rows = database.alerts.recent()
        assert len(rows) == 1
        assert rows[0]["symbol"] == "NVDA"
        assert rows[0]["dedupe_key"] == "NVDA:BULLISH"
        assert rows[0]["payload"]["signals"]

    def test_callback_channel_is_the_seam_for_webhooks(self):
        received: list[Alert] = []
        manager = AlertManager(
            [CallbackChannel(received.append, name="discord")], AlertConfig(min_score=1.0)
        )
        manager.consider(scored())
        assert len(received) == 1


class TestPriority:
    @pytest.mark.parametrize(
        "score,expected",
        [
            (5.0, AlertPriority.LOW),
            (6.1, AlertPriority.MEDIUM),
            (7.6, AlertPriority.HIGH),
        ],
    )
    def test_priority_is_relative_to_the_configured_threshold(self, score, expected):
        assert AlertPriority.from_score(score, threshold=5.0) is expected

    def test_lowering_the_threshold_relabels_the_same_score(self):
        assert AlertPriority.from_score(3.5, threshold=2.0) is AlertPriority.HIGH
        assert AlertPriority.from_score(3.5, threshold=5.0) is AlertPriority.LOW


def test_render_alert_includes_every_field_a_reader_needs():
    alert = Alert.from_score(scored(), threshold=3.0, price=182.41)
    text = render_alert(alert)
    for expected in ("Symbol: NVDA", "$182.41", "Direction: BULLISH", "Overall Score:",
                     "Market Session: regular"):
        assert expected in text
    assert "Relative Volume Spike" in text


def test_config_rejects_impossible_values():
    with pytest.raises(ValueError):
        AlertConfig(cooldown_seconds=-1)
    with pytest.raises(ValueError):
        AlertConfig(max_per_symbol_per_session=0)
    with pytest.raises(ValueError):
        AlertConfig(min_agreement=1.5)
