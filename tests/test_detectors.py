"""The five signal detectors, each exercised on its own.

Every detector gets hand-built bars that isolate exactly the behaviour under
test. Generated market-like noise would prove that the detectors run; it would
not prove that a volume spike fires the volume detector and nothing else.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from tests.conftest import session_bars
from trading_bot.detectors import (
    Bias,
    BreakoutDetector,
    DetectorEngine,
    DetectorSignal,
    GapDetector,
    MomentumDetector,
    ScoringWeights,
    SignalType,
    VolatilityDetector,
    VolumeDetector,
    build_context,
    consecutive_run,
    rank,
    scale_strength,
    score_signals,
)
from trading_bot.detectors.base import (
    BreakoutDetectorConfig,
    Detector,
    GapDetectorConfig,
    VolumeDetectorConfig,
)
from trading_bot.utils.market_hours import MarketSession


def bars_from(
    closes,
    *,
    start: str = "2026-09-08 13:30",
    freq: str = "5min",
    volumes=None,
    spans=None,
    opens=None,
) -> pd.DataFrame:
    """Build a bar frame from a close series, with optional control of the rest."""
    closes = np.asarray(closes, dtype="float64")
    count = len(closes)
    index = pd.date_range(start, periods=count, freq=freq, tz="UTC", name="timestamp")
    open_ = np.r_[closes[0], closes[:-1]] if opens is None else np.asarray(opens, dtype="float64")
    span = np.full(count, 0.05) if spans is None else np.asarray(spans, dtype="float64")
    volume = np.full(count, 1_000.0) if volumes is None else np.asarray(volumes, dtype="float64")
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, closes) + span,
            "low": np.minimum(open_, closes) - span,
            "close": closes,
            "volume": volume,
        },
        index=index,
    )


def stack(*frames: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat(frames)
    combined.index.name = "timestamp"
    return combined


def context_for(frame: pd.DataFrame, **kwargs):
    return build_context("TEST", frame, now=frame.index[-1].to_pydatetime(), **kwargs)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


class TestDetectionContext:
    def test_groups_bars_into_sessions(self, intraday_sessions):
        context = context_for(intraday_sessions)
        assert context.intraday
        assert len(context.sessions) == 12
        assert all(session.regular is not None for session in context.sessions)
        assert context.today.day == date(2026, 9, 4)
        assert context.previous.day == date(2026, 9, 3)

    def test_splits_premarket_from_regular(self):
        # 08:00-19:55 UTC covers 04:00-15:55 New York: pre-market then regular.
        frame = bars_from(np.full(144, 100.0), start="2026-09-08 08:00")
        context = context_for(frame)
        today = context.today
        assert today.premarket is not None
        assert today.premarket.bars == 66      # 04:00-09:30 in 5-minute bars
        assert today.regular is not None
        assert today.regular.bars == 78        # 09:30-16:00

    def test_bars_outside_any_session_are_excluded(self):
        """A 21:00 ET print belongs to no session and must not extend the day."""
        regular = bars_from(np.full(78, 100.0), start="2026-09-08 13:30")
        overnight = bars_from([200.0], start="2026-09-09 01:30")
        context = context_for(stack(regular, overnight))
        assert context.sessions[-1].regular is not None
        assert context.sessions[-1].regular.high < 150

    def test_daily_bars_make_each_bar_a_session(self):
        frame = bars_from(np.linspace(100, 110, 30), start="2026-08-03", freq="B")
        context = context_for(frame)
        assert not context.intraday
        assert len(context.sessions) == 21     # history cap plus today
        assert context.today.premarket is None

    def test_empty_frame_yields_an_empty_context(self):
        context = build_context("X", pd.DataFrame())
        assert context.sessions == ()
        assert context.last_price is None
        assert context.previous_close is None

    def test_volume_curve_is_cumulative(self, intraday_sessions):
        context = context_for(intraday_sessions)
        curve = context.session_volume_curve(context.today.day)
        assert curve[0] > 0
        assert np.all(np.diff(curve) >= 0)
        assert curve[-1] == pytest.approx(context.today.regular.volume)


# ---------------------------------------------------------------------------
# Unusual volume
# ---------------------------------------------------------------------------


class TestVolumeDetector:
    def test_silent_on_ordinary_volume(self, intraday_sessions):
        assert VolumeDetector().evaluate(context_for(intraday_sessions)) is None

    def test_fires_when_the_session_trades_four_times_its_norm(self):
        quiet = session_bars(sessions=11, end_day="2026-09-03", volume=(2_000, 2_100))
        busy = session_bars(
            sessions=1, end_day="2026-09-04", volume=(8_000, 8_100), start_price=105.0
        )
        signal = VolumeDetector().evaluate(context_for(stack(quiet, busy)))
        assert signal is not None
        assert signal.signal_type is SignalType.UNUSUAL_VOLUME
        assert signal.metric("relative_volume") == pytest.approx(4.0, abs=0.2)
        # 4x maps to (4.0 - 1.5) / (5.0 - 1.5) on the 0-10 scale.
        assert signal.strength == pytest.approx(7.1, abs=0.5)

    def test_compares_like_with_like_partway_through_a_session(self):
        """A half-finished session must not look quiet against full-day totals."""
        history = session_bars(sessions=8, end_day="2026-09-03", volume=(2_000, 2_001))
        # Ten bars of today, at the same per-bar volume as every prior session.
        partial = bars_from(
            np.full(10, 100.0), start="2026-09-04 13:30", volumes=np.full(10, 2_000.0)
        )
        context = context_for(stack(history, partial))
        signal = VolumeDetector().evaluate(context)
        # Ten bars against ten bars is 1.0x — ordinary, so no signal at all.
        assert signal is None

    def test_direction_comes_from_the_price_move(self):
        quiet = session_bars(sessions=11, end_day="2026-09-03", volume=(2_000, 2_100))
        falling = bars_from(
            np.linspace(100, 94, 78),
            start="2026-09-04 13:30",
            volumes=np.full(78, 9_000.0),
        )
        signal = VolumeDetector().evaluate(context_for(stack(quiet, falling)))
        assert signal is not None
        assert signal.direction is Bias.BEARISH

    def test_flat_price_on_heavy_volume_is_neutral(self):
        quiet = session_bars(sessions=11, end_day="2026-09-03", volume=(2_000, 2_100))
        flat = bars_from(
            np.full(78, 100.0), start="2026-09-04 13:30", volumes=np.full(78, 9_000.0)
        )
        signal = VolumeDetector().evaluate(context_for(stack(quiet, flat)))
        assert signal is not None
        assert signal.direction is Bias.NEUTRAL

    def test_premarket_volume_is_judged_against_prior_premarkets(self):
        """Before the bell there is no regular volume, so pre-market stands in."""
        frames = []
        for index, day in enumerate(["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04"]):
            volume = 30_000.0 if index == 3 else 1_000.0
            frames.append(
                bars_from(
                    np.full(66, 100.0 + index),
                    start=f"{day} 08:00",
                    volumes=np.full(66, volume),
                )
            )
        context = build_context(
            "TEST", stack(*frames), now=datetime(2026, 9, 4, 13, 0, tzinfo=timezone.utc)
        )
        signal = VolumeDetector().evaluate(context)
        assert signal is not None
        assert "pre-market" in signal.reason
        assert signal.metric("relative_volume") == pytest.approx(30.0, abs=0.5)

    def test_needs_a_minimum_of_history(self):
        frame = session_bars(sessions=2, end_day="2026-09-04", volume=(9_000, 9_100))
        detector = VolumeDetector(VolumeDetectorConfig(min_sessions=5))
        assert detector.evaluate(context_for(frame)) is None


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------


class TestMomentumDetector:
    def test_fires_on_a_sharp_run(self):
        closes = np.r_[np.full(40, 100.0), np.linspace(100.6, 103.0, 5)]
        signal = MomentumDetector().evaluate(context_for(bars_from(closes)))
        assert signal is not None
        assert signal.direction is Bias.BULLISH
        assert signal.metric("change_pct") == pytest.approx(3.0, abs=0.2)
        assert signal.metric("consecutive_bars") == 5

    def test_fires_bearish_on_a_sharp_drop(self):
        closes = np.r_[np.full(40, 100.0), np.linspace(99.4, 97.0, 5)]
        signal = MomentumDetector().evaluate(context_for(bars_from(closes)))
        assert signal is not None
        assert signal.direction is Bias.BEARISH

    def test_silent_on_slow_drift(self):
        closes = np.linspace(100, 100.4, 60)
        assert MomentumDetector().evaluate(context_for(bars_from(closes))) is None

    def test_atr_normalisation_ranks_a_calm_stock_higher_for_the_same_percent(self):
        """The same 3% move is a bigger event in a quiet symbol than a wild one."""
        # Identical closes, different bar ranges: only the ATR differs.
        closes = np.r_[np.full(40, 100.0), np.linspace(100.6, 103.0, 5)]
        calm = bars_from(closes, spans=np.full(45, 0.02))
        wild = bars_from(closes, spans=np.full(45, 1.5))
        calm_signal = MomentumDetector().evaluate(context_for(calm))
        wild_signal = MomentumDetector().evaluate(context_for(wild))
        assert calm_signal is not None and wild_signal is not None
        assert calm_signal.metric("atr_multiple") > wild_signal.metric("atr_multiple")

    def test_a_run_against_the_move_adds_nothing(self):
        """A move that ends in bars going the other way is weakening, not stronger."""
        # Same 2% move, same bar ranges — only the shape of the path differs.
        # Sized to land mid-scale: at full saturation both would read 10 and the
        # comparison would prove nothing.
        spans = np.full(45, 0.3)
        with_run = np.r_[np.full(40, 100.0), [100.4, 100.8, 101.2, 101.6, 102.0]]
        spiky = np.r_[np.full(40, 100.0), [102.4, 102.3, 102.2, 102.1, 102.0]]
        strong = MomentumDetector().evaluate(context_for(bars_from(with_run, spans=spans)))
        weak = MomentumDetector().evaluate(context_for(bars_from(spiky, spans=spans)))
        assert strong is not None and weak is not None
        assert strong.strength > weak.strength

    @pytest.mark.parametrize(
        "closes,expected",
        [
            ([1.0, 2, 3, 4], 3),
            ([4.0, 3, 2, 1], -3),
            ([1.0, 2, 2, 3], 1),
            ([1.0], 0),
            ([1.0, 1.0], 0),
        ],
    )
    def test_consecutive_run(self, closes, expected):
        assert consecutive_run(np.array(closes)) == expected


# ---------------------------------------------------------------------------
# Breakouts
# ---------------------------------------------------------------------------


class TestBreakoutDetector:
    def two_days(self, final_close: float, *, prior: float = 99.5):
        day_one = bars_from(np.full(78, 100.0), start="2026-09-03 13:30")
        day_two = bars_from(
            np.r_[np.full(77, prior), [final_close]], start="2026-09-04 13:30"
        )
        return stack(day_one, day_two)

    def test_fires_on_a_close_above_the_previous_day_high(self):
        signal = BreakoutDetector().evaluate(context_for(self.two_days(101.5)))
        assert signal is not None
        assert signal.direction is Bias.BULLISH
        assert "previous day high" in signal.reason

    def test_names_the_most_significant_level_broken(self):
        """Yesterday's high is the headline even when a smaller level cleared wider."""
        signal = BreakoutDetector().evaluate(context_for(self.two_days(101.5)))
        assert signal is not None
        assert signal.metric("level_significance") == 1.0
        assert signal.metric("levels_broken") == 2
        assert "intraday_high" in signal.reason

    def test_a_touch_inside_the_buffer_is_not_a_breakout(self):
        # Both days sit at 100.0, so every level is 100.05 (close plus the 0.05
        # span) and a close of 100.06 clears none of them by the 0.05% buffer.
        frame = self.two_days(100.06, prior=100.0)
        assert BreakoutDetector().evaluate(context_for(frame)) is None

    def test_the_break_is_only_reported_once(self):
        broken = self.two_days(101.5)
        assert BreakoutDetector().evaluate(context_for(broken)) is not None
        extended = stack(broken, bars_from([101.6], start="2026-09-04 20:00"))
        assert BreakoutDetector().evaluate(context_for(extended)) is None

    def test_repeat_reporting_can_be_enabled(self):
        broken = stack(self.two_days(101.5), bars_from([101.6], start="2026-09-04 20:00"))
        detector = BreakoutDetector(BreakoutDetectorConfig(require_fresh=False))
        assert detector.evaluate(context_for(broken)) is not None

    def test_breakdown_below_the_previous_day_low(self):
        signal = BreakoutDetector().evaluate(context_for(self.two_days(98.0, prior=100.0)))
        assert signal is not None
        assert signal.direction is Bias.BEARISH
        assert "previous day low" in signal.reason

    def test_intraday_range_excludes_the_bar_being_judged(self):
        """A bar is always inside a range that contains it, so it must not."""
        rising = np.r_[np.full(20, 100.0), [101.0]]
        frame = bars_from(rising, start="2026-09-04 13:30")
        signal = BreakoutDetector().evaluate(context_for(frame))
        assert signal is not None
        assert signal.metric("level") == pytest.approx(100.05)

    def test_no_intraday_level_before_the_session_has_a_range(self):
        frame = bars_from(np.r_[np.full(3, 100.0), [102.0]], start="2026-09-04 13:30")
        assert BreakoutDetector().evaluate(context_for(frame)) is None

    def test_premarket_high_is_a_level(self):
        premarket = bars_from(np.full(66, 100.0), start="2026-09-04 08:00")
        regular = bars_from([99.0, 101.0], start="2026-09-04 13:30")
        signal = BreakoutDetector().evaluate(context_for(stack(premarket, regular)))
        assert signal is not None
        assert "premarket high" in signal.reason


# ---------------------------------------------------------------------------
# Gaps
# ---------------------------------------------------------------------------


class TestGapDetector:
    def gapped(self, open_price: float, closes=None):
        yesterday = bars_from(np.full(78, 100.0), start="2026-09-03 13:30")
        series = closes if closes is not None else np.full(10, open_price)
        today = bars_from(
            series,
            start="2026-09-04 13:30",
            opens=np.r_[open_price, np.asarray(series, dtype="float64")[:-1]],
        )
        return stack(yesterday, today)

    def test_fires_on_a_gap_up(self):
        signal = GapDetector().evaluate(context_for(self.gapped(104.0)))
        assert signal is not None
        assert signal.direction is Bias.BULLISH
        assert signal.metric("gap_pct") == pytest.approx(4.0, abs=0.05)

    def test_fires_on_a_gap_down(self):
        signal = GapDetector().evaluate(context_for(self.gapped(96.0)))
        assert signal is not None
        assert signal.direction is Bias.BEARISH

    def test_silent_below_the_threshold(self):
        assert GapDetector().evaluate(context_for(self.gapped(100.5))) is None

    def test_a_filled_gap_scores_lower_than_one_that_held(self):
        held = GapDetector().evaluate(context_for(self.gapped(104.0)))
        filled = GapDetector().evaluate(
            context_for(self.gapped(104.0, closes=np.linspace(104.0, 100.0, 10)))
        )
        assert held is not None and filled is not None
        assert filled.metric("gap_fill_pct") == pytest.approx(100.0, abs=1)
        assert filled.strength < held.strength
        # The measurement itself survives the damping.
        assert filled.metric("raw_strength") == pytest.approx(held.metric("raw_strength"))

    def test_premarket_price_stands_in_before_the_open(self):
        yesterday = bars_from(np.full(78, 100.0), start="2026-09-03 13:30")
        premarket = bars_from(np.full(20, 103.0), start="2026-09-04 08:00")
        context = build_context(
            "TEST",
            stack(yesterday, premarket),
            now=datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc),
        )
        signal = GapDetector().evaluate(context)
        assert signal is not None
        assert signal.metric("premarket_basis") == 1.0
        assert "Pre-market" in signal.reason

    def test_threshold_is_configurable(self):
        detector = GapDetector(GapDetectorConfig(threshold_pct=0.2))
        assert detector.evaluate(context_for(self.gapped(100.5))) is not None


# ---------------------------------------------------------------------------
# Volatility expansion
# ---------------------------------------------------------------------------


class TestVolatilityDetector:
    def test_fires_when_bar_ranges_widen(self):
        spans = np.r_[np.full(35, 0.03), np.full(5, 0.6)]
        closes = np.r_[np.full(35, 100.0), [100.5, 101.2, 102.0, 102.6, 103.4]]
        signal = VolatilityDetector().evaluate(context_for(bars_from(closes, spans=spans)))
        assert signal is not None
        assert signal.signal_type is SignalType.VOLATILITY_EXPANSION
        assert signal.metric("expansion_ratio") > 2

    def test_silent_when_ranges_are_steady(self):
        frame = bars_from(np.full(60, 100.0), spans=np.full(60, 0.05))
        assert VolatilityDetector().evaluate(context_for(frame)) is None

    def test_two_sided_expansion_reports_no_direction(self):
        spans = np.r_[np.full(35, 0.03), np.full(5, 0.6)]
        closes = np.r_[np.full(35, 100.0), [100.4, 99.7, 100.3, 99.8, 100.02]]
        signal = VolatilityDetector().evaluate(context_for(bars_from(closes, spans=spans)))
        assert signal is not None
        assert signal.direction is Bias.NEUTRAL
        assert "two-sided" in signal.reason

    def test_needs_the_slow_window_before_it_can_speak(self):
        frame = bars_from(np.full(10, 100.0), spans=np.r_[np.full(5, 0.02), np.full(5, 1.0)])
        assert VolatilityDetector().evaluate(context_for(frame)) is None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def signal(kind: SignalType, direction: Bias, strength: float, symbol: str = "NVDA"):
    return DetectorSignal(
        symbol=symbol,
        timestamp=datetime(2026, 9, 4, 18, 0, tzinfo=timezone.utc),
        signal_type=kind,
        direction=direction,
        strength=strength,
        reason="test",
        session=MarketSession.REGULAR,
    )


class TestScoring:
    def test_a_silent_detector_counts_as_zero_not_as_missing(self):
        """Confluence is the point: four confirmations must outrank one loud reading."""
        alone = score_signals("A", [signal(SignalType.UNUSUAL_VOLUME, Bias.BULLISH, 10.0)])
        together = score_signals(
            "B",
            [
                signal(SignalType.UNUSUAL_VOLUME, Bias.BULLISH, 8.0),
                signal(SignalType.MOMENTUM, Bias.BULLISH, 8.0),
                signal(SignalType.BREAKOUT, Bias.BULLISH, 8.0),
                signal(SignalType.GAP, Bias.BULLISH, 8.0),
            ],
        )
        assert together.overall > alone.overall

    def test_agreement_is_one_when_every_directional_signal_agrees(self):
        score = score_signals(
            "A",
            [
                signal(SignalType.MOMENTUM, Bias.BULLISH, 8.0),
                signal(SignalType.BREAKOUT, Bias.BULLISH, 6.0),
            ],
        )
        assert score.agreement == pytest.approx(1.0)
        assert score.direction is Bias.BULLISH
        assert not score.conflicted

    def test_opposing_signals_are_flagged_rather_than_averaged_away(self):
        score = score_signals(
            "A",
            [
                signal(SignalType.MOMENTUM, Bias.BULLISH, 8.0),
                signal(SignalType.BREAKOUT, Bias.BEARISH, 8.0),
            ],
        )
        assert score.agreement == pytest.approx(0.0)
        assert score.conflicted
        assert score.overall > 0  # the evidence is real, the direction is not

    def test_a_neutral_signal_does_not_reduce_agreement(self):
        score = score_signals(
            "A",
            [
                signal(SignalType.MOMENTUM, Bias.BULLISH, 8.0),
                signal(SignalType.VOLATILITY_EXPANSION, Bias.NEUTRAL, 9.0),
            ],
        )
        assert score.agreement == pytest.approx(1.0)

    def test_duplicate_types_keep_the_strongest(self):
        score = score_signals(
            "A",
            [
                signal(SignalType.MOMENTUM, Bias.BULLISH, 4.0),
                signal(SignalType.MOMENTUM, Bias.BULLISH, 9.0),
            ],
        )
        assert len(score.signals) == 1
        assert score.component(SignalType.MOMENTUM) == 9.0

    def test_weights_change_the_ranking(self):
        signals = [signal(SignalType.GAP, Bias.BULLISH, 10.0)]
        low = score_signals("A", signals)
        high = score_signals("A", signals, weights=ScoringWeights(gap=1.0, unusual_volume=0.0,
                                                                 momentum=0.0, breakout=0.0,
                                                                 volatility_expansion=0.0))
        assert high.overall > low.overall
        assert high.overall == pytest.approx(10.0)

    def test_rank_breaks_ties_on_the_number_of_detectors(self):
        one = score_signals(
            "ONE",
            [signal(SignalType.MOMENTUM, Bias.BULLISH, 10.0, "ONE")],
            weights=ScoringWeights(momentum=1.0, unusual_volume=0.0, breakout=0.0,
                                   gap=0.0, volatility_expansion=0.0),
        )
        two = score_signals(
            "TWO",
            [
                signal(SignalType.MOMENTUM, Bias.BULLISH, 10.0, "TWO"),
                signal(SignalType.BREAKOUT, Bias.BULLISH, 10.0, "TWO"),
            ],
            weights=ScoringWeights(momentum=1.0, unusual_volume=0.0, breakout=0.0,
                                   gap=0.0, volatility_expansion=0.0),
        )
        assert one.overall == pytest.approx(two.overall)
        assert rank([one, two])[0].symbol == "TWO"

    def test_negative_weights_are_rejected(self):
        with pytest.raises(ValueError, match="cannot be negative"):
            ScoringWeights(momentum=-1.0)

    def test_all_zero_weights_are_rejected(self):
        with pytest.raises(ValueError, match="greater than zero"):
            ScoringWeights(unusual_volume=0, momentum=0, breakout=0, gap=0,
                           volatility_expansion=0)

    def test_describe_reads_like_the_specified_block(self):
        text = score_signals(
            "NVDA", [signal(SignalType.UNUSUAL_VOLUME, Bias.BULLISH, 9.1)]
        ).describe()
        assert "SYMBOL: NVDA" in text
        assert "OVERALL SCORE:" in text
        assert "UNUSUAL_VOLUME: 9.1" in text


class TestStrengthScale:
    def test_floor_maps_to_zero_and_ceiling_to_ten(self):
        assert scale_strength(1.5, floor=1.5, ceiling=5.0) == 0.0
        assert scale_strength(5.0, floor=1.5, ceiling=5.0) == 10.0

    def test_saturates_rather_than_letting_an_outlier_dominate(self):
        assert scale_strength(400.0, floor=1.5, ceiling=5.0) == 10.0

    def test_non_finite_values_score_zero(self):
        assert scale_strength(float("nan"), floor=1.0, ceiling=2.0) == 0.0
        assert scale_strength(float("inf"), floor=1.0, ceiling=2.0) == 0.0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ExplodingDetector(Detector):
    name = "exploding"
    signal_type = SignalType.MOMENTUM

    def evaluate(self, context):
        raise RuntimeError("boom")


class TestDetectorEngine:
    def test_one_broken_detector_does_not_end_the_scan(self, intraday_sessions, caplog):
        engine = DetectorEngine(detectors=[ExplodingDetector(), GapDetector()])
        analysis = engine.analyze("TEST", intraday_sessions)
        assert analysis.failures == ("exploding",)
        assert "boom" in caplog.text

    def test_missing_frames_still_produce_an_analysis(self):
        engine = DetectorEngine()
        results = engine.analyze_many({"AAPL": pd.DataFrame()})
        assert len(results) == 1
        assert results[0].overall == 0.0
        assert not results[0].fired

    def test_analyze_many_returns_the_best_first(self):
        quiet = session_bars(sessions=11, end_day="2026-09-03", volume=(2_000, 2_100))
        busy = stack(
            quiet,
            session_bars(sessions=1, end_day="2026-09-04", volume=(9_000, 9_100),
                         start_price=106.0),
        )
        results = DetectorEngine().analyze_many({"CALM": quiet, "BUSY": busy})
        assert results[0].symbol == "BUSY"
        assert results[0].overall > results[1].overall

    def test_signals_carry_the_detector_name_and_session(self, intraday_sessions):
        quiet = session_bars(sessions=11, end_day="2026-09-03", volume=(2_000, 2_100))
        busy = stack(
            quiet,
            session_bars(sessions=1, end_day="2026-09-04", volume=(9_000, 9_100),
                         start_price=106.0),
        )
        now = busy.index[-1].to_pydatetime()
        analysis = DetectorEngine().analyze("BUSY", busy, now=now)
        assert analysis.signals
        for produced in analysis.signals:
            assert produced.detector
            assert produced.session is MarketSession.REGULAR
            assert 0 <= produced.strength <= 10

    def test_metrics_are_plain_floats_so_they_serialise(self, intraday_sessions):
        quiet = session_bars(sessions=11, end_day="2026-09-03", volume=(2_000, 2_100))
        busy = stack(
            quiet,
            session_bars(sessions=1, end_day="2026-09-04", volume=(9_000, 9_100),
                         start_price=106.0),
        )
        analysis = DetectorEngine().analyze("BUSY", busy)
        for produced in analysis.signals:
            assert all(type(value) is float for value in produced.metrics.values())


def test_detector_signal_key_ignores_time_and_strength():
    """Two bars reporting the same event must de-duplicate to one key."""
    first = signal(SignalType.BREAKOUT, Bias.BULLISH, 6.0)
    later = DetectorSignal(
        symbol="NVDA",
        timestamp=first.timestamp + timedelta(minutes=5),
        signal_type=SignalType.BREAKOUT,
        direction=Bias.BULLISH,
        strength=9.0,
        reason="stronger now",
    )
    assert first.key == later.key
