"""Market scanner: scoring, filtering, ranking, and risk integration."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from trading_bot.indicators import (
    IndicatorConfig,
    Level,
    SupportResistance,
    TrendDirection,
    analyze_trend,
    calculate_all_indicators,
)
from trading_bot.risk import PortfolioState, RiskManager
from trading_bot.scanner import (
    DEFAULT_WEIGHTS,
    MarketScanner,
    ScannerConfig,
    score_conviction,
    score_opportunity,
    score_risk_reward,
    score_rsi_headroom,
    score_structure,
    score_trend,
    score_volume,
)
from trading_bot.strategies import (
    BaseStrategy,
    Condition,
    Signal,
    SignalDirection,
    build_strategy,
)

NOW = datetime(2026, 9, 5, 15, 0, tzinfo=timezone.utc)


def frame(closes, volumes=None) -> pd.DataFrame:
    closes = np.asarray(closes, dtype="float64")
    count = len(closes)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    index = pd.date_range("2024-01-02", periods=count, freq="15min", tz="UTC",
                          name="timestamp")
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) * 1.004,
            "low": np.minimum(opens, closes) * 0.996,
            "close": closes,
            "volume": np.asarray(
                volumes if volumes is not None else np.full(count, 500_000.0),
                dtype="float64",
            ),
        },
        index=index,
    )


def trending(drift=0.0018, seed=11, periods=400):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(drift, 0.006, periods)))


def a_signal(**overrides) -> Signal:
    payload = {
        "symbol": "AAPL",
        "direction": SignalDirection.LONG,
        "strategy": "momentum",
        "confidence": 80.0,
        "entry_price": 100.0,
        "stop_loss": 98.0,
        "take_profit": 106.0,
        "timestamp": NOW,
    }
    payload.update(overrides)
    return Signal(**payload)


class AlwaysSignals(BaseStrategy):
    """A strategy that fires on every bar, for exercising the scanner."""

    name = "always"

    def evaluate(self, symbol, data):
        return self._build_signal(
            symbol=symbol,
            direction=SignalDirection.LONG,
            data=data,
            conditions=[Condition("always", True, weight=1.0, detail="Always fires")],
        )


@pytest.fixture(scope="module")
def enriched() -> pd.DataFrame:
    return calculate_all_indicators(frame(trending()))


# ============================================================================
# Individual factors
# ============================================================================


def test_trend_factor_is_direction_aware(enriched):
    trend = analyze_trend(enriched)
    long_score = score_trend(trend, SignalDirection.LONG).score
    short_score = score_trend(trend, SignalDirection.SHORT).score
    assert long_score + short_score == pytest.approx(100.0, abs=0.01)


def test_trend_factor_is_damped_by_disagreement():
    """Components that disagreed must not produce a confident score."""
    from trading_bot.indicators import TrendAnalysis

    confident = TrendAnalysis(TrendDirection.STRONG_BULLISH, strength=90, confidence=100)
    conflicted = TrendAnalysis(TrendDirection.STRONG_BULLISH, strength=90, confidence=20)
    assert (
        score_trend(confident, SignalDirection.LONG).score
        > score_trend(conflicted, SignalDirection.LONG).score
    )


def test_risk_reward_factor_rises_with_the_ratio():
    poor = score_risk_reward(a_signal(take_profit=101.0)).score      # 0.5:1
    fair = score_risk_reward(a_signal(take_profit=104.0)).score      # 2:1
    good = score_risk_reward(a_signal(take_profit=108.0)).score      # 4:1
    assert poor < fair < good
    assert good == pytest.approx(100.0)


def test_risk_reward_factor_saturates():
    """Beyond 4:1 the difference stops being meaningful for ranking."""
    assert score_risk_reward(a_signal(take_profit=200.0)).score == 100.0


def test_rsi_headroom_penalises_buying_an_extended_move(enriched):
    """Being oversold is not penalised for a long — the strategy chose the entry."""
    config = IndicatorConfig()
    hot = enriched.copy()
    hot.iloc[-1, hot.columns.get_loc("RSI_14")] = 88.0
    cool = enriched.copy()
    cool.iloc[-1, cool.columns.get_loc("RSI_14")] = 25.0

    extended = score_rsi_headroom(hot, SignalDirection.LONG, config).score
    oversold = score_rsi_headroom(cool, SignalDirection.LONG, config).score
    assert extended < 30
    assert oversold == 100.0


def test_rsi_headroom_mirrors_for_shorts(enriched):
    config = IndicatorConfig()
    cold = enriched.copy()
    cold.iloc[-1, cold.columns.get_loc("RSI_14")] = 12.0
    assert score_rsi_headroom(cold, SignalDirection.SHORT, config).score < 30


def test_volume_factor_rewards_participation():
    from trading_bot.indicators import VolumeAnalysis, VolumeCondition

    def analysis(relative, confirms=False):
        return VolumeAnalysis(
            condition=VolumeCondition.NORMAL,
            relative_volume=relative,
            current_volume=1.0,
            average_volume=1.0,
            trend="STEADY",
            confirms_price=confirms,
        )

    assert score_volume(analysis(0.4)).score < score_volume(analysis(1.2)).score
    assert score_volume(analysis(1.2)).score < score_volume(analysis(3.0)).score


def test_volume_factor_is_absent_without_data():
    from trading_bot.indicators import VolumeAnalysis, VolumeCondition

    missing = VolumeAnalysis(
        condition=VolumeCondition.NORMAL, relative_volume=None,
        current_volume=0.0, average_volume=None, trend="UNKNOWN", confirms_price=False,
    )
    assert score_volume(missing) is None


def test_structure_factor_penalises_a_wall_overhead():
    near = SupportResistance(
        price=100.0, support=(),
        resistance=(Level(100.2, 3, 5, "resistance"),), swing_points=(),
    )
    far = SupportResistance(
        price=100.0, support=(),
        resistance=(Level(110.0, 3, 5, "resistance"),), swing_points=(),
    )
    signal = a_signal()
    assert score_structure(near, signal, atr=1.0).score < 25
    assert score_structure(far, signal, atr=1.0).score == 100.0


def test_structure_factor_needs_an_atr():
    levels = SupportResistance(price=100.0, support=(), resistance=(), swing_points=())
    assert score_structure(levels, a_signal(), atr=None) is None


def test_conviction_factor_carries_the_strategy_confidence():
    assert score_conviction(a_signal(confidence=73)).score == 73.0


# ============================================================================
# Composite score
# ============================================================================


def test_composite_stays_within_bounds(enriched):
    confidence, factors = score_opportunity(a_signal(), enriched)
    assert 0 <= confidence <= 100
    assert factors


def test_composite_renormalises_over_available_factors():
    """A missing factor reduces the evidence rather than scoring zero.

    30 bars is short enough that MACD (which needs slow + signal periods) has no
    value, so the momentum factor drops out.
    """
    short = calculate_all_indicators(frame(trending(periods=30)))
    confidence, factors = score_opportunity(a_signal(), short)

    assert 0 <= confidence <= 100
    assert len(factors) < len(DEFAULT_WEIGHTS)
    assert "momentum" not in {factor.name for factor in factors}

    # The score is the weighted mean of the factors that *did* have data.
    total_weight = sum(factor.weight for factor in factors)
    expected = sum(factor.contribution for factor in factors) / total_weight
    assert confidence == pytest.approx(expected)


def test_composite_uses_every_factor_when_history_allows():
    full = calculate_all_indicators(frame(trending(periods=400)))
    _, factors = score_opportunity(a_signal(), full)
    assert {factor.name for factor in factors} == set(DEFAULT_WEIGHTS)


def test_a_better_setup_scores_higher(enriched):
    weak = a_signal(confidence=40, take_profit=101.0)
    strong = a_signal(confidence=95, take_profit=108.0)
    assert score_opportunity(strong, enriched)[0] > score_opportunity(weak, enriched)[0]


def test_scores_are_comparable_across_strategies(enriched):
    """The reason the scanner re-scores at all."""
    momentum = a_signal(strategy="momentum", confidence=80)
    reversion = a_signal(strategy="mean_reversion", confidence=80)
    momentum_score, _ = score_opportunity(momentum, enriched)
    reversion_score, _ = score_opportunity(reversion, enriched)
    # Identical setups differing only in the label must score identically.
    assert momentum_score == pytest.approx(reversion_score)


# ============================================================================
# Scanning
# ============================================================================


def test_scanner_needs_at_least_one_strategy():
    with pytest.raises(ValueError, match="at least one strategy"):
        MarketScanner([])


def test_scan_ranks_by_confidence():
    frames = {
        "AAA": frame(trending(0.003, seed=1)),
        "BBB": frame(trending(-0.003, seed=2)),
        "CCC": frame(trending(0.0005, seed=3)),
    }
    scanner = MarketScanner([AlwaysSignals()], config=ScannerConfig(min_avg_dollar_volume=0))
    result = scanner.scan(frames)
    scores = [item.confidence for item in result.opportunities]
    assert scores == sorted(scores, reverse=True)
    assert [item.rank for item in result.opportunities] == [1, 2, 3]


def test_scan_filters_illiquid_symbols():
    frames = {
        "LIQUID": frame(trending(), volumes=np.full(400, 1_000_000.0)),
        "THIN": frame(trending(seed=5), volumes=np.full(400, 100.0)),
    }
    scanner = MarketScanner(
        [AlwaysSignals()], config=ScannerConfig(min_avg_dollar_volume=1_000_000)
    )
    result = scanner.scan(frames)
    assert "THIN" in result.skipped
    assert "turnover" in result.skipped["THIN"]
    assert result.scanned == 1


def test_scan_filters_penny_stocks():
    frames = {"CHEAP": frame(np.full(400, 0.4)), "NORMAL": frame(trending())}
    scanner = MarketScanner(
        [AlwaysSignals()],
        config=ScannerConfig(min_price=1.0, min_avg_dollar_volume=0),
    )
    result = scanner.scan(frames)
    assert "CHEAP" in result.skipped


def test_scan_records_why_nothing_fired():
    frames = {"AAA": frame(trending(0.0))}
    scanner = MarketScanner(
        [build_strategy("breakout")], config=ScannerConfig(min_avg_dollar_volume=0)
    )
    result = scanner.scan(frames)
    assert not result.opportunities
    assert result.blockers


def test_scan_survives_a_broken_symbol():
    """One bad frame must not abort the scan."""
    frames = {"GOOD": frame(trending()), "BAD": pd.DataFrame({"close": [1.0]})}
    scanner = MarketScanner([AlwaysSignals()], config=ScannerConfig(min_avg_dollar_volume=0))
    result = scanner.scan(frames)
    assert "BAD" in result.failures
    assert result.opportunities


def test_scan_respects_a_minimum_score():
    frames = {"AAA": frame(trending(0.003, seed=1)), "BBB": frame(trending(-0.004, seed=2))}
    permissive = MarketScanner(
        [AlwaysSignals()], config=ScannerConfig(min_avg_dollar_volume=0)
    ).scan(frames)
    strict = MarketScanner(
        [AlwaysSignals()],
        config=ScannerConfig(min_avg_dollar_volume=0, min_confidence=95),
    ).scan(frames)
    assert len(strict.opportunities) < len(permissive.opportunities)


def test_scan_caps_the_result_count():
    frames = {name: frame(trending(seed=index)) for index, name in enumerate("ABCDE")}
    scanner = MarketScanner(
        [AlwaysSignals()],
        config=ScannerConfig(min_avg_dollar_volume=0, max_results=2),
    )
    assert len(scanner.scan(frames).opportunities) == 2


def test_scan_reports_its_duration_and_count():
    frames = {"AAA": frame(trending())}
    result = MarketScanner(
        [AlwaysSignals()], config=ScannerConfig(min_avg_dollar_volume=0)
    ).scan(frames)
    assert result.duration_seconds >= 0
    assert "from 1 symbol" in result.summary()


# ============================================================================
# Risk integration
# ============================================================================


def test_scan_sizes_opportunities_when_risk_is_supplied():
    frames = {"AAA": frame(trending())}
    scanner = MarketScanner(
        [AlwaysSignals()],
        risk_manager=RiskManager(),
        config=ScannerConfig(min_avg_dollar_volume=0),
    )
    result = scanner.scan(
        frames, portfolio=PortfolioState(equity=Decimal("100000"),
                                         buying_power=Decimal("200000")), now=NOW
    )
    assert result.opportunities
    assert result.opportunities[0].decision is not None


def test_opportunities_are_unsized_without_a_portfolio():
    frames = {"AAA": frame(trending())}
    result = MarketScanner(
        [AlwaysSignals()], risk_manager=RiskManager(),
        config=ScannerConfig(min_avg_dollar_volume=0),
    ).scan(frames)
    assert result.opportunities[0].decision is None
    assert not result.opportunities[0].tradable


def test_rejected_opportunities_still_appear_with_their_reason():
    """A scanner that hid what risk blocked could not distinguish a quiet market
    from a mis-set limit."""
    frames = {name: frame(trending(seed=index)) for index, name in enumerate("ABC")}
    from trading_bot.config.settings import RiskSettings

    scanner = MarketScanner(
        [AlwaysSignals()],
        risk_manager=RiskManager(RiskSettings(max_open_positions=1)),
        config=ScannerConfig(min_avg_dollar_volume=0),
    )
    result = scanner.scan(
        frames,
        portfolio=PortfolioState(equity=Decimal("100000"), buying_power=Decimal("200000")),
        now=NOW,
    )
    assert len(result.opportunities) == 3
    assert len(result.tradable) <= 1
    assert any(item.rejection_reason for item in result.opportunities)


def test_scarce_slots_go_to_the_highest_scoring_opportunity():
    frames = {"AAA": frame(trending(0.004, seed=1)), "BBB": frame(trending(-0.004, seed=2))}
    from trading_bot.config.settings import RiskSettings

    scanner = MarketScanner(
        [AlwaysSignals()],
        risk_manager=RiskManager(RiskSettings(max_open_positions=1)),
        config=ScannerConfig(min_avg_dollar_volume=0),
    )
    result = scanner.scan(
        frames,
        portfolio=PortfolioState(equity=Decimal("100000"), buying_power=Decimal("200000")),
        now=NOW,
    )
    tradable = result.tradable
    if tradable:
        assert tradable[0].rank == 1


def test_scan_reports_a_trading_halt():
    frames = {"AAA": frame(trending())}
    scanner = MarketScanner(
        [AlwaysSignals()], risk_manager=RiskManager(),
        config=ScannerConfig(min_avg_dollar_volume=0),
    )
    halted = PortfolioState(
        equity=Decimal("10000"), halted=True, halt_reason="Kill switch engaged"
    )
    assert scanner.scan(frames, portfolio=halted, now=NOW).halt_reason


# ============================================================================
# Reporting
# ============================================================================


def test_opportunity_exposes_its_weakest_factor():
    frames = {"AAA": frame(trending())}
    result = MarketScanner(
        [AlwaysSignals()], config=ScannerConfig(min_avg_dollar_volume=0)
    ).scan(frames)
    weakest = result.opportunities[0].weakest_factor()
    assert weakest is not None
    assert all(weakest.score <= factor.score for factor in result.opportunities[0].factors)


def test_opportunity_exposes_its_strongest_factors():
    frames = {"AAA": frame(trending())}
    result = MarketScanner(
        [AlwaysSignals()], config=ScannerConfig(min_avg_dollar_volume=0)
    ).scan(frames)
    top = result.opportunities[0].top_factors(3)
    assert len(top) == 3
    assert top[0].contribution >= top[-1].contribution


def test_result_serializes():
    frames = {"AAA": frame(trending())}
    payload = MarketScanner(
        [AlwaysSignals()], config=ScannerConfig(min_avg_dollar_volume=0)
    ).scan(frames).as_dict()
    assert set(payload) >= {"summary", "scanned", "opportunities", "blockers"}
    assert payload["opportunities"][0]["rank"] == 1


def test_config_validation():
    with pytest.raises(ValueError):
        ScannerConfig(min_confidence=150)
    with pytest.raises(ValueError):
        ScannerConfig(max_results=0)
    with pytest.raises(ValueError):
        ScannerConfig(min_avg_dollar_volume=-1)
