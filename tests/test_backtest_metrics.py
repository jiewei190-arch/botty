"""Performance-metric tests.

The numbers here are worked out by hand in the test rather than read back from
the implementation, so a change in the maths fails rather than quietly redefining
what "Sharpe" means.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from trading_bot.backtesting import (
    MIN_MEANINGFUL_TRADES,
    calculate_metrics,
    drawdown_curve,
)
from trading_bot.utils.timeframes import Timeframe


def curve(values, freq="15min"):
    index = pd.date_range("2025-01-02 14:30", periods=len(values), freq=freq, tz="UTC")
    return pd.Series([float(v) for v in values], index=index)


def trade(pnl, *, r=None, reason="STOP_LOSS", commission=0.0, slippage=0.0, gapped=False):
    return {
        "pnl": pnl,
        "r_multiple": r,
        "exit_reason": reason,
        "commission": commission,
        "slippage": slippage,
        "gapped": gapped,
    }


def metrics_for(equity, trades, *, timeframe="15Min", start=10_000.0, in_market=0):
    return calculate_metrics(
        curve(equity), trades, timeframe=timeframe,
        starting_equity=start, bars_in_market=in_market,
    )


class TestReturns:
    def test_total_return_is_measured_against_starting_equity(self):
        result = metrics_for([10_000, 10_500, 11_000], [])
        assert result.total_return_pct == pytest.approx(10.0)
        assert result.net_profit == pytest.approx(1_000.0)

    def test_a_loss_reports_negative(self):
        assert metrics_for([10_000, 9_000], []).total_return_pct == pytest.approx(-10.0)

    def test_annualised_return_is_capped(self):
        """A few bars of gains extrapolate to nonsense; the cap says so."""
        result = metrics_for([10_000, 30_000], [])
        assert result.annualised_return_pct == 10_000.0

    def test_annualised_return_floors_at_total_loss(self):
        result = metrics_for([10_000, 1.0], [])
        assert result.annualised_return_pct == -100.0


class TestTradeStatistics:
    def test_win_rate_and_averages(self):
        trades = [trade(100), trade(50), trade(-30), trade(-20)]
        result = metrics_for([10_000, 10_100], trades)
        assert result.total_trades == 4
        assert (result.wins, result.losses) == (2, 2)
        assert result.win_rate == pytest.approx(50.0)
        assert result.average_win == pytest.approx(75.0)
        assert result.average_loss == pytest.approx(-25.0)
        assert result.largest_win == 100
        assert result.largest_loss == -30

    def test_profit_factor_is_gross_profit_over_gross_loss(self):
        result = metrics_for([10_000, 10_100], [trade(150), trade(-50)])
        assert result.profit_factor == pytest.approx(3.0)

    def test_profit_factor_is_infinite_with_no_losses(self):
        """Reported honestly rather than clipped to a plausible-looking number."""
        result = metrics_for([10_000, 10_100], [trade(150)])
        assert result.profit_factor == float("inf")

    def test_profit_factor_is_zero_when_nothing_was_made(self):
        assert metrics_for([10_000, 9_900], [trade(0.0)]).profit_factor == 0.0

    def test_a_breakeven_trade_counts_as_a_loss(self):
        """Zero P&L after costs is not a win; counting it as one flatters the rate."""
        result = metrics_for([10_000, 10_000], [trade(0.0)])
        assert (result.wins, result.losses) == (0, 1)

    def test_expectancy_averages_r_multiples(self):
        trades = [trade(200, r=2.0), trade(-100, r=-1.0), trade(-100, r=-1.0)]
        assert metrics_for([10_000, 10_000], trades).expectancy_r == pytest.approx(0.0)

    def test_expectancy_ignores_trades_without_an_r_multiple(self):
        trades = [trade(200, r=2.0), trade(-100, r=None)]
        assert metrics_for([10_000, 10_000], trades).expectancy_r == pytest.approx(2.0)

    def test_costs_and_gaps_are_totalled(self):
        trades = [
            trade(10, commission=1.0, slippage=2.0, gapped=True),
            trade(-5, commission=1.5, slippage=0.5),
        ]
        result = metrics_for([10_000, 10_005], trades)
        assert result.total_commission == pytest.approx(2.5)
        assert result.total_slippage == pytest.approx(2.5)
        assert result.gapped_stops == 1

    def test_exit_breakdown_counts_reasons(self):
        trades = [
            trade(10, reason="TAKE_PROFIT"),
            trade(-5, reason="STOP_LOSS"),
            trade(-5, reason="STOP_LOSS"),
        ]
        result = metrics_for([10_000, 10_000], trades)
        assert result.exit_breakdown == {"TAKE_PROFIT": 1, "STOP_LOSS": 2}


class TestDrawdown:
    def test_drawdown_is_measured_from_the_running_peak(self):
        result = metrics_for([10_000, 12_000, 9_000, 11_000], [])
        assert result.max_drawdown_pct == pytest.approx(25.0)  # 12000 -> 9000
        assert result.max_drawdown_value == pytest.approx(3_000.0)

    def test_underwater_duration_counts_bars_below_the_peak(self):
        # peak at index 1; bars 2, 3, 4 are below it; bar 5 makes a new high.
        result = metrics_for([10_000, 12_000, 11_000, 10_500, 11_500, 13_000], [])
        assert result.max_drawdown_bars == 3

    def test_a_curve_that_only_rises_has_no_drawdown(self):
        result = metrics_for([10_000, 10_100, 10_200], [])
        assert result.max_drawdown_pct == 0.0
        assert result.max_drawdown_bars == 0

    def test_drawdown_curve_is_zero_at_every_new_high(self):
        series = drawdown_curve(curve([100, 110, 105, 120]))
        assert list(series.round(6)) == pytest.approx(
            [0.0, 0.0, -100 * 5 / 110, 0.0], abs=1e-6
        )


class TestRiskAdjustedRatios:
    def test_sharpe_is_annualised_by_the_bar_size(self):
        """The scaling factor is sqrt(bars per year) — not sqrt(252) regardless."""
        equity = [10_000 * (1.001**i) * (1.004 if i % 2 else 1.0) for i in range(60)]
        result = metrics_for(equity, [], timeframe="15Min")
        returns = curve(equity).pct_change().dropna()
        expected = (
            returns.mean() / returns.std(ddof=1)
            * math.sqrt(Timeframe.parse("15Min").periods_per_year)
        )
        assert result.sharpe_ratio == pytest.approx(expected)

    def test_the_same_returns_on_daily_bars_give_a_smaller_sharpe(self):
        """Guards the mistake that inflates an intraday Sharpe ~5x."""
        equity = [10_000 * (1.001**i) * (1.004 if i % 2 else 1.0) for i in range(60)]
        intraday = metrics_for(equity, [], timeframe="15Min")
        daily = metrics_for(equity, [], timeframe="1Day")
        assert intraday.sharpe_ratio > daily.sharpe_ratio
        ratio = math.sqrt(
            Timeframe.parse("15Min").periods_per_year
            / Timeframe.parse("1Day").periods_per_year
        )
        assert intraday.sharpe_ratio / daily.sharpe_ratio == pytest.approx(ratio)

    def test_sortino_ignores_upside_volatility(self):
        """A curve whose gains are erratic but losses are mild scores better."""
        rng = np.random.default_rng(3)
        returns = rng.normal(0.0005, 0.002, 200)
        returns[returns > 0] *= 4  # violent upside only
        equity = 10_000 * np.cumprod(1 + returns)
        result = metrics_for(equity, [])
        assert result.sortino_ratio > result.sharpe_ratio

    def test_a_flat_curve_reports_zero_not_infinity(self):
        result = metrics_for([10_000] * 10, [])
        assert result.sharpe_ratio == 0.0
        assert result.sortino_ratio == 0.0

    def test_too_few_points_reports_zero(self):
        assert metrics_for([10_000], []).sharpe_ratio == 0.0


class TestSampleHonesty:
    def test_no_trades_says_the_ratios_are_undefined(self):
        result = metrics_for([10_000, 10_000], [])
        assert result.sample_warning is not None
        assert "undefined" in result.sample_warning
        assert not result.is_meaningful

    def test_a_small_sample_is_called_an_anecdote(self):
        result = metrics_for([10_000, 10_100], [trade(10)] * 5)
        assert result.sample_warning is not None
        assert str(MIN_MEANINGFUL_TRADES) in result.sample_warning
        assert not result.is_meaningful

    def test_a_large_enough_sample_carries_no_warning(self):
        result = metrics_for([10_000, 10_100], [trade(10)] * MIN_MEANINGFUL_TRADES)
        assert result.sample_warning is None
        assert result.is_meaningful

    def test_summary_surfaces_the_warning(self):
        text = "\n".join(metrics_for([10_000, 10_100], [trade(10)]).summary_lines())
        assert "!!" in text


class TestExposure:
    def test_exposure_is_bars_held_over_bars_simulated(self):
        result = metrics_for([10_000] * 10, [], in_market=3)
        assert result.exposure_pct == pytest.approx(30.0)

    def test_empty_curve_falls_back_to_starting_equity(self):
        result = calculate_metrics(
            pd.Series(dtype="float64"), [], timeframe="15Min", starting_equity=10_000.0
        )
        assert result.ending_equity == 10_000.0
        assert result.bars == 0
        assert result.exposure_pct == 0.0
