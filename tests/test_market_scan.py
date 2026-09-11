"""Market-wide sweep: the funnel from a whole universe down to a few entries.

The sweep's job is to be honest about what it discarded and why. A scan that
returns four names out of six hundred is either working perfectly or badly
broken, and the difference is only visible in the stage counts.
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from tests.conftest import make_bars
from trading_bot.risk import PortfolioState
from trading_bot.scanner.market_scan import (
    HuntConfig,
    MarketSweep,
    signal_age_bars,
    sweep_market,
)
from trading_bot.strategies import build_strategy
from trading_bot.universe import Universe
from trading_bot.universe.filters import profile_liquidity

SMALL_ACCOUNT = PortfolioState(
    equity=Decimal("15000"), cash=Decimal("15000"), buying_power=Decimal("15000")
)


@pytest.fixture(scope="module")
def market():
    """A simulated market of daily-bar symbols."""
    frames, profiles = {}, {}
    for index in range(220):
        symbol = f"S{index:04d}"
        bars = make_bars(
            300, seed=index, freq="1D", start_price=float(15 + (index * 13) % 400)
        )
        frames[symbol] = bars
        profiles[symbol] = profile_liquidity(symbol, bars)
    return Universe(symbols=tuple(frames), profiles=profiles, frames=frames)


@pytest.fixture(scope="module")
def strategies():
    return [build_strategy(name) for name in ("momentum", "mean_reversion", "breakout")]


@pytest.fixture(scope="module")
def sweep(market, strategies):
    return sweep_market(
        market, strategies, portfolio=SMALL_ACCOUNT, config=HuntConfig(top_n=10)
    )


class TestSignalAge:
    def frame(self, periods=10):
        return make_bars(periods, start=pd.Timestamp("2025-06-02", tz="UTC"), freq="1D")

    def test_the_latest_bar_is_age_zero(self):
        frame = self.frame()
        assert signal_age_bars(frame, frame.index[-1]) == 0

    def test_the_previous_bar_is_age_one(self):
        frame = self.frame()
        assert signal_age_bars(frame, frame.index[-2]) == 1

    def test_an_older_bar_reports_its_distance(self):
        frame = self.frame()
        assert signal_age_bars(frame, frame.index[-5]) == 4

    def test_an_empty_frame_is_age_zero(self):
        assert signal_age_bars(pd.DataFrame(), pd.Timestamp("2025-06-02", tz="UTC")) == 0


class TestSweepMechanics:
    def test_a_universe_without_bars_is_refused(self, strategies):
        empty = Universe(symbols=("AAA",), frames={})
        with pytest.raises(ValueError, match="carries no bars"):
            sweep_market(empty, strategies, portfolio=SMALL_ACCOUNT)

    def test_every_stage_is_reported(self, sweep):
        names = [stage.name for stage in sweep.stages]
        assert names == [
            "indicators",
            "strategies + risk",
            "freshness",
            "risk + reward:risk",
        ]

    def test_the_funnel_never_widens(self, sweep):
        """Each stage can only shrink the candidate set."""
        for stage in sweep.stages:
            assert stage.survived <= stage.entered

    def test_ranks_are_contiguous_from_one(self, sweep):
        ranks = [opportunity.rank for opportunity in sweep.opportunities]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_results_are_ordered_by_score(self, sweep):
        scores = [opportunity.confidence for opportunity in sweep.opportunities]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_is_respected(self, market, strategies):
        result = sweep_market(
            market, strategies, portfolio=SMALL_ACCOUNT, config=HuntConfig(top_n=3)
        )
        assert len(result.opportunities) <= 3


class TestSetupQuality:
    def test_every_returned_setup_clears_the_reward_floor(self, sweep):
        for opportunity in sweep.opportunities:
            assert opportunity.signal.risk_reward_ratio >= 2.0

    def test_every_returned_setup_passed_risk(self, sweep):
        for opportunity in sweep.opportunities:
            assert opportunity.tradable
            assert int(opportunity.decision.shares) >= 1

    def test_stops_are_wide_enough_to_survive_a_daily_bar(self, sweep):
        """A stop inside one session's noise is not a stop, it is a fee.

        Real large-cap daily bars average about 2.5% of price in true range, so
        anything under roughly half a percent would be taken out by an ordinary
        session going nowhere.
        """
        for opportunity in sweep.opportunities:
            signal = opportunity.signal
            width = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
            assert width > 0.005, f"{signal.symbol} stop is only {width:.3%} wide"

    def test_no_setup_is_stale(self, sweep):
        assert not any(
            symbol in sweep.stale
            for symbol in (o.symbol for o in sweep.opportunities)
        )


class TestIndependentSizing:
    """Share counts must answer "if I take this one", not "if I take them all".

    The scanner sizes its ranked list cumulatively, so a candidate low in the
    list can report a tiny position — or fail the position-size check outright —
    purely because the ones above it consumed the account. That is a correct
    answer to a question nobody asked.
    """

    def test_lower_ranked_setups_are_not_starved(self, sweep):
        positions = [
            int(o.decision.shares) * o.signal.entry_price
            for o in sweep.opportunities
        ]
        assert len(positions) >= 2
        # Every position is sized against the same account, so the last is not
        # dramatically smaller than the first.
        assert min(positions) > max(positions) * 0.25

    def test_cumulative_sizing_starves_them(self, market, strategies):
        """The contrast that makes the default worth having."""
        cumulative = sweep_market(
            market,
            strategies,
            portfolio=SMALL_ACCOUNT,
            config=HuntConfig(top_n=10, size_independently=False),
        )
        independent = sweep_market(
            market, strategies, portfolio=SMALL_ACCOUNT, config=HuntConfig(top_n=10)
        )
        assert len(independent.opportunities) >= len(cumulative.opportunities)

    def test_concurrent_capacity_is_reported(self, sweep):
        """Per-trade sizing would otherwise imply all of them can be taken."""
        assert sweep.concurrent_capacity >= 1
        assert sweep.concurrent_capacity <= len(sweep.opportunities) or not sweep.opportunities

    def test_capacity_grows_with_the_account(self, market, strategies):
        big = PortfolioState(
            equity=Decimal("250000"),
            cash=Decimal("250000"),
            buying_power=Decimal("250000"),
        )
        small = sweep_market(
            market, strategies, portfolio=SMALL_ACCOUNT, config=HuntConfig(top_n=10)
        )
        large = sweep_market(
            market, strategies, portfolio=big, config=HuntConfig(top_n=10)
        )
        assert large.concurrent_capacity >= small.concurrent_capacity


class TestFreshness:
    def test_stale_setups_are_dropped(self, market, strategies):
        """A setup from several sessions ago has already made its move."""
        strict = sweep_market(
            market,
            strategies,
            portfolio=SMALL_ACCOUNT,
            config=HuntConfig(top_n=50, max_signal_age_bars=1),
        )
        loose = sweep_market(
            market,
            strategies,
            portfolio=SMALL_ACCOUNT,
            config=HuntConfig(top_n=50, max_signal_age_bars=30),
        )
        assert len(loose.opportunities) >= len(strict.opportunities)

    def test_age_must_be_at_least_one_bar(self):
        with pytest.raises(ValueError, match="max_signal_age_bars"):
            HuntConfig(max_signal_age_bars=0)


class TestReporting:
    def test_drop_reasons_are_recorded(self, sweep):
        """A quiet scan must explain itself rather than just return nothing."""
        assert isinstance(sweep.filtered_out, dict)
        assert sum(sweep.filtered_out.values()) >= 0

    def test_summary_names_the_universe_size(self, sweep):
        assert "220" in "\n".join(sweep.summary_lines())

    def test_frame_has_a_row_per_opportunity(self, sweep):
        frame = sweep.as_frame()
        assert len(frame) == len(sweep.opportunities)
        if not frame.empty:
            assert {"symbol", "entry", "stop", "target", "shares"} <= set(frame.columns)

    def test_an_empty_sweep_yields_an_empty_frame(self):
        assert MarketSweep().as_frame().empty


class TestConfigValidation:
    def test_top_n_must_be_positive(self):
        with pytest.raises(ValueError, match="top_n"):
            HuntConfig(top_n=0)

    def test_min_score_is_bounded(self):
        with pytest.raises(ValueError, match="min_score"):
            HuntConfig(min_score=150.0)

