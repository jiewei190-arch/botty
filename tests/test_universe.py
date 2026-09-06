"""Universe discovery: turning ~11,000 listed symbols into a scannable set.

The filters here decide what a market-wide scan can even see, so a mistake in
them is invisible in the results — you get a plausible ranked list drawn from
the wrong half of the market. Each filter is therefore tested for what it keeps
as well as what it drops.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.conftest import make_bars
from trading_bot.universe import (
    ROBINHOOD_EXCHANGES,
    FilterReport,
    UniverseFilter,
    passes_liquidity_filters,
    passes_static_filters,
    profile_liquidity,
    screen_liquidity,
    screen_static,
)


def asset(symbol="AAPL", name="Apple Inc", exchange="NASDAQ", **overrides):
    record = {
        "symbol": symbol,
        "name": name,
        "exchange": exchange,
        "asset_class": "us_equity",
        "status": "active",
        "tradable": True,
        "shortable": True,
        "fractionable": True,
        "marginable": True,
    }
    record.update(overrides)
    return record


class _Rec:
    def __init__(self, data):
        self._data = data

    def __getattr__(self, name):
        return self._data.get(name)


def check(record, filters=None):
    return passes_static_filters(_Rec(record), filters or UniverseFilter())


class TestStaticFilters:
    def test_an_ordinary_listed_stock_passes(self):
        assert check(asset())

    @pytest.mark.parametrize("exchange", sorted(ROBINHOOD_EXCHANGES))
    def test_every_supported_exchange_passes(self, exchange):
        assert check(asset(exchange=exchange))

    def test_otc_is_excluded(self):
        """Robinhood does not trade OTC, so scanning it produces dead ends."""
        assert not check(asset(exchange="OTC"))

    def test_untradable_assets_are_excluded(self):
        assert not check(asset(tradable=False))

    def test_inactive_assets_are_excluded(self):
        assert not check(asset(status="inactive"))

    def test_enum_style_status_is_understood(self):
        """Alpaca's enums stringify as 'AssetStatus.ACTIVE'."""
        assert check(asset(status="AssetStatus.ACTIVE"))

    def test_non_equity_asset_classes_are_excluded(self):
        assert not check(asset(asset_class="crypto"))

    @pytest.mark.parametrize(
        "name",
        [
            "Direxion Daily Semiconductor Bull 3X Shares",
            "ProShares UltraShort S&P500",
            "ProShares Short QQQ -1x Shares",
            "Some Inverse Volatility Fund",
        ],
    )
    def test_leveraged_and_inverse_funds_are_excluded(self, name):
        """They decay against a multi-day holder; a swing scan should not rank them."""
        assert not check(asset(name=name))

    @pytest.mark.parametrize(
        "name",
        ["Churchill Acquisition Corp", "Foo Closed End Fund", "Bar Royalty Trust"],
    )
    def test_structures_are_excluded(self, name):
        assert not check(asset(name=name))

    @pytest.mark.parametrize("symbol", ["ABC.WS", "ABC-W", "XYZ.U", "XYZ.RT", "FOO.PA"])
    def test_warrants_rights_and_units_are_excluded(self, symbol):
        assert not check(asset(symbol=symbol))

    def test_ordinary_symbols_with_dots_are_kept(self):
        """BRK.B is a share class, not a warrant."""
        assert check(asset(symbol="BRK.B"))

    def test_leveraged_filter_can_be_disabled(self):
        filters = UniverseFilter(exclude_leveraged=False)
        assert check(asset(name="Direxion Daily 3X Bull"), filters)

    def test_always_include_overrides_every_other_check(self):
        filters = UniverseFilter(always_include=frozenset({"PINK"}))
        assert check(asset(symbol="PINK", exchange="OTC", tradable=False), filters)

    def test_never_include_wins_over_always_include(self):
        filters = UniverseFilter(
            always_include=frozenset({"AAPL"}), never_include=frozenset({"AAPL"})
        )
        assert not check(asset(), filters)

    def test_fractionable_requirement_is_off_by_default(self):
        """It is an Alpaca property and says nothing about Robinhood."""
        assert check(asset(fractionable=False))
        assert not check(
            asset(fractionable=False), UniverseFilter(require_fractionable=True)
        )

    def test_the_report_counts_each_reason(self):
        report = FilterReport()
        for record in (
            asset(exchange="OTC"),
            asset(exchange="OTC"),
            asset(tradable=False),
        ):
            passes_static_filters(_Rec(record), UniverseFilter(), report)
        assert report.dropped["exchange OTC"] == 2
        assert report.dropped["not tradable"] == 1


class TestScreenStatic:
    def test_it_returns_upper_cased_survivors(self):
        records = [asset(symbol="aapl"), asset(symbol="msft", exchange="OTC")]
        kept, report = screen_static(records, UniverseFilter())
        assert kept == ["AAPL"]
        assert report.considered == 2
        assert report.kept == 1


class TestLiquidityProfile:
    def test_turnover_is_price_times_volume(self):
        bars = make_bars(60, seed=3, freq="1D", start_price=50.0)
        bars["volume"] = 1_000_000.0
        profile = profile_liquidity("AAA", bars)
        # ~50 x 1,000,000, allowing for the price path drifting.
        assert profile.median_dollar_volume == pytest.approx(50_000_000, rel=0.4)

    def test_volume_alone_does_not_imply_liquidity(self):
        """A million shares of a $0.40 stock is $400k, not $1m."""
        cheap = make_bars(60, seed=4, freq="1D", start_price=0.40)
        cheap["volume"] = 1_000_000.0
        profile = profile_liquidity("PENNY", cheap)
        assert profile.median_dollar_volume < 1_000_000

    def test_an_empty_frame_yields_no_profile(self):
        assert profile_liquidity("AAA", pd.DataFrame()) is None

    def test_a_frame_missing_columns_yields_no_profile(self):
        assert profile_liquidity("AAA", pd.DataFrame({"close": [1.0]})) is None

    def test_atr_pct_is_reported(self):
        profile = profile_liquidity("AAA", make_bars(60, seed=5, freq="1D"))
        assert profile.atr_pct is not None and profile.atr_pct > 0


class TestLiquidityFilters:
    def make_profile(self, **overrides):
        bars = make_bars(260, seed=6, freq="1D", start_price=100.0)
        bars["volume"] = 500_000.0
        profile = profile_liquidity("AAA", bars)
        from dataclasses import replace

        return replace(profile, **overrides)

    def test_a_liquid_symbol_passes(self):
        assert passes_liquidity_filters(
            self.make_profile(median_dollar_volume=50_000_000.0), UniverseFilter()
        )

    def test_thin_turnover_is_rejected(self):
        assert not passes_liquidity_filters(
            self.make_profile(median_dollar_volume=80_000.0), UniverseFilter()
        )

    def test_a_cheap_stock_is_rejected(self):
        assert not passes_liquidity_filters(
            self.make_profile(last_close=2.0, median_dollar_volume=50_000_000.0),
            UniverseFilter(),
        )

    def test_an_expensive_stock_is_rejected(self):
        """One share of a $2,000 stock can exceed a small account's risk budget."""
        assert not passes_liquidity_filters(
            self.make_profile(last_close=2_500.0, median_dollar_volume=50_000_000.0),
            UniverseFilter(),
        )

    def test_a_recent_listing_is_rejected(self):
        """No 200-day average exists yet, so its indicators are arithmetic only."""
        assert not passes_liquidity_filters(
            self.make_profile(bars=30, median_dollar_volume=50_000_000.0),
            UniverseFilter(),
        )

    def test_screen_liquidity_ranks_by_turnover_and_caps(self):
        frames = {}
        for index in range(10):
            bars = make_bars(260, seed=index, freq="1D", start_price=100.0)
            bars["volume"] = float((index + 1) * 500_000)
            frames[f"S{index}"] = bars
        symbols, profiles, report = screen_liquidity(
            frames, UniverseFilter(min_dollar_volume=0.0, max_symbols=3)
        )
        assert len(symbols) == 3
        assert symbols[0] == "S9"  # highest turnover first
        assert report.kept == 3
        assert set(profiles) == set(symbols)


class TestFilterValidation:
    def test_max_price_must_exceed_min_price(self):
        with pytest.raises(ValueError, match="max_price"):
            UniverseFilter(min_price=100.0, max_price=10.0)

    def test_negative_turnover_is_rejected(self):
        with pytest.raises(ValueError, match="min_dollar_volume"):
            UniverseFilter(min_dollar_volume=-1.0)

    def test_max_symbols_must_be_positive(self):
        with pytest.raises(ValueError, match="max_symbols"):
            UniverseFilter(max_symbols=0)


class TestFeedWarning:
    """The turnover floor assumes consolidated volume; the free feed is one venue.

    Without this warning, a scan on the free feed returns almost nothing and
    reads as a broken tool rather than an over-strict filter.
    """

    def test_the_free_feed_is_flagged(self):
        from trading_bot.universe import feed_liquidity_warning

        warning = feed_liquidity_warning("iex", 10_000_000)
        assert warning is not None
        assert "iex" in warning
        # It says what to do, not just what is wrong.
        assert "min-dollar-volume" in warning

    def test_the_consolidated_feed_is_not_flagged(self):
        from trading_bot.universe import feed_liquidity_warning

        assert feed_liquidity_warning("sip", 10_000_000) is None

    def test_no_turnover_floor_means_no_warning(self):
        """With no filter there is nothing for the feed to distort."""
        from trading_bot.universe import feed_liquidity_warning

        assert feed_liquidity_warning("iex", 0) is None

    def test_the_warning_scales_with_the_threshold(self):
        from trading_bot.universe import feed_liquidity_warning

        small = feed_liquidity_warning("iex", 1_000_000)
        large = feed_liquidity_warning("iex", 50_000_000)
        assert "$1,000,000" in small
        assert "$50,000,000" in large

    def test_feed_names_are_matched_case_insensitively(self):
        from trading_bot.universe import feed_liquidity_warning

        assert feed_liquidity_warning("IEX", 10_000_000) is not None
        assert feed_liquidity_warning(" SIP ", 10_000_000) is None


class TestAuthFailureDiagnostics:
    """Alpaca answers every rejected credential with the same body.

    A newly created account whose compliance review has not cleared reads
    exactly like a mistyped secret, which sends people to check the wrong
    thing. The message must name the real causes rather than pass the vendor's
    single word through.
    """

    ERROR = '{"message": "unauthorized."}'

    def test_the_account_review_cause_is_listed_first(self):
        from trading_bot.universe import explain_auth_failure

        message = explain_auth_failure("PKABC123", self.ERROR)
        assert "under review" in message
        # It is cause 1, because it is the most common on a new account.
        assert message.index("under review") < message.index("different pairs")

    def test_it_admits_it_cannot_tell_the_causes_apart(self):
        """Claiming to know which one would be a guess dressed as a diagnosis."""
        from trading_bot.universe import explain_auth_failure

        assert "cannot tell the causes apart" in explain_auth_failure("PKABC", self.ERROR)

    def test_a_paper_key_rules_out_the_wrong_endpoint(self):
        from trading_bot.universe import explain_auth_failure

        message = explain_auth_failure("PKOVMPQ9UNOVE6ZVDAJB", self.ERROR)
        assert "Ruled out here" in message

    def test_a_live_key_is_called_out(self):
        """The one cause a rejected credential can still be checked against."""
        from trading_bot.universe import explain_auth_failure

        message = explain_auth_failure("AKXXXXXXXXXXXX", self.ERROR)
        assert "look like live-account keys" in message
        assert "Paper dashboard" in message

    def test_the_vendor_message_is_still_included(self):
        from trading_bot.universe import explain_auth_failure

        assert self.ERROR in explain_auth_failure("PKABC", self.ERROR)

    def test_a_missing_key_does_not_crash_the_explanation(self):
        from trading_bot.universe import explain_auth_failure

        assert "under review" in explain_auth_failure("", self.ERROR)
        assert "under review" in explain_auth_failure(None, self.ERROR)

    @pytest.mark.parametrize(
        "error", ["unauthorized.", "HTTP 401", "403 Forbidden", "Unauthorized"]
    )
    def test_auth_failures_are_recognised(self, error):
        from trading_bot.universe.discovery import _is_auth_failure

        assert _is_auth_failure(error)

    @pytest.mark.parametrize(
        "error", ["connection reset", "500 server error", "timed out"]
    )
    def test_other_failures_are_not_dressed_up_as_auth_problems(self, error):
        from trading_bot.universe.discovery import _is_auth_failure

        assert not _is_auth_failure(error)


class TestTurnoverIsRobustToSpikes:
    """Turnover is a median because the universe is *ranked* by it.

    A single abnormal print — an index rebalance, a buyout, a fat finger —
    inflates a 20-bar mean by three orders of magnitude. Ranked by that, the
    thinnest stock in the market lands at the top of the list, which is exactly
    the symbol you least want shown first.
    """

    def steady(self, seed=8, shares=2_000_000.0):
        bars = make_bars(250, seed=seed, freq="1D", start_price=40.0)
        bars["volume"] = shares
        return bars

    def test_one_enormous_print_does_not_move_it(self):
        bars = self.steady()
        clean = profile_liquidity("AAA", bars).median_dollar_volume

        spiked = bars.copy()
        spiked.loc[spiked.index[-1], "volume"] = 5e10
        after = profile_liquidity("AAA", spiked).median_dollar_volume

        assert after == pytest.approx(clean, rel=0.01)

    def test_a_mean_would_have_been_wrecked_by_it(self):
        """The contrast that justifies the choice."""
        bars = self.steady()
        spiked = bars.copy()
        spiked.loc[spiked.index[-1], "volume"] = 5e10

        recent = spiked.tail(20)
        mean = float((recent["close"] * recent["volume"]).mean())
        median = profile_liquidity("AAA", spiked).median_dollar_volume
        assert mean > median * 100

    def test_a_thin_stock_cannot_spike_its_way_into_the_universe(self):
        """The consequence that matters: it must still be filtered out."""
        thin = self.steady(shares=5_000.0)  # ~$200k/day, genuinely untradable
        thin.loc[thin.index[-1], "volume"] = 5e10
        profile = profile_liquidity("THIN", thin)
        assert not passes_liquidity_filters(profile, UniverseFilter())

    def test_sustained_volume_is_still_reflected(self):
        """It must not be so insensitive that real liquidity never registers."""
        quiet = profile_liquidity("AAA", self.steady(shares=1_000_000.0))
        busy = profile_liquidity("AAA", self.steady(shares=50_000_000.0))
        assert busy.median_dollar_volume > quiet.median_dollar_volume * 10

    def test_a_halt_does_not_read_as_liquid(self):
        """Zero-volume days pull the median down, as they should."""
        bars = self.steady()
        bars.loc[bars.index[-15:], "volume"] = 0.0  # halted for the recent window
        assert profile_liquidity("AAA", bars).median_dollar_volume == 0.0


class TestStaleAndHaltedSymbolsCannotRank:
    """A stock that stopped trading keeps its history, and every per-symbol
    check passes on that history. What gives it away is that the market moved
    on without it.
    """

    def liquid(self, seed=1, periods=250, start=None):
        bars = make_bars(periods, seed=seed, freq="1D", start_price=50.0, start=start)
        bars["volume"] = 4_000_000.0
        return bars

    def universe(self, extra):
        frames = {f"OK{i}": self.liquid(seed=i) for i in range(20)}
        frames.update(extra)
        return frames

    def test_a_symbol_that_stopped_trading_is_dropped(self):
        stale = self.liquid(seed=99, start=pd.Timestamp("2025-01-02", tz="UTC"))
        symbols, _, report = screen_liquidity(
            self.universe({"STALE": stale}), UniverseFilter(min_dollar_volume=0.0)
        )
        assert "STALE" not in symbols
        assert any("behind the market" in reason for reason in report.dropped)

    def test_staleness_is_judged_against_the_universe_not_the_clock(self):
        """So replaying a past date behaves like a live scan."""
        old = pd.Timestamp("2024-03-01", tz="UTC")
        frames = {
            f"OK{i}": self.liquid(seed=i, start=old) for i in range(10)
        }
        symbols, _, _ = screen_liquidity(frames, UniverseFilter(min_dollar_volume=0.0))
        # Everything is equally old, so nothing is stale relative to the rest.
        assert len(symbols) == 10

    def test_a_halted_symbol_is_dropped(self):
        halted = self.liquid(seed=98)
        halted.loc[halted.index[-15:], "volume"] = 0.0
        symbols, _, _ = screen_liquidity(
            self.universe({"HALTED": halted}), UniverseFilter(min_dollar_volume=0.0)
        )
        assert "HALTED" not in symbols

    def test_the_active_day_check_names_its_reason(self):
        halted = self.liquid(seed=97)
        halted.loc[halted.index[-10:], "volume"] = 0.0
        profile = profile_liquidity("HALTED", halted)
        report = FilterReport()
        assert not passes_liquidity_filters(
            profile, UniverseFilter(min_dollar_volume=0.0), report
        )
        assert any("recent sessions" in reason for reason in report.dropped)

    def test_an_occasional_quiet_day_is_tolerated(self):
        """One dead session is normal; a filter that rejects it is too strict."""
        bars = self.liquid(seed=96)
        bars.loc[bars.index[-2], "volume"] = 0.0
        profile = profile_liquidity("AAA", bars)
        assert passes_liquidity_filters(profile, UniverseFilter(min_dollar_volume=0.0))

    def test_a_fully_traded_symbol_reports_every_session_active(self):
        assert profile_liquidity("AAA", self.liquid()).active_day_pct == 1.0

    def test_the_checks_can_be_disabled(self):
        stale = self.liquid(seed=95, start=pd.Timestamp("2025-01-02", tz="UTC"))
        symbols, _, _ = screen_liquidity(
            self.universe({"STALE": stale}),
            UniverseFilter(min_dollar_volume=0.0, max_staleness_days=0),
        )
        assert "STALE" in symbols

    @pytest.mark.parametrize("bad", [-0.1, 1.5])
    def test_the_active_day_share_is_bounded(self, bad):
        with pytest.raises(ValueError, match="min_active_day_pct"):
            UniverseFilter(min_active_day_pct=bad)

    def test_negative_staleness_is_rejected(self):
        with pytest.raises(ValueError, match="max_staleness_days"):
            UniverseFilter(max_staleness_days=-1)
