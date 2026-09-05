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
        assert profile.avg_dollar_volume == pytest.approx(50_000_000, rel=0.4)

    def test_volume_alone_does_not_imply_liquidity(self):
        """A million shares of a $0.40 stock is $400k, not $1m."""
        cheap = make_bars(60, seed=4, freq="1D", start_price=0.40)
        cheap["volume"] = 1_000_000.0
        profile = profile_liquidity("PENNY", cheap)
        assert profile.avg_dollar_volume < 1_000_000

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
            self.make_profile(avg_dollar_volume=50_000_000.0), UniverseFilter()
        )

    def test_thin_turnover_is_rejected(self):
        assert not passes_liquidity_filters(
            self.make_profile(avg_dollar_volume=80_000.0), UniverseFilter()
        )

    def test_a_cheap_stock_is_rejected(self):
        assert not passes_liquidity_filters(
            self.make_profile(last_close=2.0, avg_dollar_volume=50_000_000.0),
            UniverseFilter(),
        )

    def test_an_expensive_stock_is_rejected(self):
        """One share of a $2,000 stock can exceed a small account's risk budget."""
        assert not passes_liquidity_filters(
            self.make_profile(last_close=2_500.0, avg_dollar_volume=50_000_000.0),
            UniverseFilter(),
        )

    def test_a_recent_listing_is_rejected(self):
        """No 200-day average exists yet, so its indicators are arithmetic only."""
        assert not passes_liquidity_filters(
            self.make_profile(bars=30, avg_dollar_volume=50_000_000.0),
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
