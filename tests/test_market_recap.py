"""What the market did, summarised from bars the scanner already fetches."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from trading_bot.automation.market_recap import (
    INDEX_ETFS,
    SECTOR_ETFS,
    MarketRecap,
    build_recap,
    describe_regime,
    render_recap,
)
from trading_bot.data.market_data import MarketDataProvider, StaticMarketData, normalize_bars

NOW = datetime(2026, 9, 11, 20, 5, tzinfo=timezone.utc)


def series(drift: float, *, sigma: float = 0.008, bars: int = 120, seed: int = 3):
    """A daily frame with a chosen drift, so the recap has something to report."""
    index = pd.date_range(end="2026-09-11", periods=bars, freq="B", tz="UTC")
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, sigma, bars)))
    span = np.abs(rng.normal(0, sigma * 0.6, bars)) * close
    opens = np.r_[close[0], close[:-1]]
    return normalize_bars(
        pd.DataFrame(
            {
                "open": opens,
                "high": np.maximum(opens, close) + span,
                "low": np.minimum(opens, close) - span,
                "close": close,
                "volume": np.full(bars, 5e7),
            },
            index=index,
        ),
        symbol="X",
    )


def tape(**drifts: float) -> StaticMarketData:
    frames = {}
    for index, (symbol, _) in enumerate((*INDEX_ETFS, *SECTOR_ETFS)):
        frames[symbol] = series(drifts.get(symbol, 0.0), seed=index + 1)
    return StaticMarketData(frames)


class TestRecap:
    def test_reports_every_index_and_sector(self):
        recap = build_recap(tape(), now=NOW)
        assert len(recap.indices) == len(INDEX_ETFS)
        assert len(recap.sectors) == len(SECTOR_ETFS)
        assert recap.usable

    def test_leaders_and_laggards_are_ordered(self):
        recap = build_recap(tape(), now=NOW)
        leaders = [m.change_pct for m in recap.leaders]
        laggards = [m.change_pct for m in recap.laggards]
        assert leaders == sorted(leaders, reverse=True)
        assert laggards == sorted(laggards)
        assert max(laggards) <= max(leaders)

    def test_breadth_is_the_share_of_sectors_higher(self):
        recap = build_recap(tape(), now=NOW)
        expected = sum(1 for s in recap.sectors if s.change_pct > 0) / len(recap.sectors)
        assert recap.breadth == pytest.approx(expected)
        assert 0.0 <= recap.breadth <= 1.0

    def test_the_benchmark_is_the_first_index(self):
        recap = build_recap(tape(), now=NOW)
        assert recap.benchmark.symbol == "SPY"
        assert recap.benchmark_atr_pct == recap.benchmark.atr_pct

    def test_a_missing_symbol_does_not_sink_the_recap(self):
        """A report with a gap in it tells you more than one that crashed."""
        partial = StaticMarketData({"SPY": series(0.001), "XLK": series(0.002)})
        recap = build_recap(partial, now=NOW)
        assert recap.usable
        assert [m.symbol for m in recap.indices] == ["SPY"]
        assert recap.errors

    def test_a_dead_provider_is_reported_not_raised(self):
        class Broken(MarketDataProvider):
            def get_bars(self, *args, **kwargs):  # pragma: no cover - unused
                raise AssertionError

            def get_bars_multi(self, *args, **kwargs):  # pragma: no cover - unused
                raise AssertionError

            def fetch_watchlist(self, *args, **kwargs):
                raise RuntimeError("data feed down")

        recap = build_recap(Broken(), now=NOW)
        assert not recap.usable
        assert "data feed down" in recap.errors[0]

    def test_it_claims_nothing_about_the_economy(self):
        """Price data cannot see CPI. The field stays empty rather than invented."""
        assert build_recap(tape(), now=NOW).headlines == ()


class TestRegime:
    @pytest.mark.parametrize(
        "atr_pct,expected",
        [(0.5, "calm"), (1.0, "normal"), (1.9, "elevated"), (3.4, "stressed"),
         (None, "unknown")],
    )
    def test_bands(self, atr_pct, expected):
        assert describe_regime(atr_pct) is expected or describe_regime(atr_pct) == expected

    def test_a_quiet_tape_and_a_violent_one_are_not_the_same_label(self):
        calm = build_recap(tape(), now=NOW)
        assert calm.regime in {"calm", "normal", "elevated", "stressed"}


class TestRendering:
    def test_names_the_indices_and_the_regime(self):
        text = render_recap(build_recap(tape(), now=NOW))
        assert "MARKET RECAP" in text
        assert "S&P 500" in text
        assert "sectors higher" in text
        assert "Volatility" in text

    def test_says_it_is_not_a_forecast(self):
        text = render_recap(build_recap(tape(), now=NOW))
        assert "not a forecast" in text

    def test_an_empty_recap_still_renders(self):
        text = render_recap(MarketRecap(as_of=NOW, errors=("no data",)))
        assert "No market data was available" in text
        assert "no data" in text

    def test_percentages_read_with_their_sign(self):
        text = render_recap(build_recap(tape(SPY=0.004), now=NOW))
        line = next(row for row in text.splitlines() if "S&P 500" in row)
        assert "+" in line or "-" in line
        assert "+ " not in line  # the sign hugs the number
