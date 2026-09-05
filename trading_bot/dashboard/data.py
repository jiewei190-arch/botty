"""Data access for the dashboard.

Streamlit re-runs the whole script on every interaction, so anything expensive
must be cached or the app will hammer the API on every click. Fetches are cached
for a short TTL; indicator computation is cached on the frame itself.

Nothing here writes. The dashboard is read-only in this version and has no path
to the broker's order endpoints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from trading_bot.config.settings import Settings, TradingMode, load_settings
from trading_bot.data.database import Database
from trading_bot.data.market_data import build_market_data, drop_incomplete_bars
from trading_bot.indicators import IndicatorConfig, calculate_all_indicators
from trading_bot.strategies import build_strategy, explain_blockers
from trading_bot.utils.timeframes import Timeframe

logger = logging.getLogger(__name__)

#: Seconds a fetched frame stays cached. Short enough to feel live, long enough
#: that clicking around does not burn rate limit.
FETCH_TTL = 60


@dataclass(frozen=True, slots=True)
class SymbolData:
    """Bars for one symbol, plus how they were obtained."""

    symbol: str
    frame: pd.DataFrame
    is_demo: bool
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and not self.frame.empty


@st.cache_resource
def get_settings_cached() -> Settings:
    """Load settings once per session."""
    return load_settings()


@st.cache_data(ttl=FETCH_TTL, show_spinner=False)
def fetch_bars(
    symbol: str,
    timeframe_label: str,
    bars: int,
    demo: bool,
    _settings: Settings,
) -> tuple[pd.DataFrame, str | None]:
    """Fetch bars for one symbol, returning ``(frame, error)``.

    Errors are returned rather than raised so one bad symbol cannot blank the
    whole page.
    """
    if demo:
        from trading_bot.main import _demo_bars

        return _demo_bars(symbol, bars), None

    timeframe = Timeframe.parse(timeframe_label)
    try:
        provider = build_market_data(_settings.alpaca, _settings.data)
        frames, report = provider.fetch_watchlist([symbol], timeframe, lookback_bars=bars)
        frame = frames.get(symbol)
        if frame is None or frame.empty:
            return pd.DataFrame(), report.failed.get(symbol, "no bars returned")
        # A forming bar must never reach the charts or the strategies.
        return drop_incomplete_bars(frame, timeframe), None
    except Exception as error:  # noqa: BLE001 - surfaced in the UI
        logger.exception("Dashboard fetch failed for %s", symbol)
        return pd.DataFrame(), str(error)


@st.cache_data(ttl=FETCH_TTL, show_spinner=False)
def enrich(frame: pd.DataFrame, _config: IndicatorConfig) -> pd.DataFrame:
    """Append indicator columns, cached on the frame's contents."""
    return calculate_all_indicators(frame, _config)


def load_symbol(
    symbol: str,
    timeframe_label: str,
    bars: int,
    demo: bool,
    settings: Settings,
    indicators: IndicatorConfig,
) -> SymbolData:
    """Fetch and enrich one symbol, never raising."""
    frame, error = fetch_bars(symbol, timeframe_label, bars, demo, settings)
    if error is not None or frame.empty:
        return SymbolData(symbol, pd.DataFrame(), demo, error or "no bars")
    try:
        return SymbolData(symbol, enrich(frame, indicators), demo)
    except Exception as error:  # noqa: BLE001
        logger.exception("Indicator calculation failed for %s", symbol)
        return SymbolData(symbol, pd.DataFrame(), demo, str(error))


def run_scan(
    symbols: list[str],
    strategy_names: list[str],
    timeframe_label: str,
    bars: int,
    demo: bool,
    settings: Settings,
    indicators: IndicatorConfig,
    overrides: dict[str, Any] | None = None,
) -> tuple[list, dict[str, int], dict[str, str]]:
    """Run strategies across symbols.

    Returns
    -------
    tuple
        ``(signals, blockers, failures)`` — signals sorted by confidence, a count
        of which conditions blocked entries, and per-symbol errors.
    """
    signals: list = []
    blockers: dict[str, int] = {}
    failures: dict[str, str] = {}

    strategies = [
        build_strategy(name, indicators=indicators, **(overrides or {}))
        for name in strategy_names
    ]

    for symbol in symbols:
        loaded = load_symbol(symbol, timeframe_label, bars, demo, settings, indicators)
        if not loaded.ok:
            failures[symbol] = loaded.error or "no data"
            continue
        for strategy in strategies:
            try:
                signal = strategy.generate_signal(symbol, loaded.frame)
            except Exception as error:  # noqa: BLE001 - one failure must not stop the scan
                logger.exception("%s failed on %s", strategy.name, symbol)
                failures[f"{symbol}/{strategy.name}"] = str(error)
                continue
            if signal is not None:
                signals.append(signal)
            else:
                for name in explain_blockers(strategy):
                    key = f"{strategy.name}.{name}"
                    blockers[key] = blockers.get(key, 0) + 1

    signals.sort(key=lambda item: item.confidence, reverse=True)
    return signals, blockers, failures


def account_snapshot(settings: Settings) -> tuple[Any | None, str | None]:
    """Broker account state, or a reason it is unavailable."""
    if settings.trading_mode is TradingMode.BACKTEST:
        return None, "Backtest mode does not use a broker connection"
    if not settings.alpaca.has_credentials:
        return None, "No Alpaca credentials configured"
    try:
        from trading_bot.execution.broker import build_broker

        return build_broker(settings).get_account(), None
    except Exception as error:  # noqa: BLE001
        return None, str(error)


def database_summary(settings: Settings) -> dict[str, Any]:
    """Counts from the trade database, for the overview page."""
    try:
        database = Database(settings.data.database_path)
        database.initialize()
        stats = database.trades.statistics()
        stats["open_positions"] = database.positions.count()
        stats["recent_signals"] = len(database.signals.recent(limit=50))
        database.close()
        return stats
    except Exception as error:  # noqa: BLE001
        logger.warning("Could not read the database: %s", error)
        return {"error": str(error)}
