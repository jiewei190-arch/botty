"""Data access for the dashboard.

Streamlit re-runs the whole script on every interaction, so anything expensive
must be cached or the app will hammer the API on every click. Fetches are cached
for a short TTL; indicator computation is cached on the frame itself.

Nothing here writes. The dashboard is read-only in this version and has no path
to the broker's order endpoints.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from trading_bot.backtesting.runner import (
    BacktestRequest,
    build_strategies,
    load_data,
    run_backtest,
)
from trading_bot.config.settings import Settings, TradingMode, load_settings
from trading_bot.data.database import Database
from trading_bot.data.market_data import build_market_data, drop_incomplete_bars
from trading_bot.indicators import IndicatorConfig, calculate_all_indicators
from trading_bot.risk import PortfolioState, RiskManager, build_portfolio_state
from trading_bot.scanner import MarketScanner, ScannerConfig
from trading_bot.strategies import build_strategy, explain_blockers
from trading_bot.utils.timeframes import Timeframe

logger = logging.getLogger(__name__)

#: Seconds a fetched frame stays cached. Short enough to feel live, long enough
#: that clicking around does not burn rate limit.
FETCH_TTL = 60

#: Backtest results are cached longer than quotes: the inputs are historical
#: and settled, so a result only changes when the request does.
BACKTEST_TTL = 900

#: A market-wide hunt is the most expensive thing this app does — minutes of
#: network time. Streamlit reruns the script on every widget interaction, so
#: without a cache, expanding a section would restart the scan.
HUNT_TTL = 900


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


def _adopt_streamlit_secrets() -> None:
    """Copy Streamlit secrets into the environment before settings are read.

    Settings come from environment variables, which is right for a local run
    with a ``.env``. A hosted deployment has no ``.env`` — credentials arrive
    as Streamlit secrets instead — so without this the app would start on a
    host and report missing credentials for a key that was supplied.

    Existing environment variables win: a value set explicitly on the host
    should not be silently overridden by a stale secret.
    """
    try:
        secrets = st.secrets
    except Exception:  # noqa: BLE001 - no secrets file is the normal local case
        return
    for key in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "RISK_ACCOUNT_EQUITY"):
        try:
            value = secrets[key]
        except Exception:  # noqa: BLE001 - absent key
            continue
        if value and key not in os.environ:
            os.environ[key] = str(value)


@st.cache_resource
def get_settings_cached() -> Settings:
    """Load settings once per session."""
    _adopt_streamlit_secrets()
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


def portfolio_for(settings: Settings, equity: float | None = None):
    """Portfolio state for risk sizing, from the broker when it is reachable.

    Falls back to a stated equity so the dashboard can demonstrate sizing
    without an account — clearly, rather than by inventing one.
    """
    from trading_bot.data.database import Database

    account, _ = account_snapshot(settings)
    positions: list = []
    if account is not None:
        try:
            from trading_bot.execution.broker import build_broker

            positions = build_broker(settings).get_positions()
        except Exception as error:  # noqa: BLE001
            logger.warning("Could not read broker positions: %s", error)

    database = None
    try:
        database = Database(settings.data.database_path)
        database.initialize()
    except Exception as error:  # noqa: BLE001
        logger.warning("Could not open the database: %s", error)

    state = build_portfolio_state(
        account=account,
        broker_positions=positions,
        database=database,
        equity=equity if account is None else None,
    )
    if database is not None:
        database.close()
    return state


def evaluate_risk(
    signals: list, settings: Settings, equity: float | None = None
) -> tuple[list, Any, str | None]:
    """Size and validate signals.

    Returns
    -------
    tuple
        ``(decisions, portfolio, halt_reason)``.
    """
    portfolio = portfolio_for(settings, equity)
    manager = RiskManager(settings.risk)
    halt = manager.trading_halted(portfolio)
    return manager.evaluate_many(signals, portfolio), portfolio, halt


def run_ranked_scan(
    symbols: list[str],
    strategy_names: list[str],
    timeframe_label: str,
    bars: int,
    demo: bool,
    settings: Settings,
    indicators: IndicatorConfig,
    *,
    equity: float | None = None,
    overrides: dict[str, Any] | None = None,
    min_dollar_volume: float = 0.0,
):
    """Fetch, scan and rank the watchlist.

    Returns the scanner's own :class:`~trading_bot.scanner.ScanResult` plus the
    portfolio it was sized against, so the page can show both.
    """
    frames: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    for symbol in symbols:
        loaded = load_symbol(symbol, timeframe_label, bars, demo, settings, indicators)
        if loaded.ok:
            frames[symbol] = loaded.frame
        else:
            failures[symbol] = loaded.error or "no data"

    portfolio = portfolio_for(settings, equity)
    strategies = [
        build_strategy(name, indicators=indicators, **(overrides or {}))
        for name in strategy_names
    ]
    scanner = MarketScanner(
        strategies,
        indicators=indicators,
        risk_manager=RiskManager(settings.risk),
        config=ScannerConfig(min_avg_dollar_volume=min_dollar_volume),
    )
    result = scanner.scan(frames, portfolio=portfolio)
    result.failures.update(failures)
    return result, portfolio


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


@st.cache_data(ttl=BACKTEST_TTL, show_spinner=False, hash_funcs={Settings: id})
def run_backtest_cached(_request: BacktestRequest, _settings: Settings):
    """Run a backtest, caching on the request.

    Backtests are expensive enough that re-running one on every rerender would
    make the page unusable — Streamlit reruns the whole script on each widget
    change. The request is a frozen dataclass, so caching on it is safe: two
    equal requests describe the same simulation.
    """
    return run_backtest(_request, _settings)


@st.cache_data(ttl=BACKTEST_TTL, show_spinner=False, hash_funcs={Settings: id})
def backtest_frames(
    _request: BacktestRequest, _settings: Settings
) -> dict[str, Any]:
    """Bars for the charts, over the same window the backtest ran on.

    Fetched through the same loader as the run itself, so the markers land on
    the bars that produced them rather than on a window that has since moved.
    """
    request, settings = _request, _settings
    try:
        strategies = build_strategies(request)
        warmup = max(strategy.min_bars for strategy in strategies)
        return load_data(request, settings, warmup=warmup).frames
    except Exception:  # noqa: BLE001 - charts are optional, the result is not
        logger.exception("Could not reload frames for the backtest charts")
        return {}


@st.cache_data(ttl=HUNT_TTL, show_spinner=False, hash_funcs={Settings: id})
def run_hunt(
    _settings: Settings,
    *,
    strategies: tuple[str, ...],
    equity: float,
    risk_pct: float,
    top_n: int,
    min_turnover: float,
    min_price: float,
    max_price: float,
    min_risk_reward: float,
    max_age: int,
    include_leveraged: bool,
):
    """Discover the universe and sweep it, cached on the parameters.

    A market-wide sweep takes minutes, and Streamlit reruns the whole script on
    every widget interaction. Without caching, opening an expander would start
    the scan again.
    """
    from trading_bot.risk.position_sizing import to_decimal
    from trading_bot.scanner.market_scan import HuntConfig, sweep_market
    from trading_bot.strategies import build_strategy
    from trading_bot.universe import AssetCatalogue, UniverseFilter, build_universe

    filters = UniverseFilter(
        min_price=min_price,
        max_price=max_price,
        min_dollar_volume=min_turnover,
        exclude_leveraged=not include_leveraged,
    )
    provider = build_market_data(_settings.alpaca, _settings.data)
    catalogue = AssetCatalogue(_settings.alpaca, cache_dir=_settings.data.cache_dir)
    universe = build_universe(catalogue, provider, filters)

    amount = to_decimal(equity)
    portfolio = PortfolioState(equity=amount, cash=amount, buying_power=amount)
    risk = _settings.risk.model_copy(update={"max_risk_per_trade_pct": risk_pct})

    return sweep_market(
        universe,
        [build_strategy(name) for name in strategies],
        portfolio=portfolio,
        risk_manager=RiskManager(risk),
        config=HuntConfig(
            top_n=top_n,
            min_risk_reward=min_risk_reward,
            max_signal_age_bars=max_age,
        ),
    )
