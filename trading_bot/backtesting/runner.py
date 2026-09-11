"""Running a backtest from plain parameters.

The engine takes prepared frames and strategy objects. Getting from what a
person actually specifies — symbols, a date range, a strategy name — to those
objects means fetching data, dropping forming bars, and building strategies with
overrides. That work is identical for the CLI and the dashboard, so it lives
here rather than in either.

Data is fetched once for the whole requested window and simulated bar by bar
inside it. No bar is fetched during the run, which is a small part of why the
simulation cannot see the future: there is nothing later to see.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from trading_bot.backtesting.engine import BacktestConfig, Backtester, BacktestResult
from trading_bot.backtesting.execution import CostModel
from trading_bot.config.settings import RiskSettings, Settings
from trading_bot.data.market_data import build_market_data, drop_incomplete_bars
from trading_bot.strategies import BaseStrategy, build_strategy
from trading_bot.utils.timeframes import Timeframe

logger = logging.getLogger(__name__)
WARMUP_MARGIN = 50


class BacktestDataError(RuntimeError):
    """No usable data could be loaded for the requested run."""


@dataclass(frozen=True, slots=True)
class BacktestRequest:
    """What a person asked for, before any of it is resolved."""

    symbols: tuple[str, ...]
    strategies: tuple[str, ...]
    timeframe: str = "1Day"
    start: datetime | None = None
    end: datetime | None = None
    starting_equity: float = 10_000.0
    costs: CostModel = field(default_factory=CostModel)
    risk: RiskSettings | None = None
    allow_short: bool = False
    min_confidence: float | None = None
    demo: bool = False
    use_cache: bool = True

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValueError("A backtest needs at least one symbol")
        if not self.strategies:
            raise ValueError("A backtest needs at least one strategy")
        if self.start is not None and self.end is not None and self.start >= self.end:
            raise ValueError(
                f"start ({self.start:%Y-%m-%d}) must be before end ({self.end:%Y-%m-%d})"
            )
        Timeframe.parse(self.timeframe)

    @property
    def effective_min_confidence(self) -> float | None:
        """Confidence floor actually used to build strategies.

        An explicit request override wins. Otherwise inherit the risk settings
        supplied by the dashboard/CLI. This prevents backtests from silently
        using a strategy's unrelated built-in default while the user believes
        the configured risk confidence floor is active.
        """
        if self.min_confidence is not None:
            return self.min_confidence
        if self.risk is not None:
            return float(self.risk.min_confidence)
        return None


@dataclass(slots=True)
class LoadedData:
    frames: dict[str, pd.DataFrame]
    failures: dict[str, str] = field(default_factory=dict)
    warmup_bars: int = 0


def build_strategies(request: BacktestRequest) -> list[BaseStrategy]:
    """Construct the request's strategies, applying its overrides."""
    overrides: dict[str, Any] = {}
    if request.allow_short:
        overrides["allow_short"] = True
    confidence = request.effective_min_confidence
    if confidence is not None:
        overrides["min_confidence"] = confidence
    return [build_strategy(name, **overrides) for name in request.strategies]


def load_data(
    request: BacktestRequest,
    settings: Settings | None = None,
    *,
    warmup: int,
) -> LoadedData:
    timeframe = Timeframe.parse(request.timeframe)
    end = request.end or datetime.now(timezone.utc)
    lead = warmup + WARMUP_MARGIN

    if request.demo:
        from trading_bot.main import _demo_bars

        span = _bars_between(request.start, end, timeframe) + lead
        frames = {symbol: _demo_bars(symbol, span) for symbol in request.symbols}
        return LoadedData(frames=frames, warmup_bars=lead)

    if settings is None:
        raise BacktestDataError("Settings are required to fetch market data")

    provider = build_market_data(
        settings.alpaca, settings.data, use_cache=request.use_cache
    )
    wanted = _bars_between(request.start, end, timeframe) + lead
    frames, report = provider.fetch_watchlist(
        list(request.symbols), timeframe, lookback_bars=wanted, end=end
    )
    frames = {
        symbol: drop_incomplete_bars(frame, timeframe)
        for symbol, frame in frames.items()
        if not frame.empty
    }
    if not frames:
        raise BacktestDataError(
            "No bars were returned for any symbol. Check credentials, the "
            "symbols, and that the date range covers trading days."
        )
    return LoadedData(frames=frames, failures=dict(report.failed), warmup_bars=lead)


def _bars_between(start: datetime | None, end: datetime, timeframe: Timeframe) -> int:
    if start is None:
        return 500
    span = end - start
    per_day = timeframe.periods_per_year / 252
    return max(1, int(span / timedelta(days=1) * per_day))


def _trim_to_window(
    frames: dict[str, pd.DataFrame],
    request: BacktestRequest,
    warmup: int,
) -> tuple[dict[str, pd.DataFrame], int | None]:
    lead = warmup + WARMUP_MARGIN
    trimmed: dict[str, pd.DataFrame] = {}

    for symbol, frame in frames.items():
        window = frame
        if request.end is not None:
            window = window[window.index <= pd.Timestamp(request.end)]
        if request.start is not None:
            start = pd.Timestamp(request.start)
            if window.index.tz is not None and start.tz is None:
                start = start.tz_localize(window.index.tz)
            position = int(window.index.searchsorted(start))
            if position >= len(window):
                continue
            window = window.iloc[max(0, position - lead) :]
        if not window.empty:
            trimmed[symbol] = window

    if request.start is None or not trimmed:
        return trimmed, None

    start = pd.Timestamp(request.start)
    timeline = sorted(set().union(*(frame.index for frame in trimmed.values())))
    if timeline and timeline[0].tz is not None and start.tz is None:
        start = start.tz_localize(timeline[0].tz)
    before = sum(1 for stamp in timeline if stamp < start)
    return trimmed, max(before, warmup)


def run_backtest(
    request: BacktestRequest, settings: Settings | None = None
) -> BacktestResult:
    """Resolve a request into data and strategies, then simulate it."""
    strategies = build_strategies(request)
    warmup = max(strategy.min_bars for strategy in strategies)

    data = load_data(request, settings, warmup=warmup)
    frames, engine_warmup = _trim_to_window(data.frames, request, warmup)
    if not frames:
        raise BacktestDataError(
            "No bars fall inside the requested date range. Check that it covers "
            "trading days and that history reaches that far back."
        )

    tester = Backtester(
        strategies,
        BacktestConfig(
            starting_equity=request.starting_equity,
            timeframe=request.timeframe,
            costs=request.costs,
            risk=request.risk or RiskSettings(),
            warmup_bars=engine_warmup,
        ),
    )
    result = tester.run(frames)
    if data.failures:
        logger.warning("Some symbols were not backtested: %s", data.failures)
    return result
