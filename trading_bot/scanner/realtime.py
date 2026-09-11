"""The continuously running scanner.

Two data paths, one analysis path
---------------------------------
Bars arrive either from the websocket (live, minute by minute) or from REST
polling (a cycle every ``poll_seconds``). Both end up appending to the same
per-symbol frame, and both then call the same :class:`~trading_bot.detectors.DetectorEngine`.
That is deliberate: a detector must not be able to tell which path fed it, or the
live scanner and every offline test of it would be measuring different systems.

Why bars and not ticks
----------------------
The detectors read bars. Alpaca streams *minute* bars, so when the scanner is
configured for a coarser timeframe the minute bars are aggregated here, and a
bucket is published **only once it is complete**. A partially formed 15-minute
bar has a low that has not finished falling and a volume that has not finished
accumulating; feeding one to a detector produces a signal that changes its mind
four times before the bar closes. This is the same rule the backtester enforces,
which is what makes the two comparable.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from trading_bot.alerts.alert_manager import AlertManager
from trading_bot.alerts.models import Alert
from trading_bot.data.market_data import MarketDataProvider, normalize_bars
from trading_bot.data.market_stream import StreamEvent, StreamEventType
from trading_bot.detectors.engine import DetectorEngine, SymbolAnalysis
from trading_bot.utils.market_hours import MarketSession, describe, session_at
from trading_bot.utils.timeframes import Timeframe

logger = logging.getLogger(__name__)

#: Bars kept per symbol. Enough for a 30-bar volatility baseline and twenty
#: sessions of context, bounded so a process running for weeks cannot grow
#: without limit.
DEFAULT_MAX_BARS = 1_500


@dataclass(frozen=True, slots=True)
class RealtimeScannerConfig:
    """What the scanner watches and how often."""

    symbols: tuple[str, ...]
    timeframe: str = "5Min"
    #: Bars of history fetched per symbol before live data starts.
    lookback_bars: int = 400
    #: Seconds between REST scan cycles when not streaming.
    poll_seconds: int = 60
    max_bars: int = DEFAULT_MAX_BARS
    #: Analyse only when the market can actually trade. Off by default so the
    #: scanner still reports on pre-market and after-hours moves, which is when
    #: swing entries are often planned.
    regular_hours_only: bool = False


@dataclass
class _Bucket:
    """A partially formed bar being assembled from minute bars."""

    start: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def update(
        self, price_open: float, high: float, low: float, close: float, volume: float
    ) -> None:
        self.high = max(self.high, high)
        self.low = min(self.low, low)
        self.close = close
        self.volume += volume

    def as_row(self) -> dict[str, float]:
        return {
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


class BarAggregator:
    """Rolls streamed minute bars up into the configured timeframe.

    Emits a bar only when the next one starts, so nothing downstream ever sees
    an incomplete bucket. The cost is one bar of latency, which is the price of
    not acting on data that is still changing.
    """

    def __init__(self, timeframe: str | Timeframe) -> None:
        self.timeframe = Timeframe.parse(timeframe)
        self._buckets: dict[str, _Bucket] = {}

    def _floor(self, moment: datetime) -> datetime:
        """Start of the bucket ``moment`` belongs to."""
        return pd.Timestamp(moment).floor(self.timeframe.to_pandas_freq()).to_pydatetime()

    def add(
        self, symbol: str, bar: Mapping[str, Any], timestamp: datetime
    ) -> tuple[datetime, dict[str, float]] | None:
        """Fold one bar in; return a completed bar when the bucket rolls over."""
        start = self._floor(timestamp)
        values = (
            float(bar.get("open", 0.0)),
            float(bar.get("high", 0.0)),
            float(bar.get("low", 0.0)),
            float(bar.get("close", 0.0)),
            float(bar.get("volume", 0.0)),
        )
        current = self._buckets.get(symbol)

        if current is None:
            self._buckets[symbol] = _Bucket(start, *values)
            return None
        if start == current.start:
            current.update(*values)
            return None
        if start < current.start:
            # A late or replayed bar for a bucket already closed. Dropping it is
            # safer than reopening a bar the detectors have already judged.
            logger.debug("Ignoring out-of-order bar for %s at %s", symbol, timestamp)
            return None

        completed = (current.start, current.as_row())
        self._buckets[symbol] = _Bucket(start, *values)
        return completed

    def pending(self, symbol: str) -> dict[str, float] | None:
        bucket = self._buckets.get(symbol)
        return bucket.as_row() if bucket else None


@dataclass
class ScanCycle:
    """What one pass over the universe produced."""

    started_at: datetime
    analyses: tuple[SymbolAnalysis, ...] = ()
    alerts: tuple[Alert, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def signal_count(self) -> int:
        return sum(len(analysis.signals) for analysis in self.analyses)

    def summary(self) -> str:
        return (
            f"{len(self.analyses)} symbols, {self.signal_count} signals, "
            f"{len(self.alerts)} alerts"
        )


class RealtimeScanner:
    """Keeps per-symbol frames current and runs the detectors over them."""

    def __init__(
        self,
        provider: MarketDataProvider,
        *,
        config: RealtimeScannerConfig,
        engine: DetectorEngine | None = None,
        alerts: AlertManager | None = None,
        on_analysis: Callable[[SymbolAnalysis], None] | None = None,
    ) -> None:
        self.provider = provider
        self.config = config
        self.engine = engine or DetectorEngine()
        self.alerts = alerts
        self.on_analysis = on_analysis
        self.timeframe = Timeframe.parse(config.timeframe)
        self.aggregator = BarAggregator(self.timeframe)
        self._frames: dict[str, pd.DataFrame] = {}
        self._seeded = False

    # -- data --------------------------------------------------------------

    @property
    def frames(self) -> Mapping[str, pd.DataFrame]:
        return self._frames

    def seed(self) -> dict[str, int]:
        """Fetch history so the detectors have baselines from the first bar.

        Without this the scanner spends its first hour unable to say whether
        anything is unusual, because it has nothing to compare against.
        """
        frames, report = self.provider.fetch_watchlist(
            self.config.symbols,
            self.timeframe,
            lookback_bars=self.config.lookback_bars,
        )
        for symbol, frame in frames.items():
            if frame is not None and not frame.empty:
                self._frames[symbol.upper()] = self._trim(normalize_bars(frame, symbol=symbol))
        self._seeded = True
        missing = [s for s in self.config.symbols if s not in self._frames]
        if missing:
            logger.warning(
                "No history for %d of %d symbols: %s (%s)",
                len(missing),
                len(self.config.symbols),
                ", ".join(missing[:10]),
                report.summary(),
            )
        return {symbol: len(frame) for symbol, frame in self._frames.items()}

    def _trim(self, frame: pd.DataFrame) -> pd.DataFrame:
        if len(frame) > self.config.max_bars:
            return frame.iloc[-self.config.max_bars :]
        return frame

    def append_bar(self, symbol: str, timestamp: datetime, row: Mapping[str, float]) -> None:
        """Add one completed bar to a symbol's frame, replacing any duplicate."""
        ticker = symbol.upper()
        stamp = pd.Timestamp(timestamp)
        if stamp.tz is None:
            stamp = stamp.tz_localize(timezone.utc)
        else:
            stamp = stamp.tz_convert(timezone.utc)

        addition = pd.DataFrame([dict(row)], index=pd.DatetimeIndex([stamp], name="timestamp"))
        existing = self._frames.get(ticker)
        if existing is None or existing.empty:
            self._frames[ticker] = addition
            return
        columns = [c for c in existing.columns if c in addition.columns]
        combined = pd.concat([existing.drop(index=stamp, errors="ignore"), addition[columns]])
        self._frames[ticker] = self._trim(combined.sort_index())

    # -- analysis ----------------------------------------------------------

    def analyze(self, symbol: str, *, now: datetime | None = None) -> SymbolAnalysis | None:
        """Run the detectors over one symbol's current frame."""
        frame = self._frames.get(symbol.upper())
        if frame is None or frame.empty:
            return None
        moment = now or datetime.now(timezone.utc)
        if self.config.regular_hours_only and session_at(moment) is not MarketSession.REGULAR:
            return None

        analysis = self.engine.analyze(symbol, frame, now=moment)
        if self.on_analysis is not None:
            try:
                self.on_analysis(analysis)
            except Exception:  # noqa: BLE001 - a consumer must not end the scan
                logger.exception("Analysis callback failed for %s", symbol)
        return analysis

    def handle_event(self, event: StreamEvent) -> SymbolAnalysis | None:
        """Process one streamed message.

        Only completed buckets trigger analysis. Trades and quotes update
        nothing here — they arrive far faster than the detectors can use, and
        re-running five detectors per tick would burn CPU to produce the same
        answer thousands of times.
        """
        if event.kind not in (StreamEventType.BAR, StreamEventType.DAILY_BAR):
            return None
        if not event.symbol:
            return None

        completed = self.aggregator.add(event.symbol, event.payload, event.timestamp)
        if completed is None:
            return None
        start, row = completed
        self.append_bar(event.symbol, start, row)
        analysis = self.analyze(event.symbol, now=event.timestamp)
        if analysis is not None:
            self._raise_alerts([analysis])
        return analysis

    def scan_once(self, *, now: datetime | None = None) -> ScanCycle:
        """One REST pass: refresh every frame, analyse, alert.

        This is the fallback path when streaming is unavailable — and the path
        the ``scan`` CLI command uses, so the same code is exercised whether or
        not a websocket is running.
        """
        started = now or datetime.now(timezone.utc)
        errors: list[str] = []
        try:
            self.seed()
        except Exception as exc:  # noqa: BLE001 - a fetch failure is a cycle failure, not a crash
            logger.exception("Scan cycle could not refresh market data")
            errors.append(str(exc))

        analyses = [
            analysis
            for symbol in self.config.symbols
            if (analysis := self.analyze(symbol, now=started)) is not None
        ]
        ordered = tuple(sorted(analyses, key=lambda a: a.overall, reverse=True))
        alerts = self._raise_alerts(ordered)
        return ScanCycle(
            started_at=started, analyses=ordered, alerts=alerts, errors=tuple(errors)
        )

    def _raise_alerts(self, analyses: Sequence[SymbolAnalysis]) -> tuple[Alert, ...]:
        if self.alerts is None:
            return ()
        sent: list[Alert] = []
        for analysis in analyses:
            alert = self.alerts.consider(
                analysis.score, price=analysis.context.last_price
            )
            if alert is not None:
                sent.append(alert)
        return tuple(sent)

    # -- reporting ---------------------------------------------------------

    def status(self, *, now: datetime | None = None) -> dict[str, Any]:
        """A snapshot for logging or a health endpoint."""
        moment = now or datetime.now(timezone.utc)
        newest = [frame.index[-1] for frame in self._frames.values() if not frame.empty]
        return {
            "symbols": len(self.config.symbols),
            "frames": len(self._frames),
            "timeframe": self.timeframe.label,
            "seeded": self._seeded,
            "session": session_at(moment).value,
            "market": describe(moment),
            "newest_bar": max(newest).isoformat() if newest else None,
            "alerts": self.alerts.stats if self.alerts else {},
        }


def next_poll_delay(timeframe: Timeframe, *, floor_seconds: int = 15) -> float:
    """Seconds to wait before the next REST cycle.

    Polling faster than the bar size only re-reads the bar already held, so the
    interval tracks the timeframe — with a floor, because a 1-minute scanner
    hammering the API buys nothing on a 200-request-per-minute plan.
    """
    return max(float(timeframe.duration / timedelta(seconds=1)), float(floor_seconds))


__all__ = [
    "DEFAULT_MAX_BARS",
    "BarAggregator",
    "RealtimeScanner",
    "RealtimeScannerConfig",
    "ScanCycle",
    "next_poll_delay",
]
