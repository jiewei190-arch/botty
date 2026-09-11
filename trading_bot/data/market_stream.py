"""Real-time market data over Alpaca's websocket.

What the SDK already does, and what it does not
-----------------------------------------------
``alpaca-py``'s ``StockDataStream`` reconnects on its own, with exponential
backoff from 1s to 30s, and can force a reconnect when a socket goes quiet
(``data_timeout``). Reimplementing that here would be duplicated, worse code.

What it does *not* do is anything a long-running scanner needs around that:

* ``run()`` calls :func:`asyncio.run` internally, so it owns an event loop and
  blocks the calling thread. It cannot be awaited from inside an application
  that already has a loop.
* Handlers must be coroutines, and they execute on the stream's own loop — a
  different loop from the consumer's.
* **A rejected credential retries forever.** The SDK treats an auth failure like
  any other exception: it logs it and reconnects with backoff, indefinitely. A
  scanner started with a bad key would sit there looking busy and never say why.
* Nothing reports connection state to the application, only to a logger.

So this module supervises rather than reimplements: it pre-flights the
credentials over REST (where a rejection is a clean, explainable error), runs
the SDK in a worker thread, and republishes everything onto an
:class:`asyncio.Queue` the application can consume normally.

Backpressure
------------
The queue is bounded. If the consumer falls behind, the **oldest** event is
dropped rather than the newest, and the drop is counted. For a scanner this is
the right way round: a stale quote has no value, and blocking the websocket
thread to preserve one would risk the connection itself.

Free-plan limits
----------------
The Basic plan serves the IEX feed only and caps a subscription at 30 symbols
(https://docs.alpaca.markets/docs/about-market-data-api). IEX is a single venue
carrying a low single-digit percentage of consolidated volume, so streamed
volume figures are a *sample*, not the tape. Every relative measure built on
them compares IEX with IEX and stays meaningful; absolute share counts do not.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from trading_bot.config.universe import (
    FREE_STREAM_SYMBOL_LIMIT,
    normalize_symbols,
    stream_capacity_error,
)

logger = logging.getLogger(__name__)

#: Logger the SDK writes its connection lifecycle to. Mirrored into health
#: events for observability only — no control flow reads these messages, so a
#: change to the SDK's wording degrades the logs and nothing else.
SDK_LOGGER_NAME = "alpaca.data.live.websocket"


class StreamError(RuntimeError):
    """The stream could not be started, or could not be trusted once started."""


class StreamEventType(str, Enum):
    """What arrived."""

    TRADE = "trade"
    QUOTE = "quote"
    BAR = "bar"
    DAILY_BAR = "daily_bar"
    STATUS = "status"
    HEALTH = "health"


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """One message, normalised away from the SDK's model classes."""

    kind: StreamEventType
    symbol: str
    timestamp: datetime
    payload: dict[str, Any] = field(default_factory=dict)

    def value(self, name: str, default: Any = None) -> Any:
        return self.payload.get(name, default)

    @property
    def price(self) -> float | None:
        """The event's best single price, whatever kind it is."""
        for key in ("price", "close", "bid_price"):
            value = self.payload.get(key)
            if value is not None:
                return float(value)
        return None


@dataclass(frozen=True, slots=True)
class StreamConfig:
    """What to subscribe to, and how much to buffer."""

    symbols: tuple[str, ...]
    feed: str = "iex"
    subscribe_bars: bool = True
    subscribe_trades: bool = False
    subscribe_quotes: bool = False
    subscribe_daily_bars: bool = False
    #: Events buffered before the oldest are dropped. Ten thousand is a few
    #: minutes of minute-bars for thirty symbols, or seconds of quotes — which
    #: is why quotes are off by default.
    queue_size: int = 10_000
    #: Symbols allowed on one subscription. The free plan's cap.
    symbol_limit: int = FREE_STREAM_SYMBOL_LIMIT
    #: Seconds of silence before the SDK forces a reconnect. Left off by
    #: default because a legitimately quiet subscription — thirty symbols of
    #: minute bars overnight — would otherwise reconnect in a loop forever.
    data_timeout: float | None = None
    #: Seconds of silence before *we* warn. Advisory: the market can be quiet.
    staleness_warning_seconds: float = 300.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbols", normalize_symbols(self.symbols))
        if not self.symbols:
            raise ValueError("StreamConfig needs at least one symbol")
        if not any(
            (
                self.subscribe_bars,
                self.subscribe_trades,
                self.subscribe_quotes,
                self.subscribe_daily_bars,
            )
        ):
            raise ValueError("StreamConfig must subscribe to at least one channel")
        if self.queue_size < 1:
            raise ValueError("queue_size must be positive")


@dataclass(frozen=True, slots=True)
class HealthEvent:
    """A statement about the stream itself rather than about the market."""

    component: str
    status: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


HealthCallback = Callable[[HealthEvent], None]


def _as_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return datetime.now(timezone.utc)


def _fields(model: Any, names: Iterable[str]) -> dict[str, Any]:
    """Pull named attributes off an SDK model or a raw dict."""
    if isinstance(model, dict):
        return {name: model.get(name) for name in names if model.get(name) is not None}
    return {
        name: getattr(model, name)
        for name in names
        if getattr(model, name, None) is not None
    }


class _SdkLogBridge(logging.Handler):
    """Mirrors the SDK's connection logging into health events.

    The SDK reports connects, disconnects and reconnect attempts to a logger and
    nowhere else. Rather than leave the application blind to its own transport,
    every record from that logger is forwarded verbatim. Nothing parses the
    text: the message is carried as data, the level becomes the status, and that
    is the whole contract.
    """

    def __init__(self, callback: HealthCallback) -> None:
        super().__init__(level=logging.INFO)
        self._callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._callback(
                HealthEvent(
                    component="websocket",
                    status=record.levelname,
                    message=record.getMessage(),
                    detail={"source": record.name},
                )
            )
        except Exception:  # pragma: no cover - a logging handler must never raise
            self.handleError(record)


class StreamSupervisor:
    """Runs an Alpaca data stream and republishes it onto an asyncio queue."""

    def __init__(
        self,
        config: StreamConfig,
        *,
        api_key: str | None = None,
        secret_key: str | None = None,
        stream_factory: Callable[[], Any] | None = None,
        preflight: Callable[[], None] | None = None,
        on_health: HealthCallback | None = None,
        bridge_sdk_logs: bool = True,
    ) -> None:
        self.config = config
        self._api_key = api_key
        self._secret_key = secret_key
        self._stream_factory = stream_factory or self._default_stream_factory
        self._preflight = preflight if preflight is not None else self._default_preflight
        self._on_health = on_health
        self._bridge_sdk_logs = bridge_sdk_logs

        self._stream: Any = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[StreamEvent | None] | None = None
        self._bridge: _SdkLogBridge | None = None
        self._watchdog: asyncio.Task[None] | None = None
        self._stopping = False
        self._started = False
        self._dropped = 0
        self._received = 0
        self._last_event_at: datetime | None = None

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Validate, connect and begin publishing events.

        Raises :class:`StreamError` before opening a socket when the symbol list
        exceeds the plan's cap or the credentials are rejected — both of which
        the SDK would otherwise turn into an endless silent retry.
        """
        if self._started:
            raise StreamError("Stream is already running")

        capacity = stream_capacity_error(self.config.symbols, self.config.symbol_limit)
        if capacity:
            raise StreamError(capacity)

        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=self.config.queue_size)
        self._stopping = False

        # Pre-flight off the event loop: it makes a blocking HTTP call.
        await asyncio.to_thread(self._preflight)
        self._health("preflight", "OK", "Credentials accepted by the REST API")

        self._stream = self._stream_factory()
        self._subscribe()

        if self._bridge_sdk_logs:
            self._bridge = _SdkLogBridge(self._health_event)
            logging.getLogger(SDK_LOGGER_NAME).addHandler(self._bridge)

        self._thread = threading.Thread(
            target=self._run_stream, name="alpaca-market-stream", daemon=True
        )
        self._thread.start()
        self._started = True
        self._watchdog = asyncio.create_task(self._watch_staleness())
        self._health(
            "stream",
            "STARTING",
            f"Subscribed to {len(self.config.symbols)} symbols on the {self.config.feed} feed",
            {"symbols": list(self.config.symbols), "feed": self.config.feed},
        )

    async def stop(self, *, timeout: float = 10.0) -> None:
        """Ask the stream to close and stop publishing."""
        self._stopping = True
        if self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None

        stream = self._stream
        if stream is not None:
            try:
                # ``stop()`` reaches for the loop ``run()`` created; if the
                # thread has not got that far, it raises rather than doing
                # nothing. Either way the thread is a daemon and cannot outlive
                # the process, so a failure here is logged, not fatal.
                await asyncio.to_thread(stream.stop)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Stream stop() did not complete cleanly: %s", exc)

        thread = self._thread
        if thread is not None and thread.is_alive():
            await asyncio.to_thread(thread.join, timeout)
            if thread.is_alive():
                logger.warning(
                    "Market stream thread did not stop within %.0fs; it is a daemon "
                    "and will end with the process",
                    timeout,
                )

        self._detach_bridge()
        self._started = False
        self._enqueue(None)
        self._health("stream", "STOPPED", "Market stream stopped", self.stats)

    def _detach_bridge(self) -> None:
        if self._bridge is not None:
            logging.getLogger(SDK_LOGGER_NAME).removeHandler(self._bridge)
            self._bridge = None

    # -- consumption -------------------------------------------------------

    async def events(self) -> AsyncIterator[StreamEvent]:
        """Yield events until :meth:`stop` is called.

        A ``None`` on the queue is the shutdown sentinel; it is never yielded.
        """
        if self._queue is None:
            raise StreamError("Stream has not been started")
        while True:
            event = await self._queue.get()
            if event is None:
                return
            yield event

    async def next_event(self, timeout: float | None = None) -> StreamEvent | None:
        """One event, or None on timeout or shutdown."""
        if self._queue is None:
            raise StreamError("Stream has not been started")
        try:
            event = (
                await asyncio.wait_for(self._queue.get(), timeout)
                if timeout is not None
                else await self._queue.get()
            )
        except asyncio.TimeoutError:
            return None
        return event

    @property
    def stats(self) -> dict[str, Any]:
        """Counters worth logging periodically."""
        return {
            "received": self._received,
            "dropped": self._dropped,
            "queued": self._queue.qsize() if self._queue else 0,
            "running": self.running,
            "last_event_at": self._last_event_at.isoformat() if self._last_event_at else None,
        }

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and not self._stopping)

    # -- internals ---------------------------------------------------------

    def _default_stream_factory(self) -> Any:
        from alpaca.data.enums import DataFeed
        from alpaca.data.live.stock import StockDataStream

        if not (self._api_key and self._secret_key):
            raise StreamError(
                "Streaming needs ALPACA_API_KEY and ALPACA_SECRET_KEY. "
                "Copy .env.example to .env and fill both in."
            )
        return StockDataStream(
            api_key=self._api_key,
            secret_key=self._secret_key,
            feed=DataFeed(self.config.feed),
            data_timeout=self.config.data_timeout,
        )

    def _default_preflight(self) -> None:
        """Prove the credentials over REST before opening a socket.

        The websocket answers a bad key by retrying forever. The REST API
        answers with a 401 that can be explained, so ask it first — one cheap
        request buys a real error message instead of a hang.
        """
        if not (self._api_key and self._secret_key):
            raise StreamError(
                "Streaming needs ALPACA_API_KEY and ALPACA_SECRET_KEY. "
                "Copy .env.example to .env and fill both in."
            )
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestBarRequest

        from trading_bot.universe.discovery import _is_auth_failure, explain_auth_failure

        client = StockHistoricalDataClient(self._api_key, self._secret_key)
        request = StockLatestBarRequest(
            symbol_or_symbols=[self.config.symbols[0]], feed=self.config.feed
        )
        try:
            client.get_stock_latest_bar(request)
        except Exception as exc:  # noqa: BLE001 - re-raised with a better message
            if _is_auth_failure(exc):
                raise StreamError(explain_auth_failure(self._api_key or "", exc)) from exc
            raise StreamError(f"Could not reach Alpaca market data: {exc}") from exc

    def _subscribe(self) -> None:
        symbols = self.config.symbols
        if self.config.subscribe_bars:
            self._stream.subscribe_bars(self._on_bar, *symbols)
        if self.config.subscribe_daily_bars:
            self._stream.subscribe_daily_bars(self._on_daily_bar, *symbols)
        if self.config.subscribe_trades:
            self._stream.subscribe_trades(self._on_trade, *symbols)
        if self.config.subscribe_quotes:
            self._stream.subscribe_quotes(self._on_quote, *symbols)

    def _run_stream(self) -> None:
        """The worker thread: ``run()`` blocks here until the stream stops."""
        try:
            self._stream.run()
        except BaseException as exc:  # noqa: BLE001 - the thread must report, not vanish
            logger.exception("Market stream thread exited with an error")
            self._health("stream", "ERROR", f"Stream thread failed: {exc}")
        finally:
            if not self._stopping:
                self._health(
                    "stream",
                    "DISCONNECTED",
                    "Stream thread ended without a stop request",
                    self.stats,
                )
            self._enqueue(None)

    async def _watch_staleness(self) -> None:
        """Warn when the stream goes quiet for longer than expected.

        Advisory only. Overnight, or on a thin symbol, silence is correct — so
        this reports rather than reconnects, and leaves the decision to whoever
        knows what session it is.
        """
        interval = max(self.config.staleness_warning_seconds, 5.0)
        try:
            while not self._stopping:
                await asyncio.sleep(interval)
                if self._stopping:
                    return
                last = self._last_event_at
                if last is None:
                    self._health(
                        "stream", "QUIET", f"No data in the first {interval:.0f}s", self.stats
                    )
                    continue
                idle = (datetime.now(timezone.utc) - last).total_seconds()
                if idle >= interval:
                    self._health(
                        "stream", "QUIET", f"No data for {idle:.0f}s", self.stats
                    )
        except asyncio.CancelledError:  # pragma: no cover - normal shutdown
            raise

    # -- handlers (run on the stream thread's event loop) -------------------

    async def _on_bar(self, data: Any) -> None:
        self._publish(
            StreamEventType.BAR,
            data,
            ("open", "high", "low", "close", "volume", "trade_count", "vwap"),
        )

    async def _on_daily_bar(self, data: Any) -> None:
        self._publish(
            StreamEventType.DAILY_BAR,
            data,
            ("open", "high", "low", "close", "volume", "trade_count", "vwap"),
        )

    async def _on_trade(self, data: Any) -> None:
        self._publish(
            StreamEventType.TRADE,
            data,
            ("price", "size", "exchange", "conditions", "tape"),
        )

    async def _on_quote(self, data: Any) -> None:
        self._publish(
            StreamEventType.QUOTE,
            data,
            ("bid_price", "bid_size", "ask_price", "ask_size", "bid_exchange", "ask_exchange"),
        )

    def _publish(self, kind: StreamEventType, data: Any, names: Iterable[str]) -> None:
        symbol = str(_fields(data, ("symbol",)).get("symbol", "")).upper()
        stamp = _as_utc(_fields(data, ("timestamp",)).get("timestamp"))
        event = StreamEvent(kind=kind, symbol=symbol, timestamp=stamp, payload=_fields(data, names))
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._accept, event)

    def _accept(self, event: StreamEvent) -> None:
        """Runs on the consumer's loop: count it, then queue it."""
        self._received += 1
        self._last_event_at = event.timestamp
        self._enqueue(event)

    def _enqueue(self, event: StreamEvent | None) -> None:
        queue = self._queue
        if queue is None:
            return
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            # Drop the oldest, not the newest: a scanner acts on what is
            # happening now, and an event from ten thousand messages ago has
            # already stopped being actionable.
            try:
                queue.get_nowait()
                self._dropped += 1
            except asyncio.QueueEmpty:  # pragma: no cover - racy but harmless
                return
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover - consumer wedged
                self._dropped += 1

    # -- health ------------------------------------------------------------

    def _health(
        self, component: str, status: str, message: str, detail: dict[str, Any] | None = None
    ) -> None:
        self._health_event(
            HealthEvent(component=component, status=status, message=message, detail=detail or {})
        )

    def _health_event(self, event: HealthEvent) -> None:
        logger.info(
            "stream health: %s %s — %s",
            event.component,
            event.status,
            event.message,
            extra={"health": {"component": event.component, "status": event.status}},
        )
        if self._on_health is None:
            return
        try:
            self._on_health(event)
        except Exception:  # noqa: BLE001 - health reporting must not break the stream
            logger.exception("Health callback failed")


__all__ = [
    "HealthEvent",
    "StreamConfig",
    "StreamError",
    "StreamEvent",
    "StreamEventType",
    "StreamSupervisor",
]
