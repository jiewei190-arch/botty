"""The websocket supervisor.

No test here opens a socket. The SDK's stream is replaced by a fake with the
same contract — coroutine handlers, a blocking ``run()`` that owns its own event
loop, a ``stop()`` that signals it — because what needs proving is the
supervision: that a rejected credential fails fast instead of retrying forever,
that a slow consumer drops the oldest event rather than wedging the socket, and
that a thread which dies says so.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timezone

import pytest

from trading_bot.data.market_stream import (
    SDK_LOGGER_NAME,
    HealthEvent,
    StreamConfig,
    StreamError,
    StreamEvent,
    StreamEventType,
    StreamSupervisor,
)

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


class FakeBar:
    def __init__(self, symbol: str, price: float) -> None:
        self.symbol = symbol
        self.timestamp = datetime.now(timezone.utc)
        self.open = self.high = self.low = self.close = price
        self.volume = 1_000
        self.trade_count = 10
        self.vwap = price


class FakeStream:
    """Same shape as ``StockDataStream``: coroutine handlers, blocking run()."""

    def __init__(self, *, bars: int = 5, fail_with: Exception | None = None) -> None:
        self.subscriptions: list[tuple[str, tuple[str, ...]]] = []
        self._handlers: list[tuple[object, tuple[str, ...]]] = []
        self._stop = threading.Event()
        self._bars = bars
        self._fail_with = fail_with
        self.stopped = False

    def subscribe_bars(self, handler, *symbols):
        self.subscriptions.append(("bars", symbols))
        self._handlers.append((handler, symbols))

    def subscribe_trades(self, handler, *symbols):
        self.subscriptions.append(("trades", symbols))
        self._handlers.append((handler, symbols))

    def subscribe_quotes(self, handler, *symbols):
        self.subscriptions.append(("quotes", symbols))

    def subscribe_daily_bars(self, handler, *symbols):
        self.subscriptions.append(("daily_bars", symbols))

    def run(self) -> None:
        if self._fail_with is not None:
            raise self._fail_with

        async def pump() -> None:
            for index in range(self._bars):
                for handler, symbols in self._handlers:
                    for symbol in symbols:
                        await handler(FakeBar(symbol, 100.0 + index))
                await asyncio.sleep(0)
            while not self._stop.is_set():
                await asyncio.sleep(0.01)

        asyncio.run(pump())

    def stop(self) -> None:
        self.stopped = True
        self._stop.set()


async def collect(supervisor: StreamSupervisor, count: int, timeout: float = 5.0):
    received: list[StreamEvent] = []

    async def pump() -> None:
        async for event in supervisor.events():
            received.append(event)
            if len(received) >= count:
                return

    await asyncio.wait_for(pump(), timeout)
    return received


def supervisor_for(stream: FakeStream, **kwargs) -> StreamSupervisor:
    defaults = {
        "stream_factory": lambda: stream,
        "preflight": lambda: None,
        "bridge_sdk_logs": False,
    }
    return StreamSupervisor(
        StreamConfig(symbols=("AAPL", "NVDA")), **(defaults | kwargs)
    )


class TestStartup:
    def test_rejects_more_symbols_than_the_plan_allows(self):
        async def go():
            supervisor = StreamSupervisor(
                StreamConfig(symbols=tuple(f"S{i}" for i in range(31))),
                stream_factory=FakeStream,
                preflight=lambda: None,
            )
            with pytest.raises(StreamError, match="30"):
                await supervisor.start()

        asyncio.run(go())

    def test_a_rejected_credential_fails_before_the_socket_opens(self):
        """The SDK would retry a 401 forever; the pre-flight turns it into an error."""
        stream = FakeStream()

        def reject() -> None:
            raise StreamError("Alpaca rejected these credentials.")

        async def go():
            supervisor = supervisor_for(stream, preflight=reject)
            with pytest.raises(StreamError, match="rejected"):
                await supervisor.start()
            assert stream.subscriptions == []

        asyncio.run(go())

    def test_subscribes_only_to_the_configured_channels(self):
        stream = FakeStream(bars=0)

        async def go():
            supervisor = StreamSupervisor(
                StreamConfig(symbols=("AAPL",), subscribe_bars=True, subscribe_trades=True),
                stream_factory=lambda: stream,
                preflight=lambda: None,
                bridge_sdk_logs=False,
            )
            await supervisor.start()
            await supervisor.stop()

        asyncio.run(go())
        assert {kind for kind, _ in stream.subscriptions} == {"bars", "trades"}

    def test_starting_twice_is_refused(self):
        async def go():
            supervisor = supervisor_for(FakeStream(bars=0))
            await supervisor.start()
            with pytest.raises(StreamError, match="already running"):
                await supervisor.start()
            await supervisor.stop()

        asyncio.run(go())

    def test_config_requires_symbols_and_a_channel(self):
        with pytest.raises(ValueError, match="at least one symbol"):
            StreamConfig(symbols=())
        with pytest.raises(ValueError, match="at least one channel"):
            StreamConfig(symbols=("AAPL",), subscribe_bars=False)


class TestPublishing:
    def test_bars_reach_the_consumer_normalised(self):
        async def go():
            supervisor = supervisor_for(FakeStream(bars=3))
            await supervisor.start()
            events = await collect(supervisor, 4)
            await supervisor.stop()
            return events

        events = asyncio.run(go())
        assert all(event.kind is StreamEventType.BAR for event in events)
        assert {event.symbol for event in events} == {"AAPL", "NVDA"}
        assert events[0].price == 100.0
        assert events[0].timestamp.tzinfo is timezone.utc

    def test_stats_count_what_arrived(self):
        async def go():
            supervisor = supervisor_for(FakeStream(bars=3))
            await supervisor.start()
            await collect(supervisor, 6)
            stats = supervisor.stats
            await supervisor.stop()
            return stats

        stats = asyncio.run(go())
        assert stats["received"] >= 6
        assert stats["dropped"] == 0

    def test_a_full_queue_drops_the_oldest_event(self):
        """A stale quote is worth less than a fresh one, and blocking the
        websocket thread to keep it would risk the connection."""

        async def go():
            supervisor = StreamSupervisor(
                StreamConfig(symbols=("AAPL",), queue_size=2),
                stream_factory=lambda: FakeStream(bars=0),
                preflight=lambda: None,
                bridge_sdk_logs=False,
            )
            await supervisor.start()
            for index in range(5):
                supervisor._accept(
                    StreamEvent(
                        StreamEventType.BAR,
                        "AAPL",
                        datetime.now(timezone.utc),
                        {"close": float(index)},
                    )
                )
            first = await supervisor.next_event(timeout=1)
            stats = supervisor.stats
            await supervisor.stop()
            return first, stats

        first, stats = asyncio.run(go())
        assert stats["dropped"] == 3
        assert first is not None
        assert first.payload["close"] == 3.0  # the oldest two were dropped

    def test_events_stop_when_the_stream_stops(self):
        async def go():
            supervisor = supervisor_for(FakeStream(bars=1))
            await supervisor.start()
            await collect(supervisor, 2)
            await supervisor.stop()
            # The sentinel ends the iterator rather than hanging forever.
            drained = [event async for event in supervisor.events()]
            return drained

        assert asyncio.run(go()) == []


class TestHealth:
    def test_lifecycle_is_reported(self):
        events: list[HealthEvent] = []

        async def go():
            supervisor = supervisor_for(FakeStream(bars=1), on_health=events.append)
            await supervisor.start()
            await collect(supervisor, 1)
            await supervisor.stop()

        asyncio.run(go())
        statuses = [(event.component, event.status) for event in events]
        assert ("preflight", "OK") in statuses
        assert ("stream", "STARTING") in statuses
        assert ("stream", "STOPPED") in statuses

    def test_a_thread_that_dies_is_reported_not_swallowed(self):
        events: list[HealthEvent] = []

        async def go():
            supervisor = supervisor_for(
                FakeStream(fail_with=RuntimeError("socket exploded")),
                on_health=events.append,
            )
            await supervisor.start()
            # The dying thread pushes the sentinel, so the consumer's iterator
            # ends rather than hanging on a queue nothing will ever fill again.
            return await asyncio.wait_for(collect(supervisor, 1), 5)

        assert asyncio.run(go()) == []
        assert any(event.status == "ERROR" for event in events)
        assert any("socket exploded" in event.message for event in events)
        assert any(event.status == "DISCONNECTED" for event in events)

    def test_a_failing_health_callback_does_not_break_the_stream(self, caplog):
        def explode(_event: HealthEvent) -> None:
            raise RuntimeError("health sink down")

        async def go():
            supervisor = supervisor_for(FakeStream(bars=1), on_health=explode)
            with caplog.at_level(logging.ERROR):
                await supervisor.start()
                events = await collect(supervisor, 1)
                await supervisor.stop()
            return events

        assert len(asyncio.run(go())) == 1

    def test_the_sdk_log_bridge_mirrors_connection_messages(self):
        events: list[HealthEvent] = []

        async def go():
            supervisor = supervisor_for(
                FakeStream(bars=0), on_health=events.append, bridge_sdk_logs=True
            )
            await supervisor.start()
            logging.getLogger(SDK_LOGGER_NAME).warning("data websocket error, restarting")
            await supervisor.stop()

        asyncio.run(go())
        assert any("restarting" in event.message for event in events)
        # The handler is removed on stop, so a later message is not mirrored.
        before = len(events)
        logging.getLogger(SDK_LOGGER_NAME).warning("after shutdown")
        assert len(events) == before


def test_next_event_before_start_is_an_error():
    async def go():
        supervisor = supervisor_for(FakeStream())
        with pytest.raises(StreamError, match="not been started"):
            await supervisor.next_event()

    asyncio.run(go())
