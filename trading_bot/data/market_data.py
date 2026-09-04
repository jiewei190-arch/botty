"""Market data access.

:class:`MarketDataProvider` is the contract every consumer codes against. Two
implementations ship here:

* :class:`AlpacaMarketData` — live Alpaca REST access with retry and caching.
* :class:`StaticMarketData` — serves pre-loaded frames, for tests and for
  replaying a backtest without touching the network.

Normalization guarantees
------------------------
Every frame returned by any provider satisfies:

1. ``DatetimeIndex`` named ``timestamp``, timezone-aware in **UTC**.
2. Sorted ascending, with duplicate timestamps collapsed (last wins).
3. Columns ``open, high, low, close, volume`` present as ``float64``, plus
   ``trade_count`` and ``vwap`` when the vendor supplies them.
4. Rows with impossible OHLC relationships or non-positive prices removed.

Lookahead safety
----------------
Alpaca stamps a bar with its **opening** time. A 15-minute bar labelled 14:00 is
still forming until 14:15, so acting on it at 14:05 uses information from the
future relative to the label. :func:`drop_incomplete_bars` removes such bars and
is applied on every live fetch. Backtests read closed history only, so the same
rule is enforced there by construction.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest

from trading_bot.config.settings import AlpacaSettings, DataSettings
from trading_bot.data.cache import BarCache
from trading_bot.data.models import BAR_COLUMNS, OHLCV_COLUMNS, DataFetchReport, Quote
from trading_bot.utils.retry import retry_call
from trading_bot.utils.timeframes import Timeframe

logger = logging.getLogger(__name__)

#: Alpaca caps a multi-symbol bars request; larger watchlists are chunked.
MAX_SYMBOLS_PER_REQUEST = 100


class MarketDataError(RuntimeError):
    """Raised when market data cannot be retrieved or is unusable."""


def ensure_utc(moment: datetime | str | pd.Timestamp) -> datetime:
    """Coerce ``moment`` to a timezone-aware UTC datetime.

    Naive inputs are *assumed* to be UTC rather than local time, so behaviour does
    not change with the machine's timezone.
    """
    stamp = pd.Timestamp(moment)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.to_pydatetime()


def normalize_bars(frame: pd.DataFrame, *, symbol: str | None = None) -> pd.DataFrame:
    """Apply the normalization guarantees documented in the module docstring."""
    if frame is None or frame.empty:
        return empty_bars()

    result = frame.copy()

    # Alpaca returns a (symbol, timestamp) MultiIndex even for single symbols.
    if isinstance(result.index, pd.MultiIndex):
        if symbol is not None and "symbol" in (result.index.names or []):
            level = result.index.get_level_values("symbol")
            result = result[level == symbol]
            if result.empty:
                return empty_bars()
        has_symbol_level = "symbol" in (result.index.names or [])
        result = result.droplevel("symbol") if has_symbol_level else result.droplevel(0)

    if not isinstance(result.index, pd.DatetimeIndex):
        if "timestamp" in result.columns:
            result = result.set_index("timestamp")
        else:
            result.index = pd.to_datetime(result.index, utc=True)

    index = pd.DatetimeIndex(pd.to_datetime(result.index, utc=True))
    index.name = "timestamp"
    result.index = index

    result = result.sort_index()
    result = result[~result.index.duplicated(keep="last")]

    missing = [column for column in OHLCV_COLUMNS if column not in result.columns]
    if missing:
        raise MarketDataError(
            f"Bar data for {symbol or '<unknown>'} is missing required columns: {missing}"
        )

    keep = [column for column in BAR_COLUMNS if column in result.columns]
    result = result[keep].astype("float64")

    before = len(result)
    result = _drop_invalid_rows(result)
    dropped = before - len(result)
    if dropped:
        logger.warning(
            "Dropped %d malformed bar(s) for %s during normalization", dropped, symbol or "?"
        )
    return result


def _drop_invalid_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove bars that violate OHLC invariants or contain non-positive prices."""
    prices = frame[["open", "high", "low", "close"]]
    valid = (
        prices.notna().all(axis=1)
        & (prices > 0).all(axis=1)
        & (frame["high"] >= frame["low"])
        & (frame["high"] >= frame[["open", "close"]].max(axis=1))
        & (frame["low"] <= frame[["open", "close"]].min(axis=1))
        & frame["volume"].notna()
        & (frame["volume"] >= 0)
    )
    return frame[valid]


def empty_bars() -> pd.DataFrame:
    """An empty frame with the canonical schema, so callers never special-case None."""
    index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    return pd.DataFrame({column: pd.Series(dtype="float64") for column in BAR_COLUMNS}, index=index)


def drop_incomplete_bars(
    frame: pd.DataFrame,
    timeframe: Timeframe,
    *,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Drop trailing bars whose period has not closed yet.

    This is the primary lookahead guard: a bar stamped ``T`` only closes at
    ``T + timeframe.duration``, so it must not influence a decision made before then.
    """
    if frame.empty:
        return frame
    current = ensure_utc(now or datetime.now(timezone.utc))
    cutoff = pd.Timestamp(current) - timeframe.duration
    complete = frame[frame.index <= cutoff]
    removed = len(frame) - len(complete)
    if removed:
        logger.debug(
            "Dropped %d incomplete %s bar(s) newer than %s", removed, timeframe.label, cutoff
        )
    return complete


class MarketDataProvider(ABC):
    """Interface implemented by every source of price data."""

    @abstractmethod
    def get_bars(
        self,
        symbol: str,
        timeframe: str | Timeframe,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Return normalized bars for a single symbol."""

    @abstractmethod
    def get_bars_multi(
        self,
        symbols: Sequence[str],
        timeframe: str | Timeframe,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Return normalized bars for several symbols keyed by symbol."""

    def get_latest_quote(self, symbol: str) -> Quote | None:  # pragma: no cover - optional
        """Latest top-of-book quote, when the provider supports it."""
        return None

    def get_latest_price(self, symbol: str) -> float | None:
        """Best available current price: quote mid, else last bar close."""
        quote = self.get_latest_quote(symbol)
        if quote is not None and quote.mid_price > 0:
            return quote.mid_price
        bars = self.get_bars(symbol, "1Min", limit=1)
        if bars.empty:
            return None
        return float(bars["close"].iloc[-1])

    def fetch_watchlist(
        self,
        symbols: Sequence[str],
        timeframe: str | Timeframe,
        *,
        lookback_bars: int = 300,
        end: datetime | None = None,
    ) -> tuple[dict[str, pd.DataFrame], DataFetchReport]:
        """Fetch history for a watchlist, reporting per-symbol success or failure.

        One bad symbol must never abort a scan, so failures are collected rather
        than raised.
        """
        parsed = Timeframe.parse(timeframe)
        finish = ensure_utc(end or datetime.now(timezone.utc))
        start = finish - parsed.calendar_span_for_bars(lookback_bars)
        report = DataFetchReport(requested=list(symbols))
        frames: dict[str, pd.DataFrame] = {}

        try:
            fetched = self.get_bars_multi(symbols, parsed, start=start, end=finish)
        except Exception as error:  # noqa: BLE001 - degrade to per-symbol fetches
            logger.warning("Batch fetch failed (%s); falling back to per-symbol requests", error)
            fetched = {}
            for symbol in symbols:
                try:
                    fetched[symbol] = self.get_bars(symbol, parsed, start=start, end=finish)
                except Exception as symbol_error:  # noqa: BLE001
                    report.failed[symbol] = str(symbol_error)

        cutoff = pd.Timestamp(finish)
        for symbol in symbols:
            frame = fetched.get(symbol)
            if frame is None or frame.empty:
                report.failed.setdefault(symbol, "no bars returned")
                continue
            # Never trust the vendor to have honoured `end`. When a scan is
            # simulated "as of" a past moment, a stray later bar is a lookahead
            # leak, so clamp before trimming.
            frame = frame[frame.index <= cutoff]
            if frame.empty:
                report.failed.setdefault(symbol, "no bars at or before the requested end")
                continue
            trimmed = frame.tail(lookback_bars)
            frames[symbol] = trimmed
            report.succeeded[symbol] = len(trimmed)

        report.extra["timeframe"] = parsed.label
        report.extra["start"] = start.isoformat()
        report.extra["end"] = finish.isoformat()
        return frames, report


class AlpacaMarketData(MarketDataProvider):
    """Alpaca-backed provider with retry, chunking and an optional parquet cache."""

    def __init__(
        self,
        settings: AlpacaSettings,
        data_settings: DataSettings | None = None,
        *,
        client: StockHistoricalDataClient | None = None,
        cache: BarCache | None = None,
    ) -> None:
        if client is None and not settings.has_credentials:
            raise MarketDataError(
                "Alpaca credentials are missing. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
                "in your .env file (copy .env.example to get started)."
            )
        self._settings = settings
        self._data_settings = data_settings
        self._client = client or StockHistoricalDataClient(
            api_key=settings.api_key,
            secret_key=settings.secret_key,
        )
        self._cache = cache
        self._feed = DataFeed(settings.data_feed)
        self._adjustment = Adjustment(settings.adjustment)

    # -- internals ---------------------------------------------------------------

    def _call(self, func, description: str):
        return retry_call(
            func,
            max_attempts=self._settings.max_retries,
            base_delay=self._settings.retry_base_delay_seconds,
            description=description,
        )

    def _resolve_window(
        self,
        timeframe: Timeframe,
        start: datetime | None,
        end: datetime | None,
        limit: int | None,
    ) -> tuple[datetime, datetime]:
        finish = ensure_utc(end or datetime.now(timezone.utc))
        if start is not None:
            return ensure_utc(start), finish
        span = timeframe.calendar_span_for_bars(limit or 300)
        return finish - span, finish

    def _request_bars(
        self,
        symbols: list[str],
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
        limit: int | None,
    ) -> pd.DataFrame:
        """Issue the REST call and return Alpaca's raw MultiIndex frame."""
        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe.to_alpaca(),
            start=start,
            end=end,
            limit=limit,
            feed=self._feed,
            adjustment=self._adjustment,
        )
        description = f"get_stock_bars({','.join(symbols[:3])}{'...' if len(symbols) > 3 else ''})"
        barset = self._call(lambda: self._client.get_stock_bars(request), description)
        frame = barset.df
        if frame is None or frame.empty:
            return pd.DataFrame()
        return frame

    # -- MarketDataProvider ------------------------------------------------------

    def get_bars(
        self,
        symbol: str,
        timeframe: str | Timeframe,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        symbol = symbol.strip().upper()
        parsed = Timeframe.parse(timeframe)
        window_start, window_end = self._resolve_window(parsed, start, end, limit)

        if self._cache is not None:
            cached = self._cache.get(symbol, parsed, window_start, window_end)
            if cached is not None:
                logger.debug(
                    "Cache hit: %s %s %s..%s (%d bars)",
                    symbol, parsed.label, window_start.date(), window_end.date(), len(cached),
                )
                return cached.tail(limit) if limit else cached

        raw = self._request_bars([symbol], parsed, window_start, window_end, limit)
        frame = normalize_bars(raw, symbol=symbol)
        if frame.empty:
            logger.warning(
                "No bars returned for %s %s between %s and %s (feed=%s)",
                symbol, parsed.label, window_start, window_end, self._feed.value,
            )
            return frame

        if self._cache is not None:
            self._cache.put(symbol, parsed, frame)
        return frame.tail(limit) if limit else frame

    def get_bars_multi(
        self,
        symbols: Sequence[str],
        timeframe: str | Timeframe,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        cleaned = _unique_symbols(symbols)
        if not cleaned:
            return {}
        parsed = Timeframe.parse(timeframe)
        window_start, window_end = self._resolve_window(parsed, start, end, limit)

        results: dict[str, pd.DataFrame] = {}
        outstanding: list[str] = []

        if self._cache is not None:
            for symbol in cleaned:
                cached = self._cache.get(symbol, parsed, window_start, window_end)
                if cached is not None:
                    results[symbol] = cached
                else:
                    outstanding.append(symbol)
        else:
            outstanding = list(cleaned)

        for batch in _chunk(outstanding, MAX_SYMBOLS_PER_REQUEST):
            raw = self._request_bars(batch, parsed, window_start, window_end, limit)
            if raw.empty:
                continue
            available = (
                set(raw.index.get_level_values("symbol"))
                if isinstance(raw.index, pd.MultiIndex)
                else set(batch)
            )
            for symbol in batch:
                if symbol not in available:
                    continue
                frame = normalize_bars(raw, symbol=symbol)
                if frame.empty:
                    continue
                results[symbol] = frame
                if self._cache is not None:
                    self._cache.put(symbol, parsed, frame)

        if limit:
            results = {symbol: frame.tail(limit) for symbol, frame in results.items()}
        return results

    def get_latest_quote(self, symbol: str) -> Quote | None:
        symbol = symbol.strip().upper()
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=self._feed)
        try:
            response = self._call(
                lambda: self._client.get_stock_latest_quote(request), f"latest_quote({symbol})"
            )
        except Exception as error:  # noqa: BLE001 - quotes are advisory
            logger.warning("Could not fetch latest quote for %s: %s", symbol, error)
            return None
        quote = response.get(symbol) if isinstance(response, dict) else response
        if quote is None:
            return None
        return Quote(
            symbol=symbol,
            timestamp=ensure_utc(quote.timestamp),
            bid_price=float(quote.bid_price or 0.0),
            ask_price=float(quote.ask_price or 0.0),
            bid_size=float(quote.bid_size or 0.0),
            ask_size=float(quote.ask_size or 0.0),
        )


class StaticMarketData(MarketDataProvider):
    """Provider backed by in-memory frames — used by tests and offline replays."""

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self._frames = {
            symbol.upper(): normalize_bars(frame, symbol=symbol)
            for symbol, frame in frames.items()
        }

    def get_bars(
        self,
        symbol: str,
        timeframe: str | Timeframe,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        frame = self._frames.get(symbol.upper())
        if frame is None:
            return empty_bars()
        if start is not None:
            frame = frame[frame.index >= pd.Timestamp(ensure_utc(start))]
        if end is not None:
            frame = frame[frame.index <= pd.Timestamp(ensure_utc(end))]
        return frame.tail(limit) if limit else frame

    def get_bars_multi(
        self,
        symbols: Sequence[str],
        timeframe: str | Timeframe,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        output: dict[str, pd.DataFrame] = {}
        for symbol in _unique_symbols(symbols):
            frame = self.get_bars(symbol, timeframe, start=start, end=end, limit=limit)
            if not frame.empty:
                output[symbol] = frame
        return output


def _unique_symbols(symbols: Iterable[str]) -> list[str]:
    """Upper-case, de-duplicate and drop blanks while preserving order."""
    seen: dict[str, None] = {}
    for symbol in symbols:
        cleaned = str(symbol).strip().upper()
        if cleaned:
            seen.setdefault(cleaned, None)
    return list(seen)


def _chunk(items: Sequence[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(items), size):
        yield list(items[index : index + size])


def build_market_data(
    alpaca_settings: AlpacaSettings,
    data_settings: DataSettings,
    *,
    use_cache: bool | None = None,
) -> AlpacaMarketData:
    """Construct the default provider wired to the configured cache."""
    enabled = data_settings.cache_enabled if use_cache is None else use_cache
    cache = (
        BarCache(
            data_settings.cache_dir,
            feed=alpaca_settings.data_feed,
            adjustment=alpaca_settings.adjustment,
        )
        if enabled
        else None
    )
    return AlpacaMarketData(alpaca_settings, data_settings, cache=cache)
