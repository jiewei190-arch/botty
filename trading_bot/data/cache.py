"""On-disk cache for historical bars.

Backtests are re-run constantly while a strategy is tuned. Re-downloading the
same history each time is slow and burns rate limit, so completed bars are
persisted as parquet. Bars for a closed period never change (modulo corporate
actions, which is why the adjustment mode is part of the cache key), making them
safe to cache indefinitely.

Cache key: ``{symbol}_{timeframe}_{feed}_{adjustment}.parquet``.

A read is served only when the cached range fully covers the requested range;
otherwise the caller refetches and the merged result is written back.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from trading_bot.data.models import BAR_COLUMNS
from trading_bot.utils.timeframes import Timeframe

logger = logging.getLogger(__name__)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")


class BarCache:
    """Parquet-backed bar cache."""

    def __init__(
        self,
        directory: Path | str,
        *,
        feed: str = "iex",
        adjustment: str = "all",
        max_age: timedelta | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.feed = feed
        self.adjustment = adjustment
        #: Recent intraday data can be revised by the vendor; ignore cache files
        #: older than this when set.
        self.max_age = max_age

    def path_for(self, symbol: str, timeframe: Timeframe) -> Path:
        name = f"{symbol.upper()}_{timeframe.label}_{self.feed}_{self.adjustment}.parquet"
        return self.directory / _SAFE_NAME.sub("_", name)

    def get(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame | None:
        """Return cached bars covering ``[start, end]``, or None on a miss."""
        path = self.path_for(symbol, timeframe)
        if not path.exists():
            return None
        if self.max_age is not None:
            age = datetime.now(timezone.utc) - datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            )
            if age > self.max_age:
                logger.debug("Cache expired for %s %s (age %s)", symbol, timeframe.label, age)
                return None

        frame = self._read(path)
        if frame is None or frame.empty:
            return None

        requested_start = pd.Timestamp(start).tz_convert("UTC")
        requested_end = pd.Timestamp(end).tz_convert("UTC")
        cached_start, cached_end = frame.index[0], frame.index[-1]

        # A miss unless the cached window covers what was asked for. One bar of
        # slack at the start absorbs the fact that the first bar of a range rarely
        # lands exactly on the requested boundary.
        if cached_start > requested_start + timeframe.duration:
            return None
        if cached_end < requested_end - timeframe.duration:
            return None

        window = frame[(frame.index >= requested_start) & (frame.index <= requested_end)]
        return window if not window.empty else None

    def put(self, symbol: str, timeframe: Timeframe, frame: pd.DataFrame) -> None:
        """Merge ``frame`` into the cached history for the symbol."""
        if frame is None or frame.empty:
            return
        path = self.path_for(symbol, timeframe)
        existing = self._read(path)
        if existing is not None and not existing.empty:
            combined = pd.concat([existing, frame])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        else:
            combined = frame.sort_index()

        try:
            columns = [column for column in BAR_COLUMNS if column in combined.columns]
            combined[columns].to_parquet(path, index=True)
            logger.debug("Cached %d bars for %s %s", len(combined), symbol, timeframe.label)
        except Exception as error:  # noqa: BLE001 - cache failures must not break trading
            logger.warning("Could not write cache file %s: %s", path, error)

    def _read(self, path: Path) -> pd.DataFrame | None:
        try:
            frame = pd.read_parquet(path)
        except Exception as error:  # noqa: BLE001 - a corrupt cache is recoverable
            logger.warning("Ignoring unreadable cache file %s: %s", path, error)
            return None
        if frame.empty:
            return None
        index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
        index.name = "timestamp"
        frame.index = index
        return frame.sort_index()

    def clear(self, symbol: str | None = None) -> int:
        """Delete cache files (all, or just one symbol). Returns files removed."""
        pattern = f"{symbol.upper()}_*.parquet" if symbol else "*.parquet"
        removed = 0
        for path in self.directory.glob(pattern):
            path.unlink()
            removed += 1
        logger.info("Cleared %d cache file(s) from %s", removed, self.directory)
        return removed

    def stats(self) -> dict[str, object]:
        """Summary of cache contents for the CLI and dashboard."""
        files = sorted(self.directory.glob("*.parquet"))
        total_bytes = sum(path.stat().st_size for path in files)
        return {
            "directory": str(self.directory),
            "files": len(files),
            "size_mb": round(total_bytes / (1024 * 1024), 2),
            "symbols": sorted({path.name.split("_")[0] for path in files}),
        }
