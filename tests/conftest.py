"""Shared fixtures.

Every test runs against synthetic bars and an in-memory database, so the suite
is fast, deterministic and requires no API credentials.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from trading_bot.config.settings import Settings
from trading_bot.data.database import Database


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Strip bot-related environment variables so a developer's real .env or
    exported keys cannot influence test outcomes."""
    for prefix in ("ALPACA_", "RISK_", "DATA_", "LOG_"):
        for key in list(dict(**__import__("os").environ)):
            if key.startswith(prefix):
                monkeypatch.delenv(key, raising=False)
    for key in ("TRADING_MODE", "ENABLE_LIVE_TRADING", "LIVE_TRADING_CONFIRMATION"):
        monkeypatch.delenv(key, raising=False)


#: Close-to-close standard deviation of a US large-cap trading session,
#: measured from real AAPL daily bars (1.9%). Per-bar sigma is scaled down from
#: this by the square root of the bar's share of a session.
DAILY_SIGMA = 0.019

#: Minutes in a regular US equity session, the denominator of that scaling.
MINUTES_PER_SESSION = 390


def make_bars(
    periods: int = 100,
    *,
    start: datetime | None = None,
    freq: str = "15min",
    seed: int = 7,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Generate a deterministic, internally consistent OHLCV frame.

    When ``start`` is omitted the series is anchored so that its last bar closes
    at the most recent completed period. That matches what a provider returns for
    a default "recent history" request; tests needing fixed timestamps pass an
    explicit ``start``. The price path stays deterministic either way, because it
    is driven by ``seed`` rather than by the dates.
    """
    rng = np.random.default_rng(seed)
    if start is None:
        end = pd.Timestamp.now(tz="UTC").floor(freq)
        begin = end - periods * pd.Timedelta(freq)
    else:
        begin = start
    index = pd.date_range(begin, periods=periods, freq=freq, tz="UTC", name="timestamp")

    # Per-bar volatility is derived from the bar's length rather than fixed.
    # Measured against real AAPL daily bars, a session's true range averages
    # about 2.5% of price and its close-to-close move about 1.9%; a fixed 0.4%
    # sigma happens to be right for a 15-minute bar and is roughly five times
    # too calm for a daily one. Using it for daily bars made ATR-derived stops
    # come out around 1% — narrow enough that noise would take them out the
    # same session — which reads as a strategy bug and is not one.
    minutes = max(pd.Timedelta(freq) / pd.Timedelta(minutes=1), 1e-9)
    sigma = DAILY_SIGMA * math.sqrt(min(minutes, MINUTES_PER_SESSION) / MINUTES_PER_SESSION)
    returns = rng.normal(sigma / 20, sigma, periods)
    close = start_price * np.exp(np.cumsum(returns))
    open_ = np.concatenate([[start_price], close[:-1]])
    # 0.35 calibrated so a synthetic daily bar's true range averages 2.54% of
    # price, against 2.48% measured on real AAPL sessions.
    spread = np.abs(rng.normal(0, sigma * 0.35, periods)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(50_000, 500_000, periods).astype(float)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "trade_count": rng.integers(100, 2000, periods).astype(float),
            "vwap": (high + low + close) / 3,
        },
        index=index,
    )


def make_alpaca_frame(symbols: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build the (symbol, timestamp) MultiIndex frame that ``BarSet.df`` returns."""
    parts = []
    for symbol, frame in symbols.items():
        copy = frame.copy()
        copy["symbol"] = symbol
        copy = copy.set_index("symbol", append=True).reorder_levels(["symbol", "timestamp"])
        parts.append(copy)
    return pd.concat(parts).sort_index()


@pytest.fixture
def bars() -> pd.DataFrame:
    return make_bars()


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings pointed at a temporary directory with dummy credentials."""
    base = Settings()
    return base.with_overrides(
        alpaca=base.alpaca.model_copy(update={"api_key": "test-key", "secret_key": "test-secret"}),
        data=base.data.model_copy(
            update={
                "cache_dir": tmp_path / "cache",
                "database_path": tmp_path / "test.db",
                "watchlist": ["AAPL", "MSFT"],
            }
        ),
        logging=base.logging.model_copy(update={"directory": tmp_path / "logs"}),
    )


@pytest.fixture
def database() -> Database:
    db = Database(":memory:")
    db.initialize()
    yield db
    db.close()
