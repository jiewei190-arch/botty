"""Session-aware market context.

The problem this solves
-----------------------
Four of the five detectors need to know where the trading day starts and stops.
"Volume so far today" needs today; "yesterday's high" needs yesterday; "the gap"
needs the boundary between them; "the pre-market high" needs a sub-part of one.
A raw bar frame has none of that — it is a flat sequence of timestamps.

Doing that grouping inside each detector would mean five copies of the same
fiddly calendar logic, five chances to disagree about which bar belongs to which
day, and five places to get an early close wrong. So it is done once, here, and
the detectors receive the answer.

Daily bars are handled by the same object. When the frame's bars are a day apart
each bar *is* a session, there is no pre-market, and every detector still works —
which is what lets the same code serve a swing trader on daily bars and an
intraday scanner on 5-minute bars.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from trading_bot.utils.market_hours import (
    MARKET_TZ,
    PREMARKET_OPEN,
    REGULAR_CLOSE,
    REGULAR_OPEN,
    MarketSession,
    session_at,
    session_window,
)

logger = logging.getLogger(__name__)

#: Bars spaced at least this far apart are treated as daily or coarser.
DAILY_SPACING = pd.Timedelta(hours=20)

#: Sessions kept for baselines. Twenty sessions is roughly a trading month —
#: long enough for an average to mean something, short enough that a regime from
#: six months ago does not define "normal" today.
DEFAULT_HISTORY_SESSIONS = 20


@dataclass(frozen=True, slots=True)
class Extent:
    """Open/high/low/close/volume over a contiguous slice of one day."""

    open: float
    high: float
    low: float
    close: float
    volume: float
    bars: int
    first_ts: datetime
    last_ts: datetime

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def range_pct(self) -> float:
        """Range as a percentage of the closing price (0 if the price is not positive)."""
        return (self.range / self.close * 100.0) if self.close > 0 else 0.0

    @property
    def change_pct(self) -> float:
        return ((self.close - self.open) / self.open * 100.0) if self.open > 0 else 0.0


@dataclass(frozen=True, slots=True)
class SessionStats:
    """One trading day, split into the parts detectors ask about."""

    day: date
    #: Regular hours only (09:30-16:00 ET, or to 13:00 on a half day).
    regular: Extent | None
    #: Pre-market only (04:00-09:30 ET). Always None for daily bars.
    premarket: Extent | None
    #: Every bar of the day, whatever session it fell in.
    full: Extent
    #: True once the day's regular close has passed.
    complete: bool
    early_close: bool = False

    @property
    def close(self) -> float:
        """The reference close: regular-hours if there was one, else the day's last trade."""
        return self.regular.close if self.regular else self.full.close

    @property
    def high(self) -> float:
        return self.regular.high if self.regular else self.full.high

    @property
    def low(self) -> float:
        return self.regular.low if self.regular else self.full.low

    @property
    def open(self) -> float:
        return self.regular.open if self.regular else self.full.open

    @property
    def volume(self) -> float:
        return self.regular.volume if self.regular else self.full.volume


def _extent(frame: pd.DataFrame) -> Extent | None:
    """Summarise a slice of bars, or None when the slice is empty."""
    if frame.empty:
        return None
    highs = frame["high"].to_numpy(dtype="float64")
    lows = frame["low"].to_numpy(dtype="float64")
    return Extent(
        open=float(frame["open"].iloc[0]),
        high=float(np.nanmax(highs)),
        low=float(np.nanmin(lows)),
        close=float(frame["close"].iloc[-1]),
        volume=float(frame["volume"].sum()),
        bars=int(len(frame)),
        first_ts=frame.index[0].to_pydatetime(),
        last_ts=frame.index[-1].to_pydatetime(),
    )


def _classify_intraday(index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """Map each bar to (session date, :class:`MarketSession`).

    The session date is the New York calendar date. That holds for every feed
    the free plan serves, whose bars all fall inside 04:00-20:00 ET on a single
    day; an overnight-session bar outside that window is labelled CLOSED and
    excluded from the day's regular and pre-market extents rather than silently
    inflating them.

    Early closes are applied per day rather than per bar — there are a handful of
    distinct days in any frame and thousands of bars.
    """
    local = index.tz_convert(MARKET_TZ)
    days = pd.Series(local.date, index=index, name="session_date")
    minutes = pd.Series(local.hour * 60 + local.minute, index=index, dtype="int32")

    pre_open = PREMARKET_OPEN.hour * 60 + PREMARKET_OPEN.minute
    reg_open = REGULAR_OPEN.hour * 60 + REGULAR_OPEN.minute
    default_close = REGULAR_CLOSE.hour * 60 + REGULAR_CLOSE.minute

    closes = pd.Series(default_close, index=index, dtype="int32")
    for day in pd.unique(days):
        window = session_window(day)
        if window is None or not window.early_close:
            continue
        local_close = window.regular_close.astimezone(MARKET_TZ)
        closes[days == day] = local_close.hour * 60 + local_close.minute

    sessions = pd.Series(MarketSession.CLOSED, index=index, dtype=object)
    sessions[(minutes >= pre_open) & (minutes < reg_open)] = MarketSession.PREMARKET
    sessions[(minutes >= reg_open) & (minutes < closes)] = MarketSession.REGULAR
    after = (minutes >= closes) & (minutes < 20 * 60)
    sessions[after] = MarketSession.AFTERHOURS
    return days, sessions


@dataclass(frozen=True, slots=True)
class DetectionContext:
    """Everything the detectors need about one symbol, computed once."""

    symbol: str
    bars: pd.DataFrame
    now: datetime
    session: MarketSession
    intraday: bool
    #: Oldest first. The last entry is the session in progress (or the most
    #: recent completed one when the market is shut).
    sessions: tuple[SessionStats, ...]
    #: Session date per bar, index-aligned with ``bars``.
    session_dates: pd.Series
    #: Which session each bar fell in, index-aligned with ``bars``.
    bar_sessions: pd.Series

    @property
    def today(self) -> SessionStats | None:
        """The session in progress, or the most recent one when the market is shut."""
        return self.sessions[-1] if self.sessions else None

    @property
    def previous(self) -> SessionStats | None:
        """The completed session before :attr:`today`."""
        return self.sessions[-2] if len(self.sessions) >= 2 else None

    @property
    def history(self) -> tuple[SessionStats, ...]:
        """Completed sessions before today, oldest first."""
        return self.sessions[:-1] if len(self.sessions) >= 1 else ()

    @property
    def last_price(self) -> float | None:
        if self.bars.empty:
            return None
        return float(self.bars["close"].iloc[-1])

    @property
    def last_timestamp(self) -> datetime | None:
        if self.bars.empty:
            return None
        return self.bars.index[-1].to_pydatetime()

    @property
    def previous_close(self) -> float | None:
        """Yesterday's regular-hours close — the reference every gap is measured from."""
        return self.previous.close if self.previous else None

    def regular_bars(self, day: date) -> pd.DataFrame:
        """The regular-hours bars of one session."""
        if not self.intraday:
            return self.bars[self.session_dates == day]
        mask = (self.session_dates == day) & (self.bar_sessions == MarketSession.REGULAR)
        return self.bars[mask]

    def session_volume_curve(self, day: date) -> np.ndarray:
        """Cumulative regular-hours volume through one session.

        Element ``k`` is the volume traded through the first ``k + 1`` bars of the
        day. Comparing today's element ``k`` with prior days' element ``k`` is
        what makes a relative-volume reading meaningful before the closing bell:
        against a full-day average, every morning looks quiet.
        """
        volumes = self.regular_bars(day)["volume"].to_numpy(dtype="float64")
        if volumes.size == 0:
            return np.zeros(0, dtype="float64")
        return np.nancumsum(np.nan_to_num(volumes, nan=0.0))


def build_context(
    symbol: str,
    bars: pd.DataFrame,
    *,
    now: datetime | None = None,
    history_sessions: int = DEFAULT_HISTORY_SESSIONS,
) -> DetectionContext:
    """Group ``bars`` into sessions and summarise each one.

    ``bars`` is expected in the layout the data layer produces: a UTC
    :class:`~pandas.DatetimeIndex`, ascending, with open/high/low/close/volume
    columns. An empty frame produces an empty context rather than an error —
    every detector treats "no data" as "no signal", which is the behaviour you
    want when one symbol of thirty is missing.
    """
    ticker = str(symbol).strip().upper()
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    moment = moment.astimezone(timezone.utc)

    if bars is None or bars.empty:
        empty = pd.Series(dtype=object)
        return DetectionContext(
            symbol=ticker,
            bars=bars if bars is not None else pd.DataFrame(),
            now=moment,
            session=session_at(moment),
            intraday=True,
            sessions=(),
            session_dates=empty,
            bar_sessions=empty,
        )

    frame = bars
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{ticker}: bars must be indexed by timestamp")
    if frame.index.tz is None:
        frame = frame.tz_localize(timezone.utc)
    elif str(frame.index.tz) != "UTC":
        frame = frame.tz_convert(timezone.utc)
    if not frame.index.is_monotonic_increasing:
        frame = frame.sort_index()

    spacing = pd.Timedelta(0)
    if len(frame.index) > 1:
        deltas = frame.index.to_series().diff().dropna()
        if not deltas.empty:
            spacing = deltas.median()
    intraday = spacing < DAILY_SPACING if spacing > pd.Timedelta(0) else True

    if intraday:
        session_dates, bar_sessions = _classify_intraday(frame.index)
    else:
        # One bar per session: the bar's New York date names the day, and the
        # whole bar is the regular session. Daily bars from Alpaca aggregate
        # regular hours only, so labelling them REGULAR is accurate.
        session_dates = pd.Series(
            frame.index.tz_convert(MARKET_TZ).date, index=frame.index, name="session_date"
        )
        bar_sessions = pd.Series(MarketSession.REGULAR, index=frame.index, dtype=object)

    stats: list[SessionStats] = []
    unique_days = list(pd.unique(session_dates))
    # Keep one more session than the baseline needs: the extra is today, which
    # is the thing being measured rather than part of what it is measured against.
    for day in unique_days[-(history_sessions + 1) :]:
        day_mask = session_dates == day
        day_frame = frame[day_mask]
        day_sessions = bar_sessions[day_mask]
        full = _extent(day_frame)
        if full is None:
            continue
        window = session_window(day)
        stats.append(
            SessionStats(
                day=day,
                regular=_extent(day_frame[day_sessions == MarketSession.REGULAR]),
                premarket=(
                    _extent(day_frame[day_sessions == MarketSession.PREMARKET])
                    if intraday
                    else None
                ),
                full=full,
                complete=bool(window is not None and moment >= window.regular_close),
                early_close=bool(window is not None and window.early_close),
            )
        )

    return DetectionContext(
        symbol=ticker,
        bars=frame,
        now=moment,
        session=session_at(moment),
        intraday=intraday,
        sessions=tuple(stats),
        session_dates=session_dates,
        bar_sessions=bar_sessions,
    )
