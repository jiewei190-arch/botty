"""US equity market sessions, computed locally.

Why compute this rather than ask the broker
-------------------------------------------
The scanner needs to know what session it is in *constantly* — to label a
signal, to decide whether a volume comparison is meaningful, to know that a
quiet websocket at 03:00 is normal rather than broken. Asking Alpaca's
``/v2/clock`` for that would spend a rate-limited request on a question whose
answer is a calendar lookup, and would fail exactly when the network is the
thing that has gone wrong.

So sessions are computed from the NYSE's published rules. The rules are stable
and few:

* Regular trading is 09:30-16:00 America/New_York, Monday to Friday.
* Alpaca's extended hours run 04:00-09:30 (pre-market) and 16:00-20:00 (after
  hours). Those bounds are the data feed's, not the exchange's.
* Nine or ten fixed holidays a year, plus Good Friday, plus three early closes.

What this module is *not*
-------------------------
It is not authoritative. Unscheduled closures — a national day of mourning, a
hurricane, an exchange outage — are decided on the day and cannot be computed
from a calendar. :meth:`trading_bot.execution.broker.AlpacaBroker.get_clock`
asks the exchange and is right in those cases. Use this module for cheap,
offline, high-frequency questions; use the broker's clock before doing anything
that depends on the market genuinely being open.

DST is handled by :mod:`zoneinfo`: the session boundaries are wall-clock times
in New York, so a UTC-based comparison would be an hour wrong for half the year.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from functools import lru_cache
from zoneinfo import ZoneInfo

#: All session boundaries are wall-clock times at the exchange.
MARKET_TZ = ZoneInfo("America/New_York")

PREMARKET_OPEN = time(4, 0)
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
AFTERHOURS_CLOSE = time(20, 0)
#: Half-days close at 13:00 ET; Alpaca's after-hours window ends at 17:00 on those days.
EARLY_CLOSE = time(13, 0)
EARLY_AFTERHOURS_CLOSE = time(17, 0)

#: Juneteenth became a market holiday in 2022.
JUNETEENTH_FIRST_YEAR = 2022


class MarketSession(str, Enum):
    """Which trading session a moment falls in."""

    CLOSED = "closed"
    PREMARKET = "premarket"
    REGULAR = "regular"
    AFTERHOURS = "afterhours"

    @property
    def is_open(self) -> bool:
        """True only during regular hours — what "the market is open" means."""
        return self is MarketSession.REGULAR

    @property
    def is_extended(self) -> bool:
        return self in (MarketSession.PREMARKET, MarketSession.AFTERHOURS)

    @property
    def has_quotes(self) -> bool:
        """True when trades can print at all, extended hours included.

        Liquidity in extended hours is a fraction of regular hours, so a signal
        found here deserves more scepticism, not less.
        """
        return self is not MarketSession.CLOSED


def _easter(year: int) -> date:
    """Gregorian Easter Sunday (Meeus/Jones/Butcher algorithm).

    Needed only because Good Friday — the one market holiday with no fixed date
    — is the Friday before it.
    """
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month, day = divmod(h + m - 7 * n + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The ``n``-th ``weekday`` (0=Monday) of a month; ``n=-1`` means the last."""
    if n > 0:
        first = date(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + timedelta(days=offset + 7 * (n - 1))
    next_month = date(year, month + 1, 1) if month < 12 else date(year + 1, 1, 1)
    last_day = next_month - timedelta(days=1)
    offset = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=offset)


def _observed(day: date) -> date | None:
    """Apply the NYSE's weekend-observance rule to a fixed-date holiday.

    Saturday holidays move to the preceding Friday and Sunday holidays to the
    following Monday — except a Saturday New Year's Day, which the NYSE does not
    observe at all (31 December stays a full trading day). Returning ``None``
    expresses that "no closure this year".
    """
    if day.weekday() == 5:  # Saturday
        if day.month == 1 and day.day == 1:
            return None
        return day - timedelta(days=1)
    if day.weekday() == 6:  # Sunday
        return day + timedelta(days=1)
    return day


@lru_cache(maxsize=32)
def holidays_for_year(year: int) -> dict[date, str]:
    """Scheduled full-day closures for a calendar year, keyed by date."""
    fixed: list[tuple[date, str]] = [
        (date(year, 1, 1), "New Year's Day"),
        (date(year, 7, 4), "Independence Day"),
        (date(year, 12, 25), "Christmas Day"),
    ]
    if year >= JUNETEENTH_FIRST_YEAR:
        fixed.append((date(year, 6, 19), "Juneteenth National Independence Day"))

    holidays: dict[date, str] = {}
    for day, name in fixed:
        observed = _observed(day)
        if observed is not None:
            holidays[observed] = name

    holidays[_nth_weekday(year, 1, 0, 3)] = "Martin Luther King Jr. Day"
    holidays[_nth_weekday(year, 2, 0, 3)] = "Washington's Birthday"
    holidays[_easter(year) - timedelta(days=2)] = "Good Friday"
    holidays[_nth_weekday(year, 5, 0, -1)] = "Memorial Day"
    holidays[_nth_weekday(year, 9, 0, 1)] = "Labor Day"
    holidays[_nth_weekday(year, 11, 3, 4)] = "Thanksgiving Day"
    return holidays


@lru_cache(maxsize=32)
def early_closes_for_year(year: int) -> dict[date, str]:
    """Scheduled 13:00 ET closes for a calendar year.

    Three of them: the day before Independence Day, the day after Thanksgiving,
    and Christmas Eve. Each only counts when it is itself a trading day — an
    early close on a day the market is shut is not a thing.
    """
    holidays = holidays_for_year(year)
    candidates = [
        (date(year, 7, 3), "Day before Independence Day"),
        (_nth_weekday(year, 11, 3, 4) + timedelta(days=1), "Day after Thanksgiving"),
        (date(year, 12, 24), "Christmas Eve"),
    ]
    return {
        day: name
        for day, name in candidates
        if day.weekday() < 5 and day not in holidays
    }


def holiday_name(day: date) -> str | None:
    """The holiday closing the market on ``day``, or None."""
    return holidays_for_year(day.year).get(day)


def early_close_name(day: date) -> str | None:
    """The reason ``day`` closes early, or None."""
    return early_closes_for_year(day.year).get(day)


def is_trading_day(day: date) -> bool:
    """True when the exchange holds a session on ``day``."""
    return day.weekday() < 5 and holiday_name(day) is None


@dataclass(frozen=True, slots=True)
class SessionWindow:
    """The four boundaries of one trading day, as UTC instants."""

    day: date
    premarket_open: datetime
    regular_open: datetime
    regular_close: datetime
    afterhours_close: datetime
    early_close: bool = False
    note: str | None = None

    def session_at(self, moment: datetime) -> MarketSession:
        if moment < self.premarket_open or moment >= self.afterhours_close:
            return MarketSession.CLOSED
        if moment < self.regular_open:
            return MarketSession.PREMARKET
        if moment < self.regular_close:
            return MarketSession.REGULAR
        return MarketSession.AFTERHOURS


def _at(day: date, wall: time) -> datetime:
    """A wall-clock time on ``day`` in New York, converted to UTC."""
    return datetime.combine(day, wall, tzinfo=MARKET_TZ).astimezone(timezone.utc)


def session_window(day: date) -> SessionWindow | None:
    """The day's session boundaries, or None when the market does not open."""
    if not is_trading_day(day):
        return None
    early = early_close_name(day)
    close = EARLY_CLOSE if early else REGULAR_CLOSE
    after = EARLY_AFTERHOURS_CLOSE if early else AFTERHOURS_CLOSE
    return SessionWindow(
        day=day,
        premarket_open=_at(day, PREMARKET_OPEN),
        regular_open=_at(day, REGULAR_OPEN),
        regular_close=_at(day, close),
        afterhours_close=_at(day, after),
        early_close=bool(early),
        note=early,
    )


def _as_utc(moment: datetime | None) -> datetime:
    """Normalise an instant to UTC, treating a naive one as already UTC."""
    if moment is None:
        return datetime.now(timezone.utc)
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def current_window(moment: datetime | None = None) -> SessionWindow | None:
    """The session window ``moment`` falls inside, if any.

    A moment can belong to the *previous* New York day's window: 00:30 UTC is
    19:30 the evening before in New York, still inside after-hours. Both
    candidate days are checked rather than only the one the UTC date names.
    """
    instant = _as_utc(moment)
    local_day = instant.astimezone(MARKET_TZ).date()
    for candidate in (local_day, local_day - timedelta(days=1)):
        window = session_window(candidate)
        if window is not None and window.premarket_open <= instant < window.afterhours_close:
            return window
    return None


def session_at(moment: datetime | None = None) -> MarketSession:
    """Which session ``moment`` falls in (defaults to now)."""
    window = current_window(moment)
    if window is None:
        return MarketSession.CLOSED
    return window.session_at(_as_utc(moment))


def is_market_open(moment: datetime | None = None) -> bool:
    """True during regular hours only."""
    return session_at(moment).is_open


def next_session_window(
    moment: datetime | None = None, *, horizon_days: int = 10
) -> SessionWindow | None:
    """The next window whose regular session has not yet ended.

    ``horizon_days`` bounds the search so a pathological calendar cannot spin;
    ten days clears the longest run of consecutive closures the NYSE schedules.
    """
    instant = _as_utc(moment)
    start = instant.astimezone(MARKET_TZ).date()
    for offset in range(horizon_days + 1):
        window = session_window(start + timedelta(days=offset))
        if window is not None and window.regular_close > instant:
            return window
    return None


def next_open(moment: datetime | None = None) -> datetime | None:
    """When regular trading next begins (None if already inside it)."""
    instant = _as_utc(moment)
    window = next_session_window(instant)
    if window is None:
        return None
    return window.regular_open if window.regular_open > instant else None


def next_close(moment: datetime | None = None) -> datetime | None:
    """When regular trading next ends."""
    window = next_session_window(moment)
    return None if window is None else window.regular_close


def time_until_open(moment: datetime | None = None) -> timedelta | None:
    """How long until regular trading begins, or None if it already has."""
    instant = _as_utc(moment)
    opening = next_open(instant)
    return None if opening is None else opening - instant


def describe(moment: datetime | None = None) -> str:
    """One human-readable line about the market's state.

    Used in log banners and the scanner header, where "why is nothing
    happening?" is the question being answered.
    """
    instant = _as_utc(moment)
    local = instant.astimezone(MARKET_TZ)
    session = session_at(instant)
    stamp = local.strftime("%Y-%m-%d %H:%M %Z")

    if session is MarketSession.CLOSED:
        holiday = holiday_name(local.date())
        if holiday:
            reason = f"closed for {holiday}"
        elif local.date().weekday() >= 5:
            reason = "closed for the weekend"
        else:
            reason = "closed"
        opening = next_open(instant)
        if opening is None:
            return f"{stamp} — {reason}"
        local_open = opening.astimezone(MARKET_TZ)
        remaining = opening - instant
        hours, minutes = divmod(int(remaining.total_seconds()) // 60, 60)
        return (
            f"{stamp} — {reason}; opens {local_open.strftime('%a %H:%M %Z')} "
            f"(in {hours}h {minutes:02d}m)"
        )

    window = current_window(instant)
    suffix = ""
    if window is not None and window.early_close:
        local_close = window.regular_close.astimezone(MARKET_TZ)
        suffix = f" (early close {local_close.strftime('%H:%M %Z')} — {window.note})"
    return f"{stamp} — {session.value} session{suffix}"
