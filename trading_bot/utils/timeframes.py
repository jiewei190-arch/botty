"""Single source of truth for bar sizing.

Backtests and the live bot must agree exactly on what "15Min" means, otherwise a
strategy validated on one will misbehave on the other. Every component converts
timeframe strings through this module rather than parsing them ad hoc.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

_UNIT_ALIASES: dict[str, TimeFrameUnit] = {
    "min": TimeFrameUnit.Minute,
    "mins": TimeFrameUnit.Minute,
    "minute": TimeFrameUnit.Minute,
    "minutes": TimeFrameUnit.Minute,
    "m": TimeFrameUnit.Minute,
    "h": TimeFrameUnit.Hour,
    "hour": TimeFrameUnit.Hour,
    "hours": TimeFrameUnit.Hour,
    "d": TimeFrameUnit.Day,
    "day": TimeFrameUnit.Day,
    "days": TimeFrameUnit.Day,
    "w": TimeFrameUnit.Week,
    "week": TimeFrameUnit.Week,
    "weeks": TimeFrameUnit.Week,
    "mo": TimeFrameUnit.Month,
    "month": TimeFrameUnit.Month,
    "months": TimeFrameUnit.Month,
}

_UNIT_DURATION: dict[TimeFrameUnit, timedelta] = {
    TimeFrameUnit.Minute: timedelta(minutes=1),
    TimeFrameUnit.Hour: timedelta(hours=1),
    TimeFrameUnit.Day: timedelta(days=1),
    TimeFrameUnit.Week: timedelta(weeks=1),
    TimeFrameUnit.Month: timedelta(days=30),  # nominal; calendar months vary
}

_UNIT_CANONICAL: dict[TimeFrameUnit, str] = {
    TimeFrameUnit.Minute: "Min",
    TimeFrameUnit.Hour: "Hour",
    TimeFrameUnit.Day: "Day",
    TimeFrameUnit.Week: "Week",
    TimeFrameUnit.Month: "Month",
}

#: Approximate US-equity regular-session bars per trading day, used to translate a
#: "how many bars of history" request into a calendar date range.
_BARS_PER_TRADING_DAY: dict[TimeFrameUnit, float] = {
    TimeFrameUnit.Minute: 390.0,
    TimeFrameUnit.Hour: 7.0,
    TimeFrameUnit.Day: 1.0,
    TimeFrameUnit.Week: 0.2,
    TimeFrameUnit.Month: 1 / 21,
}

_PATTERN = re.compile(r"^\s*(\d*)\s*([A-Za-z]+)\s*$")

#: Timeframes offered in the CLI and dashboard dropdowns.
SUPPORTED_TIMEFRAMES: tuple[str, ...] = (
    "1Min",
    "5Min",
    "15Min",
    "30Min",
    "1Hour",
    "4Hour",
    "1Day",
    "1Week",
)


@dataclass(frozen=True, slots=True)
class Timeframe:
    """A parsed bar size.

    ``Timeframe.parse("15Min")`` yields ``amount=15, unit=Minute``.
    """

    amount: int
    unit: TimeFrameUnit

    @classmethod
    def parse(cls, value: str | Timeframe) -> Timeframe:
        """Parse a timeframe string such as ``5Min``, ``1h`` or ``day``."""
        if isinstance(value, Timeframe):
            return value
        match = _PATTERN.match(str(value))
        if not match:
            raise ValueError(
                f"Unrecognised timeframe {value!r}. Examples: {', '.join(SUPPORTED_TIMEFRAMES)}"
            )
        amount_text, unit_text = match.groups()
        amount = int(amount_text) if amount_text else 1
        if amount < 1:
            raise ValueError(f"Timeframe amount must be >= 1, got {amount}")
        unit = _UNIT_ALIASES.get(unit_text.lower())
        if unit is None:
            raise ValueError(
                f"Unrecognised timeframe unit {unit_text!r} in {value!r}. "
                f"Examples: {', '.join(SUPPORTED_TIMEFRAMES)}"
            )
        return cls(amount=amount, unit=unit)

    @property
    def label(self) -> str:
        """Canonical string form, e.g. ``15Min``. Used as a cache key."""
        return f"{self.amount}{_UNIT_CANONICAL[self.unit]}"

    @property
    def duration(self) -> timedelta:
        """Wall-clock length of one bar (nominal for month bars)."""
        return _UNIT_DURATION[self.unit] * self.amount

    @property
    def is_intraday(self) -> bool:
        return self.unit in (TimeFrameUnit.Minute, TimeFrameUnit.Hour)

    def to_alpaca(self) -> TimeFrame:
        """Convert to the Alpaca SDK representation."""
        return TimeFrame(self.amount, self.unit)

    def to_pandas_freq(self) -> str:
        """Pandas offset alias for resampling (``15min``, ``1h``, ``1D``...)."""
        mapping = {
            TimeFrameUnit.Minute: "min",
            TimeFrameUnit.Hour: "h",
            TimeFrameUnit.Day: "D",
            TimeFrameUnit.Week: "W",
            TimeFrameUnit.Month: "MS",
        }
        return f"{self.amount}{mapping[self.unit]}"

    @property
    def periods_per_year(self) -> float:
        """Bars in a trading year, for annualising risk-adjusted returns.

        Uses 252 trading days. A Sharpe ratio computed from 15-minute bars must
        be scaled by the square root of this, not by the square root of 252 —
        getting it wrong inflates the figure by roughly five times.
        """
        return _BARS_PER_TRADING_DAY[self.unit] / self.amount * 252

    def calendar_span_for_bars(self, bars: int) -> timedelta:
        """Calendar time needed to collect ``bars`` bars.

        Intraday bars only accrue during the ~6.5h regular session, so the naive
        ``bars * duration`` badly under-estimates the range to request. This
        converts through trading days and adds a 40% buffer for weekends and
        market holidays.
        """
        if bars < 1:
            raise ValueError(f"bars must be >= 1, got {bars}")
        bars_per_day = _BARS_PER_TRADING_DAY[self.unit] / self.amount
        trading_days = bars / max(bars_per_day, 1e-9)
        calendar_days = trading_days * 1.4 + 5
        return timedelta(days=calendar_days)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.label
