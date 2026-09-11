"""Market session calculations.

The holiday dates are checked against the NYSE's published calendars rather than
against the code that produced them — a test that recomputes the same algorithm
proves only that the algorithm is deterministic.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from trading_bot.utils import market_hours as mh


def utc(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


class TestHolidays:
    @pytest.mark.parametrize(
        "day,name",
        [
            # NYSE published calendar, 2025.
            (date(2025, 1, 1), "New Year's Day"),
            (date(2025, 1, 20), "Martin Luther King Jr. Day"),
            (date(2025, 2, 17), "Washington's Birthday"),
            (date(2025, 4, 18), "Good Friday"),
            (date(2025, 5, 26), "Memorial Day"),
            (date(2025, 6, 19), "Juneteenth National Independence Day"),
            (date(2025, 7, 4), "Independence Day"),
            (date(2025, 9, 1), "Labor Day"),
            (date(2025, 11, 27), "Thanksgiving Day"),
            (date(2025, 12, 25), "Christmas Day"),
        ],
    )
    def test_2025_calendar(self, day, name):
        assert mh.holiday_name(day) == name

    @pytest.mark.parametrize(
        "day,name",
        [
            (date(2026, 1, 1), "New Year's Day"),
            (date(2026, 4, 3), "Good Friday"),
            # 4 July 2026 is a Saturday, so the closure moves to Friday the 3rd.
            (date(2026, 7, 3), "Independence Day"),
            (date(2026, 11, 26), "Thanksgiving Day"),
        ],
    )
    def test_2026_calendar(self, day, name):
        assert mh.holiday_name(day) == name

    def test_saturday_new_year_is_not_observed(self):
        """The NYSE does not close on 31 December for a Saturday New Year."""
        assert date(2022, 1, 1).weekday() == 5
        assert mh.holiday_name(date(2021, 12, 31)) is None
        assert mh.is_trading_day(date(2021, 12, 31))

    def test_sunday_holiday_moves_to_monday(self):
        assert date(2021, 7, 4).weekday() == 6
        assert mh.holiday_name(date(2021, 7, 5)) == "Independence Day"

    def test_juneteenth_only_from_2022(self):
        assert mh.holiday_name(date(2021, 6, 18)) is None
        assert mh.holiday_name(date(2022, 6, 20)) == "Juneteenth National Independence Day"

    def test_early_closes(self):
        assert mh.early_close_name(date(2025, 7, 3)) == "Day before Independence Day"
        assert mh.early_close_name(date(2025, 11, 28)) == "Day after Thanksgiving"
        assert mh.early_close_name(date(2025, 12, 24)) == "Christmas Eve"

    def test_early_close_dropped_when_it_is_itself_a_holiday(self):
        """3 July 2026 is the observed Independence Day, so it is shut, not short."""
        assert mh.holiday_name(date(2026, 7, 3)) == "Independence Day"
        assert mh.early_close_name(date(2026, 7, 3)) is None


class TestSessions:
    @pytest.mark.parametrize(
        "moment,expected",
        [
            ("2026-09-08T08:00:00", mh.MarketSession.PREMARKET),   # 04:00 ET
            ("2026-09-08T13:29:00", mh.MarketSession.PREMARKET),
            ("2026-09-08T13:30:00", mh.MarketSession.REGULAR),     # 09:30 ET
            ("2026-09-08T19:59:00", mh.MarketSession.REGULAR),
            ("2026-09-08T20:00:00", mh.MarketSession.AFTERHOURS),  # 16:00 ET
            ("2026-09-08T23:59:00", mh.MarketSession.AFTERHOURS),
            ("2026-09-09T00:00:00", mh.MarketSession.CLOSED),      # 20:00 ET
            ("2026-09-08T07:59:00", mh.MarketSession.CLOSED),
        ],
    )
    def test_session_boundaries(self, moment, expected):
        assert mh.session_at(utc(moment)) is expected

    def test_weekend_and_holiday_are_closed(self):
        assert mh.session_at(utc("2026-09-05T15:00:00")) is mh.MarketSession.CLOSED
        assert mh.session_at(utc("2026-09-07T15:00:00")) is mh.MarketSession.CLOSED

    def test_early_close_shortens_the_regular_session(self):
        window = mh.session_window(date(2025, 11, 28))
        assert window is not None and window.early_close
        # 13:00 ET is 18:00 UTC in November, so 12:30 ET still trades and
        # 13:30 ET does not.
        assert mh.session_at(utc("2025-11-28T17:30:00")) is mh.MarketSession.REGULAR
        assert mh.session_at(utc("2025-11-28T18:00:00")) is mh.MarketSession.AFTERHOURS
        assert mh.session_at(utc("2025-11-28T18:30:00")) is mh.MarketSession.AFTERHOURS

    def test_dst_boundary_uses_wall_clock_not_utc(self):
        """09:30 New York is 13:30 UTC in summer and 14:30 UTC in winter."""
        assert mh.session_at(utc("2026-01-05T14:30:00")) is mh.MarketSession.REGULAR
        assert mh.session_at(utc("2026-01-05T13:30:00")) is mh.MarketSession.PREMARKET

    def test_evening_belongs_to_the_previous_new_york_day(self):
        window = mh.current_window(utc("2026-09-08T23:30:00"))
        assert window is not None
        assert window.day == date(2026, 9, 8)

    def test_next_open_skips_the_weekend_and_labor_day(self):
        opening = mh.next_open(utc("2026-09-04T21:00:00"))
        assert opening == utc("2026-09-08T13:30:00")

    def test_next_open_is_none_inside_the_session(self):
        assert mh.next_open(utc("2026-09-08T15:00:00")) is None
        assert mh.is_market_open(utc("2026-09-08T15:00:00"))

    def test_time_until_open(self):
        remaining = mh.time_until_open(utc("2026-09-08T12:30:00"))
        assert remaining is not None
        assert remaining.total_seconds() == 3600


class TestDescribe:
    def test_names_the_holiday(self):
        text = mh.describe(utc("2026-12-25T15:00:00"))
        assert "Christmas Day" in text and "closed" in text

    def test_names_the_session(self):
        assert "regular session" in mh.describe(utc("2026-09-08T15:00:00"))
        assert "premarket session" in mh.describe(utc("2026-09-08T12:00:00"))

    def test_flags_an_early_close(self):
        assert "early close" in mh.describe(utc("2025-11-28T16:00:00"))

    def test_naive_timestamps_are_treated_as_utc(self):
        naive = datetime(2026, 9, 8, 15, 0)
        assert mh.session_at(naive) is mh.MarketSession.REGULAR
