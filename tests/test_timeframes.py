"""Timeframe parsing and calendar arithmetic."""

from __future__ import annotations

from datetime import timedelta

import pytest
from alpaca.data.timeframe import TimeFrameUnit

from trading_bot.utils.timeframes import SUPPORTED_TIMEFRAMES, Timeframe


@pytest.mark.parametrize(
    ("text", "amount", "unit"),
    [
        ("15Min", 15, TimeFrameUnit.Minute),
        ("5m", 5, TimeFrameUnit.Minute),
        ("1Hour", 1, TimeFrameUnit.Hour),
        ("4h", 4, TimeFrameUnit.Hour),
        ("1Day", 1, TimeFrameUnit.Day),
        ("day", 1, TimeFrameUnit.Day),
        ("1Week", 1, TimeFrameUnit.Week),
        ("  30 minutes ", 30, TimeFrameUnit.Minute),
    ],
)
def test_parse_accepts_common_spellings(text, amount, unit):
    parsed = Timeframe.parse(text)
    assert parsed.amount == amount
    assert parsed.unit is unit


def test_parse_is_idempotent():
    parsed = Timeframe.parse("15Min")
    assert Timeframe.parse(parsed) is parsed


@pytest.mark.parametrize("bad", ["", "Min15x", "0Min", "7Fortnights", "abc"])
def test_parse_rejects_nonsense(bad):
    with pytest.raises(ValueError):
        Timeframe.parse(bad)


def test_labels_round_trip():
    for label in SUPPORTED_TIMEFRAMES:
        assert Timeframe.parse(label).label == label


def test_duration_matches_label():
    assert Timeframe.parse("15Min").duration == timedelta(minutes=15)
    assert Timeframe.parse("4Hour").duration == timedelta(hours=4)
    assert Timeframe.parse("1Day").duration == timedelta(days=1)


def test_intraday_classification():
    assert Timeframe.parse("5Min").is_intraday
    assert Timeframe.parse("1Hour").is_intraday
    assert not Timeframe.parse("1Day").is_intraday


def test_pandas_freq_aliases_are_current():
    # pandas 2.2+ deprecated "T"/"H"; make sure we emit the modern aliases.
    assert Timeframe.parse("15Min").to_pandas_freq() == "15min"
    assert Timeframe.parse("1Hour").to_pandas_freq() == "1h"
    assert Timeframe.parse("1Day").to_pandas_freq() == "1D"


def test_alpaca_conversion():
    alpaca = Timeframe.parse("15Min").to_alpaca()
    assert alpaca.amount_value == 15
    assert alpaca.unit_value is TimeFrameUnit.Minute


def test_intraday_span_accounts_for_closed_hours():
    """300 15-minute bars need far more than 300*15 minutes of calendar time."""
    span = Timeframe.parse("15Min").calendar_span_for_bars(300)
    assert span > timedelta(days=10)
    assert span < timedelta(days=40)


def test_daily_span_covers_weekends_and_holidays():
    span = Timeframe.parse("1Day").calendar_span_for_bars(252)
    assert span > timedelta(days=252)


def test_span_grows_with_bar_count():
    timeframe = Timeframe.parse("1Day")
    assert timeframe.calendar_span_for_bars(500) > timeframe.calendar_span_for_bars(100)


def test_span_requires_positive_bars():
    with pytest.raises(ValueError):
        Timeframe.parse("1Day").calendar_span_for_bars(0)
