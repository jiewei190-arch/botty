"""Volume analysis.

Volume is the confirmation layer. A breakout on thin volume is a very different
proposition from the same breakout on three times the usual participation, and
the Phase 3 strategies will lean on that distinction.

Everything here is expressed as *relative* volume — a multiple of the symbol's
own moving average — because raw share counts cannot be compared across symbols
or across timeframes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import numpy as np
import pandas as pd

from trading_bot.indicators.technical_indicators import (
    DEFAULT_CONFIG,
    RELATIVE_VOLUME_COL,
    IndicatorConfig,
    calculate_relative_volume,
    calculate_volume_sma,
    validate_ohlcv,
    volume_sma_column,
)

logger = logging.getLogger(__name__)

VolumeTrend = Literal["RISING", "FALLING", "STEADY", "UNKNOWN"]


class VolumeCondition(str, Enum):
    """Participation relative to the symbol's own average."""

    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    SPIKE = "SPIKE"

    @property
    def is_elevated(self) -> bool:
        """True for HIGH and SPIKE — enough participation to confirm a move."""
        return self in (VolumeCondition.HIGH, VolumeCondition.SPIKE)


@dataclass(frozen=True, slots=True)
class VolumeAnalysis:
    """Result of :func:`analyze_volume`."""

    condition: VolumeCondition
    #: Current volume as a multiple of its moving average. None during warm-up.
    relative_volume: float | None
    current_volume: float
    average_volume: float | None
    trend: VolumeTrend
    #: True when the latest bar's volume supports its price direction.
    confirms_price: bool
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialisable summary.

        Example
        -------
        >>> analyze_volume(enriched).as_dict()
        {'condition': 'HIGH', 'relative_volume': 1.42, ...}
        """
        return {
            "condition": self.condition.value,
            "relative_volume": (
                round(self.relative_volume, 3) if self.relative_volume is not None else None
            ),
            "current_volume": self.current_volume,
            "average_volume": self.average_volume,
            "trend": self.trend,
            "confirms_price": self.confirms_price,
            "reasons": list(self.reasons),
        }


def _relative_volume_series(
    data: pd.DataFrame, config: IndicatorConfig
) -> pd.Series:
    """Reuse the precomputed relative-volume column when available."""
    if RELATIVE_VOLUME_COL in data.columns:
        return data[RELATIVE_VOLUME_COL]
    return calculate_relative_volume(data["volume"], config.volume_sma_period)


def classify_relative_volume(
    relative_volume: float | None,
    config: IndicatorConfig | None = None,
) -> VolumeCondition:
    """Bucket a relative-volume multiple into a condition.

    Thresholds come from ``config`` (defaults: below 0.7 is LOW, above 1.5 HIGH,
    above 2.5 SPIKE), so a strategy trading illiquid names can loosen them.

    Parameters
    ----------
    relative_volume:
        Current volume divided by its average. ``None`` during warm-up.
    config:
        Supplies the thresholds.

    Returns
    -------
    VolumeCondition
        ``NORMAL`` when the value is unknown — the non-committal answer.

    Example
    -------
    >>> classify_relative_volume(2.8)
    <VolumeCondition.SPIKE: 'SPIKE'>
    """
    settings = config or DEFAULT_CONFIG
    if relative_volume is None or pd.isna(relative_volume):
        return VolumeCondition.NORMAL
    if relative_volume >= settings.volume_spike_threshold:
        return VolumeCondition.SPIKE
    if relative_volume >= settings.volume_high_threshold:
        return VolumeCondition.HIGH
    if relative_volume < settings.volume_low_threshold:
        return VolumeCondition.LOW
    return VolumeCondition.NORMAL


def detect_volume_condition(
    data: pd.DataFrame,
    config: IndicatorConfig | None = None,
) -> VolumeCondition:
    """Classify volume on the most recent bar.

    Returns
    -------
    VolumeCondition
        One of ``LOW``, ``NORMAL``, ``HIGH``, ``SPIKE``.

    Example
    -------
    >>> detect_volume_condition(enriched)
    <VolumeCondition.HIGH: 'HIGH'>
    """
    settings = config or DEFAULT_CONFIG
    series = _relative_volume_series(data, settings)
    if series.empty:
        return VolumeCondition.NORMAL
    value = series.iloc[-1]
    return classify_relative_volume(
        None if pd.isna(value) else float(value), settings
    )


def detect_volume_trend(
    data: pd.DataFrame,
    config: IndicatorConfig | None = None,
    *,
    short_window: int = 5,
    tolerance: float = 0.1,
) -> VolumeTrend:
    """Report whether participation is building or drying up.

    Compares the mean volume of the last ``short_window`` bars against the longer
    moving average. ``tolerance`` is the fractional band around parity that still
    counts as ``STEADY``, so ordinary noise is not reported as a trend.

    Example
    -------
    >>> detect_volume_trend(enriched)
    'RISING'
    """
    settings = config or DEFAULT_CONFIG
    volume = data["volume"]
    if len(volume) < short_window:
        return "UNKNOWN"

    column = volume_sma_column(settings.volume_sma_period)
    average = (
        data[column]
        if column in data.columns
        else calculate_volume_sma(volume, settings.volume_sma_period)
    )
    if average.dropna().empty:
        return "UNKNOWN"

    recent = float(volume.tail(short_window).mean())
    baseline = float(average.dropna().iloc[-1])
    if baseline <= 0:
        return "UNKNOWN"

    ratio = recent / baseline
    if ratio > 1 + tolerance:
        return "RISING"
    if ratio < 1 - tolerance:
        return "FALLING"
    return "STEADY"


def volume_confirms_price(
    data: pd.DataFrame,
    config: IndicatorConfig | None = None,
) -> bool:
    """True when the latest bar's volume backs up its price move.

    Confirmation means the bar closed away from its open on at least average
    volume. A move on below-average volume is not confirmed — that is the whole
    point of checking.

    Example
    -------
    >>> volume_confirms_price(enriched)
    True
    """
    settings = config or DEFAULT_CONFIG
    if data.empty:
        return False

    series = _relative_volume_series(data, settings)
    if series.empty or pd.isna(series.iloc[-1]):
        return False

    row = data.iloc[-1]
    moved = float(row["close"]) != float(row["open"])
    return bool(moved and float(series.iloc[-1]) >= 1.0)


def analyze_volume(
    data: pd.DataFrame,
    config: IndicatorConfig | None = None,
) -> VolumeAnalysis:
    """Full volume picture for the most recent bar.

    Parameters
    ----------
    data:
        OHLCV frame, ideally already enriched by
        :func:`~trading_bot.indicators.technical_indicators.calculate_all_indicators`.
    config:
        Supplies the volume period and thresholds.

    Returns
    -------
    VolumeAnalysis
        Condition, relative volume, participation trend and whether volume
        confirms the latest price move.

    Raises
    ------
    InvalidDataError
        The frame is empty or malformed.

    Example
    -------
    >>> analysis = analyze_volume(enriched)
    >>> analysis.condition, analysis.relative_volume
    (<VolumeCondition.HIGH: 'HIGH'>, 1.42)
    """
    settings = config or DEFAULT_CONFIG
    validate_ohlcv(data, name="data")

    series = _relative_volume_series(data, settings)
    latest = series.iloc[-1] if len(series) else np.nan
    relative = None if pd.isna(latest) else float(latest)

    column = volume_sma_column(settings.volume_sma_period)
    average_series = (
        data[column]
        if column in data.columns
        else calculate_volume_sma(data["volume"], settings.volume_sma_period)
    )
    average_value = average_series.iloc[-1] if len(average_series) else np.nan
    average = None if pd.isna(average_value) else float(average_value)

    condition = classify_relative_volume(relative, settings)
    trend = detect_volume_trend(data, settings)
    confirms = volume_confirms_price(data, settings)

    reasons: list[str] = []
    if relative is None:
        reasons.append(
            f"Relative volume unavailable — needs {settings.volume_sma_period} bars"
        )
    else:
        descriptions = {
            VolumeCondition.SPIKE: f"Volume spike at {relative:.2f}x average",
            VolumeCondition.HIGH: f"Above-average volume at {relative:.2f}x",
            VolumeCondition.NORMAL: f"Normal volume at {relative:.2f}x",
            VolumeCondition.LOW: f"Below-average volume at {relative:.2f}x",
        }
        reasons.append(descriptions[condition])
    if trend == "RISING":
        reasons.append("Participation is increasing")
    elif trend == "FALLING":
        reasons.append("Participation is drying up")
    if confirms:
        reasons.append("Volume confirms the latest price move")

    return VolumeAnalysis(
        condition=condition,
        relative_volume=relative,
        current_volume=float(data["volume"].iloc[-1]),
        average_volume=average,
        trend=trend,
        confirms_price=confirms,
        reasons=tuple(reasons),
    )
