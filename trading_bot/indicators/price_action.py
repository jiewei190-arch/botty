"""Price-action structure: swing points, support and resistance.

Swing points are found with a symmetric pivot rule, and support/resistance
levels are built by clustering nearby pivots. No machine learning, no curve
fitting — a pivot is a bar whose high exceeds every high within ``strength``
bars either side, and a level is a price several pivots have reacted to.

Confirmation lag — the reason this module is careful
----------------------------------------------------
A swing high at bar ``i`` cannot be identified until bar ``i + strength``, because
the rule needs the bars *after* it. Treating a pivot as known at bar ``i`` is a
textbook lookahead bug: the backtest sees tops and bottoms that a live bot could
not have seen yet, and the resulting equity curve is fiction.

So every pivot here carries the index at which it became knowable, and
:func:`find_support_resistance` only uses pivots confirmed on or before the bar
it is asked about. The ``as_of`` parameter exists specifically so a backtest can
ask "what levels were visible at this bar?" and get an honest answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
import pandas as pd

from trading_bot.indicators.technical_indicators import (
    DEFAULT_CONFIG,
    IndicatorConfig,
    InvalidDataError,
    validate_ohlcv,
)

logger = logging.getLogger(__name__)

SWING_HIGH_COL = "SWING_HIGH"
SWING_LOW_COL = "SWING_LOW"

StructureLabel = Literal[
    "HIGHER_HIGHS_HIGHER_LOWS",
    "LOWER_HIGHS_LOWER_LOWS",
    "RANGING",
    "UNDETERMINED",
]


@dataclass(frozen=True, slots=True)
class SwingPoint:
    """A confirmed pivot.

    Attributes
    ----------
    index:
        Position of the pivot bar.
    timestamp:
        Timestamp of the pivot bar.
    price:
        The pivot's high (for a swing high) or low (for a swing low).
    kind:
        ``"high"`` or ``"low"``.
    confirmed_index:
        Position of the first bar at which this pivot was knowable — always
        ``index + strength``. Anything reading pivots as of bar ``t`` must ignore
        pivots whose ``confirmed_index`` exceeds ``t``.
    """

    index: int
    timestamp: pd.Timestamp
    price: float
    kind: Literal["high", "low"]
    confirmed_index: int


@dataclass(frozen=True, slots=True)
class Level:
    """A support or resistance level built from clustered pivots."""

    price: float
    #: How many pivots formed this level. More touches, more significance.
    touches: int
    #: Position of the most recent pivot in the cluster.
    last_touch_index: int
    kind: Literal["support", "resistance"]

    def distance_pct(self, price: float) -> float:
        """Signed distance from ``price`` to this level, in percent."""
        if price <= 0:
            return 0.0
        return (self.price - price) / price * 100

    def as_dict(self) -> dict[str, Any]:
        return {
            "price": self.price,
            "touches": self.touches,
            "last_touch_index": self.last_touch_index,
            "kind": self.kind,
        }


@dataclass(frozen=True, slots=True)
class SupportResistance:
    """Levels visible as of a particular bar.

    ``support`` is sorted nearest-first below the reference price, ``resistance``
    nearest-first above it.
    """

    price: float
    support: tuple[Level, ...]
    resistance: tuple[Level, ...]
    swing_points: tuple[SwingPoint, ...]

    @property
    def nearest_support(self) -> Level | None:
        """Closest level below the reference price, if any."""
        return self.support[0] if self.support else None

    @property
    def nearest_resistance(self) -> Level | None:
        """Closest level above the reference price, if any."""
        return self.resistance[0] if self.resistance else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "price": self.price,
            "support": [level.as_dict() for level in self.support],
            "resistance": [level.as_dict() for level in self.resistance],
            "nearest_support": self.nearest_support.price if self.nearest_support else None,
            "nearest_resistance": (
                self.nearest_resistance.price if self.nearest_resistance else None
            ),
        }


def find_swing_points(
    data: pd.DataFrame,
    strength: int | None = None,
    config: IndicatorConfig | None = None,
    *,
    validate: bool = True,
) -> list[SwingPoint]:
    """Locate swing highs and lows with a symmetric pivot rule.

    A swing high is a bar whose ``high`` is the maximum of the window spanning
    ``strength`` bars either side; a swing low is the mirror image.

    Parameters
    ----------
    data:
        OHLCV frame.
    strength:
        Bars required either side of the pivot. Larger values find fewer, more
        significant pivots and take longer to confirm.
    config:
        Supplies ``swing_strength`` when ``strength`` is not given.
    validate:
        Check the frame first. Internal callers that have already validated pass
        False: this runs once per bar in a backtest, and a full-column scan of
        the whole history each time makes the run O(n^2).

    Returns
    -------
    list[SwingPoint]
        Pivots in chronological order. Each carries ``confirmed_index``; the
        final ``strength`` bars can never contain a confirmed pivot.

    Example
    -------
    >>> pivots = find_swing_points(bars, strength=3)
    >>> [p.price for p in pivots if p.kind == "high"][-3:]
    """
    settings = config or DEFAULT_CONFIG
    span = strength if strength is not None else settings.swing_strength
    if span < 1:
        raise ValueError(f"strength must be >= 1, got {span}")
    if validate:
        validate_ohlcv(data, name="data")

    window = 2 * span + 1
    if len(data) < window:
        return []

    highs = data["high"]
    lows = data["low"]

    # Compare each bar against the `span` bars before and after it. The forward
    # look is what a pivot *is*; the lookahead it implies is carried by
    # confirmed_index, which every consumer must respect.
    # Neighbour extremes via shifted numpy views. The equivalent built from
    # pd.concat plus DataFrame.max(axis=1) measured as 64% of a backtest's total
    # runtime; this is the same comparison without constructing a frame per bar.
    high_values = highs.to_numpy(dtype="float64")
    low_values = lows.to_numpy(dtype="float64")
    count = len(high_values)

    prior_max = np.full(count, -np.inf)
    later_max = np.full(count, -np.inf)
    prior_min = np.full(count, np.inf)
    later_min = np.full(count, np.inf)

    for offset in range(1, span + 1):
        prior_max[offset:] = np.maximum(prior_max[offset:], high_values[:-offset])
        later_max[:-offset] = np.maximum(later_max[:-offset], high_values[offset:])
        prior_min[offset:] = np.minimum(prior_min[offset:], low_values[:-offset])
        later_min[:-offset] = np.minimum(later_min[:-offset], low_values[offset:])

    # Edges have an incomplete window and cannot host a pivot.
    complete = np.zeros(count, dtype=bool)
    if count > 2 * span:
        complete[span : count - span] = True

    # Strict on one side, inclusive on the other. Requiring strict inequality on
    # both sides would miss a flat double top; requiring neither would make every
    # bar of a flat series a pivot, fabricating levels out of nothing.
    is_high = complete & (high_values > prior_max) & (high_values >= later_max)
    is_low = complete & (low_values < prior_min) & (low_values <= later_min)

    points: list[SwingPoint] = []
    positions = np.arange(count)
    for position, timestamp, high_flag, low_flag in zip(
        positions, data.index, is_high, is_low, strict=True
    ):
        if high_flag:
            points.append(
                SwingPoint(
                    index=int(position),
                    timestamp=timestamp,
                    price=float(high_values[position]),
                    kind="high",
                    confirmed_index=int(position) + span,
                )
            )
        if low_flag:
            points.append(
                SwingPoint(
                    index=int(position),
                    timestamp=timestamp,
                    price=float(low_values[position]),
                    kind="low",
                    confirmed_index=int(position) + span,
                )
            )
    points.sort(key=lambda point: (point.index, point.kind))
    return points


def swing_point_columns(
    data: pd.DataFrame,
    strength: int | None = None,
    config: IndicatorConfig | None = None,
) -> pd.DataFrame:
    """Boolean ``SWING_HIGH`` / ``SWING_LOW`` columns aligned to ``data``.

    Flags sit on the pivot bar itself. For decisions, prefer
    :func:`find_swing_points` and honour ``confirmed_index``.

    Example
    -------
    >>> flags = swing_point_columns(bars)
    >>> flags["SWING_HIGH"].sum()
    """
    points = find_swing_points(data, strength, config)
    highs = pd.Series(False, index=data.index, name=SWING_HIGH_COL)
    lows = pd.Series(False, index=data.index, name=SWING_LOW_COL)
    for point in points:
        if point.kind == "high":
            highs.iloc[point.index] = True
        else:
            lows.iloc[point.index] = True
    return pd.DataFrame({SWING_HIGH_COL: highs, SWING_LOW_COL: lows})


def _cluster_levels(
    pivots: list[SwingPoint],
    tolerance_pct: float,
    kind: Literal["support", "resistance"],
) -> list[Level]:
    """Merge pivots that sit within ``tolerance_pct`` of each other into levels."""
    if not pivots:
        return []

    ordered = sorted(pivots, key=lambda point: point.price)
    clusters: list[list[SwingPoint]] = [[ordered[0]]]
    for point in ordered[1:]:
        current = clusters[-1]
        reference = float(np.mean([member.price for member in current]))
        if reference > 0 and abs(point.price - reference) / reference * 100 <= tolerance_pct:
            current.append(point)
        else:
            clusters.append([point])

    levels: list[Level] = []
    for cluster in clusters:
        levels.append(
            Level(
                price=float(np.mean([member.price for member in cluster])),
                touches=len(cluster),
                last_touch_index=max(member.index for member in cluster),
                kind=kind,
            )
        )
    return levels


def find_support_resistance(
    data: pd.DataFrame,
    config: IndicatorConfig | None = None,
    *,
    as_of: int | None = None,
    price: float | None = None,
    max_levels: int = 5,
) -> SupportResistance:
    """Build support and resistance levels from confirmed pivots.

    Parameters
    ----------
    data:
        OHLCV frame.
    config:
        Supplies ``swing_strength``, ``level_lookback`` and ``level_tolerance_pct``.
    as_of:
        Position of the bar to analyse from. Defaults to the last bar. **Only
        pivots confirmed on or before this bar are used**, so a backtest asking
        about bar 50 gets the levels a live bot would have had at bar 50.
    price:
        Reference price splitting support from resistance. Defaults to the close
        at ``as_of``.
    max_levels:
        Maximum levels returned on each side, nearest first.

    Returns
    -------
    SupportResistance

    Example
    -------
    >>> levels = find_support_resistance(bars)
    >>> levels.nearest_support.price if levels.nearest_support else None
    """
    settings = config or DEFAULT_CONFIG
    validate_ohlcv(data, name="data")

    position = len(data) - 1 if as_of is None else int(as_of)
    if not 0 <= position < len(data):
        raise InvalidDataError(
            f"as_of must be a bar position between 0 and {len(data) - 1}, got {position}"
        )

    reference_price = float(price if price is not None else data["close"].iloc[position])

    # Only bars inside the lookback can contribute a level, so only scan those.
    # Scanning the whole history instead would make a per-bar backtest O(n^2) for
    # results that are discarded anyway. The extra `swing_strength` bars give the
    # earliest pivot in the window its left-hand comparison bars.
    window_start = max(0, position - settings.level_lookback - settings.swing_strength)
    window = data.iloc[window_start : position + 1]
    all_points = [
        # Shift indices back to positions in the caller's frame.
        replace(
            point,
            index=point.index + window_start,
            confirmed_index=point.confirmed_index + window_start,
        )
        for point in find_swing_points(
            window, settings.swing_strength, settings, validate=False
        )
    ]

    relevant_from = max(0, position - settings.level_lookback)
    visible = [
        point
        for point in all_points
        # Confirmed by now, and recent enough to still be relevant.
        if point.confirmed_index <= position and relevant_from <= point.index <= position
    ]

    highs = [point for point in visible if point.kind == "high"]
    lows = [point for point in visible if point.kind == "low"]

    # A pivot's role depends on which side of price it sits, not on its type:
    # a broken swing high becomes support.
    above = [point for point in highs + lows if point.price > reference_price]
    below = [point for point in highs + lows if point.price < reference_price]

    resistance = _cluster_levels(above, settings.level_tolerance_pct, "resistance")
    support = _cluster_levels(below, settings.level_tolerance_pct, "support")

    resistance.sort(key=lambda level: level.price)                 # nearest above first
    support.sort(key=lambda level: level.price, reverse=True)      # nearest below first

    return SupportResistance(
        price=reference_price,
        support=tuple(support[:max_levels]),
        resistance=tuple(resistance[:max_levels]),
        swing_points=tuple(visible),
    )


def detect_market_structure(
    data: pd.DataFrame,
    config: IndicatorConfig | None = None,
    *,
    as_of: int | None = None,
) -> StructureLabel:
    """Classify swing structure as trending or ranging.

    Compares the two most recent confirmed swing highs and swing lows:
    higher highs *and* higher lows is an uptrend, lower highs *and* lower lows a
    downtrend, anything else ranging.

    Returns
    -------
    str
        ``HIGHER_HIGHS_HIGHER_LOWS``, ``LOWER_HIGHS_LOWER_LOWS``, ``RANGING`` or
        ``UNDETERMINED`` when there are too few confirmed pivots to say.

    Example
    -------
    >>> detect_market_structure(bars)
    'HIGHER_HIGHS_HIGHER_LOWS'
    """
    settings = config or DEFAULT_CONFIG
    position = len(data) - 1 if as_of is None else int(as_of)

    # Same windowing as find_support_resistance: only recent structure matters,
    # and scanning the full history per bar does not scale.
    window_start = max(0, position - settings.level_lookback - settings.swing_strength)
    window = data.iloc[window_start : position + 1]
    points = [
        replace(point, index=point.index + window_start,
                confirmed_index=point.confirmed_index + window_start)
        for point in find_swing_points(
            window, settings.swing_strength, settings, validate=False
        )
    ]
    points = [point for point in points if point.confirmed_index <= position]

    highs = [point.price for point in points if point.kind == "high"][-2:]
    lows = [point.price for point in points if point.kind == "low"][-2:]
    if len(highs) < 2 or len(lows) < 2:
        return "UNDETERMINED"

    # Equality is neither higher nor lower — treating it as "lower" would report
    # a flat market as a downtrend.
    if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
        return "HIGHER_HIGHS_HIGHER_LOWS"
    if highs[-1] < highs[-2] and lows[-1] < lows[-2]:
        return "LOWER_HIGHS_LOWER_LOWS"
    return "RANGING"
