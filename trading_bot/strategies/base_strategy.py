"""The strategy contract.

Every strategy answers two questions and nothing else:

* **Should I open a position on this bar?** — :meth:`BaseStrategy.generate_signal`
* **Should I close the one I hold?** — :meth:`BaseStrategy.should_exit`

Strategies do not size positions, check account limits, or place orders. They
report an opportunity and the price levels that define it; the risk manager
(Phase 4) decides whether it may be taken and how large, and the execution layer
(Phase 7) places it. Keeping that boundary is what lets a strategy be swapped
without touching anything that spends money.

One evaluation rule
-------------------
``generate_signal`` looks at the **last bar of the frame it is given** and
nothing else. That single rule is what makes backtest and live behaviour
identical: the backtester passes ``bars.iloc[:i + 1]`` for each ``i`` and the
live bot passes everything up to now, and the strategy cannot tell which is
which. It also makes lookahead bias structurally impossible — a strategy has no
way to reach a bar that has not been passed to it.

Performance note
----------------
Indicators are computed once, not per call. ``calculate_all_indicators`` is
causal (proved by the Phase 2 test suite), so enriching a frame once and slicing
it gives exactly the same values as enriching each slice — at O(n) instead of
O(n²). Use :meth:`BaseStrategy.prepare` before a backtest loop; ``generate_signal``
enriches on demand only when the columns are absent.

Confidence
----------
Each strategy declares its entry conditions as :class:`Condition` objects.
Required conditions are vetoes — if any fails there is no signal at all.
Optional conditions contribute weight, and confidence is the share of total
weight that passed. So a signal firing at 62/100 means "the setup is valid and
about 62% of the supporting evidence is present", not a probability of profit.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd

from trading_bot.indicators import (
    IndicatorConfig,
    InvalidDataError,
    SupportResistance,
    atr_column,
    calculate_all_indicators,
    validate_ohlcv,
)

logger = logging.getLogger(__name__)

#: Relative tolerance when checking a strategy-supplied target against the
#: reward:risk floor. The ratio is recovered from three prices, so a target
#: placed exactly on the floor can come back a fraction below it.
RATIO_TOLERANCE = 1e-9


class StrategyError(Exception):
    """Base class for strategy-layer failures."""


class SignalDirection(str, Enum):
    """Which way a signal wants to trade."""

    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def sign(self) -> int:
        """``+1`` for long, ``-1`` for short. Makes exit maths direction-agnostic."""
        return 1 if self is SignalDirection.LONG else -1

    @property
    def opposite(self) -> SignalDirection:
        return SignalDirection.SHORT if self is SignalDirection.LONG else SignalDirection.LONG


class ExitReason(str, Enum):
    """Why a position should be closed."""

    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TREND_REVERSAL = "TREND_REVERSAL"
    MOMENTUM_FADE = "MOMENTUM_FADE"
    SIGNAL_FLIP = "SIGNAL_FLIP"
    TARGET_REACHED = "TARGET_REACHED"
    TIME_STOP = "TIME_STOP"

    @property
    def is_protective(self) -> bool:
        """True for exits that must be honoured immediately, not discretionary."""
        return self in (ExitReason.STOP_LOSS, ExitReason.TAKE_PROFIT)


@dataclass(frozen=True, slots=True)
class Condition:
    """One checked entry criterion.

    Attributes
    ----------
    name:
        Short identifier, e.g. ``"ema_crossover"``.
    passed:
        Whether the criterion holds on this bar.
    weight:
        Relative contribution to confidence. Ignored for required conditions
        that fail, since those veto the signal outright.
    required:
        A failing required condition means no signal at all.
    detail:
        Human-readable explanation, surfaced in ``Signal.reasons``.
    """

    name: str
    passed: bool
    weight: float = 1.0
    required: bool = False
    detail: str = ""

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError(f"Condition weight must be >= 0, got {self.weight}")


def score_conditions(conditions: list[Condition]) -> tuple[bool, float, tuple[str, ...]]:
    """Reduce conditions to (is_valid, confidence, reasons).

    Returns
    -------
    tuple
        ``valid`` is False when any required condition failed. ``confidence`` is
        the percentage of total weight that passed. ``reasons`` describes the
        conditions that did pass.

    Example
    -------
    >>> score_conditions([Condition("trend", True, 2.0, required=True, detail="Uptrend")])
    (True, 100.0, ('Uptrend',))
    """
    if not conditions:
        return False, 0.0, ()

    failed_required = [c for c in conditions if c.required and not c.passed]
    total_weight = sum(c.weight for c in conditions)
    passed_weight = sum(c.weight for c in conditions if c.passed)
    confidence = (passed_weight / total_weight * 100) if total_weight > 0 else 0.0
    reasons = tuple(c.detail for c in conditions if c.passed and c.detail)
    return not failed_required, confidence, reasons


@dataclass(frozen=True, slots=True)
class Signal:
    """A proposed trade.

    A ``Signal`` is a *proposal*, not an order. It carries the three prices that
    define the trade — entry, stop and target — so the risk manager can size it
    without re-deriving anything.

    Construction validates that the stop and target sit on the correct sides of
    the entry for the direction. A long signal whose stop is above its entry
    would make position sizing divide by a negative risk, so it is rejected here
    rather than allowed downstream.
    """

    symbol: str
    direction: SignalDirection
    strategy: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    timestamp: datetime
    reasons: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 100:
            raise ValueError(f"confidence must be within 0-100, got {self.confidence}")
        for label, price in (
            ("entry_price", self.entry_price),
            ("stop_loss", self.stop_loss),
            ("take_profit", self.take_profit),
        ):
            if not price > 0:
                raise ValueError(f"{label} must be positive, got {price}")

        if self.direction is SignalDirection.LONG:
            if self.stop_loss >= self.entry_price:
                raise ValueError(
                    f"LONG stop_loss ({self.stop_loss}) must be below entry "
                    f"({self.entry_price})"
                )
            if self.take_profit <= self.entry_price:
                raise ValueError(
                    f"LONG take_profit ({self.take_profit}) must be above entry "
                    f"({self.entry_price})"
                )
        else:
            if self.stop_loss <= self.entry_price:
                raise ValueError(
                    f"SHORT stop_loss ({self.stop_loss}) must be above entry "
                    f"({self.entry_price})"
                )
            if self.take_profit >= self.entry_price:
                raise ValueError(
                    f"SHORT take_profit ({self.take_profit}) must be below entry "
                    f"({self.entry_price})"
                )

    @property
    def risk_per_share(self) -> float:
        """Distance from entry to stop. The denominator of position sizing."""
        return abs(self.entry_price - self.stop_loss)

    @property
    def reward_per_share(self) -> float:
        """Distance from entry to target."""
        return abs(self.take_profit - self.entry_price)

    @property
    def risk_reward_ratio(self) -> float:
        """Reward divided by risk. ``2.0`` means a 1:2 trade."""
        risk = self.risk_per_share
        return self.reward_per_share / risk if risk > 0 else 0.0

    @property
    def stop_distance_pct(self) -> float:
        """Stop distance as a percentage of entry."""
        return (self.risk_per_share / self.entry_price * 100) if self.entry_price else 0.0

    def as_dict(self) -> dict[str, Any]:
        """Serialisable view, matching the ``signals`` database table."""
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "strategy": self.strategy,
            "confidence": round(self.confidence, 2),
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "risk_reward": round(self.risk_reward_ratio, 3),
            "timestamp": self.timestamp.isoformat(),
            "reasons": list(self.reasons),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ExitSignal:
    """An instruction to close an open position."""

    reason: ExitReason
    price: float
    detail: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_protective(self) -> bool:
        return self.reason.is_protective

    def as_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason.value,
            "price": self.price,
            "detail": self.detail,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class Position:
    """The open position a strategy is asked to evaluate for exit.

    Deliberately minimal — a strategy needs the trade's shape, not the account's.
    """

    symbol: str
    direction: SignalDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: float = 0.0
    entry_timestamp: datetime | None = None
    strategy: str = ""
    bars_held: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def unrealized_pnl_per_share(self, price: float) -> float:
        """Profit per share at ``price``, signed by direction."""
        return (price - self.entry_price) * self.direction.sign

    def r_multiple(self, price: float) -> float:
        """Progress in units of initial risk. ``+2.0`` means twice the risk earned."""
        risk = abs(self.entry_price - self.stop_loss)
        if risk <= 0:
            return 0.0
        return self.unrealized_pnl_per_share(price) / risk


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    """Parameters shared by every strategy.

    Each strategy subclasses this to add its own fields. Frozen, so a backtest
    parameter sweep derives variants with :meth:`with_overrides` rather than
    mutating shared state.
    """

    #: Signals below this confidence are not emitted at all.
    min_confidence: float = 55.0
    #: Stop distance in ATR units. Wider stops survive noise but risk more per share.
    atr_stop_multiplier: float = 1.5
    #: Target distance in ATR units.
    atr_target_multiplier: float = 3.0
    #: Targets are widened if needed so reward:risk never falls below this.
    min_risk_reward: float = 2.0
    #: Fallback stop distance when ATR is unavailable (warm-up).
    fallback_stop_pct: float = 2.0
    #: Place stops just beyond a nearby structural level instead of a raw ATR distance.
    use_structure_stops: bool = True
    #: Structural levels further than this many ATRs away are ignored for stops.
    max_structure_distance_atr: float = 2.0
    #: Extra buffer beyond a structural level, as a fraction of ATR.
    structure_buffer_atr: float = 0.25
    #: Floor on the stop distance, in ATR. A stop closer than this sits inside a
    #: single bar's ordinary range and will be hit by noise rather than by the
    #: trade being wrong. It also inflates reward:risk and, because position size
    #: divides by the stop distance, inflates the position — a tight stop is not
    #: a free lunch, it is a smaller edge taken in larger size.
    min_stop_atr: float = 0.75
    allow_long: bool = True
    allow_short: bool = False
    #: Close a position after this many bars, or None to hold indefinitely.
    max_holding_bars: int | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.min_confidence <= 100:
            raise ValueError(
                f"min_confidence must be within 0-100, got {self.min_confidence}"
            )
        if self.atr_stop_multiplier <= 0:
            raise ValueError(
                f"atr_stop_multiplier must be > 0, got {self.atr_stop_multiplier}"
            )
        if self.min_stop_atr < 0:
            raise ValueError(f"min_stop_atr must be >= 0, got {self.min_stop_atr}")
        if self.atr_target_multiplier <= 0:
            raise ValueError(
                f"atr_target_multiplier must be > 0, got {self.atr_target_multiplier}"
            )
        if self.min_risk_reward <= 0:
            raise ValueError(f"min_risk_reward must be > 0, got {self.min_risk_reward}")
        if self.fallback_stop_pct <= 0:
            raise ValueError(f"fallback_stop_pct must be > 0, got {self.fallback_stop_pct}")
        if not self.allow_long and not self.allow_short:
            raise ValueError("A strategy with neither long nor short enabled can never trade")
        if self.max_holding_bars is not None and self.max_holding_bars < 1:
            raise ValueError(
                f"max_holding_bars must be >= 1 or None, got {self.max_holding_bars}"
            )

    def with_overrides(self, **overrides: Any) -> StrategyConfig:
        """Return a validated copy with fields replaced."""
        return replace(self, **overrides)

    def allows(self, direction: SignalDirection) -> bool:
        """Whether this configuration permits trading in ``direction``."""
        if direction is SignalDirection.LONG:
            return self.allow_long
        return self.allow_short


class BaseStrategy(ABC):
    """Base class for all trading strategies.

    Subclasses implement :meth:`evaluate`, which sees the last bar of the frame
    and returns a :class:`Signal` or ``None``. Everything else — validation,
    indicator preparation, confidence gating, stop and target placement, and the
    protective exit checks — is handled here so every strategy behaves
    consistently.

    Example
    -------
    >>> strategy = MomentumStrategy()
    >>> prepared = strategy.prepare(bars)
    >>> signal = strategy.generate_signal("AAPL", prepared)
    >>> if signal:
    ...     print(signal.direction, signal.confidence, signal.risk_reward_ratio)
    """

    #: Human-readable strategy name, used in signals, logs and the database.
    name: str = "base"

    def __init__(
        self,
        config: StrategyConfig | None = None,
        indicators: IndicatorConfig | None = None,
    ) -> None:
        self.config = config or StrategyConfig()
        self.indicators = indicators or IndicatorConfig()
        #: Conditions scored during the most recent :meth:`generate_signal` call,
        #: keyed by direction. Diagnostic only — nothing reads it to make a
        #: decision. It exists so the bot can answer "why didn't you trade?",
        #: which is the question that matters most when a bot sits idle.
        self.last_evaluation: dict[SignalDirection, tuple[Condition, ...]] = {}

    # -- lifecycle ---------------------------------------------------------------

    @property
    def min_bars(self) -> int:
        """Bars needed before this strategy can produce a signal.

        Defaults to the indicator warm-up. Subclasses needing more history
        (a consolidation window, say) should extend it.
        """
        return self.indicators.max_lookback

    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enrich a frame with indicator columns once, for repeated evaluation.

        A backtest should call this on the full history and then pass slices to
        :meth:`generate_signal`. Because the indicator maths is causal, slicing
        an enriched frame equals enriching each slice, at a fraction of the cost.
        """
        return calculate_all_indicators(data, self.indicators)

    def _ensure_prepared(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enrich only if the caller has not already done so."""
        if atr_column(self.indicators.atr_period) in data.columns:
            return data
        return self.prepare(data)

    # -- entry -------------------------------------------------------------------

    @abstractmethod
    def evaluate(self, symbol: str, data: pd.DataFrame) -> Signal | None:
        """Decide whether the last bar of ``data`` warrants a new position.

        Implementations must read only the final bar (and history before it),
        never a future one. Return ``None`` when there is no setup.
        """

    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Signal | None:
        """Validate, prepare and evaluate, returning a signal or ``None``.

        This is the method callers use. It guarantees that :meth:`evaluate` sees
        a valid, indicator-enriched frame with enough history, and that any
        returned signal clears the configured confidence floor.

        Raises
        ------
        InvalidDataError
            The frame is empty or malformed.
        """
        validate_ohlcv(data, name=f"{symbol} data")
        self.last_evaluation = {}
        if len(data) < self.min_bars:
            logger.debug(
                "%s: %d bars is below the %d %s needs",
                symbol, len(data), self.min_bars, self.name,
            )
            return None

        prepared = self._ensure_prepared(data)
        signal = self.evaluate(symbol, prepared)
        if signal is None:
            return None

        if signal.confidence < self.config.min_confidence:
            logger.debug(
                "%s: %s signal at %.0f confidence is below the %.0f floor",
                symbol, self.name, signal.confidence, self.config.min_confidence,
            )
            return None
        return signal

    # -- exit --------------------------------------------------------------------

    def should_exit(self, position: Position, data: pd.DataFrame) -> ExitSignal | None:
        """Decide whether an open position should be closed on this bar.

        Protective exits are checked first and are not overridable: a stop or
        target hit inside the bar closes the position regardless of what the
        strategy thinks. Only then is :meth:`evaluate_exit` consulted for
        discretionary reasons such as a fading trend.
        """
        if data.empty:
            return None

        prepared = self._ensure_prepared(data)
        bar = prepared.iloc[-1]
        high, low, close = float(bar["high"]), float(bar["low"]), float(bar["close"])

        protective = self._check_protective_exits(position, high=high, low=low)
        if protective is not None:
            return protective

        if (
            self.config.max_holding_bars is not None
            and position.bars_held >= self.config.max_holding_bars
        ):
            return ExitSignal(
                reason=ExitReason.TIME_STOP,
                price=close,
                detail=f"Held {position.bars_held} bars, limit is {self.config.max_holding_bars}",
            )

        return self.evaluate_exit(position, prepared)

    def _check_protective_exits(
        self, position: Position, *, high: float, low: float
    ) -> ExitSignal | None:
        """Check whether the bar's range touched the stop or the target.

        When a bar spans both levels the **stop wins**. Intrabar sequence is
        unknowable from OHLC data, and assuming the favourable order is how a
        backtest quietly inflates its results.
        """
        if position.direction is SignalDirection.LONG:
            hit_stop = low <= position.stop_loss
            hit_target = high >= position.take_profit
        else:
            hit_stop = high >= position.stop_loss
            hit_target = low <= position.take_profit

        if hit_stop:
            return ExitSignal(
                reason=ExitReason.STOP_LOSS,
                price=position.stop_loss,
                detail=f"Price reached the stop at {position.stop_loss:.2f}",
                metadata={"ambiguous_bar": hit_target},
            )
        if hit_target:
            return ExitSignal(
                reason=ExitReason.TAKE_PROFIT,
                price=position.take_profit,
                detail=f"Price reached the target at {position.take_profit:.2f}",
            )
        return None

    def evaluate_exit(self, position: Position, data: pd.DataFrame) -> ExitSignal | None:
        """Strategy-specific discretionary exit. Override to add one.

        Called only after protective exits have been ruled out, with an
        indicator-enriched frame.
        """
        return None

    # -- helpers for subclasses --------------------------------------------------

    def _atr_value(self, data: pd.DataFrame) -> float | None:
        """Latest ATR, or ``None`` while it is still warming up."""
        column = atr_column(self.indicators.atr_period)
        if column not in data.columns:
            return None
        value = data[column].iloc[-1]
        if pd.isna(value) or float(value) <= 0:
            return None
        return float(value)

    def build_exits(
        self,
        direction: SignalDirection,
        entry_price: float,
        atr: float | None,
        *,
        levels: SupportResistance | None = None,
    ) -> tuple[float, float, list[str]]:
        """Place the stop and target for a signal.

        Stop placement, in order of preference:

        1. Just beyond the nearest structural level, when one sits within
           ``max_structure_distance_atr`` ATRs (a stop below support is where the
           trade thesis is actually wrong).
        2. ``atr_stop_multiplier`` ATRs away — volatility-scaled, so a quiet
           stock gets a tight stop and a volatile one gets room.
        3. ``fallback_stop_pct`` percent, only while ATR is warming up.

        Whichever is chosen is then floored at ``min_stop_atr`` ATRs. A structural
        level can sit a few cents from the entry, and a stop that close is inside
        the noise: it gets hit by an ordinary bar, while the arithmetic makes the
        trade look better than it is — reward:risk is inflated, and position size,
        which divides by the stop distance, is inflated with it.

        The target is ``atr_target_multiplier`` ATRs away, then widened if needed
        so reward:risk is never below ``min_risk_reward``. That makes the ratio a
        structural guarantee rather than something to filter for later.

        Returns
        -------
        tuple
            ``(stop_loss, take_profit, notes)``.
        """
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")

        notes: list[str] = []
        sign = direction.sign

        if atr is not None and atr > 0:
            risk = atr * self.config.atr_stop_multiplier
            structural = self._structural_stop_distance(direction, entry_price, atr, levels)
            if structural is not None and structural > 0:
                risk = structural
                notes.append("Stop placed beyond nearby structure")
        else:
            risk = entry_price * self.config.fallback_stop_pct / 100
            notes.append("ATR unavailable — using a percentage stop")

        if atr is not None and atr > 0:
            floor = atr * self.config.min_stop_atr
            if risk < floor:
                notes.append(
                    f"Stop widened to {self.config.min_stop_atr:.2f} ATR — "
                    "closer than that is inside the noise"
                )
                risk = floor

        # Never let a degenerate stop through; it would divide sizing by ~zero.
        risk = max(risk, entry_price * 0.0005)
        stop_loss = entry_price - sign * risk

        # A stop cannot cross zero on a long.
        if stop_loss <= 0:
            stop_loss = entry_price * 0.5
            risk = entry_price - stop_loss

        reward = (
            atr * self.config.atr_target_multiplier
            if atr
            else risk * self.config.min_risk_reward
        )
        minimum_reward = risk * self.config.min_risk_reward
        if reward < minimum_reward:
            reward = minimum_reward
            notes.append(f"Target widened to hold {self.config.min_risk_reward:.1f}:1")
        take_profit = entry_price + sign * reward

        return stop_loss, take_profit, notes

    def _structural_stop_distance(
        self,
        direction: SignalDirection,
        entry_price: float,
        atr: float,
        levels: SupportResistance | None,
    ) -> float | None:
        """Distance to a stop placed just beyond the nearest structural level."""
        if not self.config.use_structure_stops or levels is None:
            return None

        level = levels.nearest_support if direction is SignalDirection.LONG else (
            levels.nearest_resistance
        )
        if level is None:
            return None

        distance = abs(entry_price - level.price)
        if distance > atr * self.config.max_structure_distance_atr:
            return None  # too far to be the reason this trade is wrong

        return distance + atr * self.config.structure_buffer_atr

    def _build_signal(
        self,
        *,
        symbol: str,
        direction: SignalDirection,
        data: pd.DataFrame,
        conditions: list[Condition],
        levels: SupportResistance | None = None,
        metadata: dict[str, Any] | None = None,
        target_override: float | None = None,
        target_is_mandatory: bool = True,
    ) -> Signal | None:
        """Assemble a signal from scored conditions, or return ``None``.

        Returns ``None`` when the direction is disallowed, a required condition
        failed, or the exits cannot be placed sensibly.

        ``target_override`` lets a strategy supply a target its thesis dictates.
        Such a target is never widened to meet ``min_risk_reward`` — moving it
        past the thesis would only be lying about the expected outcome. What
        happens when it does not pay depends on ``target_is_mandatory``:

        * ``True`` (mean reversion): the target *is* the thesis — price returning
          to its mean. If that does not pay for the risk, there is no trade, so
          the signal is dropped.
        * ``False`` (breakout): the measured move is *a* target, not the reason
          for the trade. When it has already been made, the setup is still valid
          on volatility terms, so the override is discarded and the ATR-based
          target stands.
        """
        self.last_evaluation[direction] = tuple(conditions)

        if not self.config.allows(direction):
            return None

        valid, confidence, reasons = score_conditions(conditions)
        if not valid:
            return None

        bar = data.iloc[-1]
        entry_price = float(bar["close"])
        atr = self._atr_value(data)
        stop_loss, take_profit, notes = self.build_exits(
            direction, entry_price, atr, levels=levels
        )

        if target_override is not None:
            risk = abs(entry_price - stop_loss)
            reward = abs(target_override - entry_price)
            pays = risk > 0 and reward / risk >= self.config.min_risk_reward * (
                1 - RATIO_TOLERANCE
            )
            if pays:
                take_profit = target_override
                notes.append("Target set by the strategy's own thesis")
            elif target_is_mandatory:
                logger.debug(
                    "%s: %s target %.4f pays %.2f:1 against a %.2f risk, below the %.1f floor",
                    symbol, self.name, target_override,
                    reward / risk if risk > 0 else 0.0, risk, self.config.min_risk_reward,
                )
                return None
            else:
                notes.append("Measured move already made — using a volatility target")

        payload: dict[str, Any] = {
            "atr": atr,
            "conditions": {c.name: c.passed for c in conditions},
        }
        if metadata:
            payload.update(metadata)

        try:
            return Signal(
                symbol=symbol.upper(),
                direction=direction,
                strategy=self.name,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timestamp=_timestamp_of(data),
                reasons=tuple(reasons) + tuple(notes),
                metadata=payload,
            )
        except ValueError as error:
            # Refuse to emit an incoherent signal rather than let sizing break later.
            logger.warning("%s: discarded a malformed %s signal: %s", symbol, self.name, error)
            return None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(name={self.name!r})"


def explain_blockers(
    strategy: BaseStrategy, direction: SignalDirection | None = None
) -> list[str]:
    """Describe what stopped the most recent evaluation from producing a signal.

    Reads :attr:`BaseStrategy.last_evaluation`, so call it straight after
    :meth:`BaseStrategy.generate_signal` returned ``None``.

    Returns
    -------
    list[str]
        Names of the failed required conditions, most restrictive first, or an
        empty list when nothing was blocking (the confidence floor, or a
        disallowed direction, will have been the reason instead).

    Example
    -------
    >>> if strategy.generate_signal("AAPL", data) is None:
    ...     print(explain_blockers(strategy))
    ['volume', 'consolidation']
    """
    blockers: list[str] = []
    for evaluated_direction, conditions in strategy.last_evaluation.items():
        if direction is not None and evaluated_direction is not direction:
            continue
        for condition in conditions:
            if condition.required and not condition.passed:
                blockers.append(condition.name)
    return blockers


def _timestamp_of(data: pd.DataFrame) -> datetime:
    """Timestamp of the last bar, as a timezone-aware datetime."""
    stamp = data.index[-1]
    if isinstance(stamp, pd.Timestamp):
        return stamp.to_pydatetime()
    raise InvalidDataError(f"Expected a timestamp index, got {type(stamp).__name__}")
