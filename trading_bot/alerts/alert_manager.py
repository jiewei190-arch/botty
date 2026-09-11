"""Deciding what is worth interrupting someone for.

The scoring engine ranks everything. Most of what it ranks does not deserve a
notification, and the difference between a scanner people use and one they mute
is entirely in this file.

Four gates, in order:

1. **Threshold** — the score must clear the configured bar.
2. **Direction** — a symbol whose signals cancel out is not an opportunity, and
   by default it is not an alert either.
3. **Cooldown** — the same setup, in the same symbol, in the same direction, is
   reported once per cooldown window. Without this, a symbol in a strong trend
   generates an alert every bar for an hour, and the one that mattered is the
   one nobody reads.
4. **Session budget** — a hard cap per symbol per day, so a pathological symbol
   cannot monopolise attention even across cooldown windows.

Escalation is the deliberate exception to the cooldown: a setup that was worth a
5.2 an hour ago and is now worth an 8.9 has genuinely changed, and staying quiet
about it would be the wrong kind of discipline. It re-alerts once, labelled as
an escalation so the reader knows it is not a duplicate.

Every suppression is counted. A scanner that silently drops alerts is
indistinguishable from a broken one, and :attr:`AlertManager.stats` is how you
tell them apart.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from trading_bot.alerts.channels import AlertChannel
from trading_bot.alerts.models import Alert, AlertPriority
from trading_bot.detectors.models import Bias
from trading_bot.detectors.scoring import OpportunityScore
from trading_bot.utils.market_hours import MARKET_TZ

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AlertConfig:
    """When an opportunity is worth saying out loud."""

    #: Overall score (0-10) an opportunity must reach. With the default weights
    #: a 5.0 needs real confluence — roughly two strong detectors agreeing, or
    #: three moderate ones.
    min_score: float = 5.0
    #: Silence for one symbol and direction after an alert.
    cooldown_seconds: int = 900
    #: Score improvement that re-opens the cooldown early.
    escalation_delta: float = 1.5
    #: Hard cap per symbol per trading day, escalations included.
    max_per_symbol_per_session: int = 5
    #: Minimum directional agreement (0-1). 0 disables the check.
    min_agreement: float = 0.0
    #: Suppress alerts whose net direction is NEUTRAL.
    require_direction: bool = True

    def __post_init__(self) -> None:
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds cannot be negative")
        if self.max_per_symbol_per_session < 1:
            raise ValueError("max_per_symbol_per_session must be at least 1")
        if not 0.0 <= self.min_agreement <= 1.0:
            raise ValueError("min_agreement must be between 0 and 1")


@dataclass
class _AlertState:
    """What the manager remembers about one alert key."""

    last_sent: datetime
    last_score: float
    session_day: date
    count: int = 1


def _session_day(moment: datetime) -> date:
    """The New York trading date a moment belongs to.

    Used for the per-day budget. UTC dates would roll over at 20:00 ET — in the
    middle of the after-hours session — and hand a symbol a fresh budget
    part-way through an evening.
    """
    return moment.astimezone(MARKET_TZ).date()


class AlertManager:
    """Applies the gates, then fans out to the channels."""

    def __init__(
        self,
        channels: Sequence[AlertChannel] = (),
        config: AlertConfig | None = None,
    ) -> None:
        self.config = config or AlertConfig()
        self.channels: list[AlertChannel] = list(channels)
        self._state: dict[str, _AlertState] = {}
        self._stats: Counter[str] = Counter()

    # -- configuration ----------------------------------------------------

    def add_channel(self, channel: AlertChannel) -> None:
        self.channels.append(channel)

    @property
    def stats(self) -> dict[str, int]:
        """Counts of sent alerts and of every reason one was suppressed."""
        return dict(self._stats)

    def prime(self, key: str, *, timestamp: datetime, score: float, count: int = 1) -> None:
        """Restore one key's cooldown state, usually from the database.

        Without this a restart re-alerts everything it already reported, which
        is the fastest way to teach someone to ignore the scanner.
        """
        self._state[key] = _AlertState(
            last_sent=timestamp,
            last_score=float(score),
            session_day=_session_day(timestamp),
            count=max(int(count), 1),
        )

    def reset(self) -> None:
        self._state.clear()
        self._stats.clear()

    # -- the decision -----------------------------------------------------

    def consider(
        self,
        score: OpportunityScore,
        *,
        price: float | None = None,
    ) -> Alert | None:
        """Alert on ``score`` if it clears every gate, else return None.

        The cooldown is measured against the score's own timestamp rather than
        the wall clock, so replaying a day of bars produces exactly the alerts
        that day would have produced — which is what makes alerting testable.
        """
        if score.overall < self.config.min_score:
            self._stats["suppressed_below_threshold"] += 1
            return None

        if self.config.require_direction and score.direction is Bias.NEUTRAL:
            self._stats["suppressed_no_direction"] += 1
            return None

        if score.agreement < self.config.min_agreement:
            self._stats["suppressed_conflicted"] += 1
            return None

        key = f"{score.symbol}:{score.direction.value}"
        moment = score.timestamp
        day = _session_day(moment)
        state = self._state.get(key)
        trigger = "new"

        if state is not None:
            if state.session_day != day:
                # A new trading day is a clean slate: yesterday's cooldown and
                # budget say nothing about today's setup.
                state = None
            else:
                elapsed = moment - state.last_sent
                if elapsed < timedelta(seconds=self.config.cooldown_seconds):
                    improvement = score.overall - state.last_score
                    if improvement < self.config.escalation_delta:
                        self._stats["suppressed_cooldown"] += 1
                        return None
                    trigger = "escalation"
                else:
                    trigger = "repeat"
                if state.count >= self.config.max_per_symbol_per_session:
                    self._stats["suppressed_session_budget"] += 1
                    return None

        alert = Alert.from_score(
            score,
            threshold=self.config.min_score,
            price=price,
            trigger=trigger,
            metadata={"agreement": round(score.agreement, 3)},
        )

        if state is None:
            self._state[key] = _AlertState(
                last_sent=moment, last_score=score.overall, session_day=day
            )
        else:
            state.last_sent = moment
            state.last_score = score.overall
            state.count += 1

        self._stats["sent"] += 1
        self._stats[f"sent_{alert.priority.value.lower()}"] += 1
        self.dispatch(alert)
        return alert

    def consider_all(
        self,
        scores: Iterable[OpportunityScore],
        *,
        prices: dict[str, float] | None = None,
    ) -> tuple[Alert, ...]:
        """Run :meth:`consider` over many scores, returning what was sent."""
        lookup = prices or {}
        sent = [
            alert
            for score in scores
            if (alert := self.consider(score, price=lookup.get(score.symbol))) is not None
        ]
        return tuple(sent)

    # -- delivery ---------------------------------------------------------

    def dispatch(self, alert: Alert) -> None:
        """Deliver to every channel, surviving any that fail."""
        for channel in self.channels:
            try:
                channel.deliver(alert)
            except Exception:  # noqa: BLE001 - a dead channel is not a dead scanner
                logger.exception(
                    "Alert channel %s failed to deliver %s; continuing",
                    channel.name,
                    alert.symbol,
                )
                self._stats[f"channel_error_{channel.name}"] += 1

    def close(self) -> None:
        for channel in self.channels:
            try:
                channel.close()
            except Exception:  # noqa: BLE001 - shutdown must not raise
                logger.exception("Alert channel %s failed to close", channel.name)


__all__ = ["AlertConfig", "AlertManager", "AlertPriority"]
