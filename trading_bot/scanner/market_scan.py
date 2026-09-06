"""Sweeping the whole market for tradable setups.

The Phase 5 scanner ranks a watchlist. This runs the same machinery across every
liquid US equity and returns the handful worth looking at — the difference is
not the analysis but the funnel in front of it, because analysing eleven
thousand symbols the way you analyse ten is not a thing that finishes.

The funnel, cheapest stage first
--------------------------------
1. **Metadata** — exchange, tradability, instrument type. No price data, cuts
   roughly half.
2. **Liquidity** — daily bars, then price and turnover floors. This is the
   expensive network stage, and its bars are kept and reused by everything
   downstream rather than fetched again.
3. **Strategies** — run over daily bars for the survivors. Most produce no
   signal on any given day; that is the point.
4. **Scoring and risk** — the survivors are ranked on one common yardstick and
   sized against the account.

Why a signal's age matters
--------------------------
A swing setup that triggered four sessions ago has already made its move. It
will still evaluate as "valid" — the conditions that fired are still true — but
entering now means paying for the part of the move you missed while carrying
the same stop. Every candidate is therefore dated, and stale ones are dropped
before ranking rather than presented as fresh.

Nothing here places an order. It produces a ranked list of proposals with the
prices to work them at.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from trading_bot.risk import RiskManager
from trading_bot.scanner.scanner import MarketScanner, Opportunity, ScannerConfig
from trading_bot.strategies import BaseStrategy
from trading_bot.universe import Universe

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HuntConfig:
    """How wide to cast the net, and what counts as worth returning.

    Attributes
    ----------
    timeframe:
        Bars the strategies analyse. ``1Day`` for swing trading: a daily bar is
        one decision per symbol per day, which is a cadence a person can act on
        manually.
    max_signal_age_bars:
        Drop setups that triggered more than this many bars ago. ``1`` means
        only setups that fired on the most recent close. Raising it surfaces
        more candidates and worse entries.
    min_score:
        Composite score floor, 0-100.
    top_n:
        How many opportunities to return.
    min_risk_reward:
        Reject setups whose target does not pay for the stop by this multiple.
    require_risk_approval:
        Only return setups the risk manager approved and sized. Turning this
        off shows rejected setups with their reasons, which is useful for
        understanding a quiet scan.
    size_independently:
        Size every opportunity as though it were the only trade taken. This is
        almost always what a person reading a ranked list wants: "if I take
        this one, how many shares?" The alternative — sizing down the list
        cumulatively, so each entry assumes the ones above it were already
        bought — answers a different question, and makes the fourth-best idea
        report a share count that reflects a depleted account rather than the
        setup. Concurrent capacity is reported separately either way.
    """

    timeframe: str = "1Day"
    max_signal_age_bars: int = 1
    min_score: float = 0.0
    top_n: int = 15
    min_risk_reward: float = 2.0
    require_risk_approval: bool = True
    size_independently: bool = True

    def __post_init__(self) -> None:
        if self.max_signal_age_bars < 1:
            raise ValueError(
                f"max_signal_age_bars must be >= 1, got {self.max_signal_age_bars}"
            )
        if not 0 <= self.min_score <= 100:
            raise ValueError(f"min_score must be within 0-100, got {self.min_score}")
        if self.top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {self.top_n}")


@dataclass(slots=True)
class SweepStage:
    """One stage of the funnel, for reporting where symbols went."""

    name: str
    entered: int
    survived: int
    seconds: float = 0.0

    @property
    def dropped(self) -> int:
        return max(0, self.entered - self.survived)


@dataclass(slots=True)
class MarketSweep:
    """Everything a market-wide sweep produced."""

    opportunities: tuple[Opportunity, ...] = ()
    stages: list[SweepStage] = field(default_factory=list)
    #: Setups that were valid but too old to enter now.
    stale: dict[str, int] = field(default_factory=dict)
    #: Why candidates with a signal failed risk, keyed by check name.
    rejections: dict[str, int] = field(default_factory=dict)
    #: Most common reasons no signal fired at all.
    blockers: dict[str, int] = field(default_factory=dict)
    #: Why candidates that *did* signal were not returned.
    filtered_out: dict[str, int] = field(default_factory=dict)
    #: How many of the returned setups the account can hold at once, given the
    #: risk limits. Share counts are per-trade, so this is what says how many
    #: of them actually fit.
    concurrent_capacity: int = 0
    universe_size: int = 0
    scanned: int = 0
    halt_reason: str | None = None
    as_of: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    elapsed_seconds: float = 0.0

    def summary_lines(self) -> list[str]:
        lines = [
            f"Swept {self.universe_size:,} symbols in {self.elapsed_seconds:,.1f}s",
        ]
        for stage in self.stages:
            lines.append(
                f"  {stage.name:<22} {stage.entered:>6,} → {stage.survived:>6,}"
                f"   ({stage.seconds:,.1f}s)"
            )
        if self.stale:
            total = sum(self.stale.values())
            lines.append(f"  {total:,} valid setup(s) dropped as stale")
        for reason, count in sorted(
            self.filtered_out.items(), key=lambda item: -item[1]
        ):
            lines.append(f"  {count:,} dropped: {reason}")
        lines.append(f"Returned {len(self.opportunities)} opportunity(ies)")
        return lines

    def as_frame(self) -> pd.DataFrame:
        """Opportunities as a table."""
        if not self.opportunities:
            return pd.DataFrame()
        rows = []
        for opportunity in self.opportunities:
            signal = opportunity.signal
            decision = opportunity.decision
            rows.append(
                {
                    "rank": opportunity.rank,
                    "symbol": signal.symbol,
                    "direction": signal.direction.value,
                    "strategy": signal.strategy,
                    "score": round(opportunity.confidence, 1),
                    "entry": signal.entry_price,
                    "stop": signal.stop_loss,
                    "target": signal.take_profit,
                    "risk_reward": round(signal.risk_reward_ratio, 2),
                    "shares": int(decision.shares) if decision else 0,
                    "risk_dollars": (
                        float(decision.risk_amount) if decision else 0.0
                    ),
                }
            )
        return pd.DataFrame(rows)


def signal_age_bars(frame: pd.DataFrame, signal_time: datetime) -> int:
    """How many bars ago the signal's bar was, counting the latest bar as 0."""
    if frame.empty:
        return 0
    stamp = pd.Timestamp(signal_time)
    if frame.index.tz is not None and stamp.tz is None:
        stamp = stamp.tz_localize(frame.index.tz)
    elif frame.index.tz is None and stamp.tz is not None:
        stamp = stamp.tz_localize(None)
    position = int(frame.index.searchsorted(stamp, side="right")) - 1
    return max(0, len(frame) - 1 - position)


def _concurrent_capacity(
    opportunities: list[Opportunity], portfolio, manager: RiskManager
) -> int:
    """How many of these the account could actually hold at the same time.

    Share counts are reported per-trade, which would otherwise imply every
    opportunity can be taken at full size. Walking the list in rank order
    against a portfolio that accumulates each fill says how many really fit.
    """
    from trading_bot.scanner.scanner import _with_position

    working = portfolio
    taken = 0
    for opportunity in opportunities:
        decision = manager.evaluate(opportunity.signal, working)
        if not decision.approved or int(decision.shares) < 1:
            break
        working = _with_position(working, opportunity.signal, decision)
        taken += 1
    return taken


def sweep_market(
    universe: Universe,
    strategies: list[BaseStrategy],
    *,
    portfolio,
    risk_manager: RiskManager | None = None,
    config: HuntConfig | None = None,
    scanner_config: ScannerConfig | None = None,
    progress: Any | None = None,
) -> MarketSweep:
    """Run every strategy over every symbol in ``universe`` and rank the results.

    Parameters
    ----------
    universe:
        Symbols and their daily bars, from :func:`build_universe`. Its frames
        are used directly — this does no fetching of its own.
    strategies:
        Strategies to run. Each symbol takes the first signal produced.
    portfolio:
        Account state the risk manager sizes against.
    progress:
        Optional ``progress(done, total)`` callable.

    Returns
    -------
    MarketSweep
    """
    options = config or HuntConfig()
    started = time.perf_counter()
    stages: list[SweepStage] = []

    frames = universe.frames
    if not frames:
        raise ValueError(
            "The universe carries no bars. Build it with build_universe(), which "
            "keeps the frames its liquidity screen already fetched."
        )

    # Prepare indicators once per symbol, then let every strategy read them.
    #
    # Memory, measured rather than guessed: about 158 MB of interpreter and
    # libraries, plus 0.15 MB per symbol. A 4,000-symbol sweep therefore needs
    # roughly 750 MB, which fits the ~1 GB a small hosted container provides.
    #
    # Releasing each raw frame as its enriched replacement is built was tried
    # and reverted. It saved only 6% — the indicator columns dominate, the raw
    # bars are a small share — and paid for that by emptying the caller's
    # universe as a side effect, which broke four unrelated tests within a
    # minute of existing. Headroom that is not needed is not worth a surprise.
    stage_started = time.perf_counter()
    prepared: dict[str, pd.DataFrame] = {}
    total = len(frames)
    for index, (symbol, frame) in enumerate(frames.items(), start=1):
        try:
            prepared[symbol] = strategies[0].prepare(frame)
        except Exception as error:  # noqa: BLE001 - one symbol must not stop a sweep
            logger.debug("Could not prepare %s: %s", symbol, error)
        if progress is not None and index % 100 == 0:
            progress(index, total)
    stages.append(
        SweepStage("indicators", total, len(prepared), time.perf_counter() - stage_started)
    )

    scanner = MarketScanner(
        strategies,
        risk_manager=risk_manager or RiskManager(),
        config=scanner_config
        or ScannerConfig(
            min_confidence=options.min_score,
            max_results=None,
            # Liquidity was already enforced when the universe was built;
            # re-applying a different floor here would silently disagree with it.
            min_avg_dollar_volume=0.0,
            min_price=0.0,
        ),
    )

    stage_started = time.perf_counter()
    result = scanner.scan(prepared, portfolio=portfolio)
    stages.append(
        SweepStage(
            "strategies + risk",
            len(prepared),
            len(result.opportunities),
            time.perf_counter() - stage_started,
        )
    )

    # Freshness. A valid setup from several bars ago is not an entry any more.
    fresh: list[Opportunity] = []
    stale: dict[str, int] = {}
    for opportunity in result.opportunities:
        frame = prepared.get(opportunity.symbol)
        if frame is None:
            continue
        age = signal_age_bars(frame, opportunity.signal.timestamp)
        if age >= options.max_signal_age_bars:
            stale[opportunity.symbol] = age
            continue
        fresh.append(opportunity)
    stages.append(
        SweepStage("freshness", len(result.opportunities), len(fresh), 0.0)
    )

    # Re-size before filtering, not after. The scanner sizes its ranked list
    # cumulatively, so a candidate far down the list can fail the position-size
    # check purely because the ones above it consumed the account. Judging
    # tradability on that decision discards setups that are perfectly fine on
    # their own — measured at 92 of 101 on a $15k account.
    manager = risk_manager or RiskManager()
    if options.size_independently:
        fresh = [
            Opportunity(
                signal=item.signal,
                confidence=item.confidence,
                factors=item.factors,
                decision=manager.evaluate(item.signal, portfolio),
                rank=item.rank,
            )
            for item in fresh
        ]

    filtered: list[Opportunity] = []
    filtered_out: dict[str, int] = {}
    for opportunity in fresh:
        ratio = opportunity.signal.risk_reward_ratio
        if ratio < options.min_risk_reward:
            key = f"reward:risk below {options.min_risk_reward:g}:1"
            filtered_out[key] = filtered_out.get(key, 0) + 1
            continue
        if options.require_risk_approval and not opportunity.tradable:
            decision = opportunity.decision
            failed = getattr(decision, "failed_checks", None) if decision else None
            key = failed[0].name if failed else "rejected by risk"
            filtered_out[key] = filtered_out.get(key, 0) + 1
            continue
        filtered.append(opportunity)
    stages.append(SweepStage("risk + reward:risk", len(fresh), len(filtered), 0.0))

    ranked = sorted(filtered, key=lambda item: -item.confidence)[: options.top_n]

    # Ranks are assigned after every filter, so #1 is genuinely the best of what
    # survived rather than a leftover position from an earlier ordering.
    ranked = [
        Opportunity(
            signal=item.signal,
            confidence=item.confidence,
            factors=item.factors,
            decision=item.decision,
            rank=position,
        )
        for position, item in enumerate(ranked, start=1)
    ]
    capacity = _concurrent_capacity(ranked, portfolio, manager)

    return MarketSweep(
        opportunities=tuple(ranked),
        stages=stages,
        stale=stale,
        rejections=dict(getattr(result, "rejections", {}) or {}),
        blockers=dict(result.blockers),
        filtered_out=filtered_out,
        concurrent_capacity=capacity,
        universe_size=len(universe),
        scanned=len(prepared),
        halt_reason=result.halt_reason,
        elapsed_seconds=time.perf_counter() - started,
    )
