"""The backtesting engine.

Walks bars one at a time, feeding each strategy only the history that existed at
that moment, and simulates an account through the same risk manager the live bot
uses.

The bar loop, and why the order matters
---------------------------------------
For each bar ``i``:

1. **Fill pending entries** at bar ``i``'s *open*. Those entries came from
   signals generated on bar ``i-1``'s close — you cannot trade at a price you
   have only just observed.
2. **Check exits** for every open position against bar ``i``'s range, including
   positions opened moments ago at this bar's open. When a bar spans both stop
   and target the stop wins, because the intrabar path is unknowable and
   assuming the favourable order is how a backtest inflates itself.
3. **Mark to market** on bar ``i``'s close.
4. **Generate signals** from history up to and including bar ``i``, queued for
   bar ``i+1``.

Step 4 comes last and its results are not actionable until the next iteration.
That is what makes lookahead structurally impossible here rather than merely
avoided: the engine has no way to act on a bar it has not finished.

What is simulated
-----------------
Cash and positions, marked to market every bar. Every entry passes through the
real :class:`~trading_bot.risk.RiskManager`, so position slots, exposure caps,
the daily loss limit and the losing-streak cooldown all apply exactly as they
would live. Realised P&L and the losing streak are tracked across the run, and
the daily loss limit resets on each new calendar day.

What is not
-----------
Partial fills, order queue position, borrow availability for shorts, and
after-hours or halted sessions. Each would make results *worse*, never better —
the model errs pessimistic, which is the right direction for a tool whose job is
to stop you trading something that does not work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import pandas as pd

from trading_bot.backtesting.execution import CostModel, Fill, FillModel, FillReason
from trading_bot.backtesting.metrics import (
    PerformanceMetrics,
    calculate_metrics,
    drawdown_curve,
)
from trading_bot.config.settings import RiskSettings
from trading_bot.indicators import IndicatorConfig
from trading_bot.risk import OpenPosition, PortfolioState, RiskManager
from trading_bot.risk.position_sizing import to_decimal
from trading_bot.strategies import (
    BaseStrategy,
    ExitReason,
    Position,
    Signal,
    SignalDirection,
)
from trading_bot.utils.timeframes import Timeframe

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """How the simulation is run."""

    starting_equity: float = 10_000.0
    timeframe: str = "15Min"
    costs: CostModel = field(default_factory=CostModel)
    risk: RiskSettings = field(default_factory=RiskSettings)
    #: Close any position still open when the data ends. Leaving them open would
    #: let an unrealised loss escape the results.
    close_at_end: bool = True
    #: Warm-up bars excluded from the results. Defaults to each strategy's own
    #: requirement, which is normally what you want.
    warmup_bars: int | None = None

    def __post_init__(self) -> None:
        if self.starting_equity <= 0:
            raise ValueError(
                f"starting_equity must be positive, got {self.starting_equity}"
            )
        Timeframe.parse(self.timeframe)


@dataclass(slots=True)
class SimulatedPosition:
    """A position held by the simulation."""

    symbol: str
    signal: Signal
    quantity: float
    entry_fill: Fill
    entry_index: int
    entry_timestamp: pd.Timestamp
    stop_loss: float
    take_profit: float
    strategy: str
    bars_held: int = 0

    @property
    def direction(self) -> SignalDirection:
        return self.signal.direction

    def market_value(self, price: float) -> float:
        return abs(self.quantity) * price

    def unrealised(self, price: float) -> float:
        return (price - self.entry_fill.price) * self.quantity * self.direction.sign

    def to_strategy_position(self) -> Position:
        """The view a strategy's exit logic expects."""
        return Position(
            symbol=self.symbol,
            direction=self.direction,
            entry_price=self.entry_fill.price,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            quantity=self.quantity,
            entry_timestamp=self.entry_timestamp.to_pydatetime(),
            strategy=self.strategy,
            bars_held=self.bars_held,
            metadata=dict(self.signal.metadata),
        )


@dataclass(slots=True)
class BacktestResult:
    """Everything one backtest produced."""

    metrics: PerformanceMetrics
    equity_curve: pd.Series
    drawdown: pd.Series
    trades: list[dict[str, Any]]
    config: BacktestConfig
    symbols: tuple[str, ...] = ()
    strategies: tuple[str, ...] = ()
    signals_generated: int = 0
    signals_rejected: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def trade_frame(self) -> pd.DataFrame:
        """Trades as a DataFrame, for tables and charts."""
        return pd.DataFrame(self.trades)

    def summary(self) -> str:
        lines = [
            "=" * 68,
            f"BACKTEST — {', '.join(self.symbols)} · {', '.join(self.strategies)}",
            f"{self.config.timeframe} bars · {self.metrics.bars} bars simulated",
            "=" * 68,
            "",
            *self.metrics.summary_lines(),
        ]
        if self.signals_rejected:
            lines.extend(
                [
                    "",
                    f"Signals generated : {self.signals_generated} "
                    f"({self.signals_rejected} rejected by risk)",
                    "Rejected by:",
                ]
            )
            for reason, count in sorted(
                self.rejection_reasons.items(), key=lambda item: -item[1]
            )[:6]:
                lines.append(f"  {count:>4}  {reason}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics.as_dict(),
            "symbols": list(self.symbols),
            "strategies": list(self.strategies),
            "signals_generated": self.signals_generated,
            "signals_rejected": self.signals_rejected,
            "rejection_reasons": dict(self.rejection_reasons),
            "trades": self.trades,
        }


class Backtester:
    """Candle-by-candle simulation over one or more symbols.

    Example
    -------
    >>> tester = Backtester([build_strategy("momentum")], BacktestConfig())
    >>> result = tester.run({"AAPL": bars})
    >>> print(result.summary())
    """

    def __init__(
        self,
        strategies: list[BaseStrategy],
        config: BacktestConfig | None = None,
        indicators: IndicatorConfig | None = None,
    ) -> None:
        if not strategies:
            raise ValueError("A backtest needs at least one strategy")
        self.strategies = strategies
        self.config = config or BacktestConfig()
        self.indicators = indicators or IndicatorConfig()
        self.fills = FillModel(self.config.costs)
        self.risk = RiskManager(self.config.risk)

    # -- state -------------------------------------------------------------------

    def _reset(self) -> None:
        self._cash = float(self.config.starting_equity)
        self._positions: dict[str, SimulatedPosition] = {}
        self._pending: list[tuple[str, Signal]] = []
        self._pending_exits: dict[str, str] = {}
        self._trades: list[dict[str, Any]] = []
        self._equity: list[float] = []
        self._timestamps: list[pd.Timestamp] = []
        self._bars_in_market = 0
        self._realised_today = 0.0
        self._current_day: date | None = None
        self._consecutive_losses = 0
        self._last_loss_at: datetime | None = None
        self._signals = 0
        self._rejected = 0
        self._rejections: dict[str, int] = {}

    def _equity_now(self, prices: dict[str, float]) -> float:
        held = sum(
            position.market_value(prices.get(symbol, position.entry_fill.price))
            for symbol, position in self._positions.items()
        )
        unrealised = sum(
            position.unrealised(prices.get(symbol, position.entry_fill.price))
            for symbol, position in self._positions.items()
        )
        # Cash already excludes the cost basis, so equity is cash plus what the
        # positions are worth: basis plus their unrealised move.
        basis = sum(
            abs(position.quantity) * position.entry_fill.price
            for position in self._positions.values()
        )
        del held
        return self._cash + basis + unrealised

    def _portfolio_state(self, prices: dict[str, float]) -> PortfolioState:
        """The state the risk manager judges, built from the simulation."""
        equity = self._equity_now(prices)
        positions = tuple(
            OpenPosition(
                symbol=symbol,
                direction=position.direction.value,
                quantity=to_decimal(position.quantity),
                entry_price=to_decimal(position.entry_fill.price),
                current_price=to_decimal(
                    prices.get(symbol, position.entry_fill.price)
                ),
                stop_loss=to_decimal(position.stop_loss),
                take_profit=to_decimal(position.take_profit),
                strategy=position.strategy,
            )
            for symbol, position in self._positions.items()
        )
        return PortfolioState(
            equity=to_decimal(equity),
            cash=to_decimal(self._cash),
            buying_power=to_decimal(max(self._cash, 0.0)),
            positions=positions,
            realized_pnl_today=to_decimal(self._realised_today),
            session_start_equity=to_decimal(equity - self._realised_today),
            consecutive_losses=self._consecutive_losses,
            last_loss_at=self._last_loss_at,
        )

    # -- the run -----------------------------------------------------------------

    def run(self, frames: dict[str, pd.DataFrame]) -> BacktestResult:
        """Simulate the strategies over ``frames``.

        Parameters
        ----------
        frames:
            Symbol to OHLCV frame. Indicators are computed if absent.

        Returns
        -------
        BacktestResult
        """
        if not frames:
            raise ValueError("No data to backtest")

        self._reset()
        timeframe = Timeframe.parse(self.config.timeframe)
        prepared = {
            symbol: self.strategies[0].prepare(frame) for symbol, frame in frames.items()
        }

        timeline = sorted(
            set().union(*(frame.index for frame in prepared.values()))
        )
        warmup = self.config.warmup_bars
        if warmup is None:
            warmup = max(strategy.min_bars for strategy in self.strategies)

        positions_by_index = {
            symbol: {stamp: index for index, stamp in enumerate(frame.index)}
            for symbol, frame in prepared.items()
        }

        for step, stamp in enumerate(timeline):
            bars = {
                symbol: frame.loc[stamp]
                for symbol, frame in prepared.items()
                if stamp in positions_by_index[symbol]
            }
            if not bars:
                continue

            self._roll_day(stamp)

            # 1. Exits queued on the previous bar fill at this bar's open,
            #    before anything else competes for the cash.
            self._fill_pending_exits(bars, stamp)

            # 2. Entries queued on the previous bar fill at this bar's open.
            self._fill_pending(bars, stamp, step)

            # 3. Protective exits: stops and targets the bar traded through,
            #    including for a position opened at this same open.
            self._process_exits(bars, stamp)

            # 4. Mark to market. Warm-up bars are excluded from the record: no
            #    position can exist yet, so they would contribute a run of flat
            #    returns that pads the bar count, understates exposure, and drags
            #    the Sharpe ratio toward zero. The results describe the period the
            #    strategy could actually trade.
            closes = {symbol: float(bar["close"]) for symbol, bar in bars.items()}
            if step >= warmup:
                self._equity.append(self._equity_now(closes))
                self._timestamps.append(stamp)
                if self._positions:
                    self._bars_in_market += 1
            for position in self._positions.values():
                position.bars_held += 1

            # 5. Decide what to do on the *next* bar. Both the exit checks and
            #    the entry signals read this bar's close, so neither can act
            #    until the next bar opens.
            self._queue_exits(prepared, positions_by_index, stamp)
            if step >= warmup:
                self._generate(prepared, positions_by_index, stamp, closes)

        if self.config.close_at_end:
            self._close_remaining(prepared, timeline)

        equity_curve = pd.Series(
            self._equity, index=pd.DatetimeIndex(self._timestamps), dtype="float64"
        )
        metrics = calculate_metrics(
            equity_curve,
            self._trades,
            timeframe=timeframe,
            starting_equity=float(self.config.starting_equity),
            bars_in_market=self._bars_in_market,
        )
        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_curve,
            drawdown=drawdown_curve(equity_curve),
            trades=self._trades,
            config=self.config,
            symbols=tuple(sorted(frames)),
            strategies=tuple(strategy.name for strategy in self.strategies),
            signals_generated=self._signals,
            signals_rejected=self._rejected,
            rejection_reasons=self._rejections,
        )

    # -- steps -------------------------------------------------------------------

    def _roll_day(self, stamp: pd.Timestamp) -> None:
        """Reset the daily loss counter when the calendar day changes."""
        day = stamp.date()
        if self._current_day is None:
            self._current_day = day
        elif day != self._current_day:
            self._current_day = day
            self._realised_today = 0.0

    def _fill_pending(
        self, bars: dict[str, Any], stamp: pd.Timestamp, step: int
    ) -> None:
        """Open positions queued on the previous bar, at this bar's open."""
        queued, self._pending = self._pending, []
        for symbol, signal in queued:
            bar = bars.get(symbol)
            if bar is None:
                continue  # symbol has no bar here; the intent expires
            if symbol in self._positions:
                continue

            closes = {name: float(row["close"]) for name, row in bars.items()}
            decision = self.risk.evaluate(
                signal, self._portfolio_state(closes), now=stamp.to_pydatetime()
            )
            if not decision.approved:
                self._rejected += 1
                # Group by the check that failed. Grouping by message would
                # produce a near-unique key per rejection, since the messages
                # carry prices and share counts.
                failed = decision.failed_checks
                reason = failed[0].name if failed else "rejected"
                self._rejections[reason] = self._rejections.get(reason, 0) + 1
                continue

            quantity = float(decision.quantity)
            fill = self.fills.entry_fill(bar, signal.direction, quantity)
            cost = fill.notional + fill.commission
            if cost > self._cash:
                self._rejections["insufficient cash at fill"] = (
                    self._rejections.get("insufficient cash at fill", 0) + 1
                )
                self._rejected += 1
                continue

            self._cash -= cost
            self._positions[symbol] = SimulatedPosition(
                symbol=symbol,
                signal=signal,
                quantity=quantity,
                entry_fill=fill,
                entry_index=step,
                entry_timestamp=stamp,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                strategy=signal.strategy,
            )

    def _fill_pending_exits(self, bars: dict[str, Any], stamp: pd.Timestamp) -> None:
        """Close positions whose strategy asked to exit on the previous bar.

        A discretionary exit is decided from a bar's close, so it cannot be
        filled at that same bar's open — that would be selling at a price that
        precedes the information behind the decision. It fills here, at the next
        bar's open, exactly as an entry does.
        """
        queued, self._pending_exits = self._pending_exits, {}
        for symbol, reason in queued.items():
            position = self._positions.get(symbol)
            bar = bars.get(symbol)
            if position is None:
                continue  # a stop or target got there first
            if bar is None:
                self._pending_exits[symbol] = reason  # no bar here; try the next
                continue
            fill = self.fills.exit_fill(
                bar, position.direction, position.quantity, FillReason.SIGNAL_EXIT
            )
            self._close(symbol, fill, stamp, reason)

    def _process_exits(self, bars: dict[str, Any], stamp: pd.Timestamp) -> None:
        """Close positions whose stop or target was touched during the bar.

        Only the protective exits belong here. They are triggered by prices the
        bar actually traded through, so acting within the bar is what really
        happens. Discretionary exits are queued instead — see
        :meth:`_queue_exits`.
        """
        for symbol in list(self._positions):
            position = self._positions[symbol]
            bar = bars.get(symbol)
            if bar is None:
                continue

            high, low = float(bar["high"]), float(bar["low"])
            is_long = position.direction is SignalDirection.LONG
            hit_stop = low <= position.stop_loss if is_long else high >= position.stop_loss
            hit_target = (
                high >= position.take_profit if is_long else low <= position.take_profit
            )

            if hit_stop:
                # Stop wins an ambiguous bar: the intrabar path is unknowable and
                # assuming the good outcome is how a backtest flatters itself.
                fill = self.fills.stop_fill(
                    bar, position.direction, position.quantity, position.stop_loss
                )
                self._close(symbol, fill, stamp, ExitReason.STOP_LOSS.value)
                self._pending_exits.pop(symbol, None)
                continue
            if hit_target:
                fill = self.fills.target_fill(
                    bar, position.direction, position.quantity, position.take_profit
                )
                self._close(symbol, fill, stamp, ExitReason.TAKE_PROFIT.value)
                self._pending_exits.pop(symbol, None)

    def _queue_exits(
        self,
        prepared: dict[str, pd.DataFrame],
        index_maps: dict[str, dict],
        stamp: pd.Timestamp,
    ) -> None:
        """Ask each strategy whether to exit, and queue it for the next bar."""
        for symbol in list(self._positions):
            if symbol in self._pending_exits:
                continue
            position = self._positions[symbol]
            row = index_maps[symbol].get(stamp)
            if row is None:
                continue
            history = prepared[symbol].iloc[: row + 1]
            for strategy in self.strategies:
                if strategy.name != position.strategy:
                    continue
                exit_signal = strategy.evaluate_exit(
                    position.to_strategy_position(), history
                )
                if exit_signal is not None:
                    self._pending_exits[symbol] = exit_signal.reason.value
                    break

    def _generate(
        self,
        prepared: dict[str, pd.DataFrame],
        index_maps: dict[str, dict],
        stamp: pd.Timestamp,
        closes: dict[str, float],
    ) -> None:
        """Queue signals from history up to this bar, for the next one."""
        halt = self.risk.trading_halted(
            self._portfolio_state(closes), now=stamp.to_pydatetime()
        )
        if halt is not None:
            return

        for symbol, frame in prepared.items():
            if symbol in self._positions:
                continue
            row = index_maps[symbol].get(stamp)
            if row is None:
                continue
            history = frame.iloc[: row + 1]

            for strategy in self.strategies:
                try:
                    signal = strategy.generate_signal(symbol, history)
                except Exception:  # noqa: BLE001 - one symbol must not stop the run
                    logger.exception("%s failed on %s at %s", strategy.name, symbol, stamp)
                    continue
                if signal is not None:
                    self._signals += 1
                    self._pending.append((symbol, signal))
                    break  # one position per symbol; first strategy wins

    def _close(
        self, symbol: str, fill: Fill, stamp: pd.Timestamp, reason: str
    ) -> None:
        """Book a closed trade and return its proceeds to cash."""
        position = self._positions.pop(symbol)
        proceeds = fill.notional - fill.commission
        self._cash += proceeds

        entry_value = position.entry_fill.price * abs(position.quantity)
        gross = (
            (fill.price - position.entry_fill.price)
            * position.quantity
            * position.direction.sign
        )
        commission = position.entry_fill.commission + fill.commission
        slippage = position.entry_fill.slippage_cost + fill.slippage_cost
        net = gross - commission

        risk_per_share = abs(position.entry_fill.price - position.stop_loss)
        r_multiple = (
            (fill.price - position.entry_fill.price)
            * position.direction.sign
            / risk_per_share
            if risk_per_share > 0
            else None
        )

        self._realised_today += net
        if net < 0:
            self._consecutive_losses += 1
            self._last_loss_at = stamp.to_pydatetime()
        else:
            self._consecutive_losses = 0

        self._trades.append(
            {
                "symbol": symbol,
                "strategy": position.strategy,
                "direction": position.direction.value,
                "quantity": position.quantity,
                "entry_time": position.entry_timestamp,
                "entry_price": position.entry_fill.price,
                "exit_time": stamp,
                "exit_price": fill.price,
                "stop_loss": position.stop_loss,
                "take_profit": position.take_profit,
                "gross_pnl": gross,
                "commission": commission,
                "slippage": slippage,
                "pnl": net,
                "pnl_pct": (net / entry_value * 100) if entry_value else 0.0,
                "r_multiple": r_multiple,
                "bars_held": position.bars_held,
                "exit_reason": reason,
                # The score this setup carried when it was entered. Without it
                # a backtest can say how the strategy did overall but not
                # whether its own ranking meant anything, which is the question
                # that decides if the ranking is worth reading.
                "confidence": position.signal.confidence,
                "gapped": fill.gapped,
            }
        )

    def _close_remaining(
        self, prepared: dict[str, pd.DataFrame], timeline: list[pd.Timestamp]
    ) -> None:
        """Liquidate anything still open when the data runs out.

        Leaving positions open would let an unrealised loss escape the results.
        """
        if not timeline:
            return
        last = timeline[-1]
        for symbol in list(self._positions):
            frame = prepared[symbol]
            if last not in frame.index:
                continue
            bar = frame.loc[last]
            position = self._positions[symbol]
            price = float(bar["close"])
            fill = Fill(
                price=self.fills.costs.slip(
                    price, position.direction, is_entry=False
                ),
                quantity=position.quantity,
                commission=self.fills.costs.commission(position.quantity, price),
                reason=FillReason.END_OF_DATA,
            )
            self._close(symbol, fill, last, FillReason.END_OF_DATA.value)
