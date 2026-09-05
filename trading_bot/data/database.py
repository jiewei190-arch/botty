"""SQLite persistence for signals, orders, trades, positions and equity.

Why SQLite: the bot is a single writer with occasional dashboard readers, which
SQLite in WAL mode handles comfortably with zero operational overhead. The schema
avoids SQLite-specific types and the access layer is plain SQL behind repository
classes, so moving to PostgreSQL later means swapping the connection factory
rather than rewriting callers.

Conventions
-----------
* Timestamps are stored as ISO-8601 UTC strings — lexicographically sortable and
  portable across engines.
* Money is stored as REAL. Position sizing works in Decimal and converts at the
  boundary; SQLite has no native decimal type.
* Schema changes go through :data:`MIGRATIONS`, tracked by ``PRAGMA user_version``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    mode          TEXT    NOT NULL,
    strategy      TEXT,
    timeframe     TEXT,
    symbols       TEXT,
    started_at    TEXT    NOT NULL,
    ended_at      TEXT,
    starting_equity REAL,
    ending_equity   REAL,
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS signals (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         INTEGER REFERENCES runs(id) ON DELETE SET NULL,
    ts             TEXT    NOT NULL,
    symbol         TEXT    NOT NULL,
    strategy       TEXT    NOT NULL,
    direction      TEXT    NOT NULL,
    confidence     REAL    NOT NULL DEFAULT 0,
    entry_price    REAL,
    stop_loss      REAL,
    take_profit    REAL,
    risk_reward    REAL,
    reasons        TEXT,
    accepted       INTEGER NOT NULL DEFAULT 0,
    rejection_reason TEXT,
    metadata       TEXT
);
CREATE INDEX IF NOT EXISTS idx_signals_symbol_ts ON signals(symbol, ts);
CREATE INDEX IF NOT EXISTS idx_signals_run ON signals(run_id);

CREATE TABLE IF NOT EXISTS orders (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         INTEGER REFERENCES runs(id) ON DELETE SET NULL,
    signal_id      INTEGER REFERENCES signals(id) ON DELETE SET NULL,
    broker_order_id TEXT UNIQUE,
    client_order_id TEXT,
    ts             TEXT    NOT NULL,
    symbol         TEXT    NOT NULL,
    side           TEXT    NOT NULL,
    qty            REAL    NOT NULL,
    order_type     TEXT    NOT NULL,
    time_in_force  TEXT,
    limit_price    REAL,
    stop_price     REAL,
    filled_qty     REAL    DEFAULT 0,
    filled_avg_price REAL,
    status         TEXT    NOT NULL,
    updated_at     TEXT,
    raw            TEXT
);
CREATE INDEX IF NOT EXISTS idx_orders_symbol_ts ON orders(symbol, ts);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);

CREATE TABLE IF NOT EXISTS trades (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER REFERENCES runs(id) ON DELETE SET NULL,
    signal_id     INTEGER REFERENCES signals(id) ON DELETE SET NULL,
    symbol        TEXT    NOT NULL,
    strategy      TEXT,
    direction     TEXT    NOT NULL,
    qty           REAL    NOT NULL,
    entry_ts      TEXT    NOT NULL,
    entry_price   REAL    NOT NULL,
    exit_ts       TEXT,
    exit_price    REAL,
    stop_loss     REAL,
    take_profit   REAL,
    fees          REAL    NOT NULL DEFAULT 0,
    slippage      REAL    NOT NULL DEFAULT 0,
    gross_pnl     REAL,
    pnl           REAL,
    pnl_pct       REAL,
    r_multiple    REAL,
    bars_held     INTEGER,
    exit_reason   TEXT,
    status        TEXT    NOT NULL DEFAULT 'open',
    metadata      TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_entry_ts ON trades(entry_ts);

CREATE TABLE IF NOT EXISTS positions (
    symbol          TEXT PRIMARY KEY,
    run_id          INTEGER REFERENCES runs(id) ON DELETE SET NULL,
    trade_id        INTEGER REFERENCES trades(id) ON DELETE SET NULL,
    strategy        TEXT,
    direction       TEXT NOT NULL,
    qty             REAL NOT NULL,
    avg_entry_price REAL NOT NULL,
    stop_loss       REAL,
    take_profit     REAL,
    opened_at       TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    metadata        TEXT
);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         INTEGER REFERENCES runs(id) ON DELETE SET NULL,
    ts             TEXT    NOT NULL,
    equity         REAL    NOT NULL,
    cash           REAL,
    unrealized_pnl REAL,
    realized_pnl   REAL,
    open_positions INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_snapshots(ts);

CREATE TABLE IF NOT EXISTS bot_events (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       TEXT NOT NULL,
    level    TEXT NOT NULL,
    category TEXT NOT NULL,
    symbol   TEXT,
    message  TEXT NOT NULL,
    payload  TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON bot_events(ts);
CREATE INDEX IF NOT EXISTS idx_events_category ON bot_events(category);
"""

#: Ordered migrations. Index 0 upgrades user_version 0 -> 1, and so on.
MIGRATIONS: tuple[str, ...] = (SCHEMA_V1,)

SCHEMA_VERSION = len(MIGRATIONS)


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def to_iso(moment: datetime | str | None) -> str | None:
    """Normalise a timestamp for storage."""
    if moment is None:
        return None
    if isinstance(moment, str):
        return moment
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat()


def _dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, default=str)


def _loads(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


class Database:
    """Connection owner and migration runner.

    Repositories are exposed as attributes: ``db.signals``, ``db.trades``,
    ``db.positions``, ``db.equity``, ``db.events``, ``db.runs``.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None

        self.runs = RunRepository(self)
        self.signals = SignalRepository(self)
        self.orders = OrderRepository(self)
        self.trades = TradeRepository(self)
        self.positions = PositionRepository(self)
        self.equity = EquityRepository(self)
        self.events = EventRepository(self)

    # -- connection management ---------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        """Open (once) and return the shared connection."""
        if self._connection is not None:
            return self._connection
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.path,
            check_same_thread=False,  # the dashboard reads from another thread
            timeout=30.0,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA foreign_keys = ON")
        self._connection = connection
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Serialised, atomic unit of work."""
        with self._lock:
            connection = self.connect()
            try:
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def initialize(self) -> int:
        """Apply pending migrations. Returns the resulting schema version."""
        with self.transaction() as connection:
            current = connection.execute("PRAGMA user_version").fetchone()[0]
            for version in range(current, len(MIGRATIONS)):
                logger.info("Applying database migration %d -> %d", version, version + 1)
                connection.executescript(MIGRATIONS[version])
                connection.execute(f"PRAGMA user_version = {version + 1}")
            final = connection.execute("PRAGMA user_version").fetchone()[0]
        logger.info("Database ready at %s (schema v%d)", self.path, final)
        return final

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self.transaction() as connection:
            return connection.execute(sql, params)

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        with self._lock:
            cursor = self.connect().execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def insert(self, table: str, values: dict[str, Any]) -> int:
        """Insert a row, returning its new id."""
        columns = ", ".join(values)
        placeholders = ", ".join("?" for _ in values)
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        with self.transaction() as connection:
            cursor = connection.execute(sql, list(values.values()))
            return int(cursor.lastrowid or 0)

    def update(self, table: str, row_id: int, values: dict[str, Any]) -> None:
        if not values:
            return
        assignments = ", ".join(f"{column} = ?" for column in values)
        sql = f"UPDATE {table} SET {assignments} WHERE id = ?"
        self.execute(sql, [*values.values(), row_id])

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def __enter__(self) -> Database:
        self.initialize()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class _Repository:
    def __init__(self, db: Database) -> None:
        self.db = db


class RunRepository(_Repository):
    """A ``run`` groups everything produced by one bot session or backtest."""

    def start(
        self,
        *,
        mode: str,
        strategy: str | None = None,
        timeframe: str | None = None,
        symbols: Sequence[str] | None = None,
        starting_equity: float | None = None,
        notes: str | None = None,
    ) -> int:
        return self.db.insert(
            "runs",
            {
                "mode": mode,
                "strategy": strategy,
                "timeframe": timeframe,
                "symbols": _dumps(list(symbols) if symbols else None),
                "started_at": utc_now_iso(),
                "starting_equity": starting_equity,
                "notes": notes,
            },
        )

    def finish(self, run_id: int, *, ending_equity: float | None = None) -> None:
        self.db.update(
            "runs", run_id, {"ended_at": utc_now_iso(), "ending_equity": ending_equity}
        )

    def latest(self, mode: str | None = None) -> dict[str, Any] | None:
        if mode:
            return self.db.query_one(
                "SELECT * FROM runs WHERE mode = ? ORDER BY id DESC LIMIT 1", (mode,)
            )
        return self.db.query_one("SELECT * FROM runs ORDER BY id DESC LIMIT 1")


class SignalRepository(_Repository):
    """Every signal is recorded — including rejected ones, which are the most
    valuable rows when diagnosing why the bot did not trade."""

    def record(
        self,
        *,
        symbol: str,
        strategy: str,
        direction: str,
        confidence: float,
        ts: datetime | str | None = None,
        run_id: int | None = None,
        entry_price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        risk_reward: float | None = None,
        reasons: Sequence[str] | None = None,
        accepted: bool = False,
        rejection_reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        return self.db.insert(
            "signals",
            {
                "run_id": run_id,
                "ts": to_iso(ts) or utc_now_iso(),
                "symbol": symbol.upper(),
                "strategy": strategy,
                "direction": direction.upper(),
                "confidence": float(confidence),
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_reward": risk_reward,
                "reasons": _dumps(list(reasons) if reasons else None),
                "accepted": int(accepted),
                "rejection_reason": rejection_reason,
                "metadata": _dumps(metadata),
            },
        )

    def recent(self, limit: int = 50, *, symbol: str | None = None) -> list[dict[str, Any]]:
        if symbol:
            rows = self.db.query(
                "SELECT * FROM signals WHERE symbol = ? ORDER BY ts DESC LIMIT ?",
                (symbol.upper(), limit),
            )
        else:
            rows = self.db.query("SELECT * FROM signals ORDER BY ts DESC LIMIT ?", (limit,))
        for row in rows:
            row["reasons"] = _loads(row.get("reasons")) or []
            row["metadata"] = _loads(row.get("metadata"))
        return rows


class OrderRepository(_Repository):
    """Broker order lifecycle."""

    def record(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        order_type: str,
        status: str,
        ts: datetime | str | None = None,
        run_id: int | None = None,
        signal_id: int | None = None,
        broker_order_id: str | None = None,
        client_order_id: str | None = None,
        time_in_force: str | None = None,
        limit_price: float | None = None,
        stop_price: float | None = None,
        raw: dict[str, Any] | None = None,
    ) -> int:
        return self.db.insert(
            "orders",
            {
                "run_id": run_id,
                "signal_id": signal_id,
                "broker_order_id": broker_order_id,
                "client_order_id": client_order_id,
                "ts": to_iso(ts) or utc_now_iso(),
                "symbol": symbol.upper(),
                "side": side.lower(),
                "qty": float(qty),
                "order_type": order_type.lower(),
                "time_in_force": time_in_force,
                "limit_price": limit_price,
                "stop_price": stop_price,
                "status": status,
                "updated_at": utc_now_iso(),
                "raw": _dumps(raw),
            },
        )

    def update_status(
        self,
        broker_order_id: str,
        *,
        status: str,
        filled_qty: float | None = None,
        filled_avg_price: float | None = None,
    ) -> None:
        self.db.execute(
            """
            UPDATE orders
               SET status = ?,
                   filled_qty = COALESCE(?, filled_qty),
                   filled_avg_price = COALESCE(?, filled_avg_price),
                   updated_at = ?
             WHERE broker_order_id = ?
            """,
            (status, filled_qty, filled_avg_price, utc_now_iso(), broker_order_id),
        )

    def open_orders(self) -> list[dict[str, Any]]:
        return self.db.query(
            "SELECT * FROM orders WHERE status NOT IN "
            "('filled', 'canceled', 'expired', 'rejected') ORDER BY ts DESC"
        )


class TradeRepository(_Repository):
    """Round-trip trades: one row opened at entry, completed at exit."""

    def open_trade(
        self,
        *,
        symbol: str,
        direction: str,
        qty: float,
        entry_price: float,
        entry_ts: datetime | str | None = None,
        run_id: int | None = None,
        signal_id: int | None = None,
        strategy: str | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        fees: float = 0.0,
        slippage: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        return self.db.insert(
            "trades",
            {
                "run_id": run_id,
                "signal_id": signal_id,
                "symbol": symbol.upper(),
                "strategy": strategy,
                "direction": direction.upper(),
                "qty": float(qty),
                "entry_ts": to_iso(entry_ts) or utc_now_iso(),
                "entry_price": float(entry_price),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "fees": fees,
                "slippage": slippage,
                "status": "open",
                "metadata": _dumps(metadata),
            },
        )

    def close_trade(
        self,
        trade_id: int,
        *,
        exit_price: float,
        exit_ts: datetime | str | None = None,
        exit_reason: str | None = None,
        fees: float = 0.0,
        slippage: float = 0.0,
        bars_held: int | None = None,
    ) -> dict[str, Any] | None:
        """Close a trade and compute realised P&L, return % and R-multiple."""
        trade = self.db.query_one("SELECT * FROM trades WHERE id = ?", (trade_id,))
        if trade is None:
            logger.warning("close_trade called for unknown trade id %s", trade_id)
            return None

        direction = str(trade["direction"]).upper()
        qty = float(trade["qty"])
        entry = float(trade["entry_price"])
        sign = 1.0 if direction in ("LONG", "BUY") else -1.0
        gross = (float(exit_price) - entry) * qty * sign
        total_fees = float(trade["fees"] or 0.0) + fees
        total_slippage = float(trade["slippage"] or 0.0) + slippage
        net = gross - total_fees
        cost_basis = entry * qty
        pnl_pct = (net / cost_basis * 100) if cost_basis else 0.0

        r_multiple = None
        stop = trade["stop_loss"]
        if stop:
            risk_per_share = abs(entry - float(stop))
            if risk_per_share > 0:
                r_multiple = (float(exit_price) - entry) * sign / risk_per_share

        self.db.update(
            "trades",
            trade_id,
            {
                "exit_ts": to_iso(exit_ts) or utc_now_iso(),
                "exit_price": float(exit_price),
                "exit_reason": exit_reason,
                "fees": total_fees,
                "slippage": total_slippage,
                "gross_pnl": gross,
                "pnl": net,
                "pnl_pct": pnl_pct,
                "r_multiple": r_multiple,
                "bars_held": bars_held,
                "status": "closed",
            },
        )
        return self.db.query_one("SELECT * FROM trades WHERE id = ?", (trade_id,))

    def open_trades(self) -> list[dict[str, Any]]:
        return self.db.query("SELECT * FROM trades WHERE status = 'open' ORDER BY entry_ts")

    def history(self, limit: int = 100, *, symbol: str | None = None) -> list[dict[str, Any]]:
        if symbol:
            return self.db.query(
                "SELECT * FROM trades WHERE status = 'closed' AND symbol = ? "
                "ORDER BY exit_ts DESC LIMIT ?",
                (symbol.upper(), limit),
            )
        return self.db.query(
            "SELECT * FROM trades WHERE status = 'closed' ORDER BY exit_ts DESC LIMIT ?",
            (limit,),
        )

    def statistics(self, *, run_id: int | None = None) -> dict[str, Any]:
        """Aggregate performance over closed trades."""
        clause = "WHERE status = 'closed'"
        params: list[Any] = []
        if run_id is not None:
            clause += " AND run_id = ?"
            params.append(run_id)

        row = self.db.query_one(
            f"""
            SELECT COUNT(*)                                   AS total_trades,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)   AS wins,
                   SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END)  AS losses,
                   COALESCE(SUM(pnl), 0)                      AS total_pnl,
                   COALESCE(AVG(CASE WHEN pnl > 0 THEN pnl END), 0)  AS avg_win,
                   COALESCE(AVG(CASE WHEN pnl <= 0 THEN pnl END), 0) AS avg_loss,
                   COALESCE(SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END), 0)      AS gross_profit,
                   COALESCE(SUM(CASE WHEN pnl <= 0 THEN -pnl ELSE 0 END), 0)    AS gross_loss,
                   COALESCE(AVG(r_multiple), 0)               AS avg_r
              FROM trades {clause}
            """,
            params,
        ) or {}

        total = int(row.get("total_trades") or 0)
        wins = int(row.get("wins") or 0)
        gross_loss = float(row.get("gross_loss") or 0.0)
        gross_profit = float(row.get("gross_profit") or 0.0)
        return {
            "total_trades": total,
            "wins": wins,
            "losses": int(row.get("losses") or 0),
            "win_rate": (wins / total * 100) if total else 0.0,
            "total_pnl": float(row.get("total_pnl") or 0.0),
            "avg_win": float(row.get("avg_win") or 0.0),
            "avg_loss": float(row.get("avg_loss") or 0.0),
            "avg_r_multiple": float(row.get("avg_r") or 0.0),
            "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
            if gross_profit > 0
            else 0.0,
        }

    def realized_pnl_since(self, since: datetime | str) -> float:
        """Realised P&L for trades closed at or after ``since`` — the daily loss guard."""
        row = self.db.query_one(
            "SELECT COALESCE(SUM(pnl), 0) AS pnl FROM trades "
            "WHERE status = 'closed' AND exit_ts >= ?",
            (to_iso(since),),
        )
        return float(row["pnl"]) if row else 0.0

    def consecutive_losses(self, limit: int = 10) -> int:
        """Length of the current losing streak — drives the cooldown rule."""
        rows = self.db.query(
            "SELECT pnl FROM trades WHERE status = 'closed' ORDER BY exit_ts DESC LIMIT ?",
            (limit,),
        )
        streak = 0
        for row in rows:
            if (row["pnl"] or 0) < 0:
                streak += 1
            else:
                break
        return streak


class PositionRepository(_Repository):
    """Bot-side view of open positions, reconciled against the broker."""

    def upsert(
        self,
        *,
        symbol: str,
        direction: str,
        qty: float,
        avg_entry_price: float,
        run_id: int | None = None,
        trade_id: int | None = None,
        strategy: str | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        opened_at: datetime | str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = utc_now_iso()
        self.db.execute(
            """
            INSERT INTO positions (symbol, run_id, trade_id, strategy, direction, qty,
                                   avg_entry_price, stop_loss, take_profit,
                                   opened_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                run_id = excluded.run_id,
                trade_id = excluded.trade_id,
                strategy = excluded.strategy,
                direction = excluded.direction,
                qty = excluded.qty,
                avg_entry_price = excluded.avg_entry_price,
                stop_loss = excluded.stop_loss,
                take_profit = excluded.take_profit,
                updated_at = excluded.updated_at,
                metadata = excluded.metadata
            """,
            (
                symbol.upper(), run_id, trade_id, strategy, direction.upper(), float(qty),
                float(avg_entry_price), stop_loss, take_profit,
                to_iso(opened_at) or now, now, _dumps(metadata),
            ),
        )

    def remove(self, symbol: str) -> None:
        self.db.execute("DELETE FROM positions WHERE symbol = ?", (symbol.upper(),))

    def all(self) -> list[dict[str, Any]]:
        return self.db.query("SELECT * FROM positions ORDER BY symbol")

    def get(self, symbol: str) -> dict[str, Any] | None:
        return self.db.query_one("SELECT * FROM positions WHERE symbol = ?", (symbol.upper(),))

    def count(self) -> int:
        row = self.db.query_one("SELECT COUNT(*) AS n FROM positions")
        return int(row["n"]) if row else 0


class EquityRepository(_Repository):
    """Equity curve samples, used for drawdown and Sharpe calculations."""

    def record(
        self,
        *,
        equity: float,
        ts: datetime | str | None = None,
        run_id: int | None = None,
        cash: float | None = None,
        unrealized_pnl: float | None = None,
        realized_pnl: float | None = None,
        open_positions: int = 0,
    ) -> int:
        return self.db.insert(
            "equity_snapshots",
            {
                "run_id": run_id,
                "ts": to_iso(ts) or utc_now_iso(),
                "equity": float(equity),
                "cash": cash,
                "unrealized_pnl": unrealized_pnl,
                "realized_pnl": realized_pnl,
                "open_positions": open_positions,
            },
        )

    def curve(self, *, run_id: int | None = None, limit: int = 5000) -> list[dict[str, Any]]:
        if run_id is not None:
            return self.db.query(
                "SELECT * FROM equity_snapshots WHERE run_id = ? ORDER BY ts LIMIT ?",
                (run_id, limit),
            )
        return self.db.query("SELECT * FROM equity_snapshots ORDER BY ts LIMIT ?", (limit,))

    def latest(self) -> dict[str, Any] | None:
        return self.db.query_one("SELECT * FROM equity_snapshots ORDER BY ts DESC LIMIT 1")


class EventRepository(_Repository):
    """Durable audit trail of bot decisions, queryable from the dashboard."""

    def record(
        self,
        *,
        category: str,
        message: str,
        level: str = "INFO",
        symbol: str | None = None,
        payload: dict[str, Any] | None = None,
        ts: datetime | str | None = None,
    ) -> int:
        return self.db.insert(
            "bot_events",
            {
                "ts": to_iso(ts) or utc_now_iso(),
                "level": level.upper(),
                "category": category,
                "symbol": symbol.upper() if symbol else None,
                "message": message,
                "payload": _dumps(payload),
            },
        )

    def recent(
        self, limit: int = 100, *, category: str | None = None, level: str | None = None
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if category:
            clauses.append("category = ?")
            params.append(category)
        if level:
            clauses.append("level = ?")
            params.append(level.upper())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self.db.query(f"SELECT * FROM bot_events {where} ORDER BY ts DESC LIMIT ?", params)
        for row in rows:
            row["payload"] = _loads(row.get("payload"))
        return rows
