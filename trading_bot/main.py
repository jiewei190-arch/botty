"""Command-line entry point.

Phase 1 commands::

    python main.py config          # show resolved configuration (secrets masked)
    python main.py check           # verify credentials, broker and data connectivity
    python main.py clock           # market session state
    python main.py fetch           # download and inspect bars
    python main.py db-init         # create or migrate the database
    python main.py cache           # inspect or clear the bar cache

Later phases add ``scan``, ``backtest``, ``run`` and ``dashboard``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

# Allow `python main.py` from the repository root without installing the package.
if __package__ in (None, ""):  # pragma: no cover - script execution path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import ValidationError

from trading_bot import __version__
from trading_bot.config.settings import Settings, TradingMode, load_settings
from trading_bot.data.cache import BarCache
from trading_bot.data.database import Database
from trading_bot.data.market_data import (
    MarketDataError,
    build_market_data,
    drop_incomplete_bars,
)
from trading_bot.utils.logging_setup import configure_logging, log_banner
from trading_bot.utils.timeframes import SUPPORTED_TIMEFRAMES, Timeframe

logger = logging.getLogger("trading_bot.cli")

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_CONFIG_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trading-bot",
        description="Modular algorithmic trading bot (backtest / paper / live).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"trading-bot {__version__}")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override LOG_LEVEL for this invocation.",
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in TradingMode],
        help="Override TRADING_MODE for this invocation.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("config", help="Print resolved configuration (secrets masked).")
    subparsers.add_parser("check", help="Run connectivity and configuration health checks.")
    subparsers.add_parser("clock", help="Show current market session state.")
    subparsers.add_parser("db-init", help="Create or migrate the SQLite database.")

    fetch = subparsers.add_parser("fetch", help="Download historical bars.")
    fetch.add_argument("--symbols", help="Comma-separated symbols (default: watchlist).")
    fetch.add_argument(
        "--timeframe",
        default=None,
        help=f"Bar size (default: DATA_TIMEFRAME). Examples: {', '.join(SUPPORTED_TIMEFRAMES)}",
    )
    fetch.add_argument("--start", help="Start date (YYYY-MM-DD).")
    fetch.add_argument("--end", help="End date (YYYY-MM-DD).")
    fetch.add_argument("--limit", type=int, default=None, help="Max bars per symbol.")
    fetch.add_argument("--no-cache", action="store_true", help="Bypass the parquet cache.")
    fetch.add_argument("--csv-dir", help="Also write each symbol to CSV in this directory.")
    fetch.add_argument("--tail", type=int, default=5, help="Rows to preview (default: 5).")

    cache = subparsers.add_parser("cache", help="Inspect or clear the bar cache.")
    cache.add_argument("--clear", action="store_true", help="Delete cached files.")
    cache.add_argument("--symbol", help="Limit --clear to one symbol.")

    return parser


def _load(args: argparse.Namespace) -> Settings:
    """Load settings, applying CLI overrides."""
    overrides: dict[str, object] = {}
    if args.mode:
        overrides["trading_mode"] = TradingMode(args.mode)
    settings = load_settings(**overrides)
    if args.log_level:
        settings = settings.with_overrides(
            logging=settings.logging.model_copy(update={"level": args.log_level})
        )
    settings.ensure_directories()
    configure_logging(
        level=settings.logging.level,
        log_dir=settings.logging.directory,
        json_enabled=settings.logging.json_enabled,
        max_bytes=settings.logging.max_bytes,
        backup_count=settings.logging.backup_count,
        force=True,
    )
    return settings


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise SystemExit(f"Could not parse date {value!r}. Use YYYY-MM-DD.")


# -- commands --------------------------------------------------------------------


def cmd_config(settings: Settings) -> int:
    import json

    print(json.dumps(settings.redacted_dict(), indent=2, default=str))
    return EXIT_OK


def cmd_check(settings: Settings) -> int:
    """Health check: configuration, database, credentials, broker, market data."""
    log_banner(
        logger,
        "TRADING BOT — HEALTH CHECK",
        {
            "Version": __version__,
            "Mode": settings.trading_mode.value.upper(),
            "Live trading armed": settings.is_live,
            "Watchlist": f"{len(settings.data.watchlist)} symbols",
            "Timeframe": settings.data.timeframe,
            "Data feed": settings.alpaca.data_feed,
            "Database": settings.data.database_path,
        },
    )

    checks: list[tuple[str, bool, str]] = []

    # 1. Database
    try:
        database = Database(settings.data.database_path)
        version = database.initialize()
        database.close()
        checks.append(("database", True, f"schema v{version} at {settings.data.database_path}"))
    except Exception as error:  # noqa: BLE001
        checks.append(("database", False, str(error)))

    # 2. Credentials
    if settings.alpaca.has_credentials:
        checks.append(("credentials", True, "ALPACA_API_KEY and ALPACA_SECRET_KEY present"))
    else:
        checks.append(
            ("credentials", False, "missing — copy .env.example to .env and fill in your keys")
        )

    # 3. Broker (skipped in backtest mode, which needs no broker)
    if settings.trading_mode is TradingMode.BACKTEST:
        checks.append(("broker", True, "skipped (backtest mode needs no broker)"))
    elif not settings.alpaca.has_credentials:
        checks.append(("broker", False, "skipped — no credentials"))
    else:
        try:
            from trading_bot.execution.broker import build_broker

            broker = build_broker(settings)
            account = broker.get_account()
            checks.append(
                (
                    "broker",
                    account.can_trade,
                    f"{'PAPER' if account.is_paper else 'LIVE'} account {account.account_id[:8]}… "
                    f"status={account.status} equity=${account.equity:,.2f} "
                    f"buying_power=${account.buying_power:,.2f}"
                    + ("" if account.can_trade else " [TRADING BLOCKED]"),
                )
            )
            clock = broker.get_clock()
            checks.append(("market clock", True, clock.describe()))
        except Exception as error:  # noqa: BLE001
            checks.append(("broker", False, str(error)))

    # 4. Market data
    if not settings.alpaca.has_credentials:
        checks.append(("market data", False, "skipped — no credentials"))
    else:
        try:
            provider = build_market_data(settings.alpaca, settings.data)
            probe = settings.data.watchlist[0]
            bars = provider.get_bars(probe, "1Day", limit=5)
            if bars.empty:
                checks.append(("market data", False, f"no bars returned for {probe}"))
            else:
                last = bars.index[-1]
                checks.append(
                    (
                        "market data",
                        True,
                        f"{len(bars)} daily bars for {probe}, "
                        f"latest {last:%Y-%m-%d} close ${bars['close'].iloc[-1]:,.2f}",
                    )
                )
        except Exception as error:  # noqa: BLE001
            checks.append(("market data", False, str(error)))

    print("\nHealth check results")
    print("-" * 72)
    for name, passed, detail in checks:
        marker = "PASS" if passed else "FAIL"
        print(f"  [{marker}] {name:<14} {detail}")
    print("-" * 72)

    failures = [name for name, passed, _ in checks if not passed]
    if failures:
        print(f"\n{len(failures)} check(s) failed: {', '.join(failures)}")
        return EXIT_FAILURE
    print("\nAll checks passed.")
    return EXIT_OK


def cmd_clock(settings: Settings) -> int:
    from trading_bot.execution.broker import build_broker

    clock = build_broker(settings).get_clock()
    print(f"Market is {clock.describe()}")
    print(f"Broker time: {clock.timestamp.isoformat()}")
    return EXIT_OK


def cmd_db_init(settings: Settings) -> int:
    database = Database(settings.data.database_path)
    version = database.initialize()
    tables = database.query(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    database.close()
    print(f"Database ready: {settings.data.database_path} (schema v{version})")
    print(f"Tables: {', '.join(row['name'] for row in tables)}")
    return EXIT_OK


def cmd_fetch(settings: Settings, args: argparse.Namespace) -> int:
    symbols = (
        [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        if args.symbols
        else list(settings.data.watchlist)
    )
    timeframe = Timeframe.parse(args.timeframe or settings.data.timeframe)
    start = _parse_date(args.start)
    end = _parse_date(args.end)
    limit = args.limit or settings.data.lookback_bars

    provider = build_market_data(
        settings.alpaca, settings.data, use_cache=not args.no_cache
    )

    logger.info(
        "Fetching %s bars for %d symbol(s): %s",
        timeframe.label, len(symbols), ", ".join(symbols),
    )

    if start or end:
        frames = provider.get_bars_multi(symbols, timeframe, start=start, end=end, limit=args.limit)
        failed = {symbol: "no bars returned" for symbol in symbols if symbol not in frames}
    else:
        frames, report = provider.fetch_watchlist(symbols, timeframe, lookback_bars=limit, end=end)
        failed = report.failed

    # Live fetches must never expose a still-forming bar to a strategy.
    frames = {
        symbol: drop_incomplete_bars(frame, timeframe) for symbol, frame in frames.items()
    }

    if not frames:
        print("No data returned. Check your credentials, date range and data feed.")
        return EXIT_FAILURE

    print(f"\nFetched {timeframe.label} bars for {len(frames)}/{len(symbols)} symbol(s)")
    print("-" * 88)
    print(f"{'SYMBOL':<8}{'BARS':>7}  {'FIRST':<20}{'LAST':<20}{'CLOSE':>10}{'CHG%':>9}")
    print("-" * 88)
    for symbol, frame in sorted(frames.items()):
        if frame.empty:
            continue
        first, last = frame.index[0], frame.index[-1]
        close = float(frame["close"].iloc[-1])
        change = (close / float(frame["close"].iloc[0]) - 1) * 100
        print(
            f"{symbol:<8}{len(frame):>7}  {first:%Y-%m-%d %H:%M}     "
            f"{last:%Y-%m-%d %H:%M}     {close:>9,.2f}{change:>9.2f}"
        )
    print("-" * 88)

    for symbol, reason in failed.items():
        print(f"  ! {symbol}: {reason}")

    if args.tail and frames:
        preview_symbol, preview = next(iter(sorted(frames.items())))
        print(f"\nLast {args.tail} bars for {preview_symbol}:")
        print(preview.tail(args.tail).to_string())

    if args.csv_dir:
        directory = Path(args.csv_dir)
        directory.mkdir(parents=True, exist_ok=True)
        for symbol, frame in frames.items():
            path = directory / f"{symbol}_{timeframe.label}.csv"
            frame.to_csv(path)
        print(f"\nWrote {len(frames)} CSV file(s) to {directory}")

    return EXIT_OK


def cmd_cache(settings: Settings, args: argparse.Namespace) -> int:
    cache = BarCache(
        settings.data.cache_dir,
        feed=settings.alpaca.data_feed,
        adjustment=settings.alpaca.adjustment,
    )
    if args.clear:
        removed = cache.clear(args.symbol)
        print(f"Removed {removed} cache file(s).")
        return EXIT_OK
    stats = cache.stats()
    print(f"Cache directory : {stats['directory']}")
    print(f"Files           : {stats['files']}")
    print(f"Size            : {stats['size_mb']} MB")
    print(f"Symbols         : {', '.join(stats['symbols']) or '(none)'}")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        settings = _load(args)
    except ValidationError as error:
        print("Configuration error:\n", file=sys.stderr)
        for issue in error.errors():
            location = ".".join(str(part) for part in issue["loc"]) or "(root)"
            print(f"  {location}: {issue['msg']}", file=sys.stderr)
        print("\nSee .env.example for the expected variables.", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    try:
        if args.command == "config":
            return cmd_config(settings)
        if args.command == "check":
            return cmd_check(settings)
        if args.command == "clock":
            return cmd_clock(settings)
        if args.command == "db-init":
            return cmd_db_init(settings)
        if args.command == "fetch":
            return cmd_fetch(settings, args)
        if args.command == "cache":
            return cmd_cache(settings, args)
        parser.error(f"Unknown command {args.command!r}")
        return EXIT_FAILURE
    except (MarketDataError, ValueError) as error:
        logger.error("%s", error)
        print(f"\nError: {error}", file=sys.stderr)
        return EXIT_FAILURE
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return EXIT_FAILURE
    except Exception as error:  # noqa: BLE001 - top-level guard
        logger.exception("Unhandled error running %s", args.command)
        print(f"\nUnexpected error: {error}", file=sys.stderr)
        return EXIT_FAILURE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
