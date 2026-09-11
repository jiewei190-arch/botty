"""Command-line entry point.

Phase 1 commands::

    python main.py config          # show resolved configuration (secrets masked)
    python main.py check           # verify credentials, broker and data connectivity
    python main.py clock           # market session state
    python main.py fetch           # download and inspect bars
    python main.py db-init         # create or migrate the database
    python main.py cache           # inspect or clear the bar cache

Phase 2 adds::

    python main.py analyze         # full technical analysis of a symbol

Phase 3 adds::

    python main.py signals         # run strategies and report trade setups
    python main.py dashboard       # launch the Streamlit monitoring dashboard

Phase 4 adds risk validation and position sizing to ``signals``.

Phase 5 adds::

    python main.py scan            # rank the watchlist by trade confidence

Later phases add ``scan``, ``backtest``, ``run`` and ``dashboard``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
import textwrap
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Allow `python main.py` from the repository root without installing the package.
if __package__ in (None, ""):  # pragma: no cover - script execution path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import ValidationError

from trading_bot import __version__
from trading_bot.backtesting import FRICTIONLESS, CostModel, calibrate
from trading_bot.backtesting.runner import (
    BacktestDataError,
    BacktestRequest,
    run_backtest,
)
from trading_bot.config.settings import Settings, TradingMode, load_settings
from trading_bot.config.universe import (
    UniverseConfigError,
    load_catalogue,
    normalize_symbols,
    stream_capacity_error,
)
from trading_bot.data.cache import BarCache
from trading_bot.data.database import Database
from trading_bot.data.market_data import (
    MarketDataError,
    build_market_data,
    drop_incomplete_bars,
)
from trading_bot.indicators import (
    BB_WIDTH_COL,
    RELATIVE_VOLUME_COL,
    IndicatorConfig,
    analyze_trend,
    analyze_volume,
    atr_column,
    calculate_all_indicators,
    detect_bollinger_condition,
    detect_ema_crossover,
    detect_macd_momentum,
    detect_macd_signal,
    detect_rsi_condition,
    ema_column,
    find_support_resistance,
    rsi_column,
)
from trading_bot.risk import RiskManager, build_portfolio_state
from trading_bot.scanner import MarketScanner, ScannerConfig
from trading_bot.scanner.market_scan import HuntConfig, sweep_market
from trading_bot.strategies import (
    StrategyError,
    available_strategies,
    build_strategy,
    explain_blockers,
)
from trading_bot.universe import (
    AssetCatalogue,
    Universe,
    UniverseError,
    UniverseFilter,
    build_universe,
    feed_liquidity_warning,
    profile_liquidity,
)
from trading_bot.utils import market_hours
from trading_bot.utils.logging_setup import (
    configure_logging,
    log_banner,
    log_signal_block,
)
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

    universe_cmd = subparsers.add_parser(
        "universe", help="List symbol categories and the resolved watchlist."
    )
    universe_cmd.add_argument(
        "--categories", help="Comma-separated categories to resolve (default: configured)."
    )

    detect = subparsers.add_parser(
        "detect", help="Run the Phase 1 detectors once over historical bars."
    )
    detect.add_argument("--symbols", help="Comma-separated symbols (default: scanner universe).")
    detect.add_argument("--categories", help="Comma-separated symbol categories.")
    detect.add_argument("--timeframe", default=None, help="Bar size (default: SCANNER_TIMEFRAME).")
    detect.add_argument("--bars", type=int, default=None, help="Bars of history per symbol.")
    detect.add_argument("--min-score", type=float, default=0.0, help="Hide symbols below this.")
    detect.add_argument("--top", type=int, default=15, help="Rows to show (default: 15).")
    detect.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    detect.add_argument("--demo", action="store_true", help="Use generated bars (no API key).")
    detect.add_argument("--no-cache", action="store_true", help="Bypass the parquet cache.")

    watch = subparsers.add_parser(
        "watch", help="Run the continuous market scanner (no orders are ever placed)."
    )
    watch.add_argument("--symbols", help="Comma-separated symbols (default: scanner universe).")
    watch.add_argument("--categories", help="Comma-separated symbol categories.")
    watch.add_argument("--timeframe", default=None, help="Bar size (default: SCANNER_TIMEFRAME).")
    watch.add_argument("--min-score", type=float, default=None, help="Alert threshold (0-10).")
    watch.add_argument("--poll", type=int, default=None, help="Seconds between REST cycles.")
    watch.add_argument(
        "--no-stream", action="store_true", help="Poll over REST instead of streaming."
    )
    watch.add_argument("--once", action="store_true", help="Run a single cycle and exit.")
    watch.add_argument(
        "--duration", type=float, default=None, help="Stop after this many minutes."
    )
    watch.add_argument(
        "--no-database", action="store_true", help="Do not persist signals or alerts."
    )

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

    analyze = subparsers.add_parser(
        "analyze", help="Run the technical indicator engine over one or more symbols."
    )
    analyze.add_argument("--symbols", help="Comma-separated symbols (default: watchlist).")
    analyze.add_argument("--timeframe", default=None, help="Bar size (default: DATA_TIMEFRAME).")
    analyze.add_argument(
        "--bars", type=int, default=None, help="Bars of history to analyse (default: lookback)."
    )
    analyze.add_argument(
        "--demo",
        action="store_true",
        help="Run on generated sample data instead of the market — no API keys needed.",
    )
    analyze.add_argument("--no-cache", action="store_true", help="Bypass the parquet cache.")

    signals = subparsers.add_parser(
        "signals", help="Run trading strategies over the watchlist and report setups."
    )
    signals.add_argument("--symbols", help="Comma-separated symbols (default: watchlist).")
    signals.add_argument(
        "--strategy",
        default="all",
        help="Strategy name, comma-separated list, or 'all' (default). "
        f"Available: {', '.join(available_strategies())}",
    )
    signals.add_argument("--timeframe", default=None, help="Bar size (default: DATA_TIMEFRAME).")
    signals.add_argument("--bars", type=int, default=None, help="Bars of history to use.")
    signals.add_argument(
        "--min-confidence", type=float, default=None, help="Override the confidence floor."
    )
    signals.add_argument(
        "--allow-short", action="store_true", help="Permit short signals (off by default)."
    )
    signals.add_argument(
        "--demo",
        action="store_true",
        help="Run on generated sample data instead of the market — no API keys needed.",
    )
    signals.add_argument("--no-cache", action="store_true", help="Bypass the parquet cache.")
    signals.add_argument(
        "--equity",
        type=float,
        default=None,
        help="Account equity to size against. Defaults to the broker's, or 10000 in demo mode.",
    )
    signals.add_argument(
        "--no-risk",
        action="store_true",
        help="Report raw signals without risk validation or sizing.",
    )

    scan = subparsers.add_parser(
        "scan", help="Rank watchlist opportunities by trade confidence."
    )
    scan.add_argument("--symbols", help="Comma-separated symbols (default: watchlist).")
    scan.add_argument(
        "--strategy", default="all",
        help=f"Strategy name, comma-separated list, or 'all'. "
        f"Available: {', '.join(available_strategies())}",
    )
    scan.add_argument("--timeframe", default=None, help="Bar size (default: DATA_TIMEFRAME).")
    scan.add_argument("--bars", type=int, default=None, help="Bars of history to use.")
    scan.add_argument(
        "--min-score", type=float, default=0.0, help="Only show scores at or above this."
    )
    scan.add_argument("--top", type=int, default=None, help="Show only the top N.")
    scan.add_argument(
        "--min-dollar-volume", type=float, default=1_000_000.0,
        help="Skip symbols whose average turnover is below this (default 1,000,000).",
    )
    scan.add_argument(
        "--equity", type=float, default=None, help="Account equity to size against."
    )
    scan.add_argument("--allow-short", action="store_true", help="Permit short signals.")
    scan.add_argument(
        "--demo", action="store_true", help="Use generated sample data — no API keys needed."
    )
    scan.add_argument("--no-cache", action="store_true", help="Bypass the parquet cache.")

    calibrate_cmd = subparsers.add_parser(
        "calibrate",
        help="Measure whether the confidence score actually predicts outcomes.",
    )
    calibrate_cmd.add_argument(
        "--symbols", help="Comma-separated symbols (default: watchlist)."
    )
    calibrate_cmd.add_argument(
        "--strategy", default="all",
        help=f"Strategy name, list, or 'all'. Available: {', '.join(available_strategies())}",
    )
    calibrate_cmd.add_argument(
        "--timeframe", default="1Day", help="Bar size (default 1Day)."
    )
    calibrate_cmd.add_argument("--start", help="First date, YYYY-MM-DD.")
    calibrate_cmd.add_argument("--end", help="Last date, YYYY-MM-DD.")
    calibrate_cmd.add_argument(
        "--capital", type=float, default=100_000.0,
        help="Starting equity. Larger than a real account on purpose: sizing "
        "limits would otherwise reject setups and bias the sample.",
    )
    calibrate_cmd.add_argument(
        "--csv", help="Write the per-trade record to this CSV file."
    )
    calibrate_cmd.add_argument("--no-cache", action="store_true",
                               help="Bypass the parquet bar cache.")

    hunt = subparsers.add_parser(
        "hunt",
        help="Scan the whole market for swing setups and rank the best entries.",
    )
    hunt.add_argument(
        "--strategy", default="all",
        help="Strategy name, comma-separated list, or 'all' (default). "
        f"Available: {', '.join(available_strategies())}",
    )
    hunt.add_argument(
        "--timeframe", default="1Day",
        help="Bars to analyse (default 1Day, for swing trades).",
    )
    hunt.add_argument(
        "--top", type=int, default=10, help="How many setups to show (default 10)."
    )
    hunt.add_argument(
        "--min-score", type=float, default=0.0, help="Only show scores at or above this."
    )
    hunt.add_argument(
        "--min-risk-reward", type=float, default=2.0,
        help="Reject setups paying less than this multiple of the risk (default 2).",
    )
    hunt.add_argument(
        "--max-age", type=int, default=1,
        help="Drop setups that triggered more than this many bars ago (default 1).",
    )
    hunt.add_argument(
        "--equity", type=float, default=None,
        help="Account equity to size against. Defaults to RISK_ACCOUNT_EQUITY.",
    )
    hunt.add_argument(
        "--risk-per-trade", type=float, default=None,
        help="Percent of equity risked per trade (default: RISK_MAX_RISK_PER_TRADE_PCT).",
    )
    hunt.add_argument(
        "--min-dollar-volume", type=float, default=10_000_000.0,
        help="Skip symbols turning over less than this per day (default 10,000,000).",
    )
    hunt.add_argument(
        "--min-price", type=float, default=5.0, help="Skip symbols below this price."
    )
    hunt.add_argument(
        "--max-price", type=float, default=1_000.0,
        help="Skip symbols above this price — one share can exceed a small "
        "account's risk budget.",
    )
    hunt.add_argument(
        "--max-symbols", type=int, default=4_000,
        help="Cap the universe after ranking by turnover (default 4000).",
    )
    hunt.add_argument(
        "--include-leveraged", action="store_true",
        help="Include leveraged and inverse ETFs, which decay over a multi-day hold.",
    )
    hunt.add_argument(
        "--allow-short", action="store_true", help="Include short setups (off by default)."
    )
    hunt.add_argument(
        "--symbols", help="Scan only these symbols instead of the whole market."
    )
    hunt.add_argument(
        "--show-all", action="store_true",
        help="Include setups that failed risk validation, with their reasons.",
    )
    hunt.add_argument("--csv", help="Write the ranked setups to this CSV file.")
    hunt.add_argument("--refresh-universe", action="store_true",
                      help="Re-download the asset list instead of using the cache.")
    hunt.add_argument("--no-cache", action="store_true", help="Bypass the parquet bar cache.")
    hunt.add_argument(
        "--paper-trade", action="store_true",
        help="Submit risk-approved setups as Alpaca paper bracket orders.",
    )
    hunt.add_argument(
        "--watch-market", action="store_true",
        help="Stay online and run once whenever Alpaca reports a new market session open.",
    )

    backtest = subparsers.add_parser(
        "backtest", help="Simulate a strategy over historical bars."
    )
    backtest.add_argument("--symbols", help="Comma-separated symbols (default: watchlist).")
    backtest.add_argument(
        "--strategy",
        default="momentum",
        help="Strategy name, comma-separated list, or 'all'. "
        f"Available: {', '.join(available_strategies())}",
    )
    backtest.add_argument(
        "--timeframe", default=None, help="Bar size (default: DATA_TIMEFRAME)."
    )
    backtest.add_argument("--start", help="First tradable date, YYYY-MM-DD.")
    backtest.add_argument("--end", help="Last date, YYYY-MM-DD (default: now).")
    backtest.add_argument(
        "--capital", type=float, default=10_000.0, help="Starting equity (default 10000)."
    )
    backtest.add_argument(
        "--commission", type=float, default=0.0,
        help="Flat commission per fill (default 0, as most US retail brokers charge).",
    )
    backtest.add_argument(
        "--commission-per-share", type=float, default=0.0,
        help="Per-share commission.",
    )
    backtest.add_argument(
        "--slippage", type=float, default=0.05,
        help="Slippage per fill as a percent, always against the trade (default 0.05).",
    )
    backtest.add_argument(
        "--risk-per-trade", type=float, default=None,
        help="Percent of equity risked per trade (default: RISK_MAX_RISK_PER_TRADE_PCT).",
    )
    backtest.add_argument(
        "--max-positions", type=int, default=None, help="Cap on concurrent positions."
    )
    backtest.add_argument(
        "--min-confidence", type=float, default=None, help="Override the confidence floor."
    )
    backtest.add_argument(
        "--allow-short", action="store_true", help="Permit short signals (off by default)."
    )
    backtest.add_argument(
        "--no-costs", action="store_true",
        help="Run frictionless. Useful for isolating strategy behaviour, never for "
        "judging whether a strategy is worth trading.",
    )
    backtest.add_argument(
        "--trades", action="store_true", help="Print every trade, not just the summary."
    )
    backtest.add_argument("--json", help="Write the full result to this JSON file.")
    backtest.add_argument("--csv", help="Write the trade list to this CSV file.")
    backtest.add_argument(
        "--demo", action="store_true",
        help="Run on generated sample data instead of the market — no API keys needed.",
    )
    backtest.add_argument("--no-cache", action="store_true", help="Bypass the parquet cache.")

    dashboard = subparsers.add_parser(
        "dashboard", help="Launch the Streamlit monitoring dashboard."
    )
    dashboard.add_argument("--port", type=int, default=8501, help="Port (default 8501).")
    dashboard.add_argument(
        "--host", default="localhost", help="Bind address (default localhost)."
    )
    dashboard.add_argument(
        "--headless", action="store_true", help="Do not open a browser automatically."
    )

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

    now = datetime.now(timezone.utc)
    # The local calendar always answers, including when the broker cannot be
    # reached — which is exactly when you most want to know whether the silence
    # is the market being shut or the connection being down.
    print(f"Local calendar: {market_hours.describe(now)}")

    try:
        clock = build_broker(settings).get_clock()
    except Exception as error:  # noqa: BLE001 - the local answer still stands
        print(f"Broker clock unavailable: {error}")
        print(
            "The local calendar cannot see unscheduled closures, so treat it as "
            "advisory until the broker answers."
        )
        return EXIT_OK

    print(f"Broker says: market is {clock.describe()}")
    print(f"Broker time: {clock.timestamp.isoformat()}")
    if clock.is_open is not market_hours.is_market_open(now):
        print(
            "\nNote: broker and local calendar disagree. The broker is "
            "authoritative — this usually means an unscheduled closure or a "
            "half day the calendar does not know about."
        )
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


def _demo_bars(symbol: str, periods: int = 400, timeframe: str | Timeframe = "15Min"):
    """Generate a deterministic sample series for ``--demo`` commands.

    This is **not** market data. It exists so the analysis layers can be
    exercised end to end without API credentials.

    The index is placed on real trading sessions rather than on a continuous
    clock. That matters to anything session-aware: with 24/7 timestamps, the
    fourteen hours between one close and the next open are filled with random
    walk, and the gap detector correctly reports a 25% overnight gap in
    synthetic data that was never meant to have one. Skipping the closed hours
    makes the demo show what the real feed would show.
    """
    import zlib

    import numpy as np
    import pandas as pd

    # crc32, not hash(): Python randomises string hashing per process, which would
    # make "deterministic" false and the demo unreproducible between runs.
    seed = zlib.crc32(symbol.encode("utf-8"))
    rng = np.random.default_rng(seed)
    parsed = Timeframe.parse(timeframe)
    index = _demo_index(periods, parsed)

    # Volatility scales with the square root of bar length, so one setting
    # produces sensible-looking bars at any timeframe.
    session_fraction = min(
        parsed.duration / timedelta(minutes=390), 1.0
    ) if parsed.is_intraday else 1.0
    sigma = 0.019 * (session_fraction ** 0.5)
    drift = float(rng.choice([1.0, -1.0, 0.0])) * sigma / 12

    close = 100 * np.exp(np.cumsum(rng.normal(drift, sigma, periods)))
    spread = np.abs(rng.normal(0, sigma * 0.4, periods)) * close
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": rng.integers(80_000, 600_000, periods).astype("float64"),
        },
        index=index,
    )


def _demo_index(periods: int, timeframe: Timeframe):
    """A timestamp index on real trading sessions, ending at the last one."""
    import pandas as pd

    if not timeframe.is_intraday:
        end = pd.Timestamp.now(tz="UTC").normalize()
        freq = "W-FRI" if timeframe.to_pandas_freq().endswith("W") else "B"
        return pd.DatetimeIndex(
            pd.date_range(end=end, periods=periods, freq=freq, tz="UTC"), name="timestamp"
        )

    stamps: list[pd.Timestamp] = []
    day = datetime.now(timezone.utc).astimezone(market_hours.MARKET_TZ).date()
    step = pd.Timedelta(timeframe.duration)
    while len(stamps) < periods:
        window = market_hours.session_window(day)
        day -= timedelta(days=1)
        if window is None:
            continue
        session = pd.date_range(
            start=pd.Timestamp(window.regular_open),
            end=pd.Timestamp(window.regular_close) - step,
            freq=step,
            tz="UTC",
        )
        stamps.extend(reversed(list(session)))
    return pd.DatetimeIndex(sorted(stamps[:periods]), name="timestamp")


def _money(value: float | None, places: int = 2) -> str:
    """Render a price, or ``n/a`` when the indicator is still warming up."""
    return "n/a" if value is None else f"${value:,.{places}f}"


def _price_note(close: float, average: float | None) -> str:
    """Say whether price sits above or below a moving average."""
    if average is None:
        return ""
    return "  (price above)" if close > average else "  (price below)"


def _level_note(level, close: float) -> str:
    """Describe a support/resistance level's distance and touch count."""
    if level is None:
        return "n/a"
    return (
        f"{_money(level.price)}  ({level.distance_pct(close):+.2f}%, "
        f"{level.touches} touch{'es' if level.touches != 1 else ''})"
    )


def _render_analysis(symbol: str, frame, config: IndicatorConfig) -> None:
    """Print the market-analysis report for one symbol."""
    import pandas as pd

    enriched = calculate_all_indicators(frame, config)
    row = enriched.iloc[-1]
    trend = analyze_trend(enriched, config)
    volume = analyze_volume(enriched, config)
    levels = find_support_resistance(enriched, config)
    close = float(row["close"])

    def value(column: str) -> float | None:
        if column not in enriched.columns:
            return None
        raw = row[column]
        return None if pd.isna(raw) else float(raw)

    width = 64
    print()
    print("=" * width)
    print(f"MARKET ANALYSIS \u2014 {symbol}")
    print("=" * width)
    print(
        f"\nBars analysed : {len(enriched)}"
        f"  ({enriched.index[0]:%Y-%m-%d %H:%M} to {enriched.index[-1]:%Y-%m-%d %H:%M} UTC)"
    )
    print(f"Price         : {_money(close)}")

    print("\nTREND")
    print(f"  Direction   : {trend.direction.value}")
    print(f"  Strength    : {trend.strength}/100   (50 = neutral)")
    print(f"  Confidence  : {trend.confidence}/100")

    print("\nMOVING AVERAGES")
    for period in sorted(config.ema_periods):
        ema = value(ema_column(period))
        print(f"  EMA {period:<4}    : {_money(ema)}{_price_note(close, ema)}")

    print("\nMOMENTUM")
    rsi_value = value(rsi_column(config.rsi_period))
    rsi_text = "n/a" if rsi_value is None else f"{rsi_value:.1f}"
    print(
        f"  RSI {config.rsi_period:<4}    : {rsi_text}"
        f"  \u2014 {detect_rsi_condition(enriched, config)}"
    )
    momentum = detect_macd_momentum(enriched, config).lower()
    print(f"  MACD        : {detect_macd_signal(enriched, config)}  ({momentum} momentum)")
    print(f"  EMA 9/20    : {detect_ema_crossover(enriched, 9, 20).signal}")

    print("\nVOLATILITY")
    atr_value = value(atr_column(config.atr_period))
    atr_note = ""
    if atr_value is not None and close > 0:
        atr_note = f"  ({atr_value / close * 100:.2f}% of price)"
    print(f"  ATR {config.atr_period:<4}    : {_money(atr_value)}{atr_note}")
    print(f"  Bollinger   : {detect_bollinger_condition(enriched, config)}")
    band_width = value(BB_WIDTH_COL)
    print(f"  Band width  : {'n/a' if band_width is None else f'{band_width:.4f}'}")

    print("\nVOLUME")
    relative = value(RELATIVE_VOLUME_COL)
    print(f"  Relative    : {'n/a' if relative is None else f'{relative:.2f}x average'}")
    print(f"  Condition   : {volume.condition.value}")
    print(f"  Trend       : {volume.trend}")

    print("\nKEY LEVELS")
    print(f"  Resistance  : {_level_note(levels.nearest_resistance, close)}")
    print(f"  Support     : {_level_note(levels.nearest_support, close)}")

    print("\nSIGNALS")
    observations = list(trend.reasons) + list(volume.reasons)
    if not observations:
        print("  (no notable conditions)")
    for observation in observations:
        print(f"  \u2713 {observation}")
    print()
    print("=" * width)


def cmd_analyze(settings: Settings, args: argparse.Namespace) -> int:
    """Run the Phase 2 indicator engine over the watchlist and print a report.

    This is the Phase 2 demonstration path: it pulls bars through the Phase 1
    market-data layer, enriches them with every configured indicator, and prints
    the trend, momentum, volatility, volume and level analysis.
    """
    symbols = (
        [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        if args.symbols
        else list(settings.data.watchlist)
    )
    timeframe = Timeframe.parse(args.timeframe or settings.data.timeframe)
    config = IndicatorConfig()
    # Indicators need warm-up before the first usable value.
    bars_wanted = args.bars or max(settings.data.lookback_bars, config.max_lookback + 50)

    if args.demo:
        banner = "!" * 64
        print(f"\n{banner}")
        print("  DEMO MODE \u2014 GENERATED SAMPLE DATA, NOT REAL MARKET DATA")
        print("  Prices below are synthetic. Supply Alpaca credentials for live analysis.")
        print(banner)
        frames = {symbol: _demo_bars(symbol, bars_wanted) for symbol in symbols}
        failed: dict[str, str] = {}
    else:
        provider = build_market_data(
            settings.alpaca, settings.data, use_cache=not args.no_cache
        )
        logger.info(
            "Fetching %s bars for %d symbol(s) to analyse", timeframe.label, len(symbols)
        )
        frames, report = provider.fetch_watchlist(
            symbols, timeframe, lookback_bars=bars_wanted
        )
        # A forming bar must never reach the indicator engine.
        frames = {
            symbol: drop_incomplete_bars(frame, timeframe)
            for symbol, frame in frames.items()
        }
        failed = dict(report.failed)

    if not frames:
        print("No data to analyse. Check credentials, symbols and timeframe.")
        return EXIT_FAILURE

    for symbol in sorted(frames):
        frame = frames[symbol]
        if frame.empty:
            failed.setdefault(symbol, "no complete bars")
            continue
        try:
            _render_analysis(symbol, frame, config)
        except Exception as error:  # noqa: BLE001 - one bad symbol must not stop the run
            logger.exception("Analysis failed for %s", symbol)
            failed[symbol] = str(error)

    if failed:
        print("\nNot analysed:")
        for symbol, reason in failed.items():
            print(f"  ! {symbol}: {reason}")
    return EXIT_OK


def _load_bars(settings: Settings, args: argparse.Namespace, symbols, timeframe, bars_wanted):
    """Fetch bars for a command, or generate demo data. Returns (frames, failures)."""
    if args.demo:
        banner = "!" * 64
        print(f"\n{banner}")
        print("  DEMO MODE \u2014 GENERATED SAMPLE DATA, NOT REAL MARKET DATA")
        print("  Prices below are synthetic. Supply Alpaca credentials for live analysis.")
        print(banner)
        return {
            symbol: _demo_bars(symbol, bars_wanted, timeframe) for symbol in symbols
        }, {}

    provider = build_market_data(settings.alpaca, settings.data, use_cache=not args.no_cache)
    logger.info("Fetching %s bars for %d symbol(s)", timeframe.label, len(symbols))
    frames, report = provider.fetch_watchlist(symbols, timeframe, lookback_bars=bars_wanted)
    # A forming bar must never reach a strategy.
    frames = {
        symbol: drop_incomplete_bars(frame, timeframe) for symbol, frame in frames.items()
    }
    return frames, dict(report.failed)


def cmd_signals(settings: Settings, args: argparse.Namespace) -> int:
    """Run the Phase 3 strategies over the watchlist and report trade setups.

    Reports opportunities only. It sizes nothing and places nothing — position
    sizing arrives with the risk manager in Phase 4, and order placement in
    Phase 7.
    """
    symbols = (
        [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        if args.symbols
        else list(settings.data.watchlist)
    )
    timeframe = Timeframe.parse(args.timeframe or settings.data.timeframe)

    names = (
        available_strategies()
        if args.strategy.strip().lower() == "all"
        else [item.strip() for item in args.strategy.split(",") if item.strip()]
    )
    overrides: dict[str, object] = {}
    if args.min_confidence is not None:
        overrides["min_confidence"] = args.min_confidence
    if args.allow_short:
        overrides["allow_short"] = True

    try:
        strategies = [build_strategy(name, **overrides) for name in names]
    except StrategyError as error:
        print(f"Error: {error}", file=sys.stderr)
        return EXIT_FAILURE

    warmup = max(strategy.min_bars for strategy in strategies)
    bars_wanted = args.bars or max(settings.data.lookback_bars, warmup + 50)

    frames, failed = _load_bars(settings, args, symbols, timeframe, bars_wanted)
    if not frames:
        print("No data to analyse. Check credentials, symbols and timeframe.")
        return EXIT_FAILURE

    found = []
    blockers: Counter[str] = Counter()
    for symbol in sorted(frames):
        frame = frames[symbol]
        if frame.empty:
            failed.setdefault(symbol, "no complete bars")
            continue
        for strategy in strategies:
            try:
                prepared = strategy.prepare(frame)
                signal = strategy.generate_signal(symbol, prepared)
            except Exception as error:  # noqa: BLE001 - one failure must not stop the scan
                logger.exception("%s failed on %s", strategy.name, symbol)
                failed[f"{symbol}/{strategy.name}"] = str(error)
                continue
            if signal is not None:
                found.append(signal)
            else:
                # Record why not. An idle bot with no explanation is
                # indistinguishable from a broken one.
                for name in explain_blockers(strategy):
                    blockers[f"{strategy.name}.{name}"] += 1

    found.sort(key=lambda item: item.confidence, reverse=True)

    # Risk validation and sizing. Signals are proposals; only the risk manager
    # attaches a quantity, and only to trades that clear every limit.
    decisions = []
    portfolio = None
    if found and not args.no_risk:
        portfolio = _portfolio_for_sizing(settings, args)
        manager = RiskManager(settings.risk)
        halt = manager.trading_halted(portfolio)
        if halt:
            print(f"\n*** TRADING HALTED: {halt} ***")
        decisions = manager.evaluate_many(found, portfolio)

    approved = {id(d.signal): d for d in decisions if d.approved}
    rejected = [d for d in decisions if not d.approved]

    print(f"\nScanned {len(frames)} symbol(s) with {len(strategies)} "
          f"strateg{'ies' if len(strategies) != 1 else 'y'} on {timeframe.label} bars")
    if portfolio is not None:
        print(f"Sizing against ${float(portfolio.equity):,.2f} equity · "
              f"{portfolio.open_count} open position(s) · "
              f"risk {settings.risk.max_risk_per_trade_pct:.2f}% per trade")
    print("=" * 64)
    if not found:
        print("\nNo setups met the entry criteria on the latest bar.")
        print("That is the normal outcome — every strategy requires several conditions")
        print("to align at once, and most bars in most markets do not qualify.")
        if blockers:
            print("\nMost common blockers (condition, times it blocked an entry):")
            for name, count in blockers.most_common(8):
                print(f"  {count:>3}x  {name}")
    for signal in found:
        decision = approved.get(id(signal))
        matching = next((d for d in decisions if d.signal is signal), None)
        if args.no_risk:
            validation, size = None, None
        elif decision is not None:
            validation, size = "PASSED", decision.shares
        else:
            reason = matching.rejection_reason if matching else "not evaluated"
            validation, size = f"REJECTED — {reason}", None
        log_signal_block(
            logger,
            symbol=signal.symbol,
            strategy=signal.strategy,
            direction=signal.direction.value,
            confidence=signal.confidence,
            entry=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            reasons=list(signal.reasons),
            timestamp=signal.timestamp,
            risk_validation=validation,
            position_size=size,
        )

    if found:
        print("\nSummary")
        print("-" * 64)
        header = f"{'SYMBOL':<8}{'STRATEGY':<15}{'DIR':<6}{'CONF':>5}{'ENTRY':>10}{'R:R':>6}"
        if not args.no_risk:
            header += f"{'QTY':>6}{'RISK $':>9}  STATUS"
        print(header)
        print("-" * 78)
        for signal in found:
            row = (
                f"{signal.symbol:<8}{signal.strategy:<15}{signal.direction.value:<6}"
                f"{signal.confidence:>5.0f}{signal.entry_price:>10,.2f}"
                f"{signal.risk_reward_ratio:>6.2f}"
            )
            if not args.no_risk:
                decision = next((d for d in decisions if d.signal is signal), None)
                if decision is not None and decision.approved:
                    row += f"{decision.shares:>6}{float(decision.risk_amount):>9,.2f}  APPROVED"
                else:
                    reason = decision.rejection_reason if decision else "not evaluated"
                    row += f"{'—':>6}{'—':>9}  {reason[:34]}"
            print(row)
        print("-" * 78)

        if not args.no_risk:
            print(f"\n{len(approved)} of {len(found)} setup(s) cleared risk validation.")
            if rejected:
                print("Rejected by:")
                blocked_by = Counter(
                    check.name
                    for decision in rejected
                    for check in decision.failed_checks
                )
                for name, count in blocked_by.most_common():
                    print(f"  {count:>3}x  {name}")
        print("\nThese are sized proposals, not orders. Order placement arrives in Phase 7.")

    if failed:
        print("\nNot analysed:")
        for key, reason in failed.items():
            print(f"  ! {key}: {reason}")
    return EXIT_OK


def _portfolio_for_sizing(settings: Settings, args: argparse.Namespace):
    """Build the portfolio the risk manager sizes against.

    Uses the live broker account and trade history when available. In demo mode,
    or without credentials, falls back to a stated equity so sizing can still be
    demonstrated — clearly, and without pretending an account exists.
    """
    from trading_bot.data.database import Database

    equity = args.equity
    account = None
    positions: list = []
    database = None

    if not args.demo and settings.alpaca.has_credentials:
        try:
            from trading_bot.execution.broker import build_broker

            broker = build_broker(settings)
            account = broker.get_account()
            positions = broker.get_positions()
        except Exception as error:  # noqa: BLE001 - fall back to stated equity
            logger.warning("Could not read the broker account for sizing: %s", error)

    try:
        database = Database(settings.data.database_path)
        database.initialize()
    except Exception as error:  # noqa: BLE001
        logger.warning("Could not open the database for risk history: %s", error)
        database = None

    if account is None and equity is None:
        equity = 10_000.0
        print(f"\nNo broker account available — sizing against a stated "
              f"${equity:,.0f} for demonstration.")

    state = build_portfolio_state(
        account=account, broker_positions=positions, database=database, equity=equity
    )
    if database is not None:
        database.close()
    return state


def cmd_scan(settings: Settings, args: argparse.Namespace) -> int:
    """Rank watchlist opportunities by trade confidence.

    Reports opportunities in ranked order with the factors behind each score.
    Places nothing — order placement arrives in Phase 7.
    """
    symbols = (
        [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        if args.symbols
        else list(settings.data.watchlist)
    )
    timeframe = Timeframe.parse(args.timeframe or settings.data.timeframe)

    names = (
        available_strategies()
        if args.strategy.strip().lower() == "all"
        else [item.strip() for item in args.strategy.split(",") if item.strip()]
    )
    overrides: dict[str, object] = {"allow_short": True} if args.allow_short else {}
    try:
        strategies = [build_strategy(name, **overrides) for name in names]
    except StrategyError as error:
        print(f"Error: {error}", file=sys.stderr)
        return EXIT_FAILURE

    warmup = max(strategy.min_bars for strategy in strategies)
    bars_wanted = args.bars or max(settings.data.lookback_bars, warmup + 50)

    frames, failures = _load_bars(settings, args, symbols, timeframe, bars_wanted)
    if not frames:
        print("No data to scan. Check credentials, symbols and timeframe.")
        return EXIT_FAILURE

    portfolio = _portfolio_for_sizing(settings, args)
    scanner = MarketScanner(
        strategies,
        risk_manager=RiskManager(settings.risk),
        config=ScannerConfig(
            min_avg_dollar_volume=args.min_dollar_volume,
            min_confidence=args.min_score,
            max_results=args.top,
        ),
    )
    result = scanner.scan(frames, portfolio=portfolio)
    result.failures.update(failures)

    width = 72
    print()
    print("=" * width)
    print(f"MARKET SCAN — {timeframe.label} bars")
    print("=" * width)
    print(result.summary())
    print(
        f"Sizing against ${float(portfolio.equity):,.2f} equity · "
        f"{portfolio.open_count} open position(s)"
    )
    if result.halt_reason:
        print(f"\n*** TRADING HALTED: {result.halt_reason} ***")

    if not result.opportunities:
        print("\nNo opportunities scored above the threshold.")
        if result.blockers:
            print("\nMost common blockers:")
            for name, count in sorted(
                result.blockers.items(), key=lambda item: -item[1]
            )[:8]:
                print(f"  {count:>3}x  {name}")
    for opportunity in result.opportunities:
        _render_opportunity(opportunity, width)

    if result.opportunities:
        print("\n" + "=" * width)
        print(f"{'#':<3}{'SYMBOL':<8}{'DIR':<6}{'STRATEGY':<15}{'SCORE':>6}"
              f"{'QTY':>6}{'R:R':>7}  STATUS")
        print("-" * width)
        for opportunity in result.opportunities:
            status = "TRADABLE" if opportunity.tradable else (
                (opportunity.rejection_reason or "not sized")[:24]
            )
            print(
                f"{opportunity.rank:<3}{opportunity.symbol:<8}"
                f"{opportunity.signal.direction.value:<6}{opportunity.signal.strategy:<15}"
                f"{opportunity.confidence:>6.1f}{opportunity.quantity:>6}"
                f"{opportunity.signal.risk_reward_ratio:>7.2f}  {status}"
            )
        print("-" * width)

    if result.skipped:
        print("\nFiltered out before analysis:")
        for symbol, reason in result.skipped.items():
            print(f"  {symbol}: {reason}")
    if result.failures:
        print("\nNot scanned:")
        for key, reason in result.failures.items():
            print(f"  ! {key}: {reason}")

    print(
        "\nThe score ranks these against each other. It is not a probability of "
        "profit.\nThese are proposals — order placement arrives in Phase 7."
    )
    return EXIT_OK


def _render_opportunity(opportunity, width: int) -> None:
    """Print one ranked opportunity."""
    signal = opportunity.signal
    print()
    print("-" * width)
    print(f"#{opportunity.rank}  {signal.symbol}")
    print(f"Direction: {signal.direction.value}")
    print(f"Confidence: {opportunity.confidence:.0f}/100")

    if opportunity.reasons:
        print("\nReasons:")
        for reason in opportunity.reasons:
            print(f"  \u2713 {reason}")

    print("\nScore breakdown:")
    for factor in sorted(opportunity.factors, key=lambda item: -item.contribution):
        bar = "\u2588" * int(round(factor.score / 10))
        print(f"  {factor.name:<14}{factor.score:>5.0f}  {bar:<10}  {factor.detail}")

    print()
    print(f"Suggested Entry : ${signal.entry_price:,.2f}")
    print(f"Stop Loss       : ${signal.stop_loss:,.2f}"
          f"  ({signal.stop_distance_pct:.2f}% away)")
    print(f"Take Profit     : ${signal.take_profit:,.2f}")
    print(f"Risk/Reward     : 1:{signal.risk_reward_ratio:.2f}")

    decision = opportunity.decision
    if decision is None:
        print("Risk            : not evaluated")
    elif decision.approved:
        print(f"Position Size   : {decision.shares} shares "
              f"(risking ${float(decision.risk_amount):,.2f})")
        print(f"Risk Validation : PASSED — limited by "
              f"{decision.sizing.binding_constraint.description}")
    else:
        print(f"Risk Validation : REJECTED — {decision.rejection_reason}")


def _stated_portfolio(settings: Settings, args: argparse.Namespace):
    """Portfolio state built from stated equity, never from a broker balance.

    The hunt reads market data from one place and assumes you trade somewhere
    else entirely. Reading equity from the data provider's account would size
    positions against a balance that has nothing to do with the money at risk —
    typically zero, on a data-only key.
    """
    from trading_bot.risk import PortfolioState
    from trading_bot.risk.position_sizing import to_decimal

    equity = float(
        getattr(args, "equity", None) or settings.risk.account_equity
    )
    if equity <= 0:
        raise ValueError(
            f"Account equity must be positive, got {equity}. Set RISK_ACCOUNT_EQUITY "
            "in your .env or pass --equity."
        )
    amount = to_decimal(equity)
    return PortfolioState(
        equity=amount,
        cash=amount,
        buying_power=amount,
        positions=(),
    )


def cmd_calibrate(settings: Settings, args: argparse.Namespace) -> int:
    """Measure whether the confidence score predicts anything.

    Backtests the requested symbols, buckets every closed trade by the score it
    entered on, and reports how each bucket actually resolved. This is the only
    honest way to attach odds to a ranking: measure them.
    """
    symbols = (
        [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        if args.symbols
        else list(settings.data.watchlist)
    )
    names = (
        available_strategies()
        if args.strategy.strip().lower() == "all"
        else [item.strip() for item in args.strategy.split(",") if item.strip()]
    )
    start = _parse_date(args.start)
    end = _parse_date(args.end)

    width = 74
    print()
    print("=" * width)
    print("SCORE CALIBRATION — does a higher score mean a better setup?")
    print("=" * width)
    print(f"\nBacktesting {len(symbols)} symbol(s) with {', '.join(names)}...")

    try:
        request = BacktestRequest(
            symbols=tuple(symbols),
            strategies=tuple(names),
            timeframe=args.timeframe,
            start=start,
            end=end,
            starting_equity=args.capital,
            risk=settings.risk,
            use_cache=not args.no_cache,
        )
        result = run_backtest(request, settings)
    except (BacktestDataError, StrategyError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return EXIT_FAILURE

    calibration = calibrate(result.trades)
    print()
    for line in calibration.summary_lines():
        print(line)

    if args.csv and result.trades:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.trade_frame.to_csv(path, index=False)
        print(f"\nPer-trade record written to {path}")

    print()
    print("-" * width)
    print(
        "These are historical frequencies, not forward probabilities. A band's\n"
        "win rate says how setups like it resolved on this data — not what the\n"
        "next one will do. Read the sample size and the error before the rate."
    )
    print("-" * width)
    return EXIT_OK


def cmd_hunt(settings: Settings, args: argparse.Namespace) -> int:
    """Scan the market for swing setups and print an entry plan for each.

    This places nothing and connects to no broker for trading. It reads market
    data, ranks what it finds, and prints the prices to work — the orders are
    yours to place, wherever you trade.
    """
    if args.watch_market:
        from trading_bot.automation.notifications import build_notifier
        from trading_bot.automation.runner import MarketOpenRunner
        from trading_bot.execution.broker import BrokerError, build_broker

        if not args.paper_trade:
            print("Error: --watch-market currently requires --paper-trade.", file=sys.stderr)
            return EXIT_FAILURE
        try:
            broker = build_broker(settings)
        except BrokerError as error:
            print(f"Error: {error}", file=sys.stderr)
            return EXIT_FAILURE
        if not broker.is_paper:
            print("Error: automated execution is paper-only in this release.", file=sys.stderr)
            return EXIT_FAILURE
        child_args = argparse.Namespace(**vars(args))
        child_args.watch_market = False
        runner = MarketOpenRunner(
            broker,
            lambda: cmd_hunt(settings, child_args),
            build_notifier(settings),
            closed_poll_seconds=settings.automation.closed_poll_seconds,
        )
        try:
            runner.run_forever()
        except KeyboardInterrupt:
            print("\nBotty automation stopped.")
        return EXIT_OK

    names = (
        available_strategies()
        if args.strategy.strip().lower() == "all"
        else [item.strip() for item in args.strategy.split(",") if item.strip()]
    )
    overrides: dict[str, Any] = {"allow_short": True} if args.allow_short else {}
    try:
        strategies = [build_strategy(name, **overrides) for name in names]
    except StrategyError as error:
        print(f"Error: {error}", file=sys.stderr)
        return EXIT_FAILURE

    filters = UniverseFilter(
        min_price=args.min_price,
        max_price=args.max_price,
        min_dollar_volume=args.min_dollar_volume,
        max_symbols=args.max_symbols,
        exclude_leveraged=not args.include_leveraged,
        always_include=frozenset(
            item.strip().upper()
            for item in (args.symbols or "").split(",")
            if item.strip()
        ),
    )

    provider = build_market_data(
        settings.alpaca, settings.data, use_cache=not args.no_cache
    )
    try:
        portfolio = _stated_portfolio(settings, args)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return EXIT_FAILURE

    width = 78
    print()
    print("=" * width)
    print("MARKET HUNT — scanning for swing entries")
    print("=" * width)

    warning = feed_liquidity_warning(settings.alpaca.data_feed, args.min_dollar_volume)
    if warning:
        print()
        for index, line in enumerate(textwrap.wrap(warning, width - 6)):
            print(f"  {'!' if index == 0 else ' '} {line}")

    def show(done: int, total: int) -> None:
        pct = done / total * 100 if total else 0
        print(f"\r  fetching daily bars… {done:,}/{total:,} ({pct:.0f}%)", end="", flush=True)

    try:
        if args.symbols:
            symbols = [
                item.strip().upper()
                for item in args.symbols.split(",")
                if item.strip()
            ]
            print(f"\nScanning {len(symbols)} named symbol(s)...")
            frames, _ = provider.fetch_watchlist(
                symbols, "1Day", lookback_bars=300, progress=show
            )
            print()
            profiles = {
                symbol: profile
                for symbol, frame in frames.items()
                if (profile := profile_liquidity(symbol, frame)) is not None
            }
            universe = Universe(
                symbols=tuple(frames), profiles=profiles, frames=frames
            )
        else:
            catalogue = AssetCatalogue(settings.alpaca, cache_dir=settings.data.cache_dir)
            print("\nDiscovering the tradable universe...")
            universe = build_universe(
                catalogue,
                provider,
                filters,
                use_cache=not args.refresh_universe,
                progress=show,
            )
            print()
            for line in universe.summary_lines():
                print(f"  {line}")
    except (UniverseError, MarketDataError) as error:
        print(f"\nError: {error}", file=sys.stderr)
        return EXIT_FAILURE

    if not universe.frames:
        print("\nNo symbols to scan.", file=sys.stderr)
        return EXIT_FAILURE

    print(f"\nAnalysing {len(universe.frames):,} symbols with "
          f"{', '.join(names)}...")
    sweep = sweep_market(
        universe,
        strategies,
        portfolio=portfolio,
        risk_manager=RiskManager(settings.risk),
        config=HuntConfig(
            timeframe=args.timeframe,
            max_signal_age_bars=args.max_age,
            min_score=args.min_score,
            top_n=args.top,
            min_risk_reward=args.min_risk_reward,
            require_risk_approval=not args.show_all,
        ),
    )

    print()
    for line in sweep.summary_lines():
        print(f"  {line}")

    if sweep.halt_reason:
        print(f"\n*** TRADING HALTED: {sweep.halt_reason} ***")

    if not sweep.opportunities:
        print("\nNo setups met the criteria today. That is a normal outcome —")
        print("most days most stocks are not at an entry.")
        if sweep.blockers:
            print("\nMost common blockers:")
            for name, count in sorted(
                sweep.blockers.items(), key=lambda item: -item[1]
            )[:6]:
                print(f"  {count:>6,}x  {name}")
        return EXIT_OK

    equity = float(portfolio.equity)
    print()
    print("=" * width)
    print(f"{len(sweep.opportunities)} SETUP(S) — sized for a ${equity:,.0f} account")
    print("=" * width)
    for opportunity in sweep.opportunities:
        _render_entry_plan(opportunity, width)

    print()
    print("-" * width)
    print(
        f"Each share count assumes this is the only trade you take. Your account "
        f"supports\nabout {sweep.concurrent_capacity} of these at once — taking "
        "more means sizing each one smaller."
    )
    print("-" * width)

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        sweep.as_frame().to_csv(path, index=False)
        print(f"\nRanked setups written to {path}")

    if args.paper_trade:
        from trading_bot.automation.notifications import build_notifier
        from trading_bot.execution.broker import BrokerError, build_broker

        notifier = build_notifier(settings)
        try:
            broker = build_broker(settings)
            if not broker.is_paper:
                raise BrokerError("automated execution is paper-only in this release")
            held = {item["symbol"] for item in broker.get_positions()}
            placed = 0
            for opportunity in sweep.opportunities[: sweep.concurrent_capacity]:
                signal = opportunity.signal
                if signal.symbol in held or opportunity.decision is None:
                    continue
                stamp = signal.timestamp.strftime("%Y%m%d")
                client_id = f"botty-{stamp}-{signal.symbol}-{signal.strategy}".lower()
                broker.submit_bracket_order(
                    symbol=signal.symbol,
                    qty=int(opportunity.decision.shares),
                    side="buy" if signal.direction.value == "LONG" else "sell",
                    take_profit=signal.take_profit,
                    stop_loss=signal.stop_loss,
                    client_order_id=client_id,
                )
                placed += 1
                notifier.send(
                    f"Botty paper order submitted: {signal.symbol} "
                    f"{signal.direction.value} x{int(opportunity.decision.shares)}; "
                    f"stop ${signal.stop_loss:,.2f}, target ${signal.take_profit:,.2f}."
                )
            print(f"\nSubmitted {placed} Alpaca paper bracket order(s).")
        except Exception as error:  # notification must survive broker failures
            with contextlib.suppress(Exception):
                notifier.send(f"BOTTY NEEDS ATTENTION: paper execution failed: {error}")
            print(f"\nPaper execution failed: {error}", file=sys.stderr)
            return EXIT_FAILURE

    if args.paper_trade:
        print(
            "\nThese were paper-trading candidates. The score ranks setups; it is "
            "not a probability of profit."
        )
    else:
        print(
            "\nThese are candidates, ranked against each other. The score is not a "
            "probability\nof profit, and nothing here has been placed as an order."
        )
    return EXIT_OK


def _render_entry_plan(opportunity, width: int) -> None:
    """Print one setup as a plan you could work from."""
    signal = opportunity.signal
    decision = opportunity.decision
    shares = int(decision.shares) if decision else 0
    is_long = signal.direction.value == "LONG"

    stop_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price * 100
    target_pct = abs(signal.take_profit - signal.entry_price) / signal.entry_price * 100

    print()
    print("-" * width)
    print(
        f"#{opportunity.rank}  {signal.symbol}  ·  {signal.direction.value}  ·  "
        f"{signal.strategy}  ·  score {opportunity.confidence:.0f}/100"
    )
    print("-" * width)

    if signal.reasons:
        for reason in signal.reasons[:4]:
            print(f"  \u2713 {reason}")
        print()

    verb = "Buy" if is_long else "Sell short"
    print(f"  {verb:<12} {shares:>6} shares near ${signal.entry_price:,.2f}"
          f"   (${shares * signal.entry_price:,.0f})")
    print(f"  {'Stop':<12} {'':>6}        ${signal.stop_loss:,.2f}"
          f"   ({stop_pct:.2f}% away)")
    print(f"  {'Target':<12} {'':>6}        ${signal.take_profit:,.2f}"
          f"   ({target_pct:.2f}% away)")

    if decision is not None and decision.approved:
        print(
            f"\n  Risking ${float(decision.risk_amount):,.2f} to make "
            f"${float(decision.risk_amount) * signal.risk_reward_ratio:,.2f} "
            f"({signal.risk_reward_ratio:.2f}:1)"
        )
        print(f"  Sized by: {decision.sizing.binding_constraint.description}")
    elif decision is not None:
        print(f"\n  NOT SIZED — {decision.rejection_reason}")


# ---------------------------------------------------------------------------
# Phase 1: universe, detectors and the continuous scanner
# ---------------------------------------------------------------------------


def _watch_symbols(settings: Settings, args: argparse.Namespace) -> tuple[str, ...]:
    """Resolve the symbol list for a scanner command."""
    from trading_bot.scanner.factory import resolve_symbols

    if getattr(args, "symbols", None):
        return normalize_symbols(args.symbols.split(","))
    if getattr(args, "categories", None):
        catalogue = load_catalogue(settings.scanner.universe_file)
        return catalogue.resolve(
            [part.strip() for part in args.categories.split(",") if part.strip()],
            include=settings.scanner.extra_symbols,
            exclude=settings.scanner.exclude_symbols,
        )
    return resolve_symbols(settings)


def cmd_universe(settings: Settings, args: argparse.Namespace) -> int:
    """Show the symbol catalogue and what the scanner would watch."""
    catalogue = load_catalogue(settings.scanner.universe_file)
    print("Symbol categories")
    print("-" * 40)
    print(catalogue.describe())

    symbols = _watch_symbols(settings, args)
    source = args.categories or ", ".join(settings.scanner.categories)
    print(f"\nResolved watchlist ({source}): {len(symbols)} symbols")
    for index in range(0, len(symbols), 10):
        print("  " + " ".join(f"{symbol:<7}" for symbol in symbols[index : index + 10]))

    warning = stream_capacity_error(symbols)
    if warning:
        print(f"\nStreaming: {warning}")
    else:
        print("\nStreaming: fits one websocket subscription on the free plan.")
    if settings.scanner.universe_file:
        print(f"Universe file: {settings.scanner.universe_file}")
    return EXIT_OK


def _render_detections(analyses, *, top: int, min_score: float) -> None:
    """Print the ranked detector table."""
    shown = [a for a in analyses if a.overall >= min_score][:top]
    if not shown:
        print("\nNothing above the score threshold.")
        return

    header = f"{'SYMBOL':<8} {'SCORE':>6} {'DIR':<8} {'AGREE':>6}  SIGNALS"
    print(f"\n{header}")
    print("-" * len(header))
    for analysis in shown:
        score = analysis.score
        kinds = ", ".join(
            f"{signal.signal_type.value.split('_')[0].title()} {signal.strength:.1f}"
            for signal in score.fired
        )
        print(
            f"{score.symbol:<8} {score.overall:>6.2f} {score.direction.value:<8} "
            f"{score.agreement:>6.0%}  {kinds or '-'}"
        )

    best = shown[0]
    if best.signals:
        print(f"\nTop candidate\n{'-' * 40}")
        print(best.score.describe())


def cmd_detect(settings: Settings, args: argparse.Namespace) -> int:
    """Run every detector once over historical bars and rank the results."""
    from trading_bot.scanner.factory import build_engine

    symbols = _watch_symbols(settings, args)
    timeframe = Timeframe.parse(args.timeframe or settings.scanner.timeframe)
    bars_wanted = args.bars or settings.scanner.lookback_bars

    print(f"Detecting on {len(symbols)} symbols, {timeframe.label} bars")
    print(market_hours.describe())

    frames, failures = _load_bars(settings, args, symbols, timeframe, bars_wanted)
    if failures:
        print(f"\n{len(failures)} symbol(s) returned no data: {', '.join(sorted(failures))}")
    if not frames:
        print("\nNo bars to analyse.", file=sys.stderr)
        return EXIT_FAILURE

    analyses = build_engine(settings).analyze_many(frames)
    failed_detectors = Counter(
        name for analysis in analyses for name in analysis.failures
    )
    if failed_detectors:
        print(f"\nDetector errors: {dict(failed_detectors)}", file=sys.stderr)

    if args.json:
        print(json.dumps([a.score.as_dict() for a in analyses], indent=2))
        return EXIT_OK

    _render_detections(analyses, top=args.top, min_score=args.min_score)
    total = sum(len(a.signals) for a in analyses)
    print(f"\n{total} signal(s) across {len(analyses)} symbol(s).")
    print("Ranking only \u2014 no order was placed and none can be from this command.")
    return EXIT_OK


def cmd_watch(settings: Settings, args: argparse.Namespace) -> int:
    """Run the continuous scanner until interrupted."""
    import asyncio

    from trading_bot.scanner.factory import (
        build_alert_manager,
        build_scanner,
        build_stream_config,
    )

    symbols = _watch_symbols(settings, args)
    if args.timeframe:
        settings = settings.with_overrides(
            scanner=settings.scanner.model_copy(update={"timeframe": args.timeframe})
        )
    if args.min_score is not None:
        settings = settings.with_overrides(
            alerts=settings.alerts.model_copy(update={"min_score": args.min_score})
        )
    if args.poll is not None:
        settings = settings.with_overrides(
            scanner=settings.scanner.model_copy(update={"poll_seconds": args.poll})
        )

    database: Database | None = None
    if not args.no_database:
        database = Database(settings.data.database_path)
        database.initialize()

    alerts = build_alert_manager(settings, database=database)
    scanner = build_scanner(settings, _watch_provider(settings), symbols=symbols, alerts=alerts)

    if database is not None:
        scanner.on_analysis = _persisting_recorder(database)

    log_banner(
        logger,
        "Market scanner starting",
        {
            "symbols": len(symbols),
            "timeframe": scanner.timeframe.label,
            "feed": settings.alpaca.data_feed,
            "alert threshold": settings.alerts.min_score,
            "streaming": not args.no_stream and settings.scanner.stream_enabled,
        },
    )
    print(f"\nWatching {len(symbols)} symbols on {scanner.timeframe.label} bars")
    print(market_hours.describe())
    print("This command never places an order. Ctrl-C to stop.\n")

    try:
        if args.once:
            cycle = scanner.scan_once()
            print(f"Cycle complete: {cycle.summary()}")
            _render_detections(cycle.analyses, top=10, min_score=0.0)
            return EXIT_OK

        streaming = settings.scanner.stream_enabled and not args.no_stream
        stream_config = build_stream_config(settings, symbols) if streaming else None
        asyncio.run(
            _watch_loop(
                settings,
                scanner,
                stream_config=stream_config,
                database=database,
                duration_minutes=args.duration,
            )
        )
        return EXIT_OK
    except KeyboardInterrupt:
        print("\nStopped.")
        return EXIT_OK
    finally:
        alerts.close()
        if database is not None:
            _record_health(database, "scanner", "STOPPED", "Scanner exited", scanner.status())
            database.close()


def _watch_provider(settings: Settings):
    """Market data for the scanner, cached like every other command."""
    return build_market_data(settings.alpaca, settings.data, use_cache=True)


def _persisting_recorder(database: Database):
    """Store detector signals and scores as they are produced."""

    def record(analysis) -> None:
        try:
            if analysis.signals:
                database.detector_signals.record_many(analysis.signals)
                database.opportunities.record(analysis.score)
        except Exception:  # noqa: BLE001 - persistence must not stop a scan
            logger.exception("Could not persist analysis for %s", analysis.symbol)

    return record


def _record_health(
    database: Database | None, component: str, status: str, message: str, detail=None
) -> None:
    if database is None:
        return
    try:
        database.health.record(
            component=component, status=status, message=message, detail=detail
        )
    except Exception:  # noqa: BLE001 - health logging must never stop the scanner
        logger.exception("Could not record health event %s/%s", component, status)


async def _watch_loop(
    settings: Settings,
    scanner,
    *,
    stream_config,
    database: Database | None,
    duration_minutes: float | None,
) -> None:
    """Seed, then either stream or poll until the deadline or an interrupt."""
    import asyncio

    from trading_bot.data.market_stream import StreamError, StreamSupervisor
    from trading_bot.scanner.realtime import next_poll_delay

    loop = asyncio.get_running_loop()
    deadline = (
        loop.time() + duration_minutes * 60 if duration_minutes is not None else None
    )

    seeded = await asyncio.to_thread(scanner.seed)
    print(f"Seeded {len(seeded)} symbols with history.")
    _record_health(database, "scanner", "STARTED", "Scanner seeded", scanner.status())

    supervisor: StreamSupervisor | None = None
    if stream_config is not None:
        supervisor = StreamSupervisor(
            stream_config,
            api_key=settings.alpaca.api_key,
            secret_key=settings.alpaca.secret_key,
            on_health=lambda event: _record_health(
                database, event.component, event.status, event.message, event.detail
            ),
        )
        try:
            await supervisor.start()
            print(f"Streaming {len(stream_config.symbols)} symbols on the "
                  f"{stream_config.feed} feed.\n")
        except StreamError as error:
            # Streaming is an optimisation, not a requirement: the REST path
            # produces the same analysis a bar later. Falling back beats
            # exiting, and the reason is printed rather than buried in a log.
            print(f"Streaming unavailable, falling back to REST polling:\n  {error}\n")
            _record_health(database, "stream", "UNAVAILABLE", str(error))
            supervisor = None

    if supervisor is not None:
        await _consume_stream(supervisor, scanner, deadline=deadline)
        await supervisor.stop()
        return

    delay = settings.scanner.poll_seconds or next_poll_delay(scanner.timeframe)
    while deadline is None or loop.time() < deadline:
        cycle = await asyncio.to_thread(scanner.scan_once)
        logger.info("Scan cycle: %s", cycle.summary())
        if deadline is not None and loop.time() + delay > deadline:
            return
        await asyncio.sleep(delay)


async def _consume_stream(supervisor, scanner, *, deadline: float | None) -> None:
    """Feed streamed bars into the scanner until the deadline."""
    import asyncio

    loop = asyncio.get_running_loop()
    while deadline is None or loop.time() < deadline:
        remaining = None if deadline is None else max(deadline - loop.time(), 0.0)
        event = await supervisor.next_event(timeout=min(remaining or 5.0, 5.0))
        if event is None:
            if not supervisor.running:
                logger.warning("Stream stopped; ending the scan loop")
                return
            continue
        try:
            scanner.handle_event(event)
        except Exception:  # noqa: BLE001 - one bad message is not a dead scanner
            logger.exception("Failed to process %s event for %s", event.kind, event.symbol)


def cmd_backtest(settings: Settings, args: argparse.Namespace) -> int:
    """Simulate a strategy over historical bars and report what it would have done.

    Every figure here is a measurement of the past under an explicit set of
    assumptions about fills and costs, not a forecast.
    """
    symbols = (
        [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
        if args.symbols
        else list(settings.data.watchlist)
    )
    names = (
        available_strategies()
        if args.strategy.strip().lower() == "all"
        else [item.strip() for item in args.strategy.split(",") if item.strip()]
    )

    start = _parse_date(args.start)
    end = _parse_date(args.end)

    costs = (
        FRICTIONLESS
        if args.no_costs
        else CostModel(
            commission_per_trade=args.commission,
            commission_per_share=args.commission_per_share,
            slippage_pct=args.slippage,
        )
    )

    risk = settings.risk
    overrides: dict[str, Any] = {}
    if args.risk_per_trade is not None:
        overrides["max_risk_per_trade_pct"] = args.risk_per_trade
    if args.max_positions is not None:
        overrides["max_open_positions"] = args.max_positions
    if overrides:
        risk = risk.model_copy(update=overrides)

    try:
        request = BacktestRequest(
            symbols=tuple(symbols),
            strategies=tuple(names),
            timeframe=args.timeframe or settings.data.timeframe,
            start=start,
            end=end,
            starting_equity=args.capital,
            costs=costs,
            risk=risk,
            allow_short=args.allow_short,
            min_confidence=args.min_confidence,
            demo=args.demo,
            use_cache=not args.no_cache,
        )
    except (ValueError, StrategyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return EXIT_FAILURE

    if args.demo:
        banner = "!" * 64
        print(f"\n{banner}")
        print("  DEMO MODE \u2014 GENERATED SAMPLE DATA, NOT REAL MARKET DATA")
        print("  Results below describe a random walk, not any real instrument.")
        print(banner)

    print(f"\nSimulating {len(symbols)} symbol(s) with {', '.join(names)}...")
    try:
        result = run_backtest(request, settings)
    except (BacktestDataError, StrategyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return EXIT_FAILURE

    print()
    print(result.summary())

    if args.trades and result.trades:
        _render_trades(result.trades)

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.trade_frame.to_csv(path, index=False)
        print(f"\nTrades written to {path}")

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.as_dict(), indent=2, default=str))
        print(f"Full result written to {path}")

    print(
        "\nThese are simulated results under the stated fill and cost assumptions.\n"
        "Past behaviour of a strategy on historical data is not a forecast."
    )
    return EXIT_OK


def _render_trades(trades: list[dict[str, Any]]) -> None:
    """Print the trade list, one row per closed trade."""
    print()
    print("-" * 100)
    print(
        f"{'SYMBOL':<8}{'DIR':<6}{'ENTRY':>10}{'EXIT':>10}{'QTY':>7}"
        f"{'P&L':>11}{'R':>7}  {'BARS':>5}  REASON"
    )
    print("-" * 100)
    for trade in trades:
        r_multiple = trade.get("r_multiple")
        print(
            f"{trade['symbol']:<8}{trade['direction']:<6}"
            f"{trade['entry_price']:>10.2f}{trade['exit_price']:>10.2f}"
            f"{trade['quantity']:>7.0f}{trade['pnl']:>11.2f}"
            f"{(f'{r_multiple:.2f}' if r_multiple is not None else '-'):>7}"
            f"  {trade['bars_held']:>5}  {trade['exit_reason']}"
        )
    print("-" * 100)


def cmd_dashboard(settings: Settings, args: argparse.Namespace) -> int:
    """Launch the Streamlit dashboard.

    Streamlit owns its own server, so this hands off to it rather than trying to
    run it in-process.
    """
    import subprocess

    app_path = Path(__file__).resolve().parent / "dashboard" / "app.py"
    if not app_path.exists():
        print(f"Dashboard not found at {app_path}", file=sys.stderr)
        return EXIT_FAILURE

    try:
        import streamlit  # noqa: F401
    except ImportError:
        print(
            "Streamlit is not installed. Run:\n\n    pip install -r requirements.txt\n",
            file=sys.stderr,
        )
        return EXIT_FAILURE

    command = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(args.port),
        "--server.address", args.host,
        "--server.headless", "true" if args.headless else "false",
        "--browser.gatherUsageStats", "false",
    ]
    logger.info("Starting the dashboard on http://%s:%d", args.host, args.port)
    print(f"\nDashboard starting at http://{args.host}:{args.port}")
    print("The dashboard is read-only — it cannot place orders.\nPress Ctrl+C to stop.\n")
    try:
        return subprocess.call(command)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
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
        if args.command == "universe":
            return cmd_universe(settings, args)
        if args.command == "detect":
            return cmd_detect(settings, args)
        if args.command == "watch":
            return cmd_watch(settings, args)
        if args.command == "fetch":
            return cmd_fetch(settings, args)
        if args.command == "analyze":
            return cmd_analyze(settings, args)
        if args.command == "signals":
            return cmd_signals(settings, args)
        if args.command == "scan":
            return cmd_scan(settings, args)
        if args.command == "calibrate":
            return cmd_calibrate(settings, args)
        if args.command == "hunt":
            return cmd_hunt(settings, args)
        if args.command == "backtest":
            return cmd_backtest(settings, args)
        if args.command == "dashboard":
            return cmd_dashboard(settings, args)
        if args.command == "cache":
            return cmd_cache(settings, args)
        parser.error(f"Unknown command {args.command!r}")
        return EXIT_FAILURE
    except (MarketDataError, UniverseConfigError, ValueError) as error:
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
