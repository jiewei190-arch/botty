"""Turning configuration into a running scanner.

One place that knows how ``Settings`` maps onto detectors, weights, channels and
stream subscriptions. Without it that mapping ends up duplicated between the CLI
and the dashboard, and the two slowly start scanning differently — which is the
sort of divergence nobody notices until a signal appears in one and not the
other.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from trading_bot.alerts.alert_manager import AlertConfig, AlertManager
from trading_bot.alerts.channels import (
    AlertChannel,
    ConsoleChannel,
    DatabaseChannel,
    LogChannel,
)
from trading_bot.config.settings import Settings
from trading_bot.config.universe import UniverseConfigError, load_catalogue, normalize_symbols
from trading_bot.data.database import Database
from trading_bot.data.market_data import MarketDataProvider
from trading_bot.data.market_stream import StreamConfig
from trading_bot.detectors.base import DetectorConfig
from trading_bot.detectors.engine import DetectorEngine
from trading_bot.detectors.scoring import ScoringWeights
from trading_bot.scanner.realtime import RealtimeScanner, RealtimeScannerConfig

logger = logging.getLogger(__name__)


def resolve_symbols(settings: Settings, *, override: list[str] | None = None) -> tuple[str, ...]:
    """The symbols to watch.

    An explicit list on the command line wins outright; otherwise the configured
    categories are expanded, extras added and exclusions removed.
    """
    if override:
        return normalize_symbols(override)
    catalogue = load_catalogue(settings.scanner.universe_file)
    symbols = catalogue.resolve(
        settings.scanner.categories,
        include=settings.scanner.extra_symbols,
        exclude=settings.scanner.exclude_symbols,
    )
    if not symbols:
        raise UniverseConfigError(
            "The configured universe is empty. Check SCANNER_CATEGORIES against "
            f"the available categories: {', '.join(catalogue.names)}"
        )
    return symbols


def build_detector_config(settings: Settings) -> DetectorConfig:
    """Apply the environment-tunable thresholds over the documented defaults."""
    base = DetectorConfig()
    tuning = settings.detectors
    return DetectorConfig(
        volume=replace(base.volume, threshold=tuning.volume_threshold),
        momentum=replace(base.momentum, atr_threshold=tuning.momentum_atr_threshold),
        breakout=replace(base.breakout, buffer_pct=tuning.breakout_buffer_pct),
        gap=replace(base.gap, threshold_pct=tuning.gap_threshold_pct),
        volatility=replace(base.volatility, threshold=tuning.volatility_threshold),
    )


def build_weights(settings: Settings) -> ScoringWeights:
    scanner = settings.scanner
    return ScoringWeights(
        unusual_volume=scanner.weight_volume,
        momentum=scanner.weight_momentum,
        breakout=scanner.weight_breakout,
        gap=scanner.weight_gap,
        volatility_expansion=scanner.weight_volatility,
    )


def build_engine(settings: Settings) -> DetectorEngine:
    return DetectorEngine(
        build_detector_config(settings), weights=build_weights(settings)
    )


def build_alert_manager(
    settings: Settings,
    *,
    database: Database | None = None,
    extra_channels: list[AlertChannel] | None = None,
    prime_from_database: bool = True,
) -> AlertManager:
    """Assemble the alert manager and its channels.

    When a database is supplied the manager is primed from it, so a restart
    honours cooldowns already in effect instead of re-announcing a morning's
    worth of setups.
    """
    config = AlertConfig(
        min_score=settings.alerts.min_score,
        cooldown_seconds=settings.alerts.cooldown_seconds,
        escalation_delta=settings.alerts.escalation_delta,
        max_per_symbol_per_session=settings.alerts.max_per_symbol_per_session,
        min_agreement=settings.alerts.min_agreement,
        require_direction=settings.alerts.require_direction,
    )
    channels: list[AlertChannel] = []
    if settings.alerts.console_enabled:
        channels.append(ConsoleChannel())
    if settings.alerts.log_enabled:
        channels.append(LogChannel())
    if settings.alerts.database_enabled and database is not None:
        channels.append(DatabaseChannel(database.alerts))
    channels.extend(extra_channels or [])

    manager = AlertManager(channels, config)
    if database is not None and prime_from_database:
        _prime(manager, database)
    return manager


def _prime(manager: AlertManager, database: Database) -> None:
    """Restore cooldown state from stored alerts, tolerating a missing table."""
    from datetime import datetime, timedelta, timezone

    since = datetime.now(timezone.utc) - timedelta(days=1)
    try:
        rows = database.alerts.latest_per_key(since=since)
    except Exception:  # noqa: BLE001 - an un-migrated database must not block a scan
        logger.warning("Could not read stored alerts to prime cooldowns", exc_info=True)
        return
    for row in rows:
        stamp = row.get("ts")
        key = row.get("dedupe_key")
        if not key or not stamp:
            continue
        try:
            moment = datetime.fromisoformat(str(stamp))
        except ValueError:
            continue
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        manager.prime(
            str(key),
            timestamp=moment,
            score=float(row.get("score") or 0.0),
            count=int(row.get("count") or 1),
        )
    if rows:
        logger.info("Primed alert cooldowns for %d keys from the database", len(rows))


def build_scanner_config(
    settings: Settings, symbols: tuple[str, ...]
) -> RealtimeScannerConfig:
    scanner = settings.scanner
    return RealtimeScannerConfig(
        symbols=symbols,
        timeframe=scanner.timeframe,
        lookback_bars=scanner.lookback_bars,
        poll_seconds=scanner.poll_seconds,
        max_bars=scanner.max_bars,
        regular_hours_only=scanner.regular_hours_only,
    )


def build_scanner(
    settings: Settings,
    provider: MarketDataProvider,
    *,
    symbols: tuple[str, ...] | None = None,
    alerts: AlertManager | None = None,
) -> RealtimeScanner:
    watched = symbols or resolve_symbols(settings)
    return RealtimeScanner(
        provider,
        config=build_scanner_config(settings, watched),
        engine=build_engine(settings),
        alerts=alerts,
    )


def build_stream_config(settings: Settings, symbols: tuple[str, ...]) -> StreamConfig:
    """Subscription settings.

    Minute bars are always requested when bars are enabled, whatever the
    analysis timeframe: Alpaca streams minute bars, and the scanner aggregates
    them up. Asking for the analysis timeframe directly is not an option the
    API offers.
    """
    scanner = settings.scanner
    return StreamConfig(
        symbols=symbols,
        feed=settings.alpaca.data_feed,
        subscribe_bars=scanner.stream_bars,
        subscribe_trades=scanner.stream_trades,
        subscribe_quotes=scanner.stream_quotes,
        staleness_warning_seconds=scanner.stream_staleness_seconds,
    )


__all__ = [
    "build_alert_manager",
    "build_detector_config",
    "build_engine",
    "build_scanner",
    "build_scanner_config",
    "build_stream_config",
    "build_weights",
    "resolve_symbols",
]
