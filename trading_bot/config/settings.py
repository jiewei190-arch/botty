"""Typed application settings loaded from environment variables and ``.env``.

Design notes
------------
* Every settings group is an independent ``BaseSettings`` with its own ``env_prefix``
  so environment variables stay flat and readable (``ALPACA_API_KEY``,
  ``RISK_MAX_RISK_PER_TRADE_PCT``, ...).
* Live trading is protected by a **double lock**: selecting ``TRADING_MODE=live`` is
  not sufficient. ``ENABLE_LIVE_TRADING`` must be true *and*
  ``LIVE_TRADING_CONFIRMATION`` must match an exact phrase. Any other combination
  raises at construction time, so a typo can never route real money orders.
* Settings objects are frozen. Runtime code must not mutate configuration; the
  dashboard edits a copy via :meth:`Settings.with_overrides`.
"""

from __future__ import annotations

import functools
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Repository root: <root>/trading_bot/config/settings.py -> parents[2] == <root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

#: Exact phrase required in ``LIVE_TRADING_CONFIRMATION`` to arm live trading.
LIVE_CONFIRMATION_PHRASE = "I UNDERSTAND THE RISKS"

_BASE_CONFIG = SettingsConfigDict(
    env_file=ENV_FILE,
    env_file_encoding="utf-8",
    extra="ignore",
    frozen=True,
)


class TradingMode(str, Enum):
    """Execution mode. Defaults to :attr:`PAPER` — never live."""

    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"

    @property
    def is_live(self) -> bool:
        return self is TradingMode.LIVE

    @property
    def uses_broker(self) -> bool:
        """True when the mode talks to a broker account (paper or live)."""
        return self in (TradingMode.PAPER, TradingMode.LIVE)


def _split_csv(value: Any) -> Any:
    """Parse ``"AAPL, MSFT"`` into ``["AAPL", "MSFT"]``, leaving lists untouched."""
    if isinstance(value, str):
        return [item.strip().upper() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip().upper() for item in value if str(item).strip()]
    return value


class AlpacaSettings(BaseSettings):
    """Alpaca API credentials and endpoint selection."""

    model_config = _BASE_CONFIG | SettingsConfigDict(env_prefix="ALPACA_")

    api_key: str | None = Field(default=None, description="Alpaca API key ID.")
    secret_key: str | None = Field(default=None, description="Alpaca API secret key.")

    paper_base_url: str = "https://paper-api.alpaca.markets"
    live_base_url: str = "https://api.alpaca.markets"

    #: Market data feed. ``iex`` is free; ``sip`` requires a paid subscription.
    data_feed: str = Field(default="iex", description="One of: iex, sip, delayed_sip, otc.")
    #: Corporate-action adjustment for historical bars. ``all`` avoids split/dividend
    #: artefacts that would otherwise create fake gaps in backtests.
    adjustment: str = Field(default="all", description="One of: raw, split, dividend, all.")

    request_timeout_seconds: float = 30.0
    max_retries: int = 4
    retry_base_delay_seconds: float = 1.0

    @field_validator("data_feed")
    @classmethod
    def _validate_feed(cls, value: str) -> str:
        allowed = {"iex", "sip", "delayed_sip", "otc", "boats", "overnight"}
        normalized = value.strip().lower()
        if normalized not in allowed:
            raise ValueError(f"data_feed must be one of {sorted(allowed)}, got {value!r}")
        return normalized

    @field_validator("adjustment")
    @classmethod
    def _validate_adjustment(cls, value: str) -> str:
        allowed = {"raw", "split", "dividend", "all"}
        normalized = value.strip().lower()
        if normalized not in allowed:
            raise ValueError(f"adjustment must be one of {sorted(allowed)}, got {value!r}")
        return normalized

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key and self.secret_key)


class RiskSettings(BaseSettings):
    """Risk limits. Enforced by the risk manager in Phase 4 — defined now so that
    every later component reads the same numbers."""

    model_config = _BASE_CONFIG | SettingsConfigDict(env_prefix="RISK_")

    #: Percentage of account equity risked on a single trade (distance to stop).
    max_risk_per_trade_pct: float = Field(default=1.0, gt=0, le=100)
    #: Trading halts for the day once realised + unrealised losses exceed this.
    max_daily_loss_pct: float = Field(default=3.0, gt=0, le=100)
    #: Hard cap on simultaneously open positions.
    max_open_positions: int = Field(default=5, ge=1, le=100)
    #: Combined market value of open positions as a share of equity.
    max_portfolio_exposure_pct: float = Field(default=60.0, gt=0, le=100)
    #: Largest share of equity a single position may consume.
    max_position_size_pct: float = Field(default=20.0, gt=0, le=100)
    #: Minimum acceptable reward-to-risk ratio; signals below this are rejected.
    min_risk_reward: float = Field(default=2.0, gt=0)
    #: Fallback stop distance when a strategy does not supply one.
    default_stop_loss_pct: float = Field(default=2.0, gt=0, le=100)
    #: Fallback take-profit distance when a strategy does not supply one.
    default_take_profit_pct: float = Field(default=5.0, gt=0, le=100)
    #: Consecutive losing trades that trigger a cooldown.
    consecutive_loss_limit: int = Field(default=3, ge=1)
    #: Duration of the cooldown after hitting the consecutive-loss limit.
    cooldown_minutes: int = Field(default=60, ge=0)
    #: Minimum confidence (0-100) a signal needs before it may be traded.
    min_confidence: float = Field(default=60.0, ge=0, le=100)
    #: Equity every position size is calculated from.
    #:
    #: Set this to the balance of the account you actually trade. It is stated
    #: rather than read from a broker on purpose: the data feed and the account
    #: you trade need not be the same place, and sizing a real position against
    #: an unrelated broker's balance — a data-only account holding nothing —
    #: would produce share counts with no relationship to the money at risk.
    account_equity: float = Field(default=10_000.0, gt=0)

    @model_validator(mode="after")
    def _validate_coherence(self) -> RiskSettings:
        if self.max_position_size_pct > self.max_portfolio_exposure_pct:
            raise ValueError(
                "RISK_MAX_POSITION_SIZE_PCT cannot exceed RISK_MAX_PORTFOLIO_EXPOSURE_PCT"
            )
        if self.max_risk_per_trade_pct > self.max_daily_loss_pct:
            raise ValueError(
                "RISK_MAX_RISK_PER_TRADE_PCT cannot exceed RISK_MAX_DAILY_LOSS_PCT: "
                "a single trade would be able to breach the daily loss limit"
            )
        return self


class DataSettings(BaseSettings):
    """Market data, watchlist and storage locations."""

    model_config = _BASE_CONFIG | SettingsConfigDict(env_prefix="DATA_")

    #: Symbols scanned for opportunities.
    watchlist: Annotated[list[str], NoDecode] = Field(
        default=["AAPL", "NVDA", "TSLA", "AMD", "MSFT", "META", "AMZN", "GOOGL", "SPY", "QQQ"]
    )
    #: Default bar size for analysis (see :mod:`trading_bot.utils.timeframes`).
    timeframe: str = "15Min"
    #: Bars of history pulled for indicator warm-up on each scan.
    lookback_bars: int = Field(default=300, ge=50, le=10_000)

    cache_enabled: bool = True
    cache_dir: Path = PROJECT_ROOT / "storage" / "cache"
    database_path: Path = PROJECT_ROOT / "storage" / "trading_bot.db"

    @field_validator("watchlist", mode="before")
    @classmethod
    def _parse_watchlist(cls, value: Any) -> Any:
        return _split_csv(value)

    @field_validator("watchlist")
    @classmethod
    def _dedupe_watchlist(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("DATA_WATCHLIST must contain at least one symbol")
        seen: dict[str, None] = {}
        for symbol in value:
            seen.setdefault(symbol, None)
        return list(seen)


class LoggingSettings(BaseSettings):
    """Logging destinations and verbosity."""

    model_config = _BASE_CONFIG | SettingsConfigDict(env_prefix="LOG_")

    level: str = "INFO"
    directory: Path = PROJECT_ROOT / "logs"
    #: Emit a machine-readable JSON-lines stream alongside the human-readable log.
    json_enabled: bool = True
    max_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)
    backup_count: int = Field(default=5, ge=0)

    @field_validator("level")
    @classmethod
    def _validate_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        normalized = value.strip().upper()
        if normalized not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(allowed)}, got {value!r}")
        return normalized


class ScannerSettings(BaseSettings):
    """The continuously running market scanner (Phase 1)."""

    model_config = _BASE_CONFIG | SettingsConfigDict(env_prefix="SCANNER_")

    #: Categories from :mod:`trading_bot.config.universe` to watch.
    categories: Annotated[list[str], NoDecode] = Field(
        default=["INDEX_ETFS", "MEGA_CAPS"]
    )
    #: Symbols added on top of the categories.
    extra_symbols: Annotated[list[str], NoDecode] = Field(default=[])
    #: Symbols removed after the categories are expanded.
    exclude_symbols: Annotated[list[str], NoDecode] = Field(default=[])
    #: Optional JSON file overriding the built-in categories.
    universe_file: Path | None = None

    #: Bar size the detectors analyse.
    timeframe: str = "5Min"
    #: Bars of history fetched before live data starts, for detector baselines.
    lookback_bars: int = Field(default=400, ge=50, le=10_000)
    #: Bars retained per symbol while running.
    max_bars: int = Field(default=1_500, ge=100, le=50_000)

    #: Open a websocket. When false the scanner polls REST on ``poll_seconds``.
    stream_enabled: bool = True
    stream_bars: bool = True
    stream_trades: bool = False
    #: Quotes arrive per tick and are off by default: thirty symbols of quotes
    #: is thousands of messages a second, and the detectors read bars.
    stream_quotes: bool = False
    #: Seconds between REST cycles. 0 tracks the timeframe automatically.
    poll_seconds: int = Field(default=0, ge=0, le=3_600)
    #: Seconds of websocket silence before a health warning is logged.
    stream_staleness_seconds: float = Field(default=300.0, gt=0)

    #: Analyse only during regular hours.
    regular_hours_only: bool = False

    #: Scoring weights. They are renormalised, so relative size is what matters.
    weight_volume: float = Field(default=0.25, ge=0)
    weight_momentum: float = Field(default=0.25, ge=0)
    weight_breakout: float = Field(default=0.25, ge=0)
    weight_gap: float = Field(default=0.15, ge=0)
    weight_volatility: float = Field(default=0.10, ge=0)

    @field_validator("categories", "extra_symbols", "exclude_symbols", mode="before")
    @classmethod
    def _parse_lists(cls, value: Any) -> Any:
        return _split_csv(value)

    @field_validator("timeframe")
    @classmethod
    def _validate_timeframe(cls, value: str) -> str:
        from trading_bot.utils.timeframes import Timeframe

        return Timeframe.parse(value).label

    @model_validator(mode="after")
    def _validate_weights(self) -> ScannerSettings:
        total = (
            self.weight_volume
            + self.weight_momentum
            + self.weight_breakout
            + self.weight_gap
            + self.weight_volatility
        )
        if total <= 0:
            raise ValueError("At least one SCANNER_WEIGHT_* value must be greater than zero")
        return self


class DetectorSettings(BaseSettings):
    """Detector thresholds worth tuning from the environment.

    Only the five that change behaviour most are exposed here. The rest live in
    :mod:`trading_bot.detectors.base`, where each default is stated next to the
    reasoning for it — a threshold nobody can justify is one nobody can safely
    change, and burying all thirty in ``.env`` would invite exactly that.
    """

    model_config = _BASE_CONFIG | SettingsConfigDict(env_prefix="DETECTOR_")

    #: Relative volume at which participation stops being ordinary.
    volume_threshold: float = Field(default=1.5, gt=0)
    #: Move size in ATR units that counts as momentum.
    momentum_atr_threshold: float = Field(default=1.5, gt=0)
    #: Percentage past a level a close must reach to count as a break.
    breakout_buffer_pct: float = Field(default=0.05, ge=0, le=10)
    #: Percentage gap that counts as a gap.
    gap_threshold_pct: float = Field(default=1.0, gt=0, le=100)
    #: Bar-range expansion ratio that counts as abnormal.
    volatility_threshold: float = Field(default=1.6, gt=0)


class AlertSettings(BaseSettings):
    """When and where the scanner interrupts you."""

    model_config = _BASE_CONFIG | SettingsConfigDict(env_prefix="ALERT_")

    #: Overall score (0-10) required before an alert is raised.
    min_score: float = Field(default=5.0, ge=0, le=10)
    #: Silence for one symbol and direction after an alert.
    cooldown_seconds: int = Field(default=900, ge=0)
    #: Score improvement that re-opens the cooldown early.
    escalation_delta: float = Field(default=1.5, ge=0)
    #: Hard cap per symbol per trading day.
    max_per_symbol_per_session: int = Field(default=5, ge=1)
    #: Minimum directional agreement (0-1) among the signals. 0 disables it.
    min_agreement: float = Field(default=0.0, ge=0, le=1)
    #: Suppress alerts whose signals cancel out to no direction.
    require_direction: bool = True

    console_enabled: bool = True
    log_enabled: bool = True
    database_enabled: bool = True


class Settings(BaseSettings):
    """Root settings object composing every configuration group."""

    model_config = _BASE_CONFIG

    trading_mode: TradingMode = TradingMode.PAPER

    # --- Live trading double lock -------------------------------------------------
    enable_live_trading: bool = Field(
        default=False,
        description="First lock. Must be true for TRADING_MODE=live to be accepted.",
    )
    live_trading_confirmation: str = Field(
        default="",
        description=f"Second lock. Must equal {LIVE_CONFIRMATION_PHRASE!r} exactly.",
    )

    alpaca: AlpacaSettings = Field(default_factory=AlpacaSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    scanner: ScannerSettings = Field(default_factory=ScannerSettings)
    detectors: DetectorSettings = Field(default_factory=DetectorSettings)
    alerts: AlertSettings = Field(default_factory=AlertSettings)

    project_root: Path = PROJECT_ROOT

    @model_validator(mode="after")
    def _enforce_live_trading_locks(self) -> Settings:
        if self.trading_mode is not TradingMode.LIVE:
            return self
        if not self.enable_live_trading:
            raise ValueError(
                "TRADING_MODE=live requires ENABLE_LIVE_TRADING=true. "
                "Refusing to start: live trading is not armed."
            )
        if self.live_trading_confirmation.strip() != LIVE_CONFIRMATION_PHRASE:
            raise ValueError(
                "TRADING_MODE=live requires LIVE_TRADING_CONFIRMATION="
                f"{LIVE_CONFIRMATION_PHRASE!r}. Refusing to start."
            )
        return self

    @property
    def is_live(self) -> bool:
        """True only when live trading passed both locks."""
        return self.trading_mode is TradingMode.LIVE

    @property
    def broker_base_url(self) -> str:
        return self.alpaca.live_base_url if self.is_live else self.alpaca.paper_base_url

    def ensure_directories(self) -> None:
        """Create the runtime directories this configuration points at."""
        self.logging.directory.mkdir(parents=True, exist_ok=True)
        self.data.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data.database_path.parent.mkdir(parents=True, exist_ok=True)

    def with_overrides(self, **overrides: Any) -> Settings:
        """Return a new Settings with ``overrides`` applied (settings are frozen)."""
        return self.model_copy(update=overrides)

    def redacted_dict(self) -> dict[str, Any]:
        """Configuration dump safe to print or render in the dashboard."""
        payload = self.model_dump(mode="json")
        alpaca = payload.get("alpaca", {})
        for key in ("api_key", "secret_key"):
            if alpaca.get(key):
                alpaca[key] = _mask(str(alpaca[key]))
        if payload.get("live_trading_confirmation"):
            payload["live_trading_confirmation"] = "***set***"
        return payload


def _mask(secret: str) -> str:
    """Show only the last 4 characters of a secret."""
    if len(secret) <= 4:
        return "*" * len(secret)
    return f"{'*' * (len(secret) - 4)}{secret[-4:]}"


def load_settings(**overrides: Any) -> Settings:
    """Build a fresh :class:`Settings` from the environment.

    Raises ``pydantic.ValidationError`` when configuration is invalid — including
    an unarmed live-trading attempt.
    """
    return Settings(**overrides)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings. Use :func:`load_settings` in tests."""
    return load_settings()
