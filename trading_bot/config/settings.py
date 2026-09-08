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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"
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
        return self in (TradingMode.PAPER, TradingMode.LIVE)


def _split_csv(value: Any) -> Any:
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
    data_feed: str = Field(default="iex", description="One of: iex, sip, delayed_sip, otc.")
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
    """Risk limits shared by backtest, paper and eventual live execution."""

    model_config = _BASE_CONFIG | SettingsConfigDict(env_prefix="RISK_")

    max_risk_per_trade_pct: float = Field(default=1.0, gt=0, le=100)
    max_daily_loss_pct: float = Field(default=3.0, gt=0, le=100)
    max_open_positions: int = Field(default=5, ge=1, le=100)
    max_portfolio_exposure_pct: float = Field(default=60.0, gt=0, le=100)
    max_position_size_pct: float = Field(default=20.0, gt=0, le=100)
    min_risk_reward: float = Field(default=2.0, gt=0)
    default_stop_loss_pct: float = Field(default=2.0, gt=0, le=100)
    default_take_profit_pct: float = Field(default=5.0, gt=0, le=100)
    consecutive_loss_limit: int = Field(default=3, ge=1)
    cooldown_minutes: int = Field(default=60, ge=0)
    # Swing mode deliberately defaults to a stricter floor than the generic bot.
    min_confidence: float = Field(default=70.0, ge=0, le=100)
    account_equity: float = Field(default=10_000.0, gt=0)

    @model_validator(mode="after")
    def _validate_coherence(self) -> "RiskSettings":
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

    watchlist: Annotated[list[str], NoDecode] = Field(
        default=["AAPL", "NVDA", "TSLA", "AMD", "MSFT", "META", "AMZN", "GOOGL", "SPY", "QQQ"]
    )
    # Botty is a swing bot: daily bars define the setup; intraday bars are for
    # later entry refinement, not for converting the system into a day trader.
    timeframe: str = "1Day"
    # 500 daily bars gives roughly two years of context and fully warms the 200 EMA.
    lookback_bars: int = Field(default=500, ge=50, le=10_000)

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
    model_config = _BASE_CONFIG | SettingsConfigDict(env_prefix="LOG_")

    level: str = "INFO"
    directory: Path = PROJECT_ROOT / "logs"
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


class AutomationSettings(BaseSettings):
    """Always-on runner and outbound notification settings."""

    model_config = _BASE_CONFIG | SettingsConfigDict(env_prefix="AUTO_")

    closed_poll_seconds: int = Field(default=300, ge=15, le=3600)
    webhook_url: str | None = None
    webhook_kind: str = "discord"
    webhook_timeout_seconds: float = Field(default=10.0, gt=0, le=60)

    @field_validator("webhook_kind")
    @classmethod
    def _validate_webhook_kind(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"discord", "slack"}:
            raise ValueError("AUTO_WEBHOOK_KIND must be 'discord' or 'slack'")
        return normalized


class Settings(BaseSettings):
    """Root settings object composing every configuration group."""

    model_config = _BASE_CONFIG

    trading_mode: TradingMode = TradingMode.PAPER
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
    automation: AutomationSettings = Field(default_factory=AutomationSettings)
    project_root: Path = PROJECT_ROOT

    @model_validator(mode="after")
    def _enforce_live_trading_locks(self) -> "Settings":
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
        return self.trading_mode is TradingMode.LIVE

    @property
    def broker_base_url(self) -> str:
        return self.alpaca.live_base_url if self.is_live else self.alpaca.paper_base_url

    def ensure_directories(self) -> None:
        self.logging.directory.mkdir(parents=True, exist_ok=True)
        self.data.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data.database_path.parent.mkdir(parents=True, exist_ok=True)

    def with_overrides(self, **overrides: Any) -> "Settings":
        return self.model_copy(update=overrides)

    def redacted_dict(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        alpaca = payload.get("alpaca", {})
        for key in ("api_key", "secret_key"):
            if alpaca.get(key):
                alpaca[key] = _mask(str(alpaca[key]))
        if payload.get("live_trading_confirmation"):
            payload["live_trading_confirmation"] = "***set***"
        automation = payload.get("automation", {})
        if automation.get("webhook_url"):
            automation["webhook_url"] = "***set***"
        return payload


def _mask(secret: str) -> str:
    if len(secret) <= 4:
        return "*" * len(secret)
    return f"{'*' * (len(secret) - 4)}{secret[-4:]}"


def load_settings(**overrides: Any) -> Settings:
    return Settings(**overrides)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
