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

    #: Option data feed. ``indicative`` is what the free (Basic) plan serves;
    #: ``opra`` is the consolidated options tape and needs a paid subscription.
    #: Stated rather than left to the SDK default so a failure depends on the
    #: configuration rather than on which plan the account happens to hold.
    options_feed: str = Field(default="indicative", description="One of: indicative, opra.")

    @field_validator("options_feed")
    @classmethod
    def _validate_options_feed(cls, value: str) -> str:
        allowed = {"indicative", "opra"}
        normalized = value.strip().lower()
        if normalized not in allowed:
            raise ValueError(
                f"options_feed must be one of {sorted(allowed)}, got {value!r}"
            )
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
    #:
    #: Daily, to match the rest of a swing bot. The detectors work on any
    #: timeframe, but a 5-minute scanner alerting into a strategy that holds for
    #: weeks is answering a different question from the one being asked, and the
    #: two disagree most at exactly the moments that matter. Set a finer bar size
    #: here only to time an entry that daily bars already justified.
    timeframe: str = "1Day"
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

class AutomationSettings(BaseSettings):
    """Always-on runner and outbound notification settings."""

    model_config = _BASE_CONFIG | SettingsConfigDict(env_prefix="AUTO_")

    closed_poll_seconds: int = Field(default=300, ge=15, le=3600)
    open_poll_seconds: int = Field(default=60, ge=15, le=900)
    max_signal_age_hours: float = Field(default=120.0, gt=0, le=720)
    max_order_failures_per_scan: int = Field(default=2, ge=1, le=20)
    #: Optional Discord or Slack incoming-webhook URL. Kept out of logs/config dumps.
    webhook_url: str | None = None
    webhook_kind: str = "discord"
    webhook_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    #: Optional Socket Mode credentials for interactive Slack status checks.
    slack_bot_token: str | None = None
    slack_app_token: str | None = None

    @field_validator("webhook_kind")
    @classmethod
    def _validate_webhook_kind(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"discord", "slack"}:
            raise ValueError("AUTO_WEBHOOK_KIND must be 'discord' or 'slack'")
        return normalized

    @model_validator(mode="after")
    def _validate_slack_socket_tokens(self) -> AutomationSettings:
        if bool(self.slack_bot_token) != bool(self.slack_app_token):
            raise ValueError(
                "AUTO_SLACK_BOT_TOKEN and AUTO_SLACK_APP_TOKEN must be set together"
            )
        return self


class OptionsSettings(BaseSettings):
    """Long-option swing rules. Loss is capped at premium paid."""

    model_config = _BASE_CONFIG | SettingsConfigDict(env_prefix="OPTIONS_")

    enabled: bool = True
    alert_only: bool = True
    #: Days to expiration, set from the holding window rather than chosen.
    #:
    #: The equity thesis runs 7-30 calendar days and the position is closed once
    #: ``exit_before_expiry_days`` remain, so the shortest contract worth buying
    #: is ``planned_max_hold_days + exit_before_expiry_days``. The old floor of 7
    #: bought contracts that were already past the exit rule on the day they were
    #: opened — a 7-day option held for three weeks is not a swing trade, it is a
    #: donation. ``_validate_options`` enforces the relationship.
    min_dte: int = Field(default=45, ge=1, le=365)
    max_dte: int = Field(default=90, ge=7, le=365)
    exceptional_max_dte: int = Field(default=120, ge=7, le=730)
    exceptional_min_confidence: float = Field(default=90.0, ge=70, le=100)
    target_dte: int = Field(default=60, ge=7, le=365)
    target_delta: float = Field(default=0.60, ge=0.35, le=0.80)
    min_abs_delta: float = Field(default=0.50, ge=0.20, le=0.80)
    max_abs_delta: float = Field(default=0.70, ge=0.40, le=0.95)
    max_spread_pct: float = Field(default=12.0, gt=0, le=50)
    #: Contracts traded today. Defaults to 0 because Alpaca's option **snapshot**
    #: does not carry a daily bar — only the latest trade, quote, IV and greeks —
    #: so nothing fills this in on the live path. It was 10, which silently
    #: rejected every contract ever offered. Left configurable for a caller that
    #: supplies volume from somewhere else; the spread and open-interest gates
    #: below are what actually keep an illiquid contract out, and the spread is
    #: the more honest measure anyway since it is illiquidity priced in real time.
    min_daily_volume: int = Field(default=0, ge=0)
    min_open_interest: int = Field(default=100, ge=0)
    #: Moneyness band used when the data feed returns no greeks. Alpaca's free
    #: (indicative) option feed may omit them, and the delta filter would then
    #: reject everything. Expressed as percent of spot: negative is in the money
    #: for a call. Roughly brackets the 0.50-0.70 delta band at these tenors.
    fallback_min_moneyness_pct: float = Field(default=-6.0, ge=-25, le=0)
    fallback_max_moneyness_pct: float = Field(default=2.0, ge=0, le=25)
    min_premium_per_trade: float = Field(default=500.0, gt=0)
    max_premium_per_trade: float = Field(default=1_000.0, gt=0)
    max_total_premium: float = Field(default=2_000.0, gt=0)
    max_contracts_per_trade: int = Field(default=4, ge=1, le=4)
    max_total_contracts: int = Field(default=4, ge=1, le=20)
    max_open_positions: int = Field(default=2, ge=1, le=5)
    #: How many underlyings a single scan may look up a chain for, and therefore
    #: how many swing alerts it can send. This bounds API usage; it is not a risk
    #: control. The premium and position caps above govern *orders* — using them
    #: to gate alerts as well meant a manually-traded account went quiet after
    #: the second name in a scan, losing the setups ranked below it.
    max_alerts_per_scan: int = Field(default=8, ge=1, le=25)
    #: The holding window the contract has to survive. Matches the equity swing
    #: strategy's 7-30 calendar days; it was 60, which no longer describes what
    #: the strategy does.
    planned_min_hold_days: int = Field(default=7, ge=1, le=90)
    planned_max_hold_days: int = Field(default=30, ge=7, le=180)
    profit_target_pct: float = Field(default=50.0, gt=0, le=500)
    stop_loss_pct: float = Field(default=35.0, gt=0, lt=100)
    #: Close the position once this many days remain. Theta and gamma both
    #: accelerate inside the last few weeks, and a swing thesis has no edge
    #: there.
    exit_before_expiry_days: int = Field(default=14, ge=7, le=60)

    @model_validator(mode="after")
    def _validate_options(self) -> OptionsSettings:
        if not self.min_dte <= self.target_dte <= self.max_dte:
            raise ValueError("OPTIONS_TARGET_DTE must be between MIN_DTE and MAX_DTE")
        if self.min_abs_delta > self.max_abs_delta:
            raise ValueError("OPTIONS_MIN_ABS_DELTA cannot exceed MAX_ABS_DELTA")
        if not self.min_abs_delta <= self.target_delta <= self.max_abs_delta:
            raise ValueError("OPTIONS_TARGET_DELTA must be inside the delta range")
        if self.planned_min_hold_days > self.planned_max_hold_days:
            raise ValueError("planned minimum hold cannot exceed maximum hold")
        if self.max_premium_per_trade > self.max_total_premium:
            raise ValueError("per-trade premium cannot exceed total premium")
        if self.min_premium_per_trade > self.max_premium_per_trade:
            raise ValueError("minimum premium cannot exceed maximum premium")
        if self.exceptional_max_dte < self.max_dte:
            raise ValueError("exceptional max DTE cannot be below normal max DTE")
        required = self.planned_max_hold_days + self.exit_before_expiry_days
        if self.min_dte < required:
            raise ValueError(
                f"OPTIONS_MIN_DTE must be at least {required} "
                f"(planned_max_hold_days {self.planned_max_hold_days} + "
                f"exit_before_expiry_days {self.exit_before_expiry_days}): a "
                "contract bought below that is already past the exit rule"
            )
        if self.fallback_min_moneyness_pct > self.fallback_max_moneyness_pct:
            raise ValueError("fallback moneyness band is inverted")
        return self


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
    scanner: ScannerSettings = Field(default_factory=ScannerSettings)
    detectors: DetectorSettings = Field(default_factory=DetectorSettings)
    alerts: AlertSettings = Field(default_factory=AlertSettings)
    automation: AutomationSettings = Field(default_factory=AutomationSettings)
    options: OptionsSettings = Field(default_factory=OptionsSettings)
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
        return self.trading_mode is TradingMode.LIVE

    @property
    def broker_base_url(self) -> str:
        return self.alpaca.live_base_url if self.is_live else self.alpaca.paper_base_url

    def ensure_directories(self) -> None:
        self.logging.directory.mkdir(parents=True, exist_ok=True)
        self.data.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data.database_path.parent.mkdir(parents=True, exist_ok=True)

    def with_overrides(self, **overrides: Any) -> Settings:
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
        for key in ("webhook_url", "slack_bot_token", "slack_app_token"):
            if automation.get(key):
                automation[key] = "***set***"
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
