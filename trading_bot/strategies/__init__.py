"""Trading strategy registry and shared strategy contract."""

from __future__ import annotations

from trading_bot.strategies.base_strategy import (
    BaseStrategy,
    Condition,
    ExitReason,
    ExitSignal,
    Position,
    Signal,
    SignalDirection,
    StrategyConfig,
    StrategyError,
    explain_blockers,
    score_conditions,
)
from trading_bot.strategies.breakout_strategy import BreakoutConfig, BreakoutStrategy
from trading_bot.strategies.mean_reversion import MeanReversionConfig, MeanReversionStrategy
from trading_bot.strategies.momentum_strategy import MomentumConfig, MomentumStrategy
from trading_bot.strategies.swing_quality import SwingQualityConfig, SwingQualityStrategy

STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    MomentumStrategy.name: MomentumStrategy,
    MeanReversionStrategy.name: MeanReversionStrategy,
    BreakoutStrategy.name: BreakoutStrategy,
    SwingQualityStrategy.name: SwingQualityStrategy,
}

CONFIG_REGISTRY: dict[str, type[StrategyConfig]] = {
    MomentumStrategy.name: MomentumConfig,
    MeanReversionStrategy.name: MeanReversionConfig,
    BreakoutStrategy.name: BreakoutConfig,
    SwingQualityStrategy.name: SwingQualityConfig,
}


def available_strategies() -> list[str]:
    """Return registered strategy names in stable sorted order."""
    return sorted(STRATEGY_REGISTRY)


def register_strategy(
    strategy: type[BaseStrategy], config: type[StrategyConfig] | None = None
) -> None:
    """Add a strategy to the registry so all callers can select it by name."""
    name = getattr(strategy, "name", "")
    if not name or name == "base":
        raise StrategyError(
            f"{strategy.__name__} must define a unique class-level `name` before registration"
        )
    if name in STRATEGY_REGISTRY and STRATEGY_REGISTRY[name] is not strategy:
        raise StrategyError(
            f"Strategy name {name!r} is already registered to "
            f"{STRATEGY_REGISTRY[name].__name__}"
        )
    STRATEGY_REGISTRY[name] = strategy
    if config is not None:
        CONFIG_REGISTRY[name] = config


def build_strategy(
    name: str,
    config: StrategyConfig | None = None,
    indicators=None,
    **overrides,
) -> BaseStrategy:
    """Construct a registered strategy by name with optional typed overrides."""
    key = name.strip().lower()
    strategy_class = STRATEGY_REGISTRY.get(key)
    if strategy_class is None:
        raise StrategyError(
            f"Unknown strategy {name!r}. Available: {', '.join(available_strategies())}"
        )
    if config is not None and overrides:
        raise StrategyError("Pass either a config object or overrides, not both")
    if overrides:
        config_class = CONFIG_REGISTRY.get(key, StrategyConfig)
        valid = {field.name for field in config_class.__dataclass_fields__.values()}
        unknown = sorted(set(overrides) - valid)
        if unknown:
            raise StrategyError(
                f"{key} has no parameter(s) {unknown}. "
                f"Valid parameters: {', '.join(sorted(valid))}"
            )
        config = config_class(**overrides)
    return strategy_class(config, indicators)


__all__ = [
    "BaseStrategy",
    "StrategyConfig",
    "StrategyError",
    "Signal",
    "SignalDirection",
    "ExitSignal",
    "ExitReason",
    "Position",
    "Condition",
    "score_conditions",
    "explain_blockers",
    "MomentumStrategy",
    "MomentumConfig",
    "MeanReversionStrategy",
    "MeanReversionConfig",
    "BreakoutStrategy",
    "BreakoutConfig",
    "SwingQualityStrategy",
    "SwingQualityConfig",
    "STRATEGY_REGISTRY",
    "CONFIG_REGISTRY",
    "available_strategies",
    "build_strategy",
    "register_strategy",
]
