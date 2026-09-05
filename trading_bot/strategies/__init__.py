"""Strategy engine (Phase 3).

Three strategies with deliberately different edges, behind one interface:

======================  ==================================  =====================
Strategy                Thesis                              Works when
======================  ==================================  =====================
:class:`MomentumStrategy`     Trends continue                Markets are trending
:class:`MeanReversionStrategy` Stretched prices snap back    Markets are ranging
:class:`BreakoutStrategy`     Compression precedes expansion  Ranges are resolving
======================  ==================================  =====================

They are meant to disagree. A momentum strategy and a mean-reversion strategy
looking at the same oversold chart should reach opposite conclusions — that is
the point of running more than one, and why the scanner (Phase 5) ranks their
output rather than averaging it.

Selecting a strategy by name::

    from trading_bot.strategies import build_strategy, available_strategies

    strategy = build_strategy("momentum")
    signal = strategy.generate_signal("AAPL", strategy.prepare(bars))

Adding your own: subclass :class:`BaseStrategy`, implement ``evaluate``, and
register it with :func:`register_strategy`. Nothing else in the system needs to
change — the scanner, backtester and dashboard all work through this registry.
"""

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
from trading_bot.strategies.mean_reversion import (
    MeanReversionConfig,
    MeanReversionStrategy,
)
from trading_bot.strategies.momentum_strategy import MomentumConfig, MomentumStrategy

#: Name to class. Keys are what the CLI, scanner and dashboard accept.
STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    MomentumStrategy.name: MomentumStrategy,
    MeanReversionStrategy.name: MeanReversionStrategy,
    BreakoutStrategy.name: BreakoutStrategy,
}

#: Name to the matching configuration class, for building typed overrides.
CONFIG_REGISTRY: dict[str, type[StrategyConfig]] = {
    MomentumStrategy.name: MomentumConfig,
    MeanReversionStrategy.name: MeanReversionConfig,
    BreakoutStrategy.name: BreakoutConfig,
}


def available_strategies() -> list[str]:
    """Registered strategy names, sorted.

    Example
    -------
    >>> available_strategies()
    ['breakout', 'mean_reversion', 'momentum']
    """
    return sorted(STRATEGY_REGISTRY)


def register_strategy(
    strategy: type[BaseStrategy], config: type[StrategyConfig] | None = None
) -> None:
    """Add a strategy to the registry so it can be selected by name.

    Raises
    ------
    StrategyError
        The name is already taken, or the class does not define one.
    """
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
    """Construct a strategy by name.

    Parameters
    ----------
    name:
        A registered strategy name, case-insensitive.
    config:
        A ready-made configuration. Mutually exclusive with ``overrides``.
    indicators:
        Indicator configuration to share with the strategy.
    **overrides:
        Field overrides applied to the strategy's default configuration, so the
        CLI and dashboard can tune parameters without importing config classes.

    Raises
    ------
    StrategyError
        The name is not registered, or an override is not a valid field.

    Example
    -------
    >>> build_strategy("momentum", min_confidence=70, rsi_entry_ceiling=60)
    MomentumStrategy(name='momentum')
    """
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
    # Contract
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
    # Strategies
    "MomentumStrategy",
    "MomentumConfig",
    "MeanReversionStrategy",
    "MeanReversionConfig",
    "BreakoutStrategy",
    "BreakoutConfig",
    # Registry
    "STRATEGY_REGISTRY",
    "CONFIG_REGISTRY",
    "available_strategies",
    "build_strategy",
    "register_strategy",
]
