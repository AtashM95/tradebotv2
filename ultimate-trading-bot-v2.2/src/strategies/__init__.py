"""
Strategies Package for Ultimate Trading Bot v2.2.

This package provides comprehensive trading strategy implementations
including technical analysis, quantitative, and ML-based strategies.
"""

from src.strategies.base_strategy import (
    BaseStrategy,
    StrategyConfig,
    StrategySignal,
    SignalAction,
    SignalSide,
    MarketData,
    StrategyContext,
)
from src.strategies.strategy_manager import (
    StrategyManager,
    StrategyManagerConfig,
    StrategyInfo,
    AggregatedSignal,
)
from src.strategies.momentum_strategy import (
    MomentumStrategy,
    MomentumConfig,
)
from src.strategies.mean_reversion_strategy import (
    MeanReversionStrategy,
    MeanReversionConfig,
)
from src.strategies.trend_following_strategy import (
    TrendFollowingStrategy,
    TrendFollowingConfig,
)
from src.strategies.breakout_strategy import (
    BreakoutStrategy,
    BreakoutConfig,
)
from src.strategies.scalping_strategy import (
    ScalpingStrategy,
    ScalpingConfig,
)
from src.strategies.swing_trading_strategy import (
    SwingTradingStrategy,
    SwingTradingConfig,
    SwingTrade,
)
from src.strategies.pairs_trading_strategy import (
    PairsTradingStrategy,
    PairsTradingConfig,
    TradingPair,
    PairPosition,
)
from src.strategies.grid_strategy import (
    GridTradingStrategy,
    GridStrategyConfig,
    GridLevel,
    GridInstance,
)
from src.strategies.dca_strategy import (
    DCAStrategy,
    DCAStrategyConfig,
    DCASchedule,
    DCAExecution,
)
from src.strategies.vwap_strategy import (
    VWAPStrategy,
    VWAPStrategyConfig,
    VWAPLevel,
    VWAPSession,
)
from src.strategies.market_making_strategy import (
    MarketMakingStrategy,
    MarketMakingConfig,
    Quote,
    InventoryPosition,
)
from src.strategies.arbitrage_strategy import (
    ArbitrageStrategy,
    ArbitrageStrategyConfig,
    ArbitrageOpportunity,
    ArbitragePosition,
)
from src.strategies.sector_rotation_strategy import (
    SectorRotationStrategy,
    SectorRotationConfig,
    SectorData,
    SectorAllocation,
    EconomicCyclePhase,
)
from src.strategies.sentiment_strategy import (
    SentimentStrategy,
    SentimentStrategyConfig,
    SentimentScore,
    SentimentAlert,
)
from src.strategies.ml_strategy import (
    MLStrategy,
    MLStrategyConfig,
    ModelPrediction,
    FeatureSet,
)
from src.strategies.options_strategy import (
    OptionsStrategy,
    OptionsStrategyConfig,
    OptionType,
    OptionPosition,
    OptionGreeks,
    OptionOpportunity,
)


__all__ = [
    # Base
    "BaseStrategy",
    "StrategyConfig",
    "StrategySignal",
    "SignalAction",
    "SignalSide",
    "MarketData",
    "StrategyContext",
    # Manager
    "StrategyManager",
    "StrategyManagerConfig",
    "StrategyInfo",
    "AggregatedSignal",
    # Momentum
    "MomentumStrategy",
    "MomentumConfig",
    # Mean Reversion
    "MeanReversionStrategy",
    "MeanReversionConfig",
    # Trend Following
    "TrendFollowingStrategy",
    "TrendFollowingConfig",
    # Breakout
    "BreakoutStrategy",
    "BreakoutConfig",
    # Scalping
    "ScalpingStrategy",
    "ScalpingConfig",
    # Swing Trading
    "SwingTradingStrategy",
    "SwingTradingConfig",
    "SwingTrade",
    # Pairs Trading
    "PairsTradingStrategy",
    "PairsTradingConfig",
    "TradingPair",
    "PairPosition",
    # Grid Trading
    "GridTradingStrategy",
    "GridStrategyConfig",
    "GridLevel",
    "GridInstance",
    # DCA
    "DCAStrategy",
    "DCAStrategyConfig",
    "DCASchedule",
    "DCAExecution",
    # VWAP
    "VWAPStrategy",
    "VWAPStrategyConfig",
    "VWAPLevel",
    "VWAPSession",
    # Market Making
    "MarketMakingStrategy",
    "MarketMakingConfig",
    "Quote",
    "InventoryPosition",
    # Arbitrage
    "ArbitrageStrategy",
    "ArbitrageStrategyConfig",
    "ArbitrageOpportunity",
    "ArbitragePosition",
    # Sector Rotation
    "SectorRotationStrategy",
    "SectorRotationConfig",
    "SectorData",
    "SectorAllocation",
    "EconomicCyclePhase",
    # Sentiment
    "SentimentStrategy",
    "SentimentStrategyConfig",
    "SentimentScore",
    "SentimentAlert",
    # ML
    "MLStrategy",
    "MLStrategyConfig",
    "ModelPrediction",
    "FeatureSet",
    # Options
    "OptionsStrategy",
    "OptionsStrategyConfig",
    "OptionType",
    "OptionPosition",
    "OptionGreeks",
    "OptionOpportunity",
]


def get_all_strategies() -> dict[str, type[BaseStrategy]]:
    """
    Get all available strategy classes.

    Returns:
        Dictionary mapping strategy names to classes
    """
    return {
        "momentum": MomentumStrategy,
        "mean_reversion": MeanReversionStrategy,
        "trend_following": TrendFollowingStrategy,
        "breakout": BreakoutStrategy,
        "scalping": ScalpingStrategy,
        "swing_trading": SwingTradingStrategy,
        "pairs_trading": PairsTradingStrategy,
        "grid": GridTradingStrategy,
        "dca": DCAStrategy,
        "vwap": VWAPStrategy,
        "market_making": MarketMakingStrategy,
        "arbitrage": ArbitrageStrategy,
        "sector_rotation": SectorRotationStrategy,
        "sentiment": SentimentStrategy,
        "ml": MLStrategy,
        "options": OptionsStrategy,
    }


def create_strategy(
    strategy_type: str,
    config: dict | None = None,
) -> BaseStrategy:
    """
    Create a strategy instance by type name.

    Args:
        strategy_type: Type of strategy to create
        config: Optional configuration dictionary

    Returns:
        Strategy instance

    Raises:
        ValueError: If strategy type is unknown
    """
    strategies = get_all_strategies()

    if strategy_type not in strategies:
        available = ", ".join(strategies.keys())
        raise ValueError(
            f"Unknown strategy type: {strategy_type}. "
            f"Available: {available}"
        )

    strategy_class = strategies[strategy_type]

    config_classes = {
        "momentum": MomentumConfig,
        "mean_reversion": MeanReversionConfig,
        "trend_following": TrendFollowingConfig,
        "breakout": BreakoutConfig,
        "scalping": ScalpingConfig,
        "swing_trading": SwingTradingConfig,
        "pairs_trading": PairsTradingConfig,
        "grid": GridStrategyConfig,
        "dca": DCAStrategyConfig,
        "vwap": VWAPStrategyConfig,
        "market_making": MarketMakingConfig,
        "arbitrage": ArbitrageStrategyConfig,
        "sector_rotation": SectorRotationConfig,
        "sentiment": SentimentStrategyConfig,
        "ml": MLStrategyConfig,
        "options": OptionsStrategyConfig,
    }

    if config:
        config_class = config_classes.get(strategy_type, StrategyConfig)
        strategy_config = config_class(**config)
        return strategy_class(config=strategy_config)
    else:
        return strategy_class()


def get_strategy_descriptions() -> dict[str, str]:
    """
    Get descriptions for all available strategies.

    Returns:
        Dictionary mapping strategy names to descriptions
    """
    return {
        "momentum": "RSI, MACD, ROC-based momentum trading",
        "mean_reversion": "Bollinger Bands, Z-score mean reversion",
        "trend_following": "MA crossovers with ADX confirmation",
        "breakout": "Support/resistance and Donchian channel breaks",
        "scalping": "High-frequency small profit scalping",
        "swing_trading": "Multi-day pattern-based trading",
        "pairs_trading": "Statistical arbitrage on correlated pairs",
        "grid": "Automated grid trading at price levels",
        "dca": "Dollar cost averaging with smart timing",
        "vwap": "VWAP-based mean reversion and execution",
        "market_making": "Automated bid-ask market making",
        "arbitrage": "Statistical and structural arbitrage",
        "sector_rotation": "Economic cycle-based sector rotation",
        "sentiment": "News and social sentiment trading",
        "ml": "Machine learning signal generation",
        "options": "Options strategies (calls, puts, spreads)",
    }
