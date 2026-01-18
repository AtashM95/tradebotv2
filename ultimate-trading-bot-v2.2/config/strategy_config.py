"""
Strategy Configuration Module for Ultimate Trading Bot v2.2.

This module provides configuration for all trading strategies including
technical analysis, AI-based, and machine learning strategies.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
from dataclasses import dataclass
from functools import lru_cache
import logging

from pydantic import BaseModel, Field, field_validator, model_validator


logger = logging.getLogger(__name__)


class StrategyType(str, Enum):
    """Strategy type enumeration."""
    TECHNICAL = "technical"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    BREAKOUT = "breakout"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    MACHINE_LEARNING = "machine_learning"
    AI_SENTIMENT = "ai_sentiment"
    AI_CONSENSUS = "ai_consensus"
    COMPOSITE = "composite"


class SignalStrength(str, Enum):
    """Signal strength enumeration."""
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class MarketRegime(str, Enum):
    """Market regime enumeration."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class BaseStrategyConfig(BaseModel):
    """Base configuration for all strategies."""

    name: str = Field(description="Strategy name")
    enabled: bool = Field(default=True, description="Enable strategy")
    weight: float = Field(default=1.0, ge=0.0, le=10.0, description="Strategy weight")
    min_signal_strength: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum signal strength")
    max_positions: int = Field(default=5, ge=1, le=50, description="Maximum positions")
    position_size_percent: float = Field(default=5.0, ge=0.1, le=100.0, description="Position size %")
    allowed_sides: List[str] = Field(default=["long"], description="Allowed position sides")
    symbols: List[str] = Field(default_factory=list, description="Symbols to trade")
    excluded_symbols: Set[str] = Field(default_factory=set, description="Excluded symbols")
    timeframes: List[str] = Field(default=["1d"], description="Timeframes to analyze")
    cooldown_minutes: int = Field(default=60, ge=0, description="Cooldown between trades")
    min_volume: int = Field(default=100000, ge=0, description="Minimum volume filter")
    min_price: float = Field(default=1.0, ge=0.0, description="Minimum price filter")
    max_spread_percent: float = Field(default=0.5, ge=0.0, le=10.0, description="Maximum spread %")

    @field_validator('allowed_sides', mode='before')
    @classmethod
    def validate_sides(cls, v: Union[str, List[str]]) -> List[str]:
        """Validate allowed sides."""
        if isinstance(v, str):
            return [v]
        valid_sides = {"long", "short"}
        return [s.lower() for s in v if s.lower() in valid_sides]


class RSIStrategyConfig(BaseStrategyConfig):
    """RSI strategy configuration."""

    name: str = Field(default="RSI Strategy")
    period: int = Field(default=14, ge=2, le=100, description="RSI period")
    overbought: float = Field(default=70.0, ge=50.0, le=100.0, description="Overbought level")
    oversold: float = Field(default=30.0, ge=0.0, le=50.0, description="Oversold level")
    exit_overbought: float = Field(default=60.0, ge=50.0, le=100.0, description="Exit overbought")
    exit_oversold: float = Field(default=40.0, ge=0.0, le=50.0, description="Exit oversold")
    use_divergence: bool = Field(default=True, description="Use RSI divergence")
    divergence_lookback: int = Field(default=14, ge=5, le=100, description="Divergence lookback")
    confirmation_candles: int = Field(default=1, ge=1, le=5, description="Confirmation candles")
    use_smoothed_rsi: bool = Field(default=False, description="Use smoothed RSI")


class MACDStrategyConfig(BaseStrategyConfig):
    """MACD strategy configuration."""

    name: str = Field(default="MACD Strategy")
    fast_period: int = Field(default=12, ge=2, le=50, description="Fast EMA period")
    slow_period: int = Field(default=26, ge=5, le=100, description="Slow EMA period")
    signal_period: int = Field(default=9, ge=2, le=50, description="Signal period")
    use_histogram: bool = Field(default=True, description="Use histogram for signals")
    histogram_threshold: float = Field(default=0.0, description="Histogram threshold")
    require_zero_cross: bool = Field(default=False, description="Require zero line cross")
    divergence_enabled: bool = Field(default=True, description="Enable divergence detection")
    confirmation_bars: int = Field(default=2, ge=1, le=10, description="Confirmation bars")

    @model_validator(mode='after')
    def validate_periods(self) -> 'MACDStrategyConfig':
        """Ensure fast period is less than slow period."""
        if self.fast_period >= self.slow_period:
            self.fast_period = self.slow_period - 1
        return self


class BollingerBandsStrategyConfig(BaseStrategyConfig):
    """Bollinger Bands strategy configuration."""

    name: str = Field(default="Bollinger Bands Strategy")
    period: int = Field(default=20, ge=5, le=100, description="BB period")
    std_dev: float = Field(default=2.0, ge=0.5, le=5.0, description="Standard deviations")
    use_percent_b: bool = Field(default=True, description="Use %B indicator")
    percent_b_lower: float = Field(default=0.0, ge=-0.5, le=0.5, description="%B lower threshold")
    percent_b_upper: float = Field(default=1.0, ge=0.5, le=1.5, description="%B upper threshold")
    use_bandwidth: bool = Field(default=True, description="Use bandwidth filter")
    bandwidth_percentile: float = Field(default=20.0, ge=0.0, le=100.0, description="Bandwidth percentile")
    squeeze_threshold: float = Field(default=0.05, ge=0.01, le=0.2, description="Squeeze threshold")
    mean_reversion_mode: bool = Field(default=True, description="Mean reversion mode")


class MomentumStrategyConfig(BaseStrategyConfig):
    """Momentum strategy configuration."""

    name: str = Field(default="Momentum Strategy")
    lookback_period: int = Field(default=20, ge=5, le=252, description="Momentum lookback")
    roc_period: int = Field(default=10, ge=1, le=100, description="Rate of change period")
    momentum_threshold: float = Field(default=0.02, ge=0.0, le=0.5, description="Momentum threshold")
    use_relative_momentum: bool = Field(default=True, description="Use relative momentum")
    benchmark_symbol: str = Field(default="SPY", description="Benchmark for relative momentum")
    ranking_period: int = Field(default=20, ge=5, le=100, description="Ranking period")
    top_n_percent: float = Field(default=20.0, ge=1.0, le=50.0, description="Top N percent")
    exclude_recent_ipo: bool = Field(default=True, description="Exclude recent IPOs")
    min_history_days: int = Field(default=252, ge=20, le=1000, description="Minimum history days")


class MeanReversionStrategyConfig(BaseStrategyConfig):
    """Mean reversion strategy configuration."""

    name: str = Field(default="Mean Reversion Strategy")
    lookback_period: int = Field(default=20, ge=5, le=252, description="Lookback period")
    z_score_threshold: float = Field(default=2.0, ge=0.5, le=5.0, description="Z-score threshold")
    exit_z_score: float = Field(default=0.5, ge=0.0, le=2.0, description="Exit z-score")
    ma_type: str = Field(default="sma", description="Moving average type (sma, ema)")
    use_bollinger: bool = Field(default=True, description="Use Bollinger Bands")
    bb_std_dev: float = Field(default=2.0, ge=0.5, le=5.0, description="BB standard deviations")
    min_mean_reversion_rate: float = Field(default=0.5, ge=0.0, le=1.0, description="Min reversion rate")
    max_holding_period: int = Field(default=10, ge=1, le=100, description="Max holding period days")


class TrendFollowingStrategyConfig(BaseStrategyConfig):
    """Trend following strategy configuration."""

    name: str = Field(default="Trend Following Strategy")
    fast_ma_period: int = Field(default=20, ge=5, le=100, description="Fast MA period")
    slow_ma_period: int = Field(default=50, ge=10, le=200, description="Slow MA period")
    trend_ma_period: int = Field(default=200, ge=50, le=500, description="Trend MA period")
    ma_type: str = Field(default="ema", description="Moving average type")
    use_adx: bool = Field(default=True, description="Use ADX filter")
    adx_period: int = Field(default=14, ge=5, le=50, description="ADX period")
    adx_threshold: float = Field(default=25.0, ge=10.0, le=50.0, description="ADX threshold")
    use_supertrend: bool = Field(default=False, description="Use Supertrend indicator")
    supertrend_period: int = Field(default=10, ge=5, le=50, description="Supertrend period")
    supertrend_multiplier: float = Field(default=3.0, ge=1.0, le=5.0, description="Supertrend multiplier")
    require_all_aligned: bool = Field(default=True, description="Require all MAs aligned")


class BreakoutStrategyConfig(BaseStrategyConfig):
    """Breakout strategy configuration."""

    name: str = Field(default="Breakout Strategy")
    lookback_period: int = Field(default=20, ge=5, le=100, description="Lookback period")
    breakout_threshold: float = Field(default=0.02, ge=0.0, le=0.2, description="Breakout threshold %")
    volume_multiplier: float = Field(default=1.5, ge=1.0, le=5.0, description="Volume multiplier")
    consolidation_period: int = Field(default=10, ge=3, le=50, description="Consolidation period")
    consolidation_threshold: float = Field(default=0.05, ge=0.01, le=0.2, description="Consolidation range %")
    use_atr_filter: bool = Field(default=True, description="Use ATR filter")
    atr_period: int = Field(default=14, ge=5, le=50, description="ATR period")
    atr_multiplier: float = Field(default=1.5, ge=0.5, le=5.0, description="ATR multiplier")
    false_breakout_filter: bool = Field(default=True, description="Filter false breakouts")
    confirmation_close: bool = Field(default=True, description="Require close above breakout")


class VWAPStrategyConfig(BaseStrategyConfig):
    """VWAP strategy configuration."""

    name: str = Field(default="VWAP Strategy")
    use_anchored_vwap: bool = Field(default=False, description="Use anchored VWAP")
    vwap_bands_std: float = Field(default=2.0, ge=0.5, le=5.0, description="VWAP bands std dev")
    mean_reversion_mode: bool = Field(default=True, description="Mean reversion mode")
    deviation_threshold: float = Field(default=0.01, ge=0.001, le=0.1, description="Deviation threshold")
    use_volume_profile: bool = Field(default=True, description="Use volume profile")
    poc_threshold: float = Field(default=0.005, ge=0.001, le=0.05, description="POC threshold")


class IchimokuStrategyConfig(BaseStrategyConfig):
    """Ichimoku strategy configuration."""

    name: str = Field(default="Ichimoku Strategy")
    tenkan_period: int = Field(default=9, ge=5, le=30, description="Tenkan-sen period")
    kijun_period: int = Field(default=26, ge=10, le=60, description="Kijun-sen period")
    senkou_b_period: int = Field(default=52, ge=20, le=120, description="Senkou Span B period")
    displacement: int = Field(default=26, ge=10, le=60, description="Cloud displacement")
    use_chikou_confirmation: bool = Field(default=True, description="Use Chikou confirmation")
    require_price_above_cloud: bool = Field(default=True, description="Require price above cloud")
    require_tk_cross: bool = Field(default=True, description="Require TK cross")
    cloud_thickness_filter: bool = Field(default=True, description="Filter by cloud thickness")
    min_cloud_thickness: float = Field(default=0.01, ge=0.001, le=0.1, description="Min cloud thickness %")


class MLStrategyConfig(BaseStrategyConfig):
    """Machine learning strategy configuration."""

    name: str = Field(default="ML Strategy")
    model_type: str = Field(default="ensemble", description="Model type")
    prediction_horizon: int = Field(default=5, ge=1, le=60, description="Prediction horizon days")
    confidence_threshold: float = Field(default=0.6, ge=0.5, le=0.99, description="Confidence threshold")
    retrain_frequency: int = Field(default=30, ge=1, le=365, description="Retrain frequency days")
    feature_lookback: int = Field(default=60, ge=10, le=252, description="Feature lookback period")
    use_technical_features: bool = Field(default=True, description="Use technical features")
    use_sentiment_features: bool = Field(default=True, description="Use sentiment features")
    use_fundamental_features: bool = Field(default=False, description="Use fundamental features")
    ensemble_method: str = Field(default="voting", description="Ensemble method")
    min_training_samples: int = Field(default=1000, ge=100, le=10000, description="Min training samples")
    validation_split: float = Field(default=0.2, ge=0.1, le=0.4, description="Validation split")


class AISentimentStrategyConfig(BaseStrategyConfig):
    """AI sentiment strategy configuration."""

    name: str = Field(default="AI Sentiment Strategy")
    use_openai: bool = Field(default=True, description="Use OpenAI for sentiment")
    sentiment_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Sentiment threshold")
    news_lookback_hours: int = Field(default=24, ge=1, le=168, description="News lookback hours")
    min_news_count: int = Field(default=3, ge=1, le=50, description="Minimum news count")
    weight_by_recency: bool = Field(default=True, description="Weight by recency")
    include_social_sentiment: bool = Field(default=False, description="Include social sentiment")
    contrarian_mode: bool = Field(default=False, description="Contrarian mode")
    max_openai_calls_per_day: int = Field(default=100, ge=1, le=1000, description="Max OpenAI calls/day")


class AIConsensusStrategyConfig(BaseStrategyConfig):
    """AI consensus strategy configuration."""

    name: str = Field(default="AI Consensus Strategy")
    use_openai_advisor: bool = Field(default=True, description="Use OpenAI advisor")
    combine_with_technical: bool = Field(default=True, description="Combine with technical analysis")
    technical_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Technical analysis weight")
    ai_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="AI analysis weight")
    require_consensus: bool = Field(default=True, description="Require consensus")
    consensus_threshold: float = Field(default=0.7, ge=0.5, le=1.0, description="Consensus threshold")
    include_risk_assessment: bool = Field(default=True, description="Include risk assessment")
    include_market_context: bool = Field(default=True, description="Include market context")


class CompositeStrategyConfig(BaseStrategyConfig):
    """Composite strategy configuration."""

    name: str = Field(default="Composite Strategy")
    strategies: List[str] = Field(
        default_factory=lambda: ["rsi", "macd", "momentum"],
        description="Sub-strategies to combine"
    )
    combination_method: str = Field(default="weighted", description="Combination method")
    weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Strategy weights"
    )
    require_majority: bool = Field(default=True, description="Require majority agreement")
    majority_threshold: float = Field(default=0.6, ge=0.5, le=1.0, description="Majority threshold")
    veto_enabled: bool = Field(default=True, description="Enable veto on strong disagreement")
    veto_threshold: float = Field(default=0.8, ge=0.5, le=1.0, description="Veto threshold")


class StrategyConfig(BaseModel):
    """
    Master strategy configuration.

    Aggregates all strategy configurations for the trading bot.
    """

    # Active strategies
    active_strategies: List[str] = Field(
        default_factory=lambda: ["rsi", "macd", "momentum"],
        description="List of active strategy names"
    )

    # Default strategy settings
    default_position_size: float = Field(default=5.0, ge=0.1, le=100.0, description="Default position size %")
    default_max_positions: int = Field(default=10, ge=1, le=100, description="Default max positions")
    use_signal_aggregation: bool = Field(default=True, description="Aggregate signals from multiple strategies")
    aggregation_method: str = Field(default="weighted", description="Signal aggregation method")
    min_aggregated_signal: float = Field(default=0.5, ge=0.0, le=1.0, description="Min aggregated signal")

    # Individual strategy configurations
    rsi: RSIStrategyConfig = Field(default_factory=RSIStrategyConfig, description="RSI strategy")
    macd: MACDStrategyConfig = Field(default_factory=MACDStrategyConfig, description="MACD strategy")
    bollinger: BollingerBandsStrategyConfig = Field(default_factory=BollingerBandsStrategyConfig, description="Bollinger Bands")
    momentum: MomentumStrategyConfig = Field(default_factory=MomentumStrategyConfig, description="Momentum strategy")
    mean_reversion: MeanReversionStrategyConfig = Field(default_factory=MeanReversionStrategyConfig, description="Mean reversion")
    trend_following: TrendFollowingStrategyConfig = Field(default_factory=TrendFollowingStrategyConfig, description="Trend following")
    breakout: BreakoutStrategyConfig = Field(default_factory=BreakoutStrategyConfig, description="Breakout strategy")
    vwap: VWAPStrategyConfig = Field(default_factory=VWAPStrategyConfig, description="VWAP strategy")
    ichimoku: IchimokuStrategyConfig = Field(default_factory=IchimokuStrategyConfig, description="Ichimoku strategy")
    ml: MLStrategyConfig = Field(default_factory=MLStrategyConfig, description="ML strategy")
    ai_sentiment: AISentimentStrategyConfig = Field(default_factory=AISentimentStrategyConfig, description="AI sentiment")
    ai_consensus: AIConsensusStrategyConfig = Field(default_factory=AIConsensusStrategyConfig, description="AI consensus")
    composite: CompositeStrategyConfig = Field(default_factory=CompositeStrategyConfig, description="Composite strategy")

    # Market regime settings
    adapt_to_regime: bool = Field(default=True, description="Adapt strategies to market regime")
    regime_detection_method: str = Field(default="volatility", description="Regime detection method")
    regime_lookback: int = Field(default=60, ge=10, le=252, description="Regime lookback period")

    def get_strategy_config(self, name: str) -> Optional[BaseStrategyConfig]:
        """
        Get configuration for a specific strategy.

        Args:
            name: Strategy name

        Returns:
            Strategy configuration or None
        """
        strategy_map = {
            "rsi": self.rsi,
            "macd": self.macd,
            "bollinger": self.bollinger,
            "momentum": self.momentum,
            "mean_reversion": self.mean_reversion,
            "trend_following": self.trend_following,
            "breakout": self.breakout,
            "vwap": self.vwap,
            "ichimoku": self.ichimoku,
            "ml": self.ml,
            "ai_sentiment": self.ai_sentiment,
            "ai_consensus": self.ai_consensus,
            "composite": self.composite,
        }
        return strategy_map.get(name.lower())

    def get_active_configs(self) -> Dict[str, BaseStrategyConfig]:
        """
        Get configurations for all active strategies.

        Returns:
            Dictionary of active strategy configurations
        """
        return {
            name: config
            for name in self.active_strategies
            if (config := self.get_strategy_config(name)) is not None
        }

    def get_enabled_strategies(self) -> List[str]:
        """
        Get list of enabled strategy names.

        Returns:
            List of enabled strategy names
        """
        enabled = []
        for name in self.active_strategies:
            config = self.get_strategy_config(name)
            if config and config.enabled:
                enabled.append(name)
        return enabled


@lru_cache()
def get_strategy_config() -> StrategyConfig:
    """
    Get cached strategy configuration.

    Returns:
        Singleton StrategyConfig instance
    """
    return StrategyConfig()


def reload_strategy_config() -> StrategyConfig:
    """
    Reload strategy configuration.

    Returns:
        New StrategyConfig instance
    """
    get_strategy_config.cache_clear()
    return get_strategy_config()


# Module-level strategy config instance
strategy_config = get_strategy_config()
