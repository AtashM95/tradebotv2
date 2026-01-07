"""
Risk Configuration Module for Ultimate Trading Bot v2.2.

This module provides comprehensive risk management configuration including
position sizing, stop losses, drawdown limits, and portfolio risk controls.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
from dataclasses import dataclass
from functools import lru_cache
import logging

from pydantic import BaseModel, Field, field_validator, model_validator


logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Risk level enumeration."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class PositionSizingMethod(str, Enum):
    """Position sizing method enumeration."""
    FIXED_DOLLAR = "fixed_dollar"
    FIXED_PERCENT = "fixed_percent"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    KELLY_CRITERION = "kelly_criterion"
    OPTIMAL_F = "optimal_f"
    RISK_PARITY = "risk_parity"
    ATR_BASED = "atr_based"


class StopLossType(str, Enum):
    """Stop loss type enumeration."""
    FIXED_PERCENT = "fixed_percent"
    ATR_BASED = "atr_based"
    SUPPORT_BASED = "support_based"
    VOLATILITY_BASED = "volatility_based"
    TIME_BASED = "time_based"
    TRAILING = "trailing"
    CHANDELIER = "chandelier"


class TakeProfitType(str, Enum):
    """Take profit type enumeration."""
    FIXED_PERCENT = "fixed_percent"
    ATR_BASED = "atr_based"
    RESISTANCE_BASED = "resistance_based"
    RISK_REWARD_RATIO = "risk_reward_ratio"
    TRAILING = "trailing"
    SCALED = "scaled"


class PositionSizingConfig(BaseModel):
    """Position sizing configuration."""

    method: PositionSizingMethod = Field(
        default=PositionSizingMethod.FIXED_PERCENT,
        description="Position sizing method"
    )

    # Fixed sizing
    fixed_dollar_amount: float = Field(default=10000.0, ge=100.0, description="Fixed dollar amount")
    fixed_percent: float = Field(default=5.0, ge=0.1, le=100.0, description="Fixed percent of portfolio")

    # Risk-based sizing
    risk_per_trade_percent: float = Field(default=1.0, ge=0.1, le=10.0, description="Risk per trade %")
    max_position_percent: float = Field(default=10.0, ge=1.0, le=100.0, description="Max position size %")
    min_position_percent: float = Field(default=1.0, ge=0.1, le=10.0, description="Min position size %")

    # Volatility adjustment
    volatility_lookback: int = Field(default=20, ge=5, le=100, description="Volatility lookback period")
    volatility_target: float = Field(default=0.15, ge=0.01, le=0.5, description="Target volatility")
    volatility_scalar: float = Field(default=1.0, ge=0.1, le=5.0, description="Volatility scalar")

    # ATR-based sizing
    atr_period: int = Field(default=14, ge=5, le=50, description="ATR period")
    atr_multiplier: float = Field(default=2.0, ge=0.5, le=5.0, description="ATR multiplier for sizing")

    # Kelly criterion
    kelly_fraction: float = Field(default=0.25, ge=0.1, le=1.0, description="Kelly fraction to use")
    min_trades_for_kelly: int = Field(default=30, ge=10, le=100, description="Min trades for Kelly calc")

    # Scale factors
    scale_by_conviction: bool = Field(default=True, description="Scale by signal conviction")
    conviction_scale_min: float = Field(default=0.5, ge=0.1, le=1.0, description="Min conviction scale")
    conviction_scale_max: float = Field(default=1.5, ge=1.0, le=3.0, description="Max conviction scale")

    # Correlation adjustment
    adjust_for_correlation: bool = Field(default=True, description="Adjust for correlated positions")
    correlation_threshold: float = Field(default=0.7, ge=0.3, le=1.0, description="Correlation threshold")
    correlation_reduction: float = Field(default=0.5, ge=0.1, le=0.9, description="Size reduction for correlated")


class StopLossConfig(BaseModel):
    """Stop loss configuration."""

    enabled: bool = Field(default=True, description="Enable stop losses")
    type: StopLossType = Field(default=StopLossType.ATR_BASED, description="Stop loss type")

    # Fixed percent stop
    fixed_percent: float = Field(default=2.0, ge=0.5, le=20.0, description="Fixed stop loss %")

    # ATR-based stop
    atr_period: int = Field(default=14, ge=5, le=50, description="ATR period")
    atr_multiplier: float = Field(default=2.0, ge=0.5, le=5.0, description="ATR multiplier")

    # Volatility-based stop
    volatility_lookback: int = Field(default=20, ge=5, le=100, description="Volatility lookback")
    volatility_multiplier: float = Field(default=2.0, ge=0.5, le=5.0, description="Volatility multiplier")

    # Time-based stop
    max_holding_days: int = Field(default=10, ge=1, le=365, description="Max holding days")
    time_decay_start_day: int = Field(default=5, ge=1, le=100, description="Day to start time decay")

    # Trailing stop
    trailing_enabled: bool = Field(default=True, description="Enable trailing stop")
    trailing_activation_percent: float = Field(default=1.0, ge=0.1, le=10.0, description="Trailing activation %")
    trailing_distance_percent: float = Field(default=1.5, ge=0.5, le=10.0, description="Trailing distance %")
    trailing_step_percent: float = Field(default=0.1, ge=0.01, le=1.0, description="Trailing step size %")

    # Chandelier stop
    chandelier_period: int = Field(default=22, ge=5, le=50, description="Chandelier period")
    chandelier_multiplier: float = Field(default=3.0, ge=1.0, le=5.0, description="Chandelier multiplier")

    # Break-even stop
    move_to_breakeven: bool = Field(default=True, description="Move stop to breakeven")
    breakeven_trigger_percent: float = Field(default=1.5, ge=0.5, le=10.0, description="Breakeven trigger %")
    breakeven_offset_percent: float = Field(default=0.1, ge=0.0, le=1.0, description="Breakeven offset %")


class TakeProfitConfig(BaseModel):
    """Take profit configuration."""

    enabled: bool = Field(default=True, description="Enable take profit")
    type: TakeProfitType = Field(default=TakeProfitType.RISK_REWARD_RATIO, description="Take profit type")

    # Fixed percent
    fixed_percent: float = Field(default=4.0, ge=1.0, le=50.0, description="Fixed take profit %")

    # Risk-reward ratio
    risk_reward_ratio: float = Field(default=2.0, ge=1.0, le=10.0, description="Risk/reward ratio")

    # ATR-based
    atr_period: int = Field(default=14, ge=5, le=50, description="ATR period")
    atr_multiplier: float = Field(default=3.0, ge=1.0, le=10.0, description="ATR multiplier")

    # Scaled take profit
    use_scaled_exit: bool = Field(default=True, description="Use scaled exit")
    scale_levels: List[Tuple[float, float]] = Field(
        default=[(0.5, 0.5), (1.0, 0.3), (1.5, 0.2)],
        description="Scale levels (profit %, exit %)"
    )

    # Trailing take profit
    trailing_enabled: bool = Field(default=False, description="Enable trailing take profit")
    trailing_activation_percent: float = Field(default=3.0, ge=1.0, le=20.0, description="Trailing activation %")
    trailing_distance_percent: float = Field(default=1.0, ge=0.5, le=10.0, description="Trailing distance %")


class DrawdownConfig(BaseModel):
    """Drawdown management configuration."""

    # Daily limits
    max_daily_loss_percent: float = Field(default=3.0, ge=0.5, le=10.0, description="Max daily loss %")
    daily_loss_action: str = Field(default="reduce_size", description="Action on daily loss limit")

    # Weekly limits
    max_weekly_loss_percent: float = Field(default=7.0, ge=1.0, le=20.0, description="Max weekly loss %")
    weekly_loss_action: str = Field(default="stop_trading", description="Action on weekly loss limit")

    # Monthly limits
    max_monthly_loss_percent: float = Field(default=12.0, ge=2.0, le=30.0, description="Max monthly loss %")
    monthly_loss_action: str = Field(default="stop_trading", description="Action on monthly loss limit")

    # Overall drawdown
    max_drawdown_percent: float = Field(default=20.0, ge=5.0, le=50.0, description="Max overall drawdown %")
    drawdown_action: str = Field(default="liquidate", description="Action on max drawdown")

    # Recovery settings
    recovery_mode_enabled: bool = Field(default=True, description="Enable recovery mode")
    recovery_reduction_percent: float = Field(default=50.0, ge=10.0, le=90.0, description="Size reduction in recovery")
    recovery_threshold_percent: float = Field(default=50.0, ge=10.0, le=100.0, description="Recovery threshold")

    # Trailing drawdown
    use_trailing_drawdown: bool = Field(default=True, description="Use trailing drawdown limit")
    trailing_drawdown_percent: float = Field(default=10.0, ge=2.0, le=30.0, description="Trailing drawdown %")


class PortfolioRiskConfig(BaseModel):
    """Portfolio-level risk configuration."""

    # Position limits
    max_positions: int = Field(default=10, ge=1, le=100, description="Maximum open positions")
    max_positions_per_sector: int = Field(default=3, ge=1, le=20, description="Max positions per sector")
    max_correlated_positions: int = Field(default=3, ge=1, le=10, description="Max correlated positions")

    # Exposure limits
    max_long_exposure_percent: float = Field(default=100.0, ge=10.0, le=200.0, description="Max long exposure %")
    max_short_exposure_percent: float = Field(default=50.0, ge=0.0, le=200.0, description="Max short exposure %")
    max_gross_exposure_percent: float = Field(default=150.0, ge=50.0, le=400.0, description="Max gross exposure %")
    max_net_exposure_percent: float = Field(default=100.0, ge=10.0, le=200.0, description="Max net exposure %")

    # Sector limits
    max_sector_exposure_percent: float = Field(default=30.0, ge=10.0, le=100.0, description="Max sector exposure %")
    sector_exposure_limits: Dict[str, float] = Field(
        default_factory=dict,
        description="Sector-specific exposure limits"
    )

    # Concentration limits
    max_single_position_percent: float = Field(default=10.0, ge=1.0, le=50.0, description="Max single position %")
    max_top3_concentration_percent: float = Field(default=30.0, ge=10.0, le=100.0, description="Max top 3 concentration %")

    # Correlation limits
    correlation_lookback_days: int = Field(default=60, ge=20, le=252, description="Correlation lookback")
    max_portfolio_correlation: float = Field(default=0.7, ge=0.3, le=1.0, description="Max portfolio correlation")

    # Beta limits
    max_portfolio_beta: float = Field(default=1.5, ge=0.5, le=3.0, description="Max portfolio beta")
    target_portfolio_beta: float = Field(default=1.0, ge=0.0, le=2.0, description="Target portfolio beta")


class VaRConfig(BaseModel):
    """Value at Risk configuration."""

    enabled: bool = Field(default=True, description="Enable VaR calculation")
    method: str = Field(default="historical", description="VaR calculation method")
    confidence_level: float = Field(default=0.95, ge=0.9, le=0.99, description="VaR confidence level")
    lookback_days: int = Field(default=252, ge=60, le=1000, description="Historical lookback days")
    max_var_percent: float = Field(default=5.0, ge=1.0, le=20.0, description="Maximum VaR % limit")

    # Monte Carlo settings
    monte_carlo_simulations: int = Field(default=10000, ge=1000, le=100000, description="MC simulations")

    # Expected shortfall
    calculate_cvar: bool = Field(default=True, description="Calculate CVaR/Expected Shortfall")
    cvar_confidence: float = Field(default=0.95, ge=0.9, le=0.99, description="CVaR confidence")

    # Stress testing
    stress_test_enabled: bool = Field(default=True, description="Enable stress testing")
    stress_scenarios: List[str] = Field(
        default=["market_crash", "flash_crash", "high_volatility"],
        description="Stress test scenarios"
    )


class RiskLimitsConfig(BaseModel):
    """Trading risk limits configuration."""

    # Order limits
    max_order_value: float = Field(default=50000.0, ge=1000.0, description="Max single order value")
    max_daily_orders: int = Field(default=100, ge=1, le=1000, description="Max daily orders")
    max_orders_per_symbol: int = Field(default=10, ge=1, le=100, description="Max orders per symbol")

    # Trading frequency limits
    min_time_between_trades_seconds: int = Field(default=60, ge=0, le=3600, description="Min time between trades")
    max_trades_per_minute: int = Field(default=5, ge=1, le=60, description="Max trades per minute")
    max_trades_per_hour: int = Field(default=20, ge=1, le=200, description="Max trades per hour")
    max_trades_per_day: int = Field(default=50, ge=1, le=500, description="Max trades per day")

    # Loss limits
    max_loss_per_trade: float = Field(default=1000.0, ge=100.0, description="Max loss per trade")
    max_consecutive_losses: int = Field(default=5, ge=2, le=20, description="Max consecutive losses")
    action_on_consecutive_losses: str = Field(default="pause", description="Action on consecutive losses")
    pause_duration_minutes: int = Field(default=60, ge=5, le=1440, description="Pause duration after losses")

    # Win rate monitoring
    min_win_rate: float = Field(default=0.3, ge=0.1, le=0.9, description="Minimum win rate")
    win_rate_lookback_trades: int = Field(default=20, ge=10, le=100, description="Win rate lookback trades")
    action_on_low_win_rate: str = Field(default="reduce_size", description="Action on low win rate")


class RiskConfig(BaseModel):
    """
    Master risk configuration.

    Aggregates all risk management configurations for the trading bot.
    """

    # Risk profile
    risk_level: RiskLevel = Field(default=RiskLevel.MEDIUM, description="Overall risk level")
    risk_tolerance: float = Field(default=0.5, ge=0.0, le=1.0, description="Risk tolerance (0-1)")

    # Component configurations
    position_sizing: PositionSizingConfig = Field(
        default_factory=PositionSizingConfig,
        description="Position sizing configuration"
    )
    stop_loss: StopLossConfig = Field(
        default_factory=StopLossConfig,
        description="Stop loss configuration"
    )
    take_profit: TakeProfitConfig = Field(
        default_factory=TakeProfitConfig,
        description="Take profit configuration"
    )
    drawdown: DrawdownConfig = Field(
        default_factory=DrawdownConfig,
        description="Drawdown management configuration"
    )
    portfolio: PortfolioRiskConfig = Field(
        default_factory=PortfolioRiskConfig,
        description="Portfolio risk configuration"
    )
    var: VaRConfig = Field(
        default_factory=VaRConfig,
        description="Value at Risk configuration"
    )
    limits: RiskLimitsConfig = Field(
        default_factory=RiskLimitsConfig,
        description="Risk limits configuration"
    )

    # Global settings
    enforce_risk_limits: bool = Field(default=True, description="Enforce all risk limits")
    log_risk_events: bool = Field(default=True, description="Log risk events")
    alert_on_breach: bool = Field(default=True, description="Alert on limit breach")

    # Market condition adjustments
    adjust_for_volatility: bool = Field(default=True, description="Adjust for market volatility")
    volatility_adjustment_factor: float = Field(default=1.0, ge=0.1, le=3.0, description="Volatility adjustment factor")
    reduce_risk_in_high_volatility: bool = Field(default=True, description="Reduce risk in high volatility")
    high_volatility_reduction: float = Field(default=0.5, ge=0.1, le=0.9, description="High volatility size reduction")

    def get_risk_multiplier(self) -> float:
        """
        Get risk multiplier based on risk level.

        Returns:
            Risk multiplier value
        """
        multipliers = {
            RiskLevel.VERY_LOW: 0.25,
            RiskLevel.LOW: 0.5,
            RiskLevel.MEDIUM: 1.0,
            RiskLevel.HIGH: 1.5,
            RiskLevel.VERY_HIGH: 2.0,
        }
        return multipliers.get(self.risk_level, 1.0)

    def get_effective_position_size(self, base_size: float) -> float:
        """
        Calculate effective position size based on risk settings.

        Args:
            base_size: Base position size

        Returns:
            Adjusted position size
        """
        multiplier = self.get_risk_multiplier()
        return base_size * multiplier * self.risk_tolerance


@lru_cache()
def get_risk_config() -> RiskConfig:
    """
    Get cached risk configuration.

    Returns:
        Singleton RiskConfig instance
    """
    return RiskConfig()


def reload_risk_config() -> RiskConfig:
    """
    Reload risk configuration.

    Returns:
        New RiskConfig instance
    """
    get_risk_config.cache_clear()
    return get_risk_config()


# Module-level risk config instance
risk_config = get_risk_config()


# Preset risk profiles
RISK_PROFILES: Dict[str, Dict[str, Any]] = {
    "conservative": {
        "risk_level": RiskLevel.LOW,
        "risk_tolerance": 0.3,
        "position_sizing": {
            "fixed_percent": 3.0,
            "risk_per_trade_percent": 0.5,
            "max_position_percent": 5.0,
        },
        "stop_loss": {
            "fixed_percent": 1.5,
            "atr_multiplier": 1.5,
        },
        "drawdown": {
            "max_daily_loss_percent": 2.0,
            "max_drawdown_percent": 10.0,
        },
        "portfolio": {
            "max_positions": 5,
            "max_single_position_percent": 5.0,
        },
    },
    "moderate": {
        "risk_level": RiskLevel.MEDIUM,
        "risk_tolerance": 0.5,
        "position_sizing": {
            "fixed_percent": 5.0,
            "risk_per_trade_percent": 1.0,
            "max_position_percent": 10.0,
        },
        "stop_loss": {
            "fixed_percent": 2.0,
            "atr_multiplier": 2.0,
        },
        "drawdown": {
            "max_daily_loss_percent": 3.0,
            "max_drawdown_percent": 20.0,
        },
        "portfolio": {
            "max_positions": 10,
            "max_single_position_percent": 10.0,
        },
    },
    "aggressive": {
        "risk_level": RiskLevel.HIGH,
        "risk_tolerance": 0.7,
        "position_sizing": {
            "fixed_percent": 10.0,
            "risk_per_trade_percent": 2.0,
            "max_position_percent": 20.0,
        },
        "stop_loss": {
            "fixed_percent": 3.0,
            "atr_multiplier": 2.5,
        },
        "drawdown": {
            "max_daily_loss_percent": 5.0,
            "max_drawdown_percent": 30.0,
        },
        "portfolio": {
            "max_positions": 15,
            "max_single_position_percent": 15.0,
        },
    },
}


def create_risk_config_from_profile(profile_name: str) -> RiskConfig:
    """
    Create risk configuration from a preset profile.

    Args:
        profile_name: Name of the risk profile

    Returns:
        RiskConfig instance with profile settings
    """
    if profile_name not in RISK_PROFILES:
        raise ValueError(f"Unknown risk profile: {profile_name}")

    profile = RISK_PROFILES[profile_name]
    return RiskConfig(**profile)
