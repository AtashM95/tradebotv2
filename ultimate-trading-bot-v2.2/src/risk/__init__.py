"""
Risk Management Package for Ultimate Trading Bot v2.2.

This package provides comprehensive risk management functionality
including position sizing, drawdown control, exposure management,
and VaR calculations.
"""

from src.risk.base_risk import (
    RiskLevel,
    RiskType,
    RiskMetric,
    RiskAlert,
    RiskLimit,
    RiskAssessment,
    RiskConfig,
    RiskContext,
    BaseRiskManager,
    calculate_var,
    calculate_cvar,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)
from src.risk.position_sizer import (
    PositionSizer,
    PositionSizerConfig,
    PositionSizeResult,
    SizingMethod,
)
from src.risk.risk_manager import (
    RiskManager,
    RiskManagerConfig,
    TradingHalt,
)
from src.risk.drawdown_manager import (
    DrawdownManager,
    DrawdownManagerConfig,
    DrawdownPeriod,
    DrawdownStats,
)
from src.risk.exposure_manager import (
    ExposureManager,
    ExposureManagerConfig,
    ExposureBreakdown,
    ExposureSnapshot,
)
from src.risk.stop_loss_manager import (
    StopLossManager,
    StopLossConfig,
    StopLossOrder,
    StopType,
)
from src.risk.var_calculator import (
    VaRCalculator,
    VaRConfig,
    VaRResult,
    VaRMethod,
)
from src.risk.correlation_risk import (
    CorrelationRiskManager,
    CorrelationRiskConfig,
    CorrelationPair,
    CorrelationCluster,
    DiversificationMetrics,
)
from src.risk.liquidity_risk import (
    LiquidityRiskManager,
    LiquidityRiskConfig,
    LiquidityMetrics,
    LiquidityLevel,
    MarketImpactEstimate,
    ExecutionRiskAssessment,
    MarketImpactModel,
)
from src.risk.margin_manager import (
    MarginManager,
    MarginManagerConfig,
    MarginRequirement,
    AccountMarginStatus,
    MarginType,
    MarginCallType,
    LeverageLevel,
)
from src.risk.trade_validator import (
    TradeValidator,
    TradeValidatorConfig,
    OrderRequest,
    ValidationResponse,
    ValidationResult,
    RejectionReason,
    OrderType,
    OrderSide,
)
from src.risk.risk_metrics import (
    RiskMetricsCalculator,
    RiskMetricsConfig,
    PerformanceMetrics,
    MetricPeriod,
)
from src.risk.stress_tester import (
    StressTester,
    StressTesterConfig,
    StressScenario,
    StressTestResult,
    ScenarioType,
    HistoricalEvent,
    MonteCarloResult,
)
from src.risk.risk_limits import (
    RiskLimitsManager,
    RiskLimit as RiskLimitDef,
    LimitCheck,
    LimitType,
    LimitAction,
    LimitStatus,
)
from src.risk.risk_reporter import (
    RiskReporter,
    RiskReporterConfig,
    RiskReport,
    ReportSection,
    ReportType,
    ReportFormat,
    ReportFrequency,
)


__all__ = [
    # Base
    "RiskLevel",
    "RiskType",
    "RiskMetric",
    "RiskAlert",
    "RiskLimit",
    "RiskAssessment",
    "RiskConfig",
    "RiskContext",
    "BaseRiskManager",
    "calculate_var",
    "calculate_cvar",
    "calculate_max_drawdown",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    # Position Sizing
    "PositionSizer",
    "PositionSizerConfig",
    "PositionSizeResult",
    "SizingMethod",
    # Risk Manager
    "RiskManager",
    "RiskManagerConfig",
    "TradingHalt",
    # Drawdown
    "DrawdownManager",
    "DrawdownManagerConfig",
    "DrawdownPeriod",
    "DrawdownStats",
    # Exposure
    "ExposureManager",
    "ExposureManagerConfig",
    "ExposureBreakdown",
    "ExposureSnapshot",
    # Stop Loss
    "StopLossManager",
    "StopLossConfig",
    "StopLossOrder",
    "StopType",
    # VaR
    "VaRCalculator",
    "VaRConfig",
    "VaRResult",
    "VaRMethod",
    # Correlation
    "CorrelationRiskManager",
    "CorrelationRiskConfig",
    "CorrelationPair",
    "CorrelationCluster",
    "DiversificationMetrics",
    # Liquidity Risk
    "LiquidityRiskManager",
    "LiquidityRiskConfig",
    "LiquidityMetrics",
    "LiquidityLevel",
    "MarketImpactEstimate",
    "ExecutionRiskAssessment",
    "MarketImpactModel",
    # Margin Manager
    "MarginManager",
    "MarginManagerConfig",
    "MarginRequirement",
    "AccountMarginStatus",
    "MarginType",
    "MarginCallType",
    "LeverageLevel",
    # Trade Validator
    "TradeValidator",
    "TradeValidatorConfig",
    "OrderRequest",
    "ValidationResponse",
    "ValidationResult",
    "RejectionReason",
    "OrderType",
    "OrderSide",
    # Risk Metrics
    "RiskMetricsCalculator",
    "RiskMetricsConfig",
    "PerformanceMetrics",
    "MetricPeriod",
    # Stress Tester
    "StressTester",
    "StressTesterConfig",
    "StressScenario",
    "StressTestResult",
    "ScenarioType",
    "HistoricalEvent",
    "MonteCarloResult",
    # Risk Limits
    "RiskLimitsManager",
    "RiskLimitDef",
    "LimitCheck",
    "LimitType",
    "LimitAction",
    "LimitStatus",
    # Risk Reporter
    "RiskReporter",
    "RiskReporterConfig",
    "RiskReport",
    "ReportSection",
    "ReportType",
    "ReportFormat",
    "ReportFrequency",
]


def create_risk_manager(
    config: dict | None = None,
    include_position_sizer: bool = True,
) -> RiskManager:
    """
    Create a configured risk manager instance.

    Args:
        config: Optional configuration dictionary
        include_position_sizer: Whether to include position sizer

    Returns:
        Configured RiskManager instance
    """
    if config:
        rm_config = RiskManagerConfig(**config)
    else:
        rm_config = RiskManagerConfig()

    return RiskManager(config=rm_config)


def create_comprehensive_risk_system(
    config: dict | None = None,
) -> dict:
    """
    Create a comprehensive risk management system.

    Args:
        config: Optional configuration dictionary

    Returns:
        Dictionary containing all risk management components
    """
    risk_config = config or {}

    return {
        "risk_manager": RiskManager(
            config=RiskManagerConfig(**risk_config.get("risk_manager", {}))
        ),
        "position_sizer": PositionSizer(
            config=PositionSizerConfig(**risk_config.get("position_sizer", {}))
        ),
        "drawdown_manager": DrawdownManager(
            config=DrawdownManagerConfig(**risk_config.get("drawdown", {}))
        ),
        "exposure_manager": ExposureManager(
            config=ExposureManagerConfig(**risk_config.get("exposure", {}))
        ),
        "stop_loss_manager": StopLossManager(
            config=StopLossConfig(**risk_config.get("stop_loss", {}))
        ),
        "var_calculator": VaRCalculator(
            config=VaRConfig(**risk_config.get("var", {}))
        ),
        "correlation_manager": CorrelationRiskManager(
            config=CorrelationRiskConfig(**risk_config.get("correlation", {}))
        ),
        "liquidity_manager": LiquidityRiskManager(
            config=LiquidityRiskConfig(**risk_config.get("liquidity", {}))
        ),
        "margin_manager": MarginManager(
            config=MarginManagerConfig(**risk_config.get("margin", {}))
        ),
        "trade_validator": TradeValidator(
            config=TradeValidatorConfig(**risk_config.get("trade_validator", {}))
        ),
        "risk_metrics": RiskMetricsCalculator(
            config=RiskMetricsConfig(**risk_config.get("risk_metrics", {}))
        ),
        "stress_tester": StressTester(
            config=StressTesterConfig(**risk_config.get("stress_test", {}))
        ),
        "risk_limits": RiskLimitsManager(),
        "risk_reporter": RiskReporter(
            config=RiskReporterConfig(**risk_config.get("reporter", {}))
        ),
    }
