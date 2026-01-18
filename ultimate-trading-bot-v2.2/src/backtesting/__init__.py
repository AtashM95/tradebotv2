"""
Backtesting Package for Ultimate Trading Bot v2.2.

This package provides comprehensive backtesting capabilities including
historical simulation, performance analysis, walk-forward optimization,
and Monte Carlo analysis.
"""

from ultimate_trading_bot.backtesting.backtest_engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestMode,
    BacktestResult,
    CommissionModel,
    FillModel,
    SlippageModel,
    create_backtest_engine,
)
from ultimate_trading_bot.backtesting.data_loader import (
    CSVDataProvider,
    DataCache,
    DataFrequency,
    DataLoader,
    DataLoaderConfig,
    DataPreprocessor,
    DataRequest,
    DataSource,
    DataType,
    DataValidationResult,
    DataValidator,
    LoadedData,
    ParquetDataProvider,
    create_data_loader,
)
from ultimate_trading_bot.backtesting.monte_carlo import (
    ConfidenceLevel,
    DrawdownDistribution,
    MonteCarloConfig,
    MonteCarloResult,
    MonteCarloSimulator,
    OptimalFCalculator,
    SimulationMethod,
    SimulationPath,
    create_monte_carlo_simulator,
)
from ultimate_trading_bot.backtesting.parameter_sweep import (
    EvaluationResult,
    OptimizationMetric,
    ParameterSweep,
    ParameterSweepConfig,
    ParameterSpec,
    ParameterSpace,
    ParameterType,
    SweepMethod,
    SweepResult,
    create_parameter_sweep,
)
from ultimate_trading_bot.backtesting.performance_analyzer import (
    BenchmarkMetrics,
    PerformanceAnalyzer,
    PerformanceReport,
    RatioMetrics,
    ReturnMetrics,
    RiskMetrics,
    TradeMetrics,
)
from ultimate_trading_bot.backtesting.report_generator import (
    HTMLReportBuilder,
    JSONReportBuilder,
    MarkdownReportBuilder,
    ReportConfig,
    ReportData,
    ReportFormat,
    ReportGenerator,
    ReportMetrics,
    ReportSection,
    TradeRecord,
    create_report_generator,
)
from ultimate_trading_bot.backtesting.scenario_generator import (
    GeneratedScenario,
    HistoricalScenarioExtractor,
    MarketRegime,
    ScenarioConfig,
    ScenarioGenerator,
    ScenarioSet,
    ScenarioType,
    StressScenarioType,
    SyntheticScenarioGenerator,
    create_scenario_generator,
)
from ultimate_trading_bot.backtesting.visualization import (
    ChartConfig,
    ChartData,
    ChartResult,
    ChartType,
    ChartVisualizer,
    ColorPalette,
    ColorScheme,
    SVGChartGenerator,
    VisualizationConfig,
    create_visualizer,
)
from ultimate_trading_bot.backtesting.walk_forward import (
    OptimizationObjective,
    OptimizationResult,
    WalkForwardAnalyzer,
    WalkForwardConfig,
    WalkForwardMethod,
    WalkForwardResult,
    WindowPeriod,
    WindowResult,
    create_walk_forward_analyzer,
)

__all__ = [
    # Backtest Engine
    "BacktestConfig",
    "BacktestEngine",
    "BacktestMode",
    "BacktestResult",
    "CommissionModel",
    "FillModel",
    "SlippageModel",
    "create_backtest_engine",
    # Data Loader
    "CSVDataProvider",
    "DataCache",
    "DataFrequency",
    "DataLoader",
    "DataLoaderConfig",
    "DataPreprocessor",
    "DataRequest",
    "DataSource",
    "DataType",
    "DataValidationResult",
    "DataValidator",
    "LoadedData",
    "ParquetDataProvider",
    "create_data_loader",
    # Monte Carlo
    "ConfidenceLevel",
    "DrawdownDistribution",
    "MonteCarloConfig",
    "MonteCarloResult",
    "MonteCarloSimulator",
    "OptimalFCalculator",
    "SimulationMethod",
    "SimulationPath",
    "create_monte_carlo_simulator",
    # Parameter Sweep
    "EvaluationResult",
    "OptimizationMetric",
    "ParameterSweep",
    "ParameterSweepConfig",
    "ParameterSpec",
    "ParameterSpace",
    "ParameterType",
    "SweepMethod",
    "SweepResult",
    "create_parameter_sweep",
    # Performance Analyzer
    "BenchmarkMetrics",
    "PerformanceAnalyzer",
    "PerformanceReport",
    "RatioMetrics",
    "ReturnMetrics",
    "RiskMetrics",
    "TradeMetrics",
    # Report Generator
    "HTMLReportBuilder",
    "JSONReportBuilder",
    "MarkdownReportBuilder",
    "ReportConfig",
    "ReportData",
    "ReportFormat",
    "ReportGenerator",
    "ReportMetrics",
    "ReportSection",
    "TradeRecord",
    "create_report_generator",
    # Scenario Generator
    "GeneratedScenario",
    "HistoricalScenarioExtractor",
    "MarketRegime",
    "ScenarioConfig",
    "ScenarioGenerator",
    "ScenarioSet",
    "ScenarioType",
    "StressScenarioType",
    "SyntheticScenarioGenerator",
    "create_scenario_generator",
    # Visualization
    "ChartConfig",
    "ChartData",
    "ChartResult",
    "ChartType",
    "ChartVisualizer",
    "ColorPalette",
    "ColorScheme",
    "SVGChartGenerator",
    "VisualizationConfig",
    "create_visualizer",
    # Walk Forward
    "OptimizationObjective",
    "OptimizationResult",
    "WalkForwardAnalyzer",
    "WalkForwardConfig",
    "WalkForwardMethod",
    "WalkForwardResult",
    "WindowPeriod",
    "WindowResult",
    "create_walk_forward_analyzer",
]


def create_full_backtest_suite(
    data_dir: str = "./data",
    output_dir: str = "./output",
    initial_capital: float = 100000.0,
) -> dict:
    """
    Create a complete backtesting suite with all components.

    Args:
        data_dir: Directory containing historical data
        output_dir: Directory for output files
        initial_capital: Initial capital for backtesting

    Returns:
        Dictionary with all backtesting components
    """
    data_loader = create_data_loader(csv_dir=data_dir)

    backtest_engine = create_backtest_engine(
        initial_capital=initial_capital,
        commission_rate=0.001,
        slippage_rate=0.0005,
    )

    performance_analyzer = PerformanceAnalyzer()

    monte_carlo = create_monte_carlo_simulator(
        num_simulations=10000,
        method="bootstrap",
    )

    walk_forward = create_walk_forward_analyzer(
        method="rolling",
        in_sample_periods=252,
        out_sample_periods=63,
    )

    scenario_generator = create_scenario_generator(
        num_scenarios=100,
        scenario_length=252,
    )

    report_generator = create_report_generator(
        output_dir=f"{output_dir}/reports",
        formats=["html", "json"],
    )

    visualizer = create_visualizer(
        output_dir=f"{output_dir}/charts",
        color_scheme="default",
    )

    parameter_sweep = create_parameter_sweep(
        method="random",
        max_iterations=1000,
        metric="sharpe_ratio",
    )

    return {
        "data_loader": data_loader,
        "backtest_engine": backtest_engine,
        "performance_analyzer": performance_analyzer,
        "monte_carlo": monte_carlo,
        "walk_forward": walk_forward,
        "scenario_generator": scenario_generator,
        "report_generator": report_generator,
        "visualizer": visualizer,
        "parameter_sweep": parameter_sweep,
    }
