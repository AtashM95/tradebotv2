"""
Analysis Package for Ultimate Trading Bot v2.2.

This package provides comprehensive technical analysis,
pattern recognition, and market scanning capabilities.

Modules:
    technical_indicators: Technical indicator calculations
    pattern_recognition: Candlestick and chart pattern detection
    trend_analysis: Trend detection and analysis
    signal_generator: Trading signal generation
    market_scanner: Market opportunity scanning
    volume_analysis: Volume-based analysis
    price_analysis: Price action analysis
    market_regime: Market regime detection
    correlation_analysis: Asset correlation analysis
"""

from src.analysis.technical_indicators import (
    TechnicalIndicators,
    MAType,
    IndicatorResult,
    MACDResult,
    BollingerBandsResult,
    IchimokuResult,
    StochResult,
)

from src.analysis.pattern_recognition import (
    PatternRecognition,
    PatternType,
    PatternReliability,
    CandlePattern,
    ChartPattern,
)

from src.analysis.trend_analysis import (
    TrendAnalyzer,
    TrendAnalysisConfig,
    TrendDirection,
    TrendStrength,
    TrendPhase,
    TrendResult,
    TrendChange,
    TrendLine,
)

from src.analysis.signal_generator import (
    SignalGenerator,
    SignalGeneratorConfig,
    SignalType,
    SignalSource,
    SignalTimeframe,
    TradingSignal,
    SignalSummary,
)

from src.analysis.market_scanner import (
    MarketScanner,
    MarketScannerConfig,
    ScanType,
    ScanResult,
    ScanCriteria,
    ScannerStats,
)

from src.analysis.volume_analysis import (
    VolumeAnalyzer,
    VolumeAnalysisConfig,
    VolumeSignal,
    VolumeProfile,
    VolumeAnalysisResult,
    VolumeZone,
)

from src.analysis.price_analysis import (
    PriceAnalyzer,
    PriceAnalysisConfig,
    PriceStructure,
    PriceLevel,
    PriceRange,
    PriceAnalysisResult,
)

from src.analysis.market_regime import (
    MarketRegimeDetector,
    MarketRegimeConfig,
    RegimeType,
    VolatilityRegime,
    TrendRegime,
    RegimeResult,
    RegimeTransition,
)

from src.analysis.correlation_analysis import (
    CorrelationAnalyzer,
    CorrelationAnalysisConfig,
    CorrelationPair,
    CorrelationMatrix,
    RollingCorrelation,
    BetaResult,
)


__all__ = [
    # Technical Indicators
    "TechnicalIndicators",
    "MAType",
    "IndicatorResult",
    "MACDResult",
    "BollingerBandsResult",
    "IchimokuResult",
    "StochResult",
    # Pattern Recognition
    "PatternRecognition",
    "PatternType",
    "PatternReliability",
    "CandlePattern",
    "ChartPattern",
    # Trend Analysis
    "TrendAnalyzer",
    "TrendAnalysisConfig",
    "TrendDirection",
    "TrendStrength",
    "TrendPhase",
    "TrendResult",
    "TrendChange",
    "TrendLine",
    # Signal Generator
    "SignalGenerator",
    "SignalGeneratorConfig",
    "SignalType",
    "SignalSource",
    "SignalTimeframe",
    "TradingSignal",
    "SignalSummary",
    # Market Scanner
    "MarketScanner",
    "MarketScannerConfig",
    "ScanType",
    "ScanResult",
    "ScanCriteria",
    "ScannerStats",
    # Volume Analysis
    "VolumeAnalyzer",
    "VolumeAnalysisConfig",
    "VolumeSignal",
    "VolumeProfile",
    "VolumeAnalysisResult",
    "VolumeZone",
    # Price Analysis
    "PriceAnalyzer",
    "PriceAnalysisConfig",
    "PriceStructure",
    "PriceLevel",
    "PriceRange",
    "PriceAnalysisResult",
    # Market Regime
    "MarketRegimeDetector",
    "MarketRegimeConfig",
    "RegimeType",
    "VolatilityRegime",
    "TrendRegime",
    "RegimeResult",
    "RegimeTransition",
    # Correlation Analysis
    "CorrelationAnalyzer",
    "CorrelationAnalysisConfig",
    "CorrelationPair",
    "CorrelationMatrix",
    "RollingCorrelation",
    "BetaResult",
]
