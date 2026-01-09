"""
Correlation Analysis Module for Ultimate Trading Bot v2.2.

This module provides correlation analysis between assets
for portfolio diversification and pair trading.
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from src.utils.date_utils import now_utc


logger = logging.getLogger(__name__)


class CorrelationPair(BaseModel):
    """Correlation pair model."""

    symbol1: str
    symbol2: str
    correlation: float = Field(ge=-1.0, le=1.0)
    period_days: int = Field(default=20)
    relationship: str = Field(default="neutral")
    stability: float = Field(ge=0.0, le=1.0, default=0.5)


class CorrelationMatrix(BaseModel):
    """Correlation matrix model."""

    symbols: list[str]
    matrix: list[list[float]]
    avg_correlation: float = Field(default=0.0)
    max_correlation: float = Field(default=0.0)
    min_correlation: float = Field(default=0.0)
    highly_correlated_pairs: list[CorrelationPair] = Field(default_factory=list)
    negatively_correlated_pairs: list[CorrelationPair] = Field(default_factory=list)


class RollingCorrelation(BaseModel):
    """Rolling correlation model."""

    symbol1: str
    symbol2: str
    correlations: list[float] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    current: float = Field(default=0.0)
    mean: float = Field(default=0.0)
    std: float = Field(default=0.0)
    is_stable: bool = Field(default=True)


class BetaResult(BaseModel):
    """Beta calculation result."""

    symbol: str
    benchmark: str
    beta: float
    alpha: float = Field(default=0.0)
    r_squared: float = Field(ge=0.0, le=1.0, default=0.0)
    interpretation: str = Field(default="")


class CorrelationAnalysisConfig(BaseModel):
    """Configuration for correlation analysis."""

    default_period: int = Field(default=20, ge=5, le=252)
    high_correlation_threshold: float = Field(default=0.7, ge=0.5, le=1.0)
    low_correlation_threshold: float = Field(default=-0.3, ge=-1.0, le=0.0)
    rolling_window: int = Field(default=20, ge=5, le=100)
    stability_threshold: float = Field(default=0.3, ge=0.1, le=0.5)


class CorrelationAnalyzer:
    """
    Correlation analyzer for assets.

    Provides:
    - Pairwise correlation calculation
    - Correlation matrix generation
    - Rolling correlation tracking
    - Beta calculation
    - Correlation stability analysis
    """

    def __init__(
        self,
        config: Optional[CorrelationAnalysisConfig] = None,
    ) -> None:
        """
        Initialize CorrelationAnalyzer.

        Args:
            config: Correlation analysis configuration
        """
        self._config = config or CorrelationAnalysisConfig()

        logger.info("CorrelationAnalyzer initialized")

    def calculate_correlation(
        self,
        returns1: list[float],
        returns2: list[float],
    ) -> float:
        """
        Calculate Pearson correlation between two return series.

        Args:
            returns1: First return series
            returns2: Second return series

        Returns:
            Correlation coefficient (-1 to 1)
        """
        n = min(len(returns1), len(returns2))
        if n < 3:
            return 0.0

        r1 = returns1[-n:]
        r2 = returns2[-n:]

        mean1 = sum(r1) / n
        mean2 = sum(r2) / n

        numerator = sum((r1[i] - mean1) * (r2[i] - mean2) for i in range(n))

        var1 = sum((r1[i] - mean1) ** 2 for i in range(n))
        var2 = sum((r2[i] - mean2) ** 2 for i in range(n))

        denominator = (var1 * var2) ** 0.5

        if denominator == 0:
            return 0.0

        correlation = numerator / denominator

        return max(-1.0, min(1.0, correlation))

    def calculate_pair_correlation(
        self,
        symbol1: str,
        symbol2: str,
        prices1: list[float],
        prices2: list[float],
        period: Optional[int] = None,
    ) -> CorrelationPair:
        """
        Calculate correlation for a pair of symbols.

        Args:
            symbol1: First symbol
            symbol2: Second symbol
            prices1: First price series
            prices2: Second price series
            period: Lookback period

        Returns:
            CorrelationPair with results
        """
        period = period or self._config.default_period

        returns1 = self._calculate_returns(prices1, period)
        returns2 = self._calculate_returns(prices2, period)

        correlation = self.calculate_correlation(returns1, returns2)

        relationship = self._classify_relationship(correlation)

        stability = self._calculate_stability(
            prices1, prices2, period,
        )

        return CorrelationPair(
            symbol1=symbol1,
            symbol2=symbol2,
            correlation=correlation,
            period_days=period,
            relationship=relationship,
            stability=stability,
        )

    def calculate_correlation_matrix(
        self,
        symbols: list[str],
        price_data: dict[str, list[float]],
        period: Optional[int] = None,
    ) -> CorrelationMatrix:
        """
        Calculate correlation matrix for multiple symbols.

        Args:
            symbols: List of symbols
            price_data: Dictionary of symbol -> prices
            period: Lookback period

        Returns:
            CorrelationMatrix with results
        """
        period = period or self._config.default_period
        n = len(symbols)

        matrix = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

        returns_data: dict[str, list[float]] = {}
        for symbol in symbols:
            if symbol in price_data:
                returns_data[symbol] = self._calculate_returns(
                    price_data[symbol], period,
                )

        highly_correlated: list[CorrelationPair] = []
        negatively_correlated: list[CorrelationPair] = []
        all_correlations: list[float] = []

        for i in range(n):
            for j in range(i + 1, n):
                sym1, sym2 = symbols[i], symbols[j]

                if sym1 not in returns_data or sym2 not in returns_data:
                    continue

                corr = self.calculate_correlation(
                    returns_data[sym1],
                    returns_data[sym2],
                )

                matrix[i][j] = corr
                matrix[j][i] = corr
                all_correlations.append(corr)

                pair = CorrelationPair(
                    symbol1=sym1,
                    symbol2=sym2,
                    correlation=corr,
                    period_days=period,
                    relationship=self._classify_relationship(corr),
                )

                if corr >= self._config.high_correlation_threshold:
                    highly_correlated.append(pair)
                elif corr <= self._config.low_correlation_threshold:
                    negatively_correlated.append(pair)

        avg_corr = sum(all_correlations) / len(all_correlations) if all_correlations else 0
        max_corr = max(all_correlations) if all_correlations else 0
        min_corr = min(all_correlations) if all_correlations else 0

        return CorrelationMatrix(
            symbols=symbols,
            matrix=matrix,
            avg_correlation=avg_corr,
            max_correlation=max_corr,
            min_correlation=min_corr,
            highly_correlated_pairs=sorted(
                highly_correlated,
                key=lambda x: x.correlation,
                reverse=True,
            ),
            negatively_correlated_pairs=sorted(
                negatively_correlated,
                key=lambda x: x.correlation,
            ),
        )

    def calculate_rolling_correlation(
        self,
        symbol1: str,
        symbol2: str,
        prices1: list[float],
        prices2: list[float],
        window: Optional[int] = None,
    ) -> RollingCorrelation:
        """
        Calculate rolling correlation over time.

        Args:
            symbol1: First symbol
            symbol2: Second symbol
            prices1: First price series
            prices2: Second price series
            window: Rolling window size

        Returns:
            RollingCorrelation with time series
        """
        window = window or self._config.rolling_window
        n = min(len(prices1), len(prices2))

        if n < window + 5:
            return RollingCorrelation(
                symbol1=symbol1,
                symbol2=symbol2,
                is_stable=True,
            )

        returns1 = self._calculate_returns(prices1)
        returns2 = self._calculate_returns(prices2)

        correlations: list[float] = []

        for i in range(window, n):
            r1_window = returns1[i - window:i]
            r2_window = returns2[i - window:i]
            corr = self.calculate_correlation(r1_window, r2_window)
            correlations.append(corr)

        current = correlations[-1] if correlations else 0
        mean = sum(correlations) / len(correlations) if correlations else 0
        variance = sum((c - mean) ** 2 for c in correlations) / len(correlations) if correlations else 0
        std = variance ** 0.5

        is_stable = std < self._config.stability_threshold

        return RollingCorrelation(
            symbol1=symbol1,
            symbol2=symbol2,
            correlations=correlations,
            current=current,
            mean=mean,
            std=std,
            is_stable=is_stable,
        )

    def calculate_beta(
        self,
        symbol: str,
        asset_prices: list[float],
        benchmark_prices: list[float],
        benchmark: str = "SPY",
        period: Optional[int] = None,
    ) -> BetaResult:
        """
        Calculate beta of an asset relative to benchmark.

        Args:
            symbol: Asset symbol
            asset_prices: Asset price series
            benchmark_prices: Benchmark price series
            benchmark: Benchmark symbol
            period: Lookback period

        Returns:
            BetaResult with beta and alpha
        """
        period = period or self._config.default_period

        asset_returns = self._calculate_returns(asset_prices, period)
        bench_returns = self._calculate_returns(benchmark_prices, period)

        n = min(len(asset_returns), len(bench_returns))
        if n < 5:
            return BetaResult(
                symbol=symbol,
                benchmark=benchmark,
                beta=1.0,
                interpretation="Insufficient data",
            )

        ar = asset_returns[-n:]
        br = bench_returns[-n:]

        bench_mean = sum(br) / n
        asset_mean = sum(ar) / n

        covariance = sum((ar[i] - asset_mean) * (br[i] - bench_mean) for i in range(n)) / n
        bench_variance = sum((br[i] - bench_mean) ** 2 for i in range(n)) / n

        if bench_variance == 0:
            beta = 1.0
        else:
            beta = covariance / bench_variance

        risk_free_rate = 0.0
        alpha = asset_mean - (risk_free_rate + beta * (bench_mean - risk_free_rate))

        predicted = [risk_free_rate + beta * (br[i] - risk_free_rate) for i in range(n)]
        ss_res = sum((ar[i] - predicted[i]) ** 2 for i in range(n))
        ss_tot = sum((ar[i] - asset_mean) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        interpretation = self._interpret_beta(beta)

        return BetaResult(
            symbol=symbol,
            benchmark=benchmark,
            beta=beta,
            alpha=alpha * 252,
            r_squared=max(0, min(1, r_squared)),
            interpretation=interpretation,
        )

    def find_uncorrelated_pairs(
        self,
        symbols: list[str],
        price_data: dict[str, list[float]],
        max_correlation: float = 0.3,
    ) -> list[CorrelationPair]:
        """
        Find pairs of uncorrelated assets for diversification.

        Args:
            symbols: List of symbols
            price_data: Price data dictionary
            max_correlation: Maximum absolute correlation

        Returns:
            List of uncorrelated pairs
        """
        uncorrelated: list[CorrelationPair] = []

        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i + 1:]:
                if sym1 not in price_data or sym2 not in price_data:
                    continue

                pair = self.calculate_pair_correlation(
                    sym1, sym2,
                    price_data[sym1],
                    price_data[sym2],
                )

                if abs(pair.correlation) <= max_correlation:
                    uncorrelated.append(pair)

        return sorted(uncorrelated, key=lambda x: abs(x.correlation))

    def find_pair_trading_candidates(
        self,
        symbols: list[str],
        price_data: dict[str, list[float]],
        min_correlation: float = 0.8,
    ) -> list[CorrelationPair]:
        """
        Find highly correlated pairs for pair trading.

        Args:
            symbols: List of symbols
            price_data: Price data dictionary
            min_correlation: Minimum correlation threshold

        Returns:
            List of pair trading candidates
        """
        candidates: list[CorrelationPair] = []

        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i + 1:]:
                if sym1 not in price_data or sym2 not in price_data:
                    continue

                pair = self.calculate_pair_correlation(
                    sym1, sym2,
                    price_data[sym1],
                    price_data[sym2],
                )

                if pair.correlation >= min_correlation and pair.stability >= 0.5:
                    candidates.append(pair)

        return sorted(candidates, key=lambda x: x.correlation, reverse=True)

    def analyze_portfolio_correlation(
        self,
        symbols: list[str],
        weights: list[float],
        price_data: dict[str, list[float]],
    ) -> dict:
        """
        Analyze correlation within a portfolio.

        Args:
            symbols: Portfolio symbols
            weights: Portfolio weights
            price_data: Price data dictionary

        Returns:
            Portfolio correlation analysis
        """
        matrix = self.calculate_correlation_matrix(symbols, price_data)

        weighted_corr = 0.0
        total_weight = 0.0

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                pair_weight = weights[i] * weights[j]
                weighted_corr += pair_weight * matrix.matrix[i][j]
                total_weight += pair_weight

        avg_weighted_corr = weighted_corr / total_weight if total_weight > 0 else 0

        if avg_weighted_corr > 0.6:
            diversification = "poor"
        elif avg_weighted_corr > 0.3:
            diversification = "moderate"
        else:
            diversification = "good"

        return {
            "average_correlation": matrix.avg_correlation,
            "weighted_correlation": avg_weighted_corr,
            "diversification_quality": diversification,
            "highly_correlated_pairs": len(matrix.highly_correlated_pairs),
            "negative_correlations": len(matrix.negatively_correlated_pairs),
            "concentration_risk": "high" if avg_weighted_corr > 0.7 else "normal",
        }

    def _calculate_returns(
        self,
        prices: list[float],
        period: Optional[int] = None,
    ) -> list[float]:
        """Calculate percentage returns from prices."""
        if period:
            prices = prices[-period - 1:]

        returns: list[float] = []
        for i in range(1, len(prices)):
            if prices[i - 1] != 0:
                ret = (prices[i] - prices[i - 1]) / prices[i - 1]
                returns.append(ret)

        return returns

    def _classify_relationship(self, correlation: float) -> str:
        """Classify correlation relationship."""
        if correlation >= 0.8:
            return "very_strong_positive"
        elif correlation >= 0.6:
            return "strong_positive"
        elif correlation >= 0.3:
            return "moderate_positive"
        elif correlation > -0.3:
            return "weak_or_none"
        elif correlation > -0.6:
            return "moderate_negative"
        elif correlation > -0.8:
            return "strong_negative"
        else:
            return "very_strong_negative"

    def _calculate_stability(
        self,
        prices1: list[float],
        prices2: list[float],
        period: int,
    ) -> float:
        """Calculate correlation stability over time."""
        rolling = self.calculate_rolling_correlation(
            "", "", prices1, prices2, period // 2,
        )

        if rolling.std == 0:
            return 1.0

        stability = 1.0 - min(1.0, rolling.std / 0.5)

        return max(0.0, stability)

    def _interpret_beta(self, beta: float) -> str:
        """Interpret beta value."""
        if beta > 1.5:
            return "Very aggressive - moves significantly more than market"
        elif beta > 1.1:
            return "Aggressive - moves more than market"
        elif beta > 0.9:
            return "Market-like volatility"
        elif beta > 0.5:
            return "Defensive - moves less than market"
        elif beta > 0:
            return "Very defensive - low market sensitivity"
        else:
            return "Inverse relationship with market"

    def __repr__(self) -> str:
        """String representation."""
        return "CorrelationAnalyzer()"
