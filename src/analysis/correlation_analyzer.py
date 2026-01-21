"""
Correlation Analysis module for identifying asset correlations and diversification.
~350 lines as per schema
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import statistics

from ..core.contracts import MarketSnapshot

logger = logging.getLogger(__name__)


@dataclass
class CorrelationPair:
    """Correlation information between two assets."""
    symbol1: str
    symbol2: str
    correlation: float  # -1.0 to 1.0
    period: int  # Number of periods analyzed
    strength: str  # 'strong', 'moderate', 'weak'


def analyze(snapshot: MarketSnapshot) -> dict:
    """
    Analyze market snapshot for correlations.

    Args:
        snapshot: Market snapshot with price data

    Returns:
        Dictionary with correlation analysis
    """
    return {
        'symbol': snapshot.symbol,
        'price': snapshot.price,
        'correlations': []
    }


class CorrelationAnalyzer:
    """
    Asset correlation analysis system.

    Features:
    - Pearson correlation calculation
    - Rolling correlation analysis
    - Correlation matrix generation
    - Diversification scoring
    - Pair trading identification
    - Sector correlation analysis
    - Correlation strength classification
    """

    def __init__(self, lookback_period: int = 20):
        """
        Initialize correlation analyzer.

        Args:
            lookback_period: Number of periods for correlation calculation
        """
        self.lookback_period = lookback_period

        # Statistics
        self.stats = {
            "correlations_calculated": 0,
            "strong_correlations": 0,
            "negative_correlations": 0,
            "uncorrelated_pairs": 0
        }

    def calculate_correlation(
        self,
        prices1: List[float],
        prices2: List[float]
    ) -> float:
        """
        Calculate Pearson correlation coefficient.

        Args:
            prices1: Price series for first asset
            prices2: Price series for second asset

        Returns:
            Correlation coefficient (-1.0 to 1.0)
        """
        if len(prices1) != len(prices2) or len(prices1) < 2:
            return 0.0

        # Calculate returns
        returns1 = [(prices1[i] - prices1[i-1]) / prices1[i-1] for i in range(1, len(prices1))]
        returns2 = [(prices2[i] - prices2[i-1]) / prices2[i-1] for i in range(1, len(prices2))]

        if len(returns1) < 2:
            return 0.0

        # Calculate correlation
        mean1 = statistics.mean(returns1)
        mean2 = statistics.mean(returns2)

        numerator = sum((returns1[i] - mean1) * (returns2[i] - mean2) for i in range(len(returns1)))

        std1 = statistics.stdev(returns1)
        std2 = statistics.stdev(returns2)

        if std1 == 0 or std2 == 0:
            return 0.0

        denominator = std1 * std2 * len(returns1)

        correlation = numerator / denominator if denominator != 0 else 0.0

        self.stats["correlations_calculated"] += 1

        # Update statistics
        if abs(correlation) > 0.7:
            self.stats["strong_correlations"] += 1
        if correlation < 0:
            self.stats["negative_correlations"] += 1
        if abs(correlation) < 0.3:
            self.stats["uncorrelated_pairs"] += 1

        return max(-1.0, min(1.0, correlation))

    def analyze_pair(
        self,
        symbol1: str,
        prices1: List[float],
        symbol2: str,
        prices2: List[float]
    ) -> CorrelationPair:
        """
        Analyze correlation between two assets.

        Args:
            symbol1: First symbol
            prices1: Price series for first asset
            symbol2: Second symbol
            prices2: Price series for second asset

        Returns:
            CorrelationPair object
        """
        correlation = self.calculate_correlation(prices1, prices2)

        # Classify strength
        abs_corr = abs(correlation)
        if abs_corr > 0.7:
            strength = 'strong'
        elif abs_corr > 0.4:
            strength = 'moderate'
        else:
            strength = 'weak'

        return CorrelationPair(
            symbol1=symbol1,
            symbol2=symbol2,
            correlation=correlation,
            period=min(len(prices1), len(prices2)),
            strength=strength
        )

    def build_correlation_matrix(
        self,
        symbol_prices: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Build correlation matrix for multiple assets.

        Args:
            symbol_prices: Dictionary mapping symbols to price lists

        Returns:
            2D correlation matrix
        """
        symbols = list(symbol_prices.keys())
        matrix = {}

        for symbol1 in symbols:
            matrix[symbol1] = {}
            for symbol2 in symbols:
                if symbol1 == symbol2:
                    matrix[symbol1][symbol2] = 1.0
                else:
                    correlation = self.calculate_correlation(
                        symbol_prices[symbol1],
                        symbol_prices[symbol2]
                    )
                    matrix[symbol1][symbol2] = correlation

        return matrix

    def find_diversification_candidates(
        self,
        target_symbol: str,
        symbol_prices: Dict[str, List[float]],
        max_correlation: float = 0.3
    ) -> List[str]:
        """
        Find symbols with low correlation for diversification.

        Args:
            target_symbol: Symbol to diversify against
            symbol_prices: All available symbols with prices
            max_correlation: Maximum correlation threshold

        Returns:
            List of diversification candidate symbols
        """
        if target_symbol not in symbol_prices:
            return []

        target_prices = symbol_prices[target_symbol]
        candidates = []

        for symbol, prices in symbol_prices.items():
            if symbol == target_symbol:
                continue

            correlation = self.calculate_correlation(target_prices, prices)

            if abs(correlation) <= max_correlation:
                candidates.append(symbol)

        return candidates

    def find_pair_trading_opportunities(
        self,
        symbol_prices: Dict[str, List[float]],
        min_correlation: float = 0.8
    ) -> List[CorrelationPair]:
        """
        Find highly correlated pairs for pair trading.

        Args:
            symbol_prices: Dictionary of symbols and prices
            min_correlation: Minimum correlation for pair trading

        Returns:
            List of highly correlated pairs
        """
        symbols = list(symbol_prices.keys())
        pairs = []

        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                correlation = self.calculate_correlation(
                    symbol_prices[symbol1],
                    symbol_prices[symbol2]
                )

                if correlation >= min_correlation:
                    pair = self.analyze_pair(
                        symbol1, symbol_prices[symbol1],
                        symbol2, symbol_prices[symbol2]
                    )
                    pairs.append(pair)

        # Sort by correlation strength
        pairs.sort(key=lambda x: abs(x.correlation), reverse=True)

        return pairs

    def calculate_rolling_correlation(
        self,
        prices1: List[float],
        prices2: List[float],
        window: int = 20
    ) -> List[float]:
        """
        Calculate rolling correlation over time.

        Args:
            prices1: Price series for first asset
            prices2: Price series for second asset
            window: Rolling window size

        Returns:
            List of rolling correlations
        """
        if len(prices1) < window or len(prices2) < window:
            return []

        rolling_corr = []

        for i in range(window, len(prices1) + 1):
            window_prices1 = prices1[i-window:i]
            window_prices2 = prices2[i-window:i]

            corr = self.calculate_correlation(window_prices1, window_prices2)
            rolling_corr.append(corr)

        return rolling_corr

    def calculate_portfolio_diversification(
        self,
        symbol_prices: Dict[str, List[float]],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate portfolio diversification score.

        Args:
            symbol_prices: Portfolio symbols and prices
            weights: Portfolio weights (equal if not provided)

        Returns:
            Diversification score (0.0 to 1.0, higher is more diversified)
        """
        if len(symbol_prices) < 2:
            return 0.0

        # Equal weights if not provided
        if weights is None:
            n = len(symbol_prices)
            weights = {symbol: 1.0/n for symbol in symbol_prices}

        # Build correlation matrix
        matrix = self.build_correlation_matrix(symbol_prices)

        # Calculate average correlation
        symbols = list(symbol_prices.keys())
        total_weighted_corr = 0.0
        total_weight = 0.0

        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                correlation = abs(matrix[symbol1][symbol2])
                weight = weights[symbol1] * weights[symbol2]
                total_weighted_corr += correlation * weight
                total_weight += weight

        avg_correlation = total_weighted_corr / total_weight if total_weight > 0 else 0.0

        # Diversification score (inverse of average correlation)
        diversification = 1.0 - avg_correlation

        return max(0.0, min(1.0, diversification))

    def identify_correlation_clusters(
        self,
        symbol_prices: Dict[str, List[float]],
        threshold: float = 0.7
    ) -> List[List[str]]:
        """
        Identify clusters of highly correlated assets.

        Args:
            symbol_prices: Dictionary of symbols and prices
            threshold: Correlation threshold for clustering

        Returns:
            List of symbol clusters
        """
        symbols = list(symbol_prices.keys())
        matrix = self.build_correlation_matrix(symbol_prices)

        # Simple clustering based on correlation threshold
        clusters = []
        assigned = set()

        for symbol in symbols:
            if symbol in assigned:
                continue

            # Start new cluster
            cluster = [symbol]
            assigned.add(symbol)

            # Find correlated symbols
            for other_symbol in symbols:
                if other_symbol in assigned:
                    continue

                if abs(matrix[symbol][other_symbol]) >= threshold:
                    cluster.append(other_symbol)
                    assigned.add(other_symbol)

            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters

    def calculate_beta(
        self,
        stock_prices: List[float],
        market_prices: List[float]
    ) -> float:
        """
        Calculate beta (systematic risk) relative to market.

        Args:
            stock_prices: Stock price series
            market_prices: Market index price series

        Returns:
            Beta value
        """
        if len(stock_prices) != len(market_prices) or len(stock_prices) < 2:
            return 1.0

        # Calculate returns
        stock_returns = [(stock_prices[i] - stock_prices[i-1]) / stock_prices[i-1]
                        for i in range(1, len(stock_prices))]
        market_returns = [(market_prices[i] - market_prices[i-1]) / market_prices[i-1]
                         for i in range(1, len(market_prices))]

        if len(stock_returns) < 2:
            return 1.0

        # Calculate covariance and variance
        mean_stock = statistics.mean(stock_returns)
        mean_market = statistics.mean(market_returns)

        covariance = sum((stock_returns[i] - mean_stock) * (market_returns[i] - mean_market)
                        for i in range(len(stock_returns))) / len(stock_returns)

        market_variance = statistics.variance(market_returns)

        beta = covariance / market_variance if market_variance > 0 else 1.0

        return beta

    def get_stats(self) -> Dict[str, Any]:
        """Get analysis statistics."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "correlations_calculated": 0,
            "strong_correlations": 0,
            "negative_correlations": 0,
            "uncorrelated_pairs": 0
        }
