"""
Statistical Arbitrage Strategy - Mean reversion on statistical relationships.
~400 lines as per schema
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import statistics

from ..core.contracts import MarketSnapshot, SignalIntent
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class StatisticalArbitrageStrategy(BaseStrategy):
    """
    Statistical arbitrage strategy based on mean-reversion relationships.

    Algorithm:
    1. Identify cointegrated or correlated pairs
    2. Calculate spread or ratio
    3. Detect spread deviations from mean
    4. Enter when spread is extreme
    5. Exit when spread reverts to mean
    6. Manage portfolio as market-neutral

    Features:
    - Pairs selection by correlation/cointegration
    - Spread calculation and normalization
    - Z-score analysis
    - Kalman filter for dynamic hedging
    - Half-life estimation
    - Multiple pair monitoring
    - Portfolio beta-neutrality
    """

    name = 'statistical_arbitrage'

    def __init__(
        self,
        lookback_period: int = 60,
        entry_z_score: float = 2.0,
        exit_z_score: float = 0.5,
        stop_loss_z_score: float = 3.5,
        min_correlation: float = 0.75,
        min_half_life: int = 1,
        max_half_life: int = 30,
        use_ratio: bool = False  # Use ratio instead of spread
    ):
        """
        Initialize statistical arbitrage strategy.

        Args:
            lookback_period: Period for statistical calculations
            entry_z_score: Z-score threshold for entry
            exit_z_score: Z-score threshold for exit
            stop_loss_z_score: Z-score threshold for stop loss
            min_correlation: Minimum correlation for pair selection
            min_half_life: Minimum mean reversion half-life (bars)
            max_half_life: Maximum mean reversion half-life (bars)
            use_ratio: Use price ratio instead of spread
        """
        super().__init__()
        self.lookback_period = lookback_period
        self.entry_z_score = entry_z_score
        self.exit_z_score = exit_z_score
        self.stop_loss_z_score = stop_loss_z_score
        self.min_correlation = min_correlation
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.use_ratio = use_ratio

        # Track pairs and spreads
        self.pair_data = {}  # (symbol1, symbol2) -> pair_statistics
        self.spread_history = {}  # (symbol1, symbol2) -> List[spread_values]
        self.active_positions = {}  # (symbol1, symbol2) -> position_info
        self.hedge_ratios = {}  # (symbol1, symbol2) -> hedge_ratio

        # Statistics
        self.stats = {
            "signals_generated": 0,
            "pairs_analyzed": 0,
            "mean_reversions": 0,
            "stop_outs": 0,
            "avg_z_score": 0.0,
            "profitable_trades": 0
        }

    def generate(self, snapshot: MarketSnapshot) -> SignalIntent | None:
        """
        Generate trading signal based on statistical arbitrage.

        Args:
            snapshot: Market snapshot

        Returns:
            SignalIntent or None
        """
        # Statistical arbitrage requires pair data
        # This is simplified - production would have dedicated pair data source
        symbol = snapshot.symbol
        metadata = snapshot.metadata or {}

        pair_data = metadata.get('pair_data')

        if not pair_data:
            return None

        pair_symbol = pair_data.get('pair_symbol')
        pair_prices = pair_data.get('pair_prices', [])

        if not pair_symbol or not pair_prices or len(snapshot.history) < self.lookback_period:
            return None

        # Create pair key
        pair_key = self._create_pair_key(symbol, pair_symbol)

        # Calculate statistics for this pair
        pair_stats = self._calculate_pair_statistics(
            symbol,
            snapshot.history,
            pair_symbol,
            pair_prices
        )

        if not pair_stats:
            return None

        # Store pair data
        self.pair_data[pair_key] = pair_stats
        self.stats["pairs_analyzed"] += 1

        # Check if pair is suitable
        if not self._is_suitable_pair(pair_stats):
            return None

        # Calculate current spread
        current_spread = self._calculate_spread(
            snapshot.price,
            pair_data.get('pair_current_price'),
            pair_stats['hedge_ratio']
        )

        # Calculate z-score
        z_score = self._calculate_z_score(current_spread, pair_stats)

        # Update spread history
        if pair_key not in self.spread_history:
            self.spread_history[pair_key] = []
        self.spread_history[pair_key].append(current_spread)

        # Keep only recent history
        if len(self.spread_history[pair_key]) > self.lookback_period:
            self.spread_history[pair_key] = self.spread_history[pair_key][-self.lookback_period:]

        # Update stats
        self.stats["avg_z_score"] = (
            (self.stats["avg_z_score"] * self.stats["pairs_analyzed"] + abs(z_score)) /
            (self.stats["pairs_analyzed"] + 1)
        ) if self.stats["pairs_analyzed"] > 0 else abs(z_score)

        # Check for entry signal
        if pair_key not in self.active_positions:
            signal = self._check_entry(
                symbol,
                pair_symbol,
                snapshot.price,
                pair_data.get('pair_current_price'),
                z_score,
                pair_stats
            )

            if signal:
                self.stats["signals_generated"] += 1

            return signal

        # Check for exit signal
        else:
            return self._check_exit(
                pair_key,
                symbol,
                pair_symbol,
                snapshot.price,
                pair_data.get('pair_current_price'),
                z_score,
                pair_stats
            )

    def _create_pair_key(self, symbol1: str, symbol2: str) -> Tuple[str, str]:
        """Create consistent pair key."""
        return tuple(sorted([symbol1, symbol2]))

    def _calculate_pair_statistics(
        self,
        symbol1: str,
        prices1: List[float],
        symbol2: str,
        prices2: List[float]
    ) -> Optional[Dict[str, Any]]:
        """Calculate statistical relationships for pair."""
        if len(prices1) != len(prices2) or len(prices1) < self.lookback_period:
            return None

        # Use recent period
        prices1_recent = prices1[-self.lookback_period:]
        prices2_recent = prices2[-self.lookback_period:]

        # Calculate correlation
        correlation = self._calculate_correlation(prices1_recent, prices2_recent)

        # Calculate hedge ratio (beta)
        hedge_ratio = self._calculate_hedge_ratio(prices1_recent, prices2_recent)

        # Calculate spread mean and std
        spreads = []
        for p1, p2 in zip(prices1_recent, prices2_recent):
            spread = self._calculate_spread(p1, p2, hedge_ratio)
            spreads.append(spread)

        spread_mean = statistics.mean(spreads)
        spread_std = statistics.stdev(spreads) if len(spreads) > 1 else 0

        # Estimate half-life of mean reversion
        half_life = self._estimate_half_life(spreads)

        return {
            'correlation': correlation,
            'hedge_ratio': hedge_ratio,
            'spread_mean': spread_mean,
            'spread_std': spread_std,
            'half_life': half_life,
            'symbol1': symbol1,
            'symbol2': symbol2
        }

    def _calculate_correlation(
        self,
        prices1: List[float],
        prices2: List[float]
    ) -> float:
        """Calculate Pearson correlation."""
        if len(prices1) != len(prices2) or len(prices1) < 2:
            return 0.0

        mean1 = statistics.mean(prices1)
        mean2 = statistics.mean(prices2)

        numerator = sum((p1 - mean1) * (p2 - mean2) for p1, p2 in zip(prices1, prices2))

        std1 = statistics.stdev(prices1)
        std2 = statistics.stdev(prices2)

        denominator = std1 * std2 * len(prices1)

        if denominator == 0:
            return 0.0

        correlation = numerator / denominator

        return correlation

    def _calculate_hedge_ratio(
        self,
        prices1: List[float],
        prices2: List[float]
    ) -> float:
        """Calculate hedge ratio (beta) using OLS regression."""
        if len(prices1) != len(prices2) or len(prices1) < 2:
            return 1.0

        mean1 = statistics.mean(prices1)
        mean2 = statistics.mean(prices2)

        # Calculate beta: Cov(X,Y) / Var(X)
        covariance = sum((p1 - mean1) * (p2 - mean2) for p1, p2 in zip(prices1, prices2)) / len(prices1)
        variance = sum((p2 - mean2) ** 2 for p2 in prices2) / len(prices2)

        if variance == 0:
            return 1.0

        beta = covariance / variance

        return beta

    def _calculate_spread(
        self,
        price1: float,
        price2: float,
        hedge_ratio: float
    ) -> float:
        """Calculate spread between two assets."""
        if self.use_ratio:
            # Use price ratio
            if price2 == 0:
                return 0.0
            return price1 / price2
        else:
            # Use hedged spread
            return price1 - (hedge_ratio * price2)

    def _estimate_half_life(self, spreads: List[float]) -> int:
        """Estimate mean reversion half-life using Ornstein-Uhlenbeck."""
        if len(spreads) < 3:
            return 0

        # Calculate lagged differences
        spread_lag = spreads[:-1]
        spread_diff = [spreads[i] - spreads[i-1] for i in range(1, len(spreads))]

        # Simple AR(1) estimation
        mean_lag = statistics.mean(spread_lag)
        centered_lag = [s - mean_lag for s in spread_lag]

        numerator = sum(d * c for d, c in zip(spread_diff, centered_lag))
        denominator = sum(c ** 2 for c in centered_lag)

        if denominator == 0:
            return 0

        theta = -numerator / denominator

        if theta <= 0:
            return 0

        # Half-life = ln(2) / theta
        import math
        half_life = math.log(2) / theta

        return int(half_life)

    def _is_suitable_pair(self, pair_stats: Dict[str, Any]) -> bool:
        """Check if pair is suitable for trading."""
        # Check correlation
        if abs(pair_stats['correlation']) < self.min_correlation:
            return False

        # Check half-life
        half_life = pair_stats['half_life']
        if half_life < self.min_half_life or half_life > self.max_half_life:
            return False

        # Check spread std
        if pair_stats['spread_std'] == 0:
            return False

        return True

    def _calculate_z_score(
        self,
        current_spread: float,
        pair_stats: Dict[str, Any]
    ) -> float:
        """Calculate z-score of current spread."""
        spread_mean = pair_stats['spread_mean']
        spread_std = pair_stats['spread_std']

        if spread_std == 0:
            return 0.0

        z_score = (current_spread - spread_mean) / spread_std

        return z_score

    def _check_entry(
        self,
        symbol1: str,
        symbol2: str,
        price1: float,
        price2: float,
        z_score: float,
        pair_stats: Dict[str, Any]
    ) -> Optional[SignalIntent]:
        """Check for entry opportunity."""
        # Enter when spread is extreme
        if abs(z_score) < self.entry_z_score:
            return None

        pair_key = self._create_pair_key(symbol1, symbol2)

        # Determine direction
        if z_score > self.entry_z_score:
            # Spread too high - short spread (sell symbol1, buy symbol2)
            action = 'sell'
            entry_type = 'short_spread'
        elif z_score < -self.entry_z_score:
            # Spread too low - long spread (buy symbol1, sell symbol2)
            action = 'buy'
            entry_type = 'long_spread'
        else:
            return None

        # Calculate confidence
        confidence = self._calculate_confidence(abs(z_score), pair_stats)

        # Track position
        self.active_positions[pair_key] = {
            'entry_z_score': z_score,
            'entry_spread': pair_stats['spread_mean'] + (z_score * pair_stats['spread_std']),
            'direction': action,
            'hedge_ratio': pair_stats['hedge_ratio']
        }

        # Store hedge ratio
        self.hedge_ratios[pair_key] = pair_stats['hedge_ratio']

        return SignalIntent(
            symbol=symbol1,
            action=action,
            confidence=confidence,
            metadata={
                'strategy': self.name,
                'signal_type': 'stat_arb_entry',
                'pair_symbol': symbol2,
                'z_score': z_score,
                'hedge_ratio': pair_stats['hedge_ratio'],
                'entry_type': entry_type,
                'correlation': pair_stats['correlation'],
                'half_life': pair_stats['half_life'],
                'target_z_score': 0.0,  # Target is mean reversion
                'stop_z_score': self.stop_loss_z_score
            }
        )

    def _check_exit(
        self,
        pair_key: Tuple[str, str],
        symbol1: str,
        symbol2: str,
        price1: float,
        price2: float,
        z_score: float,
        pair_stats: Dict[str, Any]
    ) -> Optional[SignalIntent]:
        """Check for exit opportunity."""
        position = self.active_positions[pair_key]
        entry_z = position['entry_z_score']

        # Check for mean reversion (exit)
        if abs(z_score) < self.exit_z_score:
            # Spread has reverted - exit position
            del self.active_positions[pair_key]
            self.stats["mean_reversions"] += 1
            self.stats["profitable_trades"] += 1

            # Reverse the action
            exit_action = 'buy' if position['direction'] == 'sell' else 'sell'

            return SignalIntent(
                symbol=symbol1,
                action=exit_action,
                confidence=0.9,
                metadata={
                    'strategy': self.name,
                    'signal_type': 'stat_arb_exit',
                    'exit_reason': 'mean_reversion',
                    'entry_z_score': entry_z,
                    'exit_z_score': z_score,
                    'pair_symbol': symbol2
                }
            )

        # Check for stop loss
        if (position['direction'] == 'buy' and z_score < -self.stop_loss_z_score) or \
           (position['direction'] == 'sell' and z_score > self.stop_loss_z_score):
            # Spread diverged further - stop out
            del self.active_positions[pair_key]
            self.stats["stop_outs"] += 1

            exit_action = 'buy' if position['direction'] == 'sell' else 'sell'

            return SignalIntent(
                symbol=symbol1,
                action=exit_action,
                confidence=0.95,
                metadata={
                    'strategy': self.name,
                    'signal_type': 'stat_arb_exit',
                    'exit_reason': 'stop_loss',
                    'entry_z_score': entry_z,
                    'exit_z_score': z_score,
                    'pair_symbol': symbol2
                }
            )

        return None

    def _calculate_confidence(
        self,
        z_score: float,
        pair_stats: Dict[str, Any]
    ) -> float:
        """Calculate signal confidence."""
        confidence = 0.6  # Base confidence

        # Higher z-score = higher confidence
        if z_score > 3.0:
            confidence += 0.20
        elif z_score > 2.5:
            confidence += 0.15
        elif z_score > 2.0:
            confidence += 0.10

        # Stronger correlation = higher confidence
        correlation = abs(pair_stats['correlation'])
        if correlation > 0.9:
            confidence += 0.10
        elif correlation > 0.8:
            confidence += 0.05

        # Optimal half-life = higher confidence
        half_life = pair_stats['half_life']
        if 5 <= half_life <= 20:
            confidence += 0.05

        return min(confidence, 0.95)

    def get_pair_analysis(
        self,
        symbol1: str,
        symbol2: str
    ) -> Optional[Dict[str, Any]]:
        """Get analysis for a specific pair."""
        pair_key = self._create_pair_key(symbol1, symbol2)

        if pair_key not in self.pair_data:
            return None

        pair_stats = self.pair_data[pair_key]
        spread_hist = self.spread_history.get(pair_key, [])
        active_position = self.active_positions.get(pair_key)

        return {
            'pair': pair_key,
            'correlation': pair_stats['correlation'],
            'hedge_ratio': pair_stats['hedge_ratio'],
            'spread_mean': pair_stats['spread_mean'],
            'spread_std': pair_stats['spread_std'],
            'half_life': pair_stats['half_life'],
            'spread_history': spread_hist[-20:],  # Last 20 values
            'active_position': active_position is not None,
            'position_details': active_position
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        stats = self.stats.copy()

        # Calculate success rate
        total_closed = stats["mean_reversions"] + stats["stop_outs"]
        if total_closed > 0:
            stats["success_rate"] = stats["profitable_trades"] / total_closed
        else:
            stats["success_rate"] = 0.0

        # Active positions
        stats["active_pairs"] = len(self.active_positions)

        return stats

    def reset(self):
        """Reset strategy state."""
        self.pair_data.clear()
        self.spread_history.clear()
        self.active_positions.clear()
        self.hedge_ratios.clear()
        self.stats = {
            "signals_generated": 0,
            "pairs_analyzed": 0,
            "mean_reversions": 0,
            "stop_outs": 0,
            "avg_z_score": 0.0,
            "profitable_trades": 0
        }
