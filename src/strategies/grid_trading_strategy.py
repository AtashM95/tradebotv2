"""
Grid Trading Strategy - Place buy/sell orders at regular price intervals.
~400 lines as per schema
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal, ROUND_DOWN
import statistics

from ..core.contracts import MarketSnapshot, SignalIntent
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class GridTradingStrategy(BaseStrategy):
    """
    Grid trading strategy that places orders at fixed price intervals.

    Algorithm:
    1. Define price range (upper/lower bounds)
    2. Create grid levels at regular intervals
    3. Place buy orders at lower levels
    4. Place sell orders at higher levels
    5. Rebalance grid as price moves
    6. Take profits as price oscillates

    Features:
    - Dynamic grid generation
    - Configurable grid density
    - Position tracking per level
    - Automatic rebalancing
    - Profit target management
    - Range-bound market optimization
    """

    name = 'grid_trading'

    def __init__(
        self,
        grid_levels: int = 10,
        price_range_pct: float = 0.10,  # 10% range
        profit_per_grid: float = 0.01,  # 1% profit per grid
        max_position_size: float = 1.0,
        rebalance_threshold: float = 0.05,
        auto_adjust_range: bool = True
    ):
        """
        Initialize grid trading strategy.

        Args:
            grid_levels: Number of grid levels
            price_range_pct: Price range as percentage (upper/lower bounds)
            profit_per_grid: Target profit percentage per grid level
            max_position_size: Maximum position size
            rebalance_threshold: Threshold for grid rebalancing
            auto_adjust_range: Automatically adjust range based on volatility
        """
        super().__init__()
        self.grid_levels = grid_levels
        self.price_range_pct = price_range_pct
        self.profit_per_grid = profit_per_grid
        self.max_position_size = max_position_size
        self.rebalance_threshold = rebalance_threshold
        self.auto_adjust_range = auto_adjust_range

        # Track grid state
        self.grid_prices = {}  # symbol -> List[float]
        self.grid_positions = {}  # symbol -> Dict[level, position_info]
        self.base_prices = {}  # symbol -> base_price
        self.upper_bounds = {}  # symbol -> upper_bound
        self.lower_bounds = {}  # symbol -> lower_bound

        # Statistics
        self.stats = {
            "signals_generated": 0,
            "grids_created": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "grids_rebalanced": 0,
            "profits_taken": 0
        }

    def generate(self, snapshot: MarketSnapshot) -> SignalIntent | None:
        """
        Generate trading signal based on grid logic.

        Args:
            snapshot: Market snapshot

        Returns:
            SignalIntent or None
        """
        if not snapshot.history or len(snapshot.history) < 20:
            return None

        symbol = snapshot.symbol
        current_price = snapshot.price

        # Initialize or update grid
        if symbol not in self.grid_prices or self._should_rebalance(symbol, current_price):
            self._create_grid(symbol, snapshot)

        # Check if price is at a grid level
        signal = self._check_grid_levels(symbol, current_price)

        if signal:
            self.stats["signals_generated"] += 1
            if signal.action == 'buy':
                self.stats["buy_signals"] += 1
            elif signal.action == 'sell':
                self.stats["sell_signals"] += 1

        return signal

    def _create_grid(self, symbol: str, snapshot: MarketSnapshot):
        """Create grid levels based on current price."""
        current_price = snapshot.price

        # Calculate base price (center of grid)
        if self.auto_adjust_range and len(snapshot.history) >= 20:
            # Use recent average as base
            base_price = statistics.mean(snapshot.history[-20:])

            # Adjust range based on volatility
            volatility = self._calculate_volatility(snapshot.history[-20:])
            adjusted_range = max(self.price_range_pct, volatility * 2)
        else:
            base_price = current_price
            adjusted_range = self.price_range_pct

        # Calculate bounds
        upper_bound = base_price * (1 + adjusted_range)
        lower_bound = base_price * (1 - adjusted_range)

        # Generate grid levels
        grid_step = (upper_bound - lower_bound) / (self.grid_levels - 1)
        grid_prices = [lower_bound + (i * grid_step) for i in range(self.grid_levels)]

        # Store grid configuration
        self.grid_prices[symbol] = grid_prices
        self.base_prices[symbol] = base_price
        self.upper_bounds[symbol] = upper_bound
        self.lower_bounds[symbol] = lower_bound

        # Initialize positions tracking if needed
        if symbol not in self.grid_positions:
            self.grid_positions[symbol] = {}

        self.stats["grids_created"] += 1

        logger.info(f"Created grid for {symbol}: {self.grid_levels} levels from {lower_bound:.2f} to {upper_bound:.2f}")

    def _should_rebalance(self, symbol: str, current_price: float) -> bool:
        """Check if grid should be rebalanced."""
        if symbol not in self.upper_bounds or symbol not in self.lower_bounds:
            return True

        upper = self.upper_bounds[symbol]
        lower = self.lower_bounds[symbol]

        # Rebalance if price is outside bounds
        if current_price > upper or current_price < lower:
            self.stats["grids_rebalanced"] += 1
            return True

        # Rebalance if price has moved significantly from base
        if symbol in self.base_prices:
            base = self.base_prices[symbol]
            deviation = abs(current_price - base) / base
            if deviation > self.rebalance_threshold:
                self.stats["grids_rebalanced"] += 1
                return True

        return False

    def _check_grid_levels(
        self,
        symbol: str,
        current_price: float
    ) -> Optional[SignalIntent]:
        """Check if current price is at a grid level."""
        if symbol not in self.grid_prices:
            return None

        grid_prices = self.grid_prices[symbol]
        base_price = self.base_prices[symbol]

        # Find nearest grid level
        nearest_level, nearest_price = self._find_nearest_grid(grid_prices, current_price)

        # Check if price is close enough to grid level
        tolerance = nearest_price * 0.002  # 0.2% tolerance
        if abs(current_price - nearest_price) > tolerance:
            return None

        # Determine action based on grid position
        if nearest_price < base_price:
            # Below base price - buy signal
            return self._generate_buy_signal(symbol, current_price, nearest_level, nearest_price)
        elif nearest_price > base_price:
            # Above base price - sell signal
            return self._generate_sell_signal(symbol, current_price, nearest_level, nearest_price)

        return None

    def _generate_buy_signal(
        self,
        symbol: str,
        current_price: float,
        level: int,
        grid_price: float
    ) -> Optional[SignalIntent]:
        """Generate buy signal at grid level."""
        # Check if already have position at this level
        if symbol in self.grid_positions:
            if level in self.grid_positions[symbol]:
                position = self.grid_positions[symbol][level]
                if position['action'] == 'buy' and position['active']:
                    # Already have buy position at this level
                    return None

        # Calculate position size (smaller positions further from base)
        base_price = self.base_prices[symbol]
        distance_from_base = abs(grid_price - base_price) / base_price
        position_size = self.max_position_size * (1 - distance_from_base)

        # Calculate target (next grid level up)
        grid_prices = self.grid_prices[symbol]
        if level + 1 < len(grid_prices):
            target_price = grid_prices[level + 1]
        else:
            target_price = grid_price * (1 + self.profit_per_grid)

        # Calculate stop loss (next grid level down or 2% below)
        if level > 0:
            stop_loss = grid_prices[level - 1]
        else:
            stop_loss = grid_price * 0.98

        # Track position
        if symbol not in self.grid_positions:
            self.grid_positions[symbol] = {}

        self.grid_positions[symbol][level] = {
            'action': 'buy',
            'entry_price': current_price,
            'grid_price': grid_price,
            'level': level,
            'active': True,
            'target': target_price,
            'stop_loss': stop_loss
        }

        # Calculate confidence (higher near base)
        confidence = 1.0 - (distance_from_base / self.price_range_pct)
        confidence = max(0.5, min(confidence, 0.95))

        return SignalIntent(
            symbol=symbol,
            action='buy',
            confidence=confidence,
            metadata={
                'strategy': self.name,
                'grid_level': level,
                'grid_price': grid_price,
                'target': target_price,
                'stop_loss': stop_loss,
                'position_size': position_size,
                'grid_type': 'accumulation'
            }
        )

    def _generate_sell_signal(
        self,
        symbol: str,
        current_price: float,
        level: int,
        grid_price: float
    ) -> Optional[SignalIntent]:
        """Generate sell signal at grid level."""
        # Check if already have position at this level
        if symbol in self.grid_positions:
            if level in self.grid_positions[symbol]:
                position = self.grid_positions[symbol][level]
                if position['action'] == 'sell' and position['active']:
                    # Already have sell position at this level
                    return None

        # Calculate position size
        base_price = self.base_prices[symbol]
        distance_from_base = abs(grid_price - base_price) / base_price
        position_size = self.max_position_size * (1 - distance_from_base)

        # Calculate target (next grid level down)
        grid_prices = self.grid_prices[symbol]
        if level > 0:
            target_price = grid_prices[level - 1]
        else:
            target_price = grid_price * (1 - self.profit_per_grid)

        # Calculate stop loss (next grid level up or 2% above)
        if level + 1 < len(grid_prices):
            stop_loss = grid_prices[level + 1]
        else:
            stop_loss = grid_price * 1.02

        # Track position
        if symbol not in self.grid_positions:
            self.grid_positions[symbol] = {}

        self.grid_positions[symbol][level] = {
            'action': 'sell',
            'entry_price': current_price,
            'grid_price': grid_price,
            'level': level,
            'active': True,
            'target': target_price,
            'stop_loss': stop_loss
        }

        # Calculate confidence
        confidence = 1.0 - (distance_from_base / self.price_range_pct)
        confidence = max(0.5, min(confidence, 0.95))

        return SignalIntent(
            symbol=symbol,
            action='sell',
            confidence=confidence,
            metadata={
                'strategy': self.name,
                'grid_level': level,
                'grid_price': grid_price,
                'target': target_price,
                'stop_loss': stop_loss,
                'position_size': position_size,
                'grid_type': 'distribution'
            }
        )

    def _find_nearest_grid(
        self,
        grid_prices: List[float],
        current_price: float
    ) -> Tuple[int, float]:
        """Find nearest grid level to current price."""
        min_distance = float('inf')
        nearest_level = 0
        nearest_price = grid_prices[0]

        for i, price in enumerate(grid_prices):
            distance = abs(current_price - price)
            if distance < min_distance:
                min_distance = distance
                nearest_level = i
                nearest_price = price

        return nearest_level, nearest_price

    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility."""
        if len(prices) < 2:
            return 0.05  # Default 5%

        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]

        if not returns:
            return 0.05

        volatility = statistics.stdev(returns) if len(returns) > 1 else 0.05

        return volatility

    def check_exit_conditions(
        self,
        symbol: str,
        current_price: float
    ) -> Optional[Dict[str, Any]]:
        """Check if any grid positions should be exited."""
        if symbol not in self.grid_positions:
            return None

        for level, position in self.grid_positions[symbol].items():
            if not position['active']:
                continue

            # Check if target reached
            if position['action'] == 'buy':
                if current_price >= position['target']:
                    position['active'] = False
                    self.stats["profits_taken"] += 1

                    return {
                        'level': level,
                        'action': 'close',
                        'reason': 'target_reached',
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'profit': (current_price - position['entry_price']) / position['entry_price']
                    }

                # Check stop loss
                if current_price <= position['stop_loss']:
                    position['active'] = False

                    return {
                        'level': level,
                        'action': 'close',
                        'reason': 'stop_loss',
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'profit': (current_price - position['entry_price']) / position['entry_price']
                    }

            elif position['action'] == 'sell':
                if current_price <= position['target']:
                    position['active'] = False
                    self.stats["profits_taken"] += 1

                    return {
                        'level': level,
                        'action': 'close',
                        'reason': 'target_reached',
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'profit': (position['entry_price'] - current_price) / position['entry_price']
                    }

                # Check stop loss
                if current_price >= position['stop_loss']:
                    position['active'] = False

                    return {
                        'level': level,
                        'action': 'close',
                        'reason': 'stop_loss',
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'profit': (position['entry_price'] - current_price) / position['entry_price']
                    }

        return None

    def get_grid_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get grid configuration and status."""
        if symbol not in self.grid_prices:
            return None

        active_positions = sum(
            1 for pos in self.grid_positions.get(symbol, {}).values()
            if pos['active']
        )

        return {
            'symbol': symbol,
            'base_price': self.base_prices.get(symbol),
            'upper_bound': self.upper_bounds.get(symbol),
            'lower_bound': self.lower_bounds.get(symbol),
            'grid_levels': self.grid_levels,
            'grid_prices': self.grid_prices[symbol],
            'active_positions': active_positions,
            'total_positions': len(self.grid_positions.get(symbol, {}))
        }

    def get_active_positions(self, symbol: str) -> List[Dict[str, Any]]:
        """Get all active positions for a symbol."""
        if symbol not in self.grid_positions:
            return []

        return [
            {
                'level': level,
                'action': pos['action'],
                'entry_price': pos['entry_price'],
                'grid_price': pos['grid_price'],
                'target': pos['target'],
                'stop_loss': pos['stop_loss']
            }
            for level, pos in self.grid_positions[symbol].items()
            if pos['active']
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        stats = self.stats.copy()

        # Calculate additional metrics
        if stats["buy_signals"] + stats["sell_signals"] > 0:
            stats["buy_ratio"] = stats["buy_signals"] / (stats["buy_signals"] + stats["sell_signals"])
        else:
            stats["buy_ratio"] = 0.0

        return stats

    def reset(self):
        """Reset strategy state."""
        self.grid_prices.clear()
        self.grid_positions.clear()
        self.base_prices.clear()
        self.upper_bounds.clear()
        self.lower_bounds.clear()
        self.stats = {
            "signals_generated": 0,
            "grids_created": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "grids_rebalanced": 0,
            "profits_taken": 0
        }
