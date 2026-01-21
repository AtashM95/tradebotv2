"""
Scalping Strategy - High-frequency trading for small, quick profits.
~400 lines as per schema
"""

import logging
from typing import Dict, Any, List, Optional
from collections import deque
import statistics

from ..core.contracts import MarketSnapshot, SignalIntent
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class ScalpingStrategy(BaseStrategy):
    """
    Scalping strategy for quick, small profits from market inefficiencies.

    Algorithm:
    1. Monitor order book and tick data
    2. Detect short-term price momentum
    3. Enter on favorable spread conditions
    4. Exit quickly with small profit target
    5. Use tight stops to minimize losses

    Features:
    - Bid-ask spread analysis
    - Order book imbalance detection
    - Microstructure alpha signals
    - Tick-by-tick momentum
    - Fast execution optimization
    - Time-based exits
    - Liquidity assessment
    """

    name = 'scalping'

    def __init__(
        self,
        profit_target_pct: float = 0.001,  # 0.1% profit target
        stop_loss_pct: float = 0.0005,  # 0.05% stop loss
        max_holding_seconds: int = 60,  # 1 minute max hold
        min_spread_ratio: float = 0.3,  # Min spread quality
        tick_window: int = 50,  # Ticks to analyze
        min_volume_ratio: float = 1.2,  # Volume must be 1.2x average
        order_book_depth: int = 10  # Order book levels to analyze
    ):
        """
        Initialize scalping strategy.

        Args:
            profit_target_pct: Target profit percentage
            stop_loss_pct: Stop loss percentage
            max_holding_seconds: Maximum holding period in seconds
            min_spread_ratio: Minimum acceptable spread ratio
            tick_window: Number of ticks to analyze
            min_volume_ratio: Minimum volume ratio for entry
            order_book_depth: Order book depth to analyze
        """
        super().__init__()
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_seconds = max_holding_seconds
        self.min_spread_ratio = min_spread_ratio
        self.tick_window = tick_window
        self.min_volume_ratio = min_volume_ratio
        self.order_book_depth = order_book_depth

        # Track tick data
        self.tick_prices = {}  # symbol -> deque of prices
        self.tick_volumes = {}  # symbol -> deque of volumes
        self.tick_timestamps = {}  # symbol -> deque of timestamps

        # Track positions
        self.open_positions = {}  # symbol -> position_info
        self.entry_times = {}  # symbol -> entry_timestamp

        # Statistics
        self.stats = {
            "signals_generated": 0,
            "scalps_completed": 0,
            "profitable_scalps": 0,
            "total_profit_pct": 0.0,
            "avg_holding_time": 0.0,
            "timeouts": 0,
            "stopped_out": 0
        }

    def generate(self, snapshot: MarketSnapshot) -> SignalIntent | None:
        """
        Generate trading signal based on scalping logic.

        Args:
            snapshot: Market snapshot

        Returns:
            SignalIntent or None
        """
        symbol = snapshot.symbol
        current_price = snapshot.price
        metadata = snapshot.metadata or {}

        # Update tick data
        self._update_tick_data(symbol, snapshot)

        # Check if we have enough data
        if symbol not in self.tick_prices or len(self.tick_prices[symbol]) < self.tick_window:
            return None

        # Check for position timeout
        if symbol in self.open_positions:
            if self._check_position_timeout(symbol, metadata.get('timestamp', 0)):
                return self._generate_exit_signal(symbol, current_price, 'timeout')

        # Analyze spread conditions
        spread_signal = self._analyze_spread(symbol, metadata)
        if not spread_signal:
            return None

        # Analyze tick momentum
        momentum = self._calculate_tick_momentum(symbol)

        # Analyze order book imbalance
        imbalance = self._analyze_order_book_imbalance(metadata)

        # Check volume
        volume_confirmed = self._check_volume(symbol, metadata)

        # Generate signal if conditions align
        if abs(momentum) > 0.0002 and volume_confirmed:  # 0.02% momentum threshold
            signal = self._generate_scalp_signal(
                symbol,
                current_price,
                momentum,
                imbalance,
                spread_signal,
                metadata
            )

            if signal:
                self.stats["signals_generated"] += 1

            return signal

        return None

    def _update_tick_data(self, symbol: str, snapshot: MarketSnapshot):
        """Update tick-by-tick data."""
        if symbol not in self.tick_prices:
            self.tick_prices[symbol] = deque(maxlen=self.tick_window)
            self.tick_volumes[symbol] = deque(maxlen=self.tick_window)
            self.tick_timestamps[symbol] = deque(maxlen=self.tick_window)

        self.tick_prices[symbol].append(snapshot.price)

        metadata = snapshot.metadata or {}
        volume = metadata.get('volume', metadata.get('tick_volume', 0))
        timestamp = metadata.get('timestamp', 0)

        self.tick_volumes[symbol].append(volume)
        self.tick_timestamps[symbol].append(timestamp)

    def _analyze_spread(
        self,
        symbol: str,
        metadata: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Analyze bid-ask spread quality."""
        bid = metadata.get('bid', 0)
        ask = metadata.get('ask', 0)

        if bid <= 0 or ask <= 0:
            return None

        mid_price = (bid + ask) / 2
        spread = ask - bid
        spread_pct = spread / mid_price if mid_price > 0 else 0

        # Spread should be reasonable for scalping
        if spread_pct > 0.003:  # > 0.3% is too wide
            return None

        # Calculate spread quality
        avg_price = statistics.mean(self.tick_prices[symbol])
        spread_ratio = spread / (avg_price * 0.001) if avg_price > 0 else 0

        if spread_ratio > self.min_spread_ratio:
            return None  # Spread too wide relative to volatility

        return {
            'bid': bid,
            'ask': ask,
            'mid': mid_price,
            'spread': spread,
            'spread_pct': spread_pct,
            'spread_quality': 1.0 - min(spread_ratio, 1.0)
        }

    def _calculate_tick_momentum(self, symbol: str) -> float:
        """Calculate momentum from recent ticks."""
        prices = list(self.tick_prices[symbol])

        if len(prices) < 5:
            return 0.0

        # Calculate short-term vs medium-term momentum
        recent_prices = prices[-5:]
        earlier_prices = prices[-20:-5] if len(prices) >= 20 else prices[:-5]

        recent_avg = statistics.mean(recent_prices)
        earlier_avg = statistics.mean(earlier_prices) if earlier_prices else recent_avg

        if earlier_avg == 0:
            return 0.0

        momentum = (recent_avg - earlier_avg) / earlier_avg

        return momentum

    def _analyze_order_book_imbalance(
        self,
        metadata: Dict[str, Any]
    ) -> float:
        """Analyze order book imbalance."""
        order_book = metadata.get('order_book', {})

        if not order_book:
            return 0.0

        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])

        if not bids or not asks:
            return 0.0

        # Calculate volume at top N levels
        depth = min(self.order_book_depth, len(bids), len(asks))

        bid_volume = sum(bid[1] for bid in bids[:depth])
        ask_volume = sum(ask[1] for ask in asks[:depth])

        total_volume = bid_volume + ask_volume

        if total_volume == 0:
            return 0.0

        # Positive imbalance = more bids (bullish)
        # Negative imbalance = more asks (bearish)
        imbalance = (bid_volume - ask_volume) / total_volume

        return imbalance

    def _check_volume(self, symbol: str, metadata: Dict[str, Any]) -> bool:
        """Check if volume supports entry."""
        volumes = list(self.tick_volumes[symbol])

        if len(volumes) < 10:
            return False

        current_volume = volumes[-1]
        avg_volume = statistics.mean(volumes[:-1])

        if avg_volume == 0:
            return False

        volume_ratio = current_volume / avg_volume

        return volume_ratio >= self.min_volume_ratio

    def _generate_scalp_signal(
        self,
        symbol: str,
        current_price: float,
        momentum: float,
        imbalance: float,
        spread_data: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Optional[SignalIntent]:
        """Generate scalp entry signal."""
        # Determine direction
        if momentum > 0 and imbalance > 0.1:
            action = 'buy'
            entry_price = spread_data['ask']
            exit_price = entry_price * (1 + self.profit_target_pct)
            stop_price = entry_price * (1 - self.stop_loss_pct)
        elif momentum < 0 and imbalance < -0.1:
            action = 'sell'
            entry_price = spread_data['bid']
            exit_price = entry_price * (1 - self.profit_target_pct)
            stop_price = entry_price * (1 + self.stop_loss_pct)
        else:
            return None

        # Calculate confidence
        confidence = self._calculate_confidence(
            abs(momentum),
            abs(imbalance),
            spread_data['spread_quality']
        )

        # Track position
        self.open_positions[symbol] = {
            'action': action,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'stop_price': stop_price
        }
        self.entry_times[symbol] = metadata.get('timestamp', 0)

        return SignalIntent(
            symbol=symbol,
            action=action,
            confidence=confidence,
            metadata={
                'strategy': self.name,
                'entry_price': entry_price,
                'profit_target': exit_price,
                'stop_loss': stop_price,
                'momentum': momentum,
                'order_book_imbalance': imbalance,
                'spread_quality': spread_data['spread_quality'],
                'max_holding_seconds': self.max_holding_seconds,
                'scalp_type': 'momentum' if abs(momentum) > abs(imbalance) * 0.01 else 'imbalance'
            }
        )

    def _generate_exit_signal(
        self,
        symbol: str,
        current_price: float,
        reason: str
    ) -> Optional[SignalIntent]:
        """Generate exit signal for open position."""
        if symbol not in self.open_positions:
            return None

        position = self.open_positions[symbol]
        entry_price = position['entry_price']

        # Calculate profit
        if position['action'] == 'buy':
            profit_pct = (current_price - entry_price) / entry_price
            exit_action = 'sell'
        else:
            profit_pct = (entry_price - current_price) / entry_price
            exit_action = 'buy'

        # Update statistics
        self.stats["scalps_completed"] += 1
        if profit_pct > 0:
            self.stats["profitable_scalps"] += 1
        self.stats["total_profit_pct"] += profit_pct

        if reason == 'timeout':
            self.stats["timeouts"] += 1
        elif reason == 'stop_loss':
            self.stats["stopped_out"] += 1

        # Clean up
        del self.open_positions[symbol]
        if symbol in self.entry_times:
            del self.entry_times[symbol]

        return SignalIntent(
            symbol=symbol,
            action=exit_action,
            confidence=0.9,
            metadata={
                'strategy': self.name,
                'exit_reason': reason,
                'profit_pct': profit_pct,
                'entry_price': entry_price,
                'exit_price': current_price
            }
        )

    def _check_position_timeout(self, symbol: str, current_timestamp: float) -> bool:
        """Check if position has exceeded max holding time."""
        if symbol not in self.entry_times:
            return False

        entry_time = self.entry_times[symbol]
        holding_time = current_timestamp - entry_time

        # Convert to seconds if in milliseconds
        if holding_time > 1000000:
            holding_time = holding_time / 1000

        return holding_time > self.max_holding_seconds

    def _calculate_confidence(
        self,
        momentum: float,
        imbalance: float,
        spread_quality: float
    ) -> float:
        """Calculate signal confidence."""
        confidence = 0.6  # Base confidence

        # Momentum contribution (max 0.15)
        momentum_score = min(momentum / 0.001, 1.0)  # Normalize to 0.1%
        confidence += momentum_score * 0.15

        # Imbalance contribution (max 0.15)
        imbalance_score = min(imbalance / 0.3, 1.0)  # Normalize to 30%
        confidence += imbalance_score * 0.15

        # Spread quality contribution (max 0.10)
        confidence += spread_quality * 0.10

        return min(confidence, 0.95)

    def check_exit_conditions(
        self,
        symbol: str,
        current_price: float,
        metadata: Dict[str, Any]
    ) -> Optional[SignalIntent]:
        """Check if exit conditions are met."""
        if symbol not in self.open_positions:
            return None

        position = self.open_positions[symbol]

        # Check profit target
        if position['action'] == 'buy':
            if current_price >= position['exit_price']:
                return self._generate_exit_signal(symbol, current_price, 'profit_target')
            if current_price <= position['stop_price']:
                return self._generate_exit_signal(symbol, current_price, 'stop_loss')
        else:
            if current_price <= position['exit_price']:
                return self._generate_exit_signal(symbol, current_price, 'profit_target')
            if current_price >= position['stop_price']:
                return self._generate_exit_signal(symbol, current_price, 'stop_loss')

        # Check timeout
        if self._check_position_timeout(symbol, metadata.get('timestamp', 0)):
            return self._generate_exit_signal(symbol, current_price, 'timeout')

        return None

    def get_open_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get open position for symbol."""
        return self.open_positions.get(symbol)

    def get_stats(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        stats = self.stats.copy()

        # Calculate win rate
        if stats["scalps_completed"] > 0:
            stats["win_rate"] = stats["profitable_scalps"] / stats["scalps_completed"]
            stats["avg_profit_per_scalp"] = stats["total_profit_pct"] / stats["scalps_completed"]
        else:
            stats["win_rate"] = 0.0
            stats["avg_profit_per_scalp"] = 0.0

        # Calculate timeout rate
        if stats["scalps_completed"] > 0:
            stats["timeout_rate"] = stats["timeouts"] / stats["scalps_completed"]
            stats["stop_out_rate"] = stats["stopped_out"] / stats["scalps_completed"]
        else:
            stats["timeout_rate"] = 0.0
            stats["stop_out_rate"] = 0.0

        return stats

    def reset(self):
        """Reset strategy state."""
        self.tick_prices.clear()
        self.tick_volumes.clear()
        self.tick_timestamps.clear()
        self.open_positions.clear()
        self.entry_times.clear()
        self.stats = {
            "signals_generated": 0,
            "scalps_completed": 0,
            "profitable_scalps": 0,
            "total_profit_pct": 0.0,
            "avg_holding_time": 0.0,
            "timeouts": 0,
            "stopped_out": 0
        }
