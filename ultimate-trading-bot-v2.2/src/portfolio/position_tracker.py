"""
Position Tracker for Ultimate Trading Bot v2.2.

Provides detailed position tracking, P&L calculation, and position analytics.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PositionSide(str, Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"


class PositionStatus(str, Enum):
    """Position status."""
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"


@dataclass
class PositionEntry:
    """A single position entry (fill)."""

    timestamp: datetime
    quantity: float
    price: float
    commission: float = 0.0
    side: PositionSide = PositionSide.LONG

    @property
    def cost(self) -> float:
        """Get total cost including commission."""
        return self.quantity * self.price + self.commission

    @property
    def avg_price_with_commission(self) -> float:
        """Get average price including commission."""
        if self.quantity > 0:
            return self.cost / self.quantity
        return self.price


@dataclass
class PositionExit:
    """A single position exit (sell/cover)."""

    timestamp: datetime
    quantity: float
    price: float
    commission: float = 0.0
    realized_pnl: float = 0.0

    @property
    def proceeds(self) -> float:
        """Get net proceeds after commission."""
        return self.quantity * self.price - self.commission


@dataclass
class TrackedPosition:
    """A fully tracked position with history."""

    symbol: str
    side: PositionSide = PositionSide.LONG
    status: PositionStatus = PositionStatus.OPEN

    # Current state
    quantity: float = 0.0
    avg_cost: float = 0.0
    current_price: float = 0.0

    # History
    entries: list[PositionEntry] = field(default_factory=list)
    exits: list[PositionExit] = field(default_factory=list)

    # P&L
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_commission: float = 0.0

    # Timing
    open_date: datetime | None = None
    close_date: datetime | None = None
    holding_period_days: int = 0

    # Risk metrics
    max_gain: float = 0.0
    max_loss: float = 0.0
    max_drawdown: float = 0.0

    # Price history for analytics
    price_history: list[tuple[datetime, float]] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def market_value(self) -> float:
        """Get current market value."""
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        """Get total cost basis."""
        return self.quantity * self.avg_cost

    @property
    def total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def return_pct(self) -> float:
        """Get return percentage."""
        if self.cost_basis > 0:
            return self.total_pnl / self.cost_basis
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "status": self.status.value,
            "quantity": self.quantity,
            "avg_cost": self.avg_cost,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "cost_basis": self.cost_basis,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.total_pnl,
            "return_pct": self.return_pct,
            "holding_period_days": self.holding_period_days,
            "entry_count": len(self.entries),
            "exit_count": len(self.exits),
        }


@dataclass
class PositionAnalytics:
    """Analytics for a position."""

    symbol: str

    # Performance
    total_return: float = 0.0
    annualized_return: float = 0.0
    daily_returns: list[float] = field(default_factory=list)

    # Risk
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0

    # Trading metrics
    avg_holding_period: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0

    # Efficiency
    mae: float = 0.0  # Maximum adverse excursion
    mfe: float = 0.0  # Maximum favorable excursion
    edge_ratio: float = 0.0  # MFE / MAE


class PositionTracker:
    """
    Comprehensive position tracking system.

    Tracks entries, exits, P&L, and provides analytics for each position.
    """

    def __init__(
        self,
        track_price_history: bool = True,
        max_history_days: int = 365,
    ) -> None:
        """
        Initialize position tracker.

        Args:
            track_price_history: Whether to track price history
            max_history_days: Maximum days of history to keep
        """
        self.track_price_history = track_price_history
        self.max_history_days = max_history_days

        # Active positions
        self._positions: dict[str, TrackedPosition] = {}

        # Closed positions history
        self._closed_positions: list[TrackedPosition] = []

        # Overall stats
        self._total_trades = 0
        self._winning_trades = 0
        self._losing_trades = 0
        self._total_realized_pnl = 0.0

        logger.info("PositionTracker initialized")

    def open_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: PositionSide = PositionSide.LONG,
        commission: float = 0.0,
        timestamp: datetime | None = None,
    ) -> TrackedPosition:
        """
        Open a new position or add to existing.

        Args:
            symbol: Trading symbol
            quantity: Number of units
            price: Entry price
            side: Position side
            commission: Commission paid
            timestamp: Entry timestamp

        Returns:
            Tracked position
        """
        timestamp = timestamp or datetime.now()

        entry = PositionEntry(
            timestamp=timestamp,
            quantity=quantity,
            price=price,
            commission=commission,
            side=side,
        )

        if symbol in self._positions:
            position = self._positions[symbol]

            # Update average cost
            total_cost = (
                position.quantity * position.avg_cost +
                quantity * price + commission
            )
            total_quantity = position.quantity + quantity
            position.avg_cost = total_cost / total_quantity if total_quantity > 0 else 0.0
            position.quantity = total_quantity
            position.total_commission += commission
            position.entries.append(entry)

        else:
            position = TrackedPosition(
                symbol=symbol,
                side=side,
                status=PositionStatus.OPEN,
                quantity=quantity,
                avg_cost=(price * quantity + commission) / quantity if quantity > 0 else price,
                current_price=price,
                entries=[entry],
                total_commission=commission,
                open_date=timestamp,
            )
            self._positions[symbol] = position

        logger.info(
            f"Opened position: {quantity} {symbol} @ ${price:.2f} "
            f"(avg: ${position.avg_cost:.2f})"
        )

        return position

    def close_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        commission: float = 0.0,
        timestamp: datetime | None = None,
    ) -> tuple[TrackedPosition, float]:
        """
        Close or reduce a position.

        Args:
            symbol: Trading symbol
            quantity: Number of units to close
            price: Exit price
            commission: Commission paid
            timestamp: Exit timestamp

        Returns:
            Tuple of (position, realized_pnl)
        """
        if symbol not in self._positions:
            raise ValueError(f"Position not found: {symbol}")

        timestamp = timestamp or datetime.now()
        position = self._positions[symbol]

        if quantity > position.quantity:
            raise ValueError(
                f"Close quantity exceeds position: {quantity} > {position.quantity}"
            )

        # Calculate realized P&L
        if position.side == PositionSide.LONG:
            realized_pnl = (price - position.avg_cost) * quantity - commission
        else:  # SHORT
            realized_pnl = (position.avg_cost - price) * quantity - commission

        exit_record = PositionExit(
            timestamp=timestamp,
            quantity=quantity,
            price=price,
            commission=commission,
            realized_pnl=realized_pnl,
        )

        position.exits.append(exit_record)
        position.realized_pnl += realized_pnl
        position.total_commission += commission
        position.quantity -= quantity

        self._total_realized_pnl += realized_pnl

        if position.quantity <= 0:
            # Position fully closed
            position.status = PositionStatus.CLOSED
            position.close_date = timestamp
            position.quantity = 0

            if position.open_date:
                position.holding_period_days = (timestamp - position.open_date).days

            # Track win/loss
            self._total_trades += 1
            if position.realized_pnl > 0:
                self._winning_trades += 1
            elif position.realized_pnl < 0:
                self._losing_trades += 1

            # Move to closed positions
            self._closed_positions.append(position)
            del self._positions[symbol]

            logger.info(
                f"Closed position: {symbol} P&L: ${realized_pnl:,.2f} "
                f"(total: ${position.realized_pnl:,.2f})"
            )
        else:
            position.status = PositionStatus.PARTIAL
            logger.info(
                f"Reduced position: {quantity} {symbol} @ ${price:.2f} "
                f"P&L: ${realized_pnl:,.2f}"
            )

        return position, realized_pnl

    def update_price(
        self,
        symbol: str,
        price: float,
        timestamp: datetime | None = None,
    ) -> TrackedPosition | None:
        """
        Update position with current price.

        Args:
            symbol: Trading symbol
            price: Current price
            timestamp: Price timestamp

        Returns:
            Updated position or None
        """
        if symbol not in self._positions:
            return None

        timestamp = timestamp or datetime.now()
        position = self._positions[symbol]

        old_price = position.current_price
        position.current_price = price

        # Calculate unrealized P&L
        if position.side == PositionSide.LONG:
            position.unrealized_pnl = (price - position.avg_cost) * position.quantity
        else:  # SHORT
            position.unrealized_pnl = (position.avg_cost - price) * position.quantity

        # Track max gain/loss
        if position.unrealized_pnl > position.max_gain:
            position.max_gain = position.unrealized_pnl
        if position.unrealized_pnl < position.max_loss:
            position.max_loss = position.unrealized_pnl

        # Calculate drawdown from max gain
        if position.max_gain > 0:
            drawdown = (position.max_gain - position.unrealized_pnl) / position.max_gain
            if drawdown > position.max_drawdown:
                position.max_drawdown = drawdown

        # Track price history
        if self.track_price_history:
            position.price_history.append((timestamp, price))

            # Trim history
            cutoff = datetime.now() - timedelta(days=self.max_history_days)
            position.price_history = [
                (t, p) for t, p in position.price_history if t >= cutoff
            ]

        return position

    def update_prices(
        self,
        prices: dict[str, float],
        timestamp: datetime | None = None,
    ) -> None:
        """
        Update multiple positions with prices.

        Args:
            prices: Dictionary of symbol to price
            timestamp: Price timestamp
        """
        for symbol, price in prices.items():
            self.update_price(symbol, price, timestamp)

    def get_position(self, symbol: str) -> TrackedPosition | None:
        """Get a tracked position."""
        return self._positions.get(symbol)

    def get_all_positions(self) -> list[TrackedPosition]:
        """Get all open positions."""
        return list(self._positions.values())

    def get_closed_positions(
        self,
        symbol: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[TrackedPosition]:
        """
        Get closed positions with optional filters.

        Args:
            symbol: Filter by symbol
            start_date: Filter by start date
            end_date: Filter by end date

        Returns:
            List of closed positions
        """
        positions = self._closed_positions

        if symbol:
            positions = [p for p in positions if p.symbol == symbol]

        if start_date:
            positions = [
                p for p in positions
                if p.close_date and p.close_date >= start_date
            ]

        if end_date:
            positions = [
                p for p in positions
                if p.close_date and p.close_date <= end_date
            ]

        return positions

    def get_position_analytics(self, symbol: str) -> PositionAnalytics:
        """
        Get analytics for a position.

        Args:
            symbol: Trading symbol

        Returns:
            Position analytics
        """
        # Combine open and closed positions for symbol
        positions = [p for p in self._closed_positions if p.symbol == symbol]
        if symbol in self._positions:
            positions.append(self._positions[symbol])

        if not positions:
            return PositionAnalytics(symbol=symbol)

        # Calculate metrics
        total_return = sum(p.total_pnl for p in positions)
        total_cost = sum(
            sum(e.cost for e in p.entries) for p in positions
        )
        return_pct = total_return / total_cost if total_cost > 0 else 0.0

        # Calculate daily returns from price history
        daily_returns = []
        for position in positions:
            if position.price_history and len(position.price_history) > 1:
                prices = [p for _, p in position.price_history]
                for i in range(1, len(prices)):
                    if prices[i-1] > 0:
                        daily_returns.append((prices[i] - prices[i-1]) / prices[i-1])

        # Win/loss metrics
        closed = [p for p in positions if p.status == PositionStatus.CLOSED]
        wins = [p for p in closed if p.realized_pnl > 0]
        losses = [p for p in closed if p.realized_pnl < 0]

        win_rate = len(wins) / len(closed) if closed else 0.0
        avg_win = np.mean([p.realized_pnl for p in wins]) if wins else 0.0
        avg_loss = np.mean([p.realized_pnl for p in losses]) if losses else 0.0

        total_wins = sum(p.realized_pnl for p in wins)
        total_losses = abs(sum(p.realized_pnl for p in losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Holding period
        holding_periods = [
            p.holding_period_days for p in closed if p.holding_period_days > 0
        ]
        avg_holding = np.mean(holding_periods) if holding_periods else 0.0

        # Risk metrics
        volatility = 0.0
        sharpe = 0.0
        var_95 = 0.0
        max_dd = 0.0

        if daily_returns:
            returns = np.array(daily_returns)
            volatility = float(np.std(returns) * np.sqrt(252))
            if volatility > 0:
                sharpe = float(np.mean(returns) * np.sqrt(252) / volatility)
            var_95 = float(np.percentile(returns, 5))
            max_dd = max(p.max_drawdown for p in positions)

        # MAE/MFE
        mae = abs(min(p.max_loss for p in positions)) if positions else 0.0
        mfe = max(p.max_gain for p in positions) if positions else 0.0
        edge_ratio = mfe / mae if mae > 0 else 0.0

        return PositionAnalytics(
            symbol=symbol,
            total_return=return_pct,
            annualized_return=0.0,  # Would need more data
            daily_returns=daily_returns,
            volatility=volatility,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            var_95=var_95,
            avg_holding_period=float(avg_holding),
            win_rate=win_rate,
            avg_win=float(avg_win),
            avg_loss=float(avg_loss),
            profit_factor=profit_factor,
            mae=mae,
            mfe=mfe,
            edge_ratio=edge_ratio,
        )

    def get_overall_stats(self) -> dict[str, Any]:
        """Get overall tracking statistics."""
        open_positions = list(self._positions.values())
        closed_positions = self._closed_positions

        total_unrealized = sum(p.unrealized_pnl for p in open_positions)
        total_realized = sum(p.realized_pnl for p in closed_positions)

        return {
            "open_positions": len(open_positions),
            "closed_positions": len(closed_positions),
            "total_trades": self._total_trades,
            "winning_trades": self._winning_trades,
            "losing_trades": self._losing_trades,
            "win_rate": self._winning_trades / self._total_trades if self._total_trades > 0 else 0.0,
            "total_unrealized_pnl": total_unrealized,
            "total_realized_pnl": total_realized,
            "total_pnl": total_unrealized + total_realized,
        }

    def get_symbol_summary(self, symbol: str) -> dict[str, Any]:
        """Get summary for a specific symbol."""
        position = self._positions.get(symbol)
        closed = [p for p in self._closed_positions if p.symbol == symbol]

        return {
            "symbol": symbol,
            "has_open_position": position is not None,
            "current_quantity": position.quantity if position else 0,
            "current_value": position.market_value if position else 0,
            "unrealized_pnl": position.unrealized_pnl if position else 0,
            "closed_trades": len(closed),
            "total_realized_pnl": sum(p.realized_pnl for p in closed),
        }

    def clear(self) -> None:
        """Clear all tracking data."""
        self._positions.clear()
        self._closed_positions.clear()
        self._total_trades = 0
        self._winning_trades = 0
        self._losing_trades = 0
        self._total_realized_pnl = 0.0

        logger.info("Position tracker cleared")


def create_position_tracker(
    track_price_history: bool = True,
    max_history_days: int = 365,
) -> PositionTracker:
    """
    Create a position tracker instance.

    Args:
        track_price_history: Whether to track price history
        max_history_days: Maximum days of history

    Returns:
        Position tracker instance
    """
    return PositionTracker(
        track_price_history=track_price_history,
        max_history_days=max_history_days,
    )
