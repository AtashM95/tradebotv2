
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from .contracts import TradeFill, RunContext

logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """Position side indicators."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Position:
    """
    Represents an open trading position.

    Tracks entry, current state, P&L, and position metrics.
    """
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    realized_pnl: float
    total_pnl: float
    cost_basis: float
    opened_at: datetime
    updated_at: datetime
    filled_quantity: float
    avg_entry_price: float
    trade_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'market_value': self.market_value,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.total_pnl,
            'cost_basis': self.cost_basis,
            'opened_at': self.opened_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'filled_quantity': self.filled_quantity,
            'avg_entry_price': self.avg_entry_price,
            'trade_count': self.trade_count,
            'metadata': self.metadata
        }

    def is_long(self) -> bool:
        """Check if position is long."""
        return self.side == PositionSide.LONG

    def is_short(self) -> bool:
        """Check if position is short."""
        return self.side == PositionSide.SHORT

    def is_profitable(self) -> bool:
        """Check if position is currently profitable."""
        return self.unrealized_pnl > 0


class PositionManagerMetrics:
    """Tracks position manager performance metrics."""

    def __init__(self) -> None:
        self.total_positions_opened: int = 0
        self.total_positions_closed: int = 0
        self.positions_profitable: int = 0
        self.positions_losing: int = 0
        self.total_realized_pnl: float = 0.0
        self.total_unrealized_pnl: float = 0.0
        self.largest_winner: float = 0.0
        self.largest_loser: float = 0.0
        self.avg_winner: float = 0.0
        self.avg_loser: float = 0.0
        self.win_rate: float = 0.0
        self._winners: List[float] = []
        self._losers: List[float] = []

    def record_close(self, pnl: float) -> None:
        """Record a position close."""
        self.total_positions_closed += 1
        self.total_realized_pnl += pnl

        if pnl > 0:
            self.positions_profitable += 1
            self._winners.append(pnl)
            self.largest_winner = max(self.largest_winner, pnl)
            self.avg_winner = sum(self._winners) / len(self._winners)
        else:
            self.positions_losing += 1
            self._losers.append(pnl)
            self.largest_loser = min(self.largest_loser, pnl)
            if self._losers:
                self.avg_loser = sum(self._losers) / len(self._losers)

        if self.total_positions_closed > 0:
            self.win_rate = self.positions_profitable / self.total_positions_closed

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'total_positions_opened': self.total_positions_opened,
            'total_positions_closed': self.total_positions_closed,
            'positions_profitable': self.positions_profitable,
            'positions_losing': self.positions_losing,
            'win_rate': self.win_rate,
            'total_realized_pnl': self.total_realized_pnl,
            'total_unrealized_pnl': self.total_unrealized_pnl,
            'largest_winner': self.largest_winner,
            'largest_loser': self.largest_loser,
            'avg_winner': self.avg_winner,
            'avg_loser': self.avg_loser,
            'profit_factor': self._calculate_profit_factor()
        }

    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (total wins / total losses)."""
        total_wins = sum(self._winners) if self._winners else 0
        total_losses = abs(sum(self._losers)) if self._losers else 0
        if total_losses == 0:
            return float('inf') if total_wins > 0 else 0.0
        return total_wins / total_losses


class PositionManager:
    """
    Manages trading positions and tracks P&L.

    Responsibilities:
    - Track open positions
    - Calculate unrealized and realized P&L
    - Update positions with market prices
    - Handle position opens, increases, decreases, and closes
    - Provide position analytics and reporting
    - Track position history
    """

    def __init__(self, enable_short_selling: bool = False) -> None:
        """
        Initialize the position manager.

        Args:
            enable_short_selling: Whether to allow short positions
        """
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.enable_short_selling = enable_short_selling
        self.metrics = PositionManagerMetrics()

        logger.info("PositionManager initialized", extra={
            'enable_short_selling': enable_short_selling
        })

    def process_fill(
        self,
        fill: TradeFill,
        context: Optional[RunContext] = None
    ) -> None:
        """
        Process a trade fill and update positions.

        Args:
            fill: Trade fill to process
            context: Optional run context
        """
        symbol = fill.symbol
        action = fill.action.lower()

        log_extra = {
            'symbol': symbol,
            'action': action,
            'quantity': fill.quantity,
            'price': fill.price
        }
        if context:
            log_extra['run_id'] = context.run_id

        if action in ('buy', 'cover'):
            self._process_long_entry(fill, context)
        elif action in ('sell', 'short'):
            self._process_long_exit_or_short_entry(fill, context)
        else:
            logger.warning(f"Unknown action: {action}", extra=log_extra)

    def _process_long_entry(
        self,
        fill: TradeFill,
        context: Optional[RunContext] = None
    ) -> None:
        """Process a long entry (buy) or short cover."""
        symbol = fill.symbol

        if symbol in self.positions:
            position = self.positions[symbol]

            # If covering a short position
            if position.is_short():
                self._reduce_or_close_position(fill, context)
            else:
                # Increasing long position
                self._increase_position(fill, context)
        else:
            # Opening new long position
            self._open_position(fill, PositionSide.LONG, context)

    def _process_long_exit_or_short_entry(
        self,
        fill: TradeFill,
        context: Optional[RunContext] = None
    ) -> None:
        """Process a long exit (sell) or short entry."""
        symbol = fill.symbol
        action = fill.action.lower()

        if symbol in self.positions:
            position = self.positions[symbol]

            # If selling long position
            if position.is_long():
                self._reduce_or_close_position(fill, context)
            elif action == 'short' and self.enable_short_selling:
                # Increasing short position
                self._increase_position(fill, context)
            else:
                logger.warning(f"Invalid action {action} for position side {position.side}")
        else:
            # Opening new short position
            if action == 'short' and self.enable_short_selling:
                self._open_position(fill, PositionSide.SHORT, context)
            else:
                logger.warning(f"Cannot sell without position or short selling disabled")

    def _open_position(
        self,
        fill: TradeFill,
        side: PositionSide,
        context: Optional[RunContext] = None
    ) -> None:
        """Open a new position."""
        now = datetime.utcnow()

        position = Position(
            symbol=fill.symbol,
            side=side,
            quantity=fill.quantity,
            entry_price=fill.price,
            current_price=fill.price,
            market_value=fill.quantity * fill.price,
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0,
            realized_pnl=0.0,
            total_pnl=0.0,
            cost_basis=fill.quantity * fill.price,
            opened_at=now,
            updated_at=now,
            filled_quantity=fill.quantity,
            avg_entry_price=fill.price,
            trade_count=1,
            metadata=fill.metadata.copy()
        )

        self.positions[fill.symbol] = position
        self.metrics.total_positions_opened += 1

        log_extra = {
            'symbol': fill.symbol,
            'side': side.value,
            'quantity': fill.quantity,
            'price': fill.price
        }
        if context:
            log_extra['run_id'] = context.run_id

        logger.info("Position opened", extra=log_extra)

    def _increase_position(
        self,
        fill: TradeFill,
        context: Optional[RunContext] = None
    ) -> None:
        """Increase an existing position."""
        position = self.positions[fill.symbol]

        # Calculate new average entry price
        total_qty = position.quantity + fill.quantity
        total_cost = (position.quantity * position.avg_entry_price +
                     fill.quantity * fill.price)
        new_avg_price = total_cost / total_qty

        position.quantity = total_qty
        position.avg_entry_price = new_avg_price
        position.cost_basis = total_cost
        position.filled_quantity += fill.quantity
        position.trade_count += 1
        position.updated_at = datetime.utcnow()

        # Recalculate P&L with current price
        self._update_position_pnl(position, position.current_price)

        log_extra = {
            'symbol': fill.symbol,
            'quantity': fill.quantity,
            'new_total': position.quantity,
            'new_avg_price': new_avg_price
        }
        if context:
            log_extra['run_id'] = context.run_id

        logger.info("Position increased", extra=log_extra)

    def _reduce_or_close_position(
        self,
        fill: TradeFill,
        context: Optional[RunContext] = None
    ) -> None:
        """Reduce or close an existing position."""
        position = self.positions[fill.symbol]

        # Calculate realized P&L for this partial close
        if position.is_long():
            realized_pnl = fill.quantity * (fill.price - position.avg_entry_price)
        else:
            realized_pnl = fill.quantity * (position.avg_entry_price - fill.price)

        position.realized_pnl += realized_pnl
        position.trade_count += 1

        # Check if fully closing
        if abs(fill.quantity - position.quantity) < 0.0001:
            self._close_position(fill.symbol, realized_pnl, context)
        else:
            # Partial close
            position.quantity -= fill.quantity
            position.cost_basis = position.quantity * position.avg_entry_price
            position.updated_at = datetime.utcnow()

            # Recalculate P&L
            self._update_position_pnl(position, position.current_price)

            log_extra = {
                'symbol': fill.symbol,
                'quantity_closed': fill.quantity,
                'remaining_quantity': position.quantity,
                'realized_pnl': realized_pnl
            }
            if context:
                log_extra['run_id'] = context.run_id

            logger.info("Position reduced", extra=log_extra)

    def _close_position(
        self,
        symbol: str,
        realized_pnl: float,
        context: Optional[RunContext] = None
    ) -> None:
        """Close a position completely."""
        position = self.positions[symbol]
        position.total_pnl = position.realized_pnl + position.unrealized_pnl

        # Move to closed positions
        self.closed_positions.append(position)
        del self.positions[symbol]

        # Update metrics
        self.metrics.record_close(position.total_pnl)

        log_extra = {
            'symbol': symbol,
            'total_pnl': position.total_pnl,
            'realized_pnl': position.realized_pnl,
            'duration_seconds': (position.updated_at - position.opened_at).total_seconds()
        }
        if context:
            log_extra['run_id'] = context.run_id

        logger.info("Position closed", extra=log_extra)

    def update_prices(
        self,
        prices: Dict[str, float],
        context: Optional[RunContext] = None
    ) -> None:
        """
        Update current prices for all positions.

        Args:
            prices: Dictionary of symbol -> current price
            context: Optional run context
        """
        for symbol, position in self.positions.items():
            if symbol in prices:
                self._update_position_pnl(position, prices[symbol])

        # Update total unrealized P&L metric
        self.metrics.total_unrealized_pnl = sum(
            p.unrealized_pnl for p in self.positions.values()
        )

    def _update_position_pnl(self, position: Position, current_price: float) -> None:
        """Update position P&L calculations."""
        position.current_price = current_price
        position.market_value = position.quantity * current_price
        position.updated_at = datetime.utcnow()

        if position.is_long():
            position.unrealized_pnl = position.quantity * (current_price - position.avg_entry_price)
        else:
            position.unrealized_pnl = position.quantity * (position.avg_entry_price - current_price)

        if position.cost_basis != 0:
            position.unrealized_pnl_pct = (position.unrealized_pnl / position.cost_basis) * 100
        else:
            position.unrealized_pnl_pct = 0.0

        position.total_pnl = position.realized_pnl + position.unrealized_pnl

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """Check if there's an open position for a symbol."""
        return symbol in self.positions

    def get_all_positions(self) -> List[Position]:
        """Get all open positions."""
        return list(self.positions.values())

    def get_long_positions(self) -> List[Position]:
        """Get all long positions."""
        return [p for p in self.positions.values() if p.is_long()]

    def get_short_positions(self) -> List[Position]:
        """Get all short positions."""
        return [p for p in self.positions.values() if p.is_short()]

    def get_total_market_value(self) -> float:
        """Get total market value of all positions."""
        return sum(p.market_value for p in self.positions.values())

    def get_total_unrealized_pnl(self) -> float:
        """Get total unrealized P&L across all positions."""
        return sum(p.unrealized_pnl for p in self.positions.values())

    def get_total_realized_pnl(self) -> float:
        """Get total realized P&L from closed positions."""
        return self.metrics.total_realized_pnl

    def get_total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)."""
        return self.get_total_realized_pnl() + self.get_total_unrealized_pnl()

    def get_position_count(self) -> int:
        """Get number of open positions."""
        return len(self.positions)

    def get_metrics(self) -> Dict[str, Any]:
        """Get position manager metrics."""
        metrics = self.metrics.to_dict()
        metrics['open_positions'] = len(self.positions)
        metrics['total_market_value'] = self.get_total_market_value()
        metrics['total_pnl'] = self.get_total_pnl()
        return metrics

    def get_closed_positions(self, limit: Optional[int] = None) -> List[Position]:
        """Get closed positions history."""
        if limit:
            return self.closed_positions[-limit:]
        return self.closed_positions

    def close_all_positions(
        self,
        current_prices: Dict[str, float],
        context: Optional[RunContext] = None
    ) -> int:
        """
        Close all open positions at current prices.

        Args:
            current_prices: Dictionary of symbol -> current price
            context: Optional run context

        Returns:
            Number of positions closed
        """
        count = 0
        symbols_to_close = list(self.positions.keys())

        for symbol in symbols_to_close:
            if symbol in current_prices:
                position = self.positions[symbol]

                # Calculate final realized P&L
                if position.is_long():
                    realized_pnl = position.quantity * (current_prices[symbol] - position.avg_entry_price)
                else:
                    realized_pnl = position.quantity * (position.avg_entry_price - current_prices[symbol])

                position.realized_pnl += realized_pnl
                self._close_position(symbol, realized_pnl, context)
                count += 1

        logger.info(f"Closed {count} positions", extra={'count': count})
        return count

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on position manager.

        Returns:
            Health status dictionary
        """
        return {
            'healthy': True,
            'open_positions': len(self.positions),
            'total_market_value': self.get_total_market_value(),
            'total_unrealized_pnl': self.get_total_unrealized_pnl(),
            'metrics': self.metrics.to_dict()
        }
