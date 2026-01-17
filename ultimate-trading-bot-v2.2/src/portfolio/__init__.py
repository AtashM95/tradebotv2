"""
Portfolio Management Package for Ultimate Trading Bot v2.2.

This package provides comprehensive portfolio management capabilities including:
- Portfolio tracking and performance analysis
- Position tracking with detailed P&L
- Portfolio rebalancing with multiple strategies
- Risk management and constraint checking
"""

import logging
from typing import Any

from .portfolio_manager import (
    PortfolioType,
    AssetClass,
    PortfolioConfig,
    Position,
    PortfolioSnapshot,
    PortfolioPerformance,
    PortfolioManager,
    create_portfolio_manager,
)
from .position_tracker import (
    PositionSide,
    PositionStatus,
    PositionEntry,
    PositionExit,
    TrackedPosition,
    PositionAnalytics,
    PositionTracker,
    create_position_tracker,
)
from .rebalancer import (
    RebalanceStrategy,
    RebalanceFrequency,
    RebalanceConfig,
    TargetAllocation,
    RebalanceOrder,
    RebalanceResult,
    PortfolioRebalancer,
    EqualWeightRebalancer,
    RiskParityRebalancer,
    create_rebalancer,
    create_equal_weight_rebalancer,
    create_risk_parity_rebalancer,
)


logger = logging.getLogger(__name__)


__all__ = [
    # Portfolio Manager
    "PortfolioType",
    "AssetClass",
    "PortfolioConfig",
    "Position",
    "PortfolioSnapshot",
    "PortfolioPerformance",
    "PortfolioManager",
    "create_portfolio_manager",
    # Position Tracker
    "PositionSide",
    "PositionStatus",
    "PositionEntry",
    "PositionExit",
    "TrackedPosition",
    "PositionAnalytics",
    "PositionTracker",
    "create_position_tracker",
    # Rebalancer
    "RebalanceStrategy",
    "RebalanceFrequency",
    "RebalanceConfig",
    "TargetAllocation",
    "RebalanceOrder",
    "RebalanceResult",
    "PortfolioRebalancer",
    "EqualWeightRebalancer",
    "RiskParityRebalancer",
    "create_rebalancer",
    "create_equal_weight_rebalancer",
    "create_risk_parity_rebalancer",
]


class PortfolioSystem:
    """
    Integrated portfolio management system.

    Combines portfolio management, position tracking, and rebalancing.
    """

    def __init__(
        self,
        portfolio_config: PortfolioConfig | None = None,
        rebalance_config: RebalanceConfig | None = None,
        track_positions: bool = True,
    ) -> None:
        """
        Initialize portfolio system.

        Args:
            portfolio_config: Portfolio configuration
            rebalance_config: Rebalancing configuration
            track_positions: Whether to track detailed positions
        """
        self.portfolio_config = portfolio_config or PortfolioConfig()
        self.rebalance_config = rebalance_config or RebalanceConfig()

        # Core components
        self._portfolio: PortfolioManager | None = None
        self._tracker: PositionTracker | None = None
        self._rebalancer: PortfolioRebalancer | None = None

        self._track_positions = track_positions
        self._initialized = False

        logger.info("PortfolioSystem created")

    async def initialize(self) -> None:
        """Initialize portfolio system."""
        try:
            # Initialize portfolio manager
            self._portfolio = create_portfolio_manager(self.portfolio_config)
            await self._portfolio.initialize()

            # Initialize position tracker
            if self._track_positions:
                self._tracker = create_position_tracker()

            # Initialize rebalancer
            self._rebalancer = create_rebalancer(
                self._portfolio,
                self.rebalance_config,
            )

            self._initialized = True
            logger.info("PortfolioSystem initialized")

        except Exception as e:
            logger.error(f"Failed to initialize PortfolioSystem: {e}")
            raise

    @property
    def portfolio(self) -> PortfolioManager:
        """Get portfolio manager."""
        if self._portfolio is None:
            raise RuntimeError("PortfolioSystem not initialized")
        return self._portfolio

    @property
    def tracker(self) -> PositionTracker | None:
        """Get position tracker."""
        return self._tracker

    @property
    def rebalancer(self) -> PortfolioRebalancer:
        """Get rebalancer."""
        if self._rebalancer is None:
            raise RuntimeError("PortfolioSystem not initialized")
        return self._rebalancer

    async def open_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        asset_class: AssetClass = AssetClass.EQUITY,
        sector: str | None = None,
        side: PositionSide = PositionSide.LONG,
        commission: float = 0.0,
    ) -> Position:
        """
        Open or add to a position.

        Args:
            symbol: Trading symbol
            quantity: Number of units
            price: Entry price
            asset_class: Asset class
            sector: Sector classification
            side: Position side
            commission: Commission paid

        Returns:
            Updated position
        """
        if not self._initialized:
            await self.initialize()

        # Add to portfolio
        position = await self.portfolio.add_position(
            symbol, quantity, price, asset_class, sector
        )

        # Track position
        if self._tracker:
            self._tracker.open_position(
                symbol, quantity, price, side, commission
            )

        return position

    async def close_position(
        self,
        symbol: str,
        quantity: float | None = None,
        price: float | None = None,
        commission: float = 0.0,
    ) -> float:
        """
        Close or reduce a position.

        Args:
            symbol: Trading symbol
            quantity: Number of units (None for full close)
            price: Exit price
            commission: Commission paid

        Returns:
            Realized P&L
        """
        if not self._initialized:
            await self.initialize()

        position = self.portfolio.get_position(symbol)
        if position is None:
            raise ValueError(f"Position not found: {symbol}")

        # Use position's current price if not provided
        if price is None:
            price = position.current_price

        # Use full quantity if not specified
        if quantity is None:
            quantity = position.quantity

        # Close in portfolio
        realized_pnl = await self.portfolio.reduce_position(
            symbol, quantity, price
        )

        # Track in tracker
        if self._tracker:
            self._tracker.close_position(
                symbol, quantity, price, commission
            )

        return realized_pnl

    async def update_prices(self, prices: dict[str, float]) -> None:
        """
        Update all positions with current prices.

        Args:
            prices: Dictionary of symbol to price
        """
        if not self._initialized:
            await self.initialize()

        await self.portfolio.update_prices(prices)

        if self._tracker:
            self._tracker.update_prices(prices)

    def set_target_allocation(self, target: TargetAllocation) -> None:
        """
        Set target portfolio allocation for rebalancing.

        Args:
            target: Target allocation
        """
        if self._rebalancer is None:
            raise RuntimeError("PortfolioSystem not initialized")

        self._rebalancer.set_target_allocation(target)

    async def check_rebalance(self, prices: dict[str, float]) -> RebalanceResult | None:
        """
        Check if rebalancing is needed and generate orders.

        Args:
            prices: Current prices

        Returns:
            Rebalance result if needed, None otherwise
        """
        if not self._initialized:
            await self.initialize()

        if self._rebalancer.needs_rebalance():
            return self._rebalancer.generate_orders(prices)

        return None

    async def get_performance(self) -> PortfolioPerformance:
        """Get portfolio performance metrics."""
        if not self._initialized:
            await self.initialize()

        return await self.portfolio.get_performance()

    async def get_snapshot(self) -> PortfolioSnapshot:
        """Get current portfolio snapshot."""
        if not self._initialized:
            await self.initialize()

        return await self.portfolio.get_snapshot()

    def get_position_analytics(self, symbol: str) -> PositionAnalytics | None:
        """Get analytics for a position."""
        if self._tracker is None:
            return None

        return self._tracker.get_position_analytics(symbol)

    def get_overall_stats(self) -> dict[str, Any]:
        """Get overall portfolio statistics."""
        stats = {}

        if self._portfolio:
            stats["portfolio"] = {
                "total_value": self._portfolio.total_value,
                "cash": self._portfolio.cash,
                "positions_value": self._portfolio.positions_value,
                "position_count": self._portfolio.position_count,
            }

        if self._tracker:
            stats["trading"] = self._tracker.get_overall_stats()

        if self._rebalancer:
            stats["rebalancing"] = {
                "last_rebalance": (
                    self._rebalancer.get_last_rebalance().isoformat()
                    if self._rebalancer.get_last_rebalance()
                    else None
                ),
                "max_drift": self._rebalancer.get_max_drift(),
                "needs_rebalance": self._rebalancer.needs_rebalance(),
            }

        return stats

    async def reset(self) -> None:
        """Reset portfolio system to initial state."""
        if self._portfolio:
            await self._portfolio.reset()

        if self._tracker:
            self._tracker.clear()

        logger.info("PortfolioSystem reset")


def create_portfolio_system(
    portfolio_config: PortfolioConfig | None = None,
    rebalance_config: RebalanceConfig | None = None,
    track_positions: bool = True,
) -> PortfolioSystem:
    """
    Create a portfolio system instance.

    Args:
        portfolio_config: Portfolio configuration
        rebalance_config: Rebalancing configuration
        track_positions: Whether to track detailed positions

    Returns:
        PortfolioSystem instance
    """
    return PortfolioSystem(
        portfolio_config=portfolio_config,
        rebalance_config=rebalance_config,
        track_positions=track_positions,
    )


# Module version
__version__ = "2.2.0"
