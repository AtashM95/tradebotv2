"""
Portfolio Rebalancer for Ultimate Trading Bot v2.2.

Provides portfolio rebalancing strategies including threshold-based,
calendar-based, and optimization-based rebalancing.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

from .portfolio_manager import PortfolioManager, Position, AssetClass

logger = logging.getLogger(__name__)


class RebalanceStrategy(str, Enum):
    """Rebalancing strategies."""
    THRESHOLD = "threshold"
    CALENDAR = "calendar"
    CONSTANT_MIX = "constant_mix"
    CPPI = "cppi"  # Constant Proportion Portfolio Insurance
    TACTICAL = "tactical"
    OPTIMIZATION = "optimization"


class RebalanceFrequency(str, Enum):
    """Rebalancing frequency."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


@dataclass
class RebalanceConfig:
    """Configuration for portfolio rebalancing."""

    strategy: RebalanceStrategy = RebalanceStrategy.THRESHOLD
    frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY

    # Threshold rebalancing
    threshold: float = 0.05  # 5% drift threshold
    band_width: float = 0.02  # 2% band around target

    # CPPI settings
    cppi_floor: float = 0.8  # 80% floor
    cppi_multiplier: float = 3.0

    # Constraints
    min_trade_value: float = 100.0
    max_turnover: float = 0.25  # 25% max turnover per rebalance
    consider_tax: bool = False
    consider_transaction_costs: bool = True

    # Transaction costs
    commission_rate: float = 0.001  # 0.1% commission
    spread_cost: float = 0.0005  # 0.05% bid-ask spread


@dataclass
class TargetAllocation:
    """Target portfolio allocation."""

    weights: dict[str, float]  # symbol -> target weight
    asset_class_weights: dict[str, float] = field(default_factory=dict)
    sector_weights: dict[str, float] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate that weights sum to approximately 1."""
        total = sum(self.weights.values())
        return 0.99 <= total <= 1.01


@dataclass
class RebalanceOrder:
    """Order generated for rebalancing."""

    symbol: str
    action: str  # "BUY" or "SELL"
    quantity: float
    target_value: float
    current_value: float
    value_change: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "action": self.action,
            "quantity": self.quantity,
            "target_value": self.target_value,
            "current_value": self.current_value,
            "value_change": self.value_change,
            "reason": self.reason,
        }


@dataclass
class RebalanceResult:
    """Result of a rebalancing operation."""

    timestamp: datetime
    strategy: RebalanceStrategy
    orders: list[RebalanceOrder]

    # Portfolio changes
    turnover: float = 0.0
    estimated_costs: float = 0.0
    drift_before: float = 0.0
    drift_after: float = 0.0

    # Validation
    success: bool = True
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "strategy": self.strategy.value,
            "orders": [o.to_dict() for o in self.orders],
            "turnover": self.turnover,
            "estimated_costs": self.estimated_costs,
            "drift_before": self.drift_before,
            "drift_after": self.drift_after,
            "success": self.success,
            "message": self.message,
        }


class PortfolioRebalancer:
    """
    Portfolio rebalancing system.

    Supports multiple rebalancing strategies and generates
    orders to bring portfolio back to target allocation.
    """

    def __init__(
        self,
        portfolio: PortfolioManager,
        config: RebalanceConfig | None = None,
    ) -> None:
        """
        Initialize portfolio rebalancer.

        Args:
            portfolio: Portfolio manager
            config: Rebalancing configuration
        """
        self.portfolio = portfolio
        self.config = config or RebalanceConfig()

        # Target allocation
        self._target: TargetAllocation | None = None

        # Rebalance history
        self._last_rebalance: datetime | None = None
        self._rebalance_history: list[RebalanceResult] = []

        logger.info(f"PortfolioRebalancer initialized with strategy: {self.config.strategy.value}")

    def set_target_allocation(self, target: TargetAllocation) -> None:
        """
        Set target portfolio allocation.

        Args:
            target: Target allocation
        """
        if not target.validate():
            raise ValueError("Target weights must sum to 1")

        self._target = target
        logger.info(f"Set target allocation with {len(target.weights)} positions")

    def get_target_allocation(self) -> TargetAllocation | None:
        """Get current target allocation."""
        return self._target

    def calculate_drift(self) -> dict[str, float]:
        """
        Calculate drift from target allocation.

        Returns:
            Dictionary of symbol to drift amount
        """
        if self._target is None:
            return {}

        current_allocation = self.portfolio.get_allocation()
        drift = {}

        for symbol, target_weight in self._target.weights.items():
            current_weight = current_allocation.get(symbol, 0.0)
            drift[symbol] = current_weight - target_weight

        # Check for positions not in target
        for symbol, weight in current_allocation.items():
            if symbol not in self._target.weights and symbol != "cash":
                drift[symbol] = weight  # Full weight is drift

        return drift

    def get_max_drift(self) -> float:
        """Get maximum absolute drift from target."""
        drift = self.calculate_drift()
        if not drift:
            return 0.0
        return max(abs(d) for d in drift.values())

    def needs_rebalance(self) -> bool:
        """
        Check if portfolio needs rebalancing.

        Returns:
            True if rebalancing is needed
        """
        if self._target is None:
            return False

        strategy = self.config.strategy

        if strategy == RebalanceStrategy.THRESHOLD:
            return self.get_max_drift() > self.config.threshold

        elif strategy == RebalanceStrategy.CALENDAR:
            return self._check_calendar_rebalance()

        elif strategy in [RebalanceStrategy.CONSTANT_MIX, RebalanceStrategy.CPPI]:
            # Always rebalance based on drift
            return self.get_max_drift() > self.config.threshold

        elif strategy == RebalanceStrategy.TACTICAL:
            # Would need additional market signals
            return self.get_max_drift() > self.config.threshold

        elif strategy == RebalanceStrategy.OPTIMIZATION:
            # Check if optimization would improve portfolio
            return self.get_max_drift() > self.config.threshold

        return False

    def generate_orders(
        self,
        prices: dict[str, float],
    ) -> RebalanceResult:
        """
        Generate rebalancing orders.

        Args:
            prices: Current prices for all symbols

        Returns:
            Rebalance result with orders
        """
        if self._target is None:
            return RebalanceResult(
                timestamp=datetime.now(),
                strategy=self.config.strategy,
                orders=[],
                success=False,
                message="No target allocation set",
            )

        strategy = self.config.strategy

        if strategy == RebalanceStrategy.THRESHOLD:
            orders = self._generate_threshold_orders(prices)
        elif strategy == RebalanceStrategy.CONSTANT_MIX:
            orders = self._generate_constant_mix_orders(prices)
        elif strategy == RebalanceStrategy.CPPI:
            orders = self._generate_cppi_orders(prices)
        elif strategy == RebalanceStrategy.TACTICAL:
            orders = self._generate_tactical_orders(prices)
        elif strategy == RebalanceStrategy.OPTIMIZATION:
            orders = self._generate_optimization_orders(prices)
        else:
            orders = self._generate_threshold_orders(prices)

        # Calculate metrics
        drift_before = self.get_max_drift()
        turnover = self._calculate_turnover(orders)
        costs = self._estimate_costs(orders)

        # Apply constraints
        orders = self._apply_constraints(orders)

        result = RebalanceResult(
            timestamp=datetime.now(),
            strategy=strategy,
            orders=orders,
            turnover=turnover,
            estimated_costs=costs,
            drift_before=drift_before,
            drift_after=0.0,  # Would be calculated after execution
            success=True,
        )

        self._rebalance_history.append(result)
        self._last_rebalance = datetime.now()

        return result

    def _generate_threshold_orders(
        self,
        prices: dict[str, float],
    ) -> list[RebalanceOrder]:
        """Generate orders using threshold rebalancing."""
        orders = []
        total_value = self.portfolio.total_value
        drift = self.calculate_drift()

        for symbol, target_weight in self._target.weights.items():
            if symbol == "cash":
                continue

            price = prices.get(symbol)
            if price is None:
                continue

            current_position = self.portfolio.get_position(symbol)
            current_value = current_position.market_value if current_position else 0.0
            current_weight = current_value / total_value if total_value > 0 else 0.0

            symbol_drift = abs(current_weight - target_weight)

            # Only rebalance if drift exceeds threshold
            if symbol_drift > self.config.threshold:
                target_value = total_value * target_weight
                value_change = target_value - current_value
                quantity = abs(value_change) / price

                if abs(value_change) >= self.config.min_trade_value:
                    action = "BUY" if value_change > 0 else "SELL"
                    orders.append(RebalanceOrder(
                        symbol=symbol,
                        action=action,
                        quantity=quantity,
                        target_value=target_value,
                        current_value=current_value,
                        value_change=value_change,
                        reason=f"Drift of {symbol_drift:.2%} exceeds threshold",
                    ))

        # Handle positions not in target (sell all)
        for symbol, position in self.portfolio.positions.items():
            if symbol not in self._target.weights:
                price = prices.get(symbol, position.current_price)
                orders.append(RebalanceOrder(
                    symbol=symbol,
                    action="SELL",
                    quantity=position.quantity,
                    target_value=0.0,
                    current_value=position.market_value,
                    value_change=-position.market_value,
                    reason="Not in target allocation",
                ))

        return orders

    def _generate_constant_mix_orders(
        self,
        prices: dict[str, float],
    ) -> list[RebalanceOrder]:
        """Generate orders for constant mix strategy."""
        # Same as threshold but always rebalance to exact target
        orders = []
        total_value = self.portfolio.total_value

        for symbol, target_weight in self._target.weights.items():
            if symbol == "cash":
                continue

            price = prices.get(symbol)
            if price is None:
                continue

            current_position = self.portfolio.get_position(symbol)
            current_value = current_position.market_value if current_position else 0.0

            target_value = total_value * target_weight
            value_change = target_value - current_value
            quantity = abs(value_change) / price

            if abs(value_change) >= self.config.min_trade_value:
                action = "BUY" if value_change > 0 else "SELL"
                orders.append(RebalanceOrder(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    target_value=target_value,
                    current_value=current_value,
                    value_change=value_change,
                    reason="Constant mix rebalance",
                ))

        return orders

    def _generate_cppi_orders(
        self,
        prices: dict[str, float],
    ) -> list[RebalanceOrder]:
        """Generate orders for CPPI strategy."""
        orders = []
        total_value = self.portfolio.total_value

        # Calculate floor and cushion
        floor_value = self.portfolio.config.initial_capital * self.config.cppi_floor
        cushion = max(0, total_value - floor_value)

        # Target risky asset allocation
        risky_target = min(cushion * self.config.cppi_multiplier, total_value)
        safe_target = total_value - risky_target

        # Allocate risky portion according to target weights (excluding cash)
        risky_weights = {
            k: v for k, v in self._target.weights.items()
            if k != "cash"
        }
        risky_total_weight = sum(risky_weights.values())

        if risky_total_weight > 0:
            for symbol, weight in risky_weights.items():
                price = prices.get(symbol)
                if price is None:
                    continue

                # Scale weight to risky portion only
                scaled_weight = weight / risky_total_weight
                target_value = risky_target * scaled_weight

                current_position = self.portfolio.get_position(symbol)
                current_value = current_position.market_value if current_position else 0.0

                value_change = target_value - current_value
                quantity = abs(value_change) / price

                if abs(value_change) >= self.config.min_trade_value:
                    action = "BUY" if value_change > 0 else "SELL"
                    orders.append(RebalanceOrder(
                        symbol=symbol,
                        action=action,
                        quantity=quantity,
                        target_value=target_value,
                        current_value=current_value,
                        value_change=value_change,
                        reason=f"CPPI rebalance (cushion: ${cushion:,.2f})",
                    ))

        return orders

    def _generate_tactical_orders(
        self,
        prices: dict[str, float],
    ) -> list[RebalanceOrder]:
        """Generate orders for tactical rebalancing."""
        # Start with threshold rebalancing
        orders = self._generate_threshold_orders(prices)

        # Would add tactical adjustments based on market signals
        # For now, just return threshold orders
        return orders

    def _generate_optimization_orders(
        self,
        prices: dict[str, float],
    ) -> list[RebalanceOrder]:
        """Generate orders using mean-variance optimization."""
        # Simplified optimization - would use full mean-variance in production
        # For now, use threshold with adjusted targets

        orders = self._generate_threshold_orders(prices)
        return orders

    def _check_calendar_rebalance(self) -> bool:
        """Check if calendar rebalancing is due."""
        if self._last_rebalance is None:
            return True

        now = datetime.now()
        frequency = self.config.frequency

        if frequency == RebalanceFrequency.DAILY:
            return (now - self._last_rebalance).days >= 1
        elif frequency == RebalanceFrequency.WEEKLY:
            return (now - self._last_rebalance).days >= 7
        elif frequency == RebalanceFrequency.MONTHLY:
            return (now - self._last_rebalance).days >= 30
        elif frequency == RebalanceFrequency.QUARTERLY:
            return (now - self._last_rebalance).days >= 90
        elif frequency == RebalanceFrequency.YEARLY:
            return (now - self._last_rebalance).days >= 365

        return False

    def _calculate_turnover(self, orders: list[RebalanceOrder]) -> float:
        """Calculate portfolio turnover from orders."""
        total_value = self.portfolio.total_value
        if total_value == 0:
            return 0.0

        total_traded = sum(abs(o.value_change) for o in orders)
        return total_traded / total_value

    def _estimate_costs(self, orders: list[RebalanceOrder]) -> float:
        """Estimate transaction costs for orders."""
        if not self.config.consider_transaction_costs:
            return 0.0

        total_cost = 0.0
        for order in orders:
            value = abs(order.value_change)
            commission = value * self.config.commission_rate
            spread = value * self.config.spread_cost
            total_cost += commission + spread

        return total_cost

    def _apply_constraints(
        self,
        orders: list[RebalanceOrder],
    ) -> list[RebalanceOrder]:
        """Apply constraints to orders."""
        # Filter out small trades
        orders = [
            o for o in orders
            if abs(o.value_change) >= self.config.min_trade_value
        ]

        # Check turnover constraint
        turnover = self._calculate_turnover(orders)
        if turnover > self.config.max_turnover:
            # Scale down orders proportionally
            scale = self.config.max_turnover / turnover
            for order in orders:
                order.quantity *= scale
                order.value_change *= scale

        return orders

    def get_rebalance_history(
        self,
        limit: int = 10,
    ) -> list[RebalanceResult]:
        """Get recent rebalance history."""
        return self._rebalance_history[-limit:]

    def get_last_rebalance(self) -> datetime | None:
        """Get timestamp of last rebalance."""
        return self._last_rebalance


class EqualWeightRebalancer(PortfolioRebalancer):
    """
    Rebalancer that maintains equal weights across all positions.
    """

    def __init__(
        self,
        portfolio: PortfolioManager,
        symbols: list[str],
        cash_weight: float = 0.05,
        config: RebalanceConfig | None = None,
    ) -> None:
        """
        Initialize equal weight rebalancer.

        Args:
            portfolio: Portfolio manager
            symbols: Symbols to include in portfolio
            cash_weight: Target cash weight
            config: Rebalancing configuration
        """
        super().__init__(portfolio, config)

        # Calculate equal weights
        equity_weight = 1.0 - cash_weight
        weight_per_symbol = equity_weight / len(symbols) if symbols else 0.0

        weights = {"cash": cash_weight}
        for symbol in symbols:
            weights[symbol] = weight_per_symbol

        self._target = TargetAllocation(weights=weights)

    def update_symbols(self, symbols: list[str], cash_weight: float = 0.05) -> None:
        """Update symbol list and recalculate equal weights."""
        equity_weight = 1.0 - cash_weight
        weight_per_symbol = equity_weight / len(symbols) if symbols else 0.0

        weights = {"cash": cash_weight}
        for symbol in symbols:
            weights[symbol] = weight_per_symbol

        self._target = TargetAllocation(weights=weights)


class RiskParityRebalancer(PortfolioRebalancer):
    """
    Rebalancer that targets equal risk contribution from each asset.
    """

    def __init__(
        self,
        portfolio: PortfolioManager,
        config: RebalanceConfig | None = None,
    ) -> None:
        """
        Initialize risk parity rebalancer.

        Args:
            portfolio: Portfolio manager
            config: Rebalancing configuration
        """
        super().__init__(portfolio, config)
        self._volatilities: dict[str, float] = {}

    def update_volatilities(self, volatilities: dict[str, float]) -> None:
        """
        Update asset volatilities for risk parity calculation.

        Args:
            volatilities: Dictionary of symbol to annual volatility
        """
        self._volatilities = volatilities
        self._calculate_risk_parity_weights()

    def _calculate_risk_parity_weights(self) -> None:
        """Calculate risk parity weights based on volatilities."""
        if not self._volatilities:
            return

        # Inverse volatility weighting (simplified risk parity)
        inverse_vols = {
            symbol: 1.0 / vol if vol > 0 else 0.0
            for symbol, vol in self._volatilities.items()
        }

        total_inverse = sum(inverse_vols.values())

        if total_inverse > 0:
            weights = {
                symbol: inv_vol / total_inverse
                for symbol, inv_vol in inverse_vols.items()
            }
            self._target = TargetAllocation(weights=weights)


def create_rebalancer(
    portfolio: PortfolioManager,
    config: RebalanceConfig | None = None,
) -> PortfolioRebalancer:
    """
    Create a portfolio rebalancer.

    Args:
        portfolio: Portfolio manager
        config: Rebalancing configuration

    Returns:
        Portfolio rebalancer instance
    """
    return PortfolioRebalancer(portfolio, config)


def create_equal_weight_rebalancer(
    portfolio: PortfolioManager,
    symbols: list[str],
    cash_weight: float = 0.05,
    config: RebalanceConfig | None = None,
) -> EqualWeightRebalancer:
    """
    Create an equal weight rebalancer.

    Args:
        portfolio: Portfolio manager
        symbols: Symbols to include
        cash_weight: Target cash weight
        config: Rebalancing configuration

    Returns:
        Equal weight rebalancer instance
    """
    return EqualWeightRebalancer(portfolio, symbols, cash_weight, config)


def create_risk_parity_rebalancer(
    portfolio: PortfolioManager,
    config: RebalanceConfig | None = None,
) -> RiskParityRebalancer:
    """
    Create a risk parity rebalancer.

    Args:
        portfolio: Portfolio manager
        config: Rebalancing configuration

    Returns:
        Risk parity rebalancer instance
    """
    return RiskParityRebalancer(portfolio, config)
