"""
Portfolio Manager for Ultimate Trading Bot v2.2.

Provides comprehensive portfolio management including tracking, allocation,
performance analysis, and risk management.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PortfolioType(str, Enum):
    """Types of portfolios."""
    CASH = "cash"
    MARGIN = "margin"
    RETIREMENT = "retirement"
    PAPER = "paper"


class AssetClass(str, Enum):
    """Asset classes."""
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    CRYPTO = "crypto"
    DERIVATIVE = "derivative"
    CASH = "cash"


@dataclass
class PortfolioConfig:
    """Configuration for portfolio management."""

    # Basic settings
    initial_capital: float = 100000.0
    currency: str = "USD"
    portfolio_type: PortfolioType = PortfolioType.CASH

    # Allocation limits
    max_position_size: float = 0.1  # 10% max per position
    max_sector_exposure: float = 0.3  # 30% max per sector
    max_asset_class_exposure: float = 0.5  # 50% max per asset class
    min_cash_reserve: float = 0.05  # 5% minimum cash

    # Risk settings
    max_portfolio_var: float = 0.02  # 2% daily VaR
    max_drawdown: float = 0.2  # 20% max drawdown
    target_volatility: float = 0.15  # 15% annual volatility

    # Rebalancing
    rebalance_threshold: float = 0.05  # 5% drift threshold
    rebalance_frequency: str = "weekly"  # daily, weekly, monthly

    # Performance tracking
    benchmark: str = "SPY"
    track_history: bool = True
    history_days: int = 365


@dataclass
class Position:
    """Represents a portfolio position."""

    symbol: str
    quantity: float
    avg_cost: float
    current_price: float
    asset_class: AssetClass = AssetClass.EQUITY

    # Optional fields
    sector: str | None = None
    industry: str | None = None
    entry_date: datetime | None = None

    # Computed fields
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    weight: float = 0.0

    # Risk metrics
    beta: float | None = None
    volatility: float | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Calculate derived fields."""
        self.market_value = self.quantity * self.current_price
        cost_basis = self.quantity * self.avg_cost
        self.unrealized_pnl = self.market_value - cost_basis
        if cost_basis > 0:
            self.unrealized_pnl_pct = self.unrealized_pnl / cost_basis
        else:
            self.unrealized_pnl_pct = 0.0

    def update_price(self, price: float) -> None:
        """Update position with new price."""
        self.current_price = price
        self.__post_init__()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_cost": self.avg_cost,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "weight": self.weight,
            "asset_class": self.asset_class.value,
            "sector": self.sector,
        }


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio state at a point in time."""

    timestamp: datetime
    total_value: float
    cash: float
    positions_value: float
    positions: list[Position]

    # Performance
    daily_return: float = 0.0
    total_return: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    # Risk metrics
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    beta: float = 0.0

    # Allocation
    allocation: dict[str, float] = field(default_factory=dict)
    sector_allocation: dict[str, float] = field(default_factory=dict)
    asset_class_allocation: dict[str, float] = field(default_factory=dict)


@dataclass
class PortfolioPerformance:
    """Portfolio performance metrics."""

    # Returns
    total_return: float = 0.0
    annualized_return: float = 0.0
    daily_returns: list[float] = field(default_factory=list)

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown_duration: int = 0

    # Risk
    volatility: float = 0.0
    downside_volatility: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    beta: float = 0.0

    # Other
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0

    # Benchmark comparison
    benchmark_return: float = 0.0
    alpha: float = 0.0
    tracking_error: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "volatility": self.volatility,
            "var_95": self.var_95,
            "beta": self.beta,
            "alpha": self.alpha,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
        }


class PortfolioManager:
    """
    Comprehensive portfolio management system.

    Handles position tracking, allocation, risk management,
    and performance analysis.
    """

    def __init__(self, config: PortfolioConfig | None = None) -> None:
        """
        Initialize portfolio manager.

        Args:
            config: Portfolio configuration
        """
        self.config = config or PortfolioConfig()

        # Portfolio state
        self._cash = self.config.initial_capital
        self._positions: dict[str, Position] = {}
        self._closed_positions: list[dict[str, Any]] = []

        # History
        self._snapshots: list[PortfolioSnapshot] = []
        self._daily_returns: list[float] = []
        self._value_history: list[tuple[datetime, float]] = []

        # Performance tracking
        self._initial_value = self.config.initial_capital
        self._high_water_mark = self.config.initial_capital
        self._realized_pnl = 0.0

        # Benchmark data
        self._benchmark_returns: list[float] = []

        self._initialized = False

        logger.info(
            f"PortfolioManager created with initial capital: "
            f"${self.config.initial_capital:,.2f}"
        )

    async def initialize(self) -> None:
        """Initialize portfolio manager."""
        # Take initial snapshot
        await self._take_snapshot()
        self._initialized = True
        logger.info("PortfolioManager initialized")

    @property
    def cash(self) -> float:
        """Get available cash."""
        return self._cash

    @property
    def positions(self) -> dict[str, Position]:
        """Get all positions."""
        return self._positions.copy()

    @property
    def total_value(self) -> float:
        """Get total portfolio value."""
        positions_value = sum(p.market_value for p in self._positions.values())
        return self._cash + positions_value

    @property
    def positions_value(self) -> float:
        """Get total positions value."""
        return sum(p.market_value for p in self._positions.values())

    @property
    def position_count(self) -> int:
        """Get number of positions."""
        return len(self._positions)

    async def add_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        asset_class: AssetClass = AssetClass.EQUITY,
        sector: str | None = None,
    ) -> Position:
        """
        Add a new position or add to existing position.

        Args:
            symbol: Trading symbol
            quantity: Number of shares/units
            price: Entry price
            asset_class: Asset class
            sector: Sector classification

        Returns:
            Updated position
        """
        cost = quantity * price

        # Check cash availability
        if cost > self._cash:
            raise ValueError(f"Insufficient cash: ${self._cash:,.2f} < ${cost:,.2f}")

        # Check position size limit
        if cost / self.total_value > self.config.max_position_size:
            raise ValueError(
                f"Position size exceeds limit: "
                f"{cost / self.total_value:.1%} > {self.config.max_position_size:.1%}"
            )

        # Update or create position
        if symbol in self._positions:
            existing = self._positions[symbol]
            total_quantity = existing.quantity + quantity
            total_cost = (existing.quantity * existing.avg_cost) + cost
            avg_cost = total_cost / total_quantity

            existing.quantity = total_quantity
            existing.avg_cost = avg_cost
            existing.update_price(price)
            position = existing
        else:
            position = Position(
                symbol=symbol,
                quantity=quantity,
                avg_cost=price,
                current_price=price,
                asset_class=asset_class,
                sector=sector,
                entry_date=datetime.now(),
            )
            self._positions[symbol] = position

        # Deduct cash
        self._cash -= cost

        # Update weights
        self._update_weights()

        logger.info(
            f"Added position: {quantity} {symbol} @ ${price:.2f} "
            f"(total: {position.quantity})"
        )

        return position

    async def reduce_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
    ) -> float:
        """
        Reduce or close a position.

        Args:
            symbol: Trading symbol
            quantity: Number of shares to sell
            price: Exit price

        Returns:
            Realized P&L
        """
        if symbol not in self._positions:
            raise ValueError(f"Position not found: {symbol}")

        position = self._positions[symbol]

        if quantity > position.quantity:
            raise ValueError(
                f"Quantity exceeds position: "
                f"{quantity} > {position.quantity}"
            )

        # Calculate realized P&L
        cost_basis = quantity * position.avg_cost
        proceeds = quantity * price
        realized_pnl = proceeds - cost_basis

        self._realized_pnl += realized_pnl

        # Update position
        position.quantity -= quantity

        if position.quantity <= 0:
            # Close position
            self._closed_positions.append({
                "symbol": symbol,
                "quantity": quantity,
                "avg_cost": position.avg_cost,
                "exit_price": price,
                "realized_pnl": realized_pnl,
                "entry_date": position.entry_date,
                "exit_date": datetime.now(),
            })
            del self._positions[symbol]
        else:
            position.update_price(price)

        # Add cash
        self._cash += proceeds

        # Update weights
        self._update_weights()

        logger.info(
            f"Reduced position: {quantity} {symbol} @ ${price:.2f} "
            f"(P&L: ${realized_pnl:,.2f})"
        )

        return realized_pnl

    async def close_position(
        self,
        symbol: str,
        price: float,
    ) -> float:
        """
        Close entire position.

        Args:
            symbol: Trading symbol
            price: Exit price

        Returns:
            Realized P&L
        """
        if symbol not in self._positions:
            raise ValueError(f"Position not found: {symbol}")

        position = self._positions[symbol]
        return await self.reduce_position(symbol, position.quantity, price)

    async def update_prices(
        self,
        prices: dict[str, float],
    ) -> None:
        """
        Update positions with current prices.

        Args:
            prices: Dictionary of symbol to price
        """
        for symbol, position in self._positions.items():
            if symbol in prices:
                position.update_price(prices[symbol])

        self._update_weights()
        await self._take_snapshot()

    def get_position(self, symbol: str) -> Position | None:
        """Get a specific position."""
        return self._positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """Check if position exists."""
        return symbol in self._positions

    def get_allocation(self) -> dict[str, float]:
        """Get current portfolio allocation."""
        allocation = {}
        total = self.total_value

        if total > 0:
            allocation["cash"] = self._cash / total
            for symbol, position in self._positions.items():
                allocation[symbol] = position.market_value / total

        return allocation

    def get_sector_allocation(self) -> dict[str, float]:
        """Get allocation by sector."""
        sector_values: dict[str, float] = {}
        total = self.total_value

        for position in self._positions.values():
            sector = position.sector or "Unknown"
            sector_values[sector] = (
                sector_values.get(sector, 0) + position.market_value
            )

        return {
            sector: value / total if total > 0 else 0.0
            for sector, value in sector_values.items()
        }

    def get_asset_class_allocation(self) -> dict[str, float]:
        """Get allocation by asset class."""
        class_values: dict[str, float] = {}
        total = self.total_value

        # Add cash
        class_values[AssetClass.CASH.value] = self._cash

        for position in self._positions.values():
            asset_class = position.asset_class.value
            class_values[asset_class] = (
                class_values.get(asset_class, 0) + position.market_value
            )

        return {
            ac: value / total if total > 0 else 0.0
            for ac, value in class_values.items()
        }

    async def get_performance(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> PortfolioPerformance:
        """
        Calculate portfolio performance metrics.

        Args:
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            Portfolio performance
        """
        if not self._daily_returns:
            return PortfolioPerformance()

        returns = np.array(self._daily_returns)

        # Filter by date if provided
        if start_date or end_date:
            filtered_returns = []
            for i, (ts, _) in enumerate(self._value_history):
                if start_date and ts < start_date:
                    continue
                if end_date and ts > end_date:
                    continue
                if i < len(self._daily_returns):
                    filtered_returns.append(self._daily_returns[i])
            if filtered_returns:
                returns = np.array(filtered_returns)

        # Calculate metrics
        total_return = (self.total_value - self._initial_value) / self._initial_value
        annualized_return = self._annualize_return(returns)
        volatility = float(np.std(returns) * np.sqrt(252)) if len(returns) > 1 else 0.0

        # Risk-adjusted metrics
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        excess_returns = returns - risk_free_rate

        sharpe = (
            float(np.mean(excess_returns) / np.std(returns) * np.sqrt(252))
            if len(returns) > 1 and np.std(returns) > 0 else 0.0
        )

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_vol = (
            float(np.std(downside_returns) * np.sqrt(252))
            if len(downside_returns) > 0 else 0.0
        )
        sortino = (
            float(np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252))
            if len(downside_returns) > 0 and np.std(downside_returns) > 0 else 0.0
        )

        # Drawdown
        max_dd, avg_dd, max_dd_duration = self._calculate_drawdown_metrics(returns)

        # VaR and CVaR
        var_95 = float(np.percentile(returns, 5)) if len(returns) > 0 else 0.0
        cvar_95 = float(np.mean(returns[returns <= var_95])) if len(returns) > 0 else 0.0

        # Win rate and profit factor
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        win_rate = len(wins) / len(returns) if len(returns) > 0 else 0.0
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

        total_wins = float(np.sum(wins)) if len(wins) > 0 else 0.0
        total_losses = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Calmar ratio
        calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0.0

        # Beta and alpha (vs benchmark)
        beta = 0.0
        alpha = 0.0
        tracking_error = 0.0
        benchmark_return = 0.0

        if self._benchmark_returns and len(self._benchmark_returns) == len(returns):
            benchmark = np.array(self._benchmark_returns)
            benchmark_return = float(np.prod(1 + benchmark) - 1)

            cov_matrix = np.cov(returns, benchmark)
            if cov_matrix[1, 1] > 0:
                beta = float(cov_matrix[0, 1] / cov_matrix[1, 1])

            alpha = annualized_return - beta * (benchmark_return * 252 / len(returns))
            tracking_error = float(np.std(returns - benchmark) * np.sqrt(252))

        return PortfolioPerformance(
            total_return=total_return,
            annualized_return=annualized_return,
            daily_returns=list(returns),
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            information_ratio=alpha / tracking_error if tracking_error > 0 else 0.0,
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            max_drawdown_duration=max_dd_duration,
            volatility=volatility,
            downside_volatility=downside_vol,
            var_95=var_95,
            cvar_95=cvar_95,
            beta=beta,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            benchmark_return=benchmark_return,
            alpha=alpha,
            tracking_error=tracking_error,
        )

    async def get_snapshot(self) -> PortfolioSnapshot:
        """Get current portfolio snapshot."""
        return PortfolioSnapshot(
            timestamp=datetime.now(),
            total_value=self.total_value,
            cash=self._cash,
            positions_value=self.positions_value,
            positions=list(self._positions.values()),
            daily_return=self._daily_returns[-1] if self._daily_returns else 0.0,
            total_return=(self.total_value - self._initial_value) / self._initial_value,
            unrealized_pnl=sum(p.unrealized_pnl for p in self._positions.values()),
            realized_pnl=self._realized_pnl,
            allocation=self.get_allocation(),
            sector_allocation=self.get_sector_allocation(),
            asset_class_allocation=self.get_asset_class_allocation(),
        )

    def check_constraints(self) -> dict[str, bool]:
        """
        Check portfolio constraints.

        Returns:
            Dictionary of constraint checks
        """
        total = self.total_value
        allocation = self.get_allocation()
        sector_allocation = self.get_sector_allocation()
        asset_class_allocation = self.get_asset_class_allocation()

        checks = {}

        # Cash reserve
        cash_ratio = self._cash / total if total > 0 else 1.0
        checks["cash_reserve"] = cash_ratio >= self.config.min_cash_reserve

        # Position size
        max_position = max(
            (v for k, v in allocation.items() if k != "cash"),
            default=0
        )
        checks["position_size"] = max_position <= self.config.max_position_size

        # Sector exposure
        max_sector = max(sector_allocation.values(), default=0)
        checks["sector_exposure"] = max_sector <= self.config.max_sector_exposure

        # Asset class exposure
        max_asset_class = max(
            (v for k, v in asset_class_allocation.items() if k != AssetClass.CASH.value),
            default=0
        )
        checks["asset_class_exposure"] = max_asset_class <= self.config.max_asset_class_exposure

        # Drawdown
        current_drawdown = (self._high_water_mark - self.total_value) / self._high_water_mark
        checks["max_drawdown"] = current_drawdown <= self.config.max_drawdown

        return checks

    def get_risk_metrics(self) -> dict[str, float]:
        """Get current portfolio risk metrics."""
        if not self._daily_returns:
            return {
                "volatility": 0.0,
                "var_95": 0.0,
                "current_drawdown": 0.0,
                "max_drawdown": 0.0,
            }

        returns = np.array(self._daily_returns)

        volatility = float(np.std(returns) * np.sqrt(252)) if len(returns) > 1 else 0.0
        var_95 = float(np.percentile(returns, 5)) if len(returns) > 0 else 0.0
        current_dd = (self._high_water_mark - self.total_value) / self._high_water_mark
        max_dd = self._calculate_drawdown_metrics(returns)[0]

        return {
            "volatility": volatility,
            "var_95": var_95,
            "current_drawdown": current_dd,
            "max_drawdown": max_dd,
        }

    def _update_weights(self) -> None:
        """Update position weights."""
        total = self.total_value
        for position in self._positions.values():
            position.weight = position.market_value / total if total > 0 else 0.0

    async def _take_snapshot(self) -> None:
        """Take portfolio snapshot and update history."""
        current_value = self.total_value

        # Calculate daily return
        if self._value_history:
            prev_value = self._value_history[-1][1]
            daily_return = (current_value - prev_value) / prev_value if prev_value > 0 else 0.0
            self._daily_returns.append(daily_return)

        # Update value history
        self._value_history.append((datetime.now(), current_value))

        # Update high water mark
        if current_value > self._high_water_mark:
            self._high_water_mark = current_value

        # Keep limited history
        max_days = self.config.history_days
        if len(self._value_history) > max_days:
            self._value_history = self._value_history[-max_days:]
            self._daily_returns = self._daily_returns[-max_days:]

    def _annualize_return(self, returns: np.ndarray) -> float:
        """Annualize daily returns."""
        if len(returns) == 0:
            return 0.0

        total_return = float(np.prod(1 + returns) - 1)
        days = len(returns)
        if days > 0:
            return float((1 + total_return) ** (252 / days) - 1)
        return 0.0

    def _calculate_drawdown_metrics(
        self,
        returns: np.ndarray,
    ) -> tuple[float, float, int]:
        """Calculate drawdown metrics."""
        if len(returns) == 0:
            return 0.0, 0.0, 0

        # Calculate cumulative returns
        cum_returns = np.cumprod(1 + returns)

        # Calculate running maximum
        running_max = np.maximum.accumulate(cum_returns)

        # Calculate drawdowns
        drawdowns = (running_max - cum_returns) / running_max

        max_dd = float(np.max(drawdowns))
        avg_dd = float(np.mean(drawdowns[drawdowns > 0])) if np.any(drawdowns > 0) else 0.0

        # Calculate max drawdown duration
        in_drawdown = False
        current_duration = 0
        max_duration = 0

        for dd in drawdowns:
            if dd > 0:
                if not in_drawdown:
                    in_drawdown = True
                    current_duration = 1
                else:
                    current_duration += 1
            else:
                if in_drawdown:
                    max_duration = max(max_duration, current_duration)
                    in_drawdown = False
                    current_duration = 0

        if in_drawdown:
            max_duration = max(max_duration, current_duration)

        return max_dd, avg_dd, max_duration

    def add_benchmark_return(self, daily_return: float) -> None:
        """Add benchmark daily return for comparison."""
        self._benchmark_returns.append(daily_return)

        if len(self._benchmark_returns) > self.config.history_days:
            self._benchmark_returns = self._benchmark_returns[-self.config.history_days:]

    async def reset(self) -> None:
        """Reset portfolio to initial state."""
        self._cash = self.config.initial_capital
        self._positions.clear()
        self._closed_positions.clear()
        self._snapshots.clear()
        self._daily_returns.clear()
        self._value_history.clear()
        self._benchmark_returns.clear()

        self._initial_value = self.config.initial_capital
        self._high_water_mark = self.config.initial_capital
        self._realized_pnl = 0.0

        logger.info("Portfolio reset to initial state")


def create_portfolio_manager(
    config: PortfolioConfig | None = None,
) -> PortfolioManager:
    """
    Create a portfolio manager instance.

    Args:
        config: Portfolio configuration

    Returns:
        Portfolio manager instance
    """
    return PortfolioManager(config)
