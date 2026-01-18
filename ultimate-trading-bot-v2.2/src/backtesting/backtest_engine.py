"""
Backtesting Engine for Ultimate Trading Bot v2.2.

This module provides the core backtesting engine for historical
strategy simulation with realistic execution modeling.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Iterator
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class BacktestMode(str, Enum):
    """Backtesting modes."""

    VECTORIZED = "vectorized"
    EVENT_DRIVEN = "event_driven"
    TICK_BY_TICK = "tick_by_tick"


class FillModel(str, Enum):
    """Order fill models."""

    IMMEDIATE = "immediate"
    NEXT_BAR = "next_bar"
    VWAP = "vwap"
    REALISTIC = "realistic"


class SlippageModel(str, Enum):
    """Slippage models."""

    NONE = "none"
    FIXED = "fixed"
    PERCENTAGE = "percentage"
    VOLUME_BASED = "volume_based"
    VOLATILITY_BASED = "volatility_based"


class BacktestConfig(BaseModel):
    """Configuration for backtesting."""

    model_config = {"arbitrary_types_allowed": True}

    start_date: datetime
    end_date: datetime
    initial_capital: float = Field(default=100000.0, description="Initial capital")

    mode: BacktestMode = Field(default=BacktestMode.EVENT_DRIVEN)
    fill_model: FillModel = Field(default=FillModel.NEXT_BAR)
    slippage_model: SlippageModel = Field(default=SlippageModel.PERCENTAGE)

    slippage_pct: float = Field(default=0.001, description="Slippage percentage")
    slippage_fixed: float = Field(default=0.01, description="Fixed slippage per share")
    commission_per_share: float = Field(default=0.005, description="Commission per share")
    commission_pct: float = Field(default=0.0001, description="Commission percentage")
    min_commission: float = Field(default=1.0, description="Minimum commission")

    margin_requirement: float = Field(default=0.5, description="Margin requirement")
    allow_short_selling: bool = Field(default=True, description="Allow short selling")
    short_borrow_rate: float = Field(default=0.02, description="Annual short borrow rate")

    max_position_size: float = Field(default=0.25, description="Max position as % of portfolio")
    max_leverage: float = Field(default=2.0, description="Maximum leverage")

    data_frequency: str = Field(default="1D", description="Data frequency")
    benchmark_symbol: str | None = Field(default="SPY", description="Benchmark symbol")

    enable_logging: bool = Field(default=True, description="Enable detailed logging")
    save_trades: bool = Field(default=True, description="Save trade history")
    save_portfolio_history: bool = Field(default=True, description="Save portfolio history")


class BacktestOrder(BaseModel):
    """Order in backtesting context."""

    order_id: str = Field(default_factory=lambda: str(uuid4()))
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    order_type: str = Field(default="market")
    limit_price: float | None = None
    stop_price: float | None = None

    created_at: datetime
    filled_at: datetime | None = None
    fill_price: float | None = None
    commission: float = Field(default=0.0)
    slippage: float = Field(default=0.0)

    status: str = Field(default="pending")
    strategy_id: str | None = None
    signal_id: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class BacktestTrade(BaseModel):
    """Completed trade record."""

    trade_id: str = Field(default_factory=lambda: str(uuid4()))
    symbol: str
    side: str
    quantity: float
    entry_price: float
    entry_time: datetime
    exit_price: float | None = None
    exit_time: datetime | None = None

    pnl: float = Field(default=0.0)
    pnl_pct: float = Field(default=0.0)
    commission: float = Field(default=0.0)
    slippage: float = Field(default=0.0)

    holding_period_days: int = Field(default=0)
    max_favorable_excursion: float = Field(default=0.0)
    max_adverse_excursion: float = Field(default=0.0)

    strategy_id: str | None = None
    is_closed: bool = Field(default=False)


class BacktestPosition(BaseModel):
    """Position during backtest."""

    symbol: str
    quantity: float = Field(default=0.0)
    avg_entry_price: float = Field(default=0.0)
    current_price: float = Field(default=0.0)
    market_value: float = Field(default=0.0)
    cost_basis: float = Field(default=0.0)
    unrealized_pnl: float = Field(default=0.0)
    realized_pnl: float = Field(default=0.0)
    entry_time: datetime | None = None


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio state."""

    timestamp: datetime
    cash: float
    equity: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    positions_count: int
    leverage: float = 1.0
    drawdown: float = 0.0
    high_water_mark: float = 0.0


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    backtest_id: str
    config: BacktestConfig
    start_time: datetime
    end_time: datetime

    initial_capital: float
    final_equity: float
    total_return: float = 0.0
    total_return_pct: float = 0.0

    trades: list[BacktestTrade] = field(default_factory=list)
    orders: list[BacktestOrder] = field(default_factory=list)
    portfolio_history: list[PortfolioSnapshot] = field(default_factory=list)

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    calmar_ratio: float = 0.0

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    annualized_return: float = 0.0
    annualized_volatility: float = 0.0

    total_commission: float = 0.0
    total_slippage: float = 0.0

    benchmark_return: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0

    execution_time_seconds: float = 0.0


class BacktestEngine:
    """
    Core backtesting engine for strategy simulation.

    Provides realistic historical simulation with configurable
    execution models, slippage, and commission handling.
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration
        """
        self.config = config
        self._reset_state()

        logger.info(
            f"BacktestEngine initialized: {config.start_date} to {config.end_date}"
        )

    def _reset_state(self) -> None:
        """Reset engine state."""
        self._cash = self.config.initial_capital
        self._equity = self.config.initial_capital
        self._positions: dict[str, BacktestPosition] = {}
        self._orders: list[BacktestOrder] = []
        self._trades: list[BacktestTrade] = []
        self._portfolio_history: list[PortfolioSnapshot] = []
        self._current_time: datetime = self.config.start_date
        self._high_water_mark = self.config.initial_capital
        self._benchmark_data: dict[datetime, float] = {}
        self._realized_pnl = 0.0

    async def run(
        self,
        data: dict[str, list[dict[str, Any]]],
        strategy_callback: Callable[[datetime, dict[str, Any], dict[str, Any]], list[dict[str, Any]]],
        benchmark_data: list[dict[str, Any]] | None = None,
    ) -> BacktestResult:
        """
        Run backtest with provided data and strategy.

        Args:
            data: Historical data by symbol
            strategy_callback: Strategy function that generates signals
            benchmark_data: Optional benchmark data

        Returns:
            BacktestResult
        """
        start_execution = datetime.now()

        self._reset_state()

        if benchmark_data:
            self._benchmark_data = {
                d["timestamp"]: d["close"]
                for d in benchmark_data
            }

        timestamps = self._get_timestamps(data)

        for timestamp in timestamps:
            self._current_time = timestamp

            bar_data = self._get_bar_data(data, timestamp)

            self._update_positions(bar_data)

            portfolio_state = self._get_portfolio_state()

            signals = strategy_callback(timestamp, bar_data, portfolio_state)

            for signal in signals:
                await self._process_signal(signal, bar_data)

            self._record_portfolio_snapshot()

        self._close_all_positions(data)

        result = self._compile_results(start_execution)

        return result

    def _get_timestamps(
        self,
        data: dict[str, list[dict[str, Any]]],
    ) -> list[datetime]:
        """Extract unique sorted timestamps from data."""
        all_timestamps: set[datetime] = set()

        for symbol_data in data.values():
            for bar in symbol_data:
                ts = bar.get("timestamp")
                if isinstance(ts, datetime):
                    all_timestamps.add(ts)

        return sorted(all_timestamps)

    def _get_bar_data(
        self,
        data: dict[str, list[dict[str, Any]]],
        timestamp: datetime,
    ) -> dict[str, dict[str, Any]]:
        """Get bar data for all symbols at timestamp."""
        bar_data: dict[str, dict[str, Any]] = {}

        for symbol, symbol_data in data.items():
            for bar in symbol_data:
                if bar.get("timestamp") == timestamp:
                    bar_data[symbol] = bar
                    break

        return bar_data

    def _update_positions(
        self,
        bar_data: dict[str, dict[str, Any]],
    ) -> None:
        """Update position values with current prices."""
        for symbol, position in self._positions.items():
            if symbol in bar_data:
                price = bar_data[symbol].get("close", position.current_price)
                position.current_price = price
                position.market_value = position.quantity * price
                position.unrealized_pnl = position.market_value - position.cost_basis

    def _get_portfolio_state(self) -> dict[str, Any]:
        """Get current portfolio state."""
        positions_value = sum(p.market_value for p in self._positions.values())
        self._equity = self._cash + positions_value

        return {
            "cash": self._cash,
            "equity": self._equity,
            "positions": {
                symbol: {
                    "quantity": p.quantity,
                    "avg_price": p.avg_entry_price,
                    "current_price": p.current_price,
                    "market_value": p.market_value,
                    "unrealized_pnl": p.unrealized_pnl,
                }
                for symbol, p in self._positions.items()
            },
            "realized_pnl": self._realized_pnl,
            "timestamp": self._current_time,
        }

    async def _process_signal(
        self,
        signal: dict[str, Any],
        bar_data: dict[str, dict[str, Any]],
    ) -> None:
        """Process a trading signal."""
        symbol = signal.get("symbol")
        action = signal.get("action", "").lower()
        quantity = signal.get("quantity", 0)
        order_type = signal.get("order_type", "market")
        limit_price = signal.get("limit_price")
        stop_price = signal.get("stop_price")

        if not symbol or not action or quantity <= 0:
            return

        if symbol not in bar_data:
            return

        if not self._validate_order(symbol, action, quantity, bar_data):
            return

        order = BacktestOrder(
            symbol=symbol,
            side="buy" if action == "buy" else "sell",
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            created_at=self._current_time,
            strategy_id=signal.get("strategy_id"),
            signal_id=signal.get("signal_id"),
        )

        self._orders.append(order)

        if self.config.fill_model == FillModel.IMMEDIATE:
            self._fill_order(order, bar_data)
        elif self.config.fill_model == FillModel.NEXT_BAR:
            self._fill_order(order, bar_data)

    def _validate_order(
        self,
        symbol: str,
        action: str,
        quantity: float,
        bar_data: dict[str, dict[str, Any]],
    ) -> bool:
        """Validate order against constraints."""
        price = bar_data[symbol].get("close", 0)
        order_value = quantity * price

        if action == "buy":
            if order_value > self._cash:
                return False

            positions_value = sum(p.market_value for p in self._positions.values())
            new_total = positions_value + order_value
            if new_total / self._equity > self.config.max_leverage:
                return False

            if order_value / self._equity > self.config.max_position_size:
                return False

        if action == "sell":
            if not self.config.allow_short_selling:
                position = self._positions.get(symbol)
                if not position or position.quantity < quantity:
                    return False

        return True

    def _fill_order(
        self,
        order: BacktestOrder,
        bar_data: dict[str, dict[str, Any]],
    ) -> None:
        """Fill an order."""
        if order.symbol not in bar_data:
            order.status = "rejected"
            return

        bar = bar_data[order.symbol]
        base_price = bar.get("close", 0)

        slippage = self._calculate_slippage(order, bar)
        if order.side == "buy":
            fill_price = base_price + slippage
        else:
            fill_price = base_price - slippage

        commission = self._calculate_commission(order, fill_price)

        order.filled_at = self._current_time
        order.fill_price = fill_price
        order.commission = commission
        order.slippage = slippage
        order.status = "filled"

        self._update_position_from_fill(order)

    def _calculate_slippage(
        self,
        order: BacktestOrder,
        bar: dict[str, Any],
    ) -> float:
        """Calculate slippage for an order."""
        price = bar.get("close", 0)

        if self.config.slippage_model == SlippageModel.NONE:
            return 0.0

        elif self.config.slippage_model == SlippageModel.FIXED:
            return self.config.slippage_fixed

        elif self.config.slippage_model == SlippageModel.PERCENTAGE:
            return price * self.config.slippage_pct

        elif self.config.slippage_model == SlippageModel.VOLUME_BASED:
            volume = bar.get("volume", 1e6)
            participation = order.quantity / volume
            return price * participation * 0.1

        elif self.config.slippage_model == SlippageModel.VOLATILITY_BASED:
            high = bar.get("high", price)
            low = bar.get("low", price)
            volatility = (high - low) / price if price > 0 else 0
            return price * volatility * 0.1

        return 0.0

    def _calculate_commission(
        self,
        order: BacktestOrder,
        fill_price: float,
    ) -> float:
        """Calculate commission for an order."""
        per_share = self.config.commission_per_share * order.quantity
        percentage = fill_price * order.quantity * self.config.commission_pct
        commission = max(per_share + percentage, self.config.min_commission)
        return commission

    def _update_position_from_fill(self, order: BacktestOrder) -> None:
        """Update position from order fill."""
        symbol = order.symbol
        fill_price = order.fill_price or 0
        quantity = order.quantity
        commission = order.commission

        if order.side == "buy":
            self._cash -= (quantity * fill_price + commission)

            if symbol in self._positions:
                pos = self._positions[symbol]
                total_qty = pos.quantity + quantity
                if total_qty != 0:
                    pos.avg_entry_price = (
                        pos.quantity * pos.avg_entry_price + quantity * fill_price
                    ) / total_qty
                pos.quantity = total_qty
                pos.cost_basis = abs(pos.quantity * pos.avg_entry_price)
            else:
                self._positions[symbol] = BacktestPosition(
                    symbol=symbol,
                    quantity=quantity,
                    avg_entry_price=fill_price,
                    current_price=fill_price,
                    market_value=quantity * fill_price,
                    cost_basis=quantity * fill_price,
                    entry_time=self._current_time,
                )

            trade = BacktestTrade(
                symbol=symbol,
                side="buy",
                quantity=quantity,
                entry_price=fill_price,
                entry_time=self._current_time,
                commission=commission,
                strategy_id=order.strategy_id,
            )
            self._trades.append(trade)

        else:
            self._cash += (quantity * fill_price - commission)

            if symbol in self._positions:
                pos = self._positions[symbol]

                if pos.quantity > 0:
                    realized = (fill_price - pos.avg_entry_price) * min(quantity, pos.quantity)
                    self._realized_pnl += realized
                    pos.realized_pnl += realized

                pos.quantity -= quantity

                if abs(pos.quantity) < 0.0001:
                    del self._positions[symbol]
                else:
                    pos.cost_basis = abs(pos.quantity * pos.avg_entry_price)

            for trade in reversed(self._trades):
                if trade.symbol == symbol and not trade.is_closed and trade.side == "buy":
                    trade.exit_price = fill_price
                    trade.exit_time = self._current_time
                    trade.pnl = (fill_price - trade.entry_price) * trade.quantity - trade.commission - commission
                    trade.pnl_pct = trade.pnl / (trade.entry_price * trade.quantity)
                    trade.is_closed = True
                    if trade.entry_time:
                        trade.holding_period_days = (self._current_time - trade.entry_time).days
                    break

    def _record_portfolio_snapshot(self) -> None:
        """Record portfolio snapshot."""
        positions_value = sum(p.market_value for p in self._positions.values())
        self._equity = self._cash + positions_value

        self._high_water_mark = max(self._high_water_mark, self._equity)
        drawdown = (self._high_water_mark - self._equity) / self._high_water_mark

        leverage = positions_value / self._equity if self._equity > 0 else 0

        snapshot = PortfolioSnapshot(
            timestamp=self._current_time,
            cash=self._cash,
            equity=self._equity,
            market_value=positions_value,
            unrealized_pnl=sum(p.unrealized_pnl for p in self._positions.values()),
            realized_pnl=self._realized_pnl,
            positions_count=len(self._positions),
            leverage=leverage,
            drawdown=drawdown,
            high_water_mark=self._high_water_mark,
        )

        if self.config.save_portfolio_history:
            self._portfolio_history.append(snapshot)

    def _close_all_positions(
        self,
        data: dict[str, list[dict[str, Any]]],
    ) -> None:
        """Close all remaining positions at end of backtest."""
        for symbol, position in list(self._positions.items()):
            if position.quantity != 0:
                last_price = position.current_price

                if position.quantity > 0:
                    pnl = (last_price - position.avg_entry_price) * position.quantity
                else:
                    pnl = (position.avg_entry_price - last_price) * abs(position.quantity)

                self._cash += position.market_value
                self._realized_pnl += pnl

        self._positions.clear()

    def _compile_results(
        self,
        start_execution: datetime,
    ) -> BacktestResult:
        """Compile backtest results."""
        end_execution = datetime.now()
        execution_time = (end_execution - start_execution).total_seconds()

        final_equity = self._cash

        total_return = final_equity - self.config.initial_capital
        total_return_pct = total_return / self.config.initial_capital

        closed_trades = [t for t in self._trades if t.is_closed]
        winning = [t for t in closed_trades if t.pnl > 0]
        losing = [t for t in closed_trades if t.pnl < 0]

        win_rate = len(winning) / len(closed_trades) if closed_trades else 0
        avg_win = np.mean([t.pnl for t in winning]) if winning else 0
        avg_loss = np.mean([t.pnl for t in losing]) if losing else 0
        largest_win = max([t.pnl for t in winning], default=0)
        largest_loss = min([t.pnl for t in losing], default=0)

        total_wins = sum(t.pnl for t in winning)
        total_losses = abs(sum(t.pnl for t in losing))
        profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

        max_drawdown = 0.0
        if self._portfolio_history:
            drawdowns = [s.drawdown for s in self._portfolio_history]
            max_drawdown = max(drawdowns) if drawdowns else 0

        returns = []
        if len(self._portfolio_history) > 1:
            for i in range(1, len(self._portfolio_history)):
                prev_equity = self._portfolio_history[i - 1].equity
                curr_equity = self._portfolio_history[i].equity
                if prev_equity > 0:
                    returns.append((curr_equity - prev_equity) / prev_equity)

        if returns:
            annualized_return = np.mean(returns) * 252
            annualized_volatility = np.std(returns) * np.sqrt(252)
            sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0

            negative_returns = [r for r in returns if r < 0]
            downside_std = np.std(negative_returns) * np.sqrt(252) if negative_returns else 0
            sortino_ratio = annualized_return / downside_std if downside_std > 0 else 0
        else:
            annualized_return = 0
            annualized_volatility = 0
            sharpe_ratio = 0
            sortino_ratio = 0

        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0

        total_commission = sum(o.commission for o in self._orders if o.status == "filled")
        total_slippage = sum(o.slippage * o.quantity for o in self._orders if o.status == "filled")

        return BacktestResult(
            backtest_id=str(uuid4()),
            config=self.config,
            start_time=self.config.start_date,
            end_time=self.config.end_date,
            initial_capital=self.config.initial_capital,
            final_equity=final_equity,
            total_return=total_return,
            total_return_pct=total_return_pct,
            trades=self._trades if self.config.save_trades else [],
            orders=self._orders,
            portfolio_history=self._portfolio_history,
            total_trades=len(closed_trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=float(avg_win),
            avg_loss=float(avg_loss),
            largest_win=float(largest_win),
            largest_loss=float(largest_loss),
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            annualized_return=annualized_return,
            annualized_volatility=annualized_volatility,
            total_commission=total_commission,
            total_slippage=total_slippage,
            execution_time_seconds=execution_time,
        )


def create_backtest_engine(
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 100000.0,
    config: dict | None = None,
) -> BacktestEngine:
    """
    Create a configured backtest engine.

    Args:
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital
        config: Additional configuration

    Returns:
        BacktestEngine instance
    """
    backtest_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        **(config or {}),
    )

    return BacktestEngine(config=backtest_config)
