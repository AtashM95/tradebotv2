"""
Position Management for Ultimate Trading Bot v2.2.

This module provides comprehensive position tracking and management
including P&L calculation, position sizing, and portfolio analytics.
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, Field

from src.execution.base_executor import (
    BaseExecutor,
    Order,
    OrderSide,
    OrderType,
    Position,
    Fill,
)


logger = logging.getLogger(__name__)


class PositionAction(str, Enum):
    """Position action types."""

    OPEN = "open"
    ADD = "add"
    REDUCE = "reduce"
    CLOSE = "close"
    REVERSE = "reverse"


class PnLType(str, Enum):
    """P&L calculation type."""

    FIFO = "fifo"
    LIFO = "lifo"
    AVERAGE_COST = "average_cost"


class PositionManagerConfig(BaseModel):
    """Configuration for position manager."""

    model_config = {"arbitrary_types_allowed": True}

    pnl_method: PnLType = Field(default=PnLType.AVERAGE_COST, description="P&L calculation method")
    update_interval_seconds: float = Field(default=1.0, description="Position update interval")
    track_intraday_high_low: bool = Field(default=True, description="Track intraday high/low")
    calculate_unrealized_continuously: bool = Field(default=True, description="Continuous unrealized P&L")
    enable_position_alerts: bool = Field(default=True, description="Enable position alerts")
    pnl_alert_threshold: float = Field(default=0.05, description="P&L alert threshold %")
    max_positions: int = Field(default=50, description="Maximum positions")


class PositionEntry(BaseModel):
    """Record of a position entry/exit."""

    entry_id: str
    symbol: str
    action: PositionAction
    quantity: float
    price: float
    timestamp: datetime = Field(default_factory=datetime.now)
    order_id: str | None = None
    commission: float = Field(default=0.0)
    fees: float = Field(default=0.0)


class EnhancedPosition(BaseModel):
    """Enhanced position with detailed tracking."""

    model_config = {"arbitrary_types_allowed": True}

    symbol: str
    quantity: float = Field(default=0.0)
    avg_entry_price: float = Field(default=0.0)
    current_price: float = Field(default=0.0)

    market_value: float = Field(default=0.0)
    cost_basis: float = Field(default=0.0)

    unrealized_pnl: float = Field(default=0.0)
    unrealized_pnl_pct: float = Field(default=0.0)
    realized_pnl: float = Field(default=0.0)
    total_pnl: float = Field(default=0.0)

    intraday_high: float = Field(default=0.0)
    intraday_low: float = Field(default=float("inf"))
    intraday_pnl: float = Field(default=0.0)

    day_change: float = Field(default=0.0)
    day_change_pct: float = Field(default=0.0)

    opened_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)

    weight: float = Field(default=0.0)
    beta: float = Field(default=1.0)
    sector: str = Field(default="Unknown")
    asset_class: str = Field(default="equity")

    entries: list[PositionEntry] = Field(default_factory=list)
    total_commission: float = Field(default=0.0)
    total_fees: float = Field(default=0.0)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        """Check if position is flat."""
        return self.quantity == 0

    def calculate_metrics(self) -> None:
        """Calculate position metrics."""
        self.cost_basis = abs(self.quantity) * self.avg_entry_price
        self.market_value = self.quantity * self.current_price

        if self.is_long:
            self.unrealized_pnl = self.market_value - self.cost_basis
        else:
            self.unrealized_pnl = self.cost_basis - abs(self.market_value)

        if self.cost_basis > 0:
            self.unrealized_pnl_pct = self.unrealized_pnl / self.cost_basis

        self.total_pnl = self.unrealized_pnl + self.realized_pnl


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio state."""

    timestamp: datetime
    total_value: float
    cash: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    day_pnl: float
    position_count: int


@dataclass
class PositionAlert:
    """Position-related alert."""

    symbol: str
    alert_type: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    threshold: float = 0.0
    current_value: float = 0.0
    severity: str = "info"


class PositionManager:
    """
    Manages positions and P&L tracking.

    Provides real-time position monitoring, P&L calculation,
    and portfolio analytics.
    """

    def __init__(
        self,
        executor: BaseExecutor,
        config: PositionManagerConfig | None = None,
    ):
        """
        Initialize position manager.

        Args:
            executor: Order executor to use
            config: Position manager configuration
        """
        self.executor = executor
        self.config = config or PositionManagerConfig()

        self._positions: dict[str, EnhancedPosition] = {}
        self._closed_positions: list[EnhancedPosition] = []
        self._snapshots: list[PortfolioSnapshot] = []
        self._alerts: list[PositionAlert] = []

        self._portfolio_value: float = 0.0
        self._cash: float = 0.0
        self._day_start_value: float = 0.0

        self._callbacks: dict[str, list[Callable]] = {
            "position_opened": [],
            "position_closed": [],
            "position_updated": [],
            "pnl_alert": [],
        }

        self._updating = False
        self._update_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

        logger.info("PositionManager initialized")

    async def start(self) -> None:
        """Start position monitoring."""
        self._updating = True
        self._update_task = asyncio.create_task(self._update_loop())

        await self.sync_positions()

        account = await self.executor.get_account_info()
        self._portfolio_value = account.get("equity", 0)
        self._cash = account.get("cash", 0)
        self._day_start_value = self._portfolio_value

        logger.info("PositionManager started")

    async def stop(self) -> None:
        """Stop position monitoring."""
        self._updating = False

        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

        logger.info("PositionManager stopped")

    async def sync_positions(self) -> None:
        """Sync positions with executor."""
        try:
            executor_positions = await self.executor.get_positions()

            current_symbols = set(self._positions.keys())
            executor_symbols = set(p.symbol for p in executor_positions)

            for symbol in current_symbols - executor_symbols:
                await self._close_position(symbol)

            for pos in executor_positions:
                if pos.symbol in self._positions:
                    await self._update_position_from_executor(pos)
                else:
                    await self._create_position_from_executor(pos)

            logger.debug(f"Synced {len(executor_positions)} positions")

        except Exception as e:
            logger.error(f"Failed to sync positions: {e}")

    async def _create_position_from_executor(self, pos: Position) -> None:
        """Create internal position from executor position."""
        enhanced = EnhancedPosition(
            symbol=pos.symbol,
            quantity=pos.quantity,
            avg_entry_price=pos.avg_entry_price,
            current_price=pos.current_price,
            market_value=pos.market_value,
            cost_basis=pos.cost_basis,
            unrealized_pnl=pos.unrealized_pnl,
            unrealized_pnl_pct=pos.unrealized_pnl_pct,
            realized_pnl=pos.realized_pnl,
            asset_class=pos.asset_class,
        )
        enhanced.calculate_metrics()

        async with self._lock:
            self._positions[pos.symbol] = enhanced

        await self._trigger_callbacks("position_opened", enhanced)

    async def _update_position_from_executor(self, pos: Position) -> None:
        """Update internal position from executor."""
        async with self._lock:
            if pos.symbol not in self._positions:
                return

            enhanced = self._positions[pos.symbol]
            enhanced.quantity = pos.quantity
            enhanced.current_price = pos.current_price
            enhanced.market_value = pos.market_value

            if self.config.track_intraday_high_low:
                enhanced.intraday_high = max(enhanced.intraday_high, pos.current_price)
                enhanced.intraday_low = min(enhanced.intraday_low, pos.current_price)

            enhanced.calculate_metrics()
            enhanced.last_updated = datetime.now()

        await self._check_position_alerts(enhanced)
        await self._trigger_callbacks("position_updated", enhanced)

    async def _close_position(self, symbol: str) -> None:
        """Mark position as closed."""
        async with self._lock:
            if symbol not in self._positions:
                return

            position = self._positions.pop(symbol)
            position.quantity = 0
            position.market_value = 0
            position.unrealized_pnl = 0

            self._closed_positions.append(position)

        await self._trigger_callbacks("position_closed", position)

    async def record_fill(self, fill: Fill) -> None:
        """
        Record a fill and update position.

        Args:
            fill: Fill to record
        """
        symbol = fill.symbol

        async with self._lock:
            if symbol not in self._positions:
                self._positions[symbol] = EnhancedPosition(
                    symbol=symbol,
                    opened_at=datetime.now(),
                )

            position = self._positions[symbol]

            if fill.side == OrderSide.BUY:
                action = PositionAction.OPEN if position.quantity == 0 else PositionAction.ADD
                if position.quantity < 0:
                    action = PositionAction.REDUCE if fill.quantity < abs(position.quantity) else PositionAction.REVERSE
            else:
                action = PositionAction.REDUCE if position.quantity > fill.quantity else PositionAction.CLOSE
                if position.quantity < 0:
                    action = PositionAction.ADD

            entry = PositionEntry(
                entry_id=fill.fill_id,
                symbol=symbol,
                action=action,
                quantity=fill.quantity,
                price=fill.price,
                order_id=fill.order_id,
                commission=fill.commission,
                fees=fill.fees,
            )
            position.entries.append(entry)
            position.total_commission += fill.commission
            position.total_fees += fill.fees

            self._update_position_from_fill(position, fill)

            if position.quantity == 0:
                self._closed_positions.append(position)
                del self._positions[symbol]
                await self._trigger_callbacks("position_closed", position)
            else:
                position.calculate_metrics()
                await self._trigger_callbacks("position_updated", position)

    def _update_position_from_fill(
        self,
        position: EnhancedPosition,
        fill: Fill,
    ) -> None:
        """Update position state from fill using configured P&L method."""
        old_qty = position.quantity
        fill_qty = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity

        new_qty = old_qty + fill_qty

        if self.config.pnl_method == PnLType.AVERAGE_COST:
            if old_qty >= 0 and fill_qty > 0:
                total_cost = position.avg_entry_price * old_qty + fill.price * fill_qty
                position.avg_entry_price = total_cost / new_qty if new_qty != 0 else 0
            elif old_qty <= 0 and fill_qty < 0:
                total_cost = position.avg_entry_price * abs(old_qty) + fill.price * abs(fill_qty)
                position.avg_entry_price = total_cost / abs(new_qty) if new_qty != 0 else 0
            else:
                closing_qty = min(abs(fill_qty), abs(old_qty))
                realized = closing_qty * (fill.price - position.avg_entry_price)
                if old_qty < 0:
                    realized = -realized
                position.realized_pnl += realized

                if abs(fill_qty) > abs(old_qty):
                    position.avg_entry_price = fill.price

        position.quantity = new_qty
        position.current_price = fill.price
        position.last_updated = datetime.now()

    async def _update_loop(self) -> None:
        """Continuous position update loop."""
        while self._updating:
            try:
                await self.sync_positions()

                account = await self.executor.get_account_info()
                self._portfolio_value = account.get("equity", 0)
                self._cash = account.get("cash", 0)

                await self._update_weights()

                self._record_snapshot()

                await asyncio.sleep(self.config.update_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in position update loop: {e}")
                await asyncio.sleep(5.0)

    async def _update_weights(self) -> None:
        """Update position weights."""
        if self._portfolio_value <= 0:
            return

        async with self._lock:
            for position in self._positions.values():
                position.weight = position.market_value / self._portfolio_value

    def _record_snapshot(self) -> None:
        """Record portfolio snapshot."""
        positions_value = sum(p.market_value for p in self._positions.values())
        unrealized_pnl = sum(p.unrealized_pnl for p in self._positions.values())
        realized_pnl = sum(p.realized_pnl for p in self._positions.values())

        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            total_value=self._portfolio_value,
            cash=self._cash,
            positions_value=positions_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            day_pnl=self._portfolio_value - self._day_start_value,
            position_count=len(self._positions),
        )

        self._snapshots.append(snapshot)

        cutoff = datetime.now() - timedelta(hours=24)
        self._snapshots = [s for s in self._snapshots if s.timestamp > cutoff]

    async def _check_position_alerts(self, position: EnhancedPosition) -> None:
        """Check and generate position alerts."""
        if not self.config.enable_position_alerts:
            return

        if abs(position.unrealized_pnl_pct) >= self.config.pnl_alert_threshold:
            alert = PositionAlert(
                symbol=position.symbol,
                alert_type="pnl_threshold",
                message=f"{position.symbol} unrealized P&L at {position.unrealized_pnl_pct:.1%}",
                threshold=self.config.pnl_alert_threshold,
                current_value=position.unrealized_pnl_pct,
                severity="warning" if position.unrealized_pnl_pct < 0 else "info",
            )
            self._alerts.append(alert)
            await self._trigger_callbacks("pnl_alert", alert)

    async def close_position(
        self,
        symbol: str,
        quantity: float | None = None,
    ) -> Order | None:
        """
        Close a position.

        Args:
            symbol: Symbol to close
            quantity: Quantity to close (None for full)

        Returns:
            Close order if submitted
        """
        position = self._positions.get(symbol)
        if not position or position.is_flat:
            return None

        close_qty = quantity or abs(position.quantity)
        close_side = OrderSide.SELL if position.is_long else OrderSide.BUY

        order = await self.executor.create_order(
            symbol=symbol,
            side=close_side,
            quantity=close_qty,
            order_type=OrderType.MARKET,
        )

        result = await self.executor.submit_order(order)

        if result.success:
            for fill in result.fills:
                await self.record_fill(fill)
            return order

        return None

    async def close_all_positions(self) -> list[Order]:
        """
        Close all positions.

        Returns:
            List of close orders
        """
        orders: list[Order] = []

        for symbol in list(self._positions.keys()):
            order = await self.close_position(symbol)
            if order:
                orders.append(order)

        return orders

    def get_position(self, symbol: str) -> EnhancedPosition | None:
        """Get position by symbol."""
        return self._positions.get(symbol)

    def get_all_positions(self) -> list[EnhancedPosition]:
        """Get all positions."""
        return list(self._positions.values())

    def get_long_positions(self) -> list[EnhancedPosition]:
        """Get all long positions."""
        return [p for p in self._positions.values() if p.is_long]

    def get_short_positions(self) -> list[EnhancedPosition]:
        """Get all short positions."""
        return [p for p in self._positions.values() if p.is_short]

    def get_closed_positions(self, limit: int = 100) -> list[EnhancedPosition]:
        """Get recently closed positions."""
        return self._closed_positions[-limit:]

    def get_alerts(self, limit: int = 50) -> list[PositionAlert]:
        """Get recent position alerts."""
        return self._alerts[-limit:]

    def get_snapshots(self, limit: int = 100) -> list[PortfolioSnapshot]:
        """Get recent portfolio snapshots."""
        return self._snapshots[-limit:]

    def register_callback(
        self,
        event: str,
        callback: Callable,
    ) -> None:
        """Register callback for position events."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    async def _trigger_callbacks(
        self,
        event: str,
        data: Any,
    ) -> None:
        """Trigger callbacks for an event."""
        for callback in self._callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")

    async def get_portfolio_summary(self) -> dict[str, Any]:
        """
        Get portfolio summary.

        Returns:
            Portfolio summary dictionary
        """
        positions = list(self._positions.values())

        long_value = sum(p.market_value for p in positions if p.is_long)
        short_value = sum(abs(p.market_value) for p in positions if p.is_short)
        gross_exposure = long_value + short_value
        net_exposure = long_value - short_value

        unrealized = sum(p.unrealized_pnl for p in positions)
        realized = sum(p.realized_pnl for p in positions)
        total_commission = sum(p.total_commission for p in positions)

        sector_exposure: dict[str, float] = defaultdict(float)
        for p in positions:
            sector_exposure[p.sector] += p.market_value

        return {
            "timestamp": datetime.now().isoformat(),
            "portfolio_value": self._portfolio_value,
            "cash": self._cash,
            "positions_count": len(positions),
            "long_count": sum(1 for p in positions if p.is_long),
            "short_count": sum(1 for p in positions if p.is_short),
            "exposure": {
                "long": long_value,
                "short": short_value,
                "gross": gross_exposure,
                "net": net_exposure,
                "gross_pct": gross_exposure / self._portfolio_value if self._portfolio_value > 0 else 0,
                "net_pct": net_exposure / self._portfolio_value if self._portfolio_value > 0 else 0,
            },
            "pnl": {
                "unrealized": unrealized,
                "realized": realized,
                "total": unrealized + realized,
                "day": self._portfolio_value - self._day_start_value,
                "total_commission": total_commission,
            },
            "sector_exposure": dict(sector_exposure),
            "top_positions": [
                {
                    "symbol": p.symbol,
                    "value": p.market_value,
                    "weight": p.weight,
                    "pnl": p.unrealized_pnl,
                    "pnl_pct": p.unrealized_pnl_pct,
                }
                for p in sorted(positions, key=lambda x: abs(x.market_value), reverse=True)[:10]
            ],
        }

    async def get_position_details(self, symbol: str) -> dict[str, Any] | None:
        """
        Get detailed position information.

        Args:
            symbol: Position symbol

        Returns:
            Position details dictionary
        """
        position = self._positions.get(symbol)
        if not position:
            return None

        return {
            "symbol": position.symbol,
            "quantity": position.quantity,
            "side": "long" if position.is_long else "short",
            "avg_entry_price": position.avg_entry_price,
            "current_price": position.current_price,
            "market_value": position.market_value,
            "cost_basis": position.cost_basis,
            "pnl": {
                "unrealized": position.unrealized_pnl,
                "unrealized_pct": position.unrealized_pnl_pct,
                "realized": position.realized_pnl,
                "total": position.total_pnl,
                "intraday": position.intraday_pnl,
            },
            "intraday": {
                "high": position.intraday_high,
                "low": position.intraday_low,
            },
            "weight": position.weight,
            "beta": position.beta,
            "sector": position.sector,
            "opened_at": position.opened_at.isoformat(),
            "last_updated": position.last_updated.isoformat(),
            "entry_count": len(position.entries),
            "total_commission": position.total_commission,
        }
