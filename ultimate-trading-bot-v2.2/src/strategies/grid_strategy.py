"""
Grid Trading Strategy Module for Ultimate Trading Bot v2.2.

This module implements an automated grid trading strategy
that places buy and sell orders at predetermined price levels.
"""

import logging
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.strategies.base_strategy import (
    BaseStrategy,
    StrategyConfig,
    StrategySignal,
    SignalAction,
    SignalSide,
    MarketData,
    StrategyContext,
)
from src.analysis.technical_indicators import TechnicalIndicators
from src.utils.helpers import generate_uuid


logger = logging.getLogger(__name__)


class GridLevel(BaseModel):
    """Model for a grid level."""

    level_id: str = Field(default_factory=generate_uuid)
    price: float
    type: str
    is_filled: bool = Field(default=False)
    filled_at: Optional[datetime] = None
    quantity: float = Field(default=0.0)


class GridInstance(BaseModel):
    """Model for a grid instance on a symbol."""

    symbol: str
    grid_type: str
    upper_price: float
    lower_price: float
    num_grids: int
    grid_spacing: float
    levels: list[GridLevel] = Field(default_factory=list)
    created_at: datetime
    total_invested: float = Field(default=0.0)
    total_profit: float = Field(default=0.0)
    trades_count: int = Field(default=0)


class GridStrategyConfig(StrategyConfig):
    """Configuration for grid trading strategy."""

    name: str = Field(default="Grid Trading Strategy")
    description: str = Field(
        default="Automated grid trading with buy/sell at predetermined levels"
    )

    grid_type: str = Field(default="arithmetic")
    num_grids: int = Field(default=10, ge=3, le=50)

    upper_price_pct: float = Field(default=0.1, ge=0.02, le=0.5)
    lower_price_pct: float = Field(default=0.1, ge=0.02, le=0.5)

    auto_adjust_range: bool = Field(default=True)
    atr_multiplier: float = Field(default=2.0, ge=1.0, le=5.0)

    investment_per_grid: float = Field(default=0.01, ge=0.001, le=0.1)
    total_investment_pct: float = Field(default=0.2, ge=0.05, le=0.5)

    trigger_distance_pct: float = Field(default=0.002, ge=0.001, le=0.01)

    take_profit_pct: Optional[float] = Field(default=None, ge=0.05, le=0.5)
    stop_loss_pct: Optional[float] = Field(default=None, ge=0.05, le=0.3)

    trailing_grid: bool = Field(default=False)
    trail_trigger_pct: float = Field(default=0.05, ge=0.02, le=0.2)


class GridTradingStrategy(BaseStrategy):
    """
    Automated grid trading strategy.

    Features:
    - Arithmetic or geometric grid spacing
    - Auto-adjusting grid range based on volatility
    - Multiple grid instances per symbol
    - Profit tracking per grid
    - Optional trailing grid
    """

    def __init__(
        self,
        config: Optional[GridStrategyConfig] = None,
    ) -> None:
        """
        Initialize GridTradingStrategy.

        Args:
            config: Grid trading configuration
        """
        config = config or GridStrategyConfig()
        super().__init__(config)

        self._grid_config = config
        self._indicators = TechnicalIndicators()

        self._grids: dict[str, GridInstance] = {}
        self._pending_signals: dict[str, list[StrategySignal]] = {}

        logger.info(f"GridTradingStrategy initialized: {self.name}")

    def calculate_indicators(
        self,
        symbol: str,
        data: MarketData,
    ) -> dict[str, Any]:
        """
        Calculate indicators for grid setup.

        Args:
            symbol: Trading symbol
            data: Market data

        Returns:
            Dictionary of indicator values
        """
        closes = data.closes
        highs = data.highs
        lows = data.lows

        if len(closes) < 20:
            return {}

        atr = self._indicators.atr(highs, lows, closes, 14)
        current_atr = atr[-1] if atr else 0

        current_price = closes[-1]
        atr_pct = current_atr / current_price if current_price > 0 else 0

        high_20 = max(highs[-20:])
        low_20 = min(lows[-20:])
        range_20 = (high_20 - low_20) / current_price

        sma_20 = sum(closes[-20:]) / 20

        returns = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]
        volatility = (sum(r ** 2 for r in returns[-20:]) / 20) ** 0.5 if len(returns) >= 20 else 0

        return {
            "current_price": current_price,
            "atr": current_atr,
            "atr_pct": atr_pct,
            "high_20": high_20,
            "low_20": low_20,
            "range_20": range_20,
            "sma_20": sma_20,
            "volatility": volatility,
        }

    async def evaluate(
        self,
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """
        Evaluate grid trading opportunities.

        Args:
            market_data: Market data for all symbols
            context: Strategy execution context

        Returns:
            List of grid trading signals
        """
        signals: list[StrategySignal] = []

        for symbol in self.config.symbols:
            if symbol not in market_data:
                continue

            data = market_data[symbol]
            current_price = data.closes[-1]

            if symbol not in self._grids:
                indicators = self.calculate_indicators(symbol, data)
                if indicators:
                    self._setup_grid(symbol, indicators, context)

            if symbol in self._grids:
                grid_signals = self._evaluate_grid(
                    symbol, current_price, context
                )
                signals.extend(grid_signals)

                if self._grid_config.trailing_grid:
                    self._check_trail_grid(symbol, current_price)

                if self._grid_config.take_profit_pct:
                    tp_signal = self._check_take_profit(symbol, current_price)
                    if tp_signal:
                        signals.append(tp_signal)

                if self._grid_config.stop_loss_pct:
                    sl_signal = self._check_stop_loss(symbol, current_price)
                    if sl_signal:
                        signals.append(sl_signal)

        return signals

    def _setup_grid(
        self,
        symbol: str,
        indicators: dict[str, Any],
        context: StrategyContext,
    ) -> None:
        """Set up a new grid for a symbol."""
        current_price = indicators["current_price"]

        if self._grid_config.auto_adjust_range:
            atr_pct = indicators["atr_pct"]
            range_mult = self._grid_config.atr_multiplier

            upper_pct = max(self._grid_config.upper_price_pct, atr_pct * range_mult)
            lower_pct = max(self._grid_config.lower_price_pct, atr_pct * range_mult)
        else:
            upper_pct = self._grid_config.upper_price_pct
            lower_pct = self._grid_config.lower_price_pct

        upper_price = current_price * (1 + upper_pct)
        lower_price = current_price * (1 - lower_pct)

        levels = self._create_grid_levels(
            current_price, upper_price, lower_price,
            self._grid_config.num_grids,
            self._grid_config.grid_type,
        )

        grid_spacing = (upper_price - lower_price) / self._grid_config.num_grids

        grid = GridInstance(
            symbol=symbol,
            grid_type=self._grid_config.grid_type,
            upper_price=upper_price,
            lower_price=lower_price,
            num_grids=self._grid_config.num_grids,
            grid_spacing=grid_spacing,
            levels=levels,
            created_at=context.timestamp,
        )

        self._grids[symbol] = grid

        logger.info(
            f"Grid setup for {symbol}: "
            f"range [{lower_price:.2f} - {upper_price:.2f}], "
            f"{len(levels)} levels"
        )

    def _create_grid_levels(
        self,
        current_price: float,
        upper_price: float,
        lower_price: float,
        num_grids: int,
        grid_type: str,
    ) -> list[GridLevel]:
        """Create grid levels."""
        levels: list[GridLevel] = []

        if grid_type == "geometric":
            ratio = (upper_price / lower_price) ** (1 / num_grids)
            prices = [lower_price * (ratio ** i) for i in range(num_grids + 1)]
        else:
            spacing = (upper_price - lower_price) / num_grids
            prices = [lower_price + spacing * i for i in range(num_grids + 1)]

        for price in prices:
            level_type = "buy" if price < current_price else "sell"

            levels.append(GridLevel(
                price=price,
                type=level_type,
            ))

        return levels

    def _evaluate_grid(
        self,
        symbol: str,
        current_price: float,
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """Evaluate grid levels and generate signals."""
        grid = self._grids[symbol]
        signals: list[StrategySignal] = []

        trigger_distance = current_price * self._grid_config.trigger_distance_pct

        for level in grid.levels:
            if level.is_filled:
                continue

            price_diff = abs(current_price - level.price)

            if price_diff <= trigger_distance:
                signal = self._create_grid_signal(
                    symbol, level, current_price, grid, context
                )

                if signal:
                    signals.append(signal)
                    level.is_filled = True
                    level.filled_at = context.timestamp
                    grid.trades_count += 1

                    self._update_adjacent_levels(grid, level)

        return signals

    def _create_grid_signal(
        self,
        symbol: str,
        level: GridLevel,
        current_price: float,
        grid: GridInstance,
        context: StrategyContext,
    ) -> Optional[StrategySignal]:
        """Create trading signal for grid level."""
        position_size = self._grid_config.investment_per_grid

        if level.type == "buy":
            return self.create_signal(
                symbol=symbol,
                action=SignalAction.BUY,
                side=SignalSide.LONG,
                entry_price=current_price,
                strength=0.6,
                confidence=0.75,
                reason=f"Grid buy at level {level.price:.2f}",
                position_size_pct=position_size,
                metadata={
                    "strategy_type": "grid_trading",
                    "grid_level": level.price,
                    "level_type": level.type,
                    "grid_id": grid.symbol,
                },
            )
        else:
            return self.create_signal(
                symbol=symbol,
                action=SignalAction.SELL,
                side=SignalSide.SHORT,
                entry_price=current_price,
                strength=0.6,
                confidence=0.75,
                reason=f"Grid sell at level {level.price:.2f}",
                position_size_pct=position_size,
                metadata={
                    "strategy_type": "grid_trading",
                    "grid_level": level.price,
                    "level_type": level.type,
                    "grid_id": grid.symbol,
                },
            )

    def _update_adjacent_levels(
        self,
        grid: GridInstance,
        filled_level: GridLevel,
    ) -> None:
        """Update adjacent levels after a fill."""
        sorted_levels = sorted(grid.levels, key=lambda l: l.price)

        for i, level in enumerate(sorted_levels):
            if level.level_id == filled_level.level_id:
                if filled_level.type == "buy" and i < len(sorted_levels) - 1:
                    sorted_levels[i + 1].type = "sell"
                    sorted_levels[i + 1].is_filled = False

                elif filled_level.type == "sell" and i > 0:
                    sorted_levels[i - 1].type = "buy"
                    sorted_levels[i - 1].is_filled = False

                break

    def _check_trail_grid(self, symbol: str, current_price: float) -> None:
        """Check if grid should be trailed."""
        grid = self._grids[symbol]

        mid_price = (grid.upper_price + grid.lower_price) / 2

        if current_price > grid.upper_price * (1 + self._grid_config.trail_trigger_pct):
            shift = current_price - grid.upper_price
            grid.upper_price += shift
            grid.lower_price += shift

            for level in grid.levels:
                level.price += shift
                level.is_filled = False

            logger.info(f"Grid trailed up for {symbol}: new range [{grid.lower_price:.2f} - {grid.upper_price:.2f}]")

        elif current_price < grid.lower_price * (1 - self._grid_config.trail_trigger_pct):
            shift = grid.lower_price - current_price
            grid.upper_price -= shift
            grid.lower_price -= shift

            for level in grid.levels:
                level.price -= shift
                level.is_filled = False

            logger.info(f"Grid trailed down for {symbol}: new range [{grid.lower_price:.2f} - {grid.upper_price:.2f}]")

    def _check_take_profit(
        self,
        symbol: str,
        current_price: float,
    ) -> Optional[StrategySignal]:
        """Check take profit condition."""
        grid = self._grids[symbol]

        if not self._grid_config.take_profit_pct:
            return None

        mid_price = (grid.upper_price + grid.lower_price) / 2
        tp_price = mid_price * (1 + self._grid_config.take_profit_pct)

        if current_price >= tp_price:
            del self._grids[symbol]

            return self.create_signal(
                symbol=symbol,
                action=SignalAction.SELL,
                side=SignalSide.FLAT,
                entry_price=current_price,
                strength=1.0,
                confidence=0.95,
                reason=f"Grid take profit at {current_price:.2f}",
                metadata={
                    "strategy_type": "grid_trading",
                    "exit_reason": "take_profit",
                    "trades_count": grid.trades_count,
                },
            )

        return None

    def _check_stop_loss(
        self,
        symbol: str,
        current_price: float,
    ) -> Optional[StrategySignal]:
        """Check stop loss condition."""
        grid = self._grids[symbol]

        if not self._grid_config.stop_loss_pct:
            return None

        sl_price = grid.lower_price * (1 - self._grid_config.stop_loss_pct)

        if current_price <= sl_price:
            del self._grids[symbol]

            return self.create_signal(
                symbol=symbol,
                action=SignalAction.SELL,
                side=SignalSide.FLAT,
                entry_price=current_price,
                strength=1.0,
                confidence=0.95,
                reason=f"Grid stop loss at {current_price:.2f}",
                metadata={
                    "strategy_type": "grid_trading",
                    "exit_reason": "stop_loss",
                    "trades_count": grid.trades_count,
                },
            )

        return None

    def reset_grid(self, symbol: str) -> bool:
        """Reset grid for a symbol."""
        if symbol in self._grids:
            del self._grids[symbol]
            logger.info(f"Grid reset for {symbol}")
            return True
        return False

    def get_grid(self, symbol: str) -> Optional[GridInstance]:
        """Get grid instance for a symbol."""
        return self._grids.get(symbol)

    def get_all_grids(self) -> dict[str, GridInstance]:
        """Get all grid instances."""
        return self._grids.copy()

    def get_grid_statistics(self) -> dict:
        """Get grid trading statistics."""
        total_trades = sum(g.trades_count for g in self._grids.values())
        total_profit = sum(g.total_profit for g in self._grids.values())

        return {
            "active_grids": len(self._grids),
            "total_trades": total_trades,
            "total_profit": total_profit,
            "grids": {
                symbol: {
                    "upper_price": grid.upper_price,
                    "lower_price": grid.lower_price,
                    "num_grids": grid.num_grids,
                    "trades_count": grid.trades_count,
                    "filled_levels": sum(1 for l in grid.levels if l.is_filled),
                }
                for symbol, grid in self._grids.items()
            },
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"GridTradingStrategy(grids={len(self._grids)})"
