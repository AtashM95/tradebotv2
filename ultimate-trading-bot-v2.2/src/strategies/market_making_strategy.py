"""
Market Making Strategy Module for Ultimate Trading Bot v2.2.

This module implements automated market making with dynamic
spread adjustment and inventory management.
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


class Quote(BaseModel):
    """Model for a market maker quote."""

    quote_id: str = Field(default_factory=generate_uuid)
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    spread: float
    spread_pct: float
    created_at: datetime


class InventoryPosition(BaseModel):
    """Model for inventory position tracking."""

    symbol: str
    quantity: float = Field(default=0.0)
    avg_price: float = Field(default=0.0)
    max_quantity: float = Field(default=1000.0)
    target_quantity: float = Field(default=0.0)
    pnl: float = Field(default=0.0)


class MarketMakingConfig(StrategyConfig):
    """Configuration for market making strategy."""

    name: str = Field(default="Market Making Strategy")
    description: str = Field(
        default="Automated market making with inventory management"
    )

    base_spread_pct: float = Field(default=0.002, ge=0.0005, le=0.02)
    min_spread_pct: float = Field(default=0.001, ge=0.0002, le=0.01)
    max_spread_pct: float = Field(default=0.01, ge=0.002, le=0.05)

    volatility_multiplier: float = Field(default=2.0, ge=0.5, le=5.0)
    inventory_skew_factor: float = Field(default=0.5, ge=0.1, le=2.0)

    max_inventory_pct: float = Field(default=0.1, ge=0.01, le=0.5)
    target_inventory_pct: float = Field(default=0.0, ge=-0.5, le=0.5)
    inventory_decay_rate: float = Field(default=0.1, ge=0.01, le=0.5)

    order_size_pct: float = Field(default=0.01, ge=0.001, le=0.1)
    size_scaling_factor: float = Field(default=1.5, ge=1.0, le=3.0)

    quote_refresh_seconds: int = Field(default=5, ge=1, le=60)
    max_quote_age_seconds: int = Field(default=30, ge=5, le=120)

    use_fair_value_model: bool = Field(default=True)
    fair_value_ema_period: int = Field(default=20, ge=5, le=50)

    enable_inventory_hedging: bool = Field(default=True)
    hedge_threshold_pct: float = Field(default=0.7, ge=0.3, le=1.0)


class MarketMakingStrategy(BaseStrategy):
    """
    Automated market making strategy.

    Features:
    - Dynamic spread calculation
    - Volatility-adjusted pricing
    - Inventory risk management
    - Fair value estimation
    - Quote management
    """

    def __init__(
        self,
        config: Optional[MarketMakingConfig] = None,
    ) -> None:
        """
        Initialize MarketMakingStrategy.

        Args:
            config: Market making configuration
        """
        config = config or MarketMakingConfig()
        super().__init__(config)

        self._mm_config = config
        self._indicators = TechnicalIndicators()

        self._inventory: dict[str, InventoryPosition] = {}
        self._active_quotes: dict[str, Quote] = {}
        self._quote_history: list[Quote] = []
        self._last_quote_time: dict[str, datetime] = {}

        logger.info(f"MarketMakingStrategy initialized: {self.name}")

    def calculate_indicators(
        self,
        symbol: str,
        data: MarketData,
    ) -> dict[str, Any]:
        """
        Calculate indicators for market making.

        Args:
            symbol: Trading symbol
            data: Market data

        Returns:
            Dictionary of indicator values
        """
        closes = data.closes
        highs = data.highs
        lows = data.lows
        volumes = data.volumes

        if len(closes) < 30:
            return {}

        current_price = closes[-1]

        if self._mm_config.use_fair_value_model:
            ema = self._indicators.ema(closes, self._mm_config.fair_value_ema_period)
            fair_value = ema[-1] if ema else current_price
        else:
            fair_value = current_price

        atr = self._indicators.atr(highs, lows, closes, 14)
        current_atr = atr[-1] if atr else current_price * 0.01
        atr_pct = current_atr / current_price if current_price > 0 else 0.01

        returns = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]
        volatility = (sum(r ** 2 for r in returns[-20:]) / 20) ** 0.5 if len(returns) >= 20 else 0.01

        avg_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else volumes[-1]
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        bid_ask_proxy = (highs[-1] - lows[-1]) / current_price * 0.3

        return {
            "current_price": current_price,
            "fair_value": fair_value,
            "atr": current_atr,
            "atr_pct": atr_pct,
            "volatility": volatility,
            "volume_ratio": volume_ratio,
            "bid_ask_proxy": bid_ask_proxy,
            "high": highs[-1],
            "low": lows[-1],
        }

    async def evaluate(
        self,
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """
        Evaluate market making opportunities.

        Args:
            market_data: Market data for all symbols
            context: Strategy execution context

        Returns:
            List of market making signals
        """
        signals: list[StrategySignal] = []

        for symbol in self.config.symbols:
            if symbol not in market_data:
                continue

            data = market_data[symbol]

            indicators = self.calculate_indicators(symbol, data)
            if not indicators:
                continue

            if self._should_refresh_quote(symbol, context):
                quote = self._generate_quote(symbol, indicators, context)

                if quote:
                    self._active_quotes[symbol] = quote
                    self._quote_history.append(quote)
                    self._last_quote_time[symbol] = context.timestamp

                    quote_signals = self._create_quote_signals(symbol, quote, context)
                    signals.extend(quote_signals)

            if self._mm_config.enable_inventory_hedging:
                hedge_signal = self._check_inventory_hedge(symbol, indicators, context)
                if hedge_signal:
                    signals.append(hedge_signal)

        return signals

    def _should_refresh_quote(self, symbol: str, context: StrategyContext) -> bool:
        """Check if quote should be refreshed."""
        if symbol not in self._last_quote_time:
            return True

        last_time = self._last_quote_time[symbol]
        elapsed = (context.timestamp - last_time).total_seconds()

        return elapsed >= self._mm_config.quote_refresh_seconds

    def _generate_quote(
        self,
        symbol: str,
        indicators: dict[str, Any],
        context: StrategyContext,
    ) -> Quote:
        """Generate market maker quote."""
        fair_value = indicators["fair_value"]
        volatility = indicators["volatility"]
        current_price = indicators["current_price"]

        base_spread = self._mm_config.base_spread_pct
        vol_adjustment = volatility * self._mm_config.volatility_multiplier
        spread_pct = base_spread + vol_adjustment

        inventory = self._get_inventory(symbol, context)
        inventory_ratio = inventory.quantity / inventory.max_quantity if inventory.max_quantity > 0 else 0

        bid_skew = inventory_ratio * self._mm_config.inventory_skew_factor
        ask_skew = -inventory_ratio * self._mm_config.inventory_skew_factor

        half_spread = spread_pct / 2
        bid_offset = half_spread + bid_skew * spread_pct
        ask_offset = half_spread + ask_skew * spread_pct

        bid_price = fair_value * (1 - bid_offset)
        ask_price = fair_value * (1 + ask_offset)

        final_spread_pct = (ask_price - bid_price) / fair_value
        final_spread_pct = max(
            self._mm_config.min_spread_pct,
            min(self._mm_config.max_spread_pct, final_spread_pct)
        )

        half_final = final_spread_pct / 2
        bid_price = fair_value * (1 - half_final - bid_skew * half_final)
        ask_price = fair_value * (1 + half_final - ask_skew * half_final)

        base_size = context.account_value * self._mm_config.order_size_pct / current_price

        if inventory_ratio > 0:
            bid_size = base_size * (1 - abs(inventory_ratio) * 0.5)
            ask_size = base_size * (1 + abs(inventory_ratio) * 0.5)
        else:
            bid_size = base_size * (1 + abs(inventory_ratio) * 0.5)
            ask_size = base_size * (1 - abs(inventory_ratio) * 0.5)

        return Quote(
            symbol=symbol,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=max(0.01, bid_size),
            ask_size=max(0.01, ask_size),
            spread=ask_price - bid_price,
            spread_pct=final_spread_pct,
            created_at=context.timestamp,
        )

    def _create_quote_signals(
        self,
        symbol: str,
        quote: Quote,
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """Create signals from quote."""
        signals: list[StrategySignal] = []

        bid_signal = self.create_signal(
            symbol=symbol,
            action=SignalAction.BUY,
            side=SignalSide.LONG,
            entry_price=quote.bid_price,
            strength=0.5,
            confidence=0.7,
            reason=f"MM bid at {quote.bid_price:.4f}",
            position_size_pct=self._mm_config.order_size_pct,
            metadata={
                "strategy_type": "market_making",
                "order_type": "limit",
                "quote_id": quote.quote_id,
                "side": "bid",
                "spread_pct": quote.spread_pct,
                "size": quote.bid_size,
            },
        )

        ask_signal = self.create_signal(
            symbol=symbol,
            action=SignalAction.SELL,
            side=SignalSide.SHORT,
            entry_price=quote.ask_price,
            strength=0.5,
            confidence=0.7,
            reason=f"MM ask at {quote.ask_price:.4f}",
            position_size_pct=self._mm_config.order_size_pct,
            metadata={
                "strategy_type": "market_making",
                "order_type": "limit",
                "quote_id": quote.quote_id,
                "side": "ask",
                "spread_pct": quote.spread_pct,
                "size": quote.ask_size,
            },
        )

        if bid_signal:
            signals.append(bid_signal)
        if ask_signal:
            signals.append(ask_signal)

        return signals

    def _get_inventory(self, symbol: str, context: StrategyContext) -> InventoryPosition:
        """Get or create inventory position."""
        if symbol not in self._inventory:
            max_qty = context.account_value * self._mm_config.max_inventory_pct / 100
            self._inventory[symbol] = InventoryPosition(
                symbol=symbol,
                max_quantity=max_qty,
            )

        return self._inventory[symbol]

    def update_inventory(
        self,
        symbol: str,
        quantity_change: float,
        price: float,
    ) -> None:
        """Update inventory after trade execution."""
        if symbol not in self._inventory:
            self._inventory[symbol] = InventoryPosition(symbol=symbol)

        inv = self._inventory[symbol]

        if inv.quantity == 0:
            inv.avg_price = price
        elif (inv.quantity > 0 and quantity_change > 0) or (inv.quantity < 0 and quantity_change < 0):
            total_cost = inv.quantity * inv.avg_price + quantity_change * price
            inv.quantity += quantity_change
            if inv.quantity != 0:
                inv.avg_price = total_cost / inv.quantity
        else:
            pnl = quantity_change * (inv.avg_price - price) if inv.quantity > 0 else quantity_change * (price - inv.avg_price)
            inv.pnl += pnl
            inv.quantity += quantity_change

    def _check_inventory_hedge(
        self,
        symbol: str,
        indicators: dict[str, Any],
        context: StrategyContext,
    ) -> Optional[StrategySignal]:
        """Check if inventory needs hedging."""
        inventory = self._get_inventory(symbol, context)

        inventory_ratio = abs(inventory.quantity) / inventory.max_quantity if inventory.max_quantity > 0 else 0

        if inventory_ratio < self._mm_config.hedge_threshold_pct:
            return None

        current_price = indicators["current_price"]

        if inventory.quantity > 0:
            hedge_qty = inventory.quantity * self._mm_config.inventory_decay_rate

            return self.create_signal(
                symbol=symbol,
                action=SignalAction.SELL,
                side=SignalSide.FLAT,
                entry_price=current_price,
                strength=0.8,
                confidence=0.85,
                reason=f"MM inventory hedge sell {hedge_qty:.2f}",
                metadata={
                    "strategy_type": "market_making",
                    "order_type": "market",
                    "hedge": True,
                    "inventory_ratio": inventory_ratio,
                },
            )

        else:
            hedge_qty = abs(inventory.quantity) * self._mm_config.inventory_decay_rate

            return self.create_signal(
                symbol=symbol,
                action=SignalAction.BUY,
                side=SignalSide.FLAT,
                entry_price=current_price,
                strength=0.8,
                confidence=0.85,
                reason=f"MM inventory hedge buy {hedge_qty:.2f}",
                metadata={
                    "strategy_type": "market_making",
                    "order_type": "market",
                    "hedge": True,
                    "inventory_ratio": inventory_ratio,
                },
            )

    def get_active_quote(self, symbol: str) -> Optional[Quote]:
        """Get active quote for symbol."""
        return self._active_quotes.get(symbol)

    def get_inventory(self, symbol: str) -> Optional[InventoryPosition]:
        """Get inventory position for symbol."""
        return self._inventory.get(symbol)

    def get_all_inventory(self) -> dict[str, InventoryPosition]:
        """Get all inventory positions."""
        return self._inventory.copy()

    def get_quote_history(self, symbol: Optional[str] = None, limit: int = 100) -> list[Quote]:
        """Get quote history."""
        if symbol:
            quotes = [q for q in self._quote_history if q.symbol == symbol]
        else:
            quotes = self._quote_history

        return quotes[-limit:]

    def get_market_making_statistics(self) -> dict:
        """Get market making statistics."""
        total_pnl = sum(inv.pnl for inv in self._inventory.values())

        avg_spread = 0
        if self._quote_history:
            recent_quotes = self._quote_history[-100:]
            avg_spread = sum(q.spread_pct for q in recent_quotes) / len(recent_quotes)

        return {
            "active_quotes": len(self._active_quotes),
            "total_quotes": len(self._quote_history),
            "inventory_positions": len(self._inventory),
            "total_pnl": total_pnl,
            "average_spread_pct": avg_spread,
            "inventory": {
                symbol: {
                    "quantity": inv.quantity,
                    "avg_price": inv.avg_price,
                    "pnl": inv.pnl,
                    "utilization": abs(inv.quantity) / inv.max_quantity if inv.max_quantity > 0 else 0,
                }
                for symbol, inv in self._inventory.items()
            },
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"MarketMakingStrategy(quotes={len(self._active_quotes)}, inventory={len(self._inventory)})"
