"""
Strategy Manager Module for Ultimate Trading Bot v2.2.

This module provides centralized management of all trading strategies,
including registration, execution, and coordination.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional, Type

from pydantic import BaseModel, Field

from src.strategies.base_strategy import (
    BaseStrategy,
    StrategyConfig,
    StrategySignal,
    StrategyState,
    MarketData,
    StrategyContext,
    StrategyMetrics,
)
from src.utils.helpers import generate_uuid
from src.utils.date_utils import now_utc


logger = logging.getLogger(__name__)


class StrategyWeight(BaseModel):
    """Strategy weight for signal aggregation."""

    strategy_id: str
    weight: float = Field(default=1.0, ge=0.0, le=10.0)
    priority: int = Field(default=0, ge=0, le=100)
    enabled: bool = Field(default=True)


class AggregatedSignal(BaseModel):
    """Aggregated signal from multiple strategies."""

    signal_id: str = Field(default_factory=generate_uuid)
    symbol: str
    direction: str
    aggregated_strength: float = Field(ge=0.0, le=1.0)
    aggregated_confidence: float = Field(ge=0.0, le=1.0)
    contributing_strategies: list[str] = Field(default_factory=list)
    signals: list[StrategySignal] = Field(default_factory=list)
    consensus: str = Field(default="none")
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size_pct: Optional[float] = None
    created_at: datetime = Field(default_factory=now_utc)


class StrategyManagerConfig(BaseModel):
    """Configuration for strategy manager."""

    max_concurrent_signals: int = Field(default=10, ge=1, le=50)
    signal_timeout_seconds: int = Field(default=300, ge=60, le=3600)
    require_consensus: bool = Field(default=False)
    min_consensus_pct: float = Field(default=0.5, ge=0.0, le=1.0)
    aggregate_signals: bool = Field(default=True)
    use_weighted_aggregation: bool = Field(default=True)
    conflict_resolution: str = Field(default="highest_confidence")
    max_signals_per_symbol: int = Field(default=1, ge=1, le=5)


class StrategyManager:
    """
    Centralized strategy manager.

    Provides:
    - Strategy registration and lifecycle management
    - Concurrent strategy execution
    - Signal aggregation and conflict resolution
    - Performance tracking across strategies
    - Strategy weighting and prioritization
    """

    def __init__(
        self,
        config: Optional[StrategyManagerConfig] = None,
    ) -> None:
        """
        Initialize StrategyManager.

        Args:
            config: Manager configuration
        """
        self._config = config or StrategyManagerConfig()

        self._strategies: dict[str, BaseStrategy] = {}
        self._strategy_weights: dict[str, StrategyWeight] = {}
        self._strategy_types: dict[str, Type[BaseStrategy]] = {}

        self._active_signals: dict[str, list[StrategySignal]] = {}
        self._signal_history: list[StrategySignal] = []

        self._is_running = False
        self._evaluation_count = 0
        self._total_signals_generated = 0

        logger.info("StrategyManager initialized")

    def register_strategy_type(
        self,
        name: str,
        strategy_class: Type[BaseStrategy],
    ) -> None:
        """
        Register a strategy type for dynamic instantiation.

        Args:
            name: Strategy type name
            strategy_class: Strategy class
        """
        self._strategy_types[name] = strategy_class
        logger.info(f"Registered strategy type: {name}")

    def add_strategy(
        self,
        strategy: BaseStrategy,
        weight: float = 1.0,
        priority: int = 0,
    ) -> str:
        """
        Add a strategy to the manager.

        Args:
            strategy: Strategy instance
            weight: Strategy weight for aggregation
            priority: Strategy priority (higher = more important)

        Returns:
            Strategy ID
        """
        strategy_id = strategy.strategy_id

        self._strategies[strategy_id] = strategy
        self._strategy_weights[strategy_id] = StrategyWeight(
            strategy_id=strategy_id,
            weight=weight,
            priority=priority,
            enabled=strategy.config.enabled,
        )

        logger.info(f"Added strategy: {strategy.name} (ID: {strategy_id})")
        return strategy_id

    def remove_strategy(self, strategy_id: str) -> bool:
        """
        Remove a strategy from the manager.

        Args:
            strategy_id: Strategy ID to remove

        Returns:
            True if removed
        """
        if strategy_id not in self._strategies:
            return False

        strategy = self._strategies[strategy_id]

        del self._strategies[strategy_id]
        if strategy_id in self._strategy_weights:
            del self._strategy_weights[strategy_id]

        logger.info(f"Removed strategy: {strategy.name}")
        return True

    def get_strategy(self, strategy_id: str) -> Optional[BaseStrategy]:
        """Get a strategy by ID."""
        return self._strategies.get(strategy_id)

    def get_strategy_by_name(self, name: str) -> Optional[BaseStrategy]:
        """Get a strategy by name."""
        for strategy in self._strategies.values():
            if strategy.name == name:
                return strategy
        return None

    def list_strategies(self) -> list[dict]:
        """List all registered strategies."""
        return [
            {
                "id": s.strategy_id,
                "name": s.name,
                "state": s.state.value,
                "enabled": s.config.enabled,
                "weight": self._strategy_weights.get(s.strategy_id, StrategyWeight(strategy_id=s.strategy_id)).weight,
            }
            for s in self._strategies.values()
        ]

    async def start_all(self) -> None:
        """Start all strategies."""
        self._is_running = True

        for strategy in self._strategies.values():
            if strategy.config.enabled:
                await strategy.start()

        logger.info(f"Started {len(self._strategies)} strategies")

    async def stop_all(self) -> None:
        """Stop all strategies."""
        self._is_running = False

        for strategy in self._strategies.values():
            await strategy.stop()

        logger.info("Stopped all strategies")

    async def evaluate_all(
        self,
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """
        Evaluate all strategies and collect signals.

        Args:
            market_data: Market data for all symbols
            context: Strategy execution context

        Returns:
            List of trading signals
        """
        self._evaluation_count += 1
        all_signals: list[StrategySignal] = []

        tasks = []
        for strategy in self._strategies.values():
            if strategy.is_enabled():
                tasks.append(self._evaluate_strategy(strategy, market_data, context))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Strategy evaluation error: {result}")
            elif result:
                all_signals.extend(result)

        self._total_signals_generated += len(all_signals)

        for signal in all_signals:
            self._signal_history.append(signal)

        if len(self._signal_history) > 10000:
            self._signal_history = self._signal_history[-10000:]

        if self._config.aggregate_signals:
            return self._aggregate_signals(all_signals)

        return all_signals

    async def _evaluate_strategy(
        self,
        strategy: BaseStrategy,
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """Evaluate a single strategy."""
        try:
            signals = await asyncio.wait_for(
                strategy.evaluate(market_data, context),
                timeout=30.0,
            )
            return signals

        except asyncio.TimeoutError:
            logger.error(f"Strategy {strategy.name} evaluation timed out")
            return []

        except Exception as e:
            logger.error(f"Strategy {strategy.name} error: {e}")
            return []

    def _aggregate_signals(
        self,
        signals: list[StrategySignal],
    ) -> list[StrategySignal]:
        """
        Aggregate signals from multiple strategies.

        Args:
            signals: List of all signals

        Returns:
            Aggregated/filtered signals
        """
        if not signals:
            return []

        signals_by_symbol: dict[str, list[StrategySignal]] = {}
        for signal in signals:
            if signal.symbol not in signals_by_symbol:
                signals_by_symbol[signal.symbol] = []
            signals_by_symbol[signal.symbol].append(signal)

        result: list[StrategySignal] = []

        for symbol, symbol_signals in signals_by_symbol.items():
            if self._config.require_consensus:
                consensus_signal = self._get_consensus_signal(symbol, symbol_signals)
                if consensus_signal:
                    result.append(consensus_signal)
            else:
                resolved = self._resolve_conflicts(symbol_signals)
                result.extend(resolved[:self._config.max_signals_per_symbol])

        return result

    def _get_consensus_signal(
        self,
        symbol: str,
        signals: list[StrategySignal],
    ) -> Optional[StrategySignal]:
        """Get consensus signal from multiple strategies."""
        if not signals:
            return None

        buy_signals = [s for s in signals if s.action.value in ["buy", "scale_in"]]
        sell_signals = [s for s in signals if s.action.value in ["sell", "scale_out"]]

        total_strategies = len(set(s.strategy_id for s in signals))

        if len(buy_signals) / total_strategies >= self._config.min_consensus_pct:
            return self._merge_signals(buy_signals)

        if len(sell_signals) / total_strategies >= self._config.min_consensus_pct:
            return self._merge_signals(sell_signals)

        return None

    def _merge_signals(
        self,
        signals: list[StrategySignal],
    ) -> StrategySignal:
        """Merge multiple signals into one."""
        if len(signals) == 1:
            return signals[0]

        total_weight = 0.0
        weighted_strength = 0.0
        weighted_confidence = 0.0
        weighted_entry = 0.0

        for signal in signals:
            weight = self._strategy_weights.get(
                signal.strategy_id,
                StrategyWeight(strategy_id=signal.strategy_id),
            ).weight

            weighted_strength += signal.strength * weight
            weighted_confidence += signal.confidence * weight
            if signal.entry_price:
                weighted_entry += signal.entry_price * weight
            total_weight += weight

        if total_weight > 0:
            avg_strength = weighted_strength / total_weight
            avg_confidence = weighted_confidence / total_weight
            avg_entry = weighted_entry / total_weight if weighted_entry > 0 else None
        else:
            avg_strength = sum(s.strength for s in signals) / len(signals)
            avg_confidence = sum(s.confidence for s in signals) / len(signals)
            avg_entry = signals[0].entry_price

        base_signal = max(signals, key=lambda s: s.confidence)

        merged = StrategySignal(
            strategy_id="aggregated",
            strategy_name="Aggregated",
            symbol=base_signal.symbol,
            action=base_signal.action,
            side=base_signal.side,
            strength=avg_strength,
            confidence=avg_confidence,
            entry_price=avg_entry,
            stop_loss=base_signal.stop_loss,
            take_profit=base_signal.take_profit,
            position_size_pct=base_signal.position_size_pct,
            reason=f"Consensus from {len(signals)} strategies",
            metadata={
                "contributing_strategies": [s.strategy_name for s in signals],
                "signal_count": len(signals),
            },
        )

        return merged

    def _resolve_conflicts(
        self,
        signals: list[StrategySignal],
    ) -> list[StrategySignal]:
        """Resolve conflicting signals."""
        if len(signals) <= 1:
            return signals

        resolution = self._config.conflict_resolution

        if resolution == "highest_confidence":
            return sorted(signals, key=lambda s: s.confidence, reverse=True)

        elif resolution == "highest_strength":
            return sorted(signals, key=lambda s: s.strength, reverse=True)

        elif resolution == "highest_weight":
            def get_weight(s: StrategySignal) -> float:
                return self._strategy_weights.get(
                    s.strategy_id,
                    StrategyWeight(strategy_id=s.strategy_id),
                ).weight
            return sorted(signals, key=get_weight, reverse=True)

        elif resolution == "highest_priority":
            def get_priority(s: StrategySignal) -> int:
                return self._strategy_weights.get(
                    s.strategy_id,
                    StrategyWeight(strategy_id=s.strategy_id),
                ).priority
            return sorted(signals, key=get_priority, reverse=True)

        return signals

    def update_strategy_weight(
        self,
        strategy_id: str,
        weight: Optional[float] = None,
        priority: Optional[int] = None,
        enabled: Optional[bool] = None,
    ) -> bool:
        """
        Update strategy weight and settings.

        Args:
            strategy_id: Strategy ID
            weight: New weight
            priority: New priority
            enabled: Enable/disable

        Returns:
            True if updated
        """
        if strategy_id not in self._strategies:
            return False

        if strategy_id not in self._strategy_weights:
            self._strategy_weights[strategy_id] = StrategyWeight(
                strategy_id=strategy_id,
            )

        sw = self._strategy_weights[strategy_id]

        if weight is not None:
            sw.weight = weight
        if priority is not None:
            sw.priority = priority
        if enabled is not None:
            sw.enabled = enabled
            strategy = self._strategies[strategy_id]
            strategy.update_config(enabled=enabled)

        return True

    def get_all_metrics(self) -> dict[str, StrategyMetrics]:
        """Get metrics for all strategies."""
        return {
            strategy_id: strategy.metrics
            for strategy_id, strategy in self._strategies.items()
        }

    def get_active_signals(
        self,
        symbol: Optional[str] = None,
    ) -> list[StrategySignal]:
        """Get active signals."""
        if symbol:
            return self._active_signals.get(symbol, [])

        all_signals: list[StrategySignal] = []
        for signals in self._active_signals.values():
            all_signals.extend(signals)
        return all_signals

    def get_signal_history(
        self,
        strategy_id: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> list[StrategySignal]:
        """Get signal history with optional filters."""
        signals = self._signal_history

        if strategy_id:
            signals = [s for s in signals if s.strategy_id == strategy_id]

        if symbol:
            signals = [s for s in signals if s.symbol == symbol]

        return signals[-limit:]

    def get_statistics(self) -> dict:
        """Get manager statistics."""
        running = sum(
            1 for s in self._strategies.values()
            if s.state == StrategyState.RUNNING
        )

        return {
            "total_strategies": len(self._strategies),
            "running_strategies": running,
            "evaluation_count": self._evaluation_count,
            "total_signals": self._total_signals_generated,
            "active_signals": sum(len(s) for s in self._active_signals.values()),
            "is_running": self._is_running,
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"StrategyManager(strategies={len(self._strategies)})"
