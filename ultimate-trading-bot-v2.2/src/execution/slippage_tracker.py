"""
Slippage Tracking for Ultimate Trading Bot v2.2.

This module provides comprehensive slippage analysis including
real-time tracking, historical analysis, and slippage prediction.
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from src.execution.base_executor import Order, OrderSide, OrderType, Fill


logger = logging.getLogger(__name__)


class SlippageType(str, Enum):
    """Types of slippage."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    ZERO = "zero"


class SlippageSource(str, Enum):
    """Sources of slippage."""

    MARKET_IMPACT = "market_impact"
    SPREAD = "spread"
    LATENCY = "latency"
    VOLATILITY = "volatility"
    TIMING = "timing"
    SIZE = "size"


class SlippageTrackerConfig(BaseModel):
    """Configuration for slippage tracker."""

    model_config = {"arbitrary_types_allowed": True}

    track_by_symbol: bool = Field(default=True, description="Track slippage by symbol")
    track_by_strategy: bool = Field(default=True, description="Track slippage by strategy")
    track_by_time: bool = Field(default=True, description="Track slippage by time of day")
    rolling_window_days: int = Field(default=30, description="Rolling window for stats")
    alert_threshold_bps: float = Field(default=10.0, description="Alert threshold in bps")
    outlier_threshold_std: float = Field(default=3.0, description="Outlier detection threshold")


class SlippageRecord(BaseModel):
    """Record of slippage for a single trade."""

    record_id: str
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float

    expected_price: float
    execution_price: float
    arrival_price: float | None = None

    slippage_amount: float = Field(default=0.0, description="Slippage in $ terms")
    slippage_pct: float = Field(default=0.0, description="Slippage as %")
    slippage_bps: float = Field(default=0.0, description="Slippage in basis points")

    slippage_type: SlippageType = Field(default=SlippageType.ZERO)
    estimated_sources: list[SlippageSource] = Field(default_factory=list)

    execution_time: datetime = Field(default_factory=datetime.now)
    latency_ms: float = Field(default=0.0)
    market_volume: float = Field(default=0.0)
    participation_rate: float = Field(default=0.0)

    strategy_id: str | None = None
    venue_id: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class SlippageStats:
    """Slippage statistics."""

    sample_count: int = 0
    total_slippage_bps: float = 0.0
    avg_slippage_bps: float = 0.0
    median_slippage_bps: float = 0.0
    std_slippage_bps: float = 0.0
    min_slippage_bps: float = 0.0
    max_slippage_bps: float = 0.0
    positive_count: int = 0
    negative_count: int = 0
    total_cost: float = 0.0


@dataclass
class SlippageAlert:
    """Slippage alert."""

    alert_id: str
    record: SlippageRecord
    message: str
    severity: str
    timestamp: datetime = field(default_factory=datetime.now)


class SlippageTracker:
    """
    Tracks and analyzes execution slippage.

    Provides real-time slippage monitoring, historical analysis,
    and slippage prediction capabilities.
    """

    def __init__(self, config: SlippageTrackerConfig | None = None):
        """
        Initialize slippage tracker.

        Args:
            config: Tracker configuration
        """
        self.config = config or SlippageTrackerConfig()

        self._records: list[SlippageRecord] = []
        self._by_symbol: dict[str, list[SlippageRecord]] = defaultdict(list)
        self._by_strategy: dict[str, list[SlippageRecord]] = defaultdict(list)
        self._by_hour: dict[int, list[SlippageRecord]] = defaultdict(list)

        self._alerts: list[SlippageAlert] = []
        self._overall_stats = SlippageStats()

        self._slippage_values: list[float] = []
        self._lock = asyncio.Lock()

        logger.info("SlippageTracker initialized")

    async def record_slippage(
        self,
        order: Order,
        fill: Fill,
        expected_price: float,
        arrival_price: float | None = None,
        latency_ms: float = 0.0,
        market_volume: float = 0.0,
        venue_id: str | None = None,
    ) -> SlippageRecord:
        """
        Record slippage for a fill.

        Args:
            order: Original order
            fill: Execution fill
            expected_price: Expected execution price
            arrival_price: Price at order arrival
            latency_ms: Execution latency
            market_volume: Market volume during execution
            venue_id: Execution venue ID

        Returns:
            SlippageRecord
        """
        execution_price = fill.price

        slippage_amount = execution_price - expected_price
        if order.side == OrderSide.SELL:
            slippage_amount = -slippage_amount

        slippage_pct = slippage_amount / expected_price if expected_price > 0 else 0
        slippage_bps = slippage_pct * 10000

        if slippage_bps > 0:
            slippage_type = SlippageType.POSITIVE
        elif slippage_bps < 0:
            slippage_type = SlippageType.NEGATIVE
        else:
            slippage_type = SlippageType.ZERO

        sources = self._estimate_slippage_sources(
            order, fill, expected_price, latency_ms, market_volume
        )

        participation_rate = 0.0
        if market_volume > 0:
            participation_rate = fill.quantity / market_volume

        record = SlippageRecord(
            record_id=fill.fill_id,
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=fill.quantity,
            expected_price=expected_price,
            execution_price=execution_price,
            arrival_price=arrival_price,
            slippage_amount=slippage_amount * fill.quantity,
            slippage_pct=slippage_pct,
            slippage_bps=slippage_bps,
            slippage_type=slippage_type,
            estimated_sources=sources,
            execution_time=fill.executed_at,
            latency_ms=latency_ms,
            market_volume=market_volume,
            participation_rate=participation_rate,
            strategy_id=order.strategy_id,
            venue_id=venue_id,
        )

        async with self._lock:
            self._records.append(record)
            self._slippage_values.append(slippage_bps)

            if self.config.track_by_symbol:
                self._by_symbol[order.symbol].append(record)

            if self.config.track_by_strategy and order.strategy_id:
                self._by_strategy[order.strategy_id].append(record)

            if self.config.track_by_time:
                hour = fill.executed_at.hour
                self._by_hour[hour].append(record)

            self._update_stats(record)

        if abs(slippage_bps) > self.config.alert_threshold_bps:
            await self._create_alert(record)

        if abs(slippage_bps) > self._get_outlier_threshold():
            await self._create_alert(record, is_outlier=True)

        return record

    def _estimate_slippage_sources(
        self,
        order: Order,
        fill: Fill,
        expected_price: float,
        latency_ms: float,
        market_volume: float,
    ) -> list[SlippageSource]:
        """Estimate sources of slippage."""
        sources: list[SlippageSource] = []

        if latency_ms > 100:
            sources.append(SlippageSource.LATENCY)

        if market_volume > 0 and fill.quantity / market_volume > 0.01:
            sources.append(SlippageSource.MARKET_IMPACT)

        if order.order_type == OrderType.MARKET:
            sources.append(SlippageSource.SPREAD)

        if fill.quantity > 10000:
            sources.append(SlippageSource.SIZE)

        if not sources:
            sources.append(SlippageSource.VOLATILITY)

        return sources

    def _update_stats(self, record: SlippageRecord) -> None:
        """Update running statistics."""
        self._overall_stats.sample_count += 1
        self._overall_stats.total_slippage_bps += record.slippage_bps
        self._overall_stats.total_cost += record.slippage_amount

        if record.slippage_type == SlippageType.POSITIVE:
            self._overall_stats.positive_count += 1
        elif record.slippage_type == SlippageType.NEGATIVE:
            self._overall_stats.negative_count += 1

        n = self._overall_stats.sample_count
        self._overall_stats.avg_slippage_bps = (
            self._overall_stats.total_slippage_bps / n
        )

        if len(self._slippage_values) >= 2:
            self._overall_stats.std_slippage_bps = float(np.std(self._slippage_values))
            self._overall_stats.median_slippage_bps = float(np.median(self._slippage_values))
            self._overall_stats.min_slippage_bps = float(min(self._slippage_values))
            self._overall_stats.max_slippage_bps = float(max(self._slippage_values))

    def _get_outlier_threshold(self) -> float:
        """Get outlier threshold based on historical data."""
        if len(self._slippage_values) < 30:
            return float("inf")

        return (
            abs(self._overall_stats.avg_slippage_bps) +
            self.config.outlier_threshold_std * self._overall_stats.std_slippage_bps
        )

    async def _create_alert(
        self,
        record: SlippageRecord,
        is_outlier: bool = False,
    ) -> None:
        """Create slippage alert."""
        if is_outlier:
            severity = "high"
            message = (
                f"Outlier slippage detected: {record.symbol} "
                f"{record.slippage_bps:.1f} bps"
            )
        else:
            severity = "medium" if record.slippage_bps < 0 else "low"
            message = (
                f"Slippage alert: {record.symbol} "
                f"{record.slippage_bps:.1f} bps (threshold: {self.config.alert_threshold_bps})"
            )

        alert = SlippageAlert(
            alert_id=record.record_id,
            record=record,
            message=message,
            severity=severity,
        )

        self._alerts.append(alert)
        logger.warning(message)

    def get_slippage_stats(
        self,
        symbol: str | None = None,
        strategy_id: str | None = None,
        since: datetime | None = None,
    ) -> SlippageStats:
        """
        Get slippage statistics.

        Args:
            symbol: Filter by symbol
            strategy_id: Filter by strategy
            since: Filter by time

        Returns:
            SlippageStats
        """
        if symbol:
            records = self._by_symbol.get(symbol, [])
        elif strategy_id:
            records = self._by_strategy.get(strategy_id, [])
        else:
            records = self._records

        if since:
            records = [r for r in records if r.execution_time >= since]

        if not records:
            return SlippageStats()

        slippage_values = [r.slippage_bps for r in records]

        return SlippageStats(
            sample_count=len(records),
            total_slippage_bps=sum(slippage_values),
            avg_slippage_bps=float(np.mean(slippage_values)),
            median_slippage_bps=float(np.median(slippage_values)),
            std_slippage_bps=float(np.std(slippage_values)) if len(slippage_values) > 1 else 0,
            min_slippage_bps=float(min(slippage_values)),
            max_slippage_bps=float(max(slippage_values)),
            positive_count=sum(1 for v in slippage_values if v > 0),
            negative_count=sum(1 for v in slippage_values if v < 0),
            total_cost=sum(r.slippage_amount for r in records),
        )

    def get_slippage_by_hour(self) -> dict[int, SlippageStats]:
        """Get slippage statistics by hour of day."""
        result: dict[int, SlippageStats] = {}

        for hour, records in self._by_hour.items():
            if records:
                slippage_values = [r.slippage_bps for r in records]
                result[hour] = SlippageStats(
                    sample_count=len(records),
                    avg_slippage_bps=float(np.mean(slippage_values)),
                    std_slippage_bps=float(np.std(slippage_values)) if len(slippage_values) > 1 else 0,
                    total_cost=sum(r.slippage_amount for r in records),
                )

        return result

    def get_slippage_by_source(self) -> dict[str, float]:
        """Get average slippage contribution by source."""
        source_counts: dict[SlippageSource, list[float]] = defaultdict(list)

        for record in self._records:
            for source in record.estimated_sources:
                source_counts[source].append(record.slippage_bps)

        return {
            source.value: float(np.mean(values)) if values else 0.0
            for source, values in source_counts.items()
        }

    def predict_slippage(
        self,
        symbol: str,
        quantity: float,
        order_type: OrderType,
        market_volume: float | None = None,
    ) -> dict[str, float]:
        """
        Predict expected slippage for an order.

        Args:
            symbol: Symbol
            quantity: Order quantity
            order_type: Order type
            market_volume: Expected market volume

        Returns:
            Prediction dictionary
        """
        symbol_records = self._by_symbol.get(symbol, [])

        if not symbol_records:
            return {
                "predicted_bps": 0.0,
                "confidence": 0.0,
                "data_points": 0,
            }

        slippage_values = [r.slippage_bps for r in symbol_records[-100:]]

        base_prediction = float(np.mean(slippage_values))

        type_records = [
            r for r in symbol_records
            if r.order_type == order_type
        ]
        if type_records:
            type_adjustment = float(np.mean([r.slippage_bps for r in type_records])) - base_prediction
            base_prediction += type_adjustment * 0.5

        if market_volume and market_volume > 0:
            participation = quantity / market_volume
            if participation > 0.05:
                size_adjustment = participation * 50
                base_prediction += size_adjustment

        confidence = min(len(slippage_values) / 100, 1.0)

        return {
            "predicted_bps": base_prediction,
            "predicted_pct": base_prediction / 10000,
            "confidence": confidence,
            "data_points": len(slippage_values),
            "std_bps": float(np.std(slippage_values)) if len(slippage_values) > 1 else 0,
        }

    def get_records(
        self,
        symbol: str | None = None,
        limit: int = 100,
    ) -> list[SlippageRecord]:
        """Get slippage records."""
        if symbol:
            records = self._by_symbol.get(symbol, [])
        else:
            records = self._records
        return records[-limit:]

    def get_alerts(self, limit: int = 50) -> list[SlippageAlert]:
        """Get recent alerts."""
        return self._alerts[-limit:]

    def get_overall_stats(self) -> SlippageStats:
        """Get overall slippage statistics."""
        return self._overall_stats

    async def get_slippage_summary(self) -> dict[str, Any]:
        """
        Get comprehensive slippage summary.

        Returns:
            Summary dictionary
        """
        cutoff = datetime.now() - timedelta(days=self.config.rolling_window_days)
        recent_records = [r for r in self._records if r.execution_time >= cutoff]

        top_symbols = sorted(
            self._by_symbol.items(),
            key=lambda x: abs(sum(r.slippage_bps for r in x[1])),
            reverse=True,
        )[:10]

        return {
            "timestamp": datetime.now().isoformat(),
            "overall": {
                "sample_count": self._overall_stats.sample_count,
                "avg_slippage_bps": self._overall_stats.avg_slippage_bps,
                "std_slippage_bps": self._overall_stats.std_slippage_bps,
                "total_cost": self._overall_stats.total_cost,
                "positive_rate": (
                    self._overall_stats.positive_count / self._overall_stats.sample_count
                    if self._overall_stats.sample_count > 0 else 0
                ),
            },
            "rolling_window": {
                "days": self.config.rolling_window_days,
                "sample_count": len(recent_records),
                "avg_slippage_bps": (
                    float(np.mean([r.slippage_bps for r in recent_records]))
                    if recent_records else 0
                ),
            },
            "by_source": self.get_slippage_by_source(),
            "top_symbols": [
                {
                    "symbol": symbol,
                    "avg_slippage_bps": float(np.mean([r.slippage_bps for r in records])),
                    "count": len(records),
                }
                for symbol, records in top_symbols
            ],
            "alerts_count": len(self._alerts),
            "outlier_threshold_bps": self._get_outlier_threshold(),
        }

    def clear_history(self, before: datetime | None = None) -> int:
        """
        Clear slippage history.

        Args:
            before: Clear records before this time

        Returns:
            Number of records cleared
        """
        if before:
            original_count = len(self._records)
            self._records = [r for r in self._records if r.execution_time >= before]
            cleared = original_count - len(self._records)

            for symbol in self._by_symbol:
                self._by_symbol[symbol] = [
                    r for r in self._by_symbol[symbol]
                    if r.execution_time >= before
                ]

            for strategy in self._by_strategy:
                self._by_strategy[strategy] = [
                    r for r in self._by_strategy[strategy]
                    if r.execution_time >= before
                ]

            for hour in self._by_hour:
                self._by_hour[hour] = [
                    r for r in self._by_hour[hour]
                    if r.execution_time >= before
                ]

            return cleared
        else:
            count = len(self._records)
            self._records.clear()
            self._by_symbol.clear()
            self._by_strategy.clear()
            self._by_hour.clear()
            self._slippage_values.clear()
            self._overall_stats = SlippageStats()
            return count
