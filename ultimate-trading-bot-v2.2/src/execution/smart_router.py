"""
Smart Order Router for Ultimate Trading Bot v2.2.

This module provides intelligent order routing with venue selection,
order splitting, and optimal execution path determination.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field

from src.execution.base_executor import (
    Order,
    OrderSide,
    OrderType,
    TimeInForce,
    Fill,
    ExecutionResult,
)


logger = logging.getLogger(__name__)


class VenueType(str, Enum):
    """Trading venue types."""

    EXCHANGE = "exchange"
    DARK_POOL = "dark_pool"
    ATS = "ats"
    INTERNALIZATION = "internalization"
    WHOLESALE = "wholesale"


class RoutingStrategy(str, Enum):
    """Order routing strategies."""

    SMART = "smart"
    DIRECT = "direct"
    SWEEP = "sweep"
    DARK_ONLY = "dark_only"
    LIT_ONLY = "lit_only"
    COST_OPTIMIZED = "cost_optimized"
    SPEED_OPTIMIZED = "speed_optimized"
    FILL_RATE_OPTIMIZED = "fill_rate_optimized"


class VenueStatus(str, Enum):
    """Venue connection status."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"


class SmartRouterConfig(BaseModel):
    """Configuration for smart order router."""

    model_config = {"arbitrary_types_allowed": True}

    default_strategy: RoutingStrategy = Field(
        default=RoutingStrategy.SMART,
        description="Default routing strategy"
    )
    enable_dark_pools: bool = Field(default=True, description="Enable dark pool routing")
    enable_internalization: bool = Field(default=True, description="Enable internalization")
    min_dark_pool_size: float = Field(default=100, description="Min size for dark pools")
    max_venues_per_order: int = Field(default=5, description="Max venues per order")
    venue_timeout_seconds: float = Field(default=5.0, description="Venue response timeout")
    retry_on_reject: bool = Field(default=True, description="Retry on venue rejection")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    cost_weight: float = Field(default=0.3, description="Cost factor weight")
    speed_weight: float = Field(default=0.3, description="Speed factor weight")
    fill_weight: float = Field(default=0.4, description="Fill rate factor weight")


class Venue(BaseModel):
    """Trading venue representation."""

    venue_id: str
    name: str
    venue_type: VenueType
    status: VenueStatus = Field(default=VenueStatus.CONNECTED)

    fee_per_share: float = Field(default=0.0, description="Fee per share")
    rebate_per_share: float = Field(default=0.0, description="Rebate per share")
    min_size: float = Field(default=1.0, description="Minimum order size")
    max_size: float = Field(default=float("inf"), description="Maximum order size")

    avg_fill_rate: float = Field(default=0.8, description="Average fill rate")
    avg_latency_ms: float = Field(default=10.0, description="Average latency")
    avg_price_improvement_bps: float = Field(default=0.0, description="Avg price improvement")

    supports_extended_hours: bool = Field(default=False)
    supports_odd_lots: bool = Field(default=True)
    supports_short_selling: bool = Field(default=True)

    last_updated: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RouteDecision(BaseModel):
    """Routing decision for an order."""

    decision_id: str = Field(default_factory=lambda: str(uuid4()))
    order_id: str
    symbol: str
    side: OrderSide
    total_qty: float

    routes: list[dict[str, Any]] = Field(default_factory=list)
    strategy_used: RoutingStrategy
    timestamp: datetime = Field(default_factory=datetime.now)

    expected_cost: float = Field(default=0.0)
    expected_fill_rate: float = Field(default=0.0)
    expected_latency_ms: float = Field(default=0.0)

    reasoning: list[str] = Field(default_factory=list)


@dataclass
class VenueQuote:
    """Quote from a venue."""

    venue_id: str
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RouteResult:
    """Result from a routed order."""

    route_id: str
    venue_id: str
    order_id: str
    fills: list[Fill] = field(default_factory=list)
    filled_qty: float = 0.0
    avg_price: float = 0.0
    latency_ms: float = 0.0
    status: str = "pending"
    error_message: str | None = None


class SmartOrderRouter:
    """
    Intelligent order routing system.

    Provides smart order routing with venue selection, order splitting,
    and optimal execution path determination.
    """

    def __init__(self, config: SmartRouterConfig | None = None):
        """
        Initialize smart order router.

        Args:
            config: Router configuration
        """
        self.config = config or SmartRouterConfig()
        self._venues: dict[str, Venue] = {}
        self._venue_quotes: dict[str, dict[str, VenueQuote]] = {}
        self._route_history: list[RouteDecision] = []
        self._venue_stats: dict[str, dict[str, float]] = {}
        self._lock = asyncio.Lock()

        self._initialize_default_venues()

        logger.info("SmartOrderRouter initialized")

    def _initialize_default_venues(self) -> None:
        """Initialize default venue configuration."""
        default_venues = [
            Venue(
                venue_id="NYSE",
                name="New York Stock Exchange",
                venue_type=VenueType.EXCHANGE,
                fee_per_share=0.003,
                rebate_per_share=0.002,
                avg_fill_rate=0.85,
                avg_latency_ms=5.0,
            ),
            Venue(
                venue_id="NASDAQ",
                name="NASDAQ",
                venue_type=VenueType.EXCHANGE,
                fee_per_share=0.003,
                rebate_per_share=0.002,
                avg_fill_rate=0.88,
                avg_latency_ms=3.0,
            ),
            Venue(
                venue_id="ARCA",
                name="NYSE Arca",
                venue_type=VenueType.EXCHANGE,
                fee_per_share=0.003,
                rebate_per_share=0.002,
                avg_fill_rate=0.82,
                avg_latency_ms=4.0,
                supports_extended_hours=True,
            ),
            Venue(
                venue_id="BATS",
                name="BATS Global Markets",
                venue_type=VenueType.EXCHANGE,
                fee_per_share=0.002,
                rebate_per_share=0.0025,
                avg_fill_rate=0.80,
                avg_latency_ms=2.0,
            ),
            Venue(
                venue_id="IEX",
                name="Investors Exchange",
                venue_type=VenueType.EXCHANGE,
                fee_per_share=0.0009,
                avg_fill_rate=0.75,
                avg_latency_ms=1.0,
                avg_price_improvement_bps=0.5,
            ),
            Venue(
                venue_id="SIGMA_X",
                name="Goldman Sachs Sigma X",
                venue_type=VenueType.DARK_POOL,
                fee_per_share=0.001,
                min_size=100,
                avg_fill_rate=0.45,
                avg_latency_ms=15.0,
                avg_price_improvement_bps=2.0,
            ),
            Venue(
                venue_id="CROSSFINDER",
                name="Credit Suisse CrossFinder",
                venue_type=VenueType.DARK_POOL,
                fee_per_share=0.001,
                min_size=100,
                avg_fill_rate=0.40,
                avg_latency_ms=20.0,
                avg_price_improvement_bps=1.5,
            ),
        ]

        for venue in default_venues:
            self._venues[venue.venue_id] = venue

    async def route_order(
        self,
        order: Order,
        market_data: dict[str, Any],
        strategy: RoutingStrategy | None = None,
    ) -> RouteDecision:
        """
        Route an order to optimal venues.

        Args:
            order: Order to route
            market_data: Current market data
            strategy: Routing strategy to use

        Returns:
            RouteDecision
        """
        strategy = strategy or self.config.default_strategy

        decision = RouteDecision(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            total_qty=order.quantity,
            strategy_used=strategy,
        )

        try:
            eligible_venues = self._get_eligible_venues(order, market_data)

            if not eligible_venues:
                decision.reasoning.append("No eligible venues found")
                return decision

            decision.reasoning.append(f"Found {len(eligible_venues)} eligible venues")

            if strategy == RoutingStrategy.SMART:
                routes = await self._smart_route(order, eligible_venues, market_data)
            elif strategy == RoutingStrategy.SWEEP:
                routes = self._sweep_route(order, eligible_venues, market_data)
            elif strategy == RoutingStrategy.DARK_ONLY:
                dark_venues = [v for v in eligible_venues if v.venue_type == VenueType.DARK_POOL]
                routes = await self._smart_route(order, dark_venues, market_data)
            elif strategy == RoutingStrategy.LIT_ONLY:
                lit_venues = [v for v in eligible_venues if v.venue_type == VenueType.EXCHANGE]
                routes = await self._smart_route(order, lit_venues, market_data)
            elif strategy == RoutingStrategy.COST_OPTIMIZED:
                routes = self._cost_optimized_route(order, eligible_venues, market_data)
            elif strategy == RoutingStrategy.SPEED_OPTIMIZED:
                routes = self._speed_optimized_route(order, eligible_venues, market_data)
            elif strategy == RoutingStrategy.FILL_RATE_OPTIMIZED:
                routes = self._fill_rate_optimized_route(order, eligible_venues, market_data)
            else:
                routes = [{"venue_id": eligible_venues[0].venue_id, "qty": order.quantity}]

            decision.routes = routes

            total_cost = 0.0
            total_fill_rate = 0.0
            total_latency = 0.0
            total_qty = 0.0

            for route in routes:
                venue = self._venues.get(route["venue_id"])
                if venue:
                    qty = route["qty"]
                    total_cost += venue.fee_per_share * qty
                    total_fill_rate += venue.avg_fill_rate * qty
                    total_latency += venue.avg_latency_ms * qty
                    total_qty += qty

            if total_qty > 0:
                decision.expected_cost = total_cost
                decision.expected_fill_rate = total_fill_rate / total_qty
                decision.expected_latency_ms = total_latency / total_qty

            async with self._lock:
                self._route_history.append(decision)

            return decision

        except Exception as e:
            logger.error(f"Routing error: {e}")
            decision.reasoning.append(f"Error: {str(e)}")
            return decision

    def _get_eligible_venues(
        self,
        order: Order,
        market_data: dict[str, Any],
    ) -> list[Venue]:
        """Get venues eligible for this order."""
        eligible: list[Venue] = []

        is_extended = market_data.get("is_extended_hours", False)
        is_odd_lot = order.quantity < 100
        is_short = order.side == OrderSide.SELL and order.quantity < 0

        for venue in self._venues.values():
            if venue.status != VenueStatus.CONNECTED:
                continue

            if order.quantity < venue.min_size:
                continue

            if order.quantity > venue.max_size:
                continue

            if is_extended and not venue.supports_extended_hours:
                continue

            if is_odd_lot and not venue.supports_odd_lots:
                continue

            if is_short and not venue.supports_short_selling:
                continue

            if not self.config.enable_dark_pools and venue.venue_type == VenueType.DARK_POOL:
                continue

            eligible.append(venue)

        return eligible

    async def _smart_route(
        self,
        order: Order,
        venues: list[Venue],
        market_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Smart routing using venue scoring."""
        venue_scores: list[tuple[Venue, float]] = []

        for venue in venues:
            score = self._calculate_venue_score(venue, order, market_data)
            venue_scores.append((venue, score))

        venue_scores.sort(key=lambda x: x[1], reverse=True)

        routes: list[dict[str, Any]] = []
        remaining_qty = order.quantity

        for venue, score in venue_scores[:self.config.max_venues_per_order]:
            if remaining_qty <= 0:
                break

            if venue.venue_type == VenueType.DARK_POOL:
                alloc_qty = min(remaining_qty * 0.3, remaining_qty)
            else:
                alloc_qty = remaining_qty

            if alloc_qty >= venue.min_size:
                routes.append({
                    "venue_id": venue.venue_id,
                    "qty": alloc_qty,
                    "score": score,
                })
                remaining_qty -= alloc_qty

        if remaining_qty > 0 and routes:
            routes[0]["qty"] += remaining_qty

        return routes

    def _calculate_venue_score(
        self,
        venue: Venue,
        order: Order,
        market_data: dict[str, Any],
    ) -> float:
        """Calculate venue score based on weighted factors."""
        price = market_data.get("price", 100.0)

        cost_score = 1.0 - min(venue.fee_per_share / 0.01, 1.0)
        if venue.rebate_per_share > 0:
            cost_score += venue.rebate_per_share / 0.01

        speed_score = 1.0 - min(venue.avg_latency_ms / 50.0, 1.0)

        fill_score = venue.avg_fill_rate

        price_improvement_score = venue.avg_price_improvement_bps / 5.0

        score = (
            self.config.cost_weight * cost_score +
            self.config.speed_weight * speed_score +
            self.config.fill_weight * fill_score +
            0.1 * price_improvement_score
        )

        return score

    def _sweep_route(
        self,
        order: Order,
        venues: list[Venue],
        market_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Sweep routing - send to all venues simultaneously."""
        routes: list[dict[str, Any]] = []

        qty_per_venue = order.quantity / min(len(venues), self.config.max_venues_per_order)

        for venue in venues[:self.config.max_venues_per_order]:
            if qty_per_venue >= venue.min_size:
                routes.append({
                    "venue_id": venue.venue_id,
                    "qty": qty_per_venue,
                })

        return routes

    def _cost_optimized_route(
        self,
        order: Order,
        venues: list[Venue],
        market_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Route optimizing for lowest cost."""
        venues_sorted = sorted(
            venues,
            key=lambda v: v.fee_per_share - v.rebate_per_share
        )

        return [{"venue_id": venues_sorted[0].venue_id, "qty": order.quantity}]

    def _speed_optimized_route(
        self,
        order: Order,
        venues: list[Venue],
        market_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Route optimizing for fastest execution."""
        venues_sorted = sorted(venues, key=lambda v: v.avg_latency_ms)

        return [{"venue_id": venues_sorted[0].venue_id, "qty": order.quantity}]

    def _fill_rate_optimized_route(
        self,
        order: Order,
        venues: list[Venue],
        market_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Route optimizing for highest fill rate."""
        venues_sorted = sorted(venues, key=lambda v: v.avg_fill_rate, reverse=True)

        routes: list[dict[str, Any]] = []
        remaining = order.quantity

        for venue in venues_sorted[:3]:
            if remaining <= 0:
                break
            alloc = remaining * 0.4
            routes.append({"venue_id": venue.venue_id, "qty": alloc})
            remaining -= alloc

        if remaining > 0 and routes:
            routes[0]["qty"] += remaining

        return routes

    def add_venue(self, venue: Venue) -> None:
        """Add or update a venue."""
        self._venues[venue.venue_id] = venue
        logger.info(f"Added venue: {venue.venue_id}")

    def remove_venue(self, venue_id: str) -> bool:
        """Remove a venue."""
        if venue_id in self._venues:
            del self._venues[venue_id]
            return True
        return False

    def update_venue_status(
        self,
        venue_id: str,
        status: VenueStatus,
    ) -> bool:
        """Update venue status."""
        if venue_id in self._venues:
            self._venues[venue_id].status = status
            self._venues[venue_id].last_updated = datetime.now()
            return True
        return False

    def update_venue_stats(
        self,
        venue_id: str,
        fill_rate: float | None = None,
        latency_ms: float | None = None,
        price_improvement_bps: float | None = None,
    ) -> bool:
        """Update venue statistics."""
        if venue_id not in self._venues:
            return False

        venue = self._venues[venue_id]

        if fill_rate is not None:
            venue.avg_fill_rate = venue.avg_fill_rate * 0.9 + fill_rate * 0.1

        if latency_ms is not None:
            venue.avg_latency_ms = venue.avg_latency_ms * 0.9 + latency_ms * 0.1

        if price_improvement_bps is not None:
            venue.avg_price_improvement_bps = (
                venue.avg_price_improvement_bps * 0.9 + price_improvement_bps * 0.1
            )

        venue.last_updated = datetime.now()
        return True

    def get_venue(self, venue_id: str) -> Venue | None:
        """Get venue by ID."""
        return self._venues.get(venue_id)

    def get_all_venues(self) -> list[Venue]:
        """Get all venues."""
        return list(self._venues.values())

    def get_route_history(self, limit: int = 100) -> list[RouteDecision]:
        """Get routing history."""
        return self._route_history[-limit:]

    async def get_routing_summary(self) -> dict[str, Any]:
        """
        Get routing summary.

        Returns:
            Summary dictionary
        """
        venue_usage: dict[str, int] = {}
        total_routes = len(self._route_history)

        for decision in self._route_history:
            for route in decision.routes:
                venue_id = route.get("venue_id")
                if venue_id:
                    venue_usage[venue_id] = venue_usage.get(venue_id, 0) + 1

        return {
            "timestamp": datetime.now().isoformat(),
            "total_routes": total_routes,
            "active_venues": len([v for v in self._venues.values() if v.status == VenueStatus.CONNECTED]),
            "total_venues": len(self._venues),
            "venue_usage": venue_usage,
            "default_strategy": self.config.default_strategy.value,
            "dark_pools_enabled": self.config.enable_dark_pools,
        }
