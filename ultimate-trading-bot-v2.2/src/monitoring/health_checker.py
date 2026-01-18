"""
Health Checker for Ultimate Trading Bot v2.2.

Monitors system health, component status, and provides health endpoints.
"""

import asyncio
import logging
import os
import psutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentType(str, Enum):
    """Types of system components."""
    DATABASE = "database"
    CACHE = "cache"
    BROKER = "broker"
    DATA_FEED = "data_feed"
    ML_MODEL = "ml_model"
    EXTERNAL_API = "external_api"
    INTERNAL = "internal"


@dataclass
class HealthCheckConfig:
    """Configuration for health checks."""

    # Check intervals
    check_interval: float = 30.0  # seconds
    timeout: float = 10.0  # seconds per check

    # Thresholds
    cpu_threshold: float = 90.0  # percent
    memory_threshold: float = 90.0  # percent
    disk_threshold: float = 90.0  # percent
    latency_threshold: float = 5.0  # seconds

    # Degradation settings
    consecutive_failures_for_degraded: int = 2
    consecutive_failures_for_unhealthy: int = 5

    # Recovery settings
    consecutive_success_for_healthy: int = 3


@dataclass
class ComponentHealth:
    """Health status for a component."""

    name: str
    component_type: ComponentType
    status: HealthStatus = HealthStatus.UNKNOWN
    message: str = ""
    last_check: datetime | None = None
    response_time: float = 0.0

    # History
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_checks: int = 0
    total_failures: int = 0

    # Additional details
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.component_type.value,
            "status": self.status.value,
            "message": self.message,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "response_time_ms": self.response_time * 1000,
            "consecutive_failures": self.consecutive_failures,
            "total_checks": self.total_checks,
            "success_rate": (
                (self.total_checks - self.total_failures) / self.total_checks
                if self.total_checks > 0 else 0.0
            ),
            "details": self.details,
        }


@dataclass
class SystemHealth:
    """Overall system health status."""

    status: HealthStatus = HealthStatus.UNKNOWN
    timestamp: datetime = field(default_factory=datetime.now)

    # Resource usage
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0

    # Component health
    components: dict[str, ComponentHealth] = field(default_factory=dict)

    # Counts
    healthy_count: int = 0
    degraded_count: int = 0
    unhealthy_count: int = 0

    # Uptime
    start_time: datetime | None = None
    uptime_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "resources": {
                "cpu_percent": self.cpu_percent,
                "memory_percent": self.memory_percent,
                "disk_percent": self.disk_percent,
            },
            "components": {
                name: comp.to_dict()
                for name, comp in self.components.items()
            },
            "summary": {
                "healthy": self.healthy_count,
                "degraded": self.degraded_count,
                "unhealthy": self.unhealthy_count,
            },
            "uptime_seconds": self.uptime_seconds,
        }


HealthCheckFunction = Callable[[], Coroutine[Any, Any, tuple[bool, str, dict[str, Any]]]]


class HealthChecker:
    """
    System health monitoring and checking.

    Monitors component health, system resources, and provides health endpoints.
    """

    def __init__(self, config: HealthCheckConfig | None = None) -> None:
        """
        Initialize health checker.

        Args:
            config: Health check configuration
        """
        self.config = config or HealthCheckConfig()

        # Registered health checks
        self._checks: dict[str, tuple[ComponentType, HealthCheckFunction]] = {}
        self._health: dict[str, ComponentHealth] = {}

        # System state
        self._start_time = datetime.now()
        self._running = False
        self._check_task: asyncio.Task | None = None

        # Callbacks
        self._status_callbacks: list[Callable[[str, HealthStatus, HealthStatus], None]] = []

        logger.info("HealthChecker initialized")

    def register_check(
        self,
        name: str,
        check_fn: HealthCheckFunction,
        component_type: ComponentType = ComponentType.INTERNAL,
    ) -> None:
        """
        Register a health check.

        Args:
            name: Check name
            check_fn: Async function that returns (success, message, details)
            component_type: Type of component
        """
        self._checks[name] = (component_type, check_fn)
        self._health[name] = ComponentHealth(
            name=name,
            component_type=component_type,
        )
        logger.debug(f"Registered health check: {name}")

    def unregister_check(self, name: str) -> None:
        """Unregister a health check."""
        self._checks.pop(name, None)
        self._health.pop(name, None)

    def add_status_callback(
        self,
        callback: Callable[[str, HealthStatus, HealthStatus], None],
    ) -> None:
        """
        Add callback for status changes.

        Args:
            callback: Function called with (component_name, old_status, new_status)
        """
        self._status_callbacks.append(callback)

    async def check_component(self, name: str) -> ComponentHealth:
        """
        Run health check for a specific component.

        Args:
            name: Component name

        Returns:
            Component health status
        """
        if name not in self._checks:
            return ComponentHealth(
                name=name,
                component_type=ComponentType.INTERNAL,
                status=HealthStatus.UNKNOWN,
                message="Component not registered",
            )

        component_type, check_fn = self._checks[name]
        health = self._health[name]
        old_status = health.status

        start_time = datetime.now()

        try:
            # Run check with timeout
            success, message, details = await asyncio.wait_for(
                check_fn(),
                timeout=self.config.timeout,
            )

            response_time = (datetime.now() - start_time).total_seconds()

            health.total_checks += 1
            health.last_check = datetime.now()
            health.response_time = response_time
            health.details = details

            if success:
                health.consecutive_successes += 1
                health.consecutive_failures = 0
                health.message = message or "OK"

                # Check for recovery
                if health.consecutive_successes >= self.config.consecutive_success_for_healthy:
                    health.status = HealthStatus.HEALTHY
                elif health.status == HealthStatus.UNHEALTHY:
                    health.status = HealthStatus.DEGRADED

            else:
                health.consecutive_failures += 1
                health.consecutive_successes = 0
                health.total_failures += 1
                health.message = message or "Check failed"

                # Determine status based on consecutive failures
                if health.consecutive_failures >= self.config.consecutive_failures_for_unhealthy:
                    health.status = HealthStatus.UNHEALTHY
                elif health.consecutive_failures >= self.config.consecutive_failures_for_degraded:
                    health.status = HealthStatus.DEGRADED

            # Check latency
            if response_time > self.config.latency_threshold:
                if health.status == HealthStatus.HEALTHY:
                    health.status = HealthStatus.DEGRADED
                health.message += f" (slow: {response_time:.2f}s)"

        except asyncio.TimeoutError:
            health.consecutive_failures += 1
            health.consecutive_successes = 0
            health.total_failures += 1
            health.total_checks += 1
            health.last_check = datetime.now()
            health.message = f"Timeout after {self.config.timeout}s"
            health.status = HealthStatus.UNHEALTHY

        except Exception as e:
            health.consecutive_failures += 1
            health.consecutive_successes = 0
            health.total_failures += 1
            health.total_checks += 1
            health.last_check = datetime.now()
            health.message = f"Error: {str(e)}"
            health.status = HealthStatus.UNHEALTHY
            logger.error(f"Health check error for {name}: {e}")

        # Notify on status change
        if health.status != old_status:
            for callback in self._status_callbacks:
                try:
                    callback(name, old_status, health.status)
                except Exception as e:
                    logger.error(f"Status callback error: {e}")

        return health

    async def check_all(self) -> SystemHealth:
        """
        Run all health checks.

        Returns:
            Overall system health
        """
        # Check all components
        tasks = [
            self.check_component(name)
            for name in self._checks.keys()
        ]

        if tasks:
            await asyncio.gather(*tasks)

        # Get system resources
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Determine overall status
        healthy_count = sum(
            1 for h in self._health.values()
            if h.status == HealthStatus.HEALTHY
        )
        degraded_count = sum(
            1 for h in self._health.values()
            if h.status == HealthStatus.DEGRADED
        )
        unhealthy_count = sum(
            1 for h in self._health.values()
            if h.status == HealthStatus.UNHEALTHY
        )

        # Determine overall status
        if unhealthy_count > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
        elif cpu_percent > self.config.cpu_threshold:
            overall_status = HealthStatus.DEGRADED
        elif memory.percent > self.config.memory_threshold:
            overall_status = HealthStatus.DEGRADED
        elif disk.percent > self.config.disk_threshold:
            overall_status = HealthStatus.DEGRADED
        elif healthy_count > 0:
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN

        uptime = (datetime.now() - self._start_time).total_seconds()

        return SystemHealth(
            status=overall_status,
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_percent=disk.percent,
            components=self._health.copy(),
            healthy_count=healthy_count,
            degraded_count=degraded_count,
            unhealthy_count=unhealthy_count,
            start_time=self._start_time,
            uptime_seconds=uptime,
        )

    async def get_health(self) -> SystemHealth:
        """Get current health without running checks."""
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        healthy_count = sum(
            1 for h in self._health.values()
            if h.status == HealthStatus.HEALTHY
        )
        degraded_count = sum(
            1 for h in self._health.values()
            if h.status == HealthStatus.DEGRADED
        )
        unhealthy_count = sum(
            1 for h in self._health.values()
            if h.status == HealthStatus.UNHEALTHY
        )

        if unhealthy_count > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_count > 0 or cpu_percent > self.config.cpu_threshold:
            overall_status = HealthStatus.DEGRADED
        elif healthy_count > 0:
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN

        uptime = (datetime.now() - self._start_time).total_seconds()

        return SystemHealth(
            status=overall_status,
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_percent=disk.percent,
            components=self._health.copy(),
            healthy_count=healthy_count,
            degraded_count=degraded_count,
            unhealthy_count=unhealthy_count,
            start_time=self._start_time,
            uptime_seconds=uptime,
        )

    def get_component_health(self, name: str) -> ComponentHealth | None:
        """Get health for a specific component."""
        return self._health.get(name)

    async def start(self) -> None:
        """Start periodic health checking."""
        if self._running:
            return

        self._running = True
        self._check_task = asyncio.create_task(self._check_loop())
        logger.info("Health checking started")

    async def stop(self) -> None:
        """Stop health checking."""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass

        logger.info("Health checking stopped")

    async def _check_loop(self) -> None:
        """Main health check loop."""
        while self._running:
            try:
                await self.check_all()
                await asyncio.sleep(self.config.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(self.config.check_interval)

    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        unhealthy = any(
            h.status == HealthStatus.UNHEALTHY
            for h in self._health.values()
        )
        return not unhealthy

    def is_ready(self) -> bool:
        """Check if system is ready to serve requests."""
        # At least some components should be healthy
        healthy = any(
            h.status == HealthStatus.HEALTHY
            for h in self._health.values()
        )
        return healthy or len(self._health) == 0


# Common health check functions
async def check_database_health(
    connection_check: Callable[[], Coroutine[Any, Any, bool]],
) -> tuple[bool, str, dict[str, Any]]:
    """
    Create database health check.

    Args:
        connection_check: Async function to check database connection
    """
    try:
        result = await connection_check()
        if result:
            return True, "Database connected", {}
        return False, "Database connection failed", {}
    except Exception as e:
        return False, f"Database error: {e}", {}


async def check_redis_health(
    ping_fn: Callable[[], Coroutine[Any, Any, bool]],
) -> tuple[bool, str, dict[str, Any]]:
    """
    Create Redis health check.

    Args:
        ping_fn: Async function to ping Redis
    """
    try:
        result = await ping_fn()
        if result:
            return True, "Redis connected", {}
        return False, "Redis ping failed", {}
    except Exception as e:
        return False, f"Redis error: {e}", {}


def create_health_checker(
    config: HealthCheckConfig | None = None,
) -> HealthChecker:
    """
    Create a health checker instance.

    Args:
        config: Health check configuration

    Returns:
        HealthChecker instance
    """
    return HealthChecker(config)
