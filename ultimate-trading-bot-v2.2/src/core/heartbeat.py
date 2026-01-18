"""
Heartbeat Module for Ultimate Trading Bot v2.2.

This module provides health monitoring and heartbeat functionality
for system components, detecting failures and triggering alerts.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from src.utils.exceptions import InitializationError
from src.utils.helpers import generate_uuid
from src.utils.date_utils import now_utc
from src.utils.decorators import singleton


logger = logging.getLogger(__name__)


class ComponentStatus(str, Enum):
    """Component health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    STARTING = "starting"
    STOPPING = "stopping"


class HealthCheckResult(BaseModel):
    """Result of a health check."""

    component_id: str
    component_name: str
    status: ComponentStatus = Field(default=ComponentStatus.UNKNOWN)
    message: str = Field(default="")
    latency_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=now_utc)
    details: dict = Field(default_factory=dict)


class ComponentConfig(BaseModel):
    """Configuration for a monitored component."""

    component_id: str = Field(default_factory=generate_uuid)
    name: str
    description: str = Field(default="")

    check_interval_seconds: int = Field(default=30, ge=5, le=300)
    timeout_seconds: int = Field(default=10, ge=1, le=60)
    failure_threshold: int = Field(default=3, ge=1, le=10)
    recovery_threshold: int = Field(default=2, ge=1, le=10)

    critical: bool = Field(default=False)
    enabled: bool = Field(default=True)

    metadata: dict = Field(default_factory=dict)


class ComponentState(BaseModel):
    """Runtime state of a monitored component."""

    config: ComponentConfig
    status: ComponentStatus = Field(default=ComponentStatus.UNKNOWN)
    last_check: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None

    consecutive_failures: int = Field(default=0)
    consecutive_successes: int = Field(default=0)
    total_checks: int = Field(default=0)
    total_failures: int = Field(default=0)

    last_result: Optional[HealthCheckResult] = None
    last_error: Optional[str] = None

    @property
    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self.status == ComponentStatus.HEALTHY

    @property
    def is_critical_failure(self) -> bool:
        """Check if this is a critical component failure."""
        return (
            self.config.critical
            and self.status == ComponentStatus.UNHEALTHY
        )

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_checks == 0:
            return 0.0
        return (self.total_failures / self.total_checks) * 100

    @property
    def uptime_seconds(self) -> float:
        """Calculate uptime since last failure."""
        if not self.last_failure or not self.last_success:
            return 0.0
        if self.last_success < self.last_failure:
            return 0.0
        return (now_utc() - self.last_failure).total_seconds()


class HeartbeatConfig(BaseModel):
    """Configuration for heartbeat monitor."""

    tick_interval_seconds: float = Field(default=1.0, ge=0.5, le=10.0)
    default_check_interval_seconds: int = Field(default=30, ge=5, le=300)
    default_timeout_seconds: int = Field(default=10, ge=1, le=60)
    enable_auto_recovery: bool = Field(default=True)
    max_history_per_component: int = Field(default=100, ge=10, le=1000)
    alert_on_degraded: bool = Field(default=True)
    alert_on_recovery: bool = Field(default=True)


@singleton
class HeartbeatMonitor:
    """
    Monitors health of system components.

    This class provides:
    - Periodic health checks for components
    - Failure detection and alerting
    - Recovery tracking
    - System-wide health status
    """

    def __init__(
        self,
        config: Optional[HeartbeatConfig] = None,
    ) -> None:
        """
        Initialize HeartbeatMonitor.

        Args:
            config: Heartbeat configuration
        """
        self._config = config or HeartbeatConfig()

        self._components: dict[str, ComponentState] = {}
        self._health_checks: dict[str, Callable[[], Any]] = {}

        self._check_history: dict[str, list[HealthCheckResult]] = {}

        self._alert_callbacks: list[Callable[[str, ComponentStatus, str], None]] = []
        self._status_callbacks: list[Callable[[dict[str, ComponentStatus]], None]] = []

        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = asyncio.Lock()

        logger.info("HeartbeatMonitor initialized")

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running

    @property
    def component_count(self) -> int:
        """Get number of monitored components."""
        return len(self._components)

    @property
    def system_status(self) -> ComponentStatus:
        """Get overall system health status."""
        if not self._components:
            return ComponentStatus.UNKNOWN

        statuses = [c.status for c in self._components.values()]

        if any(s == ComponentStatus.UNHEALTHY for s in statuses):
            critical_unhealthy = any(
                c.is_critical_failure for c in self._components.values()
            )
            if critical_unhealthy:
                return ComponentStatus.UNHEALTHY

        if any(s == ComponentStatus.UNHEALTHY for s in statuses):
            return ComponentStatus.DEGRADED

        if any(s == ComponentStatus.DEGRADED for s in statuses):
            return ComponentStatus.DEGRADED

        if all(s == ComponentStatus.HEALTHY for s in statuses):
            return ComponentStatus.HEALTHY

        return ComponentStatus.UNKNOWN

    def register_alert_callback(
        self,
        callback: Callable[[str, ComponentStatus, str], None]
    ) -> None:
        """Register callback for health alerts."""
        self._alert_callbacks.append(callback)

    def register_status_callback(
        self,
        callback: Callable[[dict[str, ComponentStatus]], None]
    ) -> None:
        """Register callback for status updates."""
        self._status_callbacks.append(callback)

    def register_component(
        self,
        name: str,
        health_check: Callable[[], Any],
        check_interval_seconds: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        failure_threshold: int = 3,
        recovery_threshold: int = 2,
        critical: bool = False,
        description: str = "",
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Register a component for health monitoring.

        Args:
            name: Component name
            health_check: Health check function
            check_interval_seconds: Check interval
            timeout_seconds: Check timeout
            failure_threshold: Failures before unhealthy
            recovery_threshold: Successes before healthy
            critical: Whether component is critical
            description: Component description
            metadata: Additional metadata

        Returns:
            Component ID
        """
        config = ComponentConfig(
            name=name,
            description=description,
            check_interval_seconds=check_interval_seconds or self._config.default_check_interval_seconds,
            timeout_seconds=timeout_seconds or self._config.default_timeout_seconds,
            failure_threshold=failure_threshold,
            recovery_threshold=recovery_threshold,
            critical=critical,
            metadata=metadata or {},
        )

        state = ComponentState(
            config=config,
            status=ComponentStatus.UNKNOWN,
        )

        self._components[config.component_id] = state
        self._health_checks[config.component_id] = health_check
        self._check_history[config.component_id] = []

        logger.info(
            f"Registered component '{name}' (id={config.component_id}, "
            f"critical={critical})"
        )

        return config.component_id

    def unregister_component(self, component_id: str) -> bool:
        """
        Unregister a component from monitoring.

        Args:
            component_id: Component to unregister

        Returns:
            True if unregistered
        """
        if component_id not in self._components:
            return False

        name = self._components[component_id].config.name

        del self._components[component_id]
        del self._health_checks[component_id]
        del self._check_history[component_id]

        logger.info(f"Unregistered component '{name}'")
        return True

    async def start(self) -> None:
        """Start the heartbeat monitor."""
        if self._running:
            logger.warning("HeartbeatMonitor already running")
            return

        self._running = True

        self._monitor_task = asyncio.create_task(
            self._monitor_loop(),
            name="heartbeat_monitor"
        )

        logger.info("HeartbeatMonitor started")

    async def stop(self) -> None:
        """Stop the heartbeat monitor."""
        if not self._running:
            return

        self._running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        logger.info("HeartbeatMonitor stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                now = now_utc()

                for component_id, state in self._components.items():
                    if not state.config.enabled:
                        continue

                    should_check = (
                        state.last_check is None
                        or (now - state.last_check).total_seconds()
                        >= state.config.check_interval_seconds
                    )

                    if should_check:
                        asyncio.create_task(
                            self._check_component(component_id),
                            name=f"health_check_{component_id}"
                        )

                await asyncio.sleep(self._config.tick_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(1)

    async def _check_component(self, component_id: str) -> HealthCheckResult:
        """
        Perform health check on a component.

        Args:
            component_id: Component to check

        Returns:
            Health check result
        """
        state = self._components.get(component_id)
        if not state:
            return HealthCheckResult(
                component_id=component_id,
                component_name="unknown",
                status=ComponentStatus.UNKNOWN,
                message="Component not found",
            )

        health_check = self._health_checks.get(component_id)
        if not health_check:
            return HealthCheckResult(
                component_id=component_id,
                component_name=state.config.name,
                status=ComponentStatus.UNKNOWN,
                message="No health check function",
            )

        start_time = datetime.now()
        result = HealthCheckResult(
            component_id=component_id,
            component_name=state.config.name,
        )

        try:
            if asyncio.iscoroutinefunction(health_check):
                check_result = await asyncio.wait_for(
                    health_check(),
                    timeout=state.config.timeout_seconds
                )
            else:
                check_result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, health_check),
                    timeout=state.config.timeout_seconds
                )

            end_time = datetime.now()
            result.latency_ms = (end_time - start_time).total_seconds() * 1000

            if isinstance(check_result, dict):
                result.status = ComponentStatus(
                    check_result.get("status", "healthy")
                )
                result.message = check_result.get("message", "OK")
                result.details = check_result.get("details", {})
            elif isinstance(check_result, bool):
                result.status = (
                    ComponentStatus.HEALTHY if check_result
                    else ComponentStatus.UNHEALTHY
                )
                result.message = "OK" if check_result else "Check failed"
            else:
                result.status = ComponentStatus.HEALTHY
                result.message = "OK"

        except asyncio.TimeoutError:
            result.status = ComponentStatus.UNHEALTHY
            result.message = f"Health check timeout ({state.config.timeout_seconds}s)"

        except Exception as e:
            result.status = ComponentStatus.UNHEALTHY
            result.message = f"Health check error: {e}"
            logger.error(f"Health check error for {state.config.name}: {e}")

        await self._update_component_state(component_id, result)

        return result

    async def _update_component_state(
        self,
        component_id: str,
        result: HealthCheckResult
    ) -> None:
        """Update component state based on check result."""
        async with self._lock:
            state = self._components.get(component_id)
            if not state:
                return

            previous_status = state.status
            state.last_check = result.timestamp
            state.last_result = result
            state.total_checks += 1

            if result.status == ComponentStatus.HEALTHY:
                state.consecutive_failures = 0
                state.consecutive_successes += 1
                state.last_success = result.timestamp

                if state.consecutive_successes >= state.config.recovery_threshold:
                    state.status = ComponentStatus.HEALTHY
                elif state.status == ComponentStatus.UNHEALTHY:
                    state.status = ComponentStatus.DEGRADED

            else:
                state.consecutive_successes = 0
                state.consecutive_failures += 1
                state.total_failures += 1
                state.last_failure = result.timestamp
                state.last_error = result.message

                if state.consecutive_failures >= state.config.failure_threshold:
                    state.status = ComponentStatus.UNHEALTHY
                elif state.status == ComponentStatus.HEALTHY:
                    state.status = ComponentStatus.DEGRADED

            self._check_history[component_id].append(result)
            self._trim_history(component_id)

            if state.status != previous_status:
                await self._notify_status_change(
                    component_id,
                    state,
                    previous_status
                )

    async def _notify_status_change(
        self,
        component_id: str,
        state: ComponentState,
        previous_status: ComponentStatus
    ) -> None:
        """Notify callbacks of status change."""
        message = f"{state.config.name}: {previous_status.value} -> {state.status.value}"

        if state.status == ComponentStatus.UNHEALTHY:
            logger.warning(f"Component unhealthy: {message}")
        elif state.status == ComponentStatus.HEALTHY and previous_status == ComponentStatus.UNHEALTHY:
            logger.info(f"Component recovered: {message}")

        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(component_id, state.status, message)
                else:
                    callback(component_id, state.status, message)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

        status_dict = {
            cid: c.status for cid, c in self._components.items()
        }
        for callback in self._status_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(status_dict)
                else:
                    callback(status_dict)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")

    def _trim_history(self, component_id: str) -> None:
        """Trim check history to max size."""
        max_size = self._config.max_history_per_component
        if len(self._check_history[component_id]) > max_size:
            self._check_history[component_id] = self._check_history[component_id][-max_size:]

    async def check_now(self, component_id: str) -> Optional[HealthCheckResult]:
        """
        Perform immediate health check.

        Args:
            component_id: Component to check

        Returns:
            Health check result
        """
        if component_id not in self._components:
            return None

        return await self._check_component(component_id)

    async def check_all(self) -> dict[str, HealthCheckResult]:
        """
        Check all components immediately.

        Returns:
            Dictionary of component ID to result
        """
        tasks = {
            cid: self._check_component(cid)
            for cid in self._components
        }

        results = {}
        for cid, task in tasks.items():
            results[cid] = await task

        return results

    def get_component_status(self, component_id: str) -> Optional[ComponentStatus]:
        """Get status of a specific component."""
        state = self._components.get(component_id)
        return state.status if state else None

    def get_component_state(self, component_id: str) -> Optional[ComponentState]:
        """Get full state of a component."""
        return self._components.get(component_id)

    def get_all_statuses(self) -> dict[str, ComponentStatus]:
        """Get status of all components."""
        return {
            cid: state.status
            for cid, state in self._components.items()
        }

    def get_healthy_components(self) -> list[str]:
        """Get IDs of healthy components."""
        return [
            cid for cid, state in self._components.items()
            if state.is_healthy
        ]

    def get_unhealthy_components(self) -> list[str]:
        """Get IDs of unhealthy components."""
        return [
            cid for cid, state in self._components.items()
            if state.status == ComponentStatus.UNHEALTHY
        ]

    def get_critical_components(self) -> list[str]:
        """Get IDs of critical components."""
        return [
            cid for cid, state in self._components.items()
            if state.config.critical
        ]

    def get_component_history(
        self,
        component_id: str,
        limit: int = 50
    ) -> list[HealthCheckResult]:
        """Get check history for a component."""
        history = self._check_history.get(component_id, [])
        return history[-limit:]

    def enable_component(self, component_id: str) -> bool:
        """Enable monitoring for a component."""
        if component_id not in self._components:
            return False

        self._components[component_id].config.enabled = True
        return True

    def disable_component(self, component_id: str) -> bool:
        """Disable monitoring for a component."""
        if component_id not in self._components:
            return False

        self._components[component_id].config.enabled = False
        return True

    def get_health_summary(self) -> dict:
        """Get comprehensive health summary."""
        total = len(self._components)
        healthy = len(self.get_healthy_components())
        unhealthy = len(self.get_unhealthy_components())
        degraded = sum(
            1 for c in self._components.values()
            if c.status == ComponentStatus.DEGRADED
        )

        return {
            "system_status": self.system_status.value,
            "total_components": total,
            "healthy_components": healthy,
            "unhealthy_components": unhealthy,
            "degraded_components": degraded,
            "health_percentage": (healthy / total * 100) if total > 0 else 0,
            "critical_failures": sum(
                1 for c in self._components.values()
                if c.is_critical_failure
            ),
            "components": {
                cid: {
                    "name": state.config.name,
                    "status": state.status.value,
                    "critical": state.config.critical,
                    "last_check": state.last_check.isoformat() if state.last_check else None,
                    "consecutive_failures": state.consecutive_failures,
                    "failure_rate": state.failure_rate,
                }
                for cid, state in self._components.items()
            },
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"HeartbeatMonitor(components={len(self._components)}, "
            f"status={self.system_status.value})"
        )
