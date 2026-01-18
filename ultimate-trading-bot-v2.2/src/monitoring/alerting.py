"""
Alerting System for Ultimate Trading Bot v2.2.

Provides alert generation, routing, and management for trading and system events.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertCategory(str, Enum):
    """Categories of alerts."""
    SYSTEM = "system"
    TRADING = "trading"
    RISK = "risk"
    PERFORMANCE = "performance"
    DATA = "data"
    CONNECTIVITY = "connectivity"
    SECURITY = "security"


class AlertStatus(str, Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class AlertConfig:
    """Configuration for alerting system."""

    # Rate limiting
    min_interval_seconds: float = 60.0  # Minimum time between same alerts
    max_alerts_per_hour: int = 100

    # Suppression
    suppression_window: int = 3600  # seconds
    max_repetitions: int = 5

    # Escalation
    escalation_timeout: int = 300  # seconds before escalation
    auto_resolve_timeout: int = 3600  # seconds before auto-resolve


@dataclass
class Alert:
    """Represents an alert."""

    alert_id: str
    title: str
    message: str
    severity: AlertSeverity
    category: AlertCategory

    # Timing
    created_at: datetime
    updated_at: datetime | None = None
    resolved_at: datetime | None = None

    # Status
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: str | None = None
    resolved_by: str | None = None

    # Details
    source: str = ""
    symbol: str | None = None
    value: float | None = None
    threshold: float | None = None

    # Metadata
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Tracking
    notification_count: int = 0
    escalation_level: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "source": self.source,
            "symbol": self.symbol,
            "value": self.value,
            "threshold": self.threshold,
            "tags": self.tags,
            "metadata": self.metadata,
            "notification_count": self.notification_count,
            "escalation_level": self.escalation_level,
        }


@dataclass
class AlertRule:
    """Rule for generating alerts."""

    name: str
    condition: Callable[[Any], bool]
    severity: AlertSeverity
    category: AlertCategory
    title_template: str
    message_template: str

    # Options
    enabled: bool = True
    cooldown_seconds: float = 60.0
    tags: list[str] = field(default_factory=list)

    # State
    last_triggered: datetime | None = None
    trigger_count: int = 0


AlertHandler = Callable[[Alert], Coroutine[Any, Any, bool]]


class AlertManager:
    """
    Central alert management system.

    Handles alert creation, routing, escalation, and notification.
    """

    def __init__(self, config: AlertConfig | None = None) -> None:
        """
        Initialize alert manager.

        Args:
            config: Alert configuration
        """
        self.config = config or AlertConfig()

        # Alert storage
        self._alerts: dict[str, Alert] = {}
        self._alert_history: list[Alert] = []

        # Rules
        self._rules: dict[str, AlertRule] = {}

        # Handlers
        self._handlers: dict[AlertSeverity, list[AlertHandler]] = {
            severity: [] for severity in AlertSeverity
        }
        self._global_handlers: list[AlertHandler] = []

        # Suppression
        self._suppressed: dict[str, datetime] = {}
        self._alert_counts: dict[str, list[datetime]] = {}

        # Counter for IDs
        self._id_counter = 0

        # Background tasks
        self._escalation_task: asyncio.Task | None = None
        self._running = False

        logger.info("AlertManager initialized")

    def _generate_id(self) -> str:
        """Generate unique alert ID."""
        self._id_counter += 1
        return f"alert_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._id_counter}"

    def _get_dedup_key(self, alert: Alert) -> str:
        """Get deduplication key for alert."""
        return f"{alert.category.value}:{alert.source}:{alert.title}"

    async def create_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        category: AlertCategory,
        source: str = "",
        symbol: str | None = None,
        value: float | None = None,
        threshold: float | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Alert | None:
        """
        Create and process a new alert.

        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity
            category: Alert category
            source: Alert source
            symbol: Related trading symbol
            value: Current value that triggered alert
            threshold: Threshold that was breached
            tags: Alert tags
            metadata: Additional metadata

        Returns:
            Created alert or None if suppressed
        """
        alert = Alert(
            alert_id=self._generate_id(),
            title=title,
            message=message,
            severity=severity,
            category=category,
            created_at=datetime.now(),
            source=source,
            symbol=symbol,
            value=value,
            threshold=threshold,
            tags=tags or [],
            metadata=metadata or {},
        )

        # Check suppression
        dedup_key = self._get_dedup_key(alert)
        if self._is_suppressed(dedup_key):
            logger.debug(f"Alert suppressed: {title}")
            return None

        # Check rate limit
        if not self._check_rate_limit():
            logger.warning("Alert rate limit reached")
            return None

        # Store alert
        self._alerts[alert.alert_id] = alert
        self._alert_history.append(alert)

        # Update suppression tracking
        self._update_suppression(dedup_key)

        # Send to handlers
        await self._send_to_handlers(alert)

        logger.info(
            f"Alert created: [{severity.value}] {title} ({alert.alert_id})"
        )

        return alert

    async def acknowledge_alert(
        self,
        alert_id: str,
        user: str,
    ) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert ID
            user: User acknowledging the alert

        Returns:
            True if acknowledged
        """
        if alert_id not in self._alerts:
            return False

        alert = self._alerts[alert_id]
        if alert.status != AlertStatus.ACTIVE:
            return False

        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = user
        alert.updated_at = datetime.now()

        logger.info(f"Alert acknowledged: {alert_id} by {user}")
        return True

    async def resolve_alert(
        self,
        alert_id: str,
        user: str | None = None,
        reason: str = "",
    ) -> bool:
        """
        Resolve an alert.

        Args:
            alert_id: Alert ID
            user: User resolving the alert
            reason: Resolution reason

        Returns:
            True if resolved
        """
        if alert_id not in self._alerts:
            return False

        alert = self._alerts[alert_id]
        if alert.status == AlertStatus.RESOLVED:
            return False

        alert.status = AlertStatus.RESOLVED
        alert.resolved_by = user or "system"
        alert.resolved_at = datetime.now()
        alert.updated_at = datetime.now()

        if reason:
            alert.metadata["resolution_reason"] = reason

        logger.info(f"Alert resolved: {alert_id}")
        return True

    def add_rule(self, rule: AlertRule) -> None:
        """
        Add an alert rule.

        Args:
            rule: Alert rule to add
        """
        self._rules[rule.name] = rule
        logger.debug(f"Added alert rule: {rule.name}")

    def remove_rule(self, name: str) -> None:
        """Remove an alert rule."""
        self._rules.pop(name, None)

    async def evaluate_rules(self, context: dict[str, Any]) -> list[Alert]:
        """
        Evaluate all rules against context.

        Args:
            context: Context data for rule evaluation

        Returns:
            List of created alerts
        """
        alerts = []

        for rule in self._rules.values():
            if not rule.enabled:
                continue

            # Check cooldown
            if rule.last_triggered:
                elapsed = (datetime.now() - rule.last_triggered).total_seconds()
                if elapsed < rule.cooldown_seconds:
                    continue

            try:
                if rule.condition(context):
                    # Format title and message
                    title = rule.title_template.format(**context)
                    message = rule.message_template.format(**context)

                    alert = await self.create_alert(
                        title=title,
                        message=message,
                        severity=rule.severity,
                        category=rule.category,
                        source=rule.name,
                        tags=rule.tags,
                        metadata={"rule": rule.name, "context": context},
                    )

                    if alert:
                        alerts.append(alert)
                        rule.last_triggered = datetime.now()
                        rule.trigger_count += 1

            except Exception as e:
                logger.error(f"Rule evaluation error for {rule.name}: {e}")

        return alerts

    def add_handler(
        self,
        handler: AlertHandler,
        severity: AlertSeverity | None = None,
    ) -> None:
        """
        Add alert handler.

        Args:
            handler: Handler function
            severity: Specific severity to handle (None for all)
        """
        if severity is None:
            self._global_handlers.append(handler)
        else:
            self._handlers[severity].append(handler)

    async def _send_to_handlers(self, alert: Alert) -> None:
        """Send alert to all registered handlers."""
        handlers = (
            self._handlers[alert.severity] +
            self._global_handlers
        )

        for handler in handlers:
            try:
                success = await handler(alert)
                if success:
                    alert.notification_count += 1
            except Exception as e:
                logger.error(f"Handler error: {e}")

    def _is_suppressed(self, dedup_key: str) -> bool:
        """Check if alert should be suppressed."""
        if dedup_key in self._suppressed:
            suppressed_until = self._suppressed[dedup_key]
            if datetime.now() < suppressed_until:
                return True
            else:
                del self._suppressed[dedup_key]

        # Check repetition count
        if dedup_key in self._alert_counts:
            recent = [
                t for t in self._alert_counts[dedup_key]
                if t > datetime.now() - timedelta(seconds=self.config.suppression_window)
            ]
            self._alert_counts[dedup_key] = recent

            if len(recent) >= self.config.max_repetitions:
                # Suppress for suppression window
                self._suppressed[dedup_key] = (
                    datetime.now() + timedelta(seconds=self.config.suppression_window)
                )
                return True

        return False

    def _update_suppression(self, dedup_key: str) -> None:
        """Update suppression tracking."""
        if dedup_key not in self._alert_counts:
            self._alert_counts[dedup_key] = []

        self._alert_counts[dedup_key].append(datetime.now())

    def _check_rate_limit(self) -> bool:
        """Check if within rate limit."""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)

        recent_count = sum(
            1 for alert in self._alert_history
            if alert.created_at > hour_ago
        )

        return recent_count < self.config.max_alerts_per_hour

    def get_active_alerts(
        self,
        category: AlertCategory | None = None,
        severity: AlertSeverity | None = None,
    ) -> list[Alert]:
        """
        Get active alerts with optional filters.

        Args:
            category: Filter by category
            severity: Filter by severity

        Returns:
            List of active alerts
        """
        alerts = [
            a for a in self._alerts.values()
            if a.status == AlertStatus.ACTIVE
        ]

        if category:
            alerts = [a for a in alerts if a.category == category]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return sorted(alerts, key=lambda a: a.created_at, reverse=True)

    def get_alert(self, alert_id: str) -> Alert | None:
        """Get alert by ID."""
        return self._alerts.get(alert_id)

    def get_alert_history(
        self,
        hours: int = 24,
        limit: int = 100,
    ) -> list[Alert]:
        """
        Get alert history.

        Args:
            hours: Hours of history
            limit: Maximum alerts to return

        Returns:
            List of historical alerts
        """
        cutoff = datetime.now() - timedelta(hours=hours)

        alerts = [
            a for a in self._alert_history
            if a.created_at > cutoff
        ]

        return sorted(
            alerts,
            key=lambda a: a.created_at,
            reverse=True,
        )[:limit]

    def get_statistics(self) -> dict[str, Any]:
        """Get alerting statistics."""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(hours=24)

        alerts_hour = [a for a in self._alert_history if a.created_at > hour_ago]
        alerts_day = [a for a in self._alert_history if a.created_at > day_ago]

        return {
            "active_alerts": len([a for a in self._alerts.values() if a.status == AlertStatus.ACTIVE]),
            "acknowledged_alerts": len([a for a in self._alerts.values() if a.status == AlertStatus.ACKNOWLEDGED]),
            "alerts_last_hour": len(alerts_hour),
            "alerts_last_day": len(alerts_day),
            "by_severity": {
                sev.value: len([a for a in alerts_day if a.severity == sev])
                for sev in AlertSeverity
            },
            "by_category": {
                cat.value: len([a for a in alerts_day if a.category == cat])
                for cat in AlertCategory
            },
            "active_rules": len([r for r in self._rules.values() if r.enabled]),
        }

    async def start(self) -> None:
        """Start alert manager background tasks."""
        if self._running:
            return

        self._running = True
        self._escalation_task = asyncio.create_task(self._escalation_loop())
        logger.info("Alert manager started")

    async def stop(self) -> None:
        """Stop alert manager."""
        self._running = False
        if self._escalation_task:
            self._escalation_task.cancel()
            try:
                await self._escalation_task
            except asyncio.CancelledError:
                pass

        logger.info("Alert manager stopped")

    async def _escalation_loop(self) -> None:
        """Background task for escalation and auto-resolve."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute

                now = datetime.now()

                for alert in list(self._alerts.values()):
                    if alert.status != AlertStatus.ACTIVE:
                        continue

                    age = (now - alert.created_at).total_seconds()

                    # Auto-resolve old alerts
                    if age > self.config.auto_resolve_timeout:
                        await self.resolve_alert(
                            alert.alert_id,
                            user="system",
                            reason="Auto-resolved due to timeout",
                        )
                        continue

                    # Escalate unacknowledged critical alerts
                    if (alert.severity == AlertSeverity.CRITICAL and
                        age > self.config.escalation_timeout and
                        alert.escalation_level == 0):
                        alert.escalation_level += 1
                        alert.updated_at = now
                        logger.warning(f"Alert escalated: {alert.alert_id}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Escalation loop error: {e}")


def create_alert_manager(
    config: AlertConfig | None = None,
) -> AlertManager:
    """
    Create an alert manager instance.

    Args:
        config: Alert configuration

    Returns:
        AlertManager instance
    """
    return AlertManager(config)


# Common alert rules
def create_price_alert_rule(
    symbol: str,
    threshold: float,
    direction: str = "above",
    severity: AlertSeverity = AlertSeverity.WARNING,
) -> AlertRule:
    """Create a price alert rule."""
    if direction == "above":
        condition = lambda ctx: ctx.get(f"price_{symbol}", 0) > threshold
        title = f"{symbol} price above ${threshold}"
    else:
        condition = lambda ctx: ctx.get(f"price_{symbol}", 0) < threshold
        title = f"{symbol} price below ${threshold}"

    return AlertRule(
        name=f"price_alert_{symbol}_{direction}_{threshold}",
        condition=condition,
        severity=severity,
        category=AlertCategory.TRADING,
        title_template=title,
        message_template=f"{{symbol}} price is now ${{price_{symbol}:.2f}}",
        tags=["price", symbol],
    )


def create_drawdown_alert_rule(
    threshold: float = 0.1,
    severity: AlertSeverity = AlertSeverity.ERROR,
) -> AlertRule:
    """Create a drawdown alert rule."""
    return AlertRule(
        name=f"drawdown_alert_{threshold}",
        condition=lambda ctx: ctx.get("drawdown", 0) > threshold,
        severity=severity,
        category=AlertCategory.RISK,
        title_template=f"Portfolio drawdown exceeds {threshold:.1%}",
        message_template="Current drawdown: {drawdown:.2%}",
        tags=["drawdown", "risk"],
    )
