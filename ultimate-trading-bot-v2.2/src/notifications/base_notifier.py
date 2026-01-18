"""
Base Notifier Module for Ultimate Trading Bot v2.2.

This module provides the base notification interface including:
- Abstract notifier interface
- Message formatting
- Priority and routing
- Retry logic
"""

import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable


logger = logging.getLogger(__name__)


class NotificationPriority(str, Enum):
    """Notification priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"

    def get_delay(self) -> float:
        """Get appropriate delay for batching based on priority."""
        delays = {
            "low": 300.0,      # 5 minutes
            "normal": 60.0,    # 1 minute
            "high": 10.0,      # 10 seconds
            "urgent": 1.0,     # 1 second
            "critical": 0.0,   # Immediate
        }
        return delays.get(self.value, 60.0)


class NotificationType(str, Enum):
    """Notification type enumeration."""

    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    ALERT = "alert"
    TRADE = "trade"
    ORDER = "order"
    SIGNAL = "signal"
    SYSTEM = "system"
    PERFORMANCE = "performance"


class NotificationChannel(str, Enum):
    """Notification channel enumeration."""

    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    PUSH = "push"
    WEBHOOK = "webhook"


@dataclass
class NotificationAttachment:
    """Attachment for notifications."""

    filename: str
    content: bytes
    content_type: str = "application/octet-stream"
    description: str | None = None


@dataclass
class NotificationMessage:
    """Notification message structure."""

    title: str
    body: str
    message_type: NotificationType = NotificationType.INFO
    priority: NotificationPriority = NotificationPriority.NORMAL
    channel: NotificationChannel | None = None
    recipient: str | None = None
    recipients: list[str] = field(default_factory=list)

    # Content
    html_body: str | None = None
    short_body: str | None = None
    attachments: list[NotificationAttachment] = field(default_factory=list)

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str | None = None
    correlation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Delivery options
    expires_at: datetime | None = None
    retry_count: int = 0
    max_retries: int = 3
    dedupe_key: str | None = None

    def __post_init__(self) -> None:
        """Post-init processing."""
        if self.message_id is None:
            self.message_id = self._generate_id()

        if self.recipient and self.recipient not in self.recipients:
            self.recipients.append(self.recipient)

    def _generate_id(self) -> str:
        """Generate unique message ID."""
        content = f"{self.title}{self.body}{self.timestamp.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get_dedupe_key(self) -> str:
        """Get deduplication key."""
        if self.dedupe_key:
            return self.dedupe_key
        return hashlib.sha256(f"{self.title}{self.body}".encode()).hexdigest()[:32]

    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def can_retry(self) -> bool:
        """Check if message can be retried."""
        return self.retry_count < self.max_retries


@dataclass
class DeliveryResult:
    """Result of notification delivery."""

    success: bool
    message_id: str
    channel: NotificationChannel
    recipient: str | None = None
    error: str | None = None
    delivery_time: datetime = field(default_factory=datetime.now)
    response_data: dict[str, Any] = field(default_factory=dict)
    retryable: bool = False


@dataclass
class NotifierConfig:
    """Base notifier configuration."""

    enabled: bool = True
    max_retries: int = 3
    retry_delay: float = 5.0
    retry_backoff: float = 2.0
    timeout: float = 30.0
    batch_size: int = 10
    batch_delay: float = 1.0
    rate_limit: int = 100
    rate_limit_period: float = 60.0
    dedupe_window: float = 300.0


class RateLimiter:
    """Simple rate limiter for notifications."""

    def __init__(
        self,
        max_requests: int = 100,
        period: float = 60.0,
    ) -> None:
        """Initialize rate limiter."""
        self._max_requests = max_requests
        self._period = period
        self._requests: list[datetime] = []

    def can_proceed(self) -> bool:
        """Check if request can proceed."""
        now = datetime.now()
        cutoff = now - timedelta(seconds=self._period)

        # Remove old requests
        self._requests = [r for r in self._requests if r > cutoff]

        return len(self._requests) < self._max_requests

    def record_request(self) -> None:
        """Record a request."""
        self._requests.append(datetime.now())

    async def wait_if_needed(self) -> None:
        """Wait if rate limited."""
        while not self.can_proceed():
            await asyncio.sleep(0.1)
        self.record_request()


class DeduplicationCache:
    """Cache for deduplicating notifications."""

    def __init__(self, window_seconds: float = 300.0) -> None:
        """Initialize deduplication cache."""
        self._window = window_seconds
        self._cache: dict[str, datetime] = {}

    def check_and_add(self, key: str) -> bool:
        """
        Check if key exists and add if not.

        Returns:
            True if this is a duplicate
        """
        self._cleanup()

        if key in self._cache:
            return True

        self._cache[key] = datetime.now()
        return False

    def _cleanup(self) -> None:
        """Remove expired entries."""
        cutoff = datetime.now() - timedelta(seconds=self._window)
        self._cache = {k: v for k, v in self._cache.items() if v > cutoff}


class MessageFormatter:
    """Formats notification messages for different channels."""

    @staticmethod
    def format_trade_notification(
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        pnl: float | None = None,
        **kwargs: Any,
    ) -> NotificationMessage:
        """Format a trade notification."""
        title = f"Trade Executed: {action.upper()} {symbol}"

        body_parts = [
            f"**{action.upper()} {quantity} {symbol}**",
            f"Price: ${price:,.2f}",
        ]

        if pnl is not None:
            sign = "+" if pnl >= 0 else ""
            body_parts.append(f"P&L: {sign}${pnl:,.2f}")

        if "order_id" in kwargs:
            body_parts.append(f"Order ID: {kwargs['order_id']}")

        body = "\n".join(body_parts)

        return NotificationMessage(
            title=title,
            body=body,
            message_type=NotificationType.TRADE,
            priority=NotificationPriority.HIGH,
            metadata={"symbol": symbol, "action": action, "quantity": quantity, "price": price, **kwargs},
        )

    @staticmethod
    def format_signal_notification(
        symbol: str,
        signal_type: str,
        strength: float,
        source: str,
        **kwargs: Any,
    ) -> NotificationMessage:
        """Format a signal notification."""
        title = f"Trading Signal: {signal_type.upper()} {symbol}"

        body_parts = [
            f"**{signal_type.upper()} Signal for {symbol}**",
            f"Strength: {strength:.1%}",
            f"Source: {source}",
        ]

        if "target_price" in kwargs:
            body_parts.append(f"Target: ${kwargs['target_price']:,.2f}")

        if "stop_loss" in kwargs:
            body_parts.append(f"Stop Loss: ${kwargs['stop_loss']:,.2f}")

        body = "\n".join(body_parts)

        priority = NotificationPriority.HIGH if strength > 0.8 else NotificationPriority.NORMAL

        return NotificationMessage(
            title=title,
            body=body,
            message_type=NotificationType.SIGNAL,
            priority=priority,
            metadata={"symbol": symbol, "signal_type": signal_type, "strength": strength, **kwargs},
        )

    @staticmethod
    def format_alert_notification(
        title: str,
        message: str,
        severity: str = "info",
        **kwargs: Any,
    ) -> NotificationMessage:
        """Format an alert notification."""
        priority_map = {
            "info": NotificationPriority.NORMAL,
            "warning": NotificationPriority.HIGH,
            "error": NotificationPriority.URGENT,
            "critical": NotificationPriority.CRITICAL,
        }

        type_map = {
            "info": NotificationType.INFO,
            "warning": NotificationType.WARNING,
            "error": NotificationType.ERROR,
            "critical": NotificationType.ALERT,
        }

        return NotificationMessage(
            title=f"[{severity.upper()}] {title}",
            body=message,
            message_type=type_map.get(severity, NotificationType.INFO),
            priority=priority_map.get(severity, NotificationPriority.NORMAL),
            metadata={"severity": severity, **kwargs},
        )

    @staticmethod
    def format_performance_notification(
        metric_name: str,
        value: float,
        change: float | None = None,
        period: str = "daily",
        **kwargs: Any,
    ) -> NotificationMessage:
        """Format a performance notification."""
        title = f"Performance Update: {metric_name}"

        body_parts = [
            f"**{metric_name}**",
            f"Value: {value:.2%}" if abs(value) < 10 else f"Value: {value:,.2f}",
        ]

        if change is not None:
            sign = "📈" if change >= 0 else "📉"
            body_parts.append(f"Change: {sign} {change:+.2%}")

        body_parts.append(f"Period: {period}")
        body = "\n".join(body_parts)

        return NotificationMessage(
            title=title,
            body=body,
            message_type=NotificationType.PERFORMANCE,
            priority=NotificationPriority.LOW,
            metadata={"metric": metric_name, "value": value, "change": change, **kwargs},
        )

    @staticmethod
    def format_system_notification(
        title: str,
        message: str,
        component: str | None = None,
        **kwargs: Any,
    ) -> NotificationMessage:
        """Format a system notification."""
        full_title = f"System: {title}"
        if component:
            full_title = f"System [{component}]: {title}"

        return NotificationMessage(
            title=full_title,
            body=message,
            message_type=NotificationType.SYSTEM,
            priority=NotificationPriority.NORMAL,
            metadata={"component": component, **kwargs},
        )


class BaseNotifier(ABC):
    """
    Abstract base class for notification channels.

    All notification implementations must inherit from this class.
    """

    def __init__(self, config: NotifierConfig | None = None) -> None:
        """Initialize base notifier."""
        self.config = config or NotifierConfig()
        self._rate_limiter = RateLimiter(
            self.config.rate_limit,
            self.config.rate_limit_period,
        )
        self._dedupe_cache = DeduplicationCache(self.config.dedupe_window)
        self._pending_messages: list[NotificationMessage] = []
        self._batch_task: asyncio.Task | None = None
        self._running = False
        self._callbacks: list[Callable[[DeliveryResult], None]] = []

    @property
    @abstractmethod
    def channel(self) -> NotificationChannel:
        """Get the notification channel type."""
        pass

    @abstractmethod
    async def _send(self, message: NotificationMessage) -> DeliveryResult:
        """
        Send a notification message.

        Args:
            message: Message to send

        Returns:
            Delivery result
        """
        pass

    @abstractmethod
    async def validate_config(self) -> bool:
        """
        Validate notifier configuration.

        Returns:
            True if configuration is valid
        """
        pass

    async def send(self, message: NotificationMessage) -> DeliveryResult:
        """
        Send a notification with retry and rate limiting.

        Args:
            message: Message to send

        Returns:
            Delivery result
        """
        if not self.config.enabled:
            return DeliveryResult(
                success=False,
                message_id=message.message_id or "",
                channel=self.channel,
                error="Notifier is disabled",
            )

        # Check for duplicates
        if self._dedupe_cache.check_and_add(message.get_dedupe_key()):
            logger.debug(f"Duplicate message skipped: {message.message_id}")
            return DeliveryResult(
                success=True,
                message_id=message.message_id or "",
                channel=self.channel,
                error="Duplicate message (skipped)",
            )

        # Check expiration
        if message.is_expired():
            return DeliveryResult(
                success=False,
                message_id=message.message_id or "",
                channel=self.channel,
                error="Message expired",
            )

        # Apply rate limiting
        await self._rate_limiter.wait_if_needed()

        # Try sending with retries
        last_error: str | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    self._send(message),
                    timeout=self.config.timeout,
                )

                if result.success:
                    self._notify_callbacks(result)
                    return result

                last_error = result.error

                if not result.retryable:
                    break

            except asyncio.TimeoutError:
                last_error = "Request timed out"
            except Exception as e:
                last_error = str(e)
                logger.error(f"Error sending notification: {e}")

            # Wait before retry
            if attempt < self.config.max_retries:
                delay = self.config.retry_delay * (self.config.retry_backoff ** attempt)
                await asyncio.sleep(delay)
                message.retry_count += 1

        return DeliveryResult(
            success=False,
            message_id=message.message_id or "",
            channel=self.channel,
            error=last_error or "Unknown error",
            retryable=True,
        )

    async def send_batch(self, messages: list[NotificationMessage]) -> list[DeliveryResult]:
        """
        Send multiple notifications.

        Args:
            messages: List of messages to send

        Returns:
            List of delivery results
        """
        results = []

        for message in messages:
            result = await self.send(message)
            results.append(result)

            # Small delay between messages
            if len(messages) > 1:
                await asyncio.sleep(0.1)

        return results

    def queue_message(self, message: NotificationMessage) -> None:
        """Queue a message for batched sending."""
        self._pending_messages.append(message)

    async def flush(self) -> list[DeliveryResult]:
        """Flush all pending messages."""
        messages = self._pending_messages[:]
        self._pending_messages.clear()

        if not messages:
            return []

        return await self.send_batch(messages)

    async def start(self) -> None:
        """Start the notifier (for batch processing)."""
        self._running = True
        self._batch_task = asyncio.create_task(self._batch_loop())
        logger.info(f"{self.channel.value} notifier started")

    async def stop(self) -> None:
        """Stop the notifier."""
        self._running = False

        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass

        # Flush remaining messages
        await self.flush()
        logger.info(f"{self.channel.value} notifier stopped")

    async def _batch_loop(self) -> None:
        """Background loop for processing batched messages."""
        while self._running:
            try:
                await asyncio.sleep(self.config.batch_delay)

                if self._pending_messages:
                    # Process up to batch_size messages
                    batch = self._pending_messages[:self.config.batch_size]
                    self._pending_messages = self._pending_messages[self.config.batch_size:]

                    await self.send_batch(batch)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch loop: {e}")

    def register_callback(
        self,
        callback: Callable[[DeliveryResult], None],
    ) -> None:
        """Register callback for delivery results."""
        self._callbacks.append(callback)

    def unregister_callback(
        self,
        callback: Callable[[DeliveryResult], None],
    ) -> None:
        """Unregister callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _notify_callbacks(self, result: DeliveryResult) -> None:
        """Notify registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Error in notification callback: {e}")


class ConsoleNotifier(BaseNotifier):
    """Simple console notifier for testing/debugging."""

    @property
    def channel(self) -> NotificationChannel:
        """Get channel type."""
        return NotificationChannel.WEBHOOK  # Using WEBHOOK as placeholder

    async def _send(self, message: NotificationMessage) -> DeliveryResult:
        """Send to console."""
        priority_icons = {
            NotificationPriority.LOW: "ℹ️",
            NotificationPriority.NORMAL: "📬",
            NotificationPriority.HIGH: "⚠️",
            NotificationPriority.URGENT: "🚨",
            NotificationPriority.CRITICAL: "🔴",
        }

        icon = priority_icons.get(message.priority, "📬")
        print(f"\n{icon} [{message.message_type.value.upper()}] {message.title}")
        print(f"   {message.body}")

        if message.recipients:
            print(f"   To: {', '.join(message.recipients)}")

        return DeliveryResult(
            success=True,
            message_id=message.message_id or "",
            channel=self.channel,
        )

    async def validate_config(self) -> bool:
        """Validate config (always valid for console)."""
        return True


def create_message_formatter() -> MessageFormatter:
    """
    Create a message formatter instance.

    Returns:
        MessageFormatter instance
    """
    return MessageFormatter()


def create_console_notifier(
    config: NotifierConfig | None = None,
) -> ConsoleNotifier:
    """
    Create a console notifier instance.

    Args:
        config: Notifier configuration

    Returns:
        ConsoleNotifier instance
    """
    return ConsoleNotifier(config)


# Module version
__version__ = "2.2.0"
