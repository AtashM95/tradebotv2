"""
Notifications Package for Ultimate Trading Bot v2.2.

This package provides comprehensive notification capabilities including:
- Multi-channel notifications (email, SMS, Slack, Telegram, Discord, push, webhook)
- Message formatting and templating
- Delivery tracking and retry logic
- Rate limiting and deduplication
- Unified notification manager
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable


from .base_notifier import (
    NotificationPriority,
    NotificationType,
    NotificationChannel,
    NotificationAttachment,
    NotificationMessage,
    DeliveryResult,
    NotifierConfig,
    RateLimiter,
    DeduplicationCache,
    MessageFormatter,
    BaseNotifier,
    ConsoleNotifier,
    create_message_formatter,
    create_console_notifier,
)
from .email_notifier import (
    EmailConfig,
    EmailTemplate,
    EmailNotifier,
    create_email_notifier,
    create_email_config,
)
from .sms_notifier import (
    SMSConfig,
    SMSFormatter,
    TwilioClient,
    SMSNotifier,
    create_sms_notifier,
    create_sms_config,
)
from .slack_notifier import (
    SlackConfig,
    SlackBlockBuilder,
    SlackNotifier,
    create_slack_notifier,
    create_slack_config,
)
from .telegram_notifier import (
    TelegramConfig,
    TelegramFormatter,
    TelegramBotAPI,
    TelegramNotifier,
    create_telegram_notifier,
    create_telegram_config,
)
from .discord_notifier import (
    DiscordConfig,
    DiscordEmbedBuilder,
    DiscordNotifier,
    create_discord_notifier,
    create_discord_config,
)
from .push_notifier import (
    PushProvider,
    PushConfig,
    PushSubscription,
    FCMClient,
    WebPushClient,
    PushNotifier,
    create_push_notifier,
    create_fcm_config,
    create_web_push_config,
)
from .webhook_notifier import (
    WebhookFormat,
    AuthMethod,
    WebhookConfig,
    WebhookPayloadBuilder,
    WebhookNotifier,
    create_webhook_notifier,
    create_webhook_config,
)


logger = logging.getLogger(__name__)


__all__ = [
    # Base Notifier
    "NotificationPriority",
    "NotificationType",
    "NotificationChannel",
    "NotificationAttachment",
    "NotificationMessage",
    "DeliveryResult",
    "NotifierConfig",
    "RateLimiter",
    "DeduplicationCache",
    "MessageFormatter",
    "BaseNotifier",
    "ConsoleNotifier",
    "create_message_formatter",
    "create_console_notifier",
    # Email Notifier
    "EmailConfig",
    "EmailTemplate",
    "EmailNotifier",
    "create_email_notifier",
    "create_email_config",
    # SMS Notifier
    "SMSConfig",
    "SMSFormatter",
    "TwilioClient",
    "SMSNotifier",
    "create_sms_notifier",
    "create_sms_config",
    # Slack Notifier
    "SlackConfig",
    "SlackBlockBuilder",
    "SlackNotifier",
    "create_slack_notifier",
    "create_slack_config",
    # Telegram Notifier
    "TelegramConfig",
    "TelegramFormatter",
    "TelegramBotAPI",
    "TelegramNotifier",
    "create_telegram_notifier",
    "create_telegram_config",
    # Discord Notifier
    "DiscordConfig",
    "DiscordEmbedBuilder",
    "DiscordNotifier",
    "create_discord_notifier",
    "create_discord_config",
    # Push Notifier
    "PushProvider",
    "PushConfig",
    "PushSubscription",
    "FCMClient",
    "WebPushClient",
    "PushNotifier",
    "create_push_notifier",
    "create_fcm_config",
    "create_web_push_config",
    # Webhook Notifier
    "WebhookFormat",
    "AuthMethod",
    "WebhookConfig",
    "WebhookPayloadBuilder",
    "WebhookNotifier",
    "create_webhook_notifier",
    "create_webhook_config",
    # Manager
    "NotificationManager",
    "NotificationManagerConfig",
    "create_notification_manager",
]


@dataclass
class NotificationManagerConfig:
    """Configuration for notification manager."""

    # Enable/disable channels
    email_enabled: bool = False
    sms_enabled: bool = False
    slack_enabled: bool = False
    telegram_enabled: bool = False
    discord_enabled: bool = False
    push_enabled: bool = False
    webhook_enabled: bool = False
    console_enabled: bool = True

    # Channel configurations
    email_config: EmailConfig | None = None
    sms_config: SMSConfig | None = None
    slack_config: SlackConfig | None = None
    telegram_config: TelegramConfig | None = None
    discord_config: DiscordConfig | None = None
    push_config: PushConfig | None = None
    webhook_config: WebhookConfig | None = None

    # Routing rules
    default_channels: list[NotificationChannel] = field(
        default_factory=lambda: [NotificationChannel.SLACK]
    )
    priority_routing: dict[str, list[NotificationChannel]] = field(default_factory=dict)
    type_routing: dict[str, list[NotificationChannel]] = field(default_factory=dict)

    # Manager settings
    batch_notifications: bool = True
    batch_delay: float = 1.0
    parallel_send: bool = True
    max_parallel: int = 5
    store_history: bool = True
    history_max_size: int = 1000


class NotificationManager:
    """
    Unified notification manager.

    Coordinates multiple notification channels and provides
    intelligent routing and delivery.
    """

    def __init__(self, config: NotificationManagerConfig | None = None) -> None:
        """
        Initialize notification manager.

        Args:
            config: Manager configuration
        """
        self.config = config or NotificationManagerConfig()

        # Initialize notifiers
        self._notifiers: dict[NotificationChannel, BaseNotifier] = {}
        self._formatter = MessageFormatter()

        # State
        self._running = False
        self._pending_queue: asyncio.Queue[NotificationMessage] = asyncio.Queue()
        self._history: list[tuple[NotificationMessage, list[DeliveryResult]]] = []
        self._callbacks: list[Callable[[DeliveryResult], None]] = []
        self._batch_task: asyncio.Task | None = None

        # Statistics
        self._stats = {
            "total_sent": 0,
            "total_success": 0,
            "total_failed": 0,
            "by_channel": {},
        }

        logger.info("NotificationManager initialized")

    async def initialize(self) -> None:
        """Initialize all configured notifiers."""
        # Email
        if self.config.email_enabled and self.config.email_config:
            notifier = create_email_notifier(self.config.email_config)
            if await notifier.validate_config():
                self._notifiers[NotificationChannel.EMAIL] = notifier
                logger.info("Email notifier enabled")

        # SMS
        if self.config.sms_enabled and self.config.sms_config:
            notifier = create_sms_notifier(self.config.sms_config)
            if await notifier.validate_config():
                self._notifiers[NotificationChannel.SMS] = notifier
                logger.info("SMS notifier enabled")

        # Slack
        if self.config.slack_enabled and self.config.slack_config:
            notifier = create_slack_notifier(self.config.slack_config)
            if await notifier.validate_config():
                self._notifiers[NotificationChannel.SLACK] = notifier
                logger.info("Slack notifier enabled")

        # Telegram
        if self.config.telegram_enabled and self.config.telegram_config:
            notifier = create_telegram_notifier(self.config.telegram_config)
            if await notifier.validate_config():
                self._notifiers[NotificationChannel.TELEGRAM] = notifier
                logger.info("Telegram notifier enabled")

        # Discord
        if self.config.discord_enabled and self.config.discord_config:
            notifier = create_discord_notifier(self.config.discord_config)
            if await notifier.validate_config():
                self._notifiers[NotificationChannel.DISCORD] = notifier
                logger.info("Discord notifier enabled")

        # Push
        if self.config.push_enabled and self.config.push_config:
            notifier = create_push_notifier(self.config.push_config)
            if await notifier.validate_config():
                self._notifiers[NotificationChannel.PUSH] = notifier
                logger.info("Push notifier enabled")

        # Webhook
        if self.config.webhook_enabled and self.config.webhook_config:
            notifier = create_webhook_notifier(self.config.webhook_config)
            if await notifier.validate_config():
                self._notifiers[NotificationChannel.WEBHOOK] = notifier
                logger.info("Webhook notifier enabled")

        # Console (always available for testing)
        if self.config.console_enabled:
            self._notifiers[NotificationChannel.WEBHOOK] = create_console_notifier()

        logger.info(f"Initialized {len(self._notifiers)} notification channels")

    async def start(self) -> None:
        """Start the notification manager."""
        if not self._notifiers:
            await self.initialize()

        self._running = True

        # Start batch processing
        if self.config.batch_notifications:
            self._batch_task = asyncio.create_task(self._batch_loop())

        # Start all notifiers
        for notifier in self._notifiers.values():
            await notifier.start()

        logger.info("NotificationManager started")

    async def stop(self) -> None:
        """Stop the notification manager."""
        self._running = False

        # Stop batch processing
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass

        # Process remaining queue
        while not self._pending_queue.empty():
            try:
                message = self._pending_queue.get_nowait()
                await self._send_immediately(message)
            except asyncio.QueueEmpty:
                break

        # Stop all notifiers
        for notifier in self._notifiers.values():
            await notifier.stop()

        logger.info("NotificationManager stopped")

    async def send(
        self,
        message: NotificationMessage,
        channels: list[NotificationChannel] | None = None,
        immediate: bool = False,
    ) -> list[DeliveryResult]:
        """
        Send a notification.

        Args:
            message: Notification message
            channels: Specific channels to use (overrides routing)
            immediate: Send immediately without batching

        Returns:
            List of delivery results
        """
        # Determine target channels
        target_channels = channels or self._get_target_channels(message)

        # Filter to available channels
        target_channels = [
            ch for ch in target_channels if ch in self._notifiers
        ]

        if not target_channels:
            logger.warning("No available channels for notification")
            return []

        # Set channels on message
        message.channel = target_channels[0] if len(target_channels) == 1 else None

        # Send immediately or queue
        if immediate or not self.config.batch_notifications:
            results = await self._send_to_channels(message, target_channels)
        else:
            # Queue for batching
            await self._pending_queue.put(message)
            results = []

        return results

    async def send_trade_notification(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        pnl: float | None = None,
        **kwargs: Any,
    ) -> list[DeliveryResult]:
        """Send a trade notification."""
        message = self._formatter.format_trade_notification(
            symbol, action, quantity, price, pnl, **kwargs
        )
        return await self.send(message, immediate=True)

    async def send_signal_notification(
        self,
        symbol: str,
        signal_type: str,
        strength: float,
        source: str,
        **kwargs: Any,
    ) -> list[DeliveryResult]:
        """Send a signal notification."""
        message = self._formatter.format_signal_notification(
            symbol, signal_type, strength, source, **kwargs
        )
        return await self.send(message)

    async def send_alert(
        self,
        title: str,
        message_text: str,
        severity: str = "info",
        **kwargs: Any,
    ) -> list[DeliveryResult]:
        """Send an alert notification."""
        message = self._formatter.format_alert_notification(
            title, message_text, severity, **kwargs
        )

        # Urgent/critical alerts are sent immediately
        immediate = severity in ["error", "critical"]
        return await self.send(message, immediate=immediate)

    async def send_performance_update(
        self,
        metric_name: str,
        value: float,
        change: float | None = None,
        period: str = "daily",
        **kwargs: Any,
    ) -> list[DeliveryResult]:
        """Send a performance notification."""
        message = self._formatter.format_performance_notification(
            metric_name, value, change, period, **kwargs
        )
        return await self.send(message)

    async def send_system_notification(
        self,
        title: str,
        message_text: str,
        component: str | None = None,
        **kwargs: Any,
    ) -> list[DeliveryResult]:
        """Send a system notification."""
        message = self._formatter.format_system_notification(
            title, message_text, component, **kwargs
        )
        return await self.send(message)

    def _get_target_channels(
        self,
        message: NotificationMessage,
    ) -> list[NotificationChannel]:
        """Determine target channels based on routing rules."""
        # Check priority routing
        priority_channels = self.config.priority_routing.get(message.priority.value)
        if priority_channels:
            return priority_channels

        # Check type routing
        type_channels = self.config.type_routing.get(message.message_type.value)
        if type_channels:
            return type_channels

        # Use default channels
        return list(self.config.default_channels)

    async def _send_to_channels(
        self,
        message: NotificationMessage,
        channels: list[NotificationChannel],
    ) -> list[DeliveryResult]:
        """Send message to multiple channels."""
        if self.config.parallel_send:
            # Send to all channels in parallel
            tasks = []
            for channel in channels:
                if channel in self._notifiers:
                    tasks.append(self._send_to_channel(message, channel))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions
            delivery_results = []
            for result in results:
                if isinstance(result, DeliveryResult):
                    delivery_results.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Error sending notification: {result}")

            return delivery_results
        else:
            # Send sequentially
            results = []
            for channel in channels:
                result = await self._send_to_channel(message, channel)
                results.append(result)
            return results

    async def _send_to_channel(
        self,
        message: NotificationMessage,
        channel: NotificationChannel,
    ) -> DeliveryResult:
        """Send message to specific channel."""
        notifier = self._notifiers.get(channel)
        if not notifier:
            return DeliveryResult(
                success=False,
                message_id=message.message_id or "",
                channel=channel,
                error="Channel not available",
            )

        result = await notifier.send(message)

        # Update statistics
        self._update_stats(result)

        # Notify callbacks
        self._notify_callbacks(result)

        # Store in history
        if self.config.store_history:
            self._store_history(message, result)

        return result

    async def _send_immediately(
        self,
        message: NotificationMessage,
    ) -> list[DeliveryResult]:
        """Send message immediately to all target channels."""
        channels = self._get_target_channels(message)
        return await self._send_to_channels(message, channels)

    async def _batch_loop(self) -> None:
        """Background loop for processing batched notifications."""
        while self._running:
            try:
                # Wait for batch delay
                await asyncio.sleep(self.config.batch_delay)

                # Process pending messages
                messages: list[NotificationMessage] = []
                while not self._pending_queue.empty() and len(messages) < 100:
                    try:
                        message = self._pending_queue.get_nowait()
                        messages.append(message)
                    except asyncio.QueueEmpty:
                        break

                if messages:
                    # Group by channel
                    for message in messages:
                        await self._send_immediately(message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch loop: {e}")

    def _update_stats(self, result: DeliveryResult) -> None:
        """Update statistics from delivery result."""
        self._stats["total_sent"] += 1

        if result.success:
            self._stats["total_success"] += 1
        else:
            self._stats["total_failed"] += 1

        channel_name = result.channel.value
        if channel_name not in self._stats["by_channel"]:
            self._stats["by_channel"][channel_name] = {"sent": 0, "success": 0, "failed": 0}

        self._stats["by_channel"][channel_name]["sent"] += 1
        if result.success:
            self._stats["by_channel"][channel_name]["success"] += 1
        else:
            self._stats["by_channel"][channel_name]["failed"] += 1

    def _store_history(
        self,
        message: NotificationMessage,
        result: DeliveryResult,
    ) -> None:
        """Store notification in history."""
        # Find existing entry or create new
        for hist_message, hist_results in self._history:
            if hist_message.message_id == message.message_id:
                hist_results.append(result)
                return

        self._history.append((message, [result]))

        # Trim history if needed
        if len(self._history) > self.config.history_max_size:
            self._history = self._history[-self.config.history_max_size:]

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

    def get_stats(self) -> dict[str, Any]:
        """Get notification statistics."""
        return self._stats.copy()

    def get_history(
        self,
        limit: int = 100,
    ) -> list[tuple[NotificationMessage, list[DeliveryResult]]]:
        """Get notification history."""
        return self._history[-limit:]

    def get_available_channels(self) -> list[NotificationChannel]:
        """Get list of available channels."""
        return list(self._notifiers.keys())

    def is_channel_available(self, channel: NotificationChannel) -> bool:
        """Check if channel is available."""
        return channel in self._notifiers

    async def test_channel(
        self,
        channel: NotificationChannel,
    ) -> DeliveryResult:
        """Send a test notification to specific channel."""
        message = NotificationMessage(
            title="Test Notification",
            body="This is a test notification from Ultimate Trading Bot v2.2",
            message_type=NotificationType.INFO,
            priority=NotificationPriority.LOW,
        )

        return await self._send_to_channel(message, channel)


def create_notification_manager(
    config: NotificationManagerConfig | None = None,
) -> NotificationManager:
    """
    Create a notification manager instance.

    Args:
        config: Manager configuration

    Returns:
        NotificationManager instance
    """
    return NotificationManager(config)


# Module version
__version__ = "2.2.0"
