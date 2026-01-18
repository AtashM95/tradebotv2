"""
Discord Notifier Module for Ultimate Trading Bot v2.2.

This module provides Discord notification capabilities including:
- Webhook-based messaging
- Rich embeds
- Attachment support
- Mentions and roles
"""

import asyncio
import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .base_notifier import (
    BaseNotifier,
    NotificationChannel,
    NotificationMessage,
    NotificationType,
    NotificationPriority,
    NotifierConfig,
    DeliveryResult,
)


logger = logging.getLogger(__name__)


@dataclass
class DiscordConfig(NotifierConfig):
    """Discord notifier configuration."""

    # Webhook settings
    webhook_url: str = ""

    # Bot identity
    username: str = "Trading Bot"
    avatar_url: str | None = None

    # Message settings
    use_embeds: bool = True
    include_timestamp: bool = True

    # Mention settings
    mention_everyone: bool = False
    mention_roles: list[str] = field(default_factory=list)
    mention_users: list[str] = field(default_factory=list)

    # Thread settings
    thread_id: str | None = None


class DiscordEmbedBuilder:
    """Builds Discord embed messages."""

    # Color mapping for message types (Discord uses decimal colors)
    TYPE_COLORS = {
        NotificationType.INFO: 2201331,      # Blue
        NotificationType.SUCCESS: 5025616,   # Green
        NotificationType.WARNING: 16744448,  # Orange
        NotificationType.ERROR: 16007990,    # Red
        NotificationType.ALERT: 15277667,    # Pink
        NotificationType.TRADE: 6750207,     # Purple
        NotificationType.ORDER: 4149685,     # Indigo
        NotificationType.SIGNAL: 65480,      # Teal
        NotificationType.SYSTEM: 6323595,    # Gray
        NotificationType.PERFORMANCE: 7951688,  # Brown
    }

    # Emoji mapping
    TYPE_EMOJI = {
        NotificationType.INFO: "ℹ️",
        NotificationType.SUCCESS: "✅",
        NotificationType.WARNING: "⚠️",
        NotificationType.ERROR: "❌",
        NotificationType.ALERT: "🚨",
        NotificationType.TRADE: "💰",
        NotificationType.ORDER: "📋",
        NotificationType.SIGNAL: "🔔",
        NotificationType.SYSTEM: "⚙️",
        NotificationType.PERFORMANCE: "📈",
    }

    @classmethod
    def build_embed(
        cls,
        message: NotificationMessage,
        include_timestamp: bool = True,
    ) -> dict[str, Any]:
        """
        Build Discord embed from notification.

        Args:
            message: Notification message
            include_timestamp: Include timestamp in footer

        Returns:
            Discord embed object
        """
        color = cls.TYPE_COLORS.get(message.message_type, 2201331)
        emoji = cls.TYPE_EMOJI.get(message.message_type, "📬")

        embed: dict[str, Any] = {
            "title": f"{emoji} {message.title}",
            "description": message.body[:4096],  # Discord limit
            "color": color,
        }

        # Add fields from metadata
        if message.metadata:
            fields = []
            for key, value in list(message.metadata.items())[:25]:  # Discord limit
                formatted_key = key.replace("_", " ").title()
                fields.append({
                    "name": formatted_key,
                    "value": str(value)[:1024],  # Discord field value limit
                    "inline": len(str(value)) < 30,
                })
            embed["fields"] = fields

        # Add footer with priority and timestamp
        footer_parts = []
        if message.priority in [NotificationPriority.HIGH, NotificationPriority.URGENT, NotificationPriority.CRITICAL]:
            footer_parts.append(f"Priority: {message.priority.value.upper()}")

        footer_parts.append("Ultimate Trading Bot v2.2")

        embed["footer"] = {"text": " | ".join(footer_parts)}

        if include_timestamp:
            embed["timestamp"] = message.timestamp.isoformat()

        return embed

    @classmethod
    def build_trade_embed(
        cls,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        pnl: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build trade notification embed."""
        # Color based on action and P&L
        if pnl is not None:
            color = 5025616 if pnl >= 0 else 16007990  # Green or Red
        else:
            color = 6750207 if action.lower() == "buy" else 15158332

        emoji = "📈" if action.lower() == "buy" else "📉"

        embed: dict[str, Any] = {
            "title": f"{emoji} Trade Executed: {action.upper()} {symbol}",
            "color": color,
            "fields": [
                {"name": "Action", "value": action.upper(), "inline": True},
                {"name": "Symbol", "value": symbol, "inline": True},
                {"name": "Quantity", "value": f"{quantity:,.4f}", "inline": True},
                {"name": "Price", "value": f"${price:,.2f}", "inline": True},
                {"name": "Value", "value": f"${quantity * price:,.2f}", "inline": True},
            ],
            "timestamp": datetime.now().isoformat(),
        }

        if pnl is not None:
            sign = "+" if pnl >= 0 else ""
            pnl_emoji = "💰" if pnl >= 0 else "💸"
            embed["fields"].append({
                "name": f"{pnl_emoji} P&L",
                "value": f"{sign}${pnl:,.2f}",
                "inline": True,
            })

        # Add extra fields
        for key, value in kwargs.items():
            if len(embed["fields"]) < 25:
                formatted_key = key.replace("_", " ").title()
                embed["fields"].append({
                    "name": formatted_key,
                    "value": str(value),
                    "inline": True,
                })

        embed["footer"] = {"text": "Ultimate Trading Bot v2.2"}

        return embed

    @classmethod
    def build_alert_embed(
        cls,
        title: str,
        message: str,
        severity: str = "info",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build alert notification embed."""
        color_map = {
            "info": 2201331,
            "warning": 16744448,
            "error": 16007990,
            "critical": 15277667,
        }

        emoji_map = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "critical": "🆘",
        }

        color = color_map.get(severity, 2201331)
        emoji = emoji_map.get(severity, "📬")

        embed: dict[str, Any] = {
            "title": f"{emoji} {title}",
            "description": message,
            "color": color,
            "timestamp": datetime.now().isoformat(),
            "footer": {"text": f"Severity: {severity.upper()} | Ultimate Trading Bot v2.2"},
        }

        if kwargs:
            fields = []
            for key, value in kwargs.items():
                formatted_key = key.replace("_", " ").title()
                fields.append({
                    "name": formatted_key,
                    "value": str(value),
                    "inline": True,
                })
            embed["fields"] = fields

        return embed


class DiscordNotifier(BaseNotifier):
    """
    Discord notification channel.

    Sends notifications via Discord webhooks.
    """

    def __init__(self, config: DiscordConfig | None = None) -> None:
        """Initialize Discord notifier."""
        self._discord_config = config or DiscordConfig()
        super().__init__(self._discord_config)
        self._embed_builder = DiscordEmbedBuilder()

        logger.info("DiscordNotifier initialized")

    @property
    def channel(self) -> NotificationChannel:
        """Get notification channel type."""
        return NotificationChannel.DISCORD

    async def validate_config(self) -> bool:
        """Validate Discord configuration."""
        if not self._discord_config.webhook_url:
            logger.error("Discord webhook URL not configured")
            return False

        if "discord.com/api/webhooks" not in self._discord_config.webhook_url:
            logger.error("Invalid Discord webhook URL")
            return False

        return True

    async def _send(self, message: NotificationMessage) -> DeliveryResult:
        """Send Discord notification."""
        try:
            # Build payload
            payload = self._build_payload(message)

            # Send webhook
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._send_webhook,
                payload,
            )

            if response.get("success"):
                logger.info("Discord message sent successfully")
                return DeliveryResult(
                    success=True,
                    message_id=message.message_id or "",
                    channel=self.channel,
                    response_data=response,
                )

            error = response.get("error", "Unknown error")
            logger.error(f"Discord webhook error: {error}")

            return DeliveryResult(
                success=False,
                message_id=message.message_id or "",
                channel=self.channel,
                error=error,
                retryable="rate limit" in error.lower(),
            )

        except Exception as e:
            logger.error(f"Error sending Discord message: {e}")
            return DeliveryResult(
                success=False,
                message_id=message.message_id or "",
                channel=self.channel,
                error=str(e),
                retryable=True,
            )

    def _build_payload(self, message: NotificationMessage) -> dict[str, Any]:
        """Build webhook payload."""
        payload: dict[str, Any] = {}

        # Bot identity
        if self._discord_config.username:
            payload["username"] = self._discord_config.username

        if self._discord_config.avatar_url:
            payload["avatar_url"] = self._discord_config.avatar_url

        # Content and embeds
        if self._discord_config.use_embeds:
            embed = DiscordEmbedBuilder.build_embed(
                message,
                include_timestamp=self._discord_config.include_timestamp,
            )
            payload["embeds"] = [embed]
        else:
            # Simple text message
            text_parts = [f"**{message.title}**", "", message.body]
            payload["content"] = "\n".join(text_parts)

        # Mentions
        mentions = []
        if self._discord_config.mention_everyone:
            mentions.append("@everyone")

        for role_id in self._discord_config.mention_roles:
            mentions.append(f"<@&{role_id}>")

        for user_id in self._discord_config.mention_users:
            mentions.append(f"<@{user_id}>")

        # Add mentions for high priority
        if message.priority in [NotificationPriority.URGENT, NotificationPriority.CRITICAL]:
            if mentions:
                payload["content"] = " ".join(mentions) + "\n" + payload.get("content", "")
            elif not payload.get("content"):
                payload["content"] = " ".join(mentions)

        return payload

    def _send_webhook(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send webhook request."""
        # Construct URL with thread ID if specified
        url = self._discord_config.webhook_url
        if self._discord_config.thread_id:
            url += f"?thread_id={self._discord_config.thread_id}"

        data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(url, data=data, method="POST")
        request.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                # Discord returns 204 No Content on success
                if response.status == 204:
                    return {"success": True}

                response_text = response.read().decode()
                try:
                    return {"success": True, "data": json.loads(response_text)}
                except json.JSONDecodeError:
                    return {"success": True, "response": response_text}

        except urllib.error.HTTPError as e:
            error_body = e.read().decode()
            try:
                error_data = json.loads(error_body)
                return {"success": False, "error": error_data.get("message", error_body)}
            except json.JSONDecodeError:
                return {"success": False, "error": error_body}

    async def send_trade_notification(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        pnl: float | None = None,
        **kwargs: Any,
    ) -> DeliveryResult:
        """Send a trade notification."""
        embed = DiscordEmbedBuilder.build_trade_embed(
            symbol, action, quantity, price, pnl, **kwargs
        )

        payload = {
            "embeds": [embed],
        }

        if self._discord_config.username:
            payload["username"] = self._discord_config.username

        if self._discord_config.avatar_url:
            payload["avatar_url"] = self._discord_config.avatar_url

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            self._send_webhook,
            payload,
        )

        return DeliveryResult(
            success=response.get("success", False),
            message_id=f"trade_{symbol}_{datetime.now().timestamp()}",
            channel=self.channel,
            error=response.get("error") if not response.get("success") else None,
        )

    async def send_alert(
        self,
        title: str,
        message: str,
        severity: str = "info",
        **kwargs: Any,
    ) -> DeliveryResult:
        """Send an alert notification."""
        embed = DiscordEmbedBuilder.build_alert_embed(
            title, message, severity, **kwargs
        )

        payload = {
            "embeds": [embed],
        }

        if self._discord_config.username:
            payload["username"] = self._discord_config.username

        # Add mentions for high severity
        if severity in ["error", "critical"]:
            mentions = []
            for role_id in self._discord_config.mention_roles:
                mentions.append(f"<@&{role_id}>")
            if mentions:
                payload["content"] = " ".join(mentions)

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            self._send_webhook,
            payload,
        )

        return DeliveryResult(
            success=response.get("success", False),
            message_id=f"alert_{datetime.now().timestamp()}",
            channel=self.channel,
            error=response.get("error") if not response.get("success") else None,
        )


def create_discord_notifier(
    config: DiscordConfig | None = None,
) -> DiscordNotifier:
    """
    Create a Discord notifier instance.

    Args:
        config: Discord configuration

    Returns:
        DiscordNotifier instance
    """
    return DiscordNotifier(config)


def create_discord_config(
    webhook_url: str,
    username: str = "Trading Bot",
    **kwargs: Any,
) -> DiscordConfig:
    """
    Create Discord configuration.

    Args:
        webhook_url: Discord webhook URL
        username: Bot username
        **kwargs: Additional configuration

    Returns:
        DiscordConfig instance
    """
    return DiscordConfig(
        webhook_url=webhook_url,
        username=username,
        **kwargs,
    )


# Module version
__version__ = "2.2.0"
