"""
Slack Notifier Module for Ultimate Trading Bot v2.2.

This module provides Slack notification capabilities including:
- Webhook-based messaging
- Rich message formatting with blocks
- Attachment support
- Thread replies
- Interactive messages
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
class SlackConfig(NotifierConfig):
    """Slack notifier configuration."""

    # Webhook settings
    webhook_url: str = ""
    bot_token: str | None = None  # For API-based messaging

    # Default channel
    default_channel: str = "#trading-alerts"

    # Message settings
    username: str = "Trading Bot"
    icon_emoji: str = ":chart_with_upwards_trend:"
    icon_url: str | None = None

    # Formatting
    use_blocks: bool = True
    include_footer: bool = True
    footer_text: str = "Ultimate Trading Bot v2.2"

    # Thread settings
    thread_ts: str | None = None
    reply_broadcast: bool = False


class SlackBlockBuilder:
    """Builds Slack Block Kit messages."""

    # Color mapping for message types
    TYPE_COLORS = {
        NotificationType.INFO: "#2196f3",
        NotificationType.SUCCESS: "#4caf50",
        NotificationType.WARNING: "#ff9800",
        NotificationType.ERROR: "#f44336",
        NotificationType.ALERT: "#e91e63",
        NotificationType.TRADE: "#673ab7",
        NotificationType.ORDER: "#3f51b5",
        NotificationType.SIGNAL: "#009688",
        NotificationType.SYSTEM: "#607d8b",
        NotificationType.PERFORMANCE: "#795548",
    }

    # Emoji mapping for message types
    TYPE_EMOJI = {
        NotificationType.INFO: ":information_source:",
        NotificationType.SUCCESS: ":white_check_mark:",
        NotificationType.WARNING: ":warning:",
        NotificationType.ERROR: ":x:",
        NotificationType.ALERT: ":rotating_light:",
        NotificationType.TRADE: ":moneybag:",
        NotificationType.ORDER: ":receipt:",
        NotificationType.SIGNAL: ":bell:",
        NotificationType.SYSTEM: ":gear:",
        NotificationType.PERFORMANCE: ":chart_with_upwards_trend:",
    }

    @classmethod
    def build_message(
        cls,
        message: NotificationMessage,
        include_footer: bool = True,
        footer_text: str = "",
    ) -> dict[str, Any]:
        """
        Build Slack message with blocks.

        Args:
            message: Notification message
            include_footer: Include footer section
            footer_text: Footer text

        Returns:
            Slack message payload
        """
        blocks = []
        attachments = []

        # Header block
        emoji = cls.TYPE_EMOJI.get(message.message_type, ":bell:")
        header_text = f"{emoji} *{message.title}*"

        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": message.title[:150],  # Slack header limit
                "emoji": True,
            },
        })

        # Priority indicator for high priority messages
        if message.priority in [
            NotificationPriority.HIGH,
            NotificationPriority.URGENT,
            NotificationPriority.CRITICAL,
        ]:
            priority_emoji = {
                NotificationPriority.HIGH: ":large_orange_diamond:",
                NotificationPriority.URGENT: ":red_circle:",
                NotificationPriority.CRITICAL: ":rotating_light:",
            }
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"{priority_emoji.get(message.priority, '')} *Priority: {message.priority.value.upper()}*",
                    },
                ],
            })

        # Divider
        blocks.append({"type": "divider"})

        # Body section
        body_text = message.body[:3000]  # Slack text limit
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": body_text,
            },
        })

        # Metadata fields
        if message.metadata:
            fields = []
            for key, value in list(message.metadata.items())[:10]:  # Max 10 fields
                formatted_key = key.replace("_", " ").title()
                fields.append({
                    "type": "mrkdwn",
                    "text": f"*{formatted_key}:*\n{value}",
                })

            if fields:
                # Slack allows max 10 fields per section
                for i in range(0, len(fields), 10):
                    blocks.append({
                        "type": "section",
                        "fields": fields[i:i + 10],
                    })

        # Footer
        if include_footer:
            blocks.append({"type": "divider"})
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"{footer_text} | {message.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
                    },
                ],
            })

        # Create attachment for color coding
        color = cls.TYPE_COLORS.get(message.message_type, "#2196f3")
        attachments.append({
            "color": color,
            "blocks": blocks,
        })

        return {
            "attachments": attachments,
            "text": message.title,  # Fallback text
        }

    @classmethod
    def build_simple_message(cls, message: NotificationMessage) -> dict[str, Any]:
        """Build simple text-only message."""
        emoji = cls.TYPE_EMOJI.get(message.message_type, ":bell:")

        text = f"{emoji} *{message.title}*\n{message.body}"

        if message.metadata:
            text += "\n"
            for key, value in message.metadata.items():
                formatted_key = key.replace("_", " ").title()
                text += f"\n• *{formatted_key}:* {value}"

        return {
            "text": text,
            "mrkdwn": True,
        }

    @classmethod
    def build_trade_message(
        cls,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        pnl: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build trade notification message."""
        color = "#4caf50" if action.lower() == "buy" else "#f44336"
        emoji = ":chart_with_upwards_trend:" if action.lower() == "buy" else ":chart_with_downwards_trend:"

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{action.upper()} {symbol}",
                    "emoji": True,
                },
            },
            {"type": "divider"},
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Action:*\n{emoji} {action.upper()}"},
                    {"type": "mrkdwn", "text": f"*Symbol:*\n{symbol}"},
                    {"type": "mrkdwn", "text": f"*Quantity:*\n{quantity:,.4f}"},
                    {"type": "mrkdwn", "text": f"*Price:*\n${price:,.2f}"},
                ],
            },
        ]

        if pnl is not None:
            pnl_emoji = ":moneybag:" if pnl >= 0 else ":money_with_wings:"
            pnl_color = "good" if pnl >= 0 else "danger"
            sign = "+" if pnl >= 0 else ""
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{pnl_emoji} *P&L:* {sign}${pnl:,.2f}",
                },
            })

        if kwargs:
            extra_fields = []
            for key, value in kwargs.items():
                formatted_key = key.replace("_", " ").title()
                extra_fields.append({
                    "type": "mrkdwn",
                    "text": f"*{formatted_key}:*\n{value}",
                })
            if extra_fields:
                blocks.append({
                    "type": "section",
                    "fields": extra_fields[:10],
                })

        return {
            "attachments": [{
                "color": color,
                "blocks": blocks,
            }],
            "text": f"{action.upper()} {quantity} {symbol} @ ${price:,.2f}",
        }


class SlackNotifier(BaseNotifier):
    """
    Slack notification channel.

    Sends notifications via Slack webhooks or API.
    """

    def __init__(self, config: SlackConfig | None = None) -> None:
        """Initialize Slack notifier."""
        self._slack_config = config or SlackConfig()
        super().__init__(self._slack_config)
        self._block_builder = SlackBlockBuilder()

        logger.info("SlackNotifier initialized")

    @property
    def channel(self) -> NotificationChannel:
        """Get notification channel type."""
        return NotificationChannel.SLACK

    async def validate_config(self) -> bool:
        """Validate Slack configuration."""
        if not self._slack_config.webhook_url and not self._slack_config.bot_token:
            logger.error("Neither webhook URL nor bot token configured")
            return False

        if self._slack_config.webhook_url:
            if not self._slack_config.webhook_url.startswith("https://hooks.slack.com"):
                logger.error("Invalid Slack webhook URL")
                return False

        return True

    async def _send(self, message: NotificationMessage) -> DeliveryResult:
        """Send Slack notification."""
        try:
            # Build message payload
            if self._slack_config.use_blocks:
                payload = SlackBlockBuilder.build_message(
                    message,
                    include_footer=self._slack_config.include_footer,
                    footer_text=self._slack_config.footer_text,
                )
            else:
                payload = SlackBlockBuilder.build_simple_message(message)

            # Add optional fields
            if self._slack_config.username:
                payload["username"] = self._slack_config.username

            if self._slack_config.icon_emoji:
                payload["icon_emoji"] = self._slack_config.icon_emoji
            elif self._slack_config.icon_url:
                payload["icon_url"] = self._slack_config.icon_url

            # Channel override
            if message.recipients:
                payload["channel"] = message.recipients[0]
            elif self._slack_config.default_channel:
                payload["channel"] = self._slack_config.default_channel

            # Thread reply
            if self._slack_config.thread_ts:
                payload["thread_ts"] = self._slack_config.thread_ts
                if self._slack_config.reply_broadcast:
                    payload["reply_broadcast"] = True

            # Send via webhook
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._send_webhook,
                payload,
            )

            if response.get("ok") or response.get("success"):
                logger.info("Slack message sent successfully")
                return DeliveryResult(
                    success=True,
                    message_id=message.message_id or "",
                    channel=self.channel,
                    response_data=response,
                )

            error = response.get("error", "Unknown error")
            logger.error(f"Slack API error: {error}")

            return DeliveryResult(
                success=False,
                message_id=message.message_id or "",
                channel=self.channel,
                error=error,
                retryable=error in ["rate_limited", "service_unavailable"],
            )

        except Exception as e:
            logger.error(f"Error sending Slack message: {e}")
            return DeliveryResult(
                success=False,
                message_id=message.message_id or "",
                channel=self.channel,
                error=str(e),
                retryable=True,
            )

    def _send_webhook(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send message via webhook."""
        data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            self._slack_config.webhook_url,
            data=data,
            method="POST",
        )
        request.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                response_text = response.read().decode()

                # Webhook returns "ok" on success
                if response_text == "ok":
                    return {"ok": True, "success": True}

                try:
                    return json.loads(response_text)
                except json.JSONDecodeError:
                    return {"ok": True, "response": response_text}

        except urllib.error.HTTPError as e:
            error_body = e.read().decode()
            try:
                return json.loads(error_body)
            except json.JSONDecodeError:
                return {"error": error_body, "code": e.code}

    async def send_trade_notification(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        pnl: float | None = None,
        channel: str | None = None,
        **kwargs: Any,
    ) -> DeliveryResult:
        """
        Send a trade notification.

        Args:
            symbol: Trading symbol
            action: Trade action (buy/sell)
            quantity: Trade quantity
            price: Trade price
            pnl: Profit/loss
            channel: Optional specific channel
            **kwargs: Additional trade details

        Returns:
            Delivery result
        """
        payload = SlackBlockBuilder.build_trade_message(
            symbol, action, quantity, price, pnl, **kwargs
        )

        if channel:
            payload["channel"] = channel
        elif self._slack_config.default_channel:
            payload["channel"] = self._slack_config.default_channel

        if self._slack_config.username:
            payload["username"] = self._slack_config.username

        if self._slack_config.icon_emoji:
            payload["icon_emoji"] = self._slack_config.icon_emoji

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            self._send_webhook,
            payload,
        )

        success = response.get("ok") or response.get("success")

        return DeliveryResult(
            success=bool(success),
            message_id=f"trade_{symbol}_{datetime.now().timestamp()}",
            channel=self.channel,
            error=response.get("error") if not success else None,
            response_data=response,
        )

    async def send_to_channel(
        self,
        channel: str,
        text: str,
        blocks: list[dict[str, Any]] | None = None,
    ) -> DeliveryResult:
        """
        Send message to specific channel.

        Args:
            channel: Channel name or ID
            text: Message text
            blocks: Optional Block Kit blocks

        Returns:
            Delivery result
        """
        message = NotificationMessage(
            title=text[:100],
            body=text,
            recipients=[channel],
        )
        return await self.send(message)


def create_slack_notifier(config: SlackConfig | None = None) -> SlackNotifier:
    """
    Create a Slack notifier instance.

    Args:
        config: Slack configuration

    Returns:
        SlackNotifier instance
    """
    return SlackNotifier(config)


def create_slack_config(
    webhook_url: str,
    default_channel: str = "#trading-alerts",
    **kwargs: Any,
) -> SlackConfig:
    """
    Create Slack configuration.

    Args:
        webhook_url: Slack webhook URL
        default_channel: Default channel for messages
        **kwargs: Additional configuration

    Returns:
        SlackConfig instance
    """
    return SlackConfig(
        webhook_url=webhook_url,
        default_channel=default_channel,
        **kwargs,
    )


# Module version
__version__ = "2.2.0"
