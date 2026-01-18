"""
Telegram Notifier Module for Ultimate Trading Bot v2.2.

This module provides Telegram notification capabilities including:
- Bot API integration
- Rich message formatting with Markdown/HTML
- Inline keyboards
- Photo/document sending
- Chat management
"""

import asyncio
import json
import logging
import urllib.error
import urllib.parse
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
class TelegramConfig(NotifierConfig):
    """Telegram notifier configuration."""

    # Bot settings
    bot_token: str = ""

    # Default chat
    default_chat_id: str = ""

    # Chat groups for different notification types
    chat_ids: dict[str, str] = field(default_factory=dict)

    # Message settings
    parse_mode: str = "MarkdownV2"  # MarkdownV2, HTML, or Markdown
    disable_web_page_preview: bool = True
    disable_notification: bool = False

    # Retry settings
    api_timeout: float = 30.0


class TelegramFormatter:
    """Formats messages for Telegram."""

    # Emoji mapping for message types
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

    # Priority emoji
    PRIORITY_EMOJI = {
        NotificationPriority.LOW: "",
        NotificationPriority.NORMAL: "",
        NotificationPriority.HIGH: "🔶",
        NotificationPriority.URGENT: "🔴",
        NotificationPriority.CRITICAL: "🆘",
    }

    @classmethod
    def escape_markdown_v2(cls, text: str) -> str:
        """Escape special characters for MarkdownV2."""
        special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for char in special_chars:
            text = text.replace(char, f'\\{char}')
        return text

    @classmethod
    def format_markdown_v2(cls, message: NotificationMessage) -> str:
        """Format message as MarkdownV2."""
        emoji = cls.TYPE_EMOJI.get(message.message_type, "📬")
        priority_emoji = cls.PRIORITY_EMOJI.get(message.priority, "")

        # Escape special characters
        title = cls.escape_markdown_v2(message.title)
        body = cls.escape_markdown_v2(message.body)

        parts = []

        # Header
        if priority_emoji:
            parts.append(f"{priority_emoji} *{title}*")
        else:
            parts.append(f"{emoji} *{title}*")

        parts.append("")

        # Body
        parts.append(body)

        # Metadata
        if message.metadata:
            parts.append("")
            parts.append("━━━━━━━━━━━━━━━")
            for key, value in message.metadata.items():
                formatted_key = cls.escape_markdown_v2(key.replace("_", " ").title())
                formatted_value = cls.escape_markdown_v2(str(value))
                parts.append(f"• *{formatted_key}:* {formatted_value}")

        # Timestamp
        parts.append("")
        timestamp = cls.escape_markdown_v2(message.timestamp.strftime('%Y\\--%m\\--%d %H:%M:%S'))
        parts.append(f"🕐 {timestamp}")

        return "\n".join(parts)

    @classmethod
    def format_html(cls, message: NotificationMessage) -> str:
        """Format message as HTML."""
        emoji = cls.TYPE_EMOJI.get(message.message_type, "📬")
        priority_emoji = cls.PRIORITY_EMOJI.get(message.priority, "")

        parts = []

        # Header
        if priority_emoji:
            parts.append(f"{priority_emoji} <b>{message.title}</b>")
        else:
            parts.append(f"{emoji} <b>{message.title}</b>")

        parts.append("")

        # Body
        body = message.body.replace("**", "").replace("*", "")
        parts.append(body)

        # Metadata
        if message.metadata:
            parts.append("")
            parts.append("━━━━━━━━━━━━━━━")
            for key, value in message.metadata.items():
                formatted_key = key.replace("_", " ").title()
                parts.append(f"• <b>{formatted_key}:</b> {value}")

        # Timestamp
        parts.append("")
        parts.append(f"🕐 <i>{message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</i>")

        return "\n".join(parts)

    @classmethod
    def format_trade_message(
        cls,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        pnl: float | None = None,
        parse_mode: str = "HTML",
    ) -> str:
        """Format trade notification."""
        emoji = "📈" if action.lower() == "buy" else "📉"
        pnl_emoji = "💰" if pnl and pnl >= 0 else "💸"

        if parse_mode == "HTML":
            parts = [
                f"{emoji} <b>TRADE: {action.upper()} {symbol}</b>",
                "",
                f"<b>Quantity:</b> {quantity:,.4f}",
                f"<b>Price:</b> ${price:,.2f}",
                f"<b>Value:</b> ${quantity * price:,.2f}",
            ]

            if pnl is not None:
                sign = "+" if pnl >= 0 else ""
                parts.append(f"{pnl_emoji} <b>P&L:</b> {sign}${pnl:,.2f}")

        else:
            escaped_symbol = cls.escape_markdown_v2(symbol)
            parts = [
                f"{emoji} *TRADE: {action.upper()} {escaped_symbol}*",
                "",
                f"*Quantity:* {quantity:,.4f}".replace(",", "\\,").replace(".", "\\."),
                f"*Price:* \\${price:,.2f}".replace(",", "\\,"),
                f"*Value:* \\${quantity * price:,.2f}".replace(",", "\\,"),
            ]

            if pnl is not None:
                sign = "\\+" if pnl >= 0 else ""
                pnl_str = f"{pnl:,.2f}".replace(",", "\\,")
                parts.append(f"{pnl_emoji} *P&L:* {sign}\\${pnl_str}")

        return "\n".join(parts)


class TelegramBotAPI:
    """Telegram Bot API client."""

    BASE_URL = "https://api.telegram.org/bot"

    def __init__(self, token: str, timeout: float = 30.0) -> None:
        """Initialize Telegram Bot API client."""
        self._token = token
        self._timeout = timeout
        self._base_url = f"{self.BASE_URL}{token}"

    async def send_message(
        self,
        chat_id: str,
        text: str,
        parse_mode: str | None = None,
        disable_web_page_preview: bool = True,
        disable_notification: bool = False,
        reply_to_message_id: int | None = None,
        reply_markup: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send text message."""
        params: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": disable_web_page_preview,
            "disable_notification": disable_notification,
        }

        if parse_mode:
            params["parse_mode"] = parse_mode

        if reply_to_message_id:
            params["reply_to_message_id"] = reply_to_message_id

        if reply_markup:
            params["reply_markup"] = json.dumps(reply_markup)

        return await self._make_request("sendMessage", params)

    async def send_photo(
        self,
        chat_id: str,
        photo: str,
        caption: str | None = None,
        parse_mode: str | None = None,
    ) -> dict[str, Any]:
        """Send photo."""
        params: dict[str, Any] = {
            "chat_id": chat_id,
            "photo": photo,
        }

        if caption:
            params["caption"] = caption

        if parse_mode:
            params["parse_mode"] = parse_mode

        return await self._make_request("sendPhoto", params)

    async def send_document(
        self,
        chat_id: str,
        document: str,
        caption: str | None = None,
    ) -> dict[str, Any]:
        """Send document."""
        params: dict[str, Any] = {
            "chat_id": chat_id,
            "document": document,
        }

        if caption:
            params["caption"] = caption

        return await self._make_request("sendDocument", params)

    async def get_me(self) -> dict[str, Any]:
        """Get bot information."""
        return await self._make_request("getMe", {})

    async def get_chat(self, chat_id: str) -> dict[str, Any]:
        """Get chat information."""
        return await self._make_request("getChat", {"chat_id": chat_id})

    async def _make_request(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Make API request."""
        url = f"{self._base_url}/{method}"

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._sync_request,
            url,
            params,
        )

    def _sync_request(
        self,
        url: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Make synchronous HTTP request."""
        data = urllib.parse.urlencode(params).encode("utf-8")

        request = urllib.request.Request(url, data=data, method="POST")
        request.add_header("Content-Type", "application/x-www-form-urlencoded")

        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode()
            try:
                return json.loads(error_body)
            except json.JSONDecodeError:
                return {"ok": False, "description": error_body}
        except Exception as e:
            return {"ok": False, "description": str(e)}


class TelegramNotifier(BaseNotifier):
    """
    Telegram notification channel.

    Sends notifications via Telegram Bot API.
    """

    def __init__(self, config: TelegramConfig | None = None) -> None:
        """Initialize Telegram notifier."""
        self._telegram_config = config or TelegramConfig()
        super().__init__(self._telegram_config)

        self._api: TelegramBotAPI | None = None
        self._formatter = TelegramFormatter()

        logger.info("TelegramNotifier initialized")

    @property
    def channel(self) -> NotificationChannel:
        """Get notification channel type."""
        return NotificationChannel.TELEGRAM

    def _get_api(self) -> TelegramBotAPI:
        """Get or create API client."""
        if self._api is None:
            self._api = TelegramBotAPI(
                self._telegram_config.bot_token,
                self._telegram_config.api_timeout,
            )
        return self._api

    async def validate_config(self) -> bool:
        """Validate Telegram configuration."""
        if not self._telegram_config.bot_token:
            logger.error("Telegram bot token not configured")
            return False

        if not self._telegram_config.default_chat_id:
            logger.error("Default chat ID not configured")
            return False

        # Test bot token
        try:
            api = self._get_api()
            result = await api.get_me()
            if not result.get("ok"):
                logger.error(f"Invalid bot token: {result.get('description')}")
                return False
            logger.info(f"Telegram bot validated: @{result.get('result', {}).get('username')}")
            return True
        except Exception as e:
            logger.error(f"Failed to validate Telegram config: {e}")
            return False

    async def _send(self, message: NotificationMessage) -> DeliveryResult:
        """Send Telegram notification."""
        try:
            # Format message
            if self._telegram_config.parse_mode == "HTML":
                text = TelegramFormatter.format_html(message)
            else:
                text = TelegramFormatter.format_markdown_v2(message)

            # Get chat ID
            chat_id = self._get_chat_id(message)

            # Send message
            api = self._get_api()
            result = await api.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=self._telegram_config.parse_mode,
                disable_web_page_preview=self._telegram_config.disable_web_page_preview,
                disable_notification=self._should_disable_notification(message),
            )

            if result.get("ok"):
                message_data = result.get("result", {})
                logger.info(f"Telegram message sent: {message_data.get('message_id')}")

                return DeliveryResult(
                    success=True,
                    message_id=message.message_id or "",
                    channel=self.channel,
                    recipient=chat_id,
                    response_data={
                        "telegram_message_id": message_data.get("message_id"),
                        "chat_id": chat_id,
                    },
                )

            error = result.get("description", "Unknown error")
            logger.error(f"Telegram API error: {error}")

            # Determine if retryable
            retryable = "Too Many Requests" in error or "temporarily unavailable" in error.lower()

            return DeliveryResult(
                success=False,
                message_id=message.message_id or "",
                channel=self.channel,
                error=error,
                retryable=retryable,
            )

        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return DeliveryResult(
                success=False,
                message_id=message.message_id or "",
                channel=self.channel,
                error=str(e),
                retryable=True,
            )

    def _get_chat_id(self, message: NotificationMessage) -> str:
        """Get appropriate chat ID for message."""
        # Check for specific recipient
        if message.recipients:
            return message.recipients[0]

        # Check for type-specific chat
        type_chat = self._telegram_config.chat_ids.get(message.message_type.value)
        if type_chat:
            return type_chat

        # Use default
        return self._telegram_config.default_chat_id

    def _should_disable_notification(self, message: NotificationMessage) -> bool:
        """Determine if notification sound should be disabled."""
        if self._telegram_config.disable_notification:
            return True

        # Disable for low priority
        return message.priority == NotificationPriority.LOW

    async def send_trade_notification(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        pnl: float | None = None,
        chat_id: str | None = None,
    ) -> DeliveryResult:
        """Send a trade notification."""
        text = TelegramFormatter.format_trade_message(
            symbol, action, quantity, price, pnl,
            parse_mode=self._telegram_config.parse_mode,
        )

        api = self._get_api()
        result = await api.send_message(
            chat_id=chat_id or self._telegram_config.default_chat_id,
            text=text,
            parse_mode=self._telegram_config.parse_mode,
            disable_web_page_preview=True,
        )

        success = result.get("ok", False)

        return DeliveryResult(
            success=success,
            message_id=f"trade_{symbol}_{datetime.now().timestamp()}",
            channel=self.channel,
            error=result.get("description") if not success else None,
            response_data=result,
        )

    async def send_photo(
        self,
        photo_url: str,
        caption: str | None = None,
        chat_id: str | None = None,
    ) -> DeliveryResult:
        """Send a photo."""
        api = self._get_api()
        result = await api.send_photo(
            chat_id=chat_id or self._telegram_config.default_chat_id,
            photo=photo_url,
            caption=caption,
            parse_mode=self._telegram_config.parse_mode,
        )

        success = result.get("ok", False)

        return DeliveryResult(
            success=success,
            message_id=f"photo_{datetime.now().timestamp()}",
            channel=self.channel,
            error=result.get("description") if not success else None,
        )


def create_telegram_notifier(
    config: TelegramConfig | None = None,
) -> TelegramNotifier:
    """
    Create a Telegram notifier instance.

    Args:
        config: Telegram configuration

    Returns:
        TelegramNotifier instance
    """
    return TelegramNotifier(config)


def create_telegram_config(
    bot_token: str,
    default_chat_id: str,
    **kwargs: Any,
) -> TelegramConfig:
    """
    Create Telegram configuration.

    Args:
        bot_token: Telegram bot token
        default_chat_id: Default chat ID
        **kwargs: Additional configuration

    Returns:
        TelegramConfig instance
    """
    return TelegramConfig(
        bot_token=bot_token,
        default_chat_id=default_chat_id,
        **kwargs,
    )


# Module version
__version__ = "2.2.0"
