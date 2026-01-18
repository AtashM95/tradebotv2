"""
SMS Notifier Module for Ultimate Trading Bot v2.2.

This module provides SMS notification capabilities including:
- Twilio SMS integration
- Message formatting for SMS
- Delivery status tracking
- Rate limiting
"""

import asyncio
import hashlib
import hmac
import logging
import urllib.parse
import urllib.request
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .base_notifier import (
    BaseNotifier,
    NotificationChannel,
    NotificationMessage,
    NotificationPriority,
    NotifierConfig,
    DeliveryResult,
)


logger = logging.getLogger(__name__)


@dataclass
class SMSConfig(NotifierConfig):
    """SMS notifier configuration."""

    # Twilio settings
    account_sid: str = ""
    auth_token: str = ""
    from_number: str = ""

    # Default recipients
    default_recipients: list[str] = field(default_factory=list)

    # Message settings
    max_message_length: int = 1600
    truncate_messages: bool = True
    include_timestamp: bool = True
    include_priority_prefix: bool = True

    # Delivery settings
    status_callback_url: str | None = None
    validity_period: int = 14400  # 4 hours in seconds


class SMSFormatter:
    """Formats messages for SMS delivery."""

    # Priority prefixes
    PRIORITY_PREFIXES = {
        NotificationPriority.LOW: "",
        NotificationPriority.NORMAL: "",
        NotificationPriority.HIGH: "[HIGH] ",
        NotificationPriority.URGENT: "[URGENT] ",
        NotificationPriority.CRITICAL: "[CRITICAL] ",
    }

    @classmethod
    def format_message(
        cls,
        message: NotificationMessage,
        max_length: int = 1600,
        include_timestamp: bool = True,
        include_priority: bool = True,
    ) -> str:
        """
        Format notification for SMS.

        Args:
            message: Notification message
            max_length: Maximum message length
            include_timestamp: Include timestamp in message
            include_priority: Include priority prefix

        Returns:
            Formatted SMS text
        """
        parts = []

        # Priority prefix
        if include_priority and message.priority in [
            NotificationPriority.HIGH,
            NotificationPriority.URGENT,
            NotificationPriority.CRITICAL,
        ]:
            parts.append(cls.PRIORITY_PREFIXES.get(message.priority, ""))

        # Title
        parts.append(message.title)
        parts.append("\n")

        # Body (use short_body if available)
        body = message.short_body or message.body
        parts.append(body)

        # Timestamp
        if include_timestamp:
            parts.append(f"\n{message.timestamp.strftime('%H:%M')}")

        # Join and truncate
        text = "".join(parts)

        if len(text) > max_length:
            text = text[:max_length - 3] + "..."

        return text

    @classmethod
    def format_trade_sms(
        cls,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        pnl: float | None = None,
    ) -> str:
        """Format trade notification for SMS."""
        parts = [
            f"TRADE: {action.upper()} {quantity} {symbol} @ ${price:,.2f}",
        ]

        if pnl is not None:
            sign = "+" if pnl >= 0 else ""
            parts.append(f"P&L: {sign}${pnl:,.2f}")

        return " | ".join(parts)

    @classmethod
    def format_alert_sms(
        cls,
        title: str,
        message: str,
        severity: str = "info",
    ) -> str:
        """Format alert notification for SMS."""
        prefix = ""
        if severity in ["warning", "error", "critical"]:
            prefix = f"[{severity.upper()}] "

        text = f"{prefix}{title}: {message}"

        if len(text) > 160:
            text = text[:157] + "..."

        return text


class TwilioClient:
    """Simple Twilio API client."""

    BASE_URL = "https://api.twilio.com/2010-04-01"

    def __init__(self, account_sid: str, auth_token: str) -> None:
        """Initialize Twilio client."""
        self._account_sid = account_sid
        self._auth_token = auth_token

    def _get_auth_header(self) -> str:
        """Get HTTP Basic Auth header."""
        credentials = f"{self._account_sid}:{self._auth_token}"
        encoded = b64encode(credentials.encode()).decode()
        return f"Basic {encoded}"

    async def send_message(
        self,
        to: str,
        from_number: str,
        body: str,
        status_callback: str | None = None,
        validity_period: int | None = None,
    ) -> dict[str, Any]:
        """
        Send SMS via Twilio API.

        Args:
            to: Recipient phone number
            from_number: Sender phone number
            body: Message body
            status_callback: URL for status updates
            validity_period: Message validity in seconds

        Returns:
            Twilio API response
        """
        url = f"{self.BASE_URL}/Accounts/{self._account_sid}/Messages.json"

        # Build request data
        data = {
            "To": to,
            "From": from_number,
            "Body": body,
        }

        if status_callback:
            data["StatusCallback"] = status_callback

        if validity_period:
            data["ValidityPeriod"] = str(validity_period)

        # Send request
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            self._make_request,
            url,
            data,
        )

        return response

    def _make_request(self, url: str, data: dict[str, str]) -> dict[str, Any]:
        """Make HTTP request to Twilio API."""
        encoded_data = urllib.parse.urlencode(data).encode()

        request = urllib.request.Request(url, data=encoded_data, method="POST")
        request.add_header("Authorization", self._get_auth_header())
        request.add_header("Content-Type", "application/x-www-form-urlencoded")

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                import json
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            import json
            error_body = e.read().decode()
            try:
                error_data = json.loads(error_body)
                return {"error": error_data}
            except Exception:
                return {"error": {"message": error_body, "code": e.code}}

    async def get_message_status(self, message_sid: str) -> dict[str, Any]:
        """Get status of a sent message."""
        url = f"{self.BASE_URL}/Accounts/{self._account_sid}/Messages/{message_sid}.json"

        loop = asyncio.get_event_loop()

        def make_get_request() -> dict[str, Any]:
            request = urllib.request.Request(url, method="GET")
            request.add_header("Authorization", self._get_auth_header())

            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    import json
                    return json.loads(response.read().decode())
            except Exception as e:
                return {"error": str(e)}

        return await loop.run_in_executor(None, make_get_request)

    def validate_signature(
        self,
        signature: str,
        url: str,
        params: dict[str, str],
    ) -> bool:
        """Validate Twilio webhook signature."""
        # Build string to sign
        data = url + "".join(
            f"{k}{v}" for k, v in sorted(params.items())
        )

        # Calculate expected signature
        expected = b64encode(
            hmac.new(
                self._auth_token.encode(),
                data.encode(),
                hashlib.sha1,
            ).digest()
        ).decode()

        return hmac.compare_digest(signature, expected)


class SMSNotifier(BaseNotifier):
    """
    SMS notification channel.

    Sends notifications via Twilio SMS.
    """

    def __init__(self, config: SMSConfig | None = None) -> None:
        """Initialize SMS notifier."""
        self._sms_config = config or SMSConfig()
        super().__init__(self._sms_config)

        self._client: TwilioClient | None = None
        self._formatter = SMSFormatter()
        self._message_status: dict[str, dict[str, Any]] = {}

        logger.info("SMSNotifier initialized")

    @property
    def channel(self) -> NotificationChannel:
        """Get notification channel type."""
        return NotificationChannel.SMS

    def _get_client(self) -> TwilioClient:
        """Get or create Twilio client."""
        if self._client is None:
            self._client = TwilioClient(
                self._sms_config.account_sid,
                self._sms_config.auth_token,
            )
        return self._client

    async def validate_config(self) -> bool:
        """Validate SMS configuration."""
        if not self._sms_config.account_sid:
            logger.error("Twilio account SID not configured")
            return False

        if not self._sms_config.auth_token:
            logger.error("Twilio auth token not configured")
            return False

        if not self._sms_config.from_number:
            logger.error("From phone number not configured")
            return False

        # Validate phone number format
        if not self._sms_config.from_number.startswith("+"):
            logger.error("Phone number must be in E.164 format (e.g., +1234567890)")
            return False

        return True

    async def _send(self, message: NotificationMessage) -> DeliveryResult:
        """Send SMS notification."""
        try:
            # Format message
            sms_text = SMSFormatter.format_message(
                message,
                max_length=self._sms_config.max_message_length,
                include_timestamp=self._sms_config.include_timestamp,
                include_priority=self._sms_config.include_priority_prefix,
            )

            # Get recipients
            recipients = message.recipients or self._sms_config.default_recipients
            if not recipients:
                return DeliveryResult(
                    success=False,
                    message_id=message.message_id or "",
                    channel=self.channel,
                    error="No recipients specified",
                )

            # Send to each recipient
            client = self._get_client()
            results: list[dict[str, Any]] = []
            errors: list[str] = []

            for recipient in recipients:
                # Normalize phone number
                phone = self._normalize_phone(recipient)

                response = await client.send_message(
                    to=phone,
                    from_number=self._sms_config.from_number,
                    body=sms_text,
                    status_callback=self._sms_config.status_callback_url,
                    validity_period=self._sms_config.validity_period,
                )

                if "error" in response:
                    error_msg = response.get("error", {}).get("message", "Unknown error")
                    errors.append(f"{recipient}: {error_msg}")
                else:
                    results.append(response)

                    # Track message status
                    if "sid" in response:
                        self._message_status[response["sid"]] = {
                            "recipient": recipient,
                            "status": response.get("status"),
                            "timestamp": datetime.now(),
                        }

            if errors and not results:
                return DeliveryResult(
                    success=False,
                    message_id=message.message_id or "",
                    channel=self.channel,
                    error="; ".join(errors),
                    retryable=True,
                )

            logger.info(
                f"SMS sent to {len(results)}/{len(recipients)} recipients"
            )

            return DeliveryResult(
                success=True,
                message_id=message.message_id or "",
                channel=self.channel,
                recipient=", ".join(recipients),
                response_data={
                    "sent_count": len(results),
                    "failed_count": len(errors),
                    "message_sids": [r.get("sid") for r in results if "sid" in r],
                },
            )

        except Exception as e:
            logger.error(f"Error sending SMS: {e}")
            return DeliveryResult(
                success=False,
                message_id=message.message_id or "",
                channel=self.channel,
                error=str(e),
                retryable=True,
            )

    def _normalize_phone(self, phone: str) -> str:
        """Normalize phone number to E.164 format."""
        # Remove any non-digit characters except +
        cleaned = "".join(c for c in phone if c.isdigit() or c == "+")

        # Add + if not present
        if not cleaned.startswith("+"):
            # Assume US number if 10 digits
            if len(cleaned) == 10:
                cleaned = "+1" + cleaned
            elif len(cleaned) == 11 and cleaned.startswith("1"):
                cleaned = "+" + cleaned
            else:
                cleaned = "+" + cleaned

        return cleaned

    async def get_delivery_status(self, message_sid: str) -> dict[str, Any] | None:
        """
        Get delivery status for a sent message.

        Args:
            message_sid: Twilio message SID

        Returns:
            Message status or None if not found
        """
        client = self._get_client()
        return await client.get_message_status(message_sid)

    async def send_quick_sms(
        self,
        to: str,
        text: str,
    ) -> DeliveryResult:
        """
        Send a quick SMS message.

        Args:
            to: Recipient phone number
            text: Message text

        Returns:
            Delivery result
        """
        message = NotificationMessage(
            title="",
            body=text,
            recipients=[to],
        )
        return await self.send(message)


def create_sms_notifier(config: SMSConfig | None = None) -> SMSNotifier:
    """
    Create an SMS notifier instance.

    Args:
        config: SMS configuration

    Returns:
        SMSNotifier instance
    """
    return SMSNotifier(config)


def create_sms_config(
    account_sid: str,
    auth_token: str,
    from_number: str,
    **kwargs: Any,
) -> SMSConfig:
    """
    Create SMS configuration.

    Args:
        account_sid: Twilio account SID
        auth_token: Twilio auth token
        from_number: Sender phone number
        **kwargs: Additional configuration

    Returns:
        SMSConfig instance
    """
    return SMSConfig(
        account_sid=account_sid,
        auth_token=auth_token,
        from_number=from_number,
        **kwargs,
    )


# Module version
__version__ = "2.2.0"
