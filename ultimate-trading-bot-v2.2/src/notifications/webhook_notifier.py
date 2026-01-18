"""
Webhook Notifier Module for Ultimate Trading Bot v2.2.

This module provides webhook notification capabilities including:
- Generic HTTP webhook delivery
- Customizable payload formats
- Authentication support
- Retry and error handling
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
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


class WebhookFormat(str):
    """Webhook payload format types."""

    JSON = "json"
    FORM = "form"
    CUSTOM = "custom"


class AuthMethod(str):
    """Authentication methods."""

    NONE = "none"
    BASIC = "basic"
    BEARER = "bearer"
    API_KEY = "api_key"
    HMAC = "hmac"


@dataclass
class WebhookConfig(NotifierConfig):
    """Webhook notifier configuration."""

    # Endpoint settings
    webhook_url: str = ""
    method: str = "POST"

    # Format settings
    payload_format: str = WebhookFormat.JSON
    custom_payload_template: str | None = None

    # Authentication
    auth_method: str = AuthMethod.NONE
    auth_username: str = ""
    auth_password: str = ""
    auth_token: str = ""
    api_key_header: str = "X-API-Key"
    api_key_value: str = ""
    hmac_secret: str = ""
    hmac_header: str = "X-Signature"
    hmac_algorithm: str = "sha256"

    # Headers
    custom_headers: dict[str, str] = field(default_factory=dict)
    content_type: str = "application/json"

    # Request settings
    timeout_seconds: float = 30.0
    verify_ssl: bool = True

    # Response handling
    success_status_codes: list[int] = field(default_factory=lambda: [200, 201, 202, 204])
    success_response_field: str | None = None
    success_response_value: str | None = None


class WebhookPayloadBuilder:
    """Builds webhook payloads from notifications."""

    @classmethod
    def build_json_payload(
        cls,
        message: NotificationMessage,
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        """Build JSON payload from notification."""
        payload = {
            "event_type": "notification",
            "message_id": message.message_id,
            "title": message.title,
            "body": message.body,
            "type": message.message_type.value,
            "priority": message.priority.value,
            "timestamp": message.timestamp.isoformat(),
        }

        if message.html_body:
            payload["html_body"] = message.html_body

        if message.short_body:
            payload["short_body"] = message.short_body

        if include_metadata and message.metadata:
            payload["metadata"] = message.metadata

        if message.recipients:
            payload["recipients"] = message.recipients

        return payload

    @classmethod
    def build_form_payload(
        cls,
        message: NotificationMessage,
    ) -> dict[str, str]:
        """Build form-encoded payload from notification."""
        payload = {
            "event_type": "notification",
            "message_id": message.message_id or "",
            "title": message.title,
            "body": message.body,
            "type": message.message_type.value,
            "priority": message.priority.value,
            "timestamp": message.timestamp.isoformat(),
        }

        # Flatten metadata
        if message.metadata:
            for key, value in message.metadata.items():
                payload[f"metadata_{key}"] = str(value)

        return payload

    @classmethod
    def build_custom_payload(
        cls,
        message: NotificationMessage,
        template: str,
    ) -> str:
        """Build custom payload from template."""
        # Simple template substitution
        result = template

        replacements = {
            "{{message_id}}": message.message_id or "",
            "{{title}}": message.title,
            "{{body}}": message.body,
            "{{type}}": message.message_type.value,
            "{{priority}}": message.priority.value,
            "{{timestamp}}": message.timestamp.isoformat(),
        }

        for key, value in replacements.items():
            result = result.replace(key, value)

        # Replace metadata placeholders
        if message.metadata:
            for key, value in message.metadata.items():
                result = result.replace(f"{{{{metadata.{key}}}}}", str(value))

        return result

    @classmethod
    def build_trade_payload(
        cls,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        pnl: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build trade event payload."""
        payload = {
            "event_type": "trade",
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": price,
            "value": quantity * price,
            "timestamp": datetime.now().isoformat(),
        }

        if pnl is not None:
            payload["pnl"] = pnl

        payload.update(kwargs)

        return payload

    @classmethod
    def build_alert_payload(
        cls,
        title: str,
        message: str,
        severity: str = "info",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build alert event payload."""
        return {
            "event_type": "alert",
            "title": title,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }


class WebhookNotifier(BaseNotifier):
    """
    Webhook notification channel.

    Sends notifications via HTTP webhooks.
    """

    def __init__(self, config: WebhookConfig | None = None) -> None:
        """Initialize webhook notifier."""
        self._webhook_config = config or WebhookConfig()
        super().__init__(self._webhook_config)
        self._payload_builder = WebhookPayloadBuilder()

        logger.info("WebhookNotifier initialized")

    @property
    def channel(self) -> NotificationChannel:
        """Get notification channel type."""
        return NotificationChannel.WEBHOOK

    async def validate_config(self) -> bool:
        """Validate webhook configuration."""
        if not self._webhook_config.webhook_url:
            logger.error("Webhook URL not configured")
            return False

        # Validate URL format
        try:
            parsed = urllib.parse.urlparse(self._webhook_config.webhook_url)
            if parsed.scheme not in ["http", "https"]:
                logger.error("Invalid webhook URL scheme")
                return False
        except Exception as e:
            logger.error(f"Invalid webhook URL: {e}")
            return False

        # Validate auth configuration
        if self._webhook_config.auth_method == AuthMethod.BASIC:
            if not self._webhook_config.auth_username:
                logger.error("Basic auth requires username")
                return False

        elif self._webhook_config.auth_method == AuthMethod.BEARER:
            if not self._webhook_config.auth_token:
                logger.error("Bearer auth requires token")
                return False

        elif self._webhook_config.auth_method == AuthMethod.API_KEY:
            if not self._webhook_config.api_key_value:
                logger.error("API key auth requires key value")
                return False

        elif self._webhook_config.auth_method == AuthMethod.HMAC:
            if not self._webhook_config.hmac_secret:
                logger.error("HMAC auth requires secret")
                return False

        return True

    async def _send(self, message: NotificationMessage) -> DeliveryResult:
        """Send webhook notification."""
        try:
            # Build payload
            payload = self._build_payload(message)

            # Build request
            request = self._build_request(payload)

            # Send request
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._send_request,
                request,
            )

            # Check response
            if self._is_success_response(response):
                logger.info(f"Webhook sent successfully: {response.get('status_code')}")
                return DeliveryResult(
                    success=True,
                    message_id=message.message_id or "",
                    channel=self.channel,
                    response_data=response,
                )

            error = response.get("error", f"Status {response.get('status_code')}")
            logger.error(f"Webhook error: {error}")

            return DeliveryResult(
                success=False,
                message_id=message.message_id or "",
                channel=self.channel,
                error=error,
                retryable=response.get("status_code", 0) >= 500,
            )

        except Exception as e:
            logger.error(f"Error sending webhook: {e}")
            return DeliveryResult(
                success=False,
                message_id=message.message_id or "",
                channel=self.channel,
                error=str(e),
                retryable=True,
            )

    def _build_payload(self, message: NotificationMessage) -> str | bytes:
        """Build request payload."""
        if self._webhook_config.payload_format == WebhookFormat.JSON:
            data = WebhookPayloadBuilder.build_json_payload(message)
            return json.dumps(data)

        elif self._webhook_config.payload_format == WebhookFormat.FORM:
            data = WebhookPayloadBuilder.build_form_payload(message)
            return urllib.parse.urlencode(data)

        elif self._webhook_config.payload_format == WebhookFormat.CUSTOM:
            if self._webhook_config.custom_payload_template:
                return WebhookPayloadBuilder.build_custom_payload(
                    message,
                    self._webhook_config.custom_payload_template,
                )

        # Default to JSON
        data = WebhookPayloadBuilder.build_json_payload(message)
        return json.dumps(data)

    def _build_request(self, payload: str | bytes) -> urllib.request.Request:
        """Build HTTP request."""
        if isinstance(payload, str):
            data = payload.encode("utf-8")
        else:
            data = payload

        request = urllib.request.Request(
            self._webhook_config.webhook_url,
            data=data,
            method=self._webhook_config.method,
        )

        # Set content type
        content_type = self._webhook_config.content_type
        if self._webhook_config.payload_format == WebhookFormat.FORM:
            content_type = "application/x-www-form-urlencoded"

        request.add_header("Content-Type", content_type)

        # Add custom headers
        for header, value in self._webhook_config.custom_headers.items():
            request.add_header(header, value)

        # Add authentication
        self._add_auth_headers(request, data)

        return request

    def _add_auth_headers(
        self,
        request: urllib.request.Request,
        payload: bytes,
    ) -> None:
        """Add authentication headers to request."""
        if self._webhook_config.auth_method == AuthMethod.BASIC:
            import base64
            credentials = f"{self._webhook_config.auth_username}:{self._webhook_config.auth_password}"
            encoded = base64.b64encode(credentials.encode()).decode()
            request.add_header("Authorization", f"Basic {encoded}")

        elif self._webhook_config.auth_method == AuthMethod.BEARER:
            request.add_header("Authorization", f"Bearer {self._webhook_config.auth_token}")

        elif self._webhook_config.auth_method == AuthMethod.API_KEY:
            request.add_header(
                self._webhook_config.api_key_header,
                self._webhook_config.api_key_value,
            )

        elif self._webhook_config.auth_method == AuthMethod.HMAC:
            # Generate HMAC signature
            algorithm = getattr(hashlib, self._webhook_config.hmac_algorithm)
            signature = hmac.new(
                self._webhook_config.hmac_secret.encode(),
                payload,
                algorithm,
            ).hexdigest()

            request.add_header(self._webhook_config.hmac_header, signature)

            # Also add timestamp for replay protection
            request.add_header("X-Timestamp", str(int(time.time())))

    def _send_request(self, request: urllib.request.Request) -> dict[str, Any]:
        """Send HTTP request."""
        try:
            with urllib.request.urlopen(
                request,
                timeout=self._webhook_config.timeout_seconds,
            ) as response:
                response_body = response.read().decode()

                result = {
                    "status_code": response.status,
                    "headers": dict(response.headers),
                }

                try:
                    result["body"] = json.loads(response_body)
                except json.JSONDecodeError:
                    result["body"] = response_body

                return result

        except urllib.error.HTTPError as e:
            error_body = e.read().decode()
            return {
                "status_code": e.code,
                "error": error_body,
            }

        except urllib.error.URLError as e:
            return {
                "status_code": 0,
                "error": str(e.reason),
            }

    def _is_success_response(self, response: dict[str, Any]) -> bool:
        """Check if response indicates success."""
        status_code = response.get("status_code", 0)

        if status_code not in self._webhook_config.success_status_codes:
            return False

        # Check response field if configured
        if self._webhook_config.success_response_field:
            body = response.get("body", {})
            if isinstance(body, dict):
                field_value = body.get(self._webhook_config.success_response_field)
                if self._webhook_config.success_response_value:
                    return str(field_value) == self._webhook_config.success_response_value
                return field_value is not None

        return True

    async def send_trade_event(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        pnl: float | None = None,
        **kwargs: Any,
    ) -> DeliveryResult:
        """Send a trade event webhook."""
        payload = WebhookPayloadBuilder.build_trade_payload(
            symbol, action, quantity, price, pnl, **kwargs
        )

        request = self._build_request(json.dumps(payload))

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            self._send_request,
            request,
        )

        success = self._is_success_response(response)

        return DeliveryResult(
            success=success,
            message_id=f"trade_{symbol}_{datetime.now().timestamp()}",
            channel=self.channel,
            error=response.get("error") if not success else None,
            response_data=response,
        )

    async def send_alert_event(
        self,
        title: str,
        message: str,
        severity: str = "info",
        **kwargs: Any,
    ) -> DeliveryResult:
        """Send an alert event webhook."""
        payload = WebhookPayloadBuilder.build_alert_payload(
            title, message, severity, **kwargs
        )

        request = self._build_request(json.dumps(payload))

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            self._send_request,
            request,
        )

        success = self._is_success_response(response)

        return DeliveryResult(
            success=success,
            message_id=f"alert_{datetime.now().timestamp()}",
            channel=self.channel,
            error=response.get("error") if not success else None,
            response_data=response,
        )

    async def send_custom_event(
        self,
        event_type: str,
        data: dict[str, Any],
    ) -> DeliveryResult:
        """Send a custom event webhook."""
        payload = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            **data,
        }

        request = self._build_request(json.dumps(payload))

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            self._send_request,
            request,
        )

        success = self._is_success_response(response)

        return DeliveryResult(
            success=success,
            message_id=f"{event_type}_{datetime.now().timestamp()}",
            channel=self.channel,
            error=response.get("error") if not success else None,
            response_data=response,
        )


def create_webhook_notifier(
    config: WebhookConfig | None = None,
) -> WebhookNotifier:
    """
    Create a webhook notifier instance.

    Args:
        config: Webhook configuration

    Returns:
        WebhookNotifier instance
    """
    return WebhookNotifier(config)


def create_webhook_config(
    webhook_url: str,
    auth_method: str = AuthMethod.NONE,
    **kwargs: Any,
) -> WebhookConfig:
    """
    Create webhook configuration.

    Args:
        webhook_url: Webhook URL
        auth_method: Authentication method
        **kwargs: Additional configuration

    Returns:
        WebhookConfig instance
    """
    return WebhookConfig(
        webhook_url=webhook_url,
        auth_method=auth_method,
        **kwargs,
    )


# Module version
__version__ = "2.2.0"
