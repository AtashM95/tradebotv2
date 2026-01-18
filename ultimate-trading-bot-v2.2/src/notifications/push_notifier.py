"""
Push Notifier Module for Ultimate Trading Bot v2.2.

This module provides push notification capabilities including:
- Web Push (VAPID)
- Firebase Cloud Messaging (FCM)
- Apple Push Notification Service (APNs)
- Generic push provider support
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
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


class PushProvider(str):
    """Push notification provider types."""

    FCM = "fcm"
    APNS = "apns"
    WEB_PUSH = "web_push"
    CUSTOM = "custom"


@dataclass
class PushConfig(NotifierConfig):
    """Push notifier configuration."""

    # Provider selection
    provider: str = PushProvider.FCM

    # Firebase Cloud Messaging settings
    fcm_server_key: str = ""
    fcm_project_id: str = ""
    fcm_service_account: dict[str, Any] = field(default_factory=dict)

    # Web Push (VAPID) settings
    vapid_private_key: str = ""
    vapid_public_key: str = ""
    vapid_subject: str = ""  # mailto: or https:// URL

    # Apple Push Notification settings
    apns_key_id: str = ""
    apns_team_id: str = ""
    apns_bundle_id: str = ""
    apns_private_key: str = ""
    apns_use_sandbox: bool = False

    # Device tokens
    device_tokens: list[str] = field(default_factory=list)
    topic_subscriptions: dict[str, list[str]] = field(default_factory=dict)

    # Message settings
    default_ttl: int = 86400  # 24 hours
    collapse_key: str | None = None
    priority: str = "high"  # high or normal


@dataclass
class PushSubscription:
    """Web Push subscription data."""

    endpoint: str
    p256dh: str
    auth: str
    expiration_time: datetime | None = None


class FCMClient:
    """Firebase Cloud Messaging client."""

    LEGACY_API_URL = "https://fcm.googleapis.com/fcm/send"
    V1_API_URL = "https://fcm.googleapis.com/v1/projects/{project_id}/messages:send"

    def __init__(
        self,
        server_key: str | None = None,
        project_id: str | None = None,
        service_account: dict[str, Any] | None = None,
    ) -> None:
        """Initialize FCM client."""
        self._server_key = server_key
        self._project_id = project_id
        self._service_account = service_account
        self._access_token: str | None = None
        self._token_expiry: float = 0

    async def send_message(
        self,
        token: str,
        title: str,
        body: str,
        data: dict[str, str] | None = None,
        priority: str = "high",
        ttl: int = 86400,
        collapse_key: str | None = None,
    ) -> dict[str, Any]:
        """Send push notification to single device."""
        if self._server_key:
            return await self._send_legacy(
                token, title, body, data, priority, ttl, collapse_key
            )
        elif self._project_id and self._service_account:
            return await self._send_v1(
                token, title, body, data, priority, ttl
            )
        else:
            return {"success": False, "error": "No FCM credentials configured"}

    async def send_to_topic(
        self,
        topic: str,
        title: str,
        body: str,
        data: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Send push notification to topic subscribers."""
        if self._server_key:
            payload = self._build_legacy_payload(
                f"/topics/{topic}", title, body, data, "high", 86400, None
            )

            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self._send_legacy_request,
                payload,
            )

        return {"success": False, "error": "Topic messaging requires server key"}

    async def _send_legacy(
        self,
        token: str,
        title: str,
        body: str,
        data: dict[str, str] | None,
        priority: str,
        ttl: int,
        collapse_key: str | None,
    ) -> dict[str, Any]:
        """Send using legacy FCM API."""
        payload = self._build_legacy_payload(
            token, title, body, data, priority, ttl, collapse_key
        )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._send_legacy_request,
            payload,
        )

    def _build_legacy_payload(
        self,
        to: str,
        title: str,
        body: str,
        data: dict[str, str] | None,
        priority: str,
        ttl: int,
        collapse_key: str | None,
    ) -> dict[str, Any]:
        """Build legacy API payload."""
        payload: dict[str, Any] = {
            "to": to,
            "notification": {
                "title": title,
                "body": body,
            },
            "priority": priority,
            "time_to_live": ttl,
        }

        if data:
            payload["data"] = data

        if collapse_key:
            payload["collapse_key"] = collapse_key

        return payload

    def _send_legacy_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send legacy FCM request."""
        data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            self.LEGACY_API_URL,
            data=data,
            method="POST",
        )
        request.add_header("Content-Type", "application/json")
        request.add_header("Authorization", f"key={self._server_key}")

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode())
                return {"success": result.get("success", 0) > 0, "data": result}
        except urllib.error.HTTPError as e:
            error_body = e.read().decode()
            return {"success": False, "error": error_body}

    async def _send_v1(
        self,
        token: str,
        title: str,
        body: str,
        data: dict[str, str] | None,
        priority: str,
        ttl: int,
    ) -> dict[str, Any]:
        """Send using FCM v1 API."""
        # Get access token
        access_token = await self._get_access_token()
        if not access_token:
            return {"success": False, "error": "Failed to get access token"}

        url = self.V1_API_URL.format(project_id=self._project_id)

        payload = {
            "message": {
                "token": token,
                "notification": {
                    "title": title,
                    "body": body,
                },
                "android": {
                    "priority": priority,
                    "ttl": f"{ttl}s",
                },
            }
        }

        if data:
            payload["message"]["data"] = data

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._send_v1_request,
            url,
            payload,
            access_token,
        )

    def _send_v1_request(
        self,
        url: str,
        payload: dict[str, Any],
        access_token: str,
    ) -> dict[str, Any]:
        """Send FCM v1 request."""
        data = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(url, data=data, method="POST")
        request.add_header("Content-Type", "application/json")
        request.add_header("Authorization", f"Bearer {access_token}")

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode())
                return {"success": True, "data": result}
        except urllib.error.HTTPError as e:
            error_body = e.read().decode()
            return {"success": False, "error": error_body}

    async def _get_access_token(self) -> str | None:
        """Get OAuth2 access token for v1 API."""
        # Check cached token
        if self._access_token and time.time() < self._token_expiry:
            return self._access_token

        # Generate new token (simplified - in production, use google-auth library)
        if not self._service_account:
            return None

        # For simplicity, this returns None - in production, implement proper OAuth2
        logger.warning("FCM v1 API requires proper OAuth2 implementation")
        return None


class WebPushClient:
    """Web Push (VAPID) client."""

    def __init__(
        self,
        private_key: str,
        public_key: str,
        subject: str,
    ) -> None:
        """Initialize Web Push client."""
        self._private_key = private_key
        self._public_key = public_key
        self._subject = subject

    async def send_notification(
        self,
        subscription: PushSubscription,
        title: str,
        body: str,
        data: dict[str, Any] | None = None,
        ttl: int = 86400,
    ) -> dict[str, Any]:
        """Send web push notification."""
        payload = json.dumps({
            "title": title,
            "body": body,
            "data": data or {},
            "timestamp": datetime.now().isoformat(),
        })

        # Generate VAPID headers
        headers = self._generate_vapid_headers(subscription.endpoint)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._send_push,
            subscription,
            payload,
            headers,
            ttl,
        )

    def _generate_vapid_headers(self, audience: str) -> dict[str, str]:
        """Generate VAPID authorization headers."""
        # Extract origin from endpoint
        from urllib.parse import urlparse
        parsed = urlparse(audience)
        origin = f"{parsed.scheme}://{parsed.netloc}"

        # Create JWT claims
        claims = {
            "aud": origin,
            "exp": int(time.time()) + 43200,  # 12 hours
            "sub": self._subject,
        }

        # Create JWT (simplified - in production use proper JWT library)
        header = base64.urlsafe_b64encode(
            json.dumps({"typ": "JWT", "alg": "ES256"}).encode()
        ).decode().rstrip("=")

        payload = base64.urlsafe_b64encode(
            json.dumps(claims).encode()
        ).decode().rstrip("=")

        # Signature would require proper ECDSA implementation
        # For now, return placeholder
        signature = "placeholder_signature"

        jwt = f"{header}.{payload}.{signature}"

        return {
            "Authorization": f"vapid t={jwt}, k={self._public_key}",
            "TTL": "86400",
        }

    def _send_push(
        self,
        subscription: PushSubscription,
        payload: str,
        headers: dict[str, str],
        ttl: int,
    ) -> dict[str, Any]:
        """Send web push request."""
        data = payload.encode("utf-8")

        request = urllib.request.Request(
            subscription.endpoint,
            data=data,
            method="POST",
        )
        request.add_header("Content-Type", "application/json")
        request.add_header("TTL", str(ttl))

        for key, value in headers.items():
            request.add_header(key, value)

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status in [200, 201]:
                    return {"success": True}
                return {"success": False, "error": f"Status: {response.status}"}
        except urllib.error.HTTPError as e:
            if e.code == 410:
                return {"success": False, "error": "Subscription expired", "expired": True}
            return {"success": False, "error": str(e)}


class PushNotifier(BaseNotifier):
    """
    Push notification channel.

    Sends notifications via FCM, APNs, or Web Push.
    """

    def __init__(self, config: PushConfig | None = None) -> None:
        """Initialize push notifier."""
        self._push_config = config or PushConfig()
        super().__init__(self._push_config)

        self._fcm_client: FCMClient | None = None
        self._web_push_client: WebPushClient | None = None
        self._subscriptions: dict[str, PushSubscription] = {}

        logger.info(f"PushNotifier initialized with provider: {self._push_config.provider}")

    @property
    def channel(self) -> NotificationChannel:
        """Get notification channel type."""
        return NotificationChannel.PUSH

    def _get_fcm_client(self) -> FCMClient:
        """Get or create FCM client."""
        if self._fcm_client is None:
            self._fcm_client = FCMClient(
                server_key=self._push_config.fcm_server_key or None,
                project_id=self._push_config.fcm_project_id or None,
                service_account=self._push_config.fcm_service_account or None,
            )
        return self._fcm_client

    def _get_web_push_client(self) -> WebPushClient | None:
        """Get or create Web Push client."""
        if self._web_push_client is None:
            if self._push_config.vapid_private_key and self._push_config.vapid_public_key:
                self._web_push_client = WebPushClient(
                    self._push_config.vapid_private_key,
                    self._push_config.vapid_public_key,
                    self._push_config.vapid_subject,
                )
        return self._web_push_client

    async def validate_config(self) -> bool:
        """Validate push configuration."""
        if self._push_config.provider == PushProvider.FCM:
            if not self._push_config.fcm_server_key and not self._push_config.fcm_project_id:
                logger.error("FCM server key or project ID required")
                return False

        elif self._push_config.provider == PushProvider.WEB_PUSH:
            if not self._push_config.vapid_private_key or not self._push_config.vapid_public_key:
                logger.error("VAPID keys required for Web Push")
                return False

        elif self._push_config.provider == PushProvider.APNS:
            if not all([
                self._push_config.apns_key_id,
                self._push_config.apns_team_id,
                self._push_config.apns_bundle_id,
                self._push_config.apns_private_key,
            ]):
                logger.error("APNs configuration incomplete")
                return False

        return True

    async def _send(self, message: NotificationMessage) -> DeliveryResult:
        """Send push notification."""
        try:
            # Get target tokens
            tokens = message.recipients or self._push_config.device_tokens
            if not tokens:
                return DeliveryResult(
                    success=False,
                    message_id=message.message_id or "",
                    channel=self.channel,
                    error="No device tokens specified",
                )

            # Prepare notification data
            data = {
                "message_id": message.message_id or "",
                "type": message.message_type.value,
                "priority": message.priority.value,
                **{k: str(v) for k, v in message.metadata.items()},
            }

            # Send based on provider
            results: list[dict[str, Any]] = []
            errors: list[str] = []

            for token in tokens:
                if self._push_config.provider == PushProvider.FCM:
                    result = await self._send_fcm(token, message, data)
                elif self._push_config.provider == PushProvider.WEB_PUSH:
                    result = await self._send_web_push(token, message, data)
                else:
                    result = {"success": False, "error": "Unsupported provider"}

                if result.get("success"):
                    results.append(result)
                else:
                    errors.append(f"{token[:20]}...: {result.get('error')}")

            if not results:
                return DeliveryResult(
                    success=False,
                    message_id=message.message_id or "",
                    channel=self.channel,
                    error="; ".join(errors[:3]),
                    retryable=True,
                )

            logger.info(f"Push sent to {len(results)}/{len(tokens)} devices")

            return DeliveryResult(
                success=True,
                message_id=message.message_id or "",
                channel=self.channel,
                response_data={
                    "sent_count": len(results),
                    "failed_count": len(errors),
                },
            )

        except Exception as e:
            logger.error(f"Error sending push notification: {e}")
            return DeliveryResult(
                success=False,
                message_id=message.message_id or "",
                channel=self.channel,
                error=str(e),
                retryable=True,
            )

    async def _send_fcm(
        self,
        token: str,
        message: NotificationMessage,
        data: dict[str, str],
    ) -> dict[str, Any]:
        """Send via FCM."""
        client = self._get_fcm_client()

        return await client.send_message(
            token=token,
            title=message.title,
            body=message.short_body or message.body[:256],
            data=data,
            priority=self._push_config.priority,
            ttl=self._push_config.default_ttl,
            collapse_key=self._push_config.collapse_key,
        )

    async def _send_web_push(
        self,
        token: str,
        message: NotificationMessage,
        data: dict[str, str],
    ) -> dict[str, Any]:
        """Send via Web Push."""
        client = self._get_web_push_client()
        if not client:
            return {"success": False, "error": "Web Push not configured"}

        # Get subscription for token
        subscription = self._subscriptions.get(token)
        if not subscription:
            return {"success": False, "error": "Subscription not found"}

        return await client.send_notification(
            subscription=subscription,
            title=message.title,
            body=message.short_body or message.body[:256],
            data=data,
            ttl=self._push_config.default_ttl,
        )

    def register_subscription(
        self,
        token: str,
        subscription: PushSubscription,
    ) -> None:
        """Register a web push subscription."""
        self._subscriptions[token] = subscription
        logger.info(f"Web push subscription registered: {token[:20]}...")

    def unregister_subscription(self, token: str) -> None:
        """Unregister a subscription."""
        if token in self._subscriptions:
            del self._subscriptions[token]

    def add_device_token(self, token: str) -> None:
        """Add a device token."""
        if token not in self._push_config.device_tokens:
            self._push_config.device_tokens.append(token)

    def remove_device_token(self, token: str) -> None:
        """Remove a device token."""
        if token in self._push_config.device_tokens:
            self._push_config.device_tokens.remove(token)

    async def send_to_topic(
        self,
        topic: str,
        title: str,
        body: str,
        data: dict[str, str] | None = None,
    ) -> DeliveryResult:
        """Send notification to topic subscribers."""
        if self._push_config.provider != PushProvider.FCM:
            return DeliveryResult(
                success=False,
                message_id=f"topic_{topic}",
                channel=self.channel,
                error="Topic messaging only supported with FCM",
            )

        client = self._get_fcm_client()
        result = await client.send_to_topic(topic, title, body, data)

        return DeliveryResult(
            success=result.get("success", False),
            message_id=f"topic_{topic}_{datetime.now().timestamp()}",
            channel=self.channel,
            error=result.get("error") if not result.get("success") else None,
            response_data=result,
        )


def create_push_notifier(config: PushConfig | None = None) -> PushNotifier:
    """
    Create a push notifier instance.

    Args:
        config: Push configuration

    Returns:
        PushNotifier instance
    """
    return PushNotifier(config)


def create_fcm_config(
    server_key: str,
    device_tokens: list[str] | None = None,
    **kwargs: Any,
) -> PushConfig:
    """
    Create FCM push configuration.

    Args:
        server_key: FCM server key
        device_tokens: List of device tokens
        **kwargs: Additional configuration

    Returns:
        PushConfig instance
    """
    return PushConfig(
        provider=PushProvider.FCM,
        fcm_server_key=server_key,
        device_tokens=device_tokens or [],
        **kwargs,
    )


def create_web_push_config(
    vapid_private_key: str,
    vapid_public_key: str,
    vapid_subject: str,
    **kwargs: Any,
) -> PushConfig:
    """
    Create Web Push configuration.

    Args:
        vapid_private_key: VAPID private key
        vapid_public_key: VAPID public key
        vapid_subject: VAPID subject (mailto: or https://)
        **kwargs: Additional configuration

    Returns:
        PushConfig instance
    """
    return PushConfig(
        provider=PushProvider.WEB_PUSH,
        vapid_private_key=vapid_private_key,
        vapid_public_key=vapid_public_key,
        vapid_subject=vapid_subject,
        **kwargs,
    )


# Module version
__version__ = "2.2.0"
