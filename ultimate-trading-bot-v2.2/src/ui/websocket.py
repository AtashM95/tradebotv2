"""
WebSocket Module for Ultimate Trading Bot v2.2.

This module provides WebSocket functionality including:
- Real-time data streaming
- Event broadcasting
- Client management
- Channel subscriptions
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from flask import Flask, request


logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """WebSocket message types."""

    # System messages
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    PING = "ping"
    PONG = "pong"
    ERROR = "error"

    # Subscription messages
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    SUBSCRIBED = "subscribed"
    UNSUBSCRIBED = "unsubscribed"

    # Data messages
    DATA = "data"
    UPDATE = "update"
    SNAPSHOT = "snapshot"

    # Trading messages
    TRADE = "trade"
    ORDER = "order"
    POSITION = "position"
    QUOTE = "quote"

    # System events
    ALERT = "alert"
    NOTIFICATION = "notification"
    STATUS = "status"


@dataclass
class WebSocketMessage:
    """WebSocket message structure."""

    type: MessageType
    channel: str | None = None
    data: Any = None
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps({
            "type": self.type.value,
            "channel": self.channel,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "id": self.message_id,
        })

    @classmethod
    def from_json(cls, data: str) -> "WebSocketMessage":
        """Create from JSON string."""
        parsed = json.loads(data)
        return cls(
            type=MessageType(parsed.get("type", "data")),
            channel=parsed.get("channel"),
            data=parsed.get("data"),
            message_id=parsed.get("id", str(uuid.uuid4())[:8]),
        )


@dataclass
class WebSocketClient:
    """WebSocket client information."""

    client_id: str
    user_id: str | None = None
    subscriptions: set[str] = field(default_factory=set)
    connected_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_subscribed(self, channel: str) -> bool:
        """Check if client is subscribed to channel."""
        return channel in self.subscriptions

    def subscribe(self, channel: str) -> None:
        """Subscribe to channel."""
        self.subscriptions.add(channel)

    def unsubscribe(self, channel: str) -> None:
        """Unsubscribe from channel."""
        self.subscriptions.discard(channel)


class Channel:
    """WebSocket channel."""

    def __init__(self, name: str, description: str = "") -> None:
        """Initialize channel."""
        self.name = name
        self.description = description
        self.subscribers: set[str] = set()
        self.created_at = datetime.now()
        self.message_count = 0

    def add_subscriber(self, client_id: str) -> None:
        """Add subscriber to channel."""
        self.subscribers.add(client_id)

    def remove_subscriber(self, client_id: str) -> None:
        """Remove subscriber from channel."""
        self.subscribers.discard(client_id)

    @property
    def subscriber_count(self) -> int:
        """Get number of subscribers."""
        return len(self.subscribers)


class WebSocketManager:
    """
    WebSocket connection manager.

    Manages WebSocket connections, channels, and message broadcasting.
    """

    def __init__(self) -> None:
        """Initialize WebSocket manager."""
        self._clients: dict[str, WebSocketClient] = {}
        self._channels: dict[str, Channel] = {}
        self._message_handlers: dict[MessageType, list[Callable]] = {}
        self._send_callback: Callable[[str, str], None] | None = None
        self._lock = threading.Lock()

        # Statistics
        self._stats = {
            "total_connections": 0,
            "total_messages_sent": 0,
            "total_messages_received": 0,
        }

        # Initialize default channels
        self._init_default_channels()

        logger.info("WebSocketManager initialized")

    def _init_default_channels(self) -> None:
        """Initialize default channels."""
        default_channels = [
            ("trades", "Real-time trade updates"),
            ("orders", "Order status updates"),
            ("positions", "Position changes"),
            ("quotes", "Price quotes"),
            ("alerts", "System alerts"),
            ("system", "System status updates"),
            ("portfolio", "Portfolio updates"),
            ("signals", "Trading signals"),
        ]

        for name, description in default_channels:
            self._channels[name] = Channel(name, description)

    def set_send_callback(
        self,
        callback: Callable[[str, str], None],
    ) -> None:
        """Set callback for sending messages to clients."""
        self._send_callback = callback

    def on_connect(self, client_id: str, user_id: str | None = None) -> None:
        """
        Handle client connection.

        Args:
            client_id: Client identifier
            user_id: Optional user ID
        """
        with self._lock:
            client = WebSocketClient(
                client_id=client_id,
                user_id=user_id,
            )
            self._clients[client_id] = client
            self._stats["total_connections"] += 1

        logger.info(f"Client connected: {client_id}")

        # Send welcome message
        self._send_to_client(
            client_id,
            WebSocketMessage(
                type=MessageType.CONNECT,
                data={
                    "client_id": client_id,
                    "available_channels": list(self._channels.keys()),
                },
            ),
        )

    def on_disconnect(self, client_id: str) -> None:
        """
        Handle client disconnection.

        Args:
            client_id: Client identifier
        """
        with self._lock:
            client = self._clients.pop(client_id, None)

            if client:
                # Remove from all channel subscriptions
                for channel_name in client.subscriptions:
                    if channel_name in self._channels:
                        self._channels[channel_name].remove_subscriber(client_id)

        logger.info(f"Client disconnected: {client_id}")

    def on_message(self, client_id: str, message: str) -> None:
        """
        Handle incoming message.

        Args:
            client_id: Client identifier
            message: Raw message string
        """
        try:
            msg = WebSocketMessage.from_json(message)
            self._stats["total_messages_received"] += 1

            # Update client activity
            with self._lock:
                client = self._clients.get(client_id)
                if client:
                    client.last_activity = datetime.now()

            # Handle message based on type
            if msg.type == MessageType.PING:
                self._handle_ping(client_id)
            elif msg.type == MessageType.SUBSCRIBE:
                self._handle_subscribe(client_id, msg.channel)
            elif msg.type == MessageType.UNSUBSCRIBE:
                self._handle_unsubscribe(client_id, msg.channel)
            else:
                # Call registered handlers
                self._call_handlers(msg.type, client_id, msg)

        except json.JSONDecodeError:
            self._send_error(client_id, "Invalid JSON message")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            self._send_error(client_id, str(e))

    def _handle_ping(self, client_id: str) -> None:
        """Handle ping message."""
        self._send_to_client(
            client_id,
            WebSocketMessage(type=MessageType.PONG),
        )

    def _handle_subscribe(self, client_id: str, channel: str | None) -> None:
        """Handle subscription request."""
        if not channel:
            self._send_error(client_id, "Channel name required")
            return

        if channel not in self._channels:
            self._send_error(client_id, f"Unknown channel: {channel}")
            return

        with self._lock:
            client = self._clients.get(client_id)
            if client:
                client.subscribe(channel)
                self._channels[channel].add_subscriber(client_id)

        self._send_to_client(
            client_id,
            WebSocketMessage(
                type=MessageType.SUBSCRIBED,
                channel=channel,
                data={"channel": channel},
            ),
        )

        logger.debug(f"Client {client_id} subscribed to {channel}")

    def _handle_unsubscribe(self, client_id: str, channel: str | None) -> None:
        """Handle unsubscription request."""
        if not channel:
            self._send_error(client_id, "Channel name required")
            return

        with self._lock:
            client = self._clients.get(client_id)
            if client:
                client.unsubscribe(channel)

            if channel in self._channels:
                self._channels[channel].remove_subscriber(client_id)

        self._send_to_client(
            client_id,
            WebSocketMessage(
                type=MessageType.UNSUBSCRIBED,
                channel=channel,
                data={"channel": channel},
            ),
        )

        logger.debug(f"Client {client_id} unsubscribed from {channel}")

    def _send_to_client(self, client_id: str, message: WebSocketMessage) -> None:
        """Send message to specific client."""
        if self._send_callback:
            try:
                self._send_callback(client_id, message.to_json())
                self._stats["total_messages_sent"] += 1
            except Exception as e:
                logger.error(f"Error sending to client {client_id}: {e}")

    def _send_error(self, client_id: str, error: str) -> None:
        """Send error message to client."""
        self._send_to_client(
            client_id,
            WebSocketMessage(
                type=MessageType.ERROR,
                data={"error": error},
            ),
        )

    def _call_handlers(
        self,
        message_type: MessageType,
        client_id: str,
        message: WebSocketMessage,
    ) -> None:
        """Call registered message handlers."""
        handlers = self._message_handlers.get(message_type, [])
        for handler in handlers:
            try:
                handler(client_id, message)
            except Exception as e:
                logger.error(f"Handler error: {e}")

    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable[[str, WebSocketMessage], None],
    ) -> None:
        """Register message handler."""
        if message_type not in self._message_handlers:
            self._message_handlers[message_type] = []
        self._message_handlers[message_type].append(handler)

    def broadcast(
        self,
        channel: str,
        data: Any,
        message_type: MessageType = MessageType.DATA,
    ) -> int:
        """
        Broadcast message to channel subscribers.

        Args:
            channel: Channel name
            data: Message data
            message_type: Message type

        Returns:
            Number of clients messaged
        """
        if channel not in self._channels:
            return 0

        message = WebSocketMessage(
            type=message_type,
            channel=channel,
            data=data,
        )

        channel_obj = self._channels[channel]
        channel_obj.message_count += 1

        count = 0
        for client_id in list(channel_obj.subscribers):
            self._send_to_client(client_id, message)
            count += 1

        return count

    def broadcast_all(
        self,
        data: Any,
        message_type: MessageType = MessageType.DATA,
    ) -> int:
        """
        Broadcast message to all connected clients.

        Args:
            data: Message data
            message_type: Message type

        Returns:
            Number of clients messaged
        """
        message = WebSocketMessage(
            type=message_type,
            data=data,
        )

        count = 0
        with self._lock:
            for client_id in list(self._clients.keys()):
                self._send_to_client(client_id, message)
                count += 1

        return count

    def send_to_user(
        self,
        user_id: str,
        data: Any,
        message_type: MessageType = MessageType.DATA,
        channel: str | None = None,
    ) -> int:
        """
        Send message to all connections for a user.

        Args:
            user_id: User ID
            data: Message data
            message_type: Message type
            channel: Optional channel

        Returns:
            Number of clients messaged
        """
        message = WebSocketMessage(
            type=message_type,
            channel=channel,
            data=data,
        )

        count = 0
        with self._lock:
            for client_id, client in self._clients.items():
                if client.user_id == user_id:
                    self._send_to_client(client_id, message)
                    count += 1

        return count

    def broadcast_trade(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        **kwargs: Any,
    ) -> int:
        """Broadcast trade event."""
        return self.broadcast(
            "trades",
            {
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "price": price,
                **kwargs,
            },
            MessageType.TRADE,
        )

    def broadcast_order(
        self,
        order_id: str,
        symbol: str,
        status: str,
        **kwargs: Any,
    ) -> int:
        """Broadcast order update."""
        return self.broadcast(
            "orders",
            {
                "order_id": order_id,
                "symbol": symbol,
                "status": status,
                **kwargs,
            },
            MessageType.ORDER,
        )

    def broadcast_quote(
        self,
        symbol: str,
        price: float,
        bid: float | None = None,
        ask: float | None = None,
        **kwargs: Any,
    ) -> int:
        """Broadcast price quote."""
        return self.broadcast(
            "quotes",
            {
                "symbol": symbol,
                "price": price,
                "bid": bid,
                "ask": ask,
                **kwargs,
            },
            MessageType.QUOTE,
        )

    def broadcast_alert(
        self,
        title: str,
        message: str,
        severity: str = "info",
        **kwargs: Any,
    ) -> int:
        """Broadcast alert."""
        return self.broadcast(
            "alerts",
            {
                "title": title,
                "message": message,
                "severity": severity,
                **kwargs,
            },
            MessageType.ALERT,
        )

    def get_client(self, client_id: str) -> WebSocketClient | None:
        """Get client by ID."""
        return self._clients.get(client_id)

    def get_channel(self, channel_name: str) -> Channel | None:
        """Get channel by name."""
        return self._channels.get(channel_name)

    def get_channel_subscribers(self, channel: str) -> list[str]:
        """Get list of subscriber IDs for channel."""
        if channel not in self._channels:
            return []
        return list(self._channels[channel].subscribers)

    def get_client_count(self) -> int:
        """Get number of connected clients."""
        return len(self._clients)

    def get_stats(self) -> dict[str, Any]:
        """Get WebSocket statistics."""
        with self._lock:
            return {
                **self._stats,
                "active_connections": len(self._clients),
                "channels": {
                    name: {
                        "subscribers": ch.subscriber_count,
                        "messages": ch.message_count,
                    }
                    for name, ch in self._channels.items()
                },
            }

    def cleanup_inactive(self, timeout_seconds: int = 300) -> int:
        """
        Clean up inactive connections.

        Args:
            timeout_seconds: Inactivity timeout

        Returns:
            Number of connections cleaned up
        """
        now = datetime.now()
        to_remove = []

        with self._lock:
            for client_id, client in self._clients.items():
                delta = (now - client.last_activity).total_seconds()
                if delta > timeout_seconds:
                    to_remove.append(client_id)

        for client_id in to_remove:
            self.on_disconnect(client_id)

        return len(to_remove)


# Global WebSocket manager instance
_ws_manager: WebSocketManager | None = None


def get_websocket_manager() -> WebSocketManager:
    """Get or create WebSocket manager instance."""
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = WebSocketManager()
    return _ws_manager


def create_websocket_manager() -> WebSocketManager:
    """Create a new WebSocket manager instance."""
    return WebSocketManager()


# Module version
__version__ = "2.2.0"
