"""
WebSocket Client Module for Ultimate Trading Bot v2.2.

This module provides WebSocket connectivity for real-time market data
streaming from various providers.
"""

import asyncio
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

import websockets
from websockets.client import WebSocketClientProtocol
from pydantic import BaseModel, Field

from src.data.base_provider import Quote, Bar, Trade
from src.utils.exceptions import WebSocketError, APIConnectionError
from src.utils.helpers import generate_uuid
from src.utils.date_utils import now_utc, parse_datetime
from src.utils.decorators import async_retry


logger = logging.getLogger(__name__)


class WebSocketState(str, Enum):
    """WebSocket connection state enumeration."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATING = "authenticating"
    AUTHENTICATED = "authenticated"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class StreamType(str, Enum):
    """Data stream type enumeration."""

    QUOTES = "quotes"
    TRADES = "trades"
    BARS = "bars"
    NEWS = "news"
    STATUS = "status"


class WebSocketConfig(BaseModel):
    """Configuration for WebSocket client."""

    url: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None

    auto_reconnect: bool = Field(default=True)
    reconnect_delay_seconds: float = Field(default=1.0, ge=0.5, le=30.0)
    max_reconnect_attempts: int = Field(default=10, ge=1, le=100)
    reconnect_backoff_factor: float = Field(default=1.5, ge=1.0, le=3.0)

    ping_interval_seconds: float = Field(default=30.0, ge=10.0, le=120.0)
    ping_timeout_seconds: float = Field(default=10.0, ge=5.0, le=30.0)

    message_queue_size: int = Field(default=1000, ge=100, le=10000)


class StreamSubscription(BaseModel):
    """Stream subscription model."""

    subscription_id: str = Field(default_factory=generate_uuid)
    stream_type: StreamType
    symbols: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=now_utc)
    active: bool = Field(default=True)


class WebSocketClient:
    """
    WebSocket client for real-time market data streaming.

    This class provides:
    - Automatic reconnection handling
    - Message parsing and distribution
    - Subscription management
    - Heartbeat monitoring
    """

    def __init__(
        self,
        config: WebSocketConfig,
    ) -> None:
        """
        Initialize WebSocketClient.

        Args:
            config: WebSocket configuration
        """
        self._config = config
        self._state = WebSocketState.DISCONNECTED

        self._ws: Optional[WebSocketClientProtocol] = None

        self._subscriptions: dict[str, StreamSubscription] = {}

        self._quote_callbacks: list[Callable[[Quote], None]] = []
        self._trade_callbacks: list[Callable[[Trade], None]] = []
        self._bar_callbacks: list[Callable[[Bar], None]] = []
        self._message_callbacks: list[Callable[[dict], None]] = []
        self._error_callbacks: list[Callable[[Exception], None]] = []
        self._state_callbacks: list[Callable[[WebSocketState], None]] = []

        self._message_queue: asyncio.Queue = asyncio.Queue(
            maxsize=config.message_queue_size
        )

        self._receive_task: Optional[asyncio.Task] = None
        self._process_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None

        self._reconnect_attempts = 0
        self._running = False
        self._lock = asyncio.Lock()

        self._messages_received = 0
        self._messages_processed = 0
        self._last_message_time: Optional[datetime] = None

        logger.info("WebSocketClient initialized")

    @property
    def state(self) -> WebSocketState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected and authenticated."""
        return self._state in (
            WebSocketState.CONNECTED,
            WebSocketState.AUTHENTICATED,
        )

    @property
    def subscriptions(self) -> dict[str, StreamSubscription]:
        """Get active subscriptions."""
        return {
            k: v for k, v in self._subscriptions.items()
            if v.active
        }

    def on_quote(self, callback: Callable[[Quote], None]) -> None:
        """Register callback for quote updates."""
        self._quote_callbacks.append(callback)

    def on_trade(self, callback: Callable[[Trade], None]) -> None:
        """Register callback for trade updates."""
        self._trade_callbacks.append(callback)

    def on_bar(self, callback: Callable[[Bar], None]) -> None:
        """Register callback for bar updates."""
        self._bar_callbacks.append(callback)

    def on_message(self, callback: Callable[[dict], None]) -> None:
        """Register callback for raw messages."""
        self._message_callbacks.append(callback)

    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """Register callback for errors."""
        self._error_callbacks.append(callback)

    def on_state_change(self, callback: Callable[[WebSocketState], None]) -> None:
        """Register callback for state changes."""
        self._state_callbacks.append(callback)

    async def connect(self) -> bool:
        """
        Connect to WebSocket server.

        Returns:
            True if connection successful
        """
        if self._state == WebSocketState.CONNECTING:
            return False

        await self._set_state(WebSocketState.CONNECTING)

        try:
            self._ws = await websockets.connect(
                self._config.url,
                ping_interval=self._config.ping_interval_seconds,
                ping_timeout=self._config.ping_timeout_seconds,
            )

            await self._set_state(WebSocketState.CONNECTED)
            self._reconnect_attempts = 0

            if self._config.api_key:
                await self._authenticate()

            self._running = True
            self._start_tasks()

            logger.info("WebSocket connected")
            return True

        except Exception as e:
            await self._set_state(WebSocketState.ERROR)
            logger.error(f"WebSocket connection error: {e}")
            await self._notify_error(e)

            if self._config.auto_reconnect:
                asyncio.create_task(self._reconnect())

            return False

    async def disconnect(self) -> None:
        """Disconnect from WebSocket server."""
        self._running = False

        self._stop_tasks()

        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
            self._ws = None

        await self._set_state(WebSocketState.DISCONNECTED)
        logger.info("WebSocket disconnected")

    async def _authenticate(self) -> bool:
        """Authenticate with the WebSocket server."""
        await self._set_state(WebSocketState.AUTHENTICATING)

        try:
            auth_message = {
                "action": "auth",
                "key": self._config.api_key,
                "secret": self._config.api_secret,
            }

            await self._send(auth_message)

            await self._set_state(WebSocketState.AUTHENTICATED)
            logger.info("WebSocket authenticated")
            return True

        except Exception as e:
            await self._set_state(WebSocketState.ERROR)
            logger.error(f"WebSocket authentication error: {e}")
            return False

    async def _reconnect(self) -> None:
        """Attempt to reconnect to the server."""
        if not self._config.auto_reconnect:
            return

        if self._reconnect_attempts >= self._config.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            await self._set_state(WebSocketState.ERROR)
            return

        await self._set_state(WebSocketState.RECONNECTING)
        self._reconnect_attempts += 1

        delay = self._config.reconnect_delay_seconds * (
            self._config.reconnect_backoff_factor ** (self._reconnect_attempts - 1)
        )
        delay = min(delay, 60.0)

        logger.info(
            f"Reconnecting in {delay:.1f}s "
            f"(attempt {self._reconnect_attempts}/{self._config.max_reconnect_attempts})"
        )

        await asyncio.sleep(delay)

        success = await self.connect()

        if success:
            await self._resubscribe()

    async def _resubscribe(self) -> None:
        """Resubscribe to all active subscriptions."""
        for sub_id, subscription in self._subscriptions.items():
            if subscription.active:
                try:
                    await self._send_subscription(
                        subscription.stream_type,
                        subscription.symbols,
                    )
                except Exception as e:
                    logger.error(f"Error resubscribing {sub_id}: {e}")

    def _start_tasks(self) -> None:
        """Start background tasks."""
        self._receive_task = asyncio.create_task(
            self._receive_loop(),
            name="ws_receive"
        )
        self._process_task = asyncio.create_task(
            self._process_loop(),
            name="ws_process"
        )

    def _stop_tasks(self) -> None:
        """Stop background tasks."""
        for task in [self._receive_task, self._process_task, self._ping_task]:
            if task:
                task.cancel()

        self._receive_task = None
        self._process_task = None
        self._ping_task = None

    async def _receive_loop(self) -> None:
        """Receive messages from WebSocket."""
        while self._running and self._ws:
            try:
                message = await self._ws.recv()
                self._messages_received += 1
                self._last_message_time = now_utc()

                try:
                    data = json.loads(message)
                    await self._message_queue.put(data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON message: {message[:100]}")

            except websockets.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                if self._config.auto_reconnect and self._running:
                    asyncio.create_task(self._reconnect())
                break

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                await self._notify_error(e)

    async def _process_loop(self) -> None:
        """Process messages from queue."""
        while self._running:
            try:
                data = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0
                )

                await self._handle_message(data)
                self._messages_processed += 1

            except asyncio.TimeoutError:
                continue

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.error(f"Error processing message: {e}")

    async def _handle_message(self, data: dict) -> None:
        """Handle a received message."""
        for callback in self._message_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Error in message callback: {e}")

        msg_type = data.get("T") or data.get("type") or data.get("stream")

        if msg_type in ("q", "quote", "quotes"):
            await self._handle_quote(data)
        elif msg_type in ("t", "trade", "trades"):
            await self._handle_trade(data)
        elif msg_type in ("b", "bar", "bars"):
            await self._handle_bar(data)
        elif msg_type == "error":
            logger.error(f"Server error: {data}")
        elif msg_type == "success":
            logger.debug(f"Server success: {data}")

    async def _handle_quote(self, data: dict) -> None:
        """Handle quote message."""
        try:
            quote = Quote(
                symbol=data.get("S") or data.get("symbol", ""),
                bid_price=float(data.get("bp") or data.get("bid_price", 0)),
                bid_size=int(data.get("bs") or data.get("bid_size", 0)),
                ask_price=float(data.get("ap") or data.get("ask_price", 0)),
                ask_size=int(data.get("as") or data.get("ask_size", 0)),
                timestamp=parse_datetime(data.get("t") or data.get("timestamp")),
            )

            for callback in self._quote_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(quote)
                    else:
                        callback(quote)
                except Exception as e:
                    logger.error(f"Error in quote callback: {e}")

        except Exception as e:
            logger.error(f"Error parsing quote: {e}")

    async def _handle_trade(self, data: dict) -> None:
        """Handle trade message."""
        try:
            trade = Trade(
                symbol=data.get("S") or data.get("symbol", ""),
                price=float(data.get("p") or data.get("price", 0)),
                size=int(data.get("s") or data.get("size", 0)),
                timestamp=parse_datetime(data.get("t") or data.get("timestamp")),
                exchange=data.get("x") or data.get("exchange"),
                conditions=data.get("c") or data.get("conditions"),
            )

            for callback in self._trade_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(trade)
                    else:
                        callback(trade)
                except Exception as e:
                    logger.error(f"Error in trade callback: {e}")

        except Exception as e:
            logger.error(f"Error parsing trade: {e}")

    async def _handle_bar(self, data: dict) -> None:
        """Handle bar message."""
        try:
            bar = Bar(
                symbol=data.get("S") or data.get("symbol", ""),
                timestamp=parse_datetime(data.get("t") or data.get("timestamp")),
                open=float(data.get("o") or data.get("open", 0)),
                high=float(data.get("h") or data.get("high", 0)),
                low=float(data.get("l") or data.get("low", 0)),
                close=float(data.get("c") or data.get("close", 0)),
                volume=int(data.get("v") or data.get("volume", 0)),
            )

            for callback in self._bar_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(bar)
                    else:
                        callback(bar)
                except Exception as e:
                    logger.error(f"Error in bar callback: {e}")

        except Exception as e:
            logger.error(f"Error parsing bar: {e}")

    async def _send(self, data: dict) -> None:
        """Send message to WebSocket server."""
        if not self._ws:
            raise WebSocketError("Not connected")

        try:
            message = json.dumps(data)
            await self._ws.send(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise WebSocketError(f"Send error: {e}")

    async def _send_subscription(
        self,
        stream_type: StreamType,
        symbols: list[str],
    ) -> None:
        """Send subscription message."""
        message = {
            "action": "subscribe",
            stream_type.value: symbols,
        }
        await self._send(message)

    async def _send_unsubscription(
        self,
        stream_type: StreamType,
        symbols: list[str],
    ) -> None:
        """Send unsubscription message."""
        message = {
            "action": "unsubscribe",
            stream_type.value: symbols,
        }
        await self._send(message)

    async def subscribe(
        self,
        stream_type: StreamType,
        symbols: list[str],
    ) -> str:
        """
        Subscribe to data streams.

        Args:
            stream_type: Type of data stream
            symbols: Symbols to subscribe to

        Returns:
            Subscription ID
        """
        subscription = StreamSubscription(
            stream_type=stream_type,
            symbols=symbols,
        )

        self._subscriptions[subscription.subscription_id] = subscription

        if self.is_connected:
            await self._send_subscription(stream_type, symbols)

        logger.info(f"Subscribed to {stream_type.value}: {symbols}")
        return subscription.subscription_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from data streams.

        Args:
            subscription_id: Subscription to cancel

        Returns:
            True if unsubscribed
        """
        if subscription_id not in self._subscriptions:
            return False

        subscription = self._subscriptions[subscription_id]
        subscription.active = False

        if self.is_connected:
            await self._send_unsubscription(
                subscription.stream_type,
                subscription.symbols,
            )

        del self._subscriptions[subscription_id]

        logger.info(f"Unsubscribed: {subscription_id}")
        return True

    async def subscribe_quotes(self, symbols: list[str]) -> str:
        """Subscribe to quote updates."""
        return await self.subscribe(StreamType.QUOTES, symbols)

    async def subscribe_trades(self, symbols: list[str]) -> str:
        """Subscribe to trade updates."""
        return await self.subscribe(StreamType.TRADES, symbols)

    async def subscribe_bars(self, symbols: list[str]) -> str:
        """Subscribe to bar updates."""
        return await self.subscribe(StreamType.BARS, symbols)

    async def _set_state(self, state: WebSocketState) -> None:
        """Set connection state and notify callbacks."""
        if state == self._state:
            return

        old_state = self._state
        self._state = state

        logger.debug(f"WebSocket state: {old_state.value} -> {state.value}")

        for callback in self._state_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(state)
                else:
                    callback(state)
            except Exception as e:
                logger.error(f"Error in state callback: {e}")

    async def _notify_error(self, error: Exception) -> None:
        """Notify error callbacks."""
        for callback in self._error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(error)
                else:
                    callback(error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")

    def get_statistics(self) -> dict:
        """Get WebSocket statistics."""
        return {
            "state": self._state.value,
            "is_connected": self.is_connected,
            "messages_received": self._messages_received,
            "messages_processed": self._messages_processed,
            "queue_size": self._message_queue.qsize(),
            "subscription_count": len(self.subscriptions),
            "reconnect_attempts": self._reconnect_attempts,
            "last_message_time": (
                self._last_message_time.isoformat()
                if self._last_message_time else None
            ),
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"WebSocketClient(state={self._state.value})"


class AlpacaWebSocketClient(WebSocketClient):
    """WebSocket client for Alpaca streaming data."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        feed: str = "iex",
        use_paper: bool = True,
    ) -> None:
        """
        Initialize AlpacaWebSocketClient.

        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            feed: Data feed (iex or sip)
            use_paper: Use paper trading environment
        """
        if use_paper:
            url = f"wss://stream.data.alpaca.markets/v2/{feed}"
        else:
            url = f"wss://stream.data.alpaca.markets/v2/{feed}"

        config = WebSocketConfig(
            url=url,
            api_key=api_key,
            api_secret=api_secret,
        )

        super().__init__(config)

        self._feed = feed

    async def _authenticate(self) -> bool:
        """Authenticate with Alpaca."""
        await self._set_state(WebSocketState.AUTHENTICATING)

        try:
            auth_message = {
                "action": "auth",
                "key": self._config.api_key,
                "secret": self._config.api_secret,
            }

            await self._send(auth_message)

            await self._set_state(WebSocketState.AUTHENTICATED)
            logger.info("Alpaca WebSocket authenticated")
            return True

        except Exception as e:
            await self._set_state(WebSocketState.ERROR)
            logger.error(f"Alpaca authentication error: {e}")
            return False
