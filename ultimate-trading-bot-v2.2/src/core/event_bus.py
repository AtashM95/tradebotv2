"""
Event Bus Module for Ultimate Trading Bot v2.2.

This module provides a publish-subscribe event system for decoupled
communication between components of the trading bot.
"""

import asyncio
import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    Any, Awaitable, Callable, Dict, List, Optional, Set, Type, TypeVar, Union
)
import logging
import threading
import uuid


logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event type enumeration."""

    # System events
    SYSTEM_START = auto()
    SYSTEM_STOP = auto()
    SYSTEM_ERROR = auto()
    SYSTEM_WARNING = auto()
    HEARTBEAT = auto()

    # Market events
    MARKET_OPEN = auto()
    MARKET_CLOSE = auto()
    MARKET_PRE_OPEN = auto()
    MARKET_AFTER_HOURS = auto()

    # Data events
    QUOTE_UPDATE = auto()
    BAR_UPDATE = auto()
    TRADE_UPDATE = auto()
    NEWS_UPDATE = auto()
    DATA_ERROR = auto()

    # Order events
    ORDER_SUBMITTED = auto()
    ORDER_ACCEPTED = auto()
    ORDER_FILLED = auto()
    ORDER_PARTIALLY_FILLED = auto()
    ORDER_CANCELLED = auto()
    ORDER_REJECTED = auto()
    ORDER_EXPIRED = auto()
    ORDER_REPLACED = auto()
    ORDER_ERROR = auto()

    # Position events
    POSITION_OPENED = auto()
    POSITION_CLOSED = auto()
    POSITION_UPDATED = auto()
    POSITION_STOP_TRIGGERED = auto()
    POSITION_TARGET_REACHED = auto()

    # Signal events
    SIGNAL_GENERATED = auto()
    SIGNAL_CONFIRMED = auto()
    SIGNAL_CANCELLED = auto()

    # Strategy events
    STRATEGY_STARTED = auto()
    STRATEGY_STOPPED = auto()
    STRATEGY_ERROR = auto()
    STRATEGY_SIGNAL = auto()

    # Risk events
    RISK_LIMIT_WARNING = auto()
    RISK_LIMIT_BREACH = auto()
    DRAWDOWN_WARNING = auto()
    DRAWDOWN_BREACH = auto()

    # AI events
    AI_ANALYSIS_COMPLETE = auto()
    AI_SIGNAL_GENERATED = auto()
    AI_ERROR = auto()
    AI_BUDGET_WARNING = auto()

    # Notification events
    NOTIFICATION_SENT = auto()
    NOTIFICATION_FAILED = auto()

    # Backtest events
    BACKTEST_STARTED = auto()
    BACKTEST_COMPLETED = auto()
    BACKTEST_ERROR = auto()

    # Custom events
    CUSTOM = auto()


class EventPriority(Enum):
    """Event priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Event:
    """
    Event class representing a single event.

    Attributes:
        event_type: Type of the event
        data: Event payload data
        source: Source component of the event
        timestamp: When the event was created
        event_id: Unique event identifier
        priority: Event priority level
        metadata: Additional event metadata
    """

    event_type: EventType
    data: Any = None
    source: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: EventPriority = EventPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate event after initialization."""
        if not isinstance(self.event_type, EventType):
            raise ValueError(f"Invalid event type: {self.event_type}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.name,
            "data": self.data,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.name,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary."""
        return cls(
            event_type=EventType[data["event_type"]],
            data=data.get("data"),
            source=data.get("source", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            event_id=data["event_id"],
            priority=EventPriority[data.get("priority", "NORMAL")],
            metadata=data.get("metadata", {}),
        )


# Type alias for event handlers
SyncHandler = Callable[[Event], None]
AsyncHandler = Callable[[Event], Awaitable[None]]
EventHandler = Union[SyncHandler, AsyncHandler]


@dataclass
class Subscription:
    """Subscription information for an event handler."""

    handler: EventHandler
    event_types: Set[EventType]
    priority: EventPriority
    is_async: bool
    subscriber_id: str
    filter_func: Optional[Callable[[Event], bool]] = None
    once: bool = False
    active: bool = True


class EventBus:
    """
    Central event bus for publish-subscribe pattern.

    Supports both synchronous and asynchronous event handling,
    prioritized delivery, and event filtering.
    """

    _instance: Optional['EventBus'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'EventBus':
        """Singleton pattern for event bus."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the event bus."""
        if self._initialized:
            return

        self._subscriptions: Dict[EventType, List[Subscription]] = defaultdict(list)
        self._all_subscriptions: List[Subscription] = []
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._event_history: List[Event] = []
        self._history_limit: int = 1000
        self._running: bool = False
        self._processing_task: Optional[asyncio.Task] = None
        self._sync_lock = threading.RLock()
        self._async_lock: Optional[asyncio.Lock] = None
        self._stats: Dict[str, int] = defaultdict(int)
        self._initialized = True

        logger.info("EventBus initialized")

    async def start(self) -> None:
        """Start the event bus processing loop."""
        if self._running:
            return

        self._running = True
        self._async_lock = asyncio.Lock()
        self._processing_task = asyncio.create_task(self._process_events())
        logger.info("EventBus started")

    async def stop(self) -> None:
        """Stop the event bus processing loop."""
        if not self._running:
            return

        self._running = False

        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        logger.info("EventBus stopped")

    def subscribe(
        self,
        event_types: Union[EventType, List[EventType]],
        handler: EventHandler,
        priority: EventPriority = EventPriority.NORMAL,
        filter_func: Optional[Callable[[Event], bool]] = None,
        once: bool = False
    ) -> str:
        """
        Subscribe to events.

        Args:
            event_types: Event type(s) to subscribe to
            handler: Handler function (sync or async)
            priority: Handler priority
            filter_func: Optional filter function
            once: If True, handler will be called only once

        Returns:
            Subscriber ID
        """
        if isinstance(event_types, EventType):
            event_types = [event_types]

        # Detect if handler is async
        is_async = asyncio.iscoroutinefunction(handler)

        subscriber_id = str(uuid.uuid4())

        subscription = Subscription(
            handler=handler,
            event_types=set(event_types),
            priority=priority,
            is_async=is_async,
            subscriber_id=subscriber_id,
            filter_func=filter_func,
            once=once,
        )

        with self._sync_lock:
            for event_type in event_types:
                self._subscriptions[event_type].append(subscription)
                # Sort by priority (higher priority first)
                self._subscriptions[event_type].sort(
                    key=lambda s: s.priority.value, reverse=True
                )

            self._all_subscriptions.append(subscription)

        logger.debug(f"Subscribed {subscriber_id} to {event_types}")
        return subscriber_id

    def subscribe_all(
        self,
        handler: EventHandler,
        priority: EventPriority = EventPriority.NORMAL,
        filter_func: Optional[Callable[[Event], bool]] = None
    ) -> str:
        """
        Subscribe to all events.

        Args:
            handler: Handler function
            priority: Handler priority
            filter_func: Optional filter function

        Returns:
            Subscriber ID
        """
        return self.subscribe(
            list(EventType),
            handler,
            priority=priority,
            filter_func=filter_func
        )

    def unsubscribe(self, subscriber_id: str) -> bool:
        """
        Unsubscribe a handler.

        Args:
            subscriber_id: Subscriber ID to remove

        Returns:
            True if found and removed
        """
        with self._sync_lock:
            removed = False

            # Remove from event-specific subscriptions
            for event_type in self._subscriptions:
                self._subscriptions[event_type] = [
                    s for s in self._subscriptions[event_type]
                    if s.subscriber_id != subscriber_id
                ]
                removed = True

            # Remove from all subscriptions
            self._all_subscriptions = [
                s for s in self._all_subscriptions
                if s.subscriber_id != subscriber_id
            ]

        if removed:
            logger.debug(f"Unsubscribed {subscriber_id}")

        return removed

    def publish(self, event: Event) -> None:
        """
        Publish an event synchronously.

        Args:
            event: Event to publish
        """
        self._stats["events_published"] += 1
        self._add_to_history(event)

        # Get handlers for this event type
        handlers = self._get_handlers(event)

        for subscription in handlers:
            if not subscription.active:
                continue

            if subscription.filter_func and not subscription.filter_func(event):
                continue

            try:
                if subscription.is_async:
                    # Queue async handlers
                    asyncio.create_task(self._call_async_handler(subscription, event))
                else:
                    subscription.handler(event)
                    self._stats["handlers_called"] += 1

                if subscription.once:
                    subscription.active = False

            except Exception as e:
                logger.error(f"Error in event handler: {e}")
                self._stats["handler_errors"] += 1

    async def publish_async(self, event: Event) -> None:
        """
        Publish an event asynchronously.

        Args:
            event: Event to publish
        """
        self._stats["events_published"] += 1
        self._add_to_history(event)

        # Get handlers for this event type
        handlers = self._get_handlers(event)

        tasks = []
        for subscription in handlers:
            if not subscription.active:
                continue

            if subscription.filter_func and not subscription.filter_func(event):
                continue

            try:
                if subscription.is_async:
                    tasks.append(self._call_async_handler(subscription, event))
                else:
                    subscription.handler(event)
                    self._stats["handlers_called"] += 1

                if subscription.once:
                    subscription.active = False

            except Exception as e:
                logger.error(f"Error in event handler: {e}")
                self._stats["handler_errors"] += 1

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def emit(
        self,
        event_type: EventType,
        data: Any = None,
        source: str = "",
        priority: EventPriority = EventPriority.NORMAL,
        **metadata: Any
    ) -> Event:
        """
        Convenience method to emit an event.

        Args:
            event_type: Type of event
            data: Event data
            source: Event source
            priority: Event priority
            **metadata: Additional metadata

        Returns:
            Created event
        """
        event = Event(
            event_type=event_type,
            data=data,
            source=source,
            priority=priority,
            metadata=metadata,
        )
        await self.publish_async(event)
        return event

    def emit_sync(
        self,
        event_type: EventType,
        data: Any = None,
        source: str = "",
        priority: EventPriority = EventPriority.NORMAL,
        **metadata: Any
    ) -> Event:
        """
        Convenience method to emit an event synchronously.

        Args:
            event_type: Type of event
            data: Event data
            source: Event source
            priority: Event priority
            **metadata: Additional metadata

        Returns:
            Created event
        """
        event = Event(
            event_type=event_type,
            data=data,
            source=source,
            priority=priority,
            metadata=metadata,
        )
        self.publish(event)
        return event

    def _get_handlers(self, event: Event) -> List[Subscription]:
        """Get handlers for an event."""
        with self._sync_lock:
            return list(self._subscriptions.get(event.event_type, []))

    async def _call_async_handler(
        self,
        subscription: Subscription,
        event: Event
    ) -> None:
        """Call an async handler."""
        try:
            await subscription.handler(event)
            self._stats["handlers_called"] += 1
        except Exception as e:
            logger.error(f"Error in async event handler: {e}")
            self._stats["handler_errors"] += 1

    async def _process_events(self) -> None:
        """Process events from the queue."""
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0
                )
                await self.publish_async(event)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing event: {e}")

    def _add_to_history(self, event: Event) -> None:
        """Add event to history."""
        with self._sync_lock:
            self._event_history.append(event)
            if len(self._event_history) > self._history_limit:
                self._event_history = self._event_history[-self._history_limit:]

    def get_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100
    ) -> List[Event]:
        """
        Get event history.

        Args:
            event_type: Filter by event type
            limit: Maximum events to return

        Returns:
            List of events
        """
        with self._sync_lock:
            events = self._event_history
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            return events[-limit:]

    def get_stats(self) -> Dict[str, int]:
        """Get event bus statistics."""
        return dict(self._stats)

    def clear_history(self) -> None:
        """Clear event history."""
        with self._sync_lock:
            self._event_history.clear()

    def clear_subscriptions(self) -> None:
        """Clear all subscriptions."""
        with self._sync_lock:
            self._subscriptions.clear()
            self._all_subscriptions.clear()


# Global event bus instance
event_bus = EventBus()


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    return event_bus


# Decorator for event handlers
def on_event(
    event_types: Union[EventType, List[EventType]],
    priority: EventPriority = EventPriority.NORMAL
):
    """
    Decorator to register a function as an event handler.

    Args:
        event_types: Event type(s) to handle
        priority: Handler priority

    Returns:
        Decorator function
    """
    def decorator(func: EventHandler) -> EventHandler:
        event_bus.subscribe(event_types, func, priority=priority)
        return func
    return decorator


# Convenience functions
async def emit(
    event_type: EventType,
    data: Any = None,
    source: str = "",
    **metadata: Any
) -> Event:
    """Emit an event to the global event bus."""
    return await event_bus.emit(event_type, data, source, **metadata)


def emit_sync(
    event_type: EventType,
    data: Any = None,
    source: str = "",
    **metadata: Any
) -> Event:
    """Emit an event synchronously to the global event bus."""
    return event_bus.emit_sync(event_type, data, source, **metadata)


def subscribe(
    event_types: Union[EventType, List[EventType]],
    handler: EventHandler,
    priority: EventPriority = EventPriority.NORMAL
) -> str:
    """Subscribe to events on the global event bus."""
    return event_bus.subscribe(event_types, handler, priority=priority)


def unsubscribe(subscriber_id: str) -> bool:
    """Unsubscribe from the global event bus."""
    return event_bus.unsubscribe(subscriber_id)
