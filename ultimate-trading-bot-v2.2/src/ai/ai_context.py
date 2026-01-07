"""
AI Context Manager Module for Ultimate Trading Bot v2.2.

This module manages conversation context, memory, and state
for AI interactions in trading scenarios.
"""

import asyncio
import logging
from collections import deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.utils.helpers import generate_uuid
from src.utils.date_utils import now_utc


logger = logging.getLogger(__name__)


class ContextType(str, Enum):
    """Context type enumeration."""

    CONVERSATION = "conversation"
    MARKET_DATA = "market_data"
    TRADING_STATE = "trading_state"
    USER_PREFERENCES = "user_preferences"
    ANALYSIS_HISTORY = "analysis_history"
    SIGNAL_HISTORY = "signal_history"


class MessageRole(str, Enum):
    """Message role enumeration."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class ContextMessage(BaseModel):
    """Context message model."""

    message_id: str = Field(default_factory=generate_uuid)
    role: MessageRole
    content: str
    name: Optional[str] = None
    timestamp: datetime = Field(default_factory=now_utc)
    metadata: dict = Field(default_factory=dict)
    token_count: int = Field(default=0)


class ConversationContext(BaseModel):
    """Conversation context model."""

    context_id: str = Field(default_factory=generate_uuid)
    messages: list[ContextMessage] = Field(default_factory=list)
    system_prompt: Optional[str] = None
    total_tokens: int = Field(default=0)
    created_at: datetime = Field(default_factory=now_utc)
    last_activity: datetime = Field(default_factory=now_utc)
    metadata: dict = Field(default_factory=dict)


class MarketContext(BaseModel):
    """Market context for AI analysis."""

    symbol: Optional[str] = None
    current_price: Optional[float] = None
    price_change_pct: Optional[float] = None
    volume: Optional[int] = None
    vix: Optional[float] = None
    market_trend: Optional[str] = None
    sector_performance: dict[str, float] = Field(default_factory=dict)
    recent_news: list[str] = Field(default_factory=list)
    indicators: dict[str, float] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=now_utc)


class TradingContext(BaseModel):
    """Trading context for AI decisions."""

    account_value: float = Field(default=0.0)
    buying_power: float = Field(default=0.0)
    open_positions: list[dict] = Field(default_factory=list)
    pending_orders: list[dict] = Field(default_factory=list)
    daily_pnl: float = Field(default=0.0)
    realized_pnl: float = Field(default=0.0)
    unrealized_pnl: float = Field(default=0.0)
    risk_exposure: float = Field(default=0.0)
    session_stats: dict = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=now_utc)


class UserContext(BaseModel):
    """User preferences context."""

    user_id: str = Field(default="default")
    risk_tolerance: str = Field(default="moderate")
    preferred_strategies: list[str] = Field(default_factory=list)
    watchlist: list[str] = Field(default_factory=list)
    notification_preferences: dict = Field(default_factory=dict)
    timezone: str = Field(default="UTC")
    experience_level: str = Field(default="intermediate")


class AIContextConfig(BaseModel):
    """Configuration for AI context manager."""

    max_conversation_messages: int = Field(default=50, ge=5, le=200)
    max_context_tokens: int = Field(default=8000, ge=1000, le=128000)
    context_ttl_minutes: int = Field(default=60, ge=5, le=1440)
    enable_summarization: bool = Field(default=True)
    summarize_at_token_count: int = Field(default=6000, ge=2000, le=100000)
    include_market_context: bool = Field(default=True)
    include_trading_context: bool = Field(default=True)


class AIContextManager:
    """
    AI Context Manager for maintaining conversation state.

    Provides:
    - Conversation history management
    - Market context injection
    - Trading state context
    - Token budget management
    - Context summarization
    """

    def __init__(
        self,
        config: Optional[AIContextConfig] = None,
    ) -> None:
        """
        Initialize AIContextManager.

        Args:
            config: Context manager configuration
        """
        self._config = config or AIContextConfig()

        self._conversations: dict[str, ConversationContext] = {}
        self._market_context: MarketContext = MarketContext()
        self._trading_context: TradingContext = TradingContext()
        self._user_context: UserContext = UserContext()

        self._context_summaries: dict[str, str] = {}
        self._analysis_history: deque[dict] = deque(maxlen=100)
        self._signal_history: deque[dict] = deque(maxlen=100)

        logger.info("AIContextManager initialized")

    def create_conversation(
        self,
        system_prompt: Optional[str] = None,
        context_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Create a new conversation context.

        Args:
            system_prompt: System prompt for the conversation
            context_id: Optional custom context ID
            metadata: Optional metadata

        Returns:
            Conversation context ID
        """
        context_id = context_id or generate_uuid()

        context = ConversationContext(
            context_id=context_id,
            system_prompt=system_prompt,
            metadata=metadata or {},
        )

        if system_prompt:
            system_message = ContextMessage(
                role=MessageRole.SYSTEM,
                content=system_prompt,
                token_count=self._estimate_tokens(system_prompt),
            )
            context.messages.append(system_message)
            context.total_tokens = system_message.token_count

        self._conversations[context_id] = context
        logger.debug(f"Created conversation context: {context_id}")

        return context_id

    def add_message(
        self,
        context_id: str,
        role: MessageRole,
        content: str,
        name: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> ContextMessage:
        """
        Add a message to a conversation.

        Args:
            context_id: Conversation context ID
            role: Message role
            content: Message content
            name: Optional function name
            metadata: Optional metadata

        Returns:
            Created ContextMessage
        """
        context = self._conversations.get(context_id)
        if not context:
            context_id = self.create_conversation()
            context = self._conversations[context_id]

        token_count = self._estimate_tokens(content)

        message = ContextMessage(
            role=role,
            content=content,
            name=name,
            metadata=metadata or {},
            token_count=token_count,
        )

        context.messages.append(message)
        context.total_tokens += token_count
        context.last_activity = now_utc()

        if len(context.messages) > self._config.max_conversation_messages:
            self._trim_context(context)

        if (
            self._config.enable_summarization and
            context.total_tokens > self._config.summarize_at_token_count
        ):
            self._summarize_context(context)

        return message

    def get_messages(
        self,
        context_id: str,
        include_system: bool = True,
        max_messages: Optional[int] = None,
    ) -> list[dict]:
        """
        Get messages for API call.

        Args:
            context_id: Conversation context ID
            include_system: Include system message
            max_messages: Maximum messages to return

        Returns:
            List of message dictionaries for API
        """
        context = self._conversations.get(context_id)
        if not context:
            return []

        messages: list[dict] = []

        for msg in context.messages:
            if msg.role == MessageRole.SYSTEM and not include_system:
                continue

            message_dict = {
                "role": msg.role.value,
                "content": msg.content,
            }

            if msg.name:
                message_dict["name"] = msg.name

            messages.append(message_dict)

        if max_messages and len(messages) > max_messages:
            system_messages = [m for m in messages if m["role"] == "system"]
            other_messages = [m for m in messages if m["role"] != "system"]
            messages = system_messages + other_messages[-(max_messages - len(system_messages)):]

        return messages

    def get_context_with_enrichment(
        self,
        context_id: str,
        include_market: bool = True,
        include_trading: bool = True,
        include_user: bool = True,
    ) -> list[dict]:
        """
        Get messages with context enrichment.

        Args:
            context_id: Conversation context ID
            include_market: Include market context
            include_trading: Include trading context
            include_user: Include user context

        Returns:
            Enriched message list
        """
        messages = self.get_messages(context_id)

        context_parts: list[str] = []

        if include_market and self._config.include_market_context:
            market_str = self._format_market_context()
            if market_str:
                context_parts.append(f"Market Context:\n{market_str}")

        if include_trading and self._config.include_trading_context:
            trading_str = self._format_trading_context()
            if trading_str:
                context_parts.append(f"Trading Context:\n{trading_str}")

        if include_user:
            user_str = self._format_user_context()
            if user_str:
                context_parts.append(f"User Preferences:\n{user_str}")

        if context_parts:
            context_message = {
                "role": "system",
                "content": "\n\n".join(context_parts),
            }

            if messages and messages[0]["role"] == "system":
                messages.insert(1, context_message)
            else:
                messages.insert(0, context_message)

        return messages

    def update_market_context(
        self,
        symbol: Optional[str] = None,
        current_price: Optional[float] = None,
        price_change_pct: Optional[float] = None,
        volume: Optional[int] = None,
        vix: Optional[float] = None,
        market_trend: Optional[str] = None,
        sector_performance: Optional[dict[str, float]] = None,
        recent_news: Optional[list[str]] = None,
        indicators: Optional[dict[str, float]] = None,
    ) -> None:
        """Update market context with new data."""
        if symbol is not None:
            self._market_context.symbol = symbol
        if current_price is not None:
            self._market_context.current_price = current_price
        if price_change_pct is not None:
            self._market_context.price_change_pct = price_change_pct
        if volume is not None:
            self._market_context.volume = volume
        if vix is not None:
            self._market_context.vix = vix
        if market_trend is not None:
            self._market_context.market_trend = market_trend
        if sector_performance is not None:
            self._market_context.sector_performance = sector_performance
        if recent_news is not None:
            self._market_context.recent_news = recent_news
        if indicators is not None:
            self._market_context.indicators = indicators

        self._market_context.updated_at = now_utc()

    def update_trading_context(
        self,
        account_value: Optional[float] = None,
        buying_power: Optional[float] = None,
        open_positions: Optional[list[dict]] = None,
        pending_orders: Optional[list[dict]] = None,
        daily_pnl: Optional[float] = None,
        realized_pnl: Optional[float] = None,
        unrealized_pnl: Optional[float] = None,
        risk_exposure: Optional[float] = None,
        session_stats: Optional[dict] = None,
    ) -> None:
        """Update trading context with new data."""
        if account_value is not None:
            self._trading_context.account_value = account_value
        if buying_power is not None:
            self._trading_context.buying_power = buying_power
        if open_positions is not None:
            self._trading_context.open_positions = open_positions
        if pending_orders is not None:
            self._trading_context.pending_orders = pending_orders
        if daily_pnl is not None:
            self._trading_context.daily_pnl = daily_pnl
        if realized_pnl is not None:
            self._trading_context.realized_pnl = realized_pnl
        if unrealized_pnl is not None:
            self._trading_context.unrealized_pnl = unrealized_pnl
        if risk_exposure is not None:
            self._trading_context.risk_exposure = risk_exposure
        if session_stats is not None:
            self._trading_context.session_stats = session_stats

        self._trading_context.updated_at = now_utc()

    def update_user_context(
        self,
        user_id: Optional[str] = None,
        risk_tolerance: Optional[str] = None,
        preferred_strategies: Optional[list[str]] = None,
        watchlist: Optional[list[str]] = None,
        notification_preferences: Optional[dict] = None,
        timezone: Optional[str] = None,
        experience_level: Optional[str] = None,
    ) -> None:
        """Update user context with preferences."""
        if user_id is not None:
            self._user_context.user_id = user_id
        if risk_tolerance is not None:
            self._user_context.risk_tolerance = risk_tolerance
        if preferred_strategies is not None:
            self._user_context.preferred_strategies = preferred_strategies
        if watchlist is not None:
            self._user_context.watchlist = watchlist
        if notification_preferences is not None:
            self._user_context.notification_preferences = notification_preferences
        if timezone is not None:
            self._user_context.timezone = timezone
        if experience_level is not None:
            self._user_context.experience_level = experience_level

    def add_analysis_history(
        self,
        symbol: str,
        analysis_type: str,
        result: dict,
    ) -> None:
        """Add an analysis to history."""
        entry = {
            "symbol": symbol,
            "type": analysis_type,
            "result": result,
            "timestamp": now_utc().isoformat(),
        }
        self._analysis_history.append(entry)

    def add_signal_history(
        self,
        symbol: str,
        signal_type: str,
        direction: str,
        confidence: float,
        result: Optional[str] = None,
    ) -> None:
        """Add a signal to history."""
        entry = {
            "symbol": symbol,
            "signal_type": signal_type,
            "direction": direction,
            "confidence": confidence,
            "result": result,
            "timestamp": now_utc().isoformat(),
        }
        self._signal_history.append(entry)

    def get_recent_analyses(
        self,
        symbol: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict]:
        """Get recent analyses from history."""
        analyses = list(self._analysis_history)

        if symbol:
            analyses = [a for a in analyses if a.get("symbol") == symbol]

        return analyses[-limit:]

    def get_recent_signals(
        self,
        symbol: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict]:
        """Get recent signals from history."""
        signals = list(self._signal_history)

        if symbol:
            signals = [s for s in signals if s.get("symbol") == symbol]

        return signals[-limit:]

    def _trim_context(self, context: ConversationContext) -> None:
        """Trim context to max messages."""
        system_messages = [
            m for m in context.messages
            if m.role == MessageRole.SYSTEM
        ]
        other_messages = [
            m for m in context.messages
            if m.role != MessageRole.SYSTEM
        ]

        max_others = self._config.max_conversation_messages - len(system_messages)
        trimmed_others = other_messages[-max_others:]

        context.messages = system_messages + trimmed_others
        context.total_tokens = sum(m.token_count for m in context.messages)

    def _summarize_context(self, context: ConversationContext) -> None:
        """Summarize older messages to reduce token count."""
        if len(context.messages) < 10:
            return

        system_messages = [
            m for m in context.messages
            if m.role == MessageRole.SYSTEM
        ]

        other_messages = [
            m for m in context.messages
            if m.role != MessageRole.SYSTEM
        ]

        if len(other_messages) < 6:
            return

        to_summarize = other_messages[:-4]
        to_keep = other_messages[-4:]

        summary_parts: list[str] = []
        for msg in to_summarize:
            if msg.role == MessageRole.USER:
                summary_parts.append(f"User asked: {msg.content[:100]}...")
            elif msg.role == MessageRole.ASSISTANT:
                summary_parts.append(f"Assistant: {msg.content[:100]}...")

        summary_text = "Previous conversation summary:\n" + "\n".join(summary_parts[-5:])

        summary_message = ContextMessage(
            role=MessageRole.SYSTEM,
            content=summary_text,
            token_count=self._estimate_tokens(summary_text),
            metadata={"is_summary": True},
        )

        context.messages = system_messages + [summary_message] + to_keep
        context.total_tokens = sum(m.token_count for m in context.messages)

        self._context_summaries[context.context_id] = summary_text
        logger.debug(f"Summarized context {context.context_id}")

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(text) // 4 + 1

    def _format_market_context(self) -> str:
        """Format market context as string."""
        ctx = self._market_context
        parts: list[str] = []

        if ctx.symbol:
            parts.append(f"Symbol: {ctx.symbol}")
        if ctx.current_price:
            parts.append(f"Price: ${ctx.current_price:.2f}")
        if ctx.price_change_pct is not None:
            parts.append(f"Change: {ctx.price_change_pct:+.2f}%")
        if ctx.vix:
            parts.append(f"VIX: {ctx.vix:.1f}")
        if ctx.market_trend:
            parts.append(f"Trend: {ctx.market_trend}")
        if ctx.indicators:
            ind_str = ", ".join([f"{k}: {v:.2f}" for k, v in list(ctx.indicators.items())[:5]])
            parts.append(f"Indicators: {ind_str}")

        return "; ".join(parts)

    def _format_trading_context(self) -> str:
        """Format trading context as string."""
        ctx = self._trading_context
        parts: list[str] = []

        if ctx.account_value:
            parts.append(f"Account: ${ctx.account_value:,.0f}")
        if ctx.buying_power:
            parts.append(f"Buying Power: ${ctx.buying_power:,.0f}")
        if ctx.daily_pnl:
            parts.append(f"Daily P&L: ${ctx.daily_pnl:+,.2f}")
        if ctx.open_positions:
            parts.append(f"Open Positions: {len(ctx.open_positions)}")
        if ctx.risk_exposure:
            parts.append(f"Risk Exposure: {ctx.risk_exposure:.1f}%")

        return "; ".join(parts)

    def _format_user_context(self) -> str:
        """Format user context as string."""
        ctx = self._user_context
        parts: list[str] = []

        if ctx.risk_tolerance:
            parts.append(f"Risk Tolerance: {ctx.risk_tolerance}")
        if ctx.experience_level:
            parts.append(f"Experience: {ctx.experience_level}")
        if ctx.preferred_strategies:
            parts.append(f"Strategies: {', '.join(ctx.preferred_strategies[:3])}")

        return "; ".join(parts)

    def clear_conversation(self, context_id: str) -> bool:
        """Clear a conversation context."""
        if context_id in self._conversations:
            del self._conversations[context_id]
            if context_id in self._context_summaries:
                del self._context_summaries[context_id]
            logger.debug(f"Cleared conversation {context_id}")
            return True
        return False

    def cleanup_expired_contexts(self) -> int:
        """Clean up expired conversation contexts."""
        now = now_utc()
        ttl = timedelta(minutes=self._config.context_ttl_minutes)
        expired: list[str] = []

        for context_id, context in self._conversations.items():
            if now - context.last_activity > ttl:
                expired.append(context_id)

        for context_id in expired:
            self.clear_conversation(context_id)

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired contexts")

        return len(expired)

    def get_statistics(self) -> dict:
        """Get context manager statistics."""
        total_messages = sum(
            len(c.messages) for c in self._conversations.values()
        )
        total_tokens = sum(
            c.total_tokens for c in self._conversations.values()
        )

        return {
            "active_conversations": len(self._conversations),
            "total_messages": total_messages,
            "total_tokens": total_tokens,
            "analysis_history_count": len(self._analysis_history),
            "signal_history_count": len(self._signal_history),
            "summaries_created": len(self._context_summaries),
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"AIContextManager(conversations={len(self._conversations)})"
