"""
Chat Assistant Module for Ultimate Trading Bot v2.2.

This module provides an AI-powered chat assistant for
trading queries, market analysis, and portfolio guidance.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from src.ai.openai_client import (
    OpenAIClient,
    OpenAIModel,
    ChatMessage,
    MessageRole,
    ChatCompletion,
)
from src.utils.helpers import generate_uuid
from src.utils.date_utils import now_utc


logger = logging.getLogger(__name__)


class AssistantRole(str, Enum):
    """Assistant role enumeration."""

    TRADING_ADVISOR = "trading_advisor"
    MARKET_ANALYST = "market_analyst"
    RISK_MANAGER = "risk_manager"
    PORTFOLIO_MANAGER = "portfolio_manager"
    GENERAL = "general"


ROLE_SYSTEM_PROMPTS = {
    AssistantRole.TRADING_ADVISOR: """You are an expert trading advisor for a professional trading bot system.
Your role is to:
- Provide actionable trading insights and recommendations
- Explain market conditions and their impact on trading
- Help with entry/exit timing and position sizing
- Analyze technical setups and chart patterns
- Answer questions about trading strategies

Be concise, data-driven, and always consider risk management.
Avoid generic advice - be specific based on the context provided.""",

    AssistantRole.MARKET_ANALYST: """You are a market analyst specializing in equity markets.
Your role is to:
- Analyze market trends and sector movements
- Identify potential opportunities and risks
- Explain economic indicators and their market impact
- Provide macro and micro market perspectives
- Summarize relevant news and events

Focus on factual analysis rather than predictions. Cite specific data when available.""",

    AssistantRole.RISK_MANAGER: """You are a risk management specialist for trading portfolios.
Your role is to:
- Assess portfolio risk exposure
- Recommend position sizing and hedging strategies
- Identify concentration and correlation risks
- Suggest stop-loss and take-profit levels
- Monitor risk metrics and alert thresholds

Always prioritize capital preservation and proper risk management.""",

    AssistantRole.PORTFOLIO_MANAGER: """You are a portfolio management advisor.
Your role is to:
- Analyze portfolio composition and performance
- Suggest rebalancing strategies
- Optimize asset allocation
- Track benchmark comparisons
- Provide portfolio analytics and insights

Consider diversification, correlation, and return objectives in your analysis.""",

    AssistantRole.GENERAL: """You are a helpful trading assistant for a professional trading bot.
You can help with:
- General trading questions
- Market information
- Bot configuration and usage
- Performance analysis
- Strategy explanations

Be helpful, accurate, and concise in your responses.""",
}


class ConversationMessage(BaseModel):
    """Conversation message model."""

    message_id: str = Field(default_factory=generate_uuid)
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=now_utc)
    metadata: dict = Field(default_factory=dict)


class Conversation(BaseModel):
    """Conversation session model."""

    conversation_id: str = Field(default_factory=generate_uuid)
    role: AssistantRole = Field(default=AssistantRole.GENERAL)
    messages: list[ConversationMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=now_utc)
    last_activity: datetime = Field(default_factory=now_utc)
    context: dict = Field(default_factory=dict)

    @property
    def message_count(self) -> int:
        """Get message count."""
        return len(self.messages)

    @property
    def is_active(self) -> bool:
        """Check if conversation is active (within 30 minutes)."""
        return (now_utc() - self.last_activity).total_seconds() < 1800

    def add_message(self, role: MessageRole, content: str) -> ConversationMessage:
        """Add a message to the conversation."""
        message = ConversationMessage(role=role, content=content)
        self.messages.append(message)
        self.last_activity = now_utc()
        return message

    def get_chat_messages(self, limit: int = 20) -> list[ChatMessage]:
        """Get messages in chat format."""
        recent = self.messages[-limit:] if limit else self.messages
        return [
            ChatMessage(role=msg.role, content=msg.content)
            for msg in recent
        ]

    def clear_messages(self) -> int:
        """Clear all messages."""
        count = len(self.messages)
        self.messages.clear()
        return count


class ChatAssistantConfig(BaseModel):
    """Configuration for chat assistant."""

    default_role: AssistantRole = Field(default=AssistantRole.GENERAL)
    max_conversation_messages: int = Field(default=20, ge=5, le=100)
    max_response_tokens: int = Field(default=1000, ge=100, le=4000)
    temperature: float = Field(default=0.7, ge=0.0, le=1.5)
    enable_context_injection: bool = Field(default=True)
    conversation_timeout_minutes: int = Field(default=30, ge=5, le=120)


class ChatAssistant:
    """
    AI-powered chat assistant for trading.

    Provides functionality for:
    - Interactive trading Q&A
    - Market analysis queries
    - Portfolio guidance
    - Strategy explanations
    - Multi-turn conversations
    """

    def __init__(
        self,
        config: Optional[ChatAssistantConfig] = None,
        openai_client: Optional[OpenAIClient] = None,
    ) -> None:
        """
        Initialize ChatAssistant.

        Args:
            config: Assistant configuration
            openai_client: OpenAI client instance
        """
        self._config = config or ChatAssistantConfig()
        self._client = openai_client

        self._conversations: dict[str, Conversation] = {}
        self._active_conversation_id: Optional[str] = None

        self._total_messages = 0
        self._total_conversations = 0

        logger.info("ChatAssistant initialized")

    def set_client(self, client: OpenAIClient) -> None:
        """Set the OpenAI client."""
        self._client = client

    def create_conversation(
        self,
        role: Optional[AssistantRole] = None,
        context: Optional[dict] = None,
    ) -> Conversation:
        """
        Create a new conversation.

        Args:
            role: Assistant role
            context: Initial context

        Returns:
            New conversation
        """
        conversation = Conversation(
            role=role or self._config.default_role,
            context=context or {},
        )

        self._conversations[conversation.conversation_id] = conversation
        self._active_conversation_id = conversation.conversation_id
        self._total_conversations += 1

        logger.info(f"Created conversation: {conversation.conversation_id}")
        return conversation

    def get_conversation(
        self,
        conversation_id: Optional[str] = None
    ) -> Optional[Conversation]:
        """Get a conversation by ID or active conversation."""
        conv_id = conversation_id or self._active_conversation_id
        if conv_id:
            return self._conversations.get(conv_id)
        return None

    def get_or_create_conversation(
        self,
        conversation_id: Optional[str] = None,
        role: Optional[AssistantRole] = None,
    ) -> Conversation:
        """Get existing or create new conversation."""
        if conversation_id:
            conv = self._conversations.get(conversation_id)
            if conv:
                return conv

        return self.create_conversation(role=role)

    async def chat(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        role: Optional[AssistantRole] = None,
        context: Optional[dict] = None,
        model: Optional[OpenAIModel] = None,
    ) -> str:
        """
        Send a chat message and get response.

        Args:
            message: User message
            conversation_id: Conversation ID
            role: Assistant role override
            context: Additional context
            model: Model to use

        Returns:
            Assistant response
        """
        if not self._client:
            return "Error: OpenAI client not configured"

        conversation = self.get_or_create_conversation(conversation_id, role)

        if context:
            conversation.context.update(context)

        conversation.add_message(MessageRole.USER, message)

        system_prompt = ROLE_SYSTEM_PROMPTS.get(
            role or conversation.role,
            ROLE_SYSTEM_PROMPTS[AssistantRole.GENERAL]
        )

        if self._config.enable_context_injection and conversation.context:
            context_str = self._format_context(conversation.context)
            system_prompt += f"\n\nCurrent Context:\n{context_str}"

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
        ]
        messages.extend(
            conversation.get_chat_messages(self._config.max_conversation_messages)
        )

        try:
            completion = await self._client.chat_completion(
                messages=messages,
                model=model or OpenAIModel.GPT_4O,
                temperature=self._config.temperature,
                max_tokens=self._config.max_response_tokens,
            )

            response = completion.content
            conversation.add_message(MessageRole.ASSISTANT, response)
            self._total_messages += 2

            return response

        except Exception as e:
            logger.error(f"Chat error: {e}")
            error_msg = f"I apologize, but I encountered an error: {str(e)}"
            conversation.add_message(MessageRole.ASSISTANT, error_msg)
            return error_msg

    async def ask(
        self,
        question: str,
        role: Optional[AssistantRole] = None,
        context: Optional[dict] = None,
    ) -> str:
        """
        Ask a single question (no conversation history).

        Args:
            question: Question to ask
            role: Assistant role
            context: Question context

        Returns:
            Answer
        """
        if not self._client:
            return "Error: OpenAI client not configured"

        system_prompt = ROLE_SYSTEM_PROMPTS.get(
            role or self._config.default_role,
            ROLE_SYSTEM_PROMPTS[AssistantRole.GENERAL]
        )

        if context:
            context_str = self._format_context(context)
            system_prompt += f"\n\nContext:\n{context_str}"

        try:
            response = await self._client.simple_chat(
                prompt=question,
                system_prompt=system_prompt,
                model=OpenAIModel.GPT_4O,
            )
            self._total_messages += 2
            return response

        except Exception as e:
            logger.error(f"Ask error: {e}")
            return f"Error: {str(e)}"

    async def analyze_symbol(
        self,
        symbol: str,
        data: dict,
    ) -> str:
        """
        Get analysis for a symbol.

        Args:
            symbol: Trading symbol
            data: Symbol data (price, indicators, etc.)

        Returns:
            Analysis text
        """
        context = {
            "symbol": symbol,
            **data,
        }

        question = f"""Provide a brief trading analysis for {symbol} based on the following data:

Price: ${data.get('price', 'N/A')}
Daily Change: {data.get('change', 'N/A')}%
Volume: {data.get('volume', 'N/A')}

Technical Indicators:
{self._format_indicators(data.get('indicators', {}))}

What are the key points to consider for trading this symbol?"""

        return await self.ask(
            question=question,
            role=AssistantRole.TRADING_ADVISOR,
            context=context,
        )

    async def explain_signal(
        self,
        signal: dict,
    ) -> str:
        """
        Explain a trading signal.

        Args:
            signal: Signal data

        Returns:
            Explanation text
        """
        question = f"""Explain this trading signal in simple terms:

Symbol: {signal.get('symbol')}
Direction: {signal.get('direction')}
Entry: ${signal.get('entry_price')}
Stop Loss: ${signal.get('stop_loss')}
Take Profit: ${signal.get('take_profit')}
Confidence: {signal.get('confidence', 0) * 100:.0f}%

Reasoning: {', '.join(signal.get('reasoning', []))}

Please explain:
1. What this signal means
2. The risk/reward profile
3. What could go wrong
4. How to manage this position"""

        return await self.ask(
            question=question,
            role=AssistantRole.TRADING_ADVISOR,
        )

    async def get_market_briefing(
        self,
        market_data: dict,
    ) -> str:
        """
        Get a market briefing.

        Args:
            market_data: Market data (indices, sectors, etc.)

        Returns:
            Market briefing text
        """
        question = f"""Provide a brief market overview based on:

Indices:
{self._format_dict(market_data.get('indices', {}))}

Sector Performance:
{self._format_dict(market_data.get('sectors', {}))}

VIX: {market_data.get('vix', 'N/A')}

What are the key takeaways for traders today?"""

        return await self.ask(
            question=question,
            role=AssistantRole.MARKET_ANALYST,
        )

    def _format_context(self, context: dict) -> str:
        """Format context for prompt."""
        lines = []
        for key, value in context.items():
            if isinstance(value, dict):
                lines.append(f"{key}:")
                for k, v in value.items():
                    lines.append(f"  - {k}: {v}")
            elif isinstance(value, list):
                lines.append(f"{key}: {', '.join(str(v) for v in value)}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def _format_indicators(self, indicators: dict) -> str:
        """Format indicators for prompt."""
        if not indicators:
            return "No indicators available"
        return "\n".join([
            f"- {name}: {value:.2f}" if isinstance(value, float) else f"- {name}: {value}"
            for name, value in indicators.items()
        ])

    def _format_dict(self, data: dict) -> str:
        """Format dictionary for prompt."""
        if not data:
            return "No data available"
        return "\n".join([
            f"- {key}: {value:+.2f}%" if isinstance(value, float) else f"- {key}: {value}"
            for key, value in data.items()
        ])

    def end_conversation(self, conversation_id: str) -> bool:
        """End and remove a conversation."""
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            if self._active_conversation_id == conversation_id:
                self._active_conversation_id = None
            return True
        return False

    def cleanup_inactive_conversations(self) -> int:
        """Clean up inactive conversations."""
        timeout = timedelta(minutes=self._config.conversation_timeout_minutes)
        cutoff = now_utc() - timeout

        to_remove = [
            conv_id for conv_id, conv in self._conversations.items()
            if conv.last_activity < cutoff
        ]

        for conv_id in to_remove:
            del self._conversations[conv_id]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} inactive conversations")

        return len(to_remove)

    def get_statistics(self) -> dict:
        """Get assistant statistics."""
        active_convs = [c for c in self._conversations.values() if c.is_active]

        return {
            "total_conversations": self._total_conversations,
            "active_conversations": len(active_convs),
            "total_messages": self._total_messages,
            "conversations_in_memory": len(self._conversations),
            "default_role": self._config.default_role.value,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ChatAssistant(conversations={len(self._conversations)}, "
            f"messages={self._total_messages})"
        )
