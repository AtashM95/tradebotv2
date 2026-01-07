"""
AI Signal Generator Module for Ultimate Trading Bot v2.2.

This module provides AI-powered trading signal generation
using market data and technical analysis.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from src.ai.openai_client import OpenAIClient, OpenAIModel
from src.ai.ai_analyzer import AIAnalyzer, SignalResult, SentimentResult
from src.utils.exceptions import AIAnalysisError
from src.utils.helpers import generate_uuid
from src.utils.date_utils import now_utc


logger = logging.getLogger(__name__)


class SignalStrength(str, Enum):
    """Signal strength enumeration."""

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


class SignalDirection(str, Enum):
    """Signal direction enumeration."""

    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class SignalSource(str, Enum):
    """Signal source enumeration."""

    AI = "ai"
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    COMBINED = "combined"


class AISignalConfig(BaseModel):
    """Configuration for AI signal generator."""

    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    min_signal_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    require_sentiment_confirmation: bool = Field(default=True)
    sentiment_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    technical_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    cooldown_minutes: int = Field(default=15, ge=1, le=60)
    max_signals_per_hour: int = Field(default=10, ge=1, le=100)
    enable_risk_check: bool = Field(default=True)


class AISignal(BaseModel):
    """AI-generated trading signal model."""

    signal_id: str = Field(default_factory=generate_uuid)
    symbol: str
    direction: SignalDirection = Field(default=SignalDirection.NEUTRAL)
    strength: SignalStrength = Field(default=SignalStrength.WEAK)
    source: SignalSource = Field(default=SignalSource.AI)

    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None
    take_profit_3: Optional[float] = None

    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    technical_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    combined_score: float = Field(default=0.0, ge=-1.0, le=1.0)

    risk_reward_ratio: Optional[float] = None
    position_size_pct: Optional[float] = None

    reasoning: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    generated_at: datetime = Field(default_factory=now_utc)
    valid_until: Optional[datetime] = None
    executed: bool = Field(default=False)
    executed_at: Optional[datetime] = None

    @property
    def is_valid(self) -> bool:
        """Check if signal is still valid."""
        if self.valid_until is None:
            return True
        return now_utc() < self.valid_until

    @property
    def is_actionable(self) -> bool:
        """Check if signal is actionable."""
        return (
            self.direction != SignalDirection.NEUTRAL
            and self.confidence >= 0.6
            and self.strength != SignalStrength.WEAK
            and self.is_valid
            and not self.executed
        )

    @property
    def is_long(self) -> bool:
        """Check if signal is long."""
        return self.direction == SignalDirection.LONG

    @property
    def is_short(self) -> bool:
        """Check if signal is short."""
        return self.direction == SignalDirection.SHORT

    def mark_executed(self) -> None:
        """Mark signal as executed."""
        self.executed = True
        self.executed_at = now_utc()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "strength": self.strength.value,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit_1": self.take_profit_1,
            "confidence": self.confidence,
            "combined_score": self.combined_score,
            "reasoning": self.reasoning,
            "generated_at": self.generated_at.isoformat(),
        }


class AISignalGenerator:
    """
    AI-powered trading signal generator.

    Generates trading signals by:
    - Analyzing technical indicators with AI
    - Incorporating sentiment analysis
    - Combining multiple signal sources
    - Risk-adjusted position sizing
    """

    def __init__(
        self,
        config: Optional[AISignalConfig] = None,
        analyzer: Optional[AIAnalyzer] = None,
    ) -> None:
        """
        Initialize AISignalGenerator.

        Args:
            config: Generator configuration
            analyzer: AI analyzer instance
        """
        self._config = config or AISignalConfig()
        self._analyzer = analyzer

        self._signal_history: list[AISignal] = []
        self._last_signal_time: dict[str, datetime] = {}

        self._signal_callbacks: list[Callable[[AISignal], None]] = []

        self._signals_generated = 0
        self._signals_executed = 0

        logger.info("AISignalGenerator initialized")

    def set_analyzer(self, analyzer: AIAnalyzer) -> None:
        """Set the AI analyzer."""
        self._analyzer = analyzer

    def on_signal(self, callback: Callable[[AISignal], None]) -> None:
        """Register callback for new signals."""
        self._signal_callbacks.append(callback)

    async def generate_signal(
        self,
        symbol: str,
        current_price: float,
        indicators: dict[str, float],
        price_action: list[dict],
        sentiment_text: Optional[str] = None,
        market_context: str = "",
        model: Optional[OpenAIModel] = None,
    ) -> Optional[AISignal]:
        """
        Generate trading signal for a symbol.

        Args:
            symbol: Trading symbol
            current_price: Current price
            indicators: Technical indicators
            price_action: Recent price bars
            sentiment_text: Text for sentiment analysis
            market_context: Market context description
            model: AI model to use

        Returns:
            Generated signal or None
        """
        if not self._analyzer:
            raise AIAnalysisError("AI analyzer not configured")

        if not self._check_cooldown(symbol):
            logger.debug(f"Signal cooldown active for {symbol}")
            return None

        if not self._check_signal_limit():
            logger.warning("Hourly signal limit reached")
            return None

        try:
            ai_signal_result = await self._analyzer.generate_signal(
                symbol=symbol,
                price=current_price,
                bid=current_price * 0.999,
                ask=current_price * 1.001,
                volume=indicators.get("volume", 0),
                indicators=indicators,
                price_action=price_action,
                market_context=market_context,
                model=model,
            )

            sentiment_result = None
            if self._config.require_sentiment_confirmation and sentiment_text:
                sentiment_result = await self._analyzer.analyze_sentiment(
                    symbol=symbol,
                    text=sentiment_text,
                    model=model,
                )

            signal = self._combine_signals(
                symbol=symbol,
                current_price=current_price,
                ai_signal=ai_signal_result,
                sentiment=sentiment_result,
            )

            if signal and signal.is_actionable:
                self._signal_history.append(signal)
                self._last_signal_time[symbol] = now_utc()
                self._signals_generated += 1

                await self._notify_signal(signal)

                logger.info(
                    f"Generated {signal.direction.value} signal for {symbol} "
                    f"(confidence={signal.confidence:.2f})"
                )

            return signal

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None

    def _combine_signals(
        self,
        symbol: str,
        current_price: float,
        ai_signal: SignalResult,
        sentiment: Optional[SentimentResult] = None,
    ) -> AISignal:
        """Combine AI signal and sentiment into final signal."""
        technical_score = 0.0
        if ai_signal.signal == "buy":
            technical_score = ai_signal.strength
        elif ai_signal.signal == "sell":
            technical_score = -ai_signal.strength

        sentiment_score = 0.0
        if sentiment:
            sentiment_score = sentiment.score

        combined_score = (
            technical_score * self._config.technical_weight
            + sentiment_score * self._config.sentiment_weight
        )

        if combined_score > 0.3:
            direction = SignalDirection.LONG
        elif combined_score < -0.3:
            direction = SignalDirection.SHORT
        else:
            direction = SignalDirection.NEUTRAL

        abs_score = abs(combined_score)
        if abs_score >= 0.7:
            strength = SignalStrength.STRONG
        elif abs_score >= 0.4:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        confidence = ai_signal.confidence
        if sentiment:
            confidence = (confidence + sentiment.confidence) / 2

        stop_loss = None
        take_profit_1 = None
        take_profit_2 = None
        take_profit_3 = None

        if direction == SignalDirection.LONG:
            stop_loss = ai_signal.stop_loss or current_price * 0.97
            take_profit_1 = current_price * 1.02
            take_profit_2 = current_price * 1.04
            take_profit_3 = current_price * 1.06
        elif direction == SignalDirection.SHORT:
            stop_loss = ai_signal.stop_loss or current_price * 1.03
            take_profit_1 = current_price * 0.98
            take_profit_2 = current_price * 0.96
            take_profit_3 = current_price * 0.94

        risk_reward = None
        if stop_loss and take_profit_1:
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit_1 - current_price)
            if risk > 0:
                risk_reward = round(reward / risk, 2)

        reasoning = ai_signal.reasoning.copy()
        if sentiment:
            reasoning.append(f"Sentiment: {sentiment.sentiment} (score: {sentiment.score:.2f})")

        valid_until = now_utc() + timedelta(hours=4)

        return AISignal(
            symbol=symbol,
            direction=direction,
            strength=strength,
            source=SignalSource.COMBINED if sentiment else SignalSource.AI,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=take_profit_3,
            confidence=confidence,
            technical_score=technical_score,
            sentiment_score=sentiment_score,
            combined_score=combined_score,
            risk_reward_ratio=risk_reward,
            reasoning=reasoning,
            risks=ai_signal.risks,
            valid_until=valid_until,
        )

    def _check_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown period."""
        if symbol not in self._last_signal_time:
            return True

        last_time = self._last_signal_time[symbol]
        cooldown = timedelta(minutes=self._config.cooldown_minutes)

        return now_utc() >= last_time + cooldown

    def _check_signal_limit(self) -> bool:
        """Check if under hourly signal limit."""
        one_hour_ago = now_utc() - timedelta(hours=1)
        recent_signals = [
            s for s in self._signal_history
            if s.generated_at > one_hour_ago
        ]
        return len(recent_signals) < self._config.max_signals_per_hour

    async def _notify_signal(self, signal: AISignal) -> None:
        """Notify callbacks of new signal."""
        for callback in self._signal_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(signal)
                else:
                    callback(signal)
            except Exception as e:
                logger.error(f"Error in signal callback: {e}")

    def get_signal(self, signal_id: str) -> Optional[AISignal]:
        """Get signal by ID."""
        for signal in self._signal_history:
            if signal.signal_id == signal_id:
                return signal
        return None

    def get_latest_signal(self, symbol: str) -> Optional[AISignal]:
        """Get latest signal for a symbol."""
        for signal in reversed(self._signal_history):
            if signal.symbol == symbol:
                return signal
        return None

    def get_active_signals(self) -> list[AISignal]:
        """Get all active (valid, unexecuted) signals."""
        return [
            s for s in self._signal_history
            if s.is_valid and not s.executed
        ]

    def get_signals_for_symbol(self, symbol: str) -> list[AISignal]:
        """Get all signals for a symbol."""
        return [s for s in self._signal_history if s.symbol == symbol]

    def mark_signal_executed(self, signal_id: str) -> bool:
        """Mark a signal as executed."""
        signal = self.get_signal(signal_id)
        if signal:
            signal.mark_executed()
            self._signals_executed += 1
            return True
        return False

    def invalidate_signals(self, symbol: str) -> int:
        """Invalidate all signals for a symbol."""
        count = 0
        for signal in self._signal_history:
            if signal.symbol == symbol and not signal.executed:
                signal.valid_until = now_utc()
                count += 1
        return count

    def clear_old_signals(self, hours: int = 24) -> int:
        """Clear signals older than specified hours."""
        cutoff = now_utc() - timedelta(hours=hours)
        old_count = len(self._signal_history)
        self._signal_history = [
            s for s in self._signal_history
            if s.generated_at > cutoff
        ]
        return old_count - len(self._signal_history)

    def get_signal_summary(self) -> dict:
        """Get signal summary statistics."""
        active = self.get_active_signals()
        long_signals = [s for s in active if s.is_long]
        short_signals = [s for s in active if s.is_short]

        return {
            "total_generated": self._signals_generated,
            "total_executed": self._signals_executed,
            "execution_rate": (
                self._signals_executed / self._signals_generated * 100
                if self._signals_generated > 0 else 0
            ),
            "active_signals": len(active),
            "long_signals": len(long_signals),
            "short_signals": len(short_signals),
            "history_size": len(self._signal_history),
        }

    def get_statistics(self) -> dict:
        """Get generator statistics."""
        summary = self.get_signal_summary()

        recent_signals = [
            s for s in self._signal_history
            if s.generated_at > now_utc() - timedelta(hours=24)
        ]

        avg_confidence = 0.0
        if recent_signals:
            avg_confidence = sum(s.confidence for s in recent_signals) / len(recent_signals)

        summary.update({
            "signals_last_24h": len(recent_signals),
            "avg_confidence": avg_confidence,
            "config": {
                "min_confidence": self._config.min_confidence,
                "cooldown_minutes": self._config.cooldown_minutes,
                "max_signals_per_hour": self._config.max_signals_per_hour,
            },
        })

        return summary

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"AISignalGenerator(generated={self._signals_generated}, "
            f"executed={self._signals_executed})"
        )
