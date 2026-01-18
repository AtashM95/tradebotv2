"""
AI Analyzer Module for Ultimate Trading Bot v2.2.

This module provides AI-powered market analysis functionality
using OpenAI GPT models for trading insights.
"""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.ai.openai_client import (
    OpenAIClient,
    OpenAIModel,
    ChatMessage,
    MessageRole,
)
from src.ai.prompt_manager import PromptManager, PromptCategory
from src.utils.exceptions import AIAnalysisError
from src.utils.helpers import generate_uuid
from src.utils.date_utils import now_utc


logger = logging.getLogger(__name__)


class AnalysisType(str, Enum):
    """Analysis type enumeration."""

    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    NEWS = "news"
    SIGNAL = "signal"
    RISK = "risk"
    MARKET = "market"
    COMPREHENSIVE = "comprehensive"


class SentimentResult(BaseModel):
    """Sentiment analysis result model."""

    symbol: str
    sentiment: str = Field(default="neutral")
    score: float = Field(default=0.0, ge=-1.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    key_factors: list[str] = Field(default_factory=list)
    summary: str = Field(default="")
    timestamp: datetime = Field(default_factory=now_utc)


class TechnicalResult(BaseModel):
    """Technical analysis result model."""

    symbol: str
    trend: str = Field(default="neutral")
    trend_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    key_signals: list[str] = Field(default_factory=list)
    support_levels: list[float] = Field(default_factory=list)
    resistance_levels: list[float] = Field(default_factory=list)
    recommendation: str = Field(default="hold")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    analysis_summary: str = Field(default="")
    timestamp: datetime = Field(default_factory=now_utc)


class SignalResult(BaseModel):
    """Trading signal result model."""

    signal_id: str = Field(default_factory=generate_uuid)
    symbol: str
    signal: str = Field(default="hold")
    strength: float = Field(default=0.0, ge=0.0, le=1.0)
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: list[float] = Field(default_factory=list)
    risk_reward_ratio: Optional[float] = None
    timeframe: str = Field(default="")
    reasoning: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=now_utc)

    @property
    def is_actionable(self) -> bool:
        """Check if signal is actionable."""
        return (
            self.signal in ("buy", "sell")
            and self.confidence >= 0.6
            and self.entry_price is not None
        )


class RiskResult(BaseModel):
    """Risk assessment result model."""

    symbol: str
    risk_level: str = Field(default="medium")
    risk_score: float = Field(default=0.5, ge=0.0, le=1.0)
    risk_factors: list[dict] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    suggested_stop_loss: Optional[float] = None
    max_position_size: Optional[int] = None
    summary: str = Field(default="")
    timestamp: datetime = Field(default_factory=now_utc)


class MarketOverviewResult(BaseModel):
    """Market overview result model."""

    market_sentiment: str = Field(default="neutral")
    risk_level: str = Field(default="medium")
    key_observations: list[str] = Field(default_factory=list)
    sector_leaders: list[str] = Field(default_factory=list)
    sector_laggards: list[str] = Field(default_factory=list)
    opportunities: list[str] = Field(default_factory=list)
    risks_to_watch: list[str] = Field(default_factory=list)
    outlook: str = Field(default="")
    trading_recommendations: str = Field(default="")
    timestamp: datetime = Field(default_factory=now_utc)


class AnalysisRequest(BaseModel):
    """Analysis request model."""

    request_id: str = Field(default_factory=generate_uuid)
    analysis_type: AnalysisType
    symbol: Optional[str] = None
    data: dict = Field(default_factory=dict)
    model: Optional[OpenAIModel] = None
    created_at: datetime = Field(default_factory=now_utc)


class AIAnalyzer:
    """
    AI-powered market analyzer.

    Provides functionality for:
    - Sentiment analysis
    - Technical analysis interpretation
    - Trading signal generation
    - Risk assessment
    - Market overview
    """

    def __init__(
        self,
        openai_client: Optional[OpenAIClient] = None,
        prompt_manager: Optional[PromptManager] = None,
        default_model: OpenAIModel = OpenAIModel.GPT_4O,
    ) -> None:
        """
        Initialize AIAnalyzer.

        Args:
            openai_client: OpenAI client instance
            prompt_manager: Prompt manager instance
            default_model: Default model to use
        """
        self._client = openai_client
        self._prompt_manager = prompt_manager or PromptManager()
        self._default_model = default_model

        self._analysis_count = 0
        self._error_count = 0

        self._cache: dict[str, tuple[Any, datetime]] = {}
        self._cache_ttl_seconds = 300

        logger.info("AIAnalyzer initialized")

    def set_client(self, client: OpenAIClient) -> None:
        """Set the OpenAI client."""
        self._client = client

    async def analyze_sentiment(
        self,
        symbol: str,
        text: str,
        model: Optional[OpenAIModel] = None,
    ) -> SentimentResult:
        """
        Analyze sentiment of text.

        Args:
            symbol: Trading symbol
            text: Text to analyze
            model: Model to use

        Returns:
            Sentiment analysis result
        """
        if not self._client:
            raise AIAnalysisError("OpenAI client not configured")

        try:
            prompt, system_prompt = self._prompt_manager.render_prompt(
                "sentiment_analysis",
                symbol=symbol,
                text=text,
            )

            response = await self._client.analyze_json(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model or self._default_model,
            )

            self._analysis_count += 1

            return SentimentResult(
                symbol=symbol,
                sentiment=response.get("sentiment", "neutral"),
                score=float(response.get("score", 0.0)),
                confidence=float(response.get("confidence", 0.0)),
                key_factors=response.get("key_factors", []),
                summary=response.get("summary", ""),
            )

        except Exception as e:
            self._error_count += 1
            logger.error(f"Sentiment analysis error: {e}")
            return SentimentResult(symbol=symbol)

    async def analyze_news_sentiment(
        self,
        symbol: str,
        headlines: list[str],
        model: Optional[OpenAIModel] = None,
    ) -> SentimentResult:
        """
        Analyze sentiment of news headlines.

        Args:
            symbol: Trading symbol
            headlines: News headlines
            model: Model to use

        Returns:
            Sentiment analysis result
        """
        if not self._client:
            raise AIAnalysisError("OpenAI client not configured")

        try:
            headlines_text = self._prompt_manager.format_headlines(headlines)

            prompt, system_prompt = self._prompt_manager.render_prompt(
                "news_sentiment",
                symbol=symbol,
                headlines=headlines_text,
            )

            response = await self._client.analyze_json(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model or self._default_model,
            )

            self._analysis_count += 1

            return SentimentResult(
                symbol=symbol,
                sentiment=response.get("overall_sentiment", "neutral"),
                score=float(response.get("sentiment_score", 0.0)),
                confidence=0.8,
                key_factors=response.get("key_themes", []),
                summary=response.get("trading_implication", ""),
            )

        except Exception as e:
            self._error_count += 1
            logger.error(f"News sentiment analysis error: {e}")
            return SentimentResult(symbol=symbol)

    async def analyze_technical(
        self,
        symbol: str,
        price: float,
        change: float,
        volume: int,
        indicators: dict[str, float],
        high_52w: float,
        low_52w: float,
        support: float,
        resistance: float,
        model: Optional[OpenAIModel] = None,
    ) -> TechnicalResult:
        """
        Perform technical analysis.

        Args:
            symbol: Trading symbol
            price: Current price
            change: Daily change percent
            volume: Trading volume
            indicators: Technical indicator values
            high_52w: 52-week high
            low_52w: 52-week low
            support: Support level
            resistance: Resistance level
            model: Model to use

        Returns:
            Technical analysis result
        """
        if not self._client:
            raise AIAnalysisError("OpenAI client not configured")

        try:
            indicators_text = self._prompt_manager.format_indicators(indicators)

            prompt, system_prompt = self._prompt_manager.render_prompt(
                "technical_analysis",
                symbol=symbol,
                price=f"{price:.2f}",
                change=f"{change:.2f}",
                volume=f"{volume:,}",
                indicators=indicators_text,
                high_52w=f"{high_52w:.2f}",
                low_52w=f"{low_52w:.2f}",
                support=f"{support:.2f}",
                resistance=f"{resistance:.2f}",
            )

            response = await self._client.analyze_json(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model or self._default_model,
            )

            self._analysis_count += 1

            return TechnicalResult(
                symbol=symbol,
                trend=response.get("trend", "neutral"),
                trend_strength=float(response.get("trend_strength", 0.0)),
                key_signals=response.get("key_signals", []),
                support_levels=[float(x) for x in response.get("support_levels", [])],
                resistance_levels=[float(x) for x in response.get("resistance_levels", [])],
                recommendation=response.get("recommendation", "hold"),
                confidence=float(response.get("confidence", 0.0)),
                analysis_summary=response.get("analysis_summary", ""),
            )

        except Exception as e:
            self._error_count += 1
            logger.error(f"Technical analysis error: {e}")
            return TechnicalResult(symbol=symbol)

    async def generate_signal(
        self,
        symbol: str,
        price: float,
        bid: float,
        ask: float,
        volume: int,
        indicators: dict[str, float],
        price_action: list[dict],
        market_context: str = "",
        model: Optional[OpenAIModel] = None,
    ) -> SignalResult:
        """
        Generate trading signal.

        Args:
            symbol: Trading symbol
            price: Current price
            bid: Bid price
            ask: Ask price
            volume: Trading volume
            indicators: Technical indicator values
            price_action: Recent price bars
            market_context: Market context description
            model: Model to use

        Returns:
            Trading signal result
        """
        if not self._client:
            raise AIAnalysisError("OpenAI client not configured")

        try:
            indicators_text = self._prompt_manager.format_indicators(indicators)
            price_action_text = self._prompt_manager.format_price_action(price_action)

            prompt, system_prompt = self._prompt_manager.render_prompt(
                "trading_signal",
                symbol=symbol,
                price=f"{price:.2f}",
                bid=f"{bid:.2f}",
                ask=f"{ask:.2f}",
                volume=f"{volume:,}",
                indicators=indicators_text,
                price_action=price_action_text,
                market_context=market_context or "Normal market conditions",
            )

            response = await self._client.analyze_json(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model or self._default_model,
            )

            self._analysis_count += 1

            take_profit = response.get("take_profit", [])
            if isinstance(take_profit, (int, float)):
                take_profit = [take_profit]

            return SignalResult(
                symbol=symbol,
                signal=response.get("signal", "hold"),
                strength=float(response.get("strength", 0.0)),
                entry_price=response.get("entry_price"),
                stop_loss=response.get("stop_loss"),
                take_profit=[float(x) for x in take_profit],
                risk_reward_ratio=response.get("risk_reward_ratio"),
                timeframe=response.get("timeframe", ""),
                reasoning=response.get("reasoning", []),
                risks=response.get("risks", []),
                confidence=float(response.get("confidence", 0.0)),
            )

        except Exception as e:
            self._error_count += 1
            logger.error(f"Signal generation error: {e}")
            return SignalResult(symbol=symbol)

    async def assess_risk(
        self,
        symbol: str,
        quantity: int,
        entry_price: float,
        current_price: float,
        portfolio_value: float,
        other_positions: list[dict],
        market_conditions: str = "",
        model: Optional[OpenAIModel] = None,
    ) -> RiskResult:
        """
        Assess position risk.

        Args:
            symbol: Trading symbol
            quantity: Position quantity
            entry_price: Entry price
            current_price: Current price
            portfolio_value: Total portfolio value
            other_positions: Other open positions
            market_conditions: Market conditions description
            model: Model to use

        Returns:
            Risk assessment result
        """
        if not self._client:
            raise AIAnalysisError("OpenAI client not configured")

        try:
            pnl = (current_price - entry_price) * quantity
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            position_value = current_price * quantity
            position_weight = (position_value / portfolio_value) * 100

            positions_text = self._prompt_manager.format_positions(other_positions)

            prompt, system_prompt = self._prompt_manager.render_prompt(
                "risk_assessment",
                symbol=symbol,
                quantity=str(quantity),
                entry_price=f"{entry_price:.2f}",
                current_price=f"{current_price:.2f}",
                pnl=f"{pnl:.2f}",
                pnl_pct=f"{pnl_pct:.2f}",
                portfolio_value=f"{portfolio_value:.2f}",
                position_weight=f"{position_weight:.1f}",
                other_positions=positions_text,
                market_conditions=market_conditions or "Normal market conditions",
            )

            response = await self._client.analyze_json(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model or self._default_model,
            )

            self._analysis_count += 1

            return RiskResult(
                symbol=symbol,
                risk_level=response.get("risk_level", "medium"),
                risk_score=float(response.get("risk_score", 0.5)),
                risk_factors=response.get("risk_factors", []),
                recommendations=response.get("recommendations", []),
                suggested_stop_loss=response.get("suggested_stop_loss"),
                max_position_size=response.get("max_position_size"),
                summary=response.get("summary", ""),
            )

        except Exception as e:
            self._error_count += 1
            logger.error(f"Risk assessment error: {e}")
            return RiskResult(symbol=symbol)

    async def get_market_overview(
        self,
        indices: dict[str, float],
        sectors: dict[str, float],
        advancing: int,
        declining: int,
        unchanged: int,
        vix: float,
        news: list[str],
        model: Optional[OpenAIModel] = None,
    ) -> MarketOverviewResult:
        """
        Get market overview analysis.

        Args:
            indices: Index values and changes
            sectors: Sector performance
            advancing: Number of advancing stocks
            declining: Number of declining stocks
            unchanged: Number of unchanged stocks
            vix: VIX value
            news: Recent news headlines
            model: Model to use

        Returns:
            Market overview result
        """
        if not self._client:
            raise AIAnalysisError("OpenAI client not configured")

        try:
            indices_text = "\n".join([
                f"- {name}: {value:+.2f}%"
                for name, value in indices.items()
            ])

            sectors_text = "\n".join([
                f"- {name}: {value:+.2f}%"
                for name, value in sectors.items()
            ])

            news_text = self._prompt_manager.format_headlines(news)

            prompt, system_prompt = self._prompt_manager.render_prompt(
                "market_overview",
                indices=indices_text,
                sectors=sectors_text,
                advancing=str(advancing),
                declining=str(declining),
                unchanged=str(unchanged),
                vix=f"{vix:.2f}",
                date=now_utc().strftime("%Y-%m-%d"),
                news=news_text,
            )

            response = await self._client.analyze_json(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model or self._default_model,
            )

            self._analysis_count += 1

            return MarketOverviewResult(
                market_sentiment=response.get("market_sentiment", "neutral"),
                risk_level=response.get("risk_level", "medium"),
                key_observations=response.get("key_observations", []),
                sector_leaders=response.get("sector_leaders", []),
                sector_laggards=response.get("sector_laggards", []),
                opportunities=response.get("opportunities", []),
                risks_to_watch=response.get("risks_to_watch", []),
                outlook=response.get("outlook", ""),
                trading_recommendations=response.get("trading_recommendations", ""),
            )

        except Exception as e:
            self._error_count += 1
            logger.error(f"Market overview error: {e}")
            return MarketOverviewResult()

    async def quick_sentiment(
        self,
        symbol: str,
        text: str,
    ) -> tuple[str, float]:
        """
        Quick sentiment analysis.

        Args:
            symbol: Trading symbol
            text: Text to analyze

        Returns:
            Tuple of (sentiment, score)
        """
        result = await self.analyze_sentiment(symbol, text)
        return result.sentiment, result.score

    def get_statistics(self) -> dict:
        """Get analyzer statistics."""
        return {
            "total_analyses": self._analysis_count,
            "error_count": self._error_count,
            "error_rate": (
                self._error_count / self._analysis_count * 100
                if self._analysis_count > 0 else 0
            ),
            "cache_size": len(self._cache),
            "client_connected": self._client is not None,
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"AIAnalyzer(analyses={self._analysis_count}, errors={self._error_count})"
