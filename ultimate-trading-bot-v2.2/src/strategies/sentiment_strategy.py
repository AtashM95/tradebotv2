"""
Sentiment-Based Trading Strategy Module for Ultimate Trading Bot v2.2.

This module implements trading strategies based on market sentiment
analysis from news, social media, and other sources.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.strategies.base_strategy import (
    BaseStrategy,
    StrategyConfig,
    StrategySignal,
    SignalAction,
    SignalSide,
    MarketData,
    StrategyContext,
)
from src.analysis.technical_indicators import TechnicalIndicators
from src.utils.helpers import generate_uuid


logger = logging.getLogger(__name__)


class SentimentScore(BaseModel):
    """Model for sentiment score."""

    symbol: str
    timestamp: datetime
    news_sentiment: float = Field(default=0.0, ge=-1.0, le=1.0)
    social_sentiment: float = Field(default=0.0, ge=-1.0, le=1.0)
    analyst_sentiment: float = Field(default=0.0, ge=-1.0, le=1.0)
    overall_sentiment: float = Field(default=0.0, ge=-1.0, le=1.0)
    sentiment_momentum: float = Field(default=0.0)
    volume_sentiment: float = Field(default=0.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class SentimentAlert(BaseModel):
    """Model for sentiment alert."""

    alert_id: str = Field(default_factory=generate_uuid)
    symbol: str
    alert_type: str
    sentiment_change: float
    timestamp: datetime
    details: dict = Field(default_factory=dict)


class SentimentStrategyConfig(StrategyConfig):
    """Configuration for sentiment-based strategy."""

    name: str = Field(default="Sentiment Strategy")
    description: str = Field(
        default="Trade based on market sentiment analysis"
    )

    sentiment_threshold_buy: float = Field(default=0.3, ge=0.1, le=0.9)
    sentiment_threshold_sell: float = Field(default=-0.3, ge=-0.9, le=-0.1)

    news_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    social_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    analyst_weight: float = Field(default=0.30, ge=0.0, le=1.0)

    sentiment_lookback_hours: int = Field(default=24, ge=1, le=168)
    min_data_points: int = Field(default=5, ge=1, le=50)

    use_sentiment_momentum: bool = Field(default=True)
    momentum_threshold: float = Field(default=0.1, ge=0.01, le=0.5)

    contrarian_mode: bool = Field(default=False)
    extreme_sentiment_threshold: float = Field(default=0.8, ge=0.5, le=1.0)

    combine_with_technicals: bool = Field(default=True)
    technical_confirmation_required: bool = Field(default=False)

    news_decay_hours: float = Field(default=12.0, ge=1.0, le=72.0)
    social_decay_hours: float = Field(default=6.0, ge=1.0, le=48.0)


class SentimentStrategy(BaseStrategy):
    """
    Sentiment-based trading strategy.

    Features:
    - Multi-source sentiment aggregation
    - Sentiment momentum tracking
    - Contrarian mode for extreme sentiment
    - Technical confirmation integration
    - Time-decayed sentiment weighting
    """

    def __init__(
        self,
        config: Optional[SentimentStrategyConfig] = None,
    ) -> None:
        """
        Initialize SentimentStrategy.

        Args:
            config: Sentiment strategy configuration
        """
        config = config or SentimentStrategyConfig()
        super().__init__(config)

        self._sentiment_config = config
        self._indicators = TechnicalIndicators()

        self._sentiment_scores: dict[str, list[SentimentScore]] = {}
        self._alerts: list[SentimentAlert] = []
        self._external_sentiment: dict[str, SentimentScore] = {}

        logger.info(f"SentimentStrategy initialized: {self.name}")

    def calculate_indicators(
        self,
        symbol: str,
        data: MarketData,
    ) -> dict[str, Any]:
        """
        Calculate technical indicators for sentiment confirmation.

        Args:
            symbol: Trading symbol
            data: Market data

        Returns:
            Dictionary of indicator values
        """
        closes = data.closes
        highs = data.highs
        lows = data.lows
        volumes = data.volumes

        if len(closes) < 20:
            return {}

        current_price = closes[-1]

        rsi = self._indicators.rsi(closes, 14)
        current_rsi = rsi[-1] if rsi else 50.0

        ema_20 = self._indicators.ema(closes, 20)
        ema_50 = self._indicators.ema(closes, 50)

        trend = "neutral"
        if ema_20 and ema_50:
            if ema_20[-1] > ema_50[-1]:
                trend = "bullish"
            elif ema_20[-1] < ema_50[-1]:
                trend = "bearish"

        avg_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else volumes[-1]
        volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0

        momentum_5d = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0

        return {
            "current_price": current_price,
            "rsi": current_rsi,
            "trend": trend,
            "volume_ratio": volume_ratio,
            "momentum_5d": momentum_5d,
            "ema_20": ema_20[-1] if ema_20 else current_price,
            "ema_50": ema_50[-1] if ema_50 else current_price,
        }

    async def evaluate(
        self,
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """
        Evaluate sentiment-based opportunities.

        Args:
            market_data: Market data for all symbols
            context: Strategy execution context

        Returns:
            List of sentiment-based signals
        """
        signals: list[StrategySignal] = []

        for symbol in self.config.symbols:
            if symbol not in market_data:
                continue

            data = market_data[symbol]

            sentiment = self._get_current_sentiment(symbol, context)
            if not sentiment:
                continue

            indicators = {}
            if self._sentiment_config.combine_with_technicals:
                indicators = self.calculate_indicators(symbol, data)

            signal = self._generate_sentiment_signal(
                symbol, sentiment, indicators, data, context
            )

            if signal:
                signals.append(signal)

        return signals

    def update_sentiment(
        self,
        symbol: str,
        news_sentiment: Optional[float] = None,
        social_sentiment: Optional[float] = None,
        analyst_sentiment: Optional[float] = None,
        timestamp: Optional[datetime] = None,
        confidence: float = 0.7,
    ) -> None:
        """
        Update sentiment data for a symbol.

        Args:
            symbol: Trading symbol
            news_sentiment: News sentiment score (-1 to 1)
            social_sentiment: Social media sentiment (-1 to 1)
            analyst_sentiment: Analyst sentiment (-1 to 1)
            timestamp: Timestamp of sentiment data
            confidence: Confidence in sentiment scores
        """
        from src.utils.date_utils import now_utc

        ts = timestamp or now_utc()

        existing = self._external_sentiment.get(symbol)
        if existing:
            news = news_sentiment if news_sentiment is not None else existing.news_sentiment
            social = social_sentiment if social_sentiment is not None else existing.social_sentiment
            analyst = analyst_sentiment if analyst_sentiment is not None else existing.analyst_sentiment
        else:
            news = news_sentiment or 0.0
            social = social_sentiment or 0.0
            analyst = analyst_sentiment or 0.0

        overall = (
            news * self._sentiment_config.news_weight +
            social * self._sentiment_config.social_weight +
            analyst * self._sentiment_config.analyst_weight
        )

        if symbol not in self._sentiment_scores:
            self._sentiment_scores[symbol] = []

        history = self._sentiment_scores[symbol]
        sentiment_momentum = 0.0
        if len(history) >= 2:
            prev_overall = history[-1].overall_sentiment
            sentiment_momentum = overall - prev_overall

        score = SentimentScore(
            symbol=symbol,
            timestamp=ts,
            news_sentiment=news,
            social_sentiment=social,
            analyst_sentiment=analyst,
            overall_sentiment=overall,
            sentiment_momentum=sentiment_momentum,
            confidence=confidence,
        )

        self._sentiment_scores[symbol].append(score)
        self._external_sentiment[symbol] = score

        if len(self._sentiment_scores[symbol]) > 500:
            self._sentiment_scores[symbol] = self._sentiment_scores[symbol][-500:]

        if abs(sentiment_momentum) >= self._sentiment_config.momentum_threshold:
            alert = SentimentAlert(
                symbol=symbol,
                alert_type="momentum_shift",
                sentiment_change=sentiment_momentum,
                timestamp=ts,
                details={
                    "previous": history[-1].overall_sentiment if history else 0,
                    "current": overall,
                },
            )
            self._alerts.append(alert)

    def _get_current_sentiment(
        self,
        symbol: str,
        context: StrategyContext,
    ) -> Optional[SentimentScore]:
        """Get current aggregated sentiment for symbol."""
        if symbol in self._external_sentiment:
            score = self._external_sentiment[symbol]

            age_hours = (context.timestamp - score.timestamp).total_seconds() / 3600
            if age_hours > self._sentiment_config.sentiment_lookback_hours:
                return None

            return score

        if symbol not in self._sentiment_scores:
            return None

        history = self._sentiment_scores[symbol]
        if len(history) < self._sentiment_config.min_data_points:
            return None

        cutoff = context.timestamp - timedelta(
            hours=self._sentiment_config.sentiment_lookback_hours
        )

        recent = [s for s in history if s.timestamp >= cutoff]
        if len(recent) < self._sentiment_config.min_data_points:
            return None

        weighted_news = 0.0
        weighted_social = 0.0
        weighted_analyst = 0.0
        total_weight = 0.0

        for score in recent:
            age_hours = (context.timestamp - score.timestamp).total_seconds() / 3600

            news_decay = 0.5 ** (age_hours / self._sentiment_config.news_decay_hours)
            social_decay = 0.5 ** (age_hours / self._sentiment_config.social_decay_hours)

            weight = score.confidence * max(news_decay, social_decay)

            weighted_news += score.news_sentiment * news_decay * score.confidence
            weighted_social += score.social_sentiment * social_decay * score.confidence
            weighted_analyst += score.analyst_sentiment * score.confidence
            total_weight += weight

        if total_weight == 0:
            return None

        avg_news = weighted_news / total_weight
        avg_social = weighted_social / total_weight
        avg_analyst = weighted_analyst / total_weight

        overall = (
            avg_news * self._sentiment_config.news_weight +
            avg_social * self._sentiment_config.social_weight +
            avg_analyst * self._sentiment_config.analyst_weight
        )

        momentum = 0.0
        if len(recent) >= 2:
            early_overall = sum(s.overall_sentiment for s in recent[:len(recent)//2]) / (len(recent)//2)
            late_overall = sum(s.overall_sentiment for s in recent[len(recent)//2:]) / (len(recent) - len(recent)//2)
            momentum = late_overall - early_overall

        return SentimentScore(
            symbol=symbol,
            timestamp=context.timestamp,
            news_sentiment=avg_news,
            social_sentiment=avg_social,
            analyst_sentiment=avg_analyst,
            overall_sentiment=overall,
            sentiment_momentum=momentum,
            confidence=min(0.9, total_weight / len(recent)),
        )

    def _generate_sentiment_signal(
        self,
        symbol: str,
        sentiment: SentimentScore,
        indicators: dict[str, Any],
        data: MarketData,
        context: StrategyContext,
    ) -> Optional[StrategySignal]:
        """Generate trading signal based on sentiment."""
        overall = sentiment.overall_sentiment
        momentum = sentiment.sentiment_momentum

        is_extreme = abs(overall) >= self._sentiment_config.extreme_sentiment_threshold

        if self._sentiment_config.contrarian_mode and is_extreme:
            if overall > 0:
                return self._create_contrarian_sell_signal(
                    symbol, sentiment, indicators, data, context
                )
            else:
                return self._create_contrarian_buy_signal(
                    symbol, sentiment, indicators, data, context
                )

        if self._sentiment_config.combine_with_technicals and indicators:
            technical_signal = self._get_technical_signal(indicators)

            if self._sentiment_config.technical_confirmation_required:
                if overall > self._sentiment_config.sentiment_threshold_buy:
                    if technical_signal != "bullish":
                        return None
                elif overall < self._sentiment_config.sentiment_threshold_sell:
                    if technical_signal != "bearish":
                        return None

        if overall >= self._sentiment_config.sentiment_threshold_buy:
            return self._create_buy_signal(symbol, sentiment, indicators, data, context)

        elif overall <= self._sentiment_config.sentiment_threshold_sell:
            return self._create_sell_signal(symbol, sentiment, indicators, data, context)

        if self._sentiment_config.use_sentiment_momentum:
            if momentum >= self._sentiment_config.momentum_threshold:
                return self._create_momentum_buy_signal(
                    symbol, sentiment, indicators, data, context
                )
            elif momentum <= -self._sentiment_config.momentum_threshold:
                return self._create_momentum_sell_signal(
                    symbol, sentiment, indicators, data, context
                )

        return None

    def _get_technical_signal(self, indicators: dict[str, Any]) -> str:
        """Get technical confirmation signal."""
        trend = indicators.get("trend", "neutral")
        rsi = indicators.get("rsi", 50)
        volume_ratio = indicators.get("volume_ratio", 1.0)

        bullish_points = 0
        bearish_points = 0

        if trend == "bullish":
            bullish_points += 1
        elif trend == "bearish":
            bearish_points += 1

        if rsi < 30:
            bullish_points += 1
        elif rsi > 70:
            bearish_points += 1

        if volume_ratio > 1.5:
            if trend == "bullish":
                bullish_points += 1
            elif trend == "bearish":
                bearish_points += 1

        if bullish_points >= 2:
            return "bullish"
        elif bearish_points >= 2:
            return "bearish"
        return "neutral"

    def _create_buy_signal(
        self,
        symbol: str,
        sentiment: SentimentScore,
        indicators: dict[str, Any],
        data: MarketData,
        context: StrategyContext,
    ) -> Optional[StrategySignal]:
        """Create buy signal based on positive sentiment."""
        current_price = data.closes[-1]

        strength = min(1.0, abs(sentiment.overall_sentiment))
        confidence = sentiment.confidence * 0.9

        return self.create_signal(
            symbol=symbol,
            action=SignalAction.BUY,
            side=SignalSide.LONG,
            entry_price=current_price,
            strength=strength,
            confidence=confidence,
            reason=f"Bullish sentiment: {sentiment.overall_sentiment:.2f}",
            metadata={
                "strategy_type": "sentiment",
                "news_sentiment": sentiment.news_sentiment,
                "social_sentiment": sentiment.social_sentiment,
                "analyst_sentiment": sentiment.analyst_sentiment,
                "sentiment_momentum": sentiment.sentiment_momentum,
                "technical_trend": indicators.get("trend", "unknown"),
            },
        )

    def _create_sell_signal(
        self,
        symbol: str,
        sentiment: SentimentScore,
        indicators: dict[str, Any],
        data: MarketData,
        context: StrategyContext,
    ) -> Optional[StrategySignal]:
        """Create sell signal based on negative sentiment."""
        current_price = data.closes[-1]

        strength = min(1.0, abs(sentiment.overall_sentiment))
        confidence = sentiment.confidence * 0.9

        return self.create_signal(
            symbol=symbol,
            action=SignalAction.SELL,
            side=SignalSide.SHORT,
            entry_price=current_price,
            strength=strength,
            confidence=confidence,
            reason=f"Bearish sentiment: {sentiment.overall_sentiment:.2f}",
            metadata={
                "strategy_type": "sentiment",
                "news_sentiment": sentiment.news_sentiment,
                "social_sentiment": sentiment.social_sentiment,
                "analyst_sentiment": sentiment.analyst_sentiment,
                "sentiment_momentum": sentiment.sentiment_momentum,
                "technical_trend": indicators.get("trend", "unknown"),
            },
        )

    def _create_momentum_buy_signal(
        self,
        symbol: str,
        sentiment: SentimentScore,
        indicators: dict[str, Any],
        data: MarketData,
        context: StrategyContext,
    ) -> Optional[StrategySignal]:
        """Create buy signal based on sentiment momentum."""
        current_price = data.closes[-1]

        return self.create_signal(
            symbol=symbol,
            action=SignalAction.BUY,
            side=SignalSide.LONG,
            entry_price=current_price,
            strength=min(1.0, abs(sentiment.sentiment_momentum) * 2),
            confidence=sentiment.confidence * 0.75,
            reason=f"Sentiment momentum buy: {sentiment.sentiment_momentum:.2f}",
            metadata={
                "strategy_type": "sentiment",
                "signal_subtype": "momentum",
                "sentiment_momentum": sentiment.sentiment_momentum,
            },
        )

    def _create_momentum_sell_signal(
        self,
        symbol: str,
        sentiment: SentimentScore,
        indicators: dict[str, Any],
        data: MarketData,
        context: StrategyContext,
    ) -> Optional[StrategySignal]:
        """Create sell signal based on sentiment momentum."""
        current_price = data.closes[-1]

        return self.create_signal(
            symbol=symbol,
            action=SignalAction.SELL,
            side=SignalSide.SHORT,
            entry_price=current_price,
            strength=min(1.0, abs(sentiment.sentiment_momentum) * 2),
            confidence=sentiment.confidence * 0.75,
            reason=f"Sentiment momentum sell: {sentiment.sentiment_momentum:.2f}",
            metadata={
                "strategy_type": "sentiment",
                "signal_subtype": "momentum",
                "sentiment_momentum": sentiment.sentiment_momentum,
            },
        )

    def _create_contrarian_buy_signal(
        self,
        symbol: str,
        sentiment: SentimentScore,
        indicators: dict[str, Any],
        data: MarketData,
        context: StrategyContext,
    ) -> Optional[StrategySignal]:
        """Create contrarian buy when sentiment is extremely negative."""
        current_price = data.closes[-1]

        return self.create_signal(
            symbol=symbol,
            action=SignalAction.BUY,
            side=SignalSide.LONG,
            entry_price=current_price,
            strength=0.7,
            confidence=sentiment.confidence * 0.65,
            reason=f"Contrarian buy: extreme bearish sentiment {sentiment.overall_sentiment:.2f}",
            metadata={
                "strategy_type": "sentiment",
                "signal_subtype": "contrarian",
                "overall_sentiment": sentiment.overall_sentiment,
            },
        )

    def _create_contrarian_sell_signal(
        self,
        symbol: str,
        sentiment: SentimentScore,
        indicators: dict[str, Any],
        data: MarketData,
        context: StrategyContext,
    ) -> Optional[StrategySignal]:
        """Create contrarian sell when sentiment is extremely positive."""
        current_price = data.closes[-1]

        return self.create_signal(
            symbol=symbol,
            action=SignalAction.SELL,
            side=SignalSide.SHORT,
            entry_price=current_price,
            strength=0.7,
            confidence=sentiment.confidence * 0.65,
            reason=f"Contrarian sell: extreme bullish sentiment {sentiment.overall_sentiment:.2f}",
            metadata={
                "strategy_type": "sentiment",
                "signal_subtype": "contrarian",
                "overall_sentiment": sentiment.overall_sentiment,
            },
        )

    def get_sentiment_history(
        self,
        symbol: str,
        limit: int = 50,
    ) -> list[SentimentScore]:
        """Get sentiment history for symbol."""
        if symbol not in self._sentiment_scores:
            return []

        return self._sentiment_scores[symbol][-limit:]

    def get_current_sentiment(self, symbol: str) -> Optional[SentimentScore]:
        """Get current sentiment for symbol."""
        return self._external_sentiment.get(symbol)

    def get_alerts(self, limit: int = 20) -> list[SentimentAlert]:
        """Get recent sentiment alerts."""
        return self._alerts[-limit:]

    def get_sentiment_statistics(self) -> dict:
        """Get sentiment strategy statistics."""
        return {
            "tracked_symbols": len(self._sentiment_scores),
            "total_data_points": sum(len(h) for h in self._sentiment_scores.values()),
            "alerts_count": len(self._alerts),
            "current_sentiments": {
                symbol: {
                    "overall": score.overall_sentiment,
                    "news": score.news_sentiment,
                    "social": score.social_sentiment,
                    "analyst": score.analyst_sentiment,
                    "momentum": score.sentiment_momentum,
                }
                for symbol, score in self._external_sentiment.items()
            },
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"SentimentStrategy(symbols={len(self._sentiment_scores)})"
