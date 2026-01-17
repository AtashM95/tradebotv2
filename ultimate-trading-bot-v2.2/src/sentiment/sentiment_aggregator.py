"""
Sentiment Aggregator for Ultimate Trading Bot v2.2.

Combines sentiment from multiple sources into unified sentiment scores and signals.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

from .base_analyzer import (
    SentimentConfig,
    SentimentLabel,
    SentimentResult,
    SentimentSource,
    AggregatedSentiment,
    SentimentSignal,
)
from .news_sentiment import NewsSentimentAnalyzer, NewsArticle
from .social_sentiment import SocialSentimentAnalyzer, SocialPost
from .market_sentiment import MarketSentimentAnalyzer, MarketData, OptionsFlow

logger = logging.getLogger(__name__)


class AggregationMethod(str, Enum):
    """Methods for aggregating sentiment."""
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    VOLUME_WEIGHTED = "volume_weighted"
    RECENCY_WEIGHTED = "recency_weighted"
    BAYESIAN = "bayesian"


@dataclass
class SourceWeight:
    """Weight configuration for a sentiment source."""

    source: SentimentSource
    weight: float = 1.0
    reliability: float = 1.0
    decay_rate: float = 0.95


@dataclass
class AggregationConfig:
    """Configuration for sentiment aggregation."""

    method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE
    time_window_hours: int = 24
    min_samples: int = 3
    min_confidence: float = 0.5

    # Source weights
    source_weights: dict[str, float] = field(default_factory=lambda: {
        SentimentSource.NEWS.value: 1.0,
        SentimentSource.TWITTER.value: 0.8,
        SentimentSource.REDDIT.value: 0.7,
        SentimentSource.STOCKTWITS.value: 0.9,
        SentimentSource.MARKET.value: 1.1,
        SentimentSource.ANALYST.value: 1.2,
    })

    # Trend detection
    trend_window: int = 6  # hours
    trend_threshold: float = 0.1

    # Signal generation
    signal_threshold: float = 0.3
    strong_signal_threshold: float = 0.6


@dataclass
class MultiSourceSentiment:
    """Combined sentiment from multiple sources."""

    symbol: str
    timestamp: datetime

    # Overall metrics
    overall_score: float
    overall_label: SentimentLabel
    overall_confidence: float

    # Source breakdown
    news_sentiment: AggregatedSentiment | None = None
    social_sentiment: AggregatedSentiment | None = None
    market_sentiment: AggregatedSentiment | None = None

    # Analysis
    agreement_score: float = 0.0  # How much sources agree
    trend: float = 0.0
    volatility: float = 0.0
    momentum: float = 0.0

    # Counts
    total_samples: int = 0
    source_counts: dict[str, int] = field(default_factory=dict)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "overall_score": self.overall_score,
            "overall_label": self.overall_label.value,
            "overall_confidence": self.overall_confidence,
            "agreement_score": self.agreement_score,
            "trend": self.trend,
            "volatility": self.volatility,
            "momentum": self.momentum,
            "total_samples": self.total_samples,
            "source_counts": self.source_counts,
            "metadata": self.metadata,
        }


class SentimentAggregator:
    """
    Aggregates sentiment from multiple sources.

    Combines news, social media, and market sentiment into unified scores.
    """

    def __init__(
        self,
        config: AggregationConfig | None = None,
        news_analyzer: NewsSentimentAnalyzer | None = None,
        social_analyzer: SocialSentimentAnalyzer | None = None,
        market_analyzer: MarketSentimentAnalyzer | None = None,
    ) -> None:
        """
        Initialize sentiment aggregator.

        Args:
            config: Aggregation configuration
            news_analyzer: News sentiment analyzer
            social_analyzer: Social sentiment analyzer
            market_analyzer: Market sentiment analyzer
        """
        self.config = config or AggregationConfig()

        self.news_analyzer = news_analyzer or NewsSentimentAnalyzer()
        self.social_analyzer = social_analyzer or SocialSentimentAnalyzer()
        self.market_analyzer = market_analyzer or MarketSentimentAnalyzer()

        # History for trend calculation
        self._sentiment_history: dict[str, list[tuple[datetime, float]]] = {}

        self._initialized = False

        logger.info("Initialized SentimentAggregator")

    async def initialize(self) -> None:
        """Initialize all analyzers."""
        await asyncio.gather(
            self.news_analyzer.initialize(),
            self.social_analyzer.initialize(),
            self.market_analyzer.initialize(),
        )
        self._initialized = True
        logger.info("SentimentAggregator initialized")

    async def aggregate(
        self,
        symbol: str,
        news_articles: list[NewsArticle] | None = None,
        social_posts: list[SocialPost] | None = None,
        market_data: MarketData | None = None,
        options_flow: OptionsFlow | None = None,
        prices: list[float] | None = None,
    ) -> MultiSourceSentiment:
        """
        Aggregate sentiment from all sources.

        Args:
            symbol: Trading symbol
            news_articles: News articles
            social_posts: Social media posts
            market_data: Market data
            options_flow: Options flow data
            prices: Historical prices

        Returns:
            Multi-source sentiment
        """
        if not self._initialized:
            await self.initialize()

        # Analyze each source
        news_sentiment = None
        social_sentiment = None
        market_sentiment = None

        if news_articles:
            news_sentiment = await self.news_analyzer.aggregate_news_sentiment(
                news_articles,
                symbol,
                self.config.time_window_hours,
            )

        if social_posts:
            social_sentiment = await self.social_analyzer.aggregate_social_sentiment(
                social_posts,
                symbol,
                self.config.time_window_hours,
            )

        if market_data:
            market_sentiment = await self.market_analyzer.aggregate_market_sentiment(
                market_data,
                options_flow,
                prices,
            )

        # Combine sentiments
        return self._combine_sentiments(
            symbol,
            news_sentiment,
            social_sentiment,
            market_sentiment,
        )

    async def aggregate_results(
        self,
        results: list[SentimentResult],
        symbol: str | None = None,
    ) -> AggregatedSentiment:
        """
        Aggregate a list of sentiment results.

        Args:
            results: List of sentiment results
            symbol: Trading symbol

        Returns:
            Aggregated sentiment
        """
        if not results:
            return AggregatedSentiment(
                overall_score=0.0,
                overall_label=SentimentLabel.NEUTRAL,
                overall_confidence=0.0,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )

        # Filter by confidence
        results = [r for r in results if r.confidence >= self.config.min_confidence]

        if not results:
            return AggregatedSentiment(
                overall_score=0.0,
                overall_label=SentimentLabel.NEUTRAL,
                overall_confidence=0.0,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )

        # Aggregate based on method
        if self.config.method == AggregationMethod.SIMPLE_AVERAGE:
            overall_score = self._simple_average(results)
        elif self.config.method == AggregationMethod.WEIGHTED_AVERAGE:
            overall_score = self._weighted_average(results)
        elif self.config.method == AggregationMethod.CONFIDENCE_WEIGHTED:
            overall_score = self._confidence_weighted(results)
        elif self.config.method == AggregationMethod.RECENCY_WEIGHTED:
            overall_score = self._recency_weighted(results)
        elif self.config.method == AggregationMethod.BAYESIAN:
            overall_score = self._bayesian_aggregate(results)
        else:
            overall_score = self._weighted_average(results)

        # Calculate statistics
        scores = [r.score for r in results]
        timestamps = [r.timestamp for r in results]

        volatility = float(np.std(scores)) if len(scores) > 1 else 0.0
        trend = self._calculate_trend(scores, timestamps)

        # Count by source
        source_counts: dict[str, int] = {}
        source_scores: dict[str, list[float]] = {}
        for r in results:
            source = r.source.value
            source_counts[source] = source_counts.get(source, 0) + 1
            if source not in source_scores:
                source_scores[source] = []
            source_scores[source].append(r.score)

        source_avg = {s: sum(scores) / len(scores) for s, scores in source_scores.items()}

        # Calculate sentiment ratios
        positive_count = sum(
            1 for r in results
            if r.label in [SentimentLabel.POSITIVE, SentimentLabel.VERY_POSITIVE]
        )
        negative_count = sum(
            1 for r in results
            if r.label in [SentimentLabel.NEGATIVE, SentimentLabel.VERY_NEGATIVE]
        )
        neutral_count = len(results) - positive_count - negative_count

        return AggregatedSentiment(
            overall_score=overall_score,
            overall_label=self._score_to_label(overall_score),
            overall_confidence=sum(r.confidence for r in results) / len(results),
            start_time=min(timestamps),
            end_time=max(timestamps),
            source_scores=source_avg,
            source_counts=source_counts,
            total_samples=len(results),
            positive_ratio=positive_count / len(results),
            negative_ratio=negative_count / len(results),
            neutral_ratio=neutral_count / len(results),
            trend=trend,
            volatility=volatility,
            results=results,
        )

    def _combine_sentiments(
        self,
        symbol: str,
        news: AggregatedSentiment | None,
        social: AggregatedSentiment | None,
        market: AggregatedSentiment | None,
    ) -> MultiSourceSentiment:
        """Combine sentiment from different sources."""
        scores = []
        weights = []
        source_counts: dict[str, int] = {}

        # Collect scores with weights
        if news and news.total_samples >= self.config.min_samples:
            weight = self.config.source_weights.get(SentimentSource.NEWS.value, 1.0)
            scores.append(news.overall_score)
            weights.append(weight * news.overall_confidence)
            source_counts["news"] = news.total_samples

        if social and social.total_samples >= self.config.min_samples:
            weight = self.config.source_weights.get(SentimentSource.SOCIAL_MEDIA.value, 0.8)
            scores.append(social.overall_score)
            weights.append(weight * social.overall_confidence)
            source_counts["social"] = social.total_samples

        if market and market.total_samples > 0:
            weight = self.config.source_weights.get(SentimentSource.MARKET.value, 1.1)
            scores.append(market.overall_score)
            weights.append(weight * market.overall_confidence)
            source_counts["market"] = market.total_samples

        # Calculate combined score
        if scores:
            total_weight = sum(weights)
            overall_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
            overall_confidence = total_weight / len(scores)
        else:
            overall_score = 0.0
            overall_confidence = 0.0

        # Calculate agreement score
        agreement = self._calculate_agreement(scores)

        # Calculate trend and momentum
        trend = self._get_symbol_trend(symbol)
        momentum = self._calculate_momentum(symbol, overall_score)

        # Calculate volatility
        volatility = float(np.std(scores)) if len(scores) > 1 else 0.0

        # Update history
        self._update_history(symbol, overall_score)

        return MultiSourceSentiment(
            symbol=symbol,
            timestamp=datetime.now(),
            overall_score=overall_score,
            overall_label=self._score_to_label(overall_score),
            overall_confidence=overall_confidence,
            news_sentiment=news,
            social_sentiment=social,
            market_sentiment=market,
            agreement_score=agreement,
            trend=trend,
            volatility=volatility,
            momentum=momentum,
            total_samples=sum(source_counts.values()),
            source_counts=source_counts,
        )

    def _simple_average(self, results: list[SentimentResult]) -> float:
        """Calculate simple average of scores."""
        if not results:
            return 0.0
        return sum(r.score for r in results) / len(results)

    def _weighted_average(self, results: list[SentimentResult]) -> float:
        """Calculate weighted average based on source weights."""
        if not results:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for r in results:
            weight = self.config.source_weights.get(r.source.value, 1.0)
            weighted_sum += r.score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _confidence_weighted(self, results: list[SentimentResult]) -> float:
        """Calculate confidence-weighted average."""
        if not results:
            return 0.0

        weighted_sum = sum(r.score * r.confidence for r in results)
        total_weight = sum(r.confidence for r in results)

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _recency_weighted(self, results: list[SentimentResult]) -> float:
        """Calculate recency-weighted average."""
        if not results:
            return 0.0

        now = datetime.now()
        weighted_sum = 0.0
        total_weight = 0.0

        for r in results:
            age_hours = (now - r.timestamp).total_seconds() / 3600
            decay = 0.95 ** (age_hours / 24)  # Decay over 24 hours

            weight = decay * r.confidence
            weighted_sum += r.score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _bayesian_aggregate(self, results: list[SentimentResult]) -> float:
        """Calculate Bayesian aggregate with prior."""
        if not results:
            return 0.0

        # Prior: neutral sentiment
        prior_mean = 0.0
        prior_variance = 0.5
        prior_precision = 1 / prior_variance

        # Likelihood from observations
        scores = [r.score for r in results]
        confidences = [r.confidence for r in results]

        # Precision-weighted combination
        data_precision = sum(c ** 2 for c in confidences)
        data_mean = (
            sum(s * c ** 2 for s, c in zip(scores, confidences)) / data_precision
            if data_precision > 0 else 0.0
        )

        # Posterior
        posterior_precision = prior_precision + data_precision
        posterior_mean = (
            (prior_precision * prior_mean + data_precision * data_mean) /
            posterior_precision
        )

        return posterior_mean

    def _calculate_trend(
        self,
        scores: list[float],
        timestamps: list[datetime],
    ) -> float:
        """Calculate sentiment trend."""
        if len(scores) < 2:
            return 0.0

        # Sort by timestamp
        paired = sorted(zip(timestamps, scores))
        sorted_scores = [s for _, s in paired]

        # Compare first half to second half
        mid = len(sorted_scores) // 2
        first_avg = sum(sorted_scores[:mid]) / mid if mid > 0 else 0.0
        second_avg = sum(sorted_scores[mid:]) / (len(sorted_scores) - mid)

        return second_avg - first_avg

    def _calculate_agreement(self, scores: list[float]) -> float:
        """Calculate how much sources agree."""
        if len(scores) < 2:
            return 1.0

        # Agreement based on variance
        variance = float(np.var(scores))
        # Map variance to 0-1 agreement score
        # Low variance = high agreement
        agreement = 1.0 / (1.0 + variance * 4)

        return agreement

    def _get_symbol_trend(self, symbol: str) -> float:
        """Get historical sentiment trend for symbol."""
        if symbol not in self._sentiment_history:
            return 0.0

        history = self._sentiment_history[symbol]
        if len(history) < 2:
            return 0.0

        # Look at recent history
        cutoff = datetime.now() - timedelta(hours=self.config.trend_window)
        recent = [(t, s) for t, s in history if t >= cutoff]

        if len(recent) < 2:
            return 0.0

        # Calculate trend
        first_scores = [s for _, s in recent[:len(recent)//2]]
        last_scores = [s for _, s in recent[len(recent)//2:]]

        first_avg = sum(first_scores) / len(first_scores)
        last_avg = sum(last_scores) / len(last_scores)

        return last_avg - first_avg

    def _calculate_momentum(self, symbol: str, current_score: float) -> float:
        """Calculate sentiment momentum."""
        if symbol not in self._sentiment_history:
            return 0.0

        history = self._sentiment_history[symbol]
        if not history:
            return 0.0

        # Compare to recent average
        recent = history[-10:]  # Last 10 readings
        if not recent:
            return 0.0

        avg = sum(s for _, s in recent) / len(recent)
        return current_score - avg

    def _update_history(self, symbol: str, score: float) -> None:
        """Update sentiment history for symbol."""
        if symbol not in self._sentiment_history:
            self._sentiment_history[symbol] = []

        self._sentiment_history[symbol].append((datetime.now(), score))

        # Keep last 100 entries
        if len(self._sentiment_history[symbol]) > 100:
            self._sentiment_history[symbol] = self._sentiment_history[symbol][-100:]

    def _score_to_label(self, score: float) -> SentimentLabel:
        """Convert score to sentiment label."""
        if score <= -0.6:
            return SentimentLabel.VERY_NEGATIVE
        elif score <= -0.2:
            return SentimentLabel.NEGATIVE
        elif score >= 0.6:
            return SentimentLabel.VERY_POSITIVE
        elif score >= 0.2:
            return SentimentLabel.POSITIVE
        else:
            return SentimentLabel.NEUTRAL

    def generate_signal(
        self,
        sentiment: MultiSourceSentiment,
    ) -> SentimentSignal:
        """
        Generate trading signal from multi-source sentiment.

        Args:
            sentiment: Multi-source sentiment

        Returns:
            Sentiment signal
        """
        score = sentiment.overall_score
        confidence = sentiment.overall_confidence

        # Adjust confidence by agreement
        adjusted_confidence = confidence * (0.5 + sentiment.agreement_score * 0.5)

        # Determine direction
        if score > self.config.strong_signal_threshold:
            direction = 1
            strength = min((score - self.config.signal_threshold) * 2, 1.0)
        elif score < -self.config.strong_signal_threshold:
            direction = -1
            strength = min((abs(score) - self.config.signal_threshold) * 2, 1.0)
        elif score > self.config.signal_threshold:
            direction = 1
            strength = (score - self.config.signal_threshold) / self.config.signal_threshold
        elif score < -self.config.signal_threshold:
            direction = -1
            strength = (abs(score) - self.config.signal_threshold) / self.config.signal_threshold
        else:
            direction = 0
            strength = 0.0

        # Boost strength if trend confirms
        if (direction > 0 and sentiment.trend > 0) or (direction < 0 and sentiment.trend < 0):
            strength = min(strength * 1.2, 1.0)

        # Generate reasoning
        reasoning = self._generate_reasoning(sentiment, direction)

        return SentimentSignal(
            direction=direction,
            strength=strength,
            confidence=adjusted_confidence,
            sentiment_score=score,
            sentiment_label=sentiment.overall_label,
            symbol=sentiment.symbol,
            timestamp=sentiment.timestamp,
            sources=list(SentimentSource),
            reasoning=reasoning,
            indicators={
                "agreement": sentiment.agreement_score,
                "trend": sentiment.trend,
                "volatility": sentiment.volatility,
                "momentum": sentiment.momentum,
            },
        )

    def _generate_reasoning(
        self,
        sentiment: MultiSourceSentiment,
        direction: int,
    ) -> str:
        """Generate reasoning for signal."""
        parts = []

        # Overall sentiment
        if direction > 0:
            parts.append(f"Bullish sentiment ({sentiment.overall_score:.2f})")
        elif direction < 0:
            parts.append(f"Bearish sentiment ({sentiment.overall_score:.2f})")
        else:
            parts.append(f"Neutral sentiment ({sentiment.overall_score:.2f})")

        # Source breakdown
        if sentiment.news_sentiment:
            parts.append(f"News: {sentiment.news_sentiment.overall_score:.2f}")
        if sentiment.social_sentiment:
            parts.append(f"Social: {sentiment.social_sentiment.overall_score:.2f}")
        if sentiment.market_sentiment:
            parts.append(f"Market: {sentiment.market_sentiment.overall_score:.2f}")

        # Agreement
        if sentiment.agreement_score > 0.8:
            parts.append("High agreement across sources")
        elif sentiment.agreement_score < 0.4:
            parts.append("Low agreement across sources")

        # Trend
        if abs(sentiment.trend) > 0.1:
            trend_dir = "improving" if sentiment.trend > 0 else "deteriorating"
            parts.append(f"Sentiment {trend_dir}")

        return "; ".join(parts)


def create_sentiment_aggregator(
    config: AggregationConfig | None = None,
) -> SentimentAggregator:
    """
    Create a sentiment aggregator.

    Args:
        config: Aggregation configuration

    Returns:
        Sentiment aggregator instance
    """
    return SentimentAggregator(config)
