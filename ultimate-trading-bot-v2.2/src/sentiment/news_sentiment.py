"""
News Sentiment Analyzer for Ultimate Trading Bot v2.2.

Analyzes sentiment from news articles and financial news sources.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from .base_analyzer import (
    BaseSentimentAnalyzer,
    SentimentConfig,
    SentimentLabel,
    SentimentResult,
    SentimentSource,
    AggregatedSentiment,
)
from .text_processor import FinancialTextProcessor, ProcessedText

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Represents a news article."""

    title: str
    content: str
    source: str
    url: str
    published_at: datetime

    # Optional fields
    author: str | None = None
    summary: str | None = None
    symbols: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    sentiment_score: float | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_full_text(self) -> str:
        """Get full text for analysis."""
        parts = [self.title]
        if self.summary:
            parts.append(self.summary)
        if self.content:
            parts.append(self.content)
        return " ".join(parts)


@dataclass
class NewsSource:
    """Configuration for a news source."""

    name: str
    weight: float = 1.0
    reliability: float = 0.8
    bias: float = 0.0  # -1 to 1 (negative = bearish bias, positive = bullish bias)
    categories: list[str] = field(default_factory=list)


@dataclass
class NewsSentimentResult(SentimentResult):
    """Extended sentiment result for news."""

    article: NewsArticle | None = None
    headline_sentiment: float = 0.0
    body_sentiment: float = 0.0
    relevance_score: float = 1.0


class NewsSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    Sentiment analyzer specialized for news articles.

    Analyzes headlines, body content, and combines them with source weighting.
    """

    # Default news source configurations
    DEFAULT_SOURCES: dict[str, NewsSource] = {
        "reuters": NewsSource("Reuters", weight=1.2, reliability=0.95),
        "bloomberg": NewsSource("Bloomberg", weight=1.2, reliability=0.95),
        "wsj": NewsSource("Wall Street Journal", weight=1.1, reliability=0.90),
        "cnbc": NewsSource("CNBC", weight=1.0, reliability=0.85),
        "ft": NewsSource("Financial Times", weight=1.1, reliability=0.92),
        "marketwatch": NewsSource("MarketWatch", weight=0.9, reliability=0.80),
        "yahoo": NewsSource("Yahoo Finance", weight=0.8, reliability=0.75),
        "seekingalpha": NewsSource("Seeking Alpha", weight=0.7, reliability=0.70),
        "benzinga": NewsSource("Benzinga", weight=0.8, reliability=0.75),
        "default": NewsSource("Unknown", weight=0.7, reliability=0.60),
    }

    # Headline keywords with sentiment weights
    HEADLINE_KEYWORDS: dict[str, float] = {
        # Strong positive
        "soars": 0.8, "surges": 0.8, "rockets": 0.9, "skyrockets": 0.9,
        "breakthrough": 0.7, "record": 0.6, "beats": 0.6, "exceeds": 0.6,

        # Moderate positive
        "gains": 0.4, "rises": 0.3, "climbs": 0.3, "advances": 0.3,
        "upgrades": 0.5, "bullish": 0.5, "optimistic": 0.4,

        # Strong negative
        "crashes": -0.9, "plunges": -0.8, "tumbles": -0.7, "collapses": -0.8,
        "bankruptcy": -0.9, "fraud": -0.9, "scandal": -0.8,

        # Moderate negative
        "drops": -0.4, "falls": -0.3, "declines": -0.3, "slips": -0.2,
        "downgrades": -0.5, "bearish": -0.5, "concerns": -0.4, "warns": -0.5,
    }

    def __init__(
        self,
        config: SentimentConfig | None = None,
        sources: dict[str, NewsSource] | None = None,
    ) -> None:
        """
        Initialize news sentiment analyzer.

        Args:
            config: Sentiment configuration
            sources: Custom news source configurations
        """
        super().__init__(config or SentimentConfig())

        self.sources = sources or self.DEFAULT_SOURCES.copy()
        self._text_processor = FinancialTextProcessor()
        self._headline_weight = 0.4
        self._body_weight = 0.6

    async def initialize(self) -> None:
        """Initialize the analyzer."""
        self._initialized = True
        logger.info("Initialized NewsSentimentAnalyzer")

    async def analyze(
        self,
        text: str,
        symbol: str | None = None,
    ) -> SentimentResult:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze
            symbol: Trading symbol

        Returns:
            Sentiment result
        """
        if not self._initialized:
            await self.initialize()

        # Process text
        processed = self._text_processor.process(text)

        # Analyze
        score = await self._analyze_text(processed, symbol)
        confidence = self._calculate_confidence(processed)

        return SentimentResult(
            score=score,
            label=self._score_to_label(score),
            confidence=confidence,
            source=SentimentSource.NEWS,
            timestamp=datetime.now(),
            text=text,
            symbol=symbol,
        )

    async def analyze_batch(
        self,
        texts: list[str],
        symbol: str | None = None,
    ) -> list[SentimentResult]:
        """Analyze multiple texts."""
        tasks = [self.analyze(text, symbol) for text in texts]
        return await asyncio.gather(*tasks)

    async def analyze_article(
        self,
        article: NewsArticle,
    ) -> NewsSentimentResult:
        """
        Analyze sentiment of a news article.

        Args:
            article: News article to analyze

        Returns:
            News sentiment result
        """
        if not self._initialized:
            await self.initialize()

        # Get source configuration
        source_config = self._get_source_config(article.source)

        # Analyze headline
        headline_score = await self._analyze_headline(article.title)

        # Analyze body
        body_score = await self._analyze_body(article.content)

        # Combine scores
        combined_score = (
            headline_score * self._headline_weight +
            body_score * self._body_weight
        )

        # Apply source bias correction
        combined_score = self._apply_source_bias(combined_score, source_config)

        # Calculate relevance for symbols
        relevance = self._calculate_relevance(article)

        # Calculate confidence
        confidence = self._calculate_article_confidence(
            article, source_config, headline_score, body_score
        )

        return NewsSentimentResult(
            score=combined_score,
            label=self._score_to_label(combined_score),
            confidence=confidence,
            source=SentimentSource.NEWS,
            timestamp=article.published_at,
            text=article.get_full_text(),
            symbol=article.symbols[0] if article.symbols else None,
            article=article,
            headline_sentiment=headline_score,
            body_sentiment=body_score,
            relevance_score=relevance,
            metadata={
                "source": article.source,
                "source_weight": source_config.weight,
                "source_reliability": source_config.reliability,
            },
        )

    async def analyze_articles(
        self,
        articles: list[NewsArticle],
        symbol: str | None = None,
    ) -> list[NewsSentimentResult]:
        """
        Analyze multiple news articles.

        Args:
            articles: List of articles to analyze
            symbol: Filter by symbol

        Returns:
            List of sentiment results
        """
        # Filter by symbol if provided
        if symbol:
            articles = [
                a for a in articles
                if symbol.upper() in [s.upper() for s in a.symbols]
            ]

        tasks = [self.analyze_article(article) for article in articles]
        return await asyncio.gather(*tasks)

    async def aggregate_news_sentiment(
        self,
        articles: list[NewsArticle],
        symbol: str | None = None,
        hours: int = 24,
    ) -> AggregatedSentiment:
        """
        Aggregate sentiment from multiple news articles.

        Args:
            articles: List of articles
            symbol: Trading symbol
            hours: Time window in hours

        Returns:
            Aggregated sentiment
        """
        # Filter by time window
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_articles = [a for a in articles if a.published_at >= cutoff]

        if not recent_articles:
            return AggregatedSentiment(
                overall_score=0.0,
                overall_label=SentimentLabel.NEUTRAL,
                overall_confidence=0.0,
                start_time=cutoff,
                end_time=datetime.now(),
                total_samples=0,
            )

        # Analyze all articles
        results = await self.analyze_articles(recent_articles, symbol)

        # Aggregate with time decay
        return self._aggregate_results(results, cutoff, datetime.now())

    async def _analyze_headline(self, headline: str) -> float:
        """Analyze headline sentiment."""
        if not headline:
            return 0.0

        headline_lower = headline.lower()
        score = 0.0
        matches = 0

        # Check headline keywords
        for keyword, weight in self.HEADLINE_KEYWORDS.items():
            if keyword in headline_lower:
                score += weight
                matches += 1

        # Normalize
        if matches > 0:
            score = score / matches
            score = max(-1.0, min(1.0, score))

        return score

    async def _analyze_body(self, content: str) -> float:
        """Analyze body content sentiment."""
        if not content:
            return 0.0

        # Use base analysis
        processed = self._text_processor.process(content)
        return await self._analyze_text(processed, None)

    async def _analyze_text(
        self,
        processed: ProcessedText,
        symbol: str | None,
    ) -> float:
        """Analyze processed text."""
        # Financial lexicon-based analysis
        positive_score = 0.0
        negative_score = 0.0

        # Finance positive words
        positive_words = {
            "growth", "profit", "gain", "increase", "rise", "beat",
            "exceed", "strong", "bullish", "upgrade", "buy", "long",
            "outperform", "recovery", "expansion", "dividend", "breakthrough",
        }

        # Finance negative words
        negative_words = {
            "loss", "decline", "drop", "fall", "miss", "weak",
            "bearish", "downgrade", "sell", "short", "underperform",
            "contraction", "recession", "bankruptcy", "fraud", "warning",
        }

        for token in processed.tokens:
            token_lower = token.lower()
            if token_lower in positive_words:
                positive_score += 1.0
            elif token_lower in negative_words:
                negative_score += 1.0

        total = positive_score + negative_score
        if total > 0:
            score = (positive_score - negative_score) / total
        else:
            score = 0.0

        return max(-1.0, min(1.0, score))

    def _get_source_config(self, source_name: str) -> NewsSource:
        """Get configuration for a news source."""
        source_lower = source_name.lower()

        for key, config in self.sources.items():
            if key in source_lower or source_lower in config.name.lower():
                return config

        return self.sources.get("default", NewsSource("Unknown"))

    def _apply_source_bias(
        self,
        score: float,
        source: NewsSource,
    ) -> float:
        """Apply source bias correction to score."""
        # Reduce impact of biased sources
        corrected = score - source.bias * 0.2

        # Scale by reliability
        corrected *= source.reliability

        return max(-1.0, min(1.0, corrected))

    def _calculate_relevance(self, article: NewsArticle) -> float:
        """Calculate relevance score for an article."""
        relevance = 1.0

        # More symbols mentioned = less specific
        if len(article.symbols) > 3:
            relevance *= 0.7
        elif len(article.symbols) > 1:
            relevance *= 0.9

        # Check if content is substantial
        word_count = len(article.content.split()) if article.content else 0
        if word_count < 100:
            relevance *= 0.8
        elif word_count > 1000:
            relevance *= 1.1

        return min(1.0, relevance)

    def _calculate_confidence(self, processed: ProcessedText) -> float:
        """Calculate confidence from processed text."""
        # Based on word count and quality
        word_count = len(processed.tokens)

        if word_count < 10:
            return 0.3
        elif word_count < 50:
            return 0.5
        elif word_count < 200:
            return 0.7
        else:
            return 0.9

    def _calculate_article_confidence(
        self,
        article: NewsArticle,
        source: NewsSource,
        headline_score: float,
        body_score: float,
    ) -> float:
        """Calculate confidence for article analysis."""
        # Start with source reliability
        confidence = source.reliability

        # Adjust based on headline/body agreement
        if (headline_score > 0) == (body_score > 0):
            confidence *= 1.1  # Agreement boosts confidence
        else:
            confidence *= 0.8  # Disagreement reduces confidence

        # Adjust based on content quality
        if article.content and len(article.content) > 500:
            confidence *= 1.05

        return min(1.0, confidence)

    def _aggregate_results(
        self,
        results: list[NewsSentimentResult],
        start_time: datetime,
        end_time: datetime,
    ) -> AggregatedSentiment:
        """Aggregate sentiment results with time decay."""
        if not results:
            return AggregatedSentiment(
                overall_score=0.0,
                overall_label=SentimentLabel.NEUTRAL,
                overall_confidence=0.0,
                start_time=start_time,
                end_time=end_time,
            )

        now = datetime.now()
        weighted_sum = 0.0
        total_weight = 0.0

        source_scores: dict[str, list[float]] = {}
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        scores = []

        for result in results:
            # Calculate time decay
            age_hours = (now - result.timestamp).total_seconds() / 3600
            decay = self.config.decay_rate ** (age_hours / 24)

            # Weight by confidence, relevance, and decay
            weight = (
                result.confidence *
                result.relevance_score *
                decay
            )

            weighted_sum += result.score * weight
            total_weight += weight
            scores.append(result.score)

            # Track by source
            if result.article:
                source = result.article.source
                if source not in source_scores:
                    source_scores[source] = []
                source_scores[source].append(result.score)

            # Count labels
            if result.label in [SentimentLabel.POSITIVE, SentimentLabel.VERY_POSITIVE]:
                positive_count += 1
            elif result.label in [SentimentLabel.NEGATIVE, SentimentLabel.VERY_NEGATIVE]:
                negative_count += 1
            else:
                neutral_count += 1

        # Calculate overall score
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Calculate source averages
        source_avg = {
            source: sum(scores) / len(scores)
            for source, scores in source_scores.items()
        }

        # Calculate trend
        if len(results) >= 2:
            sorted_results = sorted(results, key=lambda r: r.timestamp)
            first_half = sorted_results[:len(sorted_results)//2]
            second_half = sorted_results[len(sorted_results)//2:]

            first_avg = sum(r.score for r in first_half) / len(first_half)
            second_avg = sum(r.score for r in second_half) / len(second_half)
            trend = second_avg - first_avg
        else:
            trend = 0.0

        # Calculate volatility
        if len(scores) > 1:
            mean = sum(scores) / len(scores)
            volatility = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
        else:
            volatility = 0.0

        total_count = len(results)

        return AggregatedSentiment(
            overall_score=overall_score,
            overall_label=self._score_to_label(overall_score),
            overall_confidence=total_weight / len(results) if results else 0.0,
            start_time=start_time,
            end_time=end_time,
            source_scores=source_avg,
            source_counts={source: len(scores) for source, scores in source_scores.items()},
            total_samples=total_count,
            positive_ratio=positive_count / total_count if total_count > 0 else 0.0,
            negative_ratio=negative_count / total_count if total_count > 0 else 0.0,
            neutral_ratio=neutral_count / total_count if total_count > 0 else 0.0,
            trend=trend,
            volatility=volatility,
            results=results,
        )

    def add_source(self, source: NewsSource) -> None:
        """Add a custom news source configuration."""
        self.sources[source.name.lower()] = source
        logger.info(f"Added news source: {source.name}")

    def set_headline_weight(self, weight: float) -> None:
        """Set the weight for headline sentiment."""
        if 0 <= weight <= 1:
            self._headline_weight = weight
            self._body_weight = 1.0 - weight
        else:
            raise ValueError("Weight must be between 0 and 1")


class NewsAggregator:
    """
    Aggregates news from multiple sources for sentiment analysis.
    """

    def __init__(
        self,
        analyzer: NewsSentimentAnalyzer | None = None,
        max_articles: int = 100,
    ) -> None:
        """
        Initialize news aggregator.

        Args:
            analyzer: Sentiment analyzer to use
            max_articles: Maximum articles to keep per symbol
        """
        self.analyzer = analyzer or NewsSentimentAnalyzer()
        self.max_articles = max_articles

        self._articles: dict[str, list[NewsArticle]] = {}
        self._cache: dict[str, AggregatedSentiment] = {}
        self._cache_ttl = 300  # 5 minutes

        logger.info("Initialized NewsAggregator")

    async def add_article(self, article: NewsArticle) -> None:
        """Add an article to the aggregator."""
        for symbol in article.symbols:
            symbol = symbol.upper()
            if symbol not in self._articles:
                self._articles[symbol] = []

            self._articles[symbol].append(article)

            # Trim to max
            if len(self._articles[symbol]) > self.max_articles:
                self._articles[symbol] = sorted(
                    self._articles[symbol],
                    key=lambda a: a.published_at,
                    reverse=True
                )[:self.max_articles]

        # Invalidate cache
        for symbol in article.symbols:
            self._cache.pop(symbol.upper(), None)

    async def get_sentiment(
        self,
        symbol: str,
        hours: int = 24,
    ) -> AggregatedSentiment:
        """
        Get aggregated sentiment for a symbol.

        Args:
            symbol: Trading symbol
            hours: Time window in hours

        Returns:
            Aggregated sentiment
        """
        symbol = symbol.upper()

        # Check cache
        cache_key = f"{symbol}_{hours}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Get articles
        articles = self._articles.get(symbol, [])

        # Analyze
        result = await self.analyzer.aggregate_news_sentiment(
            articles, symbol, hours
        )

        # Cache result
        self._cache[cache_key] = result

        return result

    def get_recent_articles(
        self,
        symbol: str,
        limit: int = 10,
    ) -> list[NewsArticle]:
        """Get recent articles for a symbol."""
        symbol = symbol.upper()
        articles = self._articles.get(symbol, [])

        return sorted(
            articles,
            key=lambda a: a.published_at,
            reverse=True
        )[:limit]

    def clear(self, symbol: str | None = None) -> None:
        """Clear articles and cache."""
        if symbol:
            self._articles.pop(symbol.upper(), None)
            # Clear related cache entries
            self._cache = {
                k: v for k, v in self._cache.items()
                if not k.startswith(symbol.upper())
            }
        else:
            self._articles.clear()
            self._cache.clear()


def create_news_analyzer(
    config: SentimentConfig | None = None,
) -> NewsSentimentAnalyzer:
    """
    Create a news sentiment analyzer.

    Args:
        config: Sentiment configuration

    Returns:
        News sentiment analyzer instance
    """
    return NewsSentimentAnalyzer(config)
