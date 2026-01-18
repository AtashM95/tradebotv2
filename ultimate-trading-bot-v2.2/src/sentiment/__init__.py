"""
Sentiment Analysis Package for Ultimate Trading Bot v2.2.

This package provides comprehensive sentiment analysis capabilities including:
- Base sentiment analyzers (lexicon-based, VADER, financial)
- News sentiment analysis
- Social media sentiment (Twitter, Reddit, StockTwits)
- Market-based sentiment indicators
- Entity extraction (tickers, companies, financial metrics)
- Text preprocessing and cleaning
- Sentiment aggregation and scoring
- Signal generation from sentiment
"""

import logging
from typing import Any

from .base_analyzer import (
    SentimentLabel,
    SentimentSource,
    SentimentConfig,
    SentimentResult,
    AggregatedSentiment,
    SentimentSignal,
    BaseSentimentAnalyzer,
    LexiconBasedAnalyzer,
    VADERAnalyzer,
    FinancialSentimentAnalyzer,
    create_sentiment_analyzer,
)
from .text_processor import (
    ProcessingConfig,
    ProcessedText,
    TextProcessor,
    FinancialTextProcessor,
    BatchTextProcessor,
    create_text_processor,
)
from .entity_extractor import (
    EntityType,
    Entity,
    ExtractionResult,
    EntityExtractor,
    FinancialEntityExtractor,
    create_entity_extractor,
)
from .news_sentiment import (
    NewsArticle,
    NewsSource,
    NewsSentimentResult,
    NewsSentimentAnalyzer,
    NewsAggregator,
    create_news_analyzer,
)
from .social_sentiment import (
    SocialPlatform,
    SocialPost,
    AuthorProfile,
    SocialSentimentResult,
    SocialSentimentAnalyzer,
    TwitterAnalyzer,
    RedditAnalyzer,
    StockTwitsAnalyzer,
    create_social_analyzer,
)
from .market_sentiment import (
    MarketIndicator,
    MarketData,
    OptionsFlow,
    MarketSentimentResult,
    MarketSentimentAnalyzer,
    create_market_analyzer,
)
from .sentiment_aggregator import (
    AggregationMethod,
    SourceWeight,
    AggregationConfig,
    MultiSourceSentiment,
    SentimentAggregator,
    create_sentiment_aggregator,
)
from .sentiment_scorer import (
    NormalizationMethod,
    CalibrationMethod,
    ScoringConfig,
    ScoreDistribution,
    CalibratedScore,
    SentimentScorer,
    SourceScorer,
    create_sentiment_scorer,
    create_source_scorer,
)


logger = logging.getLogger(__name__)


__all__ = [
    # Base Analyzer
    "SentimentLabel",
    "SentimentSource",
    "SentimentConfig",
    "SentimentResult",
    "AggregatedSentiment",
    "SentimentSignal",
    "BaseSentimentAnalyzer",
    "LexiconBasedAnalyzer",
    "VADERAnalyzer",
    "FinancialSentimentAnalyzer",
    "create_sentiment_analyzer",
    # Text Processor
    "ProcessingConfig",
    "ProcessedText",
    "TextProcessor",
    "FinancialTextProcessor",
    "BatchTextProcessor",
    "create_text_processor",
    # Entity Extractor
    "EntityType",
    "Entity",
    "ExtractionResult",
    "EntityExtractor",
    "FinancialEntityExtractor",
    "create_entity_extractor",
    # News Sentiment
    "NewsArticle",
    "NewsSource",
    "NewsSentimentResult",
    "NewsSentimentAnalyzer",
    "NewsAggregator",
    "create_news_analyzer",
    # Social Sentiment
    "SocialPlatform",
    "SocialPost",
    "AuthorProfile",
    "SocialSentimentResult",
    "SocialSentimentAnalyzer",
    "TwitterAnalyzer",
    "RedditAnalyzer",
    "StockTwitsAnalyzer",
    "create_social_analyzer",
    # Market Sentiment
    "MarketIndicator",
    "MarketData",
    "OptionsFlow",
    "MarketSentimentResult",
    "MarketSentimentAnalyzer",
    "create_market_analyzer",
    # Sentiment Aggregator
    "AggregationMethod",
    "SourceWeight",
    "AggregationConfig",
    "MultiSourceSentiment",
    "SentimentAggregator",
    "create_sentiment_aggregator",
    # Sentiment Scorer
    "NormalizationMethod",
    "CalibrationMethod",
    "ScoringConfig",
    "ScoreDistribution",
    "CalibratedScore",
    "SentimentScorer",
    "SourceScorer",
    "create_sentiment_scorer",
    "create_source_scorer",
]


class SentimentManager:
    """
    Central manager for all sentiment analysis operations.

    Coordinates text processing, entity extraction, sentiment analysis,
    aggregation, and signal generation.
    """

    def __init__(
        self,
        config: SentimentConfig | None = None,
        aggregation_config: AggregationConfig | None = None,
        scoring_config: ScoringConfig | None = None,
    ) -> None:
        """
        Initialize sentiment manager.

        Args:
            config: Base sentiment configuration
            aggregation_config: Aggregation configuration
            scoring_config: Scoring configuration
        """
        self.config = config or SentimentConfig()
        self.aggregation_config = aggregation_config or AggregationConfig()
        self.scoring_config = scoring_config or ScoringConfig()

        # Components
        self._text_processor: FinancialTextProcessor | None = None
        self._entity_extractor: FinancialEntityExtractor | None = None
        self._news_analyzer: NewsSentimentAnalyzer | None = None
        self._social_analyzer: SocialSentimentAnalyzer | None = None
        self._market_analyzer: MarketSentimentAnalyzer | None = None
        self._aggregator: SentimentAggregator | None = None
        self._scorer: SourceScorer | None = None

        self._initialized = False

        logger.info("SentimentManager created")

    async def initialize(self) -> None:
        """Initialize all sentiment components."""
        try:
            # Initialize text processor
            self._text_processor = FinancialTextProcessor()

            # Initialize entity extractor
            self._entity_extractor = FinancialEntityExtractor()

            # Initialize analyzers
            self._news_analyzer = NewsSentimentAnalyzer(self.config)
            await self._news_analyzer.initialize()

            self._social_analyzer = SocialSentimentAnalyzer(self.config)
            await self._social_analyzer.initialize()

            self._market_analyzer = MarketSentimentAnalyzer(self.config)
            await self._market_analyzer.initialize()

            # Initialize aggregator
            self._aggregator = SentimentAggregator(
                self.aggregation_config,
                self._news_analyzer,
                self._social_analyzer,
                self._market_analyzer,
            )
            await self._aggregator.initialize()

            # Initialize scorer
            self._scorer = SourceScorer(self.scoring_config)

            self._initialized = True
            logger.info("SentimentManager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize SentimentManager: {e}")
            raise

    async def analyze_text(
        self,
        text: str,
        symbol: str | None = None,
    ) -> SentimentResult:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze
            symbol: Optional trading symbol

        Returns:
            Sentiment result
        """
        if not self._initialized:
            await self.initialize()

        if self._news_analyzer is None:
            raise RuntimeError("News analyzer not initialized")

        return await self._news_analyzer.analyze(text, symbol)

    async def analyze_news(
        self,
        articles: list[NewsArticle],
        symbol: str | None = None,
        hours: int = 24,
    ) -> AggregatedSentiment:
        """
        Analyze news articles.

        Args:
            articles: News articles
            symbol: Trading symbol
            hours: Time window

        Returns:
            Aggregated sentiment
        """
        if not self._initialized:
            await self.initialize()

        if self._news_analyzer is None:
            raise RuntimeError("News analyzer not initialized")

        return await self._news_analyzer.aggregate_news_sentiment(
            articles, symbol, hours
        )

    async def analyze_social(
        self,
        posts: list[SocialPost],
        symbol: str | None = None,
        hours: int = 24,
    ) -> AggregatedSentiment:
        """
        Analyze social media posts.

        Args:
            posts: Social media posts
            symbol: Trading symbol
            hours: Time window

        Returns:
            Aggregated sentiment
        """
        if not self._initialized:
            await self.initialize()

        if self._social_analyzer is None:
            raise RuntimeError("Social analyzer not initialized")

        return await self._social_analyzer.aggregate_social_sentiment(
            posts, symbol, hours
        )

    async def analyze_market(
        self,
        market_data: MarketData,
        options_flow: OptionsFlow | None = None,
        prices: list[float] | None = None,
    ) -> AggregatedSentiment:
        """
        Analyze market sentiment.

        Args:
            market_data: Market data
            options_flow: Options flow
            prices: Historical prices

        Returns:
            Aggregated sentiment
        """
        if not self._initialized:
            await self.initialize()

        if self._market_analyzer is None:
            raise RuntimeError("Market analyzer not initialized")

        return await self._market_analyzer.aggregate_market_sentiment(
            market_data, options_flow, prices
        )

    async def get_multi_source_sentiment(
        self,
        symbol: str,
        news_articles: list[NewsArticle] | None = None,
        social_posts: list[SocialPost] | None = None,
        market_data: MarketData | None = None,
        options_flow: OptionsFlow | None = None,
        prices: list[float] | None = None,
    ) -> MultiSourceSentiment:
        """
        Get aggregated sentiment from multiple sources.

        Args:
            symbol: Trading symbol
            news_articles: News articles
            social_posts: Social posts
            market_data: Market data
            options_flow: Options flow
            prices: Historical prices

        Returns:
            Multi-source sentiment
        """
        if not self._initialized:
            await self.initialize()

        if self._aggregator is None:
            raise RuntimeError("Aggregator not initialized")

        return await self._aggregator.aggregate(
            symbol,
            news_articles,
            social_posts,
            market_data,
            options_flow,
            prices,
        )

    async def generate_signal(
        self,
        symbol: str,
        news_articles: list[NewsArticle] | None = None,
        social_posts: list[SocialPost] | None = None,
        market_data: MarketData | None = None,
        options_flow: OptionsFlow | None = None,
        prices: list[float] | None = None,
    ) -> SentimentSignal:
        """
        Generate trading signal from sentiment analysis.

        Args:
            symbol: Trading symbol
            news_articles: News articles
            social_posts: Social posts
            market_data: Market data
            options_flow: Options flow
            prices: Historical prices

        Returns:
            Sentiment signal
        """
        if not self._initialized:
            await self.initialize()

        if self._aggregator is None:
            raise RuntimeError("Aggregator not initialized")

        # Get multi-source sentiment
        sentiment = await self.get_multi_source_sentiment(
            symbol,
            news_articles,
            social_posts,
            market_data,
            options_flow,
            prices,
        )

        # Generate signal
        return self._aggregator.generate_signal(sentiment)

    def extract_entities(self, text: str) -> ExtractionResult:
        """
        Extract entities from text.

        Args:
            text: Text to process

        Returns:
            Extraction result
        """
        if self._entity_extractor is None:
            self._entity_extractor = FinancialEntityExtractor()

        return self._entity_extractor.extract(text)

    def process_text(self, text: str) -> ProcessedText:
        """
        Process and clean text.

        Args:
            text: Text to process

        Returns:
            Processed text
        """
        if self._text_processor is None:
            self._text_processor = FinancialTextProcessor()

        return self._text_processor.process(text)

    def score_sentiment(
        self,
        result: SentimentResult,
    ) -> CalibratedScore:
        """
        Score and calibrate a sentiment result.

        Args:
            result: Sentiment result

        Returns:
            Calibrated score
        """
        if self._scorer is None:
            self._scorer = SourceScorer(self.scoring_config)

        return self._scorer.score_result(result)

    async def shutdown(self) -> None:
        """Shutdown sentiment manager."""
        logger.info("Shutting down SentimentManager")

        if self._news_analyzer:
            await self._news_analyzer.shutdown()

        if self._social_analyzer:
            await self._social_analyzer.shutdown()

        self._initialized = False
        logger.info("SentimentManager shutdown complete")


def create_sentiment_manager(
    config: SentimentConfig | None = None,
    aggregation_config: AggregationConfig | None = None,
    scoring_config: ScoringConfig | None = None,
) -> SentimentManager:
    """
    Create a sentiment manager instance.

    Args:
        config: Sentiment configuration
        aggregation_config: Aggregation configuration
        scoring_config: Scoring configuration

    Returns:
        Configured SentimentManager instance
    """
    return SentimentManager(
        config=config,
        aggregation_config=aggregation_config,
        scoring_config=scoring_config,
    )


# Module version
__version__ = "2.2.0"
