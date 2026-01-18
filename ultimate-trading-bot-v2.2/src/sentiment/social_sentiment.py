"""
Social Media Sentiment Analyzer for Ultimate Trading Bot v2.2.

Analyzes sentiment from social media platforms including Twitter, Reddit, and StockTwits.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from .base_analyzer import (
    BaseSentimentAnalyzer,
    SentimentConfig,
    SentimentLabel,
    SentimentResult,
    SentimentSource,
    AggregatedSentiment,
)
from .text_processor import TextProcessor, ProcessingConfig

logger = logging.getLogger(__name__)


class SocialPlatform(str, Enum):
    """Social media platforms."""
    TWITTER = "twitter"
    REDDIT = "reddit"
    STOCKTWITS = "stocktwits"
    DISCORD = "discord"
    TELEGRAM = "telegram"


@dataclass
class SocialPost:
    """Represents a social media post."""

    content: str
    platform: SocialPlatform
    author: str
    created_at: datetime
    post_id: str

    # Engagement metrics
    likes: int = 0
    shares: int = 0
    comments: int = 0
    views: int = 0

    # Optional fields
    symbols: list[str] = field(default_factory=list)
    hashtags: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    url: str | None = None
    parent_id: str | None = None  # For replies/comments

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def engagement_score(self) -> float:
        """Calculate engagement score."""
        return (
            self.likes +
            self.shares * 2 +
            self.comments * 1.5 +
            (self.views * 0.01 if self.views > 0 else 0)
        )

    @property
    def is_reply(self) -> bool:
        """Check if post is a reply."""
        return self.parent_id is not None


@dataclass
class AuthorProfile:
    """Profile of a social media author."""

    username: str
    platform: SocialPlatform
    followers: int = 0
    following: int = 0
    post_count: int = 0
    account_age_days: int = 0

    # Credibility metrics
    verified: bool = False
    credibility_score: float = 0.5
    sentiment_accuracy: float | None = None

    @property
    def influence_score(self) -> float:
        """Calculate influence score."""
        # Basic influence calculation
        follower_score = min(self.followers / 10000, 1.0)
        ratio = self.followers / max(self.following, 1)
        ratio_score = min(ratio / 10, 1.0)
        age_score = min(self.account_age_days / 365, 1.0)

        score = (follower_score * 0.4 + ratio_score * 0.3 + age_score * 0.3)

        if self.verified:
            score *= 1.5

        return min(score, 1.0)


@dataclass
class SocialSentimentResult(SentimentResult):
    """Extended sentiment result for social media."""

    post: SocialPost | None = None
    author: AuthorProfile | None = None
    engagement_weight: float = 1.0
    influence_weight: float = 1.0

    # Platform-specific metrics
    virality_score: float = 0.0
    spam_probability: float = 0.0


class SocialSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    Sentiment analyzer for social media content.

    Handles platform-specific nuances and engagement weighting.
    """

    # Platform weights
    PLATFORM_WEIGHTS = {
        SocialPlatform.TWITTER: 0.9,
        SocialPlatform.REDDIT: 0.85,
        SocialPlatform.STOCKTWITS: 1.0,
        SocialPlatform.DISCORD: 0.7,
        SocialPlatform.TELEGRAM: 0.7,
    }

    # Social media specific lexicon
    SOCIAL_LEXICON = {
        # Bullish slang
        "moon": 0.8, "mooning": 0.9, "rocket": 0.7, "lambo": 0.6,
        "tendies": 0.5, "yolo": 0.4, "diamond": 0.3, "hodl": 0.5,
        "ape": 0.3, "squeeze": 0.6, "breakout": 0.5, "bullish": 0.7,
        "pumping": 0.5, "green": 0.4, "calls": 0.3, "btd": 0.4,
        "dip": -0.1, "buying": 0.3, "accumulate": 0.4, "undervalued": 0.5,

        # Bearish slang
        "dump": -0.7, "dumping": -0.8, "crash": -0.8, "tank": -0.6,
        "bearish": -0.7, "puts": -0.3, "short": -0.4, "rekt": -0.6,
        "bagholding": -0.4, "red": -0.4, "falling": -0.5, "overvalued": -0.5,
        "scam": -0.8, "fraud": -0.9, "ponzi": -0.9, "sell": -0.3,
    }

    # Spam indicators
    SPAM_PATTERNS = [
        r'follow\s+(?:me|us|back)',
        r'dm\s+(?:me|us)\s+for',
        r'(?:buy|sell)\s+now',
        r'guaranteed\s+(?:profit|return)',
        r'(?:100|1000)x',
        r'free\s+(?:money|crypto|bitcoin)',
        r'click\s+(?:here|link)',
        r'limited\s+time\s+offer',
    ]

    def __init__(
        self,
        config: SentimentConfig | None = None,
        use_engagement_weighting: bool = True,
        use_influence_weighting: bool = True,
    ) -> None:
        """
        Initialize social sentiment analyzer.

        Args:
            config: Sentiment configuration
            use_engagement_weighting: Weight by engagement
            use_influence_weighting: Weight by author influence
        """
        super().__init__(config or SentimentConfig())

        self.use_engagement_weighting = use_engagement_weighting
        self.use_influence_weighting = use_influence_weighting

        # Text processor for social media
        self._text_processor = TextProcessor(ProcessingConfig(
            remove_urls=True,
            remove_emails=True,
            expand_hashtags=True,
            lowercase=True,
        ))

        # Compile spam patterns
        self._spam_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.SPAM_PATTERNS
        ]

    async def initialize(self) -> None:
        """Initialize the analyzer."""
        self._initialized = True
        logger.info("Initialized SocialSentimentAnalyzer")

    async def analyze(
        self,
        text: str,
        symbol: str | None = None,
    ) -> SentimentResult:
        """Analyze sentiment of text."""
        if not self._initialized:
            await self.initialize()

        # Process text
        processed = self._text_processor.process(text)

        # Calculate sentiment
        score = self._calculate_sentiment(processed.tokens)
        confidence = self._calculate_confidence(processed)

        return SentimentResult(
            score=score,
            label=self._score_to_label(score),
            confidence=confidence,
            source=SentimentSource.SOCIAL_MEDIA,
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

    async def analyze_post(
        self,
        post: SocialPost,
        author: AuthorProfile | None = None,
    ) -> SocialSentimentResult:
        """
        Analyze sentiment of a social media post.

        Args:
            post: Social media post
            author: Optional author profile

        Returns:
            Social sentiment result
        """
        if not self._initialized:
            await self.initialize()

        # Process text
        processed = self._text_processor.process(post.content)

        # Calculate base sentiment
        score = self._calculate_sentiment(processed.tokens)

        # Check for spam
        spam_prob = self._calculate_spam_probability(post.content)

        # Calculate weights
        engagement_weight = self._calculate_engagement_weight(post)
        influence_weight = (
            self._calculate_influence_weight(author)
            if author else 1.0
        )

        # Calculate virality
        virality = self._calculate_virality(post)

        # Adjust confidence based on spam
        base_confidence = self._calculate_confidence(processed)
        confidence = base_confidence * (1 - spam_prob * 0.5)

        # Apply platform weight
        platform_weight = self.PLATFORM_WEIGHTS.get(post.platform, 0.8)
        confidence *= platform_weight

        return SocialSentimentResult(
            score=score,
            label=self._score_to_label(score),
            confidence=confidence,
            source=self._platform_to_source(post.platform),
            timestamp=post.created_at,
            text=post.content,
            symbol=post.symbols[0] if post.symbols else None,
            post=post,
            author=author,
            engagement_weight=engagement_weight,
            influence_weight=influence_weight,
            virality_score=virality,
            spam_probability=spam_prob,
            metadata={
                "platform": post.platform.value,
                "engagement_score": post.engagement_score,
            },
        )

    async def analyze_posts(
        self,
        posts: list[SocialPost],
        authors: dict[str, AuthorProfile] | None = None,
    ) -> list[SocialSentimentResult]:
        """
        Analyze multiple social media posts.

        Args:
            posts: List of posts
            authors: Dictionary mapping usernames to profiles

        Returns:
            List of sentiment results
        """
        authors = authors or {}

        tasks = [
            self.analyze_post(
                post,
                authors.get(post.author)
            )
            for post in posts
        ]

        return await asyncio.gather(*tasks)

    async def aggregate_social_sentiment(
        self,
        posts: list[SocialPost],
        symbol: str | None = None,
        hours: int = 24,
        authors: dict[str, AuthorProfile] | None = None,
    ) -> AggregatedSentiment:
        """
        Aggregate sentiment from social media posts.

        Args:
            posts: List of posts
            symbol: Trading symbol filter
            hours: Time window in hours
            authors: Author profiles

        Returns:
            Aggregated sentiment
        """
        # Filter by time
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_posts = [p for p in posts if p.created_at >= cutoff]

        # Filter by symbol if provided
        if symbol:
            recent_posts = [
                p for p in recent_posts
                if symbol.upper() in [s.upper() for s in p.symbols]
            ]

        if not recent_posts:
            return AggregatedSentiment(
                overall_score=0.0,
                overall_label=SentimentLabel.NEUTRAL,
                overall_confidence=0.0,
                start_time=cutoff,
                end_time=datetime.now(),
                total_samples=0,
            )

        # Analyze all posts
        results = await self.analyze_posts(recent_posts, authors)

        # Filter out spam
        results = [r for r in results if r.spam_probability < 0.7]

        return self._aggregate_results(results, cutoff, datetime.now())

    def _calculate_sentiment(self, tokens: list[str]) -> float:
        """Calculate sentiment from tokens."""
        if not tokens:
            return 0.0

        score = 0.0
        matches = 0

        for token in tokens:
            token_lower = token.lower()
            if token_lower in self.SOCIAL_LEXICON:
                score += self.SOCIAL_LEXICON[token_lower]
                matches += 1

        if matches > 0:
            # Normalize by matches but cap it
            score = score / max(matches, 3)

        return max(-1.0, min(1.0, score))

    def _calculate_confidence(self, processed: Any) -> float:
        """Calculate confidence from processed text."""
        word_count = len(processed.tokens)

        # Social posts are shorter, adjust confidence accordingly
        if word_count < 5:
            return 0.3
        elif word_count < 15:
            return 0.5
        elif word_count < 50:
            return 0.7
        else:
            return 0.8

    def _calculate_spam_probability(self, text: str) -> float:
        """Calculate probability that text is spam."""
        spam_score = 0.0

        # Check patterns
        for pattern in self._spam_patterns:
            if pattern.search(text):
                spam_score += 0.3

        # Check for excessive caps
        if text.isupper() and len(text) > 20:
            spam_score += 0.2

        # Check for excessive exclamation marks
        exclaim_count = text.count('!')
        if exclaim_count > 5:
            spam_score += 0.2

        # Check for repeated characters
        if re.search(r'(.)\1{4,}', text):
            spam_score += 0.1

        return min(1.0, spam_score)

    def _calculate_engagement_weight(self, post: SocialPost) -> float:
        """Calculate weight based on engagement."""
        if not self.use_engagement_weighting:
            return 1.0

        engagement = post.engagement_score

        if engagement < 10:
            return 0.5
        elif engagement < 100:
            return 0.7
        elif engagement < 1000:
            return 0.9
        elif engagement < 10000:
            return 1.0
        else:
            return 1.2

    def _calculate_influence_weight(self, author: AuthorProfile) -> float:
        """Calculate weight based on author influence."""
        if not self.use_influence_weighting or author is None:
            return 1.0

        return 0.5 + author.influence_score * 0.5

    def _calculate_virality(self, post: SocialPost) -> float:
        """Calculate virality score for a post."""
        # Time-normalized engagement
        age_hours = max(
            (datetime.now() - post.created_at).total_seconds() / 3600,
            1
        )

        hourly_engagement = post.engagement_score / age_hours

        if hourly_engagement < 1:
            return 0.1
        elif hourly_engagement < 10:
            return 0.3
        elif hourly_engagement < 100:
            return 0.5
        elif hourly_engagement < 1000:
            return 0.7
        else:
            return 1.0

    def _platform_to_source(self, platform: SocialPlatform) -> SentimentSource:
        """Convert platform to sentiment source."""
        mapping = {
            SocialPlatform.TWITTER: SentimentSource.TWITTER,
            SocialPlatform.REDDIT: SentimentSource.REDDIT,
            SocialPlatform.STOCKTWITS: SentimentSource.STOCKTWITS,
        }
        return mapping.get(platform, SentimentSource.SOCIAL_MEDIA)

    def _aggregate_results(
        self,
        results: list[SocialSentimentResult],
        start_time: datetime,
        end_time: datetime,
    ) -> AggregatedSentiment:
        """Aggregate social sentiment results."""
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

        platform_scores: dict[str, list[float]] = {}
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        scores = []

        for result in results:
            # Calculate time decay
            age_hours = (now - result.timestamp).total_seconds() / 3600
            decay = self.config.decay_rate ** (age_hours / 24)

            # Calculate combined weight
            weight = (
                result.confidence *
                result.engagement_weight *
                result.influence_weight *
                decay
            )

            weighted_sum += result.score * weight
            total_weight += weight
            scores.append(result.score)

            # Track by platform
            if result.post:
                platform = result.post.platform.value
                if platform not in platform_scores:
                    platform_scores[platform] = []
                platform_scores[platform].append(result.score)

            # Count labels
            if result.label in [SentimentLabel.POSITIVE, SentimentLabel.VERY_POSITIVE]:
                positive_count += 1
            elif result.label in [SentimentLabel.NEGATIVE, SentimentLabel.VERY_NEGATIVE]:
                negative_count += 1
            else:
                neutral_count += 1

        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Calculate source averages
        source_avg = {
            platform: sum(s) / len(s)
            for platform, s in platform_scores.items()
        }

        # Calculate trend
        if len(results) >= 2:
            sorted_results = sorted(results, key=lambda r: r.timestamp)
            mid = len(sorted_results) // 2
            first_half = sorted_results[:mid]
            second_half = sorted_results[mid:]

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
            source_counts={p: len(s) for p, s in platform_scores.items()},
            total_samples=total_count,
            positive_ratio=positive_count / total_count if total_count > 0 else 0.0,
            negative_ratio=negative_count / total_count if total_count > 0 else 0.0,
            neutral_ratio=neutral_count / total_count if total_count > 0 else 0.0,
            trend=trend,
            volatility=volatility,
            results=results,
        )


class TwitterAnalyzer(SocialSentimentAnalyzer):
    """
    Specialized analyzer for Twitter/X content.
    """

    # Twitter-specific lexicon additions
    TWITTER_LEXICON = {
        "thread": 0.1, "rt": 0.0, "via": 0.0,
        "fomo": 0.3, "fud": -0.4, "ath": 0.5,
        "atl": -0.5, "dyor": 0.1, "nfa": 0.0,
        "iykyk": 0.2, "wagmi": 0.4, "ngmi": -0.4,
    }

    def __init__(self, config: SentimentConfig | None = None) -> None:
        """Initialize Twitter analyzer."""
        super().__init__(config)
        self.SOCIAL_LEXICON.update(self.TWITTER_LEXICON)


class RedditAnalyzer(SocialSentimentAnalyzer):
    """
    Specialized analyzer for Reddit content.
    """

    # Reddit-specific lexicon additions
    REDDIT_LEXICON = {
        "dd": 0.3, "tldr": 0.0, "eli5": 0.0,
        "loss porn": -0.3, "gain porn": 0.3,
        "smooth brain": -0.1, "wrinkle brain": 0.2,
        "wife's boyfriend": 0.0, "bagholder": -0.3,
    }

    def __init__(self, config: SentimentConfig | None = None) -> None:
        """Initialize Reddit analyzer."""
        super().__init__(config)
        self.SOCIAL_LEXICON.update(self.REDDIT_LEXICON)

    async def analyze_thread(
        self,
        posts: list[SocialPost],
        authors: dict[str, AuthorProfile] | None = None,
    ) -> AggregatedSentiment:
        """
        Analyze a Reddit thread.

        Args:
            posts: Posts in the thread (first is OP)
            authors: Author profiles

        Returns:
            Aggregated sentiment
        """
        if not posts:
            return AggregatedSentiment(
                overall_score=0.0,
                overall_label=SentimentLabel.NEUTRAL,
                overall_confidence=0.0,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )

        results = await self.analyze_posts(posts, authors)

        # Weight OP higher
        if results:
            results[0].confidence *= 1.5

        return self._aggregate_results(
            results,
            posts[0].created_at,
            posts[-1].created_at,
        )


class StockTwitsAnalyzer(SocialSentimentAnalyzer):
    """
    Specialized analyzer for StockTwits content.
    """

    # StockTwits-specific lexicon
    STOCKTWITS_LEXICON = {
        "bullish": 0.8, "bearish": -0.8,
        "long": 0.5, "short": -0.5,
        "oversold": 0.3, "overbought": -0.3,
        "support": 0.2, "resistance": -0.1,
    }

    def __init__(self, config: SentimentConfig | None = None) -> None:
        """Initialize StockTwits analyzer."""
        super().__init__(config)
        self.SOCIAL_LEXICON.update(self.STOCKTWITS_LEXICON)


def create_social_analyzer(
    platform: str = "general",
    config: SentimentConfig | None = None,
) -> SocialSentimentAnalyzer:
    """
    Factory function to create social sentiment analyzers.

    Args:
        platform: Target platform
        config: Sentiment configuration

    Returns:
        Social sentiment analyzer instance
    """
    analyzers = {
        "general": SocialSentimentAnalyzer,
        "twitter": TwitterAnalyzer,
        "reddit": RedditAnalyzer,
        "stocktwits": StockTwitsAnalyzer,
    }

    analyzer_class = analyzers.get(platform.lower(), SocialSentimentAnalyzer)
    return analyzer_class(config)
