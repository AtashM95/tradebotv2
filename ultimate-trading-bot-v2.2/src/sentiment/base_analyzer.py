"""
Base Sentiment Analyzer for Ultimate Trading Bot v2.2.

Provides abstract base classes and common utilities for sentiment analysis.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SentimentLabel(str, Enum):
    """Sentiment classification labels."""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class SentimentSource(str, Enum):
    """Sources of sentiment data."""
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    TWITTER = "twitter"
    REDDIT = "reddit"
    STOCKTWITS = "stocktwits"
    ANALYST = "analyst"
    MARKET = "market"
    COMBINED = "combined"


@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis."""

    # Analysis settings
    min_confidence: float = 0.5
    aggregation_window: int = 24  # hours
    decay_rate: float = 0.95
    use_volume_weighting: bool = True

    # Thresholds
    positive_threshold: float = 0.3
    negative_threshold: float = -0.3
    strong_threshold: float = 0.6

    # Processing settings
    max_text_length: int = 1000
    min_text_length: int = 10
    language: str = "en"

    # Source weights
    source_weights: dict[str, float] = field(default_factory=lambda: {
        "news": 1.0,
        "twitter": 0.8,
        "reddit": 0.7,
        "stocktwits": 0.6,
        "analyst": 1.2,
        "market": 0.9,
    })


@dataclass
class SentimentResult:
    """Result from sentiment analysis."""

    # Core metrics
    score: float  # -1 to 1
    label: SentimentLabel
    confidence: float  # 0 to 1

    # Source information
    source: SentimentSource
    timestamp: datetime

    # Additional details
    text: str | None = None
    symbol: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Component scores (for multi-aspect sentiment)
    aspects: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "label": self.label.value,
            "confidence": self.confidence,
            "source": self.source.value,
            "timestamp": self.timestamp.isoformat(),
            "text": self.text,
            "symbol": self.symbol,
            "metadata": self.metadata,
            "aspects": self.aspects,
        }


@dataclass
class AggregatedSentiment:
    """Aggregated sentiment from multiple sources."""

    # Overall metrics
    overall_score: float
    overall_label: SentimentLabel
    overall_confidence: float

    # Time period
    start_time: datetime
    end_time: datetime

    # Source breakdown
    source_scores: dict[str, float] = field(default_factory=dict)
    source_counts: dict[str, int] = field(default_factory=dict)

    # Statistics
    total_samples: int = 0
    positive_ratio: float = 0.0
    negative_ratio: float = 0.0
    neutral_ratio: float = 0.0

    # Trend
    trend: float = 0.0  # Change in sentiment over period
    volatility: float = 0.0  # Sentiment volatility

    # Individual results
    results: list[SentimentResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": self.overall_score,
            "overall_label": self.overall_label.value,
            "overall_confidence": self.overall_confidence,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "source_scores": self.source_scores,
            "source_counts": self.source_counts,
            "total_samples": self.total_samples,
            "positive_ratio": self.positive_ratio,
            "negative_ratio": self.negative_ratio,
            "neutral_ratio": self.neutral_ratio,
            "trend": self.trend,
            "volatility": self.volatility,
        }


@dataclass
class SentimentSignal:
    """Trading signal derived from sentiment."""

    # Signal details
    direction: int  # 1: bullish, -1: bearish, 0: neutral
    strength: float  # 0 to 1
    confidence: float

    # Underlying sentiment
    sentiment_score: float
    sentiment_label: SentimentLabel

    # Metadata
    symbol: str
    timestamp: datetime
    sources: list[SentimentSource] = field(default_factory=list)

    # Supporting data
    reasoning: str = ""
    indicators: dict[str, float] = field(default_factory=dict)

    def is_bullish(self) -> bool:
        """Check if signal is bullish."""
        return self.direction > 0

    def is_bearish(self) -> bool:
        """Check if signal is bearish."""
        return self.direction < 0

    def is_strong(self) -> bool:
        """Check if signal is strong."""
        return self.strength >= 0.7


class BaseSentimentAnalyzer(ABC):
    """
    Abstract base class for sentiment analyzers.

    Provides common interface for all sentiment analysis implementations.
    """

    def __init__(self, config: SentimentConfig) -> None:
        """
        Initialize sentiment analyzer.

        Args:
            config: Sentiment configuration
        """
        self.config = config
        self._initialized = False
        self._analysis_count = 0
        self._cache: dict[str, SentimentResult] = {}

        logger.info(f"Initialized {self.__class__.__name__}")

    @property
    def is_initialized(self) -> bool:
        """Check if analyzer is initialized."""
        return self._initialized

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the analyzer."""
        pass

    @abstractmethod
    async def analyze(
        self,
        text: str,
        symbol: str | None = None,
    ) -> SentimentResult:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze
            symbol: Optional trading symbol for context

        Returns:
            Sentiment result
        """
        pass

    @abstractmethod
    async def analyze_batch(
        self,
        texts: list[str],
        symbol: str | None = None,
    ) -> list[SentimentResult]:
        """
        Analyze sentiment of multiple texts.

        Args:
            texts: List of texts to analyze
            symbol: Optional trading symbol for context

        Returns:
            List of sentiment results
        """
        pass

    def _score_to_label(self, score: float) -> SentimentLabel:
        """
        Convert numerical score to sentiment label.

        Args:
            score: Sentiment score (-1 to 1)

        Returns:
            Sentiment label
        """
        if score <= -self.config.strong_threshold:
            return SentimentLabel.VERY_NEGATIVE
        elif score <= -self.config.negative_threshold:
            return SentimentLabel.NEGATIVE
        elif score >= self.config.strong_threshold:
            return SentimentLabel.VERY_POSITIVE
        elif score >= self.config.positive_threshold:
            return SentimentLabel.POSITIVE
        else:
            return SentimentLabel.NEUTRAL

    def _validate_text(self, text: str) -> str | None:
        """
        Validate and clean text for analysis.

        Args:
            text: Raw text input

        Returns:
            Cleaned text or None if invalid
        """
        if not text:
            return None

        # Strip whitespace
        text = text.strip()

        # Check length
        if len(text) < self.config.min_text_length:
            return None

        # Truncate if needed
        if len(text) > self.config.max_text_length:
            text = text[:self.config.max_text_length]

        return text

    def _compute_cache_key(self, text: str, symbol: str | None) -> str:
        """Compute cache key for text."""
        import hashlib
        content = f"{text}:{symbol or ''}"
        return hashlib.md5(content.encode()).hexdigest()

    def get_cached_result(
        self,
        text: str,
        symbol: str | None = None,
    ) -> SentimentResult | None:
        """Get cached sentiment result."""
        key = self._compute_cache_key(text, symbol)
        return self._cache.get(key)

    def cache_result(
        self,
        text: str,
        result: SentimentResult,
        symbol: str | None = None,
    ) -> None:
        """Cache sentiment result."""
        key = self._compute_cache_key(text, symbol)
        self._cache[key] = result

    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()

    async def shutdown(self) -> None:
        """Shutdown the analyzer."""
        self.clear_cache()
        self._initialized = False
        logger.info(f"Shutdown {self.__class__.__name__}")


class LexiconBasedAnalyzer(BaseSentimentAnalyzer):
    """
    Lexicon-based sentiment analyzer.

    Uses predefined word lists for sentiment classification.
    """

    def __init__(self, config: SentimentConfig) -> None:
        """
        Initialize lexicon-based analyzer.

        Args:
            config: Sentiment configuration
        """
        super().__init__(config)

        # Word sentiment lexicons
        self._positive_words: set[str] = set()
        self._negative_words: set[str] = set()
        self._intensifiers: dict[str, float] = {}
        self._negators: set[str] = set()

        # Finance-specific lexicons
        self._bullish_words: set[str] = set()
        self._bearish_words: set[str] = set()

    async def initialize(self) -> None:
        """Initialize lexicons."""
        # Basic positive words
        self._positive_words = {
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "positive", "strong", "bullish", "gain", "profit", "growth",
            "success", "improve", "beat", "exceed", "surge", "rally",
            "outperform", "upgrade", "opportunity", "optimistic", "confident",
            "breakthrough", "innovative", "efficient", "productive", "stable",
        }

        # Basic negative words
        self._negative_words = {
            "bad", "terrible", "awful", "poor", "weak", "bearish", "loss",
            "decline", "drop", "fall", "crash", "fail", "miss", "cut",
            "downgrade", "risk", "concern", "warning", "pessimistic", "fear",
            "uncertain", "volatile", "bankruptcy", "default", "scandal",
            "fraud", "investigation", "lawsuit", "layoff", "recession",
        }

        # Intensifiers
        self._intensifiers = {
            "very": 1.5, "extremely": 2.0, "highly": 1.5, "really": 1.3,
            "absolutely": 2.0, "significantly": 1.5, "substantially": 1.5,
            "slightly": 0.5, "somewhat": 0.7, "marginally": 0.5,
        }

        # Negators
        self._negators = {
            "not", "no", "never", "neither", "nobody", "nothing",
            "nowhere", "hardly", "scarcely", "barely", "doesn't",
            "don't", "didn't", "won't", "wouldn't", "couldn't",
        }

        # Finance bullish words
        self._bullish_words = {
            "buy", "long", "bullish", "accumulate", "upgrade", "outperform",
            "overweight", "growth", "expansion", "dividend", "earnings",
            "revenue", "profit", "margin", "acquisition", "partnership",
        }

        # Finance bearish words
        self._bearish_words = {
            "sell", "short", "bearish", "avoid", "downgrade", "underperform",
            "underweight", "contraction", "layoff", "debt", "loss",
            "deficit", "decline", "warning", "guidance", "bankruptcy",
        }

        self._initialized = True
        logger.info("Initialized lexicon-based analyzer")

    async def analyze(
        self,
        text: str,
        symbol: str | None = None,
    ) -> SentimentResult:
        """Analyze text sentiment using lexicons."""
        if not self._initialized:
            await self.initialize()

        # Validate text
        cleaned_text = self._validate_text(text)
        if cleaned_text is None:
            return SentimentResult(
                score=0.0,
                label=SentimentLabel.NEUTRAL,
                confidence=0.0,
                source=SentimentSource.NEWS,
                timestamp=datetime.now(),
                text=text,
                symbol=symbol,
            )

        # Check cache
        cached = self.get_cached_result(cleaned_text, symbol)
        if cached is not None:
            return cached

        # Tokenize
        words = cleaned_text.lower().split()

        # Calculate sentiment scores
        positive_score = 0.0
        negative_score = 0.0
        total_words = 0
        negation_active = False
        current_intensifier = 1.0

        for i, word in enumerate(words):
            # Check for negators
            if word in self._negators:
                negation_active = True
                continue

            # Check for intensifiers
            if word in self._intensifiers:
                current_intensifier = self._intensifiers[word]
                continue

            # Score the word
            word_score = 0.0
            if word in self._positive_words or word in self._bullish_words:
                word_score = 1.0
            elif word in self._negative_words or word in self._bearish_words:
                word_score = -1.0

            if word_score != 0.0:
                # Apply intensifier
                word_score *= current_intensifier

                # Apply negation
                if negation_active:
                    word_score *= -0.5

                # Accumulate
                if word_score > 0:
                    positive_score += word_score
                else:
                    negative_score += abs(word_score)

                total_words += 1
                negation_active = False
                current_intensifier = 1.0

        # Calculate final score
        if total_words > 0:
            score = (positive_score - negative_score) / (positive_score + negative_score + 1)
            confidence = min(total_words / 10.0, 1.0)  # More words = higher confidence
        else:
            score = 0.0
            confidence = 0.0

        # Clamp score
        score = max(-1.0, min(1.0, score))

        result = SentimentResult(
            score=score,
            label=self._score_to_label(score),
            confidence=confidence,
            source=SentimentSource.NEWS,
            timestamp=datetime.now(),
            text=cleaned_text,
            symbol=symbol,
            metadata={
                "positive_score": positive_score,
                "negative_score": negative_score,
                "word_count": total_words,
            },
        )

        # Cache result
        self.cache_result(cleaned_text, result, symbol)
        self._analysis_count += 1

        return result

    async def analyze_batch(
        self,
        texts: list[str],
        symbol: str | None = None,
    ) -> list[SentimentResult]:
        """Analyze multiple texts."""
        results = []
        for text in texts:
            result = await self.analyze(text, symbol)
            results.append(result)
        return results


class VADERAnalyzer(BaseSentimentAnalyzer):
    """
    VADER-style sentiment analyzer.

    Valence Aware Dictionary and sEntiment Reasoner.
    Optimized for social media text.
    """

    def __init__(self, config: SentimentConfig) -> None:
        """
        Initialize VADER-style analyzer.

        Args:
            config: Sentiment configuration
        """
        super().__init__(config)

        # Lexicon with valence scores
        self._lexicon: dict[str, float] = {}

        # Special patterns
        self._booster_dict: dict[str, float] = {}
        self._negate: list[str] = []

        # Constants
        self._B_INCR = 0.293
        self._B_DECR = -0.293
        self._C_INCR = 0.733
        self._N_SCALAR = -0.74

    async def initialize(self) -> None:
        """Initialize VADER lexicon."""
        # Sample finance-focused lexicon with valence scores
        self._lexicon = {
            # Strongly positive
            "excellent": 3.2, "outstanding": 3.1, "exceptional": 3.0,
            "breakthrough": 2.9, "soaring": 2.8, "surge": 2.7,

            # Moderately positive
            "good": 1.9, "profit": 1.8, "growth": 1.7, "gain": 1.6,
            "strong": 1.5, "bullish": 2.0, "rally": 1.8, "upgrade": 1.9,
            "beat": 1.7, "outperform": 1.8, "positive": 1.5,

            # Mildly positive
            "ok": 0.9, "decent": 0.8, "stable": 0.7, "steady": 0.6,

            # Mildly negative
            "concern": -0.9, "uncertain": -0.8, "volatile": -0.7,
            "miss": -0.9, "slow": -0.6,

            # Moderately negative
            "bad": -1.9, "loss": -1.8, "decline": -1.7, "drop": -1.6,
            "weak": -1.5, "bearish": -2.0, "downgrade": -1.9,
            "underperform": -1.8, "warning": -1.7, "risk": -1.5,

            # Strongly negative
            "terrible": -3.2, "crash": -3.1, "collapse": -3.0,
            "bankruptcy": -3.0, "fraud": -3.2, "scandal": -2.9,
        }

        # Booster words
        self._booster_dict = {
            "absolutely": self._B_INCR, "amazingly": self._B_INCR,
            "extremely": self._B_INCR, "incredibly": self._B_INCR,
            "really": self._B_INCR, "very": self._B_INCR,
            "highly": self._B_INCR, "significantly": self._B_INCR,
            "somewhat": self._B_DECR, "slightly": self._B_DECR,
            "marginally": self._B_DECR, "barely": self._B_DECR,
        }

        # Negation words
        self._negate = [
            "not", "isn't", "doesn't", "wasn't", "shouldn't",
            "won't", "wouldn't", "couldn't", "never", "neither",
        ]

        self._initialized = True
        logger.info("Initialized VADER-style analyzer")

    def _is_negated(self, words: list[str], index: int) -> bool:
        """Check if word at index is negated."""
        # Check previous 3 words
        for i in range(max(0, index - 3), index):
            if words[i].lower() in self._negate:
                return True
        return False

    def _get_valence(self, word: str) -> float:
        """Get valence score for a word."""
        return self._lexicon.get(word.lower(), 0.0)

    def _apply_booster(self, words: list[str], index: int, valence: float) -> float:
        """Apply booster words."""
        if index > 0:
            prev_word = words[index - 1].lower()
            if prev_word in self._booster_dict:
                booster = self._booster_dict[prev_word]
                if valence > 0:
                    valence += booster
                else:
                    valence -= booster
        return valence

    async def analyze(
        self,
        text: str,
        symbol: str | None = None,
    ) -> SentimentResult:
        """Analyze text using VADER-style analysis."""
        if not self._initialized:
            await self.initialize()

        cleaned_text = self._validate_text(text)
        if cleaned_text is None:
            return SentimentResult(
                score=0.0,
                label=SentimentLabel.NEUTRAL,
                confidence=0.0,
                source=SentimentSource.SOCIAL_MEDIA,
                timestamp=datetime.now(),
                text=text,
                symbol=symbol,
            )

        # Check cache
        cached = self.get_cached_result(cleaned_text, symbol)
        if cached is not None:
            return cached

        # Tokenize
        words = cleaned_text.split()

        # Calculate sentiment
        sentiments: list[float] = []

        for i, word in enumerate(words):
            valence = self._get_valence(word)

            if valence != 0.0:
                # Apply booster
                valence = self._apply_booster(words, i, valence)

                # Apply negation
                if self._is_negated(words, i):
                    valence *= self._N_SCALAR

                sentiments.append(valence)

        # Calculate compound score
        if sentiments:
            sum_s = sum(sentiments)
            # Normalize to -1 to 1 range
            compound = sum_s / np.sqrt(sum_s * sum_s + 15)
            confidence = min(len(sentiments) / 5.0, 1.0)
        else:
            compound = 0.0
            confidence = 0.0

        # Calculate pos/neg/neu ratios
        if sentiments:
            pos_sum = sum(s for s in sentiments if s > 0)
            neg_sum = sum(abs(s) for s in sentiments if s < 0)
            total = pos_sum + neg_sum + 0.001
            pos_ratio = pos_sum / total
            neg_ratio = neg_sum / total
            neu_ratio = 1.0 - pos_ratio - neg_ratio
        else:
            pos_ratio = neg_ratio = 0.0
            neu_ratio = 1.0

        result = SentimentResult(
            score=compound,
            label=self._score_to_label(compound),
            confidence=confidence,
            source=SentimentSource.SOCIAL_MEDIA,
            timestamp=datetime.now(),
            text=cleaned_text,
            symbol=symbol,
            aspects={
                "positive": pos_ratio,
                "negative": neg_ratio,
                "neutral": neu_ratio,
            },
        )

        self.cache_result(cleaned_text, result, symbol)
        self._analysis_count += 1

        return result

    async def analyze_batch(
        self,
        texts: list[str],
        symbol: str | None = None,
    ) -> list[SentimentResult]:
        """Analyze multiple texts."""
        results = []
        for text in texts:
            result = await self.analyze(text, symbol)
            results.append(result)
        return results


class FinancialSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    Specialized sentiment analyzer for financial text.

    Includes finance-specific vocabulary and context handling.
    """

    def __init__(self, config: SentimentConfig) -> None:
        """
        Initialize financial sentiment analyzer.

        Args:
            config: Sentiment configuration
        """
        super().__init__(config)

        self._finance_lexicon: dict[str, float] = {}
        self._entity_sentiments: dict[str, float] = {}
        self._context_modifiers: dict[str, float] = {}

    async def initialize(self) -> None:
        """Initialize financial lexicons."""
        # Comprehensive finance lexicon
        self._finance_lexicon = {
            # Earnings related
            "beat": 1.5, "exceed": 1.4, "miss": -1.5, "disappoint": -1.4,
            "guidance": 0.0, "forecast": 0.0, "estimate": 0.0,

            # Growth related
            "growth": 1.2, "expansion": 1.1, "contraction": -1.2,
            "acceleration": 1.3, "deceleration": -1.1,

            # Valuation
            "undervalued": 1.0, "overvalued": -0.8, "fair value": 0.0,

            # Technical
            "breakout": 1.4, "breakdown": -1.4, "support": 0.5,
            "resistance": -0.3, "momentum": 0.8,

            # Fundamental
            "profit": 1.0, "loss": -1.0, "revenue": 0.5, "debt": -0.5,
            "margin": 0.6, "dividend": 0.8, "buyback": 0.7,

            # Rating actions
            "upgrade": 1.6, "downgrade": -1.6, "initiate": 0.3,
            "reiterate": 0.1, "maintain": 0.0,

            # Market conditions
            "bull": 1.2, "bear": -1.2, "rally": 1.3, "correction": -0.8,
            "crash": -2.0, "bubble": -1.0, "recovery": 1.0,

            # Company actions
            "acquisition": 0.5, "merger": 0.4, "spinoff": 0.3,
            "layoff": -1.0, "restructuring": -0.5, "bankruptcy": -2.5,

            # Economic
            "inflation": -0.7, "deflation": -0.5, "recession": -1.5,
            "stimulus": 0.8, "rate hike": -0.6, "rate cut": 0.6,
        }

        # Context modifiers
        self._context_modifiers = {
            "despite": -0.5,  # Reverses following sentiment
            "although": -0.3,
            "however": -0.5,
            "but": -0.4,
            "expected": -0.3,  # Reduces impact if expected
            "unexpected": 0.5,  # Increases impact if unexpected
            "surprise": 0.4,
        }

        self._initialized = True
        logger.info("Initialized financial sentiment analyzer")

    async def analyze(
        self,
        text: str,
        symbol: str | None = None,
    ) -> SentimentResult:
        """Analyze financial text sentiment."""
        if not self._initialized:
            await self.initialize()

        cleaned_text = self._validate_text(text)
        if cleaned_text is None:
            return SentimentResult(
                score=0.0,
                label=SentimentLabel.NEUTRAL,
                confidence=0.0,
                source=SentimentSource.NEWS,
                timestamp=datetime.now(),
                text=text,
                symbol=symbol,
            )

        # Check cache
        cached = self.get_cached_result(cleaned_text, symbol)
        if cached is not None:
            return cached

        # Tokenize
        words = cleaned_text.lower().split()
        text_lower = cleaned_text.lower()

        # Calculate sentiment
        total_score = 0.0
        matches = 0
        context_modifier = 1.0

        # Check for context modifiers
        for modifier, value in self._context_modifiers.items():
            if modifier in text_lower:
                context_modifier *= (1.0 + value)

        # Score individual words
        for word in words:
            if word in self._finance_lexicon:
                score = self._finance_lexicon[word]
                total_score += score
                matches += 1

        # Check for phrases
        phrases = [
            ("beat expectations", 1.5),
            ("miss expectations", -1.5),
            ("strong earnings", 1.3),
            ("weak earnings", -1.3),
            ("record revenue", 1.4),
            ("revenue decline", -1.2),
            ("raised guidance", 1.4),
            ("lowered guidance", -1.4),
            ("price target", 0.0),
            ("buy rating", 1.2),
            ("sell rating", -1.2),
            ("hold rating", 0.0),
        ]

        for phrase, score in phrases:
            if phrase in text_lower:
                total_score += score
                matches += 1

        # Apply context modifier
        total_score *= context_modifier

        # Calculate final score
        if matches > 0:
            score = total_score / (matches + 2)  # Dampen by match count
            score = max(-1.0, min(1.0, score))
            confidence = min(matches / 5.0, 1.0)
        else:
            score = 0.0
            confidence = 0.0

        result = SentimentResult(
            score=score,
            label=self._score_to_label(score),
            confidence=confidence,
            source=SentimentSource.NEWS,
            timestamp=datetime.now(),
            text=cleaned_text,
            symbol=symbol,
            metadata={
                "matches": matches,
                "context_modifier": context_modifier,
                "raw_score": total_score,
            },
        )

        self.cache_result(cleaned_text, result, symbol)
        self._analysis_count += 1

        return result

    async def analyze_batch(
        self,
        texts: list[str],
        symbol: str | None = None,
    ) -> list[SentimentResult]:
        """Analyze multiple texts."""
        results = []
        for text in texts:
            result = await self.analyze(text, symbol)
            results.append(result)
        return results


def create_sentiment_analyzer(
    analyzer_type: str = "financial",
    config: SentimentConfig | None = None,
) -> BaseSentimentAnalyzer:
    """
    Factory function to create sentiment analyzers.

    Args:
        analyzer_type: Type of analyzer to create
        config: Optional configuration

    Returns:
        Sentiment analyzer instance
    """
    if config is None:
        config = SentimentConfig()

    analyzers = {
        "lexicon": LexiconBasedAnalyzer,
        "vader": VADERAnalyzer,
        "financial": FinancialSentimentAnalyzer,
    }

    if analyzer_type not in analyzers:
        raise ValueError(f"Unknown analyzer type: {analyzer_type}")

    return analyzers[analyzer_type](config)
