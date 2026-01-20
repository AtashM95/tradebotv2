"""
Sentiment Analyzer for news and social media content using GPT.
~500 lines
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from .openai_client import OpenAIClient
from .prompt_manager import PromptManager
from .response_validator import ResponseValidator
from config.openai_config import OpenAIConfig

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Analyzes sentiment of news articles and social media posts using GPT.

    Features:
    - Multi-dimensional sentiment (bullish/bearish/neutral)
    - Confidence scoring
    - Key entity extraction
    - Sentiment justification
    - Batch processing support
    - Caching for repeated content
    """

    SENTIMENT_SCHEMA = {
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["bullish", "bearish", "neutral", "mixed"]
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0
            },
            "score": {
                "type": "number",
                "minimum": -1.0,
                "maximum": 1.0
            },
            "reasoning": {"type": "string"},
            "entities": {
                "type": "array",
                "items": {"type": "string"}
            },
            "key_phrases": {
                "type": "array",
                "items": {"type": "string"}
            },
            "impact_level": {
                "type": "string",
                "enum": ["low", "medium", "high"]
            }
        },
        "required": ["sentiment", "confidence", "score"]
    }

    def __init__(self, config: Optional[OpenAIConfig] = None):
        self.client = OpenAIClient(config)
        self.prompt_manager = PromptManager()
        self.validator = ResponseValidator()
        self.cache: Dict[str, Dict[str, Any]] = {}

    def analyze_text(
        self,
        text: str,
        context: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text to analyze
            context: Optional context (e.g., "news", "twitter", "reddit")
            symbol: Optional stock symbol for context

        Returns:
            Sentiment analysis result with metadata
        """
        if not self.client.is_available():
            logger.warning("OpenAI not available - returning neutral sentiment")
            return self._get_fallback_sentiment()

        # Check cache
        cache_key = self._get_cache_key(text, symbol)
        if cache_key in self.cache:
            logger.debug(f"Cache hit for sentiment analysis")
            return self.cache[cache_key]

        # Build prompt
        prompt = self._build_sentiment_prompt(text, context, symbol)

        # Make request
        try:
            messages = [
                {"role": "system", "content": self.prompt_manager.get_prompt("sentiment_system")},
                {"role": "user", "content": prompt}
            ]

            response = self.client.chat_completion(
                messages,
                temperature=0.3,  # Lower temperature for more consistent results
                max_tokens=500
            )

            if not response:
                return self._get_fallback_sentiment()

            # Parse and validate response
            result = self._parse_sentiment_response(response)

            # Validate against schema
            if not self.validator.validate(result, self.SENTIMENT_SCHEMA):
                logger.warning("Invalid sentiment response, retrying with stricter prompt")
                # Retry once with stricter prompt
                result = self._retry_with_strict_prompt(text, context, symbol)

            # Add metadata
            result["analyzed_at"] = datetime.now().isoformat()
            result["text_length"] = len(text)
            if symbol:
                result["symbol"] = symbol

            # Cache result
            self.cache[cache_key] = result

            return result

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return self._get_fallback_sentiment()

    def analyze_news_article(
        self,
        headline: str,
        body: str,
        symbol: Optional[str] = None,
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze sentiment of a news article.

        Args:
            headline: Article headline
            body: Article body text
            symbol: Related stock symbol
            source: News source

        Returns:
            Sentiment analysis with article-specific metadata
        """
        # Combine headline and body (prioritize headline)
        combined_text = f"Headline: {headline}\n\nBody: {body[:1000]}"  # Limit body to 1000 chars

        result = self.analyze_text(
            combined_text,
            context="news",
            symbol=symbol
        )

        # Add article-specific metadata
        result["headline"] = headline
        if source:
            result["source"] = source

        return result

    def analyze_social_post(
        self,
        text: str,
        platform: str,
        symbol: Optional[str] = None,
        author_followers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze sentiment of social media post.

        Args:
            text: Post text
            platform: Platform name (twitter, reddit, etc.)
            symbol: Related stock symbol
            author_followers: Number of followers (for weighting)

        Returns:
            Sentiment analysis with social-specific metadata
        """
        result = self.analyze_text(
            text,
            context=f"social_{platform}",
            symbol=symbol
        )

        # Add social-specific metadata
        result["platform"] = platform
        if author_followers:
            result["author_followers"] = author_followers
            # Weight by reach
            result["weighted_score"] = self._calculate_weighted_score(
                result["score"],
                author_followers
            )

        return result

    def analyze_batch(
        self,
        texts: List[str],
        context: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple texts in batch.

        Args:
            texts: List of texts to analyze
            context: Optional context
            symbol: Optional symbol

        Returns:
            List of sentiment analysis results
        """
        results = []
        for text in texts:
            result = self.analyze_text(text, context, symbol)
            results.append(result)

        return results

    def aggregate_sentiment(
        self,
        analyses: List[Dict[str, Any]],
        weighting: str = "uniform"
    ) -> Dict[str, Any]:
        """
        Aggregate multiple sentiment analyses into overall sentiment.

        Args:
            analyses: List of sentiment analysis results
            weighting: Weighting strategy ("uniform", "confidence", "followers")

        Returns:
            Aggregated sentiment analysis
        """
        if not analyses:
            return self._get_fallback_sentiment()

        # Calculate weights
        weights = self._calculate_weights(analyses, weighting)

        # Weighted average of scores
        weighted_score = sum(
            a["score"] * w for a, w in zip(analyses, weights)
        ) / sum(weights)

        # Count sentiment distribution
        sentiment_counts = {
            "bullish": 0,
            "bearish": 0,
            "neutral": 0,
            "mixed": 0
        }
        for analysis in analyses:
            sentiment_counts[analysis["sentiment"]] += 1

        # Determine overall sentiment
        if weighted_score > 0.3:
            overall_sentiment = "bullish"
        elif weighted_score < -0.3:
            overall_sentiment = "bearish"
        elif abs(weighted_score) < 0.1:
            overall_sentiment = "neutral"
        else:
            overall_sentiment = "mixed"

        return {
            "sentiment": overall_sentiment,
            "score": weighted_score,
            "confidence": sum(a["confidence"] * w for a, w in zip(analyses, weights)) / sum(weights),
            "sample_size": len(analyses),
            "distribution": sentiment_counts,
            "aggregated_at": datetime.now().isoformat(),
            "weighting_strategy": weighting
        }

    def get_symbol_sentiment(
        self,
        symbol: str,
        news_items: List[Dict[str, str]],
        social_posts: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Get comprehensive sentiment for a symbol from multiple sources.

        Args:
            symbol: Stock symbol
            news_items: List of news items (headline, body)
            social_posts: List of social posts (text, platform)

        Returns:
            Comprehensive sentiment analysis
        """
        # Analyze news
        news_sentiments = []
        for item in news_items[:10]:  # Limit to 10 most recent
            sentiment = self.analyze_news_article(
                headline=item.get("headline", ""),
                body=item.get("body", ""),
                symbol=symbol,
                source=item.get("source")
            )
            news_sentiments.append(sentiment)

        # Analyze social
        social_sentiments = []
        for post in social_posts[:20]:  # Limit to 20 most recent
            sentiment = self.analyze_social_post(
                text=post.get("text", ""),
                platform=post.get("platform", "unknown"),
                symbol=symbol,
                author_followers=post.get("author_followers")
            )
            social_sentiments.append(sentiment)

        # Aggregate separately
        news_aggregate = self.aggregate_sentiment(news_sentiments, weighting="confidence")
        social_aggregate = self.aggregate_sentiment(social_sentiments, weighting="followers")

        # Overall aggregate (news weighted 60%, social 40%)
        overall_score = (
            news_aggregate.get("score", 0) * 0.6 +
            social_aggregate.get("score", 0) * 0.4
        )

        if overall_score > 0.3:
            overall_sentiment = "bullish"
        elif overall_score < -0.3:
            overall_sentiment = "bearish"
        else:
            overall_sentiment = "neutral"

        return {
            "symbol": symbol,
            "sentiment": overall_sentiment,
            "score": overall_score,
            "news": news_aggregate,
            "social": social_aggregate,
            "analyzed_at": datetime.now().isoformat()
        }

    def _build_sentiment_prompt(
        self,
        text: str,
        context: Optional[str],
        symbol: Optional[str]
    ) -> str:
        """Build sentiment analysis prompt."""
        prompt = f"Analyze the sentiment of the following text"

        if symbol:
            prompt += f" regarding {symbol}"

        if context:
            prompt += f" (source: {context})"

        prompt += ":\n\n"
        prompt += f'"{text}"\n\n'
        prompt += "Provide analysis in JSON format with:\n"
        prompt += '- sentiment: "bullish", "bearish", "neutral", or "mixed"\n'
        prompt += "- confidence: 0.0 to 1.0\n"
        prompt += "- score: -1.0 (very bearish) to 1.0 (very bullish)\n"
        prompt += "- reasoning: brief explanation\n"
        prompt += "- entities: list of mentioned companies/symbols\n"
        prompt += "- key_phrases: important phrases\n"
        prompt += '- impact_level: "low", "medium", or "high"\n'

        return prompt

    def _parse_sentiment_response(self, response: str) -> Dict[str, Any]:
        """Parse sentiment response from GPT."""
        try:
            # Try to parse as JSON
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()

            result = json.loads(json_str)
            return result

        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response, using fallback")
            return self._get_fallback_sentiment()

    def _retry_with_strict_prompt(
        self,
        text: str,
        context: Optional[str],
        symbol: Optional[str]
    ) -> Dict[str, Any]:
        """Retry analysis with stricter formatting requirements."""
        strict_prompt = self._build_sentiment_prompt(text, context, symbol)
        strict_prompt += "\n\nIMPORTANT: Return ONLY valid JSON. No explanatory text."

        try:
            messages = [
                {"role": "system", "content": "You are a financial sentiment analyzer. Return only valid JSON."},
                {"role": "user", "content": strict_prompt}
            ]

            response = self.client.chat_completion(messages, temperature=0.1, max_tokens=400)

            if response:
                result = self._parse_sentiment_response(response)
                if self.validator.validate(result, self.SENTIMENT_SCHEMA):
                    return result

        except Exception as e:
            logger.error(f"Strict retry failed: {e}")

        return self._get_fallback_sentiment()

    def _get_fallback_sentiment(self) -> Dict[str, Any]:
        """Return neutral sentiment as fallback."""
        return {
            "sentiment": "neutral",
            "confidence": 0.0,
            "score": 0.0,
            "reasoning": "Analysis unavailable - using neutral sentiment",
            "entities": [],
            "key_phrases": [],
            "impact_level": "low",
            "fallback": True
        }

    def _get_cache_key(self, text: str, symbol: Optional[str]) -> str:
        """Generate cache key for text."""
        import hashlib
        key_str = f"{text}:{symbol or ''}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _calculate_weighted_score(self, score: float, followers: int) -> float:
        """Calculate weighted score based on follower count."""
        # Logarithmic weighting to avoid huge accounts dominating
        import math
        if followers <= 0:
            return score
        weight = 1 + math.log10(max(1, followers)) / 10
        return score * min(weight, 3.0)  # Cap at 3x weight

    def _calculate_weights(
        self,
        analyses: List[Dict[str, Any]],
        strategy: str
    ) -> List[float]:
        """Calculate weights for aggregation."""
        if strategy == "uniform":
            return [1.0] * len(analyses)

        elif strategy == "confidence":
            return [a.get("confidence", 0.5) for a in analyses]

        elif strategy == "followers":
            weights = []
            for a in analyses:
                followers = a.get("author_followers", 1000)
                import math
                weight = 1 + math.log10(max(1, followers)) / 10
                weights.append(min(weight, 3.0))
            return weights

        else:
            return [1.0] * len(analyses)

    def clear_cache(self):
        """Clear sentiment cache."""
        self.cache.clear()
        logger.info("Sentiment cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "cache_size": len(self.cache),
            "client_stats": self.client.get_usage_stats()
        }
