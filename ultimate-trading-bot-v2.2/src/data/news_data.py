"""
News Data Module for Ultimate Trading Bot v2.2.

This module provides news data fetching and management
for trading analysis and sentiment.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional

import httpx
from pydantic import BaseModel, Field

from src.utils.exceptions import DataFetchError
from src.utils.helpers import generate_uuid
from src.utils.date_utils import now_utc, parse_datetime
from src.utils.decorators import async_retry


logger = logging.getLogger(__name__)


class NewsSource(str, Enum):
    """News source enumeration."""

    ALPACA = "alpaca"
    BENZINGA = "benzinga"
    FINNHUB = "finnhub"
    NEWS_API = "news_api"
    POLYGON = "polygon"


class NewsSentiment(str, Enum):
    """News sentiment enumeration."""

    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


class NewsArticle(BaseModel):
    """News article model."""

    article_id: str = Field(default_factory=generate_uuid)
    headline: str
    summary: str = Field(default="")
    content: str = Field(default="")

    source: str = Field(default="")
    author: str = Field(default="")
    url: Optional[str] = None

    symbols: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)

    published_at: datetime = Field(default_factory=now_utc)
    updated_at: Optional[datetime] = None

    sentiment: Optional[NewsSentiment] = None
    sentiment_score: float = Field(default=0.0)
    relevance_score: float = Field(default=0.0)

    images: list[str] = Field(default_factory=list)

    metadata: dict = Field(default_factory=dict)

    @property
    def age_minutes(self) -> int:
        """Get article age in minutes."""
        return int((now_utc() - self.published_at).total_seconds() / 60)

    @property
    def is_recent(self) -> bool:
        """Check if article is recent (within 1 hour)."""
        return self.age_minutes < 60

    def matches_symbol(self, symbol: str) -> bool:
        """Check if article mentions a symbol."""
        return symbol.upper() in [s.upper() for s in self.symbols]


class NewsConfig(BaseModel):
    """Configuration for news data manager."""

    alpaca_news_url: str = Field(default="https://data.alpaca.markets/v1beta1/news")
    max_articles_per_request: int = Field(default=50, ge=10, le=200)
    cache_ttl_seconds: int = Field(default=300, ge=60, le=3600)
    default_lookback_hours: int = Field(default=24, ge=1, le=168)
    enable_sentiment_analysis: bool = Field(default=True)


class NewsDataManager:
    """
    Manages news data for trading analysis.

    Provides functionality for:
    - Fetching news from various sources
    - News filtering and search
    - Sentiment extraction
    - News caching
    """

    def __init__(
        self,
        config: Optional[NewsConfig] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ) -> None:
        """
        Initialize NewsDataManager.

        Args:
            config: News configuration
            api_key: API key for news providers
            api_secret: API secret for news providers
        """
        self._config = config or NewsConfig()
        self._api_key = api_key
        self._api_secret = api_secret

        self._client: Optional[httpx.AsyncClient] = None
        self._cache: dict[str, tuple[list[NewsArticle], datetime]] = {}

        self._news_callbacks: list[Callable[[NewsArticle], None]] = []

        self._articles_fetched = 0
        self._cache_hits = 0

        logger.info("NewsDataManager initialized")

    @property
    def _headers(self) -> dict[str, str]:
        """Get API headers."""
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["APCA-API-KEY-ID"] = self._api_key
        if self._api_secret:
            headers["APCA-API-SECRET-KEY"] = self._api_secret
        return headers

    async def start(self) -> None:
        """Start the news manager."""
        self._client = httpx.AsyncClient(
            headers=self._headers,
            timeout=30.0,
        )
        logger.info("NewsDataManager started")

    async def stop(self) -> None:
        """Stop the news manager."""
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("NewsDataManager stopped")

    def on_news(self, callback: Callable[[NewsArticle], None]) -> None:
        """Register callback for new articles."""
        self._news_callbacks.append(callback)

    @async_retry(max_attempts=3, delay=1.0)
    async def get_news(
        self,
        symbols: Optional[list[str]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 50,
        include_content: bool = False,
    ) -> list[NewsArticle]:
        """
        Get news articles.

        Args:
            symbols: Filter by symbols
            start: Start datetime
            end: End datetime
            limit: Maximum articles
            include_content: Include full content

        Returns:
            List of news articles
        """
        cache_key = f"{','.join(symbols or ['all'])}:{limit}"
        cached = self._get_cached(cache_key)
        if cached:
            self._cache_hits += 1
            return cached

        if not self._client:
            await self.start()

        end = end or now_utc()
        start = start or (end - timedelta(hours=self._config.default_lookback_hours))

        params = {
            "limit": min(limit, self._config.max_articles_per_request),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "include_content": str(include_content).lower(),
        }

        if symbols:
            params["symbols"] = ",".join(symbols)

        try:
            response = await self._client.get(
                self._config.alpaca_news_url,
                params=params,
            )

            if response.status_code == 200:
                data = response.json()
                articles = self._parse_alpaca_news(data.get("news", []))

                self._set_cached(cache_key, articles)
                self._articles_fetched += len(articles)

                return articles

            logger.error(f"Failed to fetch news: {response.status_code}")
            return []

        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []

    def _parse_alpaca_news(self, news_data: list[dict]) -> list[NewsArticle]:
        """Parse Alpaca news response."""
        articles = []

        for item in news_data:
            try:
                article = NewsArticle(
                    article_id=item.get("id", generate_uuid()),
                    headline=item.get("headline", ""),
                    summary=item.get("summary", ""),
                    content=item.get("content", ""),
                    source=item.get("source", ""),
                    author=item.get("author", ""),
                    url=item.get("url"),
                    symbols=item.get("symbols", []),
                    published_at=parse_datetime(item.get("created_at")),
                    updated_at=parse_datetime(item.get("updated_at")) if item.get("updated_at") else None,
                    images=item.get("images", []),
                )
                articles.append(article)
            except Exception as e:
                logger.error(f"Error parsing news article: {e}")

        return articles

    async def get_news_for_symbol(
        self,
        symbol: str,
        hours_back: int = 24,
        limit: int = 20,
    ) -> list[NewsArticle]:
        """
        Get news for a specific symbol.

        Args:
            symbol: Trading symbol
            hours_back: Hours to look back
            limit: Maximum articles

        Returns:
            List of news articles
        """
        end = now_utc()
        start = end - timedelta(hours=hours_back)

        return await self.get_news(
            symbols=[symbol],
            start=start,
            end=end,
            limit=limit,
        )

    async def get_latest_headlines(
        self,
        symbols: Optional[list[str]] = None,
        limit: int = 10,
    ) -> list[str]:
        """
        Get latest headlines.

        Args:
            symbols: Filter by symbols
            limit: Maximum headlines

        Returns:
            List of headline strings
        """
        articles = await self.get_news(symbols=symbols, limit=limit)
        return [article.headline for article in articles]

    async def search_news(
        self,
        query: str,
        symbols: Optional[list[str]] = None,
        hours_back: int = 48,
        limit: int = 50,
    ) -> list[NewsArticle]:
        """
        Search news articles.

        Args:
            query: Search query
            symbols: Filter by symbols
            hours_back: Hours to look back
            limit: Maximum articles

        Returns:
            List of matching articles
        """
        articles = await self.get_news(
            symbols=symbols,
            start=now_utc() - timedelta(hours=hours_back),
            limit=limit * 2,
            include_content=True,
        )

        query_lower = query.lower()
        matching = []

        for article in articles:
            if query_lower in article.headline.lower():
                matching.append(article)
            elif query_lower in article.summary.lower():
                matching.append(article)
            elif query_lower in article.content.lower():
                matching.append(article)

        return matching[:limit]

    def filter_by_sentiment(
        self,
        articles: list[NewsArticle],
        sentiment: NewsSentiment,
    ) -> list[NewsArticle]:
        """
        Filter articles by sentiment.

        Args:
            articles: Articles to filter
            sentiment: Required sentiment

        Returns:
            Filtered articles
        """
        return [a for a in articles if a.sentiment == sentiment]

    def filter_recent(
        self,
        articles: list[NewsArticle],
        max_age_minutes: int = 60,
    ) -> list[NewsArticle]:
        """
        Filter to recent articles.

        Args:
            articles: Articles to filter
            max_age_minutes: Maximum age in minutes

        Returns:
            Recent articles
        """
        return [a for a in articles if a.age_minutes <= max_age_minutes]

    def group_by_symbol(
        self,
        articles: list[NewsArticle],
    ) -> dict[str, list[NewsArticle]]:
        """
        Group articles by symbol.

        Args:
            articles: Articles to group

        Returns:
            Dictionary of symbol to articles
        """
        grouped: dict[str, list[NewsArticle]] = {}

        for article in articles:
            for symbol in article.symbols:
                if symbol not in grouped:
                    grouped[symbol] = []
                grouped[symbol].append(article)

        return grouped

    def get_news_summary(
        self,
        articles: list[NewsArticle],
    ) -> dict:
        """
        Get summary statistics for articles.

        Args:
            articles: Articles to summarize

        Returns:
            Summary dictionary
        """
        if not articles:
            return {"count": 0}

        sentiments = [a.sentiment for a in articles if a.sentiment]
        sources = [a.source for a in articles]
        all_symbols = []
        for a in articles:
            all_symbols.extend(a.symbols)

        return {
            "count": len(articles),
            "time_range_hours": (
                (articles[0].published_at - articles[-1].published_at).total_seconds() / 3600
                if len(articles) > 1 else 0
            ),
            "unique_sources": len(set(sources)),
            "unique_symbols": len(set(all_symbols)),
            "sentiment_breakdown": {
                s.value: sentiments.count(s) for s in NewsSentiment
            } if sentiments else {},
            "average_relevance": (
                sum(a.relevance_score for a in articles) / len(articles)
            ),
        }

    def _get_cached(self, key: str) -> Optional[list[NewsArticle]]:
        """Get cached articles if valid."""
        if key not in self._cache:
            return None

        articles, cached_at = self._cache[key]
        age = (now_utc() - cached_at).total_seconds()

        if age > self._config.cache_ttl_seconds:
            del self._cache[key]
            return None

        return articles

    def _set_cached(self, key: str, articles: list[NewsArticle]) -> None:
        """Set cached articles."""
        self._cache[key] = (articles, now_utc())

    def clear_cache(self) -> int:
        """Clear the cache."""
        count = len(self._cache)
        self._cache.clear()
        return count

    async def notify_new_article(self, article: NewsArticle) -> None:
        """Notify callbacks of new article."""
        for callback in self._news_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(article)
                else:
                    callback(article)
            except Exception as e:
                logger.error(f"Error in news callback: {e}")

    def get_statistics(self) -> dict:
        """Get manager statistics."""
        return {
            "articles_fetched": self._articles_fetched,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._cache),
            "callbacks_registered": len(self._news_callbacks),
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"NewsDataManager(fetched={self._articles_fetched})"
