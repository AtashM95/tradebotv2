"""
News Sentiment Strategy - Trade based on news sentiment analysis.
~400 lines as per schema
"""

import logging
from typing import Dict, Any, List, Optional
from collections import deque
from datetime import datetime
import statistics

from ..core.contracts import MarketSnapshot, SignalIntent
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class NewsSentimentStrategy(BaseStrategy):
    """
    News sentiment-based trading strategy.

    Algorithm:
    1. Analyze news headlines and content
    2. Calculate sentiment scores
    3. Detect sentiment shifts
    4. Correlate with price action
    5. Generate signals on strong sentiment changes
    6. Filter false signals with confirmation

    Features:
    - Multi-source news aggregation
    - Sentiment scoring (positive/negative/neutral)
    - Sentiment momentum tracking
    - Event impact assessment
    - Source credibility weighting
    - Volume correlation
    - Time decay for news relevance
    """

    name = 'news_sentiment'

    def __init__(
        self,
        sentiment_threshold: float = 0.6,  # Sentiment score threshold
        min_news_count: int = 3,  # Minimum news items for signal
        news_lookback_hours: int = 24,
        sentiment_momentum_period: int = 10,
        require_price_confirmation: bool = True,
        source_weights: Dict[str, float] = None,
        decay_half_life_hours: float = 6.0
    ):
        """
        Initialize news sentiment strategy.

        Args:
            sentiment_threshold: Minimum sentiment score for signal
            min_news_count: Minimum number of news items required
            news_lookback_hours: Hours to look back for news
            sentiment_momentum_period: Period for sentiment momentum
            require_price_confirmation: Require price confirmation
            source_weights: Credibility weights for news sources
            decay_half_life_hours: Half-life for time decay
        """
        super().__init__()
        self.sentiment_threshold = sentiment_threshold
        self.min_news_count = min_news_count
        self.news_lookback_hours = news_lookback_hours
        self.sentiment_momentum_period = sentiment_momentum_period
        self.require_price_confirmation = require_price_confirmation
        self.source_weights = source_weights or {
            'bloomberg': 1.0,
            'reuters': 1.0,
            'wsj': 0.9,
            'cnbc': 0.8,
            'twitter': 0.5,
            'reddit': 0.3
        }
        self.decay_half_life_hours = decay_half_life_hours

        # Track news and sentiment
        self.news_items = {}  # symbol -> deque of news
        self.sentiment_history = {}  # symbol -> deque of sentiment scores
        self.sentiment_events = {}  # symbol -> recent significant events

        # Statistics
        self.stats = {
            "signals_generated": 0,
            "news_items_processed": 0,
            "positive_sentiment_signals": 0,
            "negative_sentiment_signals": 0,
            "avg_sentiment_score": 0.0,
            "confirmed_signals": 0
        }

    def generate(self, snapshot: MarketSnapshot) -> SignalIntent | None:
        """
        Generate trading signal based on news sentiment.

        Args:
            snapshot: Market snapshot

        Returns:
            SignalIntent or None
        """
        symbol = snapshot.symbol
        current_price = snapshot.price
        metadata = snapshot.metadata or {}

        # Get news data from metadata
        news_data = metadata.get('news', [])

        if not news_data:
            return None

        # Process news items
        self._process_news(symbol, news_data)

        # Calculate aggregate sentiment
        sentiment = self._calculate_aggregate_sentiment(symbol)

        if sentiment is None:
            return None

        # Check if we have enough news items
        if symbol not in self.news_items or len(self.news_items[symbol]) < self.min_news_count:
            return None

        # Calculate sentiment momentum
        momentum = self._calculate_sentiment_momentum(symbol)

        # Detect sentiment shifts
        shift = self._detect_sentiment_shift(symbol, sentiment)

        # Check for significant events
        event_impact = self._assess_event_impact(symbol, news_data)

        # Generate signal if conditions met
        if abs(sentiment) >= self.sentiment_threshold:
            # Confirm with price action if required
            if self.require_price_confirmation:
                if not self._confirm_with_price(snapshot, sentiment):
                    return None

            signal = self._generate_sentiment_signal(
                symbol,
                current_price,
                sentiment,
                momentum,
                shift,
                event_impact,
                metadata
            )

            if signal:
                self.stats["signals_generated"] += 1
                if sentiment > 0:
                    self.stats["positive_sentiment_signals"] += 1
                else:
                    self.stats["negative_sentiment_signals"] += 1

            return signal

        return None

    def _process_news(self, symbol: str, news_data: List[Dict[str, Any]]):
        """Process news items and calculate sentiment."""
        if symbol not in self.news_items:
            self.news_items[symbol] = deque(maxlen=100)

        for news in news_data:
            # Extract news fields
            headline = news.get('headline', '')
            content = news.get('content', '')
            source = news.get('source', 'unknown')
            timestamp = news.get('timestamp', 0)

            # Calculate sentiment score
            sentiment_score = self._calculate_sentiment_score(headline, content)

            # Apply source weighting
            source_weight = self.source_weights.get(source.lower(), 0.5)
            weighted_sentiment = sentiment_score * source_weight

            # Apply time decay
            decayed_sentiment = self._apply_time_decay(weighted_sentiment, timestamp)

            # Store news item
            news_item = {
                'headline': headline,
                'content': content,
                'source': source,
                'timestamp': timestamp,
                'raw_sentiment': sentiment_score,
                'weighted_sentiment': weighted_sentiment,
                'decayed_sentiment': decayed_sentiment
            }

            self.news_items[symbol].append(news_item)
            self.stats["news_items_processed"] += 1

    def _calculate_sentiment_score(self, headline: str, content: str) -> float:
        """
        Calculate sentiment score from text.

        Simplified implementation - in production would use NLP libraries.
        """
        # Positive keywords
        positive_keywords = [
            'surge', 'soar', 'rally', 'gain', 'profit', 'growth', 'bullish',
            'breakthrough', 'success', 'beat', 'exceed', 'strong', 'positive',
            'upgrade', 'outperform', 'buy', 'optimistic', 'record', 'high'
        ]

        # Negative keywords
        negative_keywords = [
            'plunge', 'crash', 'fall', 'loss', 'decline', 'bearish', 'weak',
            'miss', 'disappoint', 'concern', 'risk', 'downgrade', 'sell',
            'negative', 'worry', 'fear', 'low', 'cut', 'reduce'
        ]

        text = (headline + ' ' + content).lower()

        # Count keyword occurrences
        positive_count = sum(1 for keyword in positive_keywords if keyword in text)
        negative_count = sum(1 for keyword in negative_keywords if keyword in text)

        total_count = positive_count + negative_count

        if total_count == 0:
            return 0.0

        # Calculate normalized sentiment (-1 to 1)
        sentiment = (positive_count - negative_count) / total_count

        return sentiment

    def _apply_time_decay(self, sentiment: float, timestamp: float) -> float:
        """Apply exponential time decay to sentiment."""
        if timestamp == 0:
            return sentiment

        # Calculate age in hours
        current_time = datetime.now().timestamp()
        age_hours = (current_time - timestamp) / 3600

        # Exponential decay
        decay_factor = 0.5 ** (age_hours / self.decay_half_life_hours)

        return sentiment * decay_factor

    def _calculate_aggregate_sentiment(self, symbol: str) -> Optional[float]:
        """Calculate aggregate sentiment from recent news."""
        if symbol not in self.news_items or len(self.news_items[symbol]) == 0:
            return None

        news_items = list(self.news_items[symbol])

        # Use decayed sentiment
        sentiments = [item['decayed_sentiment'] for item in news_items]

        # Calculate weighted average
        aggregate = statistics.mean(sentiments) if sentiments else 0.0

        # Store in history
        if symbol not in self.sentiment_history:
            self.sentiment_history[symbol] = deque(maxlen=self.sentiment_momentum_period * 2)

        self.sentiment_history[symbol].append(aggregate)

        # Update stats
        self.stats["avg_sentiment_score"] = (
            (self.stats["avg_sentiment_score"] * self.stats["news_items_processed"] + aggregate) /
            (self.stats["news_items_processed"] + 1)
        )

        return aggregate

    def _calculate_sentiment_momentum(self, symbol: str) -> float:
        """Calculate momentum in sentiment."""
        if symbol not in self.sentiment_history:
            return 0.0

        history = list(self.sentiment_history[symbol])

        if len(history) < self.sentiment_momentum_period:
            return 0.0

        recent = history[-self.sentiment_momentum_period:]

        # Calculate trend
        if len(recent) < 2:
            return 0.0

        first_half = recent[:len(recent)//2]
        second_half = recent[len(recent)//2:]

        first_avg = statistics.mean(first_half) if first_half else 0.0
        second_avg = statistics.mean(second_half) if second_half else 0.0

        momentum = second_avg - first_avg

        return momentum

    def _detect_sentiment_shift(self, symbol: str, current_sentiment: float) -> Optional[str]:
        """Detect significant sentiment shifts."""
        if symbol not in self.sentiment_history or len(self.sentiment_history[symbol]) < 5:
            return None

        history = list(self.sentiment_history[symbol])
        previous_avg = statistics.mean(history[-6:-1]) if len(history) >= 6 else history[0]

        shift_magnitude = abs(current_sentiment - previous_avg)

        if shift_magnitude > 0.3:  # 30% shift
            if current_sentiment > previous_avg:
                return 'positive_shift'
            else:
                return 'negative_shift'

        return None

    def _assess_event_impact(
        self,
        symbol: str,
        news_data: List[Dict[str, Any]]
    ) -> float:
        """Assess impact of recent events."""
        high_impact_keywords = [
            'earnings', 'acquisition', 'merger', 'fda', 'approval',
            'bankruptcy', 'lawsuit', 'ceo', 'regulation', 'investigation'
        ]

        impact_score = 0.0

        for news in news_data:
            headline = news.get('headline', '').lower()

            for keyword in high_impact_keywords:
                if keyword in headline:
                    impact_score += 0.2

        return min(impact_score, 1.0)

    def _confirm_with_price(
        self,
        snapshot: MarketSnapshot,
        sentiment: float
    ) -> bool:
        """Confirm sentiment signal with price action."""
        if not snapshot.history or len(snapshot.history) < 10:
            return True  # Not enough data, allow signal

        recent_prices = snapshot.history[-10:]
        price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]

        # Positive sentiment should align with price increase
        if sentiment > 0:
            return price_change > -0.005  # Allow slight negative (-0.5%)

        # Negative sentiment should align with price decrease
        else:
            return price_change < 0.005  # Allow slight positive (+0.5%)

    def _generate_sentiment_signal(
        self,
        symbol: str,
        current_price: float,
        sentiment: float,
        momentum: float,
        shift: Optional[str],
        event_impact: float,
        metadata: Dict[str, Any]
    ) -> Optional[SignalIntent]:
        """Generate trading signal from sentiment."""
        # Determine action
        if sentiment > 0:
            action = 'buy'
        else:
            action = 'sell'

        # Calculate confidence
        confidence = self._calculate_confidence(
            abs(sentiment),
            abs(momentum),
            event_impact,
            shift is not None
        )

        # Calculate targets based on sentiment strength
        if action == 'buy':
            target = current_price * (1 + abs(sentiment) * 0.05)  # Up to 5% target
            stop_loss = current_price * 0.98
        else:
            target = current_price * (1 - abs(sentiment) * 0.05)
            stop_loss = current_price * 1.02

        # Track confirmation
        if self.require_price_confirmation:
            self.stats["confirmed_signals"] += 1

        return SignalIntent(
            symbol=symbol,
            action=action,
            confidence=confidence,
            metadata={
                'strategy': self.name,
                'sentiment_score': sentiment,
                'sentiment_momentum': momentum,
                'sentiment_shift': shift,
                'event_impact': event_impact,
                'news_count': len(self.news_items.get(symbol, [])),
                'target': target,
                'stop_loss': stop_loss,
                'signal_type': 'news_driven'
            }
        )

    def _calculate_confidence(
        self,
        sentiment_strength: float,
        momentum: float,
        event_impact: float,
        has_shift: bool
    ) -> float:
        """Calculate signal confidence."""
        confidence = 0.5  # Base confidence

        # Sentiment strength (max 0.25)
        confidence += min(sentiment_strength, 1.0) * 0.25

        # Momentum alignment (max 0.15)
        confidence += min(abs(momentum), 0.5) * 0.15

        # Event impact (max 0.10)
        confidence += event_impact * 0.10

        # Sentiment shift (0.10)
        if has_shift:
            confidence += 0.10

        return min(confidence, 0.95)

    def get_sentiment_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get sentiment analysis for symbol."""
        if symbol not in self.news_items or symbol not in self.sentiment_history:
            return None

        news_items = list(self.news_items[symbol])
        sentiment_history = list(self.sentiment_history[symbol])

        if not sentiment_history:
            return None

        current_sentiment = sentiment_history[-1]
        momentum = self._calculate_sentiment_momentum(symbol)

        # Categorize news by sentiment
        positive_news = [n for n in news_items if n['decayed_sentiment'] > 0.2]
        negative_news = [n for n in news_items if n['decayed_sentiment'] < -0.2]
        neutral_news = [n for n in news_items if -0.2 <= n['decayed_sentiment'] <= 0.2]

        return {
            'symbol': symbol,
            'current_sentiment': current_sentiment,
            'sentiment_momentum': momentum,
            'total_news_count': len(news_items),
            'positive_count': len(positive_news),
            'negative_count': len(negative_news),
            'neutral_count': len(neutral_news),
            'recent_headlines': [n['headline'] for n in news_items[-5:]]
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        stats = self.stats.copy()

        # Calculate sentiment bias
        total_sentiment_signals = (
            stats["positive_sentiment_signals"] +
            stats["negative_sentiment_signals"]
        )

        if total_sentiment_signals > 0:
            stats["positive_bias"] = (
                stats["positive_sentiment_signals"] / total_sentiment_signals
            )
        else:
            stats["positive_bias"] = 0.5

        # Confirmation rate
        if stats["signals_generated"] > 0:
            stats["confirmation_rate"] = (
                stats["confirmed_signals"] / stats["signals_generated"]
            )
        else:
            stats["confirmation_rate"] = 0.0

        return stats

    def reset(self):
        """Reset strategy state."""
        self.news_items.clear()
        self.sentiment_history.clear()
        self.sentiment_events.clear()
        self.stats = {
            "signals_generated": 0,
            "news_items_processed": 0,
            "positive_sentiment_signals": 0,
            "negative_sentiment_signals": 0,
            "avg_sentiment_score": 0.0,
            "confirmed_signals": 0
        }
