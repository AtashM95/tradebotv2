"""
Market Sentiment Analyzer for Ultimate Trading Bot v2.2.

Analyzes market-based sentiment indicators derived from price action,
volume, options flow, and other market data.
"""

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

logger = logging.getLogger(__name__)


class MarketIndicator(str, Enum):
    """Market-based sentiment indicators."""
    VIX = "vix"
    PUT_CALL_RATIO = "put_call_ratio"
    ADVANCE_DECLINE = "advance_decline"
    NEW_HIGHS_LOWS = "new_highs_lows"
    VOLUME_RATIO = "volume_ratio"
    BREADTH = "breadth"
    FEAR_GREED = "fear_greed"
    RSI = "rsi"
    MACD = "macd"
    MOMENTUM = "momentum"


@dataclass
class MarketData:
    """Market data for sentiment analysis."""

    symbol: str
    timestamp: datetime

    # Price data
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int

    # Optional indicators
    vix: float | None = None
    put_call_ratio: float | None = None
    advance_count: int | None = None
    decline_count: int | None = None
    new_highs: int | None = None
    new_lows: int | None = None

    # Computed fields
    change_percent: float = 0.0
    avg_volume: int | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Calculate derived fields."""
        if self.open_price > 0:
            self.change_percent = (
                (self.close_price - self.open_price) / self.open_price * 100
            )


@dataclass
class OptionsFlow:
    """Options flow data."""

    symbol: str
    timestamp: datetime

    # Volume
    call_volume: int
    put_volume: int
    total_volume: int

    # Open interest
    call_oi: int
    put_oi: int

    # Premium
    call_premium: float
    put_premium: float

    # Large trades
    large_call_trades: int = 0
    large_put_trades: int = 0
    unusual_activity: bool = False

    @property
    def put_call_ratio(self) -> float:
        """Calculate put/call ratio."""
        if self.call_volume > 0:
            return self.put_volume / self.call_volume
        return 1.0

    @property
    def premium_ratio(self) -> float:
        """Calculate premium ratio."""
        if self.call_premium > 0:
            return self.put_premium / self.call_premium
        return 1.0


@dataclass
class MarketSentimentResult(SentimentResult):
    """Extended sentiment result for market indicators."""

    indicator: MarketIndicator | None = None
    indicator_value: float = 0.0
    indicator_z_score: float = 0.0
    historical_percentile: float = 50.0


class MarketSentimentAnalyzer:
    """
    Analyzer for market-based sentiment indicators.

    Uses price action, volume, and market breadth to gauge sentiment.
    """

    # Indicator thresholds
    VIX_THRESHOLDS = {
        "extreme_fear": 35.0,
        "fear": 25.0,
        "neutral": 20.0,
        "greed": 15.0,
        "extreme_greed": 12.0,
    }

    PUT_CALL_THRESHOLDS = {
        "extreme_fear": 1.5,
        "fear": 1.2,
        "neutral": 1.0,
        "greed": 0.8,
        "extreme_greed": 0.5,
    }

    def __init__(
        self,
        config: SentimentConfig | None = None,
        lookback_period: int = 20,
    ) -> None:
        """
        Initialize market sentiment analyzer.

        Args:
            config: Sentiment configuration
            lookback_period: Period for historical comparisons
        """
        self.config = config or SentimentConfig()
        self.lookback_period = lookback_period

        # Historical data for z-score calculation
        self._vix_history: list[float] = []
        self._pcr_history: list[float] = []
        self._volume_history: list[int] = []

        self._initialized = False

        logger.info("Initialized MarketSentimentAnalyzer")

    async def initialize(self) -> None:
        """Initialize the analyzer."""
        self._initialized = True
        logger.info("MarketSentimentAnalyzer initialized")

    async def analyze_vix(
        self,
        vix_value: float,
    ) -> MarketSentimentResult:
        """
        Analyze VIX-based sentiment.

        Args:
            vix_value: Current VIX value

        Returns:
            Market sentiment result
        """
        # Update history
        self._vix_history.append(vix_value)
        if len(self._vix_history) > self.lookback_period:
            self._vix_history = self._vix_history[-self.lookback_period:]

        # Calculate z-score
        z_score = self._calculate_z_score(vix_value, self._vix_history)

        # Calculate percentile
        percentile = self._calculate_percentile(vix_value, self._vix_history)

        # Determine sentiment (VIX is inverse - high VIX = fear)
        if vix_value >= self.VIX_THRESHOLDS["extreme_fear"]:
            score = -0.9
            label = SentimentLabel.VERY_NEGATIVE
        elif vix_value >= self.VIX_THRESHOLDS["fear"]:
            score = -0.5
            label = SentimentLabel.NEGATIVE
        elif vix_value <= self.VIX_THRESHOLDS["extreme_greed"]:
            score = 0.9
            label = SentimentLabel.VERY_POSITIVE
        elif vix_value <= self.VIX_THRESHOLDS["greed"]:
            score = 0.5
            label = SentimentLabel.POSITIVE
        else:
            score = 0.0
            label = SentimentLabel.NEUTRAL

        return MarketSentimentResult(
            score=score,
            label=label,
            confidence=0.8,
            source=SentimentSource.MARKET,
            timestamp=datetime.now(),
            indicator=MarketIndicator.VIX,
            indicator_value=vix_value,
            indicator_z_score=z_score,
            historical_percentile=percentile,
        )

    async def analyze_put_call_ratio(
        self,
        pcr: float,
    ) -> MarketSentimentResult:
        """
        Analyze put/call ratio sentiment.

        Args:
            pcr: Put/call ratio

        Returns:
            Market sentiment result
        """
        # Update history
        self._pcr_history.append(pcr)
        if len(self._pcr_history) > self.lookback_period:
            self._pcr_history = self._pcr_history[-self.lookback_period:]

        z_score = self._calculate_z_score(pcr, self._pcr_history)
        percentile = self._calculate_percentile(pcr, self._pcr_history)

        # PCR: high = bearish (more puts), low = bullish (more calls)
        # But extreme readings can be contrarian
        if pcr >= self.PUT_CALL_THRESHOLDS["extreme_fear"]:
            # Extreme fear can be bullish contrarian signal
            score = 0.3  # Slightly bullish contrarian
            label = SentimentLabel.POSITIVE
        elif pcr >= self.PUT_CALL_THRESHOLDS["fear"]:
            score = -0.4
            label = SentimentLabel.NEGATIVE
        elif pcr <= self.PUT_CALL_THRESHOLDS["extreme_greed"]:
            # Extreme greed can be bearish contrarian signal
            score = -0.3  # Slightly bearish contrarian
            label = SentimentLabel.NEGATIVE
        elif pcr <= self.PUT_CALL_THRESHOLDS["greed"]:
            score = 0.4
            label = SentimentLabel.POSITIVE
        else:
            score = 0.0
            label = SentimentLabel.NEUTRAL

        return MarketSentimentResult(
            score=score,
            label=label,
            confidence=0.7,
            source=SentimentSource.MARKET,
            timestamp=datetime.now(),
            indicator=MarketIndicator.PUT_CALL_RATIO,
            indicator_value=pcr,
            indicator_z_score=z_score,
            historical_percentile=percentile,
        )

    async def analyze_advance_decline(
        self,
        advances: int,
        declines: int,
    ) -> MarketSentimentResult:
        """
        Analyze advance/decline ratio.

        Args:
            advances: Number of advancing stocks
            declines: Number of declining stocks

        Returns:
            Market sentiment result
        """
        total = advances + declines
        if total == 0:
            return MarketSentimentResult(
                score=0.0,
                label=SentimentLabel.NEUTRAL,
                confidence=0.0,
                source=SentimentSource.MARKET,
                timestamp=datetime.now(),
                indicator=MarketIndicator.ADVANCE_DECLINE,
            )

        ratio = advances / total
        ad_line = advances - declines

        # Score based on ratio
        score = (ratio - 0.5) * 2  # Map 0-1 to -1 to 1
        score = max(-1.0, min(1.0, score))

        return MarketSentimentResult(
            score=score,
            label=self._score_to_label(score),
            confidence=0.75,
            source=SentimentSource.MARKET,
            timestamp=datetime.now(),
            indicator=MarketIndicator.ADVANCE_DECLINE,
            indicator_value=ratio,
            metadata={
                "advances": advances,
                "declines": declines,
                "ad_line": ad_line,
            },
        )

    async def analyze_new_highs_lows(
        self,
        new_highs: int,
        new_lows: int,
    ) -> MarketSentimentResult:
        """
        Analyze new highs vs new lows.

        Args:
            new_highs: Number of new 52-week highs
            new_lows: Number of new 52-week lows

        Returns:
            Market sentiment result
        """
        total = new_highs + new_lows
        if total == 0:
            return MarketSentimentResult(
                score=0.0,
                label=SentimentLabel.NEUTRAL,
                confidence=0.0,
                source=SentimentSource.MARKET,
                timestamp=datetime.now(),
                indicator=MarketIndicator.NEW_HIGHS_LOWS,
            )

        ratio = new_highs / total
        score = (ratio - 0.5) * 2

        return MarketSentimentResult(
            score=max(-1.0, min(1.0, score)),
            label=self._score_to_label(score),
            confidence=0.7,
            source=SentimentSource.MARKET,
            timestamp=datetime.now(),
            indicator=MarketIndicator.NEW_HIGHS_LOWS,
            indicator_value=ratio,
            metadata={
                "new_highs": new_highs,
                "new_lows": new_lows,
            },
        )

    async def analyze_volume(
        self,
        current_volume: int,
        average_volume: int,
        price_change: float,
    ) -> MarketSentimentResult:
        """
        Analyze volume-based sentiment.

        Args:
            current_volume: Current trading volume
            average_volume: Average volume
            price_change: Price change percentage

        Returns:
            Market sentiment result
        """
        if average_volume == 0:
            return MarketSentimentResult(
                score=0.0,
                label=SentimentLabel.NEUTRAL,
                confidence=0.0,
                source=SentimentSource.MARKET,
                timestamp=datetime.now(),
                indicator=MarketIndicator.VOLUME_RATIO,
            )

        volume_ratio = current_volume / average_volume

        # Volume confirms price movement
        if volume_ratio > 1.5:
            # High volume
            if price_change > 0:
                score = min(0.8, 0.4 + volume_ratio * 0.2)
            elif price_change < 0:
                score = max(-0.8, -0.4 - volume_ratio * 0.2)
            else:
                score = 0.0
        elif volume_ratio < 0.5:
            # Low volume - weak conviction
            score = price_change * 0.01 * 0.5  # Reduced impact
        else:
            score = price_change * 0.01

        score = max(-1.0, min(1.0, score))

        return MarketSentimentResult(
            score=score,
            label=self._score_to_label(score),
            confidence=0.6 + min(volume_ratio - 1.0, 0.3) if volume_ratio > 1.0 else 0.5,
            source=SentimentSource.MARKET,
            timestamp=datetime.now(),
            indicator=MarketIndicator.VOLUME_RATIO,
            indicator_value=volume_ratio,
            metadata={
                "current_volume": current_volume,
                "average_volume": average_volume,
                "price_change": price_change,
            },
        )

    async def analyze_rsi(
        self,
        rsi: float,
    ) -> MarketSentimentResult:
        """
        Analyze RSI-based sentiment.

        Args:
            rsi: RSI value (0-100)

        Returns:
            Market sentiment result
        """
        # RSI interpretation (contrarian)
        if rsi >= 80:
            score = -0.6  # Overbought - bearish
            label = SentimentLabel.NEGATIVE
        elif rsi >= 70:
            score = -0.3
            label = SentimentLabel.NEGATIVE
        elif rsi <= 20:
            score = 0.6  # Oversold - bullish
            label = SentimentLabel.POSITIVE
        elif rsi <= 30:
            score = 0.3
            label = SentimentLabel.POSITIVE
        else:
            # Normalize 30-70 range
            score = (rsi - 50) / 50  # -0.4 to 0.4
            label = self._score_to_label(score)

        return MarketSentimentResult(
            score=score,
            label=label,
            confidence=0.65,
            source=SentimentSource.MARKET,
            timestamp=datetime.now(),
            indicator=MarketIndicator.RSI,
            indicator_value=rsi,
        )

    async def analyze_momentum(
        self,
        prices: list[float],
        period: int = 14,
    ) -> MarketSentimentResult:
        """
        Analyze price momentum.

        Args:
            prices: List of closing prices
            period: Momentum period

        Returns:
            Market sentiment result
        """
        if len(prices) < period + 1:
            return MarketSentimentResult(
                score=0.0,
                label=SentimentLabel.NEUTRAL,
                confidence=0.0,
                source=SentimentSource.MARKET,
                timestamp=datetime.now(),
                indicator=MarketIndicator.MOMENTUM,
            )

        # Calculate momentum
        momentum = prices[-1] - prices[-period-1]
        momentum_pct = momentum / prices[-period-1] * 100 if prices[-period-1] != 0 else 0

        # Normalize to -1 to 1
        score = max(-1.0, min(1.0, momentum_pct / 10))

        return MarketSentimentResult(
            score=score,
            label=self._score_to_label(score),
            confidence=0.7,
            source=SentimentSource.MARKET,
            timestamp=datetime.now(),
            indicator=MarketIndicator.MOMENTUM,
            indicator_value=momentum_pct,
        )

    async def analyze_options_flow(
        self,
        flow: OptionsFlow,
    ) -> MarketSentimentResult:
        """
        Analyze options flow sentiment.

        Args:
            flow: Options flow data

        Returns:
            Market sentiment result
        """
        # Multiple factors
        pcr_score = 0.0
        premium_score = 0.0
        large_trade_score = 0.0

        # Put/call ratio
        pcr = flow.put_call_ratio
        if pcr < 0.7:
            pcr_score = 0.5
        elif pcr > 1.3:
            pcr_score = -0.5
        else:
            pcr_score = (1.0 - pcr) * 0.5

        # Premium ratio
        pr = flow.premium_ratio
        if pr < 0.7:
            premium_score = 0.3
        elif pr > 1.3:
            premium_score = -0.3
        else:
            premium_score = (1.0 - pr) * 0.3

        # Large trades
        if flow.large_call_trades > flow.large_put_trades * 2:
            large_trade_score = 0.4
        elif flow.large_put_trades > flow.large_call_trades * 2:
            large_trade_score = -0.4

        # Combine scores
        score = pcr_score * 0.4 + premium_score * 0.3 + large_trade_score * 0.3
        score = max(-1.0, min(1.0, score))

        confidence = 0.7
        if flow.unusual_activity:
            confidence = 0.85

        return MarketSentimentResult(
            score=score,
            label=self._score_to_label(score),
            confidence=confidence,
            source=SentimentSource.MARKET,
            timestamp=flow.timestamp,
            symbol=flow.symbol,
            indicator=MarketIndicator.PUT_CALL_RATIO,
            indicator_value=pcr,
            metadata={
                "put_call_ratio": pcr,
                "premium_ratio": pr,
                "unusual_activity": flow.unusual_activity,
            },
        )

    async def aggregate_market_sentiment(
        self,
        market_data: MarketData,
        options_flow: OptionsFlow | None = None,
        prices: list[float] | None = None,
    ) -> AggregatedSentiment:
        """
        Aggregate all market-based sentiment indicators.

        Args:
            market_data: Market data
            options_flow: Options flow data
            prices: Historical prices for momentum

        Returns:
            Aggregated sentiment
        """
        results: list[MarketSentimentResult] = []

        # Analyze VIX if available
        if market_data.vix is not None:
            vix_result = await self.analyze_vix(market_data.vix)
            results.append(vix_result)

        # Analyze put/call ratio
        if market_data.put_call_ratio is not None:
            pcr_result = await self.analyze_put_call_ratio(
                market_data.put_call_ratio
            )
            results.append(pcr_result)

        # Analyze advance/decline
        if market_data.advance_count is not None and market_data.decline_count is not None:
            ad_result = await self.analyze_advance_decline(
                market_data.advance_count,
                market_data.decline_count,
            )
            results.append(ad_result)

        # Analyze new highs/lows
        if market_data.new_highs is not None and market_data.new_lows is not None:
            hl_result = await self.analyze_new_highs_lows(
                market_data.new_highs,
                market_data.new_lows,
            )
            results.append(hl_result)

        # Analyze volume
        if market_data.avg_volume is not None:
            vol_result = await self.analyze_volume(
                market_data.volume,
                market_data.avg_volume,
                market_data.change_percent,
            )
            results.append(vol_result)

        # Analyze options flow
        if options_flow is not None:
            flow_result = await self.analyze_options_flow(options_flow)
            results.append(flow_result)

        # Analyze momentum
        if prices is not None and len(prices) >= 15:
            mom_result = await self.analyze_momentum(prices)
            results.append(mom_result)

        # Aggregate results
        return self._aggregate_results(
            results,
            market_data.timestamp - timedelta(hours=24),
            market_data.timestamp,
        )

    def generate_signal(
        self,
        aggregated: AggregatedSentiment,
        symbol: str,
    ) -> SentimentSignal:
        """
        Generate trading signal from aggregated sentiment.

        Args:
            aggregated: Aggregated sentiment
            symbol: Trading symbol

        Returns:
            Sentiment signal
        """
        score = aggregated.overall_score
        confidence = aggregated.overall_confidence

        # Determine direction
        if score > self.config.positive_threshold and confidence >= self.config.min_confidence:
            direction = 1
        elif score < self.config.negative_threshold and confidence >= self.config.min_confidence:
            direction = -1
        else:
            direction = 0

        # Calculate strength
        strength = abs(score) * confidence

        return SentimentSignal(
            direction=direction,
            strength=strength,
            confidence=confidence,
            sentiment_score=score,
            sentiment_label=aggregated.overall_label,
            symbol=symbol,
            timestamp=datetime.now(),
            sources=[SentimentSource.MARKET],
            reasoning=self._generate_reasoning(aggregated),
            indicators=aggregated.source_scores,
        )

    def _calculate_z_score(
        self,
        value: float,
        history: list[float],
    ) -> float:
        """Calculate z-score for a value."""
        if len(history) < 2:
            return 0.0

        mean = np.mean(history)
        std = np.std(history)

        if std == 0:
            return 0.0

        return (value - mean) / std

    def _calculate_percentile(
        self,
        value: float,
        history: list[float],
    ) -> float:
        """Calculate percentile for a value."""
        if not history:
            return 50.0

        sorted_history = sorted(history)
        count_below = sum(1 for v in sorted_history if v < value)

        return count_below / len(sorted_history) * 100

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

    def _aggregate_results(
        self,
        results: list[MarketSentimentResult],
        start_time: datetime,
        end_time: datetime,
    ) -> AggregatedSentiment:
        """Aggregate market sentiment results."""
        if not results:
            return AggregatedSentiment(
                overall_score=0.0,
                overall_label=SentimentLabel.NEUTRAL,
                overall_confidence=0.0,
                start_time=start_time,
                end_time=end_time,
            )

        # Weight by confidence
        weighted_sum = sum(r.score * r.confidence for r in results)
        total_weight = sum(r.confidence for r in results)

        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Collect by indicator
        indicator_scores = {}
        for result in results:
            if result.indicator:
                indicator_scores[result.indicator.value] = result.score

        return AggregatedSentiment(
            overall_score=overall_score,
            overall_label=self._score_to_label(overall_score),
            overall_confidence=total_weight / len(results) if results else 0.0,
            start_time=start_time,
            end_time=end_time,
            source_scores=indicator_scores,
            total_samples=len(results),
            results=results,
        )

    def _generate_reasoning(self, aggregated: AggregatedSentiment) -> str:
        """Generate reasoning for signal."""
        reasons = []

        for indicator, score in aggregated.source_scores.items():
            if score > 0.3:
                reasons.append(f"{indicator} bullish ({score:.2f})")
            elif score < -0.3:
                reasons.append(f"{indicator} bearish ({score:.2f})")

        if reasons:
            return "; ".join(reasons)
        return "Market indicators neutral"


def create_market_analyzer(
    config: SentimentConfig | None = None,
    lookback_period: int = 20,
) -> MarketSentimentAnalyzer:
    """
    Create a market sentiment analyzer.

    Args:
        config: Sentiment configuration
        lookback_period: Historical lookback period

    Returns:
        Market sentiment analyzer instance
    """
    return MarketSentimentAnalyzer(config, lookback_period)
