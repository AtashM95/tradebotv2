"""
Market Scanner Module for Ultimate Trading Bot v2.2.

This module scans the market for trading opportunities
based on configurable criteria and filters.
"""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from src.analysis.technical_indicators import TechnicalIndicators
from src.analysis.signal_generator import SignalGenerator, SignalType, SignalTimeframe
from src.analysis.trend_analysis import TrendAnalyzer, TrendDirection
from src.utils.helpers import generate_uuid
from src.utils.date_utils import now_utc


logger = logging.getLogger(__name__)


class ScanType(str, Enum):
    """Scan type enumeration."""

    BREAKOUT = "breakout"
    OVERSOLD = "oversold"
    OVERBOUGHT = "overbought"
    VOLUME_SPIKE = "volume_spike"
    TREND_FOLLOWING = "trend_following"
    REVERSAL = "reversal"
    GAP = "gap"
    MOMENTUM = "momentum"
    CUSTOM = "custom"


class ScanResult(BaseModel):
    """Scan result model."""

    scan_id: str = Field(default_factory=generate_uuid)
    symbol: str
    scan_type: ScanType
    score: float = Field(ge=0.0, le=100.0)
    current_price: float
    signal: SignalType
    indicators: dict = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    risk_reward: Optional[float] = None
    timestamp: datetime = Field(default_factory=now_utc)


class ScanCriteria(BaseModel):
    """Scan criteria configuration."""

    scan_type: ScanType
    min_price: float = Field(default=1.0, ge=0.0)
    max_price: float = Field(default=10000.0, ge=0.0)
    min_volume: int = Field(default=100000, ge=0)
    min_avg_volume: int = Field(default=500000, ge=0)
    rsi_oversold: float = Field(default=30.0)
    rsi_overbought: float = Field(default=70.0)
    volume_spike_multiplier: float = Field(default=2.0, ge=1.0)
    breakout_threshold: float = Field(default=0.02, ge=0.0)
    atr_multiplier: float = Field(default=2.0, ge=0.5)
    custom_filter: Optional[Callable] = None


class ScannerStats(BaseModel):
    """Scanner statistics."""

    total_scans: int = Field(default=0)
    symbols_scanned: int = Field(default=0)
    opportunities_found: int = Field(default=0)
    last_scan_time: Optional[datetime] = None
    avg_scan_duration_ms: float = Field(default=0.0)


class MarketScannerConfig(BaseModel):
    """Configuration for market scanner."""

    max_results_per_scan: int = Field(default=20, ge=1, le=100)
    min_score_threshold: float = Field(default=60.0, ge=0.0, le=100.0)
    concurrent_scans: int = Field(default=10, ge=1, le=50)
    cache_results_seconds: int = Field(default=60, ge=0, le=3600)
    include_detailed_analysis: bool = Field(default=True)


class MarketScanner:
    """
    Market opportunity scanner.

    Provides:
    - Multi-criteria scanning
    - Parallel symbol processing
    - Breakout detection
    - Oversold/Overbought screening
    - Volume spike detection
    - Custom filter support
    """

    def __init__(
        self,
        config: Optional[MarketScannerConfig] = None,
    ) -> None:
        """
        Initialize MarketScanner.

        Args:
            config: Scanner configuration
        """
        self._config = config or MarketScannerConfig()
        self._indicators = TechnicalIndicators()
        self._signal_generator = SignalGenerator()
        self._trend_analyzer = TrendAnalyzer()

        self._stats = ScannerStats()
        self._cache: dict[str, tuple[list[ScanResult], datetime]] = {}

        logger.info("MarketScanner initialized")

    async def scan_market(
        self,
        symbols: list[str],
        market_data: dict[str, dict],
        criteria: ScanCriteria,
    ) -> list[ScanResult]:
        """
        Scan market for opportunities.

        Args:
            symbols: List of symbols to scan
            market_data: Market data dictionary
            criteria: Scan criteria

        Returns:
            List of scan results
        """
        start_time = datetime.now()
        self._stats.total_scans += 1
        self._stats.symbols_scanned += len(symbols)

        cache_key = f"{criteria.scan_type.value}_{len(symbols)}"
        cached = self._get_cached_results(cache_key)
        if cached:
            return cached

        results: list[ScanResult] = []

        semaphore = asyncio.Semaphore(self._config.concurrent_scans)

        async def scan_symbol(symbol: str) -> Optional[ScanResult]:
            async with semaphore:
                data = market_data.get(symbol)
                if not data:
                    return None

                return self._scan_single_symbol(symbol, data, criteria)

        tasks = [scan_symbol(symbol) for symbol in symbols]
        scan_results = await asyncio.gather(*tasks)

        for result in scan_results:
            if result and result.score >= self._config.min_score_threshold:
                results.append(result)

        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:self._config.max_results_per_scan]

        self._stats.opportunities_found += len(results)
        self._stats.last_scan_time = now_utc()

        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        self._update_avg_duration(duration_ms)

        self._cache_results(cache_key, results)

        return results

    def _scan_single_symbol(
        self,
        symbol: str,
        data: dict,
        criteria: ScanCriteria,
    ) -> Optional[ScanResult]:
        """Scan a single symbol against criteria."""
        opens = data.get("opens", [])
        highs = data.get("highs", [])
        lows = data.get("lows", [])
        closes = data.get("closes", [])
        volumes = data.get("volumes", [])

        if not closes or len(closes) < 20:
            return None

        current_price = closes[-1]
        current_volume = volumes[-1] if volumes else 0

        if current_price < criteria.min_price or current_price > criteria.max_price:
            return None

        if current_volume < criteria.min_volume:
            return None

        avg_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else current_volume
        if avg_volume < criteria.min_avg_volume:
            return None

        if criteria.scan_type == ScanType.BREAKOUT:
            return self._scan_breakout(symbol, highs, lows, closes, volumes, criteria)
        elif criteria.scan_type == ScanType.OVERSOLD:
            return self._scan_oversold(symbol, closes, volumes, criteria)
        elif criteria.scan_type == ScanType.OVERBOUGHT:
            return self._scan_overbought(symbol, closes, volumes, criteria)
        elif criteria.scan_type == ScanType.VOLUME_SPIKE:
            return self._scan_volume_spike(symbol, closes, volumes, criteria)
        elif criteria.scan_type == ScanType.TREND_FOLLOWING:
            return self._scan_trend_following(symbol, highs, lows, closes, volumes, criteria)
        elif criteria.scan_type == ScanType.REVERSAL:
            return self._scan_reversal(symbol, highs, lows, closes, volumes, criteria)
        elif criteria.scan_type == ScanType.GAP:
            return self._scan_gap(symbol, opens, closes, volumes, criteria)
        elif criteria.scan_type == ScanType.MOMENTUM:
            return self._scan_momentum(symbol, closes, volumes, criteria)
        elif criteria.scan_type == ScanType.CUSTOM and criteria.custom_filter:
            return criteria.custom_filter(symbol, data, criteria)

        return None

    def _scan_breakout(
        self,
        symbol: str,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        volumes: list[float],
        criteria: ScanCriteria,
    ) -> Optional[ScanResult]:
        """Scan for breakout opportunities."""
        current_price = closes[-1]
        current_volume = volumes[-1] if volumes else 0

        lookback = min(20, len(highs) - 1)
        recent_high = max(highs[-lookback:-1])
        recent_low = min(lows[-lookback:-1])

        breakout_up = current_price > recent_high * (1 + criteria.breakout_threshold)
        breakout_down = current_price < recent_low * (1 - criteria.breakout_threshold)

        if not breakout_up and not breakout_down:
            return None

        avg_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else current_volume
        volume_confirmation = current_volume > avg_volume * 1.5

        reasons: list[str] = []
        score = 50.0

        if breakout_up:
            signal = SignalType.BUY
            reasons.append(f"Price broke above {lookback}-day high ${recent_high:.2f}")
            breakout_pct = (current_price - recent_high) / recent_high * 100
            score += min(20, breakout_pct * 5)
        else:
            signal = SignalType.SELL
            reasons.append(f"Price broke below {lookback}-day low ${recent_low:.2f}")
            breakout_pct = (recent_low - current_price) / recent_low * 100
            score += min(20, breakout_pct * 5)

        if volume_confirmation:
            reasons.append("Volume confirmation present")
            score += 15
        else:
            reasons.append("Low volume on breakout")
            score -= 10

        atr_values = self._indicators.atr(highs, lows, closes, 14)
        atr = atr_values[-1] if atr_values else current_price * 0.02

        if breakout_up:
            stop_loss = current_price - (atr * criteria.atr_multiplier)
            target_price = current_price + (atr * criteria.atr_multiplier * 2)
        else:
            stop_loss = current_price + (atr * criteria.atr_multiplier)
            target_price = current_price - (atr * criteria.atr_multiplier * 2)

        risk_reward = abs(target_price - current_price) / abs(current_price - stop_loss) if stop_loss != current_price else 0

        return ScanResult(
            symbol=symbol,
            scan_type=ScanType.BREAKOUT,
            score=min(100, max(0, score)),
            current_price=current_price,
            signal=signal,
            indicators={
                "recent_high": recent_high,
                "recent_low": recent_low,
                "volume_ratio": current_volume / avg_volume if avg_volume > 0 else 1,
                "atr": atr,
            },
            reasons=reasons,
            entry_price=current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            risk_reward=risk_reward,
        )

    def _scan_oversold(
        self,
        symbol: str,
        closes: list[float],
        volumes: list[float],
        criteria: ScanCriteria,
    ) -> Optional[ScanResult]:
        """Scan for oversold conditions."""
        rsi_values = self._indicators.rsi(closes, 14)
        current_rsi = rsi_values[-1] if rsi_values else 50

        if current_rsi >= criteria.rsi_oversold:
            return None

        current_price = closes[-1]
        reasons: list[str] = []
        score = 50.0

        reasons.append(f"RSI oversold at {current_rsi:.1f}")
        score += (criteria.rsi_oversold - current_rsi) * 2

        if len(rsi_values) >= 2 and rsi_values[-1] > rsi_values[-2]:
            reasons.append("RSI showing bullish divergence")
            score += 10

        bb_results = self._indicators.bollinger_bands(closes, 20, 2.0)
        if bb_results:
            current_bb = bb_results[-1]
            if current_price < current_bb.lower:
                reasons.append("Price below lower Bollinger Band")
                score += 10

        sma_20 = self._indicators.sma(closes, 20)
        if sma_20 and sma_20[-1] > current_price:
            pullback_pct = (sma_20[-1] - current_price) / sma_20[-1] * 100
            if pullback_pct > 5:
                reasons.append(f"Significant pullback from 20 SMA ({pullback_pct:.1f}%)")
                score += min(15, pullback_pct)

        return ScanResult(
            symbol=symbol,
            scan_type=ScanType.OVERSOLD,
            score=min(100, max(0, score)),
            current_price=current_price,
            signal=SignalType.BUY,
            indicators={
                "rsi": current_rsi,
                "sma_20": sma_20[-1] if sma_20 else 0,
            },
            reasons=reasons,
            entry_price=current_price,
        )

    def _scan_overbought(
        self,
        symbol: str,
        closes: list[float],
        volumes: list[float],
        criteria: ScanCriteria,
    ) -> Optional[ScanResult]:
        """Scan for overbought conditions."""
        rsi_values = self._indicators.rsi(closes, 14)
        current_rsi = rsi_values[-1] if rsi_values else 50

        if current_rsi <= criteria.rsi_overbought:
            return None

        current_price = closes[-1]
        reasons: list[str] = []
        score = 50.0

        reasons.append(f"RSI overbought at {current_rsi:.1f}")
        score += (current_rsi - criteria.rsi_overbought) * 2

        if len(rsi_values) >= 2 and rsi_values[-1] < rsi_values[-2]:
            reasons.append("RSI showing bearish divergence")
            score += 10

        bb_results = self._indicators.bollinger_bands(closes, 20, 2.0)
        if bb_results:
            current_bb = bb_results[-1]
            if current_price > current_bb.upper:
                reasons.append("Price above upper Bollinger Band")
                score += 10

        return ScanResult(
            symbol=symbol,
            scan_type=ScanType.OVERBOUGHT,
            score=min(100, max(0, score)),
            current_price=current_price,
            signal=SignalType.SELL,
            indicators={
                "rsi": current_rsi,
            },
            reasons=reasons,
            entry_price=current_price,
        )

    def _scan_volume_spike(
        self,
        symbol: str,
        closes: list[float],
        volumes: list[float],
        criteria: ScanCriteria,
    ) -> Optional[ScanResult]:
        """Scan for volume spike opportunities."""
        if len(volumes) < 20:
            return None

        current_volume = volumes[-1]
        avg_volume = sum(volumes[-20:-1]) / 19

        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        if volume_ratio < criteria.volume_spike_multiplier:
            return None

        current_price = closes[-1]
        prev_price = closes[-2] if len(closes) >= 2 else current_price
        price_change = (current_price - prev_price) / prev_price * 100

        reasons: list[str] = []
        score = 50.0

        reasons.append(f"Volume spike: {volume_ratio:.1f}x average")
        score += min(30, (volume_ratio - 1) * 15)

        if price_change > 0:
            signal = SignalType.BUY
            reasons.append(f"Price up {price_change:.1f}% on high volume")
            score += min(15, abs(price_change) * 3)
        else:
            signal = SignalType.SELL
            reasons.append(f"Price down {abs(price_change):.1f}% on high volume")
            score += min(15, abs(price_change) * 3)

        return ScanResult(
            symbol=symbol,
            scan_type=ScanType.VOLUME_SPIKE,
            score=min(100, max(0, score)),
            current_price=current_price,
            signal=signal,
            indicators={
                "volume_ratio": volume_ratio,
                "avg_volume": avg_volume,
                "current_volume": current_volume,
                "price_change_pct": price_change,
            },
            reasons=reasons,
            entry_price=current_price,
        )

    def _scan_trend_following(
        self,
        symbol: str,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        volumes: list[float],
        criteria: ScanCriteria,
    ) -> Optional[ScanResult]:
        """Scan for trend following opportunities."""
        trend_result = self._trend_analyzer.analyze_trend(highs, lows, closes, volumes)

        bullish_trends = {TrendDirection.STRONG_BULLISH, TrendDirection.BULLISH}
        bearish_trends = {TrendDirection.STRONG_BEARISH, TrendDirection.BEARISH}

        if trend_result.direction not in bullish_trends and trend_result.direction not in bearish_trends:
            return None

        current_price = closes[-1]
        reasons: list[str] = []
        score = 50.0

        if trend_result.direction in bullish_trends:
            signal = SignalType.BUY
            reasons.append(f"Strong uptrend detected (ADX: {trend_result.adx_value:.1f})")
        else:
            signal = SignalType.SELL
            reasons.append(f"Strong downtrend detected (ADX: {trend_result.adx_value:.1f})")

        if trend_result.adx_value > 30:
            score += 20
            reasons.append("Very strong trend strength")
        elif trend_result.adx_value > 25:
            score += 10
            reasons.append("Strong trend strength")

        if trend_result.ma_alignment in ["bullish", "bearish"]:
            score += 15
            reasons.append(f"MAs aligned {trend_result.ma_alignment}ly")

        score += trend_result.confidence * 15

        return ScanResult(
            symbol=symbol,
            scan_type=ScanType.TREND_FOLLOWING,
            score=min(100, max(0, score)),
            current_price=current_price,
            signal=signal,
            indicators={
                "adx": trend_result.adx_value,
                "trend_direction": trend_result.direction.value,
                "trend_strength": trend_result.strength.value,
                "ma_alignment": trend_result.ma_alignment,
            },
            reasons=reasons,
            entry_price=current_price,
            target_price=trend_result.resistance_level if signal == SignalType.BUY else trend_result.support_level,
            stop_loss=trend_result.support_level if signal == SignalType.BUY else trend_result.resistance_level,
        )

    def _scan_reversal(
        self,
        symbol: str,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        volumes: list[float],
        criteria: ScanCriteria,
    ) -> Optional[ScanResult]:
        """Scan for reversal opportunities."""
        trend_result = self._trend_analyzer.analyze_trend(highs, lows, closes, volumes)

        rsi_values = self._indicators.rsi(closes, 14)
        current_rsi = rsi_values[-1] if rsi_values else 50

        reasons: list[str] = []
        score = 0.0
        signal = None

        if current_rsi < 30 and trend_result.direction in [TrendDirection.BEARISH, TrendDirection.STRONG_BEARISH]:
            signal = SignalType.BUY
            reasons.append("Potential bullish reversal: RSI oversold in downtrend")
            score = 60

            if len(rsi_values) >= 2 and rsi_values[-1] > rsi_values[-2]:
                reasons.append("RSI turning up")
                score += 15

        elif current_rsi > 70 and trend_result.direction in [TrendDirection.BULLISH, TrendDirection.STRONG_BULLISH]:
            signal = SignalType.SELL
            reasons.append("Potential bearish reversal: RSI overbought in uptrend")
            score = 60

            if len(rsi_values) >= 2 and rsi_values[-1] < rsi_values[-2]:
                reasons.append("RSI turning down")
                score += 15

        if not signal:
            return None

        current_price = closes[-1]

        return ScanResult(
            symbol=symbol,
            scan_type=ScanType.REVERSAL,
            score=min(100, max(0, score)),
            current_price=current_price,
            signal=signal,
            indicators={
                "rsi": current_rsi,
                "trend_direction": trend_result.direction.value,
            },
            reasons=reasons,
            entry_price=current_price,
        )

    def _scan_gap(
        self,
        symbol: str,
        opens: list[float],
        closes: list[float],
        volumes: list[float],
        criteria: ScanCriteria,
    ) -> Optional[ScanResult]:
        """Scan for gap opportunities."""
        if len(opens) < 2 or len(closes) < 2:
            return None

        current_open = opens[-1]
        prev_close = closes[-2]

        gap_pct = (current_open - prev_close) / prev_close * 100

        if abs(gap_pct) < 1.0:
            return None

        current_price = closes[-1]
        reasons: list[str] = []
        score = 50.0

        if gap_pct > 0:
            reasons.append(f"Gap up: {gap_pct:.1f}%")
            if current_price < current_open:
                signal = SignalType.BUY
                reasons.append("Gap fill opportunity (price retracing)")
                score += 20
            else:
                signal = SignalType.BUY
                reasons.append("Gap and go opportunity")
                score += 15
        else:
            reasons.append(f"Gap down: {abs(gap_pct):.1f}%")
            if current_price > current_open:
                signal = SignalType.SELL
                reasons.append("Gap fill opportunity (price retracing)")
                score += 20
            else:
                signal = SignalType.SELL
                reasons.append("Gap continuation")
                score += 15

        score += min(20, abs(gap_pct) * 5)

        return ScanResult(
            symbol=symbol,
            scan_type=ScanType.GAP,
            score=min(100, max(0, score)),
            current_price=current_price,
            signal=signal,
            indicators={
                "gap_pct": gap_pct,
                "open_price": current_open,
                "prev_close": prev_close,
            },
            reasons=reasons,
            entry_price=current_price,
            target_price=prev_close,
        )

    def _scan_momentum(
        self,
        symbol: str,
        closes: list[float],
        volumes: list[float],
        criteria: ScanCriteria,
    ) -> Optional[ScanResult]:
        """Scan for momentum opportunities."""
        if len(closes) < 20:
            return None

        roc_5 = (closes[-1] - closes[-6]) / closes[-6] * 100 if len(closes) >= 6 else 0
        roc_10 = (closes[-1] - closes[-11]) / closes[-11] * 100 if len(closes) >= 11 else 0

        if abs(roc_5) < 2 and abs(roc_10) < 5:
            return None

        current_price = closes[-1]
        reasons: list[str] = []
        score = 50.0

        if roc_5 > 0 and roc_10 > 0:
            signal = SignalType.BUY
            reasons.append(f"Strong upward momentum (5-day ROC: {roc_5:.1f}%, 10-day: {roc_10:.1f}%)")
            score += min(25, (roc_5 + roc_10) * 2)
        elif roc_5 < 0 and roc_10 < 0:
            signal = SignalType.SELL
            reasons.append(f"Strong downward momentum (5-day ROC: {roc_5:.1f}%, 10-day: {roc_10:.1f}%)")
            score += min(25, abs(roc_5 + roc_10) * 2)
        else:
            return None

        rsi_values = self._indicators.rsi(closes, 14)
        current_rsi = rsi_values[-1] if rsi_values else 50

        if signal == SignalType.BUY and 40 < current_rsi < 70:
            reasons.append("RSI confirms bullish momentum without being overbought")
            score += 10
        elif signal == SignalType.SELL and 30 < current_rsi < 60:
            reasons.append("RSI confirms bearish momentum without being oversold")
            score += 10

        return ScanResult(
            symbol=symbol,
            scan_type=ScanType.MOMENTUM,
            score=min(100, max(0, score)),
            current_price=current_price,
            signal=signal,
            indicators={
                "roc_5": roc_5,
                "roc_10": roc_10,
                "rsi": current_rsi,
            },
            reasons=reasons,
            entry_price=current_price,
        )

    def _get_cached_results(self, cache_key: str) -> Optional[list[ScanResult]]:
        """Get cached results if valid."""
        cached = self._cache.get(cache_key)
        if not cached:
            return None

        results, timestamp = cached
        age = (now_utc() - timestamp).total_seconds()

        if age > self._config.cache_results_seconds:
            del self._cache[cache_key]
            return None

        return results

    def _cache_results(self, cache_key: str, results: list[ScanResult]) -> None:
        """Cache scan results."""
        if self._config.cache_results_seconds > 0:
            self._cache[cache_key] = (results, now_utc())

    def _update_avg_duration(self, duration_ms: float) -> None:
        """Update average scan duration."""
        if self._stats.avg_scan_duration_ms == 0:
            self._stats.avg_scan_duration_ms = duration_ms
        else:
            self._stats.avg_scan_duration_ms = (
                self._stats.avg_scan_duration_ms * 0.9 + duration_ms * 0.1
            )

    def get_stats(self) -> ScannerStats:
        """Get scanner statistics."""
        return self._stats

    def clear_cache(self) -> int:
        """Clear scan cache."""
        count = len(self._cache)
        self._cache.clear()
        return count

    def __repr__(self) -> str:
        """String representation."""
        return f"MarketScanner(scans={self._stats.total_scans})"
