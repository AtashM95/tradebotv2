"""
VWAP Trading Strategy Module for Ultimate Trading Bot v2.2.

This module implements VWAP-based trading strategies for
institutional-style execution and mean reversion.
"""

import logging
from datetime import datetime, time
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.strategies.base_strategy import (
    BaseStrategy,
    StrategyConfig,
    StrategySignal,
    SignalAction,
    SignalSide,
    MarketData,
    StrategyContext,
)
from src.analysis.technical_indicators import TechnicalIndicators
from src.utils.helpers import generate_uuid


logger = logging.getLogger(__name__)


class VWAPLevel(BaseModel):
    """Model for VWAP level."""

    vwap: float
    upper_band_1: float
    upper_band_2: float
    lower_band_1: float
    lower_band_2: float
    std_dev: float


class VWAPSession(BaseModel):
    """Model for VWAP session tracking."""

    symbol: str
    session_date: str
    cumulative_volume: float = Field(default=0.0)
    cumulative_pv: float = Field(default=0.0)
    vwap: float = Field(default=0.0)
    vwap_history: list[float] = Field(default_factory=list)
    price_history: list[float] = Field(default_factory=list)
    volume_history: list[float] = Field(default_factory=list)


class VWAPStrategyConfig(StrategyConfig):
    """Configuration for VWAP strategy."""

    name: str = Field(default="VWAP Strategy")
    description: str = Field(
        default="VWAP-based trading for mean reversion and execution"
    )

    strategy_mode: str = Field(default="mean_reversion")

    band_multiplier_1: float = Field(default=1.0, ge=0.5, le=2.0)
    band_multiplier_2: float = Field(default=2.0, ge=1.0, le=3.0)

    entry_deviation_pct: float = Field(default=0.01, ge=0.005, le=0.05)
    exit_at_vwap: bool = Field(default=True)

    use_volume_confirmation: bool = Field(default=True)
    volume_threshold: float = Field(default=1.2, ge=1.0, le=3.0)

    session_start: str = Field(default="09:30")
    session_end: str = Field(default="16:00")
    avoid_first_minutes: int = Field(default=15, ge=0, le=60)
    avoid_last_minutes: int = Field(default=15, ge=0, le=60)

    anchor_type: str = Field(default="session")

    max_positions_per_session: int = Field(default=3, ge=1, le=10)
    position_hold_bars: int = Field(default=10, ge=1, le=100)

    stop_loss_bands: float = Field(default=2.5, ge=1.5, le=4.0)
    take_profit_bands: float = Field(default=0.5, ge=0.0, le=1.5)


class VWAPStrategy(BaseStrategy):
    """
    VWAP-based trading strategy.

    Features:
    - Session VWAP calculation
    - Standard deviation bands
    - Mean reversion entries
    - Volume confirmation
    - Session timing management
    """

    def __init__(
        self,
        config: Optional[VWAPStrategyConfig] = None,
    ) -> None:
        """
        Initialize VWAPStrategy.

        Args:
            config: VWAP configuration
        """
        config = config or VWAPStrategyConfig()
        super().__init__(config)

        self._vwap_config = config
        self._indicators = TechnicalIndicators()

        self._sessions: dict[str, VWAPSession] = {}
        self._positions_today: dict[str, int] = {}

        logger.info(f"VWAPStrategy initialized: {self.name}")

    def calculate_indicators(
        self,
        symbol: str,
        data: MarketData,
    ) -> dict[str, Any]:
        """
        Calculate VWAP indicators.

        Args:
            symbol: Trading symbol
            data: Market data

        Returns:
            Dictionary of indicator values
        """
        closes = data.closes
        highs = data.highs
        lows = data.lows
        volumes = data.volumes

        if len(closes) < 10:
            return {}

        session = self._get_or_create_session(symbol, data)

        vwap_level = self._calculate_vwap_level(session)

        current_price = closes[-1]
        deviation = (current_price - vwap_level.vwap) / vwap_level.vwap if vwap_level.vwap > 0 else 0
        deviation_bands = abs(current_price - vwap_level.vwap) / vwap_level.std_dev if vwap_level.std_dev > 0 else 0

        avg_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else volumes[-1]
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        trend = self._determine_vwap_trend(session)

        return {
            "current_price": current_price,
            "vwap": vwap_level.vwap,
            "upper_band_1": vwap_level.upper_band_1,
            "upper_band_2": vwap_level.upper_band_2,
            "lower_band_1": vwap_level.lower_band_1,
            "lower_band_2": vwap_level.lower_band_2,
            "std_dev": vwap_level.std_dev,
            "deviation": deviation,
            "deviation_bands": deviation_bands,
            "volume_ratio": volume_ratio,
            "trend": trend,
            "session_volume": session.cumulative_volume,
        }

    async def evaluate(
        self,
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """
        Evaluate VWAP trading opportunities.

        Args:
            market_data: Market data for all symbols
            context: Strategy execution context

        Returns:
            List of VWAP signals
        """
        signals: list[StrategySignal] = []

        if not self._is_valid_trading_time(context.timestamp):
            return signals

        for symbol in self.config.symbols:
            if symbol not in market_data:
                continue

            data = market_data[symbol]

            indicators = self.calculate_indicators(symbol, data)
            if not indicators:
                continue

            signal = self._generate_vwap_signal(symbol, indicators, context)

            if signal:
                signals.append(signal)

        return signals

    def _get_or_create_session(
        self,
        symbol: str,
        data: MarketData,
    ) -> VWAPSession:
        """Get or create VWAP session for symbol."""
        from src.utils.date_utils import now_utc

        today = now_utc().strftime("%Y-%m-%d")
        session_key = f"{symbol}_{today}"

        if session_key not in self._sessions:
            self._sessions[session_key] = VWAPSession(
                symbol=symbol,
                session_date=today,
            )

        session = self._sessions[session_key]

        if data.closes and data.volumes:
            typical_price = (data.highs[-1] + data.lows[-1] + data.closes[-1]) / 3
            volume = data.volumes[-1]

            session.cumulative_volume += volume
            session.cumulative_pv += typical_price * volume

            if session.cumulative_volume > 0:
                session.vwap = session.cumulative_pv / session.cumulative_volume

            session.vwap_history.append(session.vwap)
            session.price_history.append(data.closes[-1])
            session.volume_history.append(volume)

            if len(session.vwap_history) > 1000:
                session.vwap_history = session.vwap_history[-1000:]
                session.price_history = session.price_history[-1000:]
                session.volume_history = session.volume_history[-1000:]

        return session

    def _calculate_vwap_level(self, session: VWAPSession) -> VWAPLevel:
        """Calculate VWAP with standard deviation bands."""
        vwap = session.vwap if session.vwap > 0 else 0

        if len(session.price_history) >= 10:
            deviations = [p - vwap for p in session.price_history[-100:]]
            variance = sum(d ** 2 for d in deviations) / len(deviations)
            std_dev = variance ** 0.5
        else:
            std_dev = vwap * 0.01 if vwap > 0 else 1.0

        mult_1 = self._vwap_config.band_multiplier_1
        mult_2 = self._vwap_config.band_multiplier_2

        return VWAPLevel(
            vwap=vwap,
            upper_band_1=vwap + std_dev * mult_1,
            upper_band_2=vwap + std_dev * mult_2,
            lower_band_1=vwap - std_dev * mult_1,
            lower_band_2=vwap - std_dev * mult_2,
            std_dev=std_dev,
        )

    def _determine_vwap_trend(self, session: VWAPSession) -> str:
        """Determine trend relative to VWAP."""
        if len(session.price_history) < 5:
            return "neutral"

        recent_prices = session.price_history[-10:]
        above_vwap = sum(1 for p in recent_prices if p > session.vwap)

        if above_vwap >= 8:
            return "bullish"
        elif above_vwap <= 2:
            return "bearish"
        else:
            return "neutral"

    def _is_valid_trading_time(self, timestamp: datetime) -> bool:
        """Check if current time is valid for trading."""
        current_time = timestamp.time()

        start_parts = self._vwap_config.session_start.split(":")
        end_parts = self._vwap_config.session_end.split(":")

        session_start = time(int(start_parts[0]), int(start_parts[1]))
        session_end = time(int(end_parts[0]), int(end_parts[1]))

        if current_time < session_start or current_time > session_end:
            return False

        start_minutes = session_start.hour * 60 + session_start.minute
        current_minutes = current_time.hour * 60 + current_time.minute
        end_minutes = session_end.hour * 60 + session_end.minute

        if current_minutes - start_minutes < self._vwap_config.avoid_first_minutes:
            return False

        if end_minutes - current_minutes < self._vwap_config.avoid_last_minutes:
            return False

        return True

    def _generate_vwap_signal(
        self,
        symbol: str,
        indicators: dict[str, Any],
        context: StrategyContext,
    ) -> Optional[StrategySignal]:
        """Generate VWAP trading signal."""
        current_price = indicators["current_price"]
        vwap = indicators["vwap"]
        deviation = indicators["deviation"]
        deviation_bands = indicators["deviation_bands"]
        volume_ratio = indicators["volume_ratio"]
        upper_band_1 = indicators["upper_band_1"]
        upper_band_2 = indicators["upper_band_2"]
        lower_band_1 = indicators["lower_band_1"]
        lower_band_2 = indicators["lower_band_2"]
        std_dev = indicators["std_dev"]

        today = context.timestamp.strftime("%Y-%m-%d")
        position_key = f"{symbol}_{today}"
        positions_taken = self._positions_today.get(position_key, 0)

        if positions_taken >= self._vwap_config.max_positions_per_session:
            return None

        if self._vwap_config.use_volume_confirmation:
            if volume_ratio < self._vwap_config.volume_threshold:
                return None

        mode = self._vwap_config.strategy_mode

        if mode == "mean_reversion":
            return self._generate_mean_reversion_signal(
                symbol, current_price, vwap, deviation_bands,
                lower_band_1, lower_band_2, upper_band_1, upper_band_2,
                std_dev, volume_ratio, position_key
            )

        elif mode == "trend_following":
            return self._generate_trend_signal(
                symbol, current_price, vwap, deviation,
                volume_ratio, position_key
            )

        elif mode == "breakout":
            return self._generate_breakout_signal(
                symbol, current_price, vwap, deviation_bands,
                upper_band_2, lower_band_2, std_dev, position_key
            )

        return None

    def _generate_mean_reversion_signal(
        self,
        symbol: str,
        current_price: float,
        vwap: float,
        deviation_bands: float,
        lower_band_1: float,
        lower_band_2: float,
        upper_band_1: float,
        upper_band_2: float,
        std_dev: float,
        volume_ratio: float,
        position_key: str,
    ) -> Optional[StrategySignal]:
        """Generate mean reversion signal."""
        if current_price <= lower_band_1:
            stop_loss = vwap - std_dev * self._vwap_config.stop_loss_bands

            if self._vwap_config.exit_at_vwap:
                take_profit = vwap
            else:
                take_profit = vwap + std_dev * self._vwap_config.take_profit_bands

            strength = min(1.0, deviation_bands / 2)
            confidence = 0.7 + min(0.2, (volume_ratio - 1) * 0.1)

            self._positions_today[position_key] = self._positions_today.get(position_key, 0) + 1

            return self.create_signal(
                symbol=symbol,
                action=SignalAction.BUY,
                side=SignalSide.LONG,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strength=strength,
                confidence=confidence,
                reason=f"VWAP mean reversion buy at {deviation_bands:.1f} bands below",
                metadata={
                    "strategy_type": "vwap",
                    "mode": "mean_reversion",
                    "vwap": vwap,
                    "deviation_bands": deviation_bands,
                    "volume_ratio": volume_ratio,
                },
            )

        elif current_price >= upper_band_1:
            stop_loss = vwap + std_dev * self._vwap_config.stop_loss_bands

            if self._vwap_config.exit_at_vwap:
                take_profit = vwap
            else:
                take_profit = vwap - std_dev * self._vwap_config.take_profit_bands

            strength = min(1.0, deviation_bands / 2)
            confidence = 0.7 + min(0.2, (volume_ratio - 1) * 0.1)

            self._positions_today[position_key] = self._positions_today.get(position_key, 0) + 1

            return self.create_signal(
                symbol=symbol,
                action=SignalAction.SELL,
                side=SignalSide.SHORT,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strength=strength,
                confidence=confidence,
                reason=f"VWAP mean reversion sell at {deviation_bands:.1f} bands above",
                metadata={
                    "strategy_type": "vwap",
                    "mode": "mean_reversion",
                    "vwap": vwap,
                    "deviation_bands": deviation_bands,
                    "volume_ratio": volume_ratio,
                },
            )

        return None

    def _generate_trend_signal(
        self,
        symbol: str,
        current_price: float,
        vwap: float,
        deviation: float,
        volume_ratio: float,
        position_key: str,
    ) -> Optional[StrategySignal]:
        """Generate trend following signal."""
        entry_threshold = self._vwap_config.entry_deviation_pct

        if deviation > entry_threshold and volume_ratio > self._vwap_config.volume_threshold:
            stop_loss = vwap * 0.99

            self._positions_today[position_key] = self._positions_today.get(position_key, 0) + 1

            return self.create_signal(
                symbol=symbol,
                action=SignalAction.BUY,
                side=SignalSide.LONG,
                entry_price=current_price,
                stop_loss=stop_loss,
                strength=min(1.0, deviation * 20),
                confidence=0.65 + min(0.2, (volume_ratio - 1) * 0.1),
                reason=f"VWAP trend follow buy, {deviation*100:.1f}% above VWAP",
                metadata={
                    "strategy_type": "vwap",
                    "mode": "trend_following",
                    "vwap": vwap,
                    "deviation": deviation,
                },
            )

        elif deviation < -entry_threshold and volume_ratio > self._vwap_config.volume_threshold:
            stop_loss = vwap * 1.01

            self._positions_today[position_key] = self._positions_today.get(position_key, 0) + 1

            return self.create_signal(
                symbol=symbol,
                action=SignalAction.SELL,
                side=SignalSide.SHORT,
                entry_price=current_price,
                stop_loss=stop_loss,
                strength=min(1.0, abs(deviation) * 20),
                confidence=0.65 + min(0.2, (volume_ratio - 1) * 0.1),
                reason=f"VWAP trend follow sell, {abs(deviation)*100:.1f}% below VWAP",
                metadata={
                    "strategy_type": "vwap",
                    "mode": "trend_following",
                    "vwap": vwap,
                    "deviation": deviation,
                },
            )

        return None

    def _generate_breakout_signal(
        self,
        symbol: str,
        current_price: float,
        vwap: float,
        deviation_bands: float,
        upper_band_2: float,
        lower_band_2: float,
        std_dev: float,
        position_key: str,
    ) -> Optional[StrategySignal]:
        """Generate breakout signal."""
        if current_price > upper_band_2:
            stop_loss = vwap

            self._positions_today[position_key] = self._positions_today.get(position_key, 0) + 1

            return self.create_signal(
                symbol=symbol,
                action=SignalAction.BUY,
                side=SignalSide.LONG,
                entry_price=current_price,
                stop_loss=stop_loss,
                strength=min(1.0, deviation_bands / 3),
                confidence=0.6,
                reason=f"VWAP breakout buy above 2-band",
                metadata={
                    "strategy_type": "vwap",
                    "mode": "breakout",
                    "vwap": vwap,
                    "deviation_bands": deviation_bands,
                },
            )

        elif current_price < lower_band_2:
            stop_loss = vwap

            self._positions_today[position_key] = self._positions_today.get(position_key, 0) + 1

            return self.create_signal(
                symbol=symbol,
                action=SignalAction.SELL,
                side=SignalSide.SHORT,
                entry_price=current_price,
                stop_loss=stop_loss,
                strength=min(1.0, deviation_bands / 3),
                confidence=0.6,
                reason=f"VWAP breakout sell below 2-band",
                metadata={
                    "strategy_type": "vwap",
                    "mode": "breakout",
                    "vwap": vwap,
                    "deviation_bands": deviation_bands,
                },
            )

        return None

    def reset_session(self, symbol: str) -> None:
        """Reset VWAP session for symbol."""
        from src.utils.date_utils import now_utc

        today = now_utc().strftime("%Y-%m-%d")
        session_key = f"{symbol}_{today}"

        if session_key in self._sessions:
            del self._sessions[session_key]

        position_key = f"{symbol}_{today}"
        if position_key in self._positions_today:
            del self._positions_today[position_key]

    def get_session_data(self, symbol: str) -> Optional[VWAPSession]:
        """Get current session data for symbol."""
        from src.utils.date_utils import now_utc

        today = now_utc().strftime("%Y-%m-%d")
        session_key = f"{symbol}_{today}"

        return self._sessions.get(session_key)

    def get_vwap_statistics(self) -> dict:
        """Get VWAP strategy statistics."""
        return {
            "active_sessions": len(self._sessions),
            "positions_today": dict(self._positions_today),
            "sessions": {
                key: {
                    "symbol": session.symbol,
                    "vwap": session.vwap,
                    "cumulative_volume": session.cumulative_volume,
                    "data_points": len(session.vwap_history),
                }
                for key, session in self._sessions.items()
            },
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"VWAPStrategy(sessions={len(self._sessions)})"
