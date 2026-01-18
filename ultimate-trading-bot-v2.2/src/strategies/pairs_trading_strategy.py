"""
Pairs Trading Strategy Module for Ultimate Trading Bot v2.2.

This module implements statistical arbitrage through pairs trading
on correlated assets.
"""

import logging
from datetime import datetime
from typing import Any, Optional

import numpy as np
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
from src.analysis.correlation_analysis import CorrelationAnalyzer
from src.utils.helpers import generate_uuid


logger = logging.getLogger(__name__)


class TradingPair(BaseModel):
    """Model for a trading pair."""

    symbol1: str
    symbol2: str
    correlation: float = Field(ge=-1.0, le=1.0)
    hedge_ratio: float = Field(default=1.0)
    mean_spread: float = Field(default=0.0)
    std_spread: float = Field(default=1.0)
    half_life: float = Field(default=10.0)
    is_cointegrated: bool = Field(default=False)


class PairPosition(BaseModel):
    """Model for pair position."""

    pair: TradingPair
    entry_date: datetime
    entry_spread: float
    entry_zscore: float
    position_type: str
    symbol1_qty: float
    symbol2_qty: float
    symbol1_entry: float
    symbol2_entry: float


class PairsTradingConfig(StrategyConfig):
    """Configuration for pairs trading strategy."""

    name: str = Field(default="Pairs Trading Strategy")
    description: str = Field(
        default="Statistical arbitrage on correlated asset pairs"
    )

    min_correlation: float = Field(default=0.7, ge=0.5, le=0.95)
    lookback_period: int = Field(default=60, ge=20, le=252)
    zscore_entry: float = Field(default=2.0, ge=1.0, le=4.0)
    zscore_exit: float = Field(default=0.5, ge=0.0, le=1.5)
    zscore_stop: float = Field(default=3.5, ge=2.5, le=5.0)

    cointegration_pvalue: float = Field(default=0.05, ge=0.01, le=0.1)
    require_cointegration: bool = Field(default=True)

    half_life_min: float = Field(default=5.0, ge=1.0, le=20.0)
    half_life_max: float = Field(default=50.0, ge=20.0, le=100.0)

    position_size_per_leg: float = Field(default=0.05, ge=0.01, le=0.2)
    max_pairs: int = Field(default=5, ge=1, le=20)

    recalculate_interval_days: int = Field(default=5, ge=1, le=30)


class PairsTradingStrategy(BaseStrategy):
    """
    Statistical arbitrage pairs trading strategy.

    Features:
    - Cointegration testing
    - Dynamic hedge ratio calculation
    - Z-score based entry/exit
    - Half-life optimization
    - Multi-pair management
    """

    def __init__(
        self,
        config: Optional[PairsTradingConfig] = None,
    ) -> None:
        """
        Initialize PairsTradingStrategy.

        Args:
            config: Pairs trading configuration
        """
        config = config or PairsTradingConfig()
        super().__init__(config)

        self._pairs_config = config
        self._correlation_analyzer = CorrelationAnalyzer()

        self._trading_pairs: list[TradingPair] = []
        self._active_positions: dict[str, PairPosition] = {}
        self._pair_history: dict[str, list[float]] = {}
        self._last_recalculation: Optional[datetime] = None

        logger.info(f"PairsTradingStrategy initialized: {self.name}")

    def calculate_indicators(
        self,
        symbol: str,
        data: MarketData,
    ) -> dict[str, Any]:
        """
        Calculate indicators for a symbol.

        Args:
            symbol: Trading symbol
            data: Market data

        Returns:
            Dictionary of indicator values
        """
        closes = data.closes

        if len(closes) < self._pairs_config.lookback_period:
            return {}

        returns = [
            (closes[i] - closes[i - 1]) / closes[i - 1]
            for i in range(1, len(closes))
        ]

        return {
            "prices": closes,
            "returns": returns,
            "current_price": closes[-1],
            "mean_price": sum(closes[-20:]) / 20,
            "volatility": np.std(returns[-20:]) if len(returns) >= 20 else 0,
        }

    async def evaluate(
        self,
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """
        Evaluate pairs trading opportunities.

        Args:
            market_data: Market data for all symbols
            context: Strategy execution context

        Returns:
            List of pairs trading signals
        """
        signals: list[StrategySignal] = []

        if self._should_recalculate_pairs(context):
            self._recalculate_pairs(market_data)

        for pair_key, position in list(self._active_positions.items()):
            exit_signals = self._check_pair_exit(position, market_data, context)
            if exit_signals:
                signals.extend(exit_signals)
                del self._active_positions[pair_key]

        if len(self._active_positions) >= self._pairs_config.max_pairs:
            return signals

        for pair in self._trading_pairs:
            pair_key = f"{pair.symbol1}_{pair.symbol2}"

            if pair_key in self._active_positions:
                continue

            if pair.symbol1 not in market_data or pair.symbol2 not in market_data:
                continue

            entry_signals = self._check_pair_entry(pair, market_data, context)
            if entry_signals:
                signals.extend(entry_signals)

        return signals

    def _should_recalculate_pairs(self, context: StrategyContext) -> bool:
        """Check if pairs should be recalculated."""
        if not self._trading_pairs:
            return True

        if self._last_recalculation is None:
            return True

        days_since = (context.timestamp - self._last_recalculation).days
        return days_since >= self._pairs_config.recalculate_interval_days

    def _recalculate_pairs(self, market_data: dict[str, MarketData]) -> None:
        """Recalculate trading pairs from market data."""
        from src.utils.date_utils import now_utc

        symbols = list(market_data.keys())
        if len(symbols) < 2:
            return

        price_data: dict[str, list[float]] = {}
        for symbol, data in market_data.items():
            if len(data.closes) >= self._pairs_config.lookback_period:
                price_data[symbol] = data.closes

        candidates: list[TradingPair] = []

        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i + 1:]:
                if sym1 not in price_data or sym2 not in price_data:
                    continue

                pair = self._analyze_pair(
                    sym1, sym2,
                    price_data[sym1],
                    price_data[sym2],
                )

                if pair and pair.correlation >= self._pairs_config.min_correlation:
                    if not self._pairs_config.require_cointegration or pair.is_cointegrated:
                        candidates.append(pair)

        candidates.sort(key=lambda p: (p.is_cointegrated, p.correlation), reverse=True)

        self._trading_pairs = candidates[:self._pairs_config.max_pairs * 2]
        self._last_recalculation = now_utc()

        logger.info(f"Recalculated {len(self._trading_pairs)} trading pairs")

    def _analyze_pair(
        self,
        symbol1: str,
        symbol2: str,
        prices1: list[float],
        prices2: list[float],
    ) -> Optional[TradingPair]:
        """Analyze a potential trading pair."""
        n = min(len(prices1), len(prices2))
        if n < self._pairs_config.lookback_period:
            return None

        p1 = prices1[-n:]
        p2 = prices2[-n:]

        correlation = self._correlation_analyzer.calculate_correlation(
            self._calculate_returns(p1),
            self._calculate_returns(p2),
        )

        if correlation < self._pairs_config.min_correlation:
            return None

        hedge_ratio = self._calculate_hedge_ratio(p1, p2)

        spread = [p1[i] - hedge_ratio * p2[i] for i in range(n)]
        mean_spread = sum(spread) / len(spread)
        std_spread = np.std(spread)

        is_cointegrated, half_life = self._test_cointegration(spread)

        if self._pairs_config.require_cointegration and not is_cointegrated:
            return None

        if half_life < self._pairs_config.half_life_min or half_life > self._pairs_config.half_life_max:
            return None

        return TradingPair(
            symbol1=symbol1,
            symbol2=symbol2,
            correlation=correlation,
            hedge_ratio=hedge_ratio,
            mean_spread=mean_spread,
            std_spread=std_spread if std_spread > 0 else 1.0,
            half_life=half_life,
            is_cointegrated=is_cointegrated,
        )

    def _calculate_hedge_ratio(
        self,
        prices1: list[float],
        prices2: list[float],
    ) -> float:
        """Calculate optimal hedge ratio using OLS regression."""
        n = len(prices1)

        mean_x = sum(prices2) / n
        mean_y = sum(prices1) / n

        numerator = sum((prices2[i] - mean_x) * (prices1[i] - mean_y) for i in range(n))
        denominator = sum((prices2[i] - mean_x) ** 2 for i in range(n))

        if denominator == 0:
            return 1.0

        return numerator / denominator

    def _test_cointegration(
        self,
        spread: list[float],
    ) -> tuple[bool, float]:
        """
        Test for cointegration using ADF test approximation.

        Returns:
            Tuple of (is_cointegrated, half_life)
        """
        n = len(spread)
        if n < 20:
            return False, 100.0

        lagged_spread = spread[:-1]
        delta_spread = [spread[i + 1] - spread[i] for i in range(n - 1)]

        mean_lag = sum(lagged_spread) / len(lagged_spread)
        mean_delta = sum(delta_spread) / len(delta_spread)

        numerator = sum(
            (lagged_spread[i] - mean_lag) * (delta_spread[i] - mean_delta)
            for i in range(len(lagged_spread))
        )
        denominator = sum((lagged_spread[i] - mean_lag) ** 2 for i in range(len(lagged_spread)))

        if denominator == 0:
            return False, 100.0

        gamma = numerator / denominator

        if gamma >= 0:
            return False, 100.0

        half_life = -np.log(2) / gamma if gamma < 0 else 100.0

        residuals = [
            delta_spread[i] - gamma * lagged_spread[i]
            for i in range(len(lagged_spread))
        ]
        std_residual = np.std(residuals)
        std_lag = np.std(lagged_spread)

        if std_residual == 0 or std_lag == 0:
            return False, half_life

        t_stat = gamma * std_lag * np.sqrt(n) / std_residual

        critical_value_5pct = -2.86

        is_cointegrated = t_stat < critical_value_5pct

        return is_cointegrated, abs(half_life)

    def _check_pair_entry(
        self,
        pair: TradingPair,
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """Check for pair entry signal."""
        price1 = market_data[pair.symbol1].closes[-1]
        price2 = market_data[pair.symbol2].closes[-1]

        current_spread = price1 - pair.hedge_ratio * price2
        zscore = (current_spread - pair.mean_spread) / pair.std_spread if pair.std_spread > 0 else 0

        signals: list[StrategySignal] = []

        if zscore >= self._pairs_config.zscore_entry:
            signals = self._create_pair_entry_signals(
                pair, "short_spread", price1, price2, zscore, context
            )

        elif zscore <= -self._pairs_config.zscore_entry:
            signals = self._create_pair_entry_signals(
                pair, "long_spread", price1, price2, zscore, context
            )

        return signals

    def _create_pair_entry_signals(
        self,
        pair: TradingPair,
        position_type: str,
        price1: float,
        price2: float,
        zscore: float,
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """Create entry signals for pair trade."""
        pair_key = f"{pair.symbol1}_{pair.symbol2}"

        position_value = context.account_value * self._pairs_config.position_size_per_leg

        qty1 = position_value / price1
        qty2 = qty1 * pair.hedge_ratio

        position = PairPosition(
            pair=pair,
            entry_date=context.timestamp,
            entry_spread=price1 - pair.hedge_ratio * price2,
            entry_zscore=zscore,
            position_type=position_type,
            symbol1_qty=qty1,
            symbol2_qty=qty2,
            symbol1_entry=price1,
            symbol2_entry=price2,
        )

        self._active_positions[pair_key] = position

        signals: list[StrategySignal] = []

        if position_type == "long_spread":
            signal1 = self.create_signal(
                symbol=pair.symbol1,
                action=SignalAction.BUY,
                side=SignalSide.LONG,
                entry_price=price1,
                strength=abs(zscore) / 3,
                confidence=0.7 + min(0.2, pair.correlation / 5),
                reason=f"Pairs long spread: z={zscore:.2f}",
                position_size_pct=self._pairs_config.position_size_per_leg,
                metadata={
                    "strategy_type": "pairs_trading",
                    "pair": pair_key,
                    "position_type": position_type,
                    "zscore": zscore,
                    "hedge_ratio": pair.hedge_ratio,
                    "leg": "long",
                },
            )

            signal2 = self.create_signal(
                symbol=pair.symbol2,
                action=SignalAction.SELL,
                side=SignalSide.SHORT,
                entry_price=price2,
                strength=abs(zscore) / 3,
                confidence=0.7 + min(0.2, pair.correlation / 5),
                reason=f"Pairs long spread: z={zscore:.2f}",
                position_size_pct=self._pairs_config.position_size_per_leg * pair.hedge_ratio,
                metadata={
                    "strategy_type": "pairs_trading",
                    "pair": pair_key,
                    "position_type": position_type,
                    "zscore": zscore,
                    "hedge_ratio": pair.hedge_ratio,
                    "leg": "short",
                },
            )

            if signal1:
                signals.append(signal1)
            if signal2:
                signals.append(signal2)

        else:
            signal1 = self.create_signal(
                symbol=pair.symbol1,
                action=SignalAction.SELL,
                side=SignalSide.SHORT,
                entry_price=price1,
                strength=abs(zscore) / 3,
                confidence=0.7 + min(0.2, pair.correlation / 5),
                reason=f"Pairs short spread: z={zscore:.2f}",
                position_size_pct=self._pairs_config.position_size_per_leg,
                metadata={
                    "strategy_type": "pairs_trading",
                    "pair": pair_key,
                    "position_type": position_type,
                    "zscore": zscore,
                    "hedge_ratio": pair.hedge_ratio,
                    "leg": "short",
                },
            )

            signal2 = self.create_signal(
                symbol=pair.symbol2,
                action=SignalAction.BUY,
                side=SignalSide.LONG,
                entry_price=price2,
                strength=abs(zscore) / 3,
                confidence=0.7 + min(0.2, pair.correlation / 5),
                reason=f"Pairs short spread: z={zscore:.2f}",
                position_size_pct=self._pairs_config.position_size_per_leg * pair.hedge_ratio,
                metadata={
                    "strategy_type": "pairs_trading",
                    "pair": pair_key,
                    "position_type": position_type,
                    "zscore": zscore,
                    "hedge_ratio": pair.hedge_ratio,
                    "leg": "long",
                },
            )

            if signal1:
                signals.append(signal1)
            if signal2:
                signals.append(signal2)

        return signals

    def _check_pair_exit(
        self,
        position: PairPosition,
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """Check for pair exit signal."""
        pair = position.pair

        if pair.symbol1 not in market_data or pair.symbol2 not in market_data:
            return []

        price1 = market_data[pair.symbol1].closes[-1]
        price2 = market_data[pair.symbol2].closes[-1]

        current_spread = price1 - pair.hedge_ratio * price2
        zscore = (current_spread - pair.mean_spread) / pair.std_spread if pair.std_spread > 0 else 0

        should_exit = False
        exit_reason = ""

        if position.position_type == "long_spread":
            if zscore >= -self._pairs_config.zscore_exit:
                should_exit = True
                exit_reason = f"mean_reversion_complete z={zscore:.2f}"

            if zscore <= -self._pairs_config.zscore_stop:
                should_exit = True
                exit_reason = f"stop_loss z={zscore:.2f}"

        else:
            if zscore <= self._pairs_config.zscore_exit:
                should_exit = True
                exit_reason = f"mean_reversion_complete z={zscore:.2f}"

            if zscore >= self._pairs_config.zscore_stop:
                should_exit = True
                exit_reason = f"stop_loss z={zscore:.2f}"

        if not should_exit:
            return []

        signals: list[StrategySignal] = []

        if position.position_type == "long_spread":
            signal1 = self.create_signal(
                symbol=pair.symbol1,
                action=SignalAction.SELL,
                side=SignalSide.FLAT,
                entry_price=price1,
                strength=1.0,
                confidence=0.9,
                reason=f"Pairs exit: {exit_reason}",
                metadata={
                    "strategy_type": "pairs_trading",
                    "exit_reason": exit_reason,
                    "entry_zscore": position.entry_zscore,
                    "exit_zscore": zscore,
                    "leg": "close_long",
                },
            )

            signal2 = self.create_signal(
                symbol=pair.symbol2,
                action=SignalAction.BUY,
                side=SignalSide.FLAT,
                entry_price=price2,
                strength=1.0,
                confidence=0.9,
                reason=f"Pairs exit: {exit_reason}",
                metadata={
                    "strategy_type": "pairs_trading",
                    "exit_reason": exit_reason,
                    "entry_zscore": position.entry_zscore,
                    "exit_zscore": zscore,
                    "leg": "close_short",
                },
            )

        else:
            signal1 = self.create_signal(
                symbol=pair.symbol1,
                action=SignalAction.BUY,
                side=SignalSide.FLAT,
                entry_price=price1,
                strength=1.0,
                confidence=0.9,
                reason=f"Pairs exit: {exit_reason}",
                metadata={
                    "strategy_type": "pairs_trading",
                    "exit_reason": exit_reason,
                    "entry_zscore": position.entry_zscore,
                    "exit_zscore": zscore,
                    "leg": "close_short",
                },
            )

            signal2 = self.create_signal(
                symbol=pair.symbol2,
                action=SignalAction.SELL,
                side=SignalSide.FLAT,
                entry_price=price2,
                strength=1.0,
                confidence=0.9,
                reason=f"Pairs exit: {exit_reason}",
                metadata={
                    "strategy_type": "pairs_trading",
                    "exit_reason": exit_reason,
                    "entry_zscore": position.entry_zscore,
                    "exit_zscore": zscore,
                    "leg": "close_long",
                },
            )

        if signal1:
            signals.append(signal1)
        if signal2:
            signals.append(signal2)

        return signals

    def _calculate_returns(self, prices: list[float]) -> list[float]:
        """Calculate returns from prices."""
        return [
            (prices[i] - prices[i - 1]) / prices[i - 1]
            for i in range(1, len(prices))
            if prices[i - 1] != 0
        ]

    def get_trading_pairs(self) -> list[TradingPair]:
        """Get current trading pairs."""
        return self._trading_pairs.copy()

    def get_active_positions(self) -> dict[str, PairPosition]:
        """Get active pair positions."""
        return self._active_positions.copy()

    def add_trading_pair(self, pair: TradingPair) -> None:
        """Manually add a trading pair."""
        self._trading_pairs.append(pair)

    def remove_trading_pair(self, symbol1: str, symbol2: str) -> bool:
        """Remove a trading pair."""
        for i, pair in enumerate(self._trading_pairs):
            if pair.symbol1 == symbol1 and pair.symbol2 == symbol2:
                del self._trading_pairs[i]
                return True
        return False

    def get_pair_statistics(self) -> dict:
        """Get pairs trading statistics."""
        return {
            "total_pairs": len(self._trading_pairs),
            "active_positions": len(self._active_positions),
            "cointegrated_pairs": sum(1 for p in self._trading_pairs if p.is_cointegrated),
            "avg_correlation": (
                sum(p.correlation for p in self._trading_pairs) / len(self._trading_pairs)
                if self._trading_pairs else 0
            ),
            "last_recalculation": (
                self._last_recalculation.isoformat()
                if self._last_recalculation else None
            ),
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"PairsTradingStrategy(pairs={len(self._trading_pairs)}, positions={len(self._active_positions)})"
