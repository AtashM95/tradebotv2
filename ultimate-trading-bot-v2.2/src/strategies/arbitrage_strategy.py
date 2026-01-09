"""
Arbitrage Strategy Module for Ultimate Trading Bot v2.2.

This module implements various arbitrage strategies including
statistical arbitrage, triangular arbitrage, and cross-exchange
opportunities.
"""

import logging
from datetime import datetime
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
from src.utils.helpers import generate_uuid


logger = logging.getLogger(__name__)


class ArbitrageOpportunity(BaseModel):
    """Model for arbitrage opportunity."""

    opportunity_id: str = Field(default_factory=generate_uuid)
    arb_type: str
    symbols: list[str]
    spread_pct: float
    expected_profit_pct: float
    confidence: float = Field(ge=0.0, le=1.0)
    execution_risk: float = Field(ge=0.0, le=1.0)
    timestamp: datetime
    details: dict = Field(default_factory=dict)


class ArbitragePosition(BaseModel):
    """Model for arbitrage position."""

    position_id: str = Field(default_factory=generate_uuid)
    opportunity: ArbitrageOpportunity
    entry_time: datetime
    legs: list[dict] = Field(default_factory=list)
    status: str = Field(default="open")
    realized_pnl: float = Field(default=0.0)


class ArbitrageStrategyConfig(StrategyConfig):
    """Configuration for arbitrage strategy."""

    name: str = Field(default="Arbitrage Strategy")
    description: str = Field(
        default="Multi-type arbitrage including statistical and triangular"
    )

    arb_types: list[str] = Field(
        default_factory=lambda: ["statistical", "etf_nav", "index"]
    )

    min_spread_pct: float = Field(default=0.002, ge=0.0005, le=0.05)
    min_profit_after_costs: float = Field(default=0.001, ge=0.0001, le=0.01)

    execution_cost_pct: float = Field(default=0.001, ge=0.0001, le=0.01)
    slippage_estimate_pct: float = Field(default=0.0005, ge=0.0001, le=0.005)

    max_execution_time_seconds: float = Field(default=5.0, ge=0.5, le=30.0)
    max_position_hold_minutes: int = Field(default=60, ge=1, le=1440)

    position_size_pct: float = Field(default=0.05, ge=0.01, le=0.2)
    max_concurrent_positions: int = Field(default=5, ge=1, le=20)

    stat_arb_lookback: int = Field(default=60, ge=20, le=252)
    stat_arb_zscore_entry: float = Field(default=2.0, ge=1.5, le=4.0)
    stat_arb_zscore_exit: float = Field(default=0.5, ge=0.0, le=1.5)

    etf_nav_threshold_pct: float = Field(default=0.005, ge=0.001, le=0.02)
    index_tracking_error_threshold: float = Field(default=0.003, ge=0.001, le=0.01)


class ArbitrageStrategy(BaseStrategy):
    """
    Multi-type arbitrage strategy.

    Features:
    - Statistical arbitrage on correlated pairs
    - ETF vs NAV arbitrage
    - Index vs components arbitrage
    - Spread monitoring and execution
    - Risk-adjusted position sizing
    """

    def __init__(
        self,
        config: Optional[ArbitrageStrategyConfig] = None,
    ) -> None:
        """
        Initialize ArbitrageStrategy.

        Args:
            config: Arbitrage configuration
        """
        config = config or ArbitrageStrategyConfig()
        super().__init__(config)

        self._arb_config = config

        self._opportunities: list[ArbitrageOpportunity] = []
        self._active_positions: dict[str, ArbitragePosition] = {}
        self._historical_spreads: dict[str, list[float]] = {}

        self._etf_nav_mappings: dict[str, dict] = {}
        self._index_component_mappings: dict[str, list[str]] = {}

        logger.info(f"ArbitrageStrategy initialized: {self.name}")

    def calculate_indicators(
        self,
        symbol: str,
        data: MarketData,
    ) -> dict[str, Any]:
        """
        Calculate indicators for arbitrage analysis.

        Args:
            symbol: Trading symbol
            data: Market data

        Returns:
            Dictionary of indicator values
        """
        closes = data.closes
        volumes = data.volumes

        if len(closes) < 20:
            return {}

        current_price = closes[-1]

        returns = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]
        volatility = (sum(r ** 2 for r in returns[-20:]) / 20) ** 0.5 if len(returns) >= 20 else 0.01

        avg_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else volumes[-1]
        liquidity_score = min(1.0, avg_volume / 1000000)

        return {
            "current_price": current_price,
            "prices": closes,
            "volatility": volatility,
            "liquidity_score": liquidity_score,
            "avg_volume": avg_volume,
        }

    async def evaluate(
        self,
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """
        Evaluate arbitrage opportunities.

        Args:
            market_data: Market data for all symbols
            context: Strategy execution context

        Returns:
            List of arbitrage signals
        """
        signals: list[StrategySignal] = []

        exit_signals = self._check_position_exits(market_data, context)
        signals.extend(exit_signals)

        if len(self._active_positions) >= self._arb_config.max_concurrent_positions:
            return signals

        opportunities: list[ArbitrageOpportunity] = []

        if "statistical" in self._arb_config.arb_types:
            stat_opps = self._find_statistical_arbitrage(market_data, context)
            opportunities.extend(stat_opps)

        if "etf_nav" in self._arb_config.arb_types:
            etf_opps = self._find_etf_nav_arbitrage(market_data, context)
            opportunities.extend(etf_opps)

        if "index" in self._arb_config.arb_types:
            index_opps = self._find_index_arbitrage(market_data, context)
            opportunities.extend(index_opps)

        opportunities.sort(key=lambda x: x.expected_profit_pct, reverse=True)

        for opp in opportunities[:3]:
            if opp.expected_profit_pct >= self._arb_config.min_profit_after_costs:
                arb_signals = self._create_arbitrage_signals(opp, market_data, context)
                signals.extend(arb_signals)

        self._opportunities = opportunities[:20]

        return signals

    def _find_statistical_arbitrage(
        self,
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> list[ArbitrageOpportunity]:
        """Find statistical arbitrage opportunities."""
        opportunities: list[ArbitrageOpportunity] = []
        symbols = list(market_data.keys())

        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i + 1:]:
                if sym1 not in market_data or sym2 not in market_data:
                    continue

                data1 = market_data[sym1]
                data2 = market_data[sym2]

                if len(data1.closes) < self._arb_config.stat_arb_lookback:
                    continue
                if len(data2.closes) < self._arb_config.stat_arb_lookback:
                    continue

                spread_key = f"{sym1}_{sym2}"
                spread = self._calculate_price_ratio_spread(
                    data1.closes, data2.closes
                )

                if spread_key not in self._historical_spreads:
                    self._historical_spreads[spread_key] = []

                self._historical_spreads[spread_key].append(spread)

                if len(self._historical_spreads[spread_key]) > 1000:
                    self._historical_spreads[spread_key] = self._historical_spreads[spread_key][-1000:]

                hist = self._historical_spreads[spread_key]
                if len(hist) < self._arb_config.stat_arb_lookback:
                    continue

                mean_spread = sum(hist[-self._arb_config.stat_arb_lookback:]) / self._arb_config.stat_arb_lookback
                variance = sum((s - mean_spread) ** 2 for s in hist[-self._arb_config.stat_arb_lookback:]) / self._arb_config.stat_arb_lookback
                std_spread = variance ** 0.5

                if std_spread == 0:
                    continue

                zscore = (spread - mean_spread) / std_spread

                if abs(zscore) >= self._arb_config.stat_arb_zscore_entry:
                    expected_profit = abs(spread - mean_spread) / ((data1.closes[-1] + data2.closes[-1]) / 2)
                    expected_profit -= self._arb_config.execution_cost_pct * 2
                    expected_profit -= self._arb_config.slippage_estimate_pct * 2

                    if expected_profit > 0:
                        opportunities.append(ArbitrageOpportunity(
                            arb_type="statistical",
                            symbols=[sym1, sym2],
                            spread_pct=abs(zscore) * std_spread / mean_spread if mean_spread != 0 else 0,
                            expected_profit_pct=expected_profit,
                            confidence=min(0.9, 0.5 + abs(zscore) * 0.1),
                            execution_risk=0.3,
                            timestamp=context.timestamp,
                            details={
                                "zscore": zscore,
                                "mean_spread": mean_spread,
                                "std_spread": std_spread,
                                "current_spread": spread,
                                "direction": "short_spread" if zscore > 0 else "long_spread",
                            },
                        ))

        return opportunities

    def _find_etf_nav_arbitrage(
        self,
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> list[ArbitrageOpportunity]:
        """Find ETF vs NAV arbitrage opportunities."""
        opportunities: list[ArbitrageOpportunity] = []

        for etf_symbol, nav_info in self._etf_nav_mappings.items():
            if etf_symbol not in market_data:
                continue

            etf_price = market_data[etf_symbol].closes[-1]
            nav_estimate = nav_info.get("nav", etf_price)

            premium_discount = (etf_price - nav_estimate) / nav_estimate if nav_estimate > 0 else 0

            if abs(premium_discount) >= self._arb_config.etf_nav_threshold_pct:
                expected_profit = abs(premium_discount)
                expected_profit -= self._arb_config.execution_cost_pct
                expected_profit -= self._arb_config.slippage_estimate_pct

                if expected_profit > 0:
                    opportunities.append(ArbitrageOpportunity(
                        arb_type="etf_nav",
                        symbols=[etf_symbol],
                        spread_pct=abs(premium_discount),
                        expected_profit_pct=expected_profit,
                        confidence=0.75,
                        execution_risk=0.4,
                        timestamp=context.timestamp,
                        details={
                            "etf_price": etf_price,
                            "nav_estimate": nav_estimate,
                            "premium_discount": premium_discount,
                            "direction": "short_etf" if premium_discount > 0 else "long_etf",
                        },
                    ))

        return opportunities

    def _find_index_arbitrage(
        self,
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> list[ArbitrageOpportunity]:
        """Find index vs components arbitrage opportunities."""
        opportunities: list[ArbitrageOpportunity] = []

        for index_symbol, components in self._index_component_mappings.items():
            if index_symbol not in market_data:
                continue

            index_price = market_data[index_symbol].closes[-1]

            component_value = 0.0
            components_found = 0
            for comp in components:
                if comp in market_data:
                    component_value += market_data[comp].closes[-1]
                    components_found += 1

            if components_found < len(components) * 0.9:
                continue

            tracking_error = (index_price - component_value) / component_value if component_value > 0 else 0

            if abs(tracking_error) >= self._arb_config.index_tracking_error_threshold:
                expected_profit = abs(tracking_error)
                expected_profit -= self._arb_config.execution_cost_pct * (1 + len(components) * 0.1)
                expected_profit -= self._arb_config.slippage_estimate_pct * (1 + len(components) * 0.1)

                if expected_profit > 0:
                    opportunities.append(ArbitrageOpportunity(
                        arb_type="index",
                        symbols=[index_symbol] + components[:5],
                        spread_pct=abs(tracking_error),
                        expected_profit_pct=expected_profit,
                        confidence=0.7,
                        execution_risk=0.5,
                        timestamp=context.timestamp,
                        details={
                            "index_price": index_price,
                            "component_value": component_value,
                            "tracking_error": tracking_error,
                            "direction": "short_index" if tracking_error > 0 else "long_index",
                            "components_count": len(components),
                        },
                    ))

        return opportunities

    def _calculate_price_ratio_spread(
        self,
        prices1: list[float],
        prices2: list[float],
    ) -> float:
        """Calculate spread as price ratio."""
        if not prices1 or not prices2:
            return 0.0

        p1 = prices1[-1]
        p2 = prices2[-1]

        return p1 / p2 if p2 > 0 else 0.0

    def _create_arbitrage_signals(
        self,
        opportunity: ArbitrageOpportunity,
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """Create trading signals for arbitrage opportunity."""
        signals: list[StrategySignal] = []

        if opportunity.arb_type == "statistical":
            signals = self._create_stat_arb_signals(opportunity, market_data, context)

        elif opportunity.arb_type == "etf_nav":
            signals = self._create_etf_arb_signals(opportunity, market_data, context)

        elif opportunity.arb_type == "index":
            signals = self._create_index_arb_signals(opportunity, market_data, context)

        if signals:
            position = ArbitragePosition(
                opportunity=opportunity,
                entry_time=context.timestamp,
                legs=[{"symbol": s.symbol, "side": s.side.value} for s in signals],
            )
            self._active_positions[opportunity.opportunity_id] = position

        return signals

    def _create_stat_arb_signals(
        self,
        opportunity: ArbitrageOpportunity,
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """Create statistical arbitrage signals."""
        signals: list[StrategySignal] = []

        if len(opportunity.symbols) < 2:
            return signals

        sym1, sym2 = opportunity.symbols[0], opportunity.symbols[1]
        direction = opportunity.details.get("direction", "")

        price1 = market_data[sym1].closes[-1]
        price2 = market_data[sym2].closes[-1]

        if direction == "short_spread":
            signal1 = self.create_signal(
                symbol=sym1,
                action=SignalAction.SELL,
                side=SignalSide.SHORT,
                entry_price=price1,
                strength=opportunity.confidence,
                confidence=opportunity.confidence,
                reason=f"Stat arb: short {sym1} vs long {sym2}",
                position_size_pct=self._arb_config.position_size_pct / 2,
                metadata={
                    "strategy_type": "arbitrage",
                    "arb_type": "statistical",
                    "opportunity_id": opportunity.opportunity_id,
                    "leg": 1,
                    "zscore": opportunity.details.get("zscore"),
                },
            )

            signal2 = self.create_signal(
                symbol=sym2,
                action=SignalAction.BUY,
                side=SignalSide.LONG,
                entry_price=price2,
                strength=opportunity.confidence,
                confidence=opportunity.confidence,
                reason=f"Stat arb: short {sym1} vs long {sym2}",
                position_size_pct=self._arb_config.position_size_pct / 2,
                metadata={
                    "strategy_type": "arbitrage",
                    "arb_type": "statistical",
                    "opportunity_id": opportunity.opportunity_id,
                    "leg": 2,
                    "zscore": opportunity.details.get("zscore"),
                },
            )

        else:
            signal1 = self.create_signal(
                symbol=sym1,
                action=SignalAction.BUY,
                side=SignalSide.LONG,
                entry_price=price1,
                strength=opportunity.confidence,
                confidence=opportunity.confidence,
                reason=f"Stat arb: long {sym1} vs short {sym2}",
                position_size_pct=self._arb_config.position_size_pct / 2,
                metadata={
                    "strategy_type": "arbitrage",
                    "arb_type": "statistical",
                    "opportunity_id": opportunity.opportunity_id,
                    "leg": 1,
                    "zscore": opportunity.details.get("zscore"),
                },
            )

            signal2 = self.create_signal(
                symbol=sym2,
                action=SignalAction.SELL,
                side=SignalSide.SHORT,
                entry_price=price2,
                strength=opportunity.confidence,
                confidence=opportunity.confidence,
                reason=f"Stat arb: long {sym1} vs short {sym2}",
                position_size_pct=self._arb_config.position_size_pct / 2,
                metadata={
                    "strategy_type": "arbitrage",
                    "arb_type": "statistical",
                    "opportunity_id": opportunity.opportunity_id,
                    "leg": 2,
                    "zscore": opportunity.details.get("zscore"),
                },
            )

        if signal1:
            signals.append(signal1)
        if signal2:
            signals.append(signal2)

        return signals

    def _create_etf_arb_signals(
        self,
        opportunity: ArbitrageOpportunity,
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """Create ETF arbitrage signals."""
        signals: list[StrategySignal] = []

        if not opportunity.symbols:
            return signals

        etf_symbol = opportunity.symbols[0]
        direction = opportunity.details.get("direction", "")
        price = market_data[etf_symbol].closes[-1]

        if direction == "short_etf":
            signal = self.create_signal(
                symbol=etf_symbol,
                action=SignalAction.SELL,
                side=SignalSide.SHORT,
                entry_price=price,
                strength=opportunity.confidence,
                confidence=opportunity.confidence,
                reason=f"ETF NAV arb: short at premium",
                position_size_pct=self._arb_config.position_size_pct,
                metadata={
                    "strategy_type": "arbitrage",
                    "arb_type": "etf_nav",
                    "opportunity_id": opportunity.opportunity_id,
                    "premium_discount": opportunity.details.get("premium_discount"),
                },
            )
        else:
            signal = self.create_signal(
                symbol=etf_symbol,
                action=SignalAction.BUY,
                side=SignalSide.LONG,
                entry_price=price,
                strength=opportunity.confidence,
                confidence=opportunity.confidence,
                reason=f"ETF NAV arb: long at discount",
                position_size_pct=self._arb_config.position_size_pct,
                metadata={
                    "strategy_type": "arbitrage",
                    "arb_type": "etf_nav",
                    "opportunity_id": opportunity.opportunity_id,
                    "premium_discount": opportunity.details.get("premium_discount"),
                },
            )

        if signal:
            signals.append(signal)

        return signals

    def _create_index_arb_signals(
        self,
        opportunity: ArbitrageOpportunity,
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """Create index arbitrage signals."""
        signals: list[StrategySignal] = []

        if not opportunity.symbols:
            return signals

        index_symbol = opportunity.symbols[0]
        direction = opportunity.details.get("direction", "")
        price = market_data[index_symbol].closes[-1]

        if direction == "short_index":
            signal = self.create_signal(
                symbol=index_symbol,
                action=SignalAction.SELL,
                side=SignalSide.SHORT,
                entry_price=price,
                strength=opportunity.confidence,
                confidence=opportunity.confidence,
                reason=f"Index arb: short overvalued index",
                position_size_pct=self._arb_config.position_size_pct,
                metadata={
                    "strategy_type": "arbitrage",
                    "arb_type": "index",
                    "opportunity_id": opportunity.opportunity_id,
                    "tracking_error": opportunity.details.get("tracking_error"),
                },
            )
        else:
            signal = self.create_signal(
                symbol=index_symbol,
                action=SignalAction.BUY,
                side=SignalSide.LONG,
                entry_price=price,
                strength=opportunity.confidence,
                confidence=opportunity.confidence,
                reason=f"Index arb: long undervalued index",
                position_size_pct=self._arb_config.position_size_pct,
                metadata={
                    "strategy_type": "arbitrage",
                    "arb_type": "index",
                    "opportunity_id": opportunity.opportunity_id,
                    "tracking_error": opportunity.details.get("tracking_error"),
                },
            )

        if signal:
            signals.append(signal)

        return signals

    def _check_position_exits(
        self,
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """Check for arbitrage position exits."""
        signals: list[StrategySignal] = []

        for position_id, position in list(self._active_positions.items()):
            hold_minutes = (context.timestamp - position.entry_time).total_seconds() / 60

            if hold_minutes >= self._arb_config.max_position_hold_minutes:
                exit_signals = self._close_arbitrage_position(position, market_data, context)
                signals.extend(exit_signals)
                del self._active_positions[position_id]
                continue

            if position.opportunity.arb_type == "statistical":
                should_exit = self._check_stat_arb_exit(position, market_data)
                if should_exit:
                    exit_signals = self._close_arbitrage_position(position, market_data, context)
                    signals.extend(exit_signals)
                    del self._active_positions[position_id]

        return signals

    def _check_stat_arb_exit(
        self,
        position: ArbitragePosition,
        market_data: dict[str, MarketData],
    ) -> bool:
        """Check if statistical arbitrage position should exit."""
        opp = position.opportunity

        if len(opp.symbols) < 2:
            return True

        sym1, sym2 = opp.symbols[0], opp.symbols[1]
        spread_key = f"{sym1}_{sym2}"

        if spread_key not in self._historical_spreads:
            return False

        hist = self._historical_spreads[spread_key]
        if len(hist) < self._arb_config.stat_arb_lookback:
            return False

        mean_spread = sum(hist[-self._arb_config.stat_arb_lookback:]) / self._arb_config.stat_arb_lookback
        variance = sum((s - mean_spread) ** 2 for s in hist[-self._arb_config.stat_arb_lookback:]) / self._arb_config.stat_arb_lookback
        std_spread = variance ** 0.5

        if std_spread == 0:
            return True

        current_spread = hist[-1]
        zscore = (current_spread - mean_spread) / std_spread

        return abs(zscore) <= self._arb_config.stat_arb_zscore_exit

    def _close_arbitrage_position(
        self,
        position: ArbitragePosition,
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """Close arbitrage position."""
        signals: list[StrategySignal] = []

        for leg in position.legs:
            symbol = leg.get("symbol", "")
            side = leg.get("side", "")

            if symbol not in market_data:
                continue

            price = market_data[symbol].closes[-1]

            if side == "long":
                signal = self.create_signal(
                    symbol=symbol,
                    action=SignalAction.SELL,
                    side=SignalSide.FLAT,
                    entry_price=price,
                    strength=1.0,
                    confidence=0.9,
                    reason=f"Close arb position",
                    metadata={
                        "strategy_type": "arbitrage",
                        "exit": True,
                        "opportunity_id": position.opportunity.opportunity_id,
                    },
                )
            else:
                signal = self.create_signal(
                    symbol=symbol,
                    action=SignalAction.BUY,
                    side=SignalSide.FLAT,
                    entry_price=price,
                    strength=1.0,
                    confidence=0.9,
                    reason=f"Close arb position",
                    metadata={
                        "strategy_type": "arbitrage",
                        "exit": True,
                        "opportunity_id": position.opportunity.opportunity_id,
                    },
                )

            if signal:
                signals.append(signal)

        return signals

    def add_etf_nav_mapping(
        self,
        etf_symbol: str,
        nav: float,
        components: Optional[list[str]] = None,
    ) -> None:
        """Add ETF to NAV mapping."""
        self._etf_nav_mappings[etf_symbol] = {
            "nav": nav,
            "components": components or [],
        }

    def add_index_components(
        self,
        index_symbol: str,
        components: list[str],
    ) -> None:
        """Add index to components mapping."""
        self._index_component_mappings[index_symbol] = components

    def get_opportunities(self) -> list[ArbitrageOpportunity]:
        """Get recent arbitrage opportunities."""
        return self._opportunities.copy()

    def get_active_positions(self) -> dict[str, ArbitragePosition]:
        """Get active arbitrage positions."""
        return self._active_positions.copy()

    def get_arbitrage_statistics(self) -> dict:
        """Get arbitrage strategy statistics."""
        return {
            "active_positions": len(self._active_positions),
            "recent_opportunities": len(self._opportunities),
            "tracked_spreads": len(self._historical_spreads),
            "etf_mappings": len(self._etf_nav_mappings),
            "index_mappings": len(self._index_component_mappings),
            "opportunity_types": {
                "statistical": sum(1 for o in self._opportunities if o.arb_type == "statistical"),
                "etf_nav": sum(1 for o in self._opportunities if o.arb_type == "etf_nav"),
                "index": sum(1 for o in self._opportunities if o.arb_type == "index"),
            },
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"ArbitrageStrategy(positions={len(self._active_positions)}, opportunities={len(self._opportunities)})"
