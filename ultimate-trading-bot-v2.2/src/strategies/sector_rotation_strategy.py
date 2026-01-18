"""
Sector Rotation Strategy Module for Ultimate Trading Bot v2.2.

This module implements sector rotation based on economic cycles,
relative strength, and momentum across market sectors.
"""

import logging
from datetime import datetime, timedelta
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


class SectorData(BaseModel):
    """Model for sector performance data."""

    sector_id: str
    name: str
    symbols: list[str] = Field(default_factory=list)
    etf_symbol: Optional[str] = None
    relative_strength: float = Field(default=0.0)
    momentum_score: float = Field(default=0.0)
    trend_score: float = Field(default=0.0)
    economic_cycle_fit: float = Field(default=0.0)
    overall_score: float = Field(default=0.0)


class EconomicCyclePhase(BaseModel):
    """Model for economic cycle phase."""

    phase: str
    confidence: float = Field(ge=0.0, le=1.0)
    leading_indicators: dict = Field(default_factory=dict)
    favored_sectors: list[str] = Field(default_factory=list)


class SectorAllocation(BaseModel):
    """Model for sector allocation."""

    sector_id: str
    target_weight: float = Field(ge=0.0, le=1.0)
    current_weight: float = Field(default=0.0)
    adjustment_needed: float = Field(default=0.0)


class SectorRotationConfig(StrategyConfig):
    """Configuration for sector rotation strategy."""

    name: str = Field(default="Sector Rotation Strategy")
    description: str = Field(
        default="Rotate between sectors based on economic cycles and momentum"
    )

    rebalance_frequency_days: int = Field(default=30, ge=7, le=90)
    min_rebalance_threshold: float = Field(default=0.05, ge=0.01, le=0.2)

    momentum_lookback_days: int = Field(default=60, ge=20, le=252)
    relative_strength_lookback: int = Field(default=20, ge=5, le=60)

    top_sectors_count: int = Field(default=3, ge=1, le=10)
    bottom_sectors_avoid: int = Field(default=2, ge=0, le=5)

    momentum_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    relative_strength_weight: float = Field(default=0.30, ge=0.0, le=1.0)
    trend_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    economic_cycle_weight: float = Field(default=0.15, ge=0.0, le=1.0)

    use_economic_cycle: bool = Field(default=True)
    use_volatility_scaling: bool = Field(default=True)

    max_sector_weight: float = Field(default=0.40, ge=0.1, le=0.6)
    min_sector_weight: float = Field(default=0.05, ge=0.0, le=0.2)


SECTOR_DEFINITIONS = {
    "XLK": {"name": "Technology", "cycle_phases": ["expansion", "peak"]},
    "XLF": {"name": "Financials", "cycle_phases": ["early_recovery", "expansion"]},
    "XLE": {"name": "Energy", "cycle_phases": ["peak", "late_expansion"]},
    "XLV": {"name": "Healthcare", "cycle_phases": ["recession", "early_recovery"]},
    "XLP": {"name": "Consumer Staples", "cycle_phases": ["recession", "contraction"]},
    "XLY": {"name": "Consumer Discretionary", "cycle_phases": ["early_recovery", "expansion"]},
    "XLI": {"name": "Industrials", "cycle_phases": ["early_recovery", "expansion"]},
    "XLB": {"name": "Materials", "cycle_phases": ["expansion", "peak"]},
    "XLU": {"name": "Utilities", "cycle_phases": ["recession", "contraction"]},
    "XLRE": {"name": "Real Estate", "cycle_phases": ["early_recovery"]},
    "XLC": {"name": "Communications", "cycle_phases": ["expansion", "peak"]},
}


class SectorRotationStrategy(BaseStrategy):
    """
    Sector rotation strategy based on economic cycles.

    Features:
    - Multi-factor sector scoring
    - Economic cycle detection
    - Relative strength ranking
    - Momentum-based rotation
    - Volatility-adjusted allocation
    """

    def __init__(
        self,
        config: Optional[SectorRotationConfig] = None,
        sector_definitions: Optional[dict] = None,
    ) -> None:
        """
        Initialize SectorRotationStrategy.

        Args:
            config: Sector rotation configuration
            sector_definitions: Custom sector definitions
        """
        config = config or SectorRotationConfig()
        super().__init__(config)

        self._rotation_config = config
        self._indicators = TechnicalIndicators()

        self._sector_definitions = sector_definitions or SECTOR_DEFINITIONS
        self._sectors: dict[str, SectorData] = {}
        self._current_allocations: dict[str, SectorAllocation] = {}
        self._economic_cycle: Optional[EconomicCyclePhase] = None
        self._last_rebalance: Optional[datetime] = None

        self._initialize_sectors()

        logger.info(f"SectorRotationStrategy initialized: {self.name}")

    def _initialize_sectors(self) -> None:
        """Initialize sector data from definitions."""
        for symbol, info in self._sector_definitions.items():
            self._sectors[symbol] = SectorData(
                sector_id=symbol,
                name=info.get("name", symbol),
                etf_symbol=symbol,
                symbols=info.get("components", []),
            )

    def calculate_indicators(
        self,
        symbol: str,
        data: MarketData,
    ) -> dict[str, Any]:
        """
        Calculate sector indicators.

        Args:
            symbol: Trading symbol (sector ETF)
            data: Market data

        Returns:
            Dictionary of indicator values
        """
        closes = data.closes
        highs = data.highs
        lows = data.lows
        volumes = data.volumes

        if len(closes) < self._rotation_config.momentum_lookback_days:
            return {}

        current_price = closes[-1]

        momentum_period = self._rotation_config.momentum_lookback_days
        if len(closes) >= momentum_period:
            momentum = (closes[-1] - closes[-momentum_period]) / closes[-momentum_period]
        else:
            momentum = 0.0

        ema_20 = self._indicators.ema(closes, 20)
        ema_50 = self._indicators.ema(closes, 50)

        trend_score = 0.0
        if ema_20 and ema_50:
            if current_price > ema_20[-1] > ema_50[-1]:
                trend_score = 1.0
            elif current_price < ema_20[-1] < ema_50[-1]:
                trend_score = -1.0
            elif current_price > ema_50[-1]:
                trend_score = 0.5
            else:
                trend_score = -0.5

        atr = self._indicators.atr(highs, lows, closes, 14)
        volatility = atr[-1] / current_price if atr and current_price > 0 else 0.01

        rsi = self._indicators.rsi(closes, 14)
        current_rsi = rsi[-1] if rsi else 50.0

        rs_lookback = self._rotation_config.relative_strength_lookback
        if len(closes) >= rs_lookback:
            rs_momentum = (closes[-1] - closes[-rs_lookback]) / closes[-rs_lookback]
        else:
            rs_momentum = 0.0

        return {
            "current_price": current_price,
            "momentum": momentum,
            "trend_score": trend_score,
            "volatility": volatility,
            "rsi": current_rsi,
            "relative_strength_momentum": rs_momentum,
            "ema_20": ema_20[-1] if ema_20 else current_price,
            "ema_50": ema_50[-1] if ema_50 else current_price,
        }

    async def evaluate(
        self,
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """
        Evaluate sector rotation opportunities.

        Args:
            market_data: Market data for all symbols
            context: Strategy execution context

        Returns:
            List of sector rotation signals
        """
        signals: list[StrategySignal] = []

        self._update_sector_scores(market_data, context)

        if self._rotation_config.use_economic_cycle:
            self._detect_economic_cycle(market_data, context)

        if self._should_rebalance(context):
            target_allocations = self._calculate_target_allocations(context)
            signals = self._generate_rebalance_signals(
                target_allocations, market_data, context
            )

            if signals:
                self._last_rebalance = context.timestamp

        return signals

    def _update_sector_scores(
        self,
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> None:
        """Update sector scores based on market data."""
        for sector_id, sector in self._sectors.items():
            etf_symbol = sector.etf_symbol or sector_id

            if etf_symbol not in market_data:
                continue

            indicators = self.calculate_indicators(etf_symbol, market_data[etf_symbol])
            if not indicators:
                continue

            sector.momentum_score = indicators["momentum"]
            sector.trend_score = indicators["trend_score"]
            sector.relative_strength = indicators["relative_strength_momentum"]

            if self._rotation_config.use_economic_cycle and self._economic_cycle:
                sector_info = self._sector_definitions.get(sector_id, {})
                favored_phases = sector_info.get("cycle_phases", [])

                if self._economic_cycle.phase in favored_phases:
                    sector.economic_cycle_fit = 1.0
                elif any(p in self._economic_cycle.favored_sectors for p in [sector_id]):
                    sector.economic_cycle_fit = 0.7
                else:
                    sector.economic_cycle_fit = 0.3
            else:
                sector.economic_cycle_fit = 0.5

            sector.overall_score = (
                sector.momentum_score * self._rotation_config.momentum_weight +
                sector.relative_strength * self._rotation_config.relative_strength_weight +
                sector.trend_score * self._rotation_config.trend_weight +
                sector.economic_cycle_fit * self._rotation_config.economic_cycle_weight
            )

    def _detect_economic_cycle(
        self,
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> None:
        """Detect current economic cycle phase."""
        defensive_sectors = ["XLU", "XLP", "XLV"]
        cyclical_sectors = ["XLY", "XLI", "XLB", "XLF"]

        defensive_strength = 0.0
        cyclical_strength = 0.0
        defensive_count = 0
        cyclical_count = 0

        for sector_id, sector in self._sectors.items():
            if sector_id in defensive_sectors:
                defensive_strength += sector.momentum_score
                defensive_count += 1
            elif sector_id in cyclical_sectors:
                cyclical_strength += sector.momentum_score
                cyclical_count += 1

        avg_defensive = defensive_strength / defensive_count if defensive_count > 0 else 0
        avg_cyclical = cyclical_strength / cyclical_count if cyclical_count > 0 else 0

        if avg_cyclical > 0.1 and avg_cyclical > avg_defensive:
            if avg_cyclical > 0.2:
                phase = "expansion"
            else:
                phase = "early_recovery"
            favored = cyclical_sectors
        elif avg_defensive > 0.05 and avg_defensive > avg_cyclical:
            if avg_cyclical < -0.1:
                phase = "recession"
            else:
                phase = "contraction"
            favored = defensive_sectors
        elif avg_cyclical > 0.15:
            phase = "peak"
            favored = ["XLE", "XLB", "XLK"]
        else:
            phase = "late_cycle"
            favored = ["XLV", "XLU", "XLP"]

        confidence = min(0.9, abs(avg_cyclical - avg_defensive) * 5 + 0.4)

        self._economic_cycle = EconomicCyclePhase(
            phase=phase,
            confidence=confidence,
            leading_indicators={
                "defensive_momentum": avg_defensive,
                "cyclical_momentum": avg_cyclical,
            },
            favored_sectors=favored,
        )

    def _should_rebalance(self, context: StrategyContext) -> bool:
        """Check if portfolio should be rebalanced."""
        if self._last_rebalance is None:
            return True

        days_since = (context.timestamp - self._last_rebalance).days
        if days_since >= self._rotation_config.rebalance_frequency_days:
            return True

        for sector_id, allocation in self._current_allocations.items():
            if abs(allocation.adjustment_needed) >= self._rotation_config.min_rebalance_threshold:
                return True

        return False

    def _calculate_target_allocations(
        self,
        context: StrategyContext,
    ) -> dict[str, SectorAllocation]:
        """Calculate target sector allocations."""
        sorted_sectors = sorted(
            self._sectors.values(),
            key=lambda s: s.overall_score,
            reverse=True,
        )

        top_sectors = sorted_sectors[:self._rotation_config.top_sectors_count]
        avoid_sectors = sorted_sectors[-self._rotation_config.bottom_sectors_avoid:] if self._rotation_config.bottom_sectors_avoid > 0 else []
        avoid_ids = {s.sector_id for s in avoid_sectors}

        total_score = sum(
            max(0, s.overall_score) for s in top_sectors
            if s.sector_id not in avoid_ids
        )

        allocations: dict[str, SectorAllocation] = {}

        for sector in self._sectors.values():
            if sector.sector_id in avoid_ids:
                target_weight = 0.0
            elif sector in top_sectors:
                if total_score > 0:
                    raw_weight = max(0, sector.overall_score) / total_score
                else:
                    raw_weight = 1.0 / len(top_sectors)

                target_weight = min(
                    self._rotation_config.max_sector_weight,
                    max(self._rotation_config.min_sector_weight, raw_weight)
                )
            else:
                target_weight = 0.0

            current_weight = self._current_allocations.get(
                sector.sector_id, SectorAllocation(sector_id=sector.sector_id)
            ).current_weight

            allocations[sector.sector_id] = SectorAllocation(
                sector_id=sector.sector_id,
                target_weight=target_weight,
                current_weight=current_weight,
                adjustment_needed=target_weight - current_weight,
            )

        total_weight = sum(a.target_weight for a in allocations.values())
        if total_weight > 0 and total_weight != 1.0:
            for allocation in allocations.values():
                allocation.target_weight /= total_weight
                allocation.adjustment_needed = allocation.target_weight - allocation.current_weight

        return allocations

    def _generate_rebalance_signals(
        self,
        target_allocations: dict[str, SectorAllocation],
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """Generate signals for portfolio rebalancing."""
        signals: list[StrategySignal] = []

        sorted_allocations = sorted(
            target_allocations.values(),
            key=lambda a: a.adjustment_needed,
        )

        for allocation in sorted_allocations:
            if abs(allocation.adjustment_needed) < self._rotation_config.min_rebalance_threshold:
                continue

            sector = self._sectors.get(allocation.sector_id)
            if not sector:
                continue

            symbol = sector.etf_symbol or allocation.sector_id
            if symbol not in market_data:
                continue

            current_price = market_data[symbol].closes[-1]

            if allocation.adjustment_needed < 0:
                signal = self.create_signal(
                    symbol=symbol,
                    action=SignalAction.SELL,
                    side=SignalSide.FLAT,
                    entry_price=current_price,
                    strength=abs(allocation.adjustment_needed),
                    confidence=0.8,
                    reason=f"Sector rotation: reduce {sector.name}",
                    position_size_pct=abs(allocation.adjustment_needed),
                    metadata={
                        "strategy_type": "sector_rotation",
                        "sector": sector.name,
                        "sector_score": sector.overall_score,
                        "target_weight": allocation.target_weight,
                        "current_weight": allocation.current_weight,
                        "economic_cycle": self._economic_cycle.phase if self._economic_cycle else "unknown",
                    },
                )

                if signal:
                    signals.append(signal)

            else:
                signal = self.create_signal(
                    symbol=symbol,
                    action=SignalAction.BUY,
                    side=SignalSide.LONG,
                    entry_price=current_price,
                    strength=allocation.adjustment_needed,
                    confidence=0.8,
                    reason=f"Sector rotation: increase {sector.name}",
                    position_size_pct=allocation.adjustment_needed,
                    metadata={
                        "strategy_type": "sector_rotation",
                        "sector": sector.name,
                        "sector_score": sector.overall_score,
                        "target_weight": allocation.target_weight,
                        "current_weight": allocation.current_weight,
                        "economic_cycle": self._economic_cycle.phase if self._economic_cycle else "unknown",
                    },
                )

                if signal:
                    signals.append(signal)

        self._current_allocations = target_allocations

        return signals

    def update_current_allocation(
        self,
        sector_id: str,
        weight: float,
    ) -> None:
        """Update current allocation for a sector."""
        if sector_id not in self._current_allocations:
            self._current_allocations[sector_id] = SectorAllocation(sector_id=sector_id)

        self._current_allocations[sector_id].current_weight = weight

    def get_sector_rankings(self) -> list[dict]:
        """Get current sector rankings."""
        sorted_sectors = sorted(
            self._sectors.values(),
            key=lambda s: s.overall_score,
            reverse=True,
        )

        return [
            {
                "rank": i + 1,
                "sector_id": s.sector_id,
                "name": s.name,
                "overall_score": s.overall_score,
                "momentum_score": s.momentum_score,
                "trend_score": s.trend_score,
                "relative_strength": s.relative_strength,
                "economic_cycle_fit": s.economic_cycle_fit,
            }
            for i, s in enumerate(sorted_sectors)
        ]

    def get_economic_cycle(self) -> Optional[EconomicCyclePhase]:
        """Get current economic cycle phase."""
        return self._economic_cycle

    def get_target_allocations(self) -> dict[str, SectorAllocation]:
        """Get current target allocations."""
        return self._current_allocations.copy()

    def get_rotation_statistics(self) -> dict:
        """Get sector rotation statistics."""
        return {
            "sectors_count": len(self._sectors),
            "economic_cycle": self._economic_cycle.phase if self._economic_cycle else "unknown",
            "cycle_confidence": self._economic_cycle.confidence if self._economic_cycle else 0,
            "last_rebalance": self._last_rebalance.isoformat() if self._last_rebalance else None,
            "top_sectors": [
                s.name for s in sorted(
                    self._sectors.values(),
                    key=lambda x: x.overall_score,
                    reverse=True,
                )[:3]
            ],
            "current_allocations": {
                k: v.current_weight for k, v in self._current_allocations.items()
            },
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"SectorRotationStrategy(sectors={len(self._sectors)}, cycle={self._economic_cycle.phase if self._economic_cycle else 'unknown'})"
