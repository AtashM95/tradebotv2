"""
Dollar Cost Averaging (DCA) Strategy Module for Ultimate Trading Bot v2.2.

This module implements a systematic DCA strategy with optional
value averaging and smart timing adjustments.
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
from src.utils.date_utils import now_utc


logger = logging.getLogger(__name__)


class DCASchedule(BaseModel):
    """Model for DCA schedule entry."""

    schedule_id: str = Field(default_factory=generate_uuid)
    symbol: str
    frequency: str
    base_amount: float
    next_execution: datetime
    last_execution: Optional[datetime] = None
    total_invested: float = Field(default=0.0)
    total_shares: float = Field(default=0.0)
    avg_price: float = Field(default=0.0)
    execution_count: int = Field(default=0)


class DCAExecution(BaseModel):
    """Model for DCA execution record."""

    execution_id: str = Field(default_factory=generate_uuid)
    schedule_id: str
    symbol: str
    timestamp: datetime
    price: float
    amount: float
    shares: float
    adjustment_reason: Optional[str] = None


class DCAStrategyConfig(StrategyConfig):
    """Configuration for DCA strategy."""

    name: str = Field(default="DCA Strategy")
    description: str = Field(
        default="Systematic dollar cost averaging with smart timing"
    )

    frequency: str = Field(default="weekly")
    base_investment_amount: float = Field(default=100.0, ge=10.0, le=100000.0)

    use_value_averaging: bool = Field(default=False)
    target_growth_rate: float = Field(default=0.01, ge=0.0, le=0.1)

    use_smart_timing: bool = Field(default=True)
    timing_lookback: int = Field(default=20, ge=5, le=50)

    dip_threshold_pct: float = Field(default=0.05, ge=0.01, le=0.2)
    dip_bonus_pct: float = Field(default=0.5, ge=0.1, le=2.0)

    spike_threshold_pct: float = Field(default=0.05, ge=0.01, le=0.2)
    spike_reduction_pct: float = Field(default=0.3, ge=0.1, le=0.9)

    rsi_period: int = Field(default=14, ge=7, le=21)
    rsi_oversold: float = Field(default=30.0, ge=15.0, le=40.0)
    rsi_overbought: float = Field(default=70.0, ge=60.0, le=85.0)

    max_single_investment_multiplier: float = Field(default=3.0, ge=1.0, le=5.0)
    min_single_investment_multiplier: float = Field(default=0.25, ge=0.1, le=0.5)

    flexible_timing_window_days: int = Field(default=2, ge=0, le=7)


class DCAStrategy(BaseStrategy):
    """
    Dollar Cost Averaging strategy with enhancements.

    Features:
    - Regular scheduled investments
    - Value averaging option
    - Smart timing based on technicals
    - Dip buying and spike avoidance
    - Flexible execution windows
    """

    def __init__(
        self,
        config: Optional[DCAStrategyConfig] = None,
    ) -> None:
        """
        Initialize DCAStrategy.

        Args:
            config: DCA configuration
        """
        config = config or DCAStrategyConfig()
        super().__init__(config)

        self._dca_config = config
        self._indicators = TechnicalIndicators()

        self._schedules: dict[str, DCASchedule] = {}
        self._execution_history: list[DCAExecution] = []

        logger.info(f"DCAStrategy initialized: {self.name}")

    def calculate_indicators(
        self,
        symbol: str,
        data: MarketData,
    ) -> dict[str, Any]:
        """
        Calculate indicators for DCA timing.

        Args:
            symbol: Trading symbol
            data: Market data

        Returns:
            Dictionary of indicator values
        """
        closes = data.closes
        highs = data.highs
        lows = data.lows

        if len(closes) < self._dca_config.timing_lookback:
            return {}

        lookback = self._dca_config.timing_lookback
        current_price = closes[-1]

        sma = sum(closes[-lookback:]) / lookback
        price_vs_sma = (current_price - sma) / sma if sma > 0 else 0

        recent_high = max(highs[-lookback:])
        recent_low = min(lows[-lookback:])
        price_vs_high = (current_price - recent_high) / recent_high if recent_high > 0 else 0
        price_vs_low = (current_price - recent_low) / recent_low if recent_low > 0 else 0

        rsi = self._indicators.rsi(closes, self._dca_config.rsi_period)
        current_rsi = rsi[-1] if rsi else 50.0

        returns = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]
        volatility = (sum(r ** 2 for r in returns[-lookback:]) / lookback) ** 0.5 if len(returns) >= lookback else 0

        is_dip = price_vs_sma < -self._dca_config.dip_threshold_pct
        is_spike = price_vs_sma > self._dca_config.spike_threshold_pct

        return {
            "current_price": current_price,
            "sma": sma,
            "price_vs_sma": price_vs_sma,
            "price_vs_high": price_vs_high,
            "price_vs_low": price_vs_low,
            "rsi": current_rsi,
            "volatility": volatility,
            "is_dip": is_dip,
            "is_spike": is_spike,
            "recent_high": recent_high,
            "recent_low": recent_low,
        }

    async def evaluate(
        self,
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """
        Evaluate DCA opportunities.

        Args:
            market_data: Market data for all symbols
            context: Strategy execution context

        Returns:
            List of DCA signals
        """
        signals: list[StrategySignal] = []

        for symbol in self.config.symbols:
            if symbol not in market_data:
                continue

            if symbol not in self._schedules:
                self._create_schedule(symbol, context)

            schedule = self._schedules[symbol]
            data = market_data[symbol]

            if self._should_execute(schedule, context):
                indicators = self.calculate_indicators(symbol, data)

                if indicators:
                    signal = self._generate_dca_signal(
                        schedule, indicators, context
                    )

                    if signal:
                        signals.append(signal)
                        self._record_execution(schedule, signal, indicators, context)

        return signals

    def _create_schedule(self, symbol: str, context: StrategyContext) -> None:
        """Create DCA schedule for symbol."""
        next_exec = self._calculate_next_execution(context.timestamp)

        schedule = DCASchedule(
            symbol=symbol,
            frequency=self._dca_config.frequency,
            base_amount=self._dca_config.base_investment_amount,
            next_execution=next_exec,
        )

        self._schedules[symbol] = schedule
        logger.info(f"DCA schedule created for {symbol}: {self._dca_config.frequency}")

    def _calculate_next_execution(self, from_date: datetime) -> datetime:
        """Calculate next execution date."""
        freq = self._dca_config.frequency

        if freq == "daily":
            return from_date + timedelta(days=1)
        elif freq == "weekly":
            days_until_monday = (7 - from_date.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7
            return from_date + timedelta(days=days_until_monday)
        elif freq == "biweekly":
            return from_date + timedelta(days=14)
        elif freq == "monthly":
            next_month = from_date.month % 12 + 1
            year = from_date.year + (1 if next_month == 1 else 0)
            return from_date.replace(year=year, month=next_month, day=1)
        else:
            return from_date + timedelta(days=7)

    def _should_execute(self, schedule: DCASchedule, context: StrategyContext) -> bool:
        """Check if DCA should execute."""
        current_time = context.timestamp
        window_days = self._dca_config.flexible_timing_window_days

        window_start = schedule.next_execution - timedelta(days=window_days)
        window_end = schedule.next_execution + timedelta(days=window_days)

        return window_start <= current_time <= window_end

    def _generate_dca_signal(
        self,
        schedule: DCASchedule,
        indicators: dict[str, Any],
        context: StrategyContext,
    ) -> Optional[StrategySignal]:
        """Generate DCA investment signal."""
        current_price = indicators["current_price"]
        is_dip = indicators["is_dip"]
        is_spike = indicators["is_spike"]
        rsi = indicators["rsi"]
        price_vs_sma = indicators["price_vs_sma"]

        if self._dca_config.use_value_averaging:
            investment_amount = self._calculate_value_averaging_amount(
                schedule, context
            )
        else:
            investment_amount = schedule.base_amount

        adjustment_reason = None

        if self._dca_config.use_smart_timing:
            if is_dip:
                bonus = investment_amount * self._dca_config.dip_bonus_pct
                investment_amount += bonus
                adjustment_reason = f"dip_bonus_{bonus:.2f}"

            elif is_spike:
                reduction = investment_amount * self._dca_config.spike_reduction_pct
                investment_amount -= reduction
                adjustment_reason = f"spike_reduction_{reduction:.2f}"

            if rsi < self._dca_config.rsi_oversold:
                rsi_bonus = investment_amount * 0.2
                investment_amount += rsi_bonus
                adjustment_reason = (adjustment_reason or "") + f"_rsi_oversold_bonus"

            elif rsi > self._dca_config.rsi_overbought:
                rsi_reduction = investment_amount * 0.15
                investment_amount -= rsi_reduction
                adjustment_reason = (adjustment_reason or "") + f"_rsi_overbought_reduction"

        max_amount = schedule.base_amount * self._dca_config.max_single_investment_multiplier
        min_amount = schedule.base_amount * self._dca_config.min_single_investment_multiplier

        investment_amount = max(min_amount, min(max_amount, investment_amount))

        position_size_pct = investment_amount / context.account_value if context.account_value > 0 else 0

        return self.create_signal(
            symbol=schedule.symbol,
            action=SignalAction.BUY,
            side=SignalSide.LONG,
            entry_price=current_price,
            strength=0.7,
            confidence=0.85,
            reason=f"DCA investment: ${investment_amount:.2f}",
            position_size_pct=position_size_pct,
            metadata={
                "strategy_type": "dca",
                "investment_amount": investment_amount,
                "base_amount": schedule.base_amount,
                "adjustment_reason": adjustment_reason,
                "price_vs_sma": price_vs_sma,
                "rsi": rsi,
                "execution_count": schedule.execution_count + 1,
                "total_invested": schedule.total_invested + investment_amount,
            },
        )

    def _calculate_value_averaging_amount(
        self,
        schedule: DCASchedule,
        context: StrategyContext,
    ) -> float:
        """Calculate value averaging investment amount."""
        target_growth = self._dca_config.target_growth_rate
        executions = schedule.execution_count + 1

        target_value = schedule.base_amount * executions * (1 + target_growth) ** executions

        current_value = schedule.total_shares * schedule.avg_price if schedule.avg_price > 0 else 0

        required_investment = target_value - current_value

        return max(
            schedule.base_amount * self._dca_config.min_single_investment_multiplier,
            required_investment
        )

    def _record_execution(
        self,
        schedule: DCASchedule,
        signal: StrategySignal,
        indicators: dict[str, Any],
        context: StrategyContext,
    ) -> None:
        """Record DCA execution."""
        current_price = indicators["current_price"]
        investment_amount = signal.metadata.get("investment_amount", schedule.base_amount)
        shares = investment_amount / current_price if current_price > 0 else 0

        execution = DCAExecution(
            schedule_id=schedule.schedule_id,
            symbol=schedule.symbol,
            timestamp=context.timestamp,
            price=current_price,
            amount=investment_amount,
            shares=shares,
            adjustment_reason=signal.metadata.get("adjustment_reason"),
        )

        self._execution_history.append(execution)

        total_shares = schedule.total_shares + shares
        total_invested = schedule.total_invested + investment_amount

        schedule.total_shares = total_shares
        schedule.total_invested = total_invested
        schedule.avg_price = total_invested / total_shares if total_shares > 0 else 0
        schedule.execution_count += 1
        schedule.last_execution = context.timestamp
        schedule.next_execution = self._calculate_next_execution(context.timestamp)

        if len(self._execution_history) > 1000:
            self._execution_history = self._execution_history[-1000:]

    def get_schedule(self, symbol: str) -> Optional[DCASchedule]:
        """Get DCA schedule for symbol."""
        return self._schedules.get(symbol)

    def get_all_schedules(self) -> dict[str, DCASchedule]:
        """Get all DCA schedules."""
        return self._schedules.copy()

    def update_schedule(
        self,
        symbol: str,
        frequency: Optional[str] = None,
        base_amount: Optional[float] = None,
    ) -> bool:
        """Update DCA schedule parameters."""
        if symbol not in self._schedules:
            return False

        schedule = self._schedules[symbol]

        if frequency:
            schedule.frequency = frequency
        if base_amount:
            schedule.base_amount = base_amount

        return True

    def pause_schedule(self, symbol: str) -> bool:
        """Pause DCA schedule for symbol."""
        if symbol not in self._schedules:
            return False

        schedule = self._schedules[symbol]
        schedule.next_execution = schedule.next_execution + timedelta(days=365 * 10)

        return True

    def resume_schedule(self, symbol: str) -> bool:
        """Resume DCA schedule for symbol."""
        if symbol not in self._schedules:
            return False

        schedule = self._schedules[symbol]
        schedule.next_execution = self._calculate_next_execution(now_utc())

        return True

    def get_execution_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> list[DCAExecution]:
        """Get DCA execution history."""
        if symbol:
            executions = [e for e in self._execution_history if e.symbol == symbol]
        else:
            executions = self._execution_history

        return executions[-limit:]

    def get_dca_statistics(self) -> dict:
        """Get DCA statistics."""
        total_invested = sum(s.total_invested for s in self._schedules.values())
        total_executions = sum(s.execution_count for s in self._schedules.values())

        return {
            "active_schedules": len(self._schedules),
            "total_invested": total_invested,
            "total_executions": total_executions,
            "schedules": {
                symbol: {
                    "frequency": schedule.frequency,
                    "base_amount": schedule.base_amount,
                    "total_invested": schedule.total_invested,
                    "total_shares": schedule.total_shares,
                    "avg_price": schedule.avg_price,
                    "execution_count": schedule.execution_count,
                    "next_execution": schedule.next_execution.isoformat(),
                }
                for symbol, schedule in self._schedules.items()
            },
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"DCAStrategy(schedules={len(self._schedules)})"
