"""
Options Trading Strategy Module for Ultimate Trading Bot v2.2.

This module implements various options trading strategies including
covered calls, protective puts, spreads, and straddles.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Optional
from enum import Enum

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


class OptionType(str, Enum):
    """Option type enumeration."""

    CALL = "call"
    PUT = "put"


class OptionPosition(BaseModel):
    """Model for option position."""

    position_id: str = Field(default_factory=generate_uuid)
    symbol: str
    underlying: str
    option_type: OptionType
    strike: float
    expiration: datetime
    contracts: int
    premium: float
    entry_date: datetime
    is_long: bool = Field(default=True)
    strategy_name: str = Field(default="single")


class OptionGreeks(BaseModel):
    """Model for option Greeks."""

    delta: float = Field(default=0.0)
    gamma: float = Field(default=0.0)
    theta: float = Field(default=0.0)
    vega: float = Field(default=0.0)
    rho: float = Field(default=0.0)


class OptionOpportunity(BaseModel):
    """Model for option trading opportunity."""

    opportunity_id: str = Field(default_factory=generate_uuid)
    underlying: str
    strategy_name: str
    legs: list[dict] = Field(default_factory=list)
    max_profit: float = Field(default=0.0)
    max_loss: float = Field(default=0.0)
    probability_profit: float = Field(default=0.5)
    expected_return: float = Field(default=0.0)
    greeks: OptionGreeks = Field(default_factory=OptionGreeks)
    timestamp: datetime


class OptionsStrategyConfig(StrategyConfig):
    """Configuration for options trading strategy."""

    name: str = Field(default="Options Strategy")
    description: str = Field(
        default="Options trading with various strategies"
    )

    enabled_strategies: list[str] = Field(
        default_factory=lambda: [
            "covered_call",
            "protective_put",
            "bull_call_spread",
            "bear_put_spread",
            "iron_condor",
            "straddle",
        ]
    )

    min_days_to_expiration: int = Field(default=7, ge=1, le=90)
    max_days_to_expiration: int = Field(default=45, ge=7, le=365)
    target_days_to_expiration: int = Field(default=30, ge=7, le=180)

    min_delta_short: float = Field(default=0.15, ge=0.05, le=0.5)
    max_delta_short: float = Field(default=0.35, ge=0.15, le=0.5)
    target_delta_long: float = Field(default=0.50, ge=0.3, le=0.7)

    min_iv_percentile: float = Field(default=0.3, ge=0.0, le=1.0)
    max_iv_percentile: float = Field(default=0.8, ge=0.2, le=1.0)

    max_position_size_pct: float = Field(default=0.05, ge=0.01, le=0.2)
    max_total_options_pct: float = Field(default=0.20, ge=0.05, le=0.5)

    min_probability_profit: float = Field(default=0.5, ge=0.3, le=0.9)
    min_risk_reward: float = Field(default=1.0, ge=0.5, le=5.0)

    roll_days_before_expiration: int = Field(default=5, ge=1, le=14)
    close_at_profit_pct: float = Field(default=0.50, ge=0.1, le=0.9)


class OptionsStrategy(BaseStrategy):
    """
    Options trading strategy.

    Features:
    - Multiple options strategies (calls, puts, spreads)
    - Greeks-based position management
    - IV percentile analysis
    - Probability-based entry
    - Automated rolling
    """

    def __init__(
        self,
        config: Optional[OptionsStrategyConfig] = None,
    ) -> None:
        """
        Initialize OptionsStrategy.

        Args:
            config: Options strategy configuration
        """
        config = config or OptionsStrategyConfig()
        super().__init__(config)

        self._options_config = config
        self._indicators = TechnicalIndicators()

        self._positions: dict[str, OptionPosition] = {}
        self._opportunities: list[OptionOpportunity] = []
        self._iv_history: dict[str, list[float]] = {}

        logger.info(f"OptionsStrategy initialized: {self.name}")

    def calculate_indicators(
        self,
        symbol: str,
        data: MarketData,
    ) -> dict[str, Any]:
        """
        Calculate indicators for options analysis.

        Args:
            symbol: Trading symbol
            data: Market data

        Returns:
            Dictionary of indicator values
        """
        closes = data.closes
        highs = data.highs
        lows = data.lows

        if len(closes) < 30:
            return {}

        current_price = closes[-1]

        returns = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]
        hv_20 = (sum(r ** 2 for r in returns[-20:]) / 20) ** 0.5 * (252 ** 0.5) if len(returns) >= 20 else 0.2

        iv_estimate = hv_20 * 1.1

        if symbol not in self._iv_history:
            self._iv_history[symbol] = []
        self._iv_history[symbol].append(iv_estimate)
        if len(self._iv_history[symbol]) > 252:
            self._iv_history[symbol] = self._iv_history[symbol][-252:]

        iv_history = self._iv_history[symbol]
        if len(iv_history) >= 30:
            sorted_iv = sorted(iv_history)
            iv_rank = sorted_iv.index(min(sorted_iv, key=lambda x: abs(x - iv_estimate))) / len(sorted_iv)
        else:
            iv_rank = 0.5

        atr = self._indicators.atr(highs, lows, closes, 14)
        atr_value = atr[-1] if atr else current_price * 0.02

        ema_20 = self._indicators.ema(closes, 20)
        ema_50 = self._indicators.ema(closes, 50)

        trend = "neutral"
        if ema_20 and ema_50:
            if ema_20[-1] > ema_50[-1]:
                trend = "bullish"
            elif ema_20[-1] < ema_50[-1]:
                trend = "bearish"

        rsi = self._indicators.rsi(closes, 14)
        current_rsi = rsi[-1] if rsi else 50.0

        return {
            "current_price": current_price,
            "historical_volatility": hv_20,
            "implied_volatility": iv_estimate,
            "iv_percentile": iv_rank,
            "atr": atr_value,
            "trend": trend,
            "rsi": current_rsi,
            "ema_20": ema_20[-1] if ema_20 else current_price,
            "ema_50": ema_50[-1] if ema_50 else current_price,
        }

    async def evaluate(
        self,
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """
        Evaluate options opportunities.

        Args:
            market_data: Market data for all symbols
            context: Strategy execution context

        Returns:
            List of options signals
        """
        signals: list[StrategySignal] = []

        exit_signals = self._check_position_management(context)
        signals.extend(exit_signals)

        for symbol in self.config.symbols:
            if symbol not in market_data:
                continue

            data = market_data[symbol]

            indicators = self.calculate_indicators(symbol, data)
            if not indicators:
                continue

            opportunities = self._find_opportunities(symbol, indicators, context)

            for opp in opportunities:
                if opp.probability_profit >= self._options_config.min_probability_profit:
                    opp_signals = self._create_option_signals(opp, context)
                    signals.extend(opp_signals)

        return signals

    def _find_opportunities(
        self,
        symbol: str,
        indicators: dict[str, Any],
        context: StrategyContext,
    ) -> list[OptionOpportunity]:
        """Find options trading opportunities."""
        opportunities: list[OptionOpportunity] = []

        iv_percentile = indicators["iv_percentile"]
        trend = indicators["trend"]
        current_price = indicators["current_price"]
        hv = indicators["historical_volatility"]
        rsi = indicators["rsi"]

        if "covered_call" in self._options_config.enabled_strategies:
            if trend in ["bullish", "neutral"]:
                opp = self._evaluate_covered_call(symbol, indicators, context)
                if opp:
                    opportunities.append(opp)

        if "protective_put" in self._options_config.enabled_strategies:
            if iv_percentile < 0.5:
                opp = self._evaluate_protective_put(symbol, indicators, context)
                if opp:
                    opportunities.append(opp)

        if "bull_call_spread" in self._options_config.enabled_strategies:
            if trend == "bullish" and rsi < 70:
                opp = self._evaluate_bull_call_spread(symbol, indicators, context)
                if opp:
                    opportunities.append(opp)

        if "bear_put_spread" in self._options_config.enabled_strategies:
            if trend == "bearish" and rsi > 30:
                opp = self._evaluate_bear_put_spread(symbol, indicators, context)
                if opp:
                    opportunities.append(opp)

        if "iron_condor" in self._options_config.enabled_strategies:
            if iv_percentile > self._options_config.min_iv_percentile:
                opp = self._evaluate_iron_condor(symbol, indicators, context)
                if opp:
                    opportunities.append(opp)

        if "straddle" in self._options_config.enabled_strategies:
            if iv_percentile < 0.3:
                opp = self._evaluate_straddle(symbol, indicators, context)
                if opp:
                    opportunities.append(opp)

        return opportunities

    def _evaluate_covered_call(
        self,
        symbol: str,
        indicators: dict[str, Any],
        context: StrategyContext,
    ) -> Optional[OptionOpportunity]:
        """Evaluate covered call opportunity."""
        current_price = indicators["current_price"]
        iv = indicators["implied_volatility"]
        atr = indicators["atr"]

        strike = round(current_price * 1.05, 2)
        expiration = context.timestamp + timedelta(
            days=self._options_config.target_days_to_expiration
        )

        premium = self._estimate_option_premium(
            current_price, strike, iv,
            self._options_config.target_days_to_expiration,
            OptionType.CALL,
        )

        max_profit = (strike - current_price) + premium
        max_loss = current_price - premium
        probability = 0.65

        return OptionOpportunity(
            underlying=symbol,
            strategy_name="covered_call",
            legs=[
                {"type": "stock", "action": "buy", "quantity": 100, "price": current_price},
                {"type": "call", "action": "sell", "strike": strike, "expiration": expiration.isoformat(), "premium": premium},
            ],
            max_profit=max_profit * 100,
            max_loss=max_loss * 100,
            probability_profit=probability,
            expected_return=max_profit * probability - max_loss * (1 - probability),
            greeks=OptionGreeks(
                delta=0.3,
                theta=premium / self._options_config.target_days_to_expiration,
            ),
            timestamp=context.timestamp,
        )

    def _evaluate_protective_put(
        self,
        symbol: str,
        indicators: dict[str, Any],
        context: StrategyContext,
    ) -> Optional[OptionOpportunity]:
        """Evaluate protective put opportunity."""
        current_price = indicators["current_price"]
        iv = indicators["implied_volatility"]

        strike = round(current_price * 0.95, 2)
        expiration = context.timestamp + timedelta(
            days=self._options_config.target_days_to_expiration
        )

        premium = self._estimate_option_premium(
            current_price, strike, iv,
            self._options_config.target_days_to_expiration,
            OptionType.PUT,
        )

        max_loss = premium + (current_price - strike)
        probability = 0.75

        return OptionOpportunity(
            underlying=symbol,
            strategy_name="protective_put",
            legs=[
                {"type": "stock", "action": "hold", "quantity": 100, "price": current_price},
                {"type": "put", "action": "buy", "strike": strike, "expiration": expiration.isoformat(), "premium": premium},
            ],
            max_profit=float("inf"),
            max_loss=max_loss * 100,
            probability_profit=probability,
            expected_return=0,
            greeks=OptionGreeks(delta=-0.25),
            timestamp=context.timestamp,
        )

    def _evaluate_bull_call_spread(
        self,
        symbol: str,
        indicators: dict[str, Any],
        context: StrategyContext,
    ) -> Optional[OptionOpportunity]:
        """Evaluate bull call spread opportunity."""
        current_price = indicators["current_price"]
        iv = indicators["implied_volatility"]
        atr = indicators["atr"]

        lower_strike = round(current_price * 0.98, 2)
        upper_strike = round(current_price * 1.05, 2)

        expiration = context.timestamp + timedelta(
            days=self._options_config.target_days_to_expiration
        )

        long_premium = self._estimate_option_premium(
            current_price, lower_strike, iv,
            self._options_config.target_days_to_expiration,
            OptionType.CALL,
        )

        short_premium = self._estimate_option_premium(
            current_price, upper_strike, iv,
            self._options_config.target_days_to_expiration,
            OptionType.CALL,
        )

        net_debit = long_premium - short_premium
        max_profit = (upper_strike - lower_strike) - net_debit
        max_loss = net_debit
        probability = 0.55

        return OptionOpportunity(
            underlying=symbol,
            strategy_name="bull_call_spread",
            legs=[
                {"type": "call", "action": "buy", "strike": lower_strike, "expiration": expiration.isoformat(), "premium": long_premium},
                {"type": "call", "action": "sell", "strike": upper_strike, "expiration": expiration.isoformat(), "premium": short_premium},
            ],
            max_profit=max_profit * 100,
            max_loss=max_loss * 100,
            probability_profit=probability,
            expected_return=max_profit * probability - max_loss * (1 - probability),
            greeks=OptionGreeks(delta=0.35),
            timestamp=context.timestamp,
        )

    def _evaluate_bear_put_spread(
        self,
        symbol: str,
        indicators: dict[str, Any],
        context: StrategyContext,
    ) -> Optional[OptionOpportunity]:
        """Evaluate bear put spread opportunity."""
        current_price = indicators["current_price"]
        iv = indicators["implied_volatility"]

        upper_strike = round(current_price * 1.02, 2)
        lower_strike = round(current_price * 0.95, 2)

        expiration = context.timestamp + timedelta(
            days=self._options_config.target_days_to_expiration
        )

        long_premium = self._estimate_option_premium(
            current_price, upper_strike, iv,
            self._options_config.target_days_to_expiration,
            OptionType.PUT,
        )

        short_premium = self._estimate_option_premium(
            current_price, lower_strike, iv,
            self._options_config.target_days_to_expiration,
            OptionType.PUT,
        )

        net_debit = long_premium - short_premium
        max_profit = (upper_strike - lower_strike) - net_debit
        max_loss = net_debit
        probability = 0.55

        return OptionOpportunity(
            underlying=symbol,
            strategy_name="bear_put_spread",
            legs=[
                {"type": "put", "action": "buy", "strike": upper_strike, "expiration": expiration.isoformat(), "premium": long_premium},
                {"type": "put", "action": "sell", "strike": lower_strike, "expiration": expiration.isoformat(), "premium": short_premium},
            ],
            max_profit=max_profit * 100,
            max_loss=max_loss * 100,
            probability_profit=probability,
            expected_return=max_profit * probability - max_loss * (1 - probability),
            greeks=OptionGreeks(delta=-0.35),
            timestamp=context.timestamp,
        )

    def _evaluate_iron_condor(
        self,
        symbol: str,
        indicators: dict[str, Any],
        context: StrategyContext,
    ) -> Optional[OptionOpportunity]:
        """Evaluate iron condor opportunity."""
        current_price = indicators["current_price"]
        iv = indicators["implied_volatility"]
        atr = indicators["atr"]

        put_long_strike = round(current_price * 0.90, 2)
        put_short_strike = round(current_price * 0.95, 2)
        call_short_strike = round(current_price * 1.05, 2)
        call_long_strike = round(current_price * 1.10, 2)

        expiration = context.timestamp + timedelta(
            days=self._options_config.target_days_to_expiration
        )

        total_credit = iv * current_price * 0.02
        wing_width = put_short_strike - put_long_strike

        max_profit = total_credit
        max_loss = wing_width - total_credit
        probability = 0.70

        return OptionOpportunity(
            underlying=symbol,
            strategy_name="iron_condor",
            legs=[
                {"type": "put", "action": "buy", "strike": put_long_strike, "expiration": expiration.isoformat()},
                {"type": "put", "action": "sell", "strike": put_short_strike, "expiration": expiration.isoformat()},
                {"type": "call", "action": "sell", "strike": call_short_strike, "expiration": expiration.isoformat()},
                {"type": "call", "action": "buy", "strike": call_long_strike, "expiration": expiration.isoformat()},
            ],
            max_profit=max_profit * 100,
            max_loss=max_loss * 100,
            probability_profit=probability,
            expected_return=max_profit * probability - max_loss * (1 - probability),
            greeks=OptionGreeks(delta=0, theta=total_credit / self._options_config.target_days_to_expiration),
            timestamp=context.timestamp,
        )

    def _evaluate_straddle(
        self,
        symbol: str,
        indicators: dict[str, Any],
        context: StrategyContext,
    ) -> Optional[OptionOpportunity]:
        """Evaluate long straddle opportunity."""
        current_price = indicators["current_price"]
        iv = indicators["implied_volatility"]

        strike = round(current_price, 2)

        expiration = context.timestamp + timedelta(
            days=self._options_config.target_days_to_expiration
        )

        call_premium = self._estimate_option_premium(
            current_price, strike, iv,
            self._options_config.target_days_to_expiration,
            OptionType.CALL,
        )

        put_premium = self._estimate_option_premium(
            current_price, strike, iv,
            self._options_config.target_days_to_expiration,
            OptionType.PUT,
        )

        total_premium = call_premium + put_premium
        max_loss = total_premium
        breakeven_move = total_premium / current_price
        probability = 0.45

        return OptionOpportunity(
            underlying=symbol,
            strategy_name="straddle",
            legs=[
                {"type": "call", "action": "buy", "strike": strike, "expiration": expiration.isoformat(), "premium": call_premium},
                {"type": "put", "action": "buy", "strike": strike, "expiration": expiration.isoformat(), "premium": put_premium},
            ],
            max_profit=float("inf"),
            max_loss=max_loss * 100,
            probability_profit=probability,
            expected_return=0,
            greeks=OptionGreeks(delta=0, vega=total_premium * 0.1),
            timestamp=context.timestamp,
        )

    def _estimate_option_premium(
        self,
        spot: float,
        strike: float,
        iv: float,
        days: int,
        option_type: OptionType,
    ) -> float:
        """Estimate option premium using simplified Black-Scholes approximation."""
        import math

        t = days / 365
        r = 0.05

        d1 = (math.log(spot / strike) + (r + iv ** 2 / 2) * t) / (iv * math.sqrt(t))
        d2 = d1 - iv * math.sqrt(t)

        def norm_cdf(x: float) -> float:
            return (1 + math.erf(x / math.sqrt(2))) / 2

        if option_type == OptionType.CALL:
            premium = spot * norm_cdf(d1) - strike * math.exp(-r * t) * norm_cdf(d2)
        else:
            premium = strike * math.exp(-r * t) * norm_cdf(-d2) - spot * norm_cdf(-d1)

        return max(0.01, premium)

    def _create_option_signals(
        self,
        opportunity: OptionOpportunity,
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """Create trading signals from option opportunity."""
        signals: list[StrategySignal] = []

        for leg in opportunity.legs:
            if leg["type"] == "stock":
                continue

            action = SignalAction.BUY if leg["action"] == "buy" else SignalAction.SELL
            side = SignalSide.LONG if leg["action"] == "buy" else SignalSide.SHORT

            option_symbol = f"{opportunity.underlying}_{leg['strike']}_{leg['type'][0].upper()}"

            signal = self.create_signal(
                symbol=option_symbol,
                action=action,
                side=side,
                entry_price=leg.get("premium", 0),
                strength=opportunity.probability_profit,
                confidence=0.7,
                reason=f"Options: {opportunity.strategy_name}",
                metadata={
                    "strategy_type": "options",
                    "options_strategy": opportunity.strategy_name,
                    "underlying": opportunity.underlying,
                    "strike": leg["strike"],
                    "option_type": leg["type"],
                    "expiration": leg["expiration"],
                    "max_profit": opportunity.max_profit,
                    "max_loss": opportunity.max_loss,
                    "probability_profit": opportunity.probability_profit,
                },
            )

            if signal:
                signals.append(signal)

        return signals

    def _check_position_management(
        self,
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """Check positions for rolling or closing."""
        signals: list[StrategySignal] = []

        for position_id, position in list(self._positions.items()):
            days_to_exp = (position.expiration - context.timestamp).days

            if days_to_exp <= self._options_config.roll_days_before_expiration:
                roll_signals = self._create_roll_signals(position, context)
                signals.extend(roll_signals)

        return signals

    def _create_roll_signals(
        self,
        position: OptionPosition,
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """Create signals to roll an option position."""
        close_signal = self.create_signal(
            symbol=f"{position.underlying}_{position.strike}_{position.option_type.value[0].upper()}",
            action=SignalAction.BUY if not position.is_long else SignalAction.SELL,
            side=SignalSide.FLAT,
            entry_price=position.premium,
            strength=1.0,
            confidence=0.9,
            reason=f"Roll: close expiring {position.option_type.value}",
            metadata={
                "strategy_type": "options",
                "action": "roll_close",
                "position_id": position.position_id,
            },
        )

        return [close_signal] if close_signal else []

    def get_positions(self) -> dict[str, OptionPosition]:
        """Get current option positions."""
        return self._positions.copy()

    def get_opportunities(self) -> list[OptionOpportunity]:
        """Get recent opportunities."""
        return self._opportunities[-20:]

    def get_iv_percentile(self, symbol: str) -> float:
        """Get current IV percentile for symbol."""
        if symbol not in self._iv_history or len(self._iv_history[symbol]) < 30:
            return 0.5

        history = self._iv_history[symbol]
        current = history[-1]
        sorted_iv = sorted(history)
        rank = sorted_iv.index(min(sorted_iv, key=lambda x: abs(x - current)))

        return rank / len(sorted_iv)

    def get_options_statistics(self) -> dict:
        """Get options strategy statistics."""
        return {
            "active_positions": len(self._positions),
            "strategies_enabled": self._options_config.enabled_strategies,
            "iv_tracked_symbols": len(self._iv_history),
            "recent_opportunities": len(self._opportunities),
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"OptionsStrategy(positions={len(self._positions)})"
