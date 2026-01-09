"""
Position Sizing Module for Ultimate Trading Bot v2.2.

This module implements various position sizing algorithms
for optimal capital allocation.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field

from src.risk.base_risk import (
    RiskConfig,
    RiskLevel,
    RiskContext,
    calculate_var,
)
from src.utils.helpers import generate_uuid


logger = logging.getLogger(__name__)


class SizingMethod(str, Enum):
    """Position sizing method enumeration."""

    FIXED_AMOUNT = "fixed_amount"
    FIXED_PERCENT = "fixed_percent"
    RISK_PERCENT = "risk_percent"
    KELLY = "kelly"
    OPTIMAL_F = "optimal_f"
    ATR_BASED = "atr_based"
    VOLATILITY_SCALED = "volatility_scaled"
    VAR_BASED = "var_based"
    EQUAL_WEIGHT = "equal_weight"


class PositionSizeResult(BaseModel):
    """Model for position sizing result."""

    result_id: str = Field(default_factory=generate_uuid)
    symbol: str
    method: SizingMethod
    calculated_size: float
    adjusted_size: float
    shares: int
    notional_value: float
    risk_amount: float
    position_pct: float
    timestamp: datetime
    constraints_applied: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class PositionSizerConfig(RiskConfig):
    """Configuration for position sizing."""

    default_method: SizingMethod = Field(default=SizingMethod.RISK_PERCENT)

    fixed_amount: float = Field(default=1000.0, ge=100.0, le=1000000.0)
    fixed_percent: float = Field(default=0.05, ge=0.01, le=0.5)
    risk_percent: float = Field(default=0.01, ge=0.001, le=0.05)

    max_position_pct: float = Field(default=0.10, ge=0.01, le=0.5)
    min_position_pct: float = Field(default=0.01, ge=0.001, le=0.1)

    kelly_fraction: float = Field(default=0.25, ge=0.1, le=1.0)
    kelly_max_bet: float = Field(default=0.25, ge=0.1, le=0.5)

    atr_multiplier: float = Field(default=2.0, ge=0.5, le=5.0)
    atr_period: int = Field(default=14, ge=5, le=30)

    target_volatility: float = Field(default=0.15, ge=0.05, le=0.5)
    vol_lookback: int = Field(default=20, ge=5, le=60)

    var_confidence: float = Field(default=0.95, ge=0.9, le=0.99)
    max_var_pct: float = Field(default=0.02, ge=0.005, le=0.1)

    use_correlation_adjustment: bool = Field(default=True)
    max_correlation_exposure: float = Field(default=0.3, ge=0.1, le=0.5)

    round_to_lot_size: bool = Field(default=True)
    min_lot_size: int = Field(default=1, ge=1)


class PositionSizer:
    """
    Position sizing calculator.

    Features:
    - Multiple sizing algorithms
    - Risk-based position sizing
    - Kelly criterion
    - Volatility scaling
    - Correlation adjustment
    """

    def __init__(
        self,
        config: Optional[PositionSizerConfig] = None,
    ) -> None:
        """
        Initialize PositionSizer.

        Args:
            config: Position sizing configuration
        """
        self.config = config or PositionSizerConfig()
        self._sizing_history: list[PositionSizeResult] = []
        self._win_rates: dict[str, float] = {}
        self._avg_wins: dict[str, float] = {}
        self._avg_losses: dict[str, float] = {}

        logger.info("PositionSizer initialized")

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: Optional[float],
        context: RiskContext,
        method: Optional[SizingMethod] = None,
        volatility: Optional[float] = None,
        atr: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
    ) -> PositionSizeResult:
        """
        Calculate position size for a trade.

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            context: Risk context
            method: Sizing method override
            volatility: Symbol volatility
            atr: Average true range
            win_rate: Historical win rate
            avg_win: Average winning trade
            avg_loss: Average losing trade

        Returns:
            Position sizing result
        """
        from src.utils.date_utils import now_utc

        method = method or self.config.default_method
        account_value = context.account_value
        constraints: list[str] = []

        if method == SizingMethod.FIXED_AMOUNT:
            raw_size = self._fixed_amount_size(account_value)

        elif method == SizingMethod.FIXED_PERCENT:
            raw_size = self._fixed_percent_size(account_value)

        elif method == SizingMethod.RISK_PERCENT:
            raw_size = self._risk_percent_size(
                account_value, entry_price, stop_loss
            )

        elif method == SizingMethod.KELLY:
            raw_size = self._kelly_size(
                account_value,
                win_rate or self._win_rates.get(symbol, 0.5),
                avg_win or self._avg_wins.get(symbol, 1.0),
                avg_loss or self._avg_losses.get(symbol, 1.0),
            )

        elif method == SizingMethod.OPTIMAL_F:
            raw_size = self._optimal_f_size(
                account_value,
                win_rate or self._win_rates.get(symbol, 0.5),
                avg_loss or self._avg_losses.get(symbol, 1.0),
            )

        elif method == SizingMethod.ATR_BASED:
            raw_size = self._atr_based_size(
                account_value, entry_price, atr
            )

        elif method == SizingMethod.VOLATILITY_SCALED:
            raw_size = self._volatility_scaled_size(
                account_value, entry_price, volatility
            )

        elif method == SizingMethod.VAR_BASED:
            raw_size = self._var_based_size(
                account_value, entry_price, volatility
            )

        elif method == SizingMethod.EQUAL_WEIGHT:
            raw_size = self._equal_weight_size(
                account_value, context.position_count + 1
            )

        else:
            raw_size = self._fixed_percent_size(account_value)

        adjusted_size = raw_size

        max_position = account_value * self.config.max_position_pct
        if adjusted_size > max_position:
            adjusted_size = max_position
            constraints.append(f"max_position_{self.config.max_position_pct:.0%}")

        min_position = account_value * self.config.min_position_pct
        if adjusted_size < min_position and raw_size > 0:
            adjusted_size = min_position
            constraints.append(f"min_position_{self.config.min_position_pct:.0%}")

        if self.config.use_correlation_adjustment:
            adjusted_size = self._apply_correlation_adjustment(
                symbol, adjusted_size, context
            )
            if adjusted_size < raw_size:
                constraints.append("correlation_adjustment")

        shares = int(adjusted_size / entry_price) if entry_price > 0 else 0

        if self.config.round_to_lot_size:
            lot_size = self.config.min_lot_size
            shares = (shares // lot_size) * lot_size

        shares = max(0, shares)
        notional_value = shares * entry_price
        position_pct = notional_value / account_value if account_value > 0 else 0

        risk_amount = 0.0
        if stop_loss and entry_price > 0:
            risk_per_share = abs(entry_price - stop_loss)
            risk_amount = shares * risk_per_share

        result = PositionSizeResult(
            symbol=symbol,
            method=method,
            calculated_size=raw_size,
            adjusted_size=adjusted_size,
            shares=shares,
            notional_value=notional_value,
            risk_amount=risk_amount,
            position_pct=position_pct,
            timestamp=now_utc(),
            constraints_applied=constraints,
            metadata={
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "account_value": account_value,
                "volatility": volatility,
                "atr": atr,
            },
        )

        self._sizing_history.append(result)

        if len(self._sizing_history) > 1000:
            self._sizing_history = self._sizing_history[-1000:]

        return result

    def _fixed_amount_size(self, account_value: float) -> float:
        """Fixed dollar amount sizing."""
        return min(self.config.fixed_amount, account_value * 0.5)

    def _fixed_percent_size(self, account_value: float) -> float:
        """Fixed percentage of account sizing."""
        return account_value * self.config.fixed_percent

    def _risk_percent_size(
        self,
        account_value: float,
        entry_price: float,
        stop_loss: Optional[float],
    ) -> float:
        """Risk-based position sizing."""
        if not stop_loss or entry_price <= 0:
            return self._fixed_percent_size(account_value)

        risk_per_share = abs(entry_price - stop_loss)

        if risk_per_share <= 0:
            return self._fixed_percent_size(account_value)

        risk_amount = account_value * self.config.risk_percent
        shares = risk_amount / risk_per_share
        position_value = shares * entry_price

        return position_value

    def _kelly_size(
        self,
        account_value: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """Kelly criterion position sizing."""
        if avg_loss <= 0 or avg_win <= 0:
            return self._fixed_percent_size(account_value)

        win_rate = max(0.01, min(0.99, win_rate))

        b = avg_win / avg_loss

        kelly_pct = (win_rate * b - (1 - win_rate)) / b

        kelly_pct = max(0, min(kelly_pct, 1))

        fractional_kelly = kelly_pct * self.config.kelly_fraction

        fractional_kelly = min(fractional_kelly, self.config.kelly_max_bet)

        return account_value * fractional_kelly

    def _optimal_f_size(
        self,
        account_value: float,
        win_rate: float,
        avg_loss: float,
    ) -> float:
        """Optimal f position sizing."""
        if avg_loss <= 0:
            return self._fixed_percent_size(account_value)

        win_rate = max(0.01, min(0.99, win_rate))

        optimal_f = win_rate - ((1 - win_rate) / 1.5)
        optimal_f = max(0.05, min(0.25, optimal_f))

        return account_value * optimal_f

    def _atr_based_size(
        self,
        account_value: float,
        entry_price: float,
        atr: Optional[float],
    ) -> float:
        """ATR-based position sizing."""
        if not atr or atr <= 0 or entry_price <= 0:
            return self._fixed_percent_size(account_value)

        risk_amount = account_value * self.config.risk_percent
        risk_per_share = atr * self.config.atr_multiplier

        if risk_per_share <= 0:
            return self._fixed_percent_size(account_value)

        shares = risk_amount / risk_per_share
        position_value = shares * entry_price

        return position_value

    def _volatility_scaled_size(
        self,
        account_value: float,
        entry_price: float,
        volatility: Optional[float],
    ) -> float:
        """Volatility-scaled position sizing."""
        if not volatility or volatility <= 0:
            return self._fixed_percent_size(account_value)

        target_vol = self.config.target_volatility
        vol_scalar = target_vol / volatility

        vol_scalar = max(0.25, min(4.0, vol_scalar))

        base_size = account_value * self.config.fixed_percent
        scaled_size = base_size * vol_scalar

        return scaled_size

    def _var_based_size(
        self,
        account_value: float,
        entry_price: float,
        volatility: Optional[float],
    ) -> float:
        """VaR-based position sizing."""
        if not volatility or volatility <= 0:
            return self._fixed_percent_size(account_value)

        from scipy import stats
        z_score = stats.norm.ppf(self.config.var_confidence)

        daily_vol = volatility / np.sqrt(252)

        var_1d = z_score * daily_vol

        max_var = account_value * self.config.max_var_pct
        max_position = max_var / var_1d if var_1d > 0 else account_value * 0.1

        return min(max_position, account_value * self.config.max_position_pct)

    def _equal_weight_size(
        self,
        account_value: float,
        total_positions: int,
    ) -> float:
        """Equal weight position sizing."""
        total_positions = max(1, total_positions)
        weight = 1.0 / total_positions
        weight = min(weight, self.config.max_position_pct)

        return account_value * weight

    def _apply_correlation_adjustment(
        self,
        symbol: str,
        position_size: float,
        context: RiskContext,
    ) -> float:
        """Apply correlation-based position adjustment."""
        correlated_exposure = 0.0
        account_value = context.account_value

        for pos in context.positions:
            pos_symbol = pos.get("symbol", "")
            pos_value = abs(pos.get("market_value", 0))

            correlation = self._get_correlation(symbol, pos_symbol)

            if correlation > 0.5:
                correlated_exposure += pos_value * correlation

        max_correlated = account_value * self.config.max_correlation_exposure

        if correlated_exposure > 0:
            remaining_capacity = max(0, max_correlated - correlated_exposure)
            position_size = min(position_size, remaining_capacity)

        return position_size

    def _get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols (placeholder)."""
        if symbol1 == symbol2:
            return 1.0

        return 0.3

    def update_trade_statistics(
        self,
        symbol: str,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> None:
        """Update trade statistics for Kelly calculation."""
        self._win_rates[symbol] = win_rate
        self._avg_wins[symbol] = avg_win
        self._avg_losses[symbol] = avg_loss

    def get_sizing_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> list[PositionSizeResult]:
        """Get position sizing history."""
        if symbol:
            history = [h for h in self._sizing_history if h.symbol == symbol]
        else:
            history = self._sizing_history

        return history[-limit:]

    def calculate_portfolio_size(
        self,
        symbols: list[str],
        entry_prices: dict[str, float],
        context: RiskContext,
        target_weights: Optional[dict[str, float]] = None,
    ) -> dict[str, PositionSizeResult]:
        """
        Calculate position sizes for a portfolio.

        Args:
            symbols: List of symbols
            entry_prices: Entry prices for each symbol
            context: Risk context
            target_weights: Optional target weights

        Returns:
            Dictionary of position sizing results
        """
        results: dict[str, PositionSizeResult] = {}

        if target_weights is None:
            target_weights = {s: 1.0 / len(symbols) for s in symbols}

        total_weight = sum(target_weights.values())
        if total_weight > 0:
            target_weights = {
                k: v / total_weight for k, v in target_weights.items()
            }

        for symbol in symbols:
            if symbol not in entry_prices:
                continue

            weight = target_weights.get(symbol, 1.0 / len(symbols))

            modified_context = RiskContext(
                timestamp=context.timestamp,
                account_value=context.account_value * weight,
                cash_balance=context.cash_balance * weight,
                positions=[],
                open_orders=[],
                daily_pnl=0,
                unrealized_pnl=0,
            )

            result = self.calculate_position_size(
                symbol=symbol,
                entry_price=entry_prices[symbol],
                stop_loss=None,
                context=modified_context,
                method=SizingMethod.FIXED_PERCENT,
            )

            results[symbol] = result

        return results

    def get_statistics(self) -> dict:
        """Get position sizer statistics."""
        if not self._sizing_history:
            return {"total_calculations": 0}

        avg_position_pct = sum(r.position_pct for r in self._sizing_history) / len(self._sizing_history)

        method_counts = {}
        for result in self._sizing_history:
            method = result.method.value
            method_counts[method] = method_counts.get(method, 0) + 1

        return {
            "total_calculations": len(self._sizing_history),
            "avg_position_pct": avg_position_pct,
            "method_distribution": method_counts,
            "symbols_tracked": len(self._win_rates),
        }
