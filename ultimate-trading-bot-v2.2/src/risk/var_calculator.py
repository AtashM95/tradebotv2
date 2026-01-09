"""
Value at Risk (VaR) Calculator Module for Ultimate Trading Bot v2.2.

This module implements multiple VaR calculation methods including
historical, parametric, and Monte Carlo simulation.
"""

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field

from src.risk.base_risk import (
    RiskConfig,
    RiskLevel,
    RiskType,
    RiskMetric,
    RiskAlert,
    RiskContext,
)
from src.utils.helpers import generate_uuid
from src.utils.date_utils import now_utc


logger = logging.getLogger(__name__)


class VaRMethod(str, Enum):
    """VaR calculation method enumeration."""

    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"
    CORNISH_FISHER = "cornish_fisher"
    EWMA = "ewma"


class VaRResult(BaseModel):
    """Model for VaR calculation result."""

    result_id: str = Field(default_factory=generate_uuid)
    timestamp: datetime
    method: VaRMethod
    confidence_level: float
    holding_period: int

    var_absolute: float = Field(default=0.0)
    var_percentage: float = Field(default=0.0)
    cvar_absolute: float = Field(default=0.0)
    cvar_percentage: float = Field(default=0.0)

    portfolio_value: float = Field(default=0.0)
    expected_return: float = Field(default=0.0)
    volatility: float = Field(default=0.0)

    component_var: dict[str, float] = Field(default_factory=dict)
    marginal_var: dict[str, float] = Field(default_factory=dict)

    metadata: dict = Field(default_factory=dict)


class VaRConfig(RiskConfig):
    """Configuration for VaR calculator."""

    default_method: VaRMethod = Field(default=VaRMethod.HISTORICAL)
    confidence_levels: list[float] = Field(
        default_factory=lambda: [0.95, 0.99]
    )
    holding_periods: list[int] = Field(
        default_factory=lambda: [1, 5, 10]
    )

    lookback_days: int = Field(default=252, ge=30, le=1000)
    min_data_points: int = Field(default=60, ge=20, le=252)

    monte_carlo_simulations: int = Field(default=10000, ge=1000, le=100000)
    ewma_decay_factor: float = Field(default=0.94, ge=0.9, le=0.99)

    use_correlation: bool = Field(default=True)
    stress_test_multiplier: float = Field(default=2.0, ge=1.0, le=5.0)


class VaRCalculator:
    """
    Value at Risk calculator.

    Features:
    - Multiple VaR methodologies
    - Component and marginal VaR
    - CVaR (Expected Shortfall)
    - Stress testing
    - Portfolio decomposition
    """

    def __init__(
        self,
        config: Optional[VaRConfig] = None,
    ) -> None:
        """
        Initialize VaRCalculator.

        Args:
            config: VaR configuration
        """
        self.config = config or VaRConfig()
        self._returns_history: dict[str, list[float]] = {}
        self._var_history: list[VaRResult] = []
        self._correlation_matrix: Optional[np.ndarray] = None

        logger.info("VaRCalculator initialized")

    def calculate_var(
        self,
        portfolio_value: float,
        positions: list[dict],
        returns_data: dict[str, list[float]],
        method: Optional[VaRMethod] = None,
        confidence_level: float = 0.95,
        holding_period: int = 1,
    ) -> VaRResult:
        """
        Calculate Value at Risk for portfolio.

        Args:
            portfolio_value: Total portfolio value
            positions: List of position dictionaries
            returns_data: Historical returns for each symbol
            method: VaR calculation method
            confidence_level: Confidence level (e.g., 0.95)
            holding_period: Holding period in days

        Returns:
            VaR calculation result
        """
        method = method or self.config.default_method

        self._returns_history = returns_data

        if method == VaRMethod.HISTORICAL:
            var_pct, cvar_pct = self._calculate_historical_var(
                positions, returns_data, confidence_level, holding_period
            )

        elif method == VaRMethod.PARAMETRIC:
            var_pct, cvar_pct = self._calculate_parametric_var(
                positions, returns_data, confidence_level, holding_period
            )

        elif method == VaRMethod.MONTE_CARLO:
            var_pct, cvar_pct = self._calculate_monte_carlo_var(
                positions, returns_data, confidence_level, holding_period
            )

        elif method == VaRMethod.CORNISH_FISHER:
            var_pct, cvar_pct = self._calculate_cornish_fisher_var(
                positions, returns_data, confidence_level, holding_period
            )

        elif method == VaRMethod.EWMA:
            var_pct, cvar_pct = self._calculate_ewma_var(
                positions, returns_data, confidence_level, holding_period
            )

        else:
            var_pct, cvar_pct = self._calculate_historical_var(
                positions, returns_data, confidence_level, holding_period
            )

        var_absolute = portfolio_value * var_pct
        cvar_absolute = portfolio_value * cvar_pct

        component_var = self._calculate_component_var(
            positions, returns_data, confidence_level, portfolio_value
        )

        marginal_var = self._calculate_marginal_var(
            positions, returns_data, confidence_level, portfolio_value
        )

        portfolio_returns = self._calculate_portfolio_returns(positions, returns_data)
        expected_return = np.mean(portfolio_returns) if portfolio_returns else 0
        volatility = np.std(portfolio_returns) * np.sqrt(252) if portfolio_returns else 0

        result = VaRResult(
            timestamp=now_utc(),
            method=method,
            confidence_level=confidence_level,
            holding_period=holding_period,
            var_absolute=var_absolute,
            var_percentage=var_pct,
            cvar_absolute=cvar_absolute,
            cvar_percentage=cvar_pct,
            portfolio_value=portfolio_value,
            expected_return=expected_return,
            volatility=volatility,
            component_var=component_var,
            marginal_var=marginal_var,
        )

        self._var_history.append(result)
        if len(self._var_history) > 1000:
            self._var_history = self._var_history[-1000:]

        return result

    def _calculate_portfolio_returns(
        self,
        positions: list[dict],
        returns_data: dict[str, list[float]],
    ) -> list[float]:
        """Calculate weighted portfolio returns."""
        total_value = sum(abs(p.get("market_value", 0)) for p in positions)
        if total_value == 0:
            return []

        min_length = min(
            len(returns_data.get(p.get("symbol", ""), []))
            for p in positions
            if p.get("symbol") in returns_data
        )

        if min_length == 0:
            return []

        portfolio_returns = [0.0] * min_length

        for position in positions:
            symbol = position.get("symbol", "")
            market_value = position.get("market_value", 0)

            if symbol not in returns_data:
                continue

            weight = market_value / total_value
            returns = returns_data[symbol][-min_length:]

            for i, r in enumerate(returns):
                portfolio_returns[i] += weight * r

        return portfolio_returns

    def _calculate_historical_var(
        self,
        positions: list[dict],
        returns_data: dict[str, list[float]],
        confidence_level: float,
        holding_period: int,
    ) -> tuple[float, float]:
        """Calculate historical VaR."""
        portfolio_returns = self._calculate_portfolio_returns(positions, returns_data)

        if len(portfolio_returns) < self.config.min_data_points:
            return 0.0, 0.0

        if holding_period > 1:
            scaled_returns = [r * np.sqrt(holding_period) for r in portfolio_returns]
        else:
            scaled_returns = portfolio_returns

        sorted_returns = sorted(scaled_returns)
        var_index = int((1 - confidence_level) * len(sorted_returns))
        var_index = max(0, min(var_index, len(sorted_returns) - 1))

        var = abs(sorted_returns[var_index])

        tail_returns = sorted_returns[:var_index + 1]
        cvar = abs(np.mean(tail_returns)) if tail_returns else var

        return var, cvar

    def _calculate_parametric_var(
        self,
        positions: list[dict],
        returns_data: dict[str, list[float]],
        confidence_level: float,
        holding_period: int,
    ) -> tuple[float, float]:
        """Calculate parametric (variance-covariance) VaR."""
        from scipy import stats

        portfolio_returns = self._calculate_portfolio_returns(positions, returns_data)

        if len(portfolio_returns) < self.config.min_data_points:
            return 0.0, 0.0

        mean_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)

        z_score = stats.norm.ppf(1 - confidence_level)

        var = abs(mean_return + z_score * std_return)
        var *= np.sqrt(holding_period)

        pdf_at_z = stats.norm.pdf(z_score)
        cvar = abs(mean_return + std_return * pdf_at_z / (1 - confidence_level))
        cvar *= np.sqrt(holding_period)

        return var, cvar

    def _calculate_monte_carlo_var(
        self,
        positions: list[dict],
        returns_data: dict[str, list[float]],
        confidence_level: float,
        holding_period: int,
    ) -> tuple[float, float]:
        """Calculate Monte Carlo VaR."""
        portfolio_returns = self._calculate_portfolio_returns(positions, returns_data)

        if len(portfolio_returns) < self.config.min_data_points:
            return 0.0, 0.0

        mean_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)

        num_sims = self.config.monte_carlo_simulations
        simulated_returns = np.random.normal(
            mean_return * holding_period,
            std_return * np.sqrt(holding_period),
            num_sims
        )

        sorted_sims = sorted(simulated_returns)
        var_index = int((1 - confidence_level) * num_sims)

        var = abs(sorted_sims[var_index])

        tail_sims = sorted_sims[:var_index + 1]
        cvar = abs(np.mean(tail_sims))

        return var, cvar

    def _calculate_cornish_fisher_var(
        self,
        positions: list[dict],
        returns_data: dict[str, list[float]],
        confidence_level: float,
        holding_period: int,
    ) -> tuple[float, float]:
        """Calculate Cornish-Fisher adjusted VaR."""
        from scipy import stats

        portfolio_returns = self._calculate_portfolio_returns(positions, returns_data)

        if len(portfolio_returns) < self.config.min_data_points:
            return 0.0, 0.0

        mean_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)
        skewness = stats.skew(portfolio_returns)
        kurtosis = stats.kurtosis(portfolio_returns)

        z = stats.norm.ppf(1 - confidence_level)

        cf_z = (z +
                (z**2 - 1) * skewness / 6 +
                (z**3 - 3*z) * kurtosis / 24 -
                (2*z**3 - 5*z) * skewness**2 / 36)

        var = abs(mean_return + cf_z * std_return)
        var *= np.sqrt(holding_period)

        cvar = var * 1.2

        return var, cvar

    def _calculate_ewma_var(
        self,
        positions: list[dict],
        returns_data: dict[str, list[float]],
        confidence_level: float,
        holding_period: int,
    ) -> tuple[float, float]:
        """Calculate EWMA (Exponentially Weighted) VaR."""
        from scipy import stats

        portfolio_returns = self._calculate_portfolio_returns(positions, returns_data)

        if len(portfolio_returns) < self.config.min_data_points:
            return 0.0, 0.0

        decay = self.config.ewma_decay_factor
        squared_returns = [r**2 for r in portfolio_returns]

        ewma_variance = squared_returns[0]
        for r2 in squared_returns[1:]:
            ewma_variance = decay * ewma_variance + (1 - decay) * r2

        ewma_std = np.sqrt(ewma_variance)
        mean_return = np.mean(portfolio_returns)

        z_score = stats.norm.ppf(1 - confidence_level)

        var = abs(mean_return + z_score * ewma_std)
        var *= np.sqrt(holding_period)

        pdf_at_z = stats.norm.pdf(z_score)
        cvar = abs(mean_return + ewma_std * pdf_at_z / (1 - confidence_level))
        cvar *= np.sqrt(holding_period)

        return var, cvar

    def _calculate_component_var(
        self,
        positions: list[dict],
        returns_data: dict[str, list[float]],
        confidence_level: float,
        portfolio_value: float,
    ) -> dict[str, float]:
        """Calculate component VaR for each position."""
        component_var: dict[str, float] = {}

        total_value = sum(abs(p.get("market_value", 0)) for p in positions)
        if total_value == 0:
            return component_var

        portfolio_var, _ = self._calculate_historical_var(
            positions, returns_data, confidence_level, 1
        )

        for position in positions:
            symbol = position.get("symbol", "")
            market_value = abs(position.get("market_value", 0))

            weight = market_value / total_value

            component_var[symbol] = portfolio_var * weight * portfolio_value

        return component_var

    def _calculate_marginal_var(
        self,
        positions: list[dict],
        returns_data: dict[str, list[float]],
        confidence_level: float,
        portfolio_value: float,
    ) -> dict[str, float]:
        """Calculate marginal VaR for each position."""
        marginal_var: dict[str, float] = {}

        base_var, _ = self._calculate_historical_var(
            positions, returns_data, confidence_level, 1
        )

        for i, position in enumerate(positions):
            symbol = position.get("symbol", "")

            reduced_positions = positions[:i] + positions[i+1:]

            if reduced_positions:
                reduced_var, _ = self._calculate_historical_var(
                    reduced_positions, returns_data, confidence_level, 1
                )
            else:
                reduced_var = 0

            marginal_var[symbol] = (base_var - reduced_var) * portfolio_value

        return marginal_var

    def calculate_stress_var(
        self,
        portfolio_value: float,
        positions: list[dict],
        returns_data: dict[str, list[float]],
        stress_scenario: Optional[dict] = None,
    ) -> VaRResult:
        """
        Calculate stressed VaR.

        Args:
            portfolio_value: Total portfolio value
            positions: List of position dictionaries
            returns_data: Historical returns
            stress_scenario: Optional stress scenario

        Returns:
            Stressed VaR result
        """
        result = self.calculate_var(
            portfolio_value, positions, returns_data,
            method=VaRMethod.HISTORICAL,
            confidence_level=0.99,
            holding_period=1,
        )

        stress_multiplier = self.config.stress_test_multiplier

        result.var_absolute *= stress_multiplier
        result.var_percentage *= stress_multiplier
        result.cvar_absolute *= stress_multiplier
        result.cvar_percentage *= stress_multiplier

        result.metadata["stressed"] = True
        result.metadata["stress_multiplier"] = stress_multiplier

        return result

    def calculate_incremental_var(
        self,
        portfolio_value: float,
        positions: list[dict],
        returns_data: dict[str, list[float]],
        new_position: dict,
        new_position_returns: list[float],
    ) -> tuple[float, float]:
        """
        Calculate incremental VaR for adding a new position.

        Args:
            portfolio_value: Current portfolio value
            positions: Current positions
            returns_data: Historical returns
            new_position: New position to add
            new_position_returns: Returns for new position

        Returns:
            Tuple of (current_var, new_var)
        """
        current_var, _ = self._calculate_historical_var(
            positions, returns_data, 0.95, 1
        )
        current_var_abs = current_var * portfolio_value

        new_symbol = new_position.get("symbol", "new")
        new_value = new_position.get("market_value", 0)

        extended_positions = positions + [new_position]
        extended_returns = returns_data.copy()
        extended_returns[new_symbol] = new_position_returns

        new_portfolio_value = portfolio_value + new_value

        new_var, _ = self._calculate_historical_var(
            extended_positions, extended_returns, 0.95, 1
        )
        new_var_abs = new_var * new_portfolio_value

        return current_var_abs, new_var_abs

    def get_var_history(
        self,
        method: Optional[VaRMethod] = None,
        limit: int = 100,
    ) -> list[VaRResult]:
        """Get VaR calculation history."""
        if method:
            history = [r for r in self._var_history if r.method == method]
        else:
            history = self._var_history

        return history[-limit:]

    def get_var_statistics(self) -> dict:
        """Get VaR calculator statistics."""
        if not self._var_history:
            return {"calculations": 0}

        recent = self._var_history[-100:]
        avg_var = np.mean([r.var_percentage for r in recent])
        avg_cvar = np.mean([r.cvar_percentage for r in recent])

        return {
            "total_calculations": len(self._var_history),
            "avg_var_pct": avg_var,
            "avg_cvar_pct": avg_cvar,
            "methods_used": list(set(r.method.value for r in recent)),
            "symbols_tracked": len(self._returns_history),
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"VaRCalculator(history={len(self._var_history)})"
