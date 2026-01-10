"""
Objective Functions for Strategy Optimization.

This module provides various objective functions and fitness
calculators for trading strategy optimization.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ObjectiveType(str, Enum):
    """Types of optimization objectives."""

    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    OMEGA_RATIO = "omega_ratio"
    TOTAL_RETURN = "total_return"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"
    PROFIT_FACTOR = "profit_factor"
    WIN_RATE = "win_rate"
    EXPECTANCY = "expectancy"
    INFORMATION_RATIO = "information_ratio"
    TREYNOR_RATIO = "treynor_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    CAGR = "cagr"
    MAR_RATIO = "mar_ratio"
    ULCER_INDEX = "ulcer_index"
    CUSTOM = "custom"


class RiskMeasure(str, Enum):
    """Risk measures for objectives."""

    VOLATILITY = "volatility"
    DOWNSIDE_VOLATILITY = "downside_volatility"
    MAX_DRAWDOWN = "max_drawdown"
    VAR = "var"
    CVAR = "cvar"
    ULCER_INDEX = "ulcer_index"


class ObjectiveFunctionConfig(BaseModel):
    """Configuration for objective functions."""

    primary_objective: ObjectiveType = Field(
        default=ObjectiveType.SHARPE_RATIO,
        description="Primary optimization objective",
    )
    risk_measure: RiskMeasure = Field(
        default=RiskMeasure.VOLATILITY,
        description="Risk measure to use",
    )
    risk_free_rate: float = Field(default=0.0, description="Risk-free rate")
    target_return: float = Field(default=0.0, description="Target return for Sortino")
    mar: float = Field(default=0.0, description="Minimum acceptable return")
    annualization_factor: float = Field(default=252.0, description="Trading days per year")
    var_confidence: float = Field(default=0.95, description="VaR confidence level")
    penalty_factor: float = Field(default=10.0, description="Constraint violation penalty")
    min_trades: int = Field(default=10, description="Minimum trades for valid objective")


@dataclass
class ObjectiveResult:
    """Result from objective function evaluation."""

    primary_value: float
    all_metrics: dict[str, float] = field(default_factory=dict)
    is_valid: bool = True
    penalty: float = 0.0
    final_value: float = 0.0


class PerformanceMetrics:
    """Calculator for trading performance metrics."""

    def __init__(
        self,
        config: ObjectiveFunctionConfig | None = None,
    ) -> None:
        """
        Initialize performance metrics calculator.

        Args:
            config: Configuration for calculations
        """
        self.config = config or ObjectiveFunctionConfig()

    def calculate_all(
        self,
        returns: np.ndarray | pd.Series,
        trades: list[dict[str, Any]] | None = None,
        benchmark_returns: np.ndarray | pd.Series | None = None,
    ) -> dict[str, float]:
        """
        Calculate all performance metrics.

        Args:
            returns: Return series
            trades: Optional list of trade records
            benchmark_returns: Optional benchmark returns

        Returns:
            Dictionary of all metrics
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns[~np.isnan(returns)]

        if len(returns) == 0:
            return self._get_zero_metrics()

        metrics = {
            "total_return": self.total_return(returns),
            "cagr": self.cagr(returns),
            "volatility": self.volatility(returns),
            "sharpe_ratio": self.sharpe_ratio(returns),
            "sortino_ratio": self.sortino_ratio(returns),
            "calmar_ratio": self.calmar_ratio(returns),
            "max_drawdown": self.max_drawdown(returns),
            "omega_ratio": self.omega_ratio(returns),
            "ulcer_index": self.ulcer_index(returns),
            "var_95": self.value_at_risk(returns, 0.95),
            "cvar_95": self.conditional_var(returns, 0.95),
            "skewness": float(self._skewness(returns)),
            "kurtosis": float(self._kurtosis(returns)),
        }

        if trades:
            trade_metrics = self.calculate_trade_metrics(trades)
            metrics.update(trade_metrics)

        if benchmark_returns is not None:
            benchmark_metrics = self.calculate_benchmark_metrics(
                returns, benchmark_returns
            )
            metrics.update(benchmark_metrics)

        return metrics

    def _get_zero_metrics(self) -> dict[str, float]:
        """Return zero metrics for empty returns."""
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,
            "omega_ratio": 1.0,
            "ulcer_index": 0.0,
            "var_95": 0.0,
            "cvar_95": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
        }

    def total_return(self, returns: np.ndarray) -> float:
        """Calculate total return."""
        return float(np.prod(1 + returns) - 1)

    def cagr(self, returns: np.ndarray) -> float:
        """Calculate Compound Annual Growth Rate."""
        if len(returns) == 0:
            return 0.0

        total = np.prod(1 + returns)
        years = len(returns) / self.config.annualization_factor

        if years <= 0 or total <= 0:
            return 0.0

        return float(total ** (1 / years) - 1)

    def volatility(self, returns: np.ndarray) -> float:
        """Calculate annualized volatility."""
        if len(returns) < 2:
            return 0.0

        return float(np.std(returns) * np.sqrt(self.config.annualization_factor))

    def downside_volatility(
        self,
        returns: np.ndarray,
        target: float | None = None,
    ) -> float:
        """Calculate downside volatility."""
        if target is None:
            target = self.config.target_return / self.config.annualization_factor

        downside = returns[returns < target] - target
        if len(downside) == 0:
            return 0.0

        return float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(self.config.annualization_factor))

    def sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe Ratio."""
        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns) * self.config.annualization_factor
        vol = self.volatility(returns)

        if vol == 0:
            return 0.0

        return float((mean_return - self.config.risk_free_rate) / vol)

    def sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino Ratio."""
        if len(returns) < 2:
            return 0.0

        mean_return = np.mean(returns) * self.config.annualization_factor
        down_vol = self.downside_volatility(returns)

        if down_vol == 0:
            return float("inf") if mean_return > self.config.target_return else 0.0

        return float((mean_return - self.config.target_return) / down_vol)

    def calmar_ratio(self, returns: np.ndarray) -> float:
        """Calculate Calmar Ratio (CAGR / Max Drawdown)."""
        cagr_val = self.cagr(returns)
        mdd = abs(self.max_drawdown(returns))

        if mdd == 0:
            return float("inf") if cagr_val > 0 else 0.0

        return float(cagr_val / mdd)

    def max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate Maximum Drawdown."""
        if len(returns) == 0:
            return 0.0

        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - running_max) / running_max

        return float(np.min(drawdowns))

    def omega_ratio(
        self,
        returns: np.ndarray,
        threshold: float | None = None,
    ) -> float:
        """Calculate Omega Ratio."""
        if threshold is None:
            threshold = self.config.mar / self.config.annualization_factor

        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]

        if len(losses) == 0 or np.sum(losses) == 0:
            return float("inf") if len(gains) > 0 else 1.0

        return float(np.sum(gains) / np.sum(losses))

    def ulcer_index(self, returns: np.ndarray) -> float:
        """Calculate Ulcer Index."""
        if len(returns) == 0:
            return 0.0

        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = ((cum_returns - running_max) / running_max) * 100

        return float(np.sqrt(np.mean(drawdowns ** 2)))

    def value_at_risk(
        self,
        returns: np.ndarray,
        confidence: float = 0.95,
    ) -> float:
        """Calculate Value at Risk."""
        if len(returns) == 0:
            return 0.0

        return float(np.percentile(returns, (1 - confidence) * 100))

    def conditional_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.95,
    ) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        var = self.value_at_risk(returns, confidence)
        tail = returns[returns <= var]

        if len(tail) == 0:
            return var

        return float(np.mean(tail))

    def _skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness."""
        if len(returns) < 3:
            return 0.0

        mean = np.mean(returns)
        std = np.std(returns)

        if std == 0:
            return 0.0

        return float(np.mean(((returns - mean) / std) ** 3))

    def _kurtosis(self, returns: np.ndarray) -> float:
        """Calculate excess kurtosis."""
        if len(returns) < 4:
            return 0.0

        mean = np.mean(returns)
        std = np.std(returns)

        if std == 0:
            return 0.0

        return float(np.mean(((returns - mean) / std) ** 4) - 3)

    def calculate_trade_metrics(
        self,
        trades: list[dict[str, Any]],
    ) -> dict[str, float]:
        """
        Calculate trade-based metrics.

        Args:
            trades: List of trade records

        Returns:
            Trade metrics dictionary
        """
        if not trades:
            return {
                "num_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_trade": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "expectancy": 0.0,
                "max_consecutive_wins": 0,
                "max_consecutive_losses": 0,
            }

        pnls = [t.get("pnl", t.get("profit", 0)) for t in trades]

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / len(pnls) if pnls else 0.0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

        avg_trade = np.mean(pnls) if pnls else 0.0
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0

        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

        max_wins, max_losses = self._consecutive_runs(pnls)

        return {
            "num_trades": len(trades),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "avg_trade": float(avg_trade),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "expectancy": float(expectancy),
            "max_consecutive_wins": max_wins,
            "max_consecutive_losses": max_losses,
        }

    def _consecutive_runs(self, pnls: list[float]) -> tuple[int, int]:
        """Calculate maximum consecutive wins and losses."""
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for pnl in pnls:
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif pnl < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0

        return max_wins, max_losses

    def calculate_benchmark_metrics(
        self,
        returns: np.ndarray,
        benchmark_returns: np.ndarray | pd.Series,
    ) -> dict[str, float]:
        """
        Calculate benchmark-relative metrics.

        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns

        Returns:
            Benchmark metrics dictionary
        """
        if isinstance(benchmark_returns, pd.Series):
            benchmark_returns = benchmark_returns.values

        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]

        excess_returns = returns - benchmark_returns

        tracking_error = np.std(excess_returns) * np.sqrt(self.config.annualization_factor)

        information_ratio = 0.0
        if tracking_error > 0:
            information_ratio = np.mean(excess_returns) * self.config.annualization_factor / tracking_error

        if np.var(benchmark_returns) > 0:
            beta = np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
        else:
            beta = 0.0

        alpha = (
            np.mean(returns) * self.config.annualization_factor
            - self.config.risk_free_rate
            - beta * (np.mean(benchmark_returns) * self.config.annualization_factor - self.config.risk_free_rate)
        )

        treynor_ratio = 0.0
        if beta != 0:
            treynor_ratio = (np.mean(returns) * self.config.annualization_factor - self.config.risk_free_rate) / beta

        if len(returns) > 1:
            correlation = np.corrcoef(returns, benchmark_returns)[0, 1]
        else:
            correlation = 0.0

        return {
            "alpha": float(alpha),
            "beta": float(beta),
            "information_ratio": float(information_ratio),
            "treynor_ratio": float(treynor_ratio),
            "tracking_error": float(tracking_error),
            "correlation": float(correlation) if not np.isnan(correlation) else 0.0,
        }


class ObjectiveFunction:
    """Configurable objective function for optimization."""

    def __init__(
        self,
        config: ObjectiveFunctionConfig | None = None,
        custom_func: Callable[[np.ndarray], float] | None = None,
    ) -> None:
        """
        Initialize objective function.

        Args:
            config: Configuration
            custom_func: Optional custom objective function
        """
        self.config = config or ObjectiveFunctionConfig()
        self.custom_func = custom_func
        self.metrics_calculator = PerformanceMetrics(self.config)

    def __call__(
        self,
        returns: np.ndarray | pd.Series,
        trades: list[dict[str, Any]] | None = None,
        constraints: list[Callable[[], float]] | None = None,
    ) -> ObjectiveResult:
        """
        Evaluate objective function.

        Args:
            returns: Return series
            trades: Optional trade records
            constraints: Optional constraint functions

        Returns:
            Objective result
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns[~np.isnan(returns)]

        if len(returns) < self.config.min_trades:
            return ObjectiveResult(
                primary_value=float("-inf"),
                is_valid=False,
                penalty=self.config.penalty_factor,
                final_value=float("-inf"),
            )

        all_metrics = self.metrics_calculator.calculate_all(returns, trades)

        primary_value = self._get_primary_objective(returns, all_metrics)

        penalty = 0.0
        if constraints:
            for constraint in constraints:
                violation = constraint()
                if violation > 0:
                    penalty += violation * self.config.penalty_factor

        final_value = primary_value - penalty

        return ObjectiveResult(
            primary_value=primary_value,
            all_metrics=all_metrics,
            is_valid=True,
            penalty=penalty,
            final_value=final_value,
        )

    def _get_primary_objective(
        self,
        returns: np.ndarray,
        metrics: dict[str, float],
    ) -> float:
        """Get primary objective value."""
        obj_type = self.config.primary_objective

        if obj_type == ObjectiveType.SHARPE_RATIO:
            return metrics["sharpe_ratio"]
        elif obj_type == ObjectiveType.SORTINO_RATIO:
            return metrics["sortino_ratio"]
        elif obj_type == ObjectiveType.CALMAR_RATIO:
            return metrics["calmar_ratio"]
        elif obj_type == ObjectiveType.OMEGA_RATIO:
            return metrics["omega_ratio"]
        elif obj_type == ObjectiveType.TOTAL_RETURN:
            return metrics["total_return"]
        elif obj_type == ObjectiveType.CAGR:
            return metrics["cagr"]
        elif obj_type == ObjectiveType.PROFIT_FACTOR:
            return metrics.get("profit_factor", 0.0)
        elif obj_type == ObjectiveType.WIN_RATE:
            return metrics.get("win_rate", 0.0)
        elif obj_type == ObjectiveType.MAX_DRAWDOWN:
            return -abs(metrics["max_drawdown"])
        elif obj_type == ObjectiveType.ULCER_INDEX:
            return -metrics["ulcer_index"]
        elif obj_type == ObjectiveType.CUSTOM:
            if self.custom_func:
                return self.custom_func(returns)
            return 0.0
        else:
            return metrics["sharpe_ratio"]


class MultiObjectiveFunction:
    """Multi-objective function for Pareto optimization."""

    def __init__(
        self,
        objectives: list[ObjectiveType],
        weights: list[float] | None = None,
        config: ObjectiveFunctionConfig | None = None,
    ) -> None:
        """
        Initialize multi-objective function.

        Args:
            objectives: List of objectives
            weights: Optional weights for scalarization
            config: Configuration
        """
        self.objectives = objectives
        self.weights = weights or [1.0] * len(objectives)
        self.config = config or ObjectiveFunctionConfig()
        self.metrics_calculator = PerformanceMetrics(self.config)

    def __call__(
        self,
        returns: np.ndarray | pd.Series,
        trades: list[dict[str, Any]] | None = None,
    ) -> tuple[list[float], dict[str, float]]:
        """
        Evaluate multi-objective function.

        Args:
            returns: Return series
            trades: Optional trade records

        Returns:
            Tuple of objective values and all metrics
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns[~np.isnan(returns)]

        all_metrics = self.metrics_calculator.calculate_all(returns, trades)

        objective_values = []
        for obj in self.objectives:
            value = self._get_objective_value(obj, all_metrics)
            objective_values.append(value)

        return objective_values, all_metrics

    def scalarize(
        self,
        returns: np.ndarray | pd.Series,
        trades: list[dict[str, Any]] | None = None,
    ) -> float:
        """
        Scalarize multi-objective to single value.

        Args:
            returns: Return series
            trades: Optional trade records

        Returns:
            Scalarized objective value
        """
        objective_values, _ = self(returns, trades)

        return sum(w * v for w, v in zip(self.weights, objective_values))

    def _get_objective_value(
        self,
        obj_type: ObjectiveType,
        metrics: dict[str, float],
    ) -> float:
        """Get specific objective value."""
        mapping = {
            ObjectiveType.SHARPE_RATIO: "sharpe_ratio",
            ObjectiveType.SORTINO_RATIO: "sortino_ratio",
            ObjectiveType.CALMAR_RATIO: "calmar_ratio",
            ObjectiveType.TOTAL_RETURN: "total_return",
            ObjectiveType.CAGR: "cagr",
            ObjectiveType.MAX_DRAWDOWN: "max_drawdown",
            ObjectiveType.PROFIT_FACTOR: "profit_factor",
            ObjectiveType.WIN_RATE: "win_rate",
        }

        metric_name = mapping.get(obj_type, "sharpe_ratio")
        value = metrics.get(metric_name, 0.0)

        if obj_type in [ObjectiveType.MAX_DRAWDOWN, ObjectiveType.ULCER_INDEX]:
            return -abs(value)

        return value


def create_objective_function(
    objective: str = "sharpe_ratio",
    risk_free_rate: float = 0.0,
    target_return: float = 0.0,
    annualization_factor: float = 252.0,
) -> ObjectiveFunction:
    """
    Create an objective function.

    Args:
        objective: Objective type
        risk_free_rate: Risk-free rate
        target_return: Target return
        annualization_factor: Annualization factor

    Returns:
        Configured ObjectiveFunction
    """
    config = ObjectiveFunctionConfig(
        primary_objective=ObjectiveType(objective),
        risk_free_rate=risk_free_rate,
        target_return=target_return,
        annualization_factor=annualization_factor,
    )
    return ObjectiveFunction(config)
