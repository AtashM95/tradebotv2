"""
Walk-Forward Analysis for Strategy Optimization.

This module provides walk-forward analysis capabilities for validating
trading strategies and avoiding overfitting through rolling optimization.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WalkForwardMethod(str, Enum):
    """Walk-forward analysis methods."""

    ANCHORED = "anchored"
    ROLLING = "rolling"
    EXPANDING = "expanding"
    COMBINATORIAL = "combinatorial"


class OptimizationObjective(str, Enum):
    """Optimization objectives."""

    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    TOTAL_RETURN = "total_return"
    PROFIT_FACTOR = "profit_factor"
    WIN_RATE = "win_rate"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"
    CUSTOM = "custom"


class WalkForwardConfig(BaseModel):
    """Configuration for walk-forward analysis."""

    method: WalkForwardMethod = Field(
        default=WalkForwardMethod.ROLLING,
        description="Walk-forward method",
    )
    in_sample_periods: int = Field(
        default=252,
        description="Number of periods for in-sample optimization",
    )
    out_sample_periods: int = Field(
        default=63,
        description="Number of periods for out-of-sample testing",
    )
    step_size: int = Field(
        default=21,
        description="Number of periods to step forward",
    )
    min_in_sample_periods: int = Field(
        default=126,
        description="Minimum in-sample periods required",
    )
    objective: OptimizationObjective = Field(
        default=OptimizationObjective.SHARPE_RATIO,
        description="Optimization objective",
    )
    risk_free_rate: float = Field(
        default=0.0,
        description="Risk-free rate for ratio calculations",
    )
    annualization_factor: float = Field(
        default=252.0,
        description="Trading days per year",
    )
    parallel_optimization: bool = Field(
        default=True,
        description="Run optimizations in parallel",
    )
    max_workers: int = Field(
        default=4,
        description="Maximum parallel workers",
    )
    optimization_iterations: int = Field(
        default=100,
        description="Iterations per optimization",
    )
    validation_threshold: float = Field(
        default=0.5,
        description="Minimum efficiency ratio threshold",
    )


@dataclass
class WindowPeriod:
    """Represents a single walk-forward window."""

    window_id: int
    in_sample_start: datetime
    in_sample_end: datetime
    out_sample_start: datetime
    out_sample_end: datetime
    in_sample_periods: int
    out_sample_periods: int


@dataclass
class OptimizationResult:
    """Result from parameter optimization."""

    best_parameters: dict[str, Any]
    objective_value: float
    all_results: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    optimization_time: float = 0.0


@dataclass
class WindowResult:
    """Result from a single walk-forward window."""

    window: WindowPeriod
    optimization_result: OptimizationResult

    in_sample_return: float = 0.0
    in_sample_sharpe: float = 0.0
    in_sample_max_drawdown: float = 0.0
    in_sample_trades: int = 0

    out_sample_return: float = 0.0
    out_sample_sharpe: float = 0.0
    out_sample_max_drawdown: float = 0.0
    out_sample_trades: int = 0

    efficiency_ratio: float = 0.0
    degradation: float = 0.0

    is_valid: bool = True
    validation_notes: list[str] = field(default_factory=list)


@dataclass
class WalkForwardResult:
    """Complete walk-forward analysis result."""

    config: WalkForwardConfig
    windows: list[WindowResult]

    total_windows: int = 0
    valid_windows: int = 0

    combined_out_sample_return: float = 0.0
    combined_out_sample_sharpe: float = 0.0
    combined_out_sample_max_drawdown: float = 0.0

    average_efficiency_ratio: float = 0.0
    efficiency_ratio_std: float = 0.0

    average_degradation: float = 0.0

    robustness_score: float = 0.0

    optimal_parameters_stability: dict[str, float] = field(default_factory=dict)

    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    returns: np.ndarray = field(default_factory=lambda: np.array([]))

    analysis_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class ParameterSpace:
    """Defines parameter search space for optimization."""

    def __init__(self) -> None:
        """Initialize parameter space."""
        self.parameters: dict[str, dict[str, Any]] = {}

    def add_integer(
        self,
        name: str,
        min_value: int,
        max_value: int,
        step: int = 1,
    ) -> "ParameterSpace":
        """Add integer parameter."""
        self.parameters[name] = {
            "type": "integer",
            "min": min_value,
            "max": max_value,
            "step": step,
        }
        return self

    def add_float(
        self,
        name: str,
        min_value: float,
        max_value: float,
        step: float | None = None,
    ) -> "ParameterSpace":
        """Add float parameter."""
        self.parameters[name] = {
            "type": "float",
            "min": min_value,
            "max": max_value,
            "step": step,
        }
        return self

    def add_categorical(
        self,
        name: str,
        choices: list[Any],
    ) -> "ParameterSpace":
        """Add categorical parameter."""
        self.parameters[name] = {
            "type": "categorical",
            "choices": choices,
        }
        return self

    def sample_random(self) -> dict[str, Any]:
        """Sample random parameter combination."""
        params = {}

        for name, spec in self.parameters.items():
            if spec["type"] == "integer":
                if spec.get("step"):
                    values = range(spec["min"], spec["max"] + 1, spec["step"])
                    params[name] = int(np.random.choice(list(values)))
                else:
                    params[name] = np.random.randint(spec["min"], spec["max"] + 1)
            elif spec["type"] == "float":
                if spec.get("step"):
                    num_steps = int((spec["max"] - spec["min"]) / spec["step"]) + 1
                    idx = np.random.randint(0, num_steps)
                    params[name] = spec["min"] + idx * spec["step"]
                else:
                    params[name] = np.random.uniform(spec["min"], spec["max"])
            elif spec["type"] == "categorical":
                params[name] = np.random.choice(spec["choices"])

        return params

    def get_grid(self) -> list[dict[str, Any]]:
        """Generate full parameter grid."""
        from itertools import product

        param_values = {}

        for name, spec in self.parameters.items():
            if spec["type"] == "integer":
                step = spec.get("step", 1)
                param_values[name] = list(range(spec["min"], spec["max"] + 1, step))
            elif spec["type"] == "float":
                if spec.get("step"):
                    num_steps = int((spec["max"] - spec["min"]) / spec["step"]) + 1
                    param_values[name] = [
                        spec["min"] + i * spec["step"] for i in range(num_steps)
                    ]
                else:
                    param_values[name] = np.linspace(spec["min"], spec["max"], 10).tolist()
            elif spec["type"] == "categorical":
                param_values[name] = spec["choices"]

        keys = list(param_values.keys())
        values = [param_values[k] for k in keys]

        grid = []
        for combo in product(*values):
            grid.append(dict(zip(keys, combo)))

        return grid


class WalkForwardAnalyzer:
    """Walk-forward analysis engine."""

    def __init__(
        self,
        config: WalkForwardConfig | None = None,
    ) -> None:
        """
        Initialize walk-forward analyzer.

        Args:
            config: Analysis configuration
        """
        self.config = config or WalkForwardConfig()

        logger.info(
            f"WalkForwardAnalyzer initialized with {self.config.method.value} method, "
            f"IS: {self.config.in_sample_periods}, OOS: {self.config.out_sample_periods}"
        )

    async def run_analysis(
        self,
        data: pd.DataFrame,
        parameter_space: ParameterSpace,
        strategy_func: Callable[[pd.DataFrame, dict[str, Any]], pd.Series],
        objective_func: Callable[[pd.Series], float] | None = None,
    ) -> WalkForwardResult:
        """
        Run complete walk-forward analysis.

        Args:
            data: Historical market data
            parameter_space: Parameter search space
            strategy_func: Strategy function that takes data and parameters,
                          returns signal/return series
            objective_func: Optional custom objective function

        Returns:
            Complete walk-forward analysis result
        """
        start_time = datetime.now()

        windows = self._generate_windows(data)
        logger.info(f"Generated {len(windows)} walk-forward windows")

        if self.config.parallel_optimization:
            window_results = await self._run_parallel_windows(
                windows, data, parameter_space, strategy_func, objective_func
            )
        else:
            window_results = await self._run_sequential_windows(
                windows, data, parameter_space, strategy_func, objective_func
            )

        result = self._aggregate_results(window_results)
        result.analysis_time = (datetime.now() - start_time).total_seconds()

        return result

    def _generate_windows(self, data: pd.DataFrame) -> list[WindowPeriod]:
        """Generate walk-forward windows based on configuration."""
        windows = []

        n_periods = len(data)
        dates = data.index if isinstance(data.index, pd.DatetimeIndex) else None

        if self.config.method == WalkForwardMethod.ANCHORED:
            windows = self._generate_anchored_windows(n_periods, dates)
        elif self.config.method == WalkForwardMethod.ROLLING:
            windows = self._generate_rolling_windows(n_periods, dates)
        elif self.config.method == WalkForwardMethod.EXPANDING:
            windows = self._generate_expanding_windows(n_periods, dates)
        else:
            windows = self._generate_rolling_windows(n_periods, dates)

        return windows

    def _generate_anchored_windows(
        self,
        n_periods: int,
        dates: pd.DatetimeIndex | None,
    ) -> list[WindowPeriod]:
        """Generate anchored walk-forward windows."""
        windows = []
        window_id = 0

        is_start = 0
        is_end = self.config.in_sample_periods - 1

        while is_end + self.config.out_sample_periods <= n_periods:
            oos_start = is_end + 1
            oos_end = min(oos_start + self.config.out_sample_periods - 1, n_periods - 1)

            window = WindowPeriod(
                window_id=window_id,
                in_sample_start=dates[is_start] if dates is not None else datetime.now(),
                in_sample_end=dates[is_end] if dates is not None else datetime.now(),
                out_sample_start=dates[oos_start] if dates is not None else datetime.now(),
                out_sample_end=dates[oos_end] if dates is not None else datetime.now(),
                in_sample_periods=is_end - is_start + 1,
                out_sample_periods=oos_end - oos_start + 1,
            )
            windows.append(window)

            is_end += self.config.step_size
            window_id += 1

        return windows

    def _generate_rolling_windows(
        self,
        n_periods: int,
        dates: pd.DatetimeIndex | None,
    ) -> list[WindowPeriod]:
        """Generate rolling walk-forward windows."""
        windows = []
        window_id = 0

        is_start = 0

        while is_start + self.config.in_sample_periods + self.config.out_sample_periods <= n_periods:
            is_end = is_start + self.config.in_sample_periods - 1
            oos_start = is_end + 1
            oos_end = oos_start + self.config.out_sample_periods - 1

            window = WindowPeriod(
                window_id=window_id,
                in_sample_start=dates[is_start] if dates is not None else datetime.now(),
                in_sample_end=dates[is_end] if dates is not None else datetime.now(),
                out_sample_start=dates[oos_start] if dates is not None else datetime.now(),
                out_sample_end=dates[oos_end] if dates is not None else datetime.now(),
                in_sample_periods=self.config.in_sample_periods,
                out_sample_periods=self.config.out_sample_periods,
            )
            windows.append(window)

            is_start += self.config.step_size
            window_id += 1

        return windows

    def _generate_expanding_windows(
        self,
        n_periods: int,
        dates: pd.DatetimeIndex | None,
    ) -> list[WindowPeriod]:
        """Generate expanding walk-forward windows."""
        windows = []
        window_id = 0

        is_start = 0
        is_end = self.config.min_in_sample_periods - 1

        while is_end + self.config.out_sample_periods <= n_periods:
            oos_start = is_end + 1
            oos_end = min(oos_start + self.config.out_sample_periods - 1, n_periods - 1)

            window = WindowPeriod(
                window_id=window_id,
                in_sample_start=dates[is_start] if dates is not None else datetime.now(),
                in_sample_end=dates[is_end] if dates is not None else datetime.now(),
                out_sample_start=dates[oos_start] if dates is not None else datetime.now(),
                out_sample_end=dates[oos_end] if dates is not None else datetime.now(),
                in_sample_periods=is_end - is_start + 1,
                out_sample_periods=oos_end - oos_start + 1,
            )
            windows.append(window)

            is_end += self.config.step_size
            window_id += 1

        return windows

    async def _run_parallel_windows(
        self,
        windows: list[WindowPeriod],
        data: pd.DataFrame,
        parameter_space: ParameterSpace,
        strategy_func: Callable,
        objective_func: Callable | None,
    ) -> list[WindowResult]:
        """Run window optimizations in parallel."""
        semaphore = asyncio.Semaphore(self.config.max_workers)

        async def run_with_semaphore(window: WindowPeriod) -> WindowResult:
            async with semaphore:
                return await self._run_single_window(
                    window, data, parameter_space, strategy_func, objective_func
                )

        tasks = [run_with_semaphore(w) for w in windows]
        results = await asyncio.gather(*tasks)

        return list(results)

    async def _run_sequential_windows(
        self,
        windows: list[WindowPeriod],
        data: pd.DataFrame,
        parameter_space: ParameterSpace,
        strategy_func: Callable,
        objective_func: Callable | None,
    ) -> list[WindowResult]:
        """Run window optimizations sequentially."""
        results = []

        for window in windows:
            result = await self._run_single_window(
                window, data, parameter_space, strategy_func, objective_func
            )
            results.append(result)

        return results

    async def _run_single_window(
        self,
        window: WindowPeriod,
        data: pd.DataFrame,
        parameter_space: ParameterSpace,
        strategy_func: Callable,
        objective_func: Callable | None,
    ) -> WindowResult:
        """Run optimization for a single window."""
        is_data = data.loc[window.in_sample_start : window.in_sample_end]
        oos_data = data.loc[window.out_sample_start : window.out_sample_end]

        optimization_result = await self._optimize_parameters(
            is_data, parameter_space, strategy_func, objective_func
        )

        is_returns = strategy_func(is_data, optimization_result.best_parameters)
        is_metrics = self._calculate_metrics(is_returns)

        oos_returns = strategy_func(oos_data, optimization_result.best_parameters)
        oos_metrics = self._calculate_metrics(oos_returns)

        efficiency_ratio = 0.0
        if is_metrics["sharpe_ratio"] != 0:
            efficiency_ratio = oos_metrics["sharpe_ratio"] / is_metrics["sharpe_ratio"]

        degradation = is_metrics["sharpe_ratio"] - oos_metrics["sharpe_ratio"]

        is_valid = True
        validation_notes = []

        if efficiency_ratio < self.config.validation_threshold:
            is_valid = False
            validation_notes.append(
                f"Low efficiency ratio: {efficiency_ratio:.2f}"
            )

        if oos_metrics["total_return"] < 0 and is_metrics["total_return"] > 0:
            validation_notes.append("Sign reversal in returns")

        return WindowResult(
            window=window,
            optimization_result=optimization_result,
            in_sample_return=is_metrics["total_return"],
            in_sample_sharpe=is_metrics["sharpe_ratio"],
            in_sample_max_drawdown=is_metrics["max_drawdown"],
            in_sample_trades=is_metrics["num_trades"],
            out_sample_return=oos_metrics["total_return"],
            out_sample_sharpe=oos_metrics["sharpe_ratio"],
            out_sample_max_drawdown=oos_metrics["max_drawdown"],
            out_sample_trades=oos_metrics["num_trades"],
            efficiency_ratio=efficiency_ratio,
            degradation=degradation,
            is_valid=is_valid,
            validation_notes=validation_notes,
        )

    async def _optimize_parameters(
        self,
        data: pd.DataFrame,
        parameter_space: ParameterSpace,
        strategy_func: Callable,
        objective_func: Callable | None,
    ) -> OptimizationResult:
        """Optimize parameters on in-sample data."""
        start_time = datetime.now()

        all_results = []
        best_params = {}
        best_objective = float("-inf")

        for _ in range(self.config.optimization_iterations):
            params = parameter_space.sample_random()

            try:
                returns = strategy_func(data, params)

                if objective_func:
                    objective_value = objective_func(returns)
                else:
                    objective_value = self._default_objective(returns)

                all_results.append({
                    "parameters": params,
                    "objective": objective_value,
                })

                if objective_value > best_objective:
                    best_objective = objective_value
                    best_params = params

            except Exception as e:
                logger.warning(f"Optimization iteration failed: {e}")
                continue

        elapsed = (datetime.now() - start_time).total_seconds()

        return OptimizationResult(
            best_parameters=best_params,
            objective_value=best_objective,
            all_results=all_results,
            iterations=len(all_results),
            optimization_time=elapsed,
        )

    def _default_objective(self, returns: pd.Series) -> float:
        """Calculate default objective value."""
        if self.config.objective == OptimizationObjective.SHARPE_RATIO:
            return self._calculate_sharpe(returns)
        elif self.config.objective == OptimizationObjective.SORTINO_RATIO:
            return self._calculate_sortino(returns)
        elif self.config.objective == OptimizationObjective.CALMAR_RATIO:
            return self._calculate_calmar(returns)
        elif self.config.objective == OptimizationObjective.TOTAL_RETURN:
            return float(np.prod(1 + returns) - 1)
        elif self.config.objective == OptimizationObjective.PROFIT_FACTOR:
            return self._calculate_profit_factor(returns)
        elif self.config.objective == OptimizationObjective.WIN_RATE:
            return float(np.mean(returns > 0))
        else:
            return self._calculate_sharpe(returns)

    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        mean_return = returns.mean() * self.config.annualization_factor
        std_return = returns.std() * np.sqrt(self.config.annualization_factor)

        return float((mean_return - self.config.risk_free_rate) / std_return)

    def _calculate_sortino(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        if len(returns) == 0:
            return 0.0

        mean_return = returns.mean() * self.config.annualization_factor
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0:
            return float("inf") if mean_return > 0 else 0.0

        downside_std = downside_returns.std() * np.sqrt(self.config.annualization_factor)

        if downside_std == 0:
            return 0.0

        return float((mean_return - self.config.risk_free_rate) / downside_std)

    def _calculate_calmar(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio."""
        if len(returns) == 0:
            return 0.0

        total_return = np.prod(1 + returns) - 1
        annual_return = (1 + total_return) ** (
            self.config.annualization_factor / len(returns)
        ) - 1

        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - running_max) / running_max
        max_drawdown = abs(drawdowns.min())

        if max_drawdown == 0:
            return float("inf") if annual_return > 0 else 0.0

        return float(annual_return / max_drawdown)

    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor."""
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0

        return float(gross_profit / gross_loss)

    def _calculate_metrics(self, returns: pd.Series) -> dict[str, float]:
        """Calculate comprehensive metrics for a return series."""
        if len(returns) == 0:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "volatility": 0.0,
                "num_trades": 0,
            }

        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - running_max) / running_max

        signal_changes = (returns != 0).astype(int).diff().abs()
        num_trades = int(signal_changes.sum() / 2) if len(signal_changes) > 0 else 0

        return {
            "total_return": float(np.prod(1 + returns) - 1),
            "sharpe_ratio": self._calculate_sharpe(returns),
            "sortino_ratio": self._calculate_sortino(returns),
            "max_drawdown": float(drawdowns.min()),
            "volatility": float(returns.std() * np.sqrt(self.config.annualization_factor)),
            "num_trades": num_trades,
        }

    def _aggregate_results(
        self,
        window_results: list[WindowResult],
    ) -> WalkForwardResult:
        """Aggregate results from all windows."""
        valid_results = [r for r in window_results if r.is_valid]

        all_oos_returns = []
        for wr in window_results:
            if hasattr(wr, "oos_returns") and wr.oos_returns is not None:
                all_oos_returns.extend(wr.oos_returns)

        if not all_oos_returns:
            all_oos_returns = [wr.out_sample_return for wr in window_results]

        combined_return = float(np.prod([1 + r for r in all_oos_returns]) - 1)

        efficiency_ratios = [wr.efficiency_ratio for wr in window_results]
        avg_efficiency = float(np.mean(efficiency_ratios)) if efficiency_ratios else 0.0
        std_efficiency = float(np.std(efficiency_ratios)) if efficiency_ratios else 0.0

        degradations = [wr.degradation for wr in window_results]
        avg_degradation = float(np.mean(degradations)) if degradations else 0.0

        robustness_score = self._calculate_robustness_score(window_results)

        param_stability = self._analyze_parameter_stability(window_results)

        return WalkForwardResult(
            config=self.config,
            windows=window_results,
            total_windows=len(window_results),
            valid_windows=len(valid_results),
            combined_out_sample_return=combined_return,
            combined_out_sample_sharpe=float(
                np.mean([wr.out_sample_sharpe for wr in valid_results])
            ) if valid_results else 0.0,
            combined_out_sample_max_drawdown=float(
                np.min([wr.out_sample_max_drawdown for wr in window_results])
            ) if window_results else 0.0,
            average_efficiency_ratio=avg_efficiency,
            efficiency_ratio_std=std_efficiency,
            average_degradation=avg_degradation,
            robustness_score=robustness_score,
            optimal_parameters_stability=param_stability,
        )

    def _calculate_robustness_score(
        self,
        window_results: list[WindowResult],
    ) -> float:
        """Calculate overall robustness score."""
        if not window_results:
            return 0.0

        valid_ratio = len([r for r in window_results if r.is_valid]) / len(window_results)

        efficiency_ratios = [r.efficiency_ratio for r in window_results]
        avg_efficiency = np.mean(efficiency_ratios) if efficiency_ratios else 0.0
        efficiency_consistency = 1.0 - min(1.0, np.std(efficiency_ratios)) if efficiency_ratios else 0.0

        positive_oos = len([r for r in window_results if r.out_sample_return > 0])
        profit_consistency = positive_oos / len(window_results)

        robustness = (
            0.3 * valid_ratio
            + 0.3 * min(1.0, max(0.0, avg_efficiency))
            + 0.2 * efficiency_consistency
            + 0.2 * profit_consistency
        )

        return float(robustness)

    def _analyze_parameter_stability(
        self,
        window_results: list[WindowResult],
    ) -> dict[str, float]:
        """Analyze stability of optimal parameters across windows."""
        if not window_results:
            return {}

        param_values: dict[str, list[Any]] = {}

        for wr in window_results:
            for param, value in wr.optimization_result.best_parameters.items():
                if param not in param_values:
                    param_values[param] = []
                param_values[param].append(value)

        stability = {}

        for param, values in param_values.items():
            if len(values) < 2:
                stability[param] = 1.0
                continue

            try:
                numeric_values = [float(v) for v in values]
                mean_val = np.mean(numeric_values)
                if mean_val != 0:
                    cv = np.std(numeric_values) / abs(mean_val)
                    stability[param] = float(max(0.0, 1.0 - cv))
                else:
                    stability[param] = 1.0 if np.std(numeric_values) == 0 else 0.0
            except (ValueError, TypeError):
                unique_count = len(set(values))
                stability[param] = 1.0 / unique_count if unique_count > 0 else 0.0

        return stability


def create_walk_forward_analyzer(
    method: str = "rolling",
    in_sample_periods: int = 252,
    out_sample_periods: int = 63,
    config: dict | None = None,
) -> WalkForwardAnalyzer:
    """
    Create a walk-forward analyzer.

    Args:
        method: Walk-forward method
        in_sample_periods: In-sample period length
        out_sample_periods: Out-of-sample period length
        config: Additional configuration

    Returns:
        Configured WalkForwardAnalyzer
    """
    wf_config = WalkForwardConfig(
        method=WalkForwardMethod(method),
        in_sample_periods=in_sample_periods,
        out_sample_periods=out_sample_periods,
        **(config or {}),
    )
    return WalkForwardAnalyzer(config=wf_config)
