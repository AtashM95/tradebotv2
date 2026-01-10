"""
Monte Carlo Simulation for Backtesting.

This module provides Monte Carlo simulation capabilities for analyzing
strategy robustness, estimating risk metrics, and generating confidence intervals.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SimulationMethod(str, Enum):
    """Monte Carlo simulation methods."""

    BOOTSTRAP = "bootstrap"
    PARAMETRIC = "parametric"
    BLOCK_BOOTSTRAP = "block_bootstrap"
    STATIONARY_BOOTSTRAP = "stationary_bootstrap"
    PATH_PERMUTATION = "path_permutation"
    GEOMETRIC_BROWNIAN = "geometric_brownian"


class ConfidenceLevel(str, Enum):
    """Standard confidence levels."""

    LEVEL_90 = "90%"
    LEVEL_95 = "95%"
    LEVEL_99 = "99%"


class MonteCarloConfig(BaseModel):
    """Configuration for Monte Carlo simulation."""

    num_simulations: int = Field(default=10000, description="Number of simulations")
    method: SimulationMethod = Field(default=SimulationMethod.BOOTSTRAP, description="Simulation method")
    block_size: int = Field(default=20, description="Block size for block bootstrap")
    random_seed: int | None = Field(default=None, description="Random seed for reproducibility")
    confidence_levels: list[float] = Field(
        default=[0.90, 0.95, 0.99],
        description="Confidence levels to calculate",
    )
    parallel_simulations: bool = Field(default=True, description="Run simulations in parallel")
    max_workers: int = Field(default=4, description="Maximum parallel workers")
    preserve_autocorrelation: bool = Field(default=False, description="Preserve return autocorrelation")
    preserve_volatility_clustering: bool = Field(default=False, description="Preserve volatility clustering")
    annualization_factor: float = Field(default=252.0, description="Trading days per year")


@dataclass
class SimulationPath:
    """Single simulation path result."""

    path_id: int
    returns: np.ndarray
    cumulative_returns: np.ndarray
    equity_curve: np.ndarray
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    final_equity: float


@dataclass
class MonteCarloResult:
    """Complete Monte Carlo simulation result."""

    num_simulations: int
    method: SimulationMethod
    original_returns: np.ndarray
    simulated_total_returns: np.ndarray
    simulated_sharpe_ratios: np.ndarray
    simulated_max_drawdowns: np.ndarray
    simulated_volatilities: np.ndarray

    mean_return: float = 0.0
    median_return: float = 0.0
    std_return: float = 0.0

    mean_sharpe: float = 0.0
    median_sharpe: float = 0.0

    mean_max_drawdown: float = 0.0
    worst_drawdown: float = 0.0

    confidence_intervals: dict[str, dict[str, tuple[float, float]]] = field(
        default_factory=dict
    )

    var_estimates: dict[str, float] = field(default_factory=dict)
    cvar_estimates: dict[str, float] = field(default_factory=dict)

    probability_of_profit: float = 0.0
    probability_of_target: dict[float, float] = field(default_factory=dict)

    sample_paths: list[SimulationPath] = field(default_factory=list)

    simulation_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DrawdownDistribution:
    """Distribution of drawdown metrics."""

    max_drawdowns: np.ndarray
    drawdown_durations: np.ndarray
    recovery_times: np.ndarray

    mean_max_drawdown: float = 0.0
    median_max_drawdown: float = 0.0
    worst_max_drawdown: float = 0.0

    mean_duration: float = 0.0
    max_duration: float = 0.0

    percentiles: dict[str, float] = field(default_factory=dict)


class MonteCarloSimulator:
    """Monte Carlo simulator for trading strategies."""

    def __init__(
        self,
        config: MonteCarloConfig | None = None,
    ) -> None:
        """
        Initialize Monte Carlo simulator.

        Args:
            config: Simulation configuration
        """
        self.config = config or MonteCarloConfig()

        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        logger.info(
            f"MonteCarloSimulator initialized with {self.config.num_simulations} "
            f"simulations using {self.config.method.value} method"
        )

    async def run_simulation(
        self,
        returns: np.ndarray | pd.Series,
        initial_capital: float = 100000.0,
        num_paths_to_store: int = 100,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation on historical returns.

        Args:
            returns: Historical return series
            initial_capital: Starting capital
            num_paths_to_store: Number of sample paths to store

        Returns:
            Complete simulation result
        """
        start_time = datetime.now()

        if isinstance(returns, pd.Series):
            returns = returns.dropna().values

        returns = returns[~np.isnan(returns)]

        if len(returns) < 30:
            logger.warning(f"Limited data points: {len(returns)}")

        if self.config.parallel_simulations:
            sim_results = await self._run_parallel_simulations(
                returns, initial_capital
            )
        else:
            sim_results = self._run_sequential_simulations(
                returns, initial_capital
            )

        total_returns = np.array([r["total_return"] for r in sim_results])
        sharpe_ratios = np.array([r["sharpe_ratio"] for r in sim_results])
        max_drawdowns = np.array([r["max_drawdown"] for r in sim_results])
        volatilities = np.array([r["volatility"] for r in sim_results])

        confidence_intervals = self._calculate_confidence_intervals(
            total_returns, sharpe_ratios, max_drawdowns
        )

        var_estimates = self._calculate_var(total_returns)
        cvar_estimates = self._calculate_cvar(total_returns)

        prob_profit = np.mean(total_returns > 0)
        prob_targets = {
            0.10: np.mean(total_returns > 0.10),
            0.20: np.mean(total_returns > 0.20),
            0.50: np.mean(total_returns > 0.50),
            1.00: np.mean(total_returns > 1.00),
        }

        sample_indices = np.random.choice(
            len(sim_results),
            size=min(num_paths_to_store, len(sim_results)),
            replace=False,
        )
        sample_paths = [
            SimulationPath(
                path_id=int(idx),
                returns=sim_results[idx]["returns"],
                cumulative_returns=sim_results[idx]["cumulative_returns"],
                equity_curve=sim_results[idx]["equity_curve"],
                total_return=sim_results[idx]["total_return"],
                sharpe_ratio=sim_results[idx]["sharpe_ratio"],
                max_drawdown=sim_results[idx]["max_drawdown"],
                volatility=sim_results[idx]["volatility"],
                final_equity=sim_results[idx]["final_equity"],
            )
            for idx in sample_indices
        ]

        elapsed = (datetime.now() - start_time).total_seconds()

        return MonteCarloResult(
            num_simulations=self.config.num_simulations,
            method=self.config.method,
            original_returns=returns,
            simulated_total_returns=total_returns,
            simulated_sharpe_ratios=sharpe_ratios,
            simulated_max_drawdowns=max_drawdowns,
            simulated_volatilities=volatilities,
            mean_return=float(np.mean(total_returns)),
            median_return=float(np.median(total_returns)),
            std_return=float(np.std(total_returns)),
            mean_sharpe=float(np.mean(sharpe_ratios)),
            median_sharpe=float(np.median(sharpe_ratios)),
            mean_max_drawdown=float(np.mean(max_drawdowns)),
            worst_drawdown=float(np.min(max_drawdowns)),
            confidence_intervals=confidence_intervals,
            var_estimates=var_estimates,
            cvar_estimates=cvar_estimates,
            probability_of_profit=float(prob_profit),
            probability_of_target=prob_targets,
            sample_paths=sample_paths,
            simulation_time=elapsed,
        )

    async def _run_parallel_simulations(
        self,
        returns: np.ndarray,
        initial_capital: float,
    ) -> list[dict[str, Any]]:
        """Run simulations in parallel."""
        chunk_size = max(1, self.config.num_simulations // self.config.max_workers)
        chunks = [
            (i, min(i + chunk_size, self.config.num_simulations))
            for i in range(0, self.config.num_simulations, chunk_size)
        ]

        async def run_chunk(start_idx: int, end_idx: int) -> list[dict[str, Any]]:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: [
                    self._run_single_simulation(returns, initial_capital)
                    for _ in range(start_idx, end_idx)
                ],
            )

        tasks = [run_chunk(start, end) for start, end in chunks]
        chunk_results = await asyncio.gather(*tasks)

        results = []
        for chunk in chunk_results:
            results.extend(chunk)

        return results

    def _run_sequential_simulations(
        self,
        returns: np.ndarray,
        initial_capital: float,
    ) -> list[dict[str, Any]]:
        """Run simulations sequentially."""
        return [
            self._run_single_simulation(returns, initial_capital)
            for _ in range(self.config.num_simulations)
        ]

    def _run_single_simulation(
        self,
        returns: np.ndarray,
        initial_capital: float,
    ) -> dict[str, Any]:
        """Run a single simulation."""
        if self.config.method == SimulationMethod.BOOTSTRAP:
            sim_returns = self._bootstrap_resample(returns)
        elif self.config.method == SimulationMethod.BLOCK_BOOTSTRAP:
            sim_returns = self._block_bootstrap(returns)
        elif self.config.method == SimulationMethod.STATIONARY_BOOTSTRAP:
            sim_returns = self._stationary_bootstrap(returns)
        elif self.config.method == SimulationMethod.PARAMETRIC:
            sim_returns = self._parametric_simulation(returns)
        elif self.config.method == SimulationMethod.PATH_PERMUTATION:
            sim_returns = self._path_permutation(returns)
        elif self.config.method == SimulationMethod.GEOMETRIC_BROWNIAN:
            sim_returns = self._geometric_brownian_motion(returns)
        else:
            sim_returns = self._bootstrap_resample(returns)

        cum_returns = np.cumprod(1 + sim_returns) - 1
        equity_curve = initial_capital * (1 + cum_returns)

        total_return = cum_returns[-1] if len(cum_returns) > 0 else 0.0
        volatility = np.std(sim_returns) * np.sqrt(self.config.annualization_factor)

        if volatility > 0:
            mean_return = np.mean(sim_returns) * self.config.annualization_factor
            sharpe_ratio = mean_return / volatility
        else:
            sharpe_ratio = 0.0

        max_drawdown = self._calculate_max_drawdown(equity_curve)

        return {
            "returns": sim_returns,
            "cumulative_returns": cum_returns,
            "equity_curve": equity_curve,
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "final_equity": equity_curve[-1] if len(equity_curve) > 0 else initial_capital,
        }

    def _bootstrap_resample(self, returns: np.ndarray) -> np.ndarray:
        """Standard bootstrap resampling."""
        indices = np.random.randint(0, len(returns), size=len(returns))
        return returns[indices]

    def _block_bootstrap(self, returns: np.ndarray) -> np.ndarray:
        """Block bootstrap to preserve autocorrelation."""
        n = len(returns)
        block_size = self.config.block_size

        num_blocks = int(np.ceil(n / block_size))
        max_start = n - block_size

        if max_start <= 0:
            return self._bootstrap_resample(returns)

        blocks = []
        for _ in range(num_blocks):
            start = np.random.randint(0, max_start + 1)
            blocks.append(returns[start : start + block_size])

        resampled = np.concatenate(blocks)[:n]
        return resampled

    def _stationary_bootstrap(self, returns: np.ndarray) -> np.ndarray:
        """Stationary bootstrap with random block lengths."""
        n = len(returns)
        p = 1.0 / self.config.block_size

        result = []
        i = np.random.randint(0, n)

        while len(result) < n:
            result.append(returns[i])

            if np.random.random() < p:
                i = np.random.randint(0, n)
            else:
                i = (i + 1) % n

        return np.array(result[:n])

    def _parametric_simulation(self, returns: np.ndarray) -> np.ndarray:
        """Parametric simulation assuming normal distribution."""
        mean = np.mean(returns)
        std = np.std(returns)
        return np.random.normal(mean, std, size=len(returns))

    def _path_permutation(self, returns: np.ndarray) -> np.ndarray:
        """Random permutation of return sequence."""
        return np.random.permutation(returns)

    def _geometric_brownian_motion(self, returns: np.ndarray) -> np.ndarray:
        """Geometric Brownian motion simulation."""
        mu = np.mean(returns)
        sigma = np.std(returns)

        dt = 1.0 / self.config.annualization_factor
        n = len(returns)

        dW = np.random.normal(0, np.sqrt(dt), size=n)
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * dW

        return np.exp(log_returns) - 1

    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown from equity curve."""
        if len(equity_curve) == 0:
            return 0.0

        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        return float(np.min(drawdowns))

    def _calculate_confidence_intervals(
        self,
        total_returns: np.ndarray,
        sharpe_ratios: np.ndarray,
        max_drawdowns: np.ndarray,
    ) -> dict[str, dict[str, tuple[float, float]]]:
        """Calculate confidence intervals for metrics."""
        intervals: dict[str, dict[str, tuple[float, float]]] = {}

        for level in self.config.confidence_levels:
            level_key = f"{int(level * 100)}%"
            alpha = 1 - level

            lower_pct = alpha / 2 * 100
            upper_pct = (1 - alpha / 2) * 100

            intervals[level_key] = {
                "total_return": (
                    float(np.percentile(total_returns, lower_pct)),
                    float(np.percentile(total_returns, upper_pct)),
                ),
                "sharpe_ratio": (
                    float(np.percentile(sharpe_ratios, lower_pct)),
                    float(np.percentile(sharpe_ratios, upper_pct)),
                ),
                "max_drawdown": (
                    float(np.percentile(max_drawdowns, lower_pct)),
                    float(np.percentile(max_drawdowns, upper_pct)),
                ),
            }

        return intervals

    def _calculate_var(self, returns: np.ndarray) -> dict[str, float]:
        """Calculate Value at Risk at different confidence levels."""
        return {
            "VaR_90": float(np.percentile(returns, 10)),
            "VaR_95": float(np.percentile(returns, 5)),
            "VaR_99": float(np.percentile(returns, 1)),
        }

    def _calculate_cvar(self, returns: np.ndarray) -> dict[str, float]:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        var_90 = np.percentile(returns, 10)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)

        return {
            "CVaR_90": float(np.mean(returns[returns <= var_90])),
            "CVaR_95": float(np.mean(returns[returns <= var_95])),
            "CVaR_99": float(np.mean(returns[returns <= var_99])),
        }

    async def analyze_drawdown_distribution(
        self,
        returns: np.ndarray | pd.Series,
        initial_capital: float = 100000.0,
    ) -> DrawdownDistribution:
        """
        Analyze distribution of drawdown metrics across simulations.

        Args:
            returns: Historical return series
            initial_capital: Starting capital

        Returns:
            Drawdown distribution analysis
        """
        if isinstance(returns, pd.Series):
            returns = returns.dropna().values

        max_drawdowns = []
        durations = []
        recovery_times = []

        for _ in range(self.config.num_simulations):
            sim_returns = self._bootstrap_resample(returns)
            equity = initial_capital * np.cumprod(1 + sim_returns)

            max_dd, duration, recovery = self._analyze_single_drawdown(equity)
            max_drawdowns.append(max_dd)
            durations.append(duration)
            recovery_times.append(recovery)

        max_drawdowns_arr = np.array(max_drawdowns)
        durations_arr = np.array(durations)
        recovery_arr = np.array(recovery_times)

        return DrawdownDistribution(
            max_drawdowns=max_drawdowns_arr,
            drawdown_durations=durations_arr,
            recovery_times=recovery_arr,
            mean_max_drawdown=float(np.mean(max_drawdowns_arr)),
            median_max_drawdown=float(np.median(max_drawdowns_arr)),
            worst_max_drawdown=float(np.min(max_drawdowns_arr)),
            mean_duration=float(np.mean(durations_arr)),
            max_duration=float(np.max(durations_arr)),
            percentiles={
                "5%": float(np.percentile(max_drawdowns_arr, 5)),
                "25%": float(np.percentile(max_drawdowns_arr, 25)),
                "50%": float(np.percentile(max_drawdowns_arr, 50)),
                "75%": float(np.percentile(max_drawdowns_arr, 75)),
                "95%": float(np.percentile(max_drawdowns_arr, 95)),
            },
        )

    def _analyze_single_drawdown(
        self,
        equity_curve: np.ndarray,
    ) -> tuple[float, int, int]:
        """Analyze drawdown for a single simulation path."""
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max

        max_dd = float(np.min(drawdowns))

        in_drawdown = drawdowns < 0

        duration = 0
        max_duration = 0
        for is_dd in in_drawdown:
            if is_dd:
                duration += 1
                max_duration = max(max_duration, duration)
            else:
                duration = 0

        recovery = 0
        max_dd_idx = np.argmin(drawdowns)
        for i in range(max_dd_idx, len(drawdowns)):
            if drawdowns[i] >= 0:
                break
            recovery += 1

        return max_dd, max_duration, recovery

    async def run_strategy_comparison(
        self,
        returns_dict: dict[str, np.ndarray | pd.Series],
        initial_capital: float = 100000.0,
    ) -> dict[str, MonteCarloResult]:
        """
        Run Monte Carlo comparison for multiple strategies.

        Args:
            returns_dict: Dictionary mapping strategy names to returns
            initial_capital: Starting capital

        Returns:
            Dictionary of Monte Carlo results per strategy
        """
        results: dict[str, MonteCarloResult] = {}

        for name, returns in returns_dict.items():
            logger.info(f"Running Monte Carlo for strategy: {name}")
            results[name] = await self.run_simulation(
                returns, initial_capital
            )

        return results

    def calculate_probability_metrics(
        self,
        result: MonteCarloResult,
        target_returns: list[float] | None = None,
        max_acceptable_drawdown: float = -0.20,
    ) -> dict[str, Any]:
        """
        Calculate various probability metrics from simulation results.

        Args:
            result: Monte Carlo simulation result
            target_returns: Target return levels to calculate probabilities for
            max_acceptable_drawdown: Maximum acceptable drawdown threshold

        Returns:
            Dictionary of probability metrics
        """
        if target_returns is None:
            target_returns = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

        prob_targets = {
            f"prob_return_>{tr:.0%}": float(
                np.mean(result.simulated_total_returns > tr)
            )
            for tr in target_returns
        }

        prob_drawdown = float(
            np.mean(result.simulated_max_drawdowns >= max_acceptable_drawdown)
        )

        prob_sharpe_positive = float(
            np.mean(result.simulated_sharpe_ratios > 0)
        )

        prob_sharpe_good = float(
            np.mean(result.simulated_sharpe_ratios > 1.0)
        )

        prob_sharpe_excellent = float(
            np.mean(result.simulated_sharpe_ratios > 2.0)
        )

        return {
            **prob_targets,
            "prob_acceptable_drawdown": prob_drawdown,
            "prob_positive_sharpe": prob_sharpe_positive,
            "prob_sharpe_>1": prob_sharpe_good,
            "prob_sharpe_>2": prob_sharpe_excellent,
            "prob_profit": result.probability_of_profit,
            "expected_return": result.mean_return,
            "return_std": result.std_return,
            "risk_reward_ratio": (
                result.mean_return / abs(result.worst_drawdown)
                if result.worst_drawdown != 0
                else 0.0
            ),
        }


class OptimalFCalculator:
    """Calculator for optimal f (Kelly Criterion) using Monte Carlo."""

    def __init__(
        self,
        num_simulations: int = 1000,
        num_f_values: int = 50,
    ) -> None:
        """
        Initialize optimal f calculator.

        Args:
            num_simulations: Number of Monte Carlo simulations
            num_f_values: Number of f values to test
        """
        self.num_simulations = num_simulations
        self.num_f_values = num_f_values

    async def calculate_optimal_f(
        self,
        returns: np.ndarray | pd.Series,
        max_f: float = 1.0,
        risk_free_rate: float = 0.0,
    ) -> dict[str, Any]:
        """
        Calculate optimal f using Monte Carlo simulation.

        Args:
            returns: Historical return series
            max_f: Maximum f value to test
            risk_free_rate: Risk-free rate for calculations

        Returns:
            Optimal f analysis results
        """
        if isinstance(returns, pd.Series):
            returns = returns.dropna().values

        f_values = np.linspace(0.01, max_f, self.num_f_values)

        results = {
            "f_values": f_values,
            "geometric_means": [],
            "median_returns": [],
            "worst_case_returns": [],
            "prob_ruin": [],
        }

        for f in f_values:
            sim_results = self._simulate_with_f(returns, f)

            results["geometric_means"].append(sim_results["geometric_mean"])
            results["median_returns"].append(sim_results["median_return"])
            results["worst_case_returns"].append(sim_results["worst_case"])
            results["prob_ruin"].append(sim_results["prob_ruin"])

        optimal_idx = np.argmax(results["geometric_means"])
        optimal_f = f_values[optimal_idx]

        half_kelly_idx = np.argmin(np.abs(f_values - optimal_f / 2))

        return {
            "optimal_f": float(optimal_f),
            "optimal_f_geometric_mean": results["geometric_means"][optimal_idx],
            "half_kelly_f": float(f_values[half_kelly_idx]),
            "half_kelly_geometric_mean": results["geometric_means"][half_kelly_idx],
            "f_values": f_values.tolist(),
            "geometric_means": results["geometric_means"],
            "median_returns": results["median_returns"],
            "worst_case_returns": results["worst_case_returns"],
            "probability_of_ruin": results["prob_ruin"],
        }

    def _simulate_with_f(
        self,
        returns: np.ndarray,
        f: float,
    ) -> dict[str, float]:
        """Simulate returns with a specific f value."""
        final_returns = []
        ruin_count = 0

        for _ in range(self.num_simulations):
            sim_returns = np.random.choice(returns, size=len(returns), replace=True)

            portfolio_returns = f * sim_returns

            cum_return = np.prod(1 + portfolio_returns)

            if cum_return <= 0.1:
                ruin_count += 1

            final_returns.append(cum_return)

        final_returns = np.array(final_returns)

        positive_returns = final_returns[final_returns > 0]
        if len(positive_returns) > 0:
            geometric_mean = np.exp(np.mean(np.log(positive_returns))) - 1
        else:
            geometric_mean = -1.0

        return {
            "geometric_mean": float(geometric_mean),
            "median_return": float(np.median(final_returns) - 1),
            "worst_case": float(np.min(final_returns) - 1),
            "prob_ruin": float(ruin_count / self.num_simulations),
        }


def create_monte_carlo_simulator(
    num_simulations: int = 10000,
    method: str = "bootstrap",
    config: dict | None = None,
) -> MonteCarloSimulator:
    """
    Create a Monte Carlo simulator.

    Args:
        num_simulations: Number of simulations
        method: Simulation method
        config: Additional configuration

    Returns:
        Configured MonteCarloSimulator
    """
    sim_config = MonteCarloConfig(
        num_simulations=num_simulations,
        method=SimulationMethod(method),
        **(config or {}),
    )
    return MonteCarloSimulator(config=sim_config)
