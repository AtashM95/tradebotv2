"""
Parameter Sweep for Strategy Optimization.

This module provides comprehensive parameter sweep and grid search
capabilities for optimizing trading strategy parameters.
"""

import asyncio
import itertools
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SweepMethod(str, Enum):
    """Parameter sweep methods."""

    GRID = "grid"
    RANDOM = "random"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"


class ParameterType(str, Enum):
    """Types of parameters."""

    INTEGER = "integer"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


class OptimizationMetric(str, Enum):
    """Metrics for optimization."""

    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    TOTAL_RETURN = "total_return"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"
    PROFIT_FACTOR = "profit_factor"
    WIN_RATE = "win_rate"
    MAX_DRAWDOWN = "max_drawdown"
    CUSTOM = "custom"


class ParameterSweepConfig(BaseModel):
    """Configuration for parameter sweep."""

    method: SweepMethod = Field(default=SweepMethod.GRID, description="Sweep method")
    max_iterations: int = Field(default=1000, description="Maximum iterations")
    random_samples: int = Field(default=100, description="Random samples for random search")
    convergence_threshold: float = Field(default=1e-6, description="Convergence threshold")
    early_stopping_rounds: int = Field(default=50, description="Early stopping patience")
    parallel_evaluations: bool = Field(default=True, description="Parallel evaluations")
    max_workers: int = Field(default=4, description="Maximum parallel workers")
    seed: int | None = Field(default=None, description="Random seed")
    metric: OptimizationMetric = Field(
        default=OptimizationMetric.SHARPE_RATIO,
        description="Optimization metric",
    )
    minimize: bool = Field(default=False, description="Minimize metric")
    cross_validation_folds: int = Field(default=0, description="CV folds (0 = no CV)")
    annualization_factor: float = Field(default=252.0, description="Trading days per year")


@dataclass
class ParameterSpec:
    """Specification for a single parameter."""

    name: str
    param_type: ParameterType
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    choices: list[Any] | None = None
    default: Any = None
    log_scale: bool = False


@dataclass
class EvaluationResult:
    """Result from a single parameter evaluation."""

    parameters: dict[str, Any]
    metric_value: float
    all_metrics: dict[str, float] = field(default_factory=dict)
    returns: np.ndarray | None = None
    equity_curve: np.ndarray | None = None
    is_valid: bool = True
    evaluation_time: float = 0.0
    fold_results: list[float] = field(default_factory=list)


@dataclass
class SweepResult:
    """Complete parameter sweep result."""

    config: ParameterSweepConfig
    best_parameters: dict[str, Any]
    best_metric_value: float
    all_evaluations: list[EvaluationResult]

    total_evaluations: int = 0
    valid_evaluations: int = 0

    parameter_importance: dict[str, float] = field(default_factory=dict)
    parameter_correlations: dict[str, dict[str, float]] = field(default_factory=dict)

    convergence_history: list[float] = field(default_factory=list)

    sweep_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class ParameterSpace:
    """Define parameter search space."""

    def __init__(self) -> None:
        """Initialize parameter space."""
        self.parameters: dict[str, ParameterSpec] = {}

    def add_integer(
        self,
        name: str,
        min_value: int,
        max_value: int,
        step: int = 1,
        default: int | None = None,
    ) -> "ParameterSpace":
        """
        Add integer parameter.

        Args:
            name: Parameter name
            min_value: Minimum value
            max_value: Maximum value
            step: Step size
            default: Default value

        Returns:
            Self for chaining
        """
        self.parameters[name] = ParameterSpec(
            name=name,
            param_type=ParameterType.INTEGER,
            min_value=min_value,
            max_value=max_value,
            step=step,
            default=default or min_value,
        )
        return self

    def add_float(
        self,
        name: str,
        min_value: float,
        max_value: float,
        step: float | None = None,
        default: float | None = None,
        log_scale: bool = False,
    ) -> "ParameterSpace":
        """
        Add float parameter.

        Args:
            name: Parameter name
            min_value: Minimum value
            max_value: Maximum value
            step: Step size (optional)
            default: Default value
            log_scale: Use logarithmic scale

        Returns:
            Self for chaining
        """
        self.parameters[name] = ParameterSpec(
            name=name,
            param_type=ParameterType.FLOAT,
            min_value=min_value,
            max_value=max_value,
            step=step,
            default=default or min_value,
            log_scale=log_scale,
        )
        return self

    def add_categorical(
        self,
        name: str,
        choices: list[Any],
        default: Any | None = None,
    ) -> "ParameterSpace":
        """
        Add categorical parameter.

        Args:
            name: Parameter name
            choices: List of choices
            default: Default value

        Returns:
            Self for chaining
        """
        self.parameters[name] = ParameterSpec(
            name=name,
            param_type=ParameterType.CATEGORICAL,
            choices=choices,
            default=default or choices[0],
        )
        return self

    def add_boolean(
        self,
        name: str,
        default: bool = False,
    ) -> "ParameterSpace":
        """
        Add boolean parameter.

        Args:
            name: Parameter name
            default: Default value

        Returns:
            Self for chaining
        """
        self.parameters[name] = ParameterSpec(
            name=name,
            param_type=ParameterType.BOOLEAN,
            choices=[True, False],
            default=default,
        )
        return self

    def get_grid(self) -> list[dict[str, Any]]:
        """
        Generate full parameter grid.

        Returns:
            List of parameter combinations
        """
        param_values: dict[str, list[Any]] = {}

        for name, spec in self.parameters.items():
            if spec.param_type == ParameterType.INTEGER:
                step = int(spec.step or 1)
                param_values[name] = list(
                    range(int(spec.min_value), int(spec.max_value) + 1, step)
                )
            elif spec.param_type == ParameterType.FLOAT:
                if spec.step:
                    num_steps = int((spec.max_value - spec.min_value) / spec.step) + 1
                    if spec.log_scale:
                        param_values[name] = np.geomspace(
                            spec.min_value, spec.max_value, num_steps
                        ).tolist()
                    else:
                        param_values[name] = [
                            spec.min_value + i * spec.step for i in range(num_steps)
                        ]
                else:
                    if spec.log_scale:
                        param_values[name] = np.geomspace(
                            spec.min_value, spec.max_value, 10
                        ).tolist()
                    else:
                        param_values[name] = np.linspace(
                            spec.min_value, spec.max_value, 10
                        ).tolist()
            elif spec.param_type in [ParameterType.CATEGORICAL, ParameterType.BOOLEAN]:
                param_values[name] = spec.choices or [spec.default]

        keys = list(param_values.keys())
        values = [param_values[k] for k in keys]

        grid = []
        for combo in itertools.product(*values):
            grid.append(dict(zip(keys, combo)))

        return grid

    def sample_random(self, rng: np.random.Generator | None = None) -> dict[str, Any]:
        """
        Sample random parameter combination.

        Args:
            rng: Random number generator

        Returns:
            Random parameter dictionary
        """
        if rng is None:
            rng = np.random.default_rng()

        params: dict[str, Any] = {}

        for name, spec in self.parameters.items():
            if spec.param_type == ParameterType.INTEGER:
                step = int(spec.step or 1)
                possible_values = range(int(spec.min_value), int(spec.max_value) + 1, step)
                params[name] = int(rng.choice(list(possible_values)))
            elif spec.param_type == ParameterType.FLOAT:
                if spec.log_scale:
                    log_min = np.log(spec.min_value)
                    log_max = np.log(spec.max_value)
                    params[name] = float(np.exp(rng.uniform(log_min, log_max)))
                else:
                    params[name] = float(rng.uniform(spec.min_value, spec.max_value))
            elif spec.param_type in [ParameterType.CATEGORICAL, ParameterType.BOOLEAN]:
                params[name] = rng.choice(spec.choices or [spec.default])

        return params

    def get_bounds(self) -> list[tuple[float, float]]:
        """Get bounds for continuous parameters."""
        bounds = []

        for name, spec in self.parameters.items():
            if spec.param_type in [ParameterType.INTEGER, ParameterType.FLOAT]:
                bounds.append((spec.min_value, spec.max_value))

        return bounds

    def get_num_combinations(self) -> int:
        """Get total number of grid combinations."""
        total = 1

        for spec in self.parameters.values():
            if spec.param_type == ParameterType.INTEGER:
                step = int(spec.step or 1)
                count = len(range(int(spec.min_value), int(spec.max_value) + 1, step))
            elif spec.param_type == ParameterType.FLOAT:
                if spec.step:
                    count = int((spec.max_value - spec.min_value) / spec.step) + 1
                else:
                    count = 10
            elif spec.param_type in [ParameterType.CATEGORICAL, ParameterType.BOOLEAN]:
                count = len(spec.choices or [])
            else:
                count = 1

            total *= count

        return total


class ParameterSweep:
    """Parameter sweep optimizer."""

    def __init__(
        self,
        config: ParameterSweepConfig | None = None,
    ) -> None:
        """
        Initialize parameter sweep.

        Args:
            config: Sweep configuration
        """
        self.config = config or ParameterSweepConfig()

        if self.config.seed is not None:
            self.rng = np.random.default_rng(self.config.seed)
        else:
            self.rng = np.random.default_rng()

        logger.info(
            f"ParameterSweep initialized with {self.config.method.value} method"
        )

    async def run(
        self,
        parameter_space: ParameterSpace,
        objective_func: Callable[[dict[str, Any]], float | dict[str, float]],
        data: pd.DataFrame | None = None,
    ) -> SweepResult:
        """
        Run parameter sweep.

        Args:
            parameter_space: Parameter search space
            objective_func: Function that takes parameters and returns metric value
            data: Optional data for evaluation

        Returns:
            Sweep result
        """
        start_time = datetime.now()

        if self.config.method == SweepMethod.GRID:
            evaluations = await self._run_grid_search(
                parameter_space, objective_func
            )
        elif self.config.method == SweepMethod.RANDOM:
            evaluations = await self._run_random_search(
                parameter_space, objective_func
            )
        elif self.config.method == SweepMethod.GENETIC:
            evaluations = await self._run_genetic_search(
                parameter_space, objective_func
            )
        elif self.config.method == SweepMethod.DIFFERENTIAL_EVOLUTION:
            evaluations = await self._run_de_search(
                parameter_space, objective_func
            )
        else:
            evaluations = await self._run_random_search(
                parameter_space, objective_func
            )

        valid_evaluations = [e for e in evaluations if e.is_valid]

        if valid_evaluations:
            if self.config.minimize:
                best = min(valid_evaluations, key=lambda x: x.metric_value)
            else:
                best = max(valid_evaluations, key=lambda x: x.metric_value)
        else:
            best = EvaluationResult(
                parameters={},
                metric_value=float("-inf") if not self.config.minimize else float("inf"),
                is_valid=False,
            )

        importance = self._calculate_parameter_importance(evaluations, parameter_space)
        correlations = self._calculate_parameter_correlations(evaluations, parameter_space)

        convergence = [e.metric_value for e in evaluations[:100]]

        elapsed = (datetime.now() - start_time).total_seconds()

        return SweepResult(
            config=self.config,
            best_parameters=best.parameters,
            best_metric_value=best.metric_value,
            all_evaluations=evaluations,
            total_evaluations=len(evaluations),
            valid_evaluations=len(valid_evaluations),
            parameter_importance=importance,
            parameter_correlations=correlations,
            convergence_history=convergence,
            sweep_time=elapsed,
        )

    async def _run_grid_search(
        self,
        parameter_space: ParameterSpace,
        objective_func: Callable,
    ) -> list[EvaluationResult]:
        """Run grid search."""
        grid = parameter_space.get_grid()
        logger.info(f"Grid search: {len(grid)} combinations")

        if len(grid) > self.config.max_iterations:
            logger.warning(
                f"Grid size {len(grid)} exceeds max iterations {self.config.max_iterations}, "
                f"sampling randomly"
            )
            indices = self.rng.choice(len(grid), self.config.max_iterations, replace=False)
            grid = [grid[i] for i in indices]

        if self.config.parallel_evaluations:
            evaluations = await self._evaluate_parallel(grid, objective_func)
        else:
            evaluations = await self._evaluate_sequential(grid, objective_func)

        return evaluations

    async def _run_random_search(
        self,
        parameter_space: ParameterSpace,
        objective_func: Callable,
    ) -> list[EvaluationResult]:
        """Run random search."""
        num_samples = min(self.config.random_samples, self.config.max_iterations)
        logger.info(f"Random search: {num_samples} samples")

        candidates = [
            parameter_space.sample_random(self.rng) for _ in range(num_samples)
        ]

        if self.config.parallel_evaluations:
            evaluations = await self._evaluate_parallel(candidates, objective_func)
        else:
            evaluations = await self._evaluate_sequential(candidates, objective_func)

        return evaluations

    async def _run_genetic_search(
        self,
        parameter_space: ParameterSpace,
        objective_func: Callable,
    ) -> list[EvaluationResult]:
        """Run genetic algorithm search."""
        population_size = min(50, self.config.random_samples)
        num_generations = self.config.max_iterations // population_size

        logger.info(
            f"Genetic search: population {population_size}, "
            f"generations {num_generations}"
        )

        population = [
            parameter_space.sample_random(self.rng) for _ in range(population_size)
        ]

        all_evaluations = []

        for gen in range(num_generations):
            if self.config.parallel_evaluations:
                evaluations = await self._evaluate_parallel(population, objective_func)
            else:
                evaluations = await self._evaluate_sequential(population, objective_func)

            all_evaluations.extend(evaluations)

            sorted_evals = sorted(
                evaluations,
                key=lambda x: x.metric_value,
                reverse=not self.config.minimize,
            )

            elite_size = population_size // 4
            elite = [e.parameters for e in sorted_evals[:elite_size]]

            new_population = elite.copy()

            while len(new_population) < population_size:
                parent1, parent2 = self.rng.choice(elite, size=2, replace=False)

                child = self._crossover(parent1, parent2, parameter_space)
                child = self._mutate(child, parameter_space, mutation_rate=0.1)

                new_population.append(child)

            population = new_population

        return all_evaluations

    async def _run_de_search(
        self,
        parameter_space: ParameterSpace,
        objective_func: Callable,
    ) -> list[EvaluationResult]:
        """Run differential evolution search."""
        population_size = min(50, self.config.random_samples)
        num_generations = self.config.max_iterations // population_size

        logger.info(
            f"Differential evolution: population {population_size}, "
            f"generations {num_generations}"
        )

        population = [
            parameter_space.sample_random(self.rng) for _ in range(population_size)
        ]

        all_evaluations: list[EvaluationResult] = []
        fitness: list[float] = []

        initial_evals = await self._evaluate_parallel(population, objective_func)
        all_evaluations.extend(initial_evals)
        fitness = [e.metric_value for e in initial_evals]

        for gen in range(num_generations):
            new_population = []
            new_fitness = []

            for i in range(population_size):
                indices = [j for j in range(population_size) if j != i]
                a, b, c = self.rng.choice(indices, size=3, replace=False)

                mutant = self._de_mutate(
                    population[a], population[b], population[c], parameter_space
                )

                trial = self._de_crossover(population[i], mutant, parameter_space)

                trial_eval = await self._evaluate_single(trial, objective_func)
                all_evaluations.append(trial_eval)

                if self.config.minimize:
                    if trial_eval.metric_value < fitness[i]:
                        new_population.append(trial)
                        new_fitness.append(trial_eval.metric_value)
                    else:
                        new_population.append(population[i])
                        new_fitness.append(fitness[i])
                else:
                    if trial_eval.metric_value > fitness[i]:
                        new_population.append(trial)
                        new_fitness.append(trial_eval.metric_value)
                    else:
                        new_population.append(population[i])
                        new_fitness.append(fitness[i])

            population = new_population
            fitness = new_fitness

        return all_evaluations

    def _crossover(
        self,
        parent1: dict[str, Any],
        parent2: dict[str, Any],
        parameter_space: ParameterSpace,
    ) -> dict[str, Any]:
        """Perform crossover between two parents."""
        child = {}

        for name in parameter_space.parameters:
            if self.rng.random() < 0.5:
                child[name] = parent1[name]
            else:
                child[name] = parent2[name]

        return child

    def _mutate(
        self,
        individual: dict[str, Any],
        parameter_space: ParameterSpace,
        mutation_rate: float = 0.1,
    ) -> dict[str, Any]:
        """Mutate an individual."""
        mutated = individual.copy()

        for name, spec in parameter_space.parameters.items():
            if self.rng.random() < mutation_rate:
                if spec.param_type == ParameterType.INTEGER:
                    step = int(spec.step or 1)
                    delta = self.rng.integers(-2, 3) * step
                    new_val = int(mutated[name]) + delta
                    mutated[name] = int(np.clip(new_val, spec.min_value, spec.max_value))
                elif spec.param_type == ParameterType.FLOAT:
                    range_size = spec.max_value - spec.min_value
                    delta = self.rng.normal(0, range_size * 0.1)
                    new_val = float(mutated[name]) + delta
                    mutated[name] = float(np.clip(new_val, spec.min_value, spec.max_value))
                elif spec.param_type in [ParameterType.CATEGORICAL, ParameterType.BOOLEAN]:
                    mutated[name] = self.rng.choice(spec.choices)

        return mutated

    def _de_mutate(
        self,
        a: dict[str, Any],
        b: dict[str, Any],
        c: dict[str, Any],
        parameter_space: ParameterSpace,
        f: float = 0.8,
    ) -> dict[str, Any]:
        """Differential evolution mutation."""
        mutant = {}

        for name, spec in parameter_space.parameters.items():
            if spec.param_type in [ParameterType.INTEGER, ParameterType.FLOAT]:
                diff = float(b[name]) - float(c[name])
                new_val = float(a[name]) + f * diff
                new_val = np.clip(new_val, spec.min_value, spec.max_value)

                if spec.param_type == ParameterType.INTEGER:
                    mutant[name] = int(round(new_val))
                else:
                    mutant[name] = float(new_val)
            else:
                mutant[name] = a[name]

        return mutant

    def _de_crossover(
        self,
        target: dict[str, Any],
        mutant: dict[str, Any],
        parameter_space: ParameterSpace,
        cr: float = 0.9,
    ) -> dict[str, Any]:
        """Differential evolution crossover."""
        trial = {}
        param_names = list(parameter_space.parameters.keys())
        force_idx = self.rng.integers(0, len(param_names))

        for i, name in enumerate(param_names):
            if self.rng.random() < cr or i == force_idx:
                trial[name] = mutant[name]
            else:
                trial[name] = target[name]

        return trial

    async def _evaluate_parallel(
        self,
        candidates: list[dict[str, Any]],
        objective_func: Callable,
    ) -> list[EvaluationResult]:
        """Evaluate candidates in parallel."""
        semaphore = asyncio.Semaphore(self.config.max_workers)

        async def eval_with_semaphore(params: dict[str, Any]) -> EvaluationResult:
            async with semaphore:
                return await self._evaluate_single(params, objective_func)

        tasks = [eval_with_semaphore(p) for p in candidates]
        results = await asyncio.gather(*tasks)

        return list(results)

    async def _evaluate_sequential(
        self,
        candidates: list[dict[str, Any]],
        objective_func: Callable,
    ) -> list[EvaluationResult]:
        """Evaluate candidates sequentially."""
        results = []

        for params in candidates:
            result = await self._evaluate_single(params, objective_func)
            results.append(result)

        return results

    async def _evaluate_single(
        self,
        params: dict[str, Any],
        objective_func: Callable,
    ) -> EvaluationResult:
        """Evaluate a single parameter combination."""
        start_time = datetime.now()

        try:
            if asyncio.iscoroutinefunction(objective_func):
                result = await objective_func(params)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, objective_func, params)

            if isinstance(result, dict):
                metric_value = result.get(self.config.metric.value, 0.0)
                all_metrics = result
            else:
                metric_value = float(result)
                all_metrics = {self.config.metric.value: metric_value}

            elapsed = (datetime.now() - start_time).total_seconds()

            return EvaluationResult(
                parameters=params,
                metric_value=metric_value,
                all_metrics=all_metrics,
                is_valid=True,
                evaluation_time=elapsed,
            )

        except Exception as e:
            logger.warning(f"Evaluation failed for {params}: {e}")
            return EvaluationResult(
                parameters=params,
                metric_value=float("-inf") if not self.config.minimize else float("inf"),
                is_valid=False,
                evaluation_time=0.0,
            )

    def _calculate_parameter_importance(
        self,
        evaluations: list[EvaluationResult],
        parameter_space: ParameterSpace,
    ) -> dict[str, float]:
        """Calculate parameter importance based on variance contribution."""
        importance: dict[str, float] = {}

        valid_evals = [e for e in evaluations if e.is_valid]
        if len(valid_evals) < 10:
            return {name: 0.0 for name in parameter_space.parameters}

        metrics = np.array([e.metric_value for e in valid_evals])
        total_variance = np.var(metrics) if len(metrics) > 1 else 1.0

        for name, spec in parameter_space.parameters.items():
            if spec.param_type in [ParameterType.INTEGER, ParameterType.FLOAT]:
                param_values = np.array([e.parameters[name] for e in valid_evals])

                if np.std(param_values) > 0:
                    correlation = np.corrcoef(param_values, metrics)[0, 1]
                    importance[name] = abs(correlation) if not np.isnan(correlation) else 0.0
                else:
                    importance[name] = 0.0
            else:
                unique_values = list(set(e.parameters[name] for e in valid_evals))
                if len(unique_values) > 1:
                    group_means = []
                    for val in unique_values:
                        group_metrics = [
                            e.metric_value for e in valid_evals
                            if e.parameters[name] == val
                        ]
                        if group_metrics:
                            group_means.append(np.mean(group_metrics))

                    if len(group_means) > 1:
                        between_var = np.var(group_means)
                        importance[name] = between_var / total_variance if total_variance > 0 else 0.0
                    else:
                        importance[name] = 0.0
                else:
                    importance[name] = 0.0

        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}

        return importance

    def _calculate_parameter_correlations(
        self,
        evaluations: list[EvaluationResult],
        parameter_space: ParameterSpace,
    ) -> dict[str, dict[str, float]]:
        """Calculate correlations between parameters and metrics."""
        correlations: dict[str, dict[str, float]] = {}

        valid_evals = [e for e in evaluations if e.is_valid]
        if len(valid_evals) < 10:
            return {}

        metrics = np.array([e.metric_value for e in valid_evals])

        for name, spec in parameter_space.parameters.items():
            if spec.param_type in [ParameterType.INTEGER, ParameterType.FLOAT]:
                param_values = np.array([float(e.parameters[name]) for e in valid_evals])

                if np.std(param_values) > 0:
                    corr = np.corrcoef(param_values, metrics)[0, 1]
                    correlations[name] = {
                        "metric_correlation": float(corr) if not np.isnan(corr) else 0.0,
                    }
                else:
                    correlations[name] = {"metric_correlation": 0.0}

        return correlations


def create_parameter_sweep(
    method: str = "grid",
    max_iterations: int = 1000,
    metric: str = "sharpe_ratio",
    config: dict | None = None,
) -> ParameterSweep:
    """
    Create a parameter sweep optimizer.

    Args:
        method: Sweep method
        max_iterations: Maximum iterations
        metric: Optimization metric
        config: Additional configuration

    Returns:
        Configured ParameterSweep
    """
    sweep_config = ParameterSweepConfig(
        method=SweepMethod(method),
        max_iterations=max_iterations,
        metric=OptimizationMetric(metric),
        **(config or {}),
    )
    return ParameterSweep(config=sweep_config)
