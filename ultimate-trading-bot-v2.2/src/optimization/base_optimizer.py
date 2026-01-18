"""
Base Optimizer for Strategy Optimization.

This module provides abstract base classes and common functionality
for all optimization algorithms in the trading bot.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Generic, TypeVar

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

T = TypeVar("T")


class OptimizationStatus(str, Enum):
    """Status of optimization process."""

    NOT_STARTED = "not_started"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"


class OptimizationType(str, Enum):
    """Type of optimization problem."""

    MINIMIZATION = "minimization"
    MAXIMIZATION = "maximization"


class OptimizationDomain(str, Enum):
    """Domain of optimization parameters."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    MIXED = "mixed"
    CATEGORICAL = "categorical"


class BaseOptimizerConfig(BaseModel):
    """Base configuration for optimizers."""

    max_iterations: int = Field(default=1000, description="Maximum iterations")
    convergence_threshold: float = Field(default=1e-6, description="Convergence threshold")
    early_stopping_rounds: int = Field(default=50, description="Early stopping patience")
    optimization_type: OptimizationType = Field(
        default=OptimizationType.MAXIMIZATION,
        description="Optimization type",
    )
    random_seed: int | None = Field(default=None, description="Random seed")
    verbose: bool = Field(default=True, description="Verbose output")
    parallel_evaluations: bool = Field(default=True, description="Parallel evaluations")
    max_workers: int = Field(default=4, description="Maximum parallel workers")
    cache_evaluations: bool = Field(default=True, description="Cache evaluation results")
    timeout_seconds: float | None = Field(default=None, description="Optimization timeout")


@dataclass
class ParameterBounds:
    """Bounds for a single parameter."""

    name: str
    lower: float
    upper: float
    is_integer: bool = False
    is_log_scale: bool = False
    default_value: float | None = None


@dataclass
class CategoricalParameter:
    """Categorical parameter definition."""

    name: str
    choices: list[Any]
    default_value: Any | None = None


@dataclass
class SearchSpace:
    """Complete search space definition."""

    continuous_params: list[ParameterBounds] = field(default_factory=list)
    categorical_params: list[CategoricalParameter] = field(default_factory=list)

    @property
    def dimension(self) -> int:
        """Get total number of parameters."""
        return len(self.continuous_params) + len(self.categorical_params)

    @property
    def continuous_dimension(self) -> int:
        """Get number of continuous parameters."""
        return len(self.continuous_params)

    def get_bounds_array(self) -> np.ndarray:
        """Get bounds as numpy array."""
        if not self.continuous_params:
            return np.array([])

        bounds = np.array([
            [p.lower, p.upper] for p in self.continuous_params
        ])
        return bounds

    def sample_random(self, rng: np.random.Generator | None = None) -> dict[str, Any]:
        """Sample random point from search space."""
        if rng is None:
            rng = np.random.default_rng()

        params: dict[str, Any] = {}

        for p in self.continuous_params:
            if p.is_log_scale:
                log_val = rng.uniform(np.log(p.lower), np.log(p.upper))
                val = np.exp(log_val)
            else:
                val = rng.uniform(p.lower, p.upper)

            if p.is_integer:
                params[p.name] = int(round(val))
            else:
                params[p.name] = float(val)

        for p in self.categorical_params:
            params[p.name] = rng.choice(p.choices)

        return params

    def clip_to_bounds(self, params: dict[str, Any]) -> dict[str, Any]:
        """Clip parameters to valid bounds."""
        clipped = params.copy()

        for p in self.continuous_params:
            if p.name in clipped:
                val = float(clipped[p.name])
                val = np.clip(val, p.lower, p.upper)
                if p.is_integer:
                    clipped[p.name] = int(round(val))
                else:
                    clipped[p.name] = float(val)

        for p in self.categorical_params:
            if p.name in clipped and clipped[p.name] not in p.choices:
                clipped[p.name] = p.choices[0]

        return clipped


@dataclass
class EvaluationPoint:
    """Single evaluation point."""

    point_id: int
    parameters: dict[str, Any]
    objective_value: float
    all_objectives: dict[str, float] = field(default_factory=dict)
    constraint_violations: list[float] = field(default_factory=list)
    is_feasible: bool = True
    evaluation_time: float = 0.0
    iteration: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Result of optimization."""

    best_parameters: dict[str, Any]
    best_objective: float
    all_evaluations: list[EvaluationPoint]

    status: OptimizationStatus = OptimizationStatus.COMPLETED
    total_iterations: int = 0
    total_evaluations: int = 0
    convergence_iteration: int | None = None

    objective_history: list[float] = field(default_factory=list)
    best_objective_history: list[float] = field(default_factory=list)

    optimization_time: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime = field(default_factory=datetime.now)

    algorithm_name: str = ""
    config: dict[str, Any] = field(default_factory=dict)


class EvaluationCache:
    """Cache for objective function evaluations."""

    def __init__(self, tolerance: float = 1e-10) -> None:
        """
        Initialize evaluation cache.

        Args:
            tolerance: Tolerance for parameter matching
        """
        self.cache: dict[str, EvaluationPoint] = {}
        self.tolerance = tolerance
        self.hits = 0
        self.misses = 0

    def _make_key(self, params: dict[str, Any]) -> str:
        """Create cache key from parameters."""
        sorted_items = sorted(params.items())
        return str(sorted_items)

    def get(self, params: dict[str, Any]) -> EvaluationPoint | None:
        """Get cached evaluation."""
        key = self._make_key(params)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def set(self, point: EvaluationPoint) -> None:
        """Cache evaluation point."""
        key = self._make_key(point.parameters)
        self.cache[key] = point

    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class BaseOptimizer(ABC, Generic[T]):
    """Abstract base class for all optimizers."""

    def __init__(
        self,
        config: BaseOptimizerConfig | None = None,
    ) -> None:
        """
        Initialize base optimizer.

        Args:
            config: Optimizer configuration
        """
        self.config = config or BaseOptimizerConfig()

        if self.config.random_seed is not None:
            self.rng = np.random.default_rng(self.config.random_seed)
        else:
            self.rng = np.random.default_rng()

        self.cache = EvaluationCache() if self.config.cache_evaluations else None

        self.status = OptimizationStatus.NOT_STARTED
        self.current_iteration = 0
        self.best_point: EvaluationPoint | None = None
        self.all_evaluations: list[EvaluationPoint] = []
        self.objective_history: list[float] = []
        self.best_objective_history: list[float] = []

        self._start_time: datetime | None = None
        self._objective_func: Callable | None = None
        self._search_space: SearchSpace | None = None

        logger.info(f"{self.__class__.__name__} initialized")

    @property
    def algorithm_name(self) -> str:
        """Get algorithm name."""
        return self.__class__.__name__

    @abstractmethod
    async def _optimize_step(self) -> list[EvaluationPoint]:
        """
        Perform single optimization step.

        Returns:
            List of evaluation points from this step
        """
        pass

    @abstractmethod
    def _initialize(self, search_space: SearchSpace) -> None:
        """
        Initialize optimizer for new search space.

        Args:
            search_space: Search space definition
        """
        pass

    async def optimize(
        self,
        objective_func: Callable[[dict[str, Any]], float | dict[str, float]],
        search_space: SearchSpace,
        constraints: list[Callable[[dict[str, Any]], float]] | None = None,
    ) -> OptimizationResult:
        """
        Run optimization.

        Args:
            objective_func: Function to optimize
            search_space: Search space definition
            constraints: Optional constraint functions (should return <= 0 when satisfied)

        Returns:
            Optimization result
        """
        self._start_time = datetime.now()
        self.status = OptimizationStatus.RUNNING
        self._objective_func = objective_func
        self._search_space = search_space
        self._constraints = constraints or []

        self.current_iteration = 0
        self.best_point = None
        self.all_evaluations = []
        self.objective_history = []
        self.best_objective_history = []

        if self.cache:
            self.cache.clear()

        self._initialize(search_space)

        try:
            while self.current_iteration < self.config.max_iterations:
                if self.config.timeout_seconds:
                    elapsed = (datetime.now() - self._start_time).total_seconds()
                    if elapsed > self.config.timeout_seconds:
                        logger.info("Optimization timeout reached")
                        self.status = OptimizationStatus.MAX_ITERATIONS
                        break

                new_points = await self._optimize_step()

                for point in new_points:
                    self.all_evaluations.append(point)
                    self.objective_history.append(point.objective_value)

                    if self._is_better(point):
                        self.best_point = point

                if self.best_point:
                    self.best_objective_history.append(self.best_point.objective_value)

                if self._check_convergence():
                    self.status = OptimizationStatus.CONVERGED
                    logger.info(
                        f"Converged at iteration {self.current_iteration}"
                    )
                    break

                self.current_iteration += 1

                if self.config.verbose and self.current_iteration % 10 == 0:
                    self._log_progress()

            if self.status == OptimizationStatus.RUNNING:
                self.status = OptimizationStatus.MAX_ITERATIONS

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            self.status = OptimizationStatus.FAILED
            raise

        end_time = datetime.now()

        return OptimizationResult(
            best_parameters=self.best_point.parameters if self.best_point else {},
            best_objective=self.best_point.objective_value if self.best_point else float("-inf"),
            all_evaluations=self.all_evaluations,
            status=self.status,
            total_iterations=self.current_iteration,
            total_evaluations=len(self.all_evaluations),
            convergence_iteration=self.current_iteration if self.status == OptimizationStatus.CONVERGED else None,
            objective_history=self.objective_history,
            best_objective_history=self.best_objective_history,
            optimization_time=(end_time - self._start_time).total_seconds(),
            start_time=self._start_time,
            end_time=end_time,
            algorithm_name=self.algorithm_name,
            config=self.config.model_dump(),
        )

    async def evaluate_point(
        self,
        params: dict[str, Any],
        point_id: int | None = None,
    ) -> EvaluationPoint:
        """
        Evaluate a single point.

        Args:
            params: Parameters to evaluate
            point_id: Optional point ID

        Returns:
            Evaluation point
        """
        if self.cache:
            cached = self.cache.get(params)
            if cached is not None:
                return cached

        start_time = datetime.now()

        try:
            if asyncio.iscoroutinefunction(self._objective_func):
                result = await self._objective_func(params)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, self._objective_func, params
                )

            if isinstance(result, dict):
                objective_value = result.get("objective", result.get("value", 0.0))
                all_objectives = result
            else:
                objective_value = float(result)
                all_objectives = {"objective": objective_value}

            constraint_violations = []
            is_feasible = True

            for constraint_func in self._constraints:
                if asyncio.iscoroutinefunction(constraint_func):
                    violation = await constraint_func(params)
                else:
                    loop = asyncio.get_event_loop()
                    violation = await loop.run_in_executor(
                        None, constraint_func, params
                    )
                constraint_violations.append(violation)
                if violation > 0:
                    is_feasible = False

            evaluation_time = (datetime.now() - start_time).total_seconds()

            point = EvaluationPoint(
                point_id=point_id or len(self.all_evaluations),
                parameters=params,
                objective_value=objective_value,
                all_objectives=all_objectives,
                constraint_violations=constraint_violations,
                is_feasible=is_feasible,
                evaluation_time=evaluation_time,
                iteration=self.current_iteration,
            )

            if self.cache:
                self.cache.set(point)

            return point

        except Exception as e:
            logger.warning(f"Evaluation failed for {params}: {e}")

            penalty = float("-inf") if self.config.optimization_type == OptimizationType.MAXIMIZATION else float("inf")

            return EvaluationPoint(
                point_id=point_id or len(self.all_evaluations),
                parameters=params,
                objective_value=penalty,
                is_feasible=False,
                evaluation_time=0.0,
                iteration=self.current_iteration,
                metadata={"error": str(e)},
            )

    async def evaluate_batch(
        self,
        params_list: list[dict[str, Any]],
    ) -> list[EvaluationPoint]:
        """
        Evaluate batch of points.

        Args:
            params_list: List of parameter dictionaries

        Returns:
            List of evaluation points
        """
        if self.config.parallel_evaluations:
            semaphore = asyncio.Semaphore(self.config.max_workers)

            async def eval_with_semaphore(params: dict, idx: int) -> EvaluationPoint:
                async with semaphore:
                    return await self.evaluate_point(params, idx)

            tasks = [
                eval_with_semaphore(params, i)
                for i, params in enumerate(params_list)
            ]
            results = await asyncio.gather(*tasks)
            return list(results)
        else:
            results = []
            for i, params in enumerate(params_list):
                point = await self.evaluate_point(params, i)
                results.append(point)
            return results

    def _is_better(self, point: EvaluationPoint) -> bool:
        """Check if point is better than current best."""
        if self.best_point is None:
            return point.is_feasible

        if point.is_feasible and not self.best_point.is_feasible:
            return True

        if not point.is_feasible and self.best_point.is_feasible:
            return False

        if self.config.optimization_type == OptimizationType.MAXIMIZATION:
            return point.objective_value > self.best_point.objective_value
        else:
            return point.objective_value < self.best_point.objective_value

    def _check_convergence(self) -> bool:
        """Check if optimization has converged."""
        if len(self.best_objective_history) < self.config.early_stopping_rounds:
            return False

        recent = self.best_objective_history[-self.config.early_stopping_rounds:]
        improvement = abs(recent[-1] - recent[0])

        return improvement < self.config.convergence_threshold

    def _log_progress(self) -> None:
        """Log optimization progress."""
        if self.best_point:
            logger.info(
                f"Iteration {self.current_iteration}: "
                f"Best objective = {self.best_point.objective_value:.6f}, "
                f"Evaluations = {len(self.all_evaluations)}"
            )


class RandomSearchOptimizer(BaseOptimizer):
    """Simple random search optimizer."""

    def __init__(
        self,
        config: BaseOptimizerConfig | None = None,
        samples_per_iteration: int = 10,
    ) -> None:
        """
        Initialize random search optimizer.

        Args:
            config: Optimizer configuration
            samples_per_iteration: Samples per iteration
        """
        super().__init__(config)
        self.samples_per_iteration = samples_per_iteration

    def _initialize(self, search_space: SearchSpace) -> None:
        """Initialize for search space."""
        pass

    async def _optimize_step(self) -> list[EvaluationPoint]:
        """Perform single optimization step."""
        candidates = [
            self._search_space.sample_random(self.rng)
            for _ in range(self.samples_per_iteration)
        ]

        return await self.evaluate_batch(candidates)


class GridSearchOptimizer(BaseOptimizer):
    """Grid search optimizer."""

    def __init__(
        self,
        config: BaseOptimizerConfig | None = None,
        points_per_dimension: int = 10,
    ) -> None:
        """
        Initialize grid search optimizer.

        Args:
            config: Optimizer configuration
            points_per_dimension: Points per dimension
        """
        super().__init__(config)
        self.points_per_dimension = points_per_dimension
        self._grid: list[dict[str, Any]] = []
        self._grid_idx = 0

    def _initialize(self, search_space: SearchSpace) -> None:
        """Initialize grid for search space."""
        import itertools

        param_values: dict[str, list[Any]] = {}

        for p in search_space.continuous_params:
            if p.is_log_scale:
                values = np.geomspace(p.lower, p.upper, self.points_per_dimension)
            else:
                values = np.linspace(p.lower, p.upper, self.points_per_dimension)

            if p.is_integer:
                values = [int(round(v)) for v in values]
                values = list(dict.fromkeys(values))
            else:
                values = values.tolist()

            param_values[p.name] = values

        for p in search_space.categorical_params:
            param_values[p.name] = p.choices

        keys = list(param_values.keys())
        values = [param_values[k] for k in keys]

        self._grid = [
            dict(zip(keys, combo))
            for combo in itertools.product(*values)
        ]
        self._grid_idx = 0

        logger.info(f"Grid search: {len(self._grid)} total points")

    async def _optimize_step(self) -> list[EvaluationPoint]:
        """Perform single optimization step."""
        batch_size = min(self.config.max_workers, len(self._grid) - self._grid_idx)

        if batch_size <= 0:
            self.status = OptimizationStatus.COMPLETED
            return []

        batch = self._grid[self._grid_idx : self._grid_idx + batch_size]
        self._grid_idx += batch_size

        return await self.evaluate_batch(batch)


def create_random_optimizer(
    max_iterations: int = 1000,
    samples_per_iteration: int = 10,
    optimization_type: str = "maximization",
    seed: int | None = None,
) -> RandomSearchOptimizer:
    """
    Create random search optimizer.

    Args:
        max_iterations: Maximum iterations
        samples_per_iteration: Samples per iteration
        optimization_type: Optimization type
        seed: Random seed

    Returns:
        Configured RandomSearchOptimizer
    """
    config = BaseOptimizerConfig(
        max_iterations=max_iterations,
        optimization_type=OptimizationType(optimization_type),
        random_seed=seed,
    )
    return RandomSearchOptimizer(config, samples_per_iteration)


def create_grid_optimizer(
    points_per_dimension: int = 10,
    optimization_type: str = "maximization",
) -> GridSearchOptimizer:
    """
    Create grid search optimizer.

    Args:
        points_per_dimension: Points per dimension
        optimization_type: Optimization type

    Returns:
        Configured GridSearchOptimizer
    """
    config = BaseOptimizerConfig(
        max_iterations=10000,
        optimization_type=OptimizationType(optimization_type),
    )
    return GridSearchOptimizer(config, points_per_dimension)
