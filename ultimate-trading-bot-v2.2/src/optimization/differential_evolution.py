"""
Differential Evolution Optimizer.

This module provides Differential Evolution optimization
for trading strategy parameters.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, Field

from ultimate_trading_bot.optimization.base_optimizer import (
    BaseOptimizer,
    BaseOptimizerConfig,
    EvaluationPoint,
    OptimizationType,
    SearchSpace,
)

logger = logging.getLogger(__name__)


class MutationStrategy(str, Enum):
    """Mutation strategies for DE."""

    RAND_1 = "rand/1"
    RAND_2 = "rand/2"
    BEST_1 = "best/1"
    BEST_2 = "best/2"
    CURRENT_TO_BEST_1 = "current-to-best/1"
    CURRENT_TO_RAND_1 = "current-to-rand/1"
    RAND_TO_BEST_1 = "rand-to-best/1"


class CrossoverType(str, Enum):
    """Crossover types for DE."""

    BINOMIAL = "binomial"
    EXPONENTIAL = "exponential"


class DEConfig(BaseOptimizerConfig):
    """Configuration for Differential Evolution."""

    population_size: int = Field(default=50, description="Population size")
    mutation_factor: float = Field(default=0.8, description="Mutation factor (F)")
    crossover_rate: float = Field(default=0.9, description="Crossover rate (CR)")
    mutation_strategy: MutationStrategy = Field(
        default=MutationStrategy.BEST_1,
        description="Mutation strategy",
    )
    crossover_type: CrossoverType = Field(
        default=CrossoverType.BINOMIAL,
        description="Crossover type",
    )
    dither: bool = Field(default=True, description="Use dithering for F")
    dither_range: tuple[float, float] = Field(
        default=(0.5, 1.0),
        description="Dither range for F",
    )
    adaptive: bool = Field(default=False, description="Use adaptive parameters")
    polish: bool = Field(default=True, description="Polish best solution at end")
    tol: float = Field(default=1e-8, description="Tolerance for convergence")
    atol: float = Field(default=1e-8, description="Absolute tolerance")


@dataclass
class DEIndividual:
    """Individual in DE population."""

    idx: int
    position: np.ndarray
    fitness: float = float("-inf")
    trial_position: np.ndarray = field(default_factory=lambda: np.array([]))
    trial_fitness: float = float("-inf")
    is_evaluated: bool = False


@dataclass
class DEPopulation:
    """DE Population."""

    individuals: list[DEIndividual]
    best_idx: int = 0
    best_fitness: float = float("-inf")
    generation: int = 0


class DifferentialEvolution(BaseOptimizer):
    """Differential Evolution Optimizer."""

    def __init__(
        self,
        config: DEConfig | None = None,
    ) -> None:
        """
        Initialize DE optimizer.

        Args:
            config: DE configuration
        """
        self._de_config = config or DEConfig()
        super().__init__(self._de_config)

        self.population: DEPopulation | None = None
        self._bounds: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    def _initialize(self, search_space: SearchSpace) -> None:
        """Initialize DE population."""
        self._bounds = search_space.get_bounds_array()
        n_dim = len(self._bounds)

        if n_dim == 0:
            logger.warning("No continuous parameters in search space")
            return

        self._scale = self._bounds[:, 1] - self._bounds[:, 0]

        individuals = []

        for i in range(self._de_config.population_size):
            position = np.array([
                self.rng.uniform(lo, hi) for lo, hi in self._bounds
            ])

            individual = DEIndividual(
                idx=i,
                position=position,
                trial_position=np.zeros(n_dim),
            )
            individuals.append(individual)

        self.population = DEPopulation(individuals=individuals)

        logger.info(
            f"DE initialized with {self._de_config.population_size} individuals, "
            f"{n_dim} dimensions, strategy: {self._de_config.mutation_strategy.value}"
        )

    async def _optimize_step(self) -> list[EvaluationPoint]:
        """Perform single DE generation."""
        unevaluated = [
            ind for ind in self.population.individuals
            if not ind.is_evaluated
        ]

        if unevaluated:
            params_list = [
                self._array_to_params(ind.position) for ind in unevaluated
            ]
            points = await self.evaluate_batch(params_list)

            for ind, point in zip(unevaluated, points):
                ind.fitness = point.objective_value
                ind.is_evaluated = True

            self._update_best()

            return points

        for ind in self.population.individuals:
            mutant = self._mutate(ind)
            trial = self._crossover(ind.position, mutant)
            trial = self._ensure_bounds(trial)
            ind.trial_position = trial

        trial_params = [
            self._array_to_params(ind.trial_position)
            for ind in self.population.individuals
        ]
        trial_points = await self.evaluate_batch(trial_params)

        for ind, point in zip(self.population.individuals, trial_points):
            ind.trial_fitness = point.objective_value

        self._selection()
        self._update_best()

        self.population.generation += 1

        return trial_points

    def _mutate(self, target: DEIndividual) -> np.ndarray:
        """
        Generate mutant vector.

        Args:
            target: Target individual

        Returns:
            Mutant vector
        """
        pop = self.population.individuals
        n = len(pop)

        if self._de_config.dither:
            f = self.rng.uniform(*self._de_config.dither_range)
        else:
            f = self._de_config.mutation_factor

        strategy = self._de_config.mutation_strategy

        if strategy == MutationStrategy.RAND_1:
            indices = self._select_random_indices(n, target.idx, 3)
            r0, r1, r2 = [pop[i].position for i in indices]
            mutant = r0 + f * (r1 - r2)

        elif strategy == MutationStrategy.RAND_2:
            indices = self._select_random_indices(n, target.idx, 5)
            r0, r1, r2, r3, r4 = [pop[i].position for i in indices]
            mutant = r0 + f * (r1 - r2) + f * (r3 - r4)

        elif strategy == MutationStrategy.BEST_1:
            best = pop[self.population.best_idx].position
            indices = self._select_random_indices(n, target.idx, 2)
            r0, r1 = [pop[i].position for i in indices]
            mutant = best + f * (r0 - r1)

        elif strategy == MutationStrategy.BEST_2:
            best = pop[self.population.best_idx].position
            indices = self._select_random_indices(n, target.idx, 4)
            r0, r1, r2, r3 = [pop[i].position for i in indices]
            mutant = best + f * (r0 - r1) + f * (r2 - r3)

        elif strategy == MutationStrategy.CURRENT_TO_BEST_1:
            best = pop[self.population.best_idx].position
            indices = self._select_random_indices(n, target.idx, 2)
            r0, r1 = [pop[i].position for i in indices]
            mutant = target.position + f * (best - target.position) + f * (r0 - r1)

        elif strategy == MutationStrategy.CURRENT_TO_RAND_1:
            indices = self._select_random_indices(n, target.idx, 3)
            r0, r1, r2 = [pop[i].position for i in indices]
            k = self.rng.random()
            mutant = target.position + k * (r0 - target.position) + f * (r1 - r2)

        elif strategy == MutationStrategy.RAND_TO_BEST_1:
            best = pop[self.population.best_idx].position
            indices = self._select_random_indices(n, target.idx, 3)
            r0, r1, r2 = [pop[i].position for i in indices]
            mutant = r0 + f * (best - r0) + f * (r1 - r2)

        else:
            indices = self._select_random_indices(n, target.idx, 3)
            r0, r1, r2 = [pop[i].position for i in indices]
            mutant = r0 + f * (r1 - r2)

        return mutant

    def _select_random_indices(
        self,
        n: int,
        exclude: int,
        count: int,
    ) -> list[int]:
        """Select random indices excluding target."""
        available = [i for i in range(n) if i != exclude]
        return list(self.rng.choice(available, count, replace=False))

    def _crossover(
        self,
        target: np.ndarray,
        mutant: np.ndarray,
    ) -> np.ndarray:
        """
        Perform crossover between target and mutant.

        Args:
            target: Target vector
            mutant: Mutant vector

        Returns:
            Trial vector
        """
        n_dim = len(target)
        cr = self._de_config.crossover_rate

        trial = target.copy()

        if self._de_config.crossover_type == CrossoverType.BINOMIAL:
            j_rand = self.rng.integers(0, n_dim)

            for j in range(n_dim):
                if self.rng.random() < cr or j == j_rand:
                    trial[j] = mutant[j]

        elif self._de_config.crossover_type == CrossoverType.EXPONENTIAL:
            j = self.rng.integers(0, n_dim)
            L = 0

            while self.rng.random() < cr and L < n_dim:
                trial[j] = mutant[j]
                j = (j + 1) % n_dim
                L += 1

        return trial

    def _ensure_bounds(self, vector: np.ndarray) -> np.ndarray:
        """Ensure vector is within bounds."""
        for i, (lo, hi) in enumerate(self._bounds):
            if vector[i] < lo:
                vector[i] = lo + self.rng.random() * (hi - lo) * 0.1
            elif vector[i] > hi:
                vector[i] = hi - self.rng.random() * (hi - lo) * 0.1

            vector[i] = np.clip(vector[i], lo, hi)

        return vector

    def _selection(self) -> None:
        """Perform selection between target and trial."""
        for ind in self.population.individuals:
            if self._is_better_fitness(ind.trial_fitness, ind.fitness):
                ind.position = ind.trial_position.copy()
                ind.fitness = ind.trial_fitness

    def _is_better_fitness(self, new: float, old: float) -> bool:
        """Check if new fitness is better than old."""
        if self.config.optimization_type == OptimizationType.MAXIMIZATION:
            return new > old
        else:
            return new < old

    def _update_best(self) -> None:
        """Update best individual."""
        for i, ind in enumerate(self.population.individuals):
            if self._is_better_fitness(ind.fitness, self.population.best_fitness):
                self.population.best_idx = i
                self.population.best_fitness = ind.fitness

    def _array_to_params(self, array: np.ndarray) -> dict[str, Any]:
        """Convert numpy array to parameters."""
        params = {}
        for i, p in enumerate(self._search_space.continuous_params):
            val = float(array[i])
            if p.is_integer:
                params[p.name] = int(round(val))
            else:
                params[p.name] = val

        for p in self._search_space.categorical_params:
            params[p.name] = p.default_value or p.choices[0]

        return params


class AdaptiveDE(DifferentialEvolution):
    """Self-Adaptive Differential Evolution (SADE/JADE)."""

    def __init__(
        self,
        config: DEConfig | None = None,
    ) -> None:
        """
        Initialize Adaptive DE.

        Args:
            config: DE configuration
        """
        super().__init__(config)

        self._f_memory: list[float] = []
        self._cr_memory: list[float] = []
        self._success_f: list[float] = []
        self._success_cr: list[float] = []
        self._archive: list[np.ndarray] = []
        self._p = 0.1
        self._c = 0.1

    def _initialize(self, search_space: SearchSpace) -> None:
        """Initialize adaptive DE."""
        super()._initialize(search_space)

        self._f_memory = [0.5] * 5
        self._cr_memory = [0.5] * 5
        self._success_f = []
        self._success_cr = []
        self._archive = []

    async def _optimize_step(self) -> list[EvaluationPoint]:
        """Perform adaptive DE step."""
        if not all(ind.is_evaluated for ind in self.population.individuals):
            return await super()._optimize_step()

        self._success_f = []
        self._success_cr = []

        for ind in self.population.individuals:
            f_i = self._sample_f()
            cr_i = self._sample_cr()

            mutant = self._mutate_jade(ind, f_i)
            trial = self._crossover(ind.position, mutant)
            trial = self._ensure_bounds(trial)
            ind.trial_position = trial

            ind._temp_f = f_i
            ind._temp_cr = cr_i

        trial_params = [
            self._array_to_params(ind.trial_position)
            for ind in self.population.individuals
        ]
        trial_points = await self.evaluate_batch(trial_params)

        for ind, point in zip(self.population.individuals, trial_points):
            ind.trial_fitness = point.objective_value

            if self._is_better_fitness(ind.trial_fitness, ind.fitness):
                self._archive.append(ind.position.copy())
                self._success_f.append(ind._temp_f)
                self._success_cr.append(ind._temp_cr)

                ind.position = ind.trial_position.copy()
                ind.fitness = ind.trial_fitness

        max_archive = self._de_config.population_size
        while len(self._archive) > max_archive:
            idx = self.rng.integers(0, len(self._archive))
            self._archive.pop(idx)

        self._update_memories()
        self._update_best()

        self.population.generation += 1

        return trial_points

    def _sample_f(self) -> float:
        """Sample mutation factor F."""
        mu_f = self.rng.choice(self._f_memory)
        f = self.rng.standard_cauchy() * 0.1 + mu_f
        f = np.clip(f, 0, 1)
        return float(f)

    def _sample_cr(self) -> float:
        """Sample crossover rate CR."""
        mu_cr = self.rng.choice(self._cr_memory)
        cr = self.rng.normal(mu_cr, 0.1)
        cr = np.clip(cr, 0, 1)
        return float(cr)

    def _mutate_jade(self, target: DEIndividual, f: float) -> np.ndarray:
        """JADE mutation: current-to-pbest/1."""
        pop = self.population.individuals
        n = len(pop)

        p_count = max(2, int(self._p * n))

        sorted_pop = sorted(
            pop,
            key=lambda x: x.fitness,
            reverse=self.config.optimization_type == OptimizationType.MAXIMIZATION,
        )
        p_best_idx = self.rng.integers(0, p_count)
        p_best = sorted_pop[p_best_idx].position

        available = [i for i in range(n) if i != target.idx]
        r1_idx = self.rng.choice(available)
        r1 = pop[r1_idx].position

        if self._archive:
            combined = [ind.position for ind in pop] + self._archive
            available_combined = list(range(len(combined)))
            available_combined.remove(target.idx)
            if r1_idx < n:
                available_combined.remove(r1_idx)
            r2_idx = self.rng.choice(available_combined)
            r2 = combined[r2_idx]
        else:
            available = [i for i in range(n) if i != target.idx and i != r1_idx]
            r2_idx = self.rng.choice(available)
            r2 = pop[r2_idx].position

        mutant = target.position + f * (p_best - target.position) + f * (r1 - r2)
        return mutant

    def _update_memories(self) -> None:
        """Update F and CR memories."""
        if self._success_f:
            mean_f = np.sum(np.array(self._success_f) ** 2) / np.sum(self._success_f)
            self._f_memory.pop(0)
            self._f_memory.append(float(mean_f))

        if self._success_cr:
            mean_cr = np.mean(self._success_cr)
            self._cr_memory.pop(0)
            self._cr_memory.append(float(mean_cr))


class SHADE(AdaptiveDE):
    """Success-History based Adaptive DE (SHADE)."""

    def __init__(
        self,
        config: DEConfig | None = None,
        memory_size: int = 100,
    ) -> None:
        """
        Initialize SHADE.

        Args:
            config: DE configuration
            memory_size: Size of parameter memory
        """
        super().__init__(config)
        self._memory_size = memory_size
        self._memory_index = 0

    def _initialize(self, search_space: SearchSpace) -> None:
        """Initialize SHADE."""
        super()._initialize(search_space)
        self._f_memory = [0.5] * self._memory_size
        self._cr_memory = [0.5] * self._memory_size
        self._memory_index = 0

    def _update_memories(self) -> None:
        """Update memories using weighted Lehmer mean."""
        if not self._success_f or not self._success_cr:
            return

        success_f = np.array(self._success_f)
        success_cr = np.array(self._success_cr)

        improvements = np.abs(
            [ind.trial_fitness - ind.fitness for ind in self.population.individuals
             if hasattr(ind, '_temp_f')]
        )

        if len(improvements) < len(success_f):
            improvements = np.ones(len(success_f))

        weights = improvements / np.sum(improvements) if np.sum(improvements) > 0 else np.ones(len(success_f)) / len(success_f)

        mean_f = np.sum(weights * success_f ** 2) / np.sum(weights * success_f) if np.sum(weights * success_f) > 0 else 0.5
        mean_cr = np.sum(weights * success_cr)

        self._f_memory[self._memory_index] = float(mean_f)
        self._cr_memory[self._memory_index] = float(mean_cr)

        self._memory_index = (self._memory_index + 1) % self._memory_size


def create_de_optimizer(
    population_size: int = 50,
    max_iterations: int = 100,
    mutation_factor: float = 0.8,
    crossover_rate: float = 0.9,
    mutation_strategy: str = "best/1",
    crossover_type: str = "binomial",
    adaptive: bool = False,
    optimization_type: str = "maximization",
    seed: int | None = None,
) -> DifferentialEvolution:
    """
    Create Differential Evolution optimizer.

    Args:
        population_size: Population size
        max_iterations: Maximum iterations
        mutation_factor: Mutation factor F
        crossover_rate: Crossover rate CR
        mutation_strategy: Mutation strategy
        crossover_type: Crossover type
        adaptive: Use adaptive version
        optimization_type: Optimization type
        seed: Random seed

    Returns:
        Configured DifferentialEvolution optimizer
    """
    config = DEConfig(
        population_size=population_size,
        max_iterations=max_iterations,
        mutation_factor=mutation_factor,
        crossover_rate=crossover_rate,
        mutation_strategy=MutationStrategy(mutation_strategy),
        crossover_type=CrossoverType(crossover_type),
        adaptive=adaptive,
        optimization_type=OptimizationType(optimization_type),
        random_seed=seed,
    )

    if adaptive:
        return SHADE(config)
    else:
        return DifferentialEvolution(config)
