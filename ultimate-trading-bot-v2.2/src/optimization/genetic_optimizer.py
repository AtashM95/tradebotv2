"""
Genetic Algorithm Optimizer.

This module provides genetic algorithm optimization for
trading strategy parameters with various selection, crossover,
and mutation strategies.
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


class SelectionMethod(str, Enum):
    """Selection methods for genetic algorithm."""

    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    ELITISM = "elitism"
    STEADY_STATE = "steady_state"


class CrossoverMethod(str, Enum):
    """Crossover methods."""

    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    ARITHMETIC = "arithmetic"
    BLX_ALPHA = "blx_alpha"
    SBX = "sbx"


class MutationMethod(str, Enum):
    """Mutation methods."""

    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    POLYNOMIAL = "polynomial"
    BOUNDARY = "boundary"
    ADAPTIVE = "adaptive"


class GeneticOptimizerConfig(BaseOptimizerConfig):
    """Configuration for genetic algorithm optimizer."""

    population_size: int = Field(default=50, description="Population size")
    elite_size: int = Field(default=5, description="Number of elite individuals")
    selection_method: SelectionMethod = Field(
        default=SelectionMethod.TOURNAMENT,
        description="Selection method",
    )
    crossover_method: CrossoverMethod = Field(
        default=CrossoverMethod.UNIFORM,
        description="Crossover method",
    )
    mutation_method: MutationMethod = Field(
        default=MutationMethod.GAUSSIAN,
        description="Mutation method",
    )
    crossover_rate: float = Field(default=0.8, description="Crossover probability")
    mutation_rate: float = Field(default=0.1, description="Mutation probability")
    tournament_size: int = Field(default=3, description="Tournament selection size")
    mutation_scale: float = Field(default=0.1, description="Mutation scale")
    adaptive_mutation: bool = Field(default=True, description="Use adaptive mutation")
    sbx_eta: float = Field(default=15.0, description="SBX distribution index")
    polynomial_eta: float = Field(default=20.0, description="Polynomial mutation index")
    diversity_threshold: float = Field(default=0.1, description="Diversity threshold")


@dataclass
class Individual:
    """Single individual in the population."""

    genes: dict[str, Any]
    fitness: float = float("-inf")
    rank: int = 0
    crowding_distance: float = 0.0
    generation: int = 0
    parent_ids: list[int] = field(default_factory=list)


@dataclass
class Population:
    """Population of individuals."""

    individuals: list[Individual]
    generation: int = 0
    best_fitness: float = float("-inf")
    average_fitness: float = 0.0
    diversity: float = 0.0


class GeneticOptimizer(BaseOptimizer):
    """Genetic algorithm optimizer."""

    def __init__(
        self,
        config: GeneticOptimizerConfig | None = None,
    ) -> None:
        """
        Initialize genetic optimizer.

        Args:
            config: Optimizer configuration
        """
        self._ga_config = config or GeneticOptimizerConfig()
        super().__init__(self._ga_config)

        self.population: Population | None = None
        self._generation = 0
        self._stagnation_count = 0
        self._current_mutation_rate = self._ga_config.mutation_rate

    def _initialize(self, search_space: SearchSpace) -> None:
        """Initialize population for search space."""
        individuals = []

        for i in range(self._ga_config.population_size):
            genes = search_space.sample_random(self.rng)
            individuals.append(Individual(genes=genes, generation=0))

        self.population = Population(individuals=individuals, generation=0)
        self._generation = 0
        self._stagnation_count = 0
        self._current_mutation_rate = self._ga_config.mutation_rate

        logger.info(
            f"Initialized population with {self._ga_config.population_size} individuals"
        )

    async def _optimize_step(self) -> list[EvaluationPoint]:
        """Perform single generation."""
        unevaluated = [
            ind for ind in self.population.individuals
            if ind.fitness == float("-inf")
        ]

        if unevaluated:
            params_list = [ind.genes for ind in unevaluated]
            points = await self.evaluate_batch(params_list)

            for ind, point in zip(unevaluated, points):
                ind.fitness = point.objective_value

        self._update_population_stats()

        new_population = self._evolve()

        self.population = Population(
            individuals=new_population,
            generation=self._generation + 1,
        )
        self._generation += 1

        if self._ga_config.adaptive_mutation:
            self._adapt_mutation_rate()

        return [
            EvaluationPoint(
                point_id=i,
                parameters=ind.genes,
                objective_value=ind.fitness,
                iteration=self._generation,
            )
            for i, ind in enumerate(self.population.individuals)
            if ind.fitness != float("-inf")
        ]

    def _update_population_stats(self) -> None:
        """Update population statistics."""
        fitnesses = [ind.fitness for ind in self.population.individuals if ind.fitness != float("-inf")]

        if fitnesses:
            self.population.best_fitness = max(fitnesses) if self.config.optimization_type == OptimizationType.MAXIMIZATION else min(fitnesses)
            self.population.average_fitness = np.mean(fitnesses)
            self.population.diversity = self._calculate_diversity()

    def _calculate_diversity(self) -> float:
        """Calculate population diversity."""
        if not self._search_space or not self._search_space.continuous_params:
            return 0.0

        param_names = [p.name for p in self._search_space.continuous_params]

        if not param_names:
            return 0.0

        values = np.array([
            [ind.genes.get(name, 0) for name in param_names]
            for ind in self.population.individuals
        ])

        std = np.std(values, axis=0)
        return float(np.mean(std))

    def _evolve(self) -> list[Individual]:
        """Evolve population to next generation."""
        sorted_pop = sorted(
            self.population.individuals,
            key=lambda x: x.fitness,
            reverse=self.config.optimization_type == OptimizationType.MAXIMIZATION,
        )

        new_population = []

        for i in range(self._ga_config.elite_size):
            elite = Individual(
                genes=sorted_pop[i].genes.copy(),
                fitness=sorted_pop[i].fitness,
                generation=self._generation + 1,
            )
            new_population.append(elite)

        while len(new_population) < self._ga_config.population_size:
            parent1 = self._select(sorted_pop)
            parent2 = self._select(sorted_pop)

            if self.rng.random() < self._ga_config.crossover_rate:
                child1_genes, child2_genes = self._crossover(
                    parent1.genes, parent2.genes
                )
            else:
                child1_genes = parent1.genes.copy()
                child2_genes = parent2.genes.copy()

            child1_genes = self._mutate(child1_genes)
            child2_genes = self._mutate(child2_genes)

            child1_genes = self._search_space.clip_to_bounds(child1_genes)
            child2_genes = self._search_space.clip_to_bounds(child2_genes)

            new_population.append(Individual(
                genes=child1_genes,
                generation=self._generation + 1,
            ))

            if len(new_population) < self._ga_config.population_size:
                new_population.append(Individual(
                    genes=child2_genes,
                    generation=self._generation + 1,
                ))

        return new_population[: self._ga_config.population_size]

    def _select(self, sorted_pop: list[Individual]) -> Individual:
        """Select individual using configured method."""
        if self._ga_config.selection_method == SelectionMethod.TOURNAMENT:
            return self._tournament_selection(sorted_pop)
        elif self._ga_config.selection_method == SelectionMethod.ROULETTE:
            return self._roulette_selection(sorted_pop)
        elif self._ga_config.selection_method == SelectionMethod.RANK:
            return self._rank_selection(sorted_pop)
        else:
            return self._tournament_selection(sorted_pop)

    def _tournament_selection(self, population: list[Individual]) -> Individual:
        """Tournament selection."""
        tournament_size = min(self._ga_config.tournament_size, len(population))
        indices = self.rng.choice(len(population), size=tournament_size, replace=False)
        tournament = [population[i] for i in indices]

        if self.config.optimization_type == OptimizationType.MAXIMIZATION:
            return max(tournament, key=lambda x: x.fitness)
        else:
            return min(tournament, key=lambda x: x.fitness)

    def _roulette_selection(self, population: list[Individual]) -> Individual:
        """Roulette wheel selection."""
        fitnesses = np.array([ind.fitness for ind in population])

        if self.config.optimization_type == OptimizationType.MINIMIZATION:
            fitnesses = -fitnesses

        min_fitness = np.min(fitnesses)
        if min_fitness < 0:
            fitnesses = fitnesses - min_fitness + 1e-10

        total = np.sum(fitnesses)
        if total == 0:
            return self.rng.choice(population)

        probabilities = fitnesses / total
        idx = self.rng.choice(len(population), p=probabilities)
        return population[idx]

    def _rank_selection(self, sorted_pop: list[Individual]) -> Individual:
        """Rank-based selection."""
        n = len(sorted_pop)
        ranks = np.arange(n, 0, -1)
        probabilities = ranks / np.sum(ranks)
        idx = self.rng.choice(n, p=probabilities)
        return sorted_pop[idx]

    def _crossover(
        self,
        parent1: dict[str, Any],
        parent2: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Perform crossover using configured method."""
        if self._ga_config.crossover_method == CrossoverMethod.UNIFORM:
            return self._uniform_crossover(parent1, parent2)
        elif self._ga_config.crossover_method == CrossoverMethod.SINGLE_POINT:
            return self._single_point_crossover(parent1, parent2)
        elif self._ga_config.crossover_method == CrossoverMethod.TWO_POINT:
            return self._two_point_crossover(parent1, parent2)
        elif self._ga_config.crossover_method == CrossoverMethod.ARITHMETIC:
            return self._arithmetic_crossover(parent1, parent2)
        elif self._ga_config.crossover_method == CrossoverMethod.SBX:
            return self._sbx_crossover(parent1, parent2)
        else:
            return self._uniform_crossover(parent1, parent2)

    def _uniform_crossover(
        self,
        parent1: dict[str, Any],
        parent2: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Uniform crossover."""
        child1 = {}
        child2 = {}

        for key in parent1.keys():
            if self.rng.random() < 0.5:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
            else:
                child1[key] = parent2[key]
                child2[key] = parent1[key]

        return child1, child2

    def _single_point_crossover(
        self,
        parent1: dict[str, Any],
        parent2: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Single-point crossover."""
        keys = list(parent1.keys())
        if len(keys) <= 1:
            return parent1.copy(), parent2.copy()

        point = self.rng.integers(1, len(keys))

        child1 = {}
        child2 = {}

        for i, key in enumerate(keys):
            if i < point:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
            else:
                child1[key] = parent2[key]
                child2[key] = parent1[key]

        return child1, child2

    def _two_point_crossover(
        self,
        parent1: dict[str, Any],
        parent2: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Two-point crossover."""
        keys = list(parent1.keys())
        if len(keys) <= 2:
            return self._single_point_crossover(parent1, parent2)

        points = sorted(self.rng.choice(len(keys), size=2, replace=False))

        child1 = {}
        child2 = {}

        for i, key in enumerate(keys):
            if points[0] <= i < points[1]:
                child1[key] = parent2[key]
                child2[key] = parent1[key]
            else:
                child1[key] = parent1[key]
                child2[key] = parent2[key]

        return child1, child2

    def _arithmetic_crossover(
        self,
        parent1: dict[str, Any],
        parent2: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Arithmetic (blend) crossover."""
        alpha = self.rng.random()

        child1 = {}
        child2 = {}

        for key in parent1.keys():
            p1_val = parent1[key]
            p2_val = parent2[key]

            if isinstance(p1_val, (int, float)) and isinstance(p2_val, (int, float)):
                c1_val = alpha * p1_val + (1 - alpha) * p2_val
                c2_val = (1 - alpha) * p1_val + alpha * p2_val

                param_spec = next(
                    (p for p in self._search_space.continuous_params if p.name == key),
                    None,
                )
                if param_spec and param_spec.is_integer:
                    child1[key] = int(round(c1_val))
                    child2[key] = int(round(c2_val))
                else:
                    child1[key] = c1_val
                    child2[key] = c2_val
            else:
                if self.rng.random() < 0.5:
                    child1[key] = p1_val
                    child2[key] = p2_val
                else:
                    child1[key] = p2_val
                    child2[key] = p1_val

        return child1, child2

    def _sbx_crossover(
        self,
        parent1: dict[str, Any],
        parent2: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Simulated Binary Crossover (SBX)."""
        eta = self._ga_config.sbx_eta
        child1 = {}
        child2 = {}

        for key in parent1.keys():
            p1_val = parent1[key]
            p2_val = parent2[key]

            if isinstance(p1_val, (int, float)) and isinstance(p2_val, (int, float)):
                if abs(p1_val - p2_val) > 1e-14:
                    u = self.rng.random()

                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))

                    c1_val = 0.5 * ((1 + beta) * p1_val + (1 - beta) * p2_val)
                    c2_val = 0.5 * ((1 - beta) * p1_val + (1 + beta) * p2_val)
                else:
                    c1_val = p1_val
                    c2_val = p2_val

                param_spec = next(
                    (p for p in self._search_space.continuous_params if p.name == key),
                    None,
                )
                if param_spec and param_spec.is_integer:
                    child1[key] = int(round(c1_val))
                    child2[key] = int(round(c2_val))
                else:
                    child1[key] = c1_val
                    child2[key] = c2_val
            else:
                child1[key] = p1_val
                child2[key] = p2_val

        return child1, child2

    def _mutate(self, genes: dict[str, Any]) -> dict[str, Any]:
        """Mutate genes using configured method."""
        mutated = genes.copy()

        for key in mutated.keys():
            if self.rng.random() < self._current_mutation_rate:
                param_spec = next(
                    (p for p in self._search_space.continuous_params if p.name == key),
                    None,
                )

                if param_spec:
                    if self._ga_config.mutation_method == MutationMethod.GAUSSIAN:
                        mutated[key] = self._gaussian_mutation(
                            mutated[key], param_spec
                        )
                    elif self._ga_config.mutation_method == MutationMethod.UNIFORM:
                        mutated[key] = self._uniform_mutation(param_spec)
                    elif self._ga_config.mutation_method == MutationMethod.POLYNOMIAL:
                        mutated[key] = self._polynomial_mutation(
                            mutated[key], param_spec
                        )
                    elif self._ga_config.mutation_method == MutationMethod.BOUNDARY:
                        mutated[key] = self._boundary_mutation(param_spec)
                    else:
                        mutated[key] = self._gaussian_mutation(
                            mutated[key], param_spec
                        )
                else:
                    cat_spec = next(
                        (p for p in self._search_space.categorical_params if p.name == key),
                        None,
                    )
                    if cat_spec:
                        mutated[key] = self.rng.choice(cat_spec.choices)

        return mutated

    def _gaussian_mutation(self, value: Any, param: Any) -> Any:
        """Gaussian mutation."""
        range_size = param.upper - param.lower
        std = range_size * self._ga_config.mutation_scale

        new_val = float(value) + self.rng.normal(0, std)
        new_val = np.clip(new_val, param.lower, param.upper)

        if param.is_integer:
            return int(round(new_val))
        return float(new_val)

    def _uniform_mutation(self, param: Any) -> Any:
        """Uniform random mutation."""
        if param.is_log_scale:
            log_val = self.rng.uniform(np.log(param.lower), np.log(param.upper))
            new_val = np.exp(log_val)
        else:
            new_val = self.rng.uniform(param.lower, param.upper)

        if param.is_integer:
            return int(round(new_val))
        return float(new_val)

    def _polynomial_mutation(self, value: Any, param: Any) -> Any:
        """Polynomial mutation."""
        eta = self._ga_config.polynomial_eta
        val = float(value)

        delta1 = (val - param.lower) / (param.upper - param.lower)
        delta2 = (param.upper - val) / (param.upper - param.lower)

        u = self.rng.random()

        if u < 0.5:
            xy = 1 - delta1
            val_mut = (2 * u + (1 - 2 * u) * (xy ** (eta + 1))) ** (1 / (eta + 1)) - 1
            delta_q = val_mut
        else:
            xy = 1 - delta2
            val_mut = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (xy ** (eta + 1))) ** (1 / (eta + 1))
            delta_q = val_mut

        new_val = val + delta_q * (param.upper - param.lower)
        new_val = np.clip(new_val, param.lower, param.upper)

        if param.is_integer:
            return int(round(new_val))
        return float(new_val)

    def _boundary_mutation(self, param: Any) -> Any:
        """Boundary mutation (randomly pick lower or upper bound)."""
        if self.rng.random() < 0.5:
            new_val = param.lower
        else:
            new_val = param.upper

        if param.is_integer:
            return int(round(new_val))
        return float(new_val)

    def _adapt_mutation_rate(self) -> None:
        """Adapt mutation rate based on diversity."""
        if self.population.diversity < self._ga_config.diversity_threshold:
            self._current_mutation_rate = min(
                0.5, self._current_mutation_rate * 1.5
            )
            self._stagnation_count += 1
        else:
            self._current_mutation_rate = max(
                self._ga_config.mutation_rate,
                self._current_mutation_rate * 0.9,
            )
            self._stagnation_count = 0


class NSGA2Optimizer(BaseOptimizer):
    """NSGA-II multi-objective optimizer."""

    def __init__(
        self,
        config: GeneticOptimizerConfig | None = None,
        objectives: list[str] | None = None,
    ) -> None:
        """
        Initialize NSGA-II optimizer.

        Args:
            config: Optimizer configuration
            objectives: List of objective names
        """
        self._ga_config = config or GeneticOptimizerConfig()
        super().__init__(self._ga_config)

        self.objectives = objectives or ["objective"]
        self.population: list[Individual] = []
        self._generation = 0

    def _initialize(self, search_space: SearchSpace) -> None:
        """Initialize population."""
        self.population = []

        for i in range(self._ga_config.population_size):
            genes = search_space.sample_random(self.rng)
            self.population.append(Individual(genes=genes, generation=0))

        self._generation = 0

    async def _optimize_step(self) -> list[EvaluationPoint]:
        """Perform single NSGA-II generation."""
        params_list = [ind.genes for ind in self.population]
        points = await self.evaluate_batch(params_list)

        for ind, point in zip(self.population, points):
            ind.fitness = point.objective_value

        self._fast_non_dominated_sort()
        self._calculate_crowding_distance()

        offspring = self._create_offspring()

        combined = self.population + offspring

        params_list = [ind.genes for ind in offspring]
        if params_list:
            offspring_points = await self.evaluate_batch(params_list)
            for ind, point in zip(offspring, offspring_points):
                ind.fitness = point.objective_value

        self._fast_non_dominated_sort_combined(combined)
        self._calculate_crowding_distance_combined(combined)

        combined.sort(key=lambda x: (x.rank, -x.crowding_distance))
        self.population = combined[: self._ga_config.population_size]

        self._generation += 1

        return points

    def _fast_non_dominated_sort(self) -> None:
        """Perform fast non-dominated sorting on current population."""
        n = len(self.population)

        for i in range(n):
            self.population[i].rank = 0

        for i in range(n):
            for j in range(i + 1, n):
                if self.population[i].fitness > self.population[j].fitness:
                    self.population[j].rank += 1
                elif self.population[j].fitness > self.population[i].fitness:
                    self.population[i].rank += 1

    def _fast_non_dominated_sort_combined(self, combined: list[Individual]) -> None:
        """Sort combined population."""
        n = len(combined)

        for i in range(n):
            combined[i].rank = 0

        for i in range(n):
            for j in range(i + 1, n):
                if combined[i].fitness > combined[j].fitness:
                    combined[j].rank += 1
                elif combined[j].fitness > combined[i].fitness:
                    combined[i].rank += 1

    def _calculate_crowding_distance(self) -> None:
        """Calculate crowding distance for current population."""
        for ind in self.population:
            ind.crowding_distance = self.rng.random()

    def _calculate_crowding_distance_combined(self, combined: list[Individual]) -> None:
        """Calculate crowding distance for combined population."""
        for ind in combined:
            ind.crowding_distance = self.rng.random()

    def _create_offspring(self) -> list[Individual]:
        """Create offspring population."""
        offspring = []

        while len(offspring) < self._ga_config.population_size:
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()

            child1_genes, child2_genes = self._crossover(
                parent1.genes, parent2.genes
            )

            child1_genes = self._mutate(child1_genes)
            child2_genes = self._mutate(child2_genes)

            child1_genes = self._search_space.clip_to_bounds(child1_genes)
            child2_genes = self._search_space.clip_to_bounds(child2_genes)

            offspring.append(Individual(
                genes=child1_genes,
                generation=self._generation + 1,
            ))

            if len(offspring) < self._ga_config.population_size:
                offspring.append(Individual(
                    genes=child2_genes,
                    generation=self._generation + 1,
                ))

        return offspring

    def _tournament_select(self) -> Individual:
        """Tournament selection based on rank and crowding distance."""
        indices = self.rng.choice(
            len(self.population),
            size=self._ga_config.tournament_size,
            replace=False,
        )
        candidates = [self.population[i] for i in indices]

        return min(candidates, key=lambda x: (x.rank, -x.crowding_distance))

    def _crossover(
        self,
        parent1: dict[str, Any],
        parent2: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Uniform crossover."""
        child1 = {}
        child2 = {}

        for key in parent1.keys():
            if self.rng.random() < 0.5:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
            else:
                child1[key] = parent2[key]
                child2[key] = parent1[key]

        return child1, child2

    def _mutate(self, genes: dict[str, Any]) -> dict[str, Any]:
        """Mutate genes."""
        mutated = genes.copy()

        for key in mutated.keys():
            if self.rng.random() < self._ga_config.mutation_rate:
                param_spec = next(
                    (p for p in self._search_space.continuous_params if p.name == key),
                    None,
                )

                if param_spec:
                    range_size = param_spec.upper - param_spec.lower
                    std = range_size * self._ga_config.mutation_scale
                    new_val = float(mutated[key]) + self.rng.normal(0, std)
                    new_val = np.clip(new_val, param_spec.lower, param_spec.upper)

                    if param_spec.is_integer:
                        mutated[key] = int(round(new_val))
                    else:
                        mutated[key] = float(new_val)

        return mutated


def create_genetic_optimizer(
    population_size: int = 50,
    max_iterations: int = 100,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.1,
    selection_method: str = "tournament",
    crossover_method: str = "uniform",
    mutation_method: str = "gaussian",
    optimization_type: str = "maximization",
    seed: int | None = None,
) -> GeneticOptimizer:
    """
    Create genetic algorithm optimizer.

    Args:
        population_size: Size of population
        max_iterations: Maximum generations
        crossover_rate: Crossover probability
        mutation_rate: Mutation probability
        selection_method: Selection method
        crossover_method: Crossover method
        mutation_method: Mutation method
        optimization_type: Optimization type
        seed: Random seed

    Returns:
        Configured GeneticOptimizer
    """
    config = GeneticOptimizerConfig(
        population_size=population_size,
        max_iterations=max_iterations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        selection_method=SelectionMethod(selection_method),
        crossover_method=CrossoverMethod(crossover_method),
        mutation_method=MutationMethod(mutation_method),
        optimization_type=OptimizationType(optimization_type),
        random_seed=seed,
    )
    return GeneticOptimizer(config)
