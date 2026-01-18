"""
Particle Swarm Optimization (PSO).

This module provides Particle Swarm Optimization for
trading strategy parameter tuning.
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


class TopologyType(str, Enum):
    """PSO topology types."""

    GLOBAL = "global"
    RING = "ring"
    RANDOM = "random"
    VON_NEUMANN = "von_neumann"


class VelocityUpdate(str, Enum):
    """Velocity update strategies."""

    STANDARD = "standard"
    CONSTRICTION = "constriction"
    INERTIA_DECAY = "inertia_decay"
    ADAPTIVE = "adaptive"


class PSOConfig(BaseOptimizerConfig):
    """Configuration for Particle Swarm Optimization."""

    swarm_size: int = Field(default=30, description="Number of particles")
    inertia_weight: float = Field(default=0.7, description="Inertia weight (w)")
    cognitive_weight: float = Field(default=1.5, description="Cognitive weight (c1)")
    social_weight: float = Field(default=1.5, description="Social weight (c2)")
    max_velocity: float | None = Field(default=None, description="Maximum velocity")
    min_inertia: float = Field(default=0.4, description="Minimum inertia weight")
    max_inertia: float = Field(default=0.9, description="Maximum inertia weight")
    topology: TopologyType = Field(default=TopologyType.GLOBAL, description="Network topology")
    velocity_update: VelocityUpdate = Field(
        default=VelocityUpdate.INERTIA_DECAY,
        description="Velocity update strategy",
    )
    neighborhood_size: int = Field(default=3, description="Neighborhood size for local topologies")
    constriction_factor: float = Field(default=0.729, description="Constriction factor (chi)")


@dataclass
class Particle:
    """Single particle in the swarm."""

    particle_id: int
    position: np.ndarray
    velocity: np.ndarray
    fitness: float = float("-inf")
    best_position: np.ndarray = field(default_factory=lambda: np.array([]))
    best_fitness: float = float("-inf")
    neighbors: list[int] = field(default_factory=list)


@dataclass
class Swarm:
    """Swarm of particles."""

    particles: list[Particle]
    global_best_position: np.ndarray = field(default_factory=lambda: np.array([]))
    global_best_fitness: float = float("-inf")
    iteration: int = 0


class ParticleSwarmOptimizer(BaseOptimizer):
    """Particle Swarm Optimizer."""

    def __init__(
        self,
        config: PSOConfig | None = None,
    ) -> None:
        """
        Initialize PSO optimizer.

        Args:
            config: PSO configuration
        """
        self._pso_config = config or PSOConfig()
        super().__init__(self._pso_config)

        self.swarm: Swarm | None = None
        self._bounds: np.ndarray | None = None
        self._current_inertia: float = self._pso_config.max_inertia

    def _initialize(self, search_space: SearchSpace) -> None:
        """Initialize swarm for search space."""
        self._bounds = search_space.get_bounds_array()
        n_dim = len(self._bounds)

        if n_dim == 0:
            logger.warning("No continuous parameters in search space")
            return

        particles = []

        for i in range(self._pso_config.swarm_size):
            position = np.array([
                self.rng.uniform(lo, hi) for lo, hi in self._bounds
            ])

            velocity_range = (self._bounds[:, 1] - self._bounds[:, 0]) / 4
            velocity = self.rng.uniform(-velocity_range, velocity_range)

            particle = Particle(
                particle_id=i,
                position=position,
                velocity=velocity,
                best_position=position.copy(),
            )

            particles.append(particle)

        self.swarm = Swarm(particles=particles)

        self._setup_topology()

        self._current_inertia = self._pso_config.max_inertia

        logger.info(
            f"PSO initialized with {self._pso_config.swarm_size} particles, "
            f"{n_dim} dimensions"
        )

    def _setup_topology(self) -> None:
        """Setup network topology for the swarm."""
        n = len(self.swarm.particles)

        if self._pso_config.topology == TopologyType.GLOBAL:
            for particle in self.swarm.particles:
                particle.neighbors = list(range(n))

        elif self._pso_config.topology == TopologyType.RING:
            for i, particle in enumerate(self.swarm.particles):
                k = self._pso_config.neighborhood_size // 2
                particle.neighbors = [
                    (i + j) % n for j in range(-k, k + 1)
                ]

        elif self._pso_config.topology == TopologyType.RANDOM:
            for i, particle in enumerate(self.swarm.particles):
                others = [j for j in range(n) if j != i]
                k = min(self._pso_config.neighborhood_size, len(others))
                particle.neighbors = [i] + list(self.rng.choice(others, k, replace=False))

        elif self._pso_config.topology == TopologyType.VON_NEUMANN:
            side = int(np.sqrt(n))
            for i, particle in enumerate(self.swarm.particles):
                row, col = i // side, i % side
                neighbors = [i]
                if row > 0:
                    neighbors.append((row - 1) * side + col)
                if row < side - 1:
                    neighbors.append((row + 1) * side + col)
                if col > 0:
                    neighbors.append(row * side + col - 1)
                if col < side - 1:
                    neighbors.append(row * side + col + 1)
                particle.neighbors = neighbors

    async def _optimize_step(self) -> list[EvaluationPoint]:
        """Perform single PSO iteration."""
        params_list = [
            self._array_to_params(p.position) for p in self.swarm.particles
        ]
        points = await self.evaluate_batch(params_list)

        for particle, point in zip(self.swarm.particles, points):
            particle.fitness = point.objective_value

            if self._is_better_fitness(particle.fitness, particle.best_fitness):
                particle.best_fitness = particle.fitness
                particle.best_position = particle.position.copy()

        for particle in self.swarm.particles:
            neighbor_best = self._get_neighborhood_best(particle)

            if self._is_better_fitness(neighbor_best.best_fitness, self.swarm.global_best_fitness):
                self.swarm.global_best_fitness = neighbor_best.best_fitness
                self.swarm.global_best_position = neighbor_best.best_position.copy()

        for particle in self.swarm.particles:
            self._update_velocity(particle)
            self._update_position(particle)

        self._update_inertia()

        self.swarm.iteration += 1

        return points

    def _is_better_fitness(self, new: float, old: float) -> bool:
        """Check if new fitness is better than old."""
        if self.config.optimization_type == OptimizationType.MAXIMIZATION:
            return new > old
        else:
            return new < old

    def _get_neighborhood_best(self, particle: Particle) -> Particle:
        """Get best particle in neighborhood."""
        neighbors = [self.swarm.particles[i] for i in particle.neighbors]

        if self.config.optimization_type == OptimizationType.MAXIMIZATION:
            return max(neighbors, key=lambda p: p.best_fitness)
        else:
            return min(neighbors, key=lambda p: p.best_fitness)

    def _update_velocity(self, particle: Particle) -> None:
        """Update particle velocity."""
        r1 = self.rng.random(len(particle.position))
        r2 = self.rng.random(len(particle.position))

        cognitive = (
            self._pso_config.cognitive_weight
            * r1
            * (particle.best_position - particle.position)
        )

        social = (
            self._pso_config.social_weight
            * r2
            * (self.swarm.global_best_position - particle.position)
        )

        if self._pso_config.velocity_update == VelocityUpdate.STANDARD:
            particle.velocity = (
                self._current_inertia * particle.velocity + cognitive + social
            )

        elif self._pso_config.velocity_update == VelocityUpdate.CONSTRICTION:
            chi = self._pso_config.constriction_factor
            particle.velocity = chi * (particle.velocity + cognitive + social)

        elif self._pso_config.velocity_update == VelocityUpdate.INERTIA_DECAY:
            particle.velocity = (
                self._current_inertia * particle.velocity + cognitive + social
            )

        elif self._pso_config.velocity_update == VelocityUpdate.ADAPTIVE:
            self._adaptive_velocity_update(particle, cognitive, social)

        if self._pso_config.max_velocity is not None:
            speed = np.linalg.norm(particle.velocity)
            if speed > self._pso_config.max_velocity:
                particle.velocity = (
                    particle.velocity / speed * self._pso_config.max_velocity
                )

    def _adaptive_velocity_update(
        self,
        particle: Particle,
        cognitive: np.ndarray,
        social: np.ndarray,
    ) -> None:
        """Adaptive velocity update based on fitness."""
        if self._is_better_fitness(particle.fitness, particle.best_fitness * 0.95):
            inertia = self._current_inertia * 1.1
        else:
            inertia = self._current_inertia * 0.9

        inertia = np.clip(
            inertia,
            self._pso_config.min_inertia,
            self._pso_config.max_inertia,
        )

        particle.velocity = inertia * particle.velocity + cognitive + social

    def _update_position(self, particle: Particle) -> None:
        """Update particle position."""
        particle.position = particle.position + particle.velocity

        for i, (lo, hi) in enumerate(self._bounds):
            if particle.position[i] < lo:
                particle.position[i] = lo
                particle.velocity[i] *= -0.5
            elif particle.position[i] > hi:
                particle.position[i] = hi
                particle.velocity[i] *= -0.5

    def _update_inertia(self) -> None:
        """Update inertia weight."""
        if self._pso_config.velocity_update == VelocityUpdate.INERTIA_DECAY:
            progress = self.current_iteration / self.config.max_iterations
            self._current_inertia = (
                self._pso_config.max_inertia
                - progress * (self._pso_config.max_inertia - self._pso_config.min_inertia)
            )

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


class AdaptivePSO(ParticleSwarmOptimizer):
    """Adaptive PSO with self-tuning parameters."""

    def __init__(
        self,
        config: PSOConfig | None = None,
    ) -> None:
        """
        Initialize Adaptive PSO.

        Args:
            config: PSO configuration
        """
        super().__init__(config)
        self._stagnation_count = 0
        self._last_best_fitness = float("-inf")
        self._diversity_history: list[float] = []

    def _initialize(self, search_space: SearchSpace) -> None:
        """Initialize adaptive PSO."""
        super()._initialize(search_space)
        self._stagnation_count = 0
        self._last_best_fitness = float("-inf")
        self._diversity_history = []

    async def _optimize_step(self) -> list[EvaluationPoint]:
        """Perform adaptive PSO step."""
        points = await super()._optimize_step()

        self._adapt_parameters()

        return points

    def _adapt_parameters(self) -> None:
        """Adapt PSO parameters based on swarm behavior."""
        diversity = self._calculate_diversity()
        self._diversity_history.append(diversity)

        if self.swarm.global_best_fitness == self._last_best_fitness:
            self._stagnation_count += 1
        else:
            self._stagnation_count = 0
            self._last_best_fitness = self.swarm.global_best_fitness

        if self._stagnation_count > 10:
            self._pso_config.cognitive_weight = min(2.5, self._pso_config.cognitive_weight + 0.1)
            self._pso_config.social_weight = max(0.5, self._pso_config.social_weight - 0.1)
        else:
            self._pso_config.cognitive_weight = 1.5
            self._pso_config.social_weight = 1.5

        if diversity < 0.1:
            self._reinitialize_worst_particles()

    def _calculate_diversity(self) -> float:
        """Calculate swarm diversity."""
        positions = np.array([p.position for p in self.swarm.particles])
        centroid = np.mean(positions, axis=0)

        distances = np.linalg.norm(positions - centroid, axis=1)
        avg_distance = np.mean(distances)

        diagonal = np.linalg.norm(self._bounds[:, 1] - self._bounds[:, 0])
        normalized_diversity = avg_distance / diagonal if diagonal > 0 else 0

        return float(normalized_diversity)

    def _reinitialize_worst_particles(self) -> None:
        """Reinitialize worst performing particles."""
        sorted_particles = sorted(
            self.swarm.particles,
            key=lambda p: p.fitness,
            reverse=self.config.optimization_type == OptimizationType.MAXIMIZATION,
        )

        n_reinit = max(1, len(sorted_particles) // 5)

        for particle in sorted_particles[-n_reinit:]:
            particle.position = np.array([
                self.rng.uniform(lo, hi) for lo, hi in self._bounds
            ])

            velocity_range = (self._bounds[:, 1] - self._bounds[:, 0]) / 4
            particle.velocity = self.rng.uniform(-velocity_range, velocity_range)

            particle.fitness = float("-inf")


class CompetitivePSO(ParticleSwarmOptimizer):
    """Competitive PSO with multiple swarms."""

    def __init__(
        self,
        config: PSOConfig | None = None,
        num_swarms: int = 3,
    ) -> None:
        """
        Initialize Competitive PSO.

        Args:
            config: PSO configuration
            num_swarms: Number of competing swarms
        """
        super().__init__(config)
        self.num_swarms = num_swarms
        self.swarms: list[Swarm] = []

    def _initialize(self, search_space: SearchSpace) -> None:
        """Initialize multiple swarms."""
        self._bounds = search_space.get_bounds_array()
        n_dim = len(self._bounds)

        particles_per_swarm = self._pso_config.swarm_size // self.num_swarms

        self.swarms = []

        for s in range(self.num_swarms):
            particles = []

            for i in range(particles_per_swarm):
                position = np.array([
                    self.rng.uniform(lo, hi) for lo, hi in self._bounds
                ])

                velocity_range = (self._bounds[:, 1] - self._bounds[:, 0]) / 4
                velocity = self.rng.uniform(-velocity_range, velocity_range)

                particle = Particle(
                    particle_id=s * particles_per_swarm + i,
                    position=position,
                    velocity=velocity,
                    best_position=position.copy(),
                )

                particles.append(particle)

            swarm = Swarm(particles=particles)
            self.swarms.append(swarm)

        self.swarm = self.swarms[0]

        logger.info(
            f"Competitive PSO initialized with {self.num_swarms} swarms, "
            f"{particles_per_swarm} particles each"
        )

    async def _optimize_step(self) -> list[EvaluationPoint]:
        """Perform competitive PSO step."""
        all_points = []

        for swarm in self.swarms:
            self.swarm = swarm

            params_list = [
                self._array_to_params(p.position) for p in swarm.particles
            ]
            points = await self.evaluate_batch(params_list)
            all_points.extend(points)

            for particle, point in zip(swarm.particles, points):
                particle.fitness = point.objective_value

                if self._is_better_fitness(particle.fitness, particle.best_fitness):
                    particle.best_fitness = particle.fitness
                    particle.best_position = particle.position.copy()

                if self._is_better_fitness(particle.fitness, swarm.global_best_fitness):
                    swarm.global_best_fitness = particle.fitness
                    swarm.global_best_position = particle.position.copy()

            for particle in swarm.particles:
                self._update_velocity(particle)
                self._update_position(particle)

        self._compete_swarms()

        return all_points

    def _compete_swarms(self) -> None:
        """Competition between swarms."""
        sorted_swarms = sorted(
            self.swarms,
            key=lambda s: s.global_best_fitness,
            reverse=self.config.optimization_type == OptimizationType.MAXIMIZATION,
        )

        best_swarm = sorted_swarms[0]
        worst_swarm = sorted_swarms[-1]

        sorted_worst = sorted(
            worst_swarm.particles,
            key=lambda p: p.best_fitness,
            reverse=self.config.optimization_type != OptimizationType.MAXIMIZATION,
        )

        n_migrate = max(1, len(worst_swarm.particles) // 4)

        for particle in sorted_worst[:n_migrate]:
            particle.position = best_swarm.global_best_position.copy()
            particle.position += self.rng.normal(0, 0.1, len(particle.position))

            for i, (lo, hi) in enumerate(self._bounds):
                particle.position[i] = np.clip(particle.position[i], lo, hi)

            particle.velocity *= 0.5
            particle.fitness = float("-inf")


def create_pso_optimizer(
    swarm_size: int = 30,
    max_iterations: int = 100,
    inertia_weight: float = 0.7,
    cognitive_weight: float = 1.5,
    social_weight: float = 1.5,
    topology: str = "global",
    velocity_update: str = "inertia_decay",
    optimization_type: str = "maximization",
    seed: int | None = None,
) -> ParticleSwarmOptimizer:
    """
    Create Particle Swarm Optimizer.

    Args:
        swarm_size: Number of particles
        max_iterations: Maximum iterations
        inertia_weight: Inertia weight
        cognitive_weight: Cognitive weight
        social_weight: Social weight
        topology: Network topology
        velocity_update: Velocity update strategy
        optimization_type: Optimization type
        seed: Random seed

    Returns:
        Configured ParticleSwarmOptimizer
    """
    config = PSOConfig(
        swarm_size=swarm_size,
        max_iterations=max_iterations,
        inertia_weight=inertia_weight,
        cognitive_weight=cognitive_weight,
        social_weight=social_weight,
        topology=TopologyType(topology),
        velocity_update=VelocityUpdate(velocity_update),
        optimization_type=OptimizationType(optimization_type),
        random_seed=seed,
    )
    return ParticleSwarmOptimizer(config)
