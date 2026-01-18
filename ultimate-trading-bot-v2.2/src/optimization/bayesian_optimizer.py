"""
Bayesian Optimizer.

This module provides Bayesian optimization using Gaussian Process
regression for efficient hyperparameter optimization.
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


class AcquisitionFunction(str, Enum):
    """Acquisition functions for Bayesian optimization."""

    EXPECTED_IMPROVEMENT = "expected_improvement"
    PROBABILITY_OF_IMPROVEMENT = "probability_of_improvement"
    UPPER_CONFIDENCE_BOUND = "upper_confidence_bound"
    LOWER_CONFIDENCE_BOUND = "lower_confidence_bound"
    THOMPSON_SAMPLING = "thompson_sampling"


class KernelType(str, Enum):
    """Gaussian Process kernel types."""

    RBF = "rbf"
    MATERN32 = "matern32"
    MATERN52 = "matern52"
    RATIONAL_QUADRATIC = "rational_quadratic"


class BayesianOptimizerConfig(BaseOptimizerConfig):
    """Configuration for Bayesian optimizer."""

    acquisition_function: AcquisitionFunction = Field(
        default=AcquisitionFunction.EXPECTED_IMPROVEMENT,
        description="Acquisition function",
    )
    kernel_type: KernelType = Field(default=KernelType.MATERN52, description="GP kernel type")
    n_initial_points: int = Field(default=10, description="Initial random points")
    exploration_weight: float = Field(default=2.0, description="Exploration parameter")
    n_restarts: int = Field(default=10, description="Optimization restarts")
    normalize_y: bool = Field(default=True, description="Normalize target values")
    noise_variance: float = Field(default=1e-6, description="Observation noise")
    length_scale: float = Field(default=1.0, description="Initial length scale")
    length_scale_bounds: tuple[float, float] = Field(
        default=(1e-3, 1e3),
        description="Length scale bounds",
    )
    optimize_hyperparameters: bool = Field(default=True, description="Optimize GP hyperparameters")


@dataclass
class GaussianProcessModel:
    """Simple Gaussian Process model."""

    X_train: np.ndarray
    y_train: np.ndarray
    length_scale: float = 1.0
    signal_variance: float = 1.0
    noise_variance: float = 1e-6
    kernel_type: KernelType = KernelType.MATERN52

    K: np.ndarray = field(default_factory=lambda: np.array([]))
    K_inv: np.ndarray = field(default_factory=lambda: np.array([]))
    alpha: np.ndarray = field(default_factory=lambda: np.array([]))

    y_mean: float = 0.0
    y_std: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray, normalize: bool = True) -> None:
        """
        Fit GP model.

        Args:
            X: Training inputs
            y: Training targets
            normalize: Normalize targets
        """
        self.X_train = X.copy()

        if normalize and len(y) > 1:
            self.y_mean = np.mean(y)
            self.y_std = np.std(y) if np.std(y) > 0 else 1.0
            self.y_train = (y - self.y_mean) / self.y_std
        else:
            self.y_mean = 0.0
            self.y_std = 1.0
            self.y_train = y.copy()

        self.K = self._compute_kernel(X, X)
        self.K += self.noise_variance * np.eye(len(X))

        try:
            L = np.linalg.cholesky(self.K)
            self.K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(len(X))))
            self.alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train))
        except np.linalg.LinAlgError:
            self.K += 1e-4 * np.eye(len(X))
            self.K_inv = np.linalg.inv(self.K)
            self.alpha = self.K_inv @ self.y_train

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance.

        Args:
            X: Test inputs

        Returns:
            Mean and variance predictions
        """
        if len(self.X_train) == 0:
            return np.zeros(len(X)), np.ones(len(X))

        K_star = self._compute_kernel(X, self.X_train)

        mean = K_star @ self.alpha
        mean = mean * self.y_std + self.y_mean

        K_star_star = self._compute_kernel(X, X)
        var = np.diag(K_star_star) - np.sum(K_star @ self.K_inv * K_star, axis=1)
        var = np.maximum(var, 1e-10)
        var = var * self.y_std**2

        return mean, var

    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix."""
        if self.kernel_type == KernelType.RBF:
            return self._rbf_kernel(X1, X2)
        elif self.kernel_type == KernelType.MATERN32:
            return self._matern32_kernel(X1, X2)
        elif self.kernel_type == KernelType.MATERN52:
            return self._matern52_kernel(X1, X2)
        elif self.kernel_type == KernelType.RATIONAL_QUADRATIC:
            return self._rq_kernel(X1, X2)
        else:
            return self._rbf_kernel(X1, X2)

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF (squared exponential) kernel."""
        dists = self._compute_distances(X1, X2)
        return self.signal_variance * np.exp(-0.5 * dists**2 / self.length_scale**2)

    def _matern32_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Matern 3/2 kernel."""
        dists = self._compute_distances(X1, X2)
        r = np.sqrt(3) * dists / self.length_scale
        return self.signal_variance * (1 + r) * np.exp(-r)

    def _matern52_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Matern 5/2 kernel."""
        dists = self._compute_distances(X1, X2)
        r = np.sqrt(5) * dists / self.length_scale
        return self.signal_variance * (1 + r + r**2 / 3) * np.exp(-r)

    def _rq_kernel(self, X1: np.ndarray, X2: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Rational quadratic kernel."""
        dists = self._compute_distances(X1, X2)
        return self.signal_variance * (1 + dists**2 / (2 * alpha * self.length_scale**2)) ** (-alpha)

    def _compute_distances(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute pairwise Euclidean distances."""
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)

        X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)

        dists_sq = X1_sq + X2_sq - 2 * X1 @ X2.T
        dists_sq = np.maximum(dists_sq, 0)

        return np.sqrt(dists_sq)


class BayesianOptimizer(BaseOptimizer):
    """Bayesian optimizer using Gaussian Process."""

    def __init__(
        self,
        config: BayesianOptimizerConfig | None = None,
    ) -> None:
        """
        Initialize Bayesian optimizer.

        Args:
            config: Optimizer configuration
        """
        self._bo_config = config or BayesianOptimizerConfig()
        super().__init__(self._bo_config)

        self.gp_model: GaussianProcessModel | None = None
        self._X_observed: list[np.ndarray] = []
        self._y_observed: list[float] = []
        self._in_initial_phase = True

    def _initialize(self, search_space: SearchSpace) -> None:
        """Initialize Bayesian optimization."""
        self._X_observed = []
        self._y_observed = []
        self._in_initial_phase = True

        self.gp_model = GaussianProcessModel(
            X_train=np.array([]),
            y_train=np.array([]),
            length_scale=self._bo_config.length_scale,
            noise_variance=self._bo_config.noise_variance,
            kernel_type=self._bo_config.kernel_type,
        )

        logger.info(
            f"Bayesian optimization initialized with "
            f"{self._bo_config.acquisition_function.value} acquisition"
        )

    async def _optimize_step(self) -> list[EvaluationPoint]:
        """Perform single Bayesian optimization step."""
        if len(self._X_observed) < self._bo_config.n_initial_points:
            return await self._initial_sampling()
        else:
            self._in_initial_phase = False
            return await self._bayesian_step()

    async def _initial_sampling(self) -> list[EvaluationPoint]:
        """Perform initial random sampling."""
        point = self._search_space.sample_random(self.rng)
        result = await self.evaluate_point(point)

        x_array = self._params_to_array(point)
        self._X_observed.append(x_array)
        self._y_observed.append(result.objective_value)

        return [result]

    async def _bayesian_step(self) -> list[EvaluationPoint]:
        """Perform Bayesian optimization step."""
        X = np.array(self._X_observed)
        y = np.array(self._y_observed)

        self.gp_model.fit(X, y, normalize=self._bo_config.normalize_y)

        if self._bo_config.optimize_hyperparameters:
            self._optimize_gp_hyperparameters(X, y)

        next_point = self._optimize_acquisition()

        params = self._array_to_params(next_point)
        params = self._search_space.clip_to_bounds(params)

        result = await self.evaluate_point(params)

        x_array = self._params_to_array(params)
        self._X_observed.append(x_array)
        self._y_observed.append(result.objective_value)

        return [result]

    def _optimize_acquisition(self) -> np.ndarray:
        """Optimize acquisition function."""
        bounds = self._search_space.get_bounds_array()
        if len(bounds) == 0:
            return np.array([])

        best_x = None
        best_acq = float("-inf")

        for _ in range(self._bo_config.n_restarts):
            x0 = self._sample_random_point()

            x_opt, acq_opt = self._local_optimize_acquisition(x0, bounds)

            if acq_opt > best_acq:
                best_acq = acq_opt
                best_x = x_opt

        if best_x is None:
            best_x = self._sample_random_point()

        return best_x

    def _local_optimize_acquisition(
        self,
        x0: np.ndarray,
        bounds: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Local optimization of acquisition function."""
        n_iters = 50
        step_size = 0.1

        x = x0.copy()

        for _ in range(n_iters):
            acq = self._acquisition_function(x.reshape(1, -1))[0]
            grad = self._numerical_gradient(x)

            x = x + step_size * grad

            for i, (lo, hi) in enumerate(bounds):
                x[i] = np.clip(x[i], lo, hi)

        final_acq = self._acquisition_function(x.reshape(1, -1))[0]
        return x, final_acq

    def _numerical_gradient(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Compute numerical gradient of acquisition function."""
        grad = np.zeros_like(x)

        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps

            f_plus = self._acquisition_function(x_plus.reshape(1, -1))[0]
            f_minus = self._acquisition_function(x_minus.reshape(1, -1))[0]

            grad[i] = (f_plus - f_minus) / (2 * eps)

        return grad

    def _acquisition_function(self, X: np.ndarray) -> np.ndarray:
        """Evaluate acquisition function."""
        mean, var = self.gp_model.predict(X)
        std = np.sqrt(var)

        if self._bo_config.acquisition_function == AcquisitionFunction.EXPECTED_IMPROVEMENT:
            return self._expected_improvement(mean, std)
        elif self._bo_config.acquisition_function == AcquisitionFunction.PROBABILITY_OF_IMPROVEMENT:
            return self._probability_of_improvement(mean, std)
        elif self._bo_config.acquisition_function == AcquisitionFunction.UPPER_CONFIDENCE_BOUND:
            return self._upper_confidence_bound(mean, std)
        elif self._bo_config.acquisition_function == AcquisitionFunction.LOWER_CONFIDENCE_BOUND:
            return self._lower_confidence_bound(mean, std)
        elif self._bo_config.acquisition_function == AcquisitionFunction.THOMPSON_SAMPLING:
            return self._thompson_sampling(mean, std)
        else:
            return self._expected_improvement(mean, std)

    def _expected_improvement(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Expected Improvement acquisition."""
        if len(self._y_observed) == 0:
            return mean

        if self.config.optimization_type == OptimizationType.MAXIMIZATION:
            best_y = np.max(self._y_observed)
            z = (mean - best_y) / (std + 1e-10)
        else:
            best_y = np.min(self._y_observed)
            z = (best_y - mean) / (std + 1e-10)

        ei = std * (z * self._standard_normal_cdf(z) + self._standard_normal_pdf(z))
        return ei

    def _probability_of_improvement(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Probability of Improvement acquisition."""
        if len(self._y_observed) == 0:
            return mean

        if self.config.optimization_type == OptimizationType.MAXIMIZATION:
            best_y = np.max(self._y_observed)
            z = (mean - best_y) / (std + 1e-10)
        else:
            best_y = np.min(self._y_observed)
            z = (best_y - mean) / (std + 1e-10)

        return self._standard_normal_cdf(z)

    def _upper_confidence_bound(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Upper Confidence Bound acquisition."""
        beta = self._bo_config.exploration_weight

        if self.config.optimization_type == OptimizationType.MAXIMIZATION:
            return mean + beta * std
        else:
            return -(mean - beta * std)

    def _lower_confidence_bound(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Lower Confidence Bound acquisition."""
        beta = self._bo_config.exploration_weight
        return mean - beta * std

    def _thompson_sampling(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Thompson Sampling acquisition."""
        samples = self.rng.normal(mean, std)
        return samples

    def _standard_normal_pdf(self, x: np.ndarray) -> np.ndarray:
        """Standard normal PDF."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    def _standard_normal_cdf(self, x: np.ndarray) -> np.ndarray:
        """Standard normal CDF approximation."""
        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def _sample_random_point(self) -> np.ndarray:
        """Sample random point in normalized space."""
        bounds = self._search_space.get_bounds_array()
        point = np.array([
            self.rng.uniform(lo, hi) for lo, hi in bounds
        ])
        return point

    def _params_to_array(self, params: dict[str, Any]) -> np.ndarray:
        """Convert parameters to numpy array."""
        values = []
        for p in self._search_space.continuous_params:
            values.append(float(params.get(p.name, p.default_value or p.lower)))
        return np.array(values)

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

    def _optimize_gp_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> None:
        """Optimize GP hyperparameters using maximum likelihood."""
        if len(X) < 5:
            return

        best_length_scale = self.gp_model.length_scale
        best_nll = float("inf")

        lo, hi = self._bo_config.length_scale_bounds
        length_scales = np.geomspace(lo, hi, 10)

        for ls in length_scales:
            self.gp_model.length_scale = ls
            self.gp_model.fit(X, y, normalize=self._bo_config.normalize_y)

            nll = self._compute_negative_log_likelihood(X, y)

            if nll < best_nll:
                best_nll = nll
                best_length_scale = ls

        self.gp_model.length_scale = best_length_scale
        self.gp_model.fit(X, y, normalize=self._bo_config.normalize_y)

    def _compute_negative_log_likelihood(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute negative log marginal likelihood."""
        n = len(y)

        y_normalized = (y - np.mean(y)) / (np.std(y) + 1e-10)

        try:
            L = np.linalg.cholesky(self.gp_model.K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_normalized))

            nll = 0.5 * y_normalized @ alpha
            nll += np.sum(np.log(np.diag(L)))
            nll += 0.5 * n * np.log(2 * np.pi)

            return float(nll)
        except np.linalg.LinAlgError:
            return float("inf")


def create_bayesian_optimizer(
    n_initial_points: int = 10,
    max_iterations: int = 100,
    acquisition_function: str = "expected_improvement",
    kernel_type: str = "matern52",
    exploration_weight: float = 2.0,
    optimization_type: str = "maximization",
    seed: int | None = None,
) -> BayesianOptimizer:
    """
    Create Bayesian optimizer.

    Args:
        n_initial_points: Initial random samples
        max_iterations: Maximum iterations
        acquisition_function: Acquisition function
        kernel_type: GP kernel type
        exploration_weight: Exploration parameter
        optimization_type: Optimization type
        seed: Random seed

    Returns:
        Configured BayesianOptimizer
    """
    config = BayesianOptimizerConfig(
        n_initial_points=n_initial_points,
        max_iterations=max_iterations,
        acquisition_function=AcquisitionFunction(acquisition_function),
        kernel_type=KernelType(kernel_type),
        exploration_weight=exploration_weight,
        optimization_type=OptimizationType(optimization_type),
        random_seed=seed,
    )
    return BayesianOptimizer(config)
