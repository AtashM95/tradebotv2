"""
Hyperparameter Tuner for Strategy Optimization.

This module provides automated hyperparameter tuning capabilities
for trading strategies with cross-validation support.
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

from ultimate_trading_bot.optimization.base_optimizer import (
    BaseOptimizerConfig,
    EvaluationPoint,
    OptimizationResult,
    OptimizationType,
    ParameterBounds,
    SearchSpace,
)
from ultimate_trading_bot.optimization.bayesian_optimizer import (
    BayesianOptimizer,
    BayesianOptimizerConfig,
)
from ultimate_trading_bot.optimization.genetic_optimizer import (
    GeneticOptimizer,
    GeneticOptimizerConfig,
)

logger = logging.getLogger(__name__)


class TuningMethod(str, Enum):
    """Hyperparameter tuning methods."""

    RANDOM_SEARCH = "random_search"
    GRID_SEARCH = "grid_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    SUCCESSIVE_HALVING = "successive_halving"
    HYPERBAND = "hyperband"


class CrossValidation(str, Enum):
    """Cross-validation methods."""

    NONE = "none"
    KFOLD = "kfold"
    TIME_SERIES = "time_series"
    WALK_FORWARD = "walk_forward"
    COMBINATORIAL_PURGED = "combinatorial_purged"


class TunerConfig(BaseModel):
    """Configuration for hyperparameter tuner."""

    method: TuningMethod = Field(
        default=TuningMethod.BAYESIAN,
        description="Tuning method",
    )
    max_evaluations: int = Field(default=100, description="Maximum evaluations")
    cv_method: CrossValidation = Field(
        default=CrossValidation.TIME_SERIES,
        description="Cross-validation method",
    )
    cv_folds: int = Field(default=5, description="Number of CV folds")
    cv_gap: int = Field(default=0, description="Gap between train and test")
    purge_gap: int = Field(default=0, description="Purge gap for CPV")
    test_size: float = Field(default=0.2, description="Test size for splits")
    early_stopping_rounds: int = Field(default=20, description="Early stopping patience")
    n_jobs: int = Field(default=1, description="Parallel jobs")
    random_seed: int | None = Field(default=None, description="Random seed")
    verbose: bool = Field(default=True, description="Verbose output")
    optimization_metric: str = Field(default="sharpe_ratio", description="Metric to optimize")
    minimize: bool = Field(default=False, description="Minimize metric")


@dataclass
class CVFold:
    """Single cross-validation fold."""

    fold_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_start: int = 0
    train_end: int = 0
    test_start: int = 0
    test_end: int = 0


@dataclass
class CVResult:
    """Cross-validation result."""

    scores: list[float]
    mean_score: float
    std_score: float
    fold_results: list[dict[str, Any]] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class TuningResult:
    """Complete hyperparameter tuning result."""

    best_parameters: dict[str, Any]
    best_score: float
    best_cv_result: CVResult | None
    all_evaluations: list[EvaluationPoint]
    cv_results: list[CVResult]

    total_evaluations: int = 0
    total_time: float = 0.0
    convergence_iteration: int | None = None

    parameter_importance: dict[str, float] = field(default_factory=dict)
    learning_curves: dict[str, list[float]] = field(default_factory=dict)

    timestamp: datetime = field(default_factory=datetime.now)


class CrossValidator:
    """Cross-validation generator."""

    def __init__(
        self,
        method: CrossValidation = CrossValidation.TIME_SERIES,
        n_folds: int = 5,
        test_size: float = 0.2,
        gap: int = 0,
        purge_gap: int = 0,
    ) -> None:
        """
        Initialize cross-validator.

        Args:
            method: CV method
            n_folds: Number of folds
            test_size: Test size ratio
            gap: Gap between train and test
            purge_gap: Purge gap for combinatorial
        """
        self.method = method
        self.n_folds = n_folds
        self.test_size = test_size
        self.gap = gap
        self.purge_gap = purge_gap

    def split(self, n_samples: int) -> list[CVFold]:
        """
        Generate CV splits.

        Args:
            n_samples: Number of samples

        Returns:
            List of CV folds
        """
        if self.method == CrossValidation.NONE:
            return self._single_split(n_samples)
        elif self.method == CrossValidation.KFOLD:
            return self._kfold_split(n_samples)
        elif self.method == CrossValidation.TIME_SERIES:
            return self._time_series_split(n_samples)
        elif self.method == CrossValidation.WALK_FORWARD:
            return self._walk_forward_split(n_samples)
        elif self.method == CrossValidation.COMBINATORIAL_PURGED:
            return self._combinatorial_purged_split(n_samples)
        else:
            return self._time_series_split(n_samples)

    def _single_split(self, n_samples: int) -> list[CVFold]:
        """Single train/test split."""
        test_size = int(n_samples * self.test_size)
        train_size = n_samples - test_size - self.gap

        return [CVFold(
            fold_id=0,
            train_indices=np.arange(0, train_size),
            test_indices=np.arange(train_size + self.gap, n_samples),
            train_start=0,
            train_end=train_size,
            test_start=train_size + self.gap,
            test_end=n_samples,
        )]

    def _kfold_split(self, n_samples: int) -> list[CVFold]:
        """K-fold cross-validation (shuffled)."""
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        fold_size = n_samples // self.n_folds
        folds = []

        for i in range(self.n_folds):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_folds - 1 else n_samples

            test_indices = indices[test_start:test_end]
            train_indices = np.concatenate([
                indices[:test_start],
                indices[test_end:],
            ])

            folds.append(CVFold(
                fold_id=i,
                train_indices=train_indices,
                test_indices=test_indices,
                train_start=0,
                train_end=len(train_indices),
                test_start=test_start,
                test_end=test_end,
            ))

        return folds

    def _time_series_split(self, n_samples: int) -> list[CVFold]:
        """Time series cross-validation (expanding window)."""
        test_size = n_samples // (self.n_folds + 1)
        folds = []

        for i in range(self.n_folds):
            train_end = (i + 1) * test_size
            test_start = train_end + self.gap
            test_end = min(test_start + test_size, n_samples)

            if test_end <= test_start:
                continue

            folds.append(CVFold(
                fold_id=i,
                train_indices=np.arange(0, train_end),
                test_indices=np.arange(test_start, test_end),
                train_start=0,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            ))

        return folds

    def _walk_forward_split(self, n_samples: int) -> list[CVFold]:
        """Walk-forward cross-validation (rolling window)."""
        window_size = n_samples // (self.n_folds + 1)
        test_size = window_size // 3
        train_size = window_size - test_size

        folds = []

        for i in range(self.n_folds):
            train_start = i * test_size
            train_end = train_start + train_size
            test_start = train_end + self.gap
            test_end = min(test_start + test_size, n_samples)

            if test_end <= test_start or test_end > n_samples:
                continue

            folds.append(CVFold(
                fold_id=i,
                train_indices=np.arange(train_start, train_end),
                test_indices=np.arange(test_start, test_end),
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            ))

        return folds

    def _combinatorial_purged_split(self, n_samples: int) -> list[CVFold]:
        """Combinatorial purged cross-validation."""
        test_size = n_samples // self.n_folds
        folds = []

        for i in range(self.n_folds):
            test_start = i * test_size
            test_end = (i + 1) * test_size if i < self.n_folds - 1 else n_samples

            purge_start = max(0, test_start - self.purge_gap)
            purge_end = min(n_samples, test_end + self.purge_gap)

            train_indices = np.concatenate([
                np.arange(0, purge_start),
                np.arange(purge_end, n_samples),
            ])

            test_indices = np.arange(test_start, test_end)

            folds.append(CVFold(
                fold_id=i,
                train_indices=train_indices,
                test_indices=test_indices,
                train_start=0,
                train_end=len(train_indices),
                test_start=test_start,
                test_end=test_end,
            ))

        return folds


class HyperparameterTuner:
    """Hyperparameter tuner for trading strategies."""

    def __init__(
        self,
        config: TunerConfig | None = None,
    ) -> None:
        """
        Initialize hyperparameter tuner.

        Args:
            config: Tuner configuration
        """
        self.config = config or TunerConfig()

        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        self.cv = CrossValidator(
            method=self.config.cv_method,
            n_folds=self.config.cv_folds,
            gap=self.config.cv_gap,
            purge_gap=self.config.purge_gap,
        )

        self._optimizer = self._create_optimizer()
        self._cv_results: list[CVResult] = []
        self._best_cv_result: CVResult | None = None

        logger.info(
            f"HyperparameterTuner initialized with {self.config.method.value} method, "
            f"{self.config.cv_method.value} CV"
        )

    def _create_optimizer(self) -> Any:
        """Create optimizer based on configuration."""
        opt_type = (
            OptimizationType.MINIMIZATION
            if self.config.minimize
            else OptimizationType.MAXIMIZATION
        )

        if self.config.method == TuningMethod.BAYESIAN:
            config = BayesianOptimizerConfig(
                max_iterations=self.config.max_evaluations,
                optimization_type=opt_type,
                random_seed=self.config.random_seed,
            )
            return BayesianOptimizer(config)

        elif self.config.method == TuningMethod.GENETIC:
            config = GeneticOptimizerConfig(
                max_iterations=self.config.max_evaluations,
                optimization_type=opt_type,
                random_seed=self.config.random_seed,
            )
            return GeneticOptimizer(config)

        else:
            config = BaseOptimizerConfig(
                max_iterations=self.config.max_evaluations,
                optimization_type=opt_type,
                random_seed=self.config.random_seed,
            )
            from ultimate_trading_bot.optimization.base_optimizer import RandomSearchOptimizer
            return RandomSearchOptimizer(config)

    async def tune(
        self,
        objective_func: Callable[[dict[str, Any], np.ndarray, np.ndarray], float],
        search_space: SearchSpace,
        data: np.ndarray | pd.DataFrame,
    ) -> TuningResult:
        """
        Run hyperparameter tuning.

        Args:
            objective_func: Function(params, train_indices, test_indices) -> score
            search_space: Parameter search space
            data: Full dataset

        Returns:
            Tuning result
        """
        start_time = datetime.now()

        n_samples = len(data) if hasattr(data, "__len__") else data.shape[0]
        folds = self.cv.split(n_samples)

        logger.info(f"Starting tuning with {len(folds)} CV folds")

        async def cv_objective(params: dict[str, Any]) -> float:
            return await self._evaluate_cv(params, objective_func, folds)

        result = await self._optimizer.optimize(cv_objective, search_space)

        best_cv = None
        if self._cv_results:
            if self.config.minimize:
                best_cv = min(self._cv_results, key=lambda x: x.mean_score)
            else:
                best_cv = max(self._cv_results, key=lambda x: x.mean_score)

        elapsed = (datetime.now() - start_time).total_seconds()

        importance = self._calculate_importance(result.all_evaluations, search_space)

        return TuningResult(
            best_parameters=result.best_parameters,
            best_score=result.best_objective,
            best_cv_result=best_cv,
            all_evaluations=result.all_evaluations,
            cv_results=self._cv_results,
            total_evaluations=result.total_evaluations,
            total_time=elapsed,
            convergence_iteration=result.convergence_iteration,
            parameter_importance=importance,
        )

    async def _evaluate_cv(
        self,
        params: dict[str, Any],
        objective_func: Callable,
        folds: list[CVFold],
    ) -> float:
        """Evaluate parameters using cross-validation."""
        scores = []
        fold_results = []

        for fold in folds:
            try:
                if asyncio.iscoroutinefunction(objective_func):
                    score = await objective_func(
                        params, fold.train_indices, fold.test_indices
                    )
                else:
                    loop = asyncio.get_event_loop()
                    score = await loop.run_in_executor(
                        None,
                        objective_func,
                        params,
                        fold.train_indices,
                        fold.test_indices,
                    )

                scores.append(score)
                fold_results.append({
                    "fold_id": fold.fold_id,
                    "score": score,
                    "train_size": len(fold.train_indices),
                    "test_size": len(fold.test_indices),
                })

            except Exception as e:
                logger.warning(f"Fold {fold.fold_id} evaluation failed: {e}")
                penalty = float("inf") if self.config.minimize else float("-inf")
                scores.append(penalty)

        mean_score = np.mean(scores) if scores else 0.0
        std_score = np.std(scores) if scores else 0.0

        cv_result = CVResult(
            scores=scores,
            mean_score=float(mean_score),
            std_score=float(std_score),
            fold_results=fold_results,
            parameters=params,
        )
        self._cv_results.append(cv_result)

        return float(mean_score)

    def _calculate_importance(
        self,
        evaluations: list[EvaluationPoint],
        search_space: SearchSpace,
    ) -> dict[str, float]:
        """Calculate parameter importance."""
        if len(evaluations) < 10:
            return {}

        importance = {}
        scores = np.array([e.objective_value for e in evaluations])

        for param in search_space.continuous_params:
            values = np.array([
                e.parameters.get(param.name, 0) for e in evaluations
            ])

            if np.std(values) > 0:
                corr = np.corrcoef(values, scores)[0, 1]
                importance[param.name] = abs(corr) if not np.isnan(corr) else 0.0
            else:
                importance[param.name] = 0.0

        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        return importance

    def get_cv_summary(self) -> dict[str, Any]:
        """Get summary of CV results."""
        if not self._cv_results:
            return {}

        mean_scores = [cv.mean_score for cv in self._cv_results]
        std_scores = [cv.std_score for cv in self._cv_results]

        return {
            "n_evaluations": len(self._cv_results),
            "best_mean_score": max(mean_scores) if not self.config.minimize else min(mean_scores),
            "avg_mean_score": np.mean(mean_scores),
            "avg_std_score": np.mean(std_scores),
            "score_range": (min(mean_scores), max(mean_scores)),
        }


class SuccessiveHalvingTuner(HyperparameterTuner):
    """Successive Halving hyperparameter tuner."""

    def __init__(
        self,
        config: TunerConfig | None = None,
        n_configs: int = 81,
        halving_factor: int = 3,
    ) -> None:
        """
        Initialize Successive Halving tuner.

        Args:
            config: Tuner configuration
            n_configs: Initial number of configurations
            halving_factor: Factor to reduce configurations
        """
        super().__init__(config)
        self.n_configs = n_configs
        self.halving_factor = halving_factor

    async def tune(
        self,
        objective_func: Callable,
        search_space: SearchSpace,
        data: np.ndarray | pd.DataFrame,
    ) -> TuningResult:
        """Run Successive Halving tuning."""
        start_time = datetime.now()

        n_samples = len(data) if hasattr(data, "__len__") else data.shape[0]
        folds = self.cv.split(n_samples)

        configs = [
            search_space.sample_random() for _ in range(self.n_configs)
        ]

        n_resources = 1
        all_evaluations = []

        while len(configs) > 1:
            logger.info(
                f"Successive Halving: {len(configs)} configs, "
                f"{n_resources} resources"
            )

            scores = []
            for params in configs:
                score = await self._evaluate_cv(params, objective_func, folds[:n_resources])
                scores.append(score)

                all_evaluations.append(EvaluationPoint(
                    point_id=len(all_evaluations),
                    parameters=params,
                    objective_value=score,
                ))

            sorted_indices = np.argsort(scores)
            if not self.config.minimize:
                sorted_indices = sorted_indices[::-1]

            n_keep = max(1, len(configs) // self.halving_factor)
            configs = [configs[i] for i in sorted_indices[:n_keep]]

            n_resources = min(n_resources * self.halving_factor, len(folds))

        best_params = configs[0]
        best_score = await self._evaluate_cv(best_params, objective_func, folds)

        elapsed = (datetime.now() - start_time).total_seconds()

        return TuningResult(
            best_parameters=best_params,
            best_score=best_score,
            best_cv_result=self._cv_results[-1] if self._cv_results else None,
            all_evaluations=all_evaluations,
            cv_results=self._cv_results,
            total_evaluations=len(all_evaluations),
            total_time=elapsed,
        )


def create_hyperparameter_tuner(
    method: str = "bayesian",
    cv_method: str = "time_series",
    cv_folds: int = 5,
    max_evaluations: int = 100,
    optimization_metric: str = "sharpe_ratio",
    minimize: bool = False,
    seed: int | None = None,
) -> HyperparameterTuner:
    """
    Create a hyperparameter tuner.

    Args:
        method: Tuning method
        cv_method: Cross-validation method
        cv_folds: Number of CV folds
        max_evaluations: Maximum evaluations
        optimization_metric: Metric to optimize
        minimize: Minimize metric
        seed: Random seed

    Returns:
        Configured HyperparameterTuner
    """
    config = TunerConfig(
        method=TuningMethod(method),
        cv_method=CrossValidation(cv_method),
        cv_folds=cv_folds,
        max_evaluations=max_evaluations,
        optimization_metric=optimization_metric,
        minimize=minimize,
        random_seed=seed,
    )
    return HyperparameterTuner(config)
