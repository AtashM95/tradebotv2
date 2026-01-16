"""
Model Selection Module for Ultimate Trading Bot v2.2

Provides automated model selection, cross-validation, hyperparameter optimization,
and model comparison utilities for selecting the best ML model for trading.

Author: AI Assistant
Version: 2.2.0
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class SelectionCriterion(Enum):
    """Model selection criteria."""
    ACCURACY = "accuracy"
    MSE = "mse"
    MAE = "mae"
    R2 = "r2"
    SHARPE = "sharpe"
    SORTINO = "sortino"
    AIC = "aic"
    BIC = "bic"
    CROSS_ENTROPY = "cross_entropy"


class ValidationMethod(Enum):
    """Validation methods."""
    HOLDOUT = "holdout"
    KFOLD = "kfold"
    TIME_SERIES = "time_series"
    WALK_FORWARD = "walk_forward"
    NESTED = "nested"
    BOOTSTRAP = "bootstrap"


class SearchMethod(Enum):
    """Hyperparameter search methods."""
    GRID = "grid"
    RANDOM = "random"
    BAYESIAN = "bayesian"
    SUCCESSIVE_HALVING = "successive_halving"


@dataclass
class ModelCandidate:
    """Represents a model candidate for selection."""
    name: str
    model_factory: Callable[..., Any]
    param_grid: dict[str, list[Any]]
    base_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "param_grid": self.param_grid,
            "base_params": self.base_params
        }


@dataclass
class ValidationResult:
    """Result from cross-validation."""
    scores: list[float]
    mean_score: float
    std_score: float
    fold_details: list[dict[str, Any]]
    validation_method: ValidationMethod
    computation_time: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scores": self.scores,
            "mean_score": self.mean_score,
            "std_score": self.std_score,
            "fold_details": self.fold_details,
            "validation_method": self.validation_method.value,
            "computation_time": self.computation_time
        }


@dataclass
class ModelSelectionResult:
    """Result from model selection."""
    best_model_name: str
    best_params: dict[str, Any]
    best_score: float
    all_results: list[dict[str, Any]]
    selection_criterion: SelectionCriterion
    total_models_evaluated: int
    selection_time: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "best_model_name": self.best_model_name,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "all_results": self.all_results,
            "selection_criterion": self.selection_criterion.value,
            "total_models_evaluated": self.total_models_evaluated,
            "selection_time": self.selection_time
        }


@dataclass
class HyperparameterSearchResult:
    """Result from hyperparameter search."""
    best_params: dict[str, Any]
    best_score: float
    all_trials: list[dict[str, Any]]
    search_method: SearchMethod
    n_iterations: int
    search_time: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "all_trials": self.all_trials,
            "search_method": self.search_method.value,
            "n_iterations": self.n_iterations,
            "search_time": self.search_time
        }


class CrossValidator:
    """Cross-validation utilities."""

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42
    ):
        """
        Initialize cross-validator.

        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle data
            random_state: Random seed
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

        logger.info(
            f"Initialized CrossValidator with {n_splits} splits, "
            f"shuffle={shuffle}"
        )

    def k_fold_split(
        self,
        n_samples: int
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Generate K-fold cross-validation splits.

        Args:
            n_samples: Number of samples

        Returns:
            List of (train_idx, val_idx) tuples
        """
        indices = np.arange(n_samples)

        if self.shuffle:
            np.random.seed(self.random_state)
            indices = np.random.permutation(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits)
        fold_sizes[:n_samples % self.n_splits] += 1

        splits = []
        current = 0

        for fold_size in fold_sizes:
            val_indices = indices[current:current + fold_size]
            train_indices = np.concatenate([
                indices[:current],
                indices[current + fold_size:]
            ])

            splits.append((train_indices, val_indices))
            current += fold_size

        return splits

    def time_series_split(
        self,
        n_samples: int,
        gap: int = 0
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Generate time series cross-validation splits.

        Args:
            n_samples: Number of samples
            gap: Gap between train and validation

        Returns:
            List of (train_idx, val_idx) tuples
        """
        splits = []
        fold_size = n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            train_end = (i + 1) * fold_size
            val_start = train_end + gap
            val_end = val_start + fold_size

            if val_end > n_samples:
                val_end = n_samples

            train_indices = np.arange(train_end)
            val_indices = np.arange(val_start, val_end)

            if len(val_indices) > 0:
                splits.append((train_indices, val_indices))

        return splits

    def walk_forward_split(
        self,
        n_samples: int,
        train_size: int,
        val_size: int,
        step: int = 1
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Generate walk-forward validation splits.

        Args:
            n_samples: Number of samples
            train_size: Training window size
            val_size: Validation window size
            step: Step size between windows

        Returns:
            List of (train_idx, val_idx) tuples
        """
        splits = []
        start = 0

        while start + train_size + val_size <= n_samples:
            train_indices = np.arange(start, start + train_size)
            val_indices = np.arange(start + train_size, start + train_size + val_size)

            splits.append((train_indices, val_indices))
            start += step

        return splits

    def bootstrap_split(
        self,
        n_samples: int,
        n_bootstrap: int = 100
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Generate bootstrap validation splits.

        Args:
            n_samples: Number of samples
            n_bootstrap: Number of bootstrap samples

        Returns:
            List of (train_idx, val_idx) tuples
        """
        np.random.seed(self.random_state)
        splits = []

        for _ in range(n_bootstrap):
            train_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[train_indices] = False
            val_indices = np.where(oob_mask)[0]

            if len(val_indices) > 0:
                splits.append((train_indices, val_indices))

        return splits


class MetricCalculator:
    """Calculate evaluation metrics."""

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy."""
        y_pred_binary = np.round(y_pred)
        return float(np.mean(y_true == y_pred_binary))

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate mean squared error."""
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate mean absolute error."""
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot == 0:
            return 0.0

        return float(1 - ss_res / ss_tot)

    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - risk_free / 252

        if len(excess_returns) == 0:
            return 0.0

        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns)

        if std_return == 0:
            return 0.0

        return float(mean_return / std_return * np.sqrt(252))

    @staticmethod
    def sortino_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
        """Calculate Sortino ratio."""
        excess_returns = returns - risk_free / 252

        if len(excess_returns) == 0:
            return 0.0

        mean_return = np.mean(excess_returns)
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return float("inf") if mean_return > 0 else 0.0

        downside_std = np.std(downside_returns)

        if downside_std == 0:
            return 0.0

        return float(mean_return / downside_std * np.sqrt(252))

    @staticmethod
    def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate cross entropy."""
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return float(-np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        ))

    @staticmethod
    def aic(n_samples: int, n_params: int, mse: float) -> float:
        """Calculate AIC."""
        if mse <= 0:
            return float("inf")
        return float(n_samples * np.log(mse) + 2 * n_params)

    @staticmethod
    def bic(n_samples: int, n_params: int, mse: float) -> float:
        """Calculate BIC."""
        if mse <= 0:
            return float("inf")
        return float(n_samples * np.log(mse) + n_params * np.log(n_samples))

    @classmethod
    def get_metric(cls, criterion: SelectionCriterion) -> Callable:
        """Get metric function by criterion."""
        metrics = {
            SelectionCriterion.ACCURACY: cls.accuracy,
            SelectionCriterion.MSE: cls.mse,
            SelectionCriterion.MAE: cls.mae,
            SelectionCriterion.R2: cls.r2,
            SelectionCriterion.SHARPE: cls.sharpe_ratio,
            SelectionCriterion.SORTINO: cls.sortino_ratio,
            SelectionCriterion.CROSS_ENTROPY: cls.cross_entropy
        }
        return metrics.get(criterion, cls.mse)

    @staticmethod
    def is_higher_better(criterion: SelectionCriterion) -> bool:
        """Check if higher score is better."""
        higher_better = {
            SelectionCriterion.ACCURACY,
            SelectionCriterion.R2,
            SelectionCriterion.SHARPE,
            SelectionCriterion.SORTINO
        }
        return criterion in higher_better


class HyperparameterSearcher:
    """Hyperparameter search utilities."""

    def __init__(
        self,
        cross_validator: CrossValidator,
        metric_calculator: MetricCalculator,
        criterion: SelectionCriterion = SelectionCriterion.MSE
    ):
        """
        Initialize hyperparameter searcher.

        Args:
            cross_validator: Cross-validator instance
            metric_calculator: Metric calculator instance
            criterion: Selection criterion
        """
        self.cross_validator = cross_validator
        self.metric_calculator = metric_calculator
        self.criterion = criterion
        self.higher_is_better = MetricCalculator.is_higher_better(criterion)

        logger.info(f"Initialized HyperparameterSearcher with criterion={criterion.value}")

    async def grid_search(
        self,
        model_factory: Callable[..., Any],
        param_grid: dict[str, list[Any]],
        X: np.ndarray,
        y: np.ndarray,
        base_params: Optional[dict[str, Any]] = None
    ) -> HyperparameterSearchResult:
        """
        Perform grid search.

        Args:
            model_factory: Factory to create models
            param_grid: Parameter grid
            X: Features
            y: Targets
            base_params: Base model parameters

        Returns:
            HyperparameterSearchResult object
        """
        import time
        start_time = time.time()

        logger.info("Starting grid search")

        base_params = base_params or {}
        param_combinations = self._generate_param_combinations(param_grid)

        all_trials = []
        best_score = float("-inf") if self.higher_is_better else float("inf")
        best_params: dict[str, Any] = {}

        for params in param_combinations:
            merged_params = {**base_params, **params}

            score = await self._evaluate_params(
                model_factory, merged_params, X, y
            )

            trial = {
                "params": params,
                "score": score
            }
            all_trials.append(trial)

            if self._is_better_score(score, best_score):
                best_score = score
                best_params = params

        search_time = time.time() - start_time

        logger.info(f"Grid search complete: best_score={best_score:.4f}")

        return HyperparameterSearchResult(
            best_params=best_params,
            best_score=best_score,
            all_trials=all_trials,
            search_method=SearchMethod.GRID,
            n_iterations=len(param_combinations),
            search_time=search_time
        )

    async def random_search(
        self,
        model_factory: Callable[..., Any],
        param_distributions: dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        n_iterations: int = 20,
        base_params: Optional[dict[str, Any]] = None
    ) -> HyperparameterSearchResult:
        """
        Perform random search.

        Args:
            model_factory: Factory to create models
            param_distributions: Parameter distributions
            X: Features
            y: Targets
            n_iterations: Number of iterations
            base_params: Base model parameters

        Returns:
            HyperparameterSearchResult object
        """
        import time
        start_time = time.time()

        logger.info(f"Starting random search with {n_iterations} iterations")

        base_params = base_params or {}

        all_trials = []
        best_score = float("-inf") if self.higher_is_better else float("inf")
        best_params: dict[str, Any] = {}

        for i in range(n_iterations):
            params = self._sample_params(param_distributions)
            merged_params = {**base_params, **params}

            score = await self._evaluate_params(
                model_factory, merged_params, X, y
            )

            trial = {
                "params": params,
                "score": score,
                "iteration": i
            }
            all_trials.append(trial)

            if self._is_better_score(score, best_score):
                best_score = score
                best_params = params

        search_time = time.time() - start_time

        logger.info(f"Random search complete: best_score={best_score:.4f}")

        return HyperparameterSearchResult(
            best_params=best_params,
            best_score=best_score,
            all_trials=all_trials,
            search_method=SearchMethod.RANDOM,
            n_iterations=n_iterations,
            search_time=search_time
        )

    async def successive_halving(
        self,
        model_factory: Callable[..., Any],
        param_distributions: dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        n_candidates: int = 16,
        reduction_factor: int = 2,
        base_params: Optional[dict[str, Any]] = None
    ) -> HyperparameterSearchResult:
        """
        Perform successive halving search.

        Args:
            model_factory: Factory to create models
            param_distributions: Parameter distributions
            X: Features
            y: Targets
            n_candidates: Initial number of candidates
            reduction_factor: Reduction factor per round
            base_params: Base model parameters

        Returns:
            HyperparameterSearchResult object
        """
        import time
        start_time = time.time()

        logger.info(f"Starting successive halving with {n_candidates} candidates")

        base_params = base_params or {}

        candidates = [
            self._sample_params(param_distributions)
            for _ in range(n_candidates)
        ]

        all_trials = []
        n_samples = len(X)
        budget = n_samples // (2 ** int(np.log2(n_candidates)))
        round_num = 0

        while len(candidates) > 1:
            round_num += 1
            budget = min(budget * reduction_factor, n_samples)

            sample_indices = np.random.choice(n_samples, size=budget, replace=False)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]

            scores = []
            for params in candidates:
                merged_params = {**base_params, **params}
                score = await self._evaluate_params(
                    model_factory, merged_params, X_sample, y_sample
                )
                scores.append(score)

                all_trials.append({
                    "params": params,
                    "score": score,
                    "round": round_num,
                    "budget": budget
                })

            n_survivors = max(len(candidates) // reduction_factor, 1)

            if self.higher_is_better:
                sorted_indices = np.argsort(scores)[::-1]
            else:
                sorted_indices = np.argsort(scores)

            candidates = [candidates[i] for i in sorted_indices[:n_survivors]]

            logger.debug(
                f"Round {round_num}: {len(candidates)} candidates remain, "
                f"budget={budget}"
            )

        best_params = candidates[0] if candidates else {}

        merged_params = {**base_params, **best_params}
        best_score = await self._evaluate_params(model_factory, merged_params, X, y)

        search_time = time.time() - start_time

        logger.info(f"Successive halving complete: best_score={best_score:.4f}")

        return HyperparameterSearchResult(
            best_params=best_params,
            best_score=best_score,
            all_trials=all_trials,
            search_method=SearchMethod.SUCCESSIVE_HALVING,
            n_iterations=len(all_trials),
            search_time=search_time
        )

    async def _evaluate_params(
        self,
        model_factory: Callable[..., Any],
        params: dict[str, Any],
        X: np.ndarray,
        y: np.ndarray
    ) -> float:
        """Evaluate parameters using cross-validation."""
        splits = self.cross_validator.k_fold_split(len(X))
        scores = []

        metric_fn = MetricCalculator.get_metric(self.criterion)

        for train_idx, val_idx in splits:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            try:
                model = model_factory(**params)

                if hasattr(model, "fit"):
                    if hasattr(model.fit, "__self__"):
                        model.fit(X_train, y_train)
                    else:
                        await model.fit(X_train, y_train)

                if hasattr(model, "predict"):
                    if hasattr(model.predict, "__self__"):
                        predictions = model.predict(X_val)
                    else:
                        result = await model.predict(X_val)
                        predictions = result.predictions if hasattr(result, "predictions") else result
                else:
                    predictions = np.zeros(len(y_val))

                score = metric_fn(y_val, predictions)
                scores.append(score)

            except Exception as e:
                logger.warning(f"Error evaluating params: {e}")
                if self.higher_is_better:
                    scores.append(float("-inf"))
                else:
                    scores.append(float("inf"))

        return float(np.mean(scores))

    def _generate_param_combinations(
        self,
        param_grid: dict[str, list[Any]]
    ) -> list[dict[str, Any]]:
        """Generate all parameter combinations."""
        if not param_grid:
            return [{}]

        keys = list(param_grid.keys())
        values = list(param_grid.values())

        combinations = []
        indices = [0] * len(keys)

        while True:
            combo = {keys[i]: values[i][indices[i]] for i in range(len(keys))}
            combinations.append(combo)

            for i in range(len(keys) - 1, -1, -1):
                indices[i] += 1
                if indices[i] < len(values[i]):
                    break
                indices[i] = 0
            else:
                break

        return combinations

    def _sample_params(
        self,
        param_distributions: dict[str, Any]
    ) -> dict[str, Any]:
        """Sample parameters from distributions."""
        params = {}

        for name, dist in param_distributions.items():
            if isinstance(dist, list):
                params[name] = np.random.choice(dist)
            elif isinstance(dist, tuple) and len(dist) == 2:
                low, high = dist
                if isinstance(low, int) and isinstance(high, int):
                    params[name] = np.random.randint(low, high + 1)
                else:
                    params[name] = np.random.uniform(low, high)
            elif isinstance(dist, dict):
                if dist.get("type") == "log_uniform":
                    params[name] = np.exp(
                        np.random.uniform(np.log(dist["low"]), np.log(dist["high"]))
                    )
                elif dist.get("type") == "int_log_uniform":
                    val = np.exp(
                        np.random.uniform(np.log(dist["low"]), np.log(dist["high"]))
                    )
                    params[name] = int(val)
                else:
                    params[name] = dist.get("default", 0)
            else:
                params[name] = dist

        return params

    def _is_better_score(self, new_score: float, current_best: float) -> bool:
        """Check if new score is better than current best."""
        if self.higher_is_better:
            return new_score > current_best
        return new_score < current_best


class ModelSelector:
    """Automated model selection."""

    def __init__(
        self,
        candidates: list[ModelCandidate],
        cross_validator: Optional[CrossValidator] = None,
        criterion: SelectionCriterion = SelectionCriterion.MSE,
        search_method: SearchMethod = SearchMethod.RANDOM,
        n_search_iterations: int = 20
    ):
        """
        Initialize model selector.

        Args:
            candidates: List of model candidates
            cross_validator: Cross-validator instance
            criterion: Selection criterion
            search_method: Hyperparameter search method
            n_search_iterations: Number of search iterations
        """
        self.candidates = candidates
        self.cross_validator = cross_validator or CrossValidator()
        self.criterion = criterion
        self.search_method = search_method
        self.n_search_iterations = n_search_iterations

        self.metric_calculator = MetricCalculator()
        self.searcher = HyperparameterSearcher(
            self.cross_validator,
            self.metric_calculator,
            criterion
        )

        self.higher_is_better = MetricCalculator.is_higher_better(criterion)

        logger.info(
            f"Initialized ModelSelector with {len(candidates)} candidates, "
            f"criterion={criterion.value}"
        )

    async def select_best_model(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> ModelSelectionResult:
        """
        Select the best model.

        Args:
            X: Features
            y: Targets

        Returns:
            ModelSelectionResult object
        """
        import time
        start_time = time.time()

        logger.info(f"Starting model selection with {len(self.candidates)} candidates")

        all_results = []
        best_score = float("-inf") if self.higher_is_better else float("inf")
        best_model_name = ""
        best_params: dict[str, Any] = {}

        for candidate in self.candidates:
            logger.info(f"Evaluating candidate: {candidate.name}")

            if self.search_method == SearchMethod.GRID:
                search_result = await self.searcher.grid_search(
                    candidate.model_factory,
                    candidate.param_grid,
                    X, y,
                    candidate.base_params
                )
            elif self.search_method == SearchMethod.RANDOM:
                search_result = await self.searcher.random_search(
                    candidate.model_factory,
                    candidate.param_grid,
                    X, y,
                    self.n_search_iterations,
                    candidate.base_params
                )
            else:
                search_result = await self.searcher.successive_halving(
                    candidate.model_factory,
                    candidate.param_grid,
                    X, y,
                    base_params=candidate.base_params
                )

            result = {
                "model_name": candidate.name,
                "best_params": search_result.best_params,
                "best_score": search_result.best_score,
                "n_trials": search_result.n_iterations
            }
            all_results.append(result)

            if self._is_better(search_result.best_score, best_score):
                best_score = search_result.best_score
                best_model_name = candidate.name
                best_params = {**candidate.base_params, **search_result.best_params}

        selection_time = time.time() - start_time
        total_evaluated = sum(r["n_trials"] for r in all_results)

        logger.info(
            f"Model selection complete: best={best_model_name}, "
            f"score={best_score:.4f}, time={selection_time:.2f}s"
        )

        return ModelSelectionResult(
            best_model_name=best_model_name,
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            selection_criterion=self.criterion,
            total_models_evaluated=total_evaluated,
            selection_time=selection_time
        )

    async def compare_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        models: list[tuple[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Compare multiple fitted models.

        Args:
            X: Features
            y: Targets
            models: List of (name, model) tuples

        Returns:
            Comparison results
        """
        logger.info(f"Comparing {len(models)} models")

        metric_fn = MetricCalculator.get_metric(self.criterion)
        results = []

        for name, model in models:
            splits = self.cross_validator.k_fold_split(len(X))
            scores = []

            for train_idx, val_idx in splits:
                X_val = X[val_idx]
                y_val = y[val_idx]

                try:
                    if hasattr(model, "predict"):
                        predictions = model.predict(X_val)
                        if hasattr(predictions, "predictions"):
                            predictions = predictions.predictions
                    else:
                        predictions = np.zeros(len(y_val))

                    score = metric_fn(y_val, predictions)
                    scores.append(score)

                except Exception as e:
                    logger.warning(f"Error evaluating model {name}: {e}")

            if scores:
                result = {
                    "name": name,
                    "mean_score": float(np.mean(scores)),
                    "std_score": float(np.std(scores)),
                    "scores": scores
                }
            else:
                result = {
                    "name": name,
                    "mean_score": float("inf") if not self.higher_is_better else float("-inf"),
                    "std_score": 0.0,
                    "scores": []
                }

            results.append(result)

        results.sort(
            key=lambda x: x["mean_score"],
            reverse=self.higher_is_better
        )

        return results

    def _is_better(self, new_score: float, current_best: float) -> bool:
        """Check if new score is better."""
        if self.higher_is_better:
            return new_score > current_best
        return new_score < current_best


class FeatureSelector:
    """Feature selection utilities."""

    def __init__(
        self,
        n_features_to_select: Optional[int] = None,
        threshold: float = 0.01
    ):
        """
        Initialize feature selector.

        Args:
            n_features_to_select: Number of features to select
            threshold: Importance threshold
        """
        self.n_features_to_select = n_features_to_select
        self.threshold = threshold
        self._selected_features: Optional[np.ndarray] = None
        self._feature_importance: Optional[np.ndarray] = None

        logger.info("Initialized FeatureSelector")

    def select_by_variance(
        self,
        X: np.ndarray,
        threshold: float = 0.0
    ) -> np.ndarray:
        """
        Select features by variance.

        Args:
            X: Features
            threshold: Variance threshold

        Returns:
            Selected feature indices
        """
        variances = np.var(X, axis=0)
        selected = np.where(variances > threshold)[0]

        self._selected_features = selected
        logger.info(f"Selected {len(selected)} features by variance")

        return selected

    def select_by_correlation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        threshold: float = 0.1
    ) -> np.ndarray:
        """
        Select features by correlation with target.

        Args:
            X: Features
            y: Targets
            threshold: Correlation threshold

        Returns:
            Selected feature indices
        """
        correlations = np.zeros(X.shape[1])

        for i in range(X.shape[1]):
            if np.std(X[:, i]) > 0 and np.std(y) > 0:
                correlations[i] = np.abs(np.corrcoef(X[:, i], y)[0, 1])

        selected = np.where(correlations > threshold)[0]

        self._selected_features = selected
        self._feature_importance = correlations

        logger.info(f"Selected {len(selected)} features by correlation")

        return selected

    def select_by_mutual_information(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_features: Optional[int] = None
    ) -> np.ndarray:
        """
        Select features by mutual information.

        Args:
            X: Features
            y: Targets
            n_features: Number of features to select

        Returns:
            Selected feature indices
        """
        mi_scores = np.zeros(X.shape[1])

        for i in range(X.shape[1]):
            mi_scores[i] = self._estimate_mutual_information(X[:, i], y)

        if n_features is None:
            n_features = self.n_features_to_select or X.shape[1]

        n_features = min(n_features, X.shape[1])
        selected = np.argsort(mi_scores)[-n_features:][::-1]

        self._selected_features = selected
        self._feature_importance = mi_scores

        logger.info(f"Selected {len(selected)} features by mutual information")

        return selected

    def _estimate_mutual_information(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Estimate mutual information using histogram method."""
        x_bins = np.linspace(np.min(x), np.max(x), n_bins + 1)
        y_bins = np.linspace(np.min(y), np.max(y), n_bins + 1)

        x_digitized = np.digitize(x, x_bins[:-1]) - 1
        y_digitized = np.digitize(y, y_bins[:-1]) - 1

        joint_hist = np.zeros((n_bins, n_bins))
        for xi, yi in zip(x_digitized, y_digitized):
            xi = max(0, min(xi, n_bins - 1))
            yi = max(0, min(yi, n_bins - 1))
            joint_hist[xi, yi] += 1

        joint_hist = joint_hist / len(x)
        x_hist = np.sum(joint_hist, axis=1)
        y_hist = np.sum(joint_hist, axis=0)

        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if joint_hist[i, j] > 0 and x_hist[i] > 0 and y_hist[j] > 0:
                    mi += joint_hist[i, j] * np.log(
                        joint_hist[i, j] / (x_hist[i] * y_hist[j])
                    )

        return max(0, mi)

    def recursive_feature_elimination(
        self,
        model_factory: Callable[..., Any],
        X: np.ndarray,
        y: np.ndarray,
        n_features: int,
        step: int = 1
    ) -> np.ndarray:
        """
        Recursive feature elimination.

        Args:
            model_factory: Factory to create models
            X: Features
            y: Targets
            n_features: Target number of features
            step: Features to remove per iteration

        Returns:
            Selected feature indices
        """
        n_total = X.shape[1]
        selected_mask = np.ones(n_total, dtype=bool)
        importance = np.zeros(n_total)

        while np.sum(selected_mask) > n_features:
            X_subset = X[:, selected_mask]

            model = model_factory()
            model.fit(X_subset, y)

            if hasattr(model, "feature_importance"):
                imp = model.feature_importance(X_subset.shape[1])
            else:
                imp = np.random.rand(X_subset.shape[1])

            current_indices = np.where(selected_mask)[0]
            importance[current_indices] = imp

            n_to_remove = min(step, np.sum(selected_mask) - n_features)
            weakest_indices = np.argsort(imp)[:n_to_remove]

            for idx in weakest_indices:
                selected_mask[current_indices[idx]] = False

        selected = np.where(selected_mask)[0]
        self._selected_features = selected
        self._feature_importance = importance

        logger.info(f"RFE selected {len(selected)} features")

        return selected

    @property
    def selected_features(self) -> Optional[np.ndarray]:
        """Get selected feature indices."""
        return self._selected_features

    @property
    def feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores."""
        return self._feature_importance


def create_model_selector(
    candidates: list[ModelCandidate],
    criterion: str = "mse",
    search_method: str = "random",
    n_iterations: int = 20,
    n_splits: int = 5
) -> ModelSelector:
    """
    Factory function to create model selector.

    Args:
        candidates: Model candidates
        criterion: Selection criterion
        search_method: Search method
        n_iterations: Number of search iterations
        n_splits: Number of CV splits

    Returns:
        ModelSelector instance
    """
    cross_validator = CrossValidator(n_splits=n_splits)

    return ModelSelector(
        candidates=candidates,
        cross_validator=cross_validator,
        criterion=SelectionCriterion(criterion),
        search_method=SearchMethod(search_method),
        n_search_iterations=n_iterations
    )


def create_feature_selector(
    n_features: Optional[int] = None,
    threshold: float = 0.01
) -> FeatureSelector:
    """
    Factory function to create feature selector.

    Args:
        n_features: Number of features to select
        threshold: Importance threshold

    Returns:
        FeatureSelector instance
    """
    return FeatureSelector(
        n_features_to_select=n_features,
        threshold=threshold
    )
