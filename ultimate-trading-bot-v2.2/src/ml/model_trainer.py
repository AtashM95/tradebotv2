"""
Model Training Module for Machine Learning.

This module provides comprehensive model training capabilities
with cross-validation, hyperparameter tuning, and early stopping.
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

from ultimate_trading_bot.ml.base_model import (
    BaseMLModel,
    ModelMetadata,
    ModelStatus,
    TrainingResult,
)

logger = logging.getLogger(__name__)


class TrainingStrategy(str, Enum):
    """Training strategies."""

    STANDARD = "standard"
    CROSS_VALIDATION = "cross_validation"
    TIME_SERIES_CV = "time_series_cv"
    WALK_FORWARD = "walk_forward"
    BOOTSTRAP = "bootstrap"


class EarlyStoppingMode(str, Enum):
    """Early stopping modes."""

    MIN = "min"
    MAX = "max"
    AUTO = "auto"


class TrainerConfig(BaseModel):
    """Configuration for model trainer."""

    strategy: TrainingStrategy = Field(
        default=TrainingStrategy.TIME_SERIES_CV,
        description="Training strategy",
    )
    n_folds: int = Field(default=5, description="Number of CV folds")
    validation_split: float = Field(default=0.2, description="Validation split ratio")
    test_split: float = Field(default=0.1, description="Test split ratio")
    shuffle: bool = Field(default=False, description="Shuffle data (not for time series)")
    early_stopping: bool = Field(default=True, description="Use early stopping")
    early_stopping_rounds: int = Field(default=10, description="Early stopping patience")
    early_stopping_metric: str = Field(default="val_loss", description="Metric for early stopping")
    early_stopping_mode: EarlyStoppingMode = Field(
        default=EarlyStoppingMode.AUTO,
        description="Early stopping mode",
    )
    save_best_model: bool = Field(default=True, description="Save best model during training")
    verbose: bool = Field(default=True, description="Verbose output")
    random_seed: int = Field(default=42, description="Random seed")


@dataclass
class FoldResult:
    """Result from training on a single fold."""

    fold_id: int
    train_metrics: dict[str, float]
    validation_metrics: dict[str, float]
    training_time: float = 0.0
    best_iteration: int = 0


@dataclass
class CVResult:
    """Cross-validation result."""

    fold_results: list[FoldResult]
    mean_train_metrics: dict[str, float] = field(default_factory=dict)
    mean_val_metrics: dict[str, float] = field(default_factory=dict)
    std_val_metrics: dict[str, float] = field(default_factory=dict)
    best_fold: int = 0


@dataclass
class TrainingSession:
    """Complete training session result."""

    model: BaseMLModel
    training_result: TrainingResult
    cv_result: CVResult | None = None

    train_metrics: dict[str, float] = field(default_factory=dict)
    validation_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)

    training_history: dict[str, list[float]] = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)

    total_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class EarlyStopping:
    """Early stopping callback."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: EarlyStoppingMode = EarlyStoppingMode.AUTO,
        restore_best: bool = True,
    ) -> None:
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait
            min_delta: Minimum improvement
            mode: Comparison mode
            restore_best: Restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best

        self.best_score: float | None = None
        self.best_weights: Any = None
        self.counter = 0
        self.stopped = False
        self.best_epoch = 0

    def __call__(
        self,
        score: float,
        epoch: int,
        weights: Any = None,
    ) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric score
            epoch: Current epoch
            weights: Current model weights

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.best_weights = weights
            self.best_epoch = epoch
            return False

        if self.mode == EarlyStoppingMode.AUTO:
            is_better = score < self.best_score - self.min_delta
        elif self.mode == EarlyStoppingMode.MIN:
            is_better = score < self.best_score - self.min_delta
        else:
            is_better = score > self.best_score + self.min_delta

        if is_better:
            self.best_score = score
            self.best_weights = weights
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.stopped = True
            return True

        return False

    def reset(self) -> None:
        """Reset early stopping state."""
        self.best_score = None
        self.best_weights = None
        self.counter = 0
        self.stopped = False
        self.best_epoch = 0


class DataSplitter:
    """Data splitting utility."""

    def __init__(
        self,
        strategy: TrainingStrategy = TrainingStrategy.TIME_SERIES_CV,
        n_folds: int = 5,
        validation_split: float = 0.2,
        test_split: float = 0.1,
        shuffle: bool = False,
        random_seed: int = 42,
    ) -> None:
        """
        Initialize data splitter.

        Args:
            strategy: Splitting strategy
            n_folds: Number of CV folds
            validation_split: Validation ratio
            test_split: Test ratio
            shuffle: Shuffle data
            random_seed: Random seed
        """
        self.strategy = strategy
        self.n_folds = n_folds
        self.validation_split = validation_split
        self.test_split = test_split
        self.shuffle = shuffle
        self.random_seed = random_seed

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Split data into train/validation sets.

        Args:
            X: Features
            y: Targets

        Returns:
            List of (X_train, X_val, y_train, y_val) tuples
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        if self.shuffle:
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)

        if self.strategy == TrainingStrategy.STANDARD:
            return self._standard_split(X, y, indices)

        elif self.strategy == TrainingStrategy.CROSS_VALIDATION:
            return self._kfold_split(X, y, indices)

        elif self.strategy == TrainingStrategy.TIME_SERIES_CV:
            return self._time_series_split(X, y)

        elif self.strategy == TrainingStrategy.WALK_FORWARD:
            return self._walk_forward_split(X, y)

        elif self.strategy == TrainingStrategy.BOOTSTRAP:
            return self._bootstrap_split(X, y)

        return self._standard_split(X, y, indices)

    def _standard_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Standard train/validation split."""
        n_samples = len(X)
        val_size = int(n_samples * self.validation_split)
        train_size = n_samples - val_size

        train_idx = indices[:train_size]
        val_idx = indices[train_size:]

        return [(X[train_idx], X[val_idx], y[train_idx], y[val_idx])]

    def _kfold_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """K-Fold cross-validation split."""
        n_samples = len(X)
        fold_size = n_samples // self.n_folds

        splits = []

        for i in range(self.n_folds):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < self.n_folds - 1 else n_samples

            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

            splits.append((X[train_idx], X[val_idx], y[train_idx], y[val_idx]))

        return splits

    def _time_series_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Time series cross-validation split."""
        n_samples = len(X)
        test_size = n_samples // (self.n_folds + 1)

        splits = []

        for i in range(self.n_folds):
            train_end = (i + 1) * test_size
            val_start = train_end
            val_end = min(val_start + test_size, n_samples)

            if val_end <= val_start:
                continue

            X_train, y_train = X[:train_end], y[:train_end]
            X_val, y_val = X[val_start:val_end], y[val_start:val_end]

            splits.append((X_train, X_val, y_train, y_val))

        return splits

    def _walk_forward_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Walk-forward validation split."""
        n_samples = len(X)
        min_train = n_samples // 3
        step_size = (n_samples - min_train) // (self.n_folds + 1)
        val_size = step_size

        splits = []

        for i in range(self.n_folds):
            train_end = min_train + i * step_size
            val_start = train_end
            val_end = min(val_start + val_size, n_samples)

            if val_end <= val_start:
                continue

            X_train, y_train = X[:train_end], y[:train_end]
            X_val, y_val = X[val_start:val_end], y[val_start:val_end]

            splits.append((X_train, X_val, y_train, y_val))

        return splits

    def _bootstrap_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Bootstrap sampling split."""
        n_samples = len(X)
        splits = []

        np.random.seed(self.random_seed)

        for _ in range(self.n_folds):
            train_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[np.unique(train_idx)] = False
            val_idx = np.where(oob_mask)[0]

            if len(val_idx) == 0:
                val_idx = np.random.choice(n_samples, size=n_samples // 5, replace=False)

            splits.append((X[train_idx], X[val_idx], y[train_idx], y[val_idx]))

        return splits

    def train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and test sets."""
        n_samples = len(X)
        test_size = int(n_samples * self.test_split)
        train_size = n_samples - test_size

        return X[:train_size], X[train_size:], y[:train_size], y[train_size:]


class ModelTrainer:
    """Model trainer with comprehensive training pipeline."""

    def __init__(
        self,
        config: TrainerConfig | None = None,
    ) -> None:
        """
        Initialize model trainer.

        Args:
            config: Trainer configuration
        """
        self.config = config or TrainerConfig()

        self.splitter = DataSplitter(
            strategy=self.config.strategy,
            n_folds=self.config.n_folds,
            validation_split=self.config.validation_split,
            test_split=self.config.test_split,
            shuffle=self.config.shuffle,
            random_seed=self.config.random_seed,
        )

        if self.config.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=self.config.early_stopping_rounds,
                mode=self.config.early_stopping_mode,
            )
        else:
            self.early_stopping = None

        logger.info(
            f"ModelTrainer initialized with {self.config.strategy.value} strategy"
        )

    async def train(
        self,
        model: BaseMLModel,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        eval_metrics: list[Callable] | None = None,
    ) -> TrainingSession:
        """
        Train model with configured strategy.

        Args:
            model: Model to train
            X: Features
            y: Targets
            eval_metrics: Evaluation metrics

        Returns:
            Training session result
        """
        start_time = datetime.now()

        X = np.array(X)
        y = np.array(y)

        X_train, X_test, y_train, y_test = self.splitter.train_test_split(X, y)

        cv_result = await self._cross_validate(model, X_train, y_train)

        training_result = await model.fit(X_train, y_train)

        test_metrics = {}
        if len(X_test) > 0:
            test_metrics = await self._evaluate(model, X_test, y_test, eval_metrics)

        feature_importance = model.get_feature_importance()

        elapsed = (datetime.now() - start_time).total_seconds()

        return TrainingSession(
            model=model,
            training_result=training_result,
            cv_result=cv_result,
            train_metrics=training_result.train_metrics,
            validation_metrics=training_result.validation_metrics,
            test_metrics=test_metrics,
            training_history=training_result.training_history,
            feature_importance=feature_importance,
            total_time=elapsed,
        )

    async def _cross_validate(
        self,
        model: BaseMLModel,
        X: np.ndarray,
        y: np.ndarray,
    ) -> CVResult:
        """Perform cross-validation."""
        splits = self.splitter.split(X, y)
        fold_results = []

        for i, (X_train, X_val, y_train, y_val) in enumerate(splits):
            fold_start = datetime.now()

            model_copy = type(model)(model.config)

            result = await model_copy.fit(X_train, y_train, (X_val, y_val))

            fold_time = (datetime.now() - fold_start).total_seconds()

            fold_results.append(FoldResult(
                fold_id=i,
                train_metrics=result.train_metrics,
                validation_metrics=result.validation_metrics,
                training_time=fold_time,
            ))

            if self.config.verbose:
                logger.info(
                    f"Fold {i + 1}/{len(splits)}: "
                    f"val_metrics={result.validation_metrics}"
                )

        mean_train = self._aggregate_metrics([f.train_metrics for f in fold_results])
        mean_val = self._aggregate_metrics([f.validation_metrics for f in fold_results])
        std_val = self._std_metrics([f.validation_metrics for f in fold_results])

        best_fold = 0
        if fold_results:
            primary_metric = list(mean_val.keys())[0] if mean_val else None
            if primary_metric:
                scores = [f.validation_metrics.get(primary_metric, 0) for f in fold_results]
                best_fold = int(np.argmax(scores))

        return CVResult(
            fold_results=fold_results,
            mean_train_metrics=mean_train,
            mean_val_metrics=mean_val,
            std_val_metrics=std_val,
            best_fold=best_fold,
        )

    async def _evaluate(
        self,
        model: BaseMLModel,
        X: np.ndarray,
        y: np.ndarray,
        metrics: list[Callable] | None = None,
    ) -> dict[str, float]:
        """Evaluate model on data."""
        result = await model.predict(X)
        predictions = result.predictions

        eval_metrics = {}

        mse = float(np.mean((y - predictions) ** 2))
        eval_metrics["mse"] = mse
        eval_metrics["rmse"] = float(np.sqrt(mse))

        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        eval_metrics["r2"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        eval_metrics["mae"] = float(np.mean(np.abs(y - predictions)))

        if metrics:
            for metric_func in metrics:
                try:
                    metric_name = metric_func.__name__
                    eval_metrics[metric_name] = float(metric_func(y, predictions))
                except Exception as e:
                    logger.warning(f"Metric {metric_func.__name__} failed: {e}")

        return eval_metrics

    def _aggregate_metrics(
        self,
        metrics_list: list[dict[str, float]],
    ) -> dict[str, float]:
        """Aggregate metrics across folds."""
        if not metrics_list:
            return {}

        all_keys = set()
        for m in metrics_list:
            all_keys.update(m.keys())

        aggregated = {}
        for key in all_keys:
            values = [m.get(key, 0) for m in metrics_list if key in m]
            if values:
                aggregated[key] = float(np.mean(values))

        return aggregated

    def _std_metrics(
        self,
        metrics_list: list[dict[str, float]],
    ) -> dict[str, float]:
        """Calculate standard deviation of metrics."""
        if not metrics_list:
            return {}

        all_keys = set()
        for m in metrics_list:
            all_keys.update(m.keys())

        std_metrics = {}
        for key in all_keys:
            values = [m.get(key, 0) for m in metrics_list if key in m]
            if len(values) > 1:
                std_metrics[key] = float(np.std(values))
            else:
                std_metrics[key] = 0.0

        return std_metrics


def create_model_trainer(
    strategy: str = "time_series_cv",
    n_folds: int = 5,
    early_stopping: bool = True,
    config: dict | None = None,
) -> ModelTrainer:
    """
    Create a model trainer.

    Args:
        strategy: Training strategy
        n_folds: Number of CV folds
        early_stopping: Use early stopping
        config: Additional configuration

    Returns:
        Configured ModelTrainer
    """
    trainer_config = TrainerConfig(
        strategy=TrainingStrategy(strategy),
        n_folds=n_folds,
        early_stopping=early_stopping,
        **(config or {}),
    )
    return ModelTrainer(trainer_config)
