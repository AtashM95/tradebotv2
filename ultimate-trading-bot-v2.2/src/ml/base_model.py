"""
Base Model Classes for Machine Learning.

This module provides abstract base classes and common functionality
for all machine learning models in the trading bot.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

T = TypeVar("T")
ModelType = TypeVar("ModelType")


class ModelTask(str, Enum):
    """Types of ML tasks."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    TIME_SERIES = "time_series"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    REINFORCEMENT_LEARNING = "reinforcement_learning"


class ModelStatus(str, Enum):
    """Model lifecycle status."""

    UNTRAINED = "untrained"
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATING = "validating"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    FAILED = "failed"
    DEPRECATED = "deprecated"


class ScalerType(str, Enum):
    """Types of feature scalers."""

    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    NONE = "none"


class BaseModelConfig(BaseModel):
    """Base configuration for ML models."""

    model_name: str = Field(default="base_model", description="Model name")
    model_version: str = Field(default="1.0.0", description="Model version")
    task: ModelTask = Field(default=ModelTask.REGRESSION, description="Model task type")
    scaler_type: ScalerType = Field(default=ScalerType.STANDARD, description="Feature scaler type")
    feature_columns: list[str] = Field(default_factory=list, description="Feature column names")
    target_column: str = Field(default="target", description="Target column name")
    random_seed: int = Field(default=42, description="Random seed")
    validation_split: float = Field(default=0.2, description="Validation split ratio")
    early_stopping_rounds: int = Field(default=10, description="Early stopping patience")
    verbose: bool = Field(default=True, description="Verbose output")


@dataclass
class ModelMetadata:
    """Metadata about a trained model."""

    model_id: str
    model_name: str
    model_version: str
    task: ModelTask
    status: ModelStatus

    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    trained_at: datetime | None = None

    training_samples: int = 0
    validation_samples: int = 0
    feature_count: int = 0
    feature_names: list[str] = field(default_factory=list)

    training_time: float = 0.0
    model_size_bytes: int = 0

    hyperparameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    tags: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class TrainingResult:
    """Result from model training."""

    success: bool
    metadata: ModelMetadata | None = None

    train_metrics: dict[str, float] = field(default_factory=dict)
    validation_metrics: dict[str, float] = field(default_factory=dict)

    training_history: dict[str, list[float]] = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)

    error_message: str | None = None


@dataclass
class PredictionResult:
    """Result from model prediction."""

    predictions: np.ndarray
    probabilities: np.ndarray | None = None
    confidence: np.ndarray | None = None

    prediction_time: float = 0.0
    model_id: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)


class FeatureScaler:
    """Feature scaling utility."""

    def __init__(
        self,
        scaler_type: ScalerType = ScalerType.STANDARD,
    ) -> None:
        """
        Initialize feature scaler.

        Args:
            scaler_type: Type of scaler to use
        """
        self.scaler_type = scaler_type
        self._means: np.ndarray | None = None
        self._stds: np.ndarray | None = None
        self._mins: np.ndarray | None = None
        self._maxs: np.ndarray | None = None
        self._medians: np.ndarray | None = None
        self._iqrs: np.ndarray | None = None
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        """
        Fit scaler to data.

        Args:
            X: Feature matrix

        Returns:
            Self
        """
        X = np.array(X)

        if self.scaler_type == ScalerType.STANDARD:
            self._means = np.mean(X, axis=0)
            self._stds = np.std(X, axis=0)
            self._stds = np.where(self._stds == 0, 1, self._stds)

        elif self.scaler_type == ScalerType.MINMAX:
            self._mins = np.min(X, axis=0)
            self._maxs = np.max(X, axis=0)
            ranges = self._maxs - self._mins
            ranges = np.where(ranges == 0, 1, ranges)
            self._maxs = self._mins + ranges

        elif self.scaler_type == ScalerType.ROBUST:
            self._medians = np.median(X, axis=0)
            q75 = np.percentile(X, 75, axis=0)
            q25 = np.percentile(X, 25, axis=0)
            self._iqrs = q75 - q25
            self._iqrs = np.where(self._iqrs == 0, 1, self._iqrs)

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features.

        Args:
            X: Feature matrix

        Returns:
            Scaled features
        """
        if not self._is_fitted:
            raise RuntimeError("Scaler must be fitted before transform")

        X = np.array(X)

        if self.scaler_type == ScalerType.STANDARD:
            return (X - self._means) / self._stds

        elif self.scaler_type == ScalerType.MINMAX:
            return (X - self._mins) / (self._maxs - self._mins)

        elif self.scaler_type == ScalerType.ROBUST:
            return (X - self._medians) / self._iqrs

        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform features.

        Args:
            X: Feature matrix

        Returns:
            Scaled features
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled features.

        Args:
            X: Scaled feature matrix

        Returns:
            Original scale features
        """
        if not self._is_fitted:
            raise RuntimeError("Scaler must be fitted before inverse transform")

        X = np.array(X)

        if self.scaler_type == ScalerType.STANDARD:
            return X * self._stds + self._means

        elif self.scaler_type == ScalerType.MINMAX:
            return X * (self._maxs - self._mins) + self._mins

        elif self.scaler_type == ScalerType.ROBUST:
            return X * self._iqrs + self._medians

        return X

    def get_params(self) -> dict[str, Any]:
        """Get scaler parameters for persistence."""
        return {
            "scaler_type": self.scaler_type.value,
            "means": self._means.tolist() if self._means is not None else None,
            "stds": self._stds.tolist() if self._stds is not None else None,
            "mins": self._mins.tolist() if self._mins is not None else None,
            "maxs": self._maxs.tolist() if self._maxs is not None else None,
            "medians": self._medians.tolist() if self._medians is not None else None,
            "iqrs": self._iqrs.tolist() if self._iqrs is not None else None,
            "is_fitted": self._is_fitted,
        }

    def set_params(self, params: dict[str, Any]) -> None:
        """Load scaler parameters."""
        self.scaler_type = ScalerType(params.get("scaler_type", "standard"))
        self._means = np.array(params["means"]) if params.get("means") else None
        self._stds = np.array(params["stds"]) if params.get("stds") else None
        self._mins = np.array(params["mins"]) if params.get("mins") else None
        self._maxs = np.array(params["maxs"]) if params.get("maxs") else None
        self._medians = np.array(params["medians"]) if params.get("medians") else None
        self._iqrs = np.array(params["iqrs"]) if params.get("iqrs") else None
        self._is_fitted = params.get("is_fitted", False)


class BaseMLModel(ABC, Generic[ModelType]):
    """Abstract base class for all ML models."""

    def __init__(
        self,
        config: BaseModelConfig | None = None,
    ) -> None:
        """
        Initialize base ML model.

        Args:
            config: Model configuration
        """
        self.config = config or BaseModelConfig()
        self.model: ModelType | None = None
        self.scaler = FeatureScaler(self.config.scaler_type)
        self.metadata = ModelMetadata(
            model_id=f"{self.config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            task=self.config.task,
            status=ModelStatus.UNTRAINED,
        )
        self._is_fitted = False

        np.random.seed(self.config.random_seed)

        logger.info(f"Initialized {self.config.model_name} model")

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted

    @abstractmethod
    def _build_model(self) -> ModelType:
        """Build the underlying model."""
        pass

    @abstractmethod
    def _fit_internal(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Internal fitting logic."""
        pass

    @abstractmethod
    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        """Internal prediction logic."""
        pass

    async def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> TrainingResult:
        """
        Fit the model.

        Args:
            X: Feature matrix
            y: Target values
            validation_data: Optional validation data

        Returns:
            Training result
        """
        start_time = datetime.now()
        self.metadata.status = ModelStatus.TRAINING

        try:
            X = np.array(X)
            y = np.array(y)

            X_scaled = self.scaler.fit_transform(X)

            if validation_data is None and self.config.validation_split > 0:
                split_idx = int(len(X) * (1 - self.config.validation_split))
                X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]
                validation_data = (X_val, y_val)
                X_scaled = X_train
                y = y_train
            elif validation_data is not None:
                X_val, y_val = validation_data
                X_val = self.scaler.transform(np.array(X_val))
                validation_data = (X_val, np.array(y_val))

            self.model = self._build_model()

            training_info = await asyncio.get_event_loop().run_in_executor(
                None,
                self._fit_internal,
                X_scaled,
                y,
                validation_data,
            )

            self._is_fitted = True
            self.metadata.status = ModelStatus.TRAINED
            self.metadata.trained_at = datetime.now()
            self.metadata.training_samples = len(X)
            self.metadata.feature_count = X.shape[1] if len(X.shape) > 1 else 1
            self.metadata.training_time = (datetime.now() - start_time).total_seconds()

            if self.config.feature_columns:
                self.metadata.feature_names = self.config.feature_columns
            else:
                self.metadata.feature_names = [f"feature_{i}" for i in range(self.metadata.feature_count)]

            train_metrics = training_info.get("train_metrics", {})
            val_metrics = training_info.get("validation_metrics", {})
            self.metadata.metrics.update(train_metrics)
            self.metadata.metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

            logger.info(
                f"Model {self.config.model_name} trained successfully in "
                f"{self.metadata.training_time:.2f}s"
            )

            return TrainingResult(
                success=True,
                metadata=self.metadata,
                train_metrics=train_metrics,
                validation_metrics=val_metrics,
                training_history=training_info.get("history", {}),
                feature_importance=training_info.get("feature_importance", {}),
            )

        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.metadata.status = ModelStatus.FAILED

            return TrainingResult(
                success=False,
                error_message=str(e),
            )

    async def predict(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> PredictionResult:
        """
        Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Prediction result
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        start_time = datetime.now()

        X = np.array(X)
        X_scaled = self.scaler.transform(X)

        predictions = await asyncio.get_event_loop().run_in_executor(
            None,
            self._predict_internal,
            X_scaled,
        )

        prediction_time = (datetime.now() - start_time).total_seconds()

        probabilities = None
        if hasattr(self, "_predict_proba_internal"):
            probabilities = await asyncio.get_event_loop().run_in_executor(
                None,
                self._predict_proba_internal,
                X_scaled,
            )

        return PredictionResult(
            predictions=predictions,
            probabilities=probabilities,
            prediction_time=prediction_time,
            model_id=self.metadata.model_id,
        )

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""
        if not self._is_fitted:
            return {}

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            return dict(zip(self.metadata.feature_names, importances))

        return {}

    def get_params(self) -> dict[str, Any]:
        """Get model parameters for persistence."""
        return {
            "config": self.config.model_dump(),
            "scaler": self.scaler.get_params(),
            "metadata": {
                "model_id": self.metadata.model_id,
                "model_name": self.metadata.model_name,
                "model_version": self.metadata.model_version,
                "task": self.metadata.task.value,
                "status": self.metadata.status.value,
                "feature_names": self.metadata.feature_names,
                "metrics": self.metadata.metrics,
            },
            "is_fitted": self._is_fitted,
        }

    def set_params(self, params: dict[str, Any]) -> None:
        """Load model parameters."""
        if "config" in params:
            self.config = BaseModelConfig(**params["config"])

        if "scaler" in params:
            self.scaler.set_params(params["scaler"])

        if "metadata" in params:
            meta = params["metadata"]
            self.metadata.model_id = meta.get("model_id", self.metadata.model_id)
            self.metadata.model_name = meta.get("model_name", self.metadata.model_name)
            self.metadata.model_version = meta.get("model_version", self.metadata.model_version)
            self.metadata.task = ModelTask(meta.get("task", "regression"))
            self.metadata.status = ModelStatus(meta.get("status", "untrained"))
            self.metadata.feature_names = meta.get("feature_names", [])
            self.metadata.metrics = meta.get("metrics", {})

        self._is_fitted = params.get("is_fitted", False)


class SimpleLinearModel(BaseMLModel):
    """Simple linear regression model."""

    def __init__(
        self,
        config: BaseModelConfig | None = None,
        regularization: float = 0.0,
    ) -> None:
        """
        Initialize linear model.

        Args:
            config: Model configuration
            regularization: L2 regularization strength
        """
        super().__init__(config)
        self.regularization = regularization
        self._weights: np.ndarray | None = None
        self._bias: float = 0.0

    def _build_model(self) -> None:
        """Build linear model (weights initialized during fit)."""
        return None

    def _fit_internal(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Fit linear model using normal equations."""
        n_features = X.shape[1] if len(X.shape) > 1 else 1
        X = X.reshape(-1, n_features)

        X_bias = np.column_stack([np.ones(len(X)), X])

        if self.regularization > 0:
            reg_matrix = self.regularization * np.eye(X_bias.shape[1])
            reg_matrix[0, 0] = 0
            weights = np.linalg.lstsq(
                X_bias.T @ X_bias + reg_matrix,
                X_bias.T @ y,
                rcond=None,
            )[0]
        else:
            weights = np.linalg.lstsq(X_bias, y, rcond=None)[0]

        self._bias = weights[0]
        self._weights = weights[1:]

        y_pred = X @ self._weights + self._bias
        train_mse = float(np.mean((y - y_pred) ** 2))
        train_r2 = float(1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))

        val_metrics = {}
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val = X_val.reshape(-1, n_features)
            y_val_pred = X_val @ self._weights + self._bias
            val_metrics["mse"] = float(np.mean((y_val - y_val_pred) ** 2))
            val_metrics["r2"] = float(1 - np.sum((y_val - y_val_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2))

        return {
            "train_metrics": {"mse": train_mse, "r2": train_r2},
            "validation_metrics": val_metrics,
            "feature_importance": dict(
                zip(self.metadata.feature_names or [f"f{i}" for i in range(n_features)], np.abs(self._weights))
            ),
        }

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        n_features = X.shape[1] if len(X.shape) > 1 else 1
        X = X.reshape(-1, n_features)
        return X @ self._weights + self._bias


class SimpleLogisticModel(BaseMLModel):
    """Simple logistic regression model."""

    def __init__(
        self,
        config: BaseModelConfig | None = None,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        regularization: float = 0.0,
    ) -> None:
        """
        Initialize logistic model.

        Args:
            config: Model configuration
            learning_rate: Learning rate
            max_iterations: Maximum iterations
            regularization: L2 regularization
        """
        config = config or BaseModelConfig(task=ModelTask.CLASSIFICATION)
        super().__init__(config)
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization = regularization
        self._weights: np.ndarray | None = None
        self._bias: float = 0.0

    def _build_model(self) -> None:
        """Build logistic model."""
        return None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation."""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _fit_internal(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Fit logistic model using gradient descent."""
        n_samples, n_features = X.shape if len(X.shape) > 1 else (len(X), 1)
        X = X.reshape(n_samples, n_features)

        self._weights = np.zeros(n_features)
        self._bias = 0.0

        history = {"loss": [], "accuracy": []}

        for i in range(self.max_iterations):
            z = X @ self._weights + self._bias
            predictions = self._sigmoid(z)

            error = predictions - y

            dw = (1 / n_samples) * (X.T @ error) + self.regularization * self._weights
            db = (1 / n_samples) * np.sum(error)

            self._weights -= self.learning_rate * dw
            self._bias -= self.learning_rate * db

            if i % 100 == 0:
                loss = -np.mean(y * np.log(predictions + 1e-15) + (1 - y) * np.log(1 - predictions + 1e-15))
                accuracy = np.mean((predictions >= 0.5) == y)
                history["loss"].append(float(loss))
                history["accuracy"].append(float(accuracy))

        final_pred = self._sigmoid(X @ self._weights + self._bias)
        train_accuracy = float(np.mean((final_pred >= 0.5) == y))

        val_metrics = {}
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val = X_val.reshape(-1, n_features)
            val_pred = self._sigmoid(X_val @ self._weights + self._bias)
            val_metrics["accuracy"] = float(np.mean((val_pred >= 0.5) == y_val))

        return {
            "train_metrics": {"accuracy": train_accuracy},
            "validation_metrics": val_metrics,
            "history": history,
            "feature_importance": dict(
                zip(self.metadata.feature_names or [f"f{i}" for i in range(n_features)], np.abs(self._weights))
            ),
        }

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        n_features = X.shape[1] if len(X.shape) > 1 else 1
        X = X.reshape(-1, n_features)
        probabilities = self._sigmoid(X @ self._weights + self._bias)
        return (probabilities >= 0.5).astype(int)

    def _predict_proba_internal(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        n_features = X.shape[1] if len(X.shape) > 1 else 1
        X = X.reshape(-1, n_features)
        proba_1 = self._sigmoid(X @ self._weights + self._bias)
        return np.column_stack([1 - proba_1, proba_1])


def create_linear_model(
    regularization: float = 0.0,
    scaler_type: str = "standard",
) -> SimpleLinearModel:
    """
    Create a linear regression model.

    Args:
        regularization: L2 regularization strength
        scaler_type: Feature scaler type

    Returns:
        Configured SimpleLinearModel
    """
    config = BaseModelConfig(
        model_name="linear_regression",
        task=ModelTask.REGRESSION,
        scaler_type=ScalerType(scaler_type),
    )
    return SimpleLinearModel(config, regularization)


def create_logistic_model(
    learning_rate: float = 0.01,
    max_iterations: int = 1000,
    scaler_type: str = "standard",
) -> SimpleLogisticModel:
    """
    Create a logistic regression model.

    Args:
        learning_rate: Learning rate
        max_iterations: Maximum iterations
        scaler_type: Feature scaler type

    Returns:
        Configured SimpleLogisticModel
    """
    config = BaseModelConfig(
        model_name="logistic_regression",
        task=ModelTask.CLASSIFICATION,
        scaler_type=ScalerType(scaler_type),
    )
    return SimpleLogisticModel(config, learning_rate, max_iterations)
