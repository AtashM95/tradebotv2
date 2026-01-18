"""
Ensemble Models Module for Ultimate Trading Bot v2.2

Implements ensemble methods including Bagging, Boosting, Stacking, and Voting
for improved prediction accuracy and robustness.

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


class EnsembleType(Enum):
    """Types of ensemble methods."""
    BAGGING = "bagging"
    BOOSTING = "boosting"
    STACKING = "stacking"
    VOTING = "voting"
    BLENDING = "blending"


class VotingType(Enum):
    """Types of voting methods."""
    HARD = "hard"
    SOFT = "soft"
    WEIGHTED = "weighted"


class BoostingType(Enum):
    """Types of boosting methods."""
    ADABOOST = "adaboost"
    GRADIENT_BOOST = "gradient_boost"
    XGB_LIKE = "xgb_like"


@dataclass
class EnsemblePrediction:
    """Ensemble prediction result."""
    predictions: np.ndarray
    confidence: np.ndarray
    individual_predictions: list[np.ndarray]
    weights: Optional[np.ndarray] = None
    ensemble_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "predictions": self.predictions.tolist(),
            "confidence": self.confidence.tolist(),
            "individual_predictions": [p.tolist() for p in self.individual_predictions],
            "weights": self.weights.tolist() if self.weights is not None else None,
            "ensemble_name": self.ensemble_name
        }


@dataclass
class EnsembleMetrics:
    """Ensemble model metrics."""
    accuracy: float
    diversity_score: float
    correlation_matrix: np.ndarray
    individual_accuracies: list[float]
    improvement_over_best: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "diversity_score": self.diversity_score,
            "correlation_matrix": self.correlation_matrix.tolist(),
            "individual_accuracies": self.individual_accuracies,
            "improvement_over_best": self.improvement_over_best
        }


class BaseEstimator(ABC):
    """Base class for simple estimators used in ensembles."""

    def __init__(self, name: str = "estimator"):
        """
        Initialize estimator.

        Args:
            name: Estimator name
        """
        self.name = name
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseEstimator":
        """Fit the estimator."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        predictions = self.predict(X)
        proba = np.zeros((len(predictions), 2))
        proba[:, 1] = np.clip(predictions, 0, 1)
        proba[:, 0] = 1 - proba[:, 1]
        return proba

    @property
    def is_fitted(self) -> bool:
        """Check if estimator is fitted."""
        return self._is_fitted


class DecisionStump(BaseEstimator):
    """Simple decision stump (single-split decision tree)."""

    def __init__(self, name: str = "stump"):
        """
        Initialize decision stump.

        Args:
            name: Estimator name
        """
        super().__init__(name)
        self._feature_idx: int = 0
        self._threshold: float = 0.0
        self._left_value: float = 0.0
        self._right_value: float = 0.0
        self._polarity: int = 1

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> "DecisionStump":
        """
        Fit decision stump.

        Args:
            X: Features
            y: Labels
            sample_weights: Sample weights

        Returns:
            Self
        """
        if sample_weights is None:
            sample_weights = np.ones(len(y)) / len(y)

        sample_weights = sample_weights / np.sum(sample_weights)

        n_samples, n_features = X.shape
        best_error = float("inf")

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                for polarity in [1, -1]:
                    predictions = np.ones(n_samples)
                    if polarity == 1:
                        predictions[feature_values < threshold] = -1
                    else:
                        predictions[feature_values >= threshold] = -1

                    error = np.sum(sample_weights * (predictions != y))

                    if error < best_error:
                        best_error = error
                        self._feature_idx = feature_idx
                        self._threshold = threshold
                        self._polarity = polarity

        left_mask = X[:, self._feature_idx] < self._threshold
        right_mask = ~left_mask

        if np.sum(left_mask) > 0:
            self._left_value = np.average(y[left_mask], weights=sample_weights[left_mask])
        else:
            self._left_value = 0.0

        if np.sum(right_mask) > 0:
            self._right_value = np.average(y[right_mask], weights=sample_weights[right_mask])
        else:
            self._right_value = 0.0

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            Predictions
        """
        if not self._is_fitted:
            raise ValueError("Estimator must be fitted before prediction")

        predictions = np.zeros(len(X))
        left_mask = X[:, self._feature_idx] < self._threshold

        if self._polarity == 1:
            predictions[left_mask] = -1
            predictions[~left_mask] = 1
        else:
            predictions[left_mask] = 1
            predictions[~left_mask] = -1

        return predictions

    def predict_value(self, X: np.ndarray) -> np.ndarray:
        """
        Predict continuous values.

        Args:
            X: Features

        Returns:
            Predicted values
        """
        if not self._is_fitted:
            raise ValueError("Estimator must be fitted before prediction")

        predictions = np.zeros(len(X))
        left_mask = X[:, self._feature_idx] < self._threshold

        predictions[left_mask] = self._left_value
        predictions[~left_mask] = self._right_value

        return predictions


class SimpleLinearEstimator(BaseEstimator):
    """Simple linear estimator using OLS."""

    def __init__(self, regularization: float = 0.01, name: str = "linear"):
        """
        Initialize linear estimator.

        Args:
            regularization: L2 regularization strength
            name: Estimator name
        """
        super().__init__(name)
        self.regularization = regularization
        self._weights: Optional[np.ndarray] = None
        self._bias: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleLinearEstimator":
        """
        Fit linear estimator.

        Args:
            X: Features
            y: Targets

        Returns:
            Self
        """
        n_samples, n_features = X.shape

        X_bias = np.column_stack([np.ones(n_samples), X])

        regularization_matrix = np.eye(n_features + 1) * self.regularization
        regularization_matrix[0, 0] = 0

        try:
            self._weights = np.linalg.solve(
                X_bias.T @ X_bias + regularization_matrix,
                X_bias.T @ y
            )
        except np.linalg.LinAlgError:
            self._weights = np.linalg.lstsq(
                X_bias.T @ X_bias + regularization_matrix,
                X_bias.T @ y,
                rcond=None
            )[0]

        self._bias = self._weights[0]
        self._weights = self._weights[1:]

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            Predictions
        """
        if self._weights is None:
            raise ValueError("Estimator must be fitted before prediction")

        return X @ self._weights + self._bias


class BaseEnsemble(ABC):
    """Base class for ensemble methods."""

    def __init__(
        self,
        n_estimators: int = 10,
        name: str = "ensemble"
    ):
        """
        Initialize ensemble.

        Args:
            n_estimators: Number of estimators
            name: Ensemble name
        """
        self.n_estimators = n_estimators
        self.name = name
        self._estimators: list[BaseEstimator] = []
        self._weights: Optional[np.ndarray] = None
        self._is_fitted = False

        logger.info(f"Initialized {self.__class__.__name__}: {name}")

    @abstractmethod
    async def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseEnsemble":
        """Fit the ensemble."""
        pass

    @abstractmethod
    async def predict(self, X: np.ndarray) -> EnsemblePrediction:
        """Make predictions."""
        pass

    @property
    def is_fitted(self) -> bool:
        """Check if ensemble is fitted."""
        return self._is_fitted

    def _calculate_diversity(
        self,
        predictions: list[np.ndarray]
    ) -> float:
        """
        Calculate ensemble diversity.

        Args:
            predictions: List of individual predictions

        Returns:
            Diversity score
        """
        if len(predictions) < 2:
            return 0.0

        n_estimators = len(predictions)
        disagreements = []

        for i in range(n_estimators):
            for j in range(i + 1, n_estimators):
                disagreement = np.mean(predictions[i] != predictions[j])
                disagreements.append(disagreement)

        return float(np.mean(disagreements))


class BaggingEnsemble(BaseEnsemble):
    """Bagging (Bootstrap Aggregating) ensemble."""

    def __init__(
        self,
        base_estimator_factory: Callable[[], BaseEstimator],
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        name: str = "Bagging"
    ):
        """
        Initialize bagging ensemble.

        Args:
            base_estimator_factory: Factory to create base estimators
            n_estimators: Number of estimators
            max_samples: Fraction of samples for each estimator
            max_features: Fraction of features for each estimator
            bootstrap: Whether to use bootstrap sampling
            name: Ensemble name
        """
        super().__init__(n_estimators, name)

        self.base_estimator_factory = base_estimator_factory
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap

        self._feature_indices: list[np.ndarray] = []

    async def fit(self, X: np.ndarray, y: np.ndarray) -> "BaggingEnsemble":
        """
        Fit bagging ensemble.

        Args:
            X: Features
            y: Targets

        Returns:
            Self
        """
        try:
            logger.info(f"Fitting {self.name} with {self.n_estimators} estimators")

            n_samples, n_features = X.shape
            n_samples_subset = int(n_samples * self.max_samples)
            n_features_subset = int(n_features * self.max_features)

            self._estimators = []
            self._feature_indices = []

            for i in range(self.n_estimators):
                if self.bootstrap:
                    sample_indices = np.random.choice(
                        n_samples, size=n_samples_subset, replace=True
                    )
                else:
                    sample_indices = np.random.choice(
                        n_samples, size=n_samples_subset, replace=False
                    )

                feature_indices = np.random.choice(
                    n_features, size=n_features_subset, replace=False
                )
                feature_indices = np.sort(feature_indices)

                X_subset = X[sample_indices][:, feature_indices]
                y_subset = y[sample_indices]

                estimator = self.base_estimator_factory()
                estimator.fit(X_subset, y_subset)

                self._estimators.append(estimator)
                self._feature_indices.append(feature_indices)

            self._weights = np.ones(self.n_estimators) / self.n_estimators
            self._is_fitted = True

            logger.info(f"Bagging ensemble fitted with {len(self._estimators)} estimators")

            return self

        except Exception as e:
            logger.error(f"Error fitting bagging ensemble: {e}")
            raise

    async def predict(self, X: np.ndarray) -> EnsemblePrediction:
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            EnsemblePrediction object
        """
        if not self._is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        try:
            individual_predictions = []

            for estimator, feature_indices in zip(self._estimators, self._feature_indices):
                X_subset = X[:, feature_indices]
                pred = estimator.predict(X_subset)
                individual_predictions.append(pred)

            predictions_array = np.array(individual_predictions)
            mean_predictions = np.mean(predictions_array, axis=0)

            confidence = 1 - np.std(predictions_array, axis=0) / (np.abs(mean_predictions) + 1e-10)
            confidence = np.clip(confidence, 0, 1)

            return EnsemblePrediction(
                predictions=mean_predictions,
                confidence=confidence,
                individual_predictions=individual_predictions,
                weights=self._weights,
                ensemble_name=self.name
            )

        except Exception as e:
            logger.error(f"Error in bagging prediction: {e}")
            raise


class AdaBoostEnsemble(BaseEnsemble):
    """AdaBoost (Adaptive Boosting) ensemble."""

    def __init__(
        self,
        base_estimator_factory: Optional[Callable[[], BaseEstimator]] = None,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        name: str = "AdaBoost"
    ):
        """
        Initialize AdaBoost ensemble.

        Args:
            base_estimator_factory: Factory to create base estimators
            n_estimators: Number of estimators
            learning_rate: Shrinkage factor
            name: Ensemble name
        """
        super().__init__(n_estimators, name)

        self.base_estimator_factory = base_estimator_factory or (lambda: DecisionStump())
        self.learning_rate = learning_rate

        self._alphas: list[float] = []

    async def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoostEnsemble":
        """
        Fit AdaBoost ensemble.

        Args:
            X: Features
            y: Targets (should be -1 or 1 for classification)

        Returns:
            Self
        """
        try:
            logger.info(f"Fitting {self.name} with {self.n_estimators} estimators")

            n_samples = len(X)

            y_binary = np.where(y > 0, 1, -1)

            sample_weights = np.ones(n_samples) / n_samples

            self._estimators = []
            self._alphas = []

            for i in range(self.n_estimators):
                estimator = self.base_estimator_factory()

                if isinstance(estimator, DecisionStump):
                    estimator.fit(X, y_binary, sample_weights)
                else:
                    estimator.fit(X, y_binary)

                predictions = estimator.predict(X)

                incorrect = predictions != y_binary
                weighted_error = np.sum(sample_weights * incorrect) / np.sum(sample_weights)

                weighted_error = np.clip(weighted_error, 1e-10, 1 - 1e-10)

                alpha = self.learning_rate * 0.5 * np.log((1 - weighted_error) / weighted_error)

                sample_weights = sample_weights * np.exp(-alpha * y_binary * predictions)
                sample_weights = sample_weights / np.sum(sample_weights)

                self._estimators.append(estimator)
                self._alphas.append(alpha)

                if weighted_error < 1e-10:
                    logger.info(f"Early stopping at estimator {i + 1}")
                    break

            self._weights = np.array(self._alphas)
            self._weights = self._weights / np.sum(np.abs(self._weights))
            self._is_fitted = True

            logger.info(f"AdaBoost ensemble fitted with {len(self._estimators)} estimators")

            return self

        except Exception as e:
            logger.error(f"Error fitting AdaBoost ensemble: {e}")
            raise

    async def predict(self, X: np.ndarray) -> EnsemblePrediction:
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            EnsemblePrediction object
        """
        if not self._is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        try:
            individual_predictions = []
            weighted_sum = np.zeros(len(X))

            for estimator, alpha in zip(self._estimators, self._alphas):
                pred = estimator.predict(X)
                individual_predictions.append(pred)
                weighted_sum += alpha * pred

            predictions = np.sign(weighted_sum)

            confidence = np.abs(weighted_sum) / np.sum(np.abs(self._alphas))
            confidence = np.clip(confidence, 0, 1)

            return EnsemblePrediction(
                predictions=predictions,
                confidence=confidence,
                individual_predictions=individual_predictions,
                weights=self._weights,
                ensemble_name=self.name
            )

        except Exception as e:
            logger.error(f"Error in AdaBoost prediction: {e}")
            raise


class GradientBoostingEnsemble(BaseEnsemble):
    """Gradient Boosting ensemble for regression."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        subsample: float = 1.0,
        name: str = "GradientBoosting"
    ):
        """
        Initialize Gradient Boosting ensemble.

        Args:
            n_estimators: Number of estimators
            learning_rate: Shrinkage factor
            max_depth: Maximum depth of trees
            subsample: Fraction of samples for each tree
            name: Ensemble name
        """
        super().__init__(n_estimators, name)

        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample

        self._initial_prediction: float = 0.0
        self._trees: list[dict[str, Any]] = []

    async def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingEnsemble":
        """
        Fit Gradient Boosting ensemble.

        Args:
            X: Features
            y: Targets

        Returns:
            Self
        """
        try:
            logger.info(f"Fitting {self.name} with {self.n_estimators} estimators")

            n_samples = len(X)
            n_subsample = int(n_samples * self.subsample)

            self._initial_prediction = np.mean(y)
            predictions = np.full(n_samples, self._initial_prediction)

            self._trees = []

            for i in range(self.n_estimators):
                residuals = y - predictions

                if self.subsample < 1.0:
                    indices = np.random.choice(n_samples, size=n_subsample, replace=False)
                    X_subset = X[indices]
                    residuals_subset = residuals[indices]
                else:
                    X_subset = X
                    residuals_subset = residuals

                tree = self._build_tree(X_subset, residuals_subset, depth=0)
                self._trees.append(tree)

                tree_predictions = self._predict_tree(tree, X)
                predictions += self.learning_rate * tree_predictions

                if (i + 1) % 20 == 0:
                    mse = np.mean((y - predictions) ** 2)
                    logger.debug(f"Estimator {i + 1}: MSE = {mse:.6f}")

            self._is_fitted = True

            logger.info(f"Gradient Boosting ensemble fitted with {len(self._trees)} trees")

            return self

        except Exception as e:
            logger.error(f"Error fitting Gradient Boosting ensemble: {e}")
            raise

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int
    ) -> dict[str, Any]:
        """
        Build a regression tree.

        Args:
            X: Features
            y: Targets
            depth: Current depth

        Returns:
            Tree structure
        """
        if depth >= self.max_depth or len(X) < 2:
            return {"value": np.mean(y), "is_leaf": True}

        best_gain = 0.0
        best_split: Optional[dict[str, Any]] = None

        n_samples, n_features = X.shape
        current_var = np.var(y) * n_samples

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])

            if len(thresholds) > 10:
                thresholds = np.percentile(X[:, feature_idx], np.linspace(0, 100, 11))

            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)

                if n_left < 1 or n_right < 1:
                    continue

                left_var = np.var(y[left_mask]) * n_left
                right_var = np.var(y[right_mask]) * n_right

                gain = current_var - (left_var + right_var)

                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        "feature_idx": feature_idx,
                        "threshold": threshold,
                        "left_mask": left_mask,
                        "right_mask": right_mask
                    }

        if best_split is None:
            return {"value": np.mean(y), "is_leaf": True}

        left_tree = self._build_tree(
            X[best_split["left_mask"]],
            y[best_split["left_mask"]],
            depth + 1
        )

        right_tree = self._build_tree(
            X[best_split["right_mask"]],
            y[best_split["right_mask"]],
            depth + 1
        )

        return {
            "feature_idx": best_split["feature_idx"],
            "threshold": best_split["threshold"],
            "left": left_tree,
            "right": right_tree,
            "is_leaf": False
        }

    def _predict_tree(self, tree: dict[str, Any], X: np.ndarray) -> np.ndarray:
        """
        Make predictions using a tree.

        Args:
            tree: Tree structure
            X: Features

        Returns:
            Predictions
        """
        predictions = np.zeros(len(X))

        for i in range(len(X)):
            predictions[i] = self._predict_single(tree, X[i])

        return predictions

    def _predict_single(self, tree: dict[str, Any], x: np.ndarray) -> float:
        """
        Make prediction for single sample.

        Args:
            tree: Tree structure
            x: Single feature vector

        Returns:
            Prediction
        """
        if tree["is_leaf"]:
            return tree["value"]

        if x[tree["feature_idx"]] <= tree["threshold"]:
            return self._predict_single(tree["left"], x)
        else:
            return self._predict_single(tree["right"], x)

    async def predict(self, X: np.ndarray) -> EnsemblePrediction:
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            EnsemblePrediction object
        """
        if not self._is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        try:
            individual_predictions = []
            predictions = np.full(len(X), self._initial_prediction)

            for tree in self._trees:
                tree_pred = self._predict_tree(tree, X)
                individual_predictions.append(tree_pred)
                predictions += self.learning_rate * tree_pred

            confidence = np.ones(len(X)) * 0.7

            return EnsemblePrediction(
                predictions=predictions,
                confidence=confidence,
                individual_predictions=individual_predictions,
                weights=None,
                ensemble_name=self.name
            )

        except Exception as e:
            logger.error(f"Error in Gradient Boosting prediction: {e}")
            raise


class VotingEnsemble(BaseEnsemble):
    """Voting ensemble combining multiple models."""

    def __init__(
        self,
        estimators: list[tuple[str, BaseEstimator]],
        voting: VotingType = VotingType.SOFT,
        weights: Optional[list[float]] = None,
        name: str = "Voting"
    ):
        """
        Initialize voting ensemble.

        Args:
            estimators: List of (name, estimator) tuples
            voting: Type of voting
            weights: Optional weights for estimators
            name: Ensemble name
        """
        super().__init__(len(estimators), name)

        self.estimator_names = [e[0] for e in estimators]
        self._estimators = [e[1] for e in estimators]
        self.voting = voting

        if weights is not None:
            self._weights = np.array(weights)
            self._weights = self._weights / np.sum(self._weights)
        else:
            self._weights = np.ones(len(estimators)) / len(estimators)

    async def fit(self, X: np.ndarray, y: np.ndarray) -> "VotingEnsemble":
        """
        Fit voting ensemble.

        Args:
            X: Features
            y: Targets

        Returns:
            Self
        """
        try:
            logger.info(f"Fitting {self.name} with {len(self._estimators)} estimators")

            for i, (name, estimator) in enumerate(zip(self.estimator_names, self._estimators)):
                logger.debug(f"Fitting estimator: {name}")
                estimator.fit(X, y)

            self._is_fitted = True

            logger.info(f"Voting ensemble fitted")

            return self

        except Exception as e:
            logger.error(f"Error fitting voting ensemble: {e}")
            raise

    async def predict(self, X: np.ndarray) -> EnsemblePrediction:
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            EnsemblePrediction object
        """
        if not self._is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        try:
            individual_predictions = []

            for estimator in self._estimators:
                pred = estimator.predict(X)
                individual_predictions.append(pred)

            if self.voting == VotingType.HARD:
                rounded_preds = [np.round(p) for p in individual_predictions]
                predictions = np.zeros(len(X))

                for i in range(len(X)):
                    votes = [int(p[i]) for p in rounded_preds]
                    predictions[i] = max(set(votes), key=votes.count)

                confidence = np.ones(len(X)) * 0.6

            elif self.voting == VotingType.SOFT:
                predictions = np.zeros(len(X))
                for pred, weight in zip(individual_predictions, self._weights):
                    predictions += weight * pred

                predictions_array = np.array(individual_predictions)
                confidence = 1 - np.std(predictions_array, axis=0) / (np.abs(predictions) + 1e-10)
                confidence = np.clip(confidence, 0, 1)

            else:
                predictions = np.average(
                    individual_predictions,
                    axis=0,
                    weights=self._weights
                )

                predictions_array = np.array(individual_predictions)
                confidence = 1 - np.std(predictions_array, axis=0) / (np.abs(predictions) + 1e-10)
                confidence = np.clip(confidence, 0, 1)

            return EnsemblePrediction(
                predictions=predictions,
                confidence=confidence,
                individual_predictions=individual_predictions,
                weights=self._weights,
                ensemble_name=self.name
            )

        except Exception as e:
            logger.error(f"Error in voting prediction: {e}")
            raise


class StackingEnsemble(BaseEnsemble):
    """Stacking ensemble with meta-learner."""

    def __init__(
        self,
        base_estimators: list[tuple[str, BaseEstimator]],
        meta_estimator: BaseEstimator,
        cv_folds: int = 5,
        use_original_features: bool = True,
        name: str = "Stacking"
    ):
        """
        Initialize stacking ensemble.

        Args:
            base_estimators: List of (name, estimator) tuples
            meta_estimator: Meta-learner estimator
            cv_folds: Number of CV folds for generating meta-features
            use_original_features: Whether to include original features
            name: Ensemble name
        """
        super().__init__(len(base_estimators), name)

        self.estimator_names = [e[0] for e in base_estimators]
        self._estimators = [e[1] for e in base_estimators]
        self.meta_estimator = meta_estimator
        self.cv_folds = cv_folds
        self.use_original_features = use_original_features

    async def fit(self, X: np.ndarray, y: np.ndarray) -> "StackingEnsemble":
        """
        Fit stacking ensemble.

        Args:
            X: Features
            y: Targets

        Returns:
            Self
        """
        try:
            logger.info(f"Fitting {self.name} with {len(self._estimators)} base estimators")

            n_samples = len(X)

            meta_features = np.zeros((n_samples, len(self._estimators)))

            fold_size = n_samples // self.cv_folds
            indices = np.arange(n_samples)

            for fold in range(self.cv_folds):
                start_idx = fold * fold_size
                if fold == self.cv_folds - 1:
                    end_idx = n_samples
                else:
                    end_idx = (fold + 1) * fold_size

                val_indices = indices[start_idx:end_idx]
                train_indices = np.concatenate([
                    indices[:start_idx],
                    indices[end_idx:]
                ])

                X_train = X[train_indices]
                y_train = y[train_indices]
                X_val = X[val_indices]

                for i, estimator in enumerate(self._estimators):
                    estimator_clone = type(estimator)(estimator.name)
                    estimator_clone.fit(X_train, y_train)
                    meta_features[val_indices, i] = estimator_clone.predict(X_val)

            for estimator in self._estimators:
                estimator.fit(X, y)

            if self.use_original_features:
                meta_X = np.column_stack([X, meta_features])
            else:
                meta_X = meta_features

            self.meta_estimator.fit(meta_X, y)

            self._is_fitted = True

            logger.info(f"Stacking ensemble fitted")

            return self

        except Exception as e:
            logger.error(f"Error fitting stacking ensemble: {e}")
            raise

    async def predict(self, X: np.ndarray) -> EnsemblePrediction:
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            EnsemblePrediction object
        """
        if not self._is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        try:
            individual_predictions = []

            meta_features = np.zeros((len(X), len(self._estimators)))

            for i, estimator in enumerate(self._estimators):
                pred = estimator.predict(X)
                individual_predictions.append(pred)
                meta_features[:, i] = pred

            if self.use_original_features:
                meta_X = np.column_stack([X, meta_features])
            else:
                meta_X = meta_features

            predictions = self.meta_estimator.predict(meta_X)

            predictions_array = np.array(individual_predictions)
            confidence = 1 - np.std(predictions_array, axis=0) / (np.abs(predictions) + 1e-10)
            confidence = np.clip(confidence, 0, 1)

            return EnsemblePrediction(
                predictions=predictions,
                confidence=confidence,
                individual_predictions=individual_predictions,
                weights=None,
                ensemble_name=self.name
            )

        except Exception as e:
            logger.error(f"Error in stacking prediction: {e}")
            raise


class RandomForestEnsemble(BaseEnsemble):
    """Random Forest ensemble."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 2,
        max_features: str = "sqrt",
        bootstrap: bool = True,
        name: str = "RandomForest"
    ):
        """
        Initialize Random Forest.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split
            max_features: Number of features to consider
            bootstrap: Whether to use bootstrap
            name: Ensemble name
        """
        super().__init__(n_estimators, name)

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap

        self._trees: list[dict[str, Any]] = []
        self._feature_indices: list[np.ndarray] = []

    async def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestEnsemble":
        """
        Fit Random Forest.

        Args:
            X: Features
            y: Targets

        Returns:
            Self
        """
        try:
            logger.info(f"Fitting {self.name} with {self.n_estimators} trees")

            n_samples, n_features = X.shape

            if self.max_features == "sqrt":
                n_features_subset = int(np.sqrt(n_features))
            elif self.max_features == "log2":
                n_features_subset = int(np.log2(n_features))
            elif isinstance(self.max_features, float):
                n_features_subset = int(n_features * self.max_features)
            else:
                n_features_subset = n_features

            n_features_subset = max(1, n_features_subset)

            self._trees = []
            self._feature_indices = []

            for i in range(self.n_estimators):
                if self.bootstrap:
                    sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                else:
                    sample_indices = np.arange(n_samples)

                feature_indices = np.random.choice(
                    n_features, size=n_features_subset, replace=False
                )
                feature_indices = np.sort(feature_indices)

                X_subset = X[sample_indices][:, feature_indices]
                y_subset = y[sample_indices]

                tree = self._build_tree(X_subset, y_subset, depth=0)

                self._trees.append(tree)
                self._feature_indices.append(feature_indices)

            self._is_fitted = True

            logger.info(f"Random Forest fitted with {len(self._trees)} trees")

            return self

        except Exception as e:
            logger.error(f"Error fitting Random Forest: {e}")
            raise

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int
    ) -> dict[str, Any]:
        """Build a decision tree."""
        if (depth >= self.max_depth or
            len(X) < self.min_samples_split or
            len(np.unique(y)) == 1):
            return {"value": np.mean(y), "is_leaf": True}

        best_gain = 0.0
        best_split: Optional[dict[str, Any]] = None

        n_samples, n_features = X.shape
        current_var = np.var(y) * n_samples

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])

            if len(thresholds) > 20:
                thresholds = np.percentile(X[:, feature_idx], np.linspace(0, 100, 21))

            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)

                if n_left < 1 or n_right < 1:
                    continue

                left_var = np.var(y[left_mask]) * n_left if n_left > 1 else 0
                right_var = np.var(y[right_mask]) * n_right if n_right > 1 else 0

                gain = current_var - (left_var + right_var)

                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        "feature_idx": feature_idx,
                        "threshold": threshold,
                        "left_mask": left_mask,
                        "right_mask": right_mask
                    }

        if best_split is None or best_gain < 1e-7:
            return {"value": np.mean(y), "is_leaf": True}

        left_tree = self._build_tree(
            X[best_split["left_mask"]],
            y[best_split["left_mask"]],
            depth + 1
        )

        right_tree = self._build_tree(
            X[best_split["right_mask"]],
            y[best_split["right_mask"]],
            depth + 1
        )

        return {
            "feature_idx": best_split["feature_idx"],
            "threshold": best_split["threshold"],
            "left": left_tree,
            "right": right_tree,
            "is_leaf": False
        }

    def _predict_tree(self, tree: dict[str, Any], X: np.ndarray) -> np.ndarray:
        """Make predictions using a tree."""
        predictions = np.zeros(len(X))
        for i in range(len(X)):
            predictions[i] = self._predict_single(tree, X[i])
        return predictions

    def _predict_single(self, tree: dict[str, Any], x: np.ndarray) -> float:
        """Make prediction for single sample."""
        if tree["is_leaf"]:
            return tree["value"]

        if x[tree["feature_idx"]] <= tree["threshold"]:
            return self._predict_single(tree["left"], x)
        else:
            return self._predict_single(tree["right"], x)

    async def predict(self, X: np.ndarray) -> EnsemblePrediction:
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            EnsemblePrediction object
        """
        if not self._is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")

        try:
            individual_predictions = []

            for tree, feature_indices in zip(self._trees, self._feature_indices):
                X_subset = X[:, feature_indices]
                pred = self._predict_tree(tree, X_subset)
                individual_predictions.append(pred)

            predictions_array = np.array(individual_predictions)
            predictions = np.mean(predictions_array, axis=0)

            confidence = 1 - np.std(predictions_array, axis=0) / (np.abs(predictions) + 1e-10)
            confidence = np.clip(confidence, 0, 1)

            return EnsemblePrediction(
                predictions=predictions,
                confidence=confidence,
                individual_predictions=individual_predictions,
                weights=None,
                ensemble_name=self.name
            )

        except Exception as e:
            logger.error(f"Error in Random Forest prediction: {e}")
            raise

    def feature_importance(self, n_features: int) -> np.ndarray:
        """
        Calculate feature importance.

        Args:
            n_features: Total number of features

        Returns:
            Feature importance array
        """
        importance = np.zeros(n_features)
        counts = np.zeros(n_features)

        for tree, feature_indices in zip(self._trees, self._feature_indices):
            tree_importance = self._tree_feature_importance(tree)

            for local_idx, global_idx in enumerate(feature_indices):
                if local_idx < len(tree_importance):
                    importance[global_idx] += tree_importance[local_idx]
                    counts[global_idx] += 1

        mask = counts > 0
        importance[mask] = importance[mask] / counts[mask]

        if np.sum(importance) > 0:
            importance = importance / np.sum(importance)

        return importance

    def _tree_feature_importance(self, tree: dict[str, Any]) -> list[float]:
        """Calculate feature importance for a single tree."""
        importance: dict[int, float] = {}
        self._accumulate_importance(tree, importance)

        max_idx = max(importance.keys()) if importance else 0
        result = [0.0] * (max_idx + 1)
        for idx, imp in importance.items():
            result[idx] = imp

        return result

    def _accumulate_importance(
        self,
        tree: dict[str, Any],
        importance: dict[int, float]
    ) -> None:
        """Accumulate feature importance recursively."""
        if tree["is_leaf"]:
            return

        feature_idx = tree["feature_idx"]
        importance[feature_idx] = importance.get(feature_idx, 0) + 1

        self._accumulate_importance(tree["left"], importance)
        self._accumulate_importance(tree["right"], importance)


def create_bagging_ensemble(
    n_estimators: int = 10,
    max_samples: float = 1.0,
    name: str = "Bagging"
) -> BaggingEnsemble:
    """
    Factory function to create Bagging ensemble.

    Args:
        n_estimators: Number of estimators
        max_samples: Fraction of samples
        name: Ensemble name

    Returns:
        BaggingEnsemble instance
    """
    return BaggingEnsemble(
        base_estimator_factory=lambda: SimpleLinearEstimator(),
        n_estimators=n_estimators,
        max_samples=max_samples,
        name=name
    )


def create_adaboost_ensemble(
    n_estimators: int = 50,
    learning_rate: float = 1.0,
    name: str = "AdaBoost"
) -> AdaBoostEnsemble:
    """
    Factory function to create AdaBoost ensemble.

    Args:
        n_estimators: Number of estimators
        learning_rate: Learning rate
        name: Ensemble name

    Returns:
        AdaBoostEnsemble instance
    """
    return AdaBoostEnsemble(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        name=name
    )


def create_gradient_boosting_ensemble(
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    name: str = "GradientBoosting"
) -> GradientBoostingEnsemble:
    """
    Factory function to create Gradient Boosting ensemble.

    Args:
        n_estimators: Number of estimators
        learning_rate: Learning rate
        max_depth: Maximum tree depth
        name: Ensemble name

    Returns:
        GradientBoostingEnsemble instance
    """
    return GradientBoostingEnsemble(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        name=name
    )


def create_random_forest_ensemble(
    n_estimators: int = 100,
    max_depth: int = 10,
    max_features: str = "sqrt",
    name: str = "RandomForest"
) -> RandomForestEnsemble:
    """
    Factory function to create Random Forest ensemble.

    Args:
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        max_features: Number of features to consider
        name: Ensemble name

    Returns:
        RandomForestEnsemble instance
    """
    return RandomForestEnsemble(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        name=name
    )
