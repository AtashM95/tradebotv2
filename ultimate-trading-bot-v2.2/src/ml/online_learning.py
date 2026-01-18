"""
Online Learning Module for Ultimate Trading Bot v2.2

Implements online learning algorithms for continuous model updating
with streaming data, including SGD variants, adaptive learning, and
concept drift detection.

Author: AI Assistant
Version: 2.2.0
"""

import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class OnlineLearnerType(Enum):
    """Types of online learners."""
    SGD = "sgd"
    PASSIVE_AGGRESSIVE = "passive_aggressive"
    PERCEPTRON = "perceptron"
    ADAPTIVE = "adaptive"
    BANDIT = "bandit"


class DriftDetectionMethod(Enum):
    """Methods for detecting concept drift."""
    DDM = "ddm"
    EDDM = "eddm"
    ADWIN = "adwin"
    PAGE_HINKLEY = "page_hinkley"


@dataclass
class OnlineUpdate:
    """Result of an online update."""
    loss: float
    prediction: float
    actual: float
    weights_updated: bool
    learning_rate: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "loss": self.loss,
            "prediction": self.prediction,
            "actual": self.actual,
            "weights_updated": self.weights_updated,
            "learning_rate": self.learning_rate,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class DriftDetectionResult:
    """Result of drift detection."""
    drift_detected: bool
    warning_detected: bool
    drift_score: float
    detection_method: DriftDetectionMethod
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "drift_detected": self.drift_detected,
            "warning_detected": self.warning_detected,
            "drift_score": self.drift_score,
            "detection_method": self.detection_method.value,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class OnlineLearnerStats:
    """Statistics for online learner."""
    n_samples: int
    n_updates: int
    cumulative_loss: float
    recent_accuracy: float
    current_learning_rate: float
    drift_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_samples": self.n_samples,
            "n_updates": self.n_updates,
            "cumulative_loss": self.cumulative_loss,
            "recent_accuracy": self.recent_accuracy,
            "current_learning_rate": self.current_learning_rate,
            "drift_count": self.drift_count
        }


class DriftDetector(ABC):
    """Base class for concept drift detectors."""

    def __init__(self, name: str = "DriftDetector"):
        """
        Initialize drift detector.

        Args:
            name: Detector name
        """
        self.name = name
        self._n_samples = 0
        self._in_warning = False
        self._in_drift = False

        logger.info(f"Initialized {self.__class__.__name__}: {name}")

    @abstractmethod
    def update(self, error: float) -> DriftDetectionResult:
        """Update detector with new error."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset detector state."""
        pass

    @property
    def in_warning(self) -> bool:
        """Check if in warning zone."""
        return self._in_warning

    @property
    def in_drift(self) -> bool:
        """Check if drift detected."""
        return self._in_drift


class DDMDetector(DriftDetector):
    """Drift Detection Method (DDM) detector."""

    def __init__(
        self,
        min_samples: int = 30,
        warning_level: float = 2.0,
        drift_level: float = 3.0,
        name: str = "DDM"
    ):
        """
        Initialize DDM detector.

        Args:
            min_samples: Minimum samples before detection
            warning_level: Warning threshold (std devs)
            drift_level: Drift threshold (std devs)
            name: Detector name
        """
        super().__init__(name)

        self.min_samples = min_samples
        self.warning_level = warning_level
        self.drift_level = drift_level

        self._p_min = float("inf")
        self._s_min = float("inf")
        self._p = 0.0
        self._s = 0.0
        self._sum_errors = 0.0

    def update(self, error: float) -> DriftDetectionResult:
        """
        Update DDM with new error.

        Args:
            error: Prediction error (0 or 1 for classification)

        Returns:
            DriftDetectionResult object
        """
        self._n_samples += 1
        self._sum_errors += error

        self._p = self._sum_errors / self._n_samples
        self._s = np.sqrt(self._p * (1 - self._p) / self._n_samples)

        self._in_warning = False
        self._in_drift = False
        drift_score = 0.0

        if self._n_samples >= self.min_samples:
            if self._p + self._s < self._p_min + self._s_min:
                self._p_min = self._p
                self._s_min = self._s

            if self._s_min > 0:
                drift_score = (self._p + self._s - self._p_min) / self._s_min

                if self._p + self._s >= self._p_min + self.drift_level * self._s_min:
                    self._in_drift = True
                elif self._p + self._s >= self._p_min + self.warning_level * self._s_min:
                    self._in_warning = True

        return DriftDetectionResult(
            drift_detected=self._in_drift,
            warning_detected=self._in_warning,
            drift_score=drift_score,
            detection_method=DriftDetectionMethod.DDM
        )

    def reset(self) -> None:
        """Reset detector state."""
        self._n_samples = 0
        self._p_min = float("inf")
        self._s_min = float("inf")
        self._p = 0.0
        self._s = 0.0
        self._sum_errors = 0.0
        self._in_warning = False
        self._in_drift = False


class ADWINDetector(DriftDetector):
    """ADWIN (Adaptive Windowing) detector."""

    def __init__(
        self,
        delta: float = 0.002,
        name: str = "ADWIN"
    ):
        """
        Initialize ADWIN detector.

        Args:
            delta: Confidence parameter
            name: Detector name
        """
        super().__init__(name)

        self.delta = delta

        self._window: deque[float] = deque()
        self._total = 0.0
        self._variance = 0.0
        self._width = 0

    def update(self, value: float) -> DriftDetectionResult:
        """
        Update ADWIN with new value.

        Args:
            value: New observation value

        Returns:
            DriftDetectionResult object
        """
        self._n_samples += 1
        self._window.append(value)
        self._total += value
        self._width += 1

        self._in_drift = False
        self._in_warning = False
        drift_score = 0.0

        if self._width > 1:
            old_variance = self._variance
            mean = self._total / self._width
            self._variance = old_variance + (value - mean) * (value - mean)

        if self._width >= 2:
            drift_score = self._check_for_cut()
            if drift_score > 0:
                self._in_drift = True

        return DriftDetectionResult(
            drift_detected=self._in_drift,
            warning_detected=self._in_warning,
            drift_score=drift_score,
            detection_method=DriftDetectionMethod.ADWIN
        )

    def _check_for_cut(self) -> float:
        """Check if window should be cut."""
        max_cut_score = 0.0

        n = self._width
        if n < 5:
            return 0.0

        window_list = list(self._window)

        for i in range(1, n):
            n0 = i
            n1 = n - i

            if n0 < 2 or n1 < 2:
                continue

            mean0 = sum(window_list[:i]) / n0
            mean1 = sum(window_list[i:]) / n1

            delta_mean = abs(mean0 - mean1)

            m = 1.0 / n0 + 1.0 / n1
            epsilon_cut = np.sqrt(2 * m * np.log(2 / self.delta))

            if delta_mean > epsilon_cut:
                cut_score = delta_mean / epsilon_cut
                if cut_score > max_cut_score:
                    max_cut_score = cut_score

                    for _ in range(i):
                        if self._window:
                            removed = self._window.popleft()
                            self._total -= removed
                            self._width -= 1

                    break

        return max_cut_score

    def reset(self) -> None:
        """Reset detector state."""
        self._n_samples = 0
        self._window.clear()
        self._total = 0.0
        self._variance = 0.0
        self._width = 0
        self._in_warning = False
        self._in_drift = False


class PageHinkleyDetector(DriftDetector):
    """Page-Hinkley test for drift detection."""

    def __init__(
        self,
        min_samples: int = 30,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 0.9999,
        name: str = "PageHinkley"
    ):
        """
        Initialize Page-Hinkley detector.

        Args:
            min_samples: Minimum samples before detection
            delta: Minimum change magnitude
            threshold: Detection threshold
            alpha: Forgetting factor
            name: Detector name
        """
        super().__init__(name)

        self.min_samples = min_samples
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha

        self._sum = 0.0
        self._x_mean = 0.0
        self._sum_min = float("inf")

    def update(self, value: float) -> DriftDetectionResult:
        """
        Update Page-Hinkley with new value.

        Args:
            value: New observation value

        Returns:
            DriftDetectionResult object
        """
        self._n_samples += 1

        self._x_mean = self._x_mean * self.alpha + value * (1 - self.alpha)

        self._sum += value - self._x_mean - self.delta

        self._sum_min = min(self._sum_min, self._sum)

        self._in_drift = False
        self._in_warning = False
        drift_score = self._sum - self._sum_min

        if self._n_samples >= self.min_samples:
            if drift_score > self.threshold:
                self._in_drift = True
            elif drift_score > self.threshold * 0.5:
                self._in_warning = True

        return DriftDetectionResult(
            drift_detected=self._in_drift,
            warning_detected=self._in_warning,
            drift_score=drift_score,
            detection_method=DriftDetectionMethod.PAGE_HINKLEY
        )

    def reset(self) -> None:
        """Reset detector state."""
        self._n_samples = 0
        self._sum = 0.0
        self._x_mean = 0.0
        self._sum_min = float("inf")
        self._in_warning = False
        self._in_drift = False


class BaseOnlineLearner(ABC):
    """Base class for online learners."""

    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.01,
        name: str = "OnlineLearner"
    ):
        """
        Initialize online learner.

        Args:
            n_features: Number of features
            learning_rate: Initial learning rate
            name: Learner name
        """
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.name = name

        self._weights = np.zeros(n_features)
        self._bias = 0.0
        self._n_samples = 0
        self._n_updates = 0
        self._cumulative_loss = 0.0
        self._recent_errors: deque[float] = deque(maxlen=100)

        logger.info(
            f"Initialized {self.__class__.__name__}: {name}, "
            f"features={n_features}, lr={learning_rate}"
        )

    @abstractmethod
    def partial_fit(
        self,
        X: np.ndarray,
        y: float
    ) -> OnlineUpdate:
        """Update model with single sample."""
        pass

    def predict(self, X: np.ndarray) -> float:
        """
        Make prediction.

        Args:
            X: Feature vector

        Returns:
            Prediction
        """
        return float(np.dot(X, self._weights) + self._bias)

    def get_stats(self) -> OnlineLearnerStats:
        """Get learner statistics."""
        recent_accuracy = 1 - np.mean(list(self._recent_errors)) if self._recent_errors else 0.0

        return OnlineLearnerStats(
            n_samples=self._n_samples,
            n_updates=self._n_updates,
            cumulative_loss=self._cumulative_loss,
            recent_accuracy=recent_accuracy,
            current_learning_rate=self.learning_rate,
            drift_count=0
        )

    @property
    def weights(self) -> np.ndarray:
        """Get model weights."""
        return self._weights.copy()

    @property
    def bias(self) -> float:
        """Get model bias."""
        return self._bias


class SGDOnlineLearner(BaseOnlineLearner):
    """Stochastic Gradient Descent online learner."""

    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.01,
        regularization: float = 0.0001,
        loss: str = "squared",
        name: str = "SGD"
    ):
        """
        Initialize SGD online learner.

        Args:
            n_features: Number of features
            learning_rate: Learning rate
            regularization: L2 regularization strength
            loss: Loss function ("squared", "hinge", "log")
            name: Learner name
        """
        super().__init__(n_features, learning_rate, name)

        self.regularization = regularization
        self.loss = loss

    def partial_fit(
        self,
        X: np.ndarray,
        y: float
    ) -> OnlineUpdate:
        """
        Update model with single sample.

        Args:
            X: Feature vector
            y: Target value

        Returns:
            OnlineUpdate object
        """
        self._n_samples += 1

        prediction = self.predict(X)

        if self.loss == "squared":
            error = prediction - y
            loss = 0.5 * error ** 2
            gradient = error * X
            bias_gradient = error

        elif self.loss == "hinge":
            margin = y * prediction
            if margin < 1:
                loss = 1 - margin
                gradient = -y * X
                bias_gradient = -y
            else:
                loss = 0.0
                gradient = np.zeros_like(X)
                bias_gradient = 0.0

        elif self.loss == "log":
            sigmoid = 1 / (1 + np.exp(-prediction))
            loss = -y * np.log(sigmoid + 1e-15) - (1 - y) * np.log(1 - sigmoid + 1e-15)
            gradient = (sigmoid - y) * X
            bias_gradient = sigmoid - y

        else:
            error = prediction - y
            loss = 0.5 * error ** 2
            gradient = error * X
            bias_gradient = error

        self._weights -= self.learning_rate * (gradient + self.regularization * self._weights)
        self._bias -= self.learning_rate * bias_gradient

        self._n_updates += 1
        self._cumulative_loss += loss

        error_binary = 1.0 if np.sign(prediction) != np.sign(y) else 0.0
        self._recent_errors.append(error_binary)

        return OnlineUpdate(
            loss=float(loss),
            prediction=prediction,
            actual=y,
            weights_updated=True,
            learning_rate=self.learning_rate
        )


class PassiveAggressiveLearner(BaseOnlineLearner):
    """Passive-Aggressive online learner."""

    def __init__(
        self,
        n_features: int,
        C: float = 1.0,
        variant: int = 1,
        name: str = "PassiveAggressive"
    ):
        """
        Initialize Passive-Aggressive learner.

        Args:
            n_features: Number of features
            C: Aggressiveness parameter
            variant: PA variant (0, 1, or 2)
            name: Learner name
        """
        super().__init__(n_features, 0.0, name)

        self.C = C
        self.variant = variant

    def partial_fit(
        self,
        X: np.ndarray,
        y: float
    ) -> OnlineUpdate:
        """
        Update model with single sample.

        Args:
            X: Feature vector
            y: Target value (-1 or 1 for classification)

        Returns:
            OnlineUpdate object
        """
        self._n_samples += 1

        prediction = self.predict(X)

        loss = max(0, 1 - y * prediction)

        if loss > 0:
            x_norm_sq = np.dot(X, X)

            if self.variant == 0:
                tau = loss / (x_norm_sq + 1e-10)
            elif self.variant == 1:
                tau = min(self.C, loss / (x_norm_sq + 1e-10))
            else:
                tau = loss / (x_norm_sq + 1 / (2 * self.C))

            self._weights += tau * y * X
            self._bias += tau * y

            self._n_updates += 1
            weights_updated = True
        else:
            weights_updated = False

        self._cumulative_loss += loss

        error_binary = 1.0 if np.sign(prediction) != np.sign(y) else 0.0
        self._recent_errors.append(error_binary)

        return OnlineUpdate(
            loss=float(loss),
            prediction=prediction,
            actual=y,
            weights_updated=weights_updated,
            learning_rate=0.0
        )


class AdaptiveLearningRateLearner(BaseOnlineLearner):
    """Online learner with adaptive learning rate (AdaGrad-like)."""

    def __init__(
        self,
        n_features: int,
        initial_learning_rate: float = 0.1,
        epsilon: float = 1e-8,
        name: str = "AdaptiveLR"
    ):
        """
        Initialize adaptive learning rate learner.

        Args:
            n_features: Number of features
            initial_learning_rate: Initial learning rate
            epsilon: Small constant for numerical stability
            name: Learner name
        """
        super().__init__(n_features, initial_learning_rate, name)

        self.initial_learning_rate = initial_learning_rate
        self.epsilon = epsilon

        self._gradient_accumulator = np.zeros(n_features)
        self._bias_accumulator = 0.0

    def partial_fit(
        self,
        X: np.ndarray,
        y: float
    ) -> OnlineUpdate:
        """
        Update model with adaptive learning rate.

        Args:
            X: Feature vector
            y: Target value

        Returns:
            OnlineUpdate object
        """
        self._n_samples += 1

        prediction = self.predict(X)
        error = prediction - y
        loss = 0.5 * error ** 2

        gradient = error * X
        bias_gradient = error

        self._gradient_accumulator += gradient ** 2
        self._bias_accumulator += bias_gradient ** 2

        adaptive_lr = self.initial_learning_rate / (
            np.sqrt(self._gradient_accumulator) + self.epsilon
        )
        bias_lr = self.initial_learning_rate / (
            np.sqrt(self._bias_accumulator) + self.epsilon
        )

        self._weights -= adaptive_lr * gradient
        self._bias -= bias_lr * bias_gradient

        self._n_updates += 1
        self._cumulative_loss += loss

        self.learning_rate = float(np.mean(adaptive_lr))

        error_binary = 1.0 if abs(error) > 0.5 else 0.0
        self._recent_errors.append(error_binary)

        return OnlineUpdate(
            loss=float(loss),
            prediction=prediction,
            actual=y,
            weights_updated=True,
            learning_rate=self.learning_rate
        )


class OnlineLearnerWithDriftDetection(BaseOnlineLearner):
    """Online learner with concept drift detection and adaptation."""

    def __init__(
        self,
        base_learner: BaseOnlineLearner,
        drift_detector: DriftDetector,
        reset_on_drift: bool = True,
        name: str = "DriftAwareLearner"
    ):
        """
        Initialize drift-aware online learner.

        Args:
            base_learner: Base online learner
            drift_detector: Drift detector
            reset_on_drift: Whether to reset on drift detection
            name: Learner name
        """
        super().__init__(
            base_learner.n_features,
            base_learner.learning_rate,
            name
        )

        self.base_learner = base_learner
        self.drift_detector = drift_detector
        self.reset_on_drift = reset_on_drift

        self._drift_count = 0
        self._last_drift_result: Optional[DriftDetectionResult] = None

    def partial_fit(
        self,
        X: np.ndarray,
        y: float
    ) -> OnlineUpdate:
        """
        Update model with drift detection.

        Args:
            X: Feature vector
            y: Target value

        Returns:
            OnlineUpdate object
        """
        update = self.base_learner.partial_fit(X, y)

        error = 1.0 if abs(update.prediction - y) > 0.5 else 0.0
        drift_result = self.drift_detector.update(error)
        self._last_drift_result = drift_result

        if drift_result.drift_detected:
            self._drift_count += 1
            logger.info(f"Drift detected at sample {self.base_learner._n_samples}")

            if self.reset_on_drift:
                self._reset_learner()

        self._weights = self.base_learner._weights
        self._bias = self.base_learner._bias
        self._n_samples = self.base_learner._n_samples
        self._n_updates = self.base_learner._n_updates
        self._cumulative_loss = self.base_learner._cumulative_loss

        return update

    def _reset_learner(self) -> None:
        """Reset base learner after drift."""
        self.base_learner._weights = np.zeros(self.base_learner.n_features)
        self.base_learner._bias = 0.0
        self.drift_detector.reset()

        logger.info("Reset learner after drift detection")

    def get_stats(self) -> OnlineLearnerStats:
        """Get learner statistics with drift info."""
        base_stats = self.base_learner.get_stats()
        return OnlineLearnerStats(
            n_samples=base_stats.n_samples,
            n_updates=base_stats.n_updates,
            cumulative_loss=base_stats.cumulative_loss,
            recent_accuracy=base_stats.recent_accuracy,
            current_learning_rate=base_stats.current_learning_rate,
            drift_count=self._drift_count
        )

    def predict(self, X: np.ndarray) -> float:
        """Make prediction using base learner."""
        return self.base_learner.predict(X)


class OnlineEnsemble:
    """Ensemble of online learners."""

    def __init__(
        self,
        learners: list[BaseOnlineLearner],
        combination_method: str = "weighted_average",
        name: str = "OnlineEnsemble"
    ):
        """
        Initialize online ensemble.

        Args:
            learners: List of online learners
            combination_method: How to combine predictions
            name: Ensemble name
        """
        self.learners = learners
        self.combination_method = combination_method
        self.name = name

        self._learner_weights = np.ones(len(learners)) / len(learners)
        self._learner_losses: list[deque[float]] = [
            deque(maxlen=100) for _ in learners
        ]

        logger.info(
            f"Initialized OnlineEnsemble: {name} with "
            f"{len(learners)} learners"
        )

    def partial_fit(
        self,
        X: np.ndarray,
        y: float
    ) -> list[OnlineUpdate]:
        """
        Update all learners with single sample.

        Args:
            X: Feature vector
            y: Target value

        Returns:
            List of OnlineUpdate objects
        """
        updates = []

        for i, learner in enumerate(self.learners):
            update = learner.partial_fit(X, y)
            updates.append(update)
            self._learner_losses[i].append(update.loss)

        self._update_weights()

        return updates

    def _update_weights(self) -> None:
        """Update learner weights based on recent performance."""
        avg_losses = []
        for losses in self._learner_losses:
            if losses:
                avg_losses.append(np.mean(list(losses)))
            else:
                avg_losses.append(1.0)

        avg_losses = np.array(avg_losses)

        inv_losses = 1 / (avg_losses + 1e-10)
        self._learner_weights = inv_losses / np.sum(inv_losses)

    def predict(self, X: np.ndarray) -> float:
        """
        Make ensemble prediction.

        Args:
            X: Feature vector

        Returns:
            Ensemble prediction
        """
        predictions = np.array([learner.predict(X) for learner in self.learners])

        if self.combination_method == "weighted_average":
            return float(np.average(predictions, weights=self._learner_weights))
        elif self.combination_method == "median":
            return float(np.median(predictions))
        elif self.combination_method == "best":
            best_idx = np.argmax(self._learner_weights)
            return float(predictions[best_idx])
        else:
            return float(np.mean(predictions))

    def get_weights(self) -> np.ndarray:
        """Get current learner weights."""
        return self._learner_weights.copy()


class StreamingDataBuffer:
    """Buffer for managing streaming data."""

    def __init__(
        self,
        max_size: int = 10000,
        feature_size: int = 10
    ):
        """
        Initialize streaming buffer.

        Args:
            max_size: Maximum buffer size
            feature_size: Size of feature vectors
        """
        self.max_size = max_size
        self.feature_size = feature_size

        self._features: deque[np.ndarray] = deque(maxlen=max_size)
        self._targets: deque[float] = deque(maxlen=max_size)
        self._timestamps: deque[datetime] = deque(maxlen=max_size)

        logger.info(
            f"Initialized StreamingDataBuffer: max_size={max_size}, "
            f"feature_size={feature_size}"
        )

    def add(
        self,
        features: np.ndarray,
        target: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Add sample to buffer.

        Args:
            features: Feature vector
            target: Target value
            timestamp: Sample timestamp
        """
        self._features.append(features)
        self._targets.append(target)
        self._timestamps.append(timestamp or datetime.now())

    def get_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Get batch of recent samples.

        Args:
            batch_size: Number of samples

        Returns:
            Tuple of (features, targets) arrays
        """
        batch_size = min(batch_size, len(self._features))

        features = np.array(list(self._features)[-batch_size:])
        targets = np.array(list(self._targets)[-batch_size:])

        return features, targets

    def get_all(self) -> tuple[np.ndarray, np.ndarray]:
        """Get all samples in buffer."""
        return np.array(list(self._features)), np.array(list(self._targets))

    def __len__(self) -> int:
        """Get buffer size."""
        return len(self._features)


def create_sgd_learner(
    n_features: int,
    learning_rate: float = 0.01,
    loss: str = "squared",
    name: str = "SGD"
) -> SGDOnlineLearner:
    """
    Factory function to create SGD online learner.

    Args:
        n_features: Number of features
        learning_rate: Learning rate
        loss: Loss function
        name: Learner name

    Returns:
        SGDOnlineLearner instance
    """
    return SGDOnlineLearner(
        n_features=n_features,
        learning_rate=learning_rate,
        loss=loss,
        name=name
    )


def create_drift_aware_learner(
    n_features: int,
    learning_rate: float = 0.01,
    drift_method: str = "ddm",
    name: str = "DriftAwareLearner"
) -> OnlineLearnerWithDriftDetection:
    """
    Factory function to create drift-aware online learner.

    Args:
        n_features: Number of features
        learning_rate: Learning rate
        drift_method: Drift detection method
        name: Learner name

    Returns:
        OnlineLearnerWithDriftDetection instance
    """
    base_learner = SGDOnlineLearner(
        n_features=n_features,
        learning_rate=learning_rate,
        name=f"{name}_base"
    )

    if drift_method == "ddm":
        detector = DDMDetector()
    elif drift_method == "adwin":
        detector = ADWINDetector()
    elif drift_method == "page_hinkley":
        detector = PageHinkleyDetector()
    else:
        detector = DDMDetector()

    return OnlineLearnerWithDriftDetection(
        base_learner=base_learner,
        drift_detector=detector,
        name=name
    )
