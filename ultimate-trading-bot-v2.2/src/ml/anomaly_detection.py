"""
Anomaly Detection Module for Ultimate Trading Bot v2.2

Implements anomaly detection algorithms for identifying unusual market
conditions, outliers in trading data, and abnormal price movements.

Author: AI Assistant
Version: 2.2.0
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies."""
    POINT = "point"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"
    TREND = "trend"


class AnomalyDetectorType(Enum):
    """Types of anomaly detectors."""
    ZSCORE = "zscore"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"
    LOF = "lof"
    AUTOENCODER = "autoencoder"
    MAHALANOBIS = "mahalanobis"
    DBSCAN = "dbscan"


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    is_anomaly: bool
    anomaly_score: float
    anomaly_type: Optional[AnomalyType] = None
    confidence: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_anomaly": self.is_anomaly,
            "anomaly_score": self.anomaly_score,
            "anomaly_type": self.anomaly_type.value if self.anomaly_type else None,
            "confidence": self.confidence,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class AnomalyBatch:
    """Batch anomaly detection results."""
    anomaly_mask: np.ndarray
    anomaly_scores: np.ndarray
    n_anomalies: int
    anomaly_rate: float
    detector_name: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "anomaly_mask": self.anomaly_mask.tolist(),
            "anomaly_scores": self.anomaly_scores.tolist(),
            "n_anomalies": self.n_anomalies,
            "anomaly_rate": self.anomaly_rate,
            "detector_name": self.detector_name
        }


@dataclass
class AnomalyStats:
    """Anomaly detection statistics."""
    total_samples: int
    total_anomalies: int
    anomaly_rate: float
    avg_anomaly_score: float
    max_anomaly_score: float
    recent_anomalies: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_samples": self.total_samples,
            "total_anomalies": self.total_anomalies,
            "anomaly_rate": self.anomaly_rate,
            "avg_anomaly_score": self.avg_anomaly_score,
            "max_anomaly_score": self.max_anomaly_score,
            "recent_anomalies": self.recent_anomalies
        }


class BaseAnomalyDetector(ABC):
    """Base class for anomaly detectors."""

    def __init__(
        self,
        contamination: float = 0.1,
        name: str = "AnomalyDetector"
    ):
        """
        Initialize anomaly detector.

        Args:
            contamination: Expected proportion of anomalies
            name: Detector name
        """
        self.contamination = contamination
        self.name = name
        self._is_fitted = False
        self._threshold: Optional[float] = None
        self._n_samples = 0
        self._n_anomalies = 0
        self._anomaly_scores: list[float] = []

        logger.info(
            f"Initialized {self.__class__.__name__}: {name}, "
            f"contamination={contamination}"
        )

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseAnomalyDetector":
        """Fit detector to data."""
        pass

    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Score samples for anomaly."""
        pass

    def detect(self, X: np.ndarray) -> AnomalyBatch:
        """
        Detect anomalies in data.

        Args:
            X: Data to check for anomalies

        Returns:
            AnomalyBatch object
        """
        if not self._is_fitted:
            raise ValueError("Detector must be fitted before detection")

        scores = self.score_samples(X)

        if self._threshold is None:
            self._threshold = np.percentile(scores, 100 * (1 - self.contamination))

        anomaly_mask = scores > self._threshold

        n_anomalies = int(np.sum(anomaly_mask))
        anomaly_rate = n_anomalies / len(X)

        self._n_samples += len(X)
        self._n_anomalies += n_anomalies
        self._anomaly_scores.extend(scores.tolist())

        return AnomalyBatch(
            anomaly_mask=anomaly_mask,
            anomaly_scores=scores,
            n_anomalies=n_anomalies,
            anomaly_rate=anomaly_rate,
            detector_name=self.name
        )

    def detect_single(self, x: np.ndarray) -> AnomalyResult:
        """
        Detect anomaly for single sample.

        Args:
            x: Single sample

        Returns:
            AnomalyResult object
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        scores = self.score_samples(x)
        score = float(scores[0])

        if self._threshold is None:
            is_anomaly = score > 0
        else:
            is_anomaly = score > self._threshold

        if self._threshold and self._threshold > 0:
            confidence = min(1.0, score / (2 * self._threshold))
        else:
            confidence = 0.5

        self._n_samples += 1
        if is_anomaly:
            self._n_anomalies += 1
        self._anomaly_scores.append(score)

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_score=score,
            anomaly_type=AnomalyType.POINT if is_anomaly else None,
            confidence=confidence,
            details={"threshold": self._threshold}
        )

    def get_stats(self) -> AnomalyStats:
        """Get detection statistics."""
        recent_scores = self._anomaly_scores[-100:] if self._anomaly_scores else [0]

        return AnomalyStats(
            total_samples=self._n_samples,
            total_anomalies=self._n_anomalies,
            anomaly_rate=self._n_anomalies / max(self._n_samples, 1),
            avg_anomaly_score=float(np.mean(recent_scores)),
            max_anomaly_score=float(np.max(recent_scores)) if recent_scores else 0.0,
            recent_anomalies=sum(
                1 for s in recent_scores
                if self._threshold and s > self._threshold
            )
        )

    @property
    def is_fitted(self) -> bool:
        """Check if detector is fitted."""
        return self._is_fitted


class ZScoreDetector(BaseAnomalyDetector):
    """Z-Score based anomaly detector."""

    def __init__(
        self,
        threshold: float = 3.0,
        contamination: float = 0.1,
        name: str = "ZScore"
    ):
        """
        Initialize Z-Score detector.

        Args:
            threshold: Z-score threshold for anomaly
            contamination: Expected proportion of anomalies
            name: Detector name
        """
        super().__init__(contamination, name)

        self.z_threshold = threshold
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "ZScoreDetector":
        """
        Fit detector to data.

        Args:
            X: Training data

        Returns:
            Self
        """
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0)
        self._std[self._std == 0] = 1.0

        scores = self.score_samples(X)
        self._threshold = np.percentile(scores, 100 * (1 - self.contamination))

        self._is_fitted = True

        logger.info(f"Fitted ZScoreDetector: threshold={self._threshold:.4f}")

        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate anomaly scores.

        Args:
            X: Data to score

        Returns:
            Anomaly scores
        """
        if self._mean is None or self._std is None:
            return np.zeros(len(X))

        z_scores = np.abs((X - self._mean) / self._std)

        return np.max(z_scores, axis=1) if X.ndim > 1 else np.abs(z_scores)


class IQRDetector(BaseAnomalyDetector):
    """Interquartile Range (IQR) based anomaly detector."""

    def __init__(
        self,
        multiplier: float = 1.5,
        contamination: float = 0.1,
        name: str = "IQR"
    ):
        """
        Initialize IQR detector.

        Args:
            multiplier: IQR multiplier for outlier bounds
            contamination: Expected proportion of anomalies
            name: Detector name
        """
        super().__init__(contamination, name)

        self.multiplier = multiplier
        self._q1: Optional[np.ndarray] = None
        self._q3: Optional[np.ndarray] = None
        self._iqr: Optional[np.ndarray] = None
        self._lower_bound: Optional[np.ndarray] = None
        self._upper_bound: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "IQRDetector":
        """
        Fit detector to data.

        Args:
            X: Training data

        Returns:
            Self
        """
        self._q1 = np.percentile(X, 25, axis=0)
        self._q3 = np.percentile(X, 75, axis=0)
        self._iqr = self._q3 - self._q1

        self._lower_bound = self._q1 - self.multiplier * self._iqr
        self._upper_bound = self._q3 + self.multiplier * self._iqr

        scores = self.score_samples(X)
        self._threshold = np.percentile(scores, 100 * (1 - self.contamination))

        self._is_fitted = True

        logger.info(f"Fitted IQRDetector: threshold={self._threshold:.4f}")

        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate anomaly scores.

        Args:
            X: Data to score

        Returns:
            Anomaly scores
        """
        if self._lower_bound is None or self._upper_bound is None:
            return np.zeros(len(X))

        lower_deviation = np.maximum(0, self._lower_bound - X)
        upper_deviation = np.maximum(0, X - self._upper_bound)

        deviation = np.maximum(lower_deviation, upper_deviation)

        if X.ndim > 1:
            return np.max(deviation / (self._iqr + 1e-10), axis=1)
        return deviation / (self._iqr + 1e-10)


class IsolationForestDetector(BaseAnomalyDetector):
    """Isolation Forest anomaly detector."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int = 256,
        contamination: float = 0.1,
        name: str = "IsolationForest"
    ):
        """
        Initialize Isolation Forest detector.

        Args:
            n_estimators: Number of trees
            max_samples: Samples per tree
            contamination: Expected proportion of anomalies
            name: Detector name
        """
        super().__init__(contamination, name)

        self.n_estimators = n_estimators
        self.max_samples = max_samples

        self._trees: list[dict[str, Any]] = []
        self._n_features: int = 0

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        """
        Fit Isolation Forest to data.

        Args:
            X: Training data

        Returns:
            Self
        """
        n_samples, self._n_features = X.shape
        max_samples = min(self.max_samples, n_samples)

        self._trees = []

        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, size=max_samples, replace=False)
            X_sample = X[indices]

            tree = self._build_tree(X_sample, depth=0, max_depth=int(np.ceil(np.log2(max_samples))))
            self._trees.append(tree)

        scores = self.score_samples(X)
        self._threshold = np.percentile(scores, 100 * (1 - self.contamination))

        self._is_fitted = True

        logger.info(
            f"Fitted IsolationForestDetector: n_trees={len(self._trees)}, "
            f"threshold={self._threshold:.4f}"
        )

        return self

    def _build_tree(
        self,
        X: np.ndarray,
        depth: int,
        max_depth: int
    ) -> dict[str, Any]:
        """Build isolation tree."""
        n_samples = len(X)

        if depth >= max_depth or n_samples <= 1:
            return {"type": "leaf", "size": n_samples}

        feature_idx = np.random.randint(self._n_features)
        feature_values = X[:, feature_idx]

        min_val, max_val = np.min(feature_values), np.max(feature_values)

        if min_val == max_val:
            return {"type": "leaf", "size": n_samples}

        split_value = np.random.uniform(min_val, max_val)

        left_mask = feature_values < split_value
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return {"type": "leaf", "size": n_samples}

        return {
            "type": "internal",
            "feature": feature_idx,
            "threshold": split_value,
            "left": self._build_tree(X[left_mask], depth + 1, max_depth),
            "right": self._build_tree(X[right_mask], depth + 1, max_depth)
        }

    def _path_length(self, x: np.ndarray, tree: dict[str, Any], depth: int) -> float:
        """Calculate path length for sample."""
        if tree["type"] == "leaf":
            n = tree["size"]
            if n <= 1:
                return depth
            else:
                return depth + self._c(n)

        if x[tree["feature"]] < tree["threshold"]:
            return self._path_length(x, tree["left"], depth + 1)
        else:
            return self._path_length(x, tree["right"], depth + 1)

    def _c(self, n: int) -> float:
        """Average path length of unsuccessful search in BST."""
        if n <= 1:
            return 0
        return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate anomaly scores.

        Args:
            X: Data to score

        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if not self._trees:
            return np.zeros(len(X))

        scores = np.zeros(len(X))

        for i, x in enumerate(X):
            avg_path_length = np.mean([
                self._path_length(x, tree, 0) for tree in self._trees
            ])

            c_n = self._c(self.max_samples)
            scores[i] = 2 ** (-avg_path_length / c_n) if c_n > 0 else 0

        return scores


class LocalOutlierFactorDetector(BaseAnomalyDetector):
    """Local Outlier Factor (LOF) anomaly detector."""

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.1,
        name: str = "LOF"
    ):
        """
        Initialize LOF detector.

        Args:
            n_neighbors: Number of neighbors for LOF
            contamination: Expected proportion of anomalies
            name: Detector name
        """
        super().__init__(contamination, name)

        self.n_neighbors = n_neighbors
        self._training_data: Optional[np.ndarray] = None
        self._k_distances: Optional[np.ndarray] = None
        self._lrd: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "LocalOutlierFactorDetector":
        """
        Fit LOF detector to data.

        Args:
            X: Training data

        Returns:
            Self
        """
        self._training_data = X.copy()
        n_samples = len(X)

        k = min(self.n_neighbors, n_samples - 1)

        distances = self._compute_distances(X, X)

        sorted_indices = np.argsort(distances, axis=1)

        self._k_distances = np.zeros(n_samples)
        self._lrd = np.zeros(n_samples)

        for i in range(n_samples):
            k_neighbor_idx = sorted_indices[i, k]
            self._k_distances[i] = distances[i, k_neighbor_idx]

        for i in range(n_samples):
            neighbors = sorted_indices[i, 1:k + 1]
            reach_distances = np.maximum(
                distances[i, neighbors],
                self._k_distances[neighbors]
            )
            self._lrd[i] = 1 / (np.mean(reach_distances) + 1e-10)

        scores = self.score_samples(X)
        self._threshold = np.percentile(scores, 100 * (1 - self.contamination))

        self._is_fitted = True

        logger.info(f"Fitted LOFDetector: threshold={self._threshold:.4f}")

        return self

    def _compute_distances(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute pairwise Euclidean distances."""
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)
        Y_sq = np.sum(Y ** 2, axis=1, keepdims=True)
        XY = X @ Y.T

        distances = np.sqrt(np.maximum(0, X_sq - 2 * XY + Y_sq.T))

        return distances

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate LOF scores.

        Args:
            X: Data to score

        Returns:
            LOF scores (higher = more anomalous)
        """
        if self._training_data is None or self._lrd is None:
            return np.zeros(len(X))

        distances = self._compute_distances(X, self._training_data)

        k = min(self.n_neighbors, len(self._training_data) - 1)
        sorted_indices = np.argsort(distances, axis=1)

        scores = np.zeros(len(X))

        for i in range(len(X)):
            neighbors = sorted_indices[i, :k]
            reach_distances = np.maximum(
                distances[i, neighbors],
                self._k_distances[neighbors]
            )

            lrd_x = 1 / (np.mean(reach_distances) + 1e-10)

            lof = np.mean(self._lrd[neighbors]) / (lrd_x + 1e-10)
            scores[i] = lof

        return scores


class MahalanobisDetector(BaseAnomalyDetector):
    """Mahalanobis distance based anomaly detector."""

    def __init__(
        self,
        contamination: float = 0.1,
        name: str = "Mahalanobis"
    ):
        """
        Initialize Mahalanobis detector.

        Args:
            contamination: Expected proportion of anomalies
            name: Detector name
        """
        super().__init__(contamination, name)

        self._mean: Optional[np.ndarray] = None
        self._cov_inv: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "MahalanobisDetector":
        """
        Fit Mahalanobis detector to data.

        Args:
            X: Training data

        Returns:
            Self
        """
        self._mean = np.mean(X, axis=0)

        cov = np.cov(X.T)

        if cov.ndim == 0:
            cov = np.array([[cov]])
        elif cov.ndim == 1:
            cov = np.diag(cov)

        try:
            self._cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            self._cov_inv = np.linalg.pinv(cov)

        scores = self.score_samples(X)
        self._threshold = np.percentile(scores, 100 * (1 - self.contamination))

        self._is_fitted = True

        logger.info(f"Fitted MahalanobisDetector: threshold={self._threshold:.4f}")

        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate Mahalanobis distances.

        Args:
            X: Data to score

        Returns:
            Mahalanobis distances
        """
        if self._mean is None or self._cov_inv is None:
            return np.zeros(len(X))

        diff = X - self._mean

        if diff.ndim == 1:
            diff = diff.reshape(1, -1)

        distances = np.sqrt(np.sum(diff @ self._cov_inv * diff, axis=1))

        return distances


class AutoencoderDetector(BaseAnomalyDetector):
    """Autoencoder-based anomaly detector."""

    def __init__(
        self,
        encoding_dim: int = 8,
        hidden_dims: list[int] = [32, 16],
        learning_rate: float = 0.001,
        epochs: int = 100,
        contamination: float = 0.1,
        name: str = "Autoencoder"
    ):
        """
        Initialize Autoencoder detector.

        Args:
            encoding_dim: Dimension of encoding layer
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
            epochs: Training epochs
            contamination: Expected proportion of anomalies
            name: Detector name
        """
        super().__init__(contamination, name)

        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.epochs = epochs

        self._weights: dict[str, np.ndarray] = {}
        self._input_dim: int = 0

    def fit(self, X: np.ndarray) -> "AutoencoderDetector":
        """
        Fit Autoencoder to data.

        Args:
            X: Training data

        Returns:
            Self
        """
        self._input_dim = X.shape[1]

        encoder_dims = [self._input_dim] + self.hidden_dims + [self.encoding_dim]
        decoder_dims = [self.encoding_dim] + self.hidden_dims[::-1] + [self._input_dim]

        for i in range(len(encoder_dims) - 1):
            scale = np.sqrt(2.0 / encoder_dims[i])
            self._weights[f"enc_W{i}"] = np.random.randn(
                encoder_dims[i], encoder_dims[i + 1]
            ) * scale
            self._weights[f"enc_b{i}"] = np.zeros(encoder_dims[i + 1])

        for i in range(len(decoder_dims) - 1):
            scale = np.sqrt(2.0 / decoder_dims[i])
            self._weights[f"dec_W{i}"] = np.random.randn(
                decoder_dims[i], decoder_dims[i + 1]
            ) * scale
            self._weights[f"dec_b{i}"] = np.zeros(decoder_dims[i + 1])

        batch_size = min(32, len(X))
        n_batches = len(X) // batch_size

        for epoch in range(self.epochs):
            indices = np.random.permutation(len(X))
            epoch_loss = 0.0

            for batch_idx in range(n_batches):
                batch_start = batch_idx * batch_size
                batch_indices = indices[batch_start:batch_start + batch_size]
                X_batch = X[batch_indices]

                reconstructed, activations = self._forward(X_batch)

                loss = np.mean((X_batch - reconstructed) ** 2)
                epoch_loss += loss

                self._backward(X_batch, reconstructed, activations)

            if (epoch + 1) % 20 == 0:
                logger.debug(f"Epoch {epoch + 1}: loss={epoch_loss / n_batches:.6f}")

        scores = self.score_samples(X)
        self._threshold = np.percentile(scores, 100 * (1 - self.contamination))

        self._is_fitted = True

        logger.info(f"Fitted AutoencoderDetector: threshold={self._threshold:.4f}")

        return self

    def _forward(
        self,
        X: np.ndarray
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Forward pass through autoencoder."""
        activations = {"input": X}

        x = X
        n_encoder_layers = len(self.hidden_dims) + 1

        for i in range(n_encoder_layers):
            x = x @ self._weights[f"enc_W{i}"] + self._weights[f"enc_b{i}"]
            if i < n_encoder_layers - 1:
                x = np.maximum(0, x)
            activations[f"enc_{i}"] = x

        n_decoder_layers = len(self.hidden_dims) + 1

        for i in range(n_decoder_layers):
            x = x @ self._weights[f"dec_W{i}"] + self._weights[f"dec_b{i}"]
            if i < n_decoder_layers - 1:
                x = np.maximum(0, x)
            activations[f"dec_{i}"] = x

        return x, activations

    def _backward(
        self,
        X: np.ndarray,
        reconstructed: np.ndarray,
        activations: dict[str, np.ndarray]
    ) -> None:
        """Backward pass and weight update."""
        grad = 2 * (reconstructed - X) / len(X)

        n_decoder_layers = len(self.hidden_dims) + 1

        for i in range(n_decoder_layers - 1, -1, -1):
            if i < n_decoder_layers - 1:
                grad = grad * (activations[f"dec_{i}"] > 0)

            if i == 0:
                prev_activation = activations[f"enc_{len(self.hidden_dims)}"]
            else:
                prev_activation = activations[f"dec_{i - 1}"]

            dW = prev_activation.T @ grad
            db = np.sum(grad, axis=0)

            self._weights[f"dec_W{i}"] -= self.learning_rate * dW
            self._weights[f"dec_b{i}"] -= self.learning_rate * db

            grad = grad @ self._weights[f"dec_W{i}"].T

        n_encoder_layers = len(self.hidden_dims) + 1

        for i in range(n_encoder_layers - 1, -1, -1):
            if i < n_encoder_layers - 1:
                grad = grad * (activations[f"enc_{i}"] > 0)

            if i == 0:
                prev_activation = activations["input"]
            else:
                prev_activation = activations[f"enc_{i - 1}"]

            dW = prev_activation.T @ grad
            db = np.sum(grad, axis=0)

            self._weights[f"enc_W{i}"] -= self.learning_rate * dW
            self._weights[f"enc_b{i}"] -= self.learning_rate * db

            if i > 0:
                grad = grad @ self._weights[f"enc_W{i}"].T

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate reconstruction errors as anomaly scores.

        Args:
            X: Data to score

        Returns:
            Reconstruction errors
        """
        if not self._weights:
            return np.zeros(len(X))

        reconstructed, _ = self._forward(X)

        reconstruction_errors = np.mean((X - reconstructed) ** 2, axis=1)

        return reconstruction_errors


class EnsembleAnomalyDetector:
    """Ensemble of multiple anomaly detectors."""

    def __init__(
        self,
        detectors: list[BaseAnomalyDetector],
        combination_method: str = "average",
        name: str = "EnsembleDetector"
    ):
        """
        Initialize ensemble detector.

        Args:
            detectors: List of anomaly detectors
            combination_method: How to combine scores
            name: Detector name
        """
        self.detectors = detectors
        self.combination_method = combination_method
        self.name = name

        logger.info(
            f"Initialized EnsembleAnomalyDetector: {name} with "
            f"{len(detectors)} detectors"
        )

    def fit(self, X: np.ndarray) -> "EnsembleAnomalyDetector":
        """
        Fit all detectors.

        Args:
            X: Training data

        Returns:
            Self
        """
        for detector in self.detectors:
            detector.fit(X)

        logger.info(f"Fitted ensemble with {len(self.detectors)} detectors")

        return self

    def detect(self, X: np.ndarray) -> AnomalyBatch:
        """
        Detect anomalies using ensemble.

        Args:
            X: Data to check

        Returns:
            AnomalyBatch object
        """
        all_scores = []

        for detector in self.detectors:
            batch = detector.detect(X)
            all_scores.append(batch.anomaly_scores)

        scores_matrix = np.array(all_scores)

        if self.combination_method == "average":
            combined_scores = np.mean(scores_matrix, axis=0)
        elif self.combination_method == "max":
            combined_scores = np.max(scores_matrix, axis=0)
        elif self.combination_method == "voting":
            votes = np.zeros(len(X))
            for i, detector in enumerate(self.detectors):
                threshold = detector._threshold or 0
                votes += (scores_matrix[i] > threshold).astype(float)
            combined_scores = votes / len(self.detectors)
        else:
            combined_scores = np.mean(scores_matrix, axis=0)

        threshold = np.percentile(combined_scores, 90)
        anomaly_mask = combined_scores > threshold

        return AnomalyBatch(
            anomaly_mask=anomaly_mask,
            anomaly_scores=combined_scores,
            n_anomalies=int(np.sum(anomaly_mask)),
            anomaly_rate=np.mean(anomaly_mask),
            detector_name=self.name
        )


def create_zscore_detector(
    threshold: float = 3.0,
    contamination: float = 0.1,
    name: str = "ZScore"
) -> ZScoreDetector:
    """
    Factory function to create Z-Score detector.

    Args:
        threshold: Z-score threshold
        contamination: Expected anomaly rate
        name: Detector name

    Returns:
        ZScoreDetector instance
    """
    return ZScoreDetector(
        threshold=threshold,
        contamination=contamination,
        name=name
    )


def create_isolation_forest_detector(
    n_estimators: int = 100,
    contamination: float = 0.1,
    name: str = "IsolationForest"
) -> IsolationForestDetector:
    """
    Factory function to create Isolation Forest detector.

    Args:
        n_estimators: Number of trees
        contamination: Expected anomaly rate
        name: Detector name

    Returns:
        IsolationForestDetector instance
    """
    return IsolationForestDetector(
        n_estimators=n_estimators,
        contamination=contamination,
        name=name
    )


def create_autoencoder_detector(
    encoding_dim: int = 8,
    epochs: int = 100,
    contamination: float = 0.1,
    name: str = "Autoencoder"
) -> AutoencoderDetector:
    """
    Factory function to create Autoencoder detector.

    Args:
        encoding_dim: Encoding dimension
        epochs: Training epochs
        contamination: Expected anomaly rate
        name: Detector name

    Returns:
        AutoencoderDetector instance
    """
    return AutoencoderDetector(
        encoding_dim=encoding_dim,
        epochs=epochs,
        contamination=contamination,
        name=name
    )
