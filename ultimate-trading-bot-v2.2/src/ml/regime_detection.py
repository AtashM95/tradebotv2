"""
Regime Detection Module for Ultimate Trading Bot v2.2

Implements market regime detection using Hidden Markov Models,
clustering, and change point detection for adaptive trading strategies.

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


class MarketRegime(Enum):
    """Market regime types."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"


class RegimeDetectorType(Enum):
    """Types of regime detectors."""
    HMM = "hmm"
    CLUSTERING = "clustering"
    THRESHOLD = "threshold"
    CHANGE_POINT = "change_point"


@dataclass
class RegimeState:
    """Current regime state."""
    regime: MarketRegime
    probability: float
    duration: int
    regime_id: int
    features: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "regime": self.regime.value,
            "probability": self.probability,
            "duration": self.duration,
            "regime_id": self.regime_id,
            "features": self.features,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class RegimeTransition:
    """Regime transition event."""
    from_regime: MarketRegime
    to_regime: MarketRegime
    transition_probability: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "from_regime": self.from_regime.value,
            "to_regime": self.to_regime.value,
            "transition_probability": self.transition_probability,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class RegimeAnalysis:
    """Complete regime analysis."""
    current_regime: RegimeState
    regime_history: list[int]
    regime_probabilities: np.ndarray
    transition_matrix: np.ndarray
    regime_durations: dict[int, list[int]]
    change_points: list[int]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "current_regime": self.current_regime.to_dict(),
            "regime_history": self.regime_history,
            "regime_probabilities": self.regime_probabilities.tolist(),
            "transition_matrix": self.transition_matrix.tolist(),
            "regime_durations": {k: v for k, v in self.regime_durations.items()},
            "change_points": self.change_points
        }


class BaseRegimeDetector(ABC):
    """Base class for regime detectors."""

    def __init__(
        self,
        n_regimes: int = 3,
        name: str = "RegimeDetector"
    ):
        """
        Initialize regime detector.

        Args:
            n_regimes: Number of regimes to detect
            name: Detector name
        """
        self.n_regimes = n_regimes
        self.name = name
        self._is_fitted = False
        self._current_regime: int = 0
        self._regime_history: list[int] = []
        self._regime_names: dict[int, MarketRegime] = {}

        logger.info(
            f"Initialized {self.__class__.__name__}: {name}, "
            f"n_regimes={n_regimes}"
        )

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseRegimeDetector":
        """Fit detector to data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict regimes for data."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict regime probabilities."""
        pass

    def get_current_regime(self) -> RegimeState:
        """Get current regime state."""
        duration = 1
        for i in range(len(self._regime_history) - 2, -1, -1):
            if self._regime_history[i] == self._current_regime:
                duration += 1
            else:
                break

        regime_type = self._regime_names.get(
            self._current_regime,
            MarketRegime.SIDEWAYS
        )

        return RegimeState(
            regime=regime_type,
            probability=1.0,
            duration=duration,
            regime_id=self._current_regime
        )

    def _assign_regime_names(self, X: np.ndarray, labels: np.ndarray) -> None:
        """Assign meaningful names to regime clusters."""
        for regime_id in range(self.n_regimes):
            mask = labels == regime_id
            if np.sum(mask) == 0:
                self._regime_names[regime_id] = MarketRegime.SIDEWAYS
                continue

            regime_data = X[mask]

            if X.shape[1] >= 2:
                avg_return = np.mean(regime_data[:, 0])
                avg_volatility = np.mean(regime_data[:, 1])
            else:
                avg_return = np.mean(regime_data[:, 0])
                avg_volatility = np.std(regime_data[:, 0])

            if avg_return > 0.001:
                if avg_volatility > 0.02:
                    self._regime_names[regime_id] = MarketRegime.HIGH_VOLATILITY
                else:
                    self._regime_names[regime_id] = MarketRegime.BULL
            elif avg_return < -0.001:
                if avg_volatility > 0.02:
                    self._regime_names[regime_id] = MarketRegime.HIGH_VOLATILITY
                else:
                    self._regime_names[regime_id] = MarketRegime.BEAR
            else:
                if avg_volatility < 0.01:
                    self._regime_names[regime_id] = MarketRegime.LOW_VOLATILITY
                else:
                    self._regime_names[regime_id] = MarketRegime.SIDEWAYS

    @property
    def is_fitted(self) -> bool:
        """Check if detector is fitted."""
        return self._is_fitted


class HiddenMarkovModelDetector(BaseRegimeDetector):
    """Hidden Markov Model for regime detection."""

    def __init__(
        self,
        n_regimes: int = 3,
        n_iterations: int = 100,
        convergence_threshold: float = 1e-6,
        name: str = "HMM"
    ):
        """
        Initialize HMM detector.

        Args:
            n_regimes: Number of hidden states
            n_iterations: Maximum iterations for EM
            convergence_threshold: Convergence threshold
            name: Detector name
        """
        super().__init__(n_regimes, name)

        self.n_iterations = n_iterations
        self.convergence_threshold = convergence_threshold

        self._initial_probs: Optional[np.ndarray] = None
        self._transition_matrix: Optional[np.ndarray] = None
        self._means: Optional[np.ndarray] = None
        self._covariances: Optional[np.ndarray] = None
        self._n_features: int = 0

    def fit(self, X: np.ndarray) -> "HiddenMarkovModelDetector":
        """
        Fit HMM using EM algorithm.

        Args:
            X: Observation sequences (n_samples, n_features)

        Returns:
            Self
        """
        n_samples, self._n_features = X.shape
        n_states = self.n_regimes

        self._initial_probs = np.ones(n_states) / n_states

        self._transition_matrix = np.ones((n_states, n_states)) / n_states
        for i in range(n_states):
            self._transition_matrix[i, i] = 0.7
            other_prob = 0.3 / (n_states - 1)
            for j in range(n_states):
                if i != j:
                    self._transition_matrix[i, j] = other_prob

        indices = np.array_split(np.arange(n_samples), n_states)
        self._means = np.array([np.mean(X[idx], axis=0) for idx in indices])
        self._covariances = np.array([
            np.cov(X[idx].T) + np.eye(self._n_features) * 0.01
            for idx in indices
        ])

        prev_log_likelihood = float("-inf")

        for iteration in range(self.n_iterations):
            alpha, scaling_factors = self._forward(X)
            beta = self._backward(X, scaling_factors)
            gamma = self._compute_gamma(alpha, beta)
            xi = self._compute_xi(X, alpha, beta, scaling_factors)

            self._update_parameters(X, gamma, xi)

            log_likelihood = np.sum(np.log(scaling_factors + 1e-300))

            if abs(log_likelihood - prev_log_likelihood) < self.convergence_threshold:
                logger.debug(f"HMM converged at iteration {iteration}")
                break

            prev_log_likelihood = log_likelihood

        labels = self.predict(X)
        self._assign_regime_names(X, labels)
        self._regime_history = labels.tolist()
        self._current_regime = labels[-1]

        self._is_fitted = True

        logger.info(f"Fitted HMM with {n_states} states")

        return self

    def _gaussian_pdf(self, x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
        """Calculate multivariate Gaussian PDF."""
        k = len(mean)
        diff = x - mean

        try:
            cov_inv = np.linalg.inv(cov)
            det = np.linalg.det(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.eye(k)
            det = 1.0

        det = max(det, 1e-300)

        exponent = -0.5 * diff @ cov_inv @ diff
        normalizer = 1.0 / (np.sqrt((2 * np.pi) ** k * det))

        return normalizer * np.exp(exponent)

    def _emission_probs(self, x: np.ndarray) -> np.ndarray:
        """Calculate emission probabilities for observation."""
        if self._means is None or self._covariances is None:
            return np.ones(self.n_regimes) / self.n_regimes

        probs = np.array([
            self._gaussian_pdf(x, self._means[i], self._covariances[i])
            for i in range(self.n_regimes)
        ])

        return probs / (np.sum(probs) + 1e-300)

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward algorithm."""
        n_samples = len(X)
        n_states = self.n_regimes

        alpha = np.zeros((n_samples, n_states))
        scaling_factors = np.zeros(n_samples)

        emission = self._emission_probs(X[0])
        alpha[0] = self._initial_probs * emission
        scaling_factors[0] = np.sum(alpha[0]) + 1e-300
        alpha[0] /= scaling_factors[0]

        for t in range(1, n_samples):
            emission = self._emission_probs(X[t])
            alpha[t] = emission * (alpha[t - 1] @ self._transition_matrix)
            scaling_factors[t] = np.sum(alpha[t]) + 1e-300
            alpha[t] /= scaling_factors[t]

        return alpha, scaling_factors

    def _backward(
        self,
        X: np.ndarray,
        scaling_factors: np.ndarray
    ) -> np.ndarray:
        """Backward algorithm."""
        n_samples = len(X)
        n_states = self.n_regimes

        beta = np.zeros((n_samples, n_states))
        beta[-1] = 1.0

        for t in range(n_samples - 2, -1, -1):
            emission = self._emission_probs(X[t + 1])
            beta[t] = self._transition_matrix @ (emission * beta[t + 1])
            beta[t] /= scaling_factors[t + 1] + 1e-300

        return beta

    def _compute_gamma(
        self,
        alpha: np.ndarray,
        beta: np.ndarray
    ) -> np.ndarray:
        """Compute state probabilities."""
        gamma = alpha * beta
        gamma /= np.sum(gamma, axis=1, keepdims=True) + 1e-300
        return gamma

    def _compute_xi(
        self,
        X: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        scaling_factors: np.ndarray
    ) -> np.ndarray:
        """Compute transition probabilities."""
        n_samples = len(X)
        n_states = self.n_regimes

        xi = np.zeros((n_samples - 1, n_states, n_states))

        for t in range(n_samples - 1):
            emission = self._emission_probs(X[t + 1])

            numerator = np.outer(alpha[t], emission * beta[t + 1]) * self._transition_matrix
            xi[t] = numerator / (np.sum(numerator) + 1e-300)

        return xi

    def _update_parameters(
        self,
        X: np.ndarray,
        gamma: np.ndarray,
        xi: np.ndarray
    ) -> None:
        """Update HMM parameters."""
        self._initial_probs = gamma[0]

        transition_sum = np.sum(xi, axis=0)
        gamma_sum = np.sum(gamma[:-1], axis=0, keepdims=True).T
        self._transition_matrix = transition_sum / (gamma_sum + 1e-300)

        for i in range(self.n_regimes):
            weights = gamma[:, i]
            weights_sum = np.sum(weights) + 1e-300

            self._means[i] = np.average(X, axis=0, weights=weights)

            diff = X - self._means[i]
            weighted_cov = np.sum(
                weights[:, np.newaxis, np.newaxis] *
                np.einsum('ij,ik->ijk', diff, diff),
                axis=0
            )
            self._covariances[i] = weighted_cov / weights_sum + np.eye(self._n_features) * 0.01

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict most likely regime sequence (Viterbi).

        Args:
            X: Observations

        Returns:
            Predicted regime sequence
        """
        if not self._is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        n_samples = len(X)
        n_states = self.n_regimes

        viterbi = np.zeros((n_samples, n_states))
        backpointer = np.zeros((n_samples, n_states), dtype=int)

        emission = self._emission_probs(X[0])
        viterbi[0] = np.log(self._initial_probs + 1e-300) + np.log(emission + 1e-300)

        for t in range(1, n_samples):
            emission = self._emission_probs(X[t])

            for j in range(n_states):
                trans_probs = viterbi[t - 1] + np.log(self._transition_matrix[:, j] + 1e-300)
                backpointer[t, j] = np.argmax(trans_probs)
                viterbi[t, j] = trans_probs[backpointer[t, j]] + np.log(emission[j] + 1e-300)

        path = np.zeros(n_samples, dtype=int)
        path[-1] = np.argmax(viterbi[-1])

        for t in range(n_samples - 2, -1, -1):
            path[t] = backpointer[t + 1, path[t + 1]]

        self._regime_history = path.tolist()
        self._current_regime = path[-1]

        return path

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regime probabilities.

        Args:
            X: Observations

        Returns:
            Regime probabilities (n_samples, n_regimes)
        """
        if not self._is_fitted:
            raise ValueError("Detector must be fitted before prediction")

        alpha, scaling_factors = self._forward(X)
        beta = self._backward(X, scaling_factors)
        gamma = self._compute_gamma(alpha, beta)

        return gamma

    def get_transition_matrix(self) -> np.ndarray:
        """Get learned transition matrix."""
        return self._transition_matrix.copy() if self._transition_matrix is not None else np.array([])


class KMeansRegimeDetector(BaseRegimeDetector):
    """K-Means clustering for regime detection."""

    def __init__(
        self,
        n_regimes: int = 3,
        n_iterations: int = 100,
        n_init: int = 10,
        name: str = "KMeansRegime"
    ):
        """
        Initialize K-Means regime detector.

        Args:
            n_regimes: Number of clusters/regimes
            n_iterations: Maximum iterations per run
            n_init: Number of initializations
            name: Detector name
        """
        super().__init__(n_regimes, name)

        self.n_iterations = n_iterations
        self.n_init = n_init

        self._centroids: Optional[np.ndarray] = None
        self._transition_matrix: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "KMeansRegimeDetector":
        """
        Fit K-Means clustering.

        Args:
            X: Feature data

        Returns:
            Self
        """
        best_inertia = float("inf")
        best_centroids = None
        best_labels = None

        for _ in range(self.n_init):
            centroids, labels, inertia = self._kmeans_run(X)

            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels

        self._centroids = best_centroids
        labels = best_labels

        self._compute_transition_matrix(labels)

        self._assign_regime_names(X, labels)
        self._regime_history = labels.tolist()
        self._current_regime = labels[-1]

        self._is_fitted = True

        logger.info(f"Fitted KMeans with {self.n_regimes} clusters")

        return self

    def _kmeans_run(
        self,
        X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Single K-Means run."""
        n_samples = len(X)

        indices = np.random.choice(n_samples, size=self.n_regimes, replace=False)
        centroids = X[indices].copy()

        for _ in range(self.n_iterations):
            distances = self._compute_distances(X, centroids)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_regimes):
                mask = labels == k
                if np.sum(mask) > 0:
                    new_centroids[k] = np.mean(X[mask], axis=0)
                else:
                    new_centroids[k] = centroids[k]

            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        distances = self._compute_distances(X, centroids)
        labels = np.argmin(distances, axis=1)
        inertia = np.sum(np.min(distances, axis=1) ** 2)

        return centroids, labels, inertia

    def _compute_distances(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Compute distances to centroids."""
        distances = np.zeros((len(X), len(centroids)))
        for k, centroid in enumerate(centroids):
            distances[:, k] = np.sqrt(np.sum((X - centroid) ** 2, axis=1))
        return distances

    def _compute_transition_matrix(self, labels: np.ndarray) -> None:
        """Compute regime transition matrix."""
        self._transition_matrix = np.zeros((self.n_regimes, self.n_regimes))

        for i in range(len(labels) - 1):
            self._transition_matrix[labels[i], labels[i + 1]] += 1

        row_sums = self._transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self._transition_matrix /= row_sums

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regime labels.

        Args:
            X: Feature data

        Returns:
            Predicted regime labels
        """
        if self._centroids is None:
            raise ValueError("Detector must be fitted before prediction")

        distances = self._compute_distances(X, self._centroids)
        labels = np.argmin(distances, axis=1)

        self._regime_history.extend(labels.tolist())
        self._current_regime = labels[-1]

        return labels

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regime probabilities using distance-based softmax.

        Args:
            X: Feature data

        Returns:
            Regime probabilities
        """
        if self._centroids is None:
            raise ValueError("Detector must be fitted before prediction")

        distances = self._compute_distances(X, self._centroids)

        neg_distances = -distances
        max_neg = np.max(neg_distances, axis=1, keepdims=True)
        exp_neg = np.exp(neg_distances - max_neg)
        probs = exp_neg / np.sum(exp_neg, axis=1, keepdims=True)

        return probs


class ThresholdRegimeDetector(BaseRegimeDetector):
    """Threshold-based regime detection using market indicators."""

    def __init__(
        self,
        volatility_thresholds: tuple[float, float] = (0.01, 0.03),
        trend_thresholds: tuple[float, float] = (-0.001, 0.001),
        name: str = "ThresholdRegime"
    ):
        """
        Initialize threshold-based detector.

        Args:
            volatility_thresholds: (low, high) volatility thresholds
            trend_thresholds: (bear, bull) trend thresholds
            name: Detector name
        """
        super().__init__(n_regimes=5, name=name)

        self.vol_low, self.vol_high = volatility_thresholds
        self.trend_bear, self.trend_bull = trend_thresholds

        self._regime_names = {
            0: MarketRegime.BULL,
            1: MarketRegime.BEAR,
            2: MarketRegime.SIDEWAYS,
            3: MarketRegime.HIGH_VOLATILITY,
            4: MarketRegime.LOW_VOLATILITY
        }

    def fit(self, X: np.ndarray) -> "ThresholdRegimeDetector":
        """
        Fit detector (learns nothing, just validates).

        Args:
            X: Feature data (returns, volatility)

        Returns:
            Self
        """
        labels = self.predict(X)
        self._regime_history = labels.tolist()
        self._current_regime = labels[-1]

        self._is_fitted = True

        logger.info("Fitted ThresholdRegimeDetector")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regimes using thresholds.

        Args:
            X: Feature data (returns, volatility)

        Returns:
            Predicted regime labels
        """
        if X.shape[1] < 2:
            returns = X[:, 0]
            volatility = np.abs(returns)
        else:
            returns = X[:, 0]
            volatility = X[:, 1]

        labels = np.zeros(len(X), dtype=int)

        for i in range(len(X)):
            ret = returns[i]
            vol = volatility[i]

            if vol > self.vol_high:
                labels[i] = 3
            elif vol < self.vol_low:
                labels[i] = 4
            elif ret > self.trend_bull:
                labels[i] = 0
            elif ret < self.trend_bear:
                labels[i] = 1
            else:
                labels[i] = 2

        self._regime_history.extend(labels.tolist())
        self._current_regime = labels[-1]

        return labels

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regime probabilities (deterministic).

        Args:
            X: Feature data

        Returns:
            Regime probabilities
        """
        labels = self.predict(X)
        probs = np.zeros((len(X), self.n_regimes))

        for i, label in enumerate(labels):
            probs[i, label] = 0.8
            other_prob = 0.2 / (self.n_regimes - 1)
            for j in range(self.n_regimes):
                if j != label:
                    probs[i, j] = other_prob

        return probs


class ChangePointDetector(BaseRegimeDetector):
    """Change point detection for regime shifts."""

    def __init__(
        self,
        min_segment_length: int = 20,
        penalty: float = 3.0,
        name: str = "ChangePointRegime"
    ):
        """
        Initialize change point detector.

        Args:
            min_segment_length: Minimum regime length
            penalty: Penalty for adding change points
            name: Detector name
        """
        super().__init__(n_regimes=0, name=name)

        self.min_segment_length = min_segment_length
        self.penalty = penalty

        self._change_points: list[int] = []
        self._segment_means: list[np.ndarray] = []

    def fit(self, X: np.ndarray) -> "ChangePointDetector":
        """
        Fit change point detector.

        Args:
            X: Feature data

        Returns:
            Self
        """
        self._change_points = self._find_change_points(X)

        self._segment_means = []
        segments = [0] + self._change_points + [len(X)]

        for i in range(len(segments) - 1):
            segment_data = X[segments[i]:segments[i + 1]]
            self._segment_means.append(np.mean(segment_data, axis=0))

        self.n_regimes = len(self._segment_means)

        labels = self.predict(X)
        self._assign_regime_names(X, labels)
        self._regime_history = labels.tolist()
        self._current_regime = labels[-1]

        self._is_fitted = True

        logger.info(
            f"Fitted ChangePointDetector: "
            f"found {len(self._change_points)} change points"
        )

        return self

    def _find_change_points(self, X: np.ndarray) -> list[int]:
        """Find change points using PELT-like algorithm."""
        n = len(X)
        change_points = []

        self._recursive_binary_segmentation(X, 0, n, change_points)

        change_points.sort()

        return change_points

    def _recursive_binary_segmentation(
        self,
        X: np.ndarray,
        start: int,
        end: int,
        change_points: list[int]
    ) -> None:
        """Recursive binary segmentation."""
        if end - start < 2 * self.min_segment_length:
            return

        segment = X[start:end]
        segment_cost = self._segment_cost(segment)

        best_gain = 0
        best_split = None

        for split in range(self.min_segment_length, len(segment) - self.min_segment_length):
            left_cost = self._segment_cost(segment[:split])
            right_cost = self._segment_cost(segment[split:])

            gain = segment_cost - left_cost - right_cost - self.penalty

            if gain > best_gain:
                best_gain = gain
                best_split = split

        if best_split is not None:
            change_points.append(start + best_split)

            self._recursive_binary_segmentation(
                X, start, start + best_split, change_points
            )
            self._recursive_binary_segmentation(
                X, start + best_split, end, change_points
            )

    def _segment_cost(self, segment: np.ndarray) -> float:
        """Calculate cost of a segment."""
        if len(segment) <= 1:
            return 0.0

        variance = np.var(segment, axis=0)
        return float(np.sum(variance) * len(segment))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regime labels based on change points.

        Args:
            X: Feature data

        Returns:
            Predicted regime labels
        """
        labels = np.zeros(len(X), dtype=int)
        segments = [0] + self._change_points + [len(X)]

        for i in range(len(segments) - 1):
            labels[segments[i]:segments[i + 1]] = i

        self._regime_history.extend(labels.tolist())
        self._current_regime = labels[-1]

        return labels

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regime probabilities.

        Args:
            X: Feature data

        Returns:
            Regime probabilities
        """
        labels = self.predict(X)
        probs = np.zeros((len(X), self.n_regimes))

        for i, label in enumerate(labels):
            probs[i, label] = 0.9
            if self.n_regimes > 1:
                other_prob = 0.1 / (self.n_regimes - 1)
                for j in range(self.n_regimes):
                    if j != label:
                        probs[i, j] = other_prob

        return probs

    def get_change_points(self) -> list[int]:
        """Get detected change points."""
        return self._change_points.copy()


class EnsembleRegimeDetector:
    """Ensemble of regime detectors."""

    def __init__(
        self,
        detectors: list[BaseRegimeDetector],
        combination_method: str = "voting",
        name: str = "EnsembleRegime"
    ):
        """
        Initialize ensemble detector.

        Args:
            detectors: List of regime detectors
            combination_method: How to combine predictions
            name: Ensemble name
        """
        self.detectors = detectors
        self.combination_method = combination_method
        self.name = name

        logger.info(
            f"Initialized EnsembleRegimeDetector: {name} with "
            f"{len(detectors)} detectors"
        )

    def fit(self, X: np.ndarray) -> "EnsembleRegimeDetector":
        """
        Fit all detectors.

        Args:
            X: Feature data

        Returns:
            Self
        """
        for detector in self.detectors:
            detector.fit(X)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regimes using ensemble.

        Args:
            X: Feature data

        Returns:
            Predicted regime labels
        """
        all_predictions = []

        for detector in self.detectors:
            preds = detector.predict(X)
            all_predictions.append(preds)

        predictions = np.array(all_predictions)

        if self.combination_method == "voting":
            labels = np.zeros(len(X), dtype=int)
            for i in range(len(X)):
                votes = predictions[:, i]
                labels[i] = np.argmax(np.bincount(votes.astype(int)))
        else:
            labels = predictions[0]

        return labels

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regime probabilities.

        Args:
            X: Feature data

        Returns:
            Averaged regime probabilities
        """
        all_probs = []

        for detector in self.detectors:
            probs = detector.predict_proba(X)
            all_probs.append(probs)

        return np.mean(all_probs, axis=0)


def create_hmm_regime_detector(
    n_regimes: int = 3,
    n_iterations: int = 100,
    name: str = "HMM"
) -> HiddenMarkovModelDetector:
    """
    Factory function to create HMM detector.

    Args:
        n_regimes: Number of hidden states
        n_iterations: EM iterations
        name: Detector name

    Returns:
        HiddenMarkovModelDetector instance
    """
    return HiddenMarkovModelDetector(
        n_regimes=n_regimes,
        n_iterations=n_iterations,
        name=name
    )


def create_kmeans_regime_detector(
    n_regimes: int = 3,
    name: str = "KMeans"
) -> KMeansRegimeDetector:
    """
    Factory function to create K-Means detector.

    Args:
        n_regimes: Number of clusters
        name: Detector name

    Returns:
        KMeansRegimeDetector instance
    """
    return KMeansRegimeDetector(
        n_regimes=n_regimes,
        name=name
    )


def create_threshold_regime_detector(
    name: str = "Threshold"
) -> ThresholdRegimeDetector:
    """
    Factory function to create threshold detector.

    Args:
        name: Detector name

    Returns:
        ThresholdRegimeDetector instance
    """
    return ThresholdRegimeDetector(name=name)
