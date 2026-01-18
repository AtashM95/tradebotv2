"""
Sentiment Scorer for Ultimate Trading Bot v2.2.

Provides scoring, normalization, and calibration utilities for sentiment analysis.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

from .base_analyzer import (
    SentimentLabel,
    SentimentResult,
    SentimentSource,
)

logger = logging.getLogger(__name__)


class NormalizationMethod(str, Enum):
    """Methods for normalizing sentiment scores."""
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    ROBUST = "robust"
    SIGMOID = "sigmoid"
    TANH = "tanh"


class CalibrationMethod(str, Enum):
    """Methods for calibrating sentiment scores."""
    PLATT = "platt"
    ISOTONIC = "isotonic"
    HISTOGRAM = "histogram"
    BETA = "beta"


@dataclass
class ScoringConfig:
    """Configuration for sentiment scoring."""

    normalization_method: NormalizationMethod = NormalizationMethod.TANH
    calibration_method: CalibrationMethod | None = None

    # Score boundaries
    min_score: float = -1.0
    max_score: float = 1.0

    # Confidence calculation
    min_confidence: float = 0.0
    max_confidence: float = 1.0
    confidence_decay: float = 0.95

    # Label thresholds
    very_negative_threshold: float = -0.6
    negative_threshold: float = -0.2
    positive_threshold: float = 0.2
    very_positive_threshold: float = 0.6

    # Historical calibration
    use_historical_calibration: bool = True
    calibration_window: int = 1000  # samples


@dataclass
class ScoreDistribution:
    """Statistics about score distribution."""

    mean: float = 0.0
    std: float = 1.0
    median: float = 0.0
    min_value: float = -1.0
    max_value: float = 1.0
    q25: float = -0.25
    q75: float = 0.25
    skewness: float = 0.0
    kurtosis: float = 0.0
    sample_count: int = 0


@dataclass
class CalibratedScore:
    """A calibrated sentiment score."""

    raw_score: float
    calibrated_score: float
    confidence: float
    label: SentimentLabel

    # Calibration details
    calibration_method: str | None = None
    calibration_adjustment: float = 0.0

    # Probabilities
    probability_positive: float = 0.5
    probability_negative: float = 0.5
    probability_neutral: float = 0.0

    metadata: dict[str, Any] = field(default_factory=dict)


class SentimentScorer:
    """
    Scores and normalizes sentiment values.

    Provides consistent scoring across different sentiment sources.
    """

    def __init__(self, config: ScoringConfig | None = None) -> None:
        """
        Initialize sentiment scorer.

        Args:
            config: Scoring configuration
        """
        self.config = config or ScoringConfig()

        # Historical scores for calibration
        self._score_history: list[float] = []
        self._distribution: ScoreDistribution = ScoreDistribution()

        # Calibration parameters
        self._calibration_params: dict[str, Any] = {}

        logger.info("Initialized SentimentScorer")

    def score(
        self,
        raw_score: float,
        confidence: float | None = None,
    ) -> CalibratedScore:
        """
        Score and calibrate a sentiment value.

        Args:
            raw_score: Raw sentiment score
            confidence: Optional confidence value

        Returns:
            Calibrated score
        """
        # Normalize score
        normalized = self._normalize(raw_score)

        # Calibrate if configured
        if self.config.calibration_method and self._calibration_params:
            calibrated = self._calibrate(normalized)
            adjustment = calibrated - normalized
        else:
            calibrated = normalized
            adjustment = 0.0

        # Calculate confidence
        if confidence is None:
            confidence = self._estimate_confidence(calibrated)
        else:
            confidence = self._adjust_confidence(confidence, calibrated)

        # Determine label
        label = self._score_to_label(calibrated)

        # Calculate probabilities
        probs = self._calculate_probabilities(calibrated)

        # Update history
        self._update_history(calibrated)

        return CalibratedScore(
            raw_score=raw_score,
            calibrated_score=calibrated,
            confidence=confidence,
            label=label,
            calibration_method=(
                self.config.calibration_method.value
                if self.config.calibration_method else None
            ),
            calibration_adjustment=adjustment,
            probability_positive=probs["positive"],
            probability_negative=probs["negative"],
            probability_neutral=probs["neutral"],
        )

    def score_batch(
        self,
        raw_scores: list[float],
        confidences: list[float] | None = None,
    ) -> list[CalibratedScore]:
        """
        Score multiple sentiment values.

        Args:
            raw_scores: List of raw scores
            confidences: Optional list of confidences

        Returns:
            List of calibrated scores
        """
        if confidences is None:
            confidences = [None] * len(raw_scores)

        return [
            self.score(score, conf)
            for score, conf in zip(raw_scores, confidences)
        ]

    def _normalize(self, score: float) -> float:
        """Normalize a score to configured range."""
        method = self.config.normalization_method

        if method == NormalizationMethod.MIN_MAX:
            return self._min_max_normalize(score)
        elif method == NormalizationMethod.Z_SCORE:
            return self._z_score_normalize(score)
        elif method == NormalizationMethod.ROBUST:
            return self._robust_normalize(score)
        elif method == NormalizationMethod.SIGMOID:
            return self._sigmoid_normalize(score)
        elif method == NormalizationMethod.TANH:
            return self._tanh_normalize(score)
        else:
            return max(self.config.min_score, min(self.config.max_score, score))

    def _min_max_normalize(self, score: float) -> float:
        """Min-max normalization."""
        if self._distribution.max_value == self._distribution.min_value:
            return 0.0

        normalized = (score - self._distribution.min_value) / (
            self._distribution.max_value - self._distribution.min_value
        )

        # Scale to [-1, 1]
        return normalized * 2 - 1

    def _z_score_normalize(self, score: float) -> float:
        """Z-score normalization."""
        if self._distribution.std == 0:
            return 0.0

        z = (score - self._distribution.mean) / self._distribution.std

        # Clip to reasonable range
        return max(-3.0, min(3.0, z)) / 3.0

    def _robust_normalize(self, score: float) -> float:
        """Robust normalization using median and IQR."""
        iqr = self._distribution.q75 - self._distribution.q25
        if iqr == 0:
            return 0.0

        normalized = (score - self._distribution.median) / iqr
        return max(-1.0, min(1.0, normalized / 2.0))

    def _sigmoid_normalize(self, score: float) -> float:
        """Sigmoid normalization."""
        # Map to [0, 1] using sigmoid
        sigmoid = 1.0 / (1.0 + np.exp(-score))
        # Map to [-1, 1]
        return sigmoid * 2 - 1

    def _tanh_normalize(self, score: float) -> float:
        """Tanh normalization (bounded to [-1, 1])."""
        return float(np.tanh(score))

    def _calibrate(self, score: float) -> float:
        """Calibrate a normalized score."""
        method = self.config.calibration_method

        if method == CalibrationMethod.PLATT:
            return self._platt_calibrate(score)
        elif method == CalibrationMethod.ISOTONIC:
            return self._isotonic_calibrate(score)
        elif method == CalibrationMethod.HISTOGRAM:
            return self._histogram_calibrate(score)
        elif method == CalibrationMethod.BETA:
            return self._beta_calibrate(score)
        else:
            return score

    def _platt_calibrate(self, score: float) -> float:
        """Platt scaling calibration."""
        a = self._calibration_params.get("platt_a", 1.0)
        b = self._calibration_params.get("platt_b", 0.0)

        # Apply Platt scaling
        calibrated = 1.0 / (1.0 + np.exp(a * score + b))

        # Map back to [-1, 1]
        return calibrated * 2 - 1

    def _isotonic_calibrate(self, score: float) -> float:
        """Isotonic regression calibration."""
        # Simplified isotonic: use piecewise linear
        breakpoints = self._calibration_params.get(
            "isotonic_breakpoints",
            [(-1.0, -1.0), (0.0, 0.0), (1.0, 1.0)]
        )

        for i in range(len(breakpoints) - 1):
            x1, y1 = breakpoints[i]
            x2, y2 = breakpoints[i + 1]

            if x1 <= score <= x2:
                # Linear interpolation
                if x2 == x1:
                    return y1
                return y1 + (y2 - y1) * (score - x1) / (x2 - x1)

        return score

    def _histogram_calibrate(self, score: float) -> float:
        """Histogram binning calibration."""
        bins = self._calibration_params.get("histogram_bins", 10)
        bin_values = self._calibration_params.get(
            "histogram_values",
            [i / bins * 2 - 1 for i in range(bins + 1)]
        )

        # Find bin
        bin_idx = int((score + 1) / 2 * bins)
        bin_idx = max(0, min(bins - 1, bin_idx))

        return bin_values[bin_idx]

    def _beta_calibrate(self, score: float) -> float:
        """Beta calibration."""
        a = self._calibration_params.get("beta_a", 1.0)
        b = self._calibration_params.get("beta_b", 1.0)
        c = self._calibration_params.get("beta_c", 1.0)

        # Convert to [0, 1]
        p = (score + 1) / 2

        # Apply beta calibration
        calibrated = 1.0 / (1.0 + 1.0 / (c * (p ** a) * ((1 - p) ** b) + 1e-10))

        # Convert back to [-1, 1]
        return calibrated * 2 - 1

    def _estimate_confidence(self, score: float) -> float:
        """Estimate confidence from score magnitude."""
        # Higher magnitude = higher confidence
        magnitude = abs(score)

        # Non-linear mapping
        confidence = magnitude ** 0.5

        return min(self.config.max_confidence, confidence)

    def _adjust_confidence(
        self,
        confidence: float,
        score: float,
    ) -> float:
        """Adjust confidence based on score."""
        # Reduce confidence for extreme scores
        magnitude = abs(score)

        if magnitude > 0.9:
            # Very extreme scores might be less reliable
            confidence *= 0.9

        return max(
            self.config.min_confidence,
            min(self.config.max_confidence, confidence)
        )

    def _score_to_label(self, score: float) -> SentimentLabel:
        """Convert score to sentiment label."""
        if score <= self.config.very_negative_threshold:
            return SentimentLabel.VERY_NEGATIVE
        elif score <= self.config.negative_threshold:
            return SentimentLabel.NEGATIVE
        elif score >= self.config.very_positive_threshold:
            return SentimentLabel.VERY_POSITIVE
        elif score >= self.config.positive_threshold:
            return SentimentLabel.POSITIVE
        else:
            return SentimentLabel.NEUTRAL

    def _calculate_probabilities(
        self,
        score: float,
    ) -> dict[str, float]:
        """Calculate probability distribution over sentiments."""
        # Use softmax-like distribution
        # Map score to probabilities

        # Distance from thresholds
        dist_positive = max(0, score - self.config.positive_threshold)
        dist_negative = max(0, -score - abs(self.config.negative_threshold))
        dist_neutral = max(0, 1 - abs(score) / self.config.positive_threshold)

        # Normalize
        total = dist_positive + dist_negative + dist_neutral + 1e-10

        return {
            "positive": dist_positive / total,
            "negative": dist_negative / total,
            "neutral": dist_neutral / total,
        }

    def _update_history(self, score: float) -> None:
        """Update score history and distribution."""
        self._score_history.append(score)

        # Keep limited history
        if len(self._score_history) > self.config.calibration_window:
            self._score_history = self._score_history[-self.config.calibration_window:]

        # Update distribution
        if len(self._score_history) >= 10:
            self._update_distribution()

    def _update_distribution(self) -> None:
        """Update score distribution statistics."""
        scores = np.array(self._score_history)

        self._distribution = ScoreDistribution(
            mean=float(np.mean(scores)),
            std=float(np.std(scores)),
            median=float(np.median(scores)),
            min_value=float(np.min(scores)),
            max_value=float(np.max(scores)),
            q25=float(np.percentile(scores, 25)),
            q75=float(np.percentile(scores, 75)),
            skewness=self._calculate_skewness(scores),
            kurtosis=self._calculate_kurtosis(scores),
            sample_count=len(scores),
        )

    def _calculate_skewness(self, scores: np.ndarray) -> float:
        """Calculate skewness of scores."""
        if len(scores) < 3:
            return 0.0

        mean = np.mean(scores)
        std = np.std(scores)

        if std == 0:
            return 0.0

        return float(np.mean(((scores - mean) / std) ** 3))

    def _calculate_kurtosis(self, scores: np.ndarray) -> float:
        """Calculate kurtosis of scores."""
        if len(scores) < 4:
            return 0.0

        mean = np.mean(scores)
        std = np.std(scores)

        if std == 0:
            return 0.0

        return float(np.mean(((scores - mean) / std) ** 4) - 3)

    def fit_calibration(
        self,
        scores: list[float],
        labels: list[int],
    ) -> None:
        """
        Fit calibration parameters from labeled data.

        Args:
            scores: Predicted scores
            labels: True labels (1 for positive, -1 for negative)
        """
        if self.config.calibration_method == CalibrationMethod.PLATT:
            self._fit_platt(scores, labels)
        elif self.config.calibration_method == CalibrationMethod.ISOTONIC:
            self._fit_isotonic(scores, labels)
        elif self.config.calibration_method == CalibrationMethod.HISTOGRAM:
            self._fit_histogram(scores, labels)
        elif self.config.calibration_method == CalibrationMethod.BETA:
            self._fit_beta(scores, labels)

    def _fit_platt(
        self,
        scores: list[float],
        labels: list[int],
    ) -> None:
        """Fit Platt scaling parameters."""
        # Simplified Platt fitting
        scores_arr = np.array(scores)
        labels_arr = np.array(labels)

        # Convert labels to 0/1
        y = (labels_arr + 1) / 2

        # Simple logistic regression fit
        # Using gradient descent
        a, b = 1.0, 0.0
        lr = 0.01

        for _ in range(1000):
            pred = 1.0 / (1.0 + np.exp(a * scores_arr + b))
            grad_a = np.mean((pred - y) * scores_arr)
            grad_b = np.mean(pred - y)

            a -= lr * grad_a
            b -= lr * grad_b

        self._calibration_params["platt_a"] = a
        self._calibration_params["platt_b"] = b

    def _fit_isotonic(
        self,
        scores: list[float],
        labels: list[int],
    ) -> None:
        """Fit isotonic regression."""
        # Sort by score
        paired = sorted(zip(scores, labels))
        sorted_scores = [s for s, _ in paired]
        sorted_labels = [l for _, l in paired]

        # Pool adjacent violators algorithm (simplified)
        breakpoints = []
        n = len(sorted_scores)

        if n == 0:
            return

        # Create initial breakpoints
        step = max(1, n // 10)
        for i in range(0, n, step):
            x = sorted_scores[i]
            y = np.mean(sorted_labels[max(0, i-step//2):min(n, i+step//2+1)])
            breakpoints.append((x, y))

        self._calibration_params["isotonic_breakpoints"] = breakpoints

    def _fit_histogram(
        self,
        scores: list[float],
        labels: list[int],
    ) -> None:
        """Fit histogram binning."""
        bins = 10
        bin_values = []

        for i in range(bins):
            low = -1.0 + i * 2.0 / bins
            high = -1.0 + (i + 1) * 2.0 / bins

            bin_labels = [
                l for s, l in zip(scores, labels)
                if low <= s < high
            ]

            if bin_labels:
                bin_values.append(np.mean(bin_labels))
            else:
                bin_values.append(low + (high - low) / 2)

        self._calibration_params["histogram_bins"] = bins
        self._calibration_params["histogram_values"] = bin_values

    def _fit_beta(
        self,
        scores: list[float],
        labels: list[int],
    ) -> None:
        """Fit beta calibration parameters."""
        # Simplified beta calibration fitting
        # Use default parameters with slight adjustment based on data
        scores_arr = np.array(scores)
        labels_arr = np.array(labels)

        # Estimate parameters from data statistics
        pos_scores = scores_arr[labels_arr > 0]
        neg_scores = scores_arr[labels_arr < 0]

        if len(pos_scores) > 0 and len(neg_scores) > 0:
            pos_mean = np.mean(pos_scores)
            neg_mean = np.mean(neg_scores)

            # Adjust parameters based on separation
            separation = pos_mean - neg_mean
            self._calibration_params["beta_a"] = 1.0 + separation
            self._calibration_params["beta_b"] = 1.0 - separation
            self._calibration_params["beta_c"] = 1.0
        else:
            self._calibration_params["beta_a"] = 1.0
            self._calibration_params["beta_b"] = 1.0
            self._calibration_params["beta_c"] = 1.0

    def get_distribution(self) -> ScoreDistribution:
        """Get current score distribution."""
        return self._distribution

    def reset(self) -> None:
        """Reset scorer state."""
        self._score_history.clear()
        self._distribution = ScoreDistribution()
        self._calibration_params.clear()


class SourceScorer:
    """
    Scores sentiment by source with source-specific calibration.
    """

    def __init__(self, config: ScoringConfig | None = None) -> None:
        """
        Initialize source scorer.

        Args:
            config: Scoring configuration
        """
        self.config = config or ScoringConfig()

        # Scorers by source
        self._scorers: dict[SentimentSource, SentimentScorer] = {}

        # Source reliability
        self._source_reliability: dict[SentimentSource, float] = {
            SentimentSource.NEWS: 0.85,
            SentimentSource.TWITTER: 0.70,
            SentimentSource.REDDIT: 0.65,
            SentimentSource.STOCKTWITS: 0.75,
            SentimentSource.MARKET: 0.90,
            SentimentSource.ANALYST: 0.88,
            SentimentSource.COMBINED: 0.80,
        }

        logger.info("Initialized SourceScorer")

    def score(
        self,
        raw_score: float,
        source: SentimentSource,
        confidence: float | None = None,
    ) -> CalibratedScore:
        """
        Score sentiment for a specific source.

        Args:
            raw_score: Raw sentiment score
            source: Sentiment source
            confidence: Optional confidence

        Returns:
            Calibrated score
        """
        # Get or create scorer for source
        if source not in self._scorers:
            self._scorers[source] = SentimentScorer(self.config)

        scorer = self._scorers[source]
        result = scorer.score(raw_score, confidence)

        # Adjust by source reliability
        reliability = self._source_reliability.get(source, 0.7)
        result.confidence *= reliability

        return result

    def score_result(
        self,
        result: SentimentResult,
    ) -> CalibratedScore:
        """
        Score a sentiment result.

        Args:
            result: Sentiment result

        Returns:
            Calibrated score
        """
        return self.score(
            result.score,
            result.source,
            result.confidence,
        )

    def set_source_reliability(
        self,
        source: SentimentSource,
        reliability: float,
    ) -> None:
        """
        Set reliability for a source.

        Args:
            source: Sentiment source
            reliability: Reliability value (0-1)
        """
        self._source_reliability[source] = max(0.0, min(1.0, reliability))

    def get_source_reliability(
        self,
        source: SentimentSource,
    ) -> float:
        """Get reliability for a source."""
        return self._source_reliability.get(source, 0.7)


def create_sentiment_scorer(
    config: ScoringConfig | None = None,
) -> SentimentScorer:
    """
    Create a sentiment scorer.

    Args:
        config: Scoring configuration

    Returns:
        Sentiment scorer instance
    """
    return SentimentScorer(config)


def create_source_scorer(
    config: ScoringConfig | None = None,
) -> SourceScorer:
    """
    Create a source-aware sentiment scorer.

    Args:
        config: Scoring configuration

    Returns:
        Source scorer instance
    """
    return SourceScorer(config)
