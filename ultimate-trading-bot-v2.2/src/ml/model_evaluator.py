"""
Model Evaluator Module for Ultimate Trading Bot v2.2

Comprehensive model evaluation with classification, regression, and trading metrics.
Provides statistical tests, cross-validation analysis, and performance benchmarking.

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


class MetricType(Enum):
    """Types of evaluation metrics."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TRADING = "trading"
    STATISTICAL = "statistical"


class ComparisonMethod(Enum):
    """Statistical comparison methods."""
    TTEST = "ttest"
    WILCOXON = "wilcoxon"
    BOOTSTRAP = "bootstrap"
    PERMUTATION = "permutation"


@dataclass
class ClassificationMetrics:
    """Classification model metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    auc_pr: float
    log_loss: float
    confusion_matrix: np.ndarray
    class_report: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "auc_roc": self.auc_roc,
            "auc_pr": self.auc_pr,
            "log_loss": self.log_loss,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "class_report": self.class_report
        }


@dataclass
class RegressionMetrics:
    """Regression model metrics."""
    mse: float
    rmse: float
    mae: float
    mape: float
    r2: float
    adjusted_r2: float
    explained_variance: float
    max_error: float
    median_absolute_error: float
    residual_std: float

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "r2": self.r2,
            "adjusted_r2": self.adjusted_r2,
            "explained_variance": self.explained_variance,
            "max_error": self.max_error,
            "median_absolute_error": self.median_absolute_error,
            "residual_std": self.residual_std
        }


@dataclass
class TradingMetrics:
    """Trading-specific metrics."""
    hit_rate: float
    profit_factor: float
    win_loss_ratio: float
    expected_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    information_ratio: float
    directional_accuracy: float

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "hit_rate": self.hit_rate,
            "profit_factor": self.profit_factor,
            "win_loss_ratio": self.win_loss_ratio,
            "expected_return": self.expected_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
            "information_ratio": self.information_ratio,
            "directional_accuracy": self.directional_accuracy
        }


@dataclass
class StatisticalTestResult:
    """Result of statistical test."""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[tuple[float, float]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "confidence_level": self.confidence_level,
            "effect_size": self.effect_size,
            "confidence_interval": self.confidence_interval
        }


@dataclass
class ModelComparisonResult:
    """Result of model comparison."""
    model_names: list[str]
    metric_name: str
    metric_values: list[float]
    best_model: str
    statistical_test: StatisticalTestResult
    ranking: list[tuple[str, float]]

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "model_names": self.model_names,
            "metric_name": self.metric_name,
            "metric_values": self.metric_values,
            "best_model": self.best_model,
            "statistical_test": self.statistical_test.to_dict(),
            "ranking": self.ranking
        }


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    model_name: str
    evaluation_timestamp: datetime
    classification_metrics: Optional[ClassificationMetrics] = None
    regression_metrics: Optional[RegressionMetrics] = None
    trading_metrics: Optional[TradingMetrics] = None
    cross_validation_scores: Optional[dict[str, list[float]]] = None
    feature_importance: Optional[dict[str, float]] = None
    predictions: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        result: dict[str, Any] = {
            "model_name": self.model_name,
            "evaluation_timestamp": self.evaluation_timestamp.isoformat(),
            "metadata": self.metadata
        }

        if self.classification_metrics:
            result["classification_metrics"] = self.classification_metrics.to_dict()
        if self.regression_metrics:
            result["regression_metrics"] = self.regression_metrics.to_dict()
        if self.trading_metrics:
            result["trading_metrics"] = self.trading_metrics.to_dict()
        if self.cross_validation_scores:
            result["cross_validation_scores"] = self.cross_validation_scores
        if self.feature_importance:
            result["feature_importance"] = self.feature_importance

        return result


class BaseMetricCalculator(ABC):
    """Base class for metric calculators."""

    @abstractmethod
    def calculate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        **kwargs: Any
    ) -> Any:
        """Calculate metrics."""
        pass


class ClassificationMetricCalculator(BaseMetricCalculator):
    """Calculator for classification metrics."""

    def __init__(self, average: str = "weighted"):
        """
        Initialize calculator.

        Args:
            average: Averaging method for multiclass
        """
        self.average = average
        logger.info(f"Initialized ClassificationMetricCalculator with average={average}")

    def calculate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> ClassificationMetrics:
        """
        Calculate classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            **kwargs: Additional arguments

        Returns:
            ClassificationMetrics object
        """
        try:
            accuracy = self._calculate_accuracy(y_true, y_pred)
            precision = self._calculate_precision(y_true, y_pred)
            recall = self._calculate_recall(y_true, y_pred)
            f1 = self._calculate_f1(y_true, y_pred)

            confusion_mat = self._calculate_confusion_matrix(y_true, y_pred)

            if y_prob is not None:
                auc_roc = self._calculate_auc_roc(y_true, y_prob)
                auc_pr = self._calculate_auc_pr(y_true, y_prob)
                log_loss_val = self._calculate_log_loss(y_true, y_prob)
            else:
                auc_roc = 0.0
                auc_pr = 0.0
                log_loss_val = 0.0

            class_report = self._generate_class_report(y_true, y_pred)

            metrics = ClassificationMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                auc_roc=auc_roc,
                auc_pr=auc_pr,
                log_loss=log_loss_val,
                confusion_matrix=confusion_mat,
                class_report=class_report
            )

            logger.info(f"Calculated classification metrics: accuracy={accuracy:.4f}")
            return metrics

        except Exception as e:
            logger.error(f"Error calculating classification metrics: {e}")
            raise

    def _calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy score."""
        return float(np.mean(y_true == y_pred))

    def _calculate_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate precision score."""
        classes = np.unique(np.concatenate([y_true, y_pred]))
        precisions = []
        weights = []

        for cls in classes:
            tp = np.sum((y_pred == cls) & (y_true == cls))
            fp = np.sum((y_pred == cls) & (y_true != cls))

            if tp + fp > 0:
                prec = tp / (tp + fp)
            else:
                prec = 0.0

            precisions.append(prec)
            weights.append(np.sum(y_true == cls))

        if self.average == "weighted":
            total_weight = sum(weights)
            if total_weight > 0:
                return float(np.average(precisions, weights=weights))
            return 0.0
        elif self.average == "macro":
            return float(np.mean(precisions))
        else:
            return float(precisions[1]) if len(precisions) > 1 else float(precisions[0])

    def _calculate_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate recall score."""
        classes = np.unique(np.concatenate([y_true, y_pred]))
        recalls = []
        weights = []

        for cls in classes:
            tp = np.sum((y_pred == cls) & (y_true == cls))
            fn = np.sum((y_pred != cls) & (y_true == cls))

            if tp + fn > 0:
                rec = tp / (tp + fn)
            else:
                rec = 0.0

            recalls.append(rec)
            weights.append(np.sum(y_true == cls))

        if self.average == "weighted":
            total_weight = sum(weights)
            if total_weight > 0:
                return float(np.average(recalls, weights=weights))
            return 0.0
        elif self.average == "macro":
            return float(np.mean(recalls))
        else:
            return float(recalls[1]) if len(recalls) > 1 else float(recalls[0])

    def _calculate_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F1 score."""
        precision = self._calculate_precision(y_true, y_pred)
        recall = self._calculate_recall(y_true, y_pred)

        if precision + recall > 0:
            return 2 * (precision * recall) / (precision + recall)
        return 0.0

    def _calculate_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """Calculate confusion matrix."""
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)

        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

        for true_val, pred_val in zip(y_true, y_pred):
            true_idx = class_to_idx[true_val]
            pred_idx = class_to_idx[pred_val]
            matrix[true_idx, pred_idx] += 1

        return matrix

    def _calculate_auc_roc(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> float:
        """Calculate AUC-ROC score."""
        if y_prob.ndim > 1 and y_prob.shape[1] > 1:
            y_score = y_prob[:, 1]
        else:
            y_score = y_prob.flatten()

        sorted_indices = np.argsort(y_score)[::-1]
        y_true_sorted = y_true[sorted_indices]

        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        if n_pos == 0 or n_neg == 0:
            return 0.5

        tpr_values = []
        fpr_values = []

        tp = 0
        fp = 0

        for label in y_true_sorted:
            if label == 1:
                tp += 1
            else:
                fp += 1

            tpr_values.append(tp / n_pos)
            fpr_values.append(fp / n_neg)

        auc = 0.0
        for i in range(1, len(fpr_values)):
            auc += (fpr_values[i] - fpr_values[i-1]) * tpr_values[i]

        return float(auc)

    def _calculate_auc_pr(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> float:
        """Calculate AUC-PR score."""
        if y_prob.ndim > 1 and y_prob.shape[1] > 1:
            y_score = y_prob[:, 1]
        else:
            y_score = y_prob.flatten()

        sorted_indices = np.argsort(y_score)[::-1]
        y_true_sorted = y_true[sorted_indices]

        n_pos = np.sum(y_true == 1)

        if n_pos == 0:
            return 0.0

        precisions = []
        recalls = []

        tp = 0

        for i, label in enumerate(y_true_sorted, 1):
            if label == 1:
                tp += 1

            precision = tp / i
            recall = tp / n_pos

            precisions.append(precision)
            recalls.append(recall)

        auc = 0.0
        for i in range(1, len(recalls)):
            if recalls[i] != recalls[i-1]:
                auc += (recalls[i] - recalls[i-1]) * precisions[i]

        return float(auc)

    def _calculate_log_loss(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> float:
        """Calculate log loss."""
        eps = 1e-15

        if y_prob.ndim > 1 and y_prob.shape[1] > 1:
            y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
            n_samples = len(y_true)
            log_loss_val = 0.0

            for i in range(n_samples):
                true_class = int(y_true[i])
                log_loss_val -= np.log(y_prob_clipped[i, true_class])

            return float(log_loss_val / n_samples)
        else:
            y_prob_flat = y_prob.flatten()
            y_prob_clipped = np.clip(y_prob_flat, eps, 1 - eps)

            log_loss_val = -np.mean(
                y_true * np.log(y_prob_clipped) +
                (1 - y_true) * np.log(1 - y_prob_clipped)
            )

            return float(log_loss_val)

    def _generate_class_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> dict[str, dict[str, float]]:
        """Generate per-class report."""
        classes = np.unique(np.concatenate([y_true, y_pred]))
        report: dict[str, dict[str, float]] = {}

        for cls in classes:
            tp = np.sum((y_pred == cls) & (y_true == cls))
            fp = np.sum((y_pred == cls) & (y_true != cls))
            fn = np.sum((y_pred != cls) & (y_true == cls))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            support = float(np.sum(y_true == cls))

            report[str(cls)] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": support
            }

        return report


class RegressionMetricCalculator(BaseMetricCalculator):
    """Calculator for regression metrics."""

    def __init__(self, n_features: int = 1):
        """
        Initialize calculator.

        Args:
            n_features: Number of features for adjusted R2
        """
        self.n_features = n_features
        logger.info(f"Initialized RegressionMetricCalculator with n_features={n_features}")

    def calculate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        **kwargs: Any
    ) -> RegressionMetrics:
        """
        Calculate regression metrics.

        Args:
            y_true: True values
            y_pred: Predicted values
            **kwargs: Additional arguments

        Returns:
            RegressionMetrics object
        """
        try:
            n_features = kwargs.get("n_features", self.n_features)

            residuals = y_true - y_pred

            mse = float(np.mean(residuals ** 2))
            rmse = float(np.sqrt(mse))
            mae = float(np.mean(np.abs(residuals)))

            non_zero_mask = y_true != 0
            if np.any(non_zero_mask):
                mape = float(np.mean(np.abs(residuals[non_zero_mask] / y_true[non_zero_mask]))) * 100
            else:
                mape = 0.0

            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            n = len(y_true)
            if n > n_features + 1:
                adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
            else:
                adjusted_r2 = r2

            var_y = np.var(y_true)
            if var_y > 0:
                explained_var = 1 - np.var(residuals) / var_y
            else:
                explained_var = 0.0

            max_error = float(np.max(np.abs(residuals)))
            median_ae = float(np.median(np.abs(residuals)))
            residual_std = float(np.std(residuals))

            metrics = RegressionMetrics(
                mse=mse,
                rmse=rmse,
                mae=mae,
                mape=mape,
                r2=float(r2),
                adjusted_r2=float(adjusted_r2),
                explained_variance=float(explained_var),
                max_error=max_error,
                median_absolute_error=median_ae,
                residual_std=residual_std
            )

            logger.info(f"Calculated regression metrics: R2={r2:.4f}, RMSE={rmse:.4f}")
            return metrics

        except Exception as e:
            logger.error(f"Error calculating regression metrics: {e}")
            raise


class TradingMetricCalculator(BaseMetricCalculator):
    """Calculator for trading-specific metrics."""

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ):
        """
        Initialize calculator.

        Args:
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        logger.info(
            f"Initialized TradingMetricCalculator with "
            f"risk_free_rate={risk_free_rate}, periods_per_year={periods_per_year}"
        )

    def calculate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        returns: Optional[np.ndarray] = None,
        benchmark_returns: Optional[np.ndarray] = None,
        **kwargs: Any
    ) -> TradingMetrics:
        """
        Calculate trading metrics.

        Args:
            y_true: True direction or returns
            y_pred: Predicted direction or returns
            returns: Actual returns achieved
            benchmark_returns: Benchmark returns for comparison
            **kwargs: Additional arguments

        Returns:
            TradingMetrics object
        """
        try:
            directional_accuracy = self._calculate_directional_accuracy(y_true, y_pred)

            if returns is None:
                returns = y_true * np.sign(y_pred)

            hit_rate = self._calculate_hit_rate(returns)
            profit_factor = self._calculate_profit_factor(returns)
            win_loss_ratio = self._calculate_win_loss_ratio(returns)
            expected_return = self._calculate_expected_return(returns)

            sharpe = self._calculate_sharpe_ratio(returns)
            sortino = self._calculate_sortino_ratio(returns)
            max_dd = self._calculate_max_drawdown(returns)
            calmar = self._calculate_calmar_ratio(returns, max_dd)

            if benchmark_returns is not None:
                info_ratio = self._calculate_information_ratio(returns, benchmark_returns)
            else:
                info_ratio = 0.0

            metrics = TradingMetrics(
                hit_rate=hit_rate,
                profit_factor=profit_factor,
                win_loss_ratio=win_loss_ratio,
                expected_return=expected_return,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                max_drawdown=max_dd,
                calmar_ratio=calmar,
                information_ratio=info_ratio,
                directional_accuracy=directional_accuracy
            )

            logger.info(
                f"Calculated trading metrics: hit_rate={hit_rate:.4f}, "
                f"sharpe={sharpe:.4f}"
            )
            return metrics

        except Exception as e:
            logger.error(f"Error calculating trading metrics: {e}")
            raise

    def _calculate_directional_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """Calculate directional accuracy."""
        true_direction = np.sign(y_true)
        pred_direction = np.sign(y_pred)
        return float(np.mean(true_direction == pred_direction))

    def _calculate_hit_rate(self, returns: np.ndarray) -> float:
        """Calculate hit rate (win rate)."""
        wins = np.sum(returns > 0)
        total = len(returns)
        return float(wins / total) if total > 0 else 0.0

    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor."""
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = np.abs(np.sum(returns[returns < 0]))

        if gross_loss > 0:
            return float(gross_profit / gross_loss)
        elif gross_profit > 0:
            return float("inf")
        else:
            return 0.0

    def _calculate_win_loss_ratio(self, returns: np.ndarray) -> float:
        """Calculate average win/loss ratio."""
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        avg_win = np.mean(wins) if len(wins) > 0 else 0.0
        avg_loss = np.abs(np.mean(losses)) if len(losses) > 0 else 0.0

        if avg_loss > 0:
            return float(avg_win / avg_loss)
        elif avg_win > 0:
            return float("inf")
        else:
            return 0.0

    def _calculate_expected_return(self, returns: np.ndarray) -> float:
        """Calculate expected return per trade."""
        return float(np.mean(returns))

    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        excess_return = mean_return - self.risk_free_rate / self.periods_per_year
        sharpe = excess_return / std_return * np.sqrt(self.periods_per_year)

        return float(sharpe)

    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio."""
        if len(returns) == 0:
            return 0.0

        mean_return = np.mean(returns)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0:
            return float("inf") if mean_return > 0 else 0.0

        downside_std = np.std(downside_returns)

        if downside_std == 0:
            return 0.0

        excess_return = mean_return - self.risk_free_rate / self.periods_per_year
        sortino = excess_return / downside_std * np.sqrt(self.periods_per_year)

        return float(sortino)

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0

        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max

        return float(np.min(drawdowns))

    def _calculate_calmar_ratio(
        self,
        returns: np.ndarray,
        max_drawdown: float
    ) -> float:
        """Calculate Calmar ratio."""
        if max_drawdown == 0:
            return 0.0

        annual_return = np.mean(returns) * self.periods_per_year
        return float(annual_return / abs(max_drawdown))

    def _calculate_information_ratio(
        self,
        returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> float:
        """Calculate information ratio."""
        if len(returns) != len(benchmark_returns):
            min_len = min(len(returns), len(benchmark_returns))
            returns = returns[:min_len]
            benchmark_returns = benchmark_returns[:min_len]

        active_returns = returns - benchmark_returns

        if len(active_returns) == 0:
            return 0.0

        mean_active = np.mean(active_returns)
        tracking_error = np.std(active_returns)

        if tracking_error == 0:
            return 0.0

        return float(mean_active / tracking_error * np.sqrt(self.periods_per_year))


class StatisticalTester:
    """Statistical tests for model comparison."""

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize tester.

        Args:
            confidence_level: Confidence level for tests
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        logger.info(f"Initialized StatisticalTester with confidence_level={confidence_level}")

    def paired_ttest(
        self,
        scores1: np.ndarray,
        scores2: np.ndarray
    ) -> StatisticalTestResult:
        """
        Perform paired t-test.

        Args:
            scores1: First set of scores
            scores2: Second set of scores

        Returns:
            StatisticalTestResult object
        """
        try:
            differences = scores1 - scores2
            n = len(differences)

            mean_diff = np.mean(differences)
            std_diff = np.std(differences, ddof=1)

            if std_diff == 0:
                return StatisticalTestResult(
                    test_name="paired_ttest",
                    statistic=0.0,
                    p_value=1.0,
                    is_significant=False,
                    confidence_level=self.confidence_level
                )

            t_stat = mean_diff / (std_diff / np.sqrt(n))

            df = n - 1
            p_value = self._calculate_t_pvalue(t_stat, df)

            pooled_std = np.sqrt((np.std(scores1) ** 2 + np.std(scores2) ** 2) / 2)
            effect_size = mean_diff / pooled_std if pooled_std > 0 else 0.0

            t_crit = self._get_t_critical(df, self.alpha / 2)
            margin = t_crit * std_diff / np.sqrt(n)
            ci = (mean_diff - margin, mean_diff + margin)

            return StatisticalTestResult(
                test_name="paired_ttest",
                statistic=float(t_stat),
                p_value=float(p_value),
                is_significant=p_value < self.alpha,
                confidence_level=self.confidence_level,
                effect_size=float(effect_size),
                confidence_interval=ci
            )

        except Exception as e:
            logger.error(f"Error in paired t-test: {e}")
            raise

    def wilcoxon_test(
        self,
        scores1: np.ndarray,
        scores2: np.ndarray
    ) -> StatisticalTestResult:
        """
        Perform Wilcoxon signed-rank test.

        Args:
            scores1: First set of scores
            scores2: Second set of scores

        Returns:
            StatisticalTestResult object
        """
        try:
            differences = scores1 - scores2
            differences = differences[differences != 0]

            n = len(differences)
            if n == 0:
                return StatisticalTestResult(
                    test_name="wilcoxon",
                    statistic=0.0,
                    p_value=1.0,
                    is_significant=False,
                    confidence_level=self.confidence_level
                )

            abs_diff = np.abs(differences)
            ranks = self._rankdata(abs_diff)

            signed_ranks = ranks * np.sign(differences)

            w_plus = np.sum(signed_ranks[signed_ranks > 0])
            w_minus = np.abs(np.sum(signed_ranks[signed_ranks < 0]))

            w_stat = min(w_plus, w_minus)

            mean_w = n * (n + 1) / 4
            std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)

            if std_w > 0:
                z_stat = (w_stat - mean_w) / std_w
                p_value = 2 * self._normal_cdf(-abs(z_stat))
            else:
                z_stat = 0.0
                p_value = 1.0

            r_effect = 1 - (2 * w_stat) / (n * (n + 1))

            return StatisticalTestResult(
                test_name="wilcoxon",
                statistic=float(w_stat),
                p_value=float(p_value),
                is_significant=p_value < self.alpha,
                confidence_level=self.confidence_level,
                effect_size=float(r_effect)
            )

        except Exception as e:
            logger.error(f"Error in Wilcoxon test: {e}")
            raise

    def bootstrap_test(
        self,
        scores1: np.ndarray,
        scores2: np.ndarray,
        n_bootstrap: int = 10000
    ) -> StatisticalTestResult:
        """
        Perform bootstrap hypothesis test.

        Args:
            scores1: First set of scores
            scores2: Second set of scores
            n_bootstrap: Number of bootstrap samples

        Returns:
            StatisticalTestResult object
        """
        try:
            observed_diff = np.mean(scores1) - np.mean(scores2)

            combined = np.concatenate([scores1, scores2])
            n1 = len(scores1)

            bootstrap_diffs = np.zeros(n_bootstrap)

            for i in range(n_bootstrap):
                permuted = np.random.permutation(combined)
                bootstrap_diffs[i] = np.mean(permuted[:n1]) - np.mean(permuted[n1:])

            p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))

            bootstrap_diffs_paired = np.zeros(n_bootstrap)
            differences = scores1 - scores2

            for i in range(n_bootstrap):
                indices = np.random.choice(len(differences), size=len(differences), replace=True)
                bootstrap_diffs_paired[i] = np.mean(differences[indices])

            ci_lower = np.percentile(bootstrap_diffs_paired, (1 - self.confidence_level) / 2 * 100)
            ci_upper = np.percentile(bootstrap_diffs_paired, (1 + self.confidence_level) / 2 * 100)

            pooled_std = np.std(combined)
            effect_size = observed_diff / pooled_std if pooled_std > 0 else 0.0

            return StatisticalTestResult(
                test_name="bootstrap",
                statistic=float(observed_diff),
                p_value=float(p_value),
                is_significant=p_value < self.alpha,
                confidence_level=self.confidence_level,
                effect_size=float(effect_size),
                confidence_interval=(float(ci_lower), float(ci_upper))
            )

        except Exception as e:
            logger.error(f"Error in bootstrap test: {e}")
            raise

    def _rankdata(self, data: np.ndarray) -> np.ndarray:
        """Compute ranks of data."""
        sorter = np.argsort(data)
        inv = np.empty(len(data), dtype=np.intp)
        inv[sorter] = np.arange(len(data))

        sorted_data = data[sorter]
        obs = np.concatenate([[True], sorted_data[1:] != sorted_data[:-1]])
        dense = obs.cumsum()[inv]

        count = np.concatenate([[0], np.where(obs)[0], [len(data)]])
        ranks = np.zeros(len(data))

        for i in range(len(count) - 1):
            ranks[dense == i + 1] = 0.5 * (count[i] + count[i + 1] + 1)

        return ranks

    def _calculate_t_pvalue(self, t_stat: float, df: int) -> float:
        """Calculate two-tailed p-value for t-distribution."""
        x = df / (df + t_stat ** 2)
        p_value = self._incomplete_beta(df / 2, 0.5, x)
        return float(p_value)

    def _incomplete_beta(self, a: float, b: float, x: float) -> float:
        """Regularized incomplete beta function (approximation)."""
        if x < 0 or x > 1:
            return 0.0

        if x < (a + 1) / (a + b + 2):
            return self._beta_cf(a, b, x) * (x ** a) * ((1 - x) ** b) / a
        else:
            return 1 - self._beta_cf(b, a, 1 - x) * ((1 - x) ** b) * (x ** a) / b

    def _beta_cf(self, a: float, b: float, x: float) -> float:
        """Continued fraction for incomplete beta."""
        max_iter = 100
        eps = 1e-10

        c = 1.0
        d = 1.0 / max(1 - (a + b) * x / (a + 1), eps)
        h = d

        for m in range(1, max_iter):
            m2 = 2 * m

            aa = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
            d = 1.0 / max(1 + aa * d, eps)
            c = max(1 + aa / c, eps)
            h *= d * c

            aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
            d = 1.0 / max(1 + aa * d, eps)
            c = max(1 + aa / c, eps)
            h *= d * c

            if abs(d * c - 1) < eps:
                break

        return h

    def _get_t_critical(self, df: int, alpha: float) -> float:
        """Get critical t-value (approximation)."""
        a = 1 - 2 * alpha
        z = self._ppf_normal(1 - alpha)

        g1 = (z ** 3 + z) / 4
        g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96
        g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384

        t = z + g1 / df + g2 / (df ** 2) + g3 / (df ** 3)
        return float(t)

    def _ppf_normal(self, p: float) -> float:
        """Percent point function for standard normal (approximation)."""
        if p <= 0:
            return -10.0
        if p >= 1:
            return 10.0

        a = [
            -3.969683028665376e+01,
            2.209460984245205e+02,
            -2.759285104469687e+02,
            1.383577518672690e+02,
            -3.066479806614716e+01,
            2.506628277459239e+00
        ]

        b = [
            -5.447609879822406e+01,
            1.615858368580409e+02,
            -1.556989798598866e+02,
            6.680131188771972e+01,
            -1.328068155288572e+01
        ]

        c = [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e+00,
            -2.549732539343734e+00,
            4.374664141464968e+00,
            2.938163982698783e+00
        ]

        d = [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e+00,
            3.754408661907416e+00
        ]

        p_low = 0.02425
        p_high = 1 - p_low

        if p < p_low:
            q = np.sqrt(-2 * np.log(p))
            return float(
                (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
            )
        elif p <= p_high:
            q = p - 0.5
            r = q * q
            return float(
                (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
                (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
            )
        else:
            q = np.sqrt(-2 * np.log(1 - p))
            return float(
                -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
            )

    def _normal_cdf(self, x: float) -> float:
        """Standard normal CDF."""
        return 0.5 * (1 + np.tanh(x * np.sqrt(2 / np.pi) * (1 + 0.044715 * x * x)))


class ModelEvaluator:
    """Comprehensive model evaluator."""

    def __init__(
        self,
        classification_calculator: Optional[ClassificationMetricCalculator] = None,
        regression_calculator: Optional[RegressionMetricCalculator] = None,
        trading_calculator: Optional[TradingMetricCalculator] = None,
        statistical_tester: Optional[StatisticalTester] = None
    ):
        """
        Initialize evaluator.

        Args:
            classification_calculator: Classification metric calculator
            regression_calculator: Regression metric calculator
            trading_calculator: Trading metric calculator
            statistical_tester: Statistical tester
        """
        self.classification_calculator = classification_calculator or ClassificationMetricCalculator()
        self.regression_calculator = regression_calculator or RegressionMetricCalculator()
        self.trading_calculator = trading_calculator or TradingMetricCalculator()
        self.statistical_tester = statistical_tester or StatisticalTester()

        self._evaluation_history: list[EvaluationReport] = []

        logger.info("Initialized ModelEvaluator")

    async def evaluate_classification(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        cv_scores: Optional[dict[str, list[float]]] = None,
        feature_importance: Optional[dict[str, float]] = None
    ) -> EvaluationReport:
        """
        Evaluate classification model.

        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            cv_scores: Cross-validation scores
            feature_importance: Feature importance scores

        Returns:
            EvaluationReport object
        """
        try:
            logger.info(f"Evaluating classification model: {model_name}")

            metrics = self.classification_calculator.calculate(
                y_true, y_pred, y_prob=y_prob
            )

            report = EvaluationReport(
                model_name=model_name,
                evaluation_timestamp=datetime.now(),
                classification_metrics=metrics,
                cross_validation_scores=cv_scores,
                feature_importance=feature_importance,
                predictions=y_pred,
                metadata={
                    "n_samples": len(y_true),
                    "n_classes": len(np.unique(y_true))
                }
            )

            self._evaluation_history.append(report)

            logger.info(
                f"Classification evaluation complete: "
                f"accuracy={metrics.accuracy:.4f}, f1={metrics.f1_score:.4f}"
            )

            return report

        except Exception as e:
            logger.error(f"Error evaluating classification model: {e}")
            raise

    async def evaluate_regression(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_features: int = 1,
        cv_scores: Optional[dict[str, list[float]]] = None,
        feature_importance: Optional[dict[str, float]] = None
    ) -> EvaluationReport:
        """
        Evaluate regression model.

        Args:
            model_name: Name of the model
            y_true: True values
            y_pred: Predicted values
            n_features: Number of features
            cv_scores: Cross-validation scores
            feature_importance: Feature importance scores

        Returns:
            EvaluationReport object
        """
        try:
            logger.info(f"Evaluating regression model: {model_name}")

            metrics = self.regression_calculator.calculate(
                y_true, y_pred, n_features=n_features
            )

            residuals = y_true - y_pred

            report = EvaluationReport(
                model_name=model_name,
                evaluation_timestamp=datetime.now(),
                regression_metrics=metrics,
                cross_validation_scores=cv_scores,
                feature_importance=feature_importance,
                predictions=y_pred,
                residuals=residuals,
                metadata={
                    "n_samples": len(y_true),
                    "n_features": n_features
                }
            )

            self._evaluation_history.append(report)

            logger.info(
                f"Regression evaluation complete: "
                f"R2={metrics.r2:.4f}, RMSE={metrics.rmse:.4f}"
            )

            return report

        except Exception as e:
            logger.error(f"Error evaluating regression model: {e}")
            raise

    async def evaluate_trading(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        returns: Optional[np.ndarray] = None,
        benchmark_returns: Optional[np.ndarray] = None,
        cv_scores: Optional[dict[str, list[float]]] = None
    ) -> EvaluationReport:
        """
        Evaluate trading model.

        Args:
            model_name: Name of the model
            y_true: True directions/returns
            y_pred: Predicted directions/returns
            returns: Actual achieved returns
            benchmark_returns: Benchmark returns
            cv_scores: Cross-validation scores

        Returns:
            EvaluationReport object
        """
        try:
            logger.info(f"Evaluating trading model: {model_name}")

            metrics = self.trading_calculator.calculate(
                y_true, y_pred,
                returns=returns,
                benchmark_returns=benchmark_returns
            )

            report = EvaluationReport(
                model_name=model_name,
                evaluation_timestamp=datetime.now(),
                trading_metrics=metrics,
                cross_validation_scores=cv_scores,
                predictions=y_pred,
                metadata={
                    "n_samples": len(y_true),
                    "has_benchmark": benchmark_returns is not None
                }
            )

            self._evaluation_history.append(report)

            logger.info(
                f"Trading evaluation complete: "
                f"hit_rate={metrics.hit_rate:.4f}, sharpe={metrics.sharpe_ratio:.4f}"
            )

            return report

        except Exception as e:
            logger.error(f"Error evaluating trading model: {e}")
            raise

    async def compare_models(
        self,
        model_scores: dict[str, list[float]],
        metric_name: str,
        comparison_method: ComparisonMethod = ComparisonMethod.TTEST
    ) -> ModelComparisonResult:
        """
        Compare multiple models statistically.

        Args:
            model_scores: Dictionary of model name to scores
            metric_name: Name of metric being compared
            comparison_method: Statistical test to use

        Returns:
            ModelComparisonResult object
        """
        try:
            logger.info(f"Comparing {len(model_scores)} models on {metric_name}")

            model_names = list(model_scores.keys())
            mean_scores = [np.mean(scores) for scores in model_scores.values()]

            ranking = sorted(zip(model_names, mean_scores), key=lambda x: x[1], reverse=True)
            best_model = ranking[0][0]

            if len(model_names) >= 2:
                best_scores = np.array(model_scores[model_names[0]])
                second_scores = np.array(model_scores[model_names[1]])

                if comparison_method == ComparisonMethod.TTEST:
                    test_result = self.statistical_tester.paired_ttest(
                        best_scores, second_scores
                    )
                elif comparison_method == ComparisonMethod.WILCOXON:
                    test_result = self.statistical_tester.wilcoxon_test(
                        best_scores, second_scores
                    )
                else:
                    test_result = self.statistical_tester.bootstrap_test(
                        best_scores, second_scores
                    )
            else:
                test_result = StatisticalTestResult(
                    test_name="none",
                    statistic=0.0,
                    p_value=1.0,
                    is_significant=False,
                    confidence_level=0.95
                )

            result = ModelComparisonResult(
                model_names=model_names,
                metric_name=metric_name,
                metric_values=mean_scores,
                best_model=best_model,
                statistical_test=test_result,
                ranking=ranking
            )

            logger.info(f"Best model: {best_model}, significant: {test_result.is_significant}")

            return result

        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            raise

    def get_evaluation_history(self) -> list[EvaluationReport]:
        """Get evaluation history."""
        return self._evaluation_history.copy()

    def clear_history(self) -> None:
        """Clear evaluation history."""
        self._evaluation_history.clear()
        logger.info("Cleared evaluation history")


def create_model_evaluator(
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
    confidence_level: float = 0.95
) -> ModelEvaluator:
    """
    Factory function to create model evaluator.

    Args:
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year
        confidence_level: Confidence level for tests

    Returns:
        ModelEvaluator instance
    """
    classification_calc = ClassificationMetricCalculator()
    regression_calc = RegressionMetricCalculator()
    trading_calc = TradingMetricCalculator(
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year
    )
    statistical_tester = StatisticalTester(confidence_level=confidence_level)

    return ModelEvaluator(
        classification_calculator=classification_calc,
        regression_calculator=regression_calc,
        trading_calculator=trading_calc,
        statistical_tester=statistical_tester
    )
