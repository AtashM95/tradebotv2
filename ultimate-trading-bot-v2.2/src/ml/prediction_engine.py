"""
Prediction Engine Module for Ultimate Trading Bot v2.2

Centralized prediction engine that coordinates multiple ML models,
manages prediction pipelines, and generates trading signals.

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


class PredictionType(Enum):
    """Types of predictions."""
    PRICE = "price"
    DIRECTION = "direction"
    VOLATILITY = "volatility"
    RETURNS = "returns"
    PROBABILITY = "probability"
    SIGNAL = "signal"


class AggregationMethod(Enum):
    """Methods for aggregating multiple predictions."""
    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"
    MEDIAN = "median"
    VOTING = "voting"
    STACKING = "stacking"
    BEST = "best"


class ConfidenceMethod(Enum):
    """Methods for calculating prediction confidence."""
    MODEL_CONFIDENCE = "model_confidence"
    PREDICTION_VARIANCE = "prediction_variance"
    HISTORICAL_ACCURACY = "historical_accuracy"
    ENSEMBLE_AGREEMENT = "ensemble_agreement"


@dataclass
class ModelPrediction:
    """Individual model prediction."""
    model_name: str
    prediction: np.ndarray
    confidence: float
    prediction_type: PredictionType
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "prediction": self.prediction.tolist(),
            "confidence": self.confidence,
            "prediction_type": self.prediction_type.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class AggregatedPrediction:
    """Aggregated prediction from multiple models."""
    prediction: np.ndarray
    confidence: float
    individual_predictions: list[ModelPrediction]
    aggregation_method: AggregationMethod
    weights: Optional[np.ndarray] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prediction": self.prediction.tolist(),
            "confidence": self.confidence,
            "individual_predictions": [p.to_dict() for p in self.individual_predictions],
            "aggregation_method": self.aggregation_method.value,
            "weights": self.weights.tolist() if self.weights is not None else None,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class PredictionRequest:
    """Request for predictions."""
    features: np.ndarray
    prediction_type: PredictionType
    horizon: int = 1
    return_confidence: bool = True
    return_individual: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "features_shape": self.features.shape,
            "prediction_type": self.prediction_type.value,
            "horizon": self.horizon,
            "return_confidence": self.return_confidence,
            "return_individual": self.return_individual,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class PredictionResponse:
    """Response containing predictions."""
    prediction: np.ndarray
    confidence: np.ndarray
    prediction_type: PredictionType
    horizon: int
    individual_predictions: Optional[list[ModelPrediction]] = None
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prediction": self.prediction.tolist(),
            "confidence": self.confidence.tolist(),
            "prediction_type": self.prediction_type.value,
            "horizon": self.horizon,
            "individual_predictions": (
                [p.to_dict() for p in self.individual_predictions]
                if self.individual_predictions else None
            ),
            "processing_time": self.processing_time,
            "timestamp": self.timestamp.isoformat()
        }


class ModelWrapper:
    """Wrapper for ML models with prediction tracking."""

    def __init__(
        self,
        model: Any,
        name: str,
        prediction_type: PredictionType,
        weight: float = 1.0
    ):
        """
        Initialize model wrapper.

        Args:
            model: The ML model
            name: Model name
            prediction_type: Type of predictions
            weight: Model weight for aggregation
        """
        self.model = model
        self.name = name
        self.prediction_type = prediction_type
        self.weight = weight

        self._prediction_history: list[ModelPrediction] = []
        self._accuracy_history: list[float] = []
        self._is_ready = True

        logger.info(f"Initialized ModelWrapper: {name}")

    async def predict(
        self,
        features: np.ndarray
    ) -> ModelPrediction:
        """
        Make prediction.

        Args:
            features: Input features

        Returns:
            ModelPrediction object
        """
        try:
            if hasattr(self.model, "predict"):
                result = self.model.predict(features)

                if hasattr(result, "predictions"):
                    prediction = result.predictions
                    confidence = float(np.mean(result.confidence)) if hasattr(result, "confidence") else 0.5
                elif isinstance(result, tuple):
                    prediction = result[0]
                    confidence = float(np.mean(result[1])) if len(result) > 1 else 0.5
                else:
                    prediction = result
                    confidence = 0.5
            else:
                prediction = np.zeros(len(features))
                confidence = 0.0

            if isinstance(prediction, (int, float)):
                prediction = np.array([prediction])
            elif not isinstance(prediction, np.ndarray):
                prediction = np.array(prediction)

            model_prediction = ModelPrediction(
                model_name=self.name,
                prediction=prediction,
                confidence=confidence,
                prediction_type=self.prediction_type
            )

            self._prediction_history.append(model_prediction)

            if len(self._prediction_history) > 1000:
                self._prediction_history = self._prediction_history[-500:]

            return model_prediction

        except Exception as e:
            logger.error(f"Error in model {self.name} prediction: {e}")

            return ModelPrediction(
                model_name=self.name,
                prediction=np.zeros(len(features)),
                confidence=0.0,
                prediction_type=self.prediction_type,
                metadata={"error": str(e)}
            )

    def update_accuracy(self, accuracy: float) -> None:
        """
        Update model accuracy history.

        Args:
            accuracy: Recent accuracy score
        """
        self._accuracy_history.append(accuracy)

        if len(self._accuracy_history) > 100:
            self._accuracy_history = self._accuracy_history[-50:]

    def get_historical_accuracy(self) -> float:
        """Get historical accuracy."""
        if not self._accuracy_history:
            return 0.5
        return float(np.mean(self._accuracy_history))

    @property
    def is_ready(self) -> bool:
        """Check if model is ready for predictions."""
        return self._is_ready


class PredictionAggregator:
    """Aggregates predictions from multiple models."""

    def __init__(
        self,
        method: AggregationMethod = AggregationMethod.WEIGHTED_MEAN,
        confidence_method: ConfidenceMethod = ConfidenceMethod.ENSEMBLE_AGREEMENT
    ):
        """
        Initialize aggregator.

        Args:
            method: Aggregation method
            confidence_method: Confidence calculation method
        """
        self.method = method
        self.confidence_method = confidence_method

        logger.info(f"Initialized PredictionAggregator: method={method.value}")

    def aggregate(
        self,
        predictions: list[ModelPrediction],
        weights: Optional[np.ndarray] = None
    ) -> AggregatedPrediction:
        """
        Aggregate multiple predictions.

        Args:
            predictions: List of model predictions
            weights: Optional weights for each model

        Returns:
            AggregatedPrediction object
        """
        if not predictions:
            return AggregatedPrediction(
                prediction=np.array([0.0]),
                confidence=0.0,
                individual_predictions=[],
                aggregation_method=self.method
            )

        pred_arrays = [p.prediction for p in predictions]
        max_len = max(len(p) for p in pred_arrays)
        pred_arrays = [
            np.pad(p, (0, max_len - len(p)), constant_values=p[-1])
            for p in pred_arrays
        ]
        pred_matrix = np.array(pred_arrays)

        if weights is None:
            if self.method == AggregationMethod.WEIGHTED_MEAN:
                weights = np.array([p.confidence for p in predictions])
                if np.sum(weights) > 0:
                    weights = weights / np.sum(weights)
                else:
                    weights = np.ones(len(predictions)) / len(predictions)
            else:
                weights = np.ones(len(predictions)) / len(predictions)

        if self.method == AggregationMethod.MEAN:
            aggregated = np.mean(pred_matrix, axis=0)

        elif self.method == AggregationMethod.WEIGHTED_MEAN:
            aggregated = np.average(pred_matrix, axis=0, weights=weights)

        elif self.method == AggregationMethod.MEDIAN:
            aggregated = np.median(pred_matrix, axis=0)

        elif self.method == AggregationMethod.VOTING:
            rounded = np.round(pred_matrix)
            aggregated = np.zeros(max_len)
            for i in range(max_len):
                votes = rounded[:, i]
                values, counts = np.unique(votes, return_counts=True)
                aggregated[i] = values[np.argmax(counts)]

        elif self.method == AggregationMethod.BEST:
            best_idx = np.argmax([p.confidence for p in predictions])
            aggregated = pred_matrix[best_idx]

        else:
            aggregated = np.mean(pred_matrix, axis=0)

        confidence = self._calculate_confidence(predictions, pred_matrix)

        return AggregatedPrediction(
            prediction=aggregated,
            confidence=confidence,
            individual_predictions=predictions,
            aggregation_method=self.method,
            weights=weights
        )

    def _calculate_confidence(
        self,
        predictions: list[ModelPrediction],
        pred_matrix: np.ndarray
    ) -> float:
        """Calculate aggregated confidence."""
        if self.confidence_method == ConfidenceMethod.MODEL_CONFIDENCE:
            return float(np.mean([p.confidence for p in predictions]))

        elif self.confidence_method == ConfidenceMethod.PREDICTION_VARIANCE:
            variance = np.mean(np.var(pred_matrix, axis=0))
            return float(max(0, 1 - variance))

        elif self.confidence_method == ConfidenceMethod.HISTORICAL_ACCURACY:
            return float(np.mean([p.confidence for p in predictions]))

        elif self.confidence_method == ConfidenceMethod.ENSEMBLE_AGREEMENT:
            if len(predictions) < 2:
                return predictions[0].confidence if predictions else 0.0

            agreements = []
            for i in range(len(predictions)):
                for j in range(i + 1, len(predictions)):
                    sign_agreement = np.mean(
                        np.sign(predictions[i].prediction) ==
                        np.sign(predictions[j].prediction)
                    )
                    agreements.append(sign_agreement)

            base_confidence = float(np.mean(agreements))
            model_confidence = float(np.mean([p.confidence for p in predictions]))

            return 0.7 * base_confidence + 0.3 * model_confidence

        return 0.5


class PredictionPipeline:
    """Pipeline for preprocessing and prediction."""

    def __init__(
        self,
        preprocessors: Optional[list[Callable]] = None,
        postprocessors: Optional[list[Callable]] = None
    ):
        """
        Initialize pipeline.

        Args:
            preprocessors: List of preprocessing functions
            postprocessors: List of postprocessing functions
        """
        self.preprocessors = preprocessors or []
        self.postprocessors = postprocessors or []

        logger.info(
            f"Initialized PredictionPipeline with "
            f"{len(self.preprocessors)} preprocessors, "
            f"{len(self.postprocessors)} postprocessors"
        )

    def preprocess(self, features: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing steps.

        Args:
            features: Raw features

        Returns:
            Preprocessed features
        """
        result = features.copy()

        for preprocessor in self.preprocessors:
            try:
                result = preprocessor(result)
            except Exception as e:
                logger.warning(f"Preprocessor error: {e}")

        return result

    def postprocess(self, predictions: np.ndarray) -> np.ndarray:
        """
        Apply postprocessing steps.

        Args:
            predictions: Raw predictions

        Returns:
            Postprocessed predictions
        """
        result = predictions.copy()

        for postprocessor in self.postprocessors:
            try:
                result = postprocessor(result)
            except Exception as e:
                logger.warning(f"Postprocessor error: {e}")

        return result

    def add_preprocessor(self, preprocessor: Callable) -> None:
        """Add a preprocessor."""
        self.preprocessors.append(preprocessor)

    def add_postprocessor(self, postprocessor: Callable) -> None:
        """Add a postprocessor."""
        self.postprocessors.append(postprocessor)


class PredictionEngine:
    """Central prediction engine coordinating multiple models."""

    def __init__(
        self,
        aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_MEAN,
        confidence_method: ConfidenceMethod = ConfidenceMethod.ENSEMBLE_AGREEMENT,
        cache_predictions: bool = True,
        cache_size: int = 1000
    ):
        """
        Initialize prediction engine.

        Args:
            aggregation_method: Method for aggregating predictions
            confidence_method: Method for calculating confidence
            cache_predictions: Whether to cache predictions
            cache_size: Maximum cache size
        """
        self.aggregation_method = aggregation_method
        self.confidence_method = confidence_method
        self.cache_predictions = cache_predictions
        self.cache_size = cache_size

        self._models: dict[str, ModelWrapper] = {}
        self._pipelines: dict[PredictionType, PredictionPipeline] = {}
        self._aggregator = PredictionAggregator(
            method=aggregation_method,
            confidence_method=confidence_method
        )
        self._prediction_cache: dict[str, PredictionResponse] = {}
        self._prediction_history: list[PredictionResponse] = []

        logger.info(
            f"Initialized PredictionEngine: "
            f"aggregation={aggregation_method.value}, "
            f"confidence={confidence_method.value}"
        )

    def register_model(
        self,
        model: Any,
        name: str,
        prediction_type: PredictionType,
        weight: float = 1.0
    ) -> None:
        """
        Register a model with the engine.

        Args:
            model: ML model
            name: Model name
            prediction_type: Type of predictions
            weight: Model weight
        """
        wrapper = ModelWrapper(
            model=model,
            name=name,
            prediction_type=prediction_type,
            weight=weight
        )

        self._models[name] = wrapper

        logger.info(f"Registered model: {name} for {prediction_type.value}")

    def unregister_model(self, name: str) -> None:
        """
        Unregister a model.

        Args:
            name: Model name
        """
        if name in self._models:
            del self._models[name]
            logger.info(f"Unregistered model: {name}")

    def register_pipeline(
        self,
        prediction_type: PredictionType,
        pipeline: PredictionPipeline
    ) -> None:
        """
        Register a prediction pipeline.

        Args:
            prediction_type: Type of predictions
            pipeline: Prediction pipeline
        """
        self._pipelines[prediction_type] = pipeline
        logger.info(f"Registered pipeline for {prediction_type.value}")

    async def predict(
        self,
        request: PredictionRequest
    ) -> PredictionResponse:
        """
        Generate predictions.

        Args:
            request: Prediction request

        Returns:
            PredictionResponse object
        """
        import time
        start_time = time.time()

        try:
            cache_key = self._generate_cache_key(request)
            if self.cache_predictions and cache_key in self._prediction_cache:
                logger.debug("Returning cached prediction")
                return self._prediction_cache[cache_key]

            features = request.features

            if request.prediction_type in self._pipelines:
                features = self._pipelines[request.prediction_type].preprocess(features)

            relevant_models = [
                model for model in self._models.values()
                if model.prediction_type == request.prediction_type and model.is_ready
            ]

            if not relevant_models:
                logger.warning(
                    f"No models available for {request.prediction_type.value}"
                )
                return PredictionResponse(
                    prediction=np.zeros(len(features)),
                    confidence=np.zeros(len(features)),
                    prediction_type=request.prediction_type,
                    horizon=request.horizon,
                    processing_time=time.time() - start_time
                )

            individual_predictions = []
            for model in relevant_models:
                pred = await model.predict(features)
                individual_predictions.append(pred)

            weights = np.array([m.weight for m in relevant_models])
            weights = weights / np.sum(weights)

            aggregated = self._aggregator.aggregate(
                individual_predictions,
                weights
            )

            prediction = aggregated.prediction

            if request.prediction_type in self._pipelines:
                prediction = self._pipelines[request.prediction_type].postprocess(
                    prediction
                )

            confidence = np.full(len(prediction), aggregated.confidence)

            processing_time = time.time() - start_time

            response = PredictionResponse(
                prediction=prediction,
                confidence=confidence,
                prediction_type=request.prediction_type,
                horizon=request.horizon,
                individual_predictions=(
                    individual_predictions if request.return_individual else None
                ),
                processing_time=processing_time
            )

            if self.cache_predictions:
                self._prediction_cache[cache_key] = response
                if len(self._prediction_cache) > self.cache_size:
                    oldest_key = next(iter(self._prediction_cache))
                    del self._prediction_cache[oldest_key]

            self._prediction_history.append(response)
            if len(self._prediction_history) > self.cache_size:
                self._prediction_history = self._prediction_history[-self.cache_size // 2:]

            logger.debug(
                f"Prediction complete: type={request.prediction_type.value}, "
                f"confidence={aggregated.confidence:.3f}, "
                f"time={processing_time:.3f}s"
            )

            return response

        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return PredictionResponse(
                prediction=np.zeros(1),
                confidence=np.zeros(1),
                prediction_type=request.prediction_type,
                horizon=request.horizon,
                processing_time=time.time() - start_time
            )

    async def predict_batch(
        self,
        requests: list[PredictionRequest]
    ) -> list[PredictionResponse]:
        """
        Generate batch predictions.

        Args:
            requests: List of prediction requests

        Returns:
            List of PredictionResponse objects
        """
        responses = []

        for request in requests:
            response = await self.predict(request)
            responses.append(response)

        return responses

    async def predict_multi_horizon(
        self,
        features: np.ndarray,
        prediction_type: PredictionType,
        horizons: list[int]
    ) -> dict[int, PredictionResponse]:
        """
        Generate predictions for multiple horizons.

        Args:
            features: Input features
            prediction_type: Type of predictions
            horizons: List of forecast horizons

        Returns:
            Dictionary mapping horizon to response
        """
        results = {}

        for horizon in horizons:
            request = PredictionRequest(
                features=features,
                prediction_type=prediction_type,
                horizon=horizon
            )
            response = await self.predict(request)
            results[horizon] = response

        return results

    def update_model_weights(
        self,
        model_performances: dict[str, float]
    ) -> None:
        """
        Update model weights based on performance.

        Args:
            model_performances: Dictionary of model name to performance score
        """
        for name, performance in model_performances.items():
            if name in self._models:
                self._models[name].weight = max(0.1, performance)
                self._models[name].update_accuracy(performance)

        logger.info("Updated model weights based on performance")

    def get_model_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all registered models."""
        stats = {}

        for name, model in self._models.items():
            stats[name] = {
                "prediction_type": model.prediction_type.value,
                "weight": model.weight,
                "is_ready": model.is_ready,
                "historical_accuracy": model.get_historical_accuracy(),
                "n_predictions": len(model._prediction_history)
            }

        return stats

    def clear_cache(self) -> None:
        """Clear prediction cache."""
        self._prediction_cache.clear()
        logger.info("Cleared prediction cache")

    def get_prediction_history(
        self,
        n_recent: int = 100
    ) -> list[PredictionResponse]:
        """
        Get recent prediction history.

        Args:
            n_recent: Number of recent predictions

        Returns:
            List of recent predictions
        """
        return self._prediction_history[-n_recent:]

    def _generate_cache_key(self, request: PredictionRequest) -> str:
        """Generate cache key for request."""
        features_hash = hash(request.features.tobytes())
        return f"{request.prediction_type.value}_{request.horizon}_{features_hash}"


class RealTimePredictionEngine(PredictionEngine):
    """Prediction engine optimized for real-time trading."""

    def __init__(
        self,
        max_latency_ms: float = 100.0,
        fallback_on_timeout: bool = True,
        **kwargs: Any
    ):
        """
        Initialize real-time prediction engine.

        Args:
            max_latency_ms: Maximum prediction latency
            fallback_on_timeout: Whether to use fallback on timeout
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)

        self.max_latency_ms = max_latency_ms
        self.fallback_on_timeout = fallback_on_timeout

        self._last_prediction: Optional[PredictionResponse] = None
        self._latency_history: list[float] = []

        logger.info(
            f"Initialized RealTimePredictionEngine: "
            f"max_latency={max_latency_ms}ms"
        )

    async def predict(
        self,
        request: PredictionRequest
    ) -> PredictionResponse:
        """
        Generate real-time prediction with latency constraints.

        Args:
            request: Prediction request

        Returns:
            PredictionResponse object
        """
        import time
        import asyncio

        start_time = time.time()

        try:
            timeout = self.max_latency_ms / 1000.0

            prediction_task = super().predict(request)

            try:
                response = await asyncio.wait_for(prediction_task, timeout=timeout)

            except asyncio.TimeoutError:
                logger.warning(
                    f"Prediction timeout ({self.max_latency_ms}ms), "
                    f"using fallback"
                )

                if self.fallback_on_timeout and self._last_prediction is not None:
                    response = self._last_prediction
                else:
                    response = PredictionResponse(
                        prediction=np.zeros(len(request.features)),
                        confidence=np.zeros(len(request.features)),
                        prediction_type=request.prediction_type,
                        horizon=request.horizon,
                        processing_time=time.time() - start_time
                    )

            latency = (time.time() - start_time) * 1000
            self._latency_history.append(latency)

            if len(self._latency_history) > 1000:
                self._latency_history = self._latency_history[-500:]

            self._last_prediction = response

            return response

        except Exception as e:
            logger.error(f"Error in real-time prediction: {e}")

            if self._last_prediction is not None:
                return self._last_prediction

            return PredictionResponse(
                prediction=np.zeros(len(request.features)),
                confidence=np.zeros(len(request.features)),
                prediction_type=request.prediction_type,
                horizon=request.horizon,
                processing_time=time.time() - start_time
            )

    def get_latency_stats(self) -> dict[str, float]:
        """Get latency statistics."""
        if not self._latency_history:
            return {
                "mean_ms": 0.0,
                "std_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0
            }

        latencies = np.array(self._latency_history)

        return {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99))
        }


def create_prediction_engine(
    aggregation_method: str = "weighted_mean",
    confidence_method: str = "ensemble_agreement",
    cache_predictions: bool = True
) -> PredictionEngine:
    """
    Factory function to create prediction engine.

    Args:
        aggregation_method: Aggregation method name
        confidence_method: Confidence method name
        cache_predictions: Whether to cache predictions

    Returns:
        PredictionEngine instance
    """
    return PredictionEngine(
        aggregation_method=AggregationMethod(aggregation_method),
        confidence_method=ConfidenceMethod(confidence_method),
        cache_predictions=cache_predictions
    )


def create_realtime_prediction_engine(
    max_latency_ms: float = 100.0,
    fallback_on_timeout: bool = True,
    aggregation_method: str = "weighted_mean"
) -> RealTimePredictionEngine:
    """
    Factory function to create real-time prediction engine.

    Args:
        max_latency_ms: Maximum latency
        fallback_on_timeout: Use fallback on timeout
        aggregation_method: Aggregation method name

    Returns:
        RealTimePredictionEngine instance
    """
    return RealTimePredictionEngine(
        max_latency_ms=max_latency_ms,
        fallback_on_timeout=fallback_on_timeout,
        aggregation_method=AggregationMethod(aggregation_method)
    )
