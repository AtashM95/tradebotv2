"""
Machine Learning Strategy Module for Ultimate Trading Bot v2.2.

This module implements ML-based trading strategies using
trained models for price prediction and signal generation.
"""

import logging
from datetime import datetime
from typing import Any, Optional, Protocol

import numpy as np
from pydantic import BaseModel, Field

from src.strategies.base_strategy import (
    BaseStrategy,
    StrategyConfig,
    StrategySignal,
    SignalAction,
    SignalSide,
    MarketData,
    StrategyContext,
)
from src.analysis.technical_indicators import TechnicalIndicators
from src.utils.helpers import generate_uuid


logger = logging.getLogger(__name__)


class MLModel(Protocol):
    """Protocol for ML model interface."""

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions."""
        ...

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        ...


class ModelPrediction(BaseModel):
    """Model for ML prediction result."""

    symbol: str
    timestamp: datetime
    prediction: float
    confidence: float = Field(ge=0.0, le=1.0)
    direction: str
    features_used: int = Field(default=0)
    model_name: str = Field(default="default")


class FeatureSet(BaseModel):
    """Model for feature set."""

    symbol: str
    timestamp: datetime
    features: dict[str, float] = Field(default_factory=dict)
    labels: Optional[dict[str, float]] = None


class MLStrategyConfig(StrategyConfig):
    """Configuration for ML-based strategy."""

    name: str = Field(default="ML Strategy")
    description: str = Field(
        default="Machine learning-based trading signals"
    )

    model_type: str = Field(default="ensemble")
    feature_lookback: int = Field(default=30, ge=10, le=252)
    prediction_horizon: int = Field(default=5, ge=1, le=30)

    prediction_threshold_buy: float = Field(default=0.6, ge=0.5, le=0.95)
    prediction_threshold_sell: float = Field(default=0.4, ge=0.05, le=0.5)

    min_confidence: float = Field(default=0.55, ge=0.5, le=0.9)

    use_technical_features: bool = Field(default=True)
    use_price_features: bool = Field(default=True)
    use_volume_features: bool = Field(default=True)
    use_momentum_features: bool = Field(default=True)

    feature_normalization: str = Field(default="zscore")

    ensemble_models: list[str] = Field(
        default_factory=lambda: ["random_forest", "gradient_boost", "neural_net"]
    )
    ensemble_weights: list[float] = Field(
        default_factory=lambda: [0.35, 0.35, 0.30]
    )

    retrain_frequency_days: int = Field(default=30, ge=7, le=90)
    min_training_samples: int = Field(default=500, ge=100, le=5000)


class MLStrategy(BaseStrategy):
    """
    Machine learning-based trading strategy.

    Features:
    - Multi-model ensemble predictions
    - Automatic feature engineering
    - Online learning capability
    - Confidence-weighted signals
    - Model performance tracking
    """

    def __init__(
        self,
        config: Optional[MLStrategyConfig] = None,
        models: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialize MLStrategy.

        Args:
            config: ML strategy configuration
            models: Pre-trained models dictionary
        """
        config = config or MLStrategyConfig()
        super().__init__(config)

        self._ml_config = config
        self._indicators = TechnicalIndicators()

        self._models: dict[str, Any] = models or {}
        self._predictions: dict[str, list[ModelPrediction]] = {}
        self._feature_history: dict[str, list[FeatureSet]] = {}
        self._model_performance: dict[str, dict] = {}
        self._last_retrain: Optional[datetime] = None

        self._default_weights = self._create_default_weights()

        logger.info(f"MLStrategy initialized: {self.name}")

    def _create_default_weights(self) -> dict[str, float]:
        """Create default feature weights for simple prediction."""
        return {
            "momentum": 0.25,
            "trend": 0.25,
            "mean_reversion": 0.20,
            "volume": 0.15,
            "volatility": 0.15,
        }

    def calculate_indicators(
        self,
        symbol: str,
        data: MarketData,
    ) -> dict[str, Any]:
        """
        Calculate indicators and features for ML model.

        Args:
            symbol: Trading symbol
            data: Market data

        Returns:
            Dictionary of indicator values
        """
        closes = data.closes
        highs = data.highs
        lows = data.lows
        volumes = data.volumes

        lookback = self._ml_config.feature_lookback
        if len(closes) < lookback + 10:
            return {}

        current_price = closes[-1]

        features = self._extract_features(
            opens=data.opens,
            highs=highs,
            lows=lows,
            closes=closes,
            volumes=volumes,
        )

        return {
            "current_price": current_price,
            "features": features,
            "feature_count": len(features),
        }

    def _extract_features(
        self,
        opens: list[float],
        highs: list[float],
        lows: list[float],
        closes: list[float],
        volumes: list[float],
    ) -> dict[str, float]:
        """Extract features from OHLCV data."""
        features: dict[str, float] = {}

        if self._ml_config.use_price_features:
            features.update(self._extract_price_features(closes, highs, lows))

        if self._ml_config.use_technical_features:
            features.update(self._extract_technical_features(closes, highs, lows, volumes))

        if self._ml_config.use_momentum_features:
            features.update(self._extract_momentum_features(closes))

        if self._ml_config.use_volume_features:
            features.update(self._extract_volume_features(volumes, closes))

        if self._ml_config.feature_normalization == "zscore":
            features = self._normalize_features_zscore(features)

        return features

    def _extract_price_features(
        self,
        closes: list[float],
        highs: list[float],
        lows: list[float],
    ) -> dict[str, float]:
        """Extract price-based features."""
        current = closes[-1]

        features = {
            "return_1d": (closes[-1] - closes[-2]) / closes[-2] if len(closes) >= 2 else 0,
            "return_5d": (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0,
            "return_10d": (closes[-1] - closes[-10]) / closes[-10] if len(closes) >= 10 else 0,
            "return_20d": (closes[-1] - closes[-20]) / closes[-20] if len(closes) >= 20 else 0,
        }

        if len(highs) >= 20:
            high_20 = max(highs[-20:])
            low_20 = min(lows[-20:])
            features["price_position_20d"] = (current - low_20) / (high_20 - low_20) if high_20 != low_20 else 0.5

        if len(highs) >= 50:
            high_50 = max(highs[-50:])
            low_50 = min(lows[-50:])
            features["price_position_50d"] = (current - low_50) / (high_50 - low_50) if high_50 != low_50 else 0.5

        if len(closes) >= 20:
            features["volatility_20d"] = np.std(
                [(closes[i] - closes[i-1]) / closes[i-1] for i in range(-19, 0)]
            )

        if len(closes) >= 10:
            gap_returns = [(opens[i] - closes[i-1]) / closes[i-1] for i in range(-9, 0) if i < len(opens)]
            features["avg_gap"] = sum(gap_returns) / len(gap_returns) if gap_returns else 0

        return features

    def _extract_technical_features(
        self,
        closes: list[float],
        highs: list[float],
        lows: list[float],
        volumes: list[float],
    ) -> dict[str, float]:
        """Extract technical indicator features."""
        features = {}

        rsi = self._indicators.rsi(closes, 14)
        if rsi:
            features["rsi_14"] = (rsi[-1] - 50) / 50

        sma_10 = sum(closes[-10:]) / 10 if len(closes) >= 10 else closes[-1]
        sma_20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else closes[-1]
        sma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else closes[-1]

        features["price_vs_sma10"] = (closes[-1] - sma_10) / sma_10 if sma_10 != 0 else 0
        features["price_vs_sma20"] = (closes[-1] - sma_20) / sma_20 if sma_20 != 0 else 0
        features["price_vs_sma50"] = (closes[-1] - sma_50) / sma_50 if sma_50 != 0 else 0
        features["sma10_vs_sma20"] = (sma_10 - sma_20) / sma_20 if sma_20 != 0 else 0
        features["sma20_vs_sma50"] = (sma_20 - sma_50) / sma_50 if sma_50 != 0 else 0

        macd = self._indicators.macd(closes, 12, 26, 9)
        if macd:
            features["macd_histogram"] = macd[-1].histogram / closes[-1] if closes[-1] != 0 else 0

        bb = self._indicators.bollinger_bands(closes, 20, 2.0)
        if bb:
            bb_range = bb[-1].upper - bb[-1].lower
            features["bb_position"] = (closes[-1] - bb[-1].lower) / bb_range if bb_range != 0 else 0.5
            features["bb_width"] = bb_range / bb[-1].middle if bb[-1].middle != 0 else 0

        atr = self._indicators.atr(highs, lows, closes, 14)
        if atr:
            features["atr_pct"] = atr[-1] / closes[-1] if closes[-1] != 0 else 0

        adx = self._indicators.adx(highs, lows, closes, 14)
        if adx:
            features["adx"] = (adx[-1] - 25) / 25

        return features

    def _extract_momentum_features(self, closes: list[float]) -> dict[str, float]:
        """Extract momentum features."""
        features = {}

        for period in [5, 10, 20]:
            if len(closes) >= period:
                returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(-period+1, 0)]
                up_days = sum(1 for r in returns if r > 0)
                features[f"up_ratio_{period}d"] = up_days / len(returns) if returns else 0.5

        if len(closes) >= 20:
            changes = [closes[i] - closes[i-1] for i in range(-19, 0)]
            gains = [c for c in changes if c > 0]
            losses = [-c for c in changes if c < 0]

            avg_gain = sum(gains) / len(gains) if gains else 0.001
            avg_loss = sum(losses) / len(losses) if losses else 0.001

            features["gain_loss_ratio"] = avg_gain / avg_loss if avg_loss > 0 else 1.0

        if len(closes) >= 30:
            recent_vol = np.std([(closes[i] - closes[i-1]) / closes[i-1] for i in range(-9, 0)])
            older_vol = np.std([(closes[i] - closes[i-1]) / closes[i-1] for i in range(-29, -20)])
            features["vol_change"] = (recent_vol - older_vol) / older_vol if older_vol != 0 else 0

        return features

    def _extract_volume_features(
        self,
        volumes: list[float],
        closes: list[float],
    ) -> dict[str, float]:
        """Extract volume features."""
        features = {}

        if len(volumes) >= 20:
            avg_vol_20 = sum(volumes[-20:]) / 20
            features["volume_ratio"] = volumes[-1] / avg_vol_20 if avg_vol_20 > 0 else 1.0

            recent_avg = sum(volumes[-5:]) / 5
            features["volume_trend"] = (recent_avg - avg_vol_20) / avg_vol_20 if avg_vol_20 > 0 else 0

        if len(volumes) >= 10 and len(closes) >= 10:
            price_up_volume = sum(
                volumes[i] for i in range(-9, 0)
                if closes[i] > closes[i-1]
            )
            price_down_volume = sum(
                volumes[i] for i in range(-9, 0)
                if closes[i] < closes[i-1]
            )
            total_vol = price_up_volume + price_down_volume
            features["volume_price_trend"] = (
                (price_up_volume - price_down_volume) / total_vol
                if total_vol > 0 else 0
            )

        return features

    def _normalize_features_zscore(
        self,
        features: dict[str, float],
    ) -> dict[str, float]:
        """Normalize features using z-score approximation."""
        normalized = {}

        for name, value in features.items():
            clipped = max(-3, min(3, value))
            normalized[name] = clipped

        return normalized

    async def evaluate(
        self,
        market_data: dict[str, MarketData],
        context: StrategyContext,
    ) -> list[StrategySignal]:
        """
        Evaluate ML-based opportunities.

        Args:
            market_data: Market data for all symbols
            context: Strategy execution context

        Returns:
            List of ML-based signals
        """
        signals: list[StrategySignal] = []

        for symbol in self.config.symbols:
            if symbol not in market_data:
                continue

            data = market_data[symbol]

            indicators = self.calculate_indicators(symbol, data)
            if not indicators or "features" not in indicators:
                continue

            prediction = self._make_prediction(symbol, indicators, context)
            if not prediction:
                continue

            self._store_prediction(symbol, prediction)

            if prediction.confidence >= self._ml_config.min_confidence:
                signal = self._generate_ml_signal(
                    symbol, prediction, indicators, data, context
                )
                if signal:
                    signals.append(signal)

        return signals

    def _make_prediction(
        self,
        symbol: str,
        indicators: dict[str, Any],
        context: StrategyContext,
    ) -> Optional[ModelPrediction]:
        """Make prediction using ML models or heuristics."""
        features = indicators.get("features", {})
        if not features:
            return None

        if self._models:
            return self._ensemble_predict(symbol, features, context)
        else:
            return self._heuristic_predict(symbol, features, context)

    def _ensemble_predict(
        self,
        symbol: str,
        features: dict[str, float],
        context: StrategyContext,
    ) -> Optional[ModelPrediction]:
        """Make ensemble prediction using trained models."""
        feature_array = np.array([list(features.values())])

        predictions = []
        weights = []

        for model_name, weight in zip(
            self._ml_config.ensemble_models,
            self._ml_config.ensemble_weights,
        ):
            if model_name not in self._models:
                continue

            model = self._models[model_name]
            try:
                pred = model.predict(feature_array)[0]
                predictions.append(pred)
                weights.append(weight)
            except Exception as e:
                logger.warning(f"Model {model_name} prediction failed: {e}")
                continue

        if not predictions:
            return self._heuristic_predict(symbol, features, context)

        total_weight = sum(weights)
        weighted_pred = sum(p * w for p, w in zip(predictions, weights)) / total_weight

        spread = max(predictions) - min(predictions)
        confidence = max(0.5, 1 - spread)

        direction = "bullish" if weighted_pred > 0.5 else "bearish"

        return ModelPrediction(
            symbol=symbol,
            timestamp=context.timestamp,
            prediction=weighted_pred,
            confidence=confidence,
            direction=direction,
            features_used=len(features),
            model_name="ensemble",
        )

    def _heuristic_predict(
        self,
        symbol: str,
        features: dict[str, float],
        context: StrategyContext,
    ) -> Optional[ModelPrediction]:
        """Make heuristic prediction without trained models."""
        momentum_score = 0.0
        trend_score = 0.0
        mean_rev_score = 0.0
        volume_score = 0.0

        if "return_5d" in features:
            momentum_score += features["return_5d"] * 2
        if "return_10d" in features:
            momentum_score += features["return_10d"]
        if "up_ratio_10d" in features:
            momentum_score += (features["up_ratio_10d"] - 0.5) * 2

        if "price_vs_sma20" in features:
            trend_score += features["price_vs_sma20"]
        if "sma10_vs_sma20" in features:
            trend_score += features["sma10_vs_sma20"]
        if "adx" in features and features["adx"] > 0:
            trend_score *= (1 + features["adx"] * 0.5)

        if "rsi_14" in features:
            rsi_signal = -features["rsi_14"]
            mean_rev_score += rsi_signal * 0.5
        if "bb_position" in features:
            bb_signal = 0.5 - features["bb_position"]
            mean_rev_score += bb_signal

        if "volume_ratio" in features:
            volume_score = (features["volume_ratio"] - 1) * 0.2
        if "volume_price_trend" in features:
            volume_score += features["volume_price_trend"] * 0.3

        weights = self._default_weights
        combined_score = (
            momentum_score * weights["momentum"] +
            trend_score * weights["trend"] +
            mean_rev_score * weights["mean_reversion"] +
            volume_score * weights["volume"]
        )

        prediction = 1 / (1 + np.exp(-combined_score * 2))

        scores = [momentum_score, trend_score, mean_rev_score]
        agreement = 1 - np.std([1 if s > 0 else 0 for s in scores])
        confidence = 0.5 + agreement * 0.3

        direction = "bullish" if prediction > 0.5 else "bearish"

        return ModelPrediction(
            symbol=symbol,
            timestamp=context.timestamp,
            prediction=prediction,
            confidence=confidence,
            direction=direction,
            features_used=len(features),
            model_name="heuristic",
        )

    def _store_prediction(self, symbol: str, prediction: ModelPrediction) -> None:
        """Store prediction for tracking."""
        if symbol not in self._predictions:
            self._predictions[symbol] = []

        self._predictions[symbol].append(prediction)

        if len(self._predictions[symbol]) > 500:
            self._predictions[symbol] = self._predictions[symbol][-500:]

    def _generate_ml_signal(
        self,
        symbol: str,
        prediction: ModelPrediction,
        indicators: dict[str, Any],
        data: MarketData,
        context: StrategyContext,
    ) -> Optional[StrategySignal]:
        """Generate trading signal from ML prediction."""
        current_price = indicators["current_price"]

        if prediction.prediction >= self._ml_config.prediction_threshold_buy:
            return self.create_signal(
                symbol=symbol,
                action=SignalAction.BUY,
                side=SignalSide.LONG,
                entry_price=current_price,
                strength=prediction.prediction,
                confidence=prediction.confidence,
                reason=f"ML buy: {prediction.prediction:.2f} ({prediction.model_name})",
                metadata={
                    "strategy_type": "ml",
                    "prediction": prediction.prediction,
                    "model_name": prediction.model_name,
                    "features_used": prediction.features_used,
                    "prediction_horizon": self._ml_config.prediction_horizon,
                },
            )

        elif prediction.prediction <= self._ml_config.prediction_threshold_sell:
            return self.create_signal(
                symbol=symbol,
                action=SignalAction.SELL,
                side=SignalSide.SHORT,
                entry_price=current_price,
                strength=1 - prediction.prediction,
                confidence=prediction.confidence,
                reason=f"ML sell: {prediction.prediction:.2f} ({prediction.model_name})",
                metadata={
                    "strategy_type": "ml",
                    "prediction": prediction.prediction,
                    "model_name": prediction.model_name,
                    "features_used": prediction.features_used,
                    "prediction_horizon": self._ml_config.prediction_horizon,
                },
            )

        return None

    def register_model(self, name: str, model: Any) -> None:
        """Register a trained model."""
        self._models[name] = model
        logger.info(f"Model registered: {name}")

    def get_predictions(self, symbol: str, limit: int = 50) -> list[ModelPrediction]:
        """Get recent predictions for symbol."""
        if symbol not in self._predictions:
            return []
        return self._predictions[symbol][-limit:]

    def get_model_performance(self) -> dict[str, dict]:
        """Get model performance metrics."""
        return self._model_performance.copy()

    def get_ml_statistics(self) -> dict:
        """Get ML strategy statistics."""
        total_predictions = sum(len(p) for p in self._predictions.values())

        return {
            "models_registered": len(self._models),
            "model_names": list(self._models.keys()),
            "total_predictions": total_predictions,
            "symbols_tracked": len(self._predictions),
            "last_retrain": self._last_retrain.isoformat() if self._last_retrain else None,
            "config": {
                "model_type": self._ml_config.model_type,
                "feature_lookback": self._ml_config.feature_lookback,
                "prediction_horizon": self._ml_config.prediction_horizon,
            },
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"MLStrategy(models={len(self._models)}, predictions={sum(len(p) for p in self._predictions.values())})"
