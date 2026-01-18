"""
Machine Learning Package for Ultimate Trading Bot v2.2.

This package provides comprehensive machine learning capabilities including:
- Feature engineering and preprocessing
- Model training and evaluation
- Time series forecasting (ARIMA, GARCH, VAR)
- Deep learning (LSTM, GRU, Transformer, MLP)
- Ensemble methods (Bagging, Boosting, Stacking)
- Model selection and hyperparameter tuning
- Prediction engine and signal generation
- Reinforcement learning for trading
- Online learning with concept drift detection
- Anomaly detection
- Market regime detection
- Model persistence and versioning
"""

import logging
from typing import Any

from .base_model import (
    BaseModel,
    ModelConfig,
    ModelState,
    PredictionResult,
    TrainingResult,
    LinearRegressionModel,
    LogisticRegressionModel,
    DecisionTreeModel,
    SVMModel,
    KNNModel,
    NaiveBayesModel,
    create_model,
)
from .feature_engineering import (
    FeatureConfig,
    FeatureImportance,
    FeatureStatistics,
    FeatureSet,
    BaseFeatureTransformer,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LogTransformer,
    PolynomialFeatures,
    InteractionFeatures,
    LagFeatures,
    RollingFeatures,
    DifferenceFeatures,
    PCATransformer,
    FeaturePipeline,
    TechnicalFeatureGenerator,
    StatisticalFeatureGenerator,
    FeatureSelector,
    FeatureEngineer,
    create_feature_engineer,
)
from .model_trainer import (
    TrainingConfig,
    TrainingHistory,
    BatchResult,
    EpochResult,
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpointer,
    TrainingLogger,
    BaseTrainer,
    BatchTrainer,
    MiniBatchTrainer,
    OnlineTrainer,
    CrossValidationTrainer,
    EnsembleTrainer,
    TrainerFactory,
    create_trainer,
)
from .model_evaluator import (
    MetricResult,
    EvaluationReport,
    ComparisonResult,
    ClassificationMetricCalculator,
    RegressionMetricCalculator,
    TradingMetricCalculator,
    StatisticalTester,
    ModelEvaluator,
    ModelComparator,
    create_evaluator,
)
from .time_series_models import (
    TimeSeriesConfig,
    ForecastResult,
    SeasonalityResult,
    BaseTimeSeriesModel,
    ARIMAModel,
    GARCHModel,
    ExponentialSmoothingModel,
    VARModel,
    SeasonalDecomposer,
    TimeSeriesEnsemble,
    create_time_series_model,
)
from .deep_learning import (
    LayerConfig,
    NetworkConfig,
    LayerOutput,
    BaseLayer,
    DenseLayer,
    DropoutLayer,
    BatchNormLayer,
    LSTMLayer,
    GRULayer,
    MultiHeadAttention,
    BaseOptimizer,
    SGDOptimizer,
    AdamOptimizer,
    BaseDeepLearningModel,
    MLPModel,
    LSTMModel,
    GRUModel,
    TransformerModel,
    create_deep_learning_model,
)
from .ensemble_models import (
    EnsembleConfig,
    EnsemblePrediction,
    MemberResult,
    BaseEnsemble,
    BaggingEnsemble,
    AdaBoostEnsemble,
    GradientBoostingEnsemble,
    VotingEnsemble,
    StackingEnsemble,
    RandomForestEnsemble,
    create_ensemble,
)
from .model_selection import (
    CVConfig,
    SearchConfig,
    CVResult,
    SearchResult,
    FeatureRanking,
    CrossValidator,
    HyperparameterSearcher,
    FeatureSelector as ModelFeatureSelector,
    ModelSelector,
    AutoML,
    create_model_selector,
)
from .prediction_engine import (
    PredictionConfig,
    PredictionRequest,
    PredictionResponse,
    AggregatedPrediction,
    ModelWrapper,
    PredictionAggregator,
    PredictionCache,
    PredictionPipeline,
    PredictionEngine,
    RealTimePredictionEngine,
    create_prediction_engine,
)
from .signal_generator import (
    SignalConfig,
    TradingSignal,
    SignalStrength,
    SignalFilter as TradingSignalFilter,
    BaseSignalGenerator,
    DirectionSignalGenerator,
    ProbabilitySignalGenerator,
    RegressionSignalGenerator,
    MultiModelSignalGenerator,
    RiskAdjustedSignalGenerator,
    SignalFilter,
    SignalAggregator,
    create_signal_generator,
)
from .reinforcement_learning import (
    RLConfig,
    StateRepresentation,
    Action,
    Experience,
    EpisodeResult,
    ReplayBuffer,
    PrioritizedReplayBuffer,
    TradingEnvironment,
    BaseRLAgent,
    QLearningAgent,
    DQNAgent,
    PolicyGradientAgent,
    ActorCriticAgent,
    RLTrainer,
    create_rl_agent,
)
from .online_learning import (
    OnlineConfig,
    OnlineUpdate,
    DriftDetectionResult,
    DriftDetector,
    DDMDetector,
    ADWINDetector,
    PageHinkleyDetector,
    BaseOnlineLearner,
    SGDOnlineLearner,
    PassiveAggressiveLearner,
    PerceptronLearner,
    OnlineLearnerWithDriftDetection,
    AdaptiveOnlineLearner,
    create_online_learner,
    create_drift_detector,
)
from .anomaly_detection import (
    AnomalyConfig,
    AnomalyResult,
    AnomalyReport,
    BaseAnomalyDetector,
    ZScoreDetector,
    IQRDetector,
    IsolationForestDetector,
    LocalOutlierFactorDetector,
    MahalanobisDetector,
    AutoencoderDetector,
    EnsembleAnomalyDetector,
    StreamingAnomalyDetector,
    create_anomaly_detector,
)
from .regime_detection import (
    RegimeConfig,
    RegimeState,
    RegimeTransition,
    RegimeHistory,
    BaseRegimeDetector,
    HiddenMarkovModelDetector,
    KMeansRegimeDetector,
    ThresholdRegimeDetector,
    ChangePointDetector,
    EnsembleRegimeDetector,
    RegimeAnalyzer,
    create_regime_detector,
)
from .model_persistence import (
    PersistenceConfig,
    ModelMetadata,
    ModelVersion,
    CheckpointInfo,
    BaseSerializer,
    PickleSerializer,
    JSONSerializer,
    NumpySerializer,
    ModelStorage,
    ModelVersionManager,
    ModelRegistry,
    create_model_storage,
)


logger = logging.getLogger(__name__)


__all__ = [
    # Base Model
    "BaseModel",
    "ModelConfig",
    "ModelState",
    "PredictionResult",
    "TrainingResult",
    "LinearRegressionModel",
    "LogisticRegressionModel",
    "DecisionTreeModel",
    "SVMModel",
    "KNNModel",
    "NaiveBayesModel",
    "create_model",
    # Feature Engineering
    "FeatureConfig",
    "FeatureImportance",
    "FeatureStatistics",
    "FeatureSet",
    "BaseFeatureTransformer",
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "LogTransformer",
    "PolynomialFeatures",
    "InteractionFeatures",
    "LagFeatures",
    "RollingFeatures",
    "DifferenceFeatures",
    "PCATransformer",
    "FeaturePipeline",
    "TechnicalFeatureGenerator",
    "StatisticalFeatureGenerator",
    "FeatureSelector",
    "FeatureEngineer",
    "create_feature_engineer",
    # Model Trainer
    "TrainingConfig",
    "TrainingHistory",
    "BatchResult",
    "EpochResult",
    "EarlyStopping",
    "LearningRateScheduler",
    "ModelCheckpointer",
    "TrainingLogger",
    "BaseTrainer",
    "BatchTrainer",
    "MiniBatchTrainer",
    "OnlineTrainer",
    "CrossValidationTrainer",
    "EnsembleTrainer",
    "TrainerFactory",
    "create_trainer",
    # Model Evaluator
    "MetricResult",
    "EvaluationReport",
    "ComparisonResult",
    "ClassificationMetricCalculator",
    "RegressionMetricCalculator",
    "TradingMetricCalculator",
    "StatisticalTester",
    "ModelEvaluator",
    "ModelComparator",
    "create_evaluator",
    # Time Series Models
    "TimeSeriesConfig",
    "ForecastResult",
    "SeasonalityResult",
    "BaseTimeSeriesModel",
    "ARIMAModel",
    "GARCHModel",
    "ExponentialSmoothingModel",
    "VARModel",
    "SeasonalDecomposer",
    "TimeSeriesEnsemble",
    "create_time_series_model",
    # Deep Learning
    "LayerConfig",
    "NetworkConfig",
    "LayerOutput",
    "BaseLayer",
    "DenseLayer",
    "DropoutLayer",
    "BatchNormLayer",
    "LSTMLayer",
    "GRULayer",
    "MultiHeadAttention",
    "BaseOptimizer",
    "SGDOptimizer",
    "AdamOptimizer",
    "BaseDeepLearningModel",
    "MLPModel",
    "LSTMModel",
    "GRUModel",
    "TransformerModel",
    "create_deep_learning_model",
    # Ensemble Models
    "EnsembleConfig",
    "EnsemblePrediction",
    "MemberResult",
    "BaseEnsemble",
    "BaggingEnsemble",
    "AdaBoostEnsemble",
    "GradientBoostingEnsemble",
    "VotingEnsemble",
    "StackingEnsemble",
    "RandomForestEnsemble",
    "create_ensemble",
    # Model Selection
    "CVConfig",
    "SearchConfig",
    "CVResult",
    "SearchResult",
    "FeatureRanking",
    "CrossValidator",
    "HyperparameterSearcher",
    "ModelFeatureSelector",
    "ModelSelector",
    "AutoML",
    "create_model_selector",
    # Prediction Engine
    "PredictionConfig",
    "PredictionRequest",
    "PredictionResponse",
    "AggregatedPrediction",
    "ModelWrapper",
    "PredictionAggregator",
    "PredictionCache",
    "PredictionPipeline",
    "PredictionEngine",
    "RealTimePredictionEngine",
    "create_prediction_engine",
    # Signal Generator
    "SignalConfig",
    "TradingSignal",
    "SignalStrength",
    "TradingSignalFilter",
    "BaseSignalGenerator",
    "DirectionSignalGenerator",
    "ProbabilitySignalGenerator",
    "RegressionSignalGenerator",
    "MultiModelSignalGenerator",
    "RiskAdjustedSignalGenerator",
    "SignalFilter",
    "SignalAggregator",
    "create_signal_generator",
    # Reinforcement Learning
    "RLConfig",
    "StateRepresentation",
    "Action",
    "Experience",
    "EpisodeResult",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "TradingEnvironment",
    "BaseRLAgent",
    "QLearningAgent",
    "DQNAgent",
    "PolicyGradientAgent",
    "ActorCriticAgent",
    "RLTrainer",
    "create_rl_agent",
    # Online Learning
    "OnlineConfig",
    "OnlineUpdate",
    "DriftDetectionResult",
    "DriftDetector",
    "DDMDetector",
    "ADWINDetector",
    "PageHinkleyDetector",
    "BaseOnlineLearner",
    "SGDOnlineLearner",
    "PassiveAggressiveLearner",
    "PerceptronLearner",
    "OnlineLearnerWithDriftDetection",
    "AdaptiveOnlineLearner",
    "create_online_learner",
    "create_drift_detector",
    # Anomaly Detection
    "AnomalyConfig",
    "AnomalyResult",
    "AnomalyReport",
    "BaseAnomalyDetector",
    "ZScoreDetector",
    "IQRDetector",
    "IsolationForestDetector",
    "LocalOutlierFactorDetector",
    "MahalanobisDetector",
    "AutoencoderDetector",
    "EnsembleAnomalyDetector",
    "StreamingAnomalyDetector",
    "create_anomaly_detector",
    # Regime Detection
    "RegimeConfig",
    "RegimeState",
    "RegimeTransition",
    "RegimeHistory",
    "BaseRegimeDetector",
    "HiddenMarkovModelDetector",
    "KMeansRegimeDetector",
    "ThresholdRegimeDetector",
    "ChangePointDetector",
    "EnsembleRegimeDetector",
    "RegimeAnalyzer",
    "create_regime_detector",
    # Model Persistence
    "PersistenceConfig",
    "ModelMetadata",
    "ModelVersion",
    "CheckpointInfo",
    "BaseSerializer",
    "PickleSerializer",
    "JSONSerializer",
    "NumpySerializer",
    "ModelStorage",
    "ModelVersionManager",
    "ModelRegistry",
    "create_model_storage",
]


class MLManager:
    """
    Central manager for all machine learning operations.

    Coordinates feature engineering, model training, evaluation,
    prediction, and signal generation.
    """

    def __init__(
        self,
        storage_path: str = "./models",
        cache_predictions: bool = True,
        enable_online_learning: bool = False,
        enable_anomaly_detection: bool = True,
        enable_regime_detection: bool = True,
    ) -> None:
        """
        Initialize ML manager.

        Args:
            storage_path: Path for model storage
            cache_predictions: Whether to cache predictions
            enable_online_learning: Whether to enable online learning
            enable_anomaly_detection: Whether to enable anomaly detection
            enable_regime_detection: Whether to enable regime detection
        """
        self.storage_path = storage_path
        self.cache_predictions = cache_predictions
        self.enable_online_learning = enable_online_learning
        self.enable_anomaly_detection = enable_anomaly_detection
        self.enable_regime_detection = enable_regime_detection

        self._feature_engineer: FeatureEngineer | None = None
        self._prediction_engine: PredictionEngine | None = None
        self._signal_generator: MultiModelSignalGenerator | None = None
        self._model_storage: ModelStorage | None = None
        self._anomaly_detector: EnsembleAnomalyDetector | None = None
        self._regime_detector: EnsembleRegimeDetector | None = None
        self._online_learners: dict[str, BaseOnlineLearner] = {}
        self._models: dict[str, BaseModel] = {}

        self._initialized = False

        logger.info("MLManager created")

    async def initialize(self) -> None:
        """Initialize all ML components."""
        try:
            # Initialize model storage
            persistence_config = PersistenceConfig(
                base_path=self.storage_path,
                serializer_type="pickle",
                compression=True,
                max_versions=10,
            )
            self._model_storage = create_model_storage(persistence_config)

            # Initialize feature engineer
            feature_config = FeatureConfig(
                scaling_method="standard",
                handle_missing=True,
                handle_outliers=True,
            )
            self._feature_engineer = create_feature_engineer(feature_config)

            # Initialize prediction engine
            prediction_config = PredictionConfig(
                aggregation_method="weighted_average",
                confidence_threshold=0.6,
                cache_enabled=self.cache_predictions,
            )
            self._prediction_engine = create_prediction_engine(prediction_config)

            # Initialize signal generator
            signal_config = SignalConfig(
                min_confidence=0.6,
                signal_threshold=0.5,
                use_risk_adjustment=True,
            )
            self._signal_generator = MultiModelSignalGenerator(
                config=signal_config,
                combination_method="weighted_voting",
            )

            # Initialize anomaly detector if enabled
            if self.enable_anomaly_detection:
                anomaly_config = AnomalyConfig(
                    contamination=0.05,
                    threshold_multiplier=3.0,
                )
                self._anomaly_detector = EnsembleAnomalyDetector(
                    config=anomaly_config,
                    combination_method="voting",
                )

            # Initialize regime detector if enabled
            if self.enable_regime_detection:
                regime_config = RegimeConfig(
                    n_regimes=3,
                    lookback_period=100,
                )
                self._regime_detector = EnsembleRegimeDetector(
                    config=regime_config,
                    combination_method="voting",
                )

            self._initialized = True
            logger.info("MLManager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize MLManager: {e}")
            raise

    async def add_model(
        self,
        name: str,
        model: BaseModel,
        weight: float = 1.0,
    ) -> None:
        """
        Add a model to the manager.

        Args:
            name: Model name
            model: Model instance
            weight: Model weight for ensemble predictions
        """
        if not self._initialized:
            raise RuntimeError("MLManager not initialized")

        self._models[name] = model

        if self._prediction_engine is not None:
            wrapper = ModelWrapper(model=model, name=name, weight=weight)
            self._prediction_engine.add_model(wrapper)

        if self._signal_generator is not None:
            generator = DirectionSignalGenerator(
                config=SignalConfig(signal_threshold=0.5),
            )
            self._signal_generator.add_generator(generator, weight=weight)

        logger.info(f"Added model '{name}' with weight {weight}")

    async def train_model(
        self,
        model_name: str,
        model_type: str,
        X_train: Any,
        y_train: Any,
        X_val: Any | None = None,
        y_val: Any | None = None,
        config: dict[str, Any] | None = None,
    ) -> TrainingResult:
        """
        Train a new model.

        Args:
            model_name: Name for the model
            model_type: Type of model to create
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            config: Model configuration

        Returns:
            Training result
        """
        if not self._initialized:
            raise RuntimeError("MLManager not initialized")

        try:
            # Create model
            model_config = ModelConfig(**(config or {}))
            model = create_model(model_type, model_config)

            # Engineer features
            if self._feature_engineer is not None:
                X_train = await self._feature_engineer.fit_transform_async(X_train)
                if X_val is not None:
                    X_val = await self._feature_engineer.transform_async(X_val)

            # Train model
            result = await model.train_async(
                X_train, y_train,
                X_val=X_val, y_val=y_val,
            )

            # Add to manager
            await self.add_model(model_name, model)

            # Save model
            if self._model_storage is not None:
                await self._model_storage.save_async(
                    model, model_name, "1.0.0",
                    metadata={"type": model_type, "config": config},
                )

            logger.info(f"Trained model '{model_name}' successfully")
            return result

        except Exception as e:
            logger.error(f"Failed to train model '{model_name}': {e}")
            raise

    async def predict(
        self,
        X: Any,
        model_name: str | None = None,
    ) -> PredictionResponse:
        """
        Make predictions.

        Args:
            X: Input features
            model_name: Specific model to use (None for ensemble)

        Returns:
            Prediction response
        """
        if not self._initialized:
            raise RuntimeError("MLManager not initialized")

        try:
            # Engineer features
            if self._feature_engineer is not None:
                X = await self._feature_engineer.transform_async(X)

            # Check for anomalies
            if self._anomaly_detector is not None:
                anomaly_result = self._anomaly_detector.detect(X)
                if anomaly_result.is_anomaly:
                    logger.warning(f"Anomaly detected: {anomaly_result}")

            # Detect regime
            regime = None
            if self._regime_detector is not None:
                regime_result = self._regime_detector.detect(X)
                regime = regime_result.current_regime

            # Make prediction
            if model_name is not None and model_name in self._models:
                model = self._models[model_name]
                result = await model.predict_async(X)
                return PredictionResponse(
                    predictions=result.predictions,
                    confidence=result.confidence,
                    model_name=model_name,
                    metadata={"regime": regime},
                )

            if self._prediction_engine is not None:
                request = PredictionRequest(features=X, symbol="unknown")
                response = await self._prediction_engine.predict_async(request)
                response.metadata["regime"] = regime
                return response

            raise ValueError("No prediction engine or model available")

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    async def generate_signal(
        self,
        X: Any,
        current_price: float,
        symbol: str = "unknown",
    ) -> TradingSignal:
        """
        Generate trading signal.

        Args:
            X: Input features
            current_price: Current price
            symbol: Trading symbol

        Returns:
            Trading signal
        """
        if not self._initialized:
            raise RuntimeError("MLManager not initialized")

        if self._signal_generator is None:
            raise ValueError("Signal generator not initialized")

        try:
            # Get prediction
            response = await self.predict(X)

            # Generate signal
            signal = self._signal_generator.generate(
                predictions=response.predictions,
                confidence=response.confidence,
                current_price=current_price,
                symbol=symbol,
            )

            return signal

        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            raise

    async def update_online(
        self,
        X: Any,
        y: float,
        model_name: str,
    ) -> OnlineUpdate:
        """
        Update online learner with new data.

        Args:
            X: Input features
            y: Target value
            model_name: Online learner name

        Returns:
            Online update result
        """
        if not self.enable_online_learning:
            raise ValueError("Online learning not enabled")

        if model_name not in self._online_learners:
            # Create new online learner
            config = OnlineConfig(
                learning_rate=0.01,
                regularization=0.001,
            )
            self._online_learners[model_name] = create_online_learner(
                "sgd", config,
            )

        learner = self._online_learners[model_name]

        # Engineer features
        if self._feature_engineer is not None:
            X = await self._feature_engineer.transform_async(X)

        # Update learner
        result = learner.partial_fit(X, y)

        return result

    async def evaluate_model(
        self,
        model_name: str,
        X_test: Any,
        y_test: Any,
    ) -> EvaluationReport:
        """
        Evaluate a model.

        Args:
            model_name: Model to evaluate
            X_test: Test features
            y_test: Test labels

        Returns:
            Evaluation report
        """
        if not self._initialized:
            raise RuntimeError("MLManager not initialized")

        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not found")

        try:
            model = self._models[model_name]

            # Engineer features
            if self._feature_engineer is not None:
                X_test = await self._feature_engineer.transform_async(X_test)

            # Get predictions
            result = await model.predict_async(X_test)

            # Create evaluator
            evaluator = create_evaluator(task_type="trading")

            # Evaluate
            report = evaluator.evaluate(
                y_true=y_test,
                y_pred=result.predictions,
                y_prob=result.probabilities,
            )

            return report

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    async def save_model(
        self,
        model_name: str,
        version: str = "1.0.0",
    ) -> ModelMetadata:
        """
        Save a model to storage.

        Args:
            model_name: Model to save
            version: Version string

        Returns:
            Model metadata
        """
        if not self._initialized:
            raise RuntimeError("MLManager not initialized")

        if model_name not in self._models:
            raise ValueError(f"Model '{model_name}' not found")

        if self._model_storage is None:
            raise ValueError("Model storage not initialized")

        model = self._models[model_name]
        metadata = await self._model_storage.save_async(
            model, model_name, version,
        )

        logger.info(f"Saved model '{model_name}' version {version}")
        return metadata

    async def load_model(
        self,
        model_name: str,
        version: str | None = None,
    ) -> BaseModel:
        """
        Load a model from storage.

        Args:
            model_name: Model to load
            version: Version to load (None for latest)

        Returns:
            Loaded model
        """
        if not self._initialized:
            raise RuntimeError("MLManager not initialized")

        if self._model_storage is None:
            raise ValueError("Model storage not initialized")

        model = await self._model_storage.load_async(model_name, version)
        await self.add_model(model_name, model)

        logger.info(f"Loaded model '{model_name}' version {version or 'latest'}")
        return model

    async def get_regime(self, X: Any) -> RegimeState:
        """
        Get current market regime.

        Args:
            X: Input features

        Returns:
            Regime state
        """
        if not self.enable_regime_detection:
            raise ValueError("Regime detection not enabled")

        if self._regime_detector is None:
            raise ValueError("Regime detector not initialized")

        # Engineer features
        if self._feature_engineer is not None:
            X = await self._feature_engineer.transform_async(X)

        result = self._regime_detector.detect(X)
        return result.current_regime

    async def check_anomaly(self, X: Any) -> AnomalyResult:
        """
        Check for anomalies.

        Args:
            X: Input features

        Returns:
            Anomaly result
        """
        if not self.enable_anomaly_detection:
            raise ValueError("Anomaly detection not enabled")

        if self._anomaly_detector is None:
            raise ValueError("Anomaly detector not initialized")

        # Engineer features
        if self._feature_engineer is not None:
            X = await self._feature_engineer.transform_async(X)

        return self._anomaly_detector.detect(X)

    def get_model(self, name: str) -> BaseModel | None:
        """Get a model by name."""
        return self._models.get(name)

    def list_models(self) -> list[str]:
        """List all loaded models."""
        return list(self._models.keys())

    async def shutdown(self) -> None:
        """Shutdown ML manager and cleanup resources."""
        logger.info("Shutting down MLManager")

        # Clear models
        self._models.clear()
        self._online_learners.clear()

        # Clear components
        self._feature_engineer = None
        self._prediction_engine = None
        self._signal_generator = None
        self._anomaly_detector = None
        self._regime_detector = None
        self._model_storage = None

        self._initialized = False

        logger.info("MLManager shutdown complete")


def create_ml_manager(
    storage_path: str = "./models",
    cache_predictions: bool = True,
    enable_online_learning: bool = False,
    enable_anomaly_detection: bool = True,
    enable_regime_detection: bool = True,
) -> MLManager:
    """
    Create an ML manager instance.

    Args:
        storage_path: Path for model storage
        cache_predictions: Whether to cache predictions
        enable_online_learning: Whether to enable online learning
        enable_anomaly_detection: Whether to enable anomaly detection
        enable_regime_detection: Whether to enable regime detection

    Returns:
        Configured MLManager instance
    """
    return MLManager(
        storage_path=storage_path,
        cache_predictions=cache_predictions,
        enable_online_learning=enable_online_learning,
        enable_anomaly_detection=enable_anomaly_detection,
        enable_regime_detection=enable_regime_detection,
    )


# Module version
__version__ = "2.2.0"
