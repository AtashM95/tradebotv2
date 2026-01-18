"""
AI Model Manager Module for Ultimate Trading Bot v2.2.

This module provides model selection, fallback management,
and performance tracking for AI operations.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from src.ai.openai_client import OpenAIClient, OpenAIModel
from src.utils.helpers import generate_uuid
from src.utils.date_utils import now_utc


logger = logging.getLogger(__name__)


class ModelCapability(str, Enum):
    """Model capability enumeration."""

    CHAT = "chat"
    ANALYSIS = "analysis"
    CODE = "code"
    VISION = "vision"
    EMBEDDING = "embedding"
    FUNCTION_CALLING = "function_calling"


class ModelTier(str, Enum):
    """Model tier enumeration."""

    PREMIUM = "premium"
    STANDARD = "standard"
    ECONOMY = "economy"


class TaskComplexity(str, Enum):
    """Task complexity enumeration."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ModelStatus(str, Enum):
    """Model status enumeration."""

    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"


class ModelInfo(BaseModel):
    """Model information."""

    model: OpenAIModel
    tier: ModelTier
    capabilities: list[ModelCapability] = Field(default_factory=list)
    context_window: int = Field(default=8192)
    cost_per_1k_input: float = Field(default=0.0)
    cost_per_1k_output: float = Field(default=0.0)
    max_output_tokens: int = Field(default=4096)
    avg_latency_ms: float = Field(default=1000.0)
    supports_streaming: bool = Field(default=True)


class ModelPerformance(BaseModel):
    """Model performance metrics."""

    model: OpenAIModel
    total_requests: int = Field(default=0)
    successful_requests: int = Field(default=0)
    failed_requests: int = Field(default=0)
    total_tokens_used: int = Field(default=0)
    total_cost: float = Field(default=0.0)
    avg_latency_ms: float = Field(default=0.0)
    p95_latency_ms: float = Field(default=0.0)
    last_used: Optional[datetime] = None
    last_error: Optional[str] = None
    error_count_last_hour: int = Field(default=0)


class ModelSelection(BaseModel):
    """Model selection result."""

    selected_model: OpenAIModel
    fallback_models: list[OpenAIModel] = Field(default_factory=list)
    reason: str = Field(default="")
    estimated_cost: float = Field(default=0.0)
    estimated_latency_ms: float = Field(default=0.0)


class AIModelManagerConfig(BaseModel):
    """Configuration for AI model manager."""

    default_model: OpenAIModel = Field(default=OpenAIModel.GPT_4O)
    fallback_model: OpenAIModel = Field(default=OpenAIModel.GPT_4O_MINI)
    economy_model: OpenAIModel = Field(default=OpenAIModel.GPT_4O_MINI)
    enable_auto_fallback: bool = Field(default=True)
    max_retries_before_fallback: int = Field(default=2, ge=1, le=5)
    error_threshold_for_degraded: int = Field(default=5, ge=1, le=20)
    prefer_cost_optimization: bool = Field(default=False)
    latency_threshold_ms: float = Field(default=5000.0)


MODEL_REGISTRY: dict[OpenAIModel, ModelInfo] = {
    OpenAIModel.GPT_4O: ModelInfo(
        model=OpenAIModel.GPT_4O,
        tier=ModelTier.PREMIUM,
        capabilities=[
            ModelCapability.CHAT,
            ModelCapability.ANALYSIS,
            ModelCapability.CODE,
            ModelCapability.VISION,
            ModelCapability.FUNCTION_CALLING,
        ],
        context_window=128000,
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015,
        max_output_tokens=16384,
        avg_latency_ms=1500,
    ),
    OpenAIModel.GPT_4O_MINI: ModelInfo(
        model=OpenAIModel.GPT_4O_MINI,
        tier=ModelTier.STANDARD,
        capabilities=[
            ModelCapability.CHAT,
            ModelCapability.ANALYSIS,
            ModelCapability.CODE,
            ModelCapability.FUNCTION_CALLING,
        ],
        context_window=128000,
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        max_output_tokens=16384,
        avg_latency_ms=800,
    ),
    OpenAIModel.GPT_4_TURBO: ModelInfo(
        model=OpenAIModel.GPT_4_TURBO,
        tier=ModelTier.PREMIUM,
        capabilities=[
            ModelCapability.CHAT,
            ModelCapability.ANALYSIS,
            ModelCapability.CODE,
            ModelCapability.VISION,
            ModelCapability.FUNCTION_CALLING,
        ],
        context_window=128000,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
        max_output_tokens=4096,
        avg_latency_ms=2000,
    ),
    OpenAIModel.GPT_35_TURBO: ModelInfo(
        model=OpenAIModel.GPT_35_TURBO,
        tier=ModelTier.ECONOMY,
        capabilities=[
            ModelCapability.CHAT,
            ModelCapability.FUNCTION_CALLING,
        ],
        context_window=16384,
        cost_per_1k_input=0.0005,
        cost_per_1k_output=0.0015,
        max_output_tokens=4096,
        avg_latency_ms=600,
    ),
}


class AIModelManager:
    """
    AI Model Manager for intelligent model selection.

    Provides:
    - Automatic model selection based on task
    - Fallback handling for failures
    - Performance tracking
    - Cost optimization
    """

    def __init__(
        self,
        config: Optional[AIModelManagerConfig] = None,
        openai_client: Optional[OpenAIClient] = None,
    ) -> None:
        """
        Initialize AIModelManager.

        Args:
            config: Manager configuration
            openai_client: OpenAI client instance
        """
        self._config = config or AIModelManagerConfig()
        self._client = openai_client

        self._model_status: dict[OpenAIModel, ModelStatus] = {}
        self._model_performance: dict[OpenAIModel, ModelPerformance] = {}
        self._latency_history: dict[OpenAIModel, list[float]] = {}
        self._error_timestamps: dict[OpenAIModel, list[datetime]] = {}

        self._initialize_models()

        logger.info("AIModelManager initialized")

    def _initialize_models(self) -> None:
        """Initialize model tracking."""
        for model in OpenAIModel:
            self._model_status[model] = ModelStatus.AVAILABLE
            self._model_performance[model] = ModelPerformance(model=model)
            self._latency_history[model] = []
            self._error_timestamps[model] = []

    def set_client(self, client: OpenAIClient) -> None:
        """Set the OpenAI client."""
        self._client = client

    def select_model(
        self,
        task_complexity: TaskComplexity = TaskComplexity.MEDIUM,
        required_capabilities: Optional[list[ModelCapability]] = None,
        max_tokens_needed: int = 1000,
        prefer_speed: bool = False,
        prefer_quality: bool = False,
        budget_limit: Optional[float] = None,
    ) -> ModelSelection:
        """
        Select the best model for a task.

        Args:
            task_complexity: Complexity of the task
            required_capabilities: Required model capabilities
            max_tokens_needed: Estimated tokens needed
            prefer_speed: Prioritize low latency
            prefer_quality: Prioritize quality
            budget_limit: Maximum cost allowed

        Returns:
            ModelSelection with recommended model
        """
        required_caps = required_capabilities or [ModelCapability.CHAT]

        candidates = self._filter_candidates(
            required_capabilities=required_caps,
            max_tokens=max_tokens_needed,
        )

        if not candidates:
            return ModelSelection(
                selected_model=self._config.fallback_model,
                fallback_models=[self._config.economy_model],
                reason="No suitable models found, using fallback",
            )

        scored_candidates = self._score_candidates(
            candidates=candidates,
            task_complexity=task_complexity,
            prefer_speed=prefer_speed,
            prefer_quality=prefer_quality,
            budget_limit=budget_limit,
            tokens_estimate=max_tokens_needed,
        )

        sorted_candidates = sorted(
            scored_candidates.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        selected = sorted_candidates[0][0]
        fallbacks = [m for m, _ in sorted_candidates[1:3]]

        model_info = MODEL_REGISTRY.get(selected)
        estimated_cost = 0.0
        estimated_latency = 1000.0

        if model_info:
            estimated_cost = (
                (max_tokens_needed / 1000 * model_info.cost_per_1k_input) +
                (max_tokens_needed / 1000 * model_info.cost_per_1k_output)
            )
            estimated_latency = model_info.avg_latency_ms

        reason = self._generate_selection_reason(
            selected=selected,
            task_complexity=task_complexity,
            prefer_speed=prefer_speed,
            prefer_quality=prefer_quality,
        )

        return ModelSelection(
            selected_model=selected,
            fallback_models=fallbacks,
            reason=reason,
            estimated_cost=estimated_cost,
            estimated_latency_ms=estimated_latency,
        )

    def _filter_candidates(
        self,
        required_capabilities: list[ModelCapability],
        max_tokens: int,
    ) -> list[OpenAIModel]:
        """Filter models by requirements."""
        candidates: list[OpenAIModel] = []

        for model, info in MODEL_REGISTRY.items():
            if self._model_status.get(model) == ModelStatus.UNAVAILABLE:
                continue

            if not all(cap in info.capabilities for cap in required_capabilities):
                continue

            if max_tokens > info.max_output_tokens:
                continue

            candidates.append(model)

        return candidates

    def _score_candidates(
        self,
        candidates: list[OpenAIModel],
        task_complexity: TaskComplexity,
        prefer_speed: bool,
        prefer_quality: bool,
        budget_limit: Optional[float],
        tokens_estimate: int,
    ) -> dict[OpenAIModel, float]:
        """Score candidate models."""
        scores: dict[OpenAIModel, float] = {}

        for model in candidates:
            info = MODEL_REGISTRY.get(model)
            if not info:
                continue

            score = 50.0

            if task_complexity == TaskComplexity.HIGH:
                if info.tier == ModelTier.PREMIUM:
                    score += 30
                elif info.tier == ModelTier.STANDARD:
                    score += 15
            elif task_complexity == TaskComplexity.LOW:
                if info.tier == ModelTier.ECONOMY:
                    score += 25
                elif info.tier == ModelTier.STANDARD:
                    score += 20

            if prefer_speed:
                latency_score = max(0, 30 - (info.avg_latency_ms / 100))
                score += latency_score

            if prefer_quality:
                if info.tier == ModelTier.PREMIUM:
                    score += 25
                elif info.tier == ModelTier.STANDARD:
                    score += 10

            if budget_limit is not None:
                estimated_cost = (
                    (tokens_estimate / 1000 * info.cost_per_1k_input) +
                    (tokens_estimate / 1000 * info.cost_per_1k_output)
                )
                if estimated_cost > budget_limit:
                    score -= 50
                else:
                    savings = (budget_limit - estimated_cost) / budget_limit
                    score += savings * 20

            if self._config.prefer_cost_optimization:
                cost_score = max(0, 20 - (info.cost_per_1k_output * 1000))
                score += cost_score

            status = self._model_status.get(model, ModelStatus.AVAILABLE)
            if status == ModelStatus.DEGRADED:
                score -= 20
            elif status == ModelStatus.RATE_LIMITED:
                score -= 30

            perf = self._model_performance.get(model)
            if perf and perf.total_requests > 10:
                success_rate = perf.successful_requests / perf.total_requests
                score += (success_rate - 0.9) * 50

            scores[model] = max(0, score)

        return scores

    def _generate_selection_reason(
        self,
        selected: OpenAIModel,
        task_complexity: TaskComplexity,
        prefer_speed: bool,
        prefer_quality: bool,
    ) -> str:
        """Generate reason for model selection."""
        info = MODEL_REGISTRY.get(selected)
        if not info:
            return f"Selected {selected.value}"

        reasons: list[str] = []

        if task_complexity == TaskComplexity.HIGH and info.tier == ModelTier.PREMIUM:
            reasons.append("high complexity requires premium model")
        elif task_complexity == TaskComplexity.LOW and info.tier != ModelTier.PREMIUM:
            reasons.append("simple task allows cost-effective model")

        if prefer_speed:
            reasons.append(f"optimized for speed ({info.avg_latency_ms}ms avg)")

        if prefer_quality and info.tier == ModelTier.PREMIUM:
            reasons.append("prioritizing quality")

        if self._config.prefer_cost_optimization and info.tier != ModelTier.PREMIUM:
            reasons.append("cost-optimized selection")

        reason_str = ", ".join(reasons) if reasons else "best overall fit"
        return f"Selected {selected.value}: {reason_str}"

    def record_request(
        self,
        model: OpenAIModel,
        success: bool,
        latency_ms: float,
        tokens_used: int,
        cost: float,
        error: Optional[str] = None,
    ) -> None:
        """
        Record a model request for performance tracking.

        Args:
            model: Model used
            success: Whether request succeeded
            latency_ms: Request latency
            tokens_used: Tokens consumed
            cost: Request cost
            error: Error message if failed
        """
        perf = self._model_performance.get(model)
        if not perf:
            perf = ModelPerformance(model=model)
            self._model_performance[model] = perf

        perf.total_requests += 1
        if success:
            perf.successful_requests += 1
        else:
            perf.failed_requests += 1
            perf.last_error = error
            self._error_timestamps[model].append(now_utc())

        perf.total_tokens_used += tokens_used
        perf.total_cost += cost
        perf.last_used = now_utc()

        self._latency_history[model].append(latency_ms)
        if len(self._latency_history[model]) > 100:
            self._latency_history[model] = self._latency_history[model][-100:]

        latencies = self._latency_history[model]
        perf.avg_latency_ms = sum(latencies) / len(latencies)
        sorted_latencies = sorted(latencies)
        p95_idx = int(len(sorted_latencies) * 0.95)
        perf.p95_latency_ms = sorted_latencies[p95_idx] if sorted_latencies else 0

        self._update_model_status(model)

    def _update_model_status(self, model: OpenAIModel) -> None:
        """Update model status based on recent performance."""
        now = now_utc()
        hour_ago = now - timedelta(hours=1)

        recent_errors = [
            ts for ts in self._error_timestamps.get(model, [])
            if ts > hour_ago
        ]
        self._error_timestamps[model] = recent_errors

        perf = self._model_performance.get(model)
        if perf:
            perf.error_count_last_hour = len(recent_errors)

        if len(recent_errors) >= self._config.error_threshold_for_degraded:
            self._model_status[model] = ModelStatus.DEGRADED
            logger.warning(f"Model {model.value} marked as degraded")
        elif len(recent_errors) >= self._config.error_threshold_for_degraded * 2:
            self._model_status[model] = ModelStatus.UNAVAILABLE
            logger.error(f"Model {model.value} marked as unavailable")
        else:
            self._model_status[model] = ModelStatus.AVAILABLE

    async def execute_with_fallback(
        self,
        operation: Callable,
        selection: ModelSelection,
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute an operation with automatic fallback.

        Args:
            operation: Async function to execute
            selection: Model selection with fallbacks
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation

        Returns:
            Operation result
        """
        models_to_try = [selection.selected_model] + selection.fallback_models
        last_error: Optional[Exception] = None

        for model in models_to_try:
            if self._model_status.get(model) == ModelStatus.UNAVAILABLE:
                continue

            try:
                start_time = datetime.now()
                result = await operation(model=model, *args, **kwargs)
                latency_ms = (datetime.now() - start_time).total_seconds() * 1000

                self.record_request(
                    model=model,
                    success=True,
                    latency_ms=latency_ms,
                    tokens_used=kwargs.get("tokens_used", 0),
                    cost=kwargs.get("cost", 0.0),
                )

                return result

            except Exception as e:
                last_error = e
                logger.warning(f"Model {model.value} failed: {e}")

                self.record_request(
                    model=model,
                    success=False,
                    latency_ms=0,
                    tokens_used=0,
                    cost=0.0,
                    error=str(e),
                )

                if not self._config.enable_auto_fallback:
                    raise

        if last_error:
            raise last_error

        raise RuntimeError("All models failed")

    def get_model_for_task(
        self,
        task_type: str,
    ) -> OpenAIModel:
        """
        Get recommended model for a specific task type.

        Args:
            task_type: Type of task

        Returns:
            Recommended model
        """
        task_models = {
            "sentiment_analysis": OpenAIModel.GPT_4O_MINI,
            "technical_analysis": OpenAIModel.GPT_4O,
            "signal_generation": OpenAIModel.GPT_4O,
            "risk_assessment": OpenAIModel.GPT_4O,
            "chat": OpenAIModel.GPT_4O_MINI,
            "market_summary": OpenAIModel.GPT_4O_MINI,
            "strategy_advice": OpenAIModel.GPT_4O,
            "code_generation": OpenAIModel.GPT_4O,
            "simple_query": OpenAIModel.GPT_35_TURBO,
        }

        model = task_models.get(task_type, self._config.default_model)

        if self._model_status.get(model) == ModelStatus.UNAVAILABLE:
            return self._config.fallback_model

        return model

    def get_model_info(self, model: OpenAIModel) -> Optional[ModelInfo]:
        """Get information about a model."""
        return MODEL_REGISTRY.get(model)

    def get_model_status(self, model: OpenAIModel) -> ModelStatus:
        """Get current status of a model."""
        return self._model_status.get(model, ModelStatus.AVAILABLE)

    def get_model_performance(self, model: OpenAIModel) -> Optional[ModelPerformance]:
        """Get performance metrics for a model."""
        return self._model_performance.get(model)

    def reset_model_status(self, model: OpenAIModel) -> None:
        """Reset a model's status to available."""
        self._model_status[model] = ModelStatus.AVAILABLE
        self._error_timestamps[model] = []
        logger.info(f"Reset status for model {model.value}")

    def get_all_model_stats(self) -> dict[str, dict]:
        """Get statistics for all models."""
        stats: dict[str, dict] = {}

        for model in OpenAIModel:
            info = MODEL_REGISTRY.get(model)
            perf = self._model_performance.get(model)
            status = self._model_status.get(model, ModelStatus.AVAILABLE)

            stats[model.value] = {
                "status": status.value,
                "tier": info.tier.value if info else "unknown",
                "total_requests": perf.total_requests if perf else 0,
                "success_rate": (
                    perf.successful_requests / perf.total_requests * 100
                    if perf and perf.total_requests > 0 else 100.0
                ),
                "avg_latency_ms": perf.avg_latency_ms if perf else 0,
                "total_cost": perf.total_cost if perf else 0,
                "errors_last_hour": perf.error_count_last_hour if perf else 0,
            }

        return stats

    def estimate_cost(
        self,
        model: OpenAIModel,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Estimate cost for a request.

        Args:
            model: Model to use
            input_tokens: Estimated input tokens
            output_tokens: Estimated output tokens

        Returns:
            Estimated cost in USD
        """
        info = MODEL_REGISTRY.get(model)
        if not info:
            return 0.0

        input_cost = (input_tokens / 1000) * info.cost_per_1k_input
        output_cost = (output_tokens / 1000) * info.cost_per_1k_output

        return input_cost + output_cost

    def get_cheapest_model(
        self,
        capabilities: Optional[list[ModelCapability]] = None,
    ) -> OpenAIModel:
        """Get the cheapest available model with required capabilities."""
        required_caps = capabilities or [ModelCapability.CHAT]

        cheapest: Optional[OpenAIModel] = None
        lowest_cost = float("inf")

        for model, info in MODEL_REGISTRY.items():
            if self._model_status.get(model) == ModelStatus.UNAVAILABLE:
                continue

            if not all(cap in info.capabilities for cap in required_caps):
                continue

            total_cost = info.cost_per_1k_input + info.cost_per_1k_output
            if total_cost < lowest_cost:
                lowest_cost = total_cost
                cheapest = model

        return cheapest or self._config.economy_model

    def get_fastest_model(
        self,
        capabilities: Optional[list[ModelCapability]] = None,
    ) -> OpenAIModel:
        """Get the fastest available model with required capabilities."""
        required_caps = capabilities or [ModelCapability.CHAT]

        fastest: Optional[OpenAIModel] = None
        lowest_latency = float("inf")

        for model, info in MODEL_REGISTRY.items():
            if self._model_status.get(model) == ModelStatus.UNAVAILABLE:
                continue

            if not all(cap in info.capabilities for cap in required_caps):
                continue

            if info.avg_latency_ms < lowest_latency:
                lowest_latency = info.avg_latency_ms
                fastest = model

        return fastest or self._config.default_model

    def __repr__(self) -> str:
        """String representation."""
        total_requests = sum(
            p.total_requests for p in self._model_performance.values()
        )
        return f"AIModelManager(models={len(MODEL_REGISTRY)}, requests={total_requests})"
