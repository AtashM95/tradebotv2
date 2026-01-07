"""
OpenAI Configuration Module for Ultimate Trading Bot v2.2.

This module provides comprehensive configuration for OpenAI API integration
including model settings, rate limiting, cost tracking, and prompt management.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from functools import lru_cache
from datetime import datetime, timezone
import logging

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator


logger = logging.getLogger(__name__)


class OpenAIModel(str, Enum):
    """Available OpenAI models."""
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4 = "gpt-4"
    GPT_35_TURBO = "gpt-3.5-turbo"
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_ADA = "text-embedding-ada-002"


class TaskType(str, Enum):
    """AI task type enumeration."""
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    CHART_ANALYSIS = "chart_analysis"
    NEWS_SUMMARY = "news_summary"
    TRADING_SIGNAL = "trading_signal"
    RISK_ASSESSMENT = "risk_assessment"
    MARKET_ANALYSIS = "market_analysis"
    STRATEGY_ADVICE = "strategy_advice"
    CHAT = "chat"
    EMBEDDING = "embedding"


@dataclass(frozen=True)
class ModelPricing:
    """Model pricing information per 1M tokens."""
    input_price: float
    output_price: float
    context_window: int


# Model pricing as of 2024 (per 1M tokens)
MODEL_PRICING: Dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing(input_price=2.50, output_price=10.00, context_window=128000),
    "gpt-4o-mini": ModelPricing(input_price=0.15, output_price=0.60, context_window=128000),
    "gpt-4-turbo": ModelPricing(input_price=10.00, output_price=30.00, context_window=128000),
    "gpt-4": ModelPricing(input_price=30.00, output_price=60.00, context_window=8192),
    "gpt-3.5-turbo": ModelPricing(input_price=0.50, output_price=1.50, context_window=16385),
    "text-embedding-3-small": ModelPricing(input_price=0.02, output_price=0.0, context_window=8191),
    "text-embedding-3-large": ModelPricing(input_price=0.13, output_price=0.0, context_window=8191),
    "text-embedding-ada-002": ModelPricing(input_price=0.10, output_price=0.0, context_window=8191),
}


class ModelConfig(BaseModel):
    """Configuration for a specific model."""

    model: str = Field(default="gpt-4o", description="Model identifier")
    max_tokens: int = Field(default=4096, ge=1, le=128000, description="Max tokens")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    stop_sequences: List[str] = Field(default_factory=list, description="Stop sequences")

    @property
    def pricing(self) -> Optional[ModelPricing]:
        """Get pricing for this model."""
        return MODEL_PRICING.get(self.model)

    @property
    def context_window(self) -> int:
        """Get context window size."""
        pricing = self.pricing
        return pricing.context_window if pricing else 4096


class TaskModelMapping(BaseModel):
    """Mapping of tasks to model configurations."""

    sentiment_analysis: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="gpt-4o-mini",
            max_tokens=500,
            temperature=0.3
        ),
        description="Sentiment analysis model config"
    )

    chart_analysis: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="gpt-4o",
            max_tokens=2000,
            temperature=0.5
        ),
        description="Chart analysis model config (vision)"
    )

    news_summary: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="gpt-4o-mini",
            max_tokens=1000,
            temperature=0.3
        ),
        description="News summary model config"
    )

    trading_signal: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="gpt-4o",
            max_tokens=1500,
            temperature=0.2
        ),
        description="Trading signal model config"
    )

    risk_assessment: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="gpt-4o",
            max_tokens=1000,
            temperature=0.2
        ),
        description="Risk assessment model config"
    )

    market_analysis: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="gpt-4o",
            max_tokens=2000,
            temperature=0.4
        ),
        description="Market analysis model config"
    )

    strategy_advice: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="gpt-4o",
            max_tokens=2000,
            temperature=0.5
        ),
        description="Strategy advice model config"
    )

    chat: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="gpt-4o",
            max_tokens=4096,
            temperature=0.7
        ),
        description="Chat/conversation model config"
    )

    embedding: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model="text-embedding-3-small",
            max_tokens=8191,
            temperature=0.0
        ),
        description="Embedding model config"
    )

    def get_config(self, task_type: TaskType) -> ModelConfig:
        """
        Get model configuration for a task type.

        Args:
            task_type: The type of task

        Returns:
            ModelConfig for the task
        """
        task_mapping = {
            TaskType.SENTIMENT_ANALYSIS: self.sentiment_analysis,
            TaskType.CHART_ANALYSIS: self.chart_analysis,
            TaskType.NEWS_SUMMARY: self.news_summary,
            TaskType.TRADING_SIGNAL: self.trading_signal,
            TaskType.RISK_ASSESSMENT: self.risk_assessment,
            TaskType.MARKET_ANALYSIS: self.market_analysis,
            TaskType.STRATEGY_ADVICE: self.strategy_advice,
            TaskType.CHAT: self.chat,
            TaskType.EMBEDDING: self.embedding,
        }
        return task_mapping.get(task_type, self.chat)


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""

    requests_per_minute: int = Field(default=60, ge=1, description="Requests per minute limit")
    tokens_per_minute: int = Field(default=90000, ge=1, description="Tokens per minute limit")
    requests_per_day: int = Field(default=10000, ge=1, description="Requests per day limit")
    tokens_per_day: int = Field(default=2000000, ge=1, description="Tokens per day limit")

    # Burst settings
    burst_limit: int = Field(default=10, ge=1, description="Burst request limit")
    burst_window_seconds: int = Field(default=10, ge=1, description="Burst window in seconds")

    # Retry settings
    max_retries: int = Field(default=3, ge=0, le=10, description="Max retries on rate limit")
    retry_delay_seconds: float = Field(default=1.0, ge=0.1, description="Initial retry delay")
    retry_backoff_factor: float = Field(default=2.0, ge=1.0, le=5.0, description="Retry backoff factor")
    max_retry_delay_seconds: float = Field(default=60.0, ge=1.0, description="Max retry delay")

    # Queue settings
    use_request_queue: bool = Field(default=True, description="Use request queue")
    queue_timeout_seconds: float = Field(default=60.0, ge=1.0, description="Queue timeout")


class BudgetConfig(BaseModel):
    """Budget and cost management configuration."""

    # Budget limits
    daily_budget_usd: float = Field(default=50.0, ge=0.0, description="Daily budget in USD")
    weekly_budget_usd: float = Field(default=200.0, ge=0.0, description="Weekly budget in USD")
    monthly_budget_usd: float = Field(default=500.0, ge=0.0, description="Monthly budget in USD")

    # Warning thresholds
    warning_threshold_percent: float = Field(default=80.0, ge=50.0, le=100.0, description="Warning threshold %")
    critical_threshold_percent: float = Field(default=95.0, ge=80.0, le=100.0, description="Critical threshold %")

    # Actions on budget limits
    action_on_daily_limit: str = Field(default="reduce_usage", description="Action on daily limit")
    action_on_weekly_limit: str = Field(default="reduce_usage", description="Action on weekly limit")
    action_on_monthly_limit: str = Field(default="stop", description="Action on monthly limit")

    # Cost tracking
    track_costs: bool = Field(default=True, description="Enable cost tracking")
    log_costs: bool = Field(default=True, description="Log API costs")
    cost_log_file: str = Field(default="logs/openai_costs.json", description="Cost log file")

    # Task-specific budgets
    task_budgets: Dict[str, float] = Field(
        default_factory=lambda: {
            "sentiment_analysis": 10.0,
            "chart_analysis": 20.0,
            "trading_signal": 15.0,
            "market_analysis": 10.0,
        },
        description="Daily budget per task type (USD)"
    )


class CacheConfig(BaseModel):
    """Response caching configuration."""

    enabled: bool = Field(default=True, description="Enable response caching")
    backend: str = Field(default="redis", description="Cache backend (redis, memory)")

    # TTL settings (in seconds)
    default_ttl: int = Field(default=3600, ge=60, description="Default TTL")
    sentiment_ttl: int = Field(default=1800, ge=60, description="Sentiment cache TTL")
    news_summary_ttl: int = Field(default=1800, ge=60, description="News summary cache TTL")
    market_analysis_ttl: int = Field(default=900, ge=60, description="Market analysis cache TTL")
    embedding_ttl: int = Field(default=86400, ge=3600, description="Embedding cache TTL")

    # Cache size limits
    max_cache_size_mb: int = Field(default=500, ge=10, description="Max cache size in MB")
    max_entries: int = Field(default=10000, ge=100, description="Max cache entries")

    # Cache key settings
    include_model_in_key: bool = Field(default=True, description="Include model in cache key")
    include_temperature_in_key: bool = Field(default=True, description="Include temperature in key")


class SafetyConfig(BaseModel):
    """Safety and content filtering configuration."""

    # Content filtering
    filter_responses: bool = Field(default=True, description="Filter inappropriate responses")
    max_response_length: int = Field(default=10000, ge=100, description="Max response length")

    # Validation
    validate_json_responses: bool = Field(default=True, description="Validate JSON responses")
    require_structured_output: bool = Field(default=False, description="Require structured output")

    # Error handling
    fallback_on_error: bool = Field(default=True, description="Use fallback on error")
    fallback_response: str = Field(default="Unable to process request", description="Fallback response")

    # Logging
    log_all_requests: bool = Field(default=False, description="Log all requests")
    log_errors: bool = Field(default=True, description="Log errors")
    mask_sensitive_data: bool = Field(default=True, description="Mask sensitive data in logs")


class OpenAIConfig(BaseModel):
    """
    Master OpenAI configuration.

    Aggregates all OpenAI-related configurations for the trading bot.
    """

    # API credentials
    api_key: SecretStr = Field(
        default=SecretStr(os.getenv("OPENAI_API_KEY", "")),
        description="OpenAI API key"
    )
    organization_id: Optional[str] = Field(
        default=os.getenv("OPENAI_ORG_ID"),
        description="OpenAI organization ID"
    )

    # Base settings
    base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API base URL"
    )
    timeout: float = Field(default=60.0, ge=1.0, le=300.0, description="Request timeout")
    max_retries: int = Field(default=3, ge=0, le=10, description="Max retries")

    # Model configurations
    default_model: str = Field(default="gpt-4o", description="Default model")
    vision_model: str = Field(default="gpt-4o", description="Vision model for charts")
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model")
    task_models: TaskModelMapping = Field(
        default_factory=TaskModelMapping,
        description="Task-specific model mappings"
    )

    # Component configurations
    rate_limit: RateLimitConfig = Field(
        default_factory=RateLimitConfig,
        description="Rate limiting configuration"
    )
    budget: BudgetConfig = Field(
        default_factory=BudgetConfig,
        description="Budget configuration"
    )
    cache: CacheConfig = Field(
        default_factory=CacheConfig,
        description="Caching configuration"
    )
    safety: SafetyConfig = Field(
        default_factory=SafetyConfig,
        description="Safety configuration"
    )

    # Feature flags
    enabled: bool = Field(default=True, description="Enable OpenAI integration")
    use_streaming: bool = Field(default=False, description="Use streaming responses")
    use_function_calling: bool = Field(default=True, description="Use function calling")
    use_json_mode: bool = Field(default=True, description="Use JSON mode where applicable")

    @property
    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key.get_secret_value())

    def get_headers(self) -> Dict[str, str]:
        """
        Get API request headers.

        Returns:
            Dictionary of headers
        """
        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        if self.organization_id:
            headers["OpenAI-Organization"] = self.organization_id
        return headers

    def get_model_config(self, task_type: TaskType) -> ModelConfig:
        """
        Get model configuration for a task type.

        Args:
            task_type: The type of task

        Returns:
            ModelConfig for the task
        """
        return self.task_models.get_config(task_type)

    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Estimate cost for a request.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        pricing = MODEL_PRICING.get(model)
        if not pricing:
            return 0.0

        input_cost = (input_tokens / 1_000_000) * pricing.input_price
        output_cost = (output_tokens / 1_000_000) * pricing.output_price
        return input_cost + output_cost


@lru_cache()
def get_openai_config() -> OpenAIConfig:
    """
    Get cached OpenAI configuration.

    Returns:
        Singleton OpenAIConfig instance
    """
    return OpenAIConfig()


def reload_openai_config() -> OpenAIConfig:
    """
    Reload OpenAI configuration.

    Returns:
        New OpenAIConfig instance
    """
    get_openai_config.cache_clear()
    return get_openai_config()


# Module-level OpenAI config instance
openai_config = get_openai_config()


# System prompts for different tasks
SYSTEM_PROMPTS: Dict[str, str] = {
    "sentiment_analysis": """You are a financial sentiment analysis expert. Analyze the provided text and determine the sentiment towards the specified stock or market. Provide a sentiment score from -1.0 (very bearish) to 1.0 (very bullish), along with key factors influencing your assessment. Be objective and consider multiple perspectives.""",

    "chart_analysis": """You are an expert technical analyst. Analyze the provided chart image and identify key patterns, support/resistance levels, trend direction, and potential trading signals. Provide specific price levels and explain your reasoning. Consider multiple timeframes if visible.""",

    "news_summary": """You are a financial news analyst. Summarize the provided news articles focusing on their potential market impact. Identify key themes, affected sectors/stocks, and potential trading implications. Be concise but comprehensive.""",

    "trading_signal": """You are an experienced trading analyst. Based on the provided market data and analysis, generate a trading signal with entry price, stop loss, and take profit levels. Explain your reasoning and provide a confidence score. Consider risk management principles.""",

    "risk_assessment": """You are a risk management specialist. Evaluate the provided trading opportunity or portfolio position for potential risks. Consider market risk, liquidity risk, correlation risk, and external factors. Provide specific recommendations for risk mitigation.""",

    "market_analysis": """You are a market strategist. Analyze the current market conditions including trends, volatility, sector rotation, and macroeconomic factors. Provide insights on market direction and potential opportunities. Be balanced and consider multiple scenarios.""",

    "strategy_advice": """You are a trading strategy consultant. Based on the provided information about the trader's goals, risk tolerance, and current positions, provide strategic advice. Consider portfolio diversification, position sizing, and timing. Be practical and actionable.""",
}
