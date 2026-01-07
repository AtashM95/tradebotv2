"""
OpenAI Client Module for Ultimate Trading Bot v2.2.

This module provides the OpenAI API client for GPT-4o integration,
handling chat completions, function calling, and cost tracking.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional

import httpx
from pydantic import BaseModel, Field

from src.utils.exceptions import (
    APIError,
    APIConnectionError,
    APIAuthenticationError,
    APIRateLimitError,
)
from src.utils.helpers import generate_uuid
from src.utils.date_utils import now_utc
from src.utils.decorators import async_retry


logger = logging.getLogger(__name__)


class OpenAIModel(str, Enum):
    """OpenAI model enumeration."""

    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4 = "gpt-4"
    GPT_35_TURBO = "gpt-3.5-turbo"


class MessageRole(str, Enum):
    """Message role enumeration."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


MODEL_PRICING = {
    OpenAIModel.GPT_4O: {"input": 0.005, "output": 0.015},
    OpenAIModel.GPT_4O_MINI: {"input": 0.00015, "output": 0.0006},
    OpenAIModel.GPT_4_TURBO: {"input": 0.01, "output": 0.03},
    OpenAIModel.GPT_4: {"input": 0.03, "output": 0.06},
    OpenAIModel.GPT_35_TURBO: {"input": 0.0005, "output": 0.0015},
}


class OpenAIConfig(BaseModel):
    """Configuration for OpenAI client."""

    api_key: str = Field(default="")
    organization_id: Optional[str] = None
    base_url: str = Field(default="https://api.openai.com/v1")

    default_model: OpenAIModel = Field(default=OpenAIModel.GPT_4O)
    default_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    default_max_tokens: int = Field(default=1000, ge=1, le=128000)

    timeout_seconds: int = Field(default=60, ge=10, le=300)
    max_retries: int = Field(default=3, ge=1, le=10)
    retry_delay_seconds: float = Field(default=1.0, ge=0.5, le=30.0)

    rate_limit_requests_per_minute: int = Field(default=60, ge=1, le=1000)
    rate_limit_tokens_per_minute: int = Field(default=90000, ge=1000)

    daily_budget_usd: float = Field(default=50.0, ge=0.0)
    monthly_budget_usd: float = Field(default=500.0, ge=0.0)

    enable_caching: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=3600, ge=60, le=86400)


class ChatMessage(BaseModel):
    """Chat message model."""

    role: MessageRole
    content: str
    name: Optional[str] = None
    function_call: Optional[dict] = None
    tool_calls: Optional[list[dict]] = None

    def to_dict(self) -> dict:
        """Convert to API format."""
        msg = {"role": self.role.value, "content": self.content}
        if self.name:
            msg["name"] = self.name
        if self.function_call:
            msg["function_call"] = self.function_call
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg


class ChatCompletion(BaseModel):
    """Chat completion response model."""

    completion_id: str = Field(default_factory=generate_uuid)
    model: str
    content: str = Field(default="")
    finish_reason: str = Field(default="stop")

    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)

    function_call: Optional[dict] = None
    tool_calls: Optional[list[dict]] = None

    cost_usd: float = Field(default=0.0)
    latency_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=now_utc)

    @property
    def has_function_call(self) -> bool:
        """Check if response has function call."""
        return self.function_call is not None

    @property
    def has_tool_calls(self) -> bool:
        """Check if response has tool calls."""
        return self.tool_calls is not None and len(self.tool_calls) > 0


class UsageStats(BaseModel):
    """API usage statistics."""

    total_requests: int = Field(default=0)
    total_tokens: int = Field(default=0)
    total_prompt_tokens: int = Field(default=0)
    total_completion_tokens: int = Field(default=0)
    total_cost_usd: float = Field(default=0.0)

    requests_today: int = Field(default=0)
    tokens_today: int = Field(default=0)
    cost_today_usd: float = Field(default=0.0)

    requests_this_month: int = Field(default=0)
    tokens_this_month: int = Field(default=0)
    cost_this_month_usd: float = Field(default=0.0)

    last_reset_date: datetime = Field(default_factory=now_utc)
    last_request_time: Optional[datetime] = None

    def update(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
    ) -> None:
        """Update usage statistics."""
        self.total_requests += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.total_cost_usd += cost

        self.requests_today += 1
        self.tokens_today += prompt_tokens + completion_tokens
        self.cost_today_usd += cost

        self.requests_this_month += 1
        self.tokens_this_month += prompt_tokens + completion_tokens
        self.cost_this_month_usd += cost

        self.last_request_time = now_utc()

    def reset_daily(self) -> None:
        """Reset daily counters."""
        self.requests_today = 0
        self.tokens_today = 0
        self.cost_today_usd = 0.0
        self.last_reset_date = now_utc()

    def reset_monthly(self) -> None:
        """Reset monthly counters."""
        self.requests_this_month = 0
        self.tokens_this_month = 0
        self.cost_this_month_usd = 0.0


class OpenAIClient:
    """
    OpenAI API client for the trading bot.

    Provides functionality for:
    - Chat completions with GPT-4o
    - Function/tool calling
    - Cost and usage tracking
    - Rate limiting
    - Response caching
    """

    def __init__(
        self,
        config: Optional[OpenAIConfig] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize OpenAIClient.

        Args:
            config: Client configuration
            api_key: OpenAI API key (overrides config)
        """
        self._config = config or OpenAIConfig()

        if api_key:
            self._config.api_key = api_key

        self._client: Optional[httpx.AsyncClient] = None
        self._usage = UsageStats()

        self._cache: dict[str, tuple[ChatCompletion, datetime]] = {}

        self._rate_limit_requests: list[datetime] = []
        self._rate_limit_tokens: list[tuple[datetime, int]] = []

        self._lock = asyncio.Lock()

        logger.info("OpenAIClient initialized")

    @property
    def usage(self) -> UsageStats:
        """Get usage statistics."""
        return self._usage

    @property
    def is_connected(self) -> bool:
        """Check if client is initialized."""
        return self._client is not None

    @property
    def _headers(self) -> dict[str, str]:
        """Get API headers."""
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        if self._config.organization_id:
            headers["OpenAI-Organization"] = self._config.organization_id
        return headers

    async def start(self) -> None:
        """Start the client."""
        self._client = httpx.AsyncClient(
            base_url=self._config.base_url,
            headers=self._headers,
            timeout=self._config.timeout_seconds,
        )
        logger.info("OpenAI client started")

    async def stop(self) -> None:
        """Stop the client."""
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("OpenAI client stopped")

    async def chat_completion(
        self,
        messages: list[ChatMessage],
        model: Optional[OpenAIModel] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        functions: Optional[list[dict]] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[dict] = None,
        use_cache: bool = True,
    ) -> ChatCompletion:
        """
        Create a chat completion.

        Args:
            messages: Chat messages
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            functions: Function definitions (deprecated)
            tools: Tool definitions
            tool_choice: Tool choice strategy
            response_format: Response format specification
            use_cache: Whether to use response cache

        Returns:
            Chat completion response
        """
        model = model or self._config.default_model
        temperature = temperature if temperature is not None else self._config.default_temperature
        max_tokens = max_tokens or self._config.default_max_tokens

        cache_key = self._make_cache_key(messages, model, temperature)
        if use_cache and self._config.enable_caching:
            cached = self._get_cached(cache_key)
            if cached:
                return cached

        if not self._check_budget():
            raise APIError("Daily or monthly budget exceeded")

        await self._check_rate_limit()

        if not self._client:
            await self.start()

        request_body = {
            "model": model.value,
            "messages": [msg.to_dict() for msg in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if functions:
            request_body["functions"] = functions

        if tools:
            request_body["tools"] = tools
            if tool_choice:
                request_body["tool_choice"] = tool_choice

        if response_format:
            request_body["response_format"] = response_format

        start_time = datetime.now()

        try:
            response = await self._make_request(request_body)
            end_time = datetime.now()
            latency_ms = (end_time - start_time).total_seconds() * 1000

            completion = self._parse_response(response, model, latency_ms)

            self._usage.update(
                completion.prompt_tokens,
                completion.completion_tokens,
                completion.cost_usd,
            )

            if use_cache and self._config.enable_caching:
                self._set_cached(cache_key, completion)

            return completion

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise APIAuthenticationError("Invalid OpenAI API key")
            elif e.response.status_code == 429:
                raise APIRateLimitError("OpenAI rate limit exceeded")
            else:
                raise APIError(f"OpenAI API error: {e.response.status_code}")

        except httpx.TimeoutException:
            raise APIConnectionError("OpenAI API timeout")

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise APIError(f"OpenAI API error: {e}")

    @async_retry(max_attempts=3, delay=1.0)
    async def _make_request(self, request_body: dict) -> dict:
        """Make API request with retry."""
        response = await self._client.post(
            "/chat/completions",
            json=request_body,
        )
        response.raise_for_status()
        return response.json()

    def _parse_response(
        self,
        response: dict,
        model: OpenAIModel,
        latency_ms: float,
    ) -> ChatCompletion:
        """Parse API response into ChatCompletion."""
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = response.get("usage", {})

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        cost = self._calculate_cost(model, prompt_tokens, completion_tokens)

        return ChatCompletion(
            completion_id=response.get("id", generate_uuid()),
            model=response.get("model", model.value),
            content=message.get("content", ""),
            finish_reason=choice.get("finish_reason", "stop"),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            function_call=message.get("function_call"),
            tool_calls=message.get("tool_calls"),
            cost_usd=cost,
            latency_ms=latency_ms,
        )

    def _calculate_cost(
        self,
        model: OpenAIModel,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Calculate API cost."""
        pricing = MODEL_PRICING.get(model, {"input": 0.01, "output": 0.03})
        input_cost = (prompt_tokens / 1000) * pricing["input"]
        output_cost = (completion_tokens / 1000) * pricing["output"]
        return round(input_cost + output_cost, 6)

    async def _check_rate_limit(self) -> None:
        """Check and enforce rate limits."""
        now = now_utc()
        minute_ago = now - timedelta(minutes=1)

        async with self._lock:
            self._rate_limit_requests = [
                t for t in self._rate_limit_requests
                if t > minute_ago
            ]

            if len(self._rate_limit_requests) >= self._config.rate_limit_requests_per_minute:
                wait_time = (self._rate_limit_requests[0] - minute_ago).total_seconds()
                logger.warning(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time + 0.1)

            self._rate_limit_requests.append(now)

    def _check_budget(self) -> bool:
        """Check if within budget limits."""
        if self._usage.cost_today_usd >= self._config.daily_budget_usd:
            logger.warning("Daily budget exceeded")
            return False

        if self._usage.cost_this_month_usd >= self._config.monthly_budget_usd:
            logger.warning("Monthly budget exceeded")
            return False

        return True

    def _make_cache_key(
        self,
        messages: list[ChatMessage],
        model: OpenAIModel,
        temperature: float,
    ) -> str:
        """Generate cache key for request."""
        msg_str = json.dumps([msg.to_dict() for msg in messages], sort_keys=True)
        return f"{model.value}:{temperature}:{hash(msg_str)}"

    def _get_cached(self, key: str) -> Optional[ChatCompletion]:
        """Get cached response if valid."""
        if key not in self._cache:
            return None

        completion, cached_at = self._cache[key]
        age = (now_utc() - cached_at).total_seconds()

        if age > self._config.cache_ttl_seconds:
            del self._cache[key]
            return None

        logger.debug("Using cached response")
        return completion

    def _set_cached(self, key: str, completion: ChatCompletion) -> None:
        """Cache response."""
        self._cache[key] = (completion, now_utc())

    def clear_cache(self) -> int:
        """Clear the response cache."""
        count = len(self._cache)
        self._cache.clear()
        return count

    async def simple_chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[OpenAIModel] = None,
    ) -> str:
        """
        Simple chat completion.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            model: Model to use

        Returns:
            Response content
        """
        messages = []

        if system_prompt:
            messages.append(ChatMessage(
                role=MessageRole.SYSTEM,
                content=system_prompt,
            ))

        messages.append(ChatMessage(
            role=MessageRole.USER,
            content=prompt,
        ))

        completion = await self.chat_completion(messages, model=model)
        return completion.content

    async def analyze_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[OpenAIModel] = None,
    ) -> dict:
        """
        Get JSON response from model.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            model: Model to use

        Returns:
            Parsed JSON response
        """
        messages = []

        sys_content = system_prompt or "You are a helpful assistant. Respond only with valid JSON."
        messages.append(ChatMessage(
            role=MessageRole.SYSTEM,
            content=sys_content,
        ))

        messages.append(ChatMessage(
            role=MessageRole.USER,
            content=prompt,
        ))

        completion = await self.chat_completion(
            messages,
            model=model,
            response_format={"type": "json_object"},
        )

        try:
            return json.loads(completion.content)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON: {completion.content[:200]}")
            return {}

    def get_statistics(self) -> dict:
        """Get client statistics."""
        return {
            "total_requests": self._usage.total_requests,
            "total_tokens": self._usage.total_tokens,
            "total_cost_usd": self._usage.total_cost_usd,
            "cost_today_usd": self._usage.cost_today_usd,
            "cost_this_month_usd": self._usage.cost_this_month_usd,
            "cache_size": len(self._cache),
            "daily_budget_remaining": max(
                0,
                self._config.daily_budget_usd - self._usage.cost_today_usd
            ),
            "monthly_budget_remaining": max(
                0,
                self._config.monthly_budget_usd - self._usage.cost_this_month_usd
            ),
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"OpenAIClient(requests={self._usage.total_requests}, "
            f"cost=${self._usage.total_cost_usd:.4f})"
        )
