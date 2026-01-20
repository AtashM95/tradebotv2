"""
OpenAI Client with retry logic, circuit breaker, rate limiting, and token tracking.
~350 lines
"""

import os
import time
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from enum import Enum
import threading

try:
    import openai
    from openai import OpenAI, AsyncOpenAI
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
    )
    import tiktoken
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None
    OpenAI = None
    AsyncOpenAI = None

from config.openai_config import OpenAIConfig

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker pattern implementation."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED
        self._lock = threading.Lock()

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker entering HALF_OPEN state")
                else:
                    raise Exception("Circuit breaker is OPEN - service unavailable")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)

    def _on_success(self):
        """Reset circuit breaker on successful call."""
        with self._lock:
            self.failure_count = 0
            self.state = CircuitState.CLOSED
            if self.state == CircuitState.HALF_OPEN:
                logger.info("Circuit breaker recovered - entering CLOSED state")

    def _on_failure(self):
        """Handle failure and potentially open circuit."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.error(f"Circuit breaker OPEN after {self.failure_count} failures")


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.tokens = requests_per_minute
        self.last_update = time.time()
        self._lock = threading.Lock()

    def acquire(self, blocking: bool = True) -> bool:
        """Acquire a token for rate limiting."""
        while True:
            with self._lock:
                now = time.time()
                time_passed = now - self.last_update
                self.tokens = min(
                    self.requests_per_minute,
                    self.tokens + time_passed * (self.requests_per_minute / 60.0)
                )
                self.last_update = now

                if self.tokens >= 1:
                    self.tokens -= 1
                    return True
                elif not blocking:
                    return False

            if blocking:
                time.sleep(0.1)
            else:
                break

        return False


class TokenCounter:
    """Track token usage across requests."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        self._lock = threading.Lock()

        # Initialize tokenizer if available
        if OPENAI_AVAILABLE:
            try:
                self.encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                self.encoding = tiktoken.get_encoding("cl100k_base")
        else:
            self.encoding = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoding is None:
            # Rough estimate if tiktoken not available
            return len(text) // 4
        return len(self.encoding.encode(text))

    def add_usage(self, prompt_tokens: int, completion_tokens: int, model: str = None):
        """Add token usage and calculate cost."""
        with self._lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens

            # Cost calculation (approximate rates)
            cost = self._calculate_cost(prompt_tokens, completion_tokens, model or self.model)
            self.total_cost += cost

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """Calculate cost based on model pricing."""
        # Pricing as of 2024 (per 1M tokens)
        pricing = {
            "gpt-4o": {"prompt": 2.50, "completion": 10.00},
            "gpt-4o-mini": {"prompt": 0.150, "completion": 0.600},
            "gpt-4-turbo": {"prompt": 10.00, "completion": 30.00},
            "gpt-4": {"prompt": 30.00, "completion": 60.00},
            "gpt-3.5-turbo": {"prompt": 0.50, "completion": 1.50},
        }

        model_key = model.split("-")[0:2]  # Handle versioned models
        model_base = "-".join(model_key) if len(model_key) >= 2 else model

        # Default to gpt-4o-mini pricing
        rates = pricing.get(model_base, pricing["gpt-4o-mini"])

        prompt_cost = (prompt_tokens / 1_000_000) * rates["prompt"]
        completion_cost = (completion_tokens / 1_000_000) * rates["completion"]

        return prompt_cost + completion_cost

    def get_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        with self._lock:
            return {
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
                "total_cost_usd": round(self.total_cost, 4),
                "model": self.model
            }


class OpenAIClient:
    """
    Base OpenAI client with comprehensive error handling and monitoring.

    Features:
    - Exponential backoff retry logic
    - Circuit breaker pattern
    - Rate limiting
    - Token usage tracking
    - Timeout handling
    - Graceful degradation when API key missing
    """

    def __init__(self, config: Optional[OpenAIConfig] = None):
        self.config = config or OpenAIConfig()
        self.api_key = os.getenv('OPENAI_API_KEY', '')

        # Initialize components
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=Exception
        )
        self.rate_limiter = RateLimiter(requests_per_minute=60)
        self.token_counter = TokenCounter(model=self.config.model)

        # Initialize OpenAI client if available and key exists
        self.client: Optional[Any] = None
        self.available = False

        if OPENAI_AVAILABLE and self.api_key:
            try:
                self.client = OpenAI(
                    api_key=self.api_key,
                    timeout=self.config.timeout_s,
                    max_retries=0  # We handle retries ourselves
                )
                self.available = True
                logger.info(f"OpenAI client initialized with model: {self.config.model}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.available = False
        else:
            if not OPENAI_AVAILABLE:
                logger.warning("OpenAI package not available - AI features disabled")
            elif not self.api_key:
                logger.warning("OPENAI_API_KEY not set - AI features disabled")

    def is_available(self) -> bool:
        """Check if OpenAI client is available."""
        return self.available and self.circuit_breaker.state != CircuitState.OPEN

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True
    )
    def _make_request(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """Make request to OpenAI API with retry logic."""
        if not self.is_available():
            raise Exception("OpenAI client not available")

        # Rate limiting
        if not self.rate_limiter.acquire(blocking=True):
            raise Exception("Rate limit exceeded")

        # Make request through circuit breaker
        def _call():
            response = self.client.chat.completions.create(
                model=kwargs.get('model', self.config.model),
                messages=messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1000),
                timeout=self.config.timeout_s
            )
            return response

        response = self.circuit_breaker.call(_call)

        # Track token usage
        if hasattr(response, 'usage'):
            self.token_counter.add_usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                model=kwargs.get('model', self.config.model)
            )

        return response

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Optional[str]:
        """
        Send chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters for the API

        Returns:
            Response content or None if unavailable
        """
        if not self.is_available():
            logger.warning("OpenAI not available - returning None")
            return None

        try:
            response = self._make_request(messages, **kwargs)
            if response and response.choices:
                return response.choices[0].message.content
            return None
        except Exception as e:
            logger.error(f"OpenAI request failed: {e}")
            return None

    def analyze(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Simple analysis method for backward compatibility.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Analysis result or error message
        """
        if not self.is_available():
            return "no-key"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        result = self.chat_completion(messages)
        return result if result else "error"

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        stats = self.token_counter.get_stats()
        stats["circuit_state"] = self.circuit_breaker.state.value
        stats["available"] = self.available
        return stats
