"""
Decorators Module for Ultimate Trading Bot v2.2.

This module provides decorators for logging, caching, retrying, timing, and more.
"""

import asyncio
import functools
import time
import hashlib
import json
import threading
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union, Tuple
from collections import OrderedDict
import logging
import traceback


logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


# =============================================================================
# TIMING DECORATORS
# =============================================================================

def timer(func: F) -> F:
    """
    Decorator to measure and log function execution time.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start
            logger.debug(f"{func.__name__} executed in {elapsed:.4f}s")

    return wrapper  # type: ignore


def async_timer(func: F) -> F:
    """
    Decorator to measure and log async function execution time.

    Args:
        func: Async function to wrap

    Returns:
        Wrapped async function
    """
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start
            logger.debug(f"{func.__name__} executed in {elapsed:.4f}s")

    return wrapper  # type: ignore


def timeout(seconds: float):
    """
    Decorator to add timeout to synchronous functions.

    Args:
        seconds: Timeout in seconds

    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import signal

            def handler(signum: int, frame: Any) -> None:
                raise TimeoutError(f"{func.__name__} timed out after {seconds}s")

            # Set the signal handler
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.setitimer(signal.ITIMER_REAL, seconds)

            try:
                result = func(*args, **kwargs)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)

            return result

        return wrapper  # type: ignore

    return decorator


def async_timeout(seconds: float):
    """
    Decorator to add timeout to async functions.

    Args:
        seconds: Timeout in seconds

    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds
                )
            except asyncio.TimeoutError:
                raise TimeoutError(f"{func.__name__} timed out after {seconds}s")

        return wrapper  # type: ignore

    return decorator


# =============================================================================
# RETRY DECORATORS
# =============================================================================

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Decorator to retry a function on failure.

    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier for delay
        exceptions: Exceptions to catch and retry
        on_retry: Callback function on retry

    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(
                            f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        if on_retry:
                            on_retry(e, attempt)
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )

            raise last_exception  # type: ignore

        return wrapper  # type: ignore

    return decorator


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Decorator to retry an async function on failure.

    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier for delay
        exceptions: Exceptions to catch and retry
        on_retry: Callback function on retry

    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        logger.warning(
                            f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        if on_retry:
                            on_retry(e, attempt)
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )

            raise last_exception  # type: ignore

        return wrapper  # type: ignore

    return decorator


# =============================================================================
# CACHING DECORATORS
# =============================================================================

class LRUCache:
    """Thread-safe LRU cache implementation."""

    def __init__(self, maxsize: int = 128, ttl: Optional[float] = None):
        """
        Initialize LRU cache.

        Args:
            maxsize: Maximum cache size
            ttl: Time-to-live in seconds
        """
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.RLock()

    def _make_key(self, args: tuple, kwargs: dict) -> str:
        """Create cache key from arguments."""
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_str = ":".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str) -> Tuple[bool, Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                return False, None

            value, timestamp = self.cache[key]

            # Check TTL
            if self.ttl and (time.time() - timestamp) > self.ttl:
                del self.cache[key]
                return False, None

            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return True, value

    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = (value, time.time())

            # Evict oldest if over capacity
            while len(self.cache) > self.maxsize:
                self.cache.popitem(last=False)

    def clear(self) -> None:
        """Clear the cache."""
        with self.lock:
            self.cache.clear()


def cache(maxsize: int = 128, ttl: Optional[float] = None):
    """
    Decorator for caching function results.

    Args:
        maxsize: Maximum cache size
        ttl: Time-to-live in seconds

    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        cache_instance = LRUCache(maxsize=maxsize, ttl=ttl)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = cache_instance._make_key(args, kwargs)
            found, value = cache_instance.get(key)

            if found:
                logger.debug(f"Cache hit for {func.__name__}")
                return value

            result = func(*args, **kwargs)
            cache_instance.set(key, result)
            return result

        # Attach cache control methods
        wrapper.cache_clear = cache_instance.clear  # type: ignore
        wrapper.cache = cache_instance  # type: ignore

        return wrapper  # type: ignore

    return decorator


def async_cache(maxsize: int = 128, ttl: Optional[float] = None):
    """
    Decorator for caching async function results.

    Args:
        maxsize: Maximum cache size
        ttl: Time-to-live in seconds

    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        cache_instance = LRUCache(maxsize=maxsize, ttl=ttl)
        locks: Dict[str, asyncio.Lock] = {}

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = cache_instance._make_key(args, kwargs)

            # Get or create lock for this key
            if key not in locks:
                locks[key] = asyncio.Lock()

            async with locks[key]:
                found, value = cache_instance.get(key)
                if found:
                    return value

                result = await func(*args, **kwargs)
                cache_instance.set(key, result)
                return result

        wrapper.cache_clear = cache_instance.clear  # type: ignore
        wrapper.cache = cache_instance  # type: ignore

        return wrapper  # type: ignore

    return decorator


def memoize(func: F) -> F:
    """
    Simple memoization decorator (no TTL, unlimited size).

    Args:
        func: Function to memoize

    Returns:
        Memoized function
    """
    cache_dict: Dict[str, Any] = {}

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key = ":".join(key_parts)

        if key not in cache_dict:
            cache_dict[key] = func(*args, **kwargs)

        return cache_dict[key]

    wrapper.cache_clear = cache_dict.clear  # type: ignore
    return wrapper  # type: ignore


# =============================================================================
# LOGGING DECORATORS
# =============================================================================

def log_call(level: int = logging.DEBUG, include_args: bool = True):
    """
    Decorator to log function calls.

    Args:
        level: Logging level
        include_args: Whether to include arguments in log

    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = func.__name__

            if include_args:
                args_repr = [repr(a)[:100] for a in args]
                kwargs_repr = [f"{k}={repr(v)[:100]}" for k, v in kwargs.items()]
                all_args = ", ".join(args_repr + kwargs_repr)
                logger.log(level, f"Calling {func_name}({all_args})")
            else:
                logger.log(level, f"Calling {func_name}")

            result = func(*args, **kwargs)

            logger.log(level, f"{func_name} returned {repr(result)[:100]}")
            return result

        return wrapper  # type: ignore

    return decorator


def log_exceptions(
    level: int = logging.ERROR,
    reraise: bool = True,
    include_traceback: bool = True
):
    """
    Decorator to log exceptions.

    Args:
        level: Logging level
        reraise: Whether to re-raise the exception
        include_traceback: Whether to include traceback

    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                msg = f"Exception in {func.__name__}: {e}"
                if include_traceback:
                    msg += f"\n{traceback.format_exc()}"
                logger.log(level, msg)

                if reraise:
                    raise

        return wrapper  # type: ignore

    return decorator


def async_log_exceptions(
    level: int = logging.ERROR,
    reraise: bool = True,
    include_traceback: bool = True
):
    """
    Decorator to log exceptions in async functions.

    Args:
        level: Logging level
        reraise: Whether to re-raise the exception
        include_traceback: Whether to include traceback

    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                msg = f"Exception in {func.__name__}: {e}"
                if include_traceback:
                    msg += f"\n{traceback.format_exc()}"
                logger.log(level, msg)

                if reraise:
                    raise

        return wrapper  # type: ignore

    return decorator


# =============================================================================
# RATE LIMITING DECORATORS
# =============================================================================

class RateLimiter:
    """Thread-safe rate limiter."""

    def __init__(self, calls: int, period: float):
        """
        Initialize rate limiter.

        Args:
            calls: Number of calls allowed
            period: Time period in seconds
        """
        self.calls = calls
        self.period = period
        self.timestamps: list = []
        self.lock = threading.Lock()

    def acquire(self) -> float:
        """
        Acquire permission to proceed.

        Returns:
            Wait time in seconds (0 if no wait needed)
        """
        with self.lock:
            now = time.time()
            # Remove old timestamps
            self.timestamps = [t for t in self.timestamps if now - t < self.period]

            if len(self.timestamps) >= self.calls:
                # Calculate wait time
                oldest = self.timestamps[0]
                wait_time = self.period - (now - oldest)
                return max(0, wait_time)

            self.timestamps.append(now)
            return 0


def rate_limit(calls: int, period: float):
    """
    Decorator to rate limit function calls.

    Args:
        calls: Number of calls allowed
        period: Time period in seconds

    Returns:
        Decorator function
    """
    limiter = RateLimiter(calls, period)

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            wait_time = limiter.acquire()
            if wait_time > 0:
                logger.debug(f"Rate limited, waiting {wait_time:.2f}s")
                time.sleep(wait_time)
                limiter.acquire()  # Re-acquire after waiting

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def async_rate_limit(calls: int, period: float):
    """
    Decorator to rate limit async function calls.

    Args:
        calls: Number of calls allowed
        period: Time period in seconds

    Returns:
        Decorator function
    """
    limiter = RateLimiter(calls, period)

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            wait_time = limiter.acquire()
            if wait_time > 0:
                logger.debug(f"Rate limited, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                limiter.acquire()

            return await func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


# =============================================================================
# VALIDATION DECORATORS
# =============================================================================

def validate_args(**validators: Callable[[Any], bool]):
    """
    Decorator to validate function arguments.

    Args:
        **validators: Mapping of argument names to validator functions

    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for arg_name, validator in validators.items():
                if arg_name in bound.arguments:
                    value = bound.arguments[arg_name]
                    if not validator(value):
                        raise ValueError(
                            f"Validation failed for argument '{arg_name}' with value {value}"
                        )

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


# =============================================================================
# SINGLETON DECORATOR
# =============================================================================

def singleton(cls: Type) -> Type:
    """
    Decorator to make a class a singleton.

    Args:
        cls: Class to make singleton

    Returns:
        Singleton class
    """
    instances: Dict[Type, Any] = {}
    lock = threading.Lock()

    @functools.wraps(cls)
    def get_instance(*args: Any, **kwargs: Any) -> Any:
        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance  # type: ignore


# =============================================================================
# DEPRECATION DECORATOR
# =============================================================================

def deprecated(message: str = "", version: str = ""):
    """
    Decorator to mark a function as deprecated.

    Args:
        message: Deprecation message
        version: Version when deprecated

    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import warnings
            msg = f"{func.__name__} is deprecated"
            if version:
                msg += f" as of version {version}"
            if message:
                msg += f": {message}"
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


# =============================================================================
# THREAD SAFETY DECORATORS
# =============================================================================

def synchronized(lock: Optional[threading.Lock] = None):
    """
    Decorator to synchronize function execution.

    Args:
        lock: Lock to use (creates new if None)

    Returns:
        Decorator function
    """
    _lock = lock or threading.Lock()

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with _lock:
                return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def run_in_thread(func: F) -> F:
    """
    Decorator to run a function in a separate thread.

    Args:
        func: Function to run in thread

    Returns:
        Wrapped function that returns a thread
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> threading.Thread:
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
        return thread

    return wrapper  # type: ignore


# =============================================================================
# CIRCUIT BREAKER DECORATOR
# =============================================================================

class CircuitBreaker:
    """Circuit breaker pattern implementation."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        expected_exception: Type[Exception] = Exception
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening
            recovery_timeout: Time before attempting recovery
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half-open
        self.lock = threading.Lock()

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        with self.lock:
            if self.state == "closed":
                return True
            elif self.state == "open":
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = "half-open"
                    return True
                return False
            else:  # half-open
                return True

    def record_success(self) -> None:
        """Record a successful execution."""
        with self.lock:
            self.failures = 0
            self.state = "closed"

    def record_failure(self) -> None:
        """Record a failed execution."""
        with self.lock:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.failure_threshold:
                self.state = "open"


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    expected_exception: Type[Exception] = Exception
):
    """
    Decorator to apply circuit breaker pattern.

    Args:
        failure_threshold: Number of failures before opening
        recovery_timeout: Time before attempting recovery
        expected_exception: Exception type to catch

    Returns:
        Decorator function
    """
    breaker = CircuitBreaker(failure_threshold, recovery_timeout, expected_exception)

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not breaker.can_execute():
                raise RuntimeError(f"Circuit breaker is open for {func.__name__}")

            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except breaker.expected_exception as e:
                breaker.record_failure()
                raise

        return wrapper  # type: ignore

    return decorator
