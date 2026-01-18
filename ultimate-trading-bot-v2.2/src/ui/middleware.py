"""
Middleware Module for Ultimate Trading Bot v2.2.

This module provides Flask middleware including:
- Request/response logging
- Security headers
- Rate limiting
- CORS handling
- Compression
"""

import gzip
import hashlib
import io
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable

from flask import Flask, Request, Response, g, request


logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    requests: int = 100
    period: int = 60  # seconds
    burst: int = 10


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, config: RateLimitConfig | None = None) -> None:
        """Initialize rate limiter."""
        self.config = config or RateLimitConfig()
        self._buckets: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "tokens": self.config.requests,
                "last_update": time.time(),
            }
        )

    def is_allowed(self, key: str) -> tuple[bool, dict[str, Any]]:
        """
        Check if request is allowed.

        Args:
            key: Rate limit key (e.g., IP address)

        Returns:
            Tuple of (allowed, info)
        """
        bucket = self._buckets[key]
        now = time.time()

        # Refill tokens
        time_passed = now - bucket["last_update"]
        refill = time_passed * (self.config.requests / self.config.period)
        bucket["tokens"] = min(
            self.config.requests,
            bucket["tokens"] + refill,
        )
        bucket["last_update"] = now

        # Check if allowed
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True, {
                "remaining": int(bucket["tokens"]),
                "reset": int(now + self.config.period),
                "limit": self.config.requests,
            }

        return False, {
            "remaining": 0,
            "reset": int(now + self.config.period),
            "limit": self.config.requests,
            "retry_after": int(self.config.period - time_passed),
        }

    def get_key(self, request: Request) -> str:
        """Get rate limit key for request."""
        # Use IP address as key
        return request.remote_addr or "unknown"


class RequestLogger:
    """Request/response logging middleware."""

    def __init__(
        self,
        log_requests: bool = True,
        log_responses: bool = False,
        log_body: bool = False,
        exclude_paths: list[str] | None = None,
    ) -> None:
        """Initialize request logger."""
        self.log_requests = log_requests
        self.log_responses = log_responses
        self.log_body = log_body
        self.exclude_paths = exclude_paths or ["/health", "/static"]

    def should_log(self, path: str) -> bool:
        """Check if path should be logged."""
        for exclude in self.exclude_paths:
            if path.startswith(exclude):
                return False
        return True

    def log_request(self, request: Request) -> None:
        """Log incoming request."""
        if not self.log_requests or not self.should_log(request.path):
            return

        log_data = {
            "method": request.method,
            "path": request.path,
            "remote_addr": request.remote_addr,
            "user_agent": request.user_agent.string,
        }

        if self.log_body and request.data:
            try:
                log_data["body"] = request.get_json(silent=True)
            except Exception:
                pass

        logger.info(f"Request: {log_data}")

    def log_response(
        self,
        request: Request,
        response: Response,
        duration: float,
    ) -> None:
        """Log response."""
        if not self.log_responses or not self.should_log(request.path):
            return

        log_data = {
            "method": request.method,
            "path": request.path,
            "status": response.status_code,
            "duration_ms": round(duration * 1000, 2),
            "size": response.content_length,
        }

        logger.info(f"Response: {log_data}")


class SecurityHeaders:
    """Security headers middleware."""

    DEFAULT_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "SAMEORIGIN",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    }

    def __init__(
        self,
        headers: dict[str, str] | None = None,
        csp: dict[str, str] | None = None,
        hsts: bool = True,
        hsts_max_age: int = 31536000,
    ) -> None:
        """Initialize security headers."""
        self.headers = {**self.DEFAULT_HEADERS, **(headers or {})}
        self.csp = csp
        self.hsts = hsts
        self.hsts_max_age = hsts_max_age

    def apply_headers(self, response: Response) -> Response:
        """Apply security headers to response."""
        for header, value in self.headers.items():
            response.headers[header] = value

        # Content Security Policy
        if self.csp:
            csp_value = "; ".join(f"{k} {v}" for k, v in self.csp.items())
            response.headers["Content-Security-Policy"] = csp_value

        # HSTS
        if self.hsts:
            response.headers["Strict-Transport-Security"] = (
                f"max-age={self.hsts_max_age}; includeSubDomains"
            )

        return response


class CORSMiddleware:
    """CORS handling middleware."""

    def __init__(
        self,
        origins: list[str] | str = "*",
        methods: list[str] | None = None,
        headers: list[str] | None = None,
        expose_headers: list[str] | None = None,
        supports_credentials: bool = False,
        max_age: int = 86400,
    ) -> None:
        """Initialize CORS middleware."""
        self.origins = origins
        self.methods = methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.headers = headers or ["Content-Type", "Authorization"]
        self.expose_headers = expose_headers or []
        self.supports_credentials = supports_credentials
        self.max_age = max_age

    def is_origin_allowed(self, origin: str | None) -> bool:
        """Check if origin is allowed."""
        if not origin:
            return False

        if self.origins == "*":
            return True

        if isinstance(self.origins, list):
            return origin in self.origins

        return origin == self.origins

    def apply_cors(self, request: Request, response: Response) -> Response:
        """Apply CORS headers to response."""
        origin = request.headers.get("Origin")

        if not self.is_origin_allowed(origin):
            return response

        # Set allowed origin
        if self.origins == "*" and not self.supports_credentials:
            response.headers["Access-Control-Allow-Origin"] = "*"
        else:
            response.headers["Access-Control-Allow-Origin"] = origin or ""

        # Credentials
        if self.supports_credentials:
            response.headers["Access-Control-Allow-Credentials"] = "true"

        # Preflight response headers
        if request.method == "OPTIONS":
            response.headers["Access-Control-Allow-Methods"] = ", ".join(self.methods)
            response.headers["Access-Control-Allow-Headers"] = ", ".join(self.headers)
            response.headers["Access-Control-Max-Age"] = str(self.max_age)

        # Expose headers
        if self.expose_headers:
            response.headers["Access-Control-Expose-Headers"] = ", ".join(
                self.expose_headers
            )

        return response


class CompressionMiddleware:
    """Response compression middleware."""

    def __init__(
        self,
        min_size: int = 500,
        level: int = 6,
        content_types: list[str] | None = None,
    ) -> None:
        """Initialize compression middleware."""
        self.min_size = min_size
        self.level = level
        self.content_types = content_types or [
            "text/html",
            "text/css",
            "text/javascript",
            "application/javascript",
            "application/json",
            "application/xml",
        ]

    def should_compress(self, request: Request, response: Response) -> bool:
        """Check if response should be compressed."""
        # Check if client accepts gzip
        accept_encoding = request.headers.get("Accept-Encoding", "")
        if "gzip" not in accept_encoding:
            return False

        # Check content type
        content_type = response.content_type or ""
        if not any(ct in content_type for ct in self.content_types):
            return False

        # Check content length
        content_length = response.content_length or len(response.get_data())
        if content_length < self.min_size:
            return False

        # Don't compress already compressed responses
        if response.headers.get("Content-Encoding"):
            return False

        return True

    def compress(self, response: Response) -> Response:
        """Compress response body."""
        data = response.get_data()

        # Compress using gzip
        buffer = io.BytesIO()
        with gzip.GzipFile(
            mode="wb",
            fileobj=buffer,
            compresslevel=self.level,
        ) as f:
            f.write(data)

        compressed = buffer.getvalue()

        # Only use compressed if smaller
        if len(compressed) < len(data):
            response.set_data(compressed)
            response.headers["Content-Encoding"] = "gzip"
            response.headers["Content-Length"] = str(len(compressed))

        return response


class ETagMiddleware:
    """ETag caching middleware."""

    def __init__(self, weak: bool = True) -> None:
        """Initialize ETag middleware."""
        self.weak = weak

    def generate_etag(self, data: bytes) -> str:
        """Generate ETag for data."""
        hash_value = hashlib.md5(data).hexdigest()
        if self.weak:
            return f'W/"{hash_value}"'
        return f'"{hash_value}"'

    def apply_etag(self, request: Request, response: Response) -> Response:
        """Apply ETag to response."""
        # Only for successful GET requests
        if request.method != "GET" or response.status_code != 200:
            return response

        # Generate ETag
        data = response.get_data()
        etag = self.generate_etag(data)
        response.headers["ETag"] = etag

        # Check If-None-Match
        if_none_match = request.headers.get("If-None-Match")
        if if_none_match and if_none_match == etag:
            response.status_code = 304
            response.set_data(b"")

        return response


def setup_middleware(app: Flask, config: dict[str, Any] | None = None) -> None:
    """
    Set up all middleware for Flask app.

    Args:
        app: Flask application
        config: Middleware configuration
    """
    config = config or {}

    # Rate limiting
    rate_limiter = RateLimiter(
        RateLimitConfig(**config.get("rate_limit", {}))
    )

    # Request logging
    request_logger = RequestLogger(**config.get("logging", {}))

    # Security headers
    security_headers = SecurityHeaders(**config.get("security", {}))

    # CORS
    cors = CORSMiddleware(**config.get("cors", {}))

    # Compression
    compression = CompressionMiddleware(**config.get("compression", {}))

    # ETag
    etag = ETagMiddleware(**config.get("etag", {}))

    @app.before_request
    def before_request() -> Response | None:
        """Before request hook."""
        g.request_start_time = time.time()

        # Rate limiting
        if config.get("rate_limit", {}).get("enabled", True):
            key = rate_limiter.get_key(request)
            allowed, info = rate_limiter.is_allowed(key)

            if not allowed:
                from flask import jsonify
                response = jsonify({
                    "error": "Rate limit exceeded",
                    "retry_after": info.get("retry_after"),
                })
                response.status_code = 429
                response.headers["Retry-After"] = str(info.get("retry_after", 60))
                response.headers["X-RateLimit-Limit"] = str(info["limit"])
                response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
                response.headers["X-RateLimit-Reset"] = str(info["reset"])
                return response

            g.rate_limit_info = info

        # Log request
        request_logger.log_request(request)

        return None

    @app.after_request
    def after_request(response: Response) -> Response:
        """After request hook."""
        # Calculate duration
        duration = time.time() - g.get("request_start_time", time.time())

        # Apply security headers
        response = security_headers.apply_headers(response)

        # Apply CORS
        response = cors.apply_cors(request, response)

        # Apply compression
        if compression.should_compress(request, response):
            response = compression.compress(response)

        # Apply ETag
        response = etag.apply_etag(request, response)

        # Add rate limit headers
        if hasattr(g, "rate_limit_info"):
            info = g.rate_limit_info
            response.headers["X-RateLimit-Limit"] = str(info["limit"])
            response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
            response.headers["X-RateLimit-Reset"] = str(info["reset"])

        # Log response
        request_logger.log_response(request, response, duration)

        return response

    logger.info("Middleware configured")


def rate_limit(
    requests: int = 10,
    period: int = 60,
    key_func: Callable[[Request], str] | None = None,
) -> Callable:
    """
    Rate limiting decorator.

    Args:
        requests: Max requests per period
        period: Period in seconds
        key_func: Function to get rate limit key

    Returns:
        Decorator function
    """
    limiter = RateLimiter(RateLimitConfig(requests=requests, period=period))

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args: Any, **kwargs: Any) -> Any:
            if key_func:
                key = key_func(request)
            else:
                key = limiter.get_key(request)

            allowed, info = limiter.is_allowed(key)

            if not allowed:
                from flask import jsonify
                response = jsonify({
                    "error": "Rate limit exceeded",
                    "retry_after": info.get("retry_after"),
                })
                response.status_code = 429
                return response

            return f(*args, **kwargs)

        return decorated_function

    return decorator


# Module version
__version__ = "2.2.0"
