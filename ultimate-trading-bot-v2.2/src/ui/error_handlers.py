"""
Error Handlers for Ultimate Trading Bot v2.2.

This module provides error handling utilities including:
- HTTP error handlers
- Exception handlers
- Error logging
- Error response formatting
"""

import logging
import traceback
from datetime import datetime, timezone
from typing import Any
from dataclasses import dataclass
from functools import wraps

from flask import Flask, jsonify, request, render_template
from werkzeug.exceptions import HTTPException


logger = logging.getLogger(__name__)


@dataclass
class ErrorInfo:
    """Error information container."""

    code: str
    message: str
    status_code: int
    details: dict[str, Any] | None = None
    traceback: str | None = None
    request_id: str | None = None
    timestamp: str | None = None

    def __post_init__(self) -> None:
        """Set timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self, include_traceback: bool = False) -> dict[str, Any]:
        """
        Convert to dictionary.

        Args:
            include_traceback: Include traceback in response

        Returns:
            Error dictionary
        """
        result = {
            "error": True,
            "code": self.code,
            "message": self.message,
            "status_code": self.status_code,
            "timestamp": self.timestamp,
        }

        if self.details:
            result["details"] = self.details

        if self.request_id:
            result["request_id"] = self.request_id

        if include_traceback and self.traceback:
            result["traceback"] = self.traceback

        return result


class AppError(Exception):
    """Base application error."""

    def __init__(
        self,
        message: str,
        code: str = "app_error",
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize application error.

        Args:
            message: Error message
            code: Error code
            status_code: HTTP status code
            details: Additional details
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details

    def to_error_info(self) -> ErrorInfo:
        """Convert to ErrorInfo."""
        return ErrorInfo(
            code=self.code,
            message=self.message,
            status_code=self.status_code,
            details=self.details,
        )


class ValidationError(AppError):
    """Validation error."""

    def __init__(
        self,
        message: str = "Validation failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize validation error."""
        super().__init__(
            message=message,
            code="validation_error",
            status_code=400,
            details=details,
        )


class AuthenticationError(AppError):
    """Authentication error."""

    def __init__(
        self,
        message: str = "Authentication required",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize authentication error."""
        super().__init__(
            message=message,
            code="authentication_error",
            status_code=401,
            details=details,
        )


class AuthorizationError(AppError):
    """Authorization error."""

    def __init__(
        self,
        message: str = "Permission denied",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize authorization error."""
        super().__init__(
            message=message,
            code="authorization_error",
            status_code=403,
            details=details,
        )


class NotFoundError(AppError):
    """Resource not found error."""

    def __init__(
        self,
        message: str = "Resource not found",
        resource_type: str | None = None,
        resource_id: str | None = None,
    ) -> None:
        """Initialize not found error."""
        details = {}
        if resource_type:
            details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = resource_id

        super().__init__(
            message=message,
            code="not_found",
            status_code=404,
            details=details if details else None,
        )


class ConflictError(AppError):
    """Conflict error."""

    def __init__(
        self,
        message: str = "Resource conflict",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize conflict error."""
        super().__init__(
            message=message,
            code="conflict",
            status_code=409,
            details=details,
        )


class RateLimitError(AppError):
    """Rate limit exceeded error."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int | None = None,
    ) -> None:
        """Initialize rate limit error."""
        details = {}
        if retry_after:
            details["retry_after"] = retry_after

        super().__init__(
            message=message,
            code="rate_limit_exceeded",
            status_code=429,
            details=details if details else None,
        )


class ExternalServiceError(AppError):
    """External service error."""

    def __init__(
        self,
        message: str = "External service unavailable",
        service: str | None = None,
    ) -> None:
        """Initialize external service error."""
        details = {}
        if service:
            details["service"] = service

        super().__init__(
            message=message,
            code="external_service_error",
            status_code=503,
            details=details if details else None,
        )


class TradingError(AppError):
    """Trading-related error."""

    def __init__(
        self,
        message: str,
        order_id: str | None = None,
        symbol: str | None = None,
    ) -> None:
        """Initialize trading error."""
        details = {}
        if order_id:
            details["order_id"] = order_id
        if symbol:
            details["symbol"] = symbol

        super().__init__(
            message=message,
            code="trading_error",
            status_code=400,
            details=details if details else None,
        )


def get_request_id() -> str:
    """Get request ID from request headers."""
    return request.headers.get("X-Request-ID", "unknown")


def is_api_request() -> bool:
    """Check if request is an API request."""
    return (
        request.path.startswith("/api/")
        or request.accept_mimetypes.best == "application/json"
        or request.is_json
    )


def format_error_response(
    error_info: ErrorInfo,
    include_traceback: bool = False,
) -> tuple[dict[str, Any], int]:
    """
    Format error as JSON response.

    Args:
        error_info: Error information
        include_traceback: Include traceback

    Returns:
        Tuple of (response_dict, status_code)
    """
    error_info.request_id = get_request_id()
    return (
        jsonify(error_info.to_dict(include_traceback=include_traceback)),
        error_info.status_code,
    )


def log_error(
    error_info: ErrorInfo,
    exception: Exception | None = None,
) -> None:
    """
    Log error with context.

    Args:
        error_info: Error information
        exception: Original exception
    """
    log_data = {
        "error_code": error_info.code,
        "message": error_info.message,
        "status_code": error_info.status_code,
        "request_id": error_info.request_id or get_request_id(),
        "path": request.path,
        "method": request.method,
        "ip": request.headers.get("X-Forwarded-For", request.remote_addr),
    }

    if error_info.details:
        log_data["details"] = error_info.details

    if error_info.status_code >= 500:
        logger.error(f"Server error: {log_data}", exc_info=exception)
    elif error_info.status_code >= 400:
        logger.warning(f"Client error: {log_data}")
    else:
        logger.info(f"Error response: {log_data}")


def register_error_handlers(
    app: Flask,
    debug: bool = False,
) -> None:
    """
    Register error handlers for Flask app.

    Args:
        app: Flask application
        debug: Debug mode (include traceback)
    """

    @app.errorhandler(ValidationError)
    def handle_validation_error(
        error: ValidationError,
    ) -> tuple[Any, int]:
        """Handle validation errors."""
        error_info = error.to_error_info()
        log_error(error_info, error)

        if is_api_request():
            return format_error_response(error_info, include_traceback=debug)

        return render_template(
            "errors/400.html",
            error=error_info.to_dict(),
        ), 400

    @app.errorhandler(AuthenticationError)
    def handle_authentication_error(
        error: AuthenticationError,
    ) -> tuple[Any, int]:
        """Handle authentication errors."""
        error_info = error.to_error_info()
        log_error(error_info, error)

        if is_api_request():
            return format_error_response(error_info)

        return render_template(
            "errors/401.html",
            error=error_info.to_dict(),
        ), 401

    @app.errorhandler(AuthorizationError)
    def handle_authorization_error(
        error: AuthorizationError,
    ) -> tuple[Any, int]:
        """Handle authorization errors."""
        error_info = error.to_error_info()
        log_error(error_info, error)

        if is_api_request():
            return format_error_response(error_info)

        return render_template(
            "errors/403.html",
            error=error_info.to_dict(),
        ), 403

    @app.errorhandler(NotFoundError)
    def handle_not_found_error(
        error: NotFoundError,
    ) -> tuple[Any, int]:
        """Handle not found errors."""
        error_info = error.to_error_info()
        log_error(error_info, error)

        if is_api_request():
            return format_error_response(error_info)

        return render_template(
            "errors/404.html",
            error=error_info.to_dict(),
        ), 404

    @app.errorhandler(RateLimitError)
    def handle_rate_limit_error(
        error: RateLimitError,
    ) -> tuple[Any, int]:
        """Handle rate limit errors."""
        error_info = error.to_error_info()
        log_error(error_info, error)

        response, status_code = format_error_response(error_info)

        if error.details and "retry_after" in error.details:
            response.headers["Retry-After"] = str(error.details["retry_after"])

        return response, status_code

    @app.errorhandler(AppError)
    def handle_app_error(error: AppError) -> tuple[Any, int]:
        """Handle application errors."""
        error_info = error.to_error_info()
        log_error(error_info, error)

        if is_api_request():
            return format_error_response(error_info, include_traceback=debug)

        return render_template(
            "errors/500.html",
            error=error_info.to_dict(),
        ), error.status_code

    @app.errorhandler(HTTPException)
    def handle_http_exception(error: HTTPException) -> tuple[Any, int]:
        """Handle HTTP exceptions."""
        error_info = ErrorInfo(
            code=f"http_{error.code}",
            message=error.description or str(error),
            status_code=error.code or 500,
        )
        log_error(error_info)

        if is_api_request():
            return format_error_response(error_info)

        template = f"errors/{error.code}.html"
        try:
            return render_template(template, error=error_info.to_dict()), error.code or 500
        except Exception:
            return render_template("errors/500.html", error=error_info.to_dict()), 500

    @app.errorhandler(Exception)
    def handle_generic_exception(error: Exception) -> tuple[Any, int]:
        """Handle uncaught exceptions."""
        error_info = ErrorInfo(
            code="internal_error",
            message="An unexpected error occurred",
            status_code=500,
            traceback=traceback.format_exc() if debug else None,
        )
        log_error(error_info, error)

        if is_api_request():
            return format_error_response(error_info, include_traceback=debug)

        return render_template(
            "errors/500.html",
            error=error_info.to_dict(include_traceback=debug),
        ), 500

    @app.errorhandler(400)
    def handle_bad_request(error: Any) -> tuple[Any, int]:
        """Handle 400 Bad Request."""
        error_info = ErrorInfo(
            code="bad_request",
            message="Bad request",
            status_code=400,
        )

        if is_api_request():
            return format_error_response(error_info)

        return render_template("errors/400.html", error=error_info.to_dict()), 400

    @app.errorhandler(401)
    def handle_unauthorized(error: Any) -> tuple[Any, int]:
        """Handle 401 Unauthorized."""
        error_info = ErrorInfo(
            code="unauthorized",
            message="Authentication required",
            status_code=401,
        )

        if is_api_request():
            return format_error_response(error_info)

        return render_template("errors/401.html", error=error_info.to_dict()), 401

    @app.errorhandler(403)
    def handle_forbidden(error: Any) -> tuple[Any, int]:
        """Handle 403 Forbidden."""
        error_info = ErrorInfo(
            code="forbidden",
            message="Access forbidden",
            status_code=403,
        )

        if is_api_request():
            return format_error_response(error_info)

        return render_template("errors/403.html", error=error_info.to_dict()), 403

    @app.errorhandler(404)
    def handle_not_found(error: Any) -> tuple[Any, int]:
        """Handle 404 Not Found."""
        error_info = ErrorInfo(
            code="not_found",
            message="Resource not found",
            status_code=404,
        )

        if is_api_request():
            return format_error_response(error_info)

        return render_template("errors/404.html", error=error_info.to_dict()), 404

    @app.errorhandler(405)
    def handle_method_not_allowed(error: Any) -> tuple[Any, int]:
        """Handle 405 Method Not Allowed."""
        error_info = ErrorInfo(
            code="method_not_allowed",
            message="Method not allowed",
            status_code=405,
        )

        if is_api_request():
            return format_error_response(error_info)

        return render_template("errors/405.html", error=error_info.to_dict()), 405

    @app.errorhandler(429)
    def handle_too_many_requests(error: Any) -> tuple[Any, int]:
        """Handle 429 Too Many Requests."""
        error_info = ErrorInfo(
            code="too_many_requests",
            message="Too many requests",
            status_code=429,
        )

        response, status_code = format_error_response(error_info)
        response.headers["Retry-After"] = "60"

        return response, status_code

    @app.errorhandler(500)
    def handle_internal_error(error: Any) -> tuple[Any, int]:
        """Handle 500 Internal Server Error."""
        error_info = ErrorInfo(
            code="internal_error",
            message="Internal server error",
            status_code=500,
        )
        log_error(error_info)

        if is_api_request():
            return format_error_response(error_info)

        return render_template("errors/500.html", error=error_info.to_dict()), 500

    @app.errorhandler(502)
    def handle_bad_gateway(error: Any) -> tuple[Any, int]:
        """Handle 502 Bad Gateway."""
        error_info = ErrorInfo(
            code="bad_gateway",
            message="Bad gateway",
            status_code=502,
        )

        if is_api_request():
            return format_error_response(error_info)

        return render_template("errors/502.html", error=error_info.to_dict()), 502

    @app.errorhandler(503)
    def handle_service_unavailable(error: Any) -> tuple[Any, int]:
        """Handle 503 Service Unavailable."""
        error_info = ErrorInfo(
            code="service_unavailable",
            message="Service temporarily unavailable",
            status_code=503,
        )

        if is_api_request():
            return format_error_response(error_info)

        return render_template("errors/503.html", error=error_info.to_dict()), 503

    @app.errorhandler(504)
    def handle_gateway_timeout(error: Any) -> tuple[Any, int]:
        """Handle 504 Gateway Timeout."""
        error_info = ErrorInfo(
            code="gateway_timeout",
            message="Gateway timeout",
            status_code=504,
        )

        if is_api_request():
            return format_error_response(error_info)

        return render_template("errors/504.html", error=error_info.to_dict()), 504


def handle_errors(f: Any) -> Any:
    """
    Decorator to handle errors in route functions.

    Args:
        f: Function to wrap

    Returns:
        Wrapped function
    """
    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return f(*args, **kwargs)
        except AppError:
            raise
        except Exception as e:
            logger.error(f"Unhandled error in {f.__name__}: {e}", exc_info=True)
            raise AppError(
                message="An unexpected error occurred",
                code="internal_error",
                status_code=500,
            ) from e
    return wrapper


# Module version
__version__ = "2.2.0"
