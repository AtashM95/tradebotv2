"""
UI Utilities for Ultimate Trading Bot v2.2.

This module provides utility functions for:
- Response formatting
- Pagination
- Sorting and filtering
- Error handling
- Request helpers
"""

import logging
import functools
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, TypeVar
from dataclasses import dataclass, field
import json
import hashlib

from flask import request, jsonify, Response, g


logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class PaginationParams:
    """Pagination parameters."""

    page: int = 1
    per_page: int = 20
    offset: int = 0

    @classmethod
    def from_request(
        cls,
        default_per_page: int = 20,
        max_per_page: int = 100,
    ) -> "PaginationParams":
        """
        Create from request parameters.

        Args:
            default_per_page: Default items per page
            max_per_page: Maximum items per page

        Returns:
            PaginationParams instance
        """
        page = request.args.get("page", 1, type=int)
        per_page = request.args.get("per_page", default_per_page, type=int)
        offset = request.args.get("offset", type=int)

        # Validate
        page = max(1, page)
        per_page = min(max(1, per_page), max_per_page)

        if offset is None:
            offset = (page - 1) * per_page

        return cls(page=page, per_page=per_page, offset=offset)


@dataclass
class SortParams:
    """Sort parameters."""

    sort_by: str = "created_at"
    sort_order: str = "desc"

    @classmethod
    def from_request(
        cls,
        default_sort_by: str = "created_at",
        allowed_fields: list[str] | None = None,
    ) -> "SortParams":
        """
        Create from request parameters.

        Args:
            default_sort_by: Default sort field
            allowed_fields: Allowed sort fields

        Returns:
            SortParams instance
        """
        sort_by = request.args.get("sort_by", default_sort_by)
        sort_order = request.args.get("sort_order", "desc")

        # Validate sort_by
        if allowed_fields and sort_by not in allowed_fields:
            sort_by = default_sort_by

        # Validate sort_order
        if sort_order not in ["asc", "desc"]:
            sort_order = "desc"

        return cls(sort_by=sort_by, sort_order=sort_order)


@dataclass
class FilterParams:
    """Filter parameters."""

    filters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_request(
        cls,
        filter_fields: list[str] | None = None,
    ) -> "FilterParams":
        """
        Create from request parameters.

        Args:
            filter_fields: Fields to extract from request

        Returns:
            FilterParams instance
        """
        filters = {}

        if filter_fields:
            for field_name in filter_fields:
                value = request.args.get(field_name)
                if value is not None:
                    filters[field_name] = value

        return cls(filters=filters)


def paginate_list(
    items: list[Any],
    pagination: PaginationParams,
) -> dict[str, Any]:
    """
    Paginate a list of items.

    Args:
        items: List to paginate
        pagination: Pagination parameters

    Returns:
        Paginated result dict
    """
    total = len(items)
    start = pagination.offset
    end = start + pagination.per_page

    paginated = items[start:end]

    return {
        "data": paginated,
        "pagination": {
            "page": pagination.page,
            "per_page": pagination.per_page,
            "total": total,
            "total_pages": (total + pagination.per_page - 1) // pagination.per_page,
            "has_next": end < total,
            "has_prev": start > 0,
        },
    }


def sort_list(
    items: list[Any],
    sort: SortParams,
) -> list[Any]:
    """
    Sort a list of items.

    Args:
        items: List to sort
        sort: Sort parameters

    Returns:
        Sorted list
    """
    if not items:
        return items

    reverse = sort.sort_order == "desc"

    try:
        if isinstance(items[0], dict):
            return sorted(
                items,
                key=lambda x: x.get(sort.sort_by, ""),
                reverse=reverse,
            )
        else:
            return sorted(
                items,
                key=lambda x: getattr(x, sort.sort_by, ""),
                reverse=reverse,
            )
    except (TypeError, AttributeError):
        return items


def filter_list(
    items: list[Any],
    filters: FilterParams,
) -> list[Any]:
    """
    Filter a list of items.

    Args:
        items: List to filter
        filters: Filter parameters

    Returns:
        Filtered list
    """
    if not items or not filters.filters:
        return items

    result = []
    for item in items:
        match = True
        for field_name, value in filters.filters.items():
            if isinstance(item, dict):
                item_value = item.get(field_name)
            else:
                item_value = getattr(item, field_name, None)

            if item_value != value:
                match = False
                break

        if match:
            result.append(item)

    return result


def format_response(
    data: Any = None,
    message: str | None = None,
    success: bool = True,
    status_code: int = 200,
    **kwargs: Any,
) -> tuple[Response, int]:
    """
    Format a standard API response.

    Args:
        data: Response data
        message: Response message
        success: Success flag
        status_code: HTTP status code
        **kwargs: Additional response fields

    Returns:
        Tuple of (response, status_code)
    """
    response = {
        "success": success,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if message:
        response["message"] = message

    if data is not None:
        response["data"] = data

    response.update(kwargs)

    return jsonify(response), status_code


def format_error(
    message: str,
    error_code: str | None = None,
    details: dict[str, Any] | None = None,
    status_code: int = 400,
) -> tuple[Response, int]:
    """
    Format an error response.

    Args:
        message: Error message
        error_code: Error code
        details: Error details
        status_code: HTTP status code

    Returns:
        Tuple of (response, status_code)
    """
    response = {
        "success": False,
        "error": True,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if error_code:
        response["code"] = error_code

    if details:
        response["details"] = details

    return jsonify(response), status_code


def require_json(f: F) -> F:
    """
    Decorator to require JSON content type.

    Args:
        f: Function to wrap

    Returns:
        Wrapped function
    """
    @functools.wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not request.is_json:
            return format_error(
                message="Content-Type must be application/json",
                error_code="invalid_content_type",
                status_code=415,
            )
        return f(*args, **kwargs)
    return wrapper  # type: ignore


def require_auth(f: F) -> F:
    """
    Decorator to require authentication.

    Args:
        f: Function to wrap

    Returns:
        Wrapped function
    """
    @functools.wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        auth_header = request.headers.get("Authorization", "")

        if not auth_header.startswith("Bearer "):
            return format_error(
                message="Authorization required",
                error_code="unauthorized",
                status_code=401,
            )

        token = auth_header[7:]
        auth_manager = getattr(g, "auth_manager", None)

        if auth_manager:
            session = auth_manager.validate_session(token)
            if not session:
                return format_error(
                    message="Invalid or expired token",
                    error_code="invalid_token",
                    status_code=401,
                )
            g.current_user_id = session.user_id
        else:
            # Demo mode
            g.current_user_id = "demo-user"

        return f(*args, **kwargs)
    return wrapper  # type: ignore


def rate_limit(
    requests_per_minute: int = 60,
    key_func: Callable[[], str] | None = None,
) -> Callable[[F], F]:
    """
    Decorator for rate limiting.

    Args:
        requests_per_minute: Max requests per minute
        key_func: Function to get rate limit key

    Returns:
        Decorator function
    """
    def decorator(f: F) -> F:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get rate limit key
            if key_func:
                key = key_func()
            else:
                key = get_client_ip()

            rate_limiter = getattr(g, "rate_limiter", None)
            if rate_limiter:
                allowed, retry_after = rate_limiter.check(key, requests_per_minute)
                if not allowed:
                    response, _ = format_error(
                        message="Rate limit exceeded",
                        error_code="rate_limited",
                        status_code=429,
                    )
                    response.headers["Retry-After"] = str(retry_after)
                    return response, 429

            return f(*args, **kwargs)
        return wrapper  # type: ignore
    return decorator


def get_client_ip() -> str:
    """
    Get client IP address from request.

    Returns:
        Client IP address
    """
    if request.headers.get("X-Forwarded-For"):
        return request.headers["X-Forwarded-For"].split(",")[0].strip()
    if request.headers.get("X-Real-IP"):
        return request.headers["X-Real-IP"]
    return request.remote_addr or "unknown"


def get_request_id() -> str:
    """
    Get or generate request ID.

    Returns:
        Request ID
    """
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = hashlib.md5(
            f"{datetime.now().timestamp()}{get_client_ip()}".encode()
        ).hexdigest()[:16]
    return request_id


def get_user_agent() -> str:
    """
    Get user agent from request.

    Returns:
        User agent string
    """
    return request.headers.get("User-Agent", "unknown")[:256]


def parse_date_range(
    start_param: str = "start_date",
    end_param: str = "end_date",
    default_days: int = 30,
) -> tuple[datetime, datetime]:
    """
    Parse date range from request parameters.

    Args:
        start_param: Start date parameter name
        end_param: End date parameter name
        default_days: Default range in days

    Returns:
        Tuple of (start_date, end_date)
    """
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=default_days)

    start_str = request.args.get(start_param)
    end_str = request.args.get(end_param)

    if start_str:
        try:
            start_date = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
        except ValueError:
            pass

    if end_str:
        try:
            end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
        except ValueError:
            pass

    return start_date, end_date


def format_currency(
    value: float | int,
    currency: str = "USD",
    decimal_places: int = 2,
) -> str:
    """
    Format value as currency.

    Args:
        value: Numeric value
        currency: Currency code
        decimal_places: Decimal places

    Returns:
        Formatted currency string
    """
    symbols = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥",
    }

    symbol = symbols.get(currency, currency + " ")

    if value < 0:
        return f"-{symbol}{abs(value):,.{decimal_places}f}"
    return f"{symbol}{value:,.{decimal_places}f}"


def format_percent(
    value: float,
    decimal_places: int = 2,
    include_sign: bool = True,
) -> str:
    """
    Format value as percentage.

    Args:
        value: Numeric value (0.1 = 10%)
        decimal_places: Decimal places
        include_sign: Include + for positive

    Returns:
        Formatted percentage string
    """
    percent = value * 100

    if include_sign and percent > 0:
        return f"+{percent:.{decimal_places}f}%"
    return f"{percent:.{decimal_places}f}%"


def format_number(
    value: float | int,
    decimal_places: int = 2,
    abbreviate: bool = False,
) -> str:
    """
    Format number with commas and optional abbreviation.

    Args:
        value: Numeric value
        decimal_places: Decimal places
        abbreviate: Abbreviate large numbers

    Returns:
        Formatted number string
    """
    if abbreviate:
        if abs(value) >= 1_000_000_000:
            return f"{value / 1_000_000_000:.1f}B"
        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        if abs(value) >= 1_000:
            return f"{value / 1_000:.1f}K"

    if isinstance(value, int) or decimal_places == 0:
        return f"{int(value):,}"
    return f"{value:,.{decimal_places}f}"


def time_ago(dt: datetime) -> str:
    """
    Format datetime as time ago string.

    Args:
        dt: Datetime to format

    Returns:
        Time ago string
    """
    now = datetime.now(timezone.utc)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    diff = now - dt
    seconds = diff.total_seconds()

    if seconds < 60:
        return "just now"
    if seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes}m ago"
    if seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours}h ago"
    if seconds < 604800:
        days = int(seconds / 86400)
        return f"{days}d ago"
    if seconds < 2592000:
        weeks = int(seconds / 604800)
        return f"{weeks}w ago"

    return dt.strftime("%b %d, %Y")


def safe_json_loads(
    data: str | bytes,
    default: Any = None,
) -> Any:
    """
    Safely parse JSON.

    Args:
        data: JSON string or bytes
        default: Default value on error

    Returns:
        Parsed value or default
    """
    try:
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return default


def truncate_string(
    text: str,
    max_length: int = 100,
    suffix: str = "...",
) -> str:
    """
    Truncate string to max length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add

    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def generate_etag(data: Any) -> str:
    """
    Generate ETag for data.

    Args:
        data: Data to hash

    Returns:
        ETag string
    """
    if isinstance(data, (dict, list)):
        content = json.dumps(data, sort_keys=True)
    else:
        content = str(data)

    return hashlib.md5(content.encode()).hexdigest()


def check_etag(data: Any) -> bool:
    """
    Check if client's ETag matches.

    Args:
        data: Data to compare

    Returns:
        True if ETags match
    """
    client_etag = request.headers.get("If-None-Match")
    if not client_etag:
        return False

    server_etag = generate_etag(data)
    return client_etag == server_etag or client_etag == f'"{server_etag}"'


def get_request_data() -> dict[str, Any]:
    """
    Get request data from JSON or form.

    Returns:
        Request data dict
    """
    if request.is_json:
        return request.get_json() or {}
    return request.form.to_dict()


def log_request(
    level: str = "info",
    include_body: bool = False,
) -> Callable[[F], F]:
    """
    Decorator to log requests.

    Args:
        level: Log level
        include_body: Include request body

    Returns:
        Decorator function
    """
    def decorator(f: F) -> F:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = datetime.now()

            # Log request
            log_data = {
                "method": request.method,
                "path": request.path,
                "ip": get_client_ip(),
                "request_id": get_request_id(),
            }

            if include_body and request.is_json:
                log_data["body"] = request.get_json()

            log_func = getattr(logger, level)
            log_func(f"Request: {log_data}")

            # Execute function
            result = f(*args, **kwargs)

            # Log response
            duration = (datetime.now() - start_time).total_seconds() * 1000
            status = result[1] if isinstance(result, tuple) else 200

            log_func(f"Response: status={status} duration={duration:.2f}ms")

            return result
        return wrapper  # type: ignore
    return decorator


# Module version
__version__ = "2.2.0"
