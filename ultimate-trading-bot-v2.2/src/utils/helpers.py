"""
Helper Functions Module for Ultimate Trading Bot v2.2.

This module provides general utility functions used throughout the trading bot.
"""

import hashlib
import json
import os
import re
import secrets
import string
import uuid
from datetime import datetime, date, time, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterable, List, Optional,
    Tuple, Type, TypeVar, Union, overload
)
import logging


logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# STRING UTILITIES
# =============================================================================

def generate_id(prefix: str = "", length: int = 12) -> str:
    """
    Generate a unique identifier.

    Args:
        prefix: Optional prefix for the ID
        length: Length of the random part

    Returns:
        Unique identifier string
    """
    random_part = secrets.token_hex(length // 2)
    if prefix:
        return f"{prefix}_{random_part}"
    return random_part


def generate_uuid() -> str:
    """
    Generate a UUID4 string.

    Returns:
        UUID string
    """
    return str(uuid.uuid4())


def generate_short_uuid(length: int = 8) -> str:
    """
    Generate a short UUID.

    Args:
        length: Length of the short UUID

    Returns:
        Short UUID string
    """
    return uuid.uuid4().hex[:length]


def slugify(text: str) -> str:
    """
    Convert text to a URL-friendly slug.

    Args:
        text: Text to slugify

    Returns:
        Slugified text
    """
    # Convert to lowercase
    text = text.lower()
    # Replace spaces and underscores with hyphens
    text = re.sub(r'[\s_]+', '-', text)
    # Remove non-alphanumeric characters except hyphens
    text = re.sub(r'[^\w\-]', '', text)
    # Remove consecutive hyphens
    text = re.sub(r'-+', '-', text)
    # Strip leading/trailing hyphens
    return text.strip('-')


def truncate(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def camel_to_snake(name: str) -> str:
    """
    Convert camelCase to snake_case.

    Args:
        name: CamelCase string

    Returns:
        snake_case string
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def snake_to_camel(name: str, capitalize_first: bool = False) -> str:
    """
    Convert snake_case to camelCase.

    Args:
        name: snake_case string
        capitalize_first: Whether to capitalize the first letter

    Returns:
        camelCase string
    """
    components = name.split('_')
    if capitalize_first:
        return ''.join(x.title() for x in components)
    return components[0] + ''.join(x.title() for x in components[1:])


def mask_string(
    text: str,
    visible_start: int = 4,
    visible_end: int = 4,
    mask_char: str = "*"
) -> str:
    """
    Mask a string, showing only the start and end.

    Args:
        text: String to mask
        visible_start: Number of visible characters at start
        visible_end: Number of visible characters at end
        mask_char: Character to use for masking

    Returns:
        Masked string
    """
    if len(text) <= visible_start + visible_end:
        return mask_char * len(text)
    return text[:visible_start] + mask_char * (len(text) - visible_start - visible_end) + text[-visible_end:]


# =============================================================================
# NUMBER UTILITIES
# =============================================================================

def round_price(price: float, decimals: int = 2) -> float:
    """
    Round a price to specified decimal places.

    Args:
        price: Price to round
        decimals: Number of decimal places

    Returns:
        Rounded price
    """
    multiplier = 10 ** decimals
    return round(price * multiplier) / multiplier


def round_quantity(quantity: float, decimals: int = 6) -> float:
    """
    Round a quantity to specified decimal places.

    Args:
        quantity: Quantity to round
        decimals: Number of decimal places

    Returns:
        Rounded quantity
    """
    return round(quantity, decimals)


def round_decimal(
    value: Union[float, Decimal],
    places: int = 2,
    rounding: str = ROUND_HALF_UP
) -> Decimal:
    """
    Round a value using Decimal for precision.

    Args:
        value: Value to round
        places: Decimal places
        rounding: Rounding mode

    Returns:
        Rounded Decimal
    """
    d = Decimal(str(value))
    return d.quantize(Decimal(10) ** -places, rounding=rounding)


def format_currency(
    value: float,
    currency: str = "$",
    decimals: int = 2,
    show_sign: bool = False
) -> str:
    """
    Format a value as currency.

    Args:
        value: Value to format
        currency: Currency symbol
        decimals: Decimal places
        show_sign: Whether to show +/- sign

    Returns:
        Formatted currency string
    """
    sign = ""
    if show_sign and value > 0:
        sign = "+"
    elif value < 0:
        sign = "-"
        value = abs(value)

    formatted = f"{value:,.{decimals}f}"
    return f"{sign}{currency}{formatted}"


def format_percent(
    value: float,
    decimals: int = 2,
    show_sign: bool = True
) -> str:
    """
    Format a value as percentage.

    Args:
        value: Value to format (0.1 = 10%)
        decimals: Decimal places
        show_sign: Whether to show +/- sign

    Returns:
        Formatted percentage string
    """
    pct = value * 100
    sign = ""
    if show_sign and pct > 0:
        sign = "+"
    return f"{sign}{pct:.{decimals}f}%"


def format_number(
    value: float,
    decimals: int = 2,
    abbreviate: bool = False
) -> str:
    """
    Format a number with optional abbreviation.

    Args:
        value: Value to format
        decimals: Decimal places
        abbreviate: Whether to abbreviate large numbers

    Returns:
        Formatted number string
    """
    if not abbreviate:
        return f"{value:,.{decimals}f}"

    abs_value = abs(value)
    sign = "-" if value < 0 else ""

    if abs_value >= 1_000_000_000_000:
        return f"{sign}{abs_value / 1_000_000_000_000:.{decimals}f}T"
    elif abs_value >= 1_000_000_000:
        return f"{sign}{abs_value / 1_000_000_000:.{decimals}f}B"
    elif abs_value >= 1_000_000:
        return f"{sign}{abs_value / 1_000_000:.{decimals}f}M"
    elif abs_value >= 1_000:
        return f"{sign}{abs_value / 1_000:.{decimals}f}K"
    return f"{sign}{abs_value:.{decimals}f}"


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp a value between min and max.

    Args:
        value: Value to clamp
        min_value: Minimum value
        max_value: Maximum value

    Returns:
        Clamped value
    """
    return max(min_value, min(value, max_value))


def normalize(value: float, min_value: float, max_value: float) -> float:
    """
    Normalize a value to 0-1 range.

    Args:
        value: Value to normalize
        min_value: Minimum value
        max_value: Maximum value

    Returns:
        Normalized value (0-1)
    """
    if max_value == min_value:
        return 0.5
    return (value - min_value) / (max_value - min_value)


def is_close(a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 0.0) -> bool:
    """
    Check if two floats are approximately equal.

    Args:
        a: First value
        b: Second value
        rel_tol: Relative tolerance
        abs_tol: Absolute tolerance

    Returns:
        True if values are close
    """
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


# =============================================================================
# COLLECTION UTILITIES
# =============================================================================

def chunk_list(items: List[T], chunk_size: int) -> List[List[T]]:
    """
    Split a list into chunks.

    Args:
        items: List to split
        chunk_size: Size of each chunk

    Returns:
        List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def flatten(nested_list: List[List[T]]) -> List[T]:
    """
    Flatten a nested list.

    Args:
        nested_list: Nested list to flatten

    Returns:
        Flattened list
    """
    return [item for sublist in nested_list for item in sublist]


def unique(items: List[T], key: Optional[Callable[[T], Any]] = None) -> List[T]:
    """
    Get unique items from a list while preserving order.

    Args:
        items: List of items
        key: Optional key function

    Returns:
        List of unique items
    """
    seen = set()
    result = []
    for item in items:
        k = key(item) if key else item
        if k not in seen:
            seen.add(k)
            result.append(item)
    return result


def group_by(items: Iterable[T], key: Callable[[T], Any]) -> Dict[Any, List[T]]:
    """
    Group items by a key function.

    Args:
        items: Items to group
        key: Key function

    Returns:
        Dictionary of grouped items
    """
    groups: Dict[Any, List[T]] = {}
    for item in items:
        k = key(item)
        if k not in groups:
            groups[k] = []
        groups[k].append(item)
    return groups


def first(items: Iterable[T], default: Optional[T] = None) -> Optional[T]:
    """
    Get the first item from an iterable.

    Args:
        items: Iterable
        default: Default value if empty

    Returns:
        First item or default
    """
    return next(iter(items), default)


def safe_get(
    data: Dict[str, Any],
    path: str,
    default: Any = None,
    separator: str = "."
) -> Any:
    """
    Safely get a nested value from a dictionary.

    Args:
        data: Dictionary to search
        path: Dot-separated path (e.g., "a.b.c")
        default: Default value if not found
        separator: Path separator

    Returns:
        Value at path or default
    """
    keys = path.split(separator)
    result = data
    for key in keys:
        if isinstance(result, dict) and key in result:
            result = result[key]
        else:
            return default
    return result


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.

    Args:
        base: Base dictionary
        override: Override dictionary

    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# =============================================================================
# FILE UTILITIES
# =============================================================================

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_hash(filepath: Union[str, Path], algorithm: str = "sha256") -> str:
    """
    Calculate the hash of a file.

    Args:
        filepath: Path to file
        algorithm: Hash algorithm

    Returns:
        Hash string
    """
    filepath = Path(filepath)
    hasher = hashlib.new(algorithm)

    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def safe_json_loads(text: str, default: Any = None) -> Any:
    """
    Safely parse JSON string.

    Args:
        text: JSON string
        default: Default value on error

    Returns:
        Parsed value or default
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(data: Any, default: str = "{}") -> str:
    """
    Safely serialize to JSON string.

    Args:
        data: Data to serialize
        default: Default value on error

    Returns:
        JSON string or default
    """
    try:
        return json.dumps(data, default=str)
    except (TypeError, ValueError):
        return default


def read_json_file(filepath: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Read a JSON file.

    Args:
        filepath: Path to file

    Returns:
        Parsed JSON or None
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to read JSON file {filepath}: {e}")
        return None


def write_json_file(
    filepath: Union[str, Path],
    data: Any,
    indent: int = 2
) -> bool:
    """
    Write data to a JSON file.

    Args:
        filepath: Path to file
        data: Data to write
        indent: JSON indentation

    Returns:
        True if successful
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, default=str)
        return True
    except (TypeError, IOError) as e:
        logger.error(f"Failed to write JSON file {filepath}: {e}")
        return False


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def is_valid_symbol(symbol: str) -> bool:
    """
    Check if a symbol is valid.

    Args:
        symbol: Stock symbol

    Returns:
        True if valid
    """
    if not symbol or not isinstance(symbol, str):
        return False
    # Allow 1-10 alphanumeric characters, optionally with dots (e.g., BRK.B)
    return bool(re.match(r'^[A-Z]{1,5}(\.[A-Z])?$', symbol.upper()))


def is_valid_email(email: str) -> bool:
    """
    Check if an email is valid.

    Args:
        email: Email address

    Returns:
        True if valid
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def is_valid_url(url: str) -> bool:
    """
    Check if a URL is valid.

    Args:
        url: URL string

    Returns:
        True if valid
    """
    pattern = r'^https?://[^\s<>"{}|\\^`\[\]]+$'
    return bool(re.match(pattern, url))


# =============================================================================
# HASHING UTILITIES
# =============================================================================

def hash_string(text: str, algorithm: str = "sha256") -> str:
    """
    Hash a string.

    Args:
        text: String to hash
        algorithm: Hash algorithm

    Returns:
        Hash string
    """
    return hashlib.new(algorithm, text.encode()).hexdigest()


def generate_cache_key(*args: Any, prefix: str = "") -> str:
    """
    Generate a cache key from arguments.

    Args:
        *args: Arguments to include in key
        prefix: Key prefix

    Returns:
        Cache key string
    """
    key_parts = [str(arg) for arg in args]
    key_string = ":".join(key_parts)
    key_hash = hashlib.md5(key_string.encode()).hexdigest()[:16]
    return f"{prefix}:{key_hash}" if prefix else key_hash


# =============================================================================
# TYPE CONVERSION UTILITIES
# =============================================================================

def to_bool(value: Any) -> bool:
    """
    Convert a value to boolean.

    Args:
        value: Value to convert

    Returns:
        Boolean value
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    return bool(value)


def to_int(value: Any, default: int = 0) -> int:
    """
    Convert a value to integer.

    Args:
        value: Value to convert
        default: Default value on error

    Returns:
        Integer value
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def to_float(value: Any, default: float = 0.0) -> float:
    """
    Convert a value to float.

    Args:
        value: Value to convert
        default: Default value on error

    Returns:
        Float value
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_list(value: Any) -> List[Any]:
    """
    Convert a value to list.

    Args:
        value: Value to convert

    Returns:
        List value
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set, frozenset)):
        return list(value)
    if isinstance(value, str):
        return [value]
    return [value]


# =============================================================================
# ENVIRONMENT UTILITIES
# =============================================================================

def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get environment variable with default.

    Args:
        key: Environment variable name
        default: Default value

    Returns:
        Environment variable value or default
    """
    return os.environ.get(key, default)


def get_env_bool(key: str, default: bool = False) -> bool:
    """
    Get environment variable as boolean.

    Args:
        key: Environment variable name
        default: Default value

    Returns:
        Boolean value
    """
    value = os.environ.get(key)
    if value is None:
        return default
    return to_bool(value)


def get_env_int(key: str, default: int = 0) -> int:
    """
    Get environment variable as integer.

    Args:
        key: Environment variable name
        default: Default value

    Returns:
        Integer value
    """
    value = os.environ.get(key)
    if value is None:
        return default
    return to_int(value, default)


def get_env_float(key: str, default: float = 0.0) -> float:
    """
    Get environment variable as float.

    Args:
        key: Environment variable name
        default: Default value

    Returns:
        Float value
    """
    value = os.environ.get(key)
    if value is None:
        return default
    return to_float(value, default)
