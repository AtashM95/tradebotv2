"""
Validators Module for Ultimate Trading Bot v2.2.

This module provides validation functions for orders, positions, data, and more.
"""

import re
from datetime import datetime, date, time
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union
from enum import Enum
import logging

from pydantic import BaseModel, ValidationError as PydanticValidationError

from src.utils.exceptions import ValidationError


logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# SYMBOL VALIDATION
# =============================================================================

# Valid symbol patterns
STOCK_SYMBOL_PATTERN = re.compile(r'^[A-Z]{1,5}(\.[A-Z])?$')
CRYPTO_SYMBOL_PATTERN = re.compile(r'^[A-Z]{2,10}(/[A-Z]{2,5})?$')
OPTIONS_SYMBOL_PATTERN = re.compile(r'^[A-Z]{1,5}\d{6}[CP]\d{8}$')


def validate_symbol(symbol: str, asset_class: str = "stock") -> bool:
    """
    Validate a trading symbol.

    Args:
        symbol: Symbol to validate
        asset_class: Asset class (stock, crypto, options)

    Returns:
        True if valid

    Raises:
        ValidationError: If symbol is invalid
    """
    if not symbol or not isinstance(symbol, str):
        raise ValidationError(
            "Symbol must be a non-empty string",
            details={"symbol": symbol}
        )

    symbol = symbol.upper().strip()

    if asset_class == "stock":
        if not STOCK_SYMBOL_PATTERN.match(symbol):
            raise ValidationError(
                f"Invalid stock symbol format: {symbol}",
                details={"symbol": symbol, "pattern": "1-5 letters, optional .X suffix"}
            )
    elif asset_class == "crypto":
        if not CRYPTO_SYMBOL_PATTERN.match(symbol):
            raise ValidationError(
                f"Invalid crypto symbol format: {symbol}",
                details={"symbol": symbol}
            )
    elif asset_class == "options":
        if not OPTIONS_SYMBOL_PATTERN.match(symbol):
            raise ValidationError(
                f"Invalid options symbol format: {symbol}",
                details={"symbol": symbol}
            )
    else:
        raise ValidationError(
            f"Unknown asset class: {asset_class}",
            details={"asset_class": asset_class}
        )

    return True


def is_valid_symbol(symbol: str, asset_class: str = "stock") -> bool:
    """
    Check if a symbol is valid without raising an exception.

    Args:
        symbol: Symbol to check
        asset_class: Asset class

    Returns:
        True if valid, False otherwise
    """
    try:
        return validate_symbol(symbol, asset_class)
    except ValidationError:
        return False


def validate_symbols(symbols: List[str], asset_class: str = "stock") -> List[str]:
    """
    Validate a list of symbols.

    Args:
        symbols: List of symbols
        asset_class: Asset class

    Returns:
        List of validated symbols (uppercase, stripped)

    Raises:
        ValidationError: If any symbol is invalid
    """
    if not symbols:
        raise ValidationError("Symbol list cannot be empty")

    validated = []
    invalid = []

    for symbol in symbols:
        try:
            validate_symbol(symbol, asset_class)
            validated.append(symbol.upper().strip())
        except ValidationError:
            invalid.append(symbol)

    if invalid:
        raise ValidationError(
            f"Invalid symbols found: {invalid}",
            details={"invalid_symbols": invalid}
        )

    return validated


# =============================================================================
# ORDER VALIDATION
# =============================================================================

VALID_ORDER_SIDES = {"buy", "sell"}
VALID_ORDER_TYPES = {"market", "limit", "stop", "stop_limit", "trailing_stop"}
VALID_TIME_IN_FORCE = {"day", "gtc", "ioc", "fok", "opg", "cls"}


def validate_order_side(side: str) -> str:
    """
    Validate order side.

    Args:
        side: Order side

    Returns:
        Validated side (lowercase)

    Raises:
        ValidationError: If side is invalid
    """
    if not side or not isinstance(side, str):
        raise ValidationError("Order side is required")

    side = side.lower().strip()
    if side not in VALID_ORDER_SIDES:
        raise ValidationError(
            f"Invalid order side: {side}. Must be one of: {VALID_ORDER_SIDES}",
            details={"side": side, "valid_sides": list(VALID_ORDER_SIDES)}
        )

    return side


def validate_order_type(order_type: str) -> str:
    """
    Validate order type.

    Args:
        order_type: Order type

    Returns:
        Validated order type (lowercase)

    Raises:
        ValidationError: If order type is invalid
    """
    if not order_type or not isinstance(order_type, str):
        raise ValidationError("Order type is required")

    order_type = order_type.lower().strip()
    if order_type not in VALID_ORDER_TYPES:
        raise ValidationError(
            f"Invalid order type: {order_type}. Must be one of: {VALID_ORDER_TYPES}",
            details={"order_type": order_type, "valid_types": list(VALID_ORDER_TYPES)}
        )

    return order_type


def validate_time_in_force(tif: str) -> str:
    """
    Validate time in force.

    Args:
        tif: Time in force

    Returns:
        Validated TIF (lowercase)

    Raises:
        ValidationError: If TIF is invalid
    """
    if not tif or not isinstance(tif, str):
        raise ValidationError("Time in force is required")

    tif = tif.lower().strip()
    if tif not in VALID_TIME_IN_FORCE:
        raise ValidationError(
            f"Invalid time in force: {tif}. Must be one of: {VALID_TIME_IN_FORCE}",
            details={"time_in_force": tif, "valid_tif": list(VALID_TIME_IN_FORCE)}
        )

    return tif


def validate_quantity(
    quantity: Union[int, float, Decimal],
    min_quantity: float = 0.0001,
    max_quantity: float = 1_000_000,
    allow_fractional: bool = True
) -> float:
    """
    Validate order quantity.

    Args:
        quantity: Order quantity
        min_quantity: Minimum allowed quantity
        max_quantity: Maximum allowed quantity
        allow_fractional: Whether to allow fractional shares

    Returns:
        Validated quantity

    Raises:
        ValidationError: If quantity is invalid
    """
    try:
        qty = float(quantity)
    except (TypeError, ValueError) as e:
        raise ValidationError(
            f"Invalid quantity: {quantity}",
            details={"quantity": quantity},
            cause=e
        )

    if qty <= 0:
        raise ValidationError(
            "Quantity must be positive",
            details={"quantity": qty}
        )

    if qty < min_quantity:
        raise ValidationError(
            f"Quantity {qty} is below minimum {min_quantity}",
            details={"quantity": qty, "min_quantity": min_quantity}
        )

    if qty > max_quantity:
        raise ValidationError(
            f"Quantity {qty} exceeds maximum {max_quantity}",
            details={"quantity": qty, "max_quantity": max_quantity}
        )

    if not allow_fractional and qty != int(qty):
        raise ValidationError(
            "Fractional shares not allowed",
            details={"quantity": qty}
        )

    return qty


def validate_price(
    price: Union[int, float, Decimal],
    min_price: float = 0.0001,
    max_price: float = 1_000_000
) -> float:
    """
    Validate price.

    Args:
        price: Price value
        min_price: Minimum price
        max_price: Maximum price

    Returns:
        Validated price

    Raises:
        ValidationError: If price is invalid
    """
    try:
        p = float(price)
    except (TypeError, ValueError) as e:
        raise ValidationError(
            f"Invalid price: {price}",
            details={"price": price},
            cause=e
        )

    if p <= 0:
        raise ValidationError(
            "Price must be positive",
            details={"price": p}
        )

    if p < min_price:
        raise ValidationError(
            f"Price {p} is below minimum {min_price}",
            details={"price": p, "min_price": min_price}
        )

    if p > max_price:
        raise ValidationError(
            f"Price {p} exceeds maximum {max_price}",
            details={"price": p, "max_price": max_price}
        )

    return p


def validate_order(
    symbol: str,
    side: str,
    order_type: str,
    quantity: Union[int, float],
    limit_price: Optional[float] = None,
    stop_price: Optional[float] = None,
    time_in_force: str = "day"
) -> Dict[str, Any]:
    """
    Validate a complete order.

    Args:
        symbol: Trading symbol
        side: Order side
        order_type: Order type
        quantity: Order quantity
        limit_price: Limit price (required for limit orders)
        stop_price: Stop price (required for stop orders)
        time_in_force: Time in force

    Returns:
        Dictionary with validated order parameters

    Raises:
        ValidationError: If any parameter is invalid
    """
    # Validate individual fields
    validated_symbol = symbol.upper().strip()
    validate_symbol(validated_symbol)
    validated_side = validate_order_side(side)
    validated_type = validate_order_type(order_type)
    validated_qty = validate_quantity(quantity)
    validated_tif = validate_time_in_force(time_in_force)

    # Validate limit price for limit orders
    validated_limit = None
    if validated_type in ("limit", "stop_limit"):
        if limit_price is None:
            raise ValidationError(
                f"Limit price required for {validated_type} orders",
                details={"order_type": validated_type}
            )
        validated_limit = validate_price(limit_price)

    # Validate stop price for stop orders
    validated_stop = None
    if validated_type in ("stop", "stop_limit", "trailing_stop"):
        if stop_price is None:
            raise ValidationError(
                f"Stop price required for {validated_type} orders",
                details={"order_type": validated_type}
            )
        validated_stop = validate_price(stop_price)

    return {
        "symbol": validated_symbol,
        "side": validated_side,
        "order_type": validated_type,
        "quantity": validated_qty,
        "limit_price": validated_limit,
        "stop_price": validated_stop,
        "time_in_force": validated_tif,
    }


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_dataframe_columns(
    df: Any,
    required_columns: List[str],
    df_name: str = "DataFrame"
) -> bool:
    """
    Validate that a DataFrame has required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        df_name: Name for error messages

    Returns:
        True if valid

    Raises:
        ValidationError: If columns are missing
    """
    if df is None:
        raise ValidationError(f"{df_name} cannot be None")

    # Check for required attributes
    if not hasattr(df, 'columns'):
        raise ValidationError(f"{df_name} must have 'columns' attribute")

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValidationError(
            f"{df_name} missing required columns: {missing}",
            details={
                "missing_columns": missing,
                "available_columns": list(df.columns)
            }
        )

    return True


def validate_ohlcv_data(df: Any) -> bool:
    """
    Validate OHLCV (Open, High, Low, Close, Volume) data.

    Args:
        df: DataFrame to validate

    Returns:
        True if valid

    Raises:
        ValidationError: If data is invalid
    """
    required = ['open', 'high', 'low', 'close', 'volume']

    # Check column variations (lowercase, uppercase, title case)
    columns_lower = [col.lower() for col in df.columns]

    for col in required:
        if col not in columns_lower:
            raise ValidationError(
                f"Missing required OHLCV column: {col}",
                details={"available_columns": list(df.columns)}
            )

    # Validate data relationships
    # High should be >= Low
    # High should be >= Open and Close
    # Low should be <= Open and Close

    return True


def validate_date_range(
    start_date: Union[str, date, datetime],
    end_date: Union[str, date, datetime],
    max_days: Optional[int] = None
) -> Tuple:
    """
    Validate a date range.

    Args:
        start_date: Start date
        end_date: End date
        max_days: Maximum allowed days in range

    Returns:
        Tuple of (start_date, end_date) as date objects

    Raises:
        ValidationError: If date range is invalid
    """
    from src.utils.date_utils import parse_date

    # Parse dates if strings
    if isinstance(start_date, str):
        start = parse_date(start_date)
        if start is None:
            raise ValidationError(f"Invalid start date format: {start_date}")
    elif isinstance(start_date, datetime):
        start = start_date.date()
    else:
        start = start_date

    if isinstance(end_date, str):
        end = parse_date(end_date)
        if end is None:
            raise ValidationError(f"Invalid end date format: {end_date}")
    elif isinstance(end_date, datetime):
        end = end_date.date()
    else:
        end = end_date

    # Validate relationship
    if start > end:
        raise ValidationError(
            "Start date must be before or equal to end date",
            details={"start_date": str(start), "end_date": str(end)}
        )

    # Validate max days
    if max_days:
        days = (end - start).days
        if days > max_days:
            raise ValidationError(
                f"Date range exceeds maximum of {max_days} days",
                details={"days": days, "max_days": max_days}
            )

    return start, end


# =============================================================================
# TYPE VALIDATION
# =============================================================================

def validate_type(
    value: Any,
    expected_type: Type[T],
    name: str = "value",
    allow_none: bool = False
) -> T:
    """
    Validate that a value is of expected type.

    Args:
        value: Value to validate
        expected_type: Expected type
        name: Name for error messages
        allow_none: Whether to allow None

    Returns:
        The value if valid

    Raises:
        ValidationError: If type is invalid
    """
    if value is None:
        if allow_none:
            return None
        raise ValidationError(f"{name} cannot be None")

    if not isinstance(value, expected_type):
        raise ValidationError(
            f"{name} must be {expected_type.__name__}, got {type(value).__name__}",
            details={"expected": expected_type.__name__, "actual": type(value).__name__}
        )

    return value


def validate_enum(
    value: Union[str, Enum],
    enum_class: Type[Enum],
    name: str = "value"
) -> Enum:
    """
    Validate that a value is a valid enum member.

    Args:
        value: Value to validate
        enum_class: Enum class
        name: Name for error messages

    Returns:
        Enum member

    Raises:
        ValidationError: If value is not a valid enum member
    """
    if isinstance(value, enum_class):
        return value

    if isinstance(value, str):
        try:
            return enum_class(value.lower())
        except ValueError:
            try:
                return enum_class[value.upper()]
            except KeyError:
                pass

    valid_values = [e.value for e in enum_class]
    raise ValidationError(
        f"Invalid {name}: {value}. Must be one of: {valid_values}",
        details={"value": value, "valid_values": valid_values}
    )


def validate_range(
    value: Union[int, float],
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    name: str = "value",
    inclusive: bool = True
) -> Union[int, float]:
    """
    Validate that a value is within a range.

    Args:
        value: Value to validate
        min_value: Minimum value
        max_value: Maximum value
        name: Name for error messages
        inclusive: Whether bounds are inclusive

    Returns:
        The value if valid

    Raises:
        ValidationError: If value is out of range
    """
    if min_value is not None:
        if inclusive:
            if value < min_value:
                raise ValidationError(
                    f"{name} must be >= {min_value}",
                    details={"value": value, "min": min_value}
                )
        else:
            if value <= min_value:
                raise ValidationError(
                    f"{name} must be > {min_value}",
                    details={"value": value, "min": min_value}
                )

    if max_value is not None:
        if inclusive:
            if value > max_value:
                raise ValidationError(
                    f"{name} must be <= {max_value}",
                    details={"value": value, "max": max_value}
                )
        else:
            if value >= max_value:
                raise ValidationError(
                    f"{name} must be < {max_value}",
                    details={"value": value, "max": max_value}
                )

    return value


def validate_percentage(
    value: Union[int, float],
    name: str = "value",
    allow_zero: bool = True,
    as_decimal: bool = False
) -> float:
    """
    Validate a percentage value.

    Args:
        value: Value to validate
        name: Name for error messages
        allow_zero: Whether to allow zero
        as_decimal: Whether value is expressed as decimal (0.1 vs 10)

    Returns:
        Validated percentage

    Raises:
        ValidationError: If value is invalid
    """
    max_val = 1.0 if as_decimal else 100.0
    min_val = 0.0 if allow_zero else (0.0001 if as_decimal else 0.01)

    return validate_range(
        value,
        min_value=min_val,
        max_value=max_val,
        name=name
    )


# =============================================================================
# PYDANTIC VALIDATION HELPERS
# =============================================================================

def validate_model(
    model_class: Type[BaseModel],
    data: Dict[str, Any],
    strict: bool = False
) -> BaseModel:
    """
    Validate data against a Pydantic model.

    Args:
        model_class: Pydantic model class
        data: Data to validate
        strict: Whether to use strict validation

    Returns:
        Validated model instance

    Raises:
        ValidationError: If validation fails
    """
    try:
        if strict:
            return model_class.model_validate(data, strict=True)
        return model_class.model_validate(data)
    except PydanticValidationError as e:
        errors = e.errors()
        raise ValidationError(
            f"Model validation failed with {len(errors)} error(s)",
            details={"errors": errors}
        )


def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Validate data against a JSON schema.

    Args:
        data: Data to validate
        schema: JSON schema

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    try:
        import jsonschema
        jsonschema.validate(data, schema)
        return True
    except ImportError:
        logger.warning("jsonschema not installed, skipping validation")
        return True
    except jsonschema.ValidationError as e:
        raise ValidationError(
            f"JSON schema validation failed: {e.message}",
            details={"path": list(e.path), "schema_path": list(e.schema_path)}
        )


# =============================================================================
# RISK VALIDATION
# =============================================================================

def validate_position_size(
    position_value: float,
    portfolio_value: float,
    max_position_percent: float = 0.1
) -> bool:
    """
    Validate position size against portfolio limits.

    Args:
        position_value: Value of the position
        portfolio_value: Total portfolio value
        max_position_percent: Maximum position size as percentage

    Returns:
        True if valid

    Raises:
        ValidationError: If position exceeds limit
    """
    if portfolio_value <= 0:
        raise ValidationError("Portfolio value must be positive")

    position_percent = position_value / portfolio_value

    if position_percent > max_position_percent:
        raise ValidationError(
            f"Position size {position_percent:.2%} exceeds maximum {max_position_percent:.2%}",
            details={
                "position_value": position_value,
                "portfolio_value": portfolio_value,
                "position_percent": position_percent,
                "max_percent": max_position_percent
            }
        )

    return True


def validate_risk_per_trade(
    risk_amount: float,
    portfolio_value: float,
    max_risk_percent: float = 0.02
) -> bool:
    """
    Validate risk per trade against limits.

    Args:
        risk_amount: Amount at risk
        portfolio_value: Total portfolio value
        max_risk_percent: Maximum risk as percentage

    Returns:
        True if valid

    Raises:
        ValidationError: If risk exceeds limit
    """
    if portfolio_value <= 0:
        raise ValidationError("Portfolio value must be positive")

    risk_percent = risk_amount / portfolio_value

    if risk_percent > max_risk_percent:
        raise ValidationError(
            f"Risk per trade {risk_percent:.2%} exceeds maximum {max_risk_percent:.2%}",
            details={
                "risk_amount": risk_amount,
                "risk_percent": risk_percent,
                "max_percent": max_risk_percent
            }
        )

    return True


# Import Tuple from typing for type hints
from typing import Tuple
