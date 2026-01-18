"""
Validators for Ultimate Trading Bot v2.2.

This module provides validation utilities for:
- Form validation
- API request validation
- Data sanitization
- Custom validators
"""

import logging
import re
from datetime import datetime, date
from decimal import Decimal, InvalidOperation
from typing import Any, Callable
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Validation error details."""

    field: str
    message: str
    code: str = "invalid"
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "field": self.field,
            "message": self.message,
            "code": self.code,
        }


@dataclass
class ValidationResult:
    """Result of validation."""

    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    cleaned_data: dict[str, Any] = field(default_factory=dict)

    def add_error(
        self,
        field: str,
        message: str,
        code: str = "invalid",
        value: Any = None,
    ) -> None:
        """Add an error to the result."""
        self.errors.append(ValidationError(field, message, code, value))
        self.is_valid = False

    def get_errors_dict(self) -> dict[str, list[str]]:
        """Get errors as dictionary."""
        errors_dict: dict[str, list[str]] = {}
        for error in self.errors:
            if error.field not in errors_dict:
                errors_dict[error.field] = []
            errors_dict[error.field].append(error.message)
        return errors_dict


class Validator:
    """Base validator class."""

    def __init__(
        self,
        message: str | None = None,
        code: str = "invalid",
    ) -> None:
        """
        Initialize validator.

        Args:
            message: Error message
            code: Error code
        """
        self.message = message or "Invalid value"
        self.code = code

    def __call__(self, value: Any) -> tuple[bool, str]:
        """
        Validate value.

        Args:
            value: Value to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        raise NotImplementedError


class RequiredValidator(Validator):
    """Validator for required fields."""

    def __init__(self, message: str | None = None) -> None:
        """Initialize required validator."""
        super().__init__(
            message=message or "This field is required",
            code="required",
        )

    def __call__(self, value: Any) -> tuple[bool, str]:
        """Validate that value is not empty."""
        if value is None:
            return False, self.message
        if isinstance(value, str) and not value.strip():
            return False, self.message
        if isinstance(value, (list, dict)) and len(value) == 0:
            return False, self.message
        return True, ""


class LengthValidator(Validator):
    """Validator for string/list length."""

    def __init__(
        self,
        min_length: int | None = None,
        max_length: int | None = None,
        message: str | None = None,
    ) -> None:
        """
        Initialize length validator.

        Args:
            min_length: Minimum length
            max_length: Maximum length
            message: Error message
        """
        super().__init__(message=message, code="length")
        self.min_length = min_length
        self.max_length = max_length

    def __call__(self, value: Any) -> tuple[bool, str]:
        """Validate length."""
        if value is None:
            return True, ""

        length = len(value)

        if self.min_length is not None and length < self.min_length:
            msg = self.message or f"Minimum length is {self.min_length}"
            return False, msg

        if self.max_length is not None and length > self.max_length:
            msg = self.message or f"Maximum length is {self.max_length}"
            return False, msg

        return True, ""


class RangeValidator(Validator):
    """Validator for numeric range."""

    def __init__(
        self,
        min_value: float | int | None = None,
        max_value: float | int | None = None,
        message: str | None = None,
    ) -> None:
        """
        Initialize range validator.

        Args:
            min_value: Minimum value
            max_value: Maximum value
            message: Error message
        """
        super().__init__(message=message, code="range")
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, value: Any) -> tuple[bool, str]:
        """Validate range."""
        if value is None:
            return True, ""

        try:
            num_value = float(value)
        except (ValueError, TypeError):
            return False, "Must be a number"

        if self.min_value is not None and num_value < self.min_value:
            msg = self.message or f"Minimum value is {self.min_value}"
            return False, msg

        if self.max_value is not None and num_value > self.max_value:
            msg = self.message or f"Maximum value is {self.max_value}"
            return False, msg

        return True, ""


class RegexValidator(Validator):
    """Validator using regex pattern."""

    def __init__(
        self,
        pattern: str,
        flags: int = 0,
        message: str | None = None,
    ) -> None:
        """
        Initialize regex validator.

        Args:
            pattern: Regex pattern
            flags: Regex flags
            message: Error message
        """
        super().__init__(message=message or "Invalid format", code="regex")
        self.pattern = re.compile(pattern, flags)

    def __call__(self, value: Any) -> tuple[bool, str]:
        """Validate against pattern."""
        if value is None:
            return True, ""

        if not isinstance(value, str):
            value = str(value)

        if not self.pattern.match(value):
            return False, self.message

        return True, ""


class EmailValidator(Validator):
    """Validator for email addresses."""

    EMAIL_PATTERN = re.compile(
        r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    )

    def __init__(self, message: str | None = None) -> None:
        """Initialize email validator."""
        super().__init__(
            message=message or "Invalid email address",
            code="email",
        )

    def __call__(self, value: Any) -> tuple[bool, str]:
        """Validate email."""
        if value is None or value == "":
            return True, ""

        if not isinstance(value, str):
            return False, self.message

        if not self.EMAIL_PATTERN.match(value):
            return False, self.message

        return True, ""


class URLValidator(Validator):
    """Validator for URLs."""

    URL_PATTERN = re.compile(
        r"^https?://"
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"
        r"localhost|"
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
        r"(?::\d+)?"
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )

    def __init__(self, message: str | None = None) -> None:
        """Initialize URL validator."""
        super().__init__(
            message=message or "Invalid URL",
            code="url",
        )

    def __call__(self, value: Any) -> tuple[bool, str]:
        """Validate URL."""
        if value is None or value == "":
            return True, ""

        if not isinstance(value, str):
            return False, self.message

        if not self.URL_PATTERN.match(value):
            return False, self.message

        return True, ""


class ChoiceValidator(Validator):
    """Validator for choice fields."""

    def __init__(
        self,
        choices: list[Any],
        message: str | None = None,
    ) -> None:
        """
        Initialize choice validator.

        Args:
            choices: Valid choices
            message: Error message
        """
        super().__init__(message=message, code="choice")
        self.choices = choices

    def __call__(self, value: Any) -> tuple[bool, str]:
        """Validate choice."""
        if value is None:
            return True, ""

        if value not in self.choices:
            msg = self.message or f"Must be one of: {', '.join(str(c) for c in self.choices)}"
            return False, msg

        return True, ""


class DateValidator(Validator):
    """Validator for dates."""

    def __init__(
        self,
        min_date: date | None = None,
        max_date: date | None = None,
        format_str: str = "%Y-%m-%d",
        message: str | None = None,
    ) -> None:
        """
        Initialize date validator.

        Args:
            min_date: Minimum date
            max_date: Maximum date
            format_str: Date format string
            message: Error message
        """
        super().__init__(message=message, code="date")
        self.min_date = min_date
        self.max_date = max_date
        self.format_str = format_str

    def __call__(self, value: Any) -> tuple[bool, str]:
        """Validate date."""
        if value is None or value == "":
            return True, ""

        if isinstance(value, date):
            date_value = value
        elif isinstance(value, str):
            try:
                date_value = datetime.strptime(value, self.format_str).date()
            except ValueError:
                return False, self.message or f"Invalid date format. Use {self.format_str}"
        else:
            return False, "Invalid date type"

        if self.min_date and date_value < self.min_date:
            msg = self.message or f"Date must be after {self.min_date}"
            return False, msg

        if self.max_date and date_value > self.max_date:
            msg = self.message or f"Date must be before {self.max_date}"
            return False, msg

        return True, ""


class DecimalValidator(Validator):
    """Validator for decimal numbers."""

    def __init__(
        self,
        min_value: Decimal | float | None = None,
        max_value: Decimal | float | None = None,
        decimal_places: int | None = None,
        message: str | None = None,
    ) -> None:
        """
        Initialize decimal validator.

        Args:
            min_value: Minimum value
            max_value: Maximum value
            decimal_places: Maximum decimal places
            message: Error message
        """
        super().__init__(message=message, code="decimal")
        self.min_value = Decimal(str(min_value)) if min_value is not None else None
        self.max_value = Decimal(str(max_value)) if max_value is not None else None
        self.decimal_places = decimal_places

    def __call__(self, value: Any) -> tuple[bool, str]:
        """Validate decimal."""
        if value is None or value == "":
            return True, ""

        try:
            decimal_value = Decimal(str(value))
        except InvalidOperation:
            return False, self.message or "Invalid decimal number"

        if self.min_value is not None and decimal_value < self.min_value:
            msg = self.message or f"Minimum value is {self.min_value}"
            return False, msg

        if self.max_value is not None and decimal_value > self.max_value:
            msg = self.message or f"Maximum value is {self.max_value}"
            return False, msg

        if self.decimal_places is not None:
            sign, digits, exponent = decimal_value.as_tuple()
            if isinstance(exponent, int) and abs(exponent) > self.decimal_places:
                msg = self.message or f"Maximum {self.decimal_places} decimal places"
                return False, msg

        return True, ""


class SymbolValidator(Validator):
    """Validator for stock symbols."""

    SYMBOL_PATTERN = re.compile(r"^[A-Z]{1,5}$")

    def __init__(self, message: str | None = None) -> None:
        """Initialize symbol validator."""
        super().__init__(
            message=message or "Invalid stock symbol",
            code="symbol",
        )

    def __call__(self, value: Any) -> tuple[bool, str]:
        """Validate symbol."""
        if value is None or value == "":
            return True, ""

        if not isinstance(value, str):
            return False, self.message

        value = value.upper().strip()
        if not self.SYMBOL_PATTERN.match(value):
            return False, self.message

        return True, ""


class PasswordValidator(Validator):
    """Validator for passwords."""

    def __init__(
        self,
        min_length: int = 8,
        require_uppercase: bool = True,
        require_lowercase: bool = True,
        require_digit: bool = True,
        require_special: bool = False,
        message: str | None = None,
    ) -> None:
        """
        Initialize password validator.

        Args:
            min_length: Minimum password length
            require_uppercase: Require uppercase letter
            require_lowercase: Require lowercase letter
            require_digit: Require digit
            require_special: Require special character
            message: Error message
        """
        super().__init__(message=message, code="password")
        self.min_length = min_length
        self.require_uppercase = require_uppercase
        self.require_lowercase = require_lowercase
        self.require_digit = require_digit
        self.require_special = require_special

    def __call__(self, value: Any) -> tuple[bool, str]:
        """Validate password."""
        if value is None or value == "":
            return True, ""

        if not isinstance(value, str):
            return False, "Password must be a string"

        errors = []

        if len(value) < self.min_length:
            errors.append(f"at least {self.min_length} characters")

        if self.require_uppercase and not re.search(r"[A-Z]", value):
            errors.append("an uppercase letter")

        if self.require_lowercase and not re.search(r"[a-z]", value):
            errors.append("a lowercase letter")

        if self.require_digit and not re.search(r"\d", value):
            errors.append("a digit")

        if self.require_special and not re.search(r"[!@#$%^&*(),.?\":{}|<>]", value):
            errors.append("a special character")

        if errors:
            msg = self.message or f"Password must contain {', '.join(errors)}"
            return False, msg

        return True, ""


class QuantityValidator(Validator):
    """Validator for order quantities."""

    def __init__(
        self,
        min_quantity: float = 0.0001,
        max_quantity: float = 1000000,
        allow_fractional: bool = True,
        message: str | None = None,
    ) -> None:
        """
        Initialize quantity validator.

        Args:
            min_quantity: Minimum quantity
            max_quantity: Maximum quantity
            allow_fractional: Allow fractional shares
            message: Error message
        """
        super().__init__(message=message, code="quantity")
        self.min_quantity = min_quantity
        self.max_quantity = max_quantity
        self.allow_fractional = allow_fractional

    def __call__(self, value: Any) -> tuple[bool, str]:
        """Validate quantity."""
        if value is None or value == "":
            return True, ""

        try:
            qty = float(value)
        except (ValueError, TypeError):
            return False, "Quantity must be a number"

        if qty <= 0:
            return False, "Quantity must be positive"

        if qty < self.min_quantity:
            return False, f"Minimum quantity is {self.min_quantity}"

        if qty > self.max_quantity:
            return False, f"Maximum quantity is {self.max_quantity}"

        if not self.allow_fractional and qty != int(qty):
            return False, "Fractional shares not allowed"

        return True, ""


class PriceValidator(Validator):
    """Validator for prices."""

    def __init__(
        self,
        min_price: float = 0.01,
        max_price: float = 1000000,
        tick_size: float | None = None,
        message: str | None = None,
    ) -> None:
        """
        Initialize price validator.

        Args:
            min_price: Minimum price
            max_price: Maximum price
            tick_size: Minimum price increment
            message: Error message
        """
        super().__init__(message=message, code="price")
        self.min_price = min_price
        self.max_price = max_price
        self.tick_size = tick_size

    def __call__(self, value: Any) -> tuple[bool, str]:
        """Validate price."""
        if value is None or value == "":
            return True, ""

        try:
            price = float(value)
        except (ValueError, TypeError):
            return False, "Price must be a number"

        if price < self.min_price:
            return False, f"Minimum price is {self.min_price}"

        if price > self.max_price:
            return False, f"Maximum price is {self.max_price}"

        if self.tick_size:
            remainder = round(price % self.tick_size, 10)
            if remainder != 0:
                return False, f"Price must be in increments of {self.tick_size}"

        return True, ""


def validate_data(
    data: dict[str, Any],
    schema: dict[str, list[Validator]],
) -> ValidationResult:
    """
    Validate data against a schema.

    Args:
        data: Data to validate
        schema: Validation schema (field -> validators)

    Returns:
        ValidationResult
    """
    result = ValidationResult(is_valid=True)

    for field_name, validators in schema.items():
        value = data.get(field_name)
        result.cleaned_data[field_name] = value

        for validator in validators:
            is_valid, error = validator(value)
            if not is_valid:
                result.add_error(field_name, error, validator.code, value)
                break  # Stop on first error for field

    return result


def validate_order_data(data: dict[str, Any]) -> ValidationResult:
    """
    Validate order data.

    Args:
        data: Order data to validate

    Returns:
        ValidationResult
    """
    schema = {
        "symbol": [
            RequiredValidator(),
            SymbolValidator(),
        ],
        "side": [
            RequiredValidator(),
            ChoiceValidator(["buy", "sell"]),
        ],
        "quantity": [
            RequiredValidator(),
            QuantityValidator(),
        ],
        "type": [
            RequiredValidator(),
            ChoiceValidator(["market", "limit", "stop", "stop_limit"]),
        ],
        "limit_price": [
            PriceValidator(),
        ],
        "stop_price": [
            PriceValidator(),
        ],
        "time_in_force": [
            ChoiceValidator(["day", "gtc", "ioc", "fok"]),
        ],
    }

    result = validate_data(data, schema)

    # Cross-field validation
    order_type = data.get("type", "").lower()

    if order_type in ["limit", "stop_limit"]:
        if not data.get("limit_price"):
            result.add_error("limit_price", "Limit price required for limit orders")

    if order_type in ["stop", "stop_limit"]:
        if not data.get("stop_price"):
            result.add_error("stop_price", "Stop price required for stop orders")

    return result


def sanitize_string(value: str) -> str:
    """
    Sanitize string input.

    Args:
        value: String to sanitize

    Returns:
        Sanitized string
    """
    if not isinstance(value, str):
        return str(value)

    # Remove control characters
    value = "".join(c for c in value if ord(c) >= 32 or c in "\n\r\t")

    # Strip whitespace
    value = value.strip()

    return value


def sanitize_html(value: str) -> str:
    """
    Sanitize HTML content.

    Args:
        value: HTML to sanitize

    Returns:
        Sanitized HTML
    """
    if not isinstance(value, str):
        return str(value)

    # Basic HTML entity encoding
    replacements = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#x27;",
    }

    for char, entity in replacements.items():
        value = value.replace(char, entity)

    return value


# Module version
__version__ = "2.2.0"
