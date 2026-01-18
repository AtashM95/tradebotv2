"""
Forms Module for Ultimate Trading Bot v2.2.

This module provides form definitions including:
- Login and registration forms
- Trading forms
- Settings forms
- Validation helpers
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Any


class ValidationError(Exception):
    """Form validation error."""

    def __init__(self, field: str, message: str) -> None:
        """Initialize validation error."""
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")


class FieldType(str, Enum):
    """Form field types."""

    TEXT = "text"
    PASSWORD = "password"
    EMAIL = "email"
    NUMBER = "number"
    DECIMAL = "decimal"
    INTEGER = "integer"
    SELECT = "select"
    MULTISELECT = "multiselect"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    DATE = "date"
    DATETIME = "datetime"
    TEXTAREA = "textarea"
    HIDDEN = "hidden"
    FILE = "file"


@dataclass
class FormField:
    """Form field definition."""

    name: str
    field_type: FieldType
    label: str = ""
    required: bool = False
    default: Any = None
    placeholder: str = ""
    help_text: str = ""
    min_length: int | None = None
    max_length: int | None = None
    min_value: float | None = None
    max_value: float | None = None
    pattern: str | None = None
    choices: list[tuple[str, str]] = field(default_factory=list)
    validators: list[str] = field(default_factory=list)
    attrs: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set default label if not provided."""
        if not self.label:
            self.label = self.name.replace("_", " ").title()


@dataclass
class FormResult:
    """Form validation result."""

    is_valid: bool
    data: dict[str, Any]
    errors: dict[str, list[str]]


class BaseForm:
    """
    Base form class with validation.

    Provides form field definition and validation.
    """

    fields: list[FormField] = []

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        """Initialize form with optional data."""
        self._data = data or {}
        self._errors: dict[str, list[str]] = {}
        self._cleaned_data: dict[str, Any] = {}

    def validate(self) -> bool:
        """
        Validate form data.

        Returns:
            True if valid
        """
        self._errors = {}
        self._cleaned_data = {}

        for form_field in self.fields:
            value = self._data.get(form_field.name)
            field_errors = self._validate_field(form_field, value)

            if field_errors:
                self._errors[form_field.name] = field_errors
            else:
                self._cleaned_data[form_field.name] = self._clean_field(form_field, value)

        return len(self._errors) == 0

    def _validate_field(
        self,
        form_field: FormField,
        value: Any,
    ) -> list[str]:
        """Validate a single field."""
        errors = []

        # Required check
        if form_field.required and (value is None or value == ""):
            errors.append(f"{form_field.label} is required")
            return errors

        # Skip other validations if empty and not required
        if value is None or value == "":
            return errors

        # Type-specific validation
        if form_field.field_type == FieldType.EMAIL:
            if not self._validate_email(value):
                errors.append("Invalid email address")

        elif form_field.field_type in [FieldType.NUMBER, FieldType.DECIMAL]:
            try:
                float(value)
            except (TypeError, ValueError):
                errors.append("Must be a valid number")

        elif form_field.field_type == FieldType.INTEGER:
            try:
                int(value)
            except (TypeError, ValueError):
                errors.append("Must be a valid integer")

        elif form_field.field_type == FieldType.DATE:
            if not self._validate_date(value):
                errors.append("Invalid date format")

        # Length validation
        if form_field.min_length and len(str(value)) < form_field.min_length:
            errors.append(f"Must be at least {form_field.min_length} characters")

        if form_field.max_length and len(str(value)) > form_field.max_length:
            errors.append(f"Must be at most {form_field.max_length} characters")

        # Value range validation
        if form_field.min_value is not None:
            try:
                if float(value) < form_field.min_value:
                    errors.append(f"Must be at least {form_field.min_value}")
            except (TypeError, ValueError):
                pass

        if form_field.max_value is not None:
            try:
                if float(value) > form_field.max_value:
                    errors.append(f"Must be at most {form_field.max_value}")
            except (TypeError, ValueError):
                pass

        # Pattern validation
        if form_field.pattern:
            if not re.match(form_field.pattern, str(value)):
                errors.append("Invalid format")

        # Choice validation
        if form_field.choices:
            valid_values = [c[0] for c in form_field.choices]
            if form_field.field_type == FieldType.MULTISELECT:
                if isinstance(value, list):
                    for v in value:
                        if v not in valid_values:
                            errors.append(f"Invalid selection: {v}")
            else:
                if value not in valid_values:
                    errors.append("Invalid selection")

        return errors

    def _clean_field(self, form_field: FormField, value: Any) -> Any:
        """Clean and convert field value."""
        if value is None or value == "":
            return form_field.default

        if form_field.field_type == FieldType.INTEGER:
            return int(value)

        elif form_field.field_type in [FieldType.NUMBER, FieldType.DECIMAL]:
            return float(value)

        elif form_field.field_type == FieldType.CHECKBOX:
            return value in [True, "true", "True", "1", "on"]

        elif form_field.field_type == FieldType.DATE:
            if isinstance(value, date):
                return value
            return datetime.strptime(value, "%Y-%m-%d").date()

        elif form_field.field_type == FieldType.DATETIME:
            if isinstance(value, datetime):
                return value
            return datetime.fromisoformat(value)

        elif form_field.field_type == FieldType.MULTISELECT:
            if isinstance(value, list):
                return value
            return [value]

        return str(value).strip()

    def _validate_email(self, email: str) -> bool:
        """Validate email format."""
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(pattern, email))

    def _validate_date(self, value: str | date) -> bool:
        """Validate date format."""
        if isinstance(value, date):
            return True
        try:
            datetime.strptime(value, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    @property
    def errors(self) -> dict[str, list[str]]:
        """Get validation errors."""
        return self._errors

    @property
    def data(self) -> dict[str, Any]:
        """Get cleaned data."""
        return self._cleaned_data

    def get_result(self) -> FormResult:
        """Get form result."""
        return FormResult(
            is_valid=len(self._errors) == 0,
            data=self._cleaned_data,
            errors=self._errors,
        )


class LoginForm(BaseForm):
    """Login form."""

    fields = [
        FormField(
            name="username",
            field_type=FieldType.TEXT,
            label="Username",
            required=True,
            min_length=3,
            max_length=50,
            placeholder="Enter your username",
        ),
        FormField(
            name="password",
            field_type=FieldType.PASSWORD,
            label="Password",
            required=True,
            min_length=8,
            placeholder="Enter your password",
        ),
        FormField(
            name="remember_me",
            field_type=FieldType.CHECKBOX,
            label="Remember me",
            default=False,
        ),
        FormField(
            name="totp_code",
            field_type=FieldType.TEXT,
            label="2FA Code",
            required=False,
            min_length=6,
            max_length=6,
            pattern=r"^\d{6}$",
            placeholder="Enter 6-digit code",
        ),
    ]


class RegistrationForm(BaseForm):
    """User registration form."""

    fields = [
        FormField(
            name="username",
            field_type=FieldType.TEXT,
            label="Username",
            required=True,
            min_length=3,
            max_length=30,
            pattern=r"^[a-zA-Z0-9_]+$",
            placeholder="Choose a username",
            help_text="Letters, numbers, and underscores only",
        ),
        FormField(
            name="email",
            field_type=FieldType.EMAIL,
            label="Email",
            required=True,
            max_length=100,
            placeholder="Enter your email",
        ),
        FormField(
            name="password",
            field_type=FieldType.PASSWORD,
            label="Password",
            required=True,
            min_length=8,
            max_length=128,
            placeholder="Choose a password",
            help_text="At least 8 characters",
        ),
        FormField(
            name="password_confirm",
            field_type=FieldType.PASSWORD,
            label="Confirm Password",
            required=True,
            placeholder="Confirm your password",
        ),
    ]

    def validate(self) -> bool:
        """Validate form with password match check."""
        if not super().validate():
            return False

        # Check password match
        if self._data.get("password") != self._data.get("password_confirm"):
            self._errors["password_confirm"] = ["Passwords do not match"]
            return False

        return True


class OrderForm(BaseForm):
    """Order placement form."""

    fields = [
        FormField(
            name="symbol",
            field_type=FieldType.TEXT,
            label="Symbol",
            required=True,
            min_length=1,
            max_length=10,
            pattern=r"^[A-Z]+$",
            placeholder="e.g., AAPL",
        ),
        FormField(
            name="side",
            field_type=FieldType.SELECT,
            label="Side",
            required=True,
            choices=[
                ("buy", "Buy"),
                ("sell", "Sell"),
            ],
        ),
        FormField(
            name="order_type",
            field_type=FieldType.SELECT,
            label="Order Type",
            required=True,
            choices=[
                ("market", "Market"),
                ("limit", "Limit"),
                ("stop", "Stop"),
                ("stop_limit", "Stop Limit"),
            ],
            default="market",
        ),
        FormField(
            name="quantity",
            field_type=FieldType.DECIMAL,
            label="Quantity",
            required=True,
            min_value=0.0001,
            placeholder="Enter quantity",
        ),
        FormField(
            name="limit_price",
            field_type=FieldType.DECIMAL,
            label="Limit Price",
            required=False,
            min_value=0.01,
            placeholder="Enter limit price",
        ),
        FormField(
            name="stop_price",
            field_type=FieldType.DECIMAL,
            label="Stop Price",
            required=False,
            min_value=0.01,
            placeholder="Enter stop price",
        ),
        FormField(
            name="time_in_force",
            field_type=FieldType.SELECT,
            label="Time in Force",
            required=True,
            choices=[
                ("day", "Day"),
                ("gtc", "Good Till Cancelled"),
                ("ioc", "Immediate or Cancel"),
                ("fok", "Fill or Kill"),
            ],
            default="day",
        ),
    ]


class BacktestForm(BaseForm):
    """Backtesting configuration form."""

    fields = [
        FormField(
            name="strategy",
            field_type=FieldType.SELECT,
            label="Strategy",
            required=True,
            choices=[
                ("momentum", "Momentum"),
                ("mean_reversion", "Mean Reversion"),
                ("trend_following", "Trend Following"),
                ("breakout", "Breakout"),
                ("pairs", "Pairs Trading"),
            ],
        ),
        FormField(
            name="symbols",
            field_type=FieldType.TEXT,
            label="Symbols",
            required=True,
            placeholder="AAPL, GOOGL, MSFT",
            help_text="Comma-separated list of symbols",
        ),
        FormField(
            name="start_date",
            field_type=FieldType.DATE,
            label="Start Date",
            required=True,
        ),
        FormField(
            name="end_date",
            field_type=FieldType.DATE,
            label="End Date",
            required=True,
        ),
        FormField(
            name="initial_capital",
            field_type=FieldType.DECIMAL,
            label="Initial Capital",
            required=True,
            min_value=1000,
            default=100000,
            placeholder="100000",
        ),
        FormField(
            name="commission",
            field_type=FieldType.DECIMAL,
            label="Commission (%)",
            required=False,
            min_value=0,
            max_value=10,
            default=0.1,
        ),
    ]


class AlertForm(BaseForm):
    """Alert configuration form."""

    fields = [
        FormField(
            name="symbol",
            field_type=FieldType.TEXT,
            label="Symbol",
            required=True,
            pattern=r"^[A-Z]+$",
        ),
        FormField(
            name="alert_type",
            field_type=FieldType.SELECT,
            label="Alert Type",
            required=True,
            choices=[
                ("price_above", "Price Above"),
                ("price_below", "Price Below"),
                ("percent_change", "Percent Change"),
                ("volume_spike", "Volume Spike"),
            ],
        ),
        FormField(
            name="trigger_value",
            field_type=FieldType.DECIMAL,
            label="Trigger Value",
            required=True,
            min_value=0,
        ),
        FormField(
            name="notification_channels",
            field_type=FieldType.MULTISELECT,
            label="Notification Channels",
            choices=[
                ("email", "Email"),
                ("sms", "SMS"),
                ("push", "Push Notification"),
                ("slack", "Slack"),
            ],
            default=["push"],
        ),
        FormField(
            name="enabled",
            field_type=FieldType.CHECKBOX,
            label="Enable Alert",
            default=True,
        ),
    ]


class SettingsForm(BaseForm):
    """User settings form."""

    fields = [
        FormField(
            name="theme",
            field_type=FieldType.SELECT,
            label="Theme",
            choices=[
                ("light", "Light"),
                ("dark", "Dark"),
                ("system", "System Default"),
            ],
            default="dark",
        ),
        FormField(
            name="default_order_type",
            field_type=FieldType.SELECT,
            label="Default Order Type",
            choices=[
                ("market", "Market"),
                ("limit", "Limit"),
            ],
            default="market",
        ),
        FormField(
            name="confirm_orders",
            field_type=FieldType.CHECKBOX,
            label="Confirm Before Placing Orders",
            default=True,
        ),
        FormField(
            name="show_portfolio_value",
            field_type=FieldType.CHECKBOX,
            label="Show Portfolio Value",
            default=True,
        ),
        FormField(
            name="email_notifications",
            field_type=FieldType.CHECKBOX,
            label="Email Notifications",
            default=True,
        ),
        FormField(
            name="push_notifications",
            field_type=FieldType.CHECKBOX,
            label="Push Notifications",
            default=True,
        ),
        FormField(
            name="timezone",
            field_type=FieldType.SELECT,
            label="Timezone",
            choices=[
                ("America/New_York", "Eastern Time (ET)"),
                ("America/Chicago", "Central Time (CT)"),
                ("America/Denver", "Mountain Time (MT)"),
                ("America/Los_Angeles", "Pacific Time (PT)"),
                ("UTC", "UTC"),
            ],
            default="America/New_York",
        ),
    ]


def validate_form(
    form_class: type[BaseForm],
    data: dict[str, Any],
) -> FormResult:
    """
    Validate form data.

    Args:
        form_class: Form class to use
        data: Form data

    Returns:
        FormResult with validation status
    """
    form = form_class(data)
    form.validate()
    return form.get_result()


# Module version
__version__ = "2.2.0"
