"""
Custom Exceptions Module for Ultimate Trading Bot v2.2.

This module defines all custom exceptions used throughout the trading bot
for better error handling and debugging.
"""

from typing import Any, Dict, Optional, Type
from enum import IntEnum
import traceback
import logging


logger = logging.getLogger(__name__)


class ErrorCode(IntEnum):
    """Error code enumeration for all exceptions."""

    # General errors (1xxx)
    UNKNOWN = 1000
    CONFIGURATION = 1001
    INITIALIZATION = 1002
    VALIDATION = 1003
    TIMEOUT = 1004
    NOT_FOUND = 1005
    ALREADY_EXISTS = 1006
    PERMISSION_DENIED = 1007

    # Data errors (2xxx)
    DATA_FETCH = 2000
    DATA_PARSE = 2001
    DATA_MISSING = 2002
    DATA_INVALID = 2003
    DATA_STALE = 2004
    DATA_CORRUPTION = 2005

    # Trading errors (3xxx)
    ORDER_CREATION = 3000
    ORDER_SUBMISSION = 3001
    ORDER_REJECTED = 3002
    ORDER_NOT_FOUND = 3003
    ORDER_MODIFICATION = 3004
    ORDER_CANCELLATION = 3005
    INSUFFICIENT_FUNDS = 3010
    INSUFFICIENT_SHARES = 3011
    POSITION_NOT_FOUND = 3020
    POSITION_LIMIT = 3021
    RISK_LIMIT = 3030
    RISK_VIOLATION = 3031
    MARKET_CLOSED = 3040
    SYMBOL_NOT_TRADABLE = 3041

    # API errors (4xxx)
    API_CONNECTION = 4000
    API_AUTHENTICATION = 4001
    API_AUTHORIZATION = 4002
    API_RATE_LIMIT = 4003
    API_TIMEOUT = 4004
    API_RESPONSE = 4005
    API_UNAVAILABLE = 4006

    # AI/OpenAI errors (5xxx)
    AI_REQUEST = 5000
    AI_RESPONSE = 5001
    AI_BUDGET_EXCEEDED = 5002
    AI_RATE_LIMIT = 5003
    AI_CONTENT_FILTER = 5004
    AI_INVALID_MODEL = 5005

    # Database errors (6xxx)
    DB_CONNECTION = 6000
    DB_QUERY = 6001
    DB_WRITE = 6002
    DB_INTEGRITY = 6003
    DB_MIGRATION = 6004

    # Cache errors (7xxx)
    CACHE_CONNECTION = 7000
    CACHE_READ = 7001
    CACHE_WRITE = 7002
    CACHE_EXPIRED = 7003

    # Strategy errors (8xxx)
    STRATEGY_LOAD = 8000
    STRATEGY_EXECUTION = 8001
    STRATEGY_VALIDATION = 8002
    STRATEGY_NOT_FOUND = 8003

    # Backtest errors (9xxx)
    BACKTEST_DATA = 9000
    BACKTEST_EXECUTION = 9001
    BACKTEST_VALIDATION = 9002


class TradingBotException(Exception):
    """
    Base exception class for all trading bot exceptions.

    All custom exceptions should inherit from this class.
    """

    error_code: ErrorCode = ErrorCode.UNKNOWN
    default_message: str = "An unknown error occurred"

    def __init__(
        self,
        message: Optional[str] = None,
        error_code: Optional[ErrorCode] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ) -> None:
        """
        Initialize the exception.

        Args:
            message: Error message
            error_code: Error code
            details: Additional error details
            cause: Original exception that caused this one
        """
        self.message = message or self.default_message
        self.error_code = error_code or self.__class__.error_code
        self.details = details or {}
        self.cause = cause

        # Build full message
        full_message = f"[{self.error_code.name}:{self.error_code.value}] {self.message}"
        if self.details:
            full_message += f" | Details: {self.details}"

        super().__init__(full_message)

        # Log the exception
        logger.error(
            f"Exception raised: {self.__class__.__name__}",
            extra={
                "error_code": self.error_code.value,
                "message": self.message,
                "details": self.details,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary.

        Returns:
            Dictionary representation of the exception
        """
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code.value,
            "error_name": self.error_code.name,
            "message": self.message,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None,
        }

    def get_traceback(self) -> str:
        """
        Get the full traceback as a string.

        Returns:
            Traceback string
        """
        return traceback.format_exc()


# =============================================================================
# CONFIGURATION EXCEPTIONS
# =============================================================================

class ConfigurationError(TradingBotException):
    """Raised when there's a configuration error."""
    error_code = ErrorCode.CONFIGURATION
    default_message = "Configuration error"


class InitializationError(TradingBotException):
    """Raised when initialization fails."""
    error_code = ErrorCode.INITIALIZATION
    default_message = "Initialization failed"


class ValidationError(TradingBotException):
    """Raised when validation fails."""
    error_code = ErrorCode.VALIDATION
    default_message = "Validation failed"


# =============================================================================
# DATA EXCEPTIONS
# =============================================================================

class DataError(TradingBotException):
    """Base class for data-related errors."""
    error_code = ErrorCode.DATA_FETCH
    default_message = "Data error"


class DataFetchError(DataError):
    """Raised when data fetching fails."""
    error_code = ErrorCode.DATA_FETCH
    default_message = "Failed to fetch data"


class DataParseError(DataError):
    """Raised when data parsing fails."""
    error_code = ErrorCode.DATA_PARSE
    default_message = "Failed to parse data"


class DataMissingError(DataError):
    """Raised when required data is missing."""
    error_code = ErrorCode.DATA_MISSING
    default_message = "Required data is missing"


class DataInvalidError(DataError):
    """Raised when data is invalid."""
    error_code = ErrorCode.DATA_INVALID
    default_message = "Data is invalid"


class DataStaleError(DataError):
    """Raised when data is stale/outdated."""
    error_code = ErrorCode.DATA_STALE
    default_message = "Data is stale"


# =============================================================================
# TRADING EXCEPTIONS
# =============================================================================

class TradingError(TradingBotException):
    """Base class for trading-related errors."""
    error_code = ErrorCode.ORDER_CREATION
    default_message = "Trading error"


class OrderError(TradingError):
    """Base class for order-related errors."""
    error_code = ErrorCode.ORDER_CREATION
    default_message = "Order error"


class OrderCreationError(OrderError):
    """Raised when order creation fails."""
    error_code = ErrorCode.ORDER_CREATION
    default_message = "Failed to create order"


class OrderSubmissionError(OrderError):
    """Raised when order submission fails."""
    error_code = ErrorCode.ORDER_SUBMISSION
    default_message = "Failed to submit order"


class OrderRejectedError(OrderError):
    """Raised when an order is rejected."""
    error_code = ErrorCode.ORDER_REJECTED
    default_message = "Order was rejected"


class OrderNotFoundError(OrderError):
    """Raised when an order is not found."""
    error_code = ErrorCode.ORDER_NOT_FOUND
    default_message = "Order not found"


class OrderModificationError(OrderError):
    """Raised when order modification fails."""
    error_code = ErrorCode.ORDER_MODIFICATION
    default_message = "Failed to modify order"


class OrderCancellationError(OrderError):
    """Raised when order cancellation fails."""
    error_code = ErrorCode.ORDER_CANCELLATION
    default_message = "Failed to cancel order"


class InsufficientFundsError(TradingError):
    """Raised when there are insufficient funds."""
    error_code = ErrorCode.INSUFFICIENT_FUNDS
    default_message = "Insufficient funds"


class InsufficientSharesError(TradingError):
    """Raised when there are insufficient shares."""
    error_code = ErrorCode.INSUFFICIENT_SHARES
    default_message = "Insufficient shares"


class PositionError(TradingError):
    """Base class for position-related errors."""
    error_code = ErrorCode.POSITION_NOT_FOUND
    default_message = "Position error"


class PositionNotFoundError(PositionError):
    """Raised when a position is not found."""
    error_code = ErrorCode.POSITION_NOT_FOUND
    default_message = "Position not found"


class PositionLimitError(PositionError):
    """Raised when position limit is exceeded."""
    error_code = ErrorCode.POSITION_LIMIT
    default_message = "Position limit exceeded"


class RiskError(TradingError):
    """Base class for risk-related errors."""
    error_code = ErrorCode.RISK_LIMIT
    default_message = "Risk error"


class RiskLimitError(RiskError):
    """Raised when risk limit is exceeded."""
    error_code = ErrorCode.RISK_LIMIT
    default_message = "Risk limit exceeded"


class RiskViolationError(RiskError):
    """Raised when a risk rule is violated."""
    error_code = ErrorCode.RISK_VIOLATION
    default_message = "Risk violation"


class MarketClosedError(TradingError):
    """Raised when market is closed."""
    error_code = ErrorCode.MARKET_CLOSED
    default_message = "Market is closed"


class SymbolNotTradableError(TradingError):
    """Raised when symbol is not tradable."""
    error_code = ErrorCode.SYMBOL_NOT_TRADABLE
    default_message = "Symbol is not tradable"


# =============================================================================
# API EXCEPTIONS
# =============================================================================

class APIError(TradingBotException):
    """Base class for API-related errors."""
    error_code = ErrorCode.API_CONNECTION
    default_message = "API error"


class APIConnectionError(APIError):
    """Raised when API connection fails."""
    error_code = ErrorCode.API_CONNECTION
    default_message = "Failed to connect to API"


class APIAuthenticationError(APIError):
    """Raised when API authentication fails."""
    error_code = ErrorCode.API_AUTHENTICATION
    default_message = "API authentication failed"


class APIAuthorizationError(APIError):
    """Raised when API authorization fails."""
    error_code = ErrorCode.API_AUTHORIZATION
    default_message = "API authorization failed"


class APIRateLimitError(APIError):
    """Raised when API rate limit is exceeded."""
    error_code = ErrorCode.API_RATE_LIMIT
    default_message = "API rate limit exceeded"

    def __init__(
        self,
        message: Optional[str] = None,
        retry_after: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the rate limit error.

        Args:
            message: Error message
            retry_after: Seconds to wait before retrying
            **kwargs: Additional arguments
        """
        self.retry_after = retry_after
        details = kwargs.get("details", {})
        details["retry_after"] = retry_after
        kwargs["details"] = details
        super().__init__(message, **kwargs)


class APITimeoutError(APIError):
    """Raised when API request times out."""
    error_code = ErrorCode.API_TIMEOUT
    default_message = "API request timed out"


class APIResponseError(APIError):
    """Raised when API response is invalid."""
    error_code = ErrorCode.API_RESPONSE
    default_message = "Invalid API response"


class APIUnavailableError(APIError):
    """Raised when API is unavailable."""
    error_code = ErrorCode.API_UNAVAILABLE
    default_message = "API is unavailable"


# =============================================================================
# AI/OPENAI EXCEPTIONS
# =============================================================================

class AIError(TradingBotException):
    """Base class for AI-related errors."""
    error_code = ErrorCode.AI_REQUEST
    default_message = "AI error"


class AIRequestError(AIError):
    """Raised when AI request fails."""
    error_code = ErrorCode.AI_REQUEST
    default_message = "AI request failed"


class AIResponseError(AIError):
    """Raised when AI response is invalid."""
    error_code = ErrorCode.AI_RESPONSE
    default_message = "Invalid AI response"


class AIBudgetExceededError(AIError):
    """Raised when AI budget is exceeded."""
    error_code = ErrorCode.AI_BUDGET_EXCEEDED
    default_message = "AI budget exceeded"


class AIRateLimitError(AIError):
    """Raised when AI rate limit is exceeded."""
    error_code = ErrorCode.AI_RATE_LIMIT
    default_message = "AI rate limit exceeded"


class AIContentFilterError(AIError):
    """Raised when content is filtered by AI."""
    error_code = ErrorCode.AI_CONTENT_FILTER
    default_message = "Content was filtered by AI"


class AIInvalidModelError(AIError):
    """Raised when an invalid model is specified."""
    error_code = ErrorCode.AI_INVALID_MODEL
    default_message = "Invalid AI model specified"


# =============================================================================
# DATABASE EXCEPTIONS
# =============================================================================

class DatabaseError(TradingBotException):
    """Base class for database-related errors."""
    error_code = ErrorCode.DB_CONNECTION
    default_message = "Database error"


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""
    error_code = ErrorCode.DB_CONNECTION
    default_message = "Failed to connect to database"


class DatabaseQueryError(DatabaseError):
    """Raised when database query fails."""
    error_code = ErrorCode.DB_QUERY
    default_message = "Database query failed"


class DatabaseWriteError(DatabaseError):
    """Raised when database write fails."""
    error_code = ErrorCode.DB_WRITE
    default_message = "Database write failed"


class DatabaseIntegrityError(DatabaseError):
    """Raised when database integrity is violated."""
    error_code = ErrorCode.DB_INTEGRITY
    default_message = "Database integrity error"


# =============================================================================
# CACHE EXCEPTIONS
# =============================================================================

class CacheError(TradingBotException):
    """Base class for cache-related errors."""
    error_code = ErrorCode.CACHE_CONNECTION
    default_message = "Cache error"


class CacheConnectionError(CacheError):
    """Raised when cache connection fails."""
    error_code = ErrorCode.CACHE_CONNECTION
    default_message = "Failed to connect to cache"


class CacheReadError(CacheError):
    """Raised when cache read fails."""
    error_code = ErrorCode.CACHE_READ
    default_message = "Cache read failed"


class CacheWriteError(CacheError):
    """Raised when cache write fails."""
    error_code = ErrorCode.CACHE_WRITE
    default_message = "Cache write failed"


# =============================================================================
# STRATEGY EXCEPTIONS
# =============================================================================

class StrategyError(TradingBotException):
    """Base class for strategy-related errors."""
    error_code = ErrorCode.STRATEGY_EXECUTION
    default_message = "Strategy error"


class StrategyLoadError(StrategyError):
    """Raised when strategy loading fails."""
    error_code = ErrorCode.STRATEGY_LOAD
    default_message = "Failed to load strategy"


class StrategyExecutionError(StrategyError):
    """Raised when strategy execution fails."""
    error_code = ErrorCode.STRATEGY_EXECUTION
    default_message = "Strategy execution failed"


class StrategyValidationError(StrategyError):
    """Raised when strategy validation fails."""
    error_code = ErrorCode.STRATEGY_VALIDATION
    default_message = "Strategy validation failed"


class StrategyNotFoundError(StrategyError):
    """Raised when strategy is not found."""
    error_code = ErrorCode.STRATEGY_NOT_FOUND
    default_message = "Strategy not found"


# =============================================================================
# BACKTEST EXCEPTIONS
# =============================================================================

class BacktestError(TradingBotException):
    """Base class for backtest-related errors."""
    error_code = ErrorCode.BACKTEST_EXECUTION
    default_message = "Backtest error"


class BacktestDataError(BacktestError):
    """Raised when backtest data is invalid."""
    error_code = ErrorCode.BACKTEST_DATA
    default_message = "Invalid backtest data"


class BacktestExecutionError(BacktestError):
    """Raised when backtest execution fails."""
    error_code = ErrorCode.BACKTEST_EXECUTION
    default_message = "Backtest execution failed"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_exception_class(error_code: ErrorCode) -> Type[TradingBotException]:
    """
    Get the exception class for an error code.

    Args:
        error_code: Error code

    Returns:
        Exception class
    """
    code_to_class: Dict[ErrorCode, Type[TradingBotException]] = {
        ErrorCode.CONFIGURATION: ConfigurationError,
        ErrorCode.INITIALIZATION: InitializationError,
        ErrorCode.VALIDATION: ValidationError,
        ErrorCode.DATA_FETCH: DataFetchError,
        ErrorCode.DATA_PARSE: DataParseError,
        ErrorCode.DATA_MISSING: DataMissingError,
        ErrorCode.DATA_INVALID: DataInvalidError,
        ErrorCode.ORDER_CREATION: OrderCreationError,
        ErrorCode.ORDER_SUBMISSION: OrderSubmissionError,
        ErrorCode.ORDER_REJECTED: OrderRejectedError,
        ErrorCode.INSUFFICIENT_FUNDS: InsufficientFundsError,
        ErrorCode.POSITION_NOT_FOUND: PositionNotFoundError,
        ErrorCode.RISK_LIMIT: RiskLimitError,
        ErrorCode.API_CONNECTION: APIConnectionError,
        ErrorCode.API_AUTHENTICATION: APIAuthenticationError,
        ErrorCode.API_RATE_LIMIT: APIRateLimitError,
        ErrorCode.AI_REQUEST: AIRequestError,
        ErrorCode.AI_RESPONSE: AIResponseError,
        ErrorCode.AI_BUDGET_EXCEEDED: AIBudgetExceededError,
        ErrorCode.DB_CONNECTION: DatabaseConnectionError,
        ErrorCode.DB_QUERY: DatabaseQueryError,
        ErrorCode.CACHE_CONNECTION: CacheConnectionError,
        ErrorCode.STRATEGY_EXECUTION: StrategyExecutionError,
        ErrorCode.BACKTEST_EXECUTION: BacktestExecutionError,
    }
    return code_to_class.get(error_code, TradingBotException)


def raise_from_error_code(
    error_code: ErrorCode,
    message: Optional[str] = None,
    **kwargs: Any
) -> None:
    """
    Raise an exception based on error code.

    Args:
        error_code: Error code
        message: Error message
        **kwargs: Additional arguments
    """
    exception_class = get_exception_class(error_code)
    raise exception_class(message, error_code=error_code, **kwargs)
