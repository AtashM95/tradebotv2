"""
Logging Configuration Module for Ultimate Trading Bot v2.2.

This module provides comprehensive logging configuration including:
- Structured logging with JSON format
- Multiple handlers (console, file, rotating)
- Log filtering and formatting
- Context-aware logging
- Performance logging
"""

import json
import logging
import logging.handlers
import os
import sys
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class LogLevel(str, Enum):
    """Log level enumeration."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    def to_logging_level(self) -> int:
        """Convert to logging module level."""
        return getattr(logging, self.value)


class LogFormat(str, Enum):
    """Log format enumeration."""

    SIMPLE = "simple"
    DETAILED = "detailed"
    JSON = "json"
    CUSTOM = "custom"


@dataclass
class LoggingConfig:
    """Configuration for logging system."""

    # General settings
    level: LogLevel = LogLevel.INFO
    format_type: LogFormat = LogFormat.DETAILED
    custom_format: str | None = None

    # Console settings
    console_enabled: bool = True
    console_level: LogLevel | None = None
    console_colorized: bool = True

    # File settings
    file_enabled: bool = True
    file_path: str = "logs/trading_bot.log"
    file_level: LogLevel | None = None
    file_max_bytes: int = 10 * 1024 * 1024  # 10MB
    file_backup_count: int = 5

    # Error file settings
    error_file_enabled: bool = True
    error_file_path: str = "logs/errors.log"
    error_file_max_bytes: int = 10 * 1024 * 1024
    error_file_backup_count: int = 10

    # JSON logging settings
    json_file_enabled: bool = False
    json_file_path: str = "logs/trading_bot.json"
    json_file_max_bytes: int = 50 * 1024 * 1024  # 50MB
    json_file_backup_count: int = 3

    # Performance logging
    performance_logging: bool = True
    slow_operation_threshold_ms: float = 1000.0

    # Log filtering
    filter_modules: list[str] = field(default_factory=list)
    filter_levels: list[LogLevel] = field(default_factory=list)


# ANSI color codes for console output
class Colors:
    """ANSI color codes."""

    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Level-specific colors
    LEVEL_COLORS = {
        "DEBUG": CYAN,
        "INFO": GREEN,
        "WARNING": YELLOW,
        "ERROR": RED,
        "CRITICAL": f"{BOLD}{RED}",
    }


class ContextVar:
    """Thread-local context variable storage."""

    _context: dict[str, dict[str, Any]] = {}
    _lock = threading.Lock()

    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """Set a context variable for the current thread."""
        thread_id = str(threading.current_thread().ident)
        with cls._lock:
            if thread_id not in cls._context:
                cls._context[thread_id] = {}
            cls._context[thread_id][key] = value

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get a context variable for the current thread."""
        thread_id = str(threading.current_thread().ident)
        with cls._lock:
            return cls._context.get(thread_id, {}).get(key, default)

    @classmethod
    def get_all(cls) -> dict[str, Any]:
        """Get all context variables for the current thread."""
        thread_id = str(threading.current_thread().ident)
        with cls._lock:
            return cls._context.get(thread_id, {}).copy()

    @classmethod
    def clear(cls, key: str | None = None) -> None:
        """Clear context variable(s) for the current thread."""
        thread_id = str(threading.current_thread().ident)
        with cls._lock:
            if thread_id in cls._context:
                if key is None:
                    cls._context[thread_id] = {}
                elif key in cls._context[thread_id]:
                    del cls._context[thread_id][key]

    @classmethod
    def cleanup_thread(cls) -> None:
        """Clean up context for the current thread."""
        thread_id = str(threading.current_thread().ident)
        with cls._lock:
            if thread_id in cls._context:
                del cls._context[thread_id]


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to console output."""

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        colorized: bool = True,
    ) -> None:
        """Initialize colored formatter."""
        super().__init__(fmt, datefmt)
        self.colorized = colorized

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Save original values
        original_levelname = record.levelname
        original_msg = record.msg

        if self.colorized:
            # Add color to level name
            color = Colors.LEVEL_COLORS.get(record.levelname, "")
            record.levelname = f"{color}{record.levelname}{Colors.RESET}"

            # Add color to message based on level
            if record.levelno >= logging.ERROR:
                record.msg = f"{Colors.RED}{record.msg}{Colors.RESET}"
            elif record.levelno >= logging.WARNING:
                record.msg = f"{Colors.YELLOW}{record.msg}{Colors.RESET}"

        # Format the record
        result = super().format(record)

        # Restore original values
        record.levelname = original_levelname
        record.msg = original_msg

        return result


class JSONFormatter(logging.Formatter):
    """Formatter that outputs JSON-formatted log records."""

    def __init__(
        self,
        include_context: bool = True,
        include_extra: bool = True,
    ) -> None:
        """Initialize JSON formatter."""
        super().__init__()
        self.include_context = include_context
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add context variables
        if self.include_context:
            context = ContextVar.get_all()
            if context:
                log_data["context"] = context

        # Add extra fields
        if self.include_extra:
            extra = {}
            for key, value in record.__dict__.items():
                if key not in [
                    "name", "msg", "args", "created", "filename", "funcName",
                    "levelname", "levelno", "lineno", "module", "msecs",
                    "pathname", "process", "processName", "relativeCreated",
                    "stack_info", "exc_info", "exc_text", "thread", "threadName",
                    "message", "asctime",
                ]:
                    try:
                        json.dumps(value)  # Test if serializable
                        extra[key] = value
                    except (TypeError, ValueError):
                        extra[key] = str(value)

            if extra:
                log_data["extra"] = extra

        return json.dumps(log_data, default=str)


class DetailedFormatter(logging.Formatter):
    """Detailed formatter with context information."""

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        include_context: bool = True,
    ) -> None:
        """Initialize detailed formatter."""
        if fmt is None:
            fmt = (
                "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | "
                "%(message)s"
            )
        if datefmt is None:
            datefmt = "%Y-%m-%d %H:%M:%S"

        super().__init__(fmt, datefmt)
        self.include_context = include_context

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with details."""
        result = super().format(record)

        # Add context if available
        if self.include_context:
            context = ContextVar.get_all()
            if context:
                context_str = " | ".join(f"{k}={v}" for k, v in context.items())
                result = f"{result} | ctx: {context_str}"

        return result


class ModuleFilter(logging.Filter):
    """Filter that allows/denies specific modules."""

    def __init__(
        self,
        allowed_modules: list[str] | None = None,
        denied_modules: list[str] | None = None,
    ) -> None:
        """Initialize module filter."""
        super().__init__()
        self.allowed_modules = allowed_modules or []
        self.denied_modules = denied_modules or []

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log record based on module."""
        # If allowed list specified, only allow those modules
        if self.allowed_modules:
            for module in self.allowed_modules:
                if record.name.startswith(module):
                    return True
            return False

        # If denied list specified, deny those modules
        if self.denied_modules:
            for module in self.denied_modules:
                if record.name.startswith(module):
                    return False

        return True


class LevelRangeFilter(logging.Filter):
    """Filter that allows only specific level range."""

    def __init__(
        self,
        min_level: int = logging.DEBUG,
        max_level: int = logging.CRITICAL,
    ) -> None:
        """Initialize level range filter."""
        super().__init__()
        self.min_level = min_level
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log record based on level."""
        return self.min_level <= record.levelno <= self.max_level


class PerformanceLogger:
    """Logger for tracking operation performance."""

    def __init__(
        self,
        logger: logging.Logger,
        slow_threshold_ms: float = 1000.0,
    ) -> None:
        """Initialize performance logger."""
        self._logger = logger
        self._slow_threshold_ms = slow_threshold_ms
        self._operations: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def start_operation(self, operation_id: str, name: str) -> None:
        """Start timing an operation."""
        with self._lock:
            self._operations[operation_id] = {
                "name": name,
                "start_time": datetime.now(),
                "metadata": {},
            }
        self._logger.debug(f"Operation started: {name} ({operation_id})")

    def add_metadata(self, operation_id: str, **kwargs: Any) -> None:
        """Add metadata to an operation."""
        with self._lock:
            if operation_id in self._operations:
                self._operations[operation_id]["metadata"].update(kwargs)

    def end_operation(
        self,
        operation_id: str,
        success: bool = True,
        error: str | None = None,
    ) -> float:
        """End timing an operation and log results."""
        with self._lock:
            if operation_id not in self._operations:
                self._logger.warning(f"Unknown operation ID: {operation_id}")
                return 0.0

            operation = self._operations.pop(operation_id)

        duration = (datetime.now() - operation["start_time"]).total_seconds() * 1000
        name = operation["name"]
        metadata = operation["metadata"]

        # Prepare log message
        log_data = {
            "operation": name,
            "duration_ms": round(duration, 2),
            "success": success,
            **metadata,
        }

        if error:
            log_data["error"] = error

        # Log based on duration and success
        if not success:
            self._logger.error(f"Operation failed: {name}", extra=log_data)
        elif duration >= self._slow_threshold_ms:
            self._logger.warning(f"Slow operation: {name} ({duration:.2f}ms)", extra=log_data)
        else:
            self._logger.debug(f"Operation completed: {name} ({duration:.2f}ms)", extra=log_data)

        return duration


class TradingLogger(logging.Logger):
    """Extended logger with trading-specific functionality."""

    def __init__(self, name: str, level: int = logging.NOTSET) -> None:
        """Initialize trading logger."""
        super().__init__(name, level)
        self._performance_logger: PerformanceLogger | None = None

    def set_performance_logger(self, perf_logger: PerformanceLogger) -> None:
        """Set performance logger instance."""
        self._performance_logger = perf_logger

    def trade(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        **kwargs: Any,
    ) -> None:
        """Log a trade event."""
        extra = {
            "event_type": "trade",
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": price,
            **kwargs,
        }
        self.info(
            f"TRADE: {action} {quantity} {symbol} @ ${price:.2f}",
            extra=extra,
        )

    def order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        status: str,
        **kwargs: Any,
    ) -> None:
        """Log an order event."""
        extra = {
            "event_type": "order",
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "order_type": order_type,
            "status": status,
            **kwargs,
        }
        self.info(
            f"ORDER: {order_id} {side} {quantity} {symbol} ({order_type}) - {status}",
            extra=extra,
        )

    def signal(
        self,
        symbol: str,
        signal_type: str,
        strength: float,
        source: str,
        **kwargs: Any,
    ) -> None:
        """Log a trading signal."""
        extra = {
            "event_type": "signal",
            "symbol": symbol,
            "signal_type": signal_type,
            "strength": strength,
            "source": source,
            **kwargs,
        }
        self.info(
            f"SIGNAL: {symbol} {signal_type} (strength: {strength:.2f}) from {source}",
            extra=extra,
        )

    def risk(
        self,
        risk_type: str,
        level: str,
        message: str,
        **kwargs: Any,
    ) -> None:
        """Log a risk event."""
        extra = {
            "event_type": "risk",
            "risk_type": risk_type,
            "risk_level": level,
            **kwargs,
        }
        log_level = logging.WARNING if level in ["high", "critical"] else logging.INFO
        self.log(log_level, f"RISK [{level.upper()}]: {risk_type} - {message}", extra=extra)

    def performance(
        self,
        metric: str,
        value: float,
        benchmark: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Log a performance metric."""
        extra = {
            "event_type": "performance",
            "metric": metric,
            "value": value,
            "benchmark": benchmark,
            **kwargs,
        }
        msg = f"PERF: {metric} = {value:.4f}"
        if benchmark is not None:
            msg += f" (benchmark: {benchmark:.4f})"
        self.info(msg, extra=extra)


class LoggingManager:
    """
    Centralized logging configuration manager.

    Manages all logging handlers, formatters, and configuration.
    """

    _instance: "LoggingManager | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "LoggingManager":
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize logging manager."""
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._config: LoggingConfig | None = None
        self._handlers: dict[str, logging.Handler] = {}
        self._formatters: dict[str, logging.Formatter] = {}
        self._performance_loggers: dict[str, PerformanceLogger] = {}
        self._initialized = False

        # Register custom logger class
        logging.setLoggerClass(TradingLogger)

    def configure(self, config: LoggingConfig | None = None) -> None:
        """
        Configure the logging system.

        Args:
            config: Logging configuration
        """
        self._config = config or LoggingConfig()

        # Create log directories
        self._ensure_log_directories()

        # Create formatters
        self._create_formatters()

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self._config.level.to_logging_level())

        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add handlers
        if self._config.console_enabled:
            self._add_console_handler(root_logger)

        if self._config.file_enabled:
            self._add_file_handler(root_logger)

        if self._config.error_file_enabled:
            self._add_error_file_handler(root_logger)

        if self._config.json_file_enabled:
            self._add_json_file_handler(root_logger)

        self._initialized = True
        logging.info("Logging system configured successfully")

    def _ensure_log_directories(self) -> None:
        """Ensure log directories exist."""
        if self._config is None:
            return

        paths = [
            self._config.file_path,
            self._config.error_file_path,
            self._config.json_file_path,
        ]

        for path in paths:
            dir_path = Path(path).parent
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)

    def _create_formatters(self) -> None:
        """Create log formatters."""
        if self._config is None:
            return

        # Simple formatter
        self._formatters["simple"] = logging.Formatter(
            "%(levelname)s: %(message)s"
        )

        # Detailed formatter
        self._formatters["detailed"] = DetailedFormatter()

        # Colored formatter
        self._formatters["colored"] = ColoredFormatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            "%H:%M:%S",
            colorized=self._config.console_colorized,
        )

        # JSON formatter
        self._formatters["json"] = JSONFormatter()

        # Custom formatter if specified
        if self._config.custom_format:
            self._formatters["custom"] = logging.Formatter(
                self._config.custom_format
            )

    def _add_console_handler(self, logger: logging.Logger) -> None:
        """Add console handler to logger."""
        if self._config is None:
            return

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(
            (self._config.console_level or self._config.level).to_logging_level()
        )

        # Use colored formatter for console
        handler.setFormatter(self._formatters["colored"])

        # Add filters if configured
        if self._config.filter_modules:
            handler.addFilter(ModuleFilter(denied_modules=self._config.filter_modules))

        logger.addHandler(handler)
        self._handlers["console"] = handler

    def _add_file_handler(self, logger: logging.Logger) -> None:
        """Add rotating file handler to logger."""
        if self._config is None:
            return

        handler = logging.handlers.RotatingFileHandler(
            self._config.file_path,
            maxBytes=self._config.file_max_bytes,
            backupCount=self._config.file_backup_count,
            encoding="utf-8",
        )
        handler.setLevel(
            (self._config.file_level or self._config.level).to_logging_level()
        )

        # Use detailed formatter for file
        handler.setFormatter(self._formatters["detailed"])

        logger.addHandler(handler)
        self._handlers["file"] = handler

    def _add_error_file_handler(self, logger: logging.Logger) -> None:
        """Add error file handler to logger."""
        if self._config is None:
            return

        handler = logging.handlers.RotatingFileHandler(
            self._config.error_file_path,
            maxBytes=self._config.error_file_max_bytes,
            backupCount=self._config.error_file_backup_count,
            encoding="utf-8",
        )
        handler.setLevel(logging.ERROR)

        # Use detailed formatter for error file
        handler.setFormatter(self._formatters["detailed"])

        # Add level filter
        handler.addFilter(LevelRangeFilter(min_level=logging.ERROR))

        logger.addHandler(handler)
        self._handlers["error_file"] = handler

    def _add_json_file_handler(self, logger: logging.Logger) -> None:
        """Add JSON file handler to logger."""
        if self._config is None:
            return

        handler = logging.handlers.RotatingFileHandler(
            self._config.json_file_path,
            maxBytes=self._config.json_file_max_bytes,
            backupCount=self._config.json_file_backup_count,
            encoding="utf-8",
        )
        handler.setLevel(
            (self._config.file_level or self._config.level).to_logging_level()
        )

        # Use JSON formatter
        handler.setFormatter(self._formatters["json"])

        logger.addHandler(handler)
        self._handlers["json_file"] = handler

    def get_logger(self, name: str) -> TradingLogger:
        """
        Get a trading logger instance.

        Args:
            name: Logger name

        Returns:
            TradingLogger instance
        """
        logger = logging.getLogger(name)

        # Ensure it's a TradingLogger
        if not isinstance(logger, TradingLogger):
            # Create new TradingLogger
            logger = TradingLogger(name)

        # Add performance logger if enabled
        if self._config and self._config.performance_logging:
            if name not in self._performance_loggers:
                self._performance_loggers[name] = PerformanceLogger(
                    logger,
                    self._config.slow_operation_threshold_ms,
                )
            logger.set_performance_logger(self._performance_loggers[name])

        return logger

    def get_performance_logger(self, name: str) -> PerformanceLogger | None:
        """Get performance logger for a module."""
        return self._performance_loggers.get(name)

    def set_level(self, level: LogLevel, logger_name: str | None = None) -> None:
        """
        Set log level.

        Args:
            level: New log level
            logger_name: Optional specific logger name
        """
        if logger_name:
            logging.getLogger(logger_name).setLevel(level.to_logging_level())
        else:
            logging.getLogger().setLevel(level.to_logging_level())

    def add_handler(
        self,
        handler: logging.Handler,
        name: str,
        logger_name: str | None = None,
    ) -> None:
        """
        Add a custom handler.

        Args:
            handler: Handler to add
            name: Handler name
            logger_name: Optional specific logger name
        """
        logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
        logger.addHandler(handler)
        self._handlers[name] = handler

    def remove_handler(self, name: str, logger_name: str | None = None) -> None:
        """
        Remove a handler.

        Args:
            name: Handler name
            logger_name: Optional specific logger name
        """
        if name in self._handlers:
            handler = self._handlers.pop(name)
            logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
            logger.removeHandler(handler)

    def get_stats(self) -> dict[str, Any]:
        """Get logging statistics."""
        return {
            "configured": self._initialized,
            "handlers": list(self._handlers.keys()),
            "formatters": list(self._formatters.keys()),
            "performance_loggers": list(self._performance_loggers.keys()),
            "root_level": logging.getLevelName(logging.getLogger().level),
        }

    def shutdown(self) -> None:
        """Shutdown logging system."""
        logging.shutdown()
        self._handlers.clear()
        self._formatters.clear()
        self._performance_loggers.clear()
        self._initialized = False


# Module-level convenience functions
_manager: LoggingManager | None = None


def configure_logging(config: LoggingConfig | None = None) -> LoggingManager:
    """
    Configure the logging system.

    Args:
        config: Logging configuration

    Returns:
        LoggingManager instance
    """
    global _manager
    _manager = LoggingManager()
    _manager.configure(config)
    return _manager


def get_logger(name: str) -> TradingLogger:
    """
    Get a trading logger instance.

    Args:
        name: Logger name

    Returns:
        TradingLogger instance
    """
    global _manager
    if _manager is None:
        _manager = LoggingManager()
        _manager.configure()

    return _manager.get_logger(name)


def set_context(**kwargs: Any) -> None:
    """
    Set context variables for the current thread.

    Args:
        **kwargs: Context key-value pairs
    """
    for key, value in kwargs.items():
        ContextVar.set(key, value)


def clear_context(key: str | None = None) -> None:
    """
    Clear context variable(s).

    Args:
        key: Optional specific key to clear
    """
    ContextVar.clear(key)


def get_performance_logger(name: str) -> PerformanceLogger | None:
    """
    Get performance logger for a module.

    Args:
        name: Module name

    Returns:
        PerformanceLogger instance or None
    """
    global _manager
    if _manager is None:
        return None
    return _manager.get_performance_logger(name)


# Module version
__version__ = "2.2.0"
