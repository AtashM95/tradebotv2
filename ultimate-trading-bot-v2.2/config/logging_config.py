"""
Logging Configuration Module for Ultimate Trading Bot v2.2.

This module provides comprehensive logging configuration including
structured logging, file rotation, and specialized loggers for different
components of the trading bot.
"""

import logging
import logging.handlers
import sys
import json
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime, timezone
from enum import Enum
import threading
import queue
import atexit

from pydantic import BaseModel, Field


class LogFormat(str, Enum):
    """Log format enumeration."""
    SIMPLE = "simple"
    DETAILED = "detailed"
    JSON = "json"


class LogDestination(str, Enum):
    """Log destination enumeration."""
    CONSOLE = "console"
    FILE = "file"
    BOTH = "both"


class LoggingConfig(BaseModel):
    """Logging configuration model."""

    level: str = Field(default="INFO", description="Default log level")
    format: LogFormat = Field(default=LogFormat.DETAILED, description="Log format")
    destination: LogDestination = Field(default=LogDestination.BOTH, description="Log destination")
    log_dir: Path = Field(default=Path("logs"), description="Log directory")
    max_bytes: int = Field(default=10 * 1024 * 1024, description="Max log file size (10MB)")
    backup_count: int = Field(default=10, description="Number of backup files")
    enable_async: bool = Field(default=True, description="Enable async logging")
    enable_structured: bool = Field(default=True, description="Enable structured logging")
    include_stack_trace: bool = Field(default=True, description="Include stack traces")
    colorize_console: bool = Field(default=True, description="Colorize console output")


# Log format strings
SIMPLE_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DETAILED_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)-25s | "
    "%(filename)s:%(lineno)d | %(funcName)s | %(message)s"
)

# Color codes for console output
COLORS = {
    'DEBUG': '\033[36m',      # Cyan
    'INFO': '\033[32m',       # Green
    'WARNING': '\033[33m',    # Yellow
    'ERROR': '\033[31m',      # Red
    'CRITICAL': '\033[35m',   # Magenta
    'RESET': '\033[0m'        # Reset
}


class ColorizedFormatter(logging.Formatter):
    """Formatter that adds colors to log levels for console output."""

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        colorize: bool = True
    ) -> None:
        """
        Initialize the colorized formatter.

        Args:
            fmt: Log format string
            datefmt: Date format string
            colorize: Whether to colorize output
        """
        super().__init__(fmt, datefmt)
        self.colorize = colorize

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with colors.

        Args:
            record: Log record to format

        Returns:
            Formatted log string
        """
        message = super().format(record)
        if self.colorize and record.levelname in COLORS:
            color = COLORS[record.levelname]
            reset = COLORS['RESET']
            message = f"{color}{message}{reset}"
        return message


class JSONFormatter(logging.Formatter):
    """Formatter that outputs logs as JSON."""

    def __init__(
        self,
        include_stack_trace: bool = True,
        extra_fields: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize the JSON formatter.

        Args:
            include_stack_trace: Whether to include stack traces
            extra_fields: Extra fields to include in all log entries
        """
        super().__init__()
        self.include_stack_trace = include_stack_trace
        self.extra_fields = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON formatted log string
        """
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "thread_name": record.threadName,
            "process": record.process,
        }

        # Add extra fields from record
        if hasattr(record, 'extra_data'):
            log_entry['extra'] = record.extra_data

        # Add configured extra fields
        log_entry.update(self.extra_fields)

        # Add exception info if present
        if record.exc_info and self.include_stack_trace:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info)
            }

        return json.dumps(log_entry, default=str)


class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler that processes logs in a separate thread."""

    def __init__(self, handler: logging.Handler) -> None:
        """
        Initialize the async handler.

        Args:
            handler: Underlying handler to use
        """
        super().__init__()
        self.handler = handler
        self.queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._process_logs, daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def _process_logs(self) -> None:
        """Process logs from the queue in a separate thread."""
        while not self._stop_event.is_set():
            try:
                record = self.queue.get(timeout=0.5)
                if record is not None:
                    self.handler.emit(record)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                pass

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record.

        Args:
            record: Log record to emit
        """
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            # Queue is full, drop the record
            pass

    def close(self) -> None:
        """Close the handler and stop the processing thread."""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.handler.close()
        super().close()


class StructuredLogger(logging.Logger):
    """Logger that supports structured logging with extra data."""

    def _log_with_extra(
        self,
        level: int,
        msg: str,
        args: tuple,
        exc_info: Any = None,
        extra: Optional[Dict[str, Any]] = None,
        stack_info: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Log with extra structured data.

        Args:
            level: Log level
            msg: Log message
            args: Message arguments
            exc_info: Exception info
            extra: Extra data to include
            stack_info: Whether to include stack info
            **kwargs: Additional keyword arguments
        """
        if extra is None:
            extra = {}
        extra['extra_data'] = kwargs
        super()._log(level, msg, args, exc_info=exc_info, extra=extra, stack_info=stack_info)

    def debug_structured(self, msg: str, **kwargs: Any) -> None:
        """Log debug message with structured data."""
        self._log_with_extra(logging.DEBUG, msg, (), **kwargs)

    def info_structured(self, msg: str, **kwargs: Any) -> None:
        """Log info message with structured data."""
        self._log_with_extra(logging.INFO, msg, (), **kwargs)

    def warning_structured(self, msg: str, **kwargs: Any) -> None:
        """Log warning message with structured data."""
        self._log_with_extra(logging.WARNING, msg, (), **kwargs)

    def error_structured(self, msg: str, exc_info: bool = False, **kwargs: Any) -> None:
        """Log error message with structured data."""
        self._log_with_extra(logging.ERROR, msg, (), exc_info=exc_info, **kwargs)

    def critical_structured(self, msg: str, exc_info: bool = True, **kwargs: Any) -> None:
        """Log critical message with structured data."""
        self._log_with_extra(logging.CRITICAL, msg, (), exc_info=exc_info, **kwargs)


# Register structured logger class
logging.setLoggerClass(StructuredLogger)


class TradeLogger:
    """Specialized logger for trade-related events."""

    def __init__(self, logger: logging.Logger) -> None:
        """
        Initialize trade logger.

        Args:
            logger: Underlying logger instance
        """
        self.logger = logger

    def log_order_submitted(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        price: Optional[float] = None,
        order_id: Optional[str] = None
    ) -> None:
        """Log order submission."""
        self.logger.info(
            f"ORDER SUBMITTED: {side.upper()} {quantity} {symbol} @ {order_type}",
            extra={'extra_data': {
                'event': 'order_submitted',
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'order_type': order_type,
                'price': price,
                'order_id': order_id
            }}
        )

    def log_order_filled(
        self,
        symbol: str,
        side: str,
        quantity: float,
        fill_price: float,
        order_id: str
    ) -> None:
        """Log order fill."""
        self.logger.info(
            f"ORDER FILLED: {side.upper()} {quantity} {symbol} @ ${fill_price:.2f}",
            extra={'extra_data': {
                'event': 'order_filled',
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'fill_price': fill_price,
                'order_id': order_id
            }}
        )

    def log_order_cancelled(self, order_id: str, reason: str = "") -> None:
        """Log order cancellation."""
        self.logger.info(
            f"ORDER CANCELLED: {order_id} - {reason}",
            extra={'extra_data': {
                'event': 'order_cancelled',
                'order_id': order_id,
                'reason': reason
            }}
        )

    def log_position_opened(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float
    ) -> None:
        """Log position opened."""
        self.logger.info(
            f"POSITION OPENED: {side.upper()} {quantity} {symbol} @ ${entry_price:.2f}",
            extra={'extra_data': {
                'event': 'position_opened',
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'entry_price': entry_price
            }}
        )

    def log_position_closed(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_percent: float
    ) -> None:
        """Log position closed."""
        self.logger.info(
            f"POSITION CLOSED: {symbol} PnL: ${pnl:.2f} ({pnl_percent:.2f}%)",
            extra={'extra_data': {
                'event': 'position_closed',
                'symbol': symbol,
                'quantity': quantity,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_percent': pnl_percent
            }}
        )

    def log_signal_generated(
        self,
        symbol: str,
        signal_type: str,
        strength: float,
        strategy: str
    ) -> None:
        """Log trading signal generation."""
        self.logger.info(
            f"SIGNAL: {signal_type.upper()} {symbol} (strength: {strength:.2f}) - {strategy}",
            extra={'extra_data': {
                'event': 'signal_generated',
                'symbol': symbol,
                'signal_type': signal_type,
                'strength': strength,
                'strategy': strategy
            }}
        )


class LoggingManager:
    """
    Central logging manager for the trading bot.

    Manages all loggers, handlers, and formatters for the application.
    """

    _instance: Optional['LoggingManager'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'LoggingManager':
        """Singleton pattern for logging manager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the logging manager."""
        if self._initialized:
            return

        self.config: Optional[LoggingConfig] = None
        self.handlers: Dict[str, logging.Handler] = {}
        self.loggers: Dict[str, logging.Logger] = {}
        self._trade_logger: Optional[TradeLogger] = None
        self._initialized = True

    def setup(self, config: Optional[LoggingConfig] = None) -> None:
        """
        Set up logging with the given configuration.

        Args:
            config: Logging configuration
        """
        self.config = config or LoggingConfig()

        # Ensure log directory exists
        self.config.log_dir.mkdir(parents=True, exist_ok=True)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.config.level)

        # Clear existing handlers
        root_logger.handlers.clear()

        # Set up console handler
        if self.config.destination in (LogDestination.CONSOLE, LogDestination.BOTH):
            console_handler = self._create_console_handler()
            root_logger.addHandler(console_handler)
            self.handlers['console'] = console_handler

        # Set up file handlers
        if self.config.destination in (LogDestination.FILE, LogDestination.BOTH):
            file_handler = self._create_file_handler("trading_bot.log")
            root_logger.addHandler(file_handler)
            self.handlers['main_file'] = file_handler

            # Error-only file handler
            error_handler = self._create_file_handler("errors.log")
            error_handler.setLevel(logging.ERROR)
            root_logger.addHandler(error_handler)
            self.handlers['error_file'] = error_handler

        # Set up specialized loggers
        self._setup_specialized_loggers()

    def _create_console_handler(self) -> logging.Handler:
        """Create console handler with appropriate formatter."""
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(self.config.level)

        if self.config.format == LogFormat.JSON:
            formatter = JSONFormatter(include_stack_trace=self.config.include_stack_trace)
        elif self.config.format == LogFormat.DETAILED:
            formatter = ColorizedFormatter(
                DETAILED_FORMAT,
                colorize=self.config.colorize_console
            )
        else:
            formatter = ColorizedFormatter(
                SIMPLE_FORMAT,
                colorize=self.config.colorize_console
            )

        handler.setFormatter(formatter)
        return handler

    def _create_file_handler(self, filename: str) -> logging.Handler:
        """Create file handler with rotation."""
        filepath = self.config.log_dir / filename
        handler = logging.handlers.RotatingFileHandler(
            filepath,
            maxBytes=self.config.max_bytes,
            backupCount=self.config.backup_count,
            encoding='utf-8'
        )
        handler.setLevel(self.config.level)

        if self.config.format == LogFormat.JSON:
            formatter = JSONFormatter(include_stack_trace=self.config.include_stack_trace)
        else:
            formatter = logging.Formatter(DETAILED_FORMAT)

        handler.setFormatter(formatter)

        if self.config.enable_async:
            handler = AsyncLogHandler(handler)

        return handler

    def _setup_specialized_loggers(self) -> None:
        """Set up specialized loggers for different components."""
        loggers_config = {
            'trading_bot.core': logging.INFO,
            'trading_bot.data': logging.INFO,
            'trading_bot.strategies': logging.INFO,
            'trading_bot.execution': logging.INFO,
            'trading_bot.risk': logging.INFO,
            'trading_bot.ai': logging.INFO,
            'trading_bot.backtest': logging.INFO,
            'trading_bot.ml': logging.INFO,
            'trading_bot.trades': logging.INFO,
            'trading_bot.api': logging.INFO,
            'trading_bot.ui': logging.INFO,
        }

        for logger_name, level in loggers_config.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(level)
            self.loggers[logger_name] = logger

        # Set up trade logger
        trade_logger = logging.getLogger('trading_bot.trades')
        trade_file_handler = self._create_file_handler("trades.log")
        trade_logger.addHandler(trade_file_handler)
        self._trade_logger = TradeLogger(trade_logger)

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger by name.

        Args:
            name: Logger name

        Returns:
            Logger instance
        """
        if name not in self.loggers:
            self.loggers[name] = logging.getLogger(name)
        return self.loggers[name]

    def get_trade_logger(self) -> TradeLogger:
        """
        Get the specialized trade logger.

        Returns:
            TradeLogger instance
        """
        if self._trade_logger is None:
            trade_logger = logging.getLogger('trading_bot.trades')
            self._trade_logger = TradeLogger(trade_logger)
        return self._trade_logger

    def set_level(self, level: Union[str, int]) -> None:
        """
        Set log level for all loggers.

        Args:
            level: Log level (string or int)
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper())

        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        for handler in root_logger.handlers:
            handler.setLevel(level)

        for logger in self.loggers.values():
            logger.setLevel(level)

    def add_context(self, **context: Any) -> None:
        """
        Add context to all log entries.

        Args:
            **context: Context key-value pairs
        """
        for handler in logging.getLogger().handlers:
            if isinstance(handler.formatter, JSONFormatter):
                handler.formatter.extra_fields.update(context)


def setup_logging(config: Optional[LoggingConfig] = None) -> LoggingManager:
    """
    Set up logging with the given configuration.

    Args:
        config: Logging configuration

    Returns:
        LoggingManager instance
    """
    manager = LoggingManager()
    manager.setup(config)
    return manager


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger by name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return LoggingManager().get_logger(name)


def get_trade_logger() -> TradeLogger:
    """
    Get the specialized trade logger.

    Returns:
        TradeLogger instance
    """
    return LoggingManager().get_trade_logger()


# Default logger for the module
logger = logging.getLogger('trading_bot')
