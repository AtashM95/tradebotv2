"""
Constants Module for Ultimate Trading Bot v2.2.

This module defines all constant values used throughout the trading bot
including trading parameters, technical indicators, time periods, and more.
"""

from enum import Enum, IntEnum
from typing import Dict, List, Set, Tuple, FrozenSet
from dataclasses import dataclass
from decimal import Decimal


# =============================================================================
# VERSION INFORMATION
# =============================================================================

VERSION = "2.2.0"
VERSION_TUPLE = (2, 2, 0)
BUILD_DATE = "2024-01-15"
AUTHOR = "Ultimate Trading Bot Team"


# =============================================================================
# TRADING ENUMS
# =============================================================================

class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(str, Enum):
    """Order status enumeration."""
    NEW = "new"
    PENDING_NEW = "pending_new"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    PENDING_CANCEL = "pending_cancel"
    REJECTED = "rejected"
    EXPIRED = "expired"
    REPLACED = "replaced"
    PENDING_REPLACE = "pending_replace"
    DONE_FOR_DAY = "done_for_day"


class PositionSide(str, Enum):
    """Position side enumeration."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class TimeInForce(str, Enum):
    """Time in force enumeration."""
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"
    OPG = "opg"
    CLS = "cls"
    OTO = "oto"
    OCO = "oco"


class AssetClass(str, Enum):
    """Asset class enumeration."""
    US_EQUITY = "us_equity"
    CRYPTO = "crypto"
    OPTIONS = "options"
    FUTURES = "futures"


class MarketSession(str, Enum):
    """Market session enumeration."""
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    AFTER_HOURS = "after_hours"
    CLOSED = "closed"
    EXTENDED = "extended"


class SignalType(str, Enum):
    """Trading signal type enumeration."""
    BUY = "buy"
    SELL = "sell"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"
    HOLD = "hold"
    CLOSE = "close"
    NO_SIGNAL = "no_signal"


class TrendDirection(str, Enum):
    """Trend direction enumeration."""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


class VolatilityRegime(str, Enum):
    """Volatility regime enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class MarketCondition(str, Enum):
    """Market condition enumeration."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    VOLATILE = "volatile"
    RANGING = "ranging"


class StrategyMode(str, Enum):
    """Strategy mode enumeration."""
    AGGRESSIVE = "aggressive"
    MODERATE = "moderate"
    CONSERVATIVE = "conservative"
    ADAPTIVE = "adaptive"


# =============================================================================
# TIMEFRAME CONSTANTS
# =============================================================================

class Timeframe(str, Enum):
    """Timeframe enumeration."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1mo"


TIMEFRAME_MINUTES: Dict[Timeframe, int] = {
    Timeframe.MINUTE_1: 1,
    Timeframe.MINUTE_5: 5,
    Timeframe.MINUTE_15: 15,
    Timeframe.MINUTE_30: 30,
    Timeframe.HOUR_1: 60,
    Timeframe.HOUR_4: 240,
    Timeframe.DAY_1: 1440,
    Timeframe.WEEK_1: 10080,
    Timeframe.MONTH_1: 43200,
}


# =============================================================================
# TECHNICAL INDICATOR CONSTANTS
# =============================================================================

@dataclass(frozen=True)
class IndicatorDefaults:
    """Default values for technical indicators."""

    # Moving Averages
    SMA_FAST: int = 10
    SMA_SLOW: int = 50
    SMA_TREND: int = 200
    EMA_FAST: int = 12
    EMA_SLOW: int = 26
    EMA_SIGNAL: int = 9

    # RSI
    RSI_PERIOD: int = 14
    RSI_OVERBOUGHT: float = 70.0
    RSI_OVERSOLD: float = 30.0

    # MACD
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9

    # Bollinger Bands
    BB_PERIOD: int = 20
    BB_STD_DEV: float = 2.0

    # Stochastic
    STOCH_K_PERIOD: int = 14
    STOCH_D_PERIOD: int = 3
    STOCH_OVERBOUGHT: float = 80.0
    STOCH_OVERSOLD: float = 20.0

    # ATR
    ATR_PERIOD: int = 14

    # ADX
    ADX_PERIOD: int = 14
    ADX_TREND_THRESHOLD: float = 25.0

    # CCI
    CCI_PERIOD: int = 20
    CCI_OVERBOUGHT: float = 100.0
    CCI_OVERSOLD: float = -100.0

    # Williams %R
    WILLIAMS_PERIOD: int = 14
    WILLIAMS_OVERBOUGHT: float = -20.0
    WILLIAMS_OVERSOLD: float = -80.0

    # Ichimoku
    ICHIMOKU_TENKAN: int = 9
    ICHIMOKU_KIJUN: int = 26
    ICHIMOKU_SENKOU_B: int = 52

    # Pivot Points
    PIVOT_PERIOD: int = 1

    # Volume
    VOLUME_MA_PERIOD: int = 20
    OBV_PERIOD: int = 20
    VWAP_PERIOD: int = 1


INDICATOR_DEFAULTS = IndicatorDefaults()


# =============================================================================
# RISK MANAGEMENT CONSTANTS
# =============================================================================

@dataclass(frozen=True)
class RiskDefaults:
    """Default values for risk management."""

    MAX_POSITION_SIZE_PERCENT: float = 10.0
    MAX_PORTFOLIO_RISK_PERCENT: float = 2.0
    MAX_SINGLE_TRADE_RISK_PERCENT: float = 1.0
    MAX_CORRELATED_POSITIONS: int = 3
    MAX_SECTOR_EXPOSURE_PERCENT: float = 30.0
    MIN_RISK_REWARD_RATIO: float = 2.0
    DEFAULT_STOP_LOSS_PERCENT: float = 2.0
    DEFAULT_TAKE_PROFIT_PERCENT: float = 4.0
    TRAILING_STOP_ACTIVATION_PERCENT: float = 1.5
    TRAILING_STOP_DISTANCE_PERCENT: float = 1.0
    MAX_DRAWDOWN_PERCENT: float = 20.0
    DAILY_LOSS_LIMIT_PERCENT: float = 5.0
    WEEKLY_LOSS_LIMIT_PERCENT: float = 10.0
    MONTHLY_LOSS_LIMIT_PERCENT: float = 15.0
    VAR_CONFIDENCE_LEVEL: float = 0.95
    VAR_LOOKBACK_DAYS: int = 252
    CORRELATION_THRESHOLD: float = 0.7
    VOLATILITY_LOOKBACK: int = 20


RISK_DEFAULTS = RiskDefaults()


# =============================================================================
# MARKET DATA CONSTANTS
# =============================================================================

# Standard trading hours (Eastern Time)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0
PRE_MARKET_OPEN_HOUR = 4
PRE_MARKET_OPEN_MINUTE = 0
AFTER_HOURS_CLOSE_HOUR = 20
AFTER_HOURS_CLOSE_MINUTE = 0

# Market holidays (US markets) - dates as (month, day) tuples
MARKET_HOLIDAYS_FIXED: FrozenSet[Tuple[int, int]] = frozenset({
    (1, 1),    # New Year's Day
    (7, 4),    # Independence Day
    (12, 25),  # Christmas Day
})

# Common indices and benchmarks
MAJOR_INDICES: Dict[str, str] = {
    "SPY": "S&P 500 ETF",
    "QQQ": "NASDAQ 100 ETF",
    "DIA": "Dow Jones ETF",
    "IWM": "Russell 2000 ETF",
    "VTI": "Total Stock Market ETF",
}

# Sector ETFs
SECTOR_ETFS: Dict[str, str] = {
    "XLK": "Technology",
    "XLF": "Financial",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLI": "Industrial",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

# Most traded stocks
POPULAR_STOCKS: List[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK.B",
    "JPM", "JNJ", "V", "UNH", "HD", "PG", "MA", "DIS", "BAC", "ADBE",
    "CRM", "NFLX", "PYPL", "INTC", "AMD", "CSCO", "PFE", "ABT", "KO",
    "PEP", "NKE", "MCD", "WMT", "CVX", "XOM", "VZ", "T", "CMCSA"
]


# =============================================================================
# API RATE LIMITS
# =============================================================================

@dataclass(frozen=True)
class RateLimits:
    """API rate limit constants."""

    # Alpaca
    ALPACA_REQUESTS_PER_MINUTE: int = 200
    ALPACA_DATA_REQUESTS_PER_MINUTE: int = 200

    # OpenAI
    OPENAI_REQUESTS_PER_MINUTE: int = 60
    OPENAI_TOKENS_PER_MINUTE: int = 90000

    # Yahoo Finance (unofficial)
    YFINANCE_REQUESTS_PER_HOUR: int = 2000

    # Internal
    WEBSOCKET_PING_INTERVAL: int = 30
    CACHE_TTL_SECONDS: int = 300


RATE_LIMITS = RateLimits()


# =============================================================================
# DATA PROCESSING CONSTANTS
# =============================================================================

# Minimum data requirements
MIN_BARS_FOR_INDICATORS: int = 200
MIN_BARS_FOR_BACKTEST: int = 252
MIN_BARS_FOR_ML_TRAINING: int = 1000

# Maximum data limits
MAX_BARS_PER_REQUEST: int = 10000
MAX_SYMBOLS_PER_REQUEST: int = 200
MAX_NEWS_ARTICLES: int = 100

# Data quality thresholds
MAX_MISSING_DATA_PERCENT: float = 5.0
MIN_VOLUME_THRESHOLD: int = 100000
MIN_PRICE_THRESHOLD: float = 1.0

# Cache settings
CACHE_QUOTES_TTL: int = 1
CACHE_BARS_TTL: int = 60
CACHE_NEWS_TTL: int = 300
CACHE_ANALYSIS_TTL: int = 600


# =============================================================================
# NOTIFICATION CONSTANTS
# =============================================================================

class NotificationLevel(IntEnum):
    """Notification importance level."""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


class NotificationType(str, Enum):
    """Notification type enumeration."""
    TRADE = "trade"
    SIGNAL = "signal"
    ALERT = "alert"
    ERROR = "error"
    SYSTEM = "system"
    REPORT = "report"


# =============================================================================
# FILE AND PATH CONSTANTS
# =============================================================================

# File extensions
DATA_FILE_EXTENSIONS: FrozenSet[str] = frozenset({".csv", ".parquet", ".json", ".pickle"})
MODEL_FILE_EXTENSIONS: FrozenSet[str] = frozenset({".pkl", ".joblib", ".h5", ".pt", ".pth"})
CONFIG_FILE_EXTENSIONS: FrozenSet[str] = frozenset({".yaml", ".yml", ".json", ".toml"})

# Directory names
DATA_DIR = "data"
LOGS_DIR = "logs"
MODELS_DIR = "models"
CACHE_DIR = "cache"
REPORTS_DIR = "reports"
BACKTEST_DIR = "backtests"
EXPORTS_DIR = "exports"


# =============================================================================
# ML/AI CONSTANTS
# =============================================================================

@dataclass(frozen=True)
class MLDefaults:
    """Default values for ML models."""

    # Feature engineering
    LOOKBACK_PERIODS: Tuple[int, ...] = (5, 10, 20, 50, 100, 200)
    LAG_PERIODS: Tuple[int, ...] = (1, 2, 3, 5, 10)

    # Training
    TRAIN_TEST_SPLIT: float = 0.8
    VALIDATION_SPLIT: float = 0.1
    CROSS_VALIDATION_FOLDS: int = 5
    EARLY_STOPPING_PATIENCE: int = 10
    MAX_EPOCHS: int = 100
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001

    # LSTM specific
    LSTM_UNITS: int = 128
    LSTM_DROPOUT: float = 0.2
    LSTM_SEQUENCE_LENGTH: int = 60

    # XGBoost specific
    XGB_N_ESTIMATORS: int = 100
    XGB_MAX_DEPTH: int = 6
    XGB_LEARNING_RATE: float = 0.1

    # Random Forest
    RF_N_ESTIMATORS: int = 100
    RF_MAX_DEPTH: int = 10


ML_DEFAULTS = MLDefaults()


# =============================================================================
# SENTIMENT ANALYSIS CONSTANTS
# =============================================================================

@dataclass(frozen=True)
class SentimentThresholds:
    """Sentiment analysis thresholds."""

    VERY_BULLISH: float = 0.6
    BULLISH: float = 0.2
    NEUTRAL_UPPER: float = 0.2
    NEUTRAL_LOWER: float = -0.2
    BEARISH: float = -0.2
    VERY_BEARISH: float = -0.6


SENTIMENT_THRESHOLDS = SentimentThresholds()


# =============================================================================
# ERROR CODES
# =============================================================================

class ErrorCode(IntEnum):
    """Error code enumeration."""

    # General errors (1xxx)
    UNKNOWN_ERROR = 1000
    CONFIGURATION_ERROR = 1001
    INITIALIZATION_ERROR = 1002
    VALIDATION_ERROR = 1003

    # Data errors (2xxx)
    DATA_FETCH_ERROR = 2000
    DATA_PARSE_ERROR = 2001
    DATA_MISSING_ERROR = 2002
    DATA_INVALID_ERROR = 2003

    # Trading errors (3xxx)
    ORDER_ERROR = 3000
    ORDER_REJECTED = 3001
    INSUFFICIENT_FUNDS = 3002
    POSITION_ERROR = 3003
    RISK_LIMIT_EXCEEDED = 3004

    # API errors (4xxx)
    API_CONNECTION_ERROR = 4000
    API_AUTHENTICATION_ERROR = 4001
    API_RATE_LIMIT_ERROR = 4002
    API_TIMEOUT_ERROR = 4003

    # AI errors (5xxx)
    AI_REQUEST_ERROR = 5000
    AI_RESPONSE_ERROR = 5001
    AI_BUDGET_EXCEEDED = 5002

    # Database errors (6xxx)
    DATABASE_CONNECTION_ERROR = 6000
    DATABASE_QUERY_ERROR = 6001
    DATABASE_WRITE_ERROR = 6002


# =============================================================================
# HTTP STATUS CODES (for API)
# =============================================================================

class HTTPStatus(IntEnum):
    """HTTP status codes."""

    OK = 200
    CREATED = 201
    ACCEPTED = 202
    NO_CONTENT = 204
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    CONFLICT = 409
    UNPROCESSABLE_ENTITY = 422
    TOO_MANY_REQUESTS = 429
    INTERNAL_SERVER_ERROR = 500
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503


# =============================================================================
# MISC CONSTANTS
# =============================================================================

# Decimal precision
PRICE_PRECISION: int = 2
QUANTITY_PRECISION: int = 6
PERCENT_PRECISION: int = 4

# Retry settings
MAX_RETRIES: int = 3
RETRY_DELAY_SECONDS: float = 1.0
RETRY_BACKOFF_FACTOR: float = 2.0

# Timeouts
DEFAULT_TIMEOUT_SECONDS: float = 30.0
LONG_TIMEOUT_SECONDS: float = 120.0
WEBSOCKET_TIMEOUT_SECONDS: float = 60.0

# Queue sizes
EVENT_QUEUE_SIZE: int = 10000
ORDER_QUEUE_SIZE: int = 1000
DATA_QUEUE_SIZE: int = 50000

# Batch sizes
SYMBOL_BATCH_SIZE: int = 50
ORDER_BATCH_SIZE: int = 100
DATA_BATCH_SIZE: int = 1000
