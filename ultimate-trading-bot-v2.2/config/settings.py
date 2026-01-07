"""
Main Settings Module for Ultimate Trading Bot v2.2.

This module provides centralized configuration management for the entire trading bot,
including environment variables, database settings, API configurations, and runtime options.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from enum import Enum
from functools import lru_cache
from datetime import time, timezone
import logging

from pydantic import BaseModel, Field, field_validator, model_validator, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment enumeration."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class TradingMode(str, Enum):
    """Trading mode enumeration."""
    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"
    SIMULATION = "simulation"


class LogLevel(str, Enum):
    """Logging level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseType(str, Enum):
    """Database type enumeration."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"


class DatabaseSettings(BaseModel):
    """Database configuration settings."""

    type: DatabaseType = Field(default=DatabaseType.SQLITE, description="Database type")
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(default="trading_bot", description="Database name")
    user: str = Field(default="", description="Database user")
    password: SecretStr = Field(default=SecretStr(""), description="Database password")
    pool_size: int = Field(default=10, ge=1, le=100, description="Connection pool size")
    max_overflow: int = Field(default=20, ge=0, le=100, description="Max overflow connections")
    echo: bool = Field(default=False, description="Echo SQL queries")
    sqlite_path: Path = Field(default=Path("data/trading_bot.db"), description="SQLite file path")

    @property
    def connection_string(self) -> str:
        """Generate database connection string."""
        if self.type == DatabaseType.SQLITE:
            return f"sqlite:///{self.sqlite_path}"
        elif self.type == DatabaseType.POSTGRESQL:
            return (
                f"postgresql://{self.user}:{self.password.get_secret_value()}"
                f"@{self.host}:{self.port}/{self.name}"
            )
        elif self.type == DatabaseType.MYSQL:
            return (
                f"mysql+pymysql://{self.user}:{self.password.get_secret_value()}"
                f"@{self.host}:{self.port}/{self.name}"
            )
        return ""

    @property
    def async_connection_string(self) -> str:
        """Generate async database connection string."""
        if self.type == DatabaseType.SQLITE:
            return f"sqlite+aiosqlite:///{self.sqlite_path}"
        elif self.type == DatabaseType.POSTGRESQL:
            return (
                f"postgresql+asyncpg://{self.user}:{self.password.get_secret_value()}"
                f"@{self.host}:{self.port}/{self.name}"
            )
        elif self.type == DatabaseType.MYSQL:
            return (
                f"mysql+aiomysql://{self.user}:{self.password.get_secret_value()}"
                f"@{self.host}:{self.port}/{self.name}"
            )
        return ""


class RedisSettings(BaseModel):
    """Redis cache configuration settings."""

    enabled: bool = Field(default=True, description="Enable Redis caching")
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, ge=1, le=65535, description="Redis port")
    db: int = Field(default=0, ge=0, le=15, description="Redis database number")
    password: SecretStr = Field(default=SecretStr(""), description="Redis password")
    max_connections: int = Field(default=50, ge=1, le=1000, description="Max connections")
    socket_timeout: float = Field(default=5.0, ge=0.1, le=60.0, description="Socket timeout")
    socket_connect_timeout: float = Field(default=5.0, ge=0.1, le=60.0, description="Connect timeout")
    retry_on_timeout: bool = Field(default=True, description="Retry on timeout")
    decode_responses: bool = Field(default=True, description="Decode responses to strings")
    ssl: bool = Field(default=False, description="Use SSL connection")

    @property
    def connection_url(self) -> str:
        """Generate Redis connection URL."""
        protocol = "rediss" if self.ssl else "redis"
        auth = f":{self.password.get_secret_value()}@" if self.password.get_secret_value() else ""
        return f"{protocol}://{auth}{self.host}:{self.port}/{self.db}"


class AlpacaSettings(BaseModel):
    """Alpaca broker API configuration settings."""

    api_key: SecretStr = Field(default=SecretStr(""), description="Alpaca API key")
    api_secret: SecretStr = Field(default=SecretStr(""), description="Alpaca API secret")
    base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        description="Alpaca base URL"
    )
    data_url: str = Field(
        default="https://data.alpaca.markets",
        description="Alpaca data URL"
    )
    stream_url: str = Field(
        default="wss://stream.data.alpaca.markets",
        description="Alpaca streaming URL"
    )
    paper_trading: bool = Field(default=True, description="Use paper trading")
    max_retries: int = Field(default=3, ge=0, le=10, description="Max API retries")
    retry_delay: float = Field(default=1.0, ge=0.1, le=30.0, description="Retry delay seconds")
    timeout: float = Field(default=30.0, ge=1.0, le=120.0, description="API timeout seconds")

    @model_validator(mode='after')
    def set_urls_based_on_mode(self) -> 'AlpacaSettings':
        """Set appropriate URLs based on paper_trading mode."""
        if self.paper_trading:
            self.base_url = "https://paper-api.alpaca.markets"
        else:
            self.base_url = "https://api.alpaca.markets"
        return self

    @property
    def is_configured(self) -> bool:
        """Check if Alpaca credentials are configured."""
        return bool(self.api_key.get_secret_value() and self.api_secret.get_secret_value())


class OpenAISettings(BaseModel):
    """OpenAI API configuration settings."""

    api_key: SecretStr = Field(default=SecretStr(""), description="OpenAI API key")
    organization_id: Optional[str] = Field(default=None, description="OpenAI organization ID")
    model: str = Field(default="gpt-4o", description="Default model to use")
    vision_model: str = Field(default="gpt-4o", description="Vision model for chart analysis")
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model")
    max_tokens: int = Field(default=4096, ge=1, le=128000, description="Max tokens per request")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling")
    timeout: float = Field(default=60.0, ge=1.0, le=300.0, description="API timeout seconds")
    max_retries: int = Field(default=3, ge=0, le=10, description="Max API retries")
    request_timeout: float = Field(default=120.0, ge=1.0, le=600.0, description="Request timeout")
    daily_budget: float = Field(default=50.0, ge=0.0, description="Daily spending budget USD")
    monthly_budget: float = Field(default=500.0, ge=0.0, description="Monthly spending budget USD")
    rate_limit_rpm: int = Field(default=60, ge=1, description="Rate limit requests per minute")
    rate_limit_tpm: int = Field(default=90000, ge=1, description="Rate limit tokens per minute")

    @property
    def is_configured(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self.api_key.get_secret_value())


class TradingSessionSettings(BaseModel):
    """Trading session timing configuration."""

    market_open: time = Field(default=time(9, 30), description="Market open time (ET)")
    market_close: time = Field(default=time(16, 0), description="Market close time (ET)")
    pre_market_open: time = Field(default=time(4, 0), description="Pre-market open time (ET)")
    after_hours_close: time = Field(default=time(20, 0), description="After-hours close time (ET)")
    timezone: str = Field(default="America/New_York", description="Market timezone")
    trade_pre_market: bool = Field(default=False, description="Allow pre-market trading")
    trade_after_hours: bool = Field(default=False, description="Allow after-hours trading")
    buffer_minutes_open: int = Field(default=5, ge=0, le=60, description="Minutes after open to start")
    buffer_minutes_close: int = Field(default=5, ge=0, le=60, description="Minutes before close to stop")


class RiskManagementSettings(BaseModel):
    """Risk management configuration settings."""

    max_portfolio_risk: float = Field(default=0.02, ge=0.001, le=0.5, description="Max portfolio risk %")
    max_position_risk: float = Field(default=0.01, ge=0.001, le=0.2, description="Max position risk %")
    max_position_size: float = Field(default=0.1, ge=0.01, le=1.0, description="Max position size %")
    max_positions: int = Field(default=10, ge=1, le=100, description="Maximum concurrent positions")
    max_daily_loss: float = Field(default=0.05, ge=0.01, le=0.5, description="Max daily loss %")
    max_weekly_loss: float = Field(default=0.1, ge=0.01, le=0.5, description="Max weekly loss %")
    max_monthly_loss: float = Field(default=0.15, ge=0.01, le=0.5, description="Max monthly loss %")
    max_drawdown: float = Field(default=0.2, ge=0.01, le=0.5, description="Max drawdown %")
    default_stop_loss: float = Field(default=0.02, ge=0.005, le=0.2, description="Default stop loss %")
    default_take_profit: float = Field(default=0.04, ge=0.01, le=0.5, description="Default take profit %")
    trailing_stop_enabled: bool = Field(default=True, description="Enable trailing stops")
    trailing_stop_distance: float = Field(default=0.015, ge=0.005, le=0.1, description="Trailing stop distance %")
    var_confidence: float = Field(default=0.95, ge=0.9, le=0.99, description="VaR confidence level")
    var_lookback_days: int = Field(default=252, ge=20, le=1000, description="VaR lookback days")
    use_atr_stops: bool = Field(default=True, description="Use ATR-based stops")
    atr_stop_multiplier: float = Field(default=2.0, ge=0.5, le=5.0, description="ATR stop multiplier")


class NotificationSettings(BaseModel):
    """Notification configuration settings."""

    enabled: bool = Field(default=True, description="Enable notifications")

    # Email settings
    email_enabled: bool = Field(default=False, description="Enable email notifications")
    smtp_host: str = Field(default="smtp.gmail.com", description="SMTP host")
    smtp_port: int = Field(default=587, description="SMTP port")
    smtp_user: str = Field(default="", description="SMTP username")
    smtp_password: SecretStr = Field(default=SecretStr(""), description="SMTP password")
    email_from: str = Field(default="", description="From email address")
    email_to: List[str] = Field(default_factory=list, description="Recipient email addresses")

    # Telegram settings
    telegram_enabled: bool = Field(default=False, description="Enable Telegram notifications")
    telegram_bot_token: SecretStr = Field(default=SecretStr(""), description="Telegram bot token")
    telegram_chat_id: str = Field(default="", description="Telegram chat ID")

    # Discord settings
    discord_enabled: bool = Field(default=False, description="Enable Discord notifications")
    discord_webhook_url: SecretStr = Field(default=SecretStr(""), description="Discord webhook URL")

    # Notification levels
    notify_on_trade: bool = Field(default=True, description="Notify on trade execution")
    notify_on_error: bool = Field(default=True, description="Notify on errors")
    notify_on_signal: bool = Field(default=False, description="Notify on trading signals")
    notify_daily_summary: bool = Field(default=True, description="Send daily summary")
    notify_on_drawdown: bool = Field(default=True, description="Notify on drawdown threshold")
    drawdown_threshold: float = Field(default=0.05, ge=0.01, le=0.5, description="Drawdown notification threshold")


class BacktestSettings(BaseModel):
    """Backtesting configuration settings."""

    initial_capital: float = Field(default=100000.0, ge=1000.0, description="Initial capital for backtest")
    commission: float = Field(default=0.0, ge=0.0, le=0.01, description="Commission per trade")
    slippage: float = Field(default=0.0005, ge=0.0, le=0.01, description="Slippage estimation")
    data_source: str = Field(default="alpaca", description="Historical data source")
    benchmark_symbol: str = Field(default="SPY", description="Benchmark symbol")
    risk_free_rate: float = Field(default=0.05, ge=0.0, le=0.2, description="Risk-free rate for Sharpe")
    min_trade_size: float = Field(default=100.0, ge=1.0, description="Minimum trade size USD")
    use_fractional_shares: bool = Field(default=True, description="Allow fractional shares")
    fill_at_close: bool = Field(default=False, description="Fill orders at close price")


class UISettings(BaseModel):
    """UI/Dashboard configuration settings."""

    enabled: bool = Field(default=True, description="Enable web UI")
    host: str = Field(default="0.0.0.0", description="UI host")
    port: int = Field(default=5000, ge=1024, le=65535, description="UI port")
    debug: bool = Field(default=False, description="Enable debug mode")
    secret_key: SecretStr = Field(default=SecretStr("change-this-secret-key"), description="Flask secret key")
    session_lifetime_hours: int = Field(default=24, ge=1, le=720, description="Session lifetime hours")
    enable_auth: bool = Field(default=True, description="Enable authentication")
    admin_username: str = Field(default="admin", description="Admin username")
    admin_password: SecretStr = Field(default=SecretStr("admin"), description="Admin password")
    enable_api: bool = Field(default=True, description="Enable REST API")
    api_rate_limit: int = Field(default=100, ge=1, le=1000, description="API rate limit per minute")
    cors_origins: List[str] = Field(default_factory=lambda: ["*"], description="CORS allowed origins")


class MonitoringSettings(BaseModel):
    """System monitoring configuration settings."""

    enabled: bool = Field(default=True, description="Enable system monitoring")
    health_check_interval: int = Field(default=60, ge=10, le=600, description="Health check interval seconds")
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=9090, ge=1024, le=65535, description="Metrics port for Prometheus")
    log_performance: bool = Field(default=True, description="Log performance metrics")
    alert_on_high_cpu: bool = Field(default=True, description="Alert on high CPU usage")
    cpu_threshold: float = Field(default=80.0, ge=50.0, le=100.0, description="CPU alert threshold %")
    alert_on_high_memory: bool = Field(default=True, description="Alert on high memory usage")
    memory_threshold: float = Field(default=80.0, ge=50.0, le=100.0, description="Memory alert threshold %")
    heartbeat_interval: int = Field(default=30, ge=5, le=300, description="Heartbeat interval seconds")


class Settings(BaseSettings):
    """
    Main application settings.

    This class aggregates all configuration settings for the trading bot
    and provides a centralized way to access configuration values.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="TRADING_BOT_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore"
    )

    # General settings
    app_name: str = Field(default="Ultimate Trading Bot", description="Application name")
    version: str = Field(default="2.2.0", description="Application version")
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Environment")
    trading_mode: TradingMode = Field(default=TradingMode.PAPER, description="Trading mode")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")

    # Paths
    base_dir: Path = Field(default=Path("."), description="Base directory")
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    logs_dir: Path = Field(default=Path("logs"), description="Logs directory")
    models_dir: Path = Field(default=Path("models"), description="ML models directory")
    cache_dir: Path = Field(default=Path("cache"), description="Cache directory")

    # Nested settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings, description="Database settings")
    redis: RedisSettings = Field(default_factory=RedisSettings, description="Redis settings")
    alpaca: AlpacaSettings = Field(default_factory=AlpacaSettings, description="Alpaca settings")
    openai: OpenAISettings = Field(default_factory=OpenAISettings, description="OpenAI settings")
    trading_session: TradingSessionSettings = Field(default_factory=TradingSessionSettings, description="Trading session settings")
    risk: RiskManagementSettings = Field(default_factory=RiskManagementSettings, description="Risk management settings")
    notifications: NotificationSettings = Field(default_factory=NotificationSettings, description="Notification settings")
    backtest: BacktestSettings = Field(default_factory=BacktestSettings, description="Backtest settings")
    ui: UISettings = Field(default_factory=UISettings, description="UI settings")
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings, description="Monitoring settings")

    # Trading defaults
    default_symbols: List[str] = Field(
        default_factory=lambda: ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
        description="Default trading symbols"
    )
    excluded_symbols: Set[str] = Field(
        default_factory=set,
        description="Symbols to exclude from trading"
    )

    @model_validator(mode='after')
    def create_directories(self) -> 'Settings':
        """Create required directories if they don't exist."""
        for dir_path in [self.data_dir, self.logs_dir, self.models_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        return self

    @field_validator('default_symbols', mode='before')
    @classmethod
    def parse_symbols(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse symbols from string or list."""
        if isinstance(v, str):
            return [s.strip().upper() for s in v.split(',') if s.strip()]
        return [s.upper() for s in v]

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION

    @property
    def is_live_trading(self) -> bool:
        """Check if live trading is enabled."""
        return self.trading_mode == TradingMode.LIVE

    @property
    def is_paper_trading(self) -> bool:
        """Check if paper trading is enabled."""
        return self.trading_mode == TradingMode.PAPER

    def to_dict(self, exclude_secrets: bool = True) -> Dict[str, Any]:
        """
        Convert settings to dictionary.

        Args:
            exclude_secrets: Whether to exclude secret values

        Returns:
            Dictionary representation of settings
        """
        data = self.model_dump()
        if exclude_secrets:
            # Remove sensitive data
            sensitive_keys = ['password', 'secret', 'token', 'api_key']
            def mask_secrets(obj: Any) -> Any:
                if isinstance(obj, dict):
                    return {
                        k: '***MASKED***' if any(sk in k.lower() for sk in sensitive_keys) else mask_secrets(v)
                        for k, v in obj.items()
                    }
                elif isinstance(obj, list):
                    return [mask_secrets(item) for item in obj]
                return obj
            data = mask_secrets(data)
        return data

    def save_to_file(self, filepath: Path) -> None:
        """
        Save settings to JSON file.

        Args:
            filepath: Path to save settings
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(exclude_secrets=False), f, indent=2, default=str)

    @classmethod
    def load_from_file(cls, filepath: Path) -> 'Settings':
        """
        Load settings from JSON file.

        Args:
            filepath: Path to settings file

        Returns:
            Settings instance
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Settings file not found: {filepath}")
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Singleton Settings instance
    """
    return Settings()


def reload_settings() -> Settings:
    """
    Reload settings, clearing the cache.

    Returns:
        New Settings instance
    """
    get_settings.cache_clear()
    return get_settings()


# Module-level settings instance for convenience
settings = get_settings()
