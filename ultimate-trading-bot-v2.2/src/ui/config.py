"""
UI Configuration Module for Ultimate Trading Bot v2.2.

This module provides UI/web application configuration including:
- Flask application settings
- Security configuration
- Theme and display settings
- Session management
"""

import os
import secrets
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any


@dataclass
class SecurityConfig:
    """Security configuration for web application."""

    # Secret key for sessions
    secret_key: str = field(default_factory=lambda: secrets.token_hex(32))

    # CSRF protection
    csrf_enabled: bool = True
    csrf_time_limit: int = 3600  # 1 hour

    # Session settings
    session_cookie_name: str = "trading_bot_session"
    session_cookie_secure: bool = True
    session_cookie_httponly: bool = True
    session_cookie_samesite: str = "Lax"
    permanent_session_lifetime: timedelta = field(default_factory=lambda: timedelta(days=7))

    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_default: str = "100 per minute"
    rate_limit_login: str = "5 per minute"

    # Content Security Policy
    csp_enabled: bool = True
    csp_directives: dict[str, str] = field(default_factory=lambda: {
        "default-src": "'self'",
        "script-src": "'self' 'unsafe-inline' 'unsafe-eval' cdn.jsdelivr.net",
        "style-src": "'self' 'unsafe-inline' cdn.jsdelivr.net fonts.googleapis.com",
        "font-src": "'self' fonts.gstatic.com",
        "img-src": "'self' data: blob:",
        "connect-src": "'self' wss: ws:",
    })


@dataclass
class AuthConfig:
    """Authentication configuration."""

    # Auth settings
    auth_enabled: bool = True
    auth_type: str = "local"  # local, ldap, oauth2

    # Local auth
    password_min_length: int = 8
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_digit: bool = True
    password_require_special: bool = False

    # Session timeout
    session_timeout_minutes: int = 60
    remember_me_days: int = 30

    # Login settings
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15

    # Two-factor auth
    two_factor_enabled: bool = False
    two_factor_issuer: str = "TradingBot"


@dataclass
class ThemeConfig:
    """Theme and display configuration."""

    # Theme settings
    default_theme: str = "dark"
    available_themes: list[str] = field(default_factory=lambda: ["light", "dark", "system"])

    # Color schemes
    primary_color: str = "#3b82f6"  # Blue
    secondary_color: str = "#10b981"  # Green
    accent_color: str = "#f59e0b"  # Amber
    danger_color: str = "#ef4444"  # Red
    warning_color: str = "#f59e0b"  # Amber
    success_color: str = "#10b981"  # Green
    info_color: str = "#3b82f6"  # Blue

    # Chart colors
    chart_up_color: str = "#10b981"  # Green
    chart_down_color: str = "#ef4444"  # Red
    chart_neutral_color: str = "#6b7280"  # Gray
    chart_line_colors: list[str] = field(default_factory=lambda: [
        "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
        "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#6366f1",
    ])

    # Layout
    sidebar_width: int = 280
    sidebar_collapsed_width: int = 64
    header_height: int = 64
    footer_height: int = 48

    # Fonts
    font_family: str = "Inter, system-ui, sans-serif"
    font_size_base: str = "14px"
    font_size_small: str = "12px"
    font_size_large: str = "16px"

    # Animations
    animations_enabled: bool = True
    transition_duration: str = "150ms"


@dataclass
class DashboardConfig:
    """Dashboard configuration."""

    # Refresh settings
    auto_refresh_enabled: bool = True
    auto_refresh_interval: int = 5000  # milliseconds
    chart_refresh_interval: int = 1000  # milliseconds

    # Default layout
    default_layout: str = "grid"  # grid, list, compact
    default_columns: int = 3

    # Widgets
    default_widgets: list[str] = field(default_factory=lambda: [
        "portfolio_value",
        "daily_pnl",
        "positions",
        "recent_trades",
        "market_overview",
        "alerts",
    ])

    # Charts
    default_chart_type: str = "candlestick"
    default_timeframe: str = "1D"
    chart_max_points: int = 500

    # Tables
    default_page_size: int = 25
    max_page_size: int = 100


@dataclass
class WebSocketConfig:
    """WebSocket configuration."""

    enabled: bool = True
    ping_interval: int = 25  # seconds
    ping_timeout: int = 60  # seconds
    max_message_size: int = 1024 * 1024  # 1MB
    compression: bool = True

    # Channels
    channels: list[str] = field(default_factory=lambda: [
        "trades",
        "orders",
        "positions",
        "quotes",
        "alerts",
        "system",
    ])


@dataclass
class UIConfig:
    """Main UI configuration."""

    # Flask settings
    debug: bool = False
    testing: bool = False
    host: str = "0.0.0.0"
    port: int = 5000

    # Application name
    app_name: str = "Ultimate Trading Bot"
    app_version: str = "2.2.0"

    # URLs
    base_url: str = ""
    static_url_path: str = "/static"
    api_url_prefix: str = "/api/v1"

    # Templates
    template_folder: str = "templates"
    static_folder: str = "static"

    # Sub-configurations
    security: SecurityConfig = field(default_factory=SecurityConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    theme: ThemeConfig = field(default_factory=ThemeConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)

    # Feature flags
    enable_trading: bool = True
    enable_backtesting: bool = True
    enable_optimization: bool = True
    enable_ai_features: bool = True
    enable_social_features: bool = False

    # Logging
    log_requests: bool = True
    log_responses: bool = False

    def to_flask_config(self) -> dict[str, Any]:
        """Convert to Flask configuration dictionary."""
        return {
            "DEBUG": self.debug,
            "TESTING": self.testing,
            "SECRET_KEY": self.security.secret_key,
            "SESSION_COOKIE_NAME": self.security.session_cookie_name,
            "SESSION_COOKIE_SECURE": self.security.session_cookie_secure,
            "SESSION_COOKIE_HTTPONLY": self.security.session_cookie_httponly,
            "SESSION_COOKIE_SAMESITE": self.security.session_cookie_samesite,
            "PERMANENT_SESSION_LIFETIME": self.security.permanent_session_lifetime,
            "WTF_CSRF_ENABLED": self.security.csrf_enabled,
            "WTF_CSRF_TIME_LIMIT": self.security.csrf_time_limit,
            "TEMPLATES_AUTO_RELOAD": self.debug,
            "SEND_FILE_MAX_AGE_DEFAULT": 0 if self.debug else 31536000,
        }


def load_ui_config_from_env() -> UIConfig:
    """Load UI configuration from environment variables."""
    config = UIConfig()

    # Flask settings
    config.debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    config.host = os.getenv("FLASK_HOST", "0.0.0.0")
    config.port = int(os.getenv("FLASK_PORT", "5000"))

    # Security
    secret_key = os.getenv("SECRET_KEY")
    if secret_key:
        config.security.secret_key = secret_key

    config.security.session_cookie_secure = os.getenv(
        "SESSION_COOKIE_SECURE", "true"
    ).lower() == "true"

    # Auth
    config.auth.auth_enabled = os.getenv("AUTH_ENABLED", "true").lower() == "true"
    config.auth.two_factor_enabled = os.getenv(
        "TWO_FACTOR_ENABLED", "false"
    ).lower() == "true"

    # Features
    config.enable_trading = os.getenv("ENABLE_TRADING", "true").lower() == "true"
    config.enable_ai_features = os.getenv("ENABLE_AI", "true").lower() == "true"

    return config


def create_ui_config(**kwargs: Any) -> UIConfig:
    """
    Create UI configuration with overrides.

    Args:
        **kwargs: Configuration overrides

    Returns:
        UIConfig instance
    """
    config = UIConfig()

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


# Module version
__version__ = "2.2.0"
