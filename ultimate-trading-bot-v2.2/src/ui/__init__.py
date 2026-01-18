"""
UI Package for Ultimate Trading Bot v2.2.

This package provides the complete web interface including:
- Flask application with blueprints
- Authentication and authorization
- Real-time WebSocket communication
- Dashboard with widgets and charts
- Trading interface
- Portfolio management
- Backtesting interface
- Settings and configuration
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from flask import Flask

from .config import (
    UIConfig,
    SecurityConfig,
    AuthConfig,
    ThemeConfig,
    DashboardConfig,
    WebSocketConfig,
    load_ui_config_from_env,
    create_ui_config,
)
from .app import TradingBotApp, create_app
from .auth import (
    AuthManager,
    User,
    Session,
    PasswordHasher,
    TokenManager,
    TOTPManager,
    create_auth_manager,
)
from .websocket import (
    WebSocketManager,
    WebSocketClient,
    WebSocketMessage,
    create_websocket_manager,
)
from .forms import (
    BaseForm,
    LoginForm,
    RegisterForm,
    OrderForm,
    AlertForm,
    BacktestForm,
    SettingsForm,
)
from .components import (
    Component,
    Card,
    StatCard,
    Table,
    TableColumn,
    Chart,
    Badge,
    Alert as AlertComponent,
    Modal,
    Form as FormComponent,
    Button,
    create_card,
    create_stat_card,
    create_table,
    create_chart,
    render_components,
)
from .dashboard import (
    DashboardManager,
    DashboardWidget,
    DashboardLayout,
    WidgetType,
    create_dashboard_manager,
)
from .charts import (
    ChartBuilder,
    ChartConfig,
    ChartType,
    ChartTheme,
    ChartSeries,
    ChartIndicator,
    ChartAnnotation,
    CandleData,
    TimeFrame,
    create_candlestick_chart,
    create_portfolio_chart,
    create_pnl_chart,
)
from .middleware import (
    RateLimiter,
    RateLimitConfig,
    SecurityHeaders,
    CORSMiddleware,
    CompressionMiddleware,
    RequestLogger,
    setup_middleware,
)
from .templates import (
    TemplateHelpers,
    NavigationBuilder,
    NavItem,
    AssetHelper,
    create_template_helpers,
    get_navigation,
)
from .api_views import (
    api_views_bp,
    APIError,
    APIResponse,
    api_response,
    handle_api_errors,
    require_json,
    validate_params,
)
from .cache import (
    MemoryCache,
    CacheEntry,
    FragmentCache,
    get_cache,
    cached,
    invalidate_cache,
    create_cache,
    create_fragment_cache,
)
from .themes import (
    Theme,
    ThemeMode,
    ThemeColors,
    ColorPalette,
    ThemeManager,
    LIGHT_THEME,
    DARK_THEME,
    NORD_THEME,
    get_theme_manager,
    get_theme_css,
    get_available_themes,
)
from .routes import (
    dashboard_bp,
    trading_bp,
    portfolio_bp,
    auth_bp,
    api_bp,
    settings_bp,
    backtest_bp,
    register_all_blueprints,
)


logger = logging.getLogger(__name__)


@dataclass
class UIState:
    """UI application state."""

    initialized: bool = False
    started_at: datetime | None = None
    request_count: int = 0
    error_count: int = 0
    active_connections: int = 0
    last_error: str | None = None
    last_error_time: datetime | None = None

    def record_request(self) -> None:
        """Record a request."""
        self.request_count += 1

    def record_error(self, error: str) -> None:
        """Record an error."""
        self.error_count += 1
        self.last_error = error
        self.last_error_time = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "initialized": self.initialized,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "active_connections": self.active_connections,
            "last_error": self.last_error,
            "last_error_time": (
                self.last_error_time.isoformat() if self.last_error_time else None
            ),
            "uptime_seconds": (
                (datetime.now() - self.started_at).total_seconds()
                if self.started_at
                else 0
            ),
        }


class UIManager:
    """
    Main UI manager coordinating all UI components.

    This class provides a unified interface to manage:
    - Flask application
    - Authentication
    - WebSocket connections
    - Dashboard and widgets
    - Themes and caching
    """

    def __init__(
        self,
        config: UIConfig | None = None,
        trading_engine: Any = None,
        portfolio_manager: Any = None,
        data_manager: Any = None,
    ) -> None:
        """
        Initialize UI manager.

        Args:
            config: UI configuration
            trading_engine: Trading engine instance
            portfolio_manager: Portfolio manager instance
            data_manager: Data manager instance
        """
        self._config = config or load_ui_config_from_env()
        self._trading_engine = trading_engine
        self._portfolio_manager = portfolio_manager
        self._data_manager = data_manager

        # Components
        self._app: TradingBotApp | None = None
        self._flask_app: Flask | None = None
        self._auth_manager: AuthManager | None = None
        self._websocket_manager: WebSocketManager | None = None
        self._dashboard_manager: DashboardManager | None = None
        self._theme_manager: ThemeManager | None = None
        self._cache: MemoryCache | None = None
        self._rate_limiter: RateLimiter | None = None

        # State
        self._state = UIState()

        logger.info("UIManager created")

    @property
    def config(self) -> UIConfig:
        """Get UI configuration."""
        return self._config

    @property
    def app(self) -> Flask | None:
        """Get Flask application."""
        return self._flask_app

    @property
    def auth_manager(self) -> AuthManager | None:
        """Get authentication manager."""
        return self._auth_manager

    @property
    def websocket_manager(self) -> WebSocketManager | None:
        """Get WebSocket manager."""
        return self._websocket_manager

    @property
    def dashboard_manager(self) -> DashboardManager | None:
        """Get dashboard manager."""
        return self._dashboard_manager

    @property
    def theme_manager(self) -> ThemeManager | None:
        """Get theme manager."""
        return self._theme_manager

    @property
    def cache(self) -> MemoryCache | None:
        """Get cache instance."""
        return self._cache

    @property
    def state(self) -> UIState:
        """Get UI state."""
        return self._state

    def initialize(self) -> None:
        """Initialize all UI components."""
        if self._state.initialized:
            logger.warning("UIManager already initialized")
            return

        logger.info("Initializing UIManager...")

        # Create cache
        self._cache = create_cache(
            max_size=5000,
            default_ttl=300,
            cleanup_interval=60,
        )

        # Create theme manager
        self._theme_manager = get_theme_manager()

        # Create auth manager
        self._auth_manager = create_auth_manager(
            secret_key=self._config.security.secret_key,
            session_timeout=self._config.auth.session_timeout_minutes * 60,
        )

        # Create WebSocket manager
        self._websocket_manager = create_websocket_manager(
            ping_interval=self._config.websocket.ping_interval,
        )

        # Create dashboard manager
        self._dashboard_manager = create_dashboard_manager(
            self._config.dashboard.default_widgets,
        )

        # Create rate limiter
        self._rate_limiter = RateLimiter(
            RateLimitConfig(
                requests_per_minute=100,
                requests_per_hour=3000,
            )
        )

        # Create Flask application
        self._create_flask_app()

        self._state.initialized = True
        self._state.started_at = datetime.now()

        logger.info("UIManager initialized successfully")

    def _create_flask_app(self) -> None:
        """Create and configure Flask application."""
        self._app = TradingBotApp(self._config)
        self._flask_app = self._app.create_app()

        # Register blueprints
        register_all_blueprints(self._flask_app, self._config.api_url_prefix)

        # Setup middleware
        setup_middleware(self._flask_app, self._config)

        # Inject dependencies
        self._inject_dependencies()

        # Register error handlers
        self._register_error_handlers()

        # Register template helpers
        self._register_template_helpers()

        logger.info("Flask application created")

    def _inject_dependencies(self) -> None:
        """Inject dependencies into Flask app context."""
        if not self._flask_app:
            return

        @self._flask_app.before_request
        def inject_managers() -> None:
            """Inject managers into request context."""
            from flask import g

            g.auth_manager = self._auth_manager
            g.websocket_manager = self._websocket_manager
            g.dashboard_manager = self._dashboard_manager
            g.theme_manager = self._theme_manager
            g.cache = self._cache
            g.trading_engine = self._trading_engine
            g.portfolio_manager = self._portfolio_manager
            g.data_manager = self._data_manager

            # Record request
            self._state.record_request()

    def _register_error_handlers(self) -> None:
        """Register error handlers."""
        if not self._flask_app:
            return

        @self._flask_app.errorhandler(400)
        def bad_request(error: Any) -> tuple[dict[str, Any], int]:
            """Handle 400 errors."""
            self._state.record_error(str(error))
            return {"error": True, "message": "Bad request"}, 400

        @self._flask_app.errorhandler(401)
        def unauthorized(error: Any) -> tuple[dict[str, Any], int]:
            """Handle 401 errors."""
            self._state.record_error(str(error))
            return {"error": True, "message": "Unauthorized"}, 401

        @self._flask_app.errorhandler(403)
        def forbidden(error: Any) -> tuple[dict[str, Any], int]:
            """Handle 403 errors."""
            self._state.record_error(str(error))
            return {"error": True, "message": "Forbidden"}, 403

        @self._flask_app.errorhandler(404)
        def not_found(error: Any) -> tuple[dict[str, Any], int]:
            """Handle 404 errors."""
            return {"error": True, "message": "Not found"}, 404

        @self._flask_app.errorhandler(500)
        def internal_error(error: Any) -> tuple[dict[str, Any], int]:
            """Handle 500 errors."""
            self._state.record_error(str(error))
            logger.error(f"Internal server error: {error}")
            return {"error": True, "message": "Internal server error"}, 500

    def _register_template_helpers(self) -> None:
        """Register template helper functions."""
        if not self._flask_app:
            return

        helpers = create_template_helpers()

        @self._flask_app.context_processor
        def inject_helpers() -> dict[str, Any]:
            """Inject template helpers."""
            return {
                "format_currency": helpers.format_currency,
                "format_percent": helpers.format_percent,
                "format_number": helpers.format_number,
                "format_datetime": helpers.format_datetime,
                "format_date": helpers.format_date,
                "format_time": helpers.format_time,
                "time_ago": helpers.time_ago,
                "get_navigation": get_navigation,
                "get_theme_css": get_theme_css,
                "get_available_themes": get_available_themes,
                "app_name": self._config.app_name,
                "app_version": self._config.app_version,
            }

    def run(
        self,
        host: str | None = None,
        port: int | None = None,
        debug: bool | None = None,
    ) -> None:
        """
        Run the Flask application.

        Args:
            host: Host to bind to
            port: Port to bind to
            debug: Enable debug mode
        """
        if not self._state.initialized:
            self.initialize()

        if not self._flask_app:
            raise RuntimeError("Flask application not created")

        host = host or self._config.host
        port = port or self._config.port
        debug = debug if debug is not None else self._config.debug

        logger.info(f"Starting UI server on {host}:{port}")

        self._flask_app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True,
        )

    def get_wsgi_app(self) -> Flask:
        """
        Get WSGI application for production deployment.

        Returns:
            Flask application
        """
        if not self._state.initialized:
            self.initialize()

        if not self._flask_app:
            raise RuntimeError("Flask application not created")

        return self._flask_app

    def broadcast_message(
        self,
        channel: str,
        message: dict[str, Any],
    ) -> None:
        """
        Broadcast message to WebSocket clients.

        Args:
            channel: Channel name
            message: Message to broadcast
        """
        if self._websocket_manager:
            self._websocket_manager.broadcast(channel, message)

    def send_notification(
        self,
        user_id: str,
        notification: dict[str, Any],
    ) -> None:
        """
        Send notification to specific user.

        Args:
            user_id: User ID
            notification: Notification data
        """
        if self._websocket_manager:
            self._websocket_manager.send_to_user(
                user_id,
                "notification",
                notification,
            )

    def invalidate_cache(self, *tags: str) -> None:
        """
        Invalidate cache entries by tags.

        Args:
            *tags: Tags to invalidate
        """
        if self._cache:
            for tag in tags:
                self._cache.delete_by_tag(tag)

    def get_stats(self) -> dict[str, Any]:
        """Get UI statistics."""
        stats = {
            "state": self._state.to_dict(),
            "cache": self._cache.get_stats() if self._cache else {},
            "websocket": (
                self._websocket_manager.get_stats()
                if self._websocket_manager
                else {}
            ),
        }

        return stats

    def shutdown(self) -> None:
        """Shutdown UI manager."""
        logger.info("Shutting down UIManager...")

        if self._websocket_manager:
            self._websocket_manager.close_all()

        if self._cache:
            self._cache.clear()

        self._state.initialized = False

        logger.info("UIManager shutdown complete")


# Global UI manager instance
_ui_manager: UIManager | None = None


def get_ui_manager() -> UIManager:
    """
    Get or create global UI manager instance.

    Returns:
        UIManager instance
    """
    global _ui_manager
    if _ui_manager is None:
        _ui_manager = UIManager()
    return _ui_manager


def create_ui_manager(
    config: UIConfig | None = None,
    trading_engine: Any = None,
    portfolio_manager: Any = None,
    data_manager: Any = None,
    auto_initialize: bool = True,
) -> UIManager:
    """
    Create a new UI manager instance.

    Args:
        config: UI configuration
        trading_engine: Trading engine instance
        portfolio_manager: Portfolio manager instance
        data_manager: Data manager instance
        auto_initialize: Whether to initialize automatically

    Returns:
        UIManager instance
    """
    manager = UIManager(
        config=config,
        trading_engine=trading_engine,
        portfolio_manager=portfolio_manager,
        data_manager=data_manager,
    )

    if auto_initialize:
        manager.initialize()

    return manager


def run_ui_server(
    config: UIConfig | None = None,
    host: str | None = None,
    port: int | None = None,
    debug: bool = False,
) -> None:
    """
    Run the UI server.

    Args:
        config: UI configuration
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode
    """
    manager = create_ui_manager(config=config)
    manager.run(host=host, port=port, debug=debug)


__all__ = [
    # Main classes
    "UIManager",
    "UIState",
    # Factory functions
    "get_ui_manager",
    "create_ui_manager",
    "run_ui_server",
    # Config
    "UIConfig",
    "SecurityConfig",
    "AuthConfig",
    "ThemeConfig",
    "DashboardConfig",
    "WebSocketConfig",
    "load_ui_config_from_env",
    "create_ui_config",
    # App
    "TradingBotApp",
    "create_app",
    # Auth
    "AuthManager",
    "User",
    "Session",
    "PasswordHasher",
    "TokenManager",
    "TOTPManager",
    "create_auth_manager",
    # WebSocket
    "WebSocketManager",
    "WebSocketClient",
    "WebSocketMessage",
    "create_websocket_manager",
    # Forms
    "BaseForm",
    "LoginForm",
    "RegisterForm",
    "OrderForm",
    "AlertForm",
    "BacktestForm",
    "SettingsForm",
    # Components
    "Component",
    "Card",
    "StatCard",
    "Table",
    "TableColumn",
    "Chart",
    "Badge",
    "AlertComponent",
    "Modal",
    "FormComponent",
    "Button",
    "create_card",
    "create_stat_card",
    "create_table",
    "create_chart",
    "render_components",
    # Dashboard
    "DashboardManager",
    "DashboardWidget",
    "DashboardLayout",
    "WidgetType",
    "create_dashboard_manager",
    # Charts
    "ChartBuilder",
    "ChartConfig",
    "ChartType",
    "ChartTheme",
    "ChartSeries",
    "ChartIndicator",
    "ChartAnnotation",
    "CandleData",
    "TimeFrame",
    "create_candlestick_chart",
    "create_portfolio_chart",
    "create_pnl_chart",
    # Middleware
    "RateLimiter",
    "RateLimitConfig",
    "SecurityHeaders",
    "CORSMiddleware",
    "CompressionMiddleware",
    "RequestLogger",
    "setup_middleware",
    # Templates
    "TemplateHelpers",
    "NavigationBuilder",
    "NavItem",
    "AssetHelper",
    "create_template_helpers",
    "get_navigation",
    # API
    "api_views_bp",
    "APIError",
    "APIResponse",
    "api_response",
    "handle_api_errors",
    "require_json",
    "validate_params",
    # Cache
    "MemoryCache",
    "CacheEntry",
    "FragmentCache",
    "get_cache",
    "cached",
    "invalidate_cache",
    "create_cache",
    "create_fragment_cache",
    # Themes
    "Theme",
    "ThemeMode",
    "ThemeColors",
    "ColorPalette",
    "ThemeManager",
    "LIGHT_THEME",
    "DARK_THEME",
    "NORD_THEME",
    "get_theme_manager",
    "get_theme_css",
    "get_available_themes",
    # Blueprints
    "dashboard_bp",
    "trading_bp",
    "portfolio_bp",
    "auth_bp",
    "api_bp",
    "settings_bp",
    "backtest_bp",
    "register_all_blueprints",
]

# Module version
__version__ = "2.2.0"
