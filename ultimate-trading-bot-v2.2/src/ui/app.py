"""
Flask Application Module for Ultimate Trading Bot v2.2.

This module provides the main Flask application including:
- Application factory
- Blueprint registration
- Extension initialization
- Error handling setup
"""

import logging
import os
from datetime import datetime
from functools import wraps
from typing import Any, Callable

from flask import Flask, g, jsonify, render_template, request, session


from .config import UIConfig, load_ui_config_from_env


logger = logging.getLogger(__name__)


class TradingBotApp(Flask):
    """Extended Flask application for Trading Bot."""

    def __init__(
        self,
        import_name: str,
        ui_config: UIConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Trading Bot application."""
        self.ui_config = ui_config or load_ui_config_from_env()

        # Set template and static folders
        kwargs.setdefault("template_folder", self.ui_config.template_folder)
        kwargs.setdefault("static_folder", self.ui_config.static_folder)
        kwargs.setdefault("static_url_path", self.ui_config.static_url_path)

        super().__init__(import_name, **kwargs)

        # Apply Flask configuration
        self.config.update(self.ui_config.to_flask_config())

        # Initialize components
        self._init_logging()
        self._init_error_handlers()
        self._init_before_request()
        self._init_after_request()
        self._init_context_processors()
        self._init_template_filters()

        logger.info(f"TradingBotApp initialized: {self.ui_config.app_name}")

    def _init_logging(self) -> None:
        """Initialize request logging."""
        if not self.ui_config.log_requests:
            return

        @self.before_request
        def log_request() -> None:
            """Log incoming request."""
            logger.debug(
                f"Request: {request.method} {request.path} "
                f"from {request.remote_addr}"
            )

    def _init_error_handlers(self) -> None:
        """Initialize error handlers."""

        @self.errorhandler(400)
        def bad_request(error: Exception) -> tuple[Any, int]:
            """Handle 400 errors."""
            if request.is_json:
                return jsonify({"error": "Bad Request", "message": str(error)}), 400
            return render_template("errors/400.html", error=error), 400

        @self.errorhandler(401)
        def unauthorized(error: Exception) -> tuple[Any, int]:
            """Handle 401 errors."""
            if request.is_json:
                return jsonify({"error": "Unauthorized", "message": str(error)}), 401
            return render_template("errors/401.html", error=error), 401

        @self.errorhandler(403)
        def forbidden(error: Exception) -> tuple[Any, int]:
            """Handle 403 errors."""
            if request.is_json:
                return jsonify({"error": "Forbidden", "message": str(error)}), 403
            return render_template("errors/403.html", error=error), 403

        @self.errorhandler(404)
        def not_found(error: Exception) -> tuple[Any, int]:
            """Handle 404 errors."""
            if request.is_json:
                return jsonify({"error": "Not Found", "message": str(error)}), 404
            return render_template("errors/404.html", error=error), 404

        @self.errorhandler(500)
        def internal_error(error: Exception) -> tuple[Any, int]:
            """Handle 500 errors."""
            logger.error(f"Internal server error: {error}", exc_info=True)
            if request.is_json:
                return jsonify({"error": "Internal Server Error"}), 500
            return render_template("errors/500.html", error=error), 500

    def _init_before_request(self) -> None:
        """Initialize before request handlers."""

        @self.before_request
        def before_request_handler() -> None:
            """Run before each request."""
            # Store request start time
            g.request_start_time = datetime.now()

            # Set user context
            g.user = session.get("user")
            g.is_authenticated = g.user is not None

    def _init_after_request(self) -> None:
        """Initialize after request handlers."""

        @self.after_request
        def after_request_handler(response: Any) -> Any:
            """Run after each request."""
            # Add security headers
            if self.ui_config.security.csp_enabled:
                csp = "; ".join(
                    f"{k} {v}"
                    for k, v in self.ui_config.security.csp_directives.items()
                )
                response.headers["Content-Security-Policy"] = csp

            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "SAMEORIGIN"
            response.headers["X-XSS-Protection"] = "1; mode=block"

            # Log response time
            if hasattr(g, "request_start_time") and self.ui_config.log_responses:
                duration = (datetime.now() - g.request_start_time).total_seconds()
                logger.debug(
                    f"Response: {response.status_code} in {duration:.3f}s"
                )

            return response

    def _init_context_processors(self) -> None:
        """Initialize template context processors."""

        @self.context_processor
        def inject_globals() -> dict[str, Any]:
            """Inject global template variables."""
            return {
                "app_name": self.ui_config.app_name,
                "app_version": self.ui_config.app_version,
                "theme_config": self.ui_config.theme,
                "current_user": g.get("user"),
                "is_authenticated": g.get("is_authenticated", False),
                "now": datetime.now(),
                "features": {
                    "trading": self.ui_config.enable_trading,
                    "backtesting": self.ui_config.enable_backtesting,
                    "optimization": self.ui_config.enable_optimization,
                    "ai": self.ui_config.enable_ai_features,
                },
            }

    def _init_template_filters(self) -> None:
        """Initialize custom Jinja2 filters."""

        @self.template_filter("currency")
        def currency_filter(value: float | None, symbol: str = "$") -> str:
            """Format value as currency."""
            if value is None:
                return "-"
            sign = "-" if value < 0 else ""
            return f"{sign}{symbol}{abs(value):,.2f}"

        @self.template_filter("percent")
        def percent_filter(value: float | None, decimals: int = 2) -> str:
            """Format value as percentage."""
            if value is None:
                return "-"
            return f"{value * 100:.{decimals}f}%"

        @self.template_filter("number")
        def number_filter(value: float | None, decimals: int = 2) -> str:
            """Format number with commas."""
            if value is None:
                return "-"
            return f"{value:,.{decimals}f}"

        @self.template_filter("datetime")
        def datetime_filter(
            value: datetime | None,
            fmt: str = "%Y-%m-%d %H:%M:%S",
        ) -> str:
            """Format datetime."""
            if value is None:
                return "-"
            return value.strftime(fmt)

        @self.template_filter("timeago")
        def timeago_filter(value: datetime | None) -> str:
            """Format datetime as relative time."""
            if value is None:
                return "-"

            delta = datetime.now() - value
            seconds = delta.total_seconds()

            if seconds < 60:
                return "just now"
            elif seconds < 3600:
                minutes = int(seconds / 60)
                return f"{minutes}m ago"
            elif seconds < 86400:
                hours = int(seconds / 3600)
                return f"{hours}h ago"
            elif seconds < 604800:
                days = int(seconds / 86400)
                return f"{days}d ago"
            else:
                return value.strftime("%Y-%m-%d")

        @self.template_filter("pnl_class")
        def pnl_class_filter(value: float | None) -> str:
            """Get CSS class for P&L value."""
            if value is None:
                return ""
            if value > 0:
                return "text-success"
            elif value < 0:
                return "text-danger"
            return "text-muted"


def create_app(config: UIConfig | None = None) -> TradingBotApp:
    """
    Application factory function.

    Args:
        config: Optional UI configuration

    Returns:
        Configured Flask application
    """
    app = TradingBotApp(__name__, ui_config=config)

    # Register blueprints
    _register_blueprints(app)

    # Initialize extensions
    _init_extensions(app)

    logger.info("Flask application created successfully")

    return app


def _register_blueprints(app: TradingBotApp) -> None:
    """Register Flask blueprints."""
    try:
        from .routes import (
            dashboard_bp,
            trading_bp,
            portfolio_bp,
            analysis_bp,
            settings_bp,
            auth_bp,
            api_bp,
        )

        app.register_blueprint(dashboard_bp)
        app.register_blueprint(trading_bp, url_prefix="/trading")
        app.register_blueprint(portfolio_bp, url_prefix="/portfolio")
        app.register_blueprint(analysis_bp, url_prefix="/analysis")
        app.register_blueprint(settings_bp, url_prefix="/settings")
        app.register_blueprint(auth_bp, url_prefix="/auth")
        app.register_blueprint(api_bp, url_prefix=app.ui_config.api_url_prefix)

        logger.debug("Blueprints registered")
    except ImportError as e:
        logger.warning(f"Could not register all blueprints: {e}")


def _init_extensions(app: TradingBotApp) -> None:
    """Initialize Flask extensions."""
    # Extensions would be initialized here
    # e.g., Flask-Login, Flask-SocketIO, Flask-WTF, etc.
    pass


def login_required(f: Callable) -> Callable:
    """Decorator to require authentication."""

    @wraps(f)
    def decorated_function(*args: Any, **kwargs: Any) -> Any:
        if not g.get("is_authenticated"):
            if request.is_json:
                return jsonify({"error": "Unauthorized"}), 401
            return render_template("errors/401.html"), 401
        return f(*args, **kwargs)

    return decorated_function


def admin_required(f: Callable) -> Callable:
    """Decorator to require admin role."""

    @wraps(f)
    def decorated_function(*args: Any, **kwargs: Any) -> Any:
        user = g.get("user")
        if not user or not user.get("is_admin"):
            if request.is_json:
                return jsonify({"error": "Forbidden"}), 403
            return render_template("errors/403.html"), 403
        return f(*args, **kwargs)

    return decorated_function


def feature_required(feature: str) -> Callable:
    """Decorator to require a specific feature to be enabled."""

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args: Any, **kwargs: Any) -> Any:
            from flask import current_app

            app = current_app
            if isinstance(app, TradingBotApp):
                features = {
                    "trading": app.ui_config.enable_trading,
                    "backtesting": app.ui_config.enable_backtesting,
                    "optimization": app.ui_config.enable_optimization,
                    "ai": app.ui_config.enable_ai_features,
                }

                if not features.get(feature, False):
                    if request.is_json:
                        return jsonify({"error": "Feature not enabled"}), 403
                    return render_template(
                        "errors/feature_disabled.html",
                        feature=feature,
                    ), 403

            return f(*args, **kwargs)

        return decorated_function

    return decorator


# Module version
__version__ = "2.2.0"
