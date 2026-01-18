"""
Templates Module for Ultimate Trading Bot v2.2.

This module provides template utilities including:
- Template helpers
- Macros and includes
- Layout management
- Asset helpers
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import urlencode


@dataclass
class PageMeta:
    """Page metadata for templates."""

    title: str = ""
    description: str = ""
    keywords: list[str] = field(default_factory=list)
    robots: str = "index, follow"
    canonical: str = ""
    og_image: str = ""
    og_type: str = "website"


@dataclass
class BreadcrumbItem:
    """Breadcrumb navigation item."""

    label: str
    url: str | None = None
    icon: str | None = None
    active: bool = False


@dataclass
class MenuItem:
    """Navigation menu item."""

    label: str
    url: str
    icon: str | None = None
    badge: str | None = None
    badge_variant: str = "primary"
    active: bool = False
    children: list["MenuItem"] = field(default_factory=list)
    permission: str | None = None


@dataclass
class PageContext:
    """Page rendering context."""

    meta: PageMeta = field(default_factory=PageMeta)
    breadcrumbs: list[BreadcrumbItem] = field(default_factory=list)
    active_nav: str = ""
    sidebar_collapsed: bool = False
    show_header: bool = True
    show_footer: bool = True
    show_sidebar: bool = True


class TemplateHelpers:
    """Helper functions for templates."""

    @staticmethod
    def format_currency(
        value: float | None,
        symbol: str = "$",
        decimals: int = 2,
    ) -> str:
        """Format value as currency."""
        if value is None:
            return "-"
        sign = "-" if value < 0 else ""
        return f"{sign}{symbol}{abs(value):,.{decimals}f}"

    @staticmethod
    def format_percent(
        value: float | None,
        decimals: int = 2,
        show_sign: bool = True,
    ) -> str:
        """Format value as percentage."""
        if value is None:
            return "-"
        sign = "+" if show_sign and value > 0 else ""
        return f"{sign}{value * 100:.{decimals}f}%"

    @staticmethod
    def format_number(
        value: float | None,
        decimals: int = 2,
    ) -> str:
        """Format number with commas."""
        if value is None:
            return "-"
        return f"{value:,.{decimals}f}"

    @staticmethod
    def format_datetime(
        value: datetime | None,
        fmt: str = "%Y-%m-%d %H:%M:%S",
    ) -> str:
        """Format datetime."""
        if value is None:
            return "-"
        return value.strftime(fmt)

    @staticmethod
    def format_date(
        value: datetime | None,
        fmt: str = "%Y-%m-%d",
    ) -> str:
        """Format date."""
        if value is None:
            return "-"
        return value.strftime(fmt)

    @staticmethod
    def format_time(
        value: datetime | None,
        fmt: str = "%H:%M:%S",
    ) -> str:
        """Format time."""
        if value is None:
            return "-"
        return value.strftime(fmt)

    @staticmethod
    def time_ago(value: datetime | None) -> str:
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
        elif seconds < 2592000:
            weeks = int(seconds / 604800)
            return f"{weeks}w ago"
        else:
            return value.strftime("%Y-%m-%d")

    @staticmethod
    def truncate(text: str, length: int = 100, suffix: str = "...") -> str:
        """Truncate text to specified length."""
        if len(text) <= length:
            return text
        return text[:length - len(suffix)] + suffix

    @staticmethod
    def pnl_class(value: float | None) -> str:
        """Get CSS class for P&L value."""
        if value is None:
            return ""
        if value > 0:
            return "text-success"
        elif value < 0:
            return "text-danger"
        return "text-muted"

    @staticmethod
    def order_status_class(status: str) -> str:
        """Get CSS class for order status."""
        status_classes = {
            "pending": "badge-warning",
            "open": "badge-info",
            "filled": "badge-success",
            "partial": "badge-primary",
            "cancelled": "badge-secondary",
            "rejected": "badge-danger",
            "expired": "badge-dark",
        }
        return status_classes.get(status.lower(), "badge-secondary")

    @staticmethod
    def alert_severity_class(severity: str) -> str:
        """Get CSS class for alert severity."""
        severity_classes = {
            "info": "alert-info",
            "warning": "alert-warning",
            "error": "alert-danger",
            "critical": "alert-danger",
            "success": "alert-success",
        }
        return severity_classes.get(severity.lower(), "alert-info")

    @staticmethod
    def json_encode(data: Any) -> str:
        """Encode data as JSON for templates."""
        return json.dumps(data, default=str)

    @staticmethod
    def build_url(base: str, **params: Any) -> str:
        """Build URL with query parameters."""
        if not params:
            return base
        query = urlencode({k: v for k, v in params.items() if v is not None})
        return f"{base}?{query}"


class NavigationBuilder:
    """Builds navigation menus."""

    @staticmethod
    def get_main_navigation(active: str = "") -> list[MenuItem]:
        """Get main navigation menu."""
        return [
            MenuItem(
                label="Dashboard",
                url="/",
                icon="home",
                active=active == "dashboard",
            ),
            MenuItem(
                label="Trading",
                url="/trading",
                icon="trending-up",
                active=active == "trading",
                children=[
                    MenuItem(label="Overview", url="/trading"),
                    MenuItem(label="Place Order", url="/trading/order"),
                    MenuItem(label="Positions", url="/trading/positions"),
                    MenuItem(label="Orders", url="/trading/orders"),
                    MenuItem(label="Trade History", url="/trading/history"),
                ],
            ),
            MenuItem(
                label="Portfolio",
                url="/portfolio",
                icon="briefcase",
                active=active == "portfolio",
                children=[
                    MenuItem(label="Overview", url="/portfolio"),
                    MenuItem(label="Performance", url="/portfolio/performance"),
                    MenuItem(label="Allocation", url="/portfolio/allocation"),
                ],
            ),
            MenuItem(
                label="Analysis",
                url="/analysis",
                icon="bar-chart-2",
                active=active == "analysis",
                children=[
                    MenuItem(label="Overview", url="/analysis"),
                    MenuItem(label="Backtesting", url="/analysis/backtest"),
                    MenuItem(label="Optimization", url="/analysis/optimize"),
                    MenuItem(label="Signals", url="/analysis/signals"),
                ],
            ),
            MenuItem(
                label="Watchlist",
                url="/watchlist",
                icon="star",
                active=active == "watchlist",
            ),
            MenuItem(
                label="Alerts",
                url="/alerts",
                icon="bell",
                active=active == "alerts",
                badge="3",
                badge_variant="danger",
            ),
            MenuItem(
                label="Settings",
                url="/settings",
                icon="settings",
                active=active == "settings",
            ),
        ]

    @staticmethod
    def get_breadcrumbs(
        path: str,
        labels: dict[str, str] | None = None,
    ) -> list[BreadcrumbItem]:
        """Generate breadcrumbs from path."""
        labels = labels or {}

        # Default labels
        default_labels = {
            "": "Home",
            "trading": "Trading",
            "portfolio": "Portfolio",
            "analysis": "Analysis",
            "settings": "Settings",
            "backtest": "Backtesting",
            "optimize": "Optimization",
            "positions": "Positions",
            "orders": "Orders",
            "history": "History",
        }

        all_labels = {**default_labels, **labels}

        parts = [p for p in path.split("/") if p]
        breadcrumbs = [BreadcrumbItem(label="Home", url="/", icon="home")]

        current_path = ""
        for i, part in enumerate(parts):
            current_path += f"/{part}"
            is_last = i == len(parts) - 1

            label = all_labels.get(part, part.replace("-", " ").title())

            breadcrumbs.append(
                BreadcrumbItem(
                    label=label,
                    url=None if is_last else current_path,
                    active=is_last,
                )
            )

        return breadcrumbs


class AssetHelper:
    """Helper for managing static assets."""

    def __init__(
        self,
        static_folder: str = "static",
        manifest_file: str = "manifest.json",
        use_cdn: bool = False,
        cdn_url: str = "",
    ) -> None:
        """Initialize asset helper."""
        self.static_folder = static_folder
        self.manifest_file = manifest_file
        self.use_cdn = use_cdn
        self.cdn_url = cdn_url
        self._manifest: dict[str, str] = {}
        self._load_manifest()

    def _load_manifest(self) -> None:
        """Load asset manifest for cache busting."""
        manifest_path = os.path.join(self.static_folder, self.manifest_file)
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path) as f:
                    self._manifest = json.load(f)
            except Exception:
                pass

    def get_url(self, path: str) -> str:
        """Get URL for static asset."""
        # Check manifest for hashed version
        resolved_path = self._manifest.get(path, path)

        if self.use_cdn and self.cdn_url:
            return f"{self.cdn_url.rstrip('/')}/{resolved_path.lstrip('/')}"

        return f"/static/{resolved_path.lstrip('/')}"

    def get_css_tag(self, path: str, **attrs: str) -> str:
        """Generate CSS link tag."""
        url = self.get_url(path)
        attr_str = " ".join(f'{k}="{v}"' for k, v in attrs.items())
        return f'<link rel="stylesheet" href="{url}" {attr_str}>'

    def get_js_tag(
        self,
        path: str,
        async_load: bool = False,
        defer: bool = False,
        **attrs: str,
    ) -> str:
        """Generate JavaScript script tag."""
        url = self.get_url(path)
        parts = [f'<script src="{url}"']

        if async_load:
            parts.append(" async")
        if defer:
            parts.append(" defer")

        for k, v in attrs.items():
            parts.append(f' {k}="{v}"')

        parts.append("></script>")
        return "".join(parts)


def create_page_context(
    title: str = "",
    description: str = "",
    active_nav: str = "",
    **kwargs: Any,
) -> PageContext:
    """
    Create page context for template rendering.

    Args:
        title: Page title
        description: Page description
        active_nav: Active navigation item
        **kwargs: Additional context

    Returns:
        PageContext instance
    """
    meta = PageMeta(title=title, description=description)
    return PageContext(meta=meta, active_nav=active_nav, **kwargs)


def get_template_helpers() -> dict[str, Any]:
    """
    Get all template helpers for registration.

    Returns:
        Dictionary of helper functions
    """
    helpers = TemplateHelpers()
    return {
        "format_currency": helpers.format_currency,
        "format_percent": helpers.format_percent,
        "format_number": helpers.format_number,
        "format_datetime": helpers.format_datetime,
        "format_date": helpers.format_date,
        "format_time": helpers.format_time,
        "time_ago": helpers.time_ago,
        "truncate": helpers.truncate,
        "pnl_class": helpers.pnl_class,
        "order_status_class": helpers.order_status_class,
        "alert_severity_class": helpers.alert_severity_class,
        "json_encode": helpers.json_encode,
        "build_url": helpers.build_url,
    }


# Module version
__version__ = "2.2.0"
