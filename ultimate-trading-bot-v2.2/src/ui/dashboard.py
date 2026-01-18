"""
Dashboard Module for Ultimate Trading Bot v2.2.

This module provides dashboard functionality including:
- Dashboard layout management
- Widget configuration
- Data aggregation for display
- Real-time updates
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable


logger = logging.getLogger(__name__)


class WidgetType(str, Enum):
    """Dashboard widget types."""

    STAT_CARD = "stat_card"
    CHART = "chart"
    TABLE = "table"
    LIST = "list"
    WATCHLIST = "watchlist"
    POSITIONS = "positions"
    ORDERS = "orders"
    TRADES = "trades"
    ALERTS = "alerts"
    MARKET_OVERVIEW = "market_overview"
    PORTFOLIO_SUMMARY = "portfolio_summary"
    PERFORMANCE_METRICS = "performance_metrics"
    NEWS = "news"
    SIGNALS = "signals"
    CUSTOM = "custom"


class WidgetSize(str, Enum):
    """Widget size options."""

    SMALL = "small"  # 1 column
    MEDIUM = "medium"  # 2 columns
    LARGE = "large"  # 3 columns
    FULL = "full"  # Full width


@dataclass
class WidgetConfig:
    """Dashboard widget configuration."""

    widget_id: str
    widget_type: WidgetType
    title: str = ""
    size: WidgetSize = WidgetSize.MEDIUM
    row: int = 0
    column: int = 0
    height: int = 300
    refresh_interval: int = 5000  # milliseconds
    options: dict[str, Any] = field(default_factory=dict)
    visible: bool = True
    collapsible: bool = True
    collapsed: bool = False


@dataclass
class DashboardLayout:
    """Dashboard layout configuration."""

    layout_id: str
    name: str
    widgets: list[WidgetConfig] = field(default_factory=list)
    columns: int = 3
    row_height: int = 100
    margin: int = 10
    is_default: bool = False


class DashboardDataProvider:
    """Provides data for dashboard widgets."""

    def __init__(self) -> None:
        """Initialize dashboard data provider."""
        self._data_sources: dict[str, Callable[[], Any]] = {}
        self._cache: dict[str, tuple[Any, datetime]] = {}
        self._cache_ttl: dict[str, int] = {}

    def register_data_source(
        self,
        name: str,
        provider: Callable[[], Any],
        cache_ttl: int = 5,
    ) -> None:
        """
        Register a data source.

        Args:
            name: Data source name
            provider: Callable that returns data
            cache_ttl: Cache TTL in seconds
        """
        self._data_sources[name] = provider
        self._cache_ttl[name] = cache_ttl

    def get_data(self, name: str) -> Any:
        """
        Get data from a source.

        Args:
            name: Data source name

        Returns:
            Data from source
        """
        # Check cache
        if name in self._cache:
            data, timestamp = self._cache[name]
            ttl = self._cache_ttl.get(name, 5)
            if (datetime.now() - timestamp).total_seconds() < ttl:
                return data

        # Get fresh data
        if name not in self._data_sources:
            return None

        try:
            data = self._data_sources[name]()
            self._cache[name] = (data, datetime.now())
            return data
        except Exception as e:
            logger.error(f"Error getting data from {name}: {e}")
            return None

    def clear_cache(self, name: str | None = None) -> None:
        """Clear cached data."""
        if name:
            if name in self._cache:
                del self._cache[name]
        else:
            self._cache.clear()


class DashboardManager:
    """
    Dashboard manager.

    Manages dashboard layouts, widgets, and data.
    """

    def __init__(self) -> None:
        """Initialize dashboard manager."""
        self._layouts: dict[str, DashboardLayout] = {}
        self._data_provider = DashboardDataProvider()
        self._user_preferences: dict[str, dict[str, Any]] = {}

        # Create default layout
        self._create_default_layout()

        logger.info("DashboardManager initialized")

    def _create_default_layout(self) -> None:
        """Create default dashboard layout."""
        layout = DashboardLayout(
            layout_id="default",
            name="Default Dashboard",
            is_default=True,
            columns=3,
            widgets=[
                # Top row - Key metrics
                WidgetConfig(
                    widget_id="portfolio_value",
                    widget_type=WidgetType.STAT_CARD,
                    title="Portfolio Value",
                    size=WidgetSize.SMALL,
                    row=0,
                    column=0,
                    height=120,
                    options={"format": "currency", "icon": "wallet"},
                ),
                WidgetConfig(
                    widget_id="daily_pnl",
                    widget_type=WidgetType.STAT_CARD,
                    title="Daily P&L",
                    size=WidgetSize.SMALL,
                    row=0,
                    column=1,
                    height=120,
                    options={"format": "currency", "show_change": True},
                ),
                WidgetConfig(
                    widget_id="open_positions",
                    widget_type=WidgetType.STAT_CARD,
                    title="Open Positions",
                    size=WidgetSize.SMALL,
                    row=0,
                    column=2,
                    height=120,
                    options={"format": "number", "icon": "briefcase"},
                ),
                # Second row - Charts
                WidgetConfig(
                    widget_id="portfolio_chart",
                    widget_type=WidgetType.CHART,
                    title="Portfolio Performance",
                    size=WidgetSize.LARGE,
                    row=1,
                    column=0,
                    height=350,
                    options={"chart_type": "area", "timeframe": "1M"},
                ),
                # Third row - Positions and orders
                WidgetConfig(
                    widget_id="positions",
                    widget_type=WidgetType.POSITIONS,
                    title="Open Positions",
                    size=WidgetSize.MEDIUM,
                    row=2,
                    column=0,
                    height=300,
                ),
                WidgetConfig(
                    widget_id="orders",
                    widget_type=WidgetType.ORDERS,
                    title="Pending Orders",
                    size=WidgetSize.SMALL,
                    row=2,
                    column=2,
                    height=300,
                ),
                # Fourth row - Market and alerts
                WidgetConfig(
                    widget_id="watchlist",
                    widget_type=WidgetType.WATCHLIST,
                    title="Watchlist",
                    size=WidgetSize.SMALL,
                    row=3,
                    column=0,
                    height=300,
                ),
                WidgetConfig(
                    widget_id="recent_trades",
                    widget_type=WidgetType.TRADES,
                    title="Recent Trades",
                    size=WidgetSize.MEDIUM,
                    row=3,
                    column=1,
                    height=300,
                    options={"limit": 10},
                ),
            ],
        )

        self._layouts["default"] = layout

    def get_layout(self, layout_id: str) -> DashboardLayout | None:
        """Get layout by ID."""
        return self._layouts.get(layout_id)

    def get_default_layout(self) -> DashboardLayout | None:
        """Get default layout."""
        for layout in self._layouts.values():
            if layout.is_default:
                return layout
        return list(self._layouts.values())[0] if self._layouts else None

    def get_all_layouts(self) -> list[DashboardLayout]:
        """Get all layouts."""
        return list(self._layouts.values())

    def create_layout(
        self,
        name: str,
        widgets: list[WidgetConfig] | None = None,
        **kwargs: Any,
    ) -> DashboardLayout:
        """
        Create a new layout.

        Args:
            name: Layout name
            widgets: Optional widget configurations
            **kwargs: Additional layout options

        Returns:
            Created layout
        """
        import uuid
        layout_id = f"layout_{uuid.uuid4().hex[:8]}"

        layout = DashboardLayout(
            layout_id=layout_id,
            name=name,
            widgets=widgets or [],
            **kwargs,
        )

        self._layouts[layout_id] = layout
        logger.info(f"Layout created: {name}")

        return layout

    def update_layout(
        self,
        layout_id: str,
        widgets: list[WidgetConfig] | None = None,
        **kwargs: Any,
    ) -> DashboardLayout | None:
        """Update a layout."""
        layout = self._layouts.get(layout_id)
        if not layout:
            return None

        if widgets is not None:
            layout.widgets = widgets

        for key, value in kwargs.items():
            if hasattr(layout, key):
                setattr(layout, key, value)

        logger.info(f"Layout updated: {layout.name}")
        return layout

    def delete_layout(self, layout_id: str) -> bool:
        """Delete a layout."""
        if layout_id in self._layouts:
            layout = self._layouts[layout_id]
            if layout.is_default:
                logger.warning("Cannot delete default layout")
                return False

            del self._layouts[layout_id]
            logger.info(f"Layout deleted: {layout.name}")
            return True

        return False

    def add_widget(
        self,
        layout_id: str,
        widget: WidgetConfig,
    ) -> bool:
        """Add widget to layout."""
        layout = self._layouts.get(layout_id)
        if not layout:
            return False

        layout.widgets.append(widget)
        return True

    def remove_widget(
        self,
        layout_id: str,
        widget_id: str,
    ) -> bool:
        """Remove widget from layout."""
        layout = self._layouts.get(layout_id)
        if not layout:
            return False

        original_count = len(layout.widgets)
        layout.widgets = [w for w in layout.widgets if w.widget_id != widget_id]

        return len(layout.widgets) < original_count

    def update_widget(
        self,
        layout_id: str,
        widget_id: str,
        **kwargs: Any,
    ) -> bool:
        """Update widget configuration."""
        layout = self._layouts.get(layout_id)
        if not layout:
            return False

        for widget in layout.widgets:
            if widget.widget_id == widget_id:
                for key, value in kwargs.items():
                    if hasattr(widget, key):
                        setattr(widget, key, value)
                return True

        return False

    def get_widget_data(self, widget_id: str) -> dict[str, Any]:
        """
        Get data for a widget.

        Args:
            widget_id: Widget ID

        Returns:
            Widget data
        """
        return self._data_provider.get_data(widget_id) or {}

    def register_data_source(
        self,
        name: str,
        provider: Callable[[], Any],
        cache_ttl: int = 5,
    ) -> None:
        """Register a data source for widgets."""
        self._data_provider.register_data_source(name, provider, cache_ttl)

    def get_dashboard_data(self, layout_id: str | None = None) -> dict[str, Any]:
        """
        Get all data for a dashboard.

        Args:
            layout_id: Optional layout ID (uses default if not provided)

        Returns:
            Dashboard data
        """
        layout = self.get_layout(layout_id) if layout_id else self.get_default_layout()
        if not layout:
            return {}

        data = {
            "layout": {
                "id": layout.layout_id,
                "name": layout.name,
                "columns": layout.columns,
            },
            "widgets": {},
        }

        for widget in layout.widgets:
            if widget.visible:
                widget_data = self.get_widget_data(widget.widget_id)
                data["widgets"][widget.widget_id] = {
                    "config": {
                        "id": widget.widget_id,
                        "type": widget.widget_type.value,
                        "title": widget.title,
                        "size": widget.size.value,
                        "row": widget.row,
                        "column": widget.column,
                        "height": widget.height,
                        "options": widget.options,
                        "collapsible": widget.collapsible,
                        "collapsed": widget.collapsed,
                    },
                    "data": widget_data,
                }

        return data

    def set_user_preference(
        self,
        user_id: str,
        key: str,
        value: Any,
    ) -> None:
        """Set user dashboard preference."""
        if user_id not in self._user_preferences:
            self._user_preferences[user_id] = {}

        self._user_preferences[user_id][key] = value

    def get_user_preference(
        self,
        user_id: str,
        key: str,
        default: Any = None,
    ) -> Any:
        """Get user dashboard preference."""
        return self._user_preferences.get(user_id, {}).get(key, default)

    def get_user_layout(self, user_id: str) -> DashboardLayout | None:
        """Get user's preferred layout."""
        layout_id = self.get_user_preference(user_id, "layout")
        if layout_id:
            return self.get_layout(layout_id)
        return self.get_default_layout()


def create_dashboard_manager() -> DashboardManager:
    """
    Create a dashboard manager instance.

    Returns:
        DashboardManager instance
    """
    return DashboardManager()


def create_widget_config(
    widget_id: str,
    widget_type: str | WidgetType,
    **kwargs: Any,
) -> WidgetConfig:
    """
    Create a widget configuration.

    Args:
        widget_id: Widget ID
        widget_type: Widget type
        **kwargs: Additional configuration

    Returns:
        WidgetConfig instance
    """
    if isinstance(widget_type, str):
        widget_type = WidgetType(widget_type)

    return WidgetConfig(
        widget_id=widget_id,
        widget_type=widget_type,
        **kwargs,
    )


# Module version
__version__ = "2.2.0"
