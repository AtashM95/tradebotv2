"""
UI Components Module for Ultimate Trading Bot v2.2.

This module provides reusable UI components including:
- Cards and panels
- Tables and lists
- Charts and graphs
- Navigation elements
- Form components
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ComponentSize(str, Enum):
    """Component size options."""

    XS = "xs"
    SM = "sm"
    MD = "md"
    LG = "lg"
    XL = "xl"


class ComponentVariant(str, Enum):
    """Component variant options."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    SUCCESS = "success"
    DANGER = "danger"
    WARNING = "warning"
    INFO = "info"
    LIGHT = "light"
    DARK = "dark"


@dataclass
class BaseComponent:
    """Base class for UI components."""

    id: str | None = None
    css_class: str = ""
    attrs: dict[str, str] = field(default_factory=dict)

    def get_attrs_str(self) -> str:
        """Get HTML attributes string."""
        parts = []

        if self.id:
            parts.append(f'id="{self.id}"')

        if self.css_class:
            parts.append(f'class="{self.css_class}"')

        for key, value in self.attrs.items():
            parts.append(f'{key}="{value}"')

        return " ".join(parts)


@dataclass
class Card(BaseComponent):
    """Card component."""

    title: str = ""
    subtitle: str = ""
    body: str = ""
    footer: str = ""
    header_actions: list[dict[str, str]] = field(default_factory=list)
    variant: ComponentVariant = ComponentVariant.LIGHT
    collapsible: bool = False
    collapsed: bool = False

    def render(self) -> dict[str, Any]:
        """Render card to template context."""
        return {
            "type": "card",
            "id": self.id,
            "title": self.title,
            "subtitle": self.subtitle,
            "body": self.body,
            "footer": self.footer,
            "header_actions": self.header_actions,
            "variant": self.variant.value,
            "collapsible": self.collapsible,
            "collapsed": self.collapsed,
            "css_class": self.css_class,
            "attrs": self.attrs,
        }


@dataclass
class StatCard(BaseComponent):
    """Statistics card component."""

    title: str = ""
    value: Any = None
    change: float | None = None
    change_period: str = ""
    icon: str | None = None
    format_type: str = "number"  # number, currency, percent
    decimals: int = 2
    trend: str = "neutral"  # up, down, neutral
    sparkline_data: list[float] | None = None
    variant: ComponentVariant = ComponentVariant.LIGHT

    def render(self) -> dict[str, Any]:
        """Render stat card to template context."""
        # Format value
        formatted_value = self._format_value()

        return {
            "type": "stat_card",
            "id": self.id,
            "title": self.title,
            "value": self.value,
            "formatted_value": formatted_value,
            "change": self.change,
            "change_period": self.change_period,
            "icon": self.icon,
            "trend": self.trend,
            "sparkline_data": self.sparkline_data,
            "variant": self.variant.value,
            "css_class": self.css_class,
        }

    def _format_value(self) -> str:
        """Format the value based on type."""
        if self.value is None:
            return "-"

        if self.format_type == "currency":
            sign = "-" if self.value < 0 else ""
            return f"{sign}${abs(self.value):,.{self.decimals}f}"
        elif self.format_type == "percent":
            return f"{self.value * 100:.{self.decimals}f}%"
        else:
            return f"{self.value:,.{self.decimals}f}"


@dataclass
class TableColumn:
    """Table column definition."""

    key: str
    label: str
    sortable: bool = True
    searchable: bool = True
    format_type: str = "text"  # text, number, currency, percent, date, datetime
    align: str = "left"  # left, center, right
    width: str | None = None
    css_class: str = ""


@dataclass
class Table(BaseComponent):
    """Data table component."""

    columns: list[TableColumn] = field(default_factory=list)
    data: list[dict[str, Any]] = field(default_factory=list)
    title: str = ""
    sortable: bool = True
    searchable: bool = True
    paginated: bool = True
    page_size: int = 25
    current_page: int = 1
    total_rows: int = 0
    selectable: bool = False
    row_actions: list[dict[str, str]] = field(default_factory=list)
    empty_message: str = "No data available"

    def render(self) -> dict[str, Any]:
        """Render table to template context."""
        return {
            "type": "table",
            "id": self.id,
            "title": self.title,
            "columns": [
                {
                    "key": c.key,
                    "label": c.label,
                    "sortable": c.sortable,
                    "searchable": c.searchable,
                    "format_type": c.format_type,
                    "align": c.align,
                    "width": c.width,
                    "css_class": c.css_class,
                }
                for c in self.columns
            ],
            "data": self.data,
            "sortable": self.sortable,
            "searchable": self.searchable,
            "paginated": self.paginated,
            "page_size": self.page_size,
            "current_page": self.current_page,
            "total_rows": self.total_rows or len(self.data),
            "selectable": self.selectable,
            "row_actions": self.row_actions,
            "empty_message": self.empty_message,
            "css_class": self.css_class,
        }


@dataclass
class ChartSeries:
    """Chart data series."""

    name: str
    data: list[Any]
    color: str | None = None
    type: str | None = None  # Overrides chart type for this series


@dataclass
class Chart(BaseComponent):
    """Chart component."""

    chart_type: str = "line"  # line, bar, area, candlestick, pie, donut
    series: list[ChartSeries] = field(default_factory=list)
    categories: list[str] | None = None
    title: str = ""
    subtitle: str = ""
    height: int = 350
    show_legend: bool = True
    show_grid: bool = True
    show_toolbar: bool = True
    animations: bool = True
    stacked: bool = False
    x_axis_label: str = ""
    y_axis_label: str = ""
    y_axis_min: float | None = None
    y_axis_max: float | None = None

    def render(self) -> dict[str, Any]:
        """Render chart to template context."""
        return {
            "type": "chart",
            "id": self.id,
            "chart_type": self.chart_type,
            "series": [
                {
                    "name": s.name,
                    "data": s.data,
                    "color": s.color,
                    "type": s.type,
                }
                for s in self.series
            ],
            "categories": self.categories,
            "title": self.title,
            "subtitle": self.subtitle,
            "height": self.height,
            "show_legend": self.show_legend,
            "show_grid": self.show_grid,
            "show_toolbar": self.show_toolbar,
            "animations": self.animations,
            "stacked": self.stacked,
            "x_axis_label": self.x_axis_label,
            "y_axis_label": self.y_axis_label,
            "y_axis_min": self.y_axis_min,
            "y_axis_max": self.y_axis_max,
            "css_class": self.css_class,
        }


@dataclass
class Alert(BaseComponent):
    """Alert/notification component."""

    message: str = ""
    title: str | None = None
    variant: ComponentVariant = ComponentVariant.INFO
    dismissible: bool = True
    icon: str | None = None
    actions: list[dict[str, str]] = field(default_factory=list)

    def render(self) -> dict[str, Any]:
        """Render alert to template context."""
        # Default icons per variant
        default_icons = {
            ComponentVariant.SUCCESS: "check-circle",
            ComponentVariant.DANGER: "x-circle",
            ComponentVariant.WARNING: "alert-triangle",
            ComponentVariant.INFO: "info",
        }

        icon = self.icon or default_icons.get(self.variant)

        return {
            "type": "alert",
            "id": self.id,
            "message": self.message,
            "title": self.title,
            "variant": self.variant.value,
            "dismissible": self.dismissible,
            "icon": icon,
            "actions": self.actions,
            "css_class": self.css_class,
        }


@dataclass
class Badge(BaseComponent):
    """Badge/tag component."""

    text: str = ""
    variant: ComponentVariant = ComponentVariant.PRIMARY
    size: ComponentSize = ComponentSize.MD
    pill: bool = False
    dot: bool = False

    def render(self) -> dict[str, Any]:
        """Render badge to template context."""
        return {
            "type": "badge",
            "text": self.text,
            "variant": self.variant.value,
            "size": self.size.value,
            "pill": self.pill,
            "dot": self.dot,
            "css_class": self.css_class,
        }


@dataclass
class Button(BaseComponent):
    """Button component."""

    text: str = ""
    variant: ComponentVariant = ComponentVariant.PRIMARY
    size: ComponentSize = ComponentSize.MD
    icon: str | None = None
    icon_position: str = "left"  # left, right
    disabled: bool = False
    loading: bool = False
    href: str | None = None
    onclick: str | None = None
    button_type: str = "button"  # button, submit, reset

    def render(self) -> dict[str, Any]:
        """Render button to template context."""
        return {
            "type": "button",
            "id": self.id,
            "text": self.text,
            "variant": self.variant.value,
            "size": self.size.value,
            "icon": self.icon,
            "icon_position": self.icon_position,
            "disabled": self.disabled,
            "loading": self.loading,
            "href": self.href,
            "onclick": self.onclick,
            "button_type": self.button_type,
            "css_class": self.css_class,
        }


@dataclass
class ProgressBar(BaseComponent):
    """Progress bar component."""

    value: float = 0
    max_value: float = 100
    label: str | None = None
    show_percentage: bool = True
    variant: ComponentVariant = ComponentVariant.PRIMARY
    size: ComponentSize = ComponentSize.MD
    striped: bool = False
    animated: bool = False

    def render(self) -> dict[str, Any]:
        """Render progress bar to template context."""
        percentage = (self.value / self.max_value) * 100 if self.max_value > 0 else 0

        return {
            "type": "progress",
            "id": self.id,
            "value": self.value,
            "max_value": self.max_value,
            "percentage": round(percentage, 1),
            "label": self.label,
            "show_percentage": self.show_percentage,
            "variant": self.variant.value,
            "size": self.size.value,
            "striped": self.striped,
            "animated": self.animated,
            "css_class": self.css_class,
        }


@dataclass
class Tab:
    """Single tab definition."""

    id: str
    label: str
    content: str | None = None
    icon: str | None = None
    disabled: bool = False
    badge: str | None = None


@dataclass
class Tabs(BaseComponent):
    """Tabbed content component."""

    tabs: list[Tab] = field(default_factory=list)
    active_tab: str | None = None
    vertical: bool = False
    pills: bool = False

    def render(self) -> dict[str, Any]:
        """Render tabs to template context."""
        active = self.active_tab or (self.tabs[0].id if self.tabs else None)

        return {
            "type": "tabs",
            "id": self.id,
            "tabs": [
                {
                    "id": t.id,
                    "label": t.label,
                    "content": t.content,
                    "icon": t.icon,
                    "disabled": t.disabled,
                    "badge": t.badge,
                    "active": t.id == active,
                }
                for t in self.tabs
            ],
            "active_tab": active,
            "vertical": self.vertical,
            "pills": self.pills,
            "css_class": self.css_class,
        }


@dataclass
class Modal(BaseComponent):
    """Modal dialog component."""

    title: str = ""
    body: str = ""
    footer: str | None = None
    size: ComponentSize = ComponentSize.MD
    show_close: bool = True
    backdrop: str = "true"  # true, false, static
    keyboard: bool = True
    centered: bool = False
    scrollable: bool = False
    fullscreen: bool = False

    def render(self) -> dict[str, Any]:
        """Render modal to template context."""
        return {
            "type": "modal",
            "id": self.id,
            "title": self.title,
            "body": self.body,
            "footer": self.footer,
            "size": self.size.value,
            "show_close": self.show_close,
            "backdrop": self.backdrop,
            "keyboard": self.keyboard,
            "centered": self.centered,
            "scrollable": self.scrollable,
            "fullscreen": self.fullscreen,
            "css_class": self.css_class,
        }


@dataclass
class Tooltip(BaseComponent):
    """Tooltip component."""

    content: str = ""
    target: str = ""
    placement: str = "top"  # top, bottom, left, right
    trigger: str = "hover"  # hover, click, focus

    def render(self) -> dict[str, Any]:
        """Render tooltip to template context."""
        return {
            "type": "tooltip",
            "content": self.content,
            "target": self.target,
            "placement": self.placement,
            "trigger": self.trigger,
        }


@dataclass
class Dropdown(BaseComponent):
    """Dropdown menu component."""

    label: str = ""
    items: list[dict[str, Any]] = field(default_factory=list)
    variant: ComponentVariant = ComponentVariant.SECONDARY
    size: ComponentSize = ComponentSize.MD
    icon: str | None = None
    align: str = "left"  # left, right

    def render(self) -> dict[str, Any]:
        """Render dropdown to template context."""
        return {
            "type": "dropdown",
            "id": self.id,
            "label": self.label,
            "items": self.items,
            "variant": self.variant.value,
            "size": self.size.value,
            "icon": self.icon,
            "align": self.align,
            "css_class": self.css_class,
        }


class ComponentBuilder:
    """Helper class for building components."""

    @staticmethod
    def stat_card(
        title: str,
        value: Any,
        change: float | None = None,
        **kwargs: Any,
    ) -> StatCard:
        """Build a stat card."""
        trend = "neutral"
        if change is not None:
            trend = "up" if change > 0 else "down" if change < 0 else "neutral"

        return StatCard(
            title=title,
            value=value,
            change=change,
            trend=trend,
            **kwargs,
        )

    @staticmethod
    def positions_table(positions: list[dict[str, Any]]) -> Table:
        """Build positions table."""
        return Table(
            id="positions-table",
            title="Open Positions",
            columns=[
                TableColumn(key="symbol", label="Symbol"),
                TableColumn(key="quantity", label="Qty", align="right", format_type="number"),
                TableColumn(key="entry_price", label="Entry", align="right", format_type="currency"),
                TableColumn(key="current_price", label="Current", align="right", format_type="currency"),
                TableColumn(key="pnl", label="P&L", align="right", format_type="currency"),
                TableColumn(key="pnl_percent", label="P&L %", align="right", format_type="percent"),
            ],
            data=positions,
            row_actions=[
                {"action": "close", "label": "Close", "icon": "x"},
                {"action": "edit", "label": "Edit", "icon": "edit"},
            ],
        )

    @staticmethod
    def trades_table(trades: list[dict[str, Any]]) -> Table:
        """Build trades history table."""
        return Table(
            id="trades-table",
            title="Trade History",
            columns=[
                TableColumn(key="timestamp", label="Time", format_type="datetime"),
                TableColumn(key="symbol", label="Symbol"),
                TableColumn(key="side", label="Side"),
                TableColumn(key="quantity", label="Qty", align="right", format_type="number"),
                TableColumn(key="price", label="Price", align="right", format_type="currency"),
                TableColumn(key="pnl", label="P&L", align="right", format_type="currency"),
            ],
            data=trades,
            page_size=50,
        )

    @staticmethod
    def portfolio_chart(
        dates: list[str],
        values: list[float],
        benchmark: list[float] | None = None,
    ) -> Chart:
        """Build portfolio value chart."""
        series = [
            ChartSeries(name="Portfolio", data=values, color="#3b82f6"),
        ]

        if benchmark:
            series.append(
                ChartSeries(name="Benchmark", data=benchmark, color="#6b7280")
            )

        return Chart(
            id="portfolio-chart",
            chart_type="area",
            series=series,
            categories=dates,
            title="Portfolio Performance",
            height=400,
            y_axis_label="Value ($)",
        )


# Module version
__version__ = "2.2.0"
