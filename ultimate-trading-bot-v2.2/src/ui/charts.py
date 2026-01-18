"""
Charts Module for Ultimate Trading Bot v2.2.

This module provides chart generation and configuration including:
- Candlestick charts
- Line and area charts
- Technical indicator overlays
- Performance charts
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


class ChartType(str, Enum):
    """Chart type enumeration."""

    LINE = "line"
    AREA = "area"
    BAR = "bar"
    CANDLESTICK = "candlestick"
    HEIKIN_ASHI = "heikin_ashi"
    OHLC = "ohlc"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    PIE = "pie"
    DONUT = "donut"
    HEATMAP = "heatmap"
    TREEMAP = "treemap"
    RADAR = "radar"


class TimeFrame(str, Enum):
    """Chart timeframe enumeration."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1D"
    W1 = "1W"
    MN = "1M"


@dataclass
class ChartTheme:
    """Chart theme configuration."""

    # Background colors
    background: str = "#1e1e2f"
    grid_color: str = "#2d2d3d"

    # Candle colors
    up_color: str = "#10b981"
    down_color: str = "#ef4444"
    wick_up: str = "#10b981"
    wick_down: str = "#ef4444"

    # Volume colors
    volume_up: str = "rgba(16, 185, 129, 0.5)"
    volume_down: str = "rgba(239, 68, 68, 0.5)"

    # Line colors
    line_colors: list[str] = field(default_factory=lambda: [
        "#3b82f6",  # Blue
        "#10b981",  # Green
        "#f59e0b",  # Amber
        "#ef4444",  # Red
        "#8b5cf6",  # Purple
        "#ec4899",  # Pink
        "#06b6d4",  # Cyan
        "#84cc16",  # Lime
    ])

    # Text colors
    text_color: str = "#9ca3af"
    text_muted: str = "#6b7280"

    # Axis colors
    axis_color: str = "#4b5563"
    crosshair_color: str = "#6b7280"

    # Tooltip
    tooltip_bg: str = "#1f2937"
    tooltip_text: str = "#f9fafb"


@dataclass
class ChartAnnotation:
    """Chart annotation (line, marker, label)."""

    annotation_type: str  # line, marker, label, rect
    x: float | datetime | None = None
    y: float | None = None
    x2: float | datetime | None = None
    y2: float | None = None
    text: str = ""
    color: str = "#3b82f6"
    style: str = "solid"  # solid, dashed, dotted
    width: int = 1


@dataclass
class ChartIndicator:
    """Technical indicator overlay."""

    indicator_type: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    color: str = "#3b82f6"
    line_width: int = 1
    visible: bool = True
    pane: int = 0  # 0 = main pane, 1+ = separate pane


@dataclass
class ChartSeries:
    """Chart data series."""

    name: str
    data: list[Any]
    series_type: ChartType = ChartType.LINE
    color: str | None = None
    line_width: int = 2
    fill: bool = False
    fill_opacity: float = 0.2
    visible: bool = True
    y_axis: int = 0
    pane: int = 0


@dataclass
class ChartConfig:
    """Chart configuration."""

    # Chart settings
    chart_id: str = ""
    chart_type: ChartType = ChartType.CANDLESTICK
    title: str = ""
    subtitle: str = ""
    height: int = 400
    width: int | None = None  # None = responsive

    # Data
    series: list[ChartSeries] = field(default_factory=list)
    categories: list[str] | None = None

    # Indicators
    indicators: list[ChartIndicator] = field(default_factory=list)

    # Annotations
    annotations: list[ChartAnnotation] = field(default_factory=list)

    # Axes
    x_axis_type: str = "datetime"  # datetime, category, linear
    y_axis_type: str = "linear"  # linear, logarithmic
    show_x_axis: bool = True
    show_y_axis: bool = True
    y_axis_side: str = "right"

    # Grid
    show_grid: bool = True
    grid_style: str = "dashed"

    # Legend
    show_legend: bool = True
    legend_position: str = "top"  # top, bottom, left, right

    # Toolbar
    show_toolbar: bool = True
    show_zoom: bool = True
    show_pan: bool = True
    show_reset: bool = True
    show_download: bool = True

    # Tooltip
    show_tooltip: bool = True
    shared_tooltip: bool = True

    # Crosshair
    show_crosshair: bool = True

    # Volume
    show_volume: bool = True
    volume_height: int = 80

    # Theme
    theme: ChartTheme = field(default_factory=ChartTheme)

    # Animation
    animate: bool = True
    animation_duration: int = 500

    # Timeframe
    timeframe: TimeFrame = TimeFrame.D1

    # Price scale
    price_scale_mode: str = "normal"  # normal, percentage, logarithmic


class CandleData:
    """OHLCV candle data."""

    def __init__(
        self,
        timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float = 0,
    ) -> None:
        """Initialize candle data."""
        self.timestamp = timestamp
        self.open = open_price
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

    def to_list(self) -> list[Any]:
        """Convert to list format [timestamp, open, high, low, close]."""
        ts = int(self.timestamp.timestamp() * 1000)
        return [ts, self.open, self.high, self.low, self.close]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "time": self.timestamp.isoformat(),
            "timestamp": int(self.timestamp.timestamp() * 1000),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandleData":
        """Create from dictionary."""
        timestamp = data.get("timestamp") or data.get("time")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp / 1000)

        return cls(
            timestamp=timestamp,
            open_price=float(data["open"]),
            high=float(data["high"]),
            low=float(data["low"]),
            close=float(data["close"]),
            volume=float(data.get("volume", 0)),
        )


class ChartBuilder:
    """
    Builder for creating charts.

    Provides fluent interface for chart configuration.
    """

    def __init__(self) -> None:
        """Initialize chart builder."""
        self._config = ChartConfig()

    def set_type(self, chart_type: ChartType | str) -> "ChartBuilder":
        """Set chart type."""
        if isinstance(chart_type, str):
            chart_type = ChartType(chart_type)
        self._config.chart_type = chart_type
        return self

    def set_title(self, title: str, subtitle: str = "") -> "ChartBuilder":
        """Set chart title."""
        self._config.title = title
        self._config.subtitle = subtitle
        return self

    def set_size(self, height: int, width: int | None = None) -> "ChartBuilder":
        """Set chart dimensions."""
        self._config.height = height
        self._config.width = width
        return self

    def add_series(
        self,
        name: str,
        data: list[Any],
        series_type: ChartType | None = None,
        **kwargs: Any,
    ) -> "ChartBuilder":
        """Add data series."""
        series = ChartSeries(
            name=name,
            data=data,
            series_type=series_type or self._config.chart_type,
            **kwargs,
        )
        self._config.series.append(series)
        return self

    def add_candlestick_data(
        self,
        candles: list[CandleData | dict[str, Any]],
        name: str = "Price",
    ) -> "ChartBuilder":
        """Add candlestick data."""
        data = []
        for candle in candles:
            if isinstance(candle, dict):
                candle = CandleData.from_dict(candle)
            data.append(candle.to_list())

        return self.add_series(name, data, ChartType.CANDLESTICK)

    def add_line(
        self,
        name: str,
        data: list[tuple[datetime, float]] | list[float],
        color: str | None = None,
    ) -> "ChartBuilder":
        """Add line series."""
        formatted_data = []
        for item in data:
            if isinstance(item, tuple):
                ts = int(item[0].timestamp() * 1000)
                formatted_data.append([ts, item[1]])
            else:
                formatted_data.append(item)

        return self.add_series(
            name,
            formatted_data,
            ChartType.LINE,
            color=color,
        )

    def add_indicator(
        self,
        indicator_type: str,
        name: str | None = None,
        **params: Any,
    ) -> "ChartBuilder":
        """Add technical indicator."""
        indicator = ChartIndicator(
            indicator_type=indicator_type,
            name=name or indicator_type.upper(),
            params=params,
        )
        self._config.indicators.append(indicator)
        return self

    def add_annotation(
        self,
        annotation_type: str,
        **kwargs: Any,
    ) -> "ChartBuilder":
        """Add annotation."""
        annotation = ChartAnnotation(
            annotation_type=annotation_type,
            **kwargs,
        )
        self._config.annotations.append(annotation)
        return self

    def add_horizontal_line(
        self,
        y: float,
        color: str = "#6b7280",
        label: str = "",
    ) -> "ChartBuilder":
        """Add horizontal line annotation."""
        return self.add_annotation(
            "line",
            y=y,
            y2=y,
            color=color,
            text=label,
        )

    def set_timeframe(self, timeframe: TimeFrame | str) -> "ChartBuilder":
        """Set chart timeframe."""
        if isinstance(timeframe, str):
            timeframe = TimeFrame(timeframe)
        self._config.timeframe = timeframe
        return self

    def show_volume(self, show: bool = True, height: int = 80) -> "ChartBuilder":
        """Configure volume display."""
        self._config.show_volume = show
        self._config.volume_height = height
        return self

    def show_legend(self, show: bool = True, position: str = "top") -> "ChartBuilder":
        """Configure legend."""
        self._config.show_legend = show
        self._config.legend_position = position
        return self

    def show_toolbar(self, show: bool = True) -> "ChartBuilder":
        """Configure toolbar."""
        self._config.show_toolbar = show
        return self

    def set_theme(self, theme: ChartTheme) -> "ChartBuilder":
        """Set chart theme."""
        self._config.theme = theme
        return self

    def build(self) -> ChartConfig:
        """Build and return chart configuration."""
        return self._config

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for rendering."""
        return {
            "chart_id": self._config.chart_id,
            "chart_type": self._config.chart_type.value,
            "title": self._config.title,
            "subtitle": self._config.subtitle,
            "height": self._config.height,
            "width": self._config.width,
            "series": [
                {
                    "name": s.name,
                    "data": s.data,
                    "type": s.series_type.value,
                    "color": s.color,
                    "lineWidth": s.line_width,
                    "fill": s.fill,
                    "fillOpacity": s.fill_opacity,
                    "visible": s.visible,
                }
                for s in self._config.series
            ],
            "indicators": [
                {
                    "type": i.indicator_type,
                    "name": i.name,
                    "params": i.params,
                    "color": i.color,
                    "visible": i.visible,
                }
                for i in self._config.indicators
            ],
            "annotations": [
                {
                    "type": a.annotation_type,
                    "x": a.x,
                    "y": a.y,
                    "x2": a.x2,
                    "y2": a.y2,
                    "text": a.text,
                    "color": a.color,
                }
                for a in self._config.annotations
            ],
            "options": {
                "showVolume": self._config.show_volume,
                "volumeHeight": self._config.volume_height,
                "showLegend": self._config.show_legend,
                "legendPosition": self._config.legend_position,
                "showToolbar": self._config.show_toolbar,
                "showGrid": self._config.show_grid,
                "showCrosshair": self._config.show_crosshair,
                "animate": self._config.animate,
                "timeframe": self._config.timeframe.value,
            },
            "theme": {
                "background": self._config.theme.background,
                "gridColor": self._config.theme.grid_color,
                "upColor": self._config.theme.up_color,
                "downColor": self._config.theme.down_color,
                "textColor": self._config.theme.text_color,
                "lineColors": self._config.theme.line_colors,
            },
        }


def create_candlestick_chart(
    symbol: str,
    candles: list[CandleData | dict[str, Any]],
    indicators: list[str] | None = None,
    height: int = 400,
) -> dict[str, Any]:
    """
    Create a candlestick chart configuration.

    Args:
        symbol: Trading symbol
        candles: Candle data
        indicators: Optional indicator types to add
        height: Chart height

    Returns:
        Chart configuration dictionary
    """
    builder = ChartBuilder()
    builder.set_type(ChartType.CANDLESTICK)
    builder.set_title(f"{symbol} Chart")
    builder.set_size(height)
    builder.add_candlestick_data(candles)
    builder.show_volume(True)

    if indicators:
        for indicator in indicators:
            builder.add_indicator(indicator)

    return builder.to_dict()


def create_portfolio_chart(
    dates: list[datetime],
    values: list[float],
    benchmark: list[float] | None = None,
    height: int = 350,
) -> dict[str, Any]:
    """
    Create a portfolio performance chart.

    Args:
        dates: Date list
        values: Portfolio values
        benchmark: Optional benchmark values
        height: Chart height

    Returns:
        Chart configuration dictionary
    """
    builder = ChartBuilder()
    builder.set_type(ChartType.AREA)
    builder.set_title("Portfolio Performance")
    builder.set_size(height)

    # Add portfolio line
    portfolio_data = list(zip(dates, values))
    builder.add_line("Portfolio", portfolio_data, "#3b82f6")

    # Add benchmark if provided
    if benchmark:
        benchmark_data = list(zip(dates, benchmark))
        builder.add_line("Benchmark", benchmark_data, "#6b7280")

    builder.show_volume(False)
    builder.show_legend(True)

    return builder.to_dict()


def create_pnl_chart(
    dates: list[datetime],
    daily_pnl: list[float],
    cumulative_pnl: list[float],
    height: int = 300,
) -> dict[str, Any]:
    """
    Create a P&L chart.

    Args:
        dates: Date list
        daily_pnl: Daily P&L values
        cumulative_pnl: Cumulative P&L values
        height: Chart height

    Returns:
        Chart configuration dictionary
    """
    builder = ChartBuilder()
    builder.set_type(ChartType.BAR)
    builder.set_title("Profit & Loss")
    builder.set_size(height)

    # Format data
    daily_data = [[int(d.timestamp() * 1000), v] for d, v in zip(dates, daily_pnl)]
    cum_data = [[int(d.timestamp() * 1000), v] for d, v in zip(dates, cumulative_pnl)]

    builder.add_series("Daily P&L", daily_data, ChartType.BAR)
    builder.add_series("Cumulative", cum_data, ChartType.LINE, color="#3b82f6")

    builder.show_volume(False)

    return builder.to_dict()


# Module version
__version__ = "2.2.0"
