"""
Dashboard Data Provider Module for Ultimate Trading Bot v2.2.

This module provides dashboard-ready data including:
- Real-time data aggregation
- Chart data preparation
- Statistical summaries
- Time-series data formatting
- WebSocket-ready updates
"""

import asyncio
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable


logger = logging.getLogger(__name__)


class TimeFrame(str, Enum):
    """Time frame enumeration."""

    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"

    def to_seconds(self) -> int:
        """Convert to seconds."""
        mapping = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400,
            "1w": 604800,
            "1M": 2592000,
        }
        return mapping.get(self.value, 60)


class ChartType(str, Enum):
    """Chart type enumeration."""

    LINE = "line"
    BAR = "bar"
    CANDLESTICK = "candlestick"
    AREA = "area"
    PIE = "pie"
    DONUT = "donut"
    SCATTER = "scatter"
    HEATMAP = "heatmap"


@dataclass
class DataPoint:
    """Single data point for time series."""

    timestamp: datetime
    value: float
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandleData:
    """OHLCV candle data."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class ChartSeries:
    """Chart series data."""

    name: str
    data: list[DataPoint | CandleData]
    chart_type: ChartType = ChartType.LINE
    color: str | None = None
    visible: bool = True


@dataclass
class ChartConfig:
    """Chart configuration."""

    title: str
    x_axis_label: str = "Time"
    y_axis_label: str = "Value"
    series: list[ChartSeries] = field(default_factory=list)
    show_legend: bool = True
    show_grid: bool = True
    animate: bool = True
    stacked: bool = False


@dataclass
class StatCard:
    """Statistical card data for dashboard."""

    title: str
    value: Any
    change: float | None = None
    change_period: str | None = None
    trend: str = "neutral"  # up, down, neutral
    icon: str | None = None
    format_type: str = "number"  # number, currency, percent, text
    decimals: int = 2


@dataclass
class TableColumn:
    """Table column definition."""

    key: str
    label: str
    sortable: bool = True
    format_type: str = "text"
    width: str | None = None
    align: str = "left"


@dataclass
class TableData:
    """Table data for dashboard."""

    columns: list[TableColumn]
    rows: list[dict[str, Any]]
    title: str | None = None
    sortable: bool = True
    paginated: bool = True
    page_size: int = 10
    total_rows: int = 0


@dataclass
class DashboardUpdate:
    """Dashboard update message."""

    update_type: str
    section: str
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)


class TimeSeriesBuffer:
    """Thread-safe time series data buffer."""

    def __init__(
        self,
        max_points: int = 1000,
        timeframe: TimeFrame = TimeFrame.MINUTE_1,
    ) -> None:
        """Initialize time series buffer."""
        self._data: deque[DataPoint] = deque(maxlen=max_points)
        self._timeframe = timeframe
        self._lock = threading.Lock()

    def add(self, value: float, timestamp: datetime | None = None) -> None:
        """Add a data point."""
        with self._lock:
            self._data.append(
                DataPoint(
                    timestamp=timestamp or datetime.now(),
                    value=value,
                )
            )

    def get_all(self) -> list[DataPoint]:
        """Get all data points."""
        with self._lock:
            return list(self._data)

    def get_recent(self, count: int) -> list[DataPoint]:
        """Get recent data points."""
        with self._lock:
            return list(self._data)[-count:]

    def get_range(
        self,
        start: datetime,
        end: datetime | None = None,
    ) -> list[DataPoint]:
        """Get data points in time range."""
        end = end or datetime.now()
        with self._lock:
            return [
                p for p in self._data
                if start <= p.timestamp <= end
            ]

    def get_aggregated(self, timeframe: TimeFrame) -> list[DataPoint]:
        """Get aggregated data by timeframe."""
        interval_seconds = timeframe.to_seconds()
        with self._lock:
            if not self._data:
                return []

            aggregated: dict[datetime, list[float]] = {}

            for point in self._data:
                # Round timestamp to interval
                ts = point.timestamp
                rounded = datetime(
                    ts.year, ts.month, ts.day,
                    ts.hour, ts.minute - (ts.minute % (interval_seconds // 60)),
                    0,
                )
                if rounded not in aggregated:
                    aggregated[rounded] = []
                aggregated[rounded].append(point.value)

            # Calculate averages
            result = []
            for ts, values in sorted(aggregated.items()):
                result.append(
                    DataPoint(
                        timestamp=ts,
                        value=sum(values) / len(values),
                    )
                )

            return result

    def clear(self) -> None:
        """Clear all data."""
        with self._lock:
            self._data.clear()


class CandleBuffer:
    """Buffer for OHLCV candle data."""

    def __init__(
        self,
        max_candles: int = 500,
        timeframe: TimeFrame = TimeFrame.MINUTE_5,
    ) -> None:
        """Initialize candle buffer."""
        self._candles: deque[CandleData] = deque(maxlen=max_candles)
        self._timeframe = timeframe
        self._current_candle: CandleData | None = None
        self._lock = threading.Lock()

    def add_tick(self, price: float, volume: float = 0.0) -> None:
        """Add price tick and update current candle."""
        now = datetime.now()
        interval = self._timeframe.to_seconds()

        with self._lock:
            # Get candle start time
            candle_start = datetime(
                now.year, now.month, now.day,
                now.hour, now.minute - (now.minute % (interval // 60)),
                0,
            )

            if self._current_candle is None or self._current_candle.timestamp < candle_start:
                # Complete current candle and start new one
                if self._current_candle is not None:
                    self._candles.append(self._current_candle)

                self._current_candle = CandleData(
                    timestamp=candle_start,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=volume,
                )
            else:
                # Update current candle
                self._current_candle.high = max(self._current_candle.high, price)
                self._current_candle.low = min(self._current_candle.low, price)
                self._current_candle.close = price
                self._current_candle.volume += volume

    def get_candles(self, count: int | None = None) -> list[CandleData]:
        """Get candle data."""
        with self._lock:
            candles = list(self._candles)
            if self._current_candle:
                candles.append(self._current_candle)

            if count:
                return candles[-count:]
            return candles

    def clear(self) -> None:
        """Clear all candles."""
        with self._lock:
            self._candles.clear()
            self._current_candle = None


class DashboardDataProvider:
    """
    Provides data for dashboard visualizations.

    Aggregates and formats data from various sources for dashboard display.
    """

    def __init__(self) -> None:
        """Initialize dashboard data provider."""
        self._time_series: dict[str, TimeSeriesBuffer] = {}
        self._candle_buffers: dict[str, CandleBuffer] = {}
        self._stat_cards: dict[str, StatCard] = {}
        self._tables: dict[str, TableData] = {}
        self._custom_data: dict[str, Any] = {}

        self._update_callbacks: list[Callable[[DashboardUpdate], None]] = []
        self._lock = threading.Lock()

        logger.info("DashboardDataProvider initialized")

    def register_time_series(
        self,
        name: str,
        max_points: int = 1000,
        timeframe: TimeFrame = TimeFrame.MINUTE_1,
    ) -> TimeSeriesBuffer:
        """
        Register a time series data buffer.

        Args:
            name: Series name
            max_points: Maximum data points
            timeframe: Default timeframe

        Returns:
            TimeSeriesBuffer instance
        """
        buffer = TimeSeriesBuffer(max_points, timeframe)
        with self._lock:
            self._time_series[name] = buffer
        return buffer

    def register_candle_buffer(
        self,
        symbol: str,
        max_candles: int = 500,
        timeframe: TimeFrame = TimeFrame.MINUTE_5,
    ) -> CandleBuffer:
        """
        Register a candle data buffer.

        Args:
            symbol: Symbol name
            max_candles: Maximum candles
            timeframe: Candle timeframe

        Returns:
            CandleBuffer instance
        """
        buffer = CandleBuffer(max_candles, timeframe)
        with self._lock:
            self._candle_buffers[symbol] = buffer
        return buffer

    def add_data_point(self, series_name: str, value: float) -> None:
        """Add a data point to a series."""
        with self._lock:
            if series_name not in self._time_series:
                self._time_series[series_name] = TimeSeriesBuffer()
            self._time_series[series_name].add(value)

        self._notify_update(DashboardUpdate(
            update_type="data_point",
            section=series_name,
            data={"value": value, "timestamp": datetime.now().isoformat()},
        ))

    def add_tick(self, symbol: str, price: float, volume: float = 0.0) -> None:
        """Add a price tick to candle buffer."""
        with self._lock:
            if symbol not in self._candle_buffers:
                self._candle_buffers[symbol] = CandleBuffer()
            self._candle_buffers[symbol].add_tick(price, volume)

        self._notify_update(DashboardUpdate(
            update_type="tick",
            section=symbol,
            data={"price": price, "volume": volume},
        ))

    def update_stat_card(
        self,
        name: str,
        value: Any,
        change: float | None = None,
        trend: str = "neutral",
        **kwargs: Any,
    ) -> None:
        """Update a stat card."""
        with self._lock:
            if name in self._stat_cards:
                card = self._stat_cards[name]
                card.value = value
                card.change = change
                card.trend = trend
                for key, val in kwargs.items():
                    if hasattr(card, key):
                        setattr(card, key, val)
            else:
                self._stat_cards[name] = StatCard(
                    title=kwargs.get("title", name),
                    value=value,
                    change=change,
                    trend=trend,
                    **{k: v for k, v in kwargs.items() if k != "title"},
                )

        self._notify_update(DashboardUpdate(
            update_type="stat_card",
            section=name,
            data=self._stat_cards[name].__dict__,
        ))

    def set_table_data(
        self,
        name: str,
        columns: list[TableColumn],
        rows: list[dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        """Set table data."""
        with self._lock:
            self._tables[name] = TableData(
                columns=columns,
                rows=rows,
                total_rows=len(rows),
                **kwargs,
            )

        self._notify_update(DashboardUpdate(
            update_type="table",
            section=name,
            data={"columns": [c.__dict__ for c in columns], "rows": rows},
        ))

    def set_custom_data(self, key: str, data: Any) -> None:
        """Set custom dashboard data."""
        with self._lock:
            self._custom_data[key] = data

        self._notify_update(DashboardUpdate(
            update_type="custom",
            section=key,
            data=data,
        ))

    def get_line_chart(
        self,
        series_name: str,
        timeframe: TimeFrame | None = None,
        title: str | None = None,
    ) -> ChartConfig:
        """
        Get line chart data.

        Args:
            series_name: Series name
            timeframe: Optional aggregation timeframe
            title: Chart title

        Returns:
            ChartConfig for line chart
        """
        with self._lock:
            buffer = self._time_series.get(series_name)
            if buffer is None:
                return ChartConfig(title=title or series_name)

            if timeframe:
                data = buffer.get_aggregated(timeframe)
            else:
                data = buffer.get_all()

        return ChartConfig(
            title=title or series_name,
            series=[
                ChartSeries(
                    name=series_name,
                    data=data,
                    chart_type=ChartType.LINE,
                )
            ],
        )

    def get_candlestick_chart(
        self,
        symbol: str,
        count: int | None = None,
        title: str | None = None,
    ) -> ChartConfig:
        """
        Get candlestick chart data.

        Args:
            symbol: Symbol name
            count: Number of candles
            title: Chart title

        Returns:
            ChartConfig for candlestick chart
        """
        with self._lock:
            buffer = self._candle_buffers.get(symbol)
            if buffer is None:
                return ChartConfig(title=title or symbol)

            candles = buffer.get_candles(count)

        return ChartConfig(
            title=title or f"{symbol} Price",
            series=[
                ChartSeries(
                    name=symbol,
                    data=candles,
                    chart_type=ChartType.CANDLESTICK,
                )
            ],
        )

    def get_multi_line_chart(
        self,
        series_names: list[str],
        timeframe: TimeFrame | None = None,
        title: str = "Multi-Series Chart",
    ) -> ChartConfig:
        """
        Get multi-line chart data.

        Args:
            series_names: List of series names
            timeframe: Optional aggregation timeframe
            title: Chart title

        Returns:
            ChartConfig for multi-line chart
        """
        series_list = []

        with self._lock:
            for name in series_names:
                buffer = self._time_series.get(name)
                if buffer:
                    if timeframe:
                        data = buffer.get_aggregated(timeframe)
                    else:
                        data = buffer.get_all()
                    series_list.append(
                        ChartSeries(name=name, data=data, chart_type=ChartType.LINE)
                    )

        return ChartConfig(title=title, series=series_list)

    def get_pie_chart(
        self,
        data: dict[str, float],
        title: str = "Distribution",
    ) -> ChartConfig:
        """
        Get pie chart data.

        Args:
            data: Label to value mapping
            title: Chart title

        Returns:
            ChartConfig for pie chart
        """
        data_points = [
            DataPoint(timestamp=datetime.now(), value=value, label=label)
            for label, value in data.items()
        ]

        return ChartConfig(
            title=title,
            series=[
                ChartSeries(
                    name="Distribution",
                    data=data_points,
                    chart_type=ChartType.PIE,
                )
            ],
        )

    def get_stat_cards(self) -> dict[str, StatCard]:
        """Get all stat cards."""
        with self._lock:
            return self._stat_cards.copy()

    def get_stat_card(self, name: str) -> StatCard | None:
        """Get a specific stat card."""
        with self._lock:
            return self._stat_cards.get(name)

    def get_table(self, name: str) -> TableData | None:
        """Get table data."""
        with self._lock:
            return self._tables.get(name)

    def get_custom_data(self, key: str) -> Any:
        """Get custom data."""
        with self._lock:
            return self._custom_data.get(key)

    def register_update_callback(
        self,
        callback: Callable[[DashboardUpdate], None],
    ) -> None:
        """Register callback for dashboard updates."""
        self._update_callbacks.append(callback)

    def unregister_update_callback(
        self,
        callback: Callable[[DashboardUpdate], None],
    ) -> None:
        """Unregister update callback."""
        if callback in self._update_callbacks:
            self._update_callbacks.remove(callback)

    def _notify_update(self, update: DashboardUpdate) -> None:
        """Notify all registered callbacks of an update."""
        for callback in self._update_callbacks:
            try:
                callback(update)
            except Exception as e:
                logger.error(f"Error in update callback: {e}")

    def get_dashboard_summary(self) -> dict[str, Any]:
        """
        Get complete dashboard summary.

        Returns:
            Dictionary with all dashboard data
        """
        with self._lock:
            return {
                "timestamp": datetime.now().isoformat(),
                "stat_cards": {
                    name: card.__dict__
                    for name, card in self._stat_cards.items()
                },
                "time_series": list(self._time_series.keys()),
                "candle_buffers": list(self._candle_buffers.keys()),
                "tables": list(self._tables.keys()),
                "custom_data_keys": list(self._custom_data.keys()),
            }

    def clear_all(self) -> None:
        """Clear all dashboard data."""
        with self._lock:
            for buffer in self._time_series.values():
                buffer.clear()
            for buffer in self._candle_buffers.values():
                buffer.clear()
            self._stat_cards.clear()
            self._tables.clear()
            self._custom_data.clear()

        logger.info("Dashboard data cleared")


class PortfolioDataAggregator:
    """Aggregates portfolio data for dashboard display."""

    def __init__(self, dashboard: DashboardDataProvider) -> None:
        """Initialize portfolio data aggregator."""
        self._dashboard = dashboard
        self._positions: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

        # Register time series
        self._dashboard.register_time_series("portfolio_value", max_points=2000)
        self._dashboard.register_time_series("daily_pnl", max_points=500)
        self._dashboard.register_time_series("drawdown", max_points=2000)

    def update_portfolio_value(self, value: float) -> None:
        """Update portfolio value."""
        self._dashboard.add_data_point("portfolio_value", value)
        self._dashboard.update_stat_card(
            "portfolio_value",
            value,
            title="Portfolio Value",
            format_type="currency",
            icon="wallet",
        )

    def update_daily_pnl(self, pnl: float, pnl_percent: float) -> None:
        """Update daily P&L."""
        self._dashboard.add_data_point("daily_pnl", pnl)
        trend = "up" if pnl > 0 else "down" if pnl < 0 else "neutral"
        self._dashboard.update_stat_card(
            "daily_pnl",
            pnl,
            change=pnl_percent,
            trend=trend,
            title="Daily P&L",
            format_type="currency",
            change_period="today",
            icon="trending-up" if pnl >= 0 else "trending-down",
        )

    def update_drawdown(self, drawdown: float, max_drawdown: float) -> None:
        """Update drawdown metrics."""
        self._dashboard.add_data_point("drawdown", drawdown)
        self._dashboard.update_stat_card(
            "current_drawdown",
            drawdown,
            title="Current Drawdown",
            format_type="percent",
            icon="activity",
        )
        self._dashboard.update_stat_card(
            "max_drawdown",
            max_drawdown,
            title="Max Drawdown",
            format_type="percent",
            icon="alert-triangle",
        )

    def update_position(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        current_price: float,
        pnl: float,
        pnl_percent: float,
    ) -> None:
        """Update position data."""
        with self._lock:
            self._positions[symbol] = {
                "symbol": symbol,
                "quantity": quantity,
                "entry_price": entry_price,
                "current_price": current_price,
                "pnl": pnl,
                "pnl_percent": pnl_percent,
                "market_value": quantity * current_price,
            }

        self._update_positions_table()

    def remove_position(self, symbol: str) -> None:
        """Remove position from tracking."""
        with self._lock:
            if symbol in self._positions:
                del self._positions[symbol]

        self._update_positions_table()

    def _update_positions_table(self) -> None:
        """Update positions table."""
        columns = [
            TableColumn(key="symbol", label="Symbol"),
            TableColumn(key="quantity", label="Qty", format_type="number", align="right"),
            TableColumn(key="entry_price", label="Entry", format_type="currency", align="right"),
            TableColumn(key="current_price", label="Current", format_type="currency", align="right"),
            TableColumn(key="pnl", label="P&L", format_type="currency", align="right"),
            TableColumn(key="pnl_percent", label="P&L %", format_type="percent", align="right"),
            TableColumn(key="market_value", label="Value", format_type="currency", align="right"),
        ]

        with self._lock:
            rows = list(self._positions.values())

        self._dashboard.set_table_data("positions", columns, rows, title="Open Positions")

    def get_allocation_chart(self) -> ChartConfig:
        """Get portfolio allocation pie chart."""
        with self._lock:
            allocations = {
                pos["symbol"]: pos["market_value"]
                for pos in self._positions.values()
            }

        return self._dashboard.get_pie_chart(allocations, "Portfolio Allocation")


class PerformanceDataAggregator:
    """Aggregates performance data for dashboard display."""

    def __init__(self, dashboard: DashboardDataProvider) -> None:
        """Initialize performance data aggregator."""
        self._dashboard = dashboard
        self._trade_history: list[dict[str, Any]] = []
        self._lock = threading.Lock()

        # Register time series
        self._dashboard.register_time_series("win_rate", max_points=500)
        self._dashboard.register_time_series("sharpe_ratio", max_points=500)
        self._dashboard.register_time_series("profit_factor", max_points=500)

    def update_performance_metrics(
        self,
        total_return: float,
        sharpe_ratio: float,
        sortino_ratio: float,
        win_rate: float,
        profit_factor: float,
        max_drawdown: float,
    ) -> None:
        """Update all performance metrics."""
        # Add to time series
        self._dashboard.add_data_point("win_rate", win_rate)
        self._dashboard.add_data_point("sharpe_ratio", sharpe_ratio)
        self._dashboard.add_data_point("profit_factor", profit_factor)

        # Update stat cards
        self._dashboard.update_stat_card(
            "total_return",
            total_return,
            title="Total Return",
            format_type="percent",
            trend="up" if total_return > 0 else "down" if total_return < 0 else "neutral",
            icon="percent",
        )
        self._dashboard.update_stat_card(
            "sharpe_ratio",
            sharpe_ratio,
            title="Sharpe Ratio",
            format_type="number",
            decimals=2,
            icon="activity",
        )
        self._dashboard.update_stat_card(
            "sortino_ratio",
            sortino_ratio,
            title="Sortino Ratio",
            format_type="number",
            decimals=2,
            icon="shield",
        )
        self._dashboard.update_stat_card(
            "win_rate",
            win_rate,
            title="Win Rate",
            format_type="percent",
            icon="target",
        )
        self._dashboard.update_stat_card(
            "profit_factor",
            profit_factor,
            title="Profit Factor",
            format_type="number",
            decimals=2,
            icon="trending-up",
        )

    def add_trade(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        exit_price: float,
        pnl: float,
        duration: timedelta,
        timestamp: datetime,
    ) -> None:
        """Add completed trade to history."""
        trade = {
            "trade_id": trade_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_percent": ((exit_price - entry_price) / entry_price) * 100,
            "duration": str(duration),
            "timestamp": timestamp.isoformat(),
        }

        with self._lock:
            self._trade_history.append(trade)
            # Keep only last 1000 trades
            if len(self._trade_history) > 1000:
                self._trade_history = self._trade_history[-1000:]

        self._update_trades_table()

    def _update_trades_table(self) -> None:
        """Update trades history table."""
        columns = [
            TableColumn(key="timestamp", label="Time"),
            TableColumn(key="symbol", label="Symbol"),
            TableColumn(key="side", label="Side"),
            TableColumn(key="quantity", label="Qty", format_type="number", align="right"),
            TableColumn(key="entry_price", label="Entry", format_type="currency", align="right"),
            TableColumn(key="exit_price", label="Exit", format_type="currency", align="right"),
            TableColumn(key="pnl", label="P&L", format_type="currency", align="right"),
            TableColumn(key="duration", label="Duration", align="right"),
        ]

        with self._lock:
            # Show most recent first
            rows = list(reversed(self._trade_history))

        self._dashboard.set_table_data(
            "trade_history",
            columns,
            rows,
            title="Trade History",
            page_size=20,
        )


class MarketDataAggregator:
    """Aggregates market data for dashboard display."""

    def __init__(self, dashboard: DashboardDataProvider) -> None:
        """Initialize market data aggregator."""
        self._dashboard = dashboard
        self._watchlist: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def add_symbol_to_watchlist(
        self,
        symbol: str,
        timeframe: TimeFrame = TimeFrame.MINUTE_5,
    ) -> None:
        """Add symbol to watchlist with candle tracking."""
        self._dashboard.register_candle_buffer(symbol, timeframe=timeframe)
        with self._lock:
            self._watchlist[symbol] = {
                "symbol": symbol,
                "last_price": 0.0,
                "change": 0.0,
                "change_percent": 0.0,
                "volume": 0,
                "high": 0.0,
                "low": 0.0,
            }

    def update_quote(
        self,
        symbol: str,
        price: float,
        change: float,
        change_percent: float,
        volume: int,
        high: float,
        low: float,
    ) -> None:
        """Update quote data for symbol."""
        with self._lock:
            if symbol in self._watchlist:
                self._watchlist[symbol].update({
                    "last_price": price,
                    "change": change,
                    "change_percent": change_percent,
                    "volume": volume,
                    "high": high,
                    "low": low,
                })

        self._dashboard.add_tick(symbol, price, float(volume))
        self._update_watchlist_table()

    def _update_watchlist_table(self) -> None:
        """Update watchlist table."""
        columns = [
            TableColumn(key="symbol", label="Symbol"),
            TableColumn(key="last_price", label="Price", format_type="currency", align="right"),
            TableColumn(key="change", label="Change", format_type="currency", align="right"),
            TableColumn(key="change_percent", label="Change %", format_type="percent", align="right"),
            TableColumn(key="volume", label="Volume", format_type="number", align="right"),
            TableColumn(key="high", label="High", format_type="currency", align="right"),
            TableColumn(key="low", label="Low", format_type="currency", align="right"),
        ]

        with self._lock:
            rows = list(self._watchlist.values())

        self._dashboard.set_table_data("watchlist", columns, rows, title="Watchlist")

    def get_symbol_chart(
        self,
        symbol: str,
        candle_count: int = 100,
    ) -> ChartConfig:
        """Get candlestick chart for symbol."""
        return self._dashboard.get_candlestick_chart(symbol, candle_count)


class DashboardManager:
    """
    Central manager for dashboard data.

    Coordinates all data aggregators and provides unified interface.
    """

    def __init__(self) -> None:
        """Initialize dashboard manager."""
        self._provider = DashboardDataProvider()
        self._portfolio = PortfolioDataAggregator(self._provider)
        self._performance = PerformanceDataAggregator(self._provider)
        self._market = MarketDataAggregator(self._provider)

        self._running = False
        self._update_task: asyncio.Task | None = None

        logger.info("DashboardManager initialized")

    @property
    def provider(self) -> DashboardDataProvider:
        """Get dashboard data provider."""
        return self._provider

    @property
    def portfolio(self) -> PortfolioDataAggregator:
        """Get portfolio data aggregator."""
        return self._portfolio

    @property
    def performance(self) -> PerformanceDataAggregator:
        """Get performance data aggregator."""
        return self._performance

    @property
    def market(self) -> MarketDataAggregator:
        """Get market data aggregator."""
        return self._market

    async def start(self) -> None:
        """Start dashboard manager."""
        self._running = True
        logger.info("DashboardManager started")

    async def stop(self) -> None:
        """Stop dashboard manager."""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        logger.info("DashboardManager stopped")

    def get_full_dashboard(self) -> dict[str, Any]:
        """
        Get complete dashboard data.

        Returns:
            Complete dashboard data structure
        """
        return {
            "summary": self._provider.get_dashboard_summary(),
            "stat_cards": {
                name: card.__dict__
                for name, card in self._provider.get_stat_cards().items()
            },
            "positions": self._provider.get_table("positions"),
            "trade_history": self._provider.get_table("trade_history"),
            "watchlist": self._provider.get_table("watchlist"),
            "allocation": self._portfolio.get_allocation_chart().__dict__,
        }

    def reset(self) -> None:
        """Reset all dashboard data."""
        self._provider.clear_all()
        logger.info("Dashboard data reset")


def create_dashboard_manager() -> DashboardManager:
    """
    Create a dashboard manager instance.

    Returns:
        DashboardManager instance
    """
    return DashboardManager()


def create_dashboard_provider() -> DashboardDataProvider:
    """
    Create a dashboard data provider instance.

    Returns:
        DashboardDataProvider instance
    """
    return DashboardDataProvider()


# Module version
__version__ = "2.2.0"
