"""
Monitoring Package for Ultimate Trading Bot v2.2.

This package provides comprehensive monitoring capabilities including:
- Metrics collection (counters, gauges, histograms)
- Health checking and status monitoring
- Alert management and notification routing
- Performance monitoring and analytics
- System resource monitoring
- Trade and order monitoring
- Structured logging configuration
- Dashboard data aggregation
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


from .metrics_collector import (
    MetricType,
    MetricConfig,
    Counter,
    Gauge,
    Histogram,
    Summary,
    Timer,
    MetricLabel,
    MetricValue,
    MetricsRegistry,
    MetricsCollector,
    create_metrics_collector,
)
from .health_checker import (
    HealthStatus,
    ComponentHealth,
    SystemHealth,
    HealthCheckConfig,
    HealthCheckResult,
    HealthChecker,
    create_health_checker,
)
from .alerting import (
    AlertSeverity,
    AlertStatus,
    AlertCategory,
    AlertConfig,
    Alert,
    AlertRule,
    AlertEscalation,
    AlertRoute,
    AlertManager,
    create_alert_manager,
)
from .performance_monitor import (
    PerformanceMetrics,
    TradeStats,
    StrategyPerformance,
    RiskMetrics,
    PerformanceSnapshot,
    PerformanceConfig,
    PerformanceMonitor,
    create_performance_monitor,
)
from .system_monitor import (
    ResourceType,
    ResourceThreshold,
    CPUMetrics,
    MemoryMetrics,
    DiskMetrics,
    NetworkMetrics,
    ProcessMetrics,
    SystemSnapshot,
    SystemConfig,
    SystemMonitor,
    create_system_monitor,
)
from .trade_monitor import (
    OrderStatus,
    OrderEvent,
    TradeEvent,
    OrderMetrics,
    TradeMonitorConfig,
    TradeMonitor,
    create_trade_monitor,
)
from .logging_config import (
    LogLevel,
    LogFormat,
    LoggingConfig,
    ContextVar,
    ColoredFormatter,
    JSONFormatter,
    DetailedFormatter,
    ModuleFilter,
    LevelRangeFilter,
    PerformanceLogger,
    TradingLogger,
    LoggingManager,
    configure_logging,
    get_logger,
    set_context,
    clear_context,
    get_performance_logger,
)
from .dashboard_data import (
    TimeFrame,
    ChartType,
    DataPoint,
    CandleData,
    ChartSeries,
    ChartConfig,
    StatCard,
    TableColumn,
    TableData,
    DashboardUpdate,
    TimeSeriesBuffer,
    CandleBuffer,
    DashboardDataProvider,
    PortfolioDataAggregator,
    PerformanceDataAggregator,
    MarketDataAggregator,
    DashboardManager,
    create_dashboard_manager,
    create_dashboard_provider,
)


logger = logging.getLogger(__name__)


__all__ = [
    # Metrics Collector
    "MetricType",
    "MetricConfig",
    "Counter",
    "Gauge",
    "Histogram",
    "Summary",
    "Timer",
    "MetricLabel",
    "MetricValue",
    "MetricsRegistry",
    "MetricsCollector",
    "create_metrics_collector",
    # Health Checker
    "HealthStatus",
    "ComponentHealth",
    "SystemHealth",
    "HealthCheckConfig",
    "HealthCheckResult",
    "HealthChecker",
    "create_health_checker",
    # Alerting
    "AlertSeverity",
    "AlertStatus",
    "AlertCategory",
    "AlertConfig",
    "Alert",
    "AlertRule",
    "AlertEscalation",
    "AlertRoute",
    "AlertManager",
    "create_alert_manager",
    # Performance Monitor
    "PerformanceMetrics",
    "TradeStats",
    "StrategyPerformance",
    "RiskMetrics",
    "PerformanceSnapshot",
    "PerformanceConfig",
    "PerformanceMonitor",
    "create_performance_monitor",
    # System Monitor
    "ResourceType",
    "ResourceThreshold",
    "CPUMetrics",
    "MemoryMetrics",
    "DiskMetrics",
    "NetworkMetrics",
    "ProcessMetrics",
    "SystemSnapshot",
    "SystemConfig",
    "SystemMonitor",
    "create_system_monitor",
    # Trade Monitor
    "OrderStatus",
    "OrderEvent",
    "TradeEvent",
    "OrderMetrics",
    "TradeMonitorConfig",
    "TradeMonitor",
    "create_trade_monitor",
    # Logging Config
    "LogLevel",
    "LogFormat",
    "LoggingConfig",
    "ContextVar",
    "ColoredFormatter",
    "JSONFormatter",
    "DetailedFormatter",
    "ModuleFilter",
    "LevelRangeFilter",
    "PerformanceLogger",
    "TradingLogger",
    "LoggingManager",
    "configure_logging",
    "get_logger",
    "set_context",
    "clear_context",
    "get_performance_logger",
    # Dashboard Data
    "TimeFrame",
    "ChartType",
    "DataPoint",
    "CandleData",
    "ChartSeries",
    "ChartConfig",
    "StatCard",
    "TableColumn",
    "TableData",
    "DashboardUpdate",
    "TimeSeriesBuffer",
    "CandleBuffer",
    "DashboardDataProvider",
    "PortfolioDataAggregator",
    "PerformanceDataAggregator",
    "MarketDataAggregator",
    "DashboardManager",
    "create_dashboard_manager",
    "create_dashboard_provider",
    # Main System
    "MonitoringSystem",
    "MonitoringConfig",
    "create_monitoring_system",
]


@dataclass
class MonitoringConfig:
    """Configuration for the monitoring system."""

    # Metrics settings
    metrics_enabled: bool = True
    metrics_collection_interval: float = 10.0
    metrics_retention_hours: int = 24

    # Health checking settings
    health_check_enabled: bool = True
    health_check_interval: float = 30.0
    health_check_timeout: float = 5.0

    # Alerting settings
    alerting_enabled: bool = True
    alert_check_interval: float = 10.0
    alert_retention_hours: int = 72

    # Performance monitoring settings
    performance_enabled: bool = True
    performance_update_interval: float = 60.0
    track_strategy_performance: bool = True

    # System monitoring settings
    system_monitor_enabled: bool = True
    system_check_interval: float = 30.0
    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    disk_threshold: float = 90.0

    # Trade monitoring settings
    trade_monitor_enabled: bool = True
    track_order_latency: bool = True
    track_fill_rates: bool = True

    # Logging settings
    log_level: LogLevel = LogLevel.INFO
    log_to_file: bool = True
    log_to_json: bool = False
    log_performance: bool = True

    # Dashboard settings
    dashboard_enabled: bool = True
    dashboard_update_interval: float = 1.0


class MonitoringSystem:
    """
    Integrated monitoring system for the trading bot.

    Coordinates all monitoring components including metrics, health checking,
    alerting, performance tracking, and dashboard data.
    """

    def __init__(self, config: MonitoringConfig | None = None) -> None:
        """
        Initialize the monitoring system.

        Args:
            config: Monitoring configuration
        """
        self.config = config or MonitoringConfig()

        # Core components
        self._metrics: MetricsCollector | None = None
        self._health_checker: HealthChecker | None = None
        self._alert_manager: AlertManager | None = None
        self._performance_monitor: PerformanceMonitor | None = None
        self._system_monitor: SystemMonitor | None = None
        self._trade_monitor: TradeMonitor | None = None
        self._logging_manager: LoggingManager | None = None
        self._dashboard_manager: DashboardManager | None = None

        # State
        self._initialized = False
        self._running = False
        self._tasks: list[asyncio.Task] = []

        logger.info("MonitoringSystem created")

    async def initialize(self) -> None:
        """Initialize all monitoring components."""
        try:
            # Configure logging first
            self._setup_logging()

            # Initialize metrics collector
            if self.config.metrics_enabled:
                metrics_config = MetricConfig(
                    collection_interval=self.config.metrics_collection_interval,
                )
                self._metrics = create_metrics_collector(metrics_config)
                logger.debug("Metrics collector initialized")

            # Initialize health checker
            if self.config.health_check_enabled:
                health_config = HealthCheckConfig(
                    check_interval=self.config.health_check_interval,
                    timeout=self.config.health_check_timeout,
                )
                self._health_checker = create_health_checker(health_config)
                logger.debug("Health checker initialized")

            # Initialize alert manager
            if self.config.alerting_enabled:
                alert_config = AlertConfig(
                    check_interval=self.config.alert_check_interval,
                )
                self._alert_manager = create_alert_manager(alert_config)
                logger.debug("Alert manager initialized")

            # Initialize performance monitor
            if self.config.performance_enabled:
                perf_config = PerformanceConfig(
                    update_interval=self.config.performance_update_interval,
                    track_strategy_performance=self.config.track_strategy_performance,
                )
                self._performance_monitor = create_performance_monitor(perf_config)
                logger.debug("Performance monitor initialized")

            # Initialize system monitor
            if self.config.system_monitor_enabled:
                sys_config = SystemConfig(
                    check_interval=self.config.system_check_interval,
                    cpu_threshold=self.config.cpu_threshold,
                    memory_threshold=self.config.memory_threshold,
                    disk_threshold=self.config.disk_threshold,
                )
                self._system_monitor = create_system_monitor(sys_config)
                logger.debug("System monitor initialized")

            # Initialize trade monitor
            if self.config.trade_monitor_enabled:
                trade_config = TradeMonitorConfig(
                    track_latency=self.config.track_order_latency,
                    track_fill_rates=self.config.track_fill_rates,
                )
                self._trade_monitor = create_trade_monitor(trade_config)
                logger.debug("Trade monitor initialized")

            # Initialize dashboard manager
            if self.config.dashboard_enabled:
                self._dashboard_manager = create_dashboard_manager()
                logger.debug("Dashboard manager initialized")

            self._initialized = True
            logger.info("MonitoringSystem initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize MonitoringSystem: {e}")
            raise

    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        log_config = LoggingConfig(
            level=self.config.log_level,
            file_enabled=self.config.log_to_file,
            json_file_enabled=self.config.log_to_json,
            performance_logging=self.config.log_performance,
        )
        self._logging_manager = configure_logging(log_config)

    async def start(self) -> None:
        """Start all monitoring components."""
        if not self._initialized:
            await self.initialize()

        if self._running:
            return

        self._running = True

        # Start all components
        if self._metrics:
            self._tasks.append(asyncio.create_task(self._metrics_loop()))

        if self._health_checker:
            self._tasks.append(asyncio.create_task(self._health_check_loop()))

        if self._alert_manager:
            await self._alert_manager.start()

        if self._system_monitor:
            self._tasks.append(asyncio.create_task(self._system_monitor_loop()))

        if self._trade_monitor:
            await self._trade_monitor.start()

        if self._dashboard_manager:
            await self._dashboard_manager.start()
            self._tasks.append(asyncio.create_task(self._dashboard_update_loop()))

        logger.info("MonitoringSystem started")

    async def stop(self) -> None:
        """Stop all monitoring components."""
        self._running = False

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()

        # Stop components
        if self._alert_manager:
            await self._alert_manager.stop()

        if self._trade_monitor:
            await self._trade_monitor.stop()

        if self._dashboard_manager:
            await self._dashboard_manager.stop()

        logger.info("MonitoringSystem stopped")

    async def _metrics_loop(self) -> None:
        """Metrics collection loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.metrics_collection_interval)

                if self._metrics:
                    # Collect current metrics
                    snapshot = self._metrics.get_snapshot()

                    # Update dashboard if enabled
                    if self._dashboard_manager and snapshot:
                        for name, value in snapshot.items():
                            if isinstance(value, (int, float)):
                                self._dashboard_manager.provider.add_data_point(
                                    f"metric_{name}", float(value)
                                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics loop: {e}")

    async def _health_check_loop(self) -> None:
        """Health checking loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                if self._health_checker:
                    result = await self._health_checker.check_all()

                    # Generate alerts for unhealthy components
                    if self._alert_manager and result.status != HealthStatus.HEALTHY:
                        for component in result.unhealthy_components:
                            await self._alert_manager.create_alert(
                                title=f"Component Unhealthy: {component}",
                                message=f"Component {component} is reporting unhealthy status",
                                severity=AlertSeverity.WARNING,
                                category=AlertCategory.SYSTEM,
                                source="health_checker",
                            )

                    # Update dashboard
                    if self._dashboard_manager:
                        self._dashboard_manager.provider.update_stat_card(
                            "system_health",
                            result.status.value,
                            title="System Health",
                            format_type="text",
                            icon="heart",
                            trend="up" if result.status == HealthStatus.HEALTHY else "down",
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")

    async def _system_monitor_loop(self) -> None:
        """System monitoring loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.system_check_interval)

                if self._system_monitor:
                    snapshot = await self._system_monitor.get_snapshot()

                    # Check thresholds and generate alerts
                    if self._alert_manager:
                        if snapshot.cpu.usage_percent > self.config.cpu_threshold:
                            await self._alert_manager.create_alert(
                                title="High CPU Usage",
                                message=f"CPU usage is {snapshot.cpu.usage_percent:.1f}%",
                                severity=AlertSeverity.WARNING,
                                category=AlertCategory.SYSTEM,
                                source="system_monitor",
                            )

                        if snapshot.memory.percent > self.config.memory_threshold:
                            await self._alert_manager.create_alert(
                                title="High Memory Usage",
                                message=f"Memory usage is {snapshot.memory.percent:.1f}%",
                                severity=AlertSeverity.WARNING,
                                category=AlertCategory.SYSTEM,
                                source="system_monitor",
                            )

                    # Update dashboard
                    if self._dashboard_manager:
                        provider = self._dashboard_manager.provider
                        provider.add_data_point("cpu_usage", snapshot.cpu.usage_percent)
                        provider.add_data_point("memory_usage", snapshot.memory.percent)

                        provider.update_stat_card(
                            "cpu_usage",
                            snapshot.cpu.usage_percent,
                            title="CPU Usage",
                            format_type="percent",
                            icon="cpu",
                        )
                        provider.update_stat_card(
                            "memory_usage",
                            snapshot.memory.percent,
                            title="Memory Usage",
                            format_type="percent",
                            icon="hard-drive",
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in system monitor loop: {e}")

    async def _dashboard_update_loop(self) -> None:
        """Dashboard update loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.dashboard_update_interval)

                # Update performance data if available
                if self._performance_monitor and self._dashboard_manager:
                    perf_snapshot = self._performance_monitor.get_snapshot()
                    if perf_snapshot:
                        self._dashboard_manager.performance.update_performance_metrics(
                            total_return=perf_snapshot.metrics.total_return,
                            sharpe_ratio=perf_snapshot.metrics.sharpe_ratio,
                            sortino_ratio=perf_snapshot.metrics.sortino_ratio,
                            win_rate=perf_snapshot.trade_stats.win_rate,
                            profit_factor=perf_snapshot.metrics.profit_factor,
                            max_drawdown=perf_snapshot.risk.max_drawdown,
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dashboard update loop: {e}")

    # Property accessors
    @property
    def metrics(self) -> MetricsCollector | None:
        """Get metrics collector."""
        return self._metrics

    @property
    def health_checker(self) -> HealthChecker | None:
        """Get health checker."""
        return self._health_checker

    @property
    def alert_manager(self) -> AlertManager | None:
        """Get alert manager."""
        return self._alert_manager

    @property
    def performance_monitor(self) -> PerformanceMonitor | None:
        """Get performance monitor."""
        return self._performance_monitor

    @property
    def system_monitor(self) -> SystemMonitor | None:
        """Get system monitor."""
        return self._system_monitor

    @property
    def trade_monitor(self) -> TradeMonitor | None:
        """Get trade monitor."""
        return self._trade_monitor

    @property
    def dashboard(self) -> DashboardManager | None:
        """Get dashboard manager."""
        return self._dashboard_manager

    # Convenience methods for recording events
    def record_trade(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        **kwargs: Any,
    ) -> None:
        """Record a trade event."""
        if self._trade_monitor:
            self._trade_monitor.record_trade(
                trade_id=trade_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                **kwargs,
            )

        if self._metrics:
            self._metrics.increment("trades_total")
            self._metrics.observe(f"trade_value_{side}", quantity * price)

    def record_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        status: str,
        **kwargs: Any,
    ) -> None:
        """Record an order event."""
        if self._trade_monitor:
            self._trade_monitor.record_order(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                status=status,
                **kwargs,
            )

        if self._metrics:
            self._metrics.increment(f"orders_{status}")

    def update_portfolio_value(self, value: float) -> None:
        """Update current portfolio value."""
        if self._performance_monitor:
            self._performance_monitor.update_portfolio_value(value)

        if self._dashboard_manager:
            self._dashboard_manager.portfolio.update_portfolio_value(value)

        if self._metrics:
            self._metrics.set_gauge("portfolio_value", value)

    def record_signal(
        self,
        symbol: str,
        signal_type: str,
        strength: float,
        source: str,
    ) -> None:
        """Record a trading signal."""
        if self._metrics:
            self._metrics.increment(f"signals_{signal_type}")

        if self._performance_monitor:
            self._performance_monitor.record_signal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                source=source,
            )

    async def create_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        category: AlertCategory = AlertCategory.GENERAL,
        **kwargs: Any,
    ) -> Alert | None:
        """Create an alert."""
        if self._alert_manager:
            return await self._alert_manager.create_alert(
                title=title,
                message=message,
                severity=severity,
                category=category,
                **kwargs,
            )
        return None

    def register_health_check(
        self,
        name: str,
        check_func: Any,
        critical: bool = False,
    ) -> None:
        """Register a component health check."""
        if self._health_checker:
            self._health_checker.register_check(name, check_func, critical)

    def get_system_status(self) -> dict[str, Any]:
        """
        Get comprehensive system status.

        Returns:
            Dictionary with all monitoring data
        """
        status: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "initialized": self._initialized,
            "running": self._running,
        }

        if self._health_checker:
            health = self._health_checker.get_current_health()
            status["health"] = health.__dict__ if health else None

        if self._metrics:
            status["metrics"] = self._metrics.get_snapshot()

        if self._performance_monitor:
            perf = self._performance_monitor.get_snapshot()
            status["performance"] = perf.__dict__ if perf else None

        if self._system_monitor:
            status["system"] = self._system_monitor.get_current_snapshot()

        if self._alert_manager:
            status["active_alerts"] = self._alert_manager.get_active_alert_count()

        if self._trade_monitor:
            status["trade_stats"] = self._trade_monitor.get_stats()

        return status

    def get_dashboard_data(self) -> dict[str, Any]:
        """Get all dashboard data."""
        if self._dashboard_manager:
            return self._dashboard_manager.get_full_dashboard()
        return {}

    async def reset_metrics(self) -> None:
        """Reset all metrics."""
        if self._metrics:
            self._metrics.reset()

        if self._performance_monitor:
            self._performance_monitor.reset()

        logger.info("Monitoring metrics reset")


def create_monitoring_system(
    config: MonitoringConfig | None = None,
) -> MonitoringSystem:
    """
    Create a monitoring system instance.

    Args:
        config: Monitoring configuration

    Returns:
        MonitoringSystem instance
    """
    return MonitoringSystem(config)


# Module version
__version__ = "2.2.0"
