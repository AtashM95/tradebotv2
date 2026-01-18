"""
System Monitor for Ultimate Trading Bot v2.2.

Monitors system resources, processes, and infrastructure health.
"""

import asyncio
import logging
import os
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

import psutil

logger = logging.getLogger(__name__)


@dataclass
class SystemConfig:
    """Configuration for system monitoring."""

    # Monitoring intervals
    poll_interval: float = 10.0  # seconds
    history_size: int = 1000

    # Thresholds
    cpu_warning: float = 70.0
    cpu_critical: float = 90.0
    memory_warning: float = 70.0
    memory_critical: float = 90.0
    disk_warning: float = 80.0
    disk_critical: float = 95.0

    # Network monitoring
    monitor_network: bool = True
    network_interfaces: list[str] = field(default_factory=list)

    # Process monitoring
    monitor_processes: bool = True
    tracked_processes: list[str] = field(default_factory=list)


@dataclass
class CPUMetrics:
    """CPU metrics."""

    timestamp: datetime
    percent: float
    per_cpu: list[float]
    load_avg_1: float = 0.0
    load_avg_5: float = 0.0
    load_avg_15: float = 0.0
    ctx_switches: int = 0
    interrupts: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "percent": self.percent,
            "per_cpu": self.per_cpu,
            "load_avg": [self.load_avg_1, self.load_avg_5, self.load_avg_15],
            "ctx_switches": self.ctx_switches,
            "interrupts": self.interrupts,
        }


@dataclass
class MemoryMetrics:
    """Memory metrics."""

    timestamp: datetime
    total: int
    available: int
    used: int
    percent: float
    swap_total: int = 0
    swap_used: int = 0
    swap_percent: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_gb": self.total / (1024 ** 3),
            "available_gb": self.available / (1024 ** 3),
            "used_gb": self.used / (1024 ** 3),
            "percent": self.percent,
            "swap_total_gb": self.swap_total / (1024 ** 3),
            "swap_used_gb": self.swap_used / (1024 ** 3),
            "swap_percent": self.swap_percent,
        }


@dataclass
class DiskMetrics:
    """Disk metrics."""

    timestamp: datetime
    path: str
    total: int
    used: int
    free: int
    percent: float
    read_bytes: int = 0
    write_bytes: int = 0
    read_count: int = 0
    write_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "path": self.path,
            "total_gb": self.total / (1024 ** 3),
            "used_gb": self.used / (1024 ** 3),
            "free_gb": self.free / (1024 ** 3),
            "percent": self.percent,
            "read_mb": self.read_bytes / (1024 ** 2),
            "write_mb": self.write_bytes / (1024 ** 2),
        }


@dataclass
class NetworkMetrics:
    """Network metrics."""

    timestamp: datetime
    interface: str
    bytes_sent: int
    bytes_recv: int
    packets_sent: int
    packets_recv: int
    errors_in: int = 0
    errors_out: int = 0
    drops_in: int = 0
    drops_out: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "interface": self.interface,
            "bytes_sent_mb": self.bytes_sent / (1024 ** 2),
            "bytes_recv_mb": self.bytes_recv / (1024 ** 2),
            "packets_sent": self.packets_sent,
            "packets_recv": self.packets_recv,
            "errors": self.errors_in + self.errors_out,
            "drops": self.drops_in + self.drops_out,
        }


@dataclass
class ProcessMetrics:
    """Process metrics."""

    timestamp: datetime
    pid: int
    name: str
    cpu_percent: float
    memory_percent: float
    memory_rss: int
    num_threads: int
    status: str
    create_time: float
    num_fds: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "pid": self.pid,
            "name": self.name,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_mb": self.memory_rss / (1024 ** 2),
            "threads": self.num_threads,
            "status": self.status,
            "num_fds": self.num_fds,
        }


@dataclass
class SystemSnapshot:
    """Complete system snapshot."""

    timestamp: datetime
    cpu: CPUMetrics
    memory: MemoryMetrics
    disks: list[DiskMetrics]
    network: list[NetworkMetrics]
    processes: list[ProcessMetrics]

    # System info
    hostname: str = ""
    os_name: str = ""
    os_version: str = ""
    python_version: str = ""
    uptime: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "system": {
                "hostname": self.hostname,
                "os": self.os_name,
                "os_version": self.os_version,
                "python": self.python_version,
                "uptime_hours": self.uptime / 3600,
            },
            "cpu": self.cpu.to_dict(),
            "memory": self.memory.to_dict(),
            "disks": [d.to_dict() for d in self.disks],
            "network": [n.to_dict() for n in self.network],
            "processes": [p.to_dict() for p in self.processes[:10]],  # Top 10
        }


class SystemMonitor:
    """
    System resource monitoring.

    Monitors CPU, memory, disk, network, and processes.
    """

    def __init__(self, config: SystemConfig | None = None) -> None:
        """
        Initialize system monitor.

        Args:
            config: System monitoring configuration
        """
        self.config = config or SystemConfig()

        # History
        self._cpu_history: list[CPUMetrics] = []
        self._memory_history: list[MemoryMetrics] = []
        self._disk_history: dict[str, list[DiskMetrics]] = {}
        self._network_history: dict[str, list[NetworkMetrics]] = {}

        # Previous values for rate calculation
        self._prev_disk_io: dict[str, tuple[int, int]] = {}
        self._prev_net_io: dict[str, tuple[int, int]] = {}

        # Alert callbacks
        self._alert_callbacks: list[Callable[[str, str, float], None]] = []

        # Background task
        self._monitor_task: asyncio.Task | None = None
        self._running = False

        # System info
        self._hostname = platform.node()
        self._os_name = platform.system()
        self._os_version = platform.release()
        self._python_version = platform.python_version()
        self._boot_time = psutil.boot_time()

        logger.info("SystemMonitor initialized")

    def add_alert_callback(
        self,
        callback: Callable[[str, str, float], None],
    ) -> None:
        """
        Add callback for resource alerts.

        Args:
            callback: Function called with (resource, level, value)
        """
        self._alert_callbacks.append(callback)

    async def collect_cpu_metrics(self) -> CPUMetrics:
        """Collect CPU metrics."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        per_cpu = psutil.cpu_percent(percpu=True)

        try:
            load_avg = os.getloadavg()
        except (OSError, AttributeError):
            load_avg = (0.0, 0.0, 0.0)

        cpu_stats = psutil.cpu_stats()

        metrics = CPUMetrics(
            timestamp=datetime.now(),
            percent=cpu_percent,
            per_cpu=per_cpu,
            load_avg_1=load_avg[0],
            load_avg_5=load_avg[1],
            load_avg_15=load_avg[2],
            ctx_switches=cpu_stats.ctx_switches,
            interrupts=cpu_stats.interrupts,
        )

        self._cpu_history.append(metrics)
        self._trim_history(self._cpu_history)

        # Check thresholds
        if cpu_percent >= self.config.cpu_critical:
            self._notify_alert("cpu", "critical", cpu_percent)
        elif cpu_percent >= self.config.cpu_warning:
            self._notify_alert("cpu", "warning", cpu_percent)

        return metrics

    async def collect_memory_metrics(self) -> MemoryMetrics:
        """Collect memory metrics."""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        metrics = MemoryMetrics(
            timestamp=datetime.now(),
            total=mem.total,
            available=mem.available,
            used=mem.used,
            percent=mem.percent,
            swap_total=swap.total,
            swap_used=swap.used,
            swap_percent=swap.percent,
        )

        self._memory_history.append(metrics)
        self._trim_history(self._memory_history)

        # Check thresholds
        if mem.percent >= self.config.memory_critical:
            self._notify_alert("memory", "critical", mem.percent)
        elif mem.percent >= self.config.memory_warning:
            self._notify_alert("memory", "warning", mem.percent)

        return metrics

    async def collect_disk_metrics(self, path: str = "/") -> DiskMetrics:
        """Collect disk metrics for a path."""
        usage = psutil.disk_usage(path)

        # Get I/O stats
        io_counters = psutil.disk_io_counters()
        read_bytes = io_counters.read_bytes if io_counters else 0
        write_bytes = io_counters.write_bytes if io_counters else 0
        read_count = io_counters.read_count if io_counters else 0
        write_count = io_counters.write_count if io_counters else 0

        metrics = DiskMetrics(
            timestamp=datetime.now(),
            path=path,
            total=usage.total,
            used=usage.used,
            free=usage.free,
            percent=usage.percent,
            read_bytes=read_bytes,
            write_bytes=write_bytes,
            read_count=read_count,
            write_count=write_count,
        )

        if path not in self._disk_history:
            self._disk_history[path] = []

        self._disk_history[path].append(metrics)
        self._trim_history(self._disk_history[path])

        # Check thresholds
        if usage.percent >= self.config.disk_critical:
            self._notify_alert("disk", "critical", usage.percent)
        elif usage.percent >= self.config.disk_warning:
            self._notify_alert("disk", "warning", usage.percent)

        return metrics

    async def collect_network_metrics(
        self,
        interface: str | None = None,
    ) -> list[NetworkMetrics]:
        """Collect network metrics."""
        net_io = psutil.net_io_counters(pernic=True)
        metrics_list = []

        interfaces = [interface] if interface else list(net_io.keys())

        for iface in interfaces:
            if iface not in net_io:
                continue

            io = net_io[iface]

            metrics = NetworkMetrics(
                timestamp=datetime.now(),
                interface=iface,
                bytes_sent=io.bytes_sent,
                bytes_recv=io.bytes_recv,
                packets_sent=io.packets_sent,
                packets_recv=io.packets_recv,
                errors_in=io.errin,
                errors_out=io.errout,
                drops_in=io.dropin,
                drops_out=io.dropout,
            )

            if iface not in self._network_history:
                self._network_history[iface] = []

            self._network_history[iface].append(metrics)
            self._trim_history(self._network_history[iface])

            metrics_list.append(metrics)

        return metrics_list

    async def collect_process_metrics(
        self,
        process_names: list[str] | None = None,
    ) -> list[ProcessMetrics]:
        """Collect process metrics."""
        metrics_list = []
        process_names = process_names or self.config.tracked_processes

        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent',
                                          'memory_info', 'num_threads', 'status', 'create_time']):
            try:
                info = proc.info
                name = info.get('name', '')

                # Filter by name if specified
                if process_names and not any(pn.lower() in name.lower() for pn in process_names):
                    continue

                memory_info = info.get('memory_info')
                rss = memory_info.rss if memory_info else 0

                try:
                    num_fds = proc.num_fds()
                except (psutil.AccessDenied, AttributeError):
                    num_fds = 0

                metrics = ProcessMetrics(
                    timestamp=datetime.now(),
                    pid=info.get('pid', 0),
                    name=name,
                    cpu_percent=info.get('cpu_percent', 0.0) or 0.0,
                    memory_percent=info.get('memory_percent', 0.0) or 0.0,
                    memory_rss=rss,
                    num_threads=info.get('num_threads', 0) or 0,
                    status=info.get('status', 'unknown'),
                    create_time=info.get('create_time', 0.0) or 0.0,
                    num_fds=num_fds,
                )

                metrics_list.append(metrics)

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Sort by CPU usage
        return sorted(metrics_list, key=lambda p: p.cpu_percent, reverse=True)

    async def collect_snapshot(self) -> SystemSnapshot:
        """Collect complete system snapshot."""
        cpu = await self.collect_cpu_metrics()
        memory = await self.collect_memory_metrics()
        disk = await self.collect_disk_metrics()
        network = await self.collect_network_metrics()
        processes = await self.collect_process_metrics() if self.config.monitor_processes else []

        uptime = time.time() - self._boot_time

        return SystemSnapshot(
            timestamp=datetime.now(),
            cpu=cpu,
            memory=memory,
            disks=[disk],
            network=network,
            processes=processes,
            hostname=self._hostname,
            os_name=self._os_name,
            os_version=self._os_version,
            python_version=self._python_version,
            uptime=uptime,
        )

    def _notify_alert(self, resource: str, level: str, value: float) -> None:
        """Notify alert callbacks."""
        for callback in self._alert_callbacks:
            try:
                callback(resource, level, value)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def _trim_history(self, history: list) -> None:
        """Trim history to configured size."""
        while len(history) > self.config.history_size:
            history.pop(0)

    async def start(self) -> None:
        """Start monitoring."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("System monitoring started")

    async def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("System monitoring stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self.collect_snapshot()
                await asyncio.sleep(self.config.poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(self.config.poll_interval)

    def get_cpu_history(self, minutes: int = 60) -> list[CPUMetrics]:
        """Get CPU history."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self._cpu_history if m.timestamp >= cutoff]

    def get_memory_history(self, minutes: int = 60) -> list[MemoryMetrics]:
        """Get memory history."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self._memory_history if m.timestamp >= cutoff]

    def get_current_metrics(self) -> dict[str, Any]:
        """Get current system metrics."""
        return {
            "cpu": self._cpu_history[-1].to_dict() if self._cpu_history else None,
            "memory": self._memory_history[-1].to_dict() if self._memory_history else None,
            "disks": {
                path: history[-1].to_dict() if history else None
                for path, history in self._disk_history.items()
            },
        }


def create_system_monitor(
    config: SystemConfig | None = None,
) -> SystemMonitor:
    """
    Create a system monitor instance.

    Args:
        config: System monitoring configuration

    Returns:
        SystemMonitor instance
    """
    return SystemMonitor(config)
