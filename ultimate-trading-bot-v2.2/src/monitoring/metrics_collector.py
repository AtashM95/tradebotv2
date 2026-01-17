"""
Metrics Collector for Ultimate Trading Bot v2.2.

Collects and aggregates system, trading, and performance metrics.
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


@dataclass
class MetricConfig:
    """Configuration for metrics collection."""

    # Collection settings
    collection_interval: float = 60.0  # seconds
    retention_hours: int = 24
    max_samples: int = 10000

    # Histogram settings
    histogram_buckets: list[float] = field(default_factory=lambda: [
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
    ])

    # Summary settings
    summary_quantiles: list[float] = field(default_factory=lambda: [
        0.5, 0.75, 0.9, 0.95, 0.99
    ])


@dataclass
class MetricSample:
    """A single metric sample."""

    timestamp: datetime
    value: float
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class MetricStats:
    """Statistics for a metric."""

    name: str
    metric_type: MetricType
    sample_count: int = 0
    sum_value: float = 0.0
    min_value: float = float('inf')
    max_value: float = float('-inf')
    last_value: float = 0.0
    mean_value: float = 0.0
    std_value: float = 0.0

    # For histograms
    bucket_counts: dict[float, int] = field(default_factory=dict)

    # For summaries
    quantile_values: dict[float, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.metric_type.value,
            "count": self.sample_count,
            "sum": self.sum_value,
            "min": self.min_value if self.min_value != float('inf') else 0.0,
            "max": self.max_value if self.max_value != float('-inf') else 0.0,
            "last": self.last_value,
            "mean": self.mean_value,
            "std": self.std_value,
            "buckets": self.bucket_counts,
            "quantiles": self.quantile_values,
        }


class Metric:
    """
    Base class for metrics.
    """

    def __init__(
        self,
        name: str,
        metric_type: MetricType,
        description: str = "",
        labels: list[str] | None = None,
    ) -> None:
        """
        Initialize metric.

        Args:
            name: Metric name
            metric_type: Type of metric
            description: Metric description
            labels: Label names for this metric
        """
        self.name = name
        self.metric_type = metric_type
        self.description = description
        self.label_names = labels or []

        self._samples: list[MetricSample] = []
        self._lock = asyncio.Lock()

    def _get_label_key(self, labels: dict[str, str]) -> str:
        """Get unique key for label combination."""
        if not labels:
            return ""
        return ":".join(f"{k}={v}" for k, v in sorted(labels.items()))

    def get_stats(self) -> MetricStats:
        """Get metric statistics."""
        if not self._samples:
            return MetricStats(name=self.name, metric_type=self.metric_type)

        values = [s.value for s in self._samples]
        arr = np.array(values)

        return MetricStats(
            name=self.name,
            metric_type=self.metric_type,
            sample_count=len(values),
            sum_value=float(np.sum(arr)),
            min_value=float(np.min(arr)),
            max_value=float(np.max(arr)),
            last_value=values[-1],
            mean_value=float(np.mean(arr)),
            std_value=float(np.std(arr)) if len(values) > 1 else 0.0,
        )

    def clear(self) -> None:
        """Clear all samples."""
        self._samples.clear()


class Counter(Metric):
    """
    Monotonically increasing counter metric.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
    ) -> None:
        """Initialize counter."""
        super().__init__(name, MetricType.COUNTER, description, labels)
        self._values: dict[str, float] = defaultdict(float)

    async def inc(self, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """
        Increment counter.

        Args:
            value: Value to increment by
            labels: Label values
        """
        async with self._lock:
            key = self._get_label_key(labels or {})
            self._values[key] += value
            self._samples.append(MetricSample(
                timestamp=datetime.now(),
                value=self._values[key],
                labels=labels or {},
            ))

    def get_value(self, labels: dict[str, str] | None = None) -> float:
        """Get current counter value."""
        key = self._get_label_key(labels or {})
        return self._values[key]


class Gauge(Metric):
    """
    Gauge metric that can go up or down.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
    ) -> None:
        """Initialize gauge."""
        super().__init__(name, MetricType.GAUGE, description, labels)
        self._values: dict[str, float] = defaultdict(float)

    async def set(self, value: float, labels: dict[str, str] | None = None) -> None:
        """
        Set gauge value.

        Args:
            value: Value to set
            labels: Label values
        """
        async with self._lock:
            key = self._get_label_key(labels or {})
            self._values[key] = value
            self._samples.append(MetricSample(
                timestamp=datetime.now(),
                value=value,
                labels=labels or {},
            ))

    async def inc(self, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """Increment gauge."""
        async with self._lock:
            key = self._get_label_key(labels or {})
            self._values[key] += value
            self._samples.append(MetricSample(
                timestamp=datetime.now(),
                value=self._values[key],
                labels=labels or {},
            ))

    async def dec(self, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """Decrement gauge."""
        await self.inc(-value, labels)

    def get_value(self, labels: dict[str, str] | None = None) -> float:
        """Get current gauge value."""
        key = self._get_label_key(labels or {})
        return self._values[key]


class Histogram(Metric):
    """
    Histogram metric for measuring distributions.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
        buckets: list[float] | None = None,
    ) -> None:
        """Initialize histogram."""
        super().__init__(name, MetricType.HISTOGRAM, description, labels)
        self.buckets = sorted(buckets or [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0])
        self._bucket_counts: dict[str, dict[float, int]] = defaultdict(
            lambda: {b: 0 for b in self.buckets + [float('inf')]}
        )
        self._sums: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)

    async def observe(self, value: float, labels: dict[str, str] | None = None) -> None:
        """
        Observe a value.

        Args:
            value: Value to observe
            labels: Label values
        """
        async with self._lock:
            key = self._get_label_key(labels or {})

            # Update counts
            self._sums[key] += value
            self._counts[key] += 1

            # Update buckets
            for bucket in self.buckets + [float('inf')]:
                if value <= bucket:
                    self._bucket_counts[key][bucket] += 1

            self._samples.append(MetricSample(
                timestamp=datetime.now(),
                value=value,
                labels=labels or {},
            ))

    def get_stats(self) -> MetricStats:
        """Get histogram statistics."""
        stats = super().get_stats()

        # Aggregate bucket counts
        total_buckets: dict[float, int] = {b: 0 for b in self.buckets + [float('inf')]}
        for bucket_counts in self._bucket_counts.values():
            for bucket, count in bucket_counts.items():
                total_buckets[bucket] += count

        stats.bucket_counts = total_buckets
        return stats


class Summary(Metric):
    """
    Summary metric for calculating quantiles.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
        quantiles: list[float] | None = None,
        max_age_seconds: float = 600.0,
    ) -> None:
        """Initialize summary."""
        super().__init__(name, MetricType.SUMMARY, description, labels)
        self.quantiles = quantiles or [0.5, 0.9, 0.99]
        self.max_age_seconds = max_age_seconds
        self._values: dict[str, list[tuple[datetime, float]]] = defaultdict(list)

    async def observe(self, value: float, labels: dict[str, str] | None = None) -> None:
        """
        Observe a value.

        Args:
            value: Value to observe
            labels: Label values
        """
        async with self._lock:
            key = self._get_label_key(labels or {})
            now = datetime.now()

            # Add new value
            self._values[key].append((now, value))

            # Remove old values
            cutoff = now - timedelta(seconds=self.max_age_seconds)
            self._values[key] = [
                (ts, v) for ts, v in self._values[key]
                if ts >= cutoff
            ]

            self._samples.append(MetricSample(
                timestamp=now,
                value=value,
                labels=labels or {},
            ))

    def get_stats(self) -> MetricStats:
        """Get summary statistics."""
        stats = super().get_stats()

        # Calculate quantiles from all values
        all_values = []
        for values in self._values.values():
            all_values.extend(v for _, v in values)

        if all_values:
            arr = np.array(all_values)
            stats.quantile_values = {
                q: float(np.percentile(arr, q * 100))
                for q in self.quantiles
            }

        return stats


class Timer(Histogram):
    """
    Timer metric for measuring durations.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
        buckets: list[float] | None = None,
    ) -> None:
        """Initialize timer."""
        default_buckets = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        super().__init__(name, description, labels, buckets or default_buckets)
        self.metric_type = MetricType.TIMER

    def time(self) -> "TimerContext":
        """Create timer context manager."""
        return TimerContext(self)


class TimerContext:
    """Context manager for timing operations."""

    def __init__(self, timer: Timer, labels: dict[str, str] | None = None) -> None:
        """Initialize timer context."""
        self.timer = timer
        self.labels = labels
        self._start: float = 0.0

    def __enter__(self) -> "TimerContext":
        """Start timing."""
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop timing and record."""
        duration = time.perf_counter() - self._start
        asyncio.create_task(self.timer.observe(duration, self.labels))

    async def __aenter__(self) -> "TimerContext":
        """Start timing (async)."""
        self._start = time.perf_counter()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Stop timing and record (async)."""
        duration = time.perf_counter() - self._start
        await self.timer.observe(duration, self.labels)


class MetricsCollector:
    """
    Central metrics collection system.

    Manages all metrics and provides collection and export functionality.
    """

    def __init__(self, config: MetricConfig | None = None) -> None:
        """
        Initialize metrics collector.

        Args:
            config: Metrics configuration
        """
        self.config = config or MetricConfig()

        # Registered metrics
        self._metrics: dict[str, Metric] = {}

        # Collection task
        self._collection_task: asyncio.Task | None = None
        self._running = False

        # Callbacks for metric updates
        self._callbacks: list[Callable[[str, MetricStats], None]] = []

        logger.info("MetricsCollector initialized")

    def register(self, metric: Metric) -> Metric:
        """
        Register a metric.

        Args:
            metric: Metric to register

        Returns:
            Registered metric
        """
        if metric.name in self._metrics:
            logger.warning(f"Metric already registered: {metric.name}")
            return self._metrics[metric.name]

        self._metrics[metric.name] = metric
        logger.debug(f"Registered metric: {metric.name}")
        return metric

    def counter(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
    ) -> Counter:
        """Create and register a counter metric."""
        metric = Counter(name, description, labels)
        return self.register(metric)

    def gauge(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
    ) -> Gauge:
        """Create and register a gauge metric."""
        metric = Gauge(name, description, labels)
        return self.register(metric)

    def histogram(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
        buckets: list[float] | None = None,
    ) -> Histogram:
        """Create and register a histogram metric."""
        metric = Histogram(name, description, labels, buckets or self.config.histogram_buckets)
        return self.register(metric)

    def summary(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
        quantiles: list[float] | None = None,
    ) -> Summary:
        """Create and register a summary metric."""
        metric = Summary(name, description, labels, quantiles or self.config.summary_quantiles)
        return self.register(metric)

    def timer(
        self,
        name: str,
        description: str = "",
        labels: list[str] | None = None,
        buckets: list[float] | None = None,
    ) -> Timer:
        """Create and register a timer metric."""
        metric = Timer(name, description, labels, buckets)
        return self.register(metric)

    def get_metric(self, name: str) -> Metric | None:
        """Get a metric by name."""
        return self._metrics.get(name)

    def get_all_stats(self) -> dict[str, MetricStats]:
        """Get statistics for all metrics."""
        return {name: metric.get_stats() for name, metric in self._metrics.items()}

    def add_callback(self, callback: Callable[[str, MetricStats], None]) -> None:
        """Add callback for metric updates."""
        self._callbacks.append(callback)

    async def start_collection(self) -> None:
        """Start periodic metrics collection."""
        if self._running:
            return

        self._running = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Metrics collection started")

    async def stop_collection(self) -> None:
        """Stop metrics collection."""
        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass

        logger.info("Metrics collection stopped")

    async def _collection_loop(self) -> None:
        """Main collection loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.collection_interval)

                # Collect and notify
                for name, metric in self._metrics.items():
                    stats = metric.get_stats()
                    for callback in self._callbacks:
                        try:
                            callback(name, stats)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")

                # Trim old samples
                self._trim_samples()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Collection error: {e}")

    def _trim_samples(self) -> None:
        """Trim old samples based on retention settings."""
        cutoff = datetime.now() - timedelta(hours=self.config.retention_hours)

        for metric in self._metrics.values():
            metric._samples = [
                s for s in metric._samples
                if s.timestamp >= cutoff
            ][-self.config.max_samples:]

    def clear_all(self) -> None:
        """Clear all metrics."""
        for metric in self._metrics.values():
            metric.clear()

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        for name, metric in self._metrics.items():
            stats = metric.get_stats()

            # Add help and type
            lines.append(f"# HELP {name} {metric.description}")
            lines.append(f"# TYPE {name} {metric.metric_type.value}")

            if metric.metric_type == MetricType.COUNTER:
                lines.append(f"{name} {stats.last_value}")

            elif metric.metric_type == MetricType.GAUGE:
                lines.append(f"{name} {stats.last_value}")

            elif metric.metric_type in [MetricType.HISTOGRAM, MetricType.TIMER]:
                for bucket, count in stats.bucket_counts.items():
                    if bucket == float('inf'):
                        lines.append(f'{name}_bucket{{le="+Inf"}} {count}')
                    else:
                        lines.append(f'{name}_bucket{{le="{bucket}"}} {count}')
                lines.append(f"{name}_sum {stats.sum_value}")
                lines.append(f"{name}_count {stats.sample_count}")

            elif metric.metric_type == MetricType.SUMMARY:
                for quantile, value in stats.quantile_values.items():
                    lines.append(f'{name}{{quantile="{quantile}"}} {value}')
                lines.append(f"{name}_sum {stats.sum_value}")
                lines.append(f"{name}_count {stats.sample_count}")

        return "\n".join(lines)


def create_metrics_collector(
    config: MetricConfig | None = None,
) -> MetricsCollector:
    """
    Create a metrics collector instance.

    Args:
        config: Metrics configuration

    Returns:
        MetricsCollector instance
    """
    return MetricsCollector(config)
