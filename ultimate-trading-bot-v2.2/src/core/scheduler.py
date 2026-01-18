"""
Scheduler Module for Ultimate Trading Bot v2.2.

This module provides task scheduling functionality including
periodic tasks, cron-like scheduling, and market-aware scheduling.
"""

import asyncio
import logging
from datetime import datetime, time, timedelta
from enum import Enum
from heapq import heappush, heappop
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field, field_validator

from src.utils.exceptions import ValidationError
from src.utils.helpers import generate_uuid
from src.utils.date_utils import (
    now_utc,
    now_et,
    is_trading_day,
    get_next_trading_day,
    is_market_open,
)
from src.utils.decorators import singleton


logger = logging.getLogger(__name__)


class TaskPriority(int, Enum):
    """Task priority enumeration."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class TaskStatus(str, Enum):
    """Task status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ScheduleType(str, Enum):
    """Schedule type enumeration."""

    ONCE = "once"
    INTERVAL = "interval"
    DAILY = "daily"
    WEEKDAY = "weekday"
    MARKET_OPEN = "market_open"
    MARKET_CLOSE = "market_close"
    CRON = "cron"


class ScheduledTask(BaseModel):
    """Scheduled task model."""

    task_id: str = Field(default_factory=generate_uuid)
    name: str
    description: str = Field(default="")

    schedule_type: ScheduleType = Field(default=ScheduleType.ONCE)
    priority: TaskPriority = Field(default=TaskPriority.NORMAL)
    status: TaskStatus = Field(default=TaskStatus.PENDING)

    interval_seconds: Optional[int] = None
    daily_time: Optional[time] = None
    cron_expression: Optional[str] = None

    next_run: Optional[datetime] = None
    last_run: Optional[datetime] = None
    last_result: Optional[str] = None
    last_error: Optional[str] = None

    run_count: int = Field(default=0)
    error_count: int = Field(default=0)
    max_retries: int = Field(default=3)
    retry_delay_seconds: int = Field(default=5)

    timeout_seconds: Optional[int] = None
    enabled: bool = Field(default=True)
    market_hours_only: bool = Field(default=False)
    trading_days_only: bool = Field(default=False)

    created_at: datetime = Field(default_factory=now_utc)
    updated_at: datetime = Field(default_factory=now_utc)

    metadata: dict = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True

    @field_validator("interval_seconds")
    @classmethod
    def validate_interval(cls, v: Optional[int]) -> Optional[int]:
        """Validate interval is positive."""
        if v is not None and v <= 0:
            raise ValueError("Interval must be positive")
        return v

    @property
    def is_active(self) -> bool:
        """Check if task is active."""
        return self.enabled and self.status in (
            TaskStatus.PENDING,
            TaskStatus.COMPLETED
        )

    @property
    def is_recurring(self) -> bool:
        """Check if task is recurring."""
        return self.schedule_type != ScheduleType.ONCE

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "schedule_type": self.schedule_type.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "interval_seconds": self.interval_seconds,
            "daily_time": self.daily_time.isoformat() if self.daily_time else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "run_count": self.run_count,
            "error_count": self.error_count,
            "enabled": self.enabled,
        }


class TaskResult(BaseModel):
    """Result of task execution."""

    task_id: str
    success: bool = Field(default=False)
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=now_utc)


class SchedulerConfig(BaseModel):
    """Configuration for scheduler."""

    max_concurrent_tasks: int = Field(default=10, ge=1, le=100)
    default_timeout_seconds: int = Field(default=300, ge=10, le=3600)
    tick_interval_seconds: float = Field(default=1.0, ge=0.1, le=10.0)
    enable_market_aware_scheduling: bool = Field(default=True)
    max_task_history: int = Field(default=1000, ge=100, le=10000)
    cleanup_completed_after_hours: int = Field(default=24, ge=1, le=168)


@singleton
class Scheduler:
    """
    Task scheduler for the trading bot.

    This class provides:
    - One-time and recurring task scheduling
    - Priority-based execution
    - Market-aware scheduling
    - Task monitoring and history
    """

    def __init__(
        self,
        config: Optional[SchedulerConfig] = None,
    ) -> None:
        """
        Initialize Scheduler.

        Args:
            config: Scheduler configuration
        """
        self._config = config or SchedulerConfig()

        self._tasks: dict[str, ScheduledTask] = {}
        self._task_functions: dict[str, Callable] = {}

        self._task_queue: list[tuple[datetime, int, str]] = []

        self._task_results: list[TaskResult] = []

        self._running_tasks: set[str] = set()
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent_tasks)

        self._scheduler_task: Optional[asyncio.Task] = None
        self._running = False
        self._paused = False
        self._lock = asyncio.Lock()

        logger.info("Scheduler initialized")

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running

    @property
    def is_paused(self) -> bool:
        """Check if scheduler is paused."""
        return self._paused

    @property
    def task_count(self) -> int:
        """Get total task count."""
        return len(self._tasks)

    @property
    def pending_task_count(self) -> int:
        """Get pending task count."""
        return len(self._task_queue)

    @property
    def running_task_count(self) -> int:
        """Get running task count."""
        return len(self._running_tasks)

    async def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True
        self._paused = False

        self._scheduler_task = asyncio.create_task(
            self._scheduler_loop(),
            name="scheduler_loop"
        )

        logger.info("Scheduler started")

    async def stop(self) -> None:
        """Stop the scheduler."""
        if not self._running:
            return

        self._running = False

        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
            self._scheduler_task = None

        logger.info("Scheduler stopped")

    def pause(self) -> None:
        """Pause the scheduler."""
        self._paused = True
        logger.info("Scheduler paused")

    def resume(self) -> None:
        """Resume the scheduler."""
        self._paused = False
        logger.info("Scheduler resumed")

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                if self._paused:
                    await asyncio.sleep(self._config.tick_interval_seconds)
                    continue

                now = now_utc()

                tasks_to_run = []
                async with self._lock:
                    while (
                        self._task_queue
                        and self._task_queue[0][0] <= now
                    ):
                        _, _, task_id = heappop(self._task_queue)
                        if task_id in self._tasks:
                            tasks_to_run.append(task_id)

                for task_id in tasks_to_run:
                    asyncio.create_task(
                        self._execute_task(task_id),
                        name=f"task_{task_id}"
                    )

                await asyncio.sleep(self._config.tick_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(1)

    def schedule(
        self,
        name: str,
        func: Callable,
        schedule_type: ScheduleType = ScheduleType.ONCE,
        priority: TaskPriority = TaskPriority.NORMAL,
        interval_seconds: Optional[int] = None,
        daily_time: Optional[time] = None,
        delay_seconds: int = 0,
        timeout_seconds: Optional[int] = None,
        market_hours_only: bool = False,
        trading_days_only: bool = False,
        max_retries: int = 3,
        description: str = "",
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Schedule a task.

        Args:
            name: Task name
            func: Function to execute
            schedule_type: Type of schedule
            priority: Task priority
            interval_seconds: Interval for recurring tasks
            daily_time: Time for daily tasks
            delay_seconds: Initial delay
            timeout_seconds: Task timeout
            market_hours_only: Only run during market hours
            trading_days_only: Only run on trading days
            max_retries: Maximum retry attempts
            description: Task description
            metadata: Additional metadata

        Returns:
            Task ID
        """
        if schedule_type == ScheduleType.INTERVAL and not interval_seconds:
            raise ValidationError("Interval required for interval schedule")

        if schedule_type == ScheduleType.DAILY and not daily_time:
            raise ValidationError("Daily time required for daily schedule")

        task = ScheduledTask(
            name=name,
            description=description,
            schedule_type=schedule_type,
            priority=priority,
            interval_seconds=interval_seconds,
            daily_time=daily_time,
            timeout_seconds=timeout_seconds or self._config.default_timeout_seconds,
            market_hours_only=market_hours_only,
            trading_days_only=trading_days_only,
            max_retries=max_retries,
            metadata=metadata or {},
        )

        task.next_run = self._calculate_next_run(task, delay_seconds)

        self._tasks[task.task_id] = task
        self._task_functions[task.task_id] = func

        self._enqueue_task(task)

        logger.info(
            f"Scheduled task '{name}' (id={task.task_id}, "
            f"type={schedule_type.value}, next_run={task.next_run})"
        )

        return task.task_id

    def schedule_once(
        self,
        name: str,
        func: Callable,
        delay_seconds: int = 0,
        priority: TaskPriority = TaskPriority.NORMAL,
        **kwargs
    ) -> str:
        """Schedule a one-time task."""
        return self.schedule(
            name=name,
            func=func,
            schedule_type=ScheduleType.ONCE,
            delay_seconds=delay_seconds,
            priority=priority,
            **kwargs
        )

    def schedule_interval(
        self,
        name: str,
        func: Callable,
        interval_seconds: int,
        priority: TaskPriority = TaskPriority.NORMAL,
        **kwargs
    ) -> str:
        """Schedule a recurring interval task."""
        return self.schedule(
            name=name,
            func=func,
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=interval_seconds,
            priority=priority,
            **kwargs
        )

    def schedule_daily(
        self,
        name: str,
        func: Callable,
        daily_time: time,
        priority: TaskPriority = TaskPriority.NORMAL,
        **kwargs
    ) -> str:
        """Schedule a daily task."""
        return self.schedule(
            name=name,
            func=func,
            schedule_type=ScheduleType.DAILY,
            daily_time=daily_time,
            priority=priority,
            **kwargs
        )

    def schedule_at_market_open(
        self,
        name: str,
        func: Callable,
        offset_minutes: int = 0,
        priority: TaskPriority = TaskPriority.HIGH,
        **kwargs
    ) -> str:
        """Schedule a task at market open."""
        return self.schedule(
            name=name,
            func=func,
            schedule_type=ScheduleType.MARKET_OPEN,
            trading_days_only=True,
            priority=priority,
            metadata={"offset_minutes": offset_minutes},
            **kwargs
        )

    def schedule_at_market_close(
        self,
        name: str,
        func: Callable,
        offset_minutes: int = 0,
        priority: TaskPriority = TaskPriority.HIGH,
        **kwargs
    ) -> str:
        """Schedule a task at market close."""
        return self.schedule(
            name=name,
            func=func,
            schedule_type=ScheduleType.MARKET_CLOSE,
            trading_days_only=True,
            priority=priority,
            metadata={"offset_minutes": offset_minutes},
            **kwargs
        )

    def _calculate_next_run(
        self,
        task: ScheduledTask,
        delay_seconds: int = 0
    ) -> datetime:
        """Calculate the next run time for a task."""
        now = now_utc()
        base_time = now + timedelta(seconds=delay_seconds)

        if task.schedule_type == ScheduleType.ONCE:
            return base_time

        elif task.schedule_type == ScheduleType.INTERVAL:
            if task.last_run:
                return task.last_run + timedelta(seconds=task.interval_seconds or 0)
            return base_time

        elif task.schedule_type == ScheduleType.DAILY:
            if not task.daily_time:
                return base_time

            today = now_et().date()
            run_time = datetime.combine(today, task.daily_time)

            if run_time.replace(tzinfo=None) <= now_et().replace(tzinfo=None):
                run_time = datetime.combine(
                    today + timedelta(days=1),
                    task.daily_time
                )

            return run_time

        elif task.schedule_type == ScheduleType.WEEKDAY:
            if not task.daily_time:
                return base_time

            current = now_et()
            today = current.date()

            while not is_trading_day(today):
                today = get_next_trading_day(today)

            run_time = datetime.combine(today, task.daily_time)

            if run_time.replace(tzinfo=None) <= current.replace(tzinfo=None):
                today = get_next_trading_day(today)
                run_time = datetime.combine(today, task.daily_time)

            return run_time

        elif task.schedule_type == ScheduleType.MARKET_OPEN:
            from src.core.market_hours import MarketHours
            market_hours = MarketHours()

            offset = task.metadata.get("offset_minutes", 0)
            schedule = market_hours.get_schedule()

            if schedule.regular_open:
                open_time = schedule.regular_open + timedelta(minutes=offset)
                if open_time > now:
                    return open_time

            next_day = get_next_trading_day(now_et().date())
            next_schedule = market_hours.get_schedule(next_day)
            if next_schedule.regular_open:
                return next_schedule.regular_open + timedelta(minutes=offset)

            return base_time

        elif task.schedule_type == ScheduleType.MARKET_CLOSE:
            from src.core.market_hours import MarketHours
            market_hours = MarketHours()

            offset = task.metadata.get("offset_minutes", 0)
            schedule = market_hours.get_schedule()

            if schedule.regular_close:
                close_time = schedule.regular_close - timedelta(minutes=offset)
                if close_time > now:
                    return close_time

            next_day = get_next_trading_day(now_et().date())
            next_schedule = market_hours.get_schedule(next_day)
            if next_schedule.regular_close:
                return next_schedule.regular_close - timedelta(minutes=offset)

            return base_time

        return base_time

    def _enqueue_task(self, task: ScheduledTask) -> None:
        """Add task to the execution queue."""
        if task.next_run:
            heappush(
                self._task_queue,
                (task.next_run, task.priority.value, task.task_id)
            )

    async def _execute_task(self, task_id: str) -> None:
        """Execute a scheduled task."""
        task = self._tasks.get(task_id)
        if not task or not task.enabled:
            return

        func = self._task_functions.get(task_id)
        if not func:
            logger.error(f"No function for task {task_id}")
            return

        if task_id in self._running_tasks:
            logger.warning(f"Task {task_id} already running")
            return

        if not self._should_run_task(task):
            self._reschedule_task(task)
            return

        async with self._semaphore:
            self._running_tasks.add(task_id)
            task.status = TaskStatus.RUNNING

            start_time = datetime.now()
            result = TaskResult(task_id=task_id)

            try:
                if task.timeout_seconds:
                    if asyncio.iscoroutinefunction(func):
                        task_result = await asyncio.wait_for(
                            func(),
                            timeout=task.timeout_seconds
                        )
                    else:
                        task_result = await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(None, func),
                            timeout=task.timeout_seconds
                        )
                else:
                    if asyncio.iscoroutinefunction(func):
                        task_result = await func()
                    else:
                        task_result = func()

                result.success = True
                result.result = task_result
                task.status = TaskStatus.COMPLETED
                task.last_result = str(task_result)[:500] if task_result else None

            except asyncio.TimeoutError:
                result.error = "Task timeout"
                task.status = TaskStatus.FAILED
                task.last_error = "Timeout"
                task.error_count += 1

            except Exception as e:
                result.error = str(e)
                task.status = TaskStatus.FAILED
                task.last_error = str(e)
                task.error_count += 1
                logger.error(f"Task {task.name} failed: {e}")

            finally:
                end_time = datetime.now()
                result.execution_time_ms = (end_time - start_time).total_seconds() * 1000

                task.last_run = now_utc()
                task.run_count += 1
                task.updated_at = now_utc()

                self._running_tasks.discard(task_id)
                self._task_results.append(result)
                self._trim_results()

                if task.is_recurring:
                    self._reschedule_task(task)

                logger.debug(
                    f"Task '{task.name}' completed "
                    f"(success={result.success}, time={result.execution_time_ms:.2f}ms)"
                )

    def _should_run_task(self, task: ScheduledTask) -> bool:
        """Check if task should run based on constraints."""
        if task.trading_days_only:
            if not is_trading_day(now_et().date()):
                return False

        if task.market_hours_only:
            if not is_market_open():
                return False

        return True

    def _reschedule_task(self, task: ScheduledTask) -> None:
        """Reschedule a recurring task."""
        if task.is_recurring and task.enabled:
            task.next_run = self._calculate_next_run(task)
            task.status = TaskStatus.PENDING
            self._enqueue_task(task)

    def _trim_results(self) -> None:
        """Trim task results to max size."""
        if len(self._task_results) > self._config.max_task_history:
            self._task_results = self._task_results[-self._config.max_task_history:]

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a scheduled task.

        Args:
            task_id: Task to cancel

        Returns:
            True if cancelled
        """
        if task_id not in self._tasks:
            return False

        task = self._tasks[task_id]
        task.status = TaskStatus.CANCELLED
        task.enabled = False

        logger.info(f"Cancelled task: {task.name}")
        return True

    def enable_task(self, task_id: str) -> bool:
        """Enable a task."""
        if task_id not in self._tasks:
            return False

        task = self._tasks[task_id]
        task.enabled = True
        task.status = TaskStatus.PENDING

        if task.next_run is None or task.next_run < now_utc():
            task.next_run = self._calculate_next_run(task)

        self._enqueue_task(task)
        logger.info(f"Enabled task: {task.name}")
        return True

    def disable_task(self, task_id: str) -> bool:
        """Disable a task."""
        if task_id not in self._tasks:
            return False

        task = self._tasks[task_id]
        task.enabled = False
        logger.info(f"Disabled task: {task.name}")
        return True

    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> list[ScheduledTask]:
        """Get all scheduled tasks."""
        return list(self._tasks.values())

    def get_pending_tasks(self) -> list[ScheduledTask]:
        """Get pending tasks."""
        return [
            t for t in self._tasks.values()
            if t.status == TaskStatus.PENDING
        ]

    def get_running_tasks(self) -> list[ScheduledTask]:
        """Get running tasks."""
        return [
            t for t in self._tasks.values()
            if t.task_id in self._running_tasks
        ]

    def get_task_results(
        self,
        task_id: Optional[str] = None,
        limit: int = 100
    ) -> list[TaskResult]:
        """Get task execution results."""
        results = self._task_results

        if task_id:
            results = [r for r in results if r.task_id == task_id]

        return results[-limit:]

    def get_statistics(self) -> dict:
        """Get scheduler statistics."""
        total_runs = sum(t.run_count for t in self._tasks.values())
        total_errors = sum(t.error_count for t in self._tasks.values())

        return {
            "is_running": self._running,
            "is_paused": self._paused,
            "total_tasks": len(self._tasks),
            "pending_tasks": len(self.get_pending_tasks()),
            "running_tasks": len(self._running_tasks),
            "total_runs": total_runs,
            "total_errors": total_errors,
            "error_rate": (total_errors / total_runs * 100) if total_runs > 0 else 0,
            "results_in_history": len(self._task_results),
        }

    def remove_task(self, task_id: str) -> bool:
        """Remove a task completely."""
        if task_id not in self._tasks:
            return False

        del self._tasks[task_id]
        self._task_functions.pop(task_id, None)

        logger.info(f"Removed task: {task_id}")
        return True

    def clear_completed_tasks(self) -> int:
        """Clear all completed one-time tasks."""
        to_remove = [
            task_id for task_id, task in self._tasks.items()
            if task.status == TaskStatus.COMPLETED
            and task.schedule_type == ScheduleType.ONCE
        ]

        for task_id in to_remove:
            del self._tasks[task_id]
            self._task_functions.pop(task_id, None)

        return len(to_remove)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Scheduler(tasks={len(self._tasks)}, "
            f"running={self._running}, paused={self._paused})"
        )
