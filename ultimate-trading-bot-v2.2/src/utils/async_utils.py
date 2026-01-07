"""
Async Utilities Module for Ultimate Trading Bot v2.2.

This module provides async utilities for concurrent execution, task management,
and async patterns used throughout the trading bot.
"""

import asyncio
import functools
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import asynccontextmanager
from typing import (
    Any, AsyncGenerator, AsyncIterator, Awaitable, Callable, Coroutine,
    Dict, Iterable, List, Optional, Set, Tuple, TypeVar, Union
)
import logging
import signal
import weakref


logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


# =============================================================================
# ASYNC EXECUTION UTILITIES
# =============================================================================

async def gather_with_concurrency(
    n: int,
    *coros: Coroutine[Any, Any, T]
) -> List[T]:
    """
    Run coroutines with limited concurrency.

    Args:
        n: Maximum concurrent coroutines
        *coros: Coroutines to run

    Returns:
        List of results in order
    """
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro: Coroutine[Any, Any, T]) -> T:
        async with semaphore:
            return await coro

    return await asyncio.gather(*(sem_coro(c) for c in coros))


async def gather_with_exceptions(
    *coros: Coroutine[Any, Any, T],
    return_exceptions: bool = True
) -> List[Union[T, Exception]]:
    """
    Run coroutines and return results including exceptions.

    Args:
        *coros: Coroutines to run
        return_exceptions: Whether to return exceptions

    Returns:
        List of results or exceptions
    """
    return await asyncio.gather(*coros, return_exceptions=return_exceptions)


async def run_with_timeout(
    coro: Coroutine[Any, Any, T],
    timeout: float,
    default: Optional[T] = None
) -> Optional[T]:
    """
    Run a coroutine with timeout, returning default on timeout.

    Args:
        coro: Coroutine to run
        timeout: Timeout in seconds
        default: Default value on timeout

    Returns:
        Result or default
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Coroutine timed out after {timeout}s")
        return default


async def first_completed(
    *coros: Coroutine[Any, Any, T],
    timeout: Optional[float] = None
) -> Tuple[T, List[asyncio.Task]]:
    """
    Return the result of the first completed coroutine.

    Args:
        *coros: Coroutines to run
        timeout: Optional timeout

    Returns:
        Tuple of (result, pending_tasks)
    """
    tasks = [asyncio.create_task(coro) for coro in coros]

    done, pending = await asyncio.wait(
        tasks,
        timeout=timeout,
        return_when=asyncio.FIRST_COMPLETED
    )

    # Cancel pending tasks
    for task in pending:
        task.cancel()

    if done:
        task = done.pop()
        return await task, list(pending)

    raise asyncio.TimeoutError("All tasks timed out")


async def all_completed(
    *coros: Coroutine[Any, Any, T],
    timeout: Optional[float] = None
) -> List[T]:
    """
    Wait for all coroutines to complete.

    Args:
        *coros: Coroutines to run
        timeout: Optional timeout

    Returns:
        List of results

    Raises:
        asyncio.TimeoutError: If timeout exceeded
    """
    tasks = [asyncio.create_task(coro) for coro in coros]

    done, pending = await asyncio.wait(
        tasks,
        timeout=timeout,
        return_when=asyncio.ALL_COMPLETED
    )

    if pending:
        for task in pending:
            task.cancel()
        raise asyncio.TimeoutError(f"{len(pending)} tasks timed out")

    return [await task for task in done]


# =============================================================================
# ASYNC ITERATION UTILITIES
# =============================================================================

async def async_map(
    func: Callable[[T], Awaitable[R]],
    items: Iterable[T],
    concurrency: int = 10
) -> AsyncGenerator[R, None]:
    """
    Async map with concurrency control.

    Args:
        func: Async function to apply
        items: Items to process
        concurrency: Maximum concurrent operations

    Yields:
        Results as they complete
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def process(item: T) -> R:
        async with semaphore:
            return await func(item)

    tasks = [asyncio.create_task(process(item)) for item in items]

    for task in asyncio.as_completed(tasks):
        yield await task


async def async_filter(
    predicate: Callable[[T], Awaitable[bool]],
    items: Iterable[T],
    concurrency: int = 10
) -> List[T]:
    """
    Async filter with concurrency control.

    Args:
        predicate: Async predicate function
        items: Items to filter
        concurrency: Maximum concurrent operations

    Returns:
        Filtered list
    """
    semaphore = asyncio.Semaphore(concurrency)
    results: List[T] = []

    async def check(item: T) -> Optional[T]:
        async with semaphore:
            if await predicate(item):
                return item
        return None

    tasks = [asyncio.create_task(check(item)) for item in items]
    for task in asyncio.as_completed(tasks):
        result = await task
        if result is not None:
            results.append(result)

    return results


async def async_reduce(
    func: Callable[[R, T], Awaitable[R]],
    items: Iterable[T],
    initial: R
) -> R:
    """
    Async reduce/fold operation.

    Args:
        func: Async reduce function
        items: Items to reduce
        initial: Initial value

    Returns:
        Reduced value
    """
    result = initial
    for item in items:
        result = await func(result, item)
    return result


async def chunked_async(
    items: List[T],
    chunk_size: int,
    func: Callable[[List[T]], Awaitable[List[R]]]
) -> List[R]:
    """
    Process items in chunks asynchronously.

    Args:
        items: Items to process
        chunk_size: Size of each chunk
        func: Async function to process each chunk

    Returns:
        Combined results
    """
    results: List[R] = []

    for i in range(0, len(items), chunk_size):
        chunk = items[i:i + chunk_size]
        chunk_results = await func(chunk)
        results.extend(chunk_results)

    return results


# =============================================================================
# ASYNC QUEUE UTILITIES
# =============================================================================

class AsyncQueue:
    """Enhanced async queue with additional features."""

    def __init__(self, maxsize: int = 0) -> None:
        """
        Initialize async queue.

        Args:
            maxsize: Maximum queue size (0 for unlimited)
        """
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._processing: Set[asyncio.Task] = set()
        self._closed = False

    async def put(self, item: Any, timeout: Optional[float] = None) -> None:
        """
        Put an item in the queue.

        Args:
            item: Item to add
            timeout: Optional timeout
        """
        if self._closed:
            raise RuntimeError("Queue is closed")

        if timeout:
            await asyncio.wait_for(self._queue.put(item), timeout=timeout)
        else:
            await self._queue.put(item)

    async def get(self, timeout: Optional[float] = None) -> Any:
        """
        Get an item from the queue.

        Args:
            timeout: Optional timeout

        Returns:
            Queue item
        """
        if timeout:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        return await self._queue.get()

    def task_done(self) -> None:
        """Mark a task as done."""
        self._queue.task_done()

    async def join(self) -> None:
        """Wait for all items to be processed."""
        await self._queue.join()

    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()

    def qsize(self) -> int:
        """Get queue size."""
        return self._queue.qsize()

    def close(self) -> None:
        """Close the queue."""
        self._closed = True

    @property
    def closed(self) -> bool:
        """Check if queue is closed."""
        return self._closed


class AsyncWorkerPool:
    """Pool of async workers for processing tasks."""

    def __init__(
        self,
        num_workers: int,
        queue_size: int = 0,
        name: str = "worker"
    ) -> None:
        """
        Initialize worker pool.

        Args:
            num_workers: Number of workers
            queue_size: Queue size (0 for unlimited)
            name: Worker name prefix
        """
        self.num_workers = num_workers
        self.queue = AsyncQueue(maxsize=queue_size)
        self.name = name
        self._workers: List[asyncio.Task] = []
        self._handler: Optional[Callable[[Any], Awaitable[Any]]] = None
        self._running = False

    def set_handler(self, handler: Callable[[Any], Awaitable[Any]]) -> None:
        """
        Set the task handler function.

        Args:
            handler: Async function to handle tasks
        """
        self._handler = handler

    async def _worker(self, worker_id: int) -> None:
        """Worker coroutine."""
        logger.debug(f"{self.name}-{worker_id} started")

        while self._running:
            try:
                item = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                if self._handler:
                    await self._handler(item)
                self.queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"{self.name}-{worker_id} error: {e}")

        logger.debug(f"{self.name}-{worker_id} stopped")

    async def start(self) -> None:
        """Start the worker pool."""
        if self._running:
            return

        if not self._handler:
            raise RuntimeError("No handler set")

        self._running = True
        self._workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self.num_workers)
        ]
        logger.info(f"Started {self.num_workers} {self.name} workers")

    async def stop(self, wait: bool = True) -> None:
        """
        Stop the worker pool.

        Args:
            wait: Whether to wait for pending tasks
        """
        self._running = False

        if wait:
            await self.queue.join()

        for worker in self._workers:
            worker.cancel()

        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info(f"Stopped {self.name} workers")

    async def submit(self, item: Any) -> None:
        """
        Submit a task to the pool.

        Args:
            item: Task to submit
        """
        await self.queue.put(item)


# =============================================================================
# THREAD/PROCESS POOL UTILITIES
# =============================================================================

_thread_pool: Optional[ThreadPoolExecutor] = None
_process_pool: Optional[ProcessPoolExecutor] = None


def get_thread_pool(max_workers: int = 10) -> ThreadPoolExecutor:
    """
    Get or create global thread pool.

    Args:
        max_workers: Maximum workers

    Returns:
        ThreadPoolExecutor instance
    """
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolExecutor(max_workers=max_workers)
    return _thread_pool


def get_process_pool(max_workers: Optional[int] = None) -> ProcessPoolExecutor:
    """
    Get or create global process pool.

    Args:
        max_workers: Maximum workers

    Returns:
        ProcessPoolExecutor instance
    """
    global _process_pool
    if _process_pool is None:
        _process_pool = ProcessPoolExecutor(max_workers=max_workers)
    return _process_pool


async def run_in_thread(
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any
) -> T:
    """
    Run a sync function in a thread pool.

    Args:
        func: Function to run
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Function result
    """
    loop = asyncio.get_event_loop()
    pool = get_thread_pool()
    return await loop.run_in_executor(
        pool,
        functools.partial(func, *args, **kwargs)
    )


async def run_in_process(
    func: Callable[..., T],
    *args: Any
) -> T:
    """
    Run a sync function in a process pool.

    Args:
        func: Function to run
        *args: Positional arguments

    Returns:
        Function result
    """
    loop = asyncio.get_event_loop()
    pool = get_process_pool()
    return await loop.run_in_executor(pool, func, *args)


def sync_to_async(func: Callable[..., T]) -> Callable[..., Awaitable[T]]:
    """
    Convert a sync function to async by running in thread pool.

    Args:
        func: Sync function

    Returns:
        Async function wrapper
    """
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        return await run_in_thread(func, *args, **kwargs)
    return wrapper


# =============================================================================
# ASYNC EVENT UTILITIES
# =============================================================================

class AsyncEvent:
    """Enhanced async event with timeout support."""

    def __init__(self) -> None:
        """Initialize async event."""
        self._event = asyncio.Event()

    def set(self) -> None:
        """Set the event."""
        self._event.set()

    def clear(self) -> None:
        """Clear the event."""
        self._event.clear()

    def is_set(self) -> bool:
        """Check if event is set."""
        return self._event.is_set()

    async def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the event.

        Args:
            timeout: Optional timeout

        Returns:
            True if event was set, False if timeout
        """
        if timeout is None:
            await self._event.wait()
            return True

        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False


class AsyncCondition:
    """Enhanced async condition variable."""

    def __init__(self) -> None:
        """Initialize async condition."""
        self._condition = asyncio.Condition()

    async def acquire(self) -> None:
        """Acquire the condition lock."""
        await self._condition.acquire()

    def release(self) -> None:
        """Release the condition lock."""
        self._condition.release()

    async def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for notification.

        Args:
            timeout: Optional timeout

        Returns:
            True if notified, False if timeout
        """
        if timeout is None:
            await self._condition.wait()
            return True

        try:
            await asyncio.wait_for(self._condition.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def wait_for(
        self,
        predicate: Callable[[], bool],
        timeout: Optional[float] = None
    ) -> bool:
        """
        Wait for predicate to be true.

        Args:
            predicate: Predicate function
            timeout: Optional timeout

        Returns:
            True if predicate satisfied, False if timeout
        """
        if timeout is None:
            await self._condition.wait_for(predicate)
            return True

        try:
            await asyncio.wait_for(
                self._condition.wait_for(predicate),
                timeout=timeout
            )
            return True
        except asyncio.TimeoutError:
            return False

    def notify(self, n: int = 1) -> None:
        """Notify n waiters."""
        self._condition.notify(n)

    def notify_all(self) -> None:
        """Notify all waiters."""
        self._condition.notify_all()

    async def __aenter__(self) -> 'AsyncCondition':
        """Async context manager entry."""
        await self.acquire()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        self.release()


# =============================================================================
# ASYNC CONTEXT MANAGERS
# =============================================================================

@asynccontextmanager
async def timeout_context(
    seconds: float,
    exception_class: type = asyncio.TimeoutError
) -> AsyncGenerator[None, None]:
    """
    Async context manager with timeout.

    Args:
        seconds: Timeout in seconds
        exception_class: Exception to raise on timeout

    Yields:
        None
    """
    async def cancel_on_timeout() -> None:
        await asyncio.sleep(seconds)
        raise exception_class(f"Operation timed out after {seconds}s")

    task = asyncio.create_task(cancel_on_timeout())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@asynccontextmanager
async def cleanup_on_error(
    cleanup_func: Callable[[], Awaitable[None]]
) -> AsyncGenerator[None, None]:
    """
    Async context manager that runs cleanup on error.

    Args:
        cleanup_func: Async cleanup function

    Yields:
        None
    """
    try:
        yield
    except Exception:
        await cleanup_func()
        raise


# =============================================================================
# ASYNC UTILITIES
# =============================================================================

async def sleep_until(target_time: float) -> None:
    """
    Sleep until a target timestamp.

    Args:
        target_time: Target timestamp (Unix time)
    """
    now = time.time()
    if target_time > now:
        await asyncio.sleep(target_time - now)


async def periodic(
    func: Callable[[], Awaitable[None]],
    interval: float,
    stop_event: Optional[asyncio.Event] = None
) -> None:
    """
    Run a function periodically.

    Args:
        func: Async function to run
        interval: Interval in seconds
        stop_event: Optional event to stop
    """
    while True:
        if stop_event and stop_event.is_set():
            break

        start = time.time()
        try:
            await func()
        except Exception as e:
            logger.error(f"Periodic task error: {e}")

        elapsed = time.time() - start
        sleep_time = max(0, interval - elapsed)

        if stop_event:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=sleep_time)
                break
            except asyncio.TimeoutError:
                pass
        else:
            await asyncio.sleep(sleep_time)


def create_task_with_logging(
    coro: Coroutine[Any, Any, T],
    name: Optional[str] = None
) -> asyncio.Task[T]:
    """
    Create a task that logs exceptions.

    Args:
        coro: Coroutine to run
        name: Optional task name

    Returns:
        Created task
    """
    async def wrapper() -> T:
        try:
            return await coro
        except Exception as e:
            logger.error(f"Task {name or 'unknown'} failed: {e}")
            raise

    return asyncio.create_task(wrapper(), name=name)


async def cancel_tasks(tasks: Iterable[asyncio.Task]) -> None:
    """
    Cancel multiple tasks gracefully.

    Args:
        tasks: Tasks to cancel
    """
    for task in tasks:
        if not task.done():
            task.cancel()

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for task, result in zip(tasks, results):
        if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
            logger.warning(f"Task {task.get_name()} raised: {result}")
