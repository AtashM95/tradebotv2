"""
Session Manager Module for Ultimate Trading Bot v2.2.

This module manages trading sessions, including session lifecycle,
state persistence, and session analytics.
"""

import asyncio
import logging
import pickle
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from src.utils.exceptions import (
    InitializationError,
    ValidationError,
)
from src.utils.helpers import (
    generate_uuid,
    ensure_directory,
    safe_json_dumps,
    safe_json_loads,
)
from src.utils.date_utils import now_utc, format_datetime, parse_datetime
from src.utils.decorators import singleton


logger = logging.getLogger(__name__)


class SessionState(str, Enum):
    """Session state enumeration."""

    CREATED = "created"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    RECOVERING = "recovering"


class SessionType(str, Enum):
    """Session type enumeration."""

    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"
    OPTIMIZATION = "optimization"


class SessionManagerConfig(BaseModel):
    """Configuration for session manager."""

    session_data_dir: str = Field(default="data/sessions")
    auto_save_interval_seconds: int = Field(default=300, ge=60, le=3600)
    max_session_history: int = Field(default=100, ge=10, le=1000)
    enable_auto_recovery: bool = Field(default=True)
    recovery_timeout_seconds: int = Field(default=60, ge=10, le=300)
    persist_on_pause: bool = Field(default=True)
    persist_on_stop: bool = Field(default=True)
    cleanup_old_sessions_days: int = Field(default=30, ge=1, le=365)


class SessionStats(BaseModel):
    """Statistics for a trading session."""

    total_signals: int = Field(default=0)
    signals_executed: int = Field(default=0)
    signals_rejected: int = Field(default=0)

    total_orders: int = Field(default=0)
    orders_filled: int = Field(default=0)
    orders_cancelled: int = Field(default=0)
    orders_rejected: int = Field(default=0)

    trades_opened: int = Field(default=0)
    trades_closed: int = Field(default=0)

    total_profit: float = Field(default=0.0)
    total_loss: float = Field(default=0.0)
    net_pnl: float = Field(default=0.0)

    winning_trades: int = Field(default=0)
    losing_trades: int = Field(default=0)

    max_drawdown: float = Field(default=0.0)
    max_drawdown_duration_minutes: int = Field(default=0)

    peak_equity: float = Field(default=0.0)
    lowest_equity: float = Field(default=0.0)

    api_calls: int = Field(default=0)
    api_errors: int = Field(default=0)

    ai_requests: int = Field(default=0)
    ai_tokens_used: int = Field(default=0)
    ai_cost: float = Field(default=0.0)

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        total = self.winning_trades + self.losing_trades
        if total == 0:
            return 0.0
        return (self.winning_trades / total) * 100

    @property
    def profit_factor(self) -> float:
        """Calculate profit factor."""
        if self.total_loss == 0:
            return float("inf") if self.total_profit > 0 else 0.0
        return abs(self.total_profit / self.total_loss)

    @property
    def average_win(self) -> float:
        """Calculate average winning trade."""
        if self.winning_trades == 0:
            return 0.0
        return self.total_profit / self.winning_trades

    @property
    def average_loss(self) -> float:
        """Calculate average losing trade."""
        if self.losing_trades == 0:
            return 0.0
        return self.total_loss / self.losing_trades

    def record_signal(self, executed: bool = True) -> None:
        """Record a signal."""
        self.total_signals += 1
        if executed:
            self.signals_executed += 1
        else:
            self.signals_rejected += 1

    def record_order(
        self,
        filled: bool = False,
        cancelled: bool = False,
        rejected: bool = False
    ) -> None:
        """Record an order."""
        self.total_orders += 1
        if filled:
            self.orders_filled += 1
        elif cancelled:
            self.orders_cancelled += 1
        elif rejected:
            self.orders_rejected += 1

    def record_trade(
        self,
        pnl: float,
        is_close: bool = False
    ) -> None:
        """Record a trade."""
        if is_close:
            self.trades_closed += 1
            self.net_pnl += pnl

            if pnl > 0:
                self.winning_trades += 1
                self.total_profit += pnl
            else:
                self.losing_trades += 1
                self.total_loss += abs(pnl)
        else:
            self.trades_opened += 1

    def record_api_call(self, error: bool = False) -> None:
        """Record an API call."""
        self.api_calls += 1
        if error:
            self.api_errors += 1

    def record_ai_usage(
        self,
        tokens: int,
        cost: float
    ) -> None:
        """Record AI usage."""
        self.ai_requests += 1
        self.ai_tokens_used += tokens
        self.ai_cost += cost

    def update_equity_stats(self, equity: float) -> None:
        """Update equity-related statistics."""
        if equity > self.peak_equity:
            self.peak_equity = equity

        if self.lowest_equity == 0 or equity < self.lowest_equity:
            self.lowest_equity = equity

        if self.peak_equity > 0:
            drawdown = ((self.peak_equity - equity) / self.peak_equity) * 100
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown


class TradingSession(BaseModel):
    """Represents a single trading session."""

    session_id: str = Field(default_factory=generate_uuid)
    session_type: SessionType = Field(default=SessionType.PAPER)
    state: SessionState = Field(default=SessionState.CREATED)

    name: str = Field(default="")
    description: str = Field(default="")

    created_at: datetime = Field(default_factory=now_utc)
    started_at: Optional[datetime] = None
    paused_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None

    starting_equity: float = Field(default=0.0)
    current_equity: float = Field(default=0.0)

    symbols: list[str] = Field(default_factory=list)
    strategies: list[str] = Field(default_factory=list)

    stats: SessionStats = Field(default_factory=SessionStats)

    config_snapshot: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)

    error_message: Optional[str] = None
    recovery_attempts: int = Field(default=0)

    @property
    def duration(self) -> Optional[timedelta]:
        """Calculate session duration."""
        if not self.started_at:
            return None

        end = self.stopped_at or now_utc()
        return end - self.started_at

    @property
    def duration_minutes(self) -> int:
        """Get duration in minutes."""
        duration = self.duration
        if not duration:
            return 0
        return int(duration.total_seconds() / 60)

    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        return self.state in (
            SessionState.RUNNING,
            SessionState.PAUSED,
            SessionState.RECOVERING
        )

    @property
    def return_percent(self) -> float:
        """Calculate session return percentage."""
        if self.starting_equity <= 0:
            return 0.0
        return ((self.current_equity - self.starting_equity) / self.starting_equity) * 100

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "session_type": self.session_type.value,
            "state": self.state.value,
            "name": self.name,
            "description": self.description,
            "created_at": format_datetime(self.created_at),
            "started_at": format_datetime(self.started_at) if self.started_at else None,
            "paused_at": format_datetime(self.paused_at) if self.paused_at else None,
            "stopped_at": format_datetime(self.stopped_at) if self.stopped_at else None,
            "starting_equity": self.starting_equity,
            "current_equity": self.current_equity,
            "symbols": self.symbols,
            "strategies": self.strategies,
            "stats": self.stats.model_dump(),
            "config_snapshot": self.config_snapshot,
            "metadata": self.metadata,
            "error_message": self.error_message,
            "recovery_attempts": self.recovery_attempts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TradingSession":
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            session_type=SessionType(data["session_type"]),
            state=SessionState(data["state"]),
            name=data.get("name", ""),
            description=data.get("description", ""),
            created_at=parse_datetime(data["created_at"]),
            started_at=parse_datetime(data["started_at"]) if data.get("started_at") else None,
            paused_at=parse_datetime(data["paused_at"]) if data.get("paused_at") else None,
            stopped_at=parse_datetime(data["stopped_at"]) if data.get("stopped_at") else None,
            starting_equity=data.get("starting_equity", 0.0),
            current_equity=data.get("current_equity", 0.0),
            symbols=data.get("symbols", []),
            strategies=data.get("strategies", []),
            stats=SessionStats(**data.get("stats", {})),
            config_snapshot=data.get("config_snapshot", {}),
            metadata=data.get("metadata", {}),
            error_message=data.get("error_message"),
            recovery_attempts=data.get("recovery_attempts", 0),
        )


@singleton
class SessionManager:
    """
    Manages trading sessions and their lifecycle.

    This class provides:
    - Session creation and management
    - State persistence and recovery
    - Session analytics and statistics
    - Session history tracking
    """

    def __init__(
        self,
        config: Optional[SessionManagerConfig] = None,
    ) -> None:
        """
        Initialize SessionManager.

        Args:
            config: Session manager configuration
        """
        self._config = config or SessionManagerConfig()
        self._current_session: Optional[TradingSession] = None
        self._session_history: list[TradingSession] = []

        self._state_callbacks: dict[SessionState, list[Callable]] = {
            state: [] for state in SessionState
        }

        self._auto_save_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        ensure_directory(self._config.session_data_dir)

        self._load_session_history()

        logger.info("SessionManager initialized")

    @property
    def current_session(self) -> Optional[TradingSession]:
        """Get current session."""
        return self._current_session

    @property
    def session_history(self) -> list[TradingSession]:
        """Get session history."""
        return self._session_history.copy()

    @property
    def has_active_session(self) -> bool:
        """Check if there's an active session."""
        return self._current_session is not None and self._current_session.is_active

    def register_state_callback(
        self,
        state: SessionState,
        callback: Callable[[TradingSession], None]
    ) -> None:
        """Register a callback for a specific state transition."""
        self._state_callbacks[state].append(callback)
        logger.debug(f"Registered callback for state {state.value}")

    async def create_session(
        self,
        session_type: SessionType = SessionType.PAPER,
        name: str = "",
        description: str = "",
        symbols: Optional[list[str]] = None,
        strategies: Optional[list[str]] = None,
        starting_equity: float = 0.0,
        config_snapshot: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> TradingSession:
        """
        Create a new trading session.

        Args:
            session_type: Type of session
            name: Session name
            description: Session description
            symbols: Trading symbols
            strategies: Strategy names
            starting_equity: Starting account equity
            config_snapshot: Configuration snapshot
            metadata: Additional metadata

        Returns:
            Created session
        """
        async with self._lock:
            if self.has_active_session:
                raise ValidationError("Active session exists. Stop it first.")

            session = TradingSession(
                session_type=session_type,
                name=name or f"Session-{now_utc().strftime('%Y%m%d-%H%M%S')}",
                description=description,
                symbols=symbols or [],
                strategies=strategies or [],
                starting_equity=starting_equity,
                current_equity=starting_equity,
                config_snapshot=config_snapshot or {},
                metadata=metadata or {},
            )

            self._current_session = session
            logger.info(f"Created session: {session.session_id}")

            return session

    async def start_session(self) -> TradingSession:
        """
        Start the current session.

        Returns:
            Started session
        """
        async with self._lock:
            if not self._current_session:
                raise ValidationError("No session to start")

            if self._current_session.state not in (
                SessionState.CREATED,
                SessionState.PAUSED
            ):
                raise ValidationError(
                    f"Cannot start session in state: {self._current_session.state}"
                )

            self._current_session.state = SessionState.INITIALIZING
            await self._notify_state_change(SessionState.INITIALIZING)

            try:
                self._current_session.started_at = now_utc()
                self._current_session.state = SessionState.RUNNING
                self._current_session.paused_at = None

                self._start_auto_save()

                await self._notify_state_change(SessionState.RUNNING)

                logger.info(f"Session started: {self._current_session.session_id}")
                return self._current_session

            except Exception as e:
                self._current_session.state = SessionState.ERROR
                self._current_session.error_message = str(e)
                await self._notify_state_change(SessionState.ERROR)
                raise InitializationError(f"Failed to start session: {e}")

    async def pause_session(self) -> TradingSession:
        """
        Pause the current session.

        Returns:
            Paused session
        """
        async with self._lock:
            if not self._current_session:
                raise ValidationError("No session to pause")

            if self._current_session.state != SessionState.RUNNING:
                raise ValidationError(
                    f"Cannot pause session in state: {self._current_session.state}"
                )

            self._current_session.state = SessionState.PAUSED
            self._current_session.paused_at = now_utc()

            if self._config.persist_on_pause:
                await self._save_session()

            await self._notify_state_change(SessionState.PAUSED)

            logger.info(f"Session paused: {self._current_session.session_id}")
            return self._current_session

    async def resume_session(self) -> TradingSession:
        """
        Resume a paused session.

        Returns:
            Resumed session
        """
        return await self.start_session()

    async def stop_session(
        self,
        error_message: Optional[str] = None
    ) -> TradingSession:
        """
        Stop the current session.

        Args:
            error_message: Optional error message if stopping due to error

        Returns:
            Stopped session
        """
        async with self._lock:
            if not self._current_session:
                raise ValidationError("No session to stop")

            self._current_session.state = SessionState.STOPPING
            await self._notify_state_change(SessionState.STOPPING)

            self._stop_auto_save()

            self._current_session.stopped_at = now_utc()
            self._current_session.state = SessionState.STOPPED

            if error_message:
                self._current_session.error_message = error_message

            if self._config.persist_on_stop:
                await self._save_session()

            self._session_history.append(self._current_session)
            self._trim_session_history()

            await self._notify_state_change(SessionState.STOPPED)

            stopped_session = self._current_session
            self._current_session = None

            logger.info(f"Session stopped: {stopped_session.session_id}")
            return stopped_session

    async def recover_session(
        self,
        session_id: Optional[str] = None
    ) -> Optional[TradingSession]:
        """
        Attempt to recover a session.

        Args:
            session_id: Specific session to recover, or latest if None

        Returns:
            Recovered session if successful
        """
        async with self._lock:
            if self.has_active_session:
                raise ValidationError("Active session exists")

            session_data = await self._load_session_data(session_id)

            if not session_data:
                logger.warning("No session data to recover")
                return None

            try:
                session = TradingSession.from_dict(session_data)
                session.state = SessionState.RECOVERING
                session.recovery_attempts += 1

                self._current_session = session
                await self._notify_state_change(SessionState.RECOVERING)

                logger.info(f"Session recovered: {session.session_id}")
                return session

            except Exception as e:
                logger.error(f"Failed to recover session: {e}")
                return None

    async def _save_session(self) -> None:
        """Save current session to disk."""
        if not self._current_session:
            return

        try:
            session_file = Path(self._config.session_data_dir) / f"{self._current_session.session_id}.json"
            session_data = self._current_session.to_dict()

            with open(session_file, "w") as f:
                f.write(safe_json_dumps(session_data))

            latest_file = Path(self._config.session_data_dir) / "latest_session.json"
            with open(latest_file, "w") as f:
                f.write(safe_json_dumps(session_data))

            logger.debug(f"Session saved: {self._current_session.session_id}")

        except Exception as e:
            logger.error(f"Failed to save session: {e}")

    async def _load_session_data(
        self,
        session_id: Optional[str] = None
    ) -> Optional[dict]:
        """Load session data from disk."""
        try:
            if session_id:
                session_file = Path(self._config.session_data_dir) / f"{session_id}.json"
            else:
                session_file = Path(self._config.session_data_dir) / "latest_session.json"

            if not session_file.exists():
                return None

            with open(session_file, "r") as f:
                return safe_json_loads(f.read())

        except Exception as e:
            logger.error(f"Failed to load session data: {e}")
            return None

    def _load_session_history(self) -> None:
        """Load session history from disk."""
        try:
            history_file = Path(self._config.session_data_dir) / "session_history.json"

            if not history_file.exists():
                return

            with open(history_file, "r") as f:
                history_data = safe_json_loads(f.read())

            self._session_history = [
                TradingSession.from_dict(data)
                for data in history_data
            ]

            logger.debug(f"Loaded {len(self._session_history)} sessions from history")

        except Exception as e:
            logger.error(f"Failed to load session history: {e}")

    def _save_session_history(self) -> None:
        """Save session history to disk."""
        try:
            history_file = Path(self._config.session_data_dir) / "session_history.json"
            history_data = [s.to_dict() for s in self._session_history]

            with open(history_file, "w") as f:
                f.write(safe_json_dumps(history_data))

        except Exception as e:
            logger.error(f"Failed to save session history: {e}")

    def _trim_session_history(self) -> None:
        """Trim session history to max size."""
        if len(self._session_history) > self._config.max_session_history:
            self._session_history = self._session_history[-self._config.max_session_history:]
            self._save_session_history()

    def _start_auto_save(self) -> None:
        """Start auto-save task."""
        if self._auto_save_task:
            return

        self._auto_save_task = asyncio.create_task(
            self._auto_save_loop(),
            name="session_auto_save"
        )

    def _stop_auto_save(self) -> None:
        """Stop auto-save task."""
        if self._auto_save_task:
            self._auto_save_task.cancel()
            self._auto_save_task = None

    async def _auto_save_loop(self) -> None:
        """Auto-save loop."""
        while True:
            try:
                await asyncio.sleep(self._config.auto_save_interval_seconds)
                await self._save_session()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-save: {e}")

    async def _notify_state_change(self, state: SessionState) -> None:
        """Notify callbacks of state change."""
        if not self._current_session:
            return

        for callback in self._state_callbacks[state]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self._current_session)
                else:
                    callback(self._current_session)
            except Exception as e:
                logger.error(f"Error in state callback: {e}")

    def update_equity(self, equity: float) -> None:
        """Update current equity."""
        if self._current_session:
            self._current_session.current_equity = equity
            self._current_session.stats.update_equity_stats(equity)

    def record_signal(self, executed: bool = True) -> None:
        """Record a trading signal."""
        if self._current_session:
            self._current_session.stats.record_signal(executed)

    def record_order(self, **kwargs) -> None:
        """Record an order."""
        if self._current_session:
            self._current_session.stats.record_order(**kwargs)

    def record_trade(self, pnl: float, is_close: bool = False) -> None:
        """Record a trade."""
        if self._current_session:
            self._current_session.stats.record_trade(pnl, is_close)

    def record_api_call(self, error: bool = False) -> None:
        """Record an API call."""
        if self._current_session:
            self._current_session.stats.record_api_call(error)

    def record_ai_usage(self, tokens: int, cost: float) -> None:
        """Record AI usage."""
        if self._current_session:
            self._current_session.stats.record_ai_usage(tokens, cost)

    def get_session_summary(self) -> Optional[dict]:
        """Get current session summary."""
        if not self._current_session:
            return None

        return {
            "session_id": self._current_session.session_id,
            "name": self._current_session.name,
            "type": self._current_session.session_type.value,
            "state": self._current_session.state.value,
            "duration_minutes": self._current_session.duration_minutes,
            "starting_equity": self._current_session.starting_equity,
            "current_equity": self._current_session.current_equity,
            "return_percent": self._current_session.return_percent,
            "total_trades": self._current_session.stats.trades_closed,
            "win_rate": self._current_session.stats.win_rate,
            "net_pnl": self._current_session.stats.net_pnl,
            "max_drawdown": self._current_session.stats.max_drawdown,
        }

    def get_history_summary(self) -> list[dict]:
        """Get session history summary."""
        return [
            {
                "session_id": s.session_id,
                "name": s.name,
                "type": s.session_type.value,
                "state": s.state.value,
                "created_at": format_datetime(s.created_at),
                "duration_minutes": s.duration_minutes,
                "return_percent": s.return_percent,
                "total_trades": s.stats.trades_closed,
                "net_pnl": s.stats.net_pnl,
            }
            for s in self._session_history
        ]

    async def cleanup_old_sessions(self) -> int:
        """
        Clean up old session files.

        Returns:
            Number of files cleaned up
        """
        cutoff = now_utc() - timedelta(days=self._config.cleanup_old_sessions_days)
        cleaned = 0

        session_dir = Path(self._config.session_data_dir)

        for session_file in session_dir.glob("*.json"):
            if session_file.name in ("latest_session.json", "session_history.json"):
                continue

            try:
                mtime = datetime.fromtimestamp(session_file.stat().st_mtime)
                if mtime < cutoff:
                    session_file.unlink()
                    cleaned += 1
            except Exception as e:
                logger.error(f"Error cleaning up {session_file}: {e}")

        logger.info(f"Cleaned up {cleaned} old session files")
        return cleaned

    def __repr__(self) -> str:
        """String representation."""
        if self._current_session:
            return (
                f"SessionManager(session={self._current_session.session_id}, "
                f"state={self._current_session.state.value})"
            )
        return "SessionManager(no_session)"
