"""
Market Hours Module for Ultimate Trading Bot v2.2.

This module provides comprehensive market hours tracking, including
regular trading hours, pre-market, after-hours, and holiday schedules.
"""

import asyncio
import logging
from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from src.utils.date_utils import (
    ET,
    UTC,
    MARKET_TZ,
    MARKET_OPEN,
    MARKET_CLOSE,
    PRE_MARKET_OPEN,
    AFTER_HOURS_CLOSE,
    now_utc,
    now_et,
    to_et,
    is_weekend,
    is_market_holiday,
    is_early_close_day,
    get_market_close_time,
    get_next_trading_day,
    get_previous_trading_day,
)
from src.utils.decorators import singleton


logger = logging.getLogger(__name__)


class MarketSession(str, Enum):
    """Market session enumeration."""

    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    AFTER_HOURS = "after_hours"
    CLOSED = "closed"


class MarketStatus(str, Enum):
    """Market status enumeration."""

    OPEN = "open"
    CLOSED = "closed"
    HOLIDAY = "holiday"
    EARLY_CLOSE = "early_close"
    WEEKEND = "weekend"


class MarketHoursConfig(BaseModel):
    """Configuration for market hours tracking."""

    timezone: str = Field(default="America/New_York")

    regular_open: time = Field(default=time(9, 30))
    regular_close: time = Field(default=time(16, 0))
    pre_market_open: time = Field(default=time(4, 0))
    after_hours_close: time = Field(default=time(20, 0))
    early_close_time: time = Field(default=time(13, 0))

    enable_pre_market: bool = Field(default=True)
    enable_after_hours: bool = Field(default=True)

    buffer_minutes_before_open: int = Field(default=5, ge=0, le=30)
    buffer_minutes_before_close: int = Field(default=5, ge=0, le=30)

    status_check_interval_seconds: int = Field(default=60, ge=10, le=300)


class MarketSchedule(BaseModel):
    """Schedule for a specific trading day."""

    date: date
    is_trading_day: bool = Field(default=True)
    is_holiday: bool = Field(default=False)
    is_early_close: bool = Field(default=False)
    holiday_name: Optional[str] = None

    pre_market_open: Optional[datetime] = None
    regular_open: Optional[datetime] = None
    regular_close: Optional[datetime] = None
    after_hours_close: Optional[datetime] = None

    @property
    def trading_minutes(self) -> int:
        """Calculate total regular trading minutes."""
        if not self.is_trading_day or not self.regular_open or not self.regular_close:
            return 0
        delta = self.regular_close - self.regular_open
        return int(delta.total_seconds() / 60)


US_MARKET_HOLIDAYS_2024 = {
    date(2024, 1, 1): "New Year's Day",
    date(2024, 1, 15): "Martin Luther King Jr. Day",
    date(2024, 2, 19): "Presidents' Day",
    date(2024, 3, 29): "Good Friday",
    date(2024, 5, 27): "Memorial Day",
    date(2024, 6, 19): "Juneteenth",
    date(2024, 7, 4): "Independence Day",
    date(2024, 9, 2): "Labor Day",
    date(2024, 11, 28): "Thanksgiving Day",
    date(2024, 12, 25): "Christmas Day",
}

US_MARKET_HOLIDAYS_2025 = {
    date(2025, 1, 1): "New Year's Day",
    date(2025, 1, 20): "Martin Luther King Jr. Day",
    date(2025, 2, 17): "Presidents' Day",
    date(2025, 4, 18): "Good Friday",
    date(2025, 5, 26): "Memorial Day",
    date(2025, 6, 19): "Juneteenth",
    date(2025, 7, 4): "Independence Day",
    date(2025, 9, 1): "Labor Day",
    date(2025, 11, 27): "Thanksgiving Day",
    date(2025, 12, 25): "Christmas Day",
}

US_MARKET_HOLIDAYS_2026 = {
    date(2026, 1, 1): "New Year's Day",
    date(2026, 1, 19): "Martin Luther King Jr. Day",
    date(2026, 2, 16): "Presidents' Day",
    date(2026, 4, 3): "Good Friday",
    date(2026, 5, 25): "Memorial Day",
    date(2026, 6, 19): "Juneteenth",
    date(2026, 7, 3): "Independence Day (Observed)",
    date(2026, 9, 7): "Labor Day",
    date(2026, 11, 26): "Thanksgiving Day",
    date(2026, 12, 25): "Christmas Day",
}

US_MARKET_HOLIDAYS = {
    **US_MARKET_HOLIDAYS_2024,
    **US_MARKET_HOLIDAYS_2025,
    **US_MARKET_HOLIDAYS_2026,
}

US_EARLY_CLOSE_DATES = {
    date(2024, 7, 3),
    date(2024, 11, 29),
    date(2024, 12, 24),
    date(2025, 7, 3),
    date(2025, 11, 28),
    date(2025, 12, 24),
    date(2026, 11, 27),
    date(2026, 12, 24),
}


@singleton
class MarketHours:
    """
    Tracks market hours and provides session information.

    This class provides:
    - Current market session detection
    - Market schedule generation
    - Time until open/close calculations
    - Market status callbacks
    """

    def __init__(
        self,
        config: Optional[MarketHoursConfig] = None,
    ) -> None:
        """
        Initialize MarketHours.

        Args:
            config: Market hours configuration
        """
        self._config = config or MarketHoursConfig()
        self._tz = ZoneInfo(self._config.timezone)

        self._session_callbacks: dict[MarketSession, list[Callable]] = {
            session: [] for session in MarketSession
        }
        self._status_callbacks: list[Callable[[MarketStatus], None]] = []

        self._current_session: MarketSession = MarketSession.CLOSED
        self._current_status: MarketStatus = MarketStatus.CLOSED

        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False

        self._schedule_cache: dict[date, MarketSchedule] = {}

        logger.info("MarketHours initialized")

    @property
    def current_session(self) -> MarketSession:
        """Get current market session."""
        return self._determine_session()

    @property
    def current_status(self) -> MarketStatus:
        """Get current market status."""
        return self._determine_status()

    @property
    def is_market_open(self) -> bool:
        """Check if regular market is open."""
        return self.current_session == MarketSession.REGULAR

    @property
    def is_extended_hours(self) -> bool:
        """Check if extended hours trading is available."""
        session = self.current_session
        return session in (
            MarketSession.PRE_MARKET,
            MarketSession.AFTER_HOURS
        )

    @property
    def can_trade(self) -> bool:
        """Check if any trading is available."""
        return self.current_session != MarketSession.CLOSED

    def register_session_callback(
        self,
        session: MarketSession,
        callback: Callable[[], None]
    ) -> None:
        """Register a callback for session transitions."""
        self._session_callbacks[session].append(callback)
        logger.debug(f"Registered callback for session {session.value}")

    def register_status_callback(
        self,
        callback: Callable[[MarketStatus], None]
    ) -> None:
        """Register a callback for status changes."""
        self._status_callbacks.append(callback)

    async def start_monitoring(self) -> None:
        """Start market hours monitoring."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(),
            name="market_hours_monitor"
        )

        logger.info("Market hours monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop market hours monitoring."""
        self._running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        logger.info("Market hours monitoring stopped")

    async def _monitor_loop(self) -> None:
        """Monitor market hours and trigger callbacks."""
        while self._running:
            try:
                new_session = self._determine_session()
                new_status = self._determine_status()

                if new_session != self._current_session:
                    old_session = self._current_session
                    self._current_session = new_session
                    await self._notify_session_change(new_session)
                    logger.info(
                        f"Market session changed: {old_session.value} -> {new_session.value}"
                    )

                if new_status != self._current_status:
                    old_status = self._current_status
                    self._current_status = new_status
                    await self._notify_status_change(new_status)
                    logger.info(
                        f"Market status changed: {old_status.value} -> {new_status.value}"
                    )

                await asyncio.sleep(self._config.status_check_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in market hours monitor: {e}")
                await asyncio.sleep(5)

    async def _notify_session_change(self, session: MarketSession) -> None:
        """Notify callbacks of session change."""
        for callback in self._session_callbacks[session]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Error in session callback: {e}")

    async def _notify_status_change(self, status: MarketStatus) -> None:
        """Notify callbacks of status change."""
        for callback in self._status_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(status)
                else:
                    callback(status)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")

    def _determine_session(self) -> MarketSession:
        """Determine current market session."""
        now = datetime.now(self._tz)
        today = now.date()

        if is_weekend(today) or self._is_holiday(today):
            return MarketSession.CLOSED

        current_time = now.time()

        if self._is_early_close(today):
            close_time = self._config.early_close_time
        else:
            close_time = self._config.regular_close

        if self._config.regular_open <= current_time < close_time:
            return MarketSession.REGULAR

        if (
            self._config.enable_pre_market
            and self._config.pre_market_open <= current_time < self._config.regular_open
        ):
            return MarketSession.PRE_MARKET

        if (
            self._config.enable_after_hours
            and close_time <= current_time < self._config.after_hours_close
        ):
            return MarketSession.AFTER_HOURS

        return MarketSession.CLOSED

    def _determine_status(self) -> MarketStatus:
        """Determine current market status."""
        now = datetime.now(self._tz)
        today = now.date()

        if is_weekend(today):
            return MarketStatus.WEEKEND

        if self._is_holiday(today):
            return MarketStatus.HOLIDAY

        if self._is_early_close(today):
            if self.is_market_open:
                return MarketStatus.EARLY_CLOSE
            return MarketStatus.CLOSED

        if self.is_market_open:
            return MarketStatus.OPEN

        return MarketStatus.CLOSED

    def _is_holiday(self, check_date: date) -> bool:
        """Check if date is a market holiday."""
        return check_date in US_MARKET_HOLIDAYS

    def _is_early_close(self, check_date: date) -> bool:
        """Check if date is an early close day."""
        return check_date in US_EARLY_CLOSE_DATES

    def get_holiday_name(self, check_date: date) -> Optional[str]:
        """Get holiday name if date is a holiday."""
        return US_MARKET_HOLIDAYS.get(check_date)

    def get_schedule(self, for_date: Optional[date] = None) -> MarketSchedule:
        """
        Get market schedule for a specific date.

        Args:
            for_date: Date to get schedule for (default: today)

        Returns:
            Market schedule for the date
        """
        if for_date is None:
            for_date = datetime.now(self._tz).date()

        if for_date in self._schedule_cache:
            return self._schedule_cache[for_date]

        schedule = self._build_schedule(for_date)
        self._schedule_cache[for_date] = schedule

        return schedule

    def _build_schedule(self, for_date: date) -> MarketSchedule:
        """Build market schedule for a date."""
        is_holiday = self._is_holiday(for_date)
        is_weekend_day = is_weekend(for_date)
        is_trading = not is_holiday and not is_weekend_day
        is_early = self._is_early_close(for_date)

        if not is_trading:
            return MarketSchedule(
                date=for_date,
                is_trading_day=False,
                is_holiday=is_holiday,
                holiday_name=self.get_holiday_name(for_date),
            )

        close_time = (
            self._config.early_close_time
            if is_early
            else self._config.regular_close
        )

        after_hours_close = (
            datetime.combine(for_date, close_time, tzinfo=self._tz)
            + timedelta(hours=4)
        )
        if after_hours_close.time() > self._config.after_hours_close:
            after_hours_close = datetime.combine(
                for_date,
                self._config.after_hours_close,
                tzinfo=self._tz
            )

        return MarketSchedule(
            date=for_date,
            is_trading_day=True,
            is_holiday=False,
            is_early_close=is_early,
            pre_market_open=datetime.combine(
                for_date,
                self._config.pre_market_open,
                tzinfo=self._tz
            ),
            regular_open=datetime.combine(
                for_date,
                self._config.regular_open,
                tzinfo=self._tz
            ),
            regular_close=datetime.combine(
                for_date,
                close_time,
                tzinfo=self._tz
            ),
            after_hours_close=after_hours_close,
        )

    def time_until_open(self) -> Optional[timedelta]:
        """
        Get time until market opens.

        Returns:
            Time until open, or None if market is open
        """
        if self.is_market_open:
            return None

        now = datetime.now(self._tz)
        today = now.date()
        current_time = now.time()

        if not is_weekend(today) and not self._is_holiday(today):
            if current_time < self._config.regular_open:
                open_dt = datetime.combine(
                    today,
                    self._config.regular_open,
                    tzinfo=self._tz
                )
                return open_dt - now

        next_day = get_next_trading_day(today)
        next_open = datetime.combine(
            next_day,
            self._config.regular_open,
            tzinfo=self._tz
        )

        return next_open - now

    def time_until_close(self) -> Optional[timedelta]:
        """
        Get time until market closes.

        Returns:
            Time until close, or None if market is closed
        """
        if not self.is_market_open:
            return None

        now = datetime.now(self._tz)
        today = now.date()

        close_time = (
            self._config.early_close_time
            if self._is_early_close(today)
            else self._config.regular_close
        )

        close_dt = datetime.combine(today, close_time, tzinfo=self._tz)
        return close_dt - now

    def time_until_session_end(self) -> Optional[timedelta]:
        """
        Get time until current session ends.

        Returns:
            Time until session end, or None if market is closed
        """
        session = self.current_session

        if session == MarketSession.CLOSED:
            return None

        now = datetime.now(self._tz)
        today = now.date()

        if session == MarketSession.PRE_MARKET:
            end_dt = datetime.combine(
                today,
                self._config.regular_open,
                tzinfo=self._tz
            )
        elif session == MarketSession.REGULAR:
            close_time = (
                self._config.early_close_time
                if self._is_early_close(today)
                else self._config.regular_close
            )
            end_dt = datetime.combine(today, close_time, tzinfo=self._tz)
        else:
            end_dt = datetime.combine(
                today,
                self._config.after_hours_close,
                tzinfo=self._tz
            )

        return end_dt - now

    def minutes_until_close(self) -> int:
        """Get minutes until market close."""
        delta = self.time_until_close()
        if delta is None:
            return 0
        return max(0, int(delta.total_seconds() / 60))

    def minutes_since_open(self) -> int:
        """Get minutes since market opened."""
        if not self.is_market_open:
            return 0

        now = datetime.now(self._tz)
        today = now.date()

        open_dt = datetime.combine(
            today,
            self._config.regular_open,
            tzinfo=self._tz
        )

        delta = now - open_dt
        return max(0, int(delta.total_seconds() / 60))

    def is_near_close(self, minutes: int = 15) -> bool:
        """Check if market is within N minutes of close."""
        return 0 < self.minutes_until_close() <= minutes

    def is_near_open(self, minutes: int = 15) -> bool:
        """Check if market just opened within N minutes."""
        return 0 < self.minutes_since_open() <= minutes

    def should_avoid_trading(self) -> tuple[bool, str]:
        """
        Check if trading should be avoided.

        Returns:
            Tuple of (should_avoid, reason)
        """
        if not self.can_trade:
            return True, "Market is closed"

        if self.is_near_close(self._config.buffer_minutes_before_close):
            return True, f"Within {self._config.buffer_minutes_before_close} minutes of close"

        session = self.current_session
        if session == MarketSession.PRE_MARKET:
            if self.is_near_open(self._config.buffer_minutes_before_open):
                return True, "Pre-market about to end"

        return False, ""

    def get_trading_days_range(
        self,
        start: date,
        end: date
    ) -> list[date]:
        """Get list of trading days in a range."""
        days = []
        current = start

        while current <= end:
            if not is_weekend(current) and not self._is_holiday(current):
                days.append(current)
            current += timedelta(days=1)

        return days

    def get_next_n_trading_days(self, n: int) -> list[date]:
        """Get next N trading days."""
        today = datetime.now(self._tz).date()
        days = []
        current = today

        while len(days) < n:
            current += timedelta(days=1)
            if not is_weekend(current) and not self._is_holiday(current):
                days.append(current)

        return days

    def get_status_summary(self) -> dict:
        """Get comprehensive market status summary."""
        now = datetime.now(self._tz)
        schedule = self.get_schedule()

        return {
            "current_time": now.isoformat(),
            "timezone": self._config.timezone,
            "session": self.current_session.value,
            "status": self.current_status.value,
            "is_market_open": self.is_market_open,
            "is_extended_hours": self.is_extended_hours,
            "can_trade": self.can_trade,
            "is_trading_day": schedule.is_trading_day,
            "is_holiday": schedule.is_holiday,
            "is_early_close": schedule.is_early_close,
            "holiday_name": schedule.holiday_name,
            "minutes_until_close": self.minutes_until_close(),
            "minutes_since_open": self.minutes_since_open(),
            "time_until_open": str(self.time_until_open()) if self.time_until_open() else None,
            "time_until_close": str(self.time_until_close()) if self.time_until_close() else None,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MarketHours(session={self.current_session.value}, "
            f"status={self.current_status.value})"
        )
