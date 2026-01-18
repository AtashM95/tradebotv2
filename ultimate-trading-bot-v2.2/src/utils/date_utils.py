"""
Date and Time Utilities Module for Ultimate Trading Bot v2.2.

This module provides comprehensive date and time utilities for trading,
including market hours, timezone handling, and date calculations.
"""

from datetime import datetime, date, time, timedelta, timezone
from typing import List, Optional, Tuple, Union, Iterator
from enum import Enum
import calendar
from zoneinfo import ZoneInfo
import logging


logger = logging.getLogger(__name__)


# =============================================================================
# TIMEZONE CONSTANTS
# =============================================================================

# Common timezones
ET = ZoneInfo("America/New_York")  # Eastern Time
CT = ZoneInfo("America/Chicago")   # Central Time
PT = ZoneInfo("America/Los_Angeles")  # Pacific Time
UTC = timezone.utc

# Market timezone
MARKET_TZ = ET


class Weekday(Enum):
    """Weekday enumeration."""
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


# =============================================================================
# MARKET HOURS
# =============================================================================

# Regular trading hours (Eastern Time)
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

# Extended hours
PRE_MARKET_OPEN = time(4, 0)
AFTER_HOURS_CLOSE = time(20, 0)

# NYSE/NASDAQ holidays for 2024-2025 (add more as needed)
MARKET_HOLIDAYS: List[date] = [
    # 2024
    date(2024, 1, 1),    # New Year's Day
    date(2024, 1, 15),   # MLK Day
    date(2024, 2, 19),   # Presidents Day
    date(2024, 3, 29),   # Good Friday
    date(2024, 5, 27),   # Memorial Day
    date(2024, 6, 19),   # Juneteenth
    date(2024, 7, 4),    # Independence Day
    date(2024, 9, 2),    # Labor Day
    date(2024, 11, 28),  # Thanksgiving
    date(2024, 12, 25),  # Christmas
    # 2025
    date(2025, 1, 1),    # New Year's Day
    date(2025, 1, 20),   # MLK Day
    date(2025, 2, 17),   # Presidents Day
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 26),   # Memorial Day
    date(2025, 6, 19),   # Juneteenth
    date(2025, 7, 4),    # Independence Day
    date(2025, 9, 1),    # Labor Day
    date(2025, 11, 27),  # Thanksgiving
    date(2025, 12, 25),  # Christmas
]

# Early close days (1 PM ET)
EARLY_CLOSE_DAYS: List[date] = [
    date(2024, 7, 3),    # Day before Independence Day
    date(2024, 11, 29),  # Day after Thanksgiving
    date(2024, 12, 24),  # Christmas Eve
    date(2025, 7, 3),    # Day before Independence Day
    date(2025, 11, 28),  # Day after Thanksgiving
    date(2025, 12, 24),  # Christmas Eve
]


# =============================================================================
# CURRENT TIME FUNCTIONS
# =============================================================================

def now_utc() -> datetime:
    """
    Get current UTC datetime.

    Returns:
        Current datetime in UTC
    """
    return datetime.now(UTC)


def now_et() -> datetime:
    """
    Get current Eastern Time datetime.

    Returns:
        Current datetime in ET
    """
    return datetime.now(ET)


def now_market() -> datetime:
    """
    Get current market time (Eastern).

    Returns:
        Current datetime in market timezone
    """
    return datetime.now(MARKET_TZ)


def today_utc() -> date:
    """
    Get today's date in UTC.

    Returns:
        Today's date in UTC
    """
    return now_utc().date()


def today_market() -> date:
    """
    Get today's date in market timezone.

    Returns:
        Today's date in market timezone
    """
    return now_market().date()


# =============================================================================
# TIMEZONE CONVERSION
# =============================================================================

def to_utc(dt: datetime) -> datetime:
    """
    Convert datetime to UTC.

    Args:
        dt: Datetime to convert

    Returns:
        Datetime in UTC
    """
    if dt.tzinfo is None:
        # Assume ET if naive
        dt = dt.replace(tzinfo=ET)
    return dt.astimezone(UTC)


def to_et(dt: datetime) -> datetime:
    """
    Convert datetime to Eastern Time.

    Args:
        dt: Datetime to convert

    Returns:
        Datetime in ET
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(ET)


def to_market_tz(dt: datetime) -> datetime:
    """
    Convert datetime to market timezone.

    Args:
        dt: Datetime to convert

    Returns:
        Datetime in market timezone
    """
    return to_et(dt)


def localize(dt: datetime, tz: ZoneInfo = ET) -> datetime:
    """
    Add timezone info to a naive datetime.

    Args:
        dt: Datetime to localize
        tz: Timezone to use

    Returns:
        Localized datetime
    """
    if dt.tzinfo is not None:
        return dt.astimezone(tz)
    return dt.replace(tzinfo=tz)


def make_aware(dt: datetime, tz: ZoneInfo = UTC) -> datetime:
    """
    Make a naive datetime timezone-aware.

    Args:
        dt: Datetime to make aware
        tz: Timezone to use

    Returns:
        Timezone-aware datetime
    """
    if dt.tzinfo is not None:
        return dt
    return dt.replace(tzinfo=tz)


def make_naive(dt: datetime) -> datetime:
    """
    Remove timezone info from datetime.

    Args:
        dt: Datetime to make naive

    Returns:
        Naive datetime
    """
    return dt.replace(tzinfo=None)


# =============================================================================
# MARKET HOURS FUNCTIONS
# =============================================================================

def is_market_holiday(d: date) -> bool:
    """
    Check if a date is a market holiday.

    Args:
        d: Date to check

    Returns:
        True if market holiday
    """
    return d in MARKET_HOLIDAYS


def is_early_close_day(d: date) -> bool:
    """
    Check if a date is an early close day.

    Args:
        d: Date to check

    Returns:
        True if early close day
    """
    return d in EARLY_CLOSE_DAYS


def is_weekend(d: date) -> bool:
    """
    Check if a date is a weekend.

    Args:
        d: Date to check

    Returns:
        True if weekend
    """
    return d.weekday() >= 5


def is_trading_day(d: date) -> bool:
    """
    Check if a date is a trading day.

    Args:
        d: Date to check

    Returns:
        True if trading day
    """
    return not is_weekend(d) and not is_market_holiday(d)


def get_market_close_time(d: date) -> time:
    """
    Get market close time for a date.

    Args:
        d: Date to check

    Returns:
        Close time (1 PM for early close, 4 PM normally)
    """
    if is_early_close_day(d):
        return time(13, 0)
    return MARKET_CLOSE


def is_market_open(dt: Optional[datetime] = None) -> bool:
    """
    Check if the market is currently open.

    Args:
        dt: Datetime to check (defaults to now)

    Returns:
        True if market is open
    """
    if dt is None:
        dt = now_market()
    else:
        dt = to_market_tz(dt)

    # Check if trading day
    if not is_trading_day(dt.date()):
        return False

    # Get close time for the day
    close_time = get_market_close_time(dt.date())

    # Check if within trading hours
    current_time = dt.time()
    return MARKET_OPEN <= current_time < close_time


def is_pre_market(dt: Optional[datetime] = None) -> bool:
    """
    Check if currently in pre-market hours.

    Args:
        dt: Datetime to check (defaults to now)

    Returns:
        True if in pre-market
    """
    if dt is None:
        dt = now_market()
    else:
        dt = to_market_tz(dt)

    if not is_trading_day(dt.date()):
        return False

    current_time = dt.time()
    return PRE_MARKET_OPEN <= current_time < MARKET_OPEN


def is_after_hours(dt: Optional[datetime] = None) -> bool:
    """
    Check if currently in after-hours trading.

    Args:
        dt: Datetime to check (defaults to now)

    Returns:
        True if in after-hours
    """
    if dt is None:
        dt = now_market()
    else:
        dt = to_market_tz(dt)

    if not is_trading_day(dt.date()):
        return False

    close_time = get_market_close_time(dt.date())
    current_time = dt.time()
    return close_time <= current_time < AFTER_HOURS_CLOSE


def is_extended_hours(dt: Optional[datetime] = None) -> bool:
    """
    Check if in extended hours (pre-market or after-hours).

    Args:
        dt: Datetime to check (defaults to now)

    Returns:
        True if in extended hours
    """
    return is_pre_market(dt) or is_after_hours(dt)


def get_market_session(dt: Optional[datetime] = None) -> str:
    """
    Get the current market session.

    Args:
        dt: Datetime to check (defaults to now)

    Returns:
        Session name: 'pre_market', 'regular', 'after_hours', 'closed'
    """
    if dt is None:
        dt = now_market()
    else:
        dt = to_market_tz(dt)

    if not is_trading_day(dt.date()):
        return "closed"

    if is_pre_market(dt):
        return "pre_market"
    elif is_market_open(dt):
        return "regular"
    elif is_after_hours(dt):
        return "after_hours"
    else:
        return "closed"


def time_until_market_open(dt: Optional[datetime] = None) -> timedelta:
    """
    Get time until market opens.

    Args:
        dt: Reference datetime (defaults to now)

    Returns:
        Time until market open
    """
    if dt is None:
        dt = now_market()
    else:
        dt = to_market_tz(dt)

    # Get next trading day
    next_day = get_next_trading_day(dt.date())

    # Create market open datetime
    market_open_dt = datetime.combine(next_day, MARKET_OPEN, tzinfo=MARKET_TZ)

    # If market is already open today, get tomorrow's open
    if is_market_open(dt):
        next_day = get_next_trading_day(dt.date() + timedelta(days=1))
        market_open_dt = datetime.combine(next_day, MARKET_OPEN, tzinfo=MARKET_TZ)

    return market_open_dt - dt


def time_until_market_close(dt: Optional[datetime] = None) -> Optional[timedelta]:
    """
    Get time until market closes.

    Args:
        dt: Reference datetime (defaults to now)

    Returns:
        Time until market close, or None if market is closed
    """
    if dt is None:
        dt = now_market()
    else:
        dt = to_market_tz(dt)

    if not is_market_open(dt):
        return None

    close_time = get_market_close_time(dt.date())
    market_close_dt = datetime.combine(dt.date(), close_time, tzinfo=MARKET_TZ)

    return market_close_dt - dt


# =============================================================================
# TRADING DAY FUNCTIONS
# =============================================================================

def get_next_trading_day(d: Optional[date] = None) -> date:
    """
    Get the next trading day.

    Args:
        d: Reference date (defaults to today)

    Returns:
        Next trading day
    """
    if d is None:
        d = today_market()

    next_day = d
    while True:
        next_day += timedelta(days=1)
        if is_trading_day(next_day):
            return next_day


def get_previous_trading_day(d: Optional[date] = None) -> date:
    """
    Get the previous trading day.

    Args:
        d: Reference date (defaults to today)

    Returns:
        Previous trading day
    """
    if d is None:
        d = today_market()

    prev_day = d
    while True:
        prev_day -= timedelta(days=1)
        if is_trading_day(prev_day):
            return prev_day


def get_trading_days(
    start: date,
    end: date,
    include_start: bool = True,
    include_end: bool = True
) -> List[date]:
    """
    Get list of trading days in a range.

    Args:
        start: Start date
        end: End date
        include_start: Include start date
        include_end: Include end date

    Returns:
        List of trading days
    """
    trading_days = []
    current = start

    while current <= end:
        if is_trading_day(current):
            if (current > start or include_start) and (current < end or include_end):
                trading_days.append(current)
        current += timedelta(days=1)

    return trading_days


def get_trading_day_count(start: date, end: date) -> int:
    """
    Get number of trading days in a range.

    Args:
        start: Start date
        end: End date

    Returns:
        Number of trading days
    """
    return len(get_trading_days(start, end))


def get_n_trading_days_ago(n: int, d: Optional[date] = None) -> date:
    """
    Get the date N trading days ago.

    Args:
        n: Number of trading days
        d: Reference date (defaults to today)

    Returns:
        Date N trading days ago
    """
    if d is None:
        d = today_market()

    result = d
    count = 0
    while count < n:
        result -= timedelta(days=1)
        if is_trading_day(result):
            count += 1

    return result


def get_n_trading_days_from(n: int, d: Optional[date] = None) -> date:
    """
    Get the date N trading days from reference.

    Args:
        n: Number of trading days
        d: Reference date (defaults to today)

    Returns:
        Date N trading days from reference
    """
    if d is None:
        d = today_market()

    result = d
    count = 0
    while count < n:
        result += timedelta(days=1)
        if is_trading_day(result):
            count += 1

    return result


# =============================================================================
# DATE RANGE FUNCTIONS
# =============================================================================

def get_week_start(d: Optional[date] = None) -> date:
    """
    Get the start of the week (Monday).

    Args:
        d: Reference date (defaults to today)

    Returns:
        Monday of the week
    """
    if d is None:
        d = today_market()
    return d - timedelta(days=d.weekday())


def get_week_end(d: Optional[date] = None) -> date:
    """
    Get the end of the week (Friday for trading).

    Args:
        d: Reference date (defaults to today)

    Returns:
        Friday of the week
    """
    if d is None:
        d = today_market()
    monday = get_week_start(d)
    return monday + timedelta(days=4)


def get_month_start(d: Optional[date] = None) -> date:
    """
    Get the start of the month.

    Args:
        d: Reference date (defaults to today)

    Returns:
        First day of the month
    """
    if d is None:
        d = today_market()
    return d.replace(day=1)


def get_month_end(d: Optional[date] = None) -> date:
    """
    Get the end of the month.

    Args:
        d: Reference date (defaults to today)

    Returns:
        Last day of the month
    """
    if d is None:
        d = today_market()
    last_day = calendar.monthrange(d.year, d.month)[1]
    return d.replace(day=last_day)


def get_quarter_start(d: Optional[date] = None) -> date:
    """
    Get the start of the quarter.

    Args:
        d: Reference date (defaults to today)

    Returns:
        First day of the quarter
    """
    if d is None:
        d = today_market()
    quarter = (d.month - 1) // 3
    return date(d.year, quarter * 3 + 1, 1)


def get_quarter_end(d: Optional[date] = None) -> date:
    """
    Get the end of the quarter.

    Args:
        d: Reference date (defaults to today)

    Returns:
        Last day of the quarter
    """
    if d is None:
        d = today_market()
    quarter = (d.month - 1) // 3
    end_month = (quarter + 1) * 3
    last_day = calendar.monthrange(d.year, end_month)[1]
    return date(d.year, end_month, last_day)


def get_year_start(d: Optional[date] = None) -> date:
    """
    Get the start of the year.

    Args:
        d: Reference date (defaults to today)

    Returns:
        First day of the year
    """
    if d is None:
        d = today_market()
    return date(d.year, 1, 1)


def get_year_end(d: Optional[date] = None) -> date:
    """
    Get the end of the year.

    Args:
        d: Reference date (defaults to today)

    Returns:
        Last day of the year
    """
    if d is None:
        d = today_market()
    return date(d.year, 12, 31)


# =============================================================================
# PARSING AND FORMATTING
# =============================================================================

def parse_date(
    date_str: str,
    formats: Optional[List[str]] = None
) -> Optional[date]:
    """
    Parse a date string.

    Args:
        date_str: Date string to parse
        formats: List of formats to try

    Returns:
        Parsed date or None
    """
    if formats is None:
        formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%Y%m%d",
            "%b %d, %Y",
            "%B %d, %Y",
        ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue

    return None


def parse_datetime(
    dt_str: str,
    formats: Optional[List[str]] = None
) -> Optional[datetime]:
    """
    Parse a datetime string.

    Args:
        dt_str: Datetime string to parse
        formats: List of formats to try

    Returns:
        Parsed datetime or None
    """
    if formats is None:
        formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
        ]

    for fmt in formats:
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue

    return None


def format_date(d: date, fmt: str = "%Y-%m-%d") -> str:
    """
    Format a date.

    Args:
        d: Date to format
        fmt: Format string

    Returns:
        Formatted date string
    """
    return d.strftime(fmt)


def format_datetime(dt: datetime, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format a datetime.

    Args:
        dt: Datetime to format
        fmt: Format string

    Returns:
        Formatted datetime string
    """
    return dt.strftime(fmt)


def format_timestamp(dt: datetime) -> str:
    """
    Format datetime as ISO 8601 timestamp.

    Args:
        dt: Datetime to format

    Returns:
        ISO 8601 timestamp string
    """
    return dt.isoformat()


def to_timestamp(dt: datetime) -> float:
    """
    Convert datetime to Unix timestamp.

    Args:
        dt: Datetime to convert

    Returns:
        Unix timestamp (seconds since epoch)
    """
    return dt.timestamp()


def from_timestamp(ts: float) -> datetime:
    """
    Convert Unix timestamp to datetime.

    Args:
        ts: Unix timestamp

    Returns:
        Datetime in UTC
    """
    return datetime.fromtimestamp(ts, tz=UTC)


# =============================================================================
# RELATIVE TIME
# =============================================================================

def time_ago(dt: datetime) -> str:
    """
    Get human-readable relative time.

    Args:
        dt: Datetime to compare

    Returns:
        Human-readable string (e.g., "5 minutes ago")
    """
    now = now_utc()
    if dt.tzinfo is None:
        dt = make_aware(dt, UTC)

    diff = now - dt

    if diff.total_seconds() < 0:
        return "in the future"

    seconds = int(diff.total_seconds())

    if seconds < 60:
        return f"{seconds} seconds ago"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif seconds < 86400:
        hours = seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif seconds < 604800:
        days = seconds // 86400
        return f"{days} day{'s' if days != 1 else ''} ago"
    elif seconds < 2592000:
        weeks = seconds // 604800
        return f"{weeks} week{'s' if weeks != 1 else ''} ago"
    elif seconds < 31536000:
        months = seconds // 2592000
        return f"{months} month{'s' if months != 1 else ''} ago"
    else:
        years = seconds // 31536000
        return f"{years} year{'s' if years != 1 else ''} ago"


def time_until(dt: datetime) -> str:
    """
    Get human-readable time until future datetime.

    Args:
        dt: Future datetime

    Returns:
        Human-readable string (e.g., "in 5 minutes")
    """
    now = now_utc()
    if dt.tzinfo is None:
        dt = make_aware(dt, UTC)

    diff = dt - now

    if diff.total_seconds() < 0:
        return "in the past"

    seconds = int(diff.total_seconds())

    if seconds < 60:
        return f"in {seconds} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"in {minutes} minute{'s' if minutes != 1 else ''}"
    elif seconds < 86400:
        hours = seconds // 3600
        return f"in {hours} hour{'s' if hours != 1 else ''}"
    elif seconds < 604800:
        days = seconds // 86400
        return f"in {days} day{'s' if days != 1 else ''}"
    else:
        weeks = seconds // 604800
        return f"in {weeks} week{'s' if weeks != 1 else ''}"
