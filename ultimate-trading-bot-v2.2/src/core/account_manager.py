"""
Account Manager Module for Ultimate Trading Bot v2.2.

This module provides account management functionality including balance tracking,
buying power calculations, margin management, and account status monitoring.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field, field_validator

from src.utils.exceptions import (
    APIError,
    InsufficientFundsError,
    ValidationError,
)
from src.utils.helpers import generate_uuid, round_decimal
from src.utils.date_utils import now_utc
from src.utils.math_utils import safe_divide, percentage_change
from src.utils.decorators import async_retry, singleton


logger = logging.getLogger(__name__)


class AccountStatus(str, Enum):
    """Account status enumeration."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    RESTRICTED = "restricted"
    CLOSED = "closed"
    PENDING = "pending"


class AccountType(str, Enum):
    """Account type enumeration."""

    CASH = "cash"
    MARGIN = "margin"
    PAPER = "paper"


class MarginStatus(str, Enum):
    """Margin status enumeration."""

    NORMAL = "normal"
    MARGIN_CALL = "margin_call"
    MAINTENANCE_CALL = "maintenance_call"
    LIQUIDATION = "liquidation"


class AccountManagerConfig(BaseModel):
    """Configuration for account manager."""

    update_interval_seconds: int = Field(default=60, ge=5, le=300)
    cache_ttl_seconds: int = Field(default=30, ge=5, le=120)
    min_buying_power_buffer: float = Field(default=0.05, ge=0.0, le=0.5)
    enable_margin_monitoring: bool = Field(default=True)
    margin_warning_threshold: float = Field(default=0.3, ge=0.1, le=0.5)
    margin_critical_threshold: float = Field(default=0.15, ge=0.05, le=0.3)
    enable_balance_alerts: bool = Field(default=True)
    low_balance_threshold: float = Field(default=1000.0, ge=0.0)
    max_retry_attempts: int = Field(default=3, ge=1, le=10)

    @field_validator("margin_critical_threshold")
    @classmethod
    def validate_critical_threshold(cls, v: float, info) -> float:
        """Ensure critical threshold is less than warning threshold."""
        warning = info.data.get("margin_warning_threshold", 0.3)
        if v >= warning:
            raise ValueError("Critical threshold must be less than warning threshold")
        return v


class AccountSnapshot(BaseModel):
    """Point-in-time snapshot of account state."""

    snapshot_id: str = Field(default_factory=generate_uuid)
    timestamp: datetime = Field(default_factory=now_utc)

    equity: Decimal = Field(default=Decimal("0"))
    cash: Decimal = Field(default=Decimal("0"))
    buying_power: Decimal = Field(default=Decimal("0"))
    portfolio_value: Decimal = Field(default=Decimal("0"))

    long_market_value: Decimal = Field(default=Decimal("0"))
    short_market_value: Decimal = Field(default=Decimal("0"))

    initial_margin: Decimal = Field(default=Decimal("0"))
    maintenance_margin: Decimal = Field(default=Decimal("0"))

    unrealized_pl: Decimal = Field(default=Decimal("0"))
    unrealized_pl_percent: Decimal = Field(default=Decimal("0"))

    daily_pl: Decimal = Field(default=Decimal("0"))
    daily_pl_percent: Decimal = Field(default=Decimal("0"))


class AccountInfo(BaseModel):
    """Complete account information model."""

    account_id: str
    account_type: AccountType = Field(default=AccountType.CASH)
    status: AccountStatus = Field(default=AccountStatus.ACTIVE)
    margin_status: MarginStatus = Field(default=MarginStatus.NORMAL)

    currency: str = Field(default="USD")
    created_at: Optional[datetime] = None

    equity: Decimal = Field(default=Decimal("0"))
    last_equity: Decimal = Field(default=Decimal("0"))
    cash: Decimal = Field(default=Decimal("0"))
    buying_power: Decimal = Field(default=Decimal("0"))
    daytrading_buying_power: Decimal = Field(default=Decimal("0"))
    regt_buying_power: Decimal = Field(default=Decimal("0"))

    portfolio_value: Decimal = Field(default=Decimal("0"))
    long_market_value: Decimal = Field(default=Decimal("0"))
    short_market_value: Decimal = Field(default=Decimal("0"))

    initial_margin: Decimal = Field(default=Decimal("0"))
    maintenance_margin: Decimal = Field(default=Decimal("0"))
    sma: Decimal = Field(default=Decimal("0"))

    multiplier: int = Field(default=1)
    pattern_day_trader: bool = Field(default=False)
    trading_blocked: bool = Field(default=False)
    transfers_blocked: bool = Field(default=False)
    account_blocked: bool = Field(default=False)
    shorting_enabled: bool = Field(default=False)

    pending_transfer_in: Decimal = Field(default=Decimal("0"))
    pending_transfer_out: Decimal = Field(default=Decimal("0"))

    accrued_fees: Decimal = Field(default=Decimal("0"))

    last_updated: datetime = Field(default_factory=now_utc)

    @property
    def is_active(self) -> bool:
        """Check if account is active and can trade."""
        return (
            self.status == AccountStatus.ACTIVE
            and not self.trading_blocked
            and not self.account_blocked
        )

    @property
    def can_trade(self) -> bool:
        """Check if trading is allowed."""
        return self.is_active and self.buying_power > 0

    @property
    def margin_utilization(self) -> Decimal:
        """Calculate margin utilization percentage."""
        if self.equity <= 0:
            return Decimal("0")
        return round_decimal(
            (self.initial_margin / self.equity) * 100,
            2
        )

    @property
    def available_margin(self) -> Decimal:
        """Calculate available margin."""
        return self.equity - self.initial_margin

    def get_daily_change(self) -> tuple[Decimal, Decimal]:
        """Get daily P&L change in absolute and percentage terms."""
        daily_pl = self.equity - self.last_equity
        daily_pl_pct = Decimal("0")
        if self.last_equity > 0:
            daily_pl_pct = round_decimal(
                (daily_pl / self.last_equity) * 100,
                2
            )
        return daily_pl, daily_pl_pct


class AccountHistory(BaseModel):
    """Account history tracking."""

    snapshots: list[AccountSnapshot] = Field(default_factory=list)
    max_snapshots: int = Field(default=1000)

    def add_snapshot(self, snapshot: AccountSnapshot) -> None:
        """Add a new snapshot to history."""
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots = self.snapshots[-self.max_snapshots:]

    def get_snapshots_since(
        self,
        since: datetime
    ) -> list[AccountSnapshot]:
        """Get snapshots since a specific time."""
        return [s for s in self.snapshots if s.timestamp >= since]

    def get_latest_snapshot(self) -> Optional[AccountSnapshot]:
        """Get the most recent snapshot."""
        return self.snapshots[-1] if self.snapshots else None

    def get_equity_series(self) -> list[tuple[datetime, Decimal]]:
        """Get equity time series."""
        return [(s.timestamp, s.equity) for s in self.snapshots]

    def calculate_period_return(
        self,
        start: datetime,
        end: datetime
    ) -> Optional[Decimal]:
        """Calculate return over a period."""
        period_snapshots = [
            s for s in self.snapshots
            if start <= s.timestamp <= end
        ]
        if len(period_snapshots) < 2:
            return None

        start_equity = period_snapshots[0].equity
        end_equity = period_snapshots[-1].equity

        if start_equity <= 0:
            return None

        return round_decimal(
            ((end_equity - start_equity) / start_equity) * 100,
            4
        )


@singleton
class AccountManager:
    """
    Manages account information, balances, and buying power.

    This class provides centralized account management including:
    - Real-time balance tracking
    - Buying power calculations
    - Margin monitoring
    - Account history
    - Balance reservations for pending orders
    """

    def __init__(
        self,
        config: Optional[AccountManagerConfig] = None,
        broker_client: Optional[Any] = None,
    ) -> None:
        """
        Initialize AccountManager.

        Args:
            config: Account manager configuration
            broker_client: Broker API client for fetching account data
        """
        self._config = config or AccountManagerConfig()
        self._broker_client = broker_client

        self._account_info: Optional[AccountInfo] = None
        self._history = AccountHistory()
        self._last_update: Optional[datetime] = None

        self._reserved_buying_power: Decimal = Decimal("0")
        self._reservations: dict[str, Decimal] = {}

        self._update_callbacks: list[Callable[[AccountInfo], None]] = []
        self._alert_callbacks: list[Callable[[str, str, dict], None]] = []

        self._update_task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = asyncio.Lock()

        logger.info("AccountManager initialized")

    @property
    def account_info(self) -> Optional[AccountInfo]:
        """Get current account information."""
        return self._account_info

    @property
    def is_running(self) -> bool:
        """Check if manager is running."""
        return self._running

    @property
    def available_buying_power(self) -> Decimal:
        """Get available buying power after reservations."""
        if not self._account_info:
            return Decimal("0")

        available = self._account_info.buying_power - self._reserved_buying_power
        buffer = self._account_info.buying_power * Decimal(str(self._config.min_buying_power_buffer))

        return max(Decimal("0"), available - buffer)

    @property
    def effective_buying_power(self) -> Decimal:
        """Get effective buying power for calculations."""
        return self.available_buying_power

    def set_broker_client(self, broker_client: Any) -> None:
        """Set the broker client."""
        self._broker_client = broker_client
        logger.debug("Broker client set")

    def register_update_callback(
        self,
        callback: Callable[[AccountInfo], None]
    ) -> None:
        """Register a callback for account updates."""
        self._update_callbacks.append(callback)
        logger.debug(f"Registered update callback: {callback.__name__}")

    def register_alert_callback(
        self,
        callback: Callable[[str, str, dict], None]
    ) -> None:
        """Register a callback for account alerts."""
        self._alert_callbacks.append(callback)
        logger.debug(f"Registered alert callback: {callback.__name__}")

    async def start(self) -> None:
        """Start the account manager."""
        if self._running:
            logger.warning("AccountManager already running")
            return

        self._running = True

        await self.refresh_account()

        self._update_task = asyncio.create_task(
            self._update_loop(),
            name="account_update_loop"
        )

        logger.info("AccountManager started")

    async def stop(self) -> None:
        """Stop the account manager."""
        if not self._running:
            return

        self._running = False

        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
            self._update_task = None

        logger.info("AccountManager stopped")

    async def _update_loop(self) -> None:
        """Periodic account update loop."""
        while self._running:
            try:
                await asyncio.sleep(self._config.update_interval_seconds)
                await self.refresh_account()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in account update loop: {e}")
                await asyncio.sleep(5)

    @async_retry(max_attempts=3, delay=1.0)
    async def refresh_account(self) -> Optional[AccountInfo]:
        """
        Refresh account information from broker.

        Returns:
            Updated account information
        """
        if not self._broker_client:
            logger.warning("No broker client configured")
            return self._account_info

        async with self._lock:
            try:
                account_data = await self._fetch_account_data()

                if account_data:
                    self._account_info = self._parse_account_data(account_data)
                    self._last_update = now_utc()

                    snapshot = self._create_snapshot()
                    self._history.add_snapshot(snapshot)

                    await self._check_alerts()
                    await self._notify_update()

                    logger.debug("Account refreshed successfully")

                return self._account_info

            except Exception as e:
                logger.error(f"Failed to refresh account: {e}")
                raise

    async def _fetch_account_data(self) -> Optional[dict]:
        """Fetch account data from broker."""
        try:
            if hasattr(self._broker_client, "get_account"):
                return await self._broker_client.get_account()
            elif hasattr(self._broker_client, "fetch_account"):
                return await self._broker_client.fetch_account()
            else:
                logger.error("Broker client has no account method")
                return None
        except Exception as e:
            logger.error(f"Error fetching account data: {e}")
            raise APIError(f"Failed to fetch account: {e}")

    def _parse_account_data(self, data: dict) -> AccountInfo:
        """Parse raw account data into AccountInfo model."""
        status_map = {
            "ACTIVE": AccountStatus.ACTIVE,
            "SUSPENDED": AccountStatus.SUSPENDED,
            "RESTRICTED": AccountStatus.RESTRICTED,
            "CLOSED": AccountStatus.CLOSED,
            "PENDING": AccountStatus.PENDING,
        }

        account_type = AccountType.CASH
        if data.get("account_type", "").upper() == "MARGIN":
            account_type = AccountType.MARGIN
        elif data.get("is_paper", False):
            account_type = AccountType.PAPER

        return AccountInfo(
            account_id=data.get("account_number", data.get("id", "")),
            account_type=account_type,
            status=status_map.get(
                data.get("status", "ACTIVE").upper(),
                AccountStatus.ACTIVE
            ),
            currency=data.get("currency", "USD"),
            created_at=data.get("created_at"),
            equity=Decimal(str(data.get("equity", 0))),
            last_equity=Decimal(str(data.get("last_equity", 0))),
            cash=Decimal(str(data.get("cash", 0))),
            buying_power=Decimal(str(data.get("buying_power", 0))),
            daytrading_buying_power=Decimal(str(data.get("daytrading_buying_power", 0))),
            regt_buying_power=Decimal(str(data.get("regt_buying_power", 0))),
            portfolio_value=Decimal(str(data.get("portfolio_value", 0))),
            long_market_value=Decimal(str(data.get("long_market_value", 0))),
            short_market_value=Decimal(str(data.get("short_market_value", 0))),
            initial_margin=Decimal(str(data.get("initial_margin", 0))),
            maintenance_margin=Decimal(str(data.get("maintenance_margin", 0))),
            sma=Decimal(str(data.get("sma", 0))),
            multiplier=int(data.get("multiplier", 1)),
            pattern_day_trader=data.get("pattern_day_trader", False),
            trading_blocked=data.get("trading_blocked", False),
            transfers_blocked=data.get("transfers_blocked", False),
            account_blocked=data.get("account_blocked", False),
            shorting_enabled=data.get("shorting_enabled", False),
            pending_transfer_in=Decimal(str(data.get("pending_transfer_in", 0))),
            pending_transfer_out=Decimal(str(data.get("pending_transfer_out", 0))),
            accrued_fees=Decimal(str(data.get("accrued_fees", 0))),
            last_updated=now_utc(),
        )

    def _create_snapshot(self) -> AccountSnapshot:
        """Create a snapshot of current account state."""
        if not self._account_info:
            return AccountSnapshot()

        daily_pl, daily_pl_pct = self._account_info.get_daily_change()

        return AccountSnapshot(
            equity=self._account_info.equity,
            cash=self._account_info.cash,
            buying_power=self._account_info.buying_power,
            portfolio_value=self._account_info.portfolio_value,
            long_market_value=self._account_info.long_market_value,
            short_market_value=self._account_info.short_market_value,
            initial_margin=self._account_info.initial_margin,
            maintenance_margin=self._account_info.maintenance_margin,
            daily_pl=daily_pl,
            daily_pl_percent=daily_pl_pct,
        )

    async def _check_alerts(self) -> None:
        """Check for and trigger account alerts."""
        if not self._account_info or not self._config.enable_balance_alerts:
            return

        if self._account_info.trading_blocked:
            await self._trigger_alert(
                "TRADING_BLOCKED",
                "Trading is blocked on this account",
                {"account_id": self._account_info.account_id}
            )

        if float(self._account_info.buying_power) < self._config.low_balance_threshold:
            await self._trigger_alert(
                "LOW_BUYING_POWER",
                f"Buying power below threshold: ${self._account_info.buying_power}",
                {
                    "buying_power": float(self._account_info.buying_power),
                    "threshold": self._config.low_balance_threshold
                }
            )

        if self._config.enable_margin_monitoring and self._account_info.account_type == AccountType.MARGIN:
            await self._check_margin_alerts()

    async def _check_margin_alerts(self) -> None:
        """Check for margin-related alerts."""
        if not self._account_info:
            return

        if self._account_info.equity <= 0:
            return

        margin_ratio = float(self._account_info.maintenance_margin / self._account_info.equity)

        if margin_ratio >= (1 - self._config.margin_critical_threshold):
            self._account_info.margin_status = MarginStatus.LIQUIDATION
            await self._trigger_alert(
                "MARGIN_LIQUIDATION",
                "Account at risk of liquidation",
                {"margin_ratio": margin_ratio}
            )
        elif margin_ratio >= (1 - self._config.margin_warning_threshold):
            self._account_info.margin_status = MarginStatus.MARGIN_CALL
            await self._trigger_alert(
                "MARGIN_CALL",
                "Margin call warning",
                {"margin_ratio": margin_ratio}
            )
        else:
            self._account_info.margin_status = MarginStatus.NORMAL

    async def _trigger_alert(
        self,
        alert_type: str,
        message: str,
        data: dict
    ) -> None:
        """Trigger an alert to all registered callbacks."""
        logger.warning(f"Account alert [{alert_type}]: {message}")

        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert_type, message, data)
                else:
                    callback(alert_type, message, data)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    async def _notify_update(self) -> None:
        """Notify all update callbacks of account change."""
        if not self._account_info:
            return

        for callback in self._update_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self._account_info)
                else:
                    callback(self._account_info)
            except Exception as e:
                logger.error(f"Error in update callback: {e}")

    def reserve_buying_power(
        self,
        reservation_id: str,
        amount: Decimal
    ) -> bool:
        """
        Reserve buying power for a pending order.

        Args:
            reservation_id: Unique identifier for the reservation
            amount: Amount to reserve

        Returns:
            True if reservation successful
        """
        if amount <= 0:
            raise ValidationError("Reservation amount must be positive")

        if reservation_id in self._reservations:
            logger.warning(f"Reservation {reservation_id} already exists")
            return False

        if amount > self.available_buying_power:
            logger.warning(
                f"Insufficient buying power for reservation: "
                f"requested={amount}, available={self.available_buying_power}"
            )
            return False

        self._reservations[reservation_id] = amount
        self._reserved_buying_power += amount

        logger.debug(
            f"Reserved ${amount} buying power (id={reservation_id}), "
            f"total reserved=${self._reserved_buying_power}"
        )

        return True

    def release_reservation(self, reservation_id: str) -> bool:
        """
        Release a buying power reservation.

        Args:
            reservation_id: Reservation to release

        Returns:
            True if reservation was released
        """
        if reservation_id not in self._reservations:
            logger.warning(f"Reservation {reservation_id} not found")
            return False

        amount = self._reservations.pop(reservation_id)
        self._reserved_buying_power -= amount

        logger.debug(
            f"Released ${amount} buying power (id={reservation_id}), "
            f"total reserved=${self._reserved_buying_power}"
        )

        return True

    def get_reservation(self, reservation_id: str) -> Optional[Decimal]:
        """Get a specific reservation amount."""
        return self._reservations.get(reservation_id)

    def clear_all_reservations(self) -> int:
        """Clear all buying power reservations."""
        count = len(self._reservations)
        self._reservations.clear()
        self._reserved_buying_power = Decimal("0")
        logger.info(f"Cleared {count} reservations")
        return count

    def can_afford(
        self,
        amount: Decimal,
        include_buffer: bool = True
    ) -> bool:
        """
        Check if account can afford a purchase.

        Args:
            amount: Amount to check
            include_buffer: Include safety buffer

        Returns:
            True if affordable
        """
        if not self._account_info:
            return False

        available = (
            self.available_buying_power
            if include_buffer
            else self._account_info.buying_power - self._reserved_buying_power
        )

        return amount <= available

    def validate_order_buying_power(
        self,
        symbol: str,
        quantity: int,
        price: Decimal,
        side: str = "buy"
    ) -> tuple[bool, str]:
        """
        Validate if account has sufficient buying power for an order.

        Args:
            symbol: Trading symbol
            quantity: Order quantity
            price: Order price
            side: Order side (buy/sell)

        Returns:
            Tuple of (is_valid, message)
        """
        if side.lower() == "sell":
            return True, "Sell orders don't require buying power"

        required = Decimal(str(quantity)) * price

        if not self.can_afford(required):
            return False, (
                f"Insufficient buying power: required=${required:.2f}, "
                f"available=${self.available_buying_power:.2f}"
            )

        return True, "Sufficient buying power available"

    def get_max_shares(
        self,
        price: Decimal,
        max_position_value: Optional[Decimal] = None
    ) -> int:
        """
        Calculate maximum shares that can be purchased.

        Args:
            price: Share price
            max_position_value: Optional maximum position value limit

        Returns:
            Maximum number of shares
        """
        if price <= 0:
            return 0

        available = self.available_buying_power

        if max_position_value:
            available = min(available, max_position_value)

        max_shares = int(available / price)
        return max(0, max_shares)

    def get_account_summary(self) -> dict:
        """Get a summary of account status."""
        if not self._account_info:
            return {"status": "no_data"}

        daily_pl, daily_pl_pct = self._account_info.get_daily_change()

        return {
            "account_id": self._account_info.account_id,
            "account_type": self._account_info.account_type.value,
            "status": self._account_info.status.value,
            "equity": float(self._account_info.equity),
            "cash": float(self._account_info.cash),
            "buying_power": float(self._account_info.buying_power),
            "available_buying_power": float(self.available_buying_power),
            "reserved_buying_power": float(self._reserved_buying_power),
            "portfolio_value": float(self._account_info.portfolio_value),
            "daily_pl": float(daily_pl),
            "daily_pl_percent": float(daily_pl_pct),
            "margin_utilization": float(self._account_info.margin_utilization),
            "can_trade": self._account_info.can_trade,
            "last_updated": self._last_update.isoformat() if self._last_update else None,
        }

    def get_history(
        self,
        hours: int = 24
    ) -> list[AccountSnapshot]:
        """Get account history for the specified period."""
        since = now_utc() - timedelta(hours=hours)
        return self._history.get_snapshots_since(since)

    def get_period_return(
        self,
        start: datetime,
        end: Optional[datetime] = None
    ) -> Optional[float]:
        """Calculate return over a period."""
        end = end or now_utc()
        result = self._history.calculate_period_return(start, end)
        return float(result) if result is not None else None

    async def wait_for_buying_power(
        self,
        required: Decimal,
        timeout_seconds: int = 60
    ) -> bool:
        """
        Wait for sufficient buying power to become available.

        Args:
            required: Required buying power
            timeout_seconds: Maximum wait time

        Returns:
            True if buying power became available
        """
        start = now_utc()
        deadline = start + timedelta(seconds=timeout_seconds)

        while now_utc() < deadline:
            if self.can_afford(required):
                return True

            await self.refresh_account()

            if self.can_afford(required):
                return True

            await asyncio.sleep(5)

        logger.warning(
            f"Timeout waiting for buying power: "
            f"required=${required}, available=${self.available_buying_power}"
        )
        return False

    def __repr__(self) -> str:
        """String representation."""
        if self._account_info:
            return (
                f"AccountManager(account_id={self._account_info.account_id}, "
                f"equity=${self._account_info.equity}, "
                f"buying_power=${self._account_info.buying_power})"
            )
        return "AccountManager(no_account)"
