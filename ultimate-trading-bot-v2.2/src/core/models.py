"""
Core Models Module for Ultimate Trading Bot v2.2.

This module defines all core Pydantic models for orders, positions,
accounts, signals, and other trading-related data structures.
"""

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import uuid

from pydantic import BaseModel, Field, field_validator, model_validator, computed_field


# =============================================================================
# ENUMS
# =============================================================================

class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(str, Enum):
    """Order status enumeration."""
    NEW = "new"
    PENDING_NEW = "pending_new"
    ACCEPTED = "accepted"
    PENDING_CANCEL = "pending_cancel"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    REPLACED = "replaced"
    PENDING_REPLACE = "pending_replace"
    DONE_FOR_DAY = "done_for_day"


class TimeInForce(str, Enum):
    """Time in force enumeration."""
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"
    OPG = "opg"
    CLS = "cls"


class PositionSide(str, Enum):
    """Position side enumeration."""
    LONG = "long"
    SHORT = "short"


class SignalType(str, Enum):
    """Trading signal type enumeration."""
    BUY = "buy"
    SELL = "sell"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"
    HOLD = "hold"
    CLOSE = "close"
    NO_SIGNAL = "no_signal"


class AssetClass(str, Enum):
    """Asset class enumeration."""
    US_EQUITY = "us_equity"
    CRYPTO = "crypto"
    OPTIONS = "options"
    FUTURES = "futures"


# =============================================================================
# BASE MODELS
# =============================================================================

class BaseModelWithTimestamp(BaseModel):
    """Base model with timestamp fields."""

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp"
    )

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now(timezone.utc)


# =============================================================================
# ORDER MODELS
# =============================================================================

class Order(BaseModelWithTimestamp):
    """Order model representing a trading order."""

    order_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique order identifier"
    )
    client_order_id: Optional[str] = Field(
        default=None,
        description="Client-specified order ID"
    )
    symbol: str = Field(description="Trading symbol")
    side: OrderSide = Field(description="Order side (buy/sell)")
    order_type: OrderType = Field(description="Order type")
    quantity: float = Field(gt=0, description="Order quantity")
    filled_quantity: float = Field(default=0.0, ge=0, description="Filled quantity")
    limit_price: Optional[float] = Field(default=None, description="Limit price")
    stop_price: Optional[float] = Field(default=None, description="Stop price")
    trail_percent: Optional[float] = Field(default=None, description="Trailing stop percent")
    trail_price: Optional[float] = Field(default=None, description="Trailing stop price")
    time_in_force: TimeInForce = Field(default=TimeInForce.DAY, description="Time in force")
    status: OrderStatus = Field(default=OrderStatus.NEW, description="Order status")
    avg_fill_price: Optional[float] = Field(default=None, description="Average fill price")
    submitted_at: Optional[datetime] = Field(default=None, description="Submission time")
    filled_at: Optional[datetime] = Field(default=None, description="Fill time")
    cancelled_at: Optional[datetime] = Field(default=None, description="Cancellation time")
    expired_at: Optional[datetime] = Field(default=None, description="Expiration time")
    rejected_at: Optional[datetime] = Field(default=None, description="Rejection time")
    reject_reason: Optional[str] = Field(default=None, description="Rejection reason")
    extended_hours: bool = Field(default=False, description="Extended hours trading")
    strategy_id: Optional[str] = Field(default=None, description="Associated strategy")
    tags: Dict[str, str] = Field(default_factory=dict, description="Custom tags")

    @computed_field
    @property
    def remaining_quantity(self) -> float:
        """Calculate remaining unfilled quantity."""
        return self.quantity - self.filled_quantity

    @computed_field
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED

    @computed_field
    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in (
            OrderStatus.NEW,
            OrderStatus.PENDING_NEW,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.PENDING_CANCEL,
            OrderStatus.PENDING_REPLACE,
        )

    @computed_field
    @property
    def is_terminal(self) -> bool:
        """Check if order is in terminal state."""
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.DONE_FOR_DAY,
        )

    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and uppercase symbol."""
        return v.upper().strip()

    @model_validator(mode='after')
    def validate_prices(self) -> 'Order':
        """Validate price fields based on order type."""
        if self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            if self.limit_price is None:
                raise ValueError(f"Limit price required for {self.order_type} orders")

        if self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            if self.stop_price is None:
                raise ValueError(f"Stop price required for {self.order_type} orders")

        return self

    def to_api_format(self) -> Dict[str, Any]:
        """Convert to API submission format."""
        data = {
            "symbol": self.symbol,
            "side": self.side.value,
            "type": self.order_type.value,
            "qty": str(self.quantity),
            "time_in_force": self.time_in_force.value,
            "extended_hours": self.extended_hours,
        }

        if self.client_order_id:
            data["client_order_id"] = self.client_order_id

        if self.limit_price is not None:
            data["limit_price"] = str(self.limit_price)

        if self.stop_price is not None:
            data["stop_price"] = str(self.stop_price)

        if self.trail_percent is not None:
            data["trail_percent"] = str(self.trail_percent)

        if self.trail_price is not None:
            data["trail_price"] = str(self.trail_price)

        return data


class OrderFill(BaseModel):
    """Order fill/execution model."""

    fill_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Fill identifier"
    )
    order_id: str = Field(description="Parent order ID")
    symbol: str = Field(description="Symbol")
    side: OrderSide = Field(description="Order side")
    quantity: float = Field(gt=0, description="Fill quantity")
    price: float = Field(gt=0, description="Fill price")
    commission: float = Field(default=0.0, ge=0, description="Commission")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Fill timestamp"
    )

    @computed_field
    @property
    def notional_value(self) -> float:
        """Calculate notional value of fill."""
        return self.quantity * self.price

    @computed_field
    @property
    def net_value(self) -> float:
        """Calculate net value including commission."""
        value = self.notional_value
        if self.side == OrderSide.BUY:
            return value + self.commission
        return value - self.commission


# =============================================================================
# POSITION MODELS
# =============================================================================

class Position(BaseModelWithTimestamp):
    """Position model representing an open position."""

    position_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Position identifier"
    )
    symbol: str = Field(description="Trading symbol")
    side: PositionSide = Field(description="Position side")
    quantity: float = Field(gt=0, description="Position quantity")
    avg_entry_price: float = Field(gt=0, description="Average entry price")
    current_price: float = Field(default=0.0, ge=0, description="Current market price")
    market_value: float = Field(default=0.0, description="Current market value")
    cost_basis: float = Field(default=0.0, description="Total cost basis")
    unrealized_pnl: float = Field(default=0.0, description="Unrealized P&L")
    unrealized_pnl_percent: float = Field(default=0.0, description="Unrealized P&L %")
    realized_pnl: float = Field(default=0.0, description="Realized P&L")
    stop_loss: Optional[float] = Field(default=None, description="Stop loss price")
    take_profit: Optional[float] = Field(default=None, description="Take profit price")
    trailing_stop: Optional[float] = Field(default=None, description="Trailing stop distance")
    trailing_stop_price: Optional[float] = Field(default=None, description="Current trailing stop")
    high_water_mark: float = Field(default=0.0, description="Highest unrealized value")
    strategy_id: Optional[str] = Field(default=None, description="Associated strategy")
    entry_order_id: Optional[str] = Field(default=None, description="Entry order ID")
    exit_order_id: Optional[str] = Field(default=None, description="Exit order ID")
    tags: Dict[str, str] = Field(default_factory=dict, description="Custom tags")

    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and uppercase symbol."""
        return v.upper().strip()

    def update_price(self, price: float) -> None:
        """Update position with new market price."""
        self.current_price = price
        self.market_value = self.quantity * price

        if self.side == PositionSide.LONG:
            self.unrealized_pnl = self.market_value - self.cost_basis
        else:
            self.unrealized_pnl = self.cost_basis - self.market_value

        if self.cost_basis > 0:
            self.unrealized_pnl_percent = self.unrealized_pnl / self.cost_basis

        # Update high water mark
        if self.unrealized_pnl > self.high_water_mark:
            self.high_water_mark = self.unrealized_pnl

        # Update trailing stop if applicable
        if self.trailing_stop is not None:
            if self.side == PositionSide.LONG:
                new_stop = price - self.trailing_stop
                if self.trailing_stop_price is None or new_stop > self.trailing_stop_price:
                    self.trailing_stop_price = new_stop
            else:
                new_stop = price + self.trailing_stop
                if self.trailing_stop_price is None or new_stop < self.trailing_stop_price:
                    self.trailing_stop_price = new_stop

        self.update_timestamp()

    def should_stop_out(self) -> bool:
        """Check if position should be stopped out."""
        if self.current_price <= 0:
            return False

        # Check stop loss
        if self.stop_loss is not None:
            if self.side == PositionSide.LONG and self.current_price <= self.stop_loss:
                return True
            if self.side == PositionSide.SHORT and self.current_price >= self.stop_loss:
                return True

        # Check trailing stop
        if self.trailing_stop_price is not None:
            if self.side == PositionSide.LONG and self.current_price <= self.trailing_stop_price:
                return True
            if self.side == PositionSide.SHORT and self.current_price >= self.trailing_stop_price:
                return True

        return False

    def should_take_profit(self) -> bool:
        """Check if position should take profit."""
        if self.current_price <= 0 or self.take_profit is None:
            return False

        if self.side == PositionSide.LONG and self.current_price >= self.take_profit:
            return True
        if self.side == PositionSide.SHORT and self.current_price <= self.take_profit:
            return True

        return False


class ClosedPosition(BaseModel):
    """Closed position model for historical tracking."""

    position_id: str = Field(description="Original position ID")
    symbol: str = Field(description="Trading symbol")
    side: PositionSide = Field(description="Position side")
    quantity: float = Field(gt=0, description="Position quantity")
    entry_price: float = Field(gt=0, description="Entry price")
    exit_price: float = Field(gt=0, description="Exit price")
    entry_time: datetime = Field(description="Entry time")
    exit_time: datetime = Field(description="Exit time")
    realized_pnl: float = Field(description="Realized P&L")
    realized_pnl_percent: float = Field(description="Realized P&L %")
    commission: float = Field(default=0.0, description="Total commission")
    strategy_id: Optional[str] = Field(default=None, description="Associated strategy")
    exit_reason: Optional[str] = Field(default=None, description="Exit reason")
    tags: Dict[str, str] = Field(default_factory=dict, description="Custom tags")

    @computed_field
    @property
    def holding_period(self) -> float:
        """Calculate holding period in days."""
        delta = self.exit_time - self.entry_time
        return delta.total_seconds() / 86400

    @computed_field
    @property
    def net_pnl(self) -> float:
        """Calculate net P&L after commission."""
        return self.realized_pnl - self.commission


# =============================================================================
# SIGNAL MODELS
# =============================================================================

class TradingSignal(BaseModel):
    """Trading signal model."""

    signal_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Signal identifier"
    )
    symbol: str = Field(description="Trading symbol")
    signal_type: SignalType = Field(description="Signal type")
    strength: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Signal strength (0-1)"
    )
    price: float = Field(gt=0, description="Signal price")
    entry_price: Optional[float] = Field(default=None, description="Suggested entry")
    stop_loss: Optional[float] = Field(default=None, description="Suggested stop loss")
    take_profit: Optional[float] = Field(default=None, description="Suggested take profit")
    strategy_id: str = Field(description="Source strategy")
    strategy_name: str = Field(default="", description="Strategy name")
    timeframe: str = Field(default="1d", description="Signal timeframe")
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence level"
    )
    indicators: Dict[str, Any] = Field(
        default_factory=dict,
        description="Indicator values"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Signal timestamp"
    )
    expires_at: Optional[datetime] = Field(
        default=None,
        description="Signal expiration"
    )
    is_active: bool = Field(default=True, description="Signal is active")

    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and uppercase symbol."""
        return v.upper().strip()

    @computed_field
    @property
    def is_buy_signal(self) -> bool:
        """Check if this is a buy signal."""
        return self.signal_type in (SignalType.BUY, SignalType.STRONG_BUY)

    @computed_field
    @property
    def is_sell_signal(self) -> bool:
        """Check if this is a sell signal."""
        return self.signal_type in (SignalType.SELL, SignalType.STRONG_SELL)

    @computed_field
    @property
    def is_expired(self) -> bool:
        """Check if signal has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def get_order_side(self) -> Optional[OrderSide]:
        """Get order side for this signal."""
        if self.is_buy_signal:
            return OrderSide.BUY
        if self.is_sell_signal:
            return OrderSide.SELL
        return None


# =============================================================================
# ACCOUNT MODELS
# =============================================================================

class AccountInfo(BaseModel):
    """Account information model."""

    account_id: str = Field(description="Account identifier")
    account_number: Optional[str] = Field(default=None, description="Account number")
    status: str = Field(default="active", description="Account status")
    currency: str = Field(default="USD", description="Account currency")
    cash: float = Field(default=0.0, description="Cash balance")
    portfolio_value: float = Field(default=0.0, description="Total portfolio value")
    buying_power: float = Field(default=0.0, description="Available buying power")
    daytrading_buying_power: float = Field(default=0.0, description="Day trading buying power")
    regt_buying_power: float = Field(default=0.0, description="RegT buying power")
    initial_margin: float = Field(default=0.0, description="Initial margin")
    maintenance_margin: float = Field(default=0.0, description="Maintenance margin")
    last_equity: float = Field(default=0.0, description="Previous day equity")
    equity: float = Field(default=0.0, description="Current equity")
    long_market_value: float = Field(default=0.0, description="Long market value")
    short_market_value: float = Field(default=0.0, description="Short market value")
    pending_transfer_in: float = Field(default=0.0, description="Pending transfers in")
    pending_transfer_out: float = Field(default=0.0, description="Pending transfers out")
    pattern_day_trader: bool = Field(default=False, description="PDT flag")
    trading_blocked: bool = Field(default=False, description="Trading blocked flag")
    transfers_blocked: bool = Field(default=False, description="Transfers blocked flag")
    account_blocked: bool = Field(default=False, description="Account blocked flag")
    shorting_enabled: bool = Field(default=False, description="Shorting enabled")
    multiplier: float = Field(default=1.0, description="Leverage multiplier")
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update time"
    )

    @computed_field
    @property
    def net_liquidation(self) -> float:
        """Calculate net liquidation value."""
        return self.equity

    @computed_field
    @property
    def can_trade(self) -> bool:
        """Check if trading is allowed."""
        return (
            not self.trading_blocked and
            not self.account_blocked and
            self.status == "active"
        )


# =============================================================================
# QUOTE/BAR MODELS
# =============================================================================

class Quote(BaseModel):
    """Real-time quote model."""

    symbol: str = Field(description="Symbol")
    bid_price: float = Field(default=0.0, ge=0, description="Bid price")
    bid_size: int = Field(default=0, ge=0, description="Bid size")
    ask_price: float = Field(default=0.0, ge=0, description="Ask price")
    ask_size: int = Field(default=0, ge=0, description="Ask size")
    last_price: float = Field(default=0.0, ge=0, description="Last price")
    last_size: int = Field(default=0, ge=0, description="Last size")
    volume: int = Field(default=0, ge=0, description="Volume")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Quote timestamp"
    )

    @computed_field
    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        if self.bid_price > 0 and self.ask_price > 0:
            return (self.bid_price + self.ask_price) / 2
        return self.last_price

    @computed_field
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        if self.bid_price > 0 and self.ask_price > 0:
            return self.ask_price - self.bid_price
        return 0.0

    @computed_field
    @property
    def spread_percent(self) -> float:
        """Calculate spread as percentage."""
        mid = self.mid_price
        if mid > 0:
            return self.spread / mid
        return 0.0


class Bar(BaseModel):
    """OHLCV bar model."""

    symbol: str = Field(description="Symbol")
    timestamp: datetime = Field(description="Bar timestamp")
    open: float = Field(gt=0, description="Open price")
    high: float = Field(gt=0, description="High price")
    low: float = Field(gt=0, description="Low price")
    close: float = Field(gt=0, description="Close price")
    volume: int = Field(ge=0, description="Volume")
    vwap: Optional[float] = Field(default=None, description="VWAP")
    trade_count: Optional[int] = Field(default=None, description="Trade count")
    timeframe: str = Field(default="1d", description="Bar timeframe")

    @model_validator(mode='after')
    def validate_ohlc(self) -> 'Bar':
        """Validate OHLC relationships."""
        if self.high < max(self.open, self.close):
            self.high = max(self.open, self.close, self.high)
        if self.low > min(self.open, self.close):
            self.low = min(self.open, self.close, self.low)
        return self

    @computed_field
    @property
    def range(self) -> float:
        """Calculate price range."""
        return self.high - self.low

    @computed_field
    @property
    def body(self) -> float:
        """Calculate candle body."""
        return abs(self.close - self.open)

    @computed_field
    @property
    def is_bullish(self) -> bool:
        """Check if bar is bullish."""
        return self.close > self.open

    @computed_field
    @property
    def is_bearish(self) -> bool:
        """Check if bar is bearish."""
        return self.close < self.open

    @computed_field
    @property
    def change(self) -> float:
        """Calculate price change."""
        return self.close - self.open

    @computed_field
    @property
    def change_percent(self) -> float:
        """Calculate price change percent."""
        if self.open > 0:
            return (self.close - self.open) / self.open
        return 0.0
