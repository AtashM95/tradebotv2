
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from .contracts import TradeFill, RunContext

logger = logging.getLogger(__name__)


@dataclass
class AccountState:
    """
    Represents the current state of a trading account.

    Tracks equity, cash, margin, and performance metrics.
    """
    equity: float
    cash: float
    margin_used: float
    margin_available: float
    buying_power: float
    portfolio_value: float
    open_positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    daily_pnl: float
    commission_paid: float
    last_updated: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert account state to dictionary."""
        return {
            'equity': self.equity,
            'cash': self.cash,
            'margin_used': self.margin_used,
            'margin_available': self.margin_available,
            'buying_power': self.buying_power,
            'portfolio_value': self.portfolio_value,
            'open_positions_value': self.open_positions_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.total_pnl,
            'daily_pnl': self.daily_pnl,
            'commission_paid': self.commission_paid,
            'last_updated': self.last_updated.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class Transaction:
    """Represents a transaction in the account history."""
    transaction_id: str
    timestamp: datetime
    transaction_type: str  # trade, deposit, withdrawal, dividend, fee
    symbol: Optional[str]
    quantity: float
    price: float
    amount: float
    commission: float
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction to dictionary."""
        return {
            'transaction_id': self.transaction_id,
            'timestamp': self.timestamp.isoformat(),
            'transaction_type': self.transaction_type,
            'symbol': self.symbol,
            'quantity': self.quantity,
            'price': self.price,
            'amount': self.amount,
            'commission': self.commission,
            'description': self.description,
            'metadata': self.metadata
        }


class AccountManagerMetrics:
    """Tracks account manager performance metrics."""

    def __init__(self) -> None:
        self.total_trades: int = 0
        self.total_deposits: int = 0
        self.total_withdrawals: int = 0
        self.total_commission: float = 0.0
        self.total_dividends: float = 0.0
        self.peak_equity: float = 0.0
        self.max_drawdown: float = 0.0
        self.max_drawdown_pct: float = 0.0
        self.equity_history: List[float] = []
        self.daily_returns: List[float] = []

    def record_trade(self, commission: float) -> None:
        """Record a trade execution."""
        self.total_trades += 1
        self.total_commission += commission

    def record_deposit(self, amount: float) -> None:
        """Record a deposit."""
        self.total_deposits += 1

    def record_withdrawal(self, amount: float) -> None:
        """Record a withdrawal."""
        self.total_withdrawals += 1

    def update_equity(self, equity: float) -> None:
        """Update equity and calculate metrics."""
        self.equity_history.append(equity)
        self.peak_equity = max(self.peak_equity, equity)

        # Calculate drawdown
        if self.peak_equity > 0:
            drawdown = self.peak_equity - equity
            drawdown_pct = (drawdown / self.peak_equity) * 100
            self.max_drawdown = max(self.max_drawdown, drawdown)
            self.max_drawdown_pct = max(self.max_drawdown_pct, drawdown_pct)

        # Calculate daily return
        if len(self.equity_history) > 1:
            prev_equity = self.equity_history[-2]
            if prev_equity > 0:
                daily_return = ((equity - prev_equity) / prev_equity) * 100
                self.daily_returns.append(daily_return)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'total_trades': self.total_trades,
            'total_deposits': self.total_deposits,
            'total_withdrawals': self.total_withdrawals,
            'total_commission': self.total_commission,
            'total_dividends': self.total_dividends,
            'peak_equity': self.peak_equity,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct,
            'avg_daily_return': sum(self.daily_returns) / len(self.daily_returns) if self.daily_returns else 0.0
        }


class AccountManager:
    """
    Manages trading account state and transactions.

    Responsibilities:
    - Track account equity, cash, and margin
    - Record and manage transactions
    - Calculate buying power and margin requirements
    - Track account performance metrics
    - Provide account state snapshots
    - Handle deposits and withdrawals
    - Track commissions and fees
    """

    def __init__(
        self,
        initial_equity: float = 100000.0,
        margin_multiplier: float = 1.0,
        commission_per_trade: float = 0.0,
        enable_margin: bool = False
    ) -> None:
        """
        Initialize the account manager.

        Args:
            initial_equity: Starting account equity
            margin_multiplier: Margin multiplier (1.0 = no margin, 2.0 = 2x leverage)
            commission_per_trade: Commission charged per trade
            enable_margin: Whether to enable margin trading
        """
        self.initial_equity = initial_equity
        self.margin_multiplier = margin_multiplier if enable_margin else 1.0
        self.commission_per_trade = commission_per_trade
        self.enable_margin = enable_margin

        # Account state
        self.equity = initial_equity
        self.cash = initial_equity
        self.margin_used = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.daily_start_equity = initial_equity

        # Transaction history
        self.transactions: List[Transaction] = []
        self.transaction_count = 0

        # Metrics
        self.metrics = AccountManagerMetrics()
        self.metrics.peak_equity = initial_equity
        self.metrics.update_equity(initial_equity)

        logger.info("AccountManager initialized", extra={
            'initial_equity': initial_equity,
            'margin_multiplier': margin_multiplier,
            'enable_margin': enable_margin
        })

    def process_fill(
        self,
        fill: TradeFill,
        context: Optional[RunContext] = None
    ) -> None:
        """
        Process a trade fill and update account state.

        Args:
            fill: Trade fill to process
            context: Optional run context
        """
        action = fill.action.lower()
        amount = fill.quantity * fill.price
        commission = self.commission_per_trade

        # Create transaction record
        transaction = Transaction(
            transaction_id=f"TXN-{self.transaction_count:08d}",
            timestamp=fill.timestamp,
            transaction_type='trade',
            symbol=fill.symbol,
            quantity=fill.quantity,
            price=fill.price,
            amount=amount,
            commission=commission,
            description=f"{action.upper()} {fill.quantity} {fill.symbol} @ ${fill.price:.2f}",
            metadata=fill.metadata.copy()
        )

        self.transactions.append(transaction)
        self.transaction_count += 1

        # Update cash based on action
        if action in ('buy', 'cover'):
            # Cash outflow
            self.cash -= (amount + commission)
        elif action in ('sell', 'short'):
            # Cash inflow
            self.cash += (amount - commission)

        # Update metrics
        self.metrics.record_trade(commission)

        log_extra = {
            'transaction_id': transaction.transaction_id,
            'symbol': fill.symbol,
            'action': action,
            'amount': amount,
            'cash': self.cash
        }
        if context:
            log_extra['run_id'] = context.run_id

        logger.info("Trade processed", extra=log_extra)

    def update_positions_value(
        self,
        positions_value: float,
        unrealized_pnl: float,
        context: Optional[RunContext] = None
    ) -> None:
        """
        Update account with current positions value.

        Args:
            positions_value: Current market value of all positions
            unrealized_pnl: Current unrealized P&L
            context: Optional run context
        """
        self.unrealized_pnl = unrealized_pnl

        # Calculate equity
        self.equity = self.cash + positions_value

        # Calculate margin used
        if positions_value > 0:
            self.margin_used = positions_value / self.margin_multiplier
        else:
            self.margin_used = 0.0

        # Update metrics
        self.metrics.update_equity(self.equity)

    def deposit(
        self,
        amount: float,
        description: str = "Deposit",
        context: Optional[RunContext] = None
    ) -> None:
        """
        Deposit funds into the account.

        Args:
            amount: Amount to deposit
            description: Description of the deposit
            context: Optional run context
        """
        if amount <= 0:
            logger.warning(f"Invalid deposit amount: {amount}")
            return

        transaction = Transaction(
            transaction_id=f"TXN-{self.transaction_count:08d}",
            timestamp=datetime.utcnow(),
            transaction_type='deposit',
            symbol=None,
            quantity=0.0,
            price=0.0,
            amount=amount,
            commission=0.0,
            description=description
        )

        self.transactions.append(transaction)
        self.transaction_count += 1

        self.cash += amount
        self.equity += amount
        self.metrics.record_deposit(amount)

        log_extra = {'transaction_id': transaction.transaction_id, 'amount': amount}
        if context:
            log_extra['run_id'] = context.run_id

        logger.info("Deposit processed", extra=log_extra)

    def withdraw(
        self,
        amount: float,
        description: str = "Withdrawal",
        context: Optional[RunContext] = None
    ) -> bool:
        """
        Withdraw funds from the account.

        Args:
            amount: Amount to withdraw
            description: Description of the withdrawal
            context: Optional run context

        Returns:
            True if successful, False if insufficient funds
        """
        if amount <= 0:
            logger.warning(f"Invalid withdrawal amount: {amount}")
            return False

        if amount > self.cash:
            logger.warning(f"Insufficient funds for withdrawal: {amount} > {self.cash}")
            return False

        transaction = Transaction(
            transaction_id=f"TXN-{self.transaction_count:08d}",
            timestamp=datetime.utcnow(),
            transaction_type='withdrawal',
            symbol=None,
            quantity=0.0,
            price=0.0,
            amount=-amount,
            commission=0.0,
            description=description
        )

        self.transactions.append(transaction)
        self.transaction_count += 1

        self.cash -= amount
        self.equity -= amount
        self.metrics.record_withdrawal(amount)

        log_extra = {'transaction_id': transaction.transaction_id, 'amount': amount}
        if context:
            log_extra['run_id'] = context.run_id

        logger.info("Withdrawal processed", extra=log_extra)
        return True

    def record_dividend(
        self,
        symbol: str,
        amount: float,
        context: Optional[RunContext] = None
    ) -> None:
        """
        Record a dividend payment.

        Args:
            symbol: Symbol that paid the dividend
            amount: Dividend amount
            context: Optional run context
        """
        transaction = Transaction(
            transaction_id=f"TXN-{self.transaction_count:08d}",
            timestamp=datetime.utcnow(),
            transaction_type='dividend',
            symbol=symbol,
            quantity=0.0,
            price=0.0,
            amount=amount,
            commission=0.0,
            description=f"Dividend from {symbol}"
        )

        self.transactions.append(transaction)
        self.transaction_count += 1

        self.cash += amount
        self.equity += amount
        self.realized_pnl += amount
        self.metrics.total_dividends += amount

        log_extra = {
            'transaction_id': transaction.transaction_id,
            'symbol': symbol,
            'amount': amount
        }
        if context:
            log_extra['run_id'] = context.run_id

        logger.info("Dividend recorded", extra=log_extra)

    def get_state(self) -> AccountState:
        """Get current account state snapshot."""
        margin_available = self.cash * self.margin_multiplier - self.margin_used
        buying_power = self.cash * self.margin_multiplier
        portfolio_value = self.equity
        open_positions_value = self.equity - self.cash
        total_pnl = self.realized_pnl + self.unrealized_pnl
        daily_pnl = self.equity - self.daily_start_equity

        return AccountState(
            equity=self.equity,
            cash=self.cash,
            margin_used=self.margin_used,
            margin_available=margin_available,
            buying_power=buying_power,
            portfolio_value=portfolio_value,
            open_positions_value=open_positions_value,
            unrealized_pnl=self.unrealized_pnl,
            realized_pnl=self.realized_pnl,
            total_pnl=total_pnl,
            daily_pnl=daily_pnl,
            commission_paid=self.metrics.total_commission,
            last_updated=datetime.utcnow()
        )

    def get_equity(self) -> float:
        """Get current equity."""
        return self.equity

    def get_cash(self) -> float:
        """Get available cash."""
        return self.cash

    def get_buying_power(self) -> float:
        """Get available buying power."""
        return self.cash * self.margin_multiplier

    def get_margin_available(self) -> float:
        """Get available margin."""
        return self.cash * self.margin_multiplier - self.margin_used

    def can_afford(self, amount: float) -> bool:
        """
        Check if account can afford a purchase.

        Args:
            amount: Amount needed

        Returns:
            True if affordable, False otherwise
        """
        buying_power = self.get_buying_power()
        return amount <= buying_power

    def get_max_position_size(self, price: float) -> float:
        """
        Calculate maximum position size affordable.

        Args:
            price: Price per share

        Returns:
            Maximum quantity affordable
        """
        buying_power = self.get_buying_power()
        if price <= 0:
            return 0.0
        return (buying_power - self.commission_per_trade) / price

    def get_transactions(self, limit: Optional[int] = None) -> List[Transaction]:
        """Get transaction history."""
        if limit:
            return self.transactions[-limit:]
        return self.transactions

    def get_transactions_by_symbol(self, symbol: str) -> List[Transaction]:
        """Get transactions for a specific symbol."""
        return [t for t in self.transactions if t.symbol == symbol]

    def get_metrics(self) -> Dict[str, Any]:
        """Get account metrics."""
        metrics = self.metrics.to_dict()
        metrics['current_equity'] = self.equity
        metrics['return_pct'] = ((self.equity - self.initial_equity) / self.initial_equity) * 100
        metrics['total_pnl'] = self.realized_pnl + self.unrealized_pnl
        return metrics

    def reset_daily_tracking(self) -> None:
        """Reset daily tracking metrics (call at start of each trading day)."""
        self.daily_start_equity = self.equity
        logger.info("Daily tracking reset", extra={'equity': self.equity})

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on account manager.

        Returns:
            Health status dictionary
        """
        state = self.get_state()
        return {
            'healthy': True,
            'equity': state.equity,
            'cash': state.cash,
            'buying_power': state.buying_power,
            'margin_used': state.margin_used,
            'total_pnl': state.total_pnl,
            'metrics': self.metrics.to_dict()
        }
