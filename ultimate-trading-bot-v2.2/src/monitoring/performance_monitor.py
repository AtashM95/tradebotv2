"""
Performance Monitor for Ultimate Trading Bot v2.2.

Monitors trading performance, strategy metrics, and provides performance analytics.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class TimeFrame(str, Enum):
    """Time frames for performance analysis."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    ALL_TIME = "all_time"


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring."""

    # Update intervals
    update_interval: float = 60.0  # seconds
    snapshot_interval: float = 3600.0  # hourly snapshots

    # Risk-free rate for Sharpe calculation
    risk_free_rate: float = 0.02  # 2% annual

    # Benchmark
    benchmark_symbol: str = "SPY"

    # History settings
    max_history_days: int = 365
    min_trades_for_stats: int = 10


@dataclass
class TradeRecord:
    """Record of a single trade."""

    trade_id: str
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime

    # P&L
    gross_pnl: float
    commission: float
    net_pnl: float

    # Optional
    strategy: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def return_pct(self) -> float:
        """Get return percentage."""
        cost = self.entry_price * self.quantity
        if cost > 0:
            return self.net_pnl / cost
        return 0.0

    @property
    def holding_period(self) -> timedelta:
        """Get holding period."""
        return self.exit_time - self.entry_time

    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.net_pnl > 0


@dataclass
class PerformanceSnapshot:
    """Snapshot of performance metrics."""

    timestamp: datetime
    total_value: float
    cash: float
    positions_value: float

    # Daily metrics
    daily_return: float = 0.0
    daily_pnl: float = 0.0

    # Cumulative metrics
    total_return: float = 0.0
    total_pnl: float = 0.0

    # Trade metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # Risk metrics
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""

    # Time period
    start_date: datetime
    end_date: datetime
    trading_days: int = 0

    # Returns
    total_return: float = 0.0
    annualized_return: float = 0.0
    average_daily_return: float = 0.0
    best_day: float = 0.0
    worst_day: float = 0.0

    # Risk metrics
    volatility: float = 0.0
    downside_volatility: float = 0.0
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    var_95: float = 0.0
    cvar_95: float = 0.0

    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_holding_period: float = 0.0

    # P&L
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    total_commission: float = 0.0

    # Benchmark comparison
    benchmark_return: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    correlation: float = 0.0
    tracking_error: float = 0.0
    information_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "period": {
                "start": self.start_date.isoformat(),
                "end": self.end_date.isoformat(),
                "trading_days": self.trading_days,
            },
            "returns": {
                "total": self.total_return,
                "annualized": self.annualized_return,
                "daily_avg": self.average_daily_return,
                "best_day": self.best_day,
                "worst_day": self.worst_day,
            },
            "risk": {
                "volatility": self.volatility,
                "max_drawdown": self.max_drawdown,
                "var_95": self.var_95,
                "cvar_95": self.cvar_95,
            },
            "risk_adjusted": {
                "sharpe": self.sharpe_ratio,
                "sortino": self.sortino_ratio,
                "calmar": self.calmar_ratio,
            },
            "trades": {
                "total": self.total_trades,
                "winners": self.winning_trades,
                "losers": self.losing_trades,
                "win_rate": self.win_rate,
                "profit_factor": self.profit_factor,
            },
            "pnl": {
                "net_profit": self.net_profit,
                "gross_profit": self.gross_profit,
                "gross_loss": self.gross_loss,
                "commission": self.total_commission,
            },
            "benchmark": {
                "return": self.benchmark_return,
                "alpha": self.alpha,
                "beta": self.beta,
            },
        }


class PerformanceMonitor:
    """
    Comprehensive trading performance monitoring.

    Tracks trades, calculates metrics, and provides performance analytics.
    """

    def __init__(self, config: PerformanceConfig | None = None) -> None:
        """
        Initialize performance monitor.

        Args:
            config: Performance configuration
        """
        self.config = config or PerformanceConfig()

        # Trade history
        self._trades: list[TradeRecord] = []

        # Value history
        self._snapshots: list[PerformanceSnapshot] = []
        self._daily_returns: list[float] = []
        self._value_history: list[tuple[datetime, float]] = []

        # Benchmark returns
        self._benchmark_returns: list[float] = []

        # Peak tracking
        self._peak_value = 0.0
        self._initial_value = 0.0

        # Running totals
        self._total_commission = 0.0

        # Background task
        self._update_task: asyncio.Task | None = None
        self._running = False

        logger.info("PerformanceMonitor initialized")

    def set_initial_value(self, value: float) -> None:
        """Set initial portfolio value."""
        self._initial_value = value
        self._peak_value = value

    def record_trade(self, trade: TradeRecord) -> None:
        """
        Record a completed trade.

        Args:
            trade: Trade record
        """
        self._trades.append(trade)
        self._total_commission += trade.commission

        logger.debug(
            f"Recorded trade: {trade.symbol} "
            f"P&L: ${trade.net_pnl:,.2f}"
        )

    def record_snapshot(
        self,
        total_value: float,
        cash: float,
        positions_value: float,
    ) -> PerformanceSnapshot:
        """
        Record a performance snapshot.

        Args:
            total_value: Total portfolio value
            cash: Cash balance
            positions_value: Value of positions

        Returns:
            Performance snapshot
        """
        now = datetime.now()

        # Calculate daily return
        if self._snapshots:
            prev_value = self._snapshots[-1].total_value
            daily_return = (total_value - prev_value) / prev_value if prev_value > 0 else 0.0
            daily_pnl = total_value - prev_value
        else:
            daily_return = 0.0
            daily_pnl = 0.0

        self._daily_returns.append(daily_return)

        # Update peak
        if total_value > self._peak_value:
            self._peak_value = total_value

        # Calculate drawdown
        current_dd = (self._peak_value - total_value) / self._peak_value if self._peak_value > 0 else 0.0

        # Calculate max drawdown
        max_dd = max(s.current_drawdown for s in self._snapshots) if self._snapshots else 0.0
        max_dd = max(max_dd, current_dd)

        # Calculate totals
        total_return = (total_value - self._initial_value) / self._initial_value if self._initial_value > 0 else 0.0
        total_pnl = total_value - self._initial_value

        # Trade counts
        total_trades = len(self._trades)
        winning_trades = sum(1 for t in self._trades if t.is_winner)
        losing_trades = sum(1 for t in self._trades if not t.is_winner)

        snapshot = PerformanceSnapshot(
            timestamp=now,
            total_value=total_value,
            cash=cash,
            positions_value=positions_value,
            daily_return=daily_return,
            daily_pnl=daily_pnl,
            total_return=total_return,
            total_pnl=total_pnl,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            current_drawdown=current_dd,
            max_drawdown=max_dd,
        )

        self._snapshots.append(snapshot)
        self._value_history.append((now, total_value))

        # Trim old data
        self._trim_history()

        return snapshot

    def add_benchmark_return(self, daily_return: float) -> None:
        """Add benchmark daily return."""
        self._benchmark_returns.append(daily_return)

    def calculate_metrics(
        self,
        timeframe: TimeFrame = TimeFrame.ALL_TIME,
    ) -> PerformanceMetrics:
        """
        Calculate performance metrics for a time frame.

        Args:
            timeframe: Time frame for analysis

        Returns:
            Performance metrics
        """
        # Filter data by timeframe
        trades, returns, benchmark = self._filter_by_timeframe(timeframe)

        if not returns:
            return PerformanceMetrics(
                start_date=datetime.now(),
                end_date=datetime.now(),
            )

        returns_arr = np.array(returns)
        trading_days = len(returns)

        # Date range
        if self._snapshots:
            start_date = self._snapshots[0].timestamp
            end_date = self._snapshots[-1].timestamp
        else:
            start_date = datetime.now()
            end_date = datetime.now()

        # Calculate returns
        total_return = float(np.prod(1 + returns_arr) - 1)
        annualized_return = self._annualize_return(returns_arr)
        avg_daily = float(np.mean(returns_arr))
        best_day = float(np.max(returns_arr))
        worst_day = float(np.min(returns_arr))

        # Calculate risk metrics
        volatility = float(np.std(returns_arr) * np.sqrt(252))
        downside_returns = returns_arr[returns_arr < 0]
        downside_vol = (
            float(np.std(downside_returns) * np.sqrt(252))
            if len(downside_returns) > 0 else 0.0
        )

        # Drawdown metrics
        max_dd, avg_dd, max_dd_duration = self._calculate_drawdown_metrics(returns_arr)

        # VaR and CVaR
        var_95 = float(np.percentile(returns_arr, 5))
        cvar_95 = float(np.mean(returns_arr[returns_arr <= var_95])) if len(returns_arr) > 0 else 0.0

        # Risk-adjusted metrics
        risk_free_daily = self.config.risk_free_rate / 252
        excess_returns = returns_arr - risk_free_daily

        sharpe = (
            float(np.mean(excess_returns) / np.std(returns_arr) * np.sqrt(252))
            if np.std(returns_arr) > 0 else 0.0
        )

        sortino = (
            float(np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252))
            if len(downside_returns) > 0 and np.std(downside_returns) > 0 else 0.0
        )

        calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0.0

        # Trade statistics
        trade_metrics = self._calculate_trade_metrics(trades)

        # Benchmark comparison
        benchmark_metrics = self._calculate_benchmark_metrics(returns_arr, benchmark)

        return PerformanceMetrics(
            start_date=start_date,
            end_date=end_date,
            trading_days=trading_days,
            total_return=total_return,
            annualized_return=annualized_return,
            average_daily_return=avg_daily,
            best_day=best_day,
            worst_day=worst_day,
            volatility=volatility,
            downside_volatility=downside_vol,
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            max_drawdown_duration=max_dd_duration,
            var_95=var_95,
            cvar_95=cvar_95,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            **trade_metrics,
            **benchmark_metrics,
        )

    def _filter_by_timeframe(
        self,
        timeframe: TimeFrame,
    ) -> tuple[list[TradeRecord], list[float], list[float]]:
        """Filter data by timeframe."""
        now = datetime.now()

        if timeframe == TimeFrame.HOURLY:
            cutoff = now - timedelta(hours=1)
        elif timeframe == TimeFrame.DAILY:
            cutoff = now - timedelta(days=1)
        elif timeframe == TimeFrame.WEEKLY:
            cutoff = now - timedelta(weeks=1)
        elif timeframe == TimeFrame.MONTHLY:
            cutoff = now - timedelta(days=30)
        elif timeframe == TimeFrame.QUARTERLY:
            cutoff = now - timedelta(days=90)
        elif timeframe == TimeFrame.YEARLY:
            cutoff = now - timedelta(days=365)
        else:
            # All time
            return self._trades, self._daily_returns, self._benchmark_returns

        trades = [t for t in self._trades if t.exit_time >= cutoff]

        # Filter returns by snapshot timestamps
        filtered_returns = []
        filtered_benchmark = []
        for i, snapshot in enumerate(self._snapshots):
            if snapshot.timestamp >= cutoff:
                if i < len(self._daily_returns):
                    filtered_returns.append(self._daily_returns[i])
                if i < len(self._benchmark_returns):
                    filtered_benchmark.append(self._benchmark_returns[i])

        return trades, filtered_returns, filtered_benchmark

    def _calculate_trade_metrics(
        self,
        trades: list[TradeRecord],
    ) -> dict[str, Any]:
        """Calculate trade statistics."""
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "average_win": 0.0,
                "average_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "avg_holding_period": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "net_profit": 0.0,
                "total_commission": 0.0,
            }

        winners = [t for t in trades if t.is_winner]
        losers = [t for t in trades if not t.is_winner]

        gross_profit = sum(t.net_pnl for t in winners)
        gross_loss = abs(sum(t.net_pnl for t in losers))
        net_profit = sum(t.net_pnl for t in trades)
        total_commission = sum(t.commission for t in trades)

        win_rate = len(winners) / len(trades) if trades else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_win = gross_profit / len(winners) if winners else 0.0
        avg_loss = gross_loss / len(losers) if losers else 0.0

        largest_win = max(t.net_pnl for t in winners) if winners else 0.0
        largest_loss = min(t.net_pnl for t in losers) if losers else 0.0

        holding_periods = [t.holding_period.total_seconds() / 3600 for t in trades]
        avg_holding = np.mean(holding_periods) if holding_periods else 0.0

        return {
            "total_trades": len(trades),
            "winning_trades": len(winners),
            "losing_trades": len(losers),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "average_win": avg_win,
            "average_loss": avg_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "avg_holding_period": float(avg_holding),
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "net_profit": net_profit,
            "total_commission": total_commission,
        }

    def _calculate_benchmark_metrics(
        self,
        returns: np.ndarray,
        benchmark: list[float],
    ) -> dict[str, Any]:
        """Calculate benchmark comparison metrics."""
        if not benchmark or len(benchmark) != len(returns):
            return {
                "benchmark_return": 0.0,
                "alpha": 0.0,
                "beta": 0.0,
                "correlation": 0.0,
                "tracking_error": 0.0,
                "information_ratio": 0.0,
            }

        benchmark_arr = np.array(benchmark)

        benchmark_return = float(np.prod(1 + benchmark_arr) - 1)

        # Beta and alpha
        cov_matrix = np.cov(returns, benchmark_arr)
        if cov_matrix[1, 1] > 0:
            beta = float(cov_matrix[0, 1] / cov_matrix[1, 1])
        else:
            beta = 0.0

        portfolio_annualized = self._annualize_return(returns)
        benchmark_annualized = self._annualize_return(benchmark_arr)
        risk_free = self.config.risk_free_rate

        alpha = portfolio_annualized - (risk_free + beta * (benchmark_annualized - risk_free))

        # Correlation
        correlation = float(np.corrcoef(returns, benchmark_arr)[0, 1])

        # Tracking error and information ratio
        tracking_diff = returns - benchmark_arr
        tracking_error = float(np.std(tracking_diff) * np.sqrt(252))
        information_ratio = (
            float(np.mean(tracking_diff) * np.sqrt(252) / np.std(tracking_diff))
            if np.std(tracking_diff) > 0 else 0.0
        )

        return {
            "benchmark_return": benchmark_return,
            "alpha": alpha,
            "beta": beta,
            "correlation": correlation,
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
        }

    def _annualize_return(self, returns: np.ndarray) -> float:
        """Annualize daily returns."""
        if len(returns) == 0:
            return 0.0

        total_return = float(np.prod(1 + returns) - 1)
        days = len(returns)

        if days > 0:
            return float((1 + total_return) ** (252 / days) - 1)
        return 0.0

    def _calculate_drawdown_metrics(
        self,
        returns: np.ndarray,
    ) -> tuple[float, float, int]:
        """Calculate drawdown metrics."""
        if len(returns) == 0:
            return 0.0, 0.0, 0

        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (running_max - cum_returns) / running_max

        max_dd = float(np.max(drawdowns))
        avg_dd = float(np.mean(drawdowns[drawdowns > 0])) if np.any(drawdowns > 0) else 0.0

        # Max duration
        in_drawdown = False
        current_duration = 0
        max_duration = 0

        for dd in drawdowns:
            if dd > 0:
                if not in_drawdown:
                    in_drawdown = True
                    current_duration = 1
                else:
                    current_duration += 1
            else:
                if in_drawdown:
                    max_duration = max(max_duration, current_duration)
                    in_drawdown = False
                    current_duration = 0

        if in_drawdown:
            max_duration = max(max_duration, current_duration)

        return max_dd, avg_dd, max_duration

    def _trim_history(self) -> None:
        """Trim old history data."""
        cutoff = datetime.now() - timedelta(days=self.config.max_history_days)

        self._snapshots = [s for s in self._snapshots if s.timestamp >= cutoff]
        self._trades = [t for t in self._trades if t.exit_time >= cutoff]

    def get_current_snapshot(self) -> PerformanceSnapshot | None:
        """Get most recent snapshot."""
        return self._snapshots[-1] if self._snapshots else None

    def get_trade_history(
        self,
        symbol: str | None = None,
        strategy: str | None = None,
        limit: int = 100,
    ) -> list[TradeRecord]:
        """Get trade history with optional filters."""
        trades = self._trades

        if symbol:
            trades = [t for t in trades if t.symbol == symbol]

        if strategy:
            trades = [t for t in trades if t.strategy == strategy]

        return sorted(
            trades,
            key=lambda t: t.exit_time,
            reverse=True,
        )[:limit]

    def clear(self) -> None:
        """Clear all history."""
        self._trades.clear()
        self._snapshots.clear()
        self._daily_returns.clear()
        self._value_history.clear()
        self._benchmark_returns.clear()
        self._peak_value = 0.0
        self._total_commission = 0.0


def create_performance_monitor(
    config: PerformanceConfig | None = None,
) -> PerformanceMonitor:
    """
    Create a performance monitor instance.

    Args:
        config: Performance configuration

    Returns:
        PerformanceMonitor instance
    """
    return PerformanceMonitor(config)
