"""
Performance Analyzer for Ultimate Trading Bot v2.2.

This module provides comprehensive performance analysis for
backtest results including metrics calculation and attribution.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from src.backtesting.backtest_engine import BacktestResult, BacktestTrade, PortfolioSnapshot


logger = logging.getLogger(__name__)


class PerformanceConfig(BaseModel):
    """Configuration for performance analysis."""

    model_config = {"arbitrary_types_allowed": True}

    risk_free_rate: float = Field(default=0.05, description="Annual risk-free rate")
    trading_days_per_year: int = Field(default=252, description="Trading days per year")
    confidence_level: float = Field(default=0.95, description="Confidence level for VaR")
    benchmark_symbol: str | None = Field(default="SPY", description="Benchmark symbol")
    rolling_window_days: int = Field(default=60, description="Rolling window for metrics")


@dataclass
class RiskMetrics:
    """Risk-related performance metrics."""

    volatility_annual: float = 0.0
    downside_volatility: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    avg_drawdown: float = 0.0
    ulcer_index: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    tail_ratio: float = 0.0


@dataclass
class ReturnMetrics:
    """Return-related performance metrics."""

    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0
    cagr: float = 0.0
    monthly_return_avg: float = 0.0
    monthly_return_std: float = 0.0
    best_month: float = 0.0
    worst_month: float = 0.0
    positive_months: int = 0
    negative_months: int = 0


@dataclass
class RatioMetrics:
    """Ratio-based performance metrics."""

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    treynor_ratio: float = 0.0
    information_ratio: float = 0.0
    recovery_factor: float = 0.0
    profit_factor: float = 0.0
    payoff_ratio: float = 0.0
    common_sense_ratio: float = 0.0


@dataclass
class TradeMetrics:
    """Trade-related performance metrics."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_holding_period_days: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    expectancy: float = 0.0
    trade_frequency: float = 0.0


@dataclass
class BenchmarkMetrics:
    """Benchmark comparison metrics."""

    benchmark_return: float = 0.0
    excess_return: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    r_squared: float = 0.0
    correlation: float = 0.0
    tracking_error: float = 0.0
    information_ratio: float = 0.0
    up_capture: float = 0.0
    down_capture: float = 0.0


@dataclass
class PerformanceReport:
    """Complete performance report."""

    backtest_id: str
    strategy_name: str = ""
    period_start: datetime | None = None
    period_end: datetime | None = None
    trading_days: int = 0

    return_metrics: ReturnMetrics = field(default_factory=ReturnMetrics)
    risk_metrics: RiskMetrics = field(default_factory=RiskMetrics)
    ratio_metrics: RatioMetrics = field(default_factory=RatioMetrics)
    trade_metrics: TradeMetrics = field(default_factory=TradeMetrics)
    benchmark_metrics: BenchmarkMetrics = field(default_factory=BenchmarkMetrics)

    monthly_returns: list[tuple[str, float]] = field(default_factory=list)
    rolling_sharpe: list[tuple[datetime, float]] = field(default_factory=list)
    drawdown_series: list[tuple[datetime, float]] = field(default_factory=list)


class PerformanceAnalyzer:
    """
    Analyzes backtest performance.

    Provides comprehensive performance metrics, attribution,
    and comparison analysis.
    """

    def __init__(self, config: PerformanceConfig | None = None):
        """
        Initialize performance analyzer.

        Args:
            config: Analysis configuration
        """
        self.config = config or PerformanceConfig()

        logger.info("PerformanceAnalyzer initialized")

    def analyze(
        self,
        result: BacktestResult,
        benchmark_returns: list[float] | None = None,
        strategy_name: str = "",
    ) -> PerformanceReport:
        """
        Analyze backtest results.

        Args:
            result: Backtest result to analyze
            benchmark_returns: Optional benchmark returns
            strategy_name: Strategy name for report

        Returns:
            PerformanceReport
        """
        report = PerformanceReport(
            backtest_id=result.backtest_id,
            strategy_name=strategy_name,
            period_start=result.start_time,
            period_end=result.end_time,
        )

        returns = self._calculate_returns(result.portfolio_history)
        report.trading_days = len(returns)

        report.return_metrics = self._calculate_return_metrics(
            result, returns
        )

        report.risk_metrics = self._calculate_risk_metrics(returns)

        report.ratio_metrics = self._calculate_ratio_metrics(
            returns, report.return_metrics, report.risk_metrics
        )

        report.trade_metrics = self._calculate_trade_metrics(result.trades)

        if benchmark_returns:
            report.benchmark_metrics = self._calculate_benchmark_metrics(
                returns, benchmark_returns
            )

        report.monthly_returns = self._calculate_monthly_returns(
            result.portfolio_history
        )

        report.rolling_sharpe = self._calculate_rolling_sharpe(
            returns, result.portfolio_history
        )

        report.drawdown_series = [
            (s.timestamp, s.drawdown)
            for s in result.portfolio_history
        ]

        return report

    def _calculate_returns(
        self,
        portfolio_history: list[PortfolioSnapshot],
    ) -> list[float]:
        """Calculate daily returns from portfolio history."""
        returns: list[float] = []

        for i in range(1, len(portfolio_history)):
            prev_equity = portfolio_history[i - 1].equity
            curr_equity = portfolio_history[i].equity

            if prev_equity > 0:
                daily_return = (curr_equity - prev_equity) / prev_equity
                returns.append(daily_return)

        return returns

    def _calculate_return_metrics(
        self,
        result: BacktestResult,
        returns: list[float],
    ) -> ReturnMetrics:
        """Calculate return metrics."""
        metrics = ReturnMetrics()

        metrics.total_return = result.total_return
        metrics.total_return_pct = result.total_return_pct

        if returns:
            days = len(returns)
            years = days / self.config.trading_days_per_year

            metrics.annualized_return = np.mean(returns) * self.config.trading_days_per_year

            if years > 0:
                metrics.cagr = (
                    (result.final_equity / result.initial_capital) ** (1 / years) - 1
                )

        return metrics

    def _calculate_risk_metrics(
        self,
        returns: list[float],
    ) -> RiskMetrics:
        """Calculate risk metrics."""
        metrics = RiskMetrics()

        if not returns:
            return metrics

        returns_arr = np.array(returns)

        metrics.volatility_annual = float(
            np.std(returns_arr) * np.sqrt(self.config.trading_days_per_year)
        )

        negative_returns = returns_arr[returns_arr < 0]
        if len(negative_returns) > 0:
            metrics.downside_volatility = float(
                np.std(negative_returns) * np.sqrt(self.config.trading_days_per_year)
            )

        equity = np.cumprod(1 + returns_arr)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max

        metrics.max_drawdown = float(np.min(drawdowns))
        metrics.avg_drawdown = float(np.mean(drawdowns[drawdowns < 0])) if any(drawdowns < 0) else 0

        in_drawdown = drawdowns < 0
        max_duration = 0
        current_duration = 0
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        metrics.max_drawdown_duration_days = max_duration

        metrics.ulcer_index = float(np.sqrt(np.mean(drawdowns ** 2)))

        metrics.var_95 = float(np.percentile(returns_arr, 5))
        metrics.var_99 = float(np.percentile(returns_arr, 1))

        var_95_threshold = np.percentile(returns_arr, 5)
        tail_returns = returns_arr[returns_arr <= var_95_threshold]
        metrics.cvar_95 = float(np.mean(tail_returns)) if len(tail_returns) > 0 else metrics.var_95

        var_99_threshold = np.percentile(returns_arr, 1)
        tail_returns_99 = returns_arr[returns_arr <= var_99_threshold]
        metrics.cvar_99 = float(np.mean(tail_returns_99)) if len(tail_returns_99) > 0 else metrics.var_99

        if len(returns_arr) >= 3:
            mean = np.mean(returns_arr)
            std = np.std(returns_arr)
            if std > 0:
                metrics.skewness = float(np.mean(((returns_arr - mean) / std) ** 3))

        if len(returns_arr) >= 4:
            mean = np.mean(returns_arr)
            std = np.std(returns_arr)
            if std > 0:
                metrics.kurtosis = float(np.mean(((returns_arr - mean) / std) ** 4) - 3)

        p95 = np.percentile(returns_arr, 95)
        p5 = abs(np.percentile(returns_arr, 5))
        metrics.tail_ratio = float(p95 / p5) if p5 > 0 else 0

        return metrics

    def _calculate_ratio_metrics(
        self,
        returns: list[float],
        return_metrics: ReturnMetrics,
        risk_metrics: RiskMetrics,
    ) -> RatioMetrics:
        """Calculate ratio metrics."""
        metrics = RatioMetrics()

        if not returns:
            return metrics

        daily_rf = self.config.risk_free_rate / self.config.trading_days_per_year
        excess_returns = [r - daily_rf for r in returns]

        if risk_metrics.volatility_annual > 0:
            metrics.sharpe_ratio = float(
                (return_metrics.annualized_return - self.config.risk_free_rate) /
                risk_metrics.volatility_annual
            )

        if risk_metrics.downside_volatility > 0:
            metrics.sortino_ratio = float(
                (return_metrics.annualized_return - self.config.risk_free_rate) /
                risk_metrics.downside_volatility
            )

        if risk_metrics.max_drawdown != 0:
            metrics.calmar_ratio = float(
                return_metrics.annualized_return / abs(risk_metrics.max_drawdown)
            )

        threshold = 0
        gains = sum(r - threshold for r in returns if r > threshold)
        losses = abs(sum(r - threshold for r in returns if r < threshold))
        metrics.omega_ratio = float(gains / losses) if losses > 0 else float("inf")

        if risk_metrics.max_drawdown != 0:
            metrics.recovery_factor = float(
                return_metrics.total_return_pct / abs(risk_metrics.max_drawdown)
            )

        return metrics

    def _calculate_trade_metrics(
        self,
        trades: list[BacktestTrade],
    ) -> TradeMetrics:
        """Calculate trade metrics."""
        metrics = TradeMetrics()

        closed_trades = [t for t in trades if t.is_closed]

        if not closed_trades:
            return metrics

        metrics.total_trades = len(closed_trades)

        winning = [t for t in closed_trades if t.pnl > 0]
        losing = [t for t in closed_trades if t.pnl < 0]

        metrics.winning_trades = len(winning)
        metrics.losing_trades = len(losing)
        metrics.win_rate = len(winning) / len(closed_trades)

        pnls = [t.pnl for t in closed_trades]
        metrics.avg_trade_pnl = float(np.mean(pnls))

        if winning:
            metrics.avg_win = float(np.mean([t.pnl for t in winning]))
            metrics.largest_win = float(max(t.pnl for t in winning))

        if losing:
            metrics.avg_loss = float(np.mean([t.pnl for t in losing]))
            metrics.largest_loss = float(min(t.pnl for t in losing))

        total_wins = sum(t.pnl for t in winning)
        total_losses = abs(sum(t.pnl for t in losing))
        if total_losses > 0:
            metrics.expectancy = (total_wins - total_losses) / len(closed_trades)

        holding_periods = [t.holding_period_days for t in closed_trades]
        metrics.avg_holding_period_days = float(np.mean(holding_periods))

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in closed_trades:
            if trade.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        metrics.max_consecutive_wins = max_wins
        metrics.max_consecutive_losses = max_losses

        if metrics.avg_loss != 0:
            metrics.payoff_ratio = abs(metrics.avg_win / metrics.avg_loss)

        return metrics

    def _calculate_benchmark_metrics(
        self,
        returns: list[float],
        benchmark_returns: list[float],
    ) -> BenchmarkMetrics:
        """Calculate benchmark comparison metrics."""
        metrics = BenchmarkMetrics()

        min_len = min(len(returns), len(benchmark_returns))
        if min_len < 2:
            return metrics

        returns_arr = np.array(returns[:min_len])
        bench_arr = np.array(benchmark_returns[:min_len])

        metrics.benchmark_return = float(np.prod(1 + bench_arr) - 1)

        strategy_return = float(np.prod(1 + returns_arr) - 1)
        metrics.excess_return = strategy_return - metrics.benchmark_return

        metrics.correlation = float(np.corrcoef(returns_arr, bench_arr)[0, 1])

        bench_var = np.var(bench_arr)
        if bench_var > 0:
            covariance = np.cov(returns_arr, bench_arr)[0, 1]
            metrics.beta = float(covariance / bench_var)

        daily_rf = self.config.risk_free_rate / self.config.trading_days_per_year
        strategy_excess = np.mean(returns_arr) - daily_rf
        bench_excess = np.mean(bench_arr) - daily_rf
        metrics.alpha = float(
            (strategy_excess - metrics.beta * bench_excess) *
            self.config.trading_days_per_year
        )

        metrics.r_squared = metrics.correlation ** 2

        tracking_diff = returns_arr - bench_arr
        metrics.tracking_error = float(
            np.std(tracking_diff) * np.sqrt(self.config.trading_days_per_year)
        )

        if metrics.tracking_error > 0:
            metrics.information_ratio = float(
                metrics.excess_return / metrics.tracking_error
            )

        up_periods = bench_arr > 0
        down_periods = bench_arr < 0

        if np.sum(up_periods) > 0:
            strategy_up = np.mean(returns_arr[up_periods])
            bench_up = np.mean(bench_arr[up_periods])
            metrics.up_capture = float(strategy_up / bench_up) if bench_up > 0 else 0

        if np.sum(down_periods) > 0:
            strategy_down = np.mean(returns_arr[down_periods])
            bench_down = np.mean(bench_arr[down_periods])
            metrics.down_capture = float(strategy_down / bench_down) if bench_down != 0 else 0

        return metrics

    def _calculate_monthly_returns(
        self,
        portfolio_history: list[PortfolioSnapshot],
    ) -> list[tuple[str, float]]:
        """Calculate monthly returns."""
        if not portfolio_history:
            return []

        monthly: dict[str, float] = {}

        for i, snapshot in enumerate(portfolio_history):
            month_key = snapshot.timestamp.strftime("%Y-%m")

            if month_key not in monthly:
                if i > 0:
                    monthly[month_key] = portfolio_history[i - 1].equity
                else:
                    monthly[month_key] = snapshot.equity

        monthly_returns: list[tuple[str, float]] = []
        months = sorted(monthly.keys())

        for i in range(1, len(months)):
            prev_equity = monthly[months[i - 1]]
            curr_equity = monthly[months[i]]

            if prev_equity > 0:
                monthly_return = (curr_equity - prev_equity) / prev_equity
                monthly_returns.append((months[i], monthly_return))

        return monthly_returns

    def _calculate_rolling_sharpe(
        self,
        returns: list[float],
        portfolio_history: list[PortfolioSnapshot],
    ) -> list[tuple[datetime, float]]:
        """Calculate rolling Sharpe ratio."""
        window = self.config.rolling_window_days
        if len(returns) < window:
            return []

        rolling_sharpe: list[tuple[datetime, float]] = []
        daily_rf = self.config.risk_free_rate / self.config.trading_days_per_year

        for i in range(window, len(returns)):
            window_returns = returns[i - window:i]
            excess = [r - daily_rf for r in window_returns]

            mean_excess = np.mean(excess)
            std_excess = np.std(excess)

            sharpe = (mean_excess * np.sqrt(self.config.trading_days_per_year) /
                     std_excess) if std_excess > 0 else 0

            if i < len(portfolio_history):
                timestamp = portfolio_history[i].timestamp
                rolling_sharpe.append((timestamp, float(sharpe)))

        return rolling_sharpe

    def compare_results(
        self,
        results: list[BacktestResult],
        names: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Compare multiple backtest results.

        Args:
            results: List of backtest results
            names: Optional names for each result

        Returns:
            Comparison dictionary
        """
        if not names:
            names = [f"Strategy_{i + 1}" for i in range(len(results))]

        comparison: dict[str, dict[str, Any]] = {}

        for result, name in zip(results, names):
            report = self.analyze(result, strategy_name=name)

            comparison[name] = {
                "total_return_pct": report.return_metrics.total_return_pct,
                "annualized_return": report.return_metrics.annualized_return,
                "sharpe_ratio": report.ratio_metrics.sharpe_ratio,
                "sortino_ratio": report.ratio_metrics.sortino_ratio,
                "max_drawdown": report.risk_metrics.max_drawdown,
                "win_rate": report.trade_metrics.win_rate,
                "total_trades": report.trade_metrics.total_trades,
                "profit_factor": report.trade_metrics.expectancy,
            }

        rankings: dict[str, dict[str, int]] = {}
        metrics_to_rank = [
            ("total_return_pct", True),
            ("sharpe_ratio", True),
            ("max_drawdown", False),
            ("win_rate", True),
        ]

        for metric, higher_is_better in metrics_to_rank:
            values = [(name, comparison[name][metric]) for name in names]
            sorted_values = sorted(values, key=lambda x: x[1], reverse=higher_is_better)

            for rank, (name, _) in enumerate(sorted_values, 1):
                if name not in rankings:
                    rankings[name] = {}
                rankings[name][metric] = rank

        return {
            "comparison": comparison,
            "rankings": rankings,
            "best_by_metric": {
                "return": max(names, key=lambda n: comparison[n]["total_return_pct"]),
                "sharpe": max(names, key=lambda n: comparison[n]["sharpe_ratio"]),
                "drawdown": min(names, key=lambda n: abs(comparison[n]["max_drawdown"])),
            },
        }

    def generate_report_summary(
        self,
        report: PerformanceReport,
    ) -> str:
        """
        Generate text summary of performance report.

        Args:
            report: Performance report

        Returns:
            Text summary
        """
        lines = [
            "=" * 60,
            f"Performance Report: {report.strategy_name or 'Unnamed Strategy'}",
            "=" * 60,
            f"Period: {report.period_start} to {report.period_end}",
            f"Trading Days: {report.trading_days}",
            "",
            "RETURNS",
            "-" * 40,
            f"  Total Return: {report.return_metrics.total_return_pct:.2%}",
            f"  Annualized Return: {report.return_metrics.annualized_return:.2%}",
            f"  CAGR: {report.return_metrics.cagr:.2%}",
            "",
            "RISK",
            "-" * 40,
            f"  Volatility: {report.risk_metrics.volatility_annual:.2%}",
            f"  Max Drawdown: {report.risk_metrics.max_drawdown:.2%}",
            f"  Max DD Duration: {report.risk_metrics.max_drawdown_duration_days} days",
            f"  VaR (95%): {report.risk_metrics.var_95:.2%}",
            "",
            "RATIOS",
            "-" * 40,
            f"  Sharpe Ratio: {report.ratio_metrics.sharpe_ratio:.2f}",
            f"  Sortino Ratio: {report.ratio_metrics.sortino_ratio:.2f}",
            f"  Calmar Ratio: {report.ratio_metrics.calmar_ratio:.2f}",
            "",
            "TRADES",
            "-" * 40,
            f"  Total Trades: {report.trade_metrics.total_trades}",
            f"  Win Rate: {report.trade_metrics.win_rate:.2%}",
            f"  Avg Win: ${report.trade_metrics.avg_win:,.2f}",
            f"  Avg Loss: ${report.trade_metrics.avg_loss:,.2f}",
            f"  Expectancy: ${report.trade_metrics.expectancy:,.2f}",
            "",
            "=" * 60,
        ]

        return "\n".join(lines)
