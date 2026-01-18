"""
Advanced Risk Metrics for Ultimate Trading Bot v2.2.

This module provides comprehensive risk metrics calculation including
Sharpe ratio, Sortino ratio, Calmar ratio, and other performance metrics.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from src.risk.base_risk import RiskLevel


logger = logging.getLogger(__name__)


class MetricPeriod(str, Enum):
    """Time periods for metrics calculation."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    ALL_TIME = "all_time"


class RiskMetricsConfig(BaseModel):
    """Configuration for risk metrics calculation."""

    model_config = {"arbitrary_types_allowed": True}

    risk_free_rate: float = Field(default=0.05, description="Annual risk-free rate")
    trading_days_per_year: int = Field(default=252, description="Trading days per year")
    min_data_points: int = Field(default=30, description="Minimum data points for calculations")
    mar: float = Field(default=0.0, description="Minimum acceptable return for Sortino")
    confidence_level: float = Field(default=0.95, description="Confidence level for VaR/CVaR")
    rolling_window_days: int = Field(default=60, description="Rolling window for metrics")
    benchmark_symbol: str | None = Field(default="SPY", description="Benchmark symbol")
    use_exponential_weighting: bool = Field(default=False, description="Use EWMA for volatility")
    ewma_lambda: float = Field(default=0.94, description="EWMA decay factor")


class PerformanceMetrics(BaseModel):
    """Performance metrics for a portfolio or strategy."""

    period: MetricPeriod = Field(default=MetricPeriod.ALL_TIME)
    start_date: datetime | None = None
    end_date: datetime | None = None
    data_points: int = Field(default=0)

    total_return: float = Field(default=0.0, description="Total return")
    annualized_return: float = Field(default=0.0, description="Annualized return")
    cagr: float = Field(default=0.0, description="Compound annual growth rate")

    volatility: float = Field(default=0.0, description="Annualized volatility")
    downside_volatility: float = Field(default=0.0, description="Downside deviation")

    sharpe_ratio: float = Field(default=0.0, description="Sharpe ratio")
    sortino_ratio: float = Field(default=0.0, description="Sortino ratio")
    calmar_ratio: float = Field(default=0.0, description="Calmar ratio")
    omega_ratio: float = Field(default=0.0, description="Omega ratio")
    treynor_ratio: float = Field(default=0.0, description="Treynor ratio")
    information_ratio: float = Field(default=0.0, description="Information ratio")

    max_drawdown: float = Field(default=0.0, description="Maximum drawdown")
    avg_drawdown: float = Field(default=0.0, description="Average drawdown")
    max_drawdown_duration_days: int = Field(default=0, description="Max drawdown duration")
    recovery_factor: float = Field(default=0.0, description="Recovery factor")
    ulcer_index: float = Field(default=0.0, description="Ulcer index")

    var_95: float = Field(default=0.0, description="95% VaR")
    var_99: float = Field(default=0.0, description="99% VaR")
    cvar_95: float = Field(default=0.0, description="95% CVaR/ES")
    cvar_99: float = Field(default=0.0, description="99% CVaR/ES")

    beta: float = Field(default=0.0, description="Beta to benchmark")
    alpha: float = Field(default=0.0, description="Alpha (Jensen's)")
    r_squared: float = Field(default=0.0, description="R-squared")
    correlation: float = Field(default=0.0, description="Correlation to benchmark")

    win_rate: float = Field(default=0.0, description="Win rate")
    profit_factor: float = Field(default=0.0, description="Profit factor")
    avg_win: float = Field(default=0.0, description="Average win")
    avg_loss: float = Field(default=0.0, description="Average loss")
    win_loss_ratio: float = Field(default=0.0, description="Win/loss ratio")
    expectancy: float = Field(default=0.0, description="Expectancy per trade")

    skewness: float = Field(default=0.0, description="Return skewness")
    kurtosis: float = Field(default=0.0, description="Return kurtosis")

    tail_ratio: float = Field(default=0.0, description="Tail ratio")
    common_sense_ratio: float = Field(default=0.0, description="Common sense ratio")


@dataclass
class DrawdownInfo:
    """Drawdown information."""

    start_date: datetime
    end_date: datetime | None = None
    recovery_date: datetime | None = None
    peak_value: float = 0.0
    trough_value: float = 0.0
    drawdown_pct: float = 0.0
    duration_days: int = 0
    recovery_days: int | None = None
    is_recovered: bool = False


@dataclass
class RollingMetrics:
    """Rolling window metrics."""

    timestamps: list[datetime] = field(default_factory=list)
    sharpe_ratios: list[float] = field(default_factory=list)
    volatilities: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    max_drawdowns: list[float] = field(default_factory=list)
    betas: list[float] = field(default_factory=list)


class RiskMetricsCalculator:
    """
    Calculates comprehensive risk and performance metrics.

    Provides detailed analytics for portfolio and strategy evaluation.
    """

    def __init__(self, config: RiskMetricsConfig | None = None):
        """
        Initialize risk metrics calculator.

        Args:
            config: Metrics configuration
        """
        self.config = config or RiskMetricsConfig()
        self._returns_history: list[tuple[datetime, float]] = []
        self._benchmark_returns: list[tuple[datetime, float]] = []
        self._equity_curve: list[tuple[datetime, float]] = []
        self._trades: list[dict[str, Any]] = []
        self._drawdowns: list[DrawdownInfo] = []
        self._rolling_metrics = RollingMetrics()
        self._lock = asyncio.Lock()

        logger.info("RiskMetricsCalculator initialized")

    async def calculate_metrics(
        self,
        returns: list[float] | None = None,
        benchmark_returns: list[float] | None = None,
        period: MetricPeriod = MetricPeriod.ALL_TIME,
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.

        Args:
            returns: List of period returns (optional, uses stored if None)
            benchmark_returns: Benchmark returns for relative metrics
            period: Time period for calculations

        Returns:
            PerformanceMetrics object
        """
        try:
            if returns is None:
                returns = [r for _, r in self._returns_history]

            if len(returns) < self.config.min_data_points:
                logger.warning(
                    f"Insufficient data: {len(returns)} points, "
                    f"need {self.config.min_data_points}"
                )
                return PerformanceMetrics(
                    period=period,
                    data_points=len(returns),
                )

            returns_arr = np.array(returns)

            if benchmark_returns is None:
                benchmark_returns = [r for _, r in self._benchmark_returns]
            bench_arr = np.array(benchmark_returns) if benchmark_returns else None

            total_return = np.prod(1 + returns_arr) - 1
            years = len(returns) / self.config.trading_days_per_year
            annualized_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1 if years > 0 else 0
            cagr = annualized_return

            volatility = np.std(returns_arr) * np.sqrt(self.config.trading_days_per_year)

            negative_returns = returns_arr[returns_arr < self.config.mar]
            if len(negative_returns) > 0:
                downside_vol = np.std(negative_returns) * np.sqrt(self.config.trading_days_per_year)
            else:
                downside_vol = 0.0

            daily_rf = self.config.risk_free_rate / self.config.trading_days_per_year
            excess_returns = returns_arr - daily_rf

            sharpe = 0.0
            if volatility > 0:
                sharpe = (annualized_return - self.config.risk_free_rate) / volatility

            sortino = 0.0
            if downside_vol > 0:
                sortino = (annualized_return - self.config.risk_free_rate) / downside_vol

            max_dd, avg_dd, max_dd_duration = self._calculate_drawdowns(returns_arr)

            calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0.0

            omega = self._calculate_omega_ratio(returns_arr)

            recovery_factor = total_return / abs(max_dd) if max_dd != 0 else 0.0

            ulcer = self._calculate_ulcer_index(returns_arr)

            var_95 = self._calculate_var(returns_arr, 0.95)
            var_99 = self._calculate_var(returns_arr, 0.99)
            cvar_95 = self._calculate_cvar(returns_arr, 0.95)
            cvar_99 = self._calculate_cvar(returns_arr, 0.99)

            beta, alpha, r_squared, correlation, treynor, info_ratio = (
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            )
            if bench_arr is not None and len(bench_arr) == len(returns_arr):
                beta, alpha, r_squared, correlation = self._calculate_beta_alpha(
                    returns_arr, bench_arr
                )
                if beta != 0:
                    treynor = (annualized_return - self.config.risk_free_rate) / beta

                tracking_error = np.std(returns_arr - bench_arr) * np.sqrt(
                    self.config.trading_days_per_year
                )
                if tracking_error > 0:
                    info_ratio = (annualized_return - np.mean(bench_arr) * self.config.trading_days_per_year) / tracking_error

            win_rate, profit_factor, avg_win, avg_loss, wl_ratio, expectancy = (
                self._calculate_trade_metrics(returns_arr)
            )

            skewness = float(self._calculate_skewness(returns_arr))
            kurtosis = float(self._calculate_kurtosis(returns_arr))

            tail_ratio = self._calculate_tail_ratio(returns_arr)
            common_sense = tail_ratio * profit_factor if profit_factor > 0 else 0.0

            metrics = PerformanceMetrics(
                period=period,
                data_points=len(returns),
                total_return=total_return,
                annualized_return=annualized_return,
                cagr=cagr,
                volatility=volatility,
                downside_volatility=downside_vol,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                calmar_ratio=calmar,
                omega_ratio=omega,
                treynor_ratio=treynor,
                information_ratio=info_ratio,
                max_drawdown=max_dd,
                avg_drawdown=avg_dd,
                max_drawdown_duration_days=max_dd_duration,
                recovery_factor=recovery_factor,
                ulcer_index=ulcer,
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                beta=beta,
                alpha=alpha,
                r_squared=r_squared,
                correlation=correlation,
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_win=avg_win,
                avg_loss=avg_loss,
                win_loss_ratio=wl_ratio,
                expectancy=expectancy,
                skewness=skewness,
                kurtosis=kurtosis,
                tail_ratio=tail_ratio,
                common_sense_ratio=common_sense,
            )

            return metrics

        except Exception as e:
            logger.error(f"Failed to calculate metrics: {e}")
            return PerformanceMetrics(period=period)

    def _calculate_drawdowns(
        self,
        returns: np.ndarray,
    ) -> tuple[float, float, int]:
        """Calculate drawdown statistics."""
        equity = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max

        max_dd = float(np.min(drawdowns))

        dd_periods = drawdowns[drawdowns < 0]
        avg_dd = float(np.mean(dd_periods)) if len(dd_periods) > 0 else 0.0

        in_drawdown = drawdowns < 0
        max_duration = 0
        current_duration = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_dd, avg_dd, max_duration

    def _calculate_omega_ratio(
        self,
        returns: np.ndarray,
        threshold: float = 0.0,
    ) -> float:
        """Calculate Omega ratio."""
        excess = returns - threshold
        gains = excess[excess > 0].sum()
        losses = abs(excess[excess < 0].sum())

        return gains / losses if losses > 0 else float("inf") if gains > 0 else 0.0

    def _calculate_ulcer_index(self, returns: np.ndarray) -> float:
        """Calculate Ulcer Index."""
        equity = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max

        squared_dd = drawdowns ** 2
        return float(np.sqrt(np.mean(squared_dd)))

    def _calculate_var(
        self,
        returns: np.ndarray,
        confidence: float,
    ) -> float:
        """Calculate Value at Risk."""
        return float(np.percentile(returns, (1 - confidence) * 100))

    def _calculate_cvar(
        self,
        returns: np.ndarray,
        confidence: float,
    ) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        var = self._calculate_var(returns, confidence)
        tail_returns = returns[returns <= var]
        return float(np.mean(tail_returns)) if len(tail_returns) > 0 else var

    def _calculate_beta_alpha(
        self,
        returns: np.ndarray,
        benchmark: np.ndarray,
    ) -> tuple[float, float, float, float]:
        """Calculate beta, alpha, R-squared, and correlation."""
        if len(returns) != len(benchmark) or len(returns) < 2:
            return 0.0, 0.0, 0.0, 0.0

        correlation = float(np.corrcoef(returns, benchmark)[0, 1])
        if np.isnan(correlation):
            correlation = 0.0

        bench_var = np.var(benchmark)
        if bench_var > 0:
            covariance = np.cov(returns, benchmark)[0, 1]
            beta = float(covariance / bench_var)
        else:
            beta = 0.0

        daily_rf = self.config.risk_free_rate / self.config.trading_days_per_year
        alpha = float(np.mean(returns) - daily_rf - beta * (np.mean(benchmark) - daily_rf))
        alpha *= self.config.trading_days_per_year

        r_squared = correlation ** 2

        return beta, alpha, r_squared, correlation

    def _calculate_trade_metrics(
        self,
        returns: np.ndarray,
    ) -> tuple[float, float, float, float, float, float]:
        """Calculate trade-level metrics."""
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        win_rate = len(wins) / len(returns) if len(returns) > 0 else 0.0

        total_wins = np.sum(wins) if len(wins) > 0 else 0.0
        total_losses = abs(np.sum(losses)) if len(losses) > 0 else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else float("inf") if total_wins > 0 else 0.0

        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

        wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf") if avg_win > 0 else 0.0

        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

        return win_rate, profit_factor, avg_win, avg_loss, wl_ratio, expectancy

    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns."""
        n = len(returns)
        if n < 3:
            return 0.0

        mean = np.mean(returns)
        std = np.std(returns)

        if std == 0:
            return 0.0

        return float(np.mean(((returns - mean) / std) ** 3))

    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate excess kurtosis of returns."""
        n = len(returns)
        if n < 4:
            return 0.0

        mean = np.mean(returns)
        std = np.std(returns)

        if std == 0:
            return 0.0

        return float(np.mean(((returns - mean) / std) ** 4) - 3)

    def _calculate_tail_ratio(self, returns: np.ndarray) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)."""
        p95 = np.percentile(returns, 95)
        p5 = abs(np.percentile(returns, 5))

        return p95 / p5 if p5 > 0 else float("inf") if p95 > 0 else 0.0

    async def add_return(
        self,
        return_value: float,
        timestamp: datetime | None = None,
    ) -> None:
        """
        Add a return observation.

        Args:
            return_value: Period return
            timestamp: Timestamp of return
        """
        async with self._lock:
            ts = timestamp or datetime.now()
            self._returns_history.append((ts, return_value))

    async def add_benchmark_return(
        self,
        return_value: float,
        timestamp: datetime | None = None,
    ) -> None:
        """
        Add a benchmark return observation.

        Args:
            return_value: Benchmark period return
            timestamp: Timestamp of return
        """
        async with self._lock:
            ts = timestamp or datetime.now()
            self._benchmark_returns.append((ts, return_value))

    async def add_equity_point(
        self,
        equity: float,
        timestamp: datetime | None = None,
    ) -> None:
        """
        Add an equity curve point.

        Args:
            equity: Portfolio equity value
            timestamp: Timestamp
        """
        async with self._lock:
            ts = timestamp or datetime.now()
            self._equity_curve.append((ts, equity))

            if len(self._equity_curve) >= 2:
                prev_equity = self._equity_curve[-2][1]
                if prev_equity > 0:
                    ret = (equity - prev_equity) / prev_equity
                    self._returns_history.append((ts, ret))

    async def calculate_rolling_metrics(
        self,
        window_days: int | None = None,
    ) -> RollingMetrics:
        """
        Calculate rolling window metrics.

        Args:
            window_days: Rolling window size

        Returns:
            RollingMetrics object
        """
        window = window_days or self.config.rolling_window_days

        if len(self._returns_history) < window:
            return self._rolling_metrics

        returns = [r for _, r in self._returns_history]
        timestamps = [t for t, _ in self._returns_history]

        rolling = RollingMetrics()

        for i in range(window, len(returns) + 1):
            window_returns = np.array(returns[i - window:i])
            ts = timestamps[i - 1]

            rolling.timestamps.append(ts)
            rolling.returns.append(float(np.mean(window_returns)))

            vol = float(np.std(window_returns) * np.sqrt(self.config.trading_days_per_year))
            rolling.volatilities.append(vol)

            ann_ret = float(np.mean(window_returns) * self.config.trading_days_per_year)
            sharpe = (ann_ret - self.config.risk_free_rate) / vol if vol > 0 else 0.0
            rolling.sharpe_ratios.append(sharpe)

            max_dd, _, _ = self._calculate_drawdowns(window_returns)
            rolling.max_drawdowns.append(max_dd)

        self._rolling_metrics = rolling
        return rolling

    async def get_period_metrics(
        self,
        period: MetricPeriod,
    ) -> PerformanceMetrics:
        """
        Get metrics for a specific period.

        Args:
            period: Time period

        Returns:
            PerformanceMetrics for the period
        """
        now = datetime.now()

        if period == MetricPeriod.DAILY:
            cutoff = now - timedelta(days=1)
        elif period == MetricPeriod.WEEKLY:
            cutoff = now - timedelta(weeks=1)
        elif period == MetricPeriod.MONTHLY:
            cutoff = now - timedelta(days=30)
        elif period == MetricPeriod.QUARTERLY:
            cutoff = now - timedelta(days=90)
        elif period == MetricPeriod.YEARLY:
            cutoff = now - timedelta(days=365)
        else:
            cutoff = datetime.min

        filtered_returns = [r for t, r in self._returns_history if t >= cutoff]
        filtered_benchmark = [r for t, r in self._benchmark_returns if t >= cutoff]

        return await self.calculate_metrics(
            returns=filtered_returns,
            benchmark_returns=filtered_benchmark if filtered_benchmark else None,
            period=period,
        )

    async def get_risk_assessment(self) -> dict[str, Any]:
        """
        Get overall risk assessment.

        Returns:
            Risk assessment dictionary
        """
        metrics = await self.calculate_metrics()

        risk_score = 50.0

        if metrics.volatility > 0.3:
            risk_score += 20
        elif metrics.volatility > 0.2:
            risk_score += 10
        elif metrics.volatility < 0.1:
            risk_score -= 10

        if metrics.max_drawdown < -0.2:
            risk_score += 20
        elif metrics.max_drawdown < -0.1:
            risk_score += 10
        elif metrics.max_drawdown > -0.05:
            risk_score -= 10

        if abs(metrics.var_95) > 0.03:
            risk_score += 10
        elif abs(metrics.var_95) < 0.01:
            risk_score -= 10

        if metrics.beta > 1.5:
            risk_score += 10
        elif metrics.beta < 0.5:
            risk_score -= 10

        if metrics.kurtosis > 3:
            risk_score += 10

        if metrics.skewness < -1:
            risk_score += 10
        elif metrics.skewness > 0.5:
            risk_score -= 5

        risk_score = max(0, min(100, risk_score))

        if risk_score >= 80:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= 60:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 40:
            risk_level = RiskLevel.MEDIUM
        elif risk_score >= 20:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.MINIMAL

        return {
            "risk_score": risk_score,
            "risk_level": risk_level.value,
            "key_metrics": {
                "sharpe_ratio": metrics.sharpe_ratio,
                "volatility": metrics.volatility,
                "max_drawdown": metrics.max_drawdown,
                "var_95": metrics.var_95,
                "beta": metrics.beta,
            },
            "concerns": self._identify_concerns(metrics),
            "strengths": self._identify_strengths(metrics),
        }

    def _identify_concerns(self, metrics: PerformanceMetrics) -> list[str]:
        """Identify risk concerns."""
        concerns: list[str] = []

        if metrics.volatility > 0.25:
            concerns.append(f"High volatility: {metrics.volatility:.1%}")

        if metrics.max_drawdown < -0.15:
            concerns.append(f"Significant drawdown: {metrics.max_drawdown:.1%}")

        if metrics.sharpe_ratio < 0.5:
            concerns.append(f"Low risk-adjusted returns: Sharpe {metrics.sharpe_ratio:.2f}")

        if metrics.skewness < -0.5:
            concerns.append(f"Negative skew: {metrics.skewness:.2f}")

        if metrics.kurtosis > 3:
            concerns.append(f"Fat tails: Kurtosis {metrics.kurtosis:.2f}")

        if metrics.max_drawdown_duration_days > 60:
            concerns.append(f"Long drawdown duration: {metrics.max_drawdown_duration_days} days")

        return concerns

    def _identify_strengths(self, metrics: PerformanceMetrics) -> list[str]:
        """Identify portfolio strengths."""
        strengths: list[str] = []

        if metrics.sharpe_ratio > 1.5:
            strengths.append(f"Excellent risk-adjusted returns: Sharpe {metrics.sharpe_ratio:.2f}")
        elif metrics.sharpe_ratio > 1.0:
            strengths.append(f"Good risk-adjusted returns: Sharpe {metrics.sharpe_ratio:.2f}")

        if metrics.sortino_ratio > metrics.sharpe_ratio * 1.3:
            strengths.append(f"Low downside risk: Sortino {metrics.sortino_ratio:.2f}")

        if metrics.win_rate > 0.6:
            strengths.append(f"High win rate: {metrics.win_rate:.1%}")

        if metrics.profit_factor > 2:
            strengths.append(f"Strong profit factor: {metrics.profit_factor:.2f}")

        if metrics.alpha > 0.05:
            strengths.append(f"Positive alpha: {metrics.alpha:.1%}")

        if metrics.skewness > 0.3:
            strengths.append(f"Positive skew: {metrics.skewness:.2f}")

        return strengths

    def clear_history(self) -> None:
        """Clear all historical data."""
        self._returns_history.clear()
        self._benchmark_returns.clear()
        self._equity_curve.clear()
        self._trades.clear()
        self._drawdowns.clear()
        self._rolling_metrics = RollingMetrics()
        logger.info("Metrics history cleared")

    async def get_metrics_summary(self) -> dict[str, Any]:
        """
        Get comprehensive metrics summary.

        Returns:
            Summary dictionary
        """
        all_time = await self.calculate_metrics()
        monthly = await self.get_period_metrics(MetricPeriod.MONTHLY)
        yearly = await self.get_period_metrics(MetricPeriod.YEARLY)

        return {
            "timestamp": datetime.now().isoformat(),
            "data_points": len(self._returns_history),
            "all_time": {
                "total_return": all_time.total_return,
                "cagr": all_time.cagr,
                "sharpe_ratio": all_time.sharpe_ratio,
                "sortino_ratio": all_time.sortino_ratio,
                "max_drawdown": all_time.max_drawdown,
                "volatility": all_time.volatility,
            },
            "monthly": {
                "return": monthly.total_return,
                "sharpe_ratio": monthly.sharpe_ratio,
                "max_drawdown": monthly.max_drawdown,
            },
            "yearly": {
                "return": yearly.total_return,
                "sharpe_ratio": yearly.sharpe_ratio,
                "max_drawdown": yearly.max_drawdown,
            },
            "risk_metrics": {
                "var_95": all_time.var_95,
                "cvar_95": all_time.cvar_95,
                "beta": all_time.beta,
                "alpha": all_time.alpha,
            },
            "trade_metrics": {
                "win_rate": all_time.win_rate,
                "profit_factor": all_time.profit_factor,
                "expectancy": all_time.expectancy,
            },
        }
