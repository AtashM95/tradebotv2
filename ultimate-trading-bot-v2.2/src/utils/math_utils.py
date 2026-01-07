"""
Math Utilities Module for Ultimate Trading Bot v2.2.

This module provides mathematical and statistical functions for trading analysis.
"""

import math
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Optional, Sequence, Tuple, Union
import statistics
import logging


logger = logging.getLogger(__name__)

Number = Union[int, float, Decimal]


# =============================================================================
# BASIC MATH OPERATIONS
# =============================================================================

def safe_divide(
    numerator: Number,
    denominator: Number,
    default: Number = 0.0
) -> float:
    """
    Safely divide two numbers, returning default on zero division.

    Args:
        numerator: The numerator
        denominator: The denominator
        default: Value to return if denominator is zero

    Returns:
        Result of division or default
    """
    if denominator == 0:
        return float(default)
    return float(numerator) / float(denominator)


def percentage_change(
    old_value: Number,
    new_value: Number,
    default: Number = 0.0
) -> float:
    """
    Calculate percentage change between two values.

    Args:
        old_value: Original value
        new_value: New value
        default: Default if old_value is zero

    Returns:
        Percentage change as decimal (0.1 = 10%)
    """
    if old_value == 0:
        return float(default)
    return (float(new_value) - float(old_value)) / float(old_value)


def percentage_difference(value1: Number, value2: Number) -> float:
    """
    Calculate percentage difference between two values.

    Args:
        value1: First value
        value2: Second value

    Returns:
        Percentage difference
    """
    avg = (abs(float(value1)) + abs(float(value2))) / 2
    if avg == 0:
        return 0.0
    return abs(float(value1) - float(value2)) / avg


def compound_return(returns: Sequence[Number]) -> float:
    """
    Calculate compound return from a series of returns.

    Args:
        returns: Series of returns as decimals

    Returns:
        Compound return
    """
    if not returns:
        return 0.0

    product = 1.0
    for r in returns:
        product *= (1.0 + float(r))
    return product - 1.0


def annualized_return(total_return: Number, periods: int, periods_per_year: int = 252) -> float:
    """
    Calculate annualized return.

    Args:
        total_return: Total return as decimal
        periods: Number of periods
        periods_per_year: Periods per year (252 for daily)

    Returns:
        Annualized return
    """
    if periods <= 0:
        return 0.0

    years = periods / periods_per_year
    return (1.0 + float(total_return)) ** (1.0 / years) - 1.0


def log_return(start_price: Number, end_price: Number) -> float:
    """
    Calculate logarithmic return.

    Args:
        start_price: Starting price
        end_price: Ending price

    Returns:
        Log return
    """
    if float(start_price) <= 0 or float(end_price) <= 0:
        return 0.0
    return math.log(float(end_price) / float(start_price))


def simple_return(start_price: Number, end_price: Number) -> float:
    """
    Calculate simple return.

    Args:
        start_price: Starting price
        end_price: Ending price

    Returns:
        Simple return
    """
    if float(start_price) == 0:
        return 0.0
    return (float(end_price) - float(start_price)) / float(start_price)


# =============================================================================
# STATISTICAL FUNCTIONS
# =============================================================================

def mean(values: Sequence[Number]) -> float:
    """
    Calculate arithmetic mean.

    Args:
        values: Sequence of values

    Returns:
        Mean value
    """
    if not values:
        return 0.0
    return statistics.mean(float(v) for v in values)


def weighted_mean(values: Sequence[Number], weights: Sequence[Number]) -> float:
    """
    Calculate weighted mean.

    Args:
        values: Sequence of values
        weights: Sequence of weights

    Returns:
        Weighted mean
    """
    if not values or not weights or len(values) != len(weights):
        return 0.0

    total_weight = sum(float(w) for w in weights)
    if total_weight == 0:
        return 0.0

    weighted_sum = sum(float(v) * float(w) for v, w in zip(values, weights))
    return weighted_sum / total_weight


def median(values: Sequence[Number]) -> float:
    """
    Calculate median.

    Args:
        values: Sequence of values

    Returns:
        Median value
    """
    if not values:
        return 0.0
    return statistics.median(float(v) for v in values)


def mode(values: Sequence[Number]) -> float:
    """
    Calculate mode (most frequent value).

    Args:
        values: Sequence of values

    Returns:
        Mode value
    """
    if not values:
        return 0.0
    try:
        return float(statistics.mode(values))
    except statistics.StatisticsError:
        return mean(values)


def variance(values: Sequence[Number], population: bool = False) -> float:
    """
    Calculate variance.

    Args:
        values: Sequence of values
        population: If True, calculate population variance

    Returns:
        Variance
    """
    if len(values) < 2:
        return 0.0

    float_values = [float(v) for v in values]
    if population:
        return statistics.pvariance(float_values)
    return statistics.variance(float_values)


def std_dev(values: Sequence[Number], population: bool = False) -> float:
    """
    Calculate standard deviation.

    Args:
        values: Sequence of values
        population: If True, calculate population std dev

    Returns:
        Standard deviation
    """
    if len(values) < 2:
        return 0.0

    float_values = [float(v) for v in values]
    if population:
        return statistics.pstdev(float_values)
    return statistics.stdev(float_values)


def covariance(x: Sequence[Number], y: Sequence[Number]) -> float:
    """
    Calculate covariance between two sequences.

    Args:
        x: First sequence
        y: Second sequence

    Returns:
        Covariance
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    x_float = [float(v) for v in x]
    y_float = [float(v) for v in y]

    x_mean = mean(x_float)
    y_mean = mean(y_float)

    n = len(x_float)
    cov = sum((x_float[i] - x_mean) * (y_float[i] - y_mean) for i in range(n))
    return cov / (n - 1)


def correlation(x: Sequence[Number], y: Sequence[Number]) -> float:
    """
    Calculate Pearson correlation coefficient.

    Args:
        x: First sequence
        y: Second sequence

    Returns:
        Correlation coefficient (-1 to 1)
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    x_float = [float(v) for v in x]
    y_float = [float(v) for v in y]

    x_std = std_dev(x_float)
    y_std = std_dev(y_float)

    if x_std == 0 or y_std == 0:
        return 0.0

    cov = covariance(x_float, y_float)
    return cov / (x_std * y_std)


def percentile(values: Sequence[Number], p: float) -> float:
    """
    Calculate percentile.

    Args:
        values: Sequence of values
        p: Percentile (0-100)

    Returns:
        Percentile value
    """
    if not values:
        return 0.0

    sorted_values = sorted(float(v) for v in values)
    n = len(sorted_values)

    if p <= 0:
        return sorted_values[0]
    if p >= 100:
        return sorted_values[-1]

    index = (p / 100) * (n - 1)
    lower = int(index)
    upper = lower + 1

    if upper >= n:
        return sorted_values[-1]

    fraction = index - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def z_score(value: Number, mean_val: Number, std_val: Number) -> float:
    """
    Calculate z-score.

    Args:
        value: Value to calculate z-score for
        mean_val: Population mean
        std_val: Population standard deviation

    Returns:
        Z-score
    """
    if float(std_val) == 0:
        return 0.0
    return (float(value) - float(mean_val)) / float(std_val)


def z_scores(values: Sequence[Number]) -> List[float]:
    """
    Calculate z-scores for a sequence.

    Args:
        values: Sequence of values

    Returns:
        List of z-scores
    """
    if len(values) < 2:
        return [0.0] * len(values)

    float_values = [float(v) for v in values]
    mean_val = mean(float_values)
    std_val = std_dev(float_values)

    if std_val == 0:
        return [0.0] * len(values)

    return [(v - mean_val) / std_val for v in float_values]


# =============================================================================
# FINANCIAL CALCULATIONS
# =============================================================================

def sharpe_ratio(
    returns: Sequence[Number],
    risk_free_rate: Number = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: Sequence of period returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year

    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    float_returns = [float(r) for r in returns]
    mean_return = mean(float_returns)
    std_return = std_dev(float_returns)

    if std_return == 0:
        return 0.0

    # Convert risk-free rate to period rate
    rf_period = float(risk_free_rate) / periods_per_year

    # Annualize
    excess_return = (mean_return - rf_period) * periods_per_year
    annualized_std = std_return * math.sqrt(periods_per_year)

    return excess_return / annualized_std


def sortino_ratio(
    returns: Sequence[Number],
    risk_free_rate: Number = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio (uses downside deviation).

    Args:
        returns: Sequence of period returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year

    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return 0.0

    float_returns = [float(r) for r in returns]
    rf_period = float(risk_free_rate) / periods_per_year

    # Calculate downside returns
    downside_returns = [min(0, r - rf_period) for r in float_returns]
    downside_std = std_dev(downside_returns)

    if downside_std == 0:
        return 0.0

    mean_return = mean(float_returns)
    excess_return = (mean_return - rf_period) * periods_per_year
    annualized_downside_std = downside_std * math.sqrt(periods_per_year)

    return excess_return / annualized_downside_std


def calmar_ratio(
    total_return: Number,
    max_drawdown: Number,
    years: Number = 1.0
) -> float:
    """
    Calculate Calmar ratio.

    Args:
        total_return: Total return
        max_drawdown: Maximum drawdown (positive value)
        years: Number of years

    Returns:
        Calmar ratio
    """
    if float(max_drawdown) == 0 or float(years) == 0:
        return 0.0

    annualized_return = float(total_return) / float(years)
    return annualized_return / abs(float(max_drawdown))


def information_ratio(
    returns: Sequence[Number],
    benchmark_returns: Sequence[Number],
    periods_per_year: int = 252
) -> float:
    """
    Calculate Information ratio.

    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns
        periods_per_year: Number of periods per year

    Returns:
        Information ratio
    """
    if len(returns) != len(benchmark_returns) or len(returns) < 2:
        return 0.0

    # Calculate tracking difference (excess returns)
    tracking_diff = [float(r) - float(b) for r, b in zip(returns, benchmark_returns)]

    mean_diff = mean(tracking_diff)
    tracking_error = std_dev(tracking_diff)

    if tracking_error == 0:
        return 0.0

    # Annualize
    return (mean_diff * periods_per_year) / (tracking_error * math.sqrt(periods_per_year))


def max_drawdown(values: Sequence[Number]) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown.

    Args:
        values: Sequence of portfolio values

    Returns:
        Tuple of (max_drawdown, peak_index, trough_index)
    """
    if not values:
        return 0.0, 0, 0

    float_values = [float(v) for v in values]
    peak = float_values[0]
    peak_idx = 0
    max_dd = 0.0
    max_dd_peak_idx = 0
    max_dd_trough_idx = 0

    for i, value in enumerate(float_values):
        if value > peak:
            peak = value
            peak_idx = i

        drawdown = (peak - value) / peak if peak > 0 else 0

        if drawdown > max_dd:
            max_dd = drawdown
            max_dd_peak_idx = peak_idx
            max_dd_trough_idx = i

    return max_dd, max_dd_peak_idx, max_dd_trough_idx


def drawdown_series(values: Sequence[Number]) -> List[float]:
    """
    Calculate drawdown series.

    Args:
        values: Sequence of portfolio values

    Returns:
        List of drawdown values
    """
    if not values:
        return []

    float_values = [float(v) for v in values]
    peak = float_values[0]
    drawdowns = []

    for value in float_values:
        if value > peak:
            peak = value
        dd = (peak - value) / peak if peak > 0 else 0
        drawdowns.append(dd)

    return drawdowns


def win_rate(returns: Sequence[Number]) -> float:
    """
    Calculate win rate.

    Args:
        returns: Sequence of trade returns

    Returns:
        Win rate (0-1)
    """
    if not returns:
        return 0.0

    wins = sum(1 for r in returns if float(r) > 0)
    return wins / len(returns)


def profit_factor(returns: Sequence[Number]) -> float:
    """
    Calculate profit factor (gross profits / gross losses).

    Args:
        returns: Sequence of trade returns

    Returns:
        Profit factor
    """
    float_returns = [float(r) for r in returns]

    gross_profit = sum(r for r in float_returns if r > 0)
    gross_loss = abs(sum(r for r in float_returns if r < 0))

    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def expectancy(returns: Sequence[Number]) -> float:
    """
    Calculate trade expectancy.

    Args:
        returns: Sequence of trade returns

    Returns:
        Expectancy per trade
    """
    if not returns:
        return 0.0

    float_returns = [float(r) for r in returns]

    wins = [r for r in float_returns if r > 0]
    losses = [r for r in float_returns if r < 0]

    avg_win = mean(wins) if wins else 0
    avg_loss = abs(mean(losses)) if losses else 0

    win_rate_val = len(wins) / len(float_returns)
    loss_rate = 1 - win_rate_val

    return (win_rate_val * avg_win) - (loss_rate * avg_loss)


def beta(
    returns: Sequence[Number],
    market_returns: Sequence[Number]
) -> float:
    """
    Calculate beta (market sensitivity).

    Args:
        returns: Asset returns
        market_returns: Market returns

    Returns:
        Beta coefficient
    """
    if len(returns) != len(market_returns) or len(returns) < 2:
        return 1.0

    float_returns = [float(r) for r in returns]
    float_market = [float(r) for r in market_returns]

    market_var = variance(float_market)
    if market_var == 0:
        return 1.0

    cov = covariance(float_returns, float_market)
    return cov / market_var


def alpha(
    returns: Sequence[Number],
    market_returns: Sequence[Number],
    risk_free_rate: Number = 0.0
) -> float:
    """
    Calculate Jensen's alpha.

    Args:
        returns: Asset returns
        market_returns: Market returns
        risk_free_rate: Risk-free rate

    Returns:
        Alpha (excess return)
    """
    if len(returns) != len(market_returns) or len(returns) < 2:
        return 0.0

    float_returns = [float(r) for r in returns]
    float_market = [float(r) for r in market_returns]
    rf = float(risk_free_rate)

    asset_beta = beta(float_returns, float_market)
    avg_return = mean(float_returns)
    avg_market = mean(float_market)

    expected_return = rf + asset_beta * (avg_market - rf)
    return avg_return - expected_return


def value_at_risk(
    returns: Sequence[Number],
    confidence: float = 0.95,
    method: str = "historical"
) -> float:
    """
    Calculate Value at Risk (VaR).

    Args:
        returns: Sequence of returns
        confidence: Confidence level (e.g., 0.95 for 95%)
        method: Calculation method ("historical" or "parametric")

    Returns:
        VaR as positive value
    """
    if not returns:
        return 0.0

    float_returns = [float(r) for r in returns]

    if method == "historical":
        # Historical simulation
        sorted_returns = sorted(float_returns)
        index = int((1 - confidence) * len(sorted_returns))
        return abs(sorted_returns[index])
    else:
        # Parametric (normal distribution)
        mean_ret = mean(float_returns)
        std_ret = std_dev(float_returns)
        # Z-score for confidence level
        z_scores_map = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
        z = z_scores_map.get(confidence, 1.645)
        return abs(mean_ret - z * std_ret)


def expected_shortfall(
    returns: Sequence[Number],
    confidence: float = 0.95
) -> float:
    """
    Calculate Expected Shortfall (CVaR).

    Args:
        returns: Sequence of returns
        confidence: Confidence level

    Returns:
        Expected Shortfall as positive value
    """
    if not returns:
        return 0.0

    float_returns = [float(r) for r in returns]
    sorted_returns = sorted(float_returns)
    cutoff_index = int((1 - confidence) * len(sorted_returns))

    # Average of worst returns
    worst_returns = sorted_returns[:max(1, cutoff_index)]
    return abs(mean(worst_returns))


# =============================================================================
# POSITION SIZING
# =============================================================================

def kelly_criterion(win_rate: Number, win_loss_ratio: Number) -> float:
    """
    Calculate Kelly criterion for optimal bet sizing.

    Args:
        win_rate: Probability of winning (0-1)
        win_loss_ratio: Average win / average loss

    Returns:
        Optimal fraction of capital to bet (0-1)
    """
    w = float(win_rate)
    r = float(win_loss_ratio)

    if r <= 0 or w <= 0 or w >= 1:
        return 0.0

    kelly = w - (1 - w) / r
    return max(0.0, min(1.0, kelly))


def position_size_risk_based(
    account_value: Number,
    risk_per_trade: Number,
    entry_price: Number,
    stop_price: Number
) -> float:
    """
    Calculate position size based on risk per trade.

    Args:
        account_value: Total account value
        risk_per_trade: Risk per trade as decimal (e.g., 0.01 for 1%)
        entry_price: Entry price
        stop_price: Stop loss price

    Returns:
        Number of shares to buy
    """
    account = float(account_value)
    risk = float(risk_per_trade)
    entry = float(entry_price)
    stop = float(stop_price)

    risk_amount = account * risk
    price_risk = abs(entry - stop)

    if price_risk == 0:
        return 0.0

    shares = risk_amount / price_risk
    return shares


def volatility_adjusted_size(
    account_value: Number,
    target_volatility: Number,
    asset_volatility: Number,
    max_position: Number = 0.2
) -> float:
    """
    Calculate volatility-adjusted position size.

    Args:
        account_value: Total account value
        target_volatility: Target portfolio volatility
        asset_volatility: Asset's volatility
        max_position: Maximum position size as fraction

    Returns:
        Position size in dollars
    """
    account = float(account_value)
    target = float(target_volatility)
    asset_vol = float(asset_volatility)
    max_pos = float(max_position)

    if asset_vol == 0:
        return 0.0

    # Size = (Target Vol / Asset Vol) * Account
    position = (target / asset_vol) * account

    # Apply max position limit
    max_position_value = account * max_pos
    return min(position, max_position_value)
