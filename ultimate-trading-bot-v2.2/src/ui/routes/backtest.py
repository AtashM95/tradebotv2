"""
Backtest Routes for Ultimate Trading Bot v2.2.

This module provides backtesting-related routes including:
- Backtest creation and execution
- Results visualization
- Strategy comparison
- Performance analysis
"""

import logging
from datetime import datetime, timedelta
from typing import Any
import uuid

from flask import Blueprint, render_template, jsonify, request, g

from ..cache import cached, get_cache


logger = logging.getLogger(__name__)

# Create blueprint
backtest_routes_bp = Blueprint(
    "backtest_routes",
    __name__,
    url_prefix="/backtest",
)

# In-memory storage for demo backtests
_backtests: dict[str, dict[str, Any]] = {}


@backtest_routes_bp.route("/")
def index() -> str:
    """
    Render backtest page.

    Returns:
        Rendered backtest template
    """
    return render_template(
        "backtest/index.html",
        page_title="Backtesting",
    )


@backtest_routes_bp.route("/new")
def new_backtest() -> str:
    """
    Render new backtest form.

    Returns:
        Rendered new backtest template
    """
    return render_template(
        "backtest/new.html",
        page_title="New Backtest",
    )


@backtest_routes_bp.route("/results/<backtest_id>")
def results_page(backtest_id: str) -> str:
    """
    Render backtest results page.

    Args:
        backtest_id: Backtest ID

    Returns:
        Rendered results template
    """
    return render_template(
        "backtest/results.html",
        backtest_id=backtest_id,
        page_title="Backtest Results",
    )


@backtest_routes_bp.route("/api/strategies")
@cached(ttl=300, tags=["backtest", "strategies"])
def get_strategies() -> tuple[dict[str, Any], int]:
    """
    Get available strategies for backtesting.

    Returns:
        Strategies list JSON response
    """
    strategies = [
        {
            "id": "momentum",
            "name": "Momentum Strategy",
            "description": "Trend-following strategy based on price momentum",
            "parameters": [
                {"name": "lookback_period", "type": "int", "default": 20, "min": 5, "max": 100},
                {"name": "threshold", "type": "float", "default": 0.02, "min": 0.01, "max": 0.1},
            ],
        },
        {
            "id": "mean_reversion",
            "name": "Mean Reversion Strategy",
            "description": "Strategy based on price returning to mean",
            "parameters": [
                {"name": "window", "type": "int", "default": 20, "min": 5, "max": 100},
                {"name": "std_dev", "type": "float", "default": 2.0, "min": 1.0, "max": 3.0},
            ],
        },
        {
            "id": "rsi",
            "name": "RSI Strategy",
            "description": "Strategy based on Relative Strength Index",
            "parameters": [
                {"name": "period", "type": "int", "default": 14, "min": 5, "max": 50},
                {"name": "oversold", "type": "int", "default": 30, "min": 10, "max": 40},
                {"name": "overbought", "type": "int", "default": 70, "min": 60, "max": 90},
            ],
        },
        {
            "id": "macd",
            "name": "MACD Strategy",
            "description": "Moving Average Convergence Divergence strategy",
            "parameters": [
                {"name": "fast_period", "type": "int", "default": 12, "min": 5, "max": 50},
                {"name": "slow_period", "type": "int", "default": 26, "min": 10, "max": 100},
                {"name": "signal_period", "type": "int", "default": 9, "min": 5, "max": 30},
            ],
        },
        {
            "id": "bollinger",
            "name": "Bollinger Bands Strategy",
            "description": "Strategy using Bollinger Bands for entry/exit",
            "parameters": [
                {"name": "window", "type": "int", "default": 20, "min": 10, "max": 50},
                {"name": "num_std", "type": "float", "default": 2.0, "min": 1.0, "max": 3.0},
            ],
        },
        {
            "id": "breakout",
            "name": "Breakout Strategy",
            "description": "Strategy based on price breakouts from ranges",
            "parameters": [
                {"name": "lookback", "type": "int", "default": 20, "min": 5, "max": 100},
                {"name": "volume_threshold", "type": "float", "default": 1.5, "min": 1.0, "max": 3.0},
            ],
        },
        {
            "id": "ml_ensemble",
            "name": "ML Ensemble Strategy",
            "description": "Machine learning ensemble for signal generation",
            "parameters": [
                {"name": "confidence_threshold", "type": "float", "default": 0.7, "min": 0.5, "max": 0.95},
                {"name": "retrain_frequency", "type": "int", "default": 30, "min": 7, "max": 90},
            ],
        },
        {
            "id": "sentiment",
            "name": "Sentiment Strategy",
            "description": "Strategy based on news and social sentiment",
            "parameters": [
                {"name": "sentiment_threshold", "type": "float", "default": 0.3, "min": 0.1, "max": 0.5},
                {"name": "hold_period", "type": "int", "default": 5, "min": 1, "max": 20},
            ],
        },
    ]

    return jsonify({
        "success": True,
        "data": strategies,
        "count": len(strategies),
    }), 200


@backtest_routes_bp.route("/api/run", methods=["POST"])
def run_backtest() -> tuple[dict[str, Any], int]:
    """
    Start a new backtest.

    Returns:
        Backtest info JSON response
    """
    data = request.get_json()
    if not data:
        return jsonify({
            "success": False,
            "message": "No backtest configuration provided",
        }), 400

    # Validate required fields
    required = ["strategy_id", "symbols", "start_date", "end_date"]
    for field in required:
        if field not in data:
            return jsonify({
                "success": False,
                "message": f"Missing required field: {field}",
            }), 400

    # Validate dates
    try:
        start_date = datetime.fromisoformat(data["start_date"].replace("Z", "+00:00"))
        end_date = datetime.fromisoformat(data["end_date"].replace("Z", "+00:00"))

        if start_date >= end_date:
            return jsonify({
                "success": False,
                "message": "Start date must be before end date",
            }), 400

        if end_date > datetime.now():
            return jsonify({
                "success": False,
                "message": "End date cannot be in the future",
            }), 400

    except ValueError:
        return jsonify({
            "success": False,
            "message": "Invalid date format",
        }), 400

    # Generate backtest ID
    backtest_id = f"bt-{uuid.uuid4().hex[:12]}"

    # Create backtest record
    backtest = {
        "backtest_id": backtest_id,
        "strategy_id": data["strategy_id"],
        "symbols": data["symbols"],
        "start_date": data["start_date"],
        "end_date": data["end_date"],
        "initial_capital": data.get("initial_capital", 100000),
        "parameters": data.get("parameters", {}),
        "status": "running",
        "progress": 0,
        "created_at": datetime.now().isoformat(),
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "results": None,
    }

    _backtests[backtest_id] = backtest

    # In production, queue backtest job
    logger.info(f"Backtest started: {backtest_id}")

    # Simulate completion for demo
    _simulate_backtest_completion(backtest_id)

    return jsonify({
        "success": True,
        "message": "Backtest started",
        "data": {
            "backtest_id": backtest_id,
            "status": "running",
        },
    }), 202


def _simulate_backtest_completion(backtest_id: str) -> None:
    """Simulate backtest completion with demo results."""
    import random

    if backtest_id not in _backtests:
        return

    backtest = _backtests[backtest_id]
    initial_capital = backtest.get("initial_capital", 100000)

    # Generate realistic demo results
    total_return = random.uniform(-0.1, 0.3)
    final_value = initial_capital * (1 + total_return)
    num_trades = random.randint(50, 300)
    win_rate = random.uniform(0.45, 0.65)
    winning = int(num_trades * win_rate)
    losing = num_trades - winning

    avg_win = random.uniform(200, 600)
    avg_loss = random.uniform(-150, -400)

    backtest["status"] = "completed"
    backtest["progress"] = 100
    backtest["completed_at"] = datetime.now().isoformat()
    backtest["results"] = {
        "summary": {
            "initial_capital": initial_capital,
            "final_value": round(final_value, 2),
            "total_return": round(total_return, 4),
            "annualized_return": round(total_return * 2.5, 4),
            "max_drawdown": round(random.uniform(-0.05, -0.25), 4),
            "sharpe_ratio": round(random.uniform(0.5, 2.5), 2),
            "sortino_ratio": round(random.uniform(0.7, 3.0), 2),
            "calmar_ratio": round(random.uniform(0.5, 2.0), 2),
            "volatility": round(random.uniform(0.1, 0.25), 4),
            "beta": round(random.uniform(0.8, 1.2), 2),
            "alpha": round(random.uniform(-0.02, 0.05), 4),
        },
        "trades": {
            "total": num_trades,
            "winning": winning,
            "losing": losing,
            "win_rate": round(win_rate, 4),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(abs(winning * avg_win / (losing * avg_loss)), 2),
            "largest_win": round(avg_win * random.uniform(2, 5), 2),
            "largest_loss": round(avg_loss * random.uniform(2, 4), 2),
            "avg_holding_period": round(random.uniform(1, 10), 1),
        },
        "monthly_returns": _generate_monthly_returns(backtest["start_date"], backtest["end_date"]),
    }


def _generate_monthly_returns(start_date: str, end_date: str) -> list[dict[str, Any]]:
    """Generate monthly returns data."""
    import random

    returns = []
    current = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
    end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))

    while current < end:
        returns.append({
            "month": current.strftime("%Y-%m"),
            "return": round(random.uniform(-0.08, 0.12), 4),
            "benchmark": round(random.uniform(-0.05, 0.08), 4),
        })
        current = (current.replace(day=1) + timedelta(days=32)).replace(day=1)

    return returns


@backtest_routes_bp.route("/api/status/<backtest_id>")
def get_backtest_status(backtest_id: str) -> tuple[dict[str, Any], int]:
    """
    Get backtest status.

    Args:
        backtest_id: Backtest ID

    Returns:
        Status JSON response
    """
    backtest = _backtests.get(backtest_id)
    if not backtest:
        return jsonify({
            "success": False,
            "message": "Backtest not found",
        }), 404

    return jsonify({
        "success": True,
        "data": {
            "backtest_id": backtest_id,
            "status": backtest["status"],
            "progress": backtest["progress"],
            "created_at": backtest["created_at"],
            "started_at": backtest["started_at"],
            "completed_at": backtest["completed_at"],
        },
    }), 200


@backtest_routes_bp.route("/api/results/<backtest_id>")
def get_backtest_results(backtest_id: str) -> tuple[dict[str, Any], int]:
    """
    Get backtest results.

    Args:
        backtest_id: Backtest ID

    Returns:
        Results JSON response
    """
    backtest = _backtests.get(backtest_id)
    if not backtest:
        return jsonify({
            "success": False,
            "message": "Backtest not found",
        }), 404

    if backtest["status"] != "completed":
        return jsonify({
            "success": False,
            "message": f"Backtest status: {backtest['status']}",
            "data": {
                "status": backtest["status"],
                "progress": backtest["progress"],
            },
        }), 202

    return jsonify({
        "success": True,
        "data": {
            "backtest_id": backtest_id,
            "config": {
                "strategy_id": backtest["strategy_id"],
                "symbols": backtest["symbols"],
                "start_date": backtest["start_date"],
                "end_date": backtest["end_date"],
                "initial_capital": backtest["initial_capital"],
                "parameters": backtest["parameters"],
            },
            "results": backtest["results"],
            "completed_at": backtest["completed_at"],
        },
    }), 200


@backtest_routes_bp.route("/api/results/<backtest_id>/equity-curve")
def get_equity_curve(backtest_id: str) -> tuple[dict[str, Any], int]:
    """
    Get equity curve data for charting.

    Args:
        backtest_id: Backtest ID

    Returns:
        Equity curve JSON response
    """
    backtest = _backtests.get(backtest_id)
    if not backtest:
        return jsonify({
            "success": False,
            "message": "Backtest not found",
        }), 404

    if backtest["status"] != "completed":
        return jsonify({
            "success": False,
            "message": "Backtest not completed",
        }), 400

    # Generate equity curve data
    import random

    start = datetime.fromisoformat(backtest["start_date"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(backtest["end_date"].replace("Z", "+00:00"))
    days = (end - start).days
    points = min(days, 500)
    step = max(1, days // points)

    initial = backtest["initial_capital"]
    final = backtest["results"]["summary"]["final_value"]
    daily_return = (final / initial) ** (1 / days) - 1

    data = []
    equity = initial
    benchmark = initial
    current = start

    for i in range(points):
        noise = random.uniform(-0.02, 0.02)
        equity *= (1 + daily_return + noise)
        benchmark *= (1 + daily_return * 0.7 + random.uniform(-0.01, 0.01))

        data.append({
            "date": current.strftime("%Y-%m-%d"),
            "timestamp": int(current.timestamp() * 1000),
            "equity": round(equity, 2),
            "benchmark": round(benchmark, 2),
            "drawdown": round(min(0, (equity / max(e["equity"] for e in data + [{"equity": equity}]) - 1) * 100), 2) if data else 0,
        })
        current += timedelta(days=step)

    return jsonify({
        "success": True,
        "data": data,
    }), 200


@backtest_routes_bp.route("/api/results/<backtest_id>/trades")
def get_backtest_trades(backtest_id: str) -> tuple[dict[str, Any], int]:
    """
    Get backtest trades list.

    Args:
        backtest_id: Backtest ID

    Returns:
        Trades list JSON response
    """
    limit = request.args.get("limit", 50, type=int)
    offset = request.args.get("offset", 0, type=int)

    backtest = _backtests.get(backtest_id)
    if not backtest:
        return jsonify({
            "success": False,
            "message": "Backtest not found",
        }), 404

    if backtest["status"] != "completed":
        return jsonify({
            "success": False,
            "message": "Backtest not completed",
        }), 400

    # Generate sample trades
    import random

    trades = []
    symbols = backtest["symbols"]
    start = datetime.fromisoformat(backtest["start_date"].replace("Z", "+00:00"))
    num_trades = backtest["results"]["trades"]["total"]

    for i in range(min(num_trades, 100)):
        symbol = random.choice(symbols)
        side = random.choice(["buy", "sell"])
        quantity = random.randint(10, 100)
        entry_price = random.uniform(100, 500)
        exit_price = entry_price * random.uniform(0.95, 1.1)
        pnl = (exit_price - entry_price) * quantity if side == "buy" else (entry_price - exit_price) * quantity

        trade_date = start + timedelta(days=random.randint(0, (datetime.fromisoformat(backtest["end_date"].replace("Z", "+00:00")) - start).days))

        trades.append({
            "trade_id": f"bt-trd-{i+1:04d}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "pnl": round(pnl, 2),
            "pnl_percent": round((exit_price / entry_price - 1) * 100, 2),
            "entry_date": trade_date.isoformat(),
            "exit_date": (trade_date + timedelta(days=random.randint(1, 10))).isoformat(),
            "holding_period": random.randint(1, 10),
        })

    # Sort by date
    trades.sort(key=lambda x: x["entry_date"], reverse=True)

    return jsonify({
        "success": True,
        "data": trades[offset:offset + limit],
        "total": num_trades,
        "offset": offset,
        "limit": limit,
    }), 200


@backtest_routes_bp.route("/api/list")
def list_backtests() -> tuple[dict[str, Any], int]:
    """
    List all backtests.

    Returns:
        Backtests list JSON response
    """
    limit = request.args.get("limit", 20, type=int)
    offset = request.args.get("offset", 0, type=int)
    status = request.args.get("status")

    backtests = list(_backtests.values())

    if status:
        backtests = [b for b in backtests if b["status"] == status]

    # Sort by created_at descending
    backtests.sort(key=lambda x: x["created_at"], reverse=True)

    # Pagination
    paginated = backtests[offset:offset + limit]

    # Return summary only (no full results)
    summary = []
    for bt in paginated:
        item = {
            "backtest_id": bt["backtest_id"],
            "strategy_id": bt["strategy_id"],
            "symbols": bt["symbols"],
            "start_date": bt["start_date"],
            "end_date": bt["end_date"],
            "status": bt["status"],
            "progress": bt["progress"],
            "created_at": bt["created_at"],
            "completed_at": bt["completed_at"],
        }
        if bt["status"] == "completed" and bt["results"]:
            item["total_return"] = bt["results"]["summary"]["total_return"]
            item["sharpe_ratio"] = bt["results"]["summary"]["sharpe_ratio"]
        summary.append(item)

    return jsonify({
        "success": True,
        "data": summary,
        "total": len(backtests),
        "offset": offset,
        "limit": limit,
    }), 200


@backtest_routes_bp.route("/api/<backtest_id>", methods=["DELETE"])
def delete_backtest(backtest_id: str) -> tuple[dict[str, Any], int]:
    """
    Delete a backtest.

    Args:
        backtest_id: Backtest ID

    Returns:
        Deletion result JSON response
    """
    if backtest_id not in _backtests:
        return jsonify({
            "success": False,
            "message": "Backtest not found",
        }), 404

    del _backtests[backtest_id]
    logger.info(f"Backtest deleted: {backtest_id}")

    return jsonify({
        "success": True,
        "message": "Backtest deleted successfully",
    }), 200


@backtest_routes_bp.route("/api/compare", methods=["POST"])
def compare_backtests() -> tuple[dict[str, Any], int]:
    """
    Compare multiple backtests.

    Returns:
        Comparison results JSON response
    """
    data = request.get_json()
    if not data:
        return jsonify({
            "success": False,
            "message": "No backtest IDs provided",
        }), 400

    backtest_ids = data.get("backtest_ids", [])
    if len(backtest_ids) < 2:
        return jsonify({
            "success": False,
            "message": "At least 2 backtests required for comparison",
        }), 400

    if len(backtest_ids) > 5:
        return jsonify({
            "success": False,
            "message": "Maximum 5 backtests can be compared",
        }), 400

    comparison = []
    for bt_id in backtest_ids:
        backtest = _backtests.get(bt_id)
        if not backtest:
            return jsonify({
                "success": False,
                "message": f"Backtest not found: {bt_id}",
            }), 404

        if backtest["status"] != "completed":
            return jsonify({
                "success": False,
                "message": f"Backtest not completed: {bt_id}",
            }), 400

        comparison.append({
            "backtest_id": bt_id,
            "strategy_id": backtest["strategy_id"],
            "symbols": backtest["symbols"],
            "period": f"{backtest['start_date'][:10]} to {backtest['end_date'][:10]}",
            "metrics": backtest["results"]["summary"],
            "trades": backtest["results"]["trades"],
        })

    return jsonify({
        "success": True,
        "data": comparison,
    }), 200


@backtest_routes_bp.route("/api/optimize", methods=["POST"])
def optimize_strategy() -> tuple[dict[str, Any], int]:
    """
    Run strategy optimization.

    Returns:
        Optimization job info JSON response
    """
    data = request.get_json()
    if not data:
        return jsonify({
            "success": False,
            "message": "No optimization config provided",
        }), 400

    required = ["strategy_id", "symbols", "start_date", "end_date", "parameter_ranges"]
    for field in required:
        if field not in data:
            return jsonify({
                "success": False,
                "message": f"Missing required field: {field}",
            }), 400

    optimization_id = f"opt-{uuid.uuid4().hex[:12]}"

    # In production, queue optimization job
    logger.info(f"Optimization started: {optimization_id}")

    return jsonify({
        "success": True,
        "message": "Optimization started",
        "data": {
            "optimization_id": optimization_id,
            "status": "running",
            "estimated_time": "5-10 minutes",
        },
    }), 202


# Module version
__version__ = "2.2.0"
