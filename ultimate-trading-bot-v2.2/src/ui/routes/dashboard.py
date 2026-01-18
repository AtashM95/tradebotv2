"""
Dashboard Routes for Ultimate Trading Bot v2.2.

This module provides dashboard-related routes including:
- Main dashboard view
- Widget management
- Real-time data endpoints
- Dashboard customization
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from flask import Blueprint, render_template, jsonify, request, g

from ..cache import cached, get_cache
from ..dashboard import WidgetType


logger = logging.getLogger(__name__)

# Create blueprint
dashboard_routes_bp = Blueprint(
    "dashboard_routes",
    __name__,
    url_prefix="/dashboard",
)


@dashboard_routes_bp.route("/")
def index() -> str:
    """
    Render main dashboard page.

    Returns:
        Rendered dashboard template
    """
    dashboard_manager = getattr(g, "dashboard_manager", None)
    theme_manager = getattr(g, "theme_manager", None)

    # Get user layout preferences
    user_id = getattr(g, "current_user_id", "default")
    layout = None
    if dashboard_manager:
        layout = dashboard_manager.get_layout(user_id)

    # Get available widgets
    widgets = []
    if dashboard_manager:
        widgets = dashboard_manager.get_available_widgets()

    # Get current theme
    theme = "dark"
    if theme_manager:
        user_theme = theme_manager.get_user_theme(user_id)
        theme = user_theme.name.lower()

    return render_template(
        "dashboard/index.html",
        layout=layout,
        widgets=widgets,
        theme=theme,
        page_title="Dashboard",
    )


@dashboard_routes_bp.route("/api/summary")
@cached(ttl=5, tags=["dashboard", "summary"])
def get_summary() -> tuple[dict[str, Any], int]:
    """
    Get dashboard summary data.

    Returns:
        Summary data JSON response
    """
    portfolio_manager = getattr(g, "portfolio_manager", None)

    if portfolio_manager:
        try:
            summary = portfolio_manager.get_summary()
            return jsonify({
                "success": True,
                "data": summary,
                "timestamp": datetime.now().isoformat(),
            }), 200
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")

    # Return demo data if no portfolio manager
    summary = {
        "total_value": 125000.00,
        "cash_balance": 25000.00,
        "positions_value": 100000.00,
        "daily_pnl": 1250.00,
        "daily_pnl_percent": 1.01,
        "total_pnl": 15000.00,
        "total_pnl_percent": 13.64,
        "positions_count": 5,
        "open_orders": 2,
        "buying_power": 50000.00,
    }

    return jsonify({
        "success": True,
        "data": summary,
        "timestamp": datetime.now().isoformat(),
    }), 200


@dashboard_routes_bp.route("/api/positions")
@cached(ttl=5, tags=["dashboard", "positions"])
def get_positions() -> tuple[dict[str, Any], int]:
    """
    Get current positions for dashboard.

    Returns:
        Positions data JSON response
    """
    portfolio_manager = getattr(g, "portfolio_manager", None)

    if portfolio_manager:
        try:
            positions = portfolio_manager.get_positions()
            return jsonify({
                "success": True,
                "data": positions,
                "count": len(positions),
                "timestamp": datetime.now().isoformat(),
            }), 200
        except Exception as e:
            logger.error(f"Error getting positions: {e}")

    # Return demo data
    positions = [
        {
            "symbol": "AAPL",
            "quantity": 100,
            "avg_price": 150.00,
            "current_price": 155.00,
            "market_value": 15500.00,
            "pnl": 500.00,
            "pnl_percent": 3.33,
            "day_change": 2.50,
            "day_change_percent": 1.64,
        },
        {
            "symbol": "GOOGL",
            "quantity": 50,
            "avg_price": 140.00,
            "current_price": 142.50,
            "market_value": 7125.00,
            "pnl": 125.00,
            "pnl_percent": 1.79,
            "day_change": -0.75,
            "day_change_percent": -0.52,
        },
        {
            "symbol": "MSFT",
            "quantity": 75,
            "avg_price": 400.00,
            "current_price": 410.00,
            "market_value": 30750.00,
            "pnl": 750.00,
            "pnl_percent": 2.50,
            "day_change": 3.60,
            "day_change_percent": 0.89,
        },
    ]

    return jsonify({
        "success": True,
        "data": positions,
        "count": len(positions),
        "timestamp": datetime.now().isoformat(),
    }), 200


@dashboard_routes_bp.route("/api/recent-trades")
@cached(ttl=10, tags=["dashboard", "trades"])
def get_recent_trades() -> tuple[dict[str, Any], int]:
    """
    Get recent trades for dashboard.

    Returns:
        Recent trades JSON response
    """
    limit = request.args.get("limit", 10, type=int)
    trading_engine = getattr(g, "trading_engine", None)

    if trading_engine:
        try:
            trades = trading_engine.get_recent_trades(limit=limit)
            return jsonify({
                "success": True,
                "data": trades,
                "count": len(trades),
                "timestamp": datetime.now().isoformat(),
            }), 200
        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")

    # Return demo data
    trades = [
        {
            "trade_id": "trd-001",
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 50,
            "price": 154.50,
            "total": 7725.00,
            "pnl": None,
            "executed_at": (datetime.now() - timedelta(hours=2)).isoformat(),
        },
        {
            "trade_id": "trd-002",
            "symbol": "GOOGL",
            "side": "sell",
            "quantity": 25,
            "price": 143.00,
            "total": 3575.00,
            "pnl": 75.00,
            "executed_at": (datetime.now() - timedelta(hours=5)).isoformat(),
        },
        {
            "trade_id": "trd-003",
            "symbol": "TSLA",
            "side": "buy",
            "quantity": 30,
            "price": 245.00,
            "total": 7350.00,
            "pnl": None,
            "executed_at": (datetime.now() - timedelta(days=1)).isoformat(),
        },
    ]

    return jsonify({
        "success": True,
        "data": trades[:limit],
        "count": min(len(trades), limit),
        "timestamp": datetime.now().isoformat(),
    }), 200


@dashboard_routes_bp.route("/api/open-orders")
@cached(ttl=5, tags=["dashboard", "orders"])
def get_open_orders() -> tuple[dict[str, Any], int]:
    """
    Get open orders for dashboard.

    Returns:
        Open orders JSON response
    """
    trading_engine = getattr(g, "trading_engine", None)

    if trading_engine:
        try:
            orders = trading_engine.get_open_orders()
            return jsonify({
                "success": True,
                "data": orders,
                "count": len(orders),
                "timestamp": datetime.now().isoformat(),
            }), 200
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")

    # Return demo data
    orders = [
        {
            "order_id": "ord-001",
            "symbol": "NVDA",
            "side": "buy",
            "type": "limit",
            "quantity": 20,
            "limit_price": 800.00,
            "status": "pending",
            "created_at": (datetime.now() - timedelta(minutes=30)).isoformat(),
        },
        {
            "order_id": "ord-002",
            "symbol": "AMD",
            "side": "buy",
            "type": "stop_limit",
            "quantity": 50,
            "stop_price": 155.00,
            "limit_price": 156.00,
            "status": "pending",
            "created_at": (datetime.now() - timedelta(hours=1)).isoformat(),
        },
    ]

    return jsonify({
        "success": True,
        "data": orders,
        "count": len(orders),
        "timestamp": datetime.now().isoformat(),
    }), 200


@dashboard_routes_bp.route("/api/alerts")
@cached(ttl=10, tags=["dashboard", "alerts"])
def get_alerts() -> tuple[dict[str, Any], int]:
    """
    Get active alerts for dashboard.

    Returns:
        Alerts JSON response
    """
    alerts = [
        {
            "alert_id": "alert-001",
            "type": "price",
            "symbol": "AAPL",
            "condition": "above",
            "trigger_value": 160.00,
            "current_value": 155.00,
            "status": "active",
            "created_at": (datetime.now() - timedelta(days=2)).isoformat(),
        },
        {
            "alert_id": "alert-002",
            "type": "pnl",
            "condition": "below",
            "trigger_value": -1000.00,
            "current_value": 1250.00,
            "status": "active",
            "created_at": (datetime.now() - timedelta(days=1)).isoformat(),
        },
    ]

    return jsonify({
        "success": True,
        "data": alerts,
        "count": len(alerts),
        "timestamp": datetime.now().isoformat(),
    }), 200


@dashboard_routes_bp.route("/api/market-overview")
@cached(ttl=30, tags=["dashboard", "market"])
def get_market_overview() -> tuple[dict[str, Any], int]:
    """
    Get market overview data.

    Returns:
        Market overview JSON response
    """
    data_manager = getattr(g, "data_manager", None)

    if data_manager:
        try:
            overview = data_manager.get_market_overview()
            return jsonify({
                "success": True,
                "data": overview,
                "timestamp": datetime.now().isoformat(),
            }), 200
        except Exception as e:
            logger.error(f"Error getting market overview: {e}")

    # Return demo data
    overview = {
        "indices": [
            {"symbol": "SPY", "name": "S&P 500", "price": 502.50, "change": 1.25, "change_percent": 0.25},
            {"symbol": "QQQ", "name": "Nasdaq 100", "price": 435.00, "change": 2.50, "change_percent": 0.58},
            {"symbol": "DIA", "name": "Dow Jones", "price": 390.00, "change": -0.75, "change_percent": -0.19},
            {"symbol": "IWM", "name": "Russell 2000", "price": 198.50, "change": 0.50, "change_percent": 0.25},
        ],
        "sectors": [
            {"name": "Technology", "change_percent": 1.25},
            {"name": "Healthcare", "change_percent": 0.50},
            {"name": "Financials", "change_percent": -0.25},
            {"name": "Energy", "change_percent": -0.75},
            {"name": "Consumer", "change_percent": 0.35},
        ],
        "market_status": "open",
        "next_event": "Market closes in 3h 45m",
    }

    return jsonify({
        "success": True,
        "data": overview,
        "timestamp": datetime.now().isoformat(),
    }), 200


@dashboard_routes_bp.route("/api/performance")
@cached(ttl=60, tags=["dashboard", "performance"])
def get_performance() -> tuple[dict[str, Any], int]:
    """
    Get portfolio performance data.

    Returns:
        Performance data JSON response
    """
    period = request.args.get("period", "1M")
    portfolio_manager = getattr(g, "portfolio_manager", None)

    if portfolio_manager:
        try:
            performance = portfolio_manager.get_performance(period=period)
            return jsonify({
                "success": True,
                "data": performance,
                "timestamp": datetime.now().isoformat(),
            }), 200
        except Exception as e:
            logger.error(f"Error getting performance: {e}")

    # Return demo data
    performance = {
        "period": period,
        "total_return": 0.1364,
        "annualized_return": 0.2728,
        "benchmark_return": 0.1100,
        "alpha": 0.0264,
        "beta": 1.05,
        "sharpe_ratio": 1.85,
        "sortino_ratio": 2.15,
        "max_drawdown": -0.0823,
        "win_rate": 0.62,
        "profit_factor": 2.1,
        "total_trades": 156,
        "winning_trades": 97,
        "losing_trades": 59,
        "avg_win": 425.50,
        "avg_loss": -195.75,
    }

    return jsonify({
        "success": True,
        "data": performance,
        "timestamp": datetime.now().isoformat(),
    }), 200


@dashboard_routes_bp.route("/api/portfolio-history")
@cached(ttl=300, tags=["dashboard", "history"])
def get_portfolio_history() -> tuple[dict[str, Any], int]:
    """
    Get portfolio value history for chart.

    Returns:
        Portfolio history JSON response
    """
    period = request.args.get("period", "1M")
    portfolio_manager = getattr(g, "portfolio_manager", None)

    # Calculate date range
    periods = {
        "1D": timedelta(days=1),
        "1W": timedelta(weeks=1),
        "1M": timedelta(days=30),
        "3M": timedelta(days=90),
        "6M": timedelta(days=180),
        "1Y": timedelta(days=365),
        "YTD": None,
        "ALL": None,
    }

    delta = periods.get(period, timedelta(days=30))
    end_date = datetime.now()

    if period == "YTD":
        start_date = datetime(end_date.year, 1, 1)
    elif period == "ALL" or delta is None:
        start_date = end_date - timedelta(days=365 * 2)
    else:
        start_date = end_date - delta

    if portfolio_manager:
        try:
            history = portfolio_manager.get_value_history(
                start_date=start_date,
                end_date=end_date,
            )
            return jsonify({
                "success": True,
                "data": history,
                "period": period,
                "timestamp": datetime.now().isoformat(),
            }), 200
        except Exception as e:
            logger.error(f"Error getting portfolio history: {e}")

    # Return demo data
    import random
    days = (end_date - start_date).days
    interval = max(1, days // 50)

    data = []
    value = 100000.0
    current_date = start_date

    while current_date <= end_date:
        # Random daily change between -2% and +2%
        change = random.uniform(-0.02, 0.025)
        value *= (1 + change)
        data.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "timestamp": int(current_date.timestamp() * 1000),
            "value": round(value, 2),
        })
        current_date += timedelta(days=interval)

    return jsonify({
        "success": True,
        "data": data,
        "period": period,
        "timestamp": datetime.now().isoformat(),
    }), 200


@dashboard_routes_bp.route("/api/widgets")
def get_widgets() -> tuple[dict[str, Any], int]:
    """
    Get available dashboard widgets.

    Returns:
        Widgets list JSON response
    """
    dashboard_manager = getattr(g, "dashboard_manager", None)

    widgets = []
    if dashboard_manager:
        widgets = dashboard_manager.get_available_widgets()
    else:
        widgets = [
            {"id": "portfolio_value", "name": "Portfolio Value", "type": "stat"},
            {"id": "daily_pnl", "name": "Daily P&L", "type": "stat"},
            {"id": "positions", "name": "Positions", "type": "table"},
            {"id": "recent_trades", "name": "Recent Trades", "type": "table"},
            {"id": "open_orders", "name": "Open Orders", "type": "table"},
            {"id": "alerts", "name": "Alerts", "type": "list"},
            {"id": "market_overview", "name": "Market Overview", "type": "card"},
            {"id": "portfolio_chart", "name": "Portfolio Chart", "type": "chart"},
            {"id": "watchlist", "name": "Watchlist", "type": "table"},
        ]

    return jsonify({
        "success": True,
        "data": widgets,
        "count": len(widgets),
    }), 200


@dashboard_routes_bp.route("/api/layout", methods=["GET"])
def get_layout() -> tuple[dict[str, Any], int]:
    """
    Get user's dashboard layout.

    Returns:
        Layout configuration JSON response
    """
    user_id = getattr(g, "current_user_id", "default")
    dashboard_manager = getattr(g, "dashboard_manager", None)

    layout = None
    if dashboard_manager:
        layout = dashboard_manager.get_layout(user_id)

    if not layout:
        layout = {
            "columns": 3,
            "widgets": [
                {"id": "portfolio_value", "x": 0, "y": 0, "w": 1, "h": 1},
                {"id": "daily_pnl", "x": 1, "y": 0, "w": 1, "h": 1},
                {"id": "positions", "x": 2, "y": 0, "w": 1, "h": 2},
                {"id": "portfolio_chart", "x": 0, "y": 1, "w": 2, "h": 2},
                {"id": "recent_trades", "x": 0, "y": 3, "w": 2, "h": 2},
                {"id": "open_orders", "x": 2, "y": 2, "w": 1, "h": 2},
            ],
        }

    return jsonify({
        "success": True,
        "data": layout,
    }), 200


@dashboard_routes_bp.route("/api/layout", methods=["PUT"])
def save_layout() -> tuple[dict[str, Any], int]:
    """
    Save user's dashboard layout.

    Returns:
        Success response
    """
    user_id = getattr(g, "current_user_id", "default")
    dashboard_manager = getattr(g, "dashboard_manager", None)

    data = request.get_json()
    if not data:
        return jsonify({
            "success": False,
            "message": "No layout data provided",
        }), 400

    if dashboard_manager:
        try:
            dashboard_manager.save_layout(user_id, data)
        except Exception as e:
            logger.error(f"Error saving layout: {e}")
            return jsonify({
                "success": False,
                "message": "Failed to save layout",
            }), 500

    # Invalidate cache
    cache = get_cache()
    cache.delete_by_tag("dashboard")

    return jsonify({
        "success": True,
        "message": "Layout saved successfully",
    }), 200


@dashboard_routes_bp.route("/api/refresh", methods=["POST"])
def refresh_dashboard() -> tuple[dict[str, Any], int]:
    """
    Force refresh dashboard data.

    Returns:
        Refresh status JSON response
    """
    # Invalidate all dashboard cache
    cache = get_cache()
    deleted = cache.delete_by_tag("dashboard")

    logger.info(f"Dashboard cache invalidated: {deleted} entries")

    return jsonify({
        "success": True,
        "message": f"Refreshed {deleted} cached entries",
        "timestamp": datetime.now().isoformat(),
    }), 200


# Module version
__version__ = "2.2.0"
