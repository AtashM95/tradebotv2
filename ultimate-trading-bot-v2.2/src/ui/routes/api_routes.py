"""
API Routes for Ultimate Trading Bot v2.2.

This module provides REST API routes for external integrations including:
- Account information
- Market data
- Order management
- Webhook handling
"""

import logging
from datetime import datetime, timedelta
from typing import Any
import hmac
import hashlib

from flask import Blueprint, jsonify, request, g


logger = logging.getLogger(__name__)

# Create blueprint
api_routes_bp = Blueprint(
    "api_routes",
    __name__,
    url_prefix="/api/v1",
)


def verify_api_key() -> tuple[bool, str | None]:
    """
    Verify API key from request headers.

    Returns:
        Tuple of (is_valid, error_message)
    """
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        return False, "API key required"

    # In production, validate against database
    # For demo, accept any key starting with "sk_"
    if not api_key.startswith("sk_"):
        return False, "Invalid API key format"

    return True, None


def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    """
    Verify webhook signature.

    Args:
        payload: Request body
        signature: Provided signature
        secret: Webhook secret

    Returns:
        True if signature is valid
    """
    expected = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(f"sha256={expected}", signature)


@api_routes_bp.before_request
def check_api_auth() -> tuple[dict[str, Any], int] | None:
    """Check API authentication for all routes."""
    # Skip auth for health check
    if request.endpoint == "api_routes.health":
        return None

    # Skip auth for webhooks (they have their own signature verification)
    if request.path.startswith("/api/v1/webhooks/"):
        return None

    is_valid, error = verify_api_key()
    if not is_valid:
        return jsonify({
            "error": True,
            "message": error,
        }), 401

    return None


@api_routes_bp.route("/health")
def health() -> tuple[dict[str, Any], int]:
    """
    Health check endpoint.

    Returns:
        Health status JSON response
    """
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.2.0",
    }), 200


@api_routes_bp.route("/account")
def get_account() -> tuple[dict[str, Any], int]:
    """
    Get account information.

    Returns:
        Account info JSON response
    """
    portfolio_manager = getattr(g, "portfolio_manager", None)

    if portfolio_manager:
        try:
            account = portfolio_manager.get_account()
            return jsonify({
                "success": True,
                "data": account,
            }), 200
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            return jsonify({
                "success": False,
                "message": str(e),
            }), 500

    # Demo account
    return jsonify({
        "success": True,
        "data": {
            "account_id": "acc-demo-001",
            "status": "active",
            "currency": "USD",
            "buying_power": 50000.00,
            "cash": 25000.00,
            "portfolio_value": 125000.00,
            "equity": 125000.00,
            "last_equity": 123750.00,
            "long_market_value": 100000.00,
            "short_market_value": 0.00,
            "initial_margin": 0.00,
            "maintenance_margin": 0.00,
            "daytrade_count": 1,
            "pattern_day_trader": False,
            "trading_blocked": False,
            "transfers_blocked": False,
            "account_blocked": False,
            "created_at": "2024-01-01T00:00:00Z",
        },
    }), 200


@api_routes_bp.route("/positions")
def get_positions() -> tuple[dict[str, Any], int]:
    """
    Get all positions.

    Returns:
        Positions list JSON response
    """
    portfolio_manager = getattr(g, "portfolio_manager", None)

    if portfolio_manager:
        try:
            positions = portfolio_manager.get_positions()
            return jsonify({
                "success": True,
                "data": positions,
                "count": len(positions),
            }), 200
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return jsonify({
                "success": False,
                "message": str(e),
            }), 500

    # Demo positions
    return jsonify({
        "success": True,
        "data": [
            {
                "symbol": "AAPL",
                "qty": 100,
                "side": "long",
                "avg_entry_price": 150.00,
                "market_value": 15550.00,
                "cost_basis": 15000.00,
                "unrealized_pl": 550.00,
                "unrealized_plpc": 0.0367,
                "current_price": 155.50,
                "lastday_price": 153.00,
                "change_today": 0.0163,
            },
            {
                "symbol": "GOOGL",
                "qty": 75,
                "side": "long",
                "avg_entry_price": 138.00,
                "market_value": 10687.50,
                "cost_basis": 10350.00,
                "unrealized_pl": 337.50,
                "unrealized_plpc": 0.0326,
                "current_price": 142.50,
                "lastday_price": 143.25,
                "change_today": -0.0052,
            },
        ],
        "count": 2,
    }), 200


@api_routes_bp.route("/positions/<symbol>")
def get_position(symbol: str) -> tuple[dict[str, Any], int]:
    """
    Get position for a symbol.

    Args:
        symbol: Stock symbol

    Returns:
        Position JSON response
    """
    portfolio_manager = getattr(g, "portfolio_manager", None)

    if portfolio_manager:
        try:
            position = portfolio_manager.get_position(symbol.upper())
            if not position:
                return jsonify({
                    "success": False,
                    "message": f"No position for {symbol}",
                }), 404
            return jsonify({
                "success": True,
                "data": position,
            }), 200
        except Exception as e:
            logger.error(f"Error getting position: {e}")
            return jsonify({
                "success": False,
                "message": str(e),
            }), 500

    # Demo position
    return jsonify({
        "success": True,
        "data": {
            "symbol": symbol.upper(),
            "qty": 100,
            "side": "long",
            "avg_entry_price": 150.00,
            "market_value": 15550.00,
            "cost_basis": 15000.00,
            "unrealized_pl": 550.00,
            "unrealized_plpc": 0.0367,
            "current_price": 155.50,
        },
    }), 200


@api_routes_bp.route("/orders", methods=["GET"])
def get_orders() -> tuple[dict[str, Any], int]:
    """
    Get orders.

    Returns:
        Orders list JSON response
    """
    status = request.args.get("status", "open")
    limit = request.args.get("limit", 50, type=int)

    trading_engine = getattr(g, "trading_engine", None)

    if trading_engine:
        try:
            orders = trading_engine.get_orders(status=status, limit=limit)
            return jsonify({
                "success": True,
                "data": orders,
                "count": len(orders),
            }), 200
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return jsonify({
                "success": False,
                "message": str(e),
            }), 500

    # Demo orders
    return jsonify({
        "success": True,
        "data": [
            {
                "id": "ord-001",
                "symbol": "NVDA",
                "side": "buy",
                "type": "limit",
                "qty": 20,
                "limit_price": 800.00,
                "status": "new",
                "created_at": datetime.now().isoformat(),
            },
        ],
        "count": 1,
    }), 200


@api_routes_bp.route("/orders", methods=["POST"])
def create_order() -> tuple[dict[str, Any], int]:
    """
    Create a new order.

    Returns:
        Order JSON response
    """
    data = request.get_json()
    if not data:
        return jsonify({
            "success": False,
            "message": "No order data provided",
        }), 400

    required = ["symbol", "qty", "side", "type"]
    for field in required:
        if field not in data:
            return jsonify({
                "success": False,
                "message": f"Missing required field: {field}",
            }), 400

    trading_engine = getattr(g, "trading_engine", None)

    if trading_engine:
        try:
            order = trading_engine.place_order(
                symbol=data["symbol"],
                side=data["side"],
                quantity=data["qty"],
                order_type=data["type"],
                limit_price=data.get("limit_price"),
                stop_price=data.get("stop_price"),
                time_in_force=data.get("time_in_force", "day"),
            )
            return jsonify({
                "success": True,
                "data": order,
            }), 201
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            return jsonify({
                "success": False,
                "message": str(e),
            }), 500

    # Demo order
    return jsonify({
        "success": True,
        "data": {
            "id": f"ord-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "symbol": data["symbol"],
            "qty": data["qty"],
            "side": data["side"],
            "type": data["type"],
            "limit_price": data.get("limit_price"),
            "stop_price": data.get("stop_price"),
            "time_in_force": data.get("time_in_force", "day"),
            "status": "new",
            "created_at": datetime.now().isoformat(),
        },
    }), 201


@api_routes_bp.route("/orders/<order_id>", methods=["DELETE"])
def cancel_order(order_id: str) -> tuple[dict[str, Any], int]:
    """
    Cancel an order.

    Args:
        order_id: Order ID

    Returns:
        Cancellation result JSON response
    """
    trading_engine = getattr(g, "trading_engine", None)

    if trading_engine:
        try:
            result = trading_engine.cancel_order(order_id)
            return jsonify({
                "success": True,
                "data": result,
            }), 200
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return jsonify({
                "success": False,
                "message": str(e),
            }), 500

    return jsonify({
        "success": True,
        "data": {
            "id": order_id,
            "status": "cancelled",
        },
    }), 200


@api_routes_bp.route("/market/quotes/<symbol>")
def get_quote(symbol: str) -> tuple[dict[str, Any], int]:
    """
    Get quote for a symbol.

    Args:
        symbol: Stock symbol

    Returns:
        Quote JSON response
    """
    data_manager = getattr(g, "data_manager", None)

    if data_manager:
        try:
            quote = data_manager.get_quote(symbol.upper())
            return jsonify({
                "success": True,
                "data": quote,
            }), 200
        except Exception as e:
            logger.error(f"Error getting quote: {e}")
            return jsonify({
                "success": False,
                "message": str(e),
            }), 500

    # Demo quote
    import random
    price = random.uniform(100, 500)
    return jsonify({
        "success": True,
        "data": {
            "symbol": symbol.upper(),
            "bid": round(price - 0.01, 2),
            "ask": round(price + 0.01, 2),
            "last": round(price, 2),
            "volume": random.randint(1000000, 10000000),
            "timestamp": datetime.now().isoformat(),
        },
    }), 200


@api_routes_bp.route("/market/bars/<symbol>")
def get_bars(symbol: str) -> tuple[dict[str, Any], int]:
    """
    Get OHLCV bars for a symbol.

    Args:
        symbol: Stock symbol

    Returns:
        Bars JSON response
    """
    timeframe = request.args.get("timeframe", "1D")
    limit = request.args.get("limit", 100, type=int)
    start = request.args.get("start")
    end = request.args.get("end")

    data_manager = getattr(g, "data_manager", None)

    if data_manager:
        try:
            bars = data_manager.get_bars(
                symbol=symbol.upper(),
                timeframe=timeframe,
                limit=limit,
                start=start,
                end=end,
            )
            return jsonify({
                "success": True,
                "data": bars,
                "count": len(bars),
            }), 200
        except Exception as e:
            logger.error(f"Error getting bars: {e}")
            return jsonify({
                "success": False,
                "message": str(e),
            }), 500

    # Demo bars
    import random
    bars = []
    current = datetime.now()
    price = random.uniform(100, 500)

    for i in range(min(limit, 100)):
        open_price = price
        high = open_price * random.uniform(1.0, 1.02)
        low = open_price * random.uniform(0.98, 1.0)
        close = random.uniform(low, high)
        price = close

        bars.append({
            "t": (current - timedelta(days=i)).strftime("%Y-%m-%dT00:00:00Z"),
            "o": round(open_price, 2),
            "h": round(high, 2),
            "l": round(low, 2),
            "c": round(close, 2),
            "v": random.randint(1000000, 10000000),
        })

    bars.reverse()

    return jsonify({
        "success": True,
        "data": bars,
        "count": len(bars),
    }), 200


@api_routes_bp.route("/market/snapshot")
def get_market_snapshot() -> tuple[dict[str, Any], int]:
    """
    Get market snapshot for multiple symbols.

    Returns:
        Snapshot JSON response
    """
    symbols = request.args.get("symbols", "").split(",")
    symbols = [s.strip().upper() for s in symbols if s.strip()]

    if not symbols:
        return jsonify({
            "success": False,
            "message": "No symbols provided",
        }), 400

    if len(symbols) > 50:
        return jsonify({
            "success": False,
            "message": "Maximum 50 symbols allowed",
        }), 400

    data_manager = getattr(g, "data_manager", None)

    if data_manager:
        try:
            snapshots = data_manager.get_snapshots(symbols)
            return jsonify({
                "success": True,
                "data": snapshots,
            }), 200
        except Exception as e:
            logger.error(f"Error getting snapshots: {e}")
            return jsonify({
                "success": False,
                "message": str(e),
            }), 500

    # Demo snapshots
    import random
    snapshots = {}
    for symbol in symbols:
        price = random.uniform(100, 500)
        change = random.uniform(-5, 5)
        snapshots[symbol] = {
            "symbol": symbol,
            "price": round(price, 2),
            "change": round(change, 2),
            "change_percent": round(change / price * 100, 2),
            "volume": random.randint(1000000, 10000000),
            "timestamp": datetime.now().isoformat(),
        }

    return jsonify({
        "success": True,
        "data": snapshots,
    }), 200


@api_routes_bp.route("/webhooks/alpaca", methods=["POST"])
def alpaca_webhook() -> tuple[dict[str, Any], int]:
    """
    Handle Alpaca webhook events.

    Returns:
        Acknowledgment JSON response
    """
    # Verify signature if configured
    signature = request.headers.get("X-Alpaca-Signature")
    if signature:
        secret = "your_webhook_secret"  # From config in production
        if not verify_signature(request.data, signature, secret):
            logger.warning("Invalid Alpaca webhook signature")
            return jsonify({"error": "Invalid signature"}), 401

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400

    event_type = data.get("event")
    logger.info(f"Received Alpaca webhook: {event_type}")

    # Process different event types
    if event_type == "fill":
        # Order filled
        logger.info(f"Order filled: {data.get('order', {}).get('id')}")
    elif event_type == "partial_fill":
        # Partial fill
        logger.info(f"Partial fill: {data.get('order', {}).get('id')}")
    elif event_type == "canceled":
        # Order cancelled
        logger.info(f"Order cancelled: {data.get('order', {}).get('id')}")

    return jsonify({"received": True}), 200


@api_routes_bp.route("/webhooks/tradingview", methods=["POST"])
def tradingview_webhook() -> tuple[dict[str, Any], int]:
    """
    Handle TradingView webhook alerts.

    Returns:
        Acknowledgment JSON response
    """
    data = request.get_json()
    if not data:
        # TradingView may send plain text
        data = {"message": request.data.decode()}

    logger.info(f"Received TradingView webhook: {data}")

    # Parse alert message
    message = data.get("message", "")

    # In production, parse and execute trading signals
    # Example: "BUY AAPL 100 @ market"

    return jsonify({
        "received": True,
        "message": message,
        "timestamp": datetime.now().isoformat(),
    }), 200


@api_routes_bp.route("/signals", methods=["POST"])
def receive_signal() -> tuple[dict[str, Any], int]:
    """
    Receive trading signal.

    Returns:
        Signal acknowledgment JSON response
    """
    data = request.get_json()
    if not data:
        return jsonify({
            "success": False,
            "message": "No signal data",
        }), 400

    required = ["symbol", "action"]
    for field in required:
        if field not in data:
            return jsonify({
                "success": False,
                "message": f"Missing field: {field}",
            }), 400

    symbol = data["symbol"].upper()
    action = data["action"].lower()

    if action not in ["buy", "sell", "close"]:
        return jsonify({
            "success": False,
            "message": "Invalid action",
        }), 400

    logger.info(f"Signal received: {action} {symbol}")

    # In production, process signal through trading engine

    return jsonify({
        "success": True,
        "message": f"Signal received: {action} {symbol}",
        "signal_id": f"sig-{datetime.now().strftime('%Y%m%d%H%M%S')}",
    }), 200


@api_routes_bp.route("/activity")
def get_activity() -> tuple[dict[str, Any], int]:
    """
    Get account activity.

    Returns:
        Activity list JSON response
    """
    activity_type = request.args.get("type")
    limit = request.args.get("limit", 50, type=int)

    # Demo activity
    activities = [
        {
            "id": "act-001",
            "type": "FILL",
            "symbol": "AAPL",
            "side": "buy",
            "qty": 50,
            "price": 154.50,
            "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
        },
        {
            "id": "act-002",
            "type": "DIV",
            "symbol": "MSFT",
            "amount": 45.00,
            "timestamp": (datetime.now() - timedelta(days=5)).isoformat(),
        },
        {
            "id": "act-003",
            "type": "FILL",
            "symbol": "GOOGL",
            "side": "sell",
            "qty": 25,
            "price": 143.00,
            "timestamp": (datetime.now() - timedelta(hours=5)).isoformat(),
        },
    ]

    if activity_type:
        activities = [a for a in activities if a["type"] == activity_type]

    return jsonify({
        "success": True,
        "data": activities[:limit],
        "count": len(activities),
    }), 200


@api_routes_bp.route("/performance")
def get_performance() -> tuple[dict[str, Any], int]:
    """
    Get performance metrics.

    Returns:
        Performance JSON response
    """
    period = request.args.get("period", "1M")

    portfolio_manager = getattr(g, "portfolio_manager", None)

    if portfolio_manager:
        try:
            performance = portfolio_manager.get_performance(period=period)
            return jsonify({
                "success": True,
                "data": performance,
            }), 200
        except Exception as e:
            logger.error(f"Error getting performance: {e}")
            return jsonify({
                "success": False,
                "message": str(e),
            }), 500

    # Demo performance
    return jsonify({
        "success": True,
        "data": {
            "period": period,
            "total_return": 0.1364,
            "sharpe_ratio": 1.85,
            "max_drawdown": -0.0823,
            "win_rate": 0.62,
            "profit_factor": 2.1,
        },
    }), 200


# Module version
__version__ = "2.2.0"
