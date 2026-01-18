"""
Trading Routes for Ultimate Trading Bot v2.2.

This module provides trading-related routes including:
- Order placement and management
- Order history
- Trade execution
- Position management
"""

import logging
from datetime import datetime, timedelta
from typing import Any
from enum import Enum

from flask import Blueprint, render_template, jsonify, request, g


logger = logging.getLogger(__name__)

# Create blueprint
trading_routes_bp = Blueprint(
    "trading_routes",
    __name__,
    url_prefix="/trading",
)


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


class TimeInForce(str, Enum):
    """Time in force enumeration."""
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"
    OPG = "opg"
    CLS = "cls"


def validate_order_request(data: dict[str, Any]) -> tuple[bool, str | None]:
    """
    Validate order request data.

    Args:
        data: Order request data

    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ["symbol", "side", "quantity"]
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"

    # Validate symbol
    symbol = data.get("symbol", "")
    if not symbol or len(symbol) > 10:
        return False, "Invalid symbol"

    # Validate side
    side = data.get("side", "").lower()
    if side not in ["buy", "sell"]:
        return False, "Side must be 'buy' or 'sell'"

    # Validate quantity
    try:
        quantity = float(data.get("quantity", 0))
        if quantity <= 0:
            return False, "Quantity must be positive"
    except (ValueError, TypeError):
        return False, "Invalid quantity"

    # Validate order type
    order_type = data.get("type", "market").lower()
    valid_types = ["market", "limit", "stop", "stop_limit", "trailing_stop"]
    if order_type not in valid_types:
        return False, f"Invalid order type. Must be one of: {', '.join(valid_types)}"

    # Validate limit price for limit orders
    if order_type in ["limit", "stop_limit"]:
        if "limit_price" not in data:
            return False, "Limit price required for limit orders"
        try:
            limit_price = float(data.get("limit_price", 0))
            if limit_price <= 0:
                return False, "Limit price must be positive"
        except (ValueError, TypeError):
            return False, "Invalid limit price"

    # Validate stop price for stop orders
    if order_type in ["stop", "stop_limit"]:
        if "stop_price" not in data:
            return False, "Stop price required for stop orders"
        try:
            stop_price = float(data.get("stop_price", 0))
            if stop_price <= 0:
                return False, "Stop price must be positive"
        except (ValueError, TypeError):
            return False, "Invalid stop price"

    return True, None


@trading_routes_bp.route("/")
def index() -> str:
    """
    Render trading page.

    Returns:
        Rendered trading template
    """
    return render_template(
        "trading/index.html",
        page_title="Trading",
    )


@trading_routes_bp.route("/order-form")
def order_form() -> str:
    """
    Render order form page.

    Returns:
        Rendered order form template
    """
    symbol = request.args.get("symbol", "")
    side = request.args.get("side", "buy")

    return render_template(
        "trading/order_form.html",
        symbol=symbol,
        side=side,
        page_title="Place Order",
    )


@trading_routes_bp.route("/api/place-order", methods=["POST"])
def place_order() -> tuple[dict[str, Any], int]:
    """
    Place a new order.

    Returns:
        Order confirmation JSON response
    """
    data = request.get_json()
    if not data:
        return jsonify({
            "success": False,
            "message": "No order data provided",
        }), 400

    # Validate order
    is_valid, error = validate_order_request(data)
    if not is_valid:
        return jsonify({
            "success": False,
            "message": error,
        }), 400

    trading_engine = getattr(g, "trading_engine", None)

    if trading_engine:
        try:
            order = trading_engine.place_order(
                symbol=data["symbol"].upper(),
                side=data["side"].lower(),
                quantity=float(data["quantity"]),
                order_type=data.get("type", "market").lower(),
                limit_price=data.get("limit_price"),
                stop_price=data.get("stop_price"),
                time_in_force=data.get("time_in_force", "day").lower(),
                extended_hours=data.get("extended_hours", False),
                client_order_id=data.get("client_order_id"),
            )

            logger.info(f"Order placed: {order.get('order_id')}")

            return jsonify({
                "success": True,
                "message": "Order placed successfully",
                "data": order,
            }), 201

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return jsonify({
                "success": False,
                "message": str(e),
            }), 500

    # Demo response
    order_id = f"ord-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    order = {
        "order_id": order_id,
        "client_order_id": data.get("client_order_id"),
        "symbol": data["symbol"].upper(),
        "side": data["side"].lower(),
        "type": data.get("type", "market").lower(),
        "quantity": float(data["quantity"]),
        "limit_price": data.get("limit_price"),
        "stop_price": data.get("stop_price"),
        "time_in_force": data.get("time_in_force", "day").lower(),
        "status": "pending" if data.get("type", "market") != "market" else "filled",
        "filled_quantity": float(data["quantity"]) if data.get("type", "market") == "market" else 0,
        "filled_avg_price": 155.00 if data.get("type", "market") == "market" else None,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }

    return jsonify({
        "success": True,
        "message": "Order placed successfully (demo)",
        "data": order,
    }), 201


@trading_routes_bp.route("/api/cancel-order/<order_id>", methods=["DELETE"])
def cancel_order(order_id: str) -> tuple[dict[str, Any], int]:
    """
    Cancel an order.

    Args:
        order_id: Order ID to cancel

    Returns:
        Cancellation status JSON response
    """
    if not order_id:
        return jsonify({
            "success": False,
            "message": "Order ID required",
        }), 400

    trading_engine = getattr(g, "trading_engine", None)

    if trading_engine:
        try:
            result = trading_engine.cancel_order(order_id)
            logger.info(f"Order cancelled: {order_id}")

            return jsonify({
                "success": True,
                "message": "Order cancelled successfully",
                "data": result,
            }), 200

        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return jsonify({
                "success": False,
                "message": str(e),
            }), 500

    # Demo response
    return jsonify({
        "success": True,
        "message": "Order cancelled successfully (demo)",
        "data": {
            "order_id": order_id,
            "status": "cancelled",
            "cancelled_at": datetime.now().isoformat(),
        },
    }), 200


@trading_routes_bp.route("/api/cancel-all-orders", methods=["DELETE"])
def cancel_all_orders() -> tuple[dict[str, Any], int]:
    """
    Cancel all open orders.

    Returns:
        Cancellation status JSON response
    """
    trading_engine = getattr(g, "trading_engine", None)

    if trading_engine:
        try:
            result = trading_engine.cancel_all_orders()
            logger.info(f"All orders cancelled: {result.get('cancelled_count', 0)}")

            return jsonify({
                "success": True,
                "message": "All orders cancelled successfully",
                "data": result,
            }), 200

        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            return jsonify({
                "success": False,
                "message": str(e),
            }), 500

    # Demo response
    return jsonify({
        "success": True,
        "message": "All orders cancelled (demo)",
        "data": {
            "cancelled_count": 2,
            "cancelled_at": datetime.now().isoformat(),
        },
    }), 200


@trading_routes_bp.route("/api/orders")
def get_orders() -> tuple[dict[str, Any], int]:
    """
    Get orders list.

    Returns:
        Orders list JSON response
    """
    status = request.args.get("status", "all")
    limit = request.args.get("limit", 50, type=int)
    offset = request.args.get("offset", 0, type=int)

    trading_engine = getattr(g, "trading_engine", None)

    if trading_engine:
        try:
            orders = trading_engine.get_orders(
                status=status if status != "all" else None,
                limit=limit,
                offset=offset,
            )

            return jsonify({
                "success": True,
                "data": orders,
                "count": len(orders),
                "offset": offset,
                "limit": limit,
            }), 200

        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return jsonify({
                "success": False,
                "message": str(e),
            }), 500

    # Demo data
    orders = [
        {
            "order_id": "ord-001",
            "symbol": "AAPL",
            "side": "buy",
            "type": "limit",
            "quantity": 50,
            "limit_price": 152.00,
            "filled_quantity": 0,
            "status": "pending",
            "created_at": (datetime.now() - timedelta(hours=2)).isoformat(),
        },
        {
            "order_id": "ord-002",
            "symbol": "GOOGL",
            "side": "sell",
            "type": "market",
            "quantity": 25,
            "filled_quantity": 25,
            "filled_avg_price": 143.50,
            "status": "filled",
            "created_at": (datetime.now() - timedelta(hours=5)).isoformat(),
            "filled_at": (datetime.now() - timedelta(hours=5)).isoformat(),
        },
        {
            "order_id": "ord-003",
            "symbol": "MSFT",
            "side": "buy",
            "type": "stop_limit",
            "quantity": 30,
            "stop_price": 415.00,
            "limit_price": 416.00,
            "filled_quantity": 0,
            "status": "pending",
            "created_at": (datetime.now() - timedelta(days=1)).isoformat(),
        },
    ]

    if status != "all":
        orders = [o for o in orders if o["status"] == status]

    return jsonify({
        "success": True,
        "data": orders[offset:offset + limit],
        "count": len(orders),
        "offset": offset,
        "limit": limit,
    }), 200


@trading_routes_bp.route("/api/orders/<order_id>")
def get_order(order_id: str) -> tuple[dict[str, Any], int]:
    """
    Get order details.

    Args:
        order_id: Order ID

    Returns:
        Order details JSON response
    """
    trading_engine = getattr(g, "trading_engine", None)

    if trading_engine:
        try:
            order = trading_engine.get_order(order_id)
            if not order:
                return jsonify({
                    "success": False,
                    "message": "Order not found",
                }), 404

            return jsonify({
                "success": True,
                "data": order,
            }), 200

        except Exception as e:
            logger.error(f"Error getting order {order_id}: {e}")
            return jsonify({
                "success": False,
                "message": str(e),
            }), 500

    # Demo data
    order = {
        "order_id": order_id,
        "symbol": "AAPL",
        "side": "buy",
        "type": "limit",
        "quantity": 50,
        "limit_price": 152.00,
        "filled_quantity": 0,
        "status": "pending",
        "time_in_force": "day",
        "extended_hours": False,
        "created_at": (datetime.now() - timedelta(hours=2)).isoformat(),
        "updated_at": (datetime.now() - timedelta(hours=2)).isoformat(),
    }

    return jsonify({
        "success": True,
        "data": order,
    }), 200


@trading_routes_bp.route("/api/modify-order/<order_id>", methods=["PUT"])
def modify_order(order_id: str) -> tuple[dict[str, Any], int]:
    """
    Modify an existing order.

    Args:
        order_id: Order ID to modify

    Returns:
        Modified order JSON response
    """
    data = request.get_json()
    if not data:
        return jsonify({
            "success": False,
            "message": "No modification data provided",
        }), 400

    trading_engine = getattr(g, "trading_engine", None)

    if trading_engine:
        try:
            order = trading_engine.modify_order(
                order_id=order_id,
                quantity=data.get("quantity"),
                limit_price=data.get("limit_price"),
                stop_price=data.get("stop_price"),
                time_in_force=data.get("time_in_force"),
            )

            logger.info(f"Order modified: {order_id}")

            return jsonify({
                "success": True,
                "message": "Order modified successfully",
                "data": order,
            }), 200

        except Exception as e:
            logger.error(f"Error modifying order {order_id}: {e}")
            return jsonify({
                "success": False,
                "message": str(e),
            }), 500

    # Demo response
    return jsonify({
        "success": True,
        "message": "Order modified successfully (demo)",
        "data": {
            "order_id": order_id,
            "quantity": data.get("quantity"),
            "limit_price": data.get("limit_price"),
            "stop_price": data.get("stop_price"),
            "updated_at": datetime.now().isoformat(),
        },
    }), 200


@trading_routes_bp.route("/api/trades")
def get_trades() -> tuple[dict[str, Any], int]:
    """
    Get trade history.

    Returns:
        Trade history JSON response
    """
    symbol = request.args.get("symbol")
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    limit = request.args.get("limit", 100, type=int)
    offset = request.args.get("offset", 0, type=int)

    trading_engine = getattr(g, "trading_engine", None)

    if trading_engine:
        try:
            trades = trading_engine.get_trades(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
                offset=offset,
            )

            return jsonify({
                "success": True,
                "data": trades,
                "count": len(trades),
                "offset": offset,
                "limit": limit,
            }), 200

        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return jsonify({
                "success": False,
                "message": str(e),
            }), 500

    # Demo data
    trades = [
        {
            "trade_id": "trd-001",
            "order_id": "ord-001",
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 50,
            "price": 154.50,
            "total": 7725.00,
            "commission": 0.00,
            "executed_at": (datetime.now() - timedelta(hours=2)).isoformat(),
        },
        {
            "trade_id": "trd-002",
            "order_id": "ord-002",
            "symbol": "GOOGL",
            "side": "sell",
            "quantity": 25,
            "price": 143.00,
            "total": 3575.00,
            "commission": 0.00,
            "pnl": 75.00,
            "executed_at": (datetime.now() - timedelta(hours=5)).isoformat(),
        },
        {
            "trade_id": "trd-003",
            "order_id": "ord-003",
            "symbol": "TSLA",
            "side": "buy",
            "quantity": 30,
            "price": 245.00,
            "total": 7350.00,
            "commission": 0.00,
            "executed_at": (datetime.now() - timedelta(days=1)).isoformat(),
        },
        {
            "trade_id": "trd-004",
            "order_id": "ord-004",
            "symbol": "MSFT",
            "side": "sell",
            "quantity": 20,
            "price": 412.00,
            "total": 8240.00,
            "commission": 0.00,
            "pnl": 240.00,
            "executed_at": (datetime.now() - timedelta(days=2)).isoformat(),
        },
    ]

    if symbol:
        trades = [t for t in trades if t["symbol"] == symbol.upper()]

    return jsonify({
        "success": True,
        "data": trades[offset:offset + limit],
        "count": len(trades),
        "offset": offset,
        "limit": limit,
    }), 200


@trading_routes_bp.route("/api/quote/<symbol>")
def get_quote(symbol: str) -> tuple[dict[str, Any], int]:
    """
    Get real-time quote for a symbol.

    Args:
        symbol: Stock symbol

    Returns:
        Quote data JSON response
    """
    if not symbol:
        return jsonify({
            "success": False,
            "message": "Symbol required",
        }), 400

    data_manager = getattr(g, "data_manager", None)

    if data_manager:
        try:
            quote = data_manager.get_quote(symbol.upper())

            return jsonify({
                "success": True,
                "data": quote,
                "timestamp": datetime.now().isoformat(),
            }), 200

        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return jsonify({
                "success": False,
                "message": str(e),
            }), 500

    # Demo data
    import random
    base_price = 150.00
    change = random.uniform(-5, 5)

    quote = {
        "symbol": symbol.upper(),
        "bid": round(base_price - 0.01, 2),
        "ask": round(base_price + 0.01, 2),
        "bid_size": random.randint(100, 1000),
        "ask_size": random.randint(100, 1000),
        "last": round(base_price, 2),
        "last_size": random.randint(10, 100),
        "volume": random.randint(1000000, 10000000),
        "high": round(base_price + abs(change) + 1, 2),
        "low": round(base_price - abs(change) - 1, 2),
        "open": round(base_price - change, 2),
        "close": round(base_price - change, 2),
        "change": round(change, 2),
        "change_percent": round(change / base_price * 100, 2),
        "timestamp": datetime.now().isoformat(),
    }

    return jsonify({
        "success": True,
        "data": quote,
        "timestamp": datetime.now().isoformat(),
    }), 200


@trading_routes_bp.route("/api/buying-power")
def get_buying_power() -> tuple[dict[str, Any], int]:
    """
    Get current buying power.

    Returns:
        Buying power JSON response
    """
    portfolio_manager = getattr(g, "portfolio_manager", None)

    if portfolio_manager:
        try:
            buying_power = portfolio_manager.get_buying_power()

            return jsonify({
                "success": True,
                "data": buying_power,
            }), 200

        except Exception as e:
            logger.error(f"Error getting buying power: {e}")
            return jsonify({
                "success": False,
                "message": str(e),
            }), 500

    # Demo data
    return jsonify({
        "success": True,
        "data": {
            "buying_power": 50000.00,
            "cash": 25000.00,
            "margin_available": 25000.00,
            "margin_used": 0.00,
            "day_trades_remaining": 3,
            "pattern_day_trader": False,
        },
    }), 200


@trading_routes_bp.route("/api/positions/<symbol>/close", methods=["POST"])
def close_position(symbol: str) -> tuple[dict[str, Any], int]:
    """
    Close position for a symbol.

    Args:
        symbol: Stock symbol

    Returns:
        Close order JSON response
    """
    if not symbol:
        return jsonify({
            "success": False,
            "message": "Symbol required",
        }), 400

    data = request.get_json() or {}
    quantity = data.get("quantity")  # None = close all

    trading_engine = getattr(g, "trading_engine", None)

    if trading_engine:
        try:
            order = trading_engine.close_position(
                symbol=symbol.upper(),
                quantity=quantity,
            )

            logger.info(f"Position closed for {symbol}")

            return jsonify({
                "success": True,
                "message": f"Position closed for {symbol}",
                "data": order,
            }), 200

        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            return jsonify({
                "success": False,
                "message": str(e),
            }), 500

    # Demo response
    return jsonify({
        "success": True,
        "message": f"Position closed for {symbol} (demo)",
        "data": {
            "order_id": f"ord-close-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "symbol": symbol.upper(),
            "side": "sell",
            "type": "market",
            "quantity": quantity or 100,
            "status": "filled",
            "filled_avg_price": 155.00,
            "executed_at": datetime.now().isoformat(),
        },
    }), 200


@trading_routes_bp.route("/api/close-all-positions", methods=["POST"])
def close_all_positions() -> tuple[dict[str, Any], int]:
    """
    Close all positions.

    Returns:
        Close status JSON response
    """
    trading_engine = getattr(g, "trading_engine", None)

    if trading_engine:
        try:
            result = trading_engine.close_all_positions()
            logger.info(f"All positions closed: {result.get('closed_count', 0)}")

            return jsonify({
                "success": True,
                "message": "All positions closed successfully",
                "data": result,
            }), 200

        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return jsonify({
                "success": False,
                "message": str(e),
            }), 500

    # Demo response
    return jsonify({
        "success": True,
        "message": "All positions closed (demo)",
        "data": {
            "closed_count": 3,
            "closed_at": datetime.now().isoformat(),
        },
    }), 200


# Module version
__version__ = "2.2.0"
