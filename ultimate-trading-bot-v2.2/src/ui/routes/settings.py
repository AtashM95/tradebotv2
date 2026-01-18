"""
Settings Routes for Ultimate Trading Bot v2.2.

This module provides settings-related routes including:
- User preferences
- Trading configuration
- Notification settings
- API key management
- Theme settings
"""

import logging
from datetime import datetime
from typing import Any
import secrets

from flask import Blueprint, render_template, jsonify, request, g

from ..cache import get_cache


logger = logging.getLogger(__name__)

# Create blueprint
settings_routes_bp = Blueprint(
    "settings_routes",
    __name__,
    url_prefix="/settings",
)


@settings_routes_bp.route("/")
def index() -> str:
    """
    Render settings page.

    Returns:
        Rendered settings template
    """
    return render_template(
        "settings/index.html",
        page_title="Settings",
    )


@settings_routes_bp.route("/profile")
def profile_page() -> str:
    """
    Render profile settings page.

    Returns:
        Rendered profile template
    """
    return render_template(
        "settings/profile.html",
        page_title="Profile Settings",
    )


@settings_routes_bp.route("/trading")
def trading_page() -> str:
    """
    Render trading settings page.

    Returns:
        Rendered trading settings template
    """
    return render_template(
        "settings/trading.html",
        page_title="Trading Settings",
    )


@settings_routes_bp.route("/notifications")
def notifications_page() -> str:
    """
    Render notification settings page.

    Returns:
        Rendered notifications template
    """
    return render_template(
        "settings/notifications.html",
        page_title="Notification Settings",
    )


@settings_routes_bp.route("/api-keys")
def api_keys_page() -> str:
    """
    Render API keys page.

    Returns:
        Rendered API keys template
    """
    return render_template(
        "settings/api_keys.html",
        page_title="API Keys",
    )


@settings_routes_bp.route("/api/profile", methods=["GET"])
def get_profile() -> tuple[dict[str, Any], int]:
    """
    Get user profile settings.

    Returns:
        Profile settings JSON response
    """
    current_user_id = getattr(g, "current_user_id", "demo")

    # In production, fetch from database
    profile = {
        "user_id": current_user_id,
        "username": "demo_user",
        "email": "demo@example.com",
        "full_name": "Demo User",
        "timezone": "America/New_York",
        "date_format": "MM/DD/YYYY",
        "time_format": "12h",
        "language": "en",
        "currency": "USD",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": datetime.now().isoformat(),
    }

    return jsonify({
        "success": True,
        "data": profile,
    }), 200


@settings_routes_bp.route("/api/profile", methods=["PUT"])
def update_profile() -> tuple[dict[str, Any], int]:
    """
    Update user profile settings.

    Returns:
        Update result JSON response
    """
    data = request.get_json()
    if not data:
        return jsonify({
            "success": False,
            "message": "No data provided",
        }), 400

    # Validate allowed fields
    allowed_fields = {
        "full_name", "timezone", "date_format", "time_format",
        "language", "currency",
    }

    updates = {k: v for k, v in data.items() if k in allowed_fields}

    if not updates:
        return jsonify({
            "success": False,
            "message": "No valid fields to update",
        }), 400

    # Validate timezone
    if "timezone" in updates:
        valid_timezones = [
            "America/New_York", "America/Chicago", "America/Denver",
            "America/Los_Angeles", "Europe/London", "Europe/Berlin",
            "Asia/Tokyo", "Asia/Shanghai", "Australia/Sydney", "UTC",
        ]
        if updates["timezone"] not in valid_timezones:
            return jsonify({
                "success": False,
                "message": "Invalid timezone",
            }), 400

    # In production, update database
    logger.info(f"Profile updated: {updates}")

    return jsonify({
        "success": True,
        "message": "Profile updated successfully",
        "data": updates,
    }), 200


@settings_routes_bp.route("/api/trading-preferences", methods=["GET"])
def get_trading_preferences() -> tuple[dict[str, Any], int]:
    """
    Get trading preferences.

    Returns:
        Trading preferences JSON response
    """
    preferences = {
        "default_order_type": "limit",
        "default_time_in_force": "day",
        "confirm_orders": True,
        "show_order_preview": True,
        "default_quantity": 100,
        "max_position_size": 10000.00,
        "max_daily_loss": 1000.00,
        "auto_stop_loss": True,
        "default_stop_loss_percent": 5.0,
        "auto_take_profit": False,
        "default_take_profit_percent": 10.0,
        "extended_hours_trading": False,
        "paper_trading_mode": False,
        "risk_level": "moderate",
        "alerts_enabled": True,
    }

    return jsonify({
        "success": True,
        "data": preferences,
    }), 200


@settings_routes_bp.route("/api/trading-preferences", methods=["PUT"])
def update_trading_preferences() -> tuple[dict[str, Any], int]:
    """
    Update trading preferences.

    Returns:
        Update result JSON response
    """
    data = request.get_json()
    if not data:
        return jsonify({
            "success": False,
            "message": "No data provided",
        }), 400

    # Validate settings
    allowed_fields = {
        "default_order_type", "default_time_in_force", "confirm_orders",
        "show_order_preview", "default_quantity", "max_position_size",
        "max_daily_loss", "auto_stop_loss", "default_stop_loss_percent",
        "auto_take_profit", "default_take_profit_percent",
        "extended_hours_trading", "paper_trading_mode", "risk_level",
        "alerts_enabled",
    }

    updates = {k: v for k, v in data.items() if k in allowed_fields}

    if not updates:
        return jsonify({
            "success": False,
            "message": "No valid fields to update",
        }), 400

    # Validate order type
    if "default_order_type" in updates:
        valid_types = ["market", "limit", "stop", "stop_limit"]
        if updates["default_order_type"] not in valid_types:
            return jsonify({
                "success": False,
                "message": "Invalid order type",
            }), 400

    # Validate risk level
    if "risk_level" in updates:
        valid_levels = ["conservative", "moderate", "aggressive"]
        if updates["risk_level"] not in valid_levels:
            return jsonify({
                "success": False,
                "message": "Invalid risk level",
            }), 400

    # In production, update database
    logger.info(f"Trading preferences updated: {updates}")

    # Invalidate related cache
    cache = get_cache()
    cache.delete_by_tag("trading")

    return jsonify({
        "success": True,
        "message": "Trading preferences updated successfully",
        "data": updates,
    }), 200


@settings_routes_bp.route("/api/notification-settings", methods=["GET"])
def get_notification_settings() -> tuple[dict[str, Any], int]:
    """
    Get notification settings.

    Returns:
        Notification settings JSON response
    """
    settings = {
        "email": {
            "enabled": True,
            "address": "demo@example.com",
            "trade_executions": True,
            "order_fills": True,
            "price_alerts": True,
            "daily_summary": True,
            "weekly_report": True,
            "account_updates": True,
        },
        "push": {
            "enabled": True,
            "trade_executions": True,
            "order_fills": True,
            "price_alerts": True,
            "market_news": False,
        },
        "sms": {
            "enabled": False,
            "phone_number": None,
            "critical_alerts_only": True,
        },
        "telegram": {
            "enabled": False,
            "chat_id": None,
        },
        "slack": {
            "enabled": False,
            "webhook_url": None,
            "channel": None,
        },
        "quiet_hours": {
            "enabled": True,
            "start": "22:00",
            "end": "08:00",
            "timezone": "America/New_York",
            "allow_critical": True,
        },
    }

    return jsonify({
        "success": True,
        "data": settings,
    }), 200


@settings_routes_bp.route("/api/notification-settings", methods=["PUT"])
def update_notification_settings() -> tuple[dict[str, Any], int]:
    """
    Update notification settings.

    Returns:
        Update result JSON response
    """
    data = request.get_json()
    if not data:
        return jsonify({
            "success": False,
            "message": "No data provided",
        }), 400

    # Validate channel
    channel = data.get("channel")
    valid_channels = ["email", "push", "sms", "telegram", "slack", "quiet_hours"]

    if channel and channel not in valid_channels:
        return jsonify({
            "success": False,
            "message": "Invalid notification channel",
        }), 400

    # In production, update database
    logger.info(f"Notification settings updated: {data}")

    return jsonify({
        "success": True,
        "message": "Notification settings updated successfully",
    }), 200


@settings_routes_bp.route("/api/api-keys", methods=["GET"])
def get_api_keys() -> tuple[dict[str, Any], int]:
    """
    Get API keys list.

    Returns:
        API keys list JSON response
    """
    # In production, fetch from database
    api_keys = [
        {
            "key_id": "key-001",
            "name": "Trading Bot",
            "prefix": "sk_live_xxxx1234",
            "permissions": ["read", "trade"],
            "created_at": "2024-01-15T10:00:00Z",
            "last_used": "2024-01-20T15:30:00Z",
            "status": "active",
        },
        {
            "key_id": "key-002",
            "name": "Read-only Access",
            "prefix": "sk_live_xxxx5678",
            "permissions": ["read"],
            "created_at": "2024-01-10T08:00:00Z",
            "last_used": "2024-01-19T12:00:00Z",
            "status": "active",
        },
    ]

    return jsonify({
        "success": True,
        "data": api_keys,
        "count": len(api_keys),
    }), 200


@settings_routes_bp.route("/api/api-keys", methods=["POST"])
def create_api_key() -> tuple[dict[str, Any], int]:
    """
    Create new API key.

    Returns:
        New API key JSON response
    """
    data = request.get_json()
    if not data:
        return jsonify({
            "success": False,
            "message": "No data provided",
        }), 400

    name = data.get("name", "").strip()
    permissions = data.get("permissions", ["read"])

    if not name:
        return jsonify({
            "success": False,
            "message": "Key name required",
        }), 400

    # Validate permissions
    valid_permissions = ["read", "trade", "withdraw", "admin"]
    for perm in permissions:
        if perm not in valid_permissions:
            return jsonify({
                "success": False,
                "message": f"Invalid permission: {perm}",
            }), 400

    # Generate API key
    key_id = f"key-{secrets.token_hex(8)}"
    api_key = f"sk_live_{secrets.token_urlsafe(32)}"

    # In production, store hashed key in database
    logger.info(f"API key created: {key_id}")

    return jsonify({
        "success": True,
        "message": "API key created successfully",
        "data": {
            "key_id": key_id,
            "name": name,
            "api_key": api_key,  # Only shown once!
            "permissions": permissions,
            "created_at": datetime.now().isoformat(),
        },
        "warning": "Save this API key now. It won't be shown again!",
    }), 201


@settings_routes_bp.route("/api/api-keys/<key_id>", methods=["DELETE"])
def delete_api_key(key_id: str) -> tuple[dict[str, Any], int]:
    """
    Delete API key.

    Args:
        key_id: API key ID to delete

    Returns:
        Deletion result JSON response
    """
    # In production, delete from database
    logger.info(f"API key deleted: {key_id}")

    return jsonify({
        "success": True,
        "message": "API key deleted successfully",
    }), 200


@settings_routes_bp.route("/api/api-keys/<key_id>", methods=["PUT"])
def update_api_key(key_id: str) -> tuple[dict[str, Any], int]:
    """
    Update API key.

    Args:
        key_id: API key ID to update

    Returns:
        Update result JSON response
    """
    data = request.get_json()
    if not data:
        return jsonify({
            "success": False,
            "message": "No data provided",
        }), 400

    name = data.get("name")
    permissions = data.get("permissions")
    status = data.get("status")

    updates = {}
    if name:
        updates["name"] = name.strip()
    if permissions:
        valid_permissions = ["read", "trade", "withdraw", "admin"]
        for perm in permissions:
            if perm not in valid_permissions:
                return jsonify({
                    "success": False,
                    "message": f"Invalid permission: {perm}",
                }), 400
        updates["permissions"] = permissions
    if status in ["active", "disabled"]:
        updates["status"] = status

    if not updates:
        return jsonify({
            "success": False,
            "message": "No valid fields to update",
        }), 400

    # In production, update database
    logger.info(f"API key updated: {key_id} - {updates}")

    return jsonify({
        "success": True,
        "message": "API key updated successfully",
        "data": updates,
    }), 200


@settings_routes_bp.route("/api/theme", methods=["GET"])
def get_theme() -> tuple[dict[str, Any], int]:
    """
    Get current theme settings.

    Returns:
        Theme settings JSON response
    """
    user_id = getattr(g, "current_user_id", "default")
    theme_manager = getattr(g, "theme_manager", None)

    if theme_manager:
        theme = theme_manager.get_user_theme(user_id)
        return jsonify({
            "success": True,
            "data": {
                "name": theme.name,
                "mode": theme.mode.value,
                "colors": theme.colors.__dict__,
            },
        }), 200

    # Default theme
    return jsonify({
        "success": True,
        "data": {
            "name": "Dark",
            "mode": "dark",
            "primary_color": "#3b82f6",
            "accent_color": "#10b981",
        },
    }), 200


@settings_routes_bp.route("/api/theme", methods=["PUT"])
def update_theme() -> tuple[dict[str, Any], int]:
    """
    Update theme settings.

    Returns:
        Update result JSON response
    """
    data = request.get_json()
    if not data:
        return jsonify({
            "success": False,
            "message": "No data provided",
        }), 400

    theme_name = data.get("theme")
    valid_themes = ["light", "dark", "nord", "solarized", "dracula"]

    if theme_name and theme_name.lower() not in valid_themes:
        return jsonify({
            "success": False,
            "message": f"Invalid theme. Choose from: {', '.join(valid_themes)}",
        }), 400

    user_id = getattr(g, "current_user_id", "default")
    theme_manager = getattr(g, "theme_manager", None)

    if theme_manager and theme_name:
        theme_manager.set_user_theme(user_id, theme_name.lower())

    logger.info(f"Theme updated to: {theme_name}")

    return jsonify({
        "success": True,
        "message": f"Theme changed to {theme_name}",
    }), 200


@settings_routes_bp.route("/api/export-settings", methods=["GET"])
def export_settings() -> tuple[dict[str, Any], int]:
    """
    Export all user settings.

    Returns:
        All settings JSON response
    """
    settings = {
        "profile": {
            "timezone": "America/New_York",
            "language": "en",
            "currency": "USD",
        },
        "trading": {
            "default_order_type": "limit",
            "confirm_orders": True,
            "risk_level": "moderate",
        },
        "notifications": {
            "email_enabled": True,
            "push_enabled": True,
        },
        "theme": {
            "name": "dark",
        },
        "exported_at": datetime.now().isoformat(),
    }

    return jsonify({
        "success": True,
        "data": settings,
    }), 200


@settings_routes_bp.route("/api/import-settings", methods=["POST"])
def import_settings() -> tuple[dict[str, Any], int]:
    """
    Import user settings.

    Returns:
        Import result JSON response
    """
    data = request.get_json()
    if not data:
        return jsonify({
            "success": False,
            "message": "No settings data provided",
        }), 400

    # Validate settings structure
    valid_sections = ["profile", "trading", "notifications", "theme"]
    imported_sections = []

    for section in valid_sections:
        if section in data:
            imported_sections.append(section)
            # In production, validate and apply each section

    if not imported_sections:
        return jsonify({
            "success": False,
            "message": "No valid settings sections found",
        }), 400

    logger.info(f"Settings imported: {imported_sections}")

    return jsonify({
        "success": True,
        "message": f"Imported settings for: {', '.join(imported_sections)}",
        "data": {"imported_sections": imported_sections},
    }), 200


@settings_routes_bp.route("/api/reset-settings", methods=["POST"])
def reset_settings() -> tuple[dict[str, Any], int]:
    """
    Reset settings to defaults.

    Returns:
        Reset result JSON response
    """
    data = request.get_json() or {}
    section = data.get("section", "all")

    valid_sections = ["profile", "trading", "notifications", "theme", "all"]

    if section not in valid_sections:
        return jsonify({
            "success": False,
            "message": f"Invalid section. Choose from: {', '.join(valid_sections)}",
        }), 400

    # In production, reset to defaults
    logger.info(f"Settings reset: {section}")

    # Invalidate cache
    cache = get_cache()
    cache.delete_by_tag("settings")

    return jsonify({
        "success": True,
        "message": f"Settings reset to defaults: {section}",
    }), 200


# Module version
__version__ = "2.2.0"
