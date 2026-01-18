"""
Authentication Routes for Ultimate Trading Bot v2.2.

This module provides authentication-related routes including:
- Login/logout
- Registration
- Password reset
- Two-factor authentication
- Session management
"""

import logging
from datetime import datetime, timedelta
from typing import Any
import secrets

from flask import Blueprint, render_template, jsonify, request, g, redirect, url_for


logger = logging.getLogger(__name__)

# Create blueprint
auth_routes_bp = Blueprint(
    "auth_routes",
    __name__,
    url_prefix="/auth",
)


def get_client_ip() -> str:
    """Get client IP address."""
    if request.headers.get("X-Forwarded-For"):
        return request.headers["X-Forwarded-For"].split(",")[0].strip()
    return request.remote_addr or "unknown"


def get_user_agent() -> str:
    """Get user agent string."""
    return request.headers.get("User-Agent", "unknown")[:256]


@auth_routes_bp.route("/login", methods=["GET"])
def login_page() -> str:
    """
    Render login page.

    Returns:
        Rendered login template
    """
    next_url = request.args.get("next", "/dashboard")
    error = request.args.get("error")

    return render_template(
        "auth/login.html",
        page_title="Login",
        next_url=next_url,
        error=error,
    )


@auth_routes_bp.route("/login", methods=["POST"])
def login() -> tuple[dict[str, Any], int]:
    """
    Process login request.

    Returns:
        Login result JSON response
    """
    data = request.get_json()
    if not data:
        return jsonify({
            "success": False,
            "message": "No credentials provided",
        }), 400

    username = data.get("username", "").strip()
    password = data.get("password", "")
    remember = data.get("remember", False)
    totp_code = data.get("totp_code")

    if not username or not password:
        return jsonify({
            "success": False,
            "message": "Username and password required",
        }), 400

    auth_manager = getattr(g, "auth_manager", None)

    if auth_manager:
        try:
            # Authenticate user
            user = auth_manager.authenticate(username, password)
            if not user:
                logger.warning(f"Failed login attempt for user: {username}")
                return jsonify({
                    "success": False,
                    "message": "Invalid credentials",
                }), 401

            # Check if 2FA is enabled
            if user.totp_enabled:
                if not totp_code:
                    return jsonify({
                        "success": False,
                        "message": "Two-factor authentication required",
                        "requires_2fa": True,
                    }), 401

                if not auth_manager.verify_totp(user.user_id, totp_code):
                    logger.warning(f"Invalid 2FA code for user: {username}")
                    return jsonify({
                        "success": False,
                        "message": "Invalid two-factor code",
                    }), 401

            # Create session
            session = auth_manager.create_session(
                user_id=user.user_id,
                ip_address=get_client_ip(),
                user_agent=get_user_agent(),
                remember=remember,
            )

            logger.info(f"User logged in: {username}")

            return jsonify({
                "success": True,
                "message": "Login successful",
                "data": {
                    "user_id": user.user_id,
                    "username": user.username,
                    "email": user.email,
                    "token": session.token,
                    "expires_at": session.expires_at.isoformat(),
                },
            }), 200

        except Exception as e:
            logger.error(f"Login error: {e}")
            return jsonify({
                "success": False,
                "message": "Authentication failed",
            }), 500

    # Demo login
    if username == "demo" and password == "demo":
        token = secrets.token_urlsafe(32)
        return jsonify({
            "success": True,
            "message": "Login successful (demo)",
            "data": {
                "user_id": "user-demo",
                "username": "demo",
                "email": "demo@example.com",
                "token": token,
                "expires_at": (datetime.now() + timedelta(hours=24)).isoformat(),
            },
        }), 200

    return jsonify({
        "success": False,
        "message": "Invalid credentials",
    }), 401


@auth_routes_bp.route("/logout", methods=["POST"])
def logout() -> tuple[dict[str, Any], int]:
    """
    Process logout request.

    Returns:
        Logout result JSON response
    """
    auth_header = request.headers.get("Authorization", "")
    token = None

    if auth_header.startswith("Bearer "):
        token = auth_header[7:]

    auth_manager = getattr(g, "auth_manager", None)

    if auth_manager and token:
        try:
            auth_manager.invalidate_session(token)
            logger.info("User logged out")
        except Exception as e:
            logger.error(f"Logout error: {e}")

    return jsonify({
        "success": True,
        "message": "Logged out successfully",
    }), 200


@auth_routes_bp.route("/register", methods=["GET"])
def register_page() -> str:
    """
    Render registration page.

    Returns:
        Rendered registration template
    """
    return render_template(
        "auth/register.html",
        page_title="Register",
    )


@auth_routes_bp.route("/register", methods=["POST"])
def register() -> tuple[dict[str, Any], int]:
    """
    Process registration request.

    Returns:
        Registration result JSON response
    """
    data = request.get_json()
    if not data:
        return jsonify({
            "success": False,
            "message": "No registration data provided",
        }), 400

    username = data.get("username", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    confirm_password = data.get("confirm_password", "")

    # Validation
    if not username or len(username) < 3:
        return jsonify({
            "success": False,
            "message": "Username must be at least 3 characters",
        }), 400

    if not email or "@" not in email:
        return jsonify({
            "success": False,
            "message": "Valid email required",
        }), 400

    if len(password) < 8:
        return jsonify({
            "success": False,
            "message": "Password must be at least 8 characters",
        }), 400

    if password != confirm_password:
        return jsonify({
            "success": False,
            "message": "Passwords do not match",
        }), 400

    auth_manager = getattr(g, "auth_manager", None)

    if auth_manager:
        try:
            # Check if username/email exists
            if auth_manager.user_exists(username=username):
                return jsonify({
                    "success": False,
                    "message": "Username already taken",
                }), 409

            if auth_manager.user_exists(email=email):
                return jsonify({
                    "success": False,
                    "message": "Email already registered",
                }), 409

            # Create user
            user = auth_manager.create_user(
                username=username,
                email=email,
                password=password,
            )

            logger.info(f"User registered: {username}")

            return jsonify({
                "success": True,
                "message": "Registration successful",
                "data": {
                    "user_id": user.user_id,
                    "username": user.username,
                    "email": user.email,
                },
            }), 201

        except Exception as e:
            logger.error(f"Registration error: {e}")
            return jsonify({
                "success": False,
                "message": "Registration failed",
            }), 500

    # Demo registration
    return jsonify({
        "success": True,
        "message": "Registration successful (demo)",
        "data": {
            "user_id": f"user-{secrets.token_hex(8)}",
            "username": username,
            "email": email,
        },
    }), 201


@auth_routes_bp.route("/forgot-password", methods=["GET"])
def forgot_password_page() -> str:
    """
    Render forgot password page.

    Returns:
        Rendered forgot password template
    """
    return render_template(
        "auth/forgot_password.html",
        page_title="Forgot Password",
    )


@auth_routes_bp.route("/forgot-password", methods=["POST"])
def forgot_password() -> tuple[dict[str, Any], int]:
    """
    Process forgot password request.

    Returns:
        Result JSON response
    """
    data = request.get_json()
    if not data:
        return jsonify({
            "success": False,
            "message": "No email provided",
        }), 400

    email = data.get("email", "").strip().lower()

    if not email or "@" not in email:
        return jsonify({
            "success": False,
            "message": "Valid email required",
        }), 400

    auth_manager = getattr(g, "auth_manager", None)

    if auth_manager:
        try:
            # Generate reset token (always return success to prevent enumeration)
            auth_manager.create_password_reset_token(email)
            logger.info(f"Password reset requested for: {email}")
        except Exception as e:
            logger.error(f"Password reset error: {e}")

    # Always return success to prevent email enumeration
    return jsonify({
        "success": True,
        "message": "If an account exists with this email, a reset link has been sent",
    }), 200


@auth_routes_bp.route("/reset-password", methods=["GET"])
def reset_password_page() -> str:
    """
    Render reset password page.

    Returns:
        Rendered reset password template
    """
    token = request.args.get("token", "")

    return render_template(
        "auth/reset_password.html",
        page_title="Reset Password",
        token=token,
    )


@auth_routes_bp.route("/reset-password", methods=["POST"])
def reset_password() -> tuple[dict[str, Any], int]:
    """
    Process password reset.

    Returns:
        Result JSON response
    """
    data = request.get_json()
    if not data:
        return jsonify({
            "success": False,
            "message": "No data provided",
        }), 400

    token = data.get("token", "")
    password = data.get("password", "")
    confirm_password = data.get("confirm_password", "")

    if not token:
        return jsonify({
            "success": False,
            "message": "Reset token required",
        }), 400

    if len(password) < 8:
        return jsonify({
            "success": False,
            "message": "Password must be at least 8 characters",
        }), 400

    if password != confirm_password:
        return jsonify({
            "success": False,
            "message": "Passwords do not match",
        }), 400

    auth_manager = getattr(g, "auth_manager", None)

    if auth_manager:
        try:
            result = auth_manager.reset_password(token, password)
            if not result:
                return jsonify({
                    "success": False,
                    "message": "Invalid or expired reset token",
                }), 400

            logger.info("Password reset successful")

            return jsonify({
                "success": True,
                "message": "Password reset successful",
            }), 200

        except Exception as e:
            logger.error(f"Password reset error: {e}")
            return jsonify({
                "success": False,
                "message": "Password reset failed",
            }), 500

    # Demo reset
    return jsonify({
        "success": True,
        "message": "Password reset successful (demo)",
    }), 200


@auth_routes_bp.route("/api/me")
def get_current_user() -> tuple[dict[str, Any], int]:
    """
    Get current user info.

    Returns:
        Current user JSON response
    """
    auth_header = request.headers.get("Authorization", "")
    token = None

    if auth_header.startswith("Bearer "):
        token = auth_header[7:]

    auth_manager = getattr(g, "auth_manager", None)

    if auth_manager and token:
        try:
            session = auth_manager.validate_session(token)
            if not session:
                return jsonify({
                    "success": False,
                    "message": "Invalid or expired session",
                }), 401

            user = auth_manager.get_user(session.user_id)
            if not user:
                return jsonify({
                    "success": False,
                    "message": "User not found",
                }), 404

            return jsonify({
                "success": True,
                "data": {
                    "user_id": user.user_id,
                    "username": user.username,
                    "email": user.email,
                    "created_at": user.created_at.isoformat(),
                    "totp_enabled": user.totp_enabled,
                    "roles": user.roles,
                },
            }), 200

        except Exception as e:
            logger.error(f"Get current user error: {e}")
            return jsonify({
                "success": False,
                "message": "Authentication error",
            }), 500

    # Demo user
    if token:
        return jsonify({
            "success": True,
            "data": {
                "user_id": "user-demo",
                "username": "demo",
                "email": "demo@example.com",
                "created_at": datetime.now().isoformat(),
                "totp_enabled": False,
                "roles": ["user"],
            },
        }), 200

    return jsonify({
        "success": False,
        "message": "Not authenticated",
    }), 401


@auth_routes_bp.route("/api/change-password", methods=["POST"])
def change_password() -> tuple[dict[str, Any], int]:
    """
    Change user password.

    Returns:
        Result JSON response
    """
    data = request.get_json()
    if not data:
        return jsonify({
            "success": False,
            "message": "No data provided",
        }), 400

    current_password = data.get("current_password", "")
    new_password = data.get("new_password", "")
    confirm_password = data.get("confirm_password", "")

    if not current_password:
        return jsonify({
            "success": False,
            "message": "Current password required",
        }), 400

    if len(new_password) < 8:
        return jsonify({
            "success": False,
            "message": "New password must be at least 8 characters",
        }), 400

    if new_password != confirm_password:
        return jsonify({
            "success": False,
            "message": "Passwords do not match",
        }), 400

    if current_password == new_password:
        return jsonify({
            "success": False,
            "message": "New password must be different from current",
        }), 400

    auth_manager = getattr(g, "auth_manager", None)
    current_user_id = getattr(g, "current_user_id", None)

    if auth_manager and current_user_id:
        try:
            result = auth_manager.change_password(
                user_id=current_user_id,
                current_password=current_password,
                new_password=new_password,
            )

            if not result:
                return jsonify({
                    "success": False,
                    "message": "Current password is incorrect",
                }), 401

            logger.info(f"Password changed for user: {current_user_id}")

            return jsonify({
                "success": True,
                "message": "Password changed successfully",
            }), 200

        except Exception as e:
            logger.error(f"Change password error: {e}")
            return jsonify({
                "success": False,
                "message": "Password change failed",
            }), 500

    # Demo change
    return jsonify({
        "success": True,
        "message": "Password changed successfully (demo)",
    }), 200


@auth_routes_bp.route("/api/enable-2fa", methods=["POST"])
def enable_2fa() -> tuple[dict[str, Any], int]:
    """
    Enable two-factor authentication.

    Returns:
        2FA setup JSON response
    """
    auth_manager = getattr(g, "auth_manager", None)
    current_user_id = getattr(g, "current_user_id", None)

    if auth_manager and current_user_id:
        try:
            setup = auth_manager.setup_totp(current_user_id)

            return jsonify({
                "success": True,
                "message": "2FA setup initiated",
                "data": {
                    "secret": setup["secret"],
                    "qr_code": setup["qr_code"],
                    "backup_codes": setup["backup_codes"],
                },
            }), 200

        except Exception as e:
            logger.error(f"Enable 2FA error: {e}")
            return jsonify({
                "success": False,
                "message": "Failed to enable 2FA",
            }), 500

    # Demo setup
    return jsonify({
        "success": True,
        "message": "2FA setup initiated (demo)",
        "data": {
            "secret": "JBSWY3DPEHPK3PXP",
            "qr_code": "data:image/png;base64,...",
            "backup_codes": [
                "12345678",
                "23456789",
                "34567890",
                "45678901",
                "56789012",
            ],
        },
    }), 200


@auth_routes_bp.route("/api/verify-2fa", methods=["POST"])
def verify_2fa() -> tuple[dict[str, Any], int]:
    """
    Verify 2FA setup.

    Returns:
        Verification result JSON response
    """
    data = request.get_json()
    if not data:
        return jsonify({
            "success": False,
            "message": "No code provided",
        }), 400

    code = data.get("code", "")

    if not code or len(code) != 6:
        return jsonify({
            "success": False,
            "message": "Invalid code format",
        }), 400

    auth_manager = getattr(g, "auth_manager", None)
    current_user_id = getattr(g, "current_user_id", None)

    if auth_manager and current_user_id:
        try:
            result = auth_manager.verify_and_enable_totp(current_user_id, code)

            if not result:
                return jsonify({
                    "success": False,
                    "message": "Invalid verification code",
                }), 400

            logger.info(f"2FA enabled for user: {current_user_id}")

            return jsonify({
                "success": True,
                "message": "2FA enabled successfully",
            }), 200

        except Exception as e:
            logger.error(f"Verify 2FA error: {e}")
            return jsonify({
                "success": False,
                "message": "Failed to verify 2FA",
            }), 500

    # Demo verify
    if code == "123456":
        return jsonify({
            "success": True,
            "message": "2FA enabled successfully (demo)",
        }), 200

    return jsonify({
        "success": False,
        "message": "Invalid verification code",
    }), 400


@auth_routes_bp.route("/api/disable-2fa", methods=["POST"])
def disable_2fa() -> tuple[dict[str, Any], int]:
    """
    Disable two-factor authentication.

    Returns:
        Result JSON response
    """
    data = request.get_json()
    if not data:
        return jsonify({
            "success": False,
            "message": "No password provided",
        }), 400

    password = data.get("password", "")

    if not password:
        return jsonify({
            "success": False,
            "message": "Password required",
        }), 400

    auth_manager = getattr(g, "auth_manager", None)
    current_user_id = getattr(g, "current_user_id", None)

    if auth_manager and current_user_id:
        try:
            result = auth_manager.disable_totp(current_user_id, password)

            if not result:
                return jsonify({
                    "success": False,
                    "message": "Invalid password",
                }), 401

            logger.info(f"2FA disabled for user: {current_user_id}")

            return jsonify({
                "success": True,
                "message": "2FA disabled successfully",
            }), 200

        except Exception as e:
            logger.error(f"Disable 2FA error: {e}")
            return jsonify({
                "success": False,
                "message": "Failed to disable 2FA",
            }), 500

    # Demo disable
    return jsonify({
        "success": True,
        "message": "2FA disabled successfully (demo)",
    }), 200


@auth_routes_bp.route("/api/sessions")
def get_sessions() -> tuple[dict[str, Any], int]:
    """
    Get user sessions.

    Returns:
        Sessions list JSON response
    """
    auth_manager = getattr(g, "auth_manager", None)
    current_user_id = getattr(g, "current_user_id", None)

    if auth_manager and current_user_id:
        try:
            sessions = auth_manager.get_user_sessions(current_user_id)

            return jsonify({
                "success": True,
                "data": sessions,
                "count": len(sessions),
            }), 200

        except Exception as e:
            logger.error(f"Get sessions error: {e}")
            return jsonify({
                "success": False,
                "message": "Failed to get sessions",
            }), 500

    # Demo sessions
    return jsonify({
        "success": True,
        "data": [
            {
                "session_id": "sess-001",
                "ip_address": "192.168.1.1",
                "user_agent": "Chrome on Windows",
                "created_at": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat(),
                "is_current": True,
            },
            {
                "session_id": "sess-002",
                "ip_address": "10.0.0.1",
                "user_agent": "Safari on macOS",
                "created_at": (datetime.now() - timedelta(days=1)).isoformat(),
                "last_activity": (datetime.now() - timedelta(hours=2)).isoformat(),
                "is_current": False,
            },
        ],
        "count": 2,
    }), 200


@auth_routes_bp.route("/api/sessions/<session_id>", methods=["DELETE"])
def revoke_session(session_id: str) -> tuple[dict[str, Any], int]:
    """
    Revoke a session.

    Args:
        session_id: Session ID to revoke

    Returns:
        Result JSON response
    """
    auth_manager = getattr(g, "auth_manager", None)
    current_user_id = getattr(g, "current_user_id", None)

    if auth_manager and current_user_id:
        try:
            result = auth_manager.revoke_session(current_user_id, session_id)

            if not result:
                return jsonify({
                    "success": False,
                    "message": "Session not found",
                }), 404

            logger.info(f"Session revoked: {session_id}")

            return jsonify({
                "success": True,
                "message": "Session revoked successfully",
            }), 200

        except Exception as e:
            logger.error(f"Revoke session error: {e}")
            return jsonify({
                "success": False,
                "message": "Failed to revoke session",
            }), 500

    # Demo revoke
    return jsonify({
        "success": True,
        "message": "Session revoked successfully (demo)",
    }), 200


@auth_routes_bp.route("/api/sessions/revoke-all", methods=["POST"])
def revoke_all_sessions() -> tuple[dict[str, Any], int]:
    """
    Revoke all other sessions.

    Returns:
        Result JSON response
    """
    auth_manager = getattr(g, "auth_manager", None)
    current_user_id = getattr(g, "current_user_id", None)

    if auth_manager and current_user_id:
        try:
            count = auth_manager.revoke_all_sessions(
                current_user_id,
                except_current=True,
            )

            logger.info(f"Revoked {count} sessions for user: {current_user_id}")

            return jsonify({
                "success": True,
                "message": f"Revoked {count} sessions",
                "data": {"revoked_count": count},
            }), 200

        except Exception as e:
            logger.error(f"Revoke all sessions error: {e}")
            return jsonify({
                "success": False,
                "message": "Failed to revoke sessions",
            }), 500

    # Demo revoke all
    return jsonify({
        "success": True,
        "message": "Revoked 1 session (demo)",
        "data": {"revoked_count": 1},
    }), 200


# Module version
__version__ = "2.2.0"
