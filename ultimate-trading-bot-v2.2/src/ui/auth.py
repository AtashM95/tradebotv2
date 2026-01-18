"""
Authentication Module for Ultimate Trading Bot v2.2.

This module provides authentication functionality including:
- User authentication
- Session management
- Password hashing
- Token generation
- Two-factor authentication
"""

import hashlib
import hmac
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from flask import session, request


logger = logging.getLogger(__name__)


@dataclass
class User:
    """User data class."""

    id: str
    username: str
    email: str
    password_hash: str
    is_admin: bool = False
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_login: datetime | None = None
    two_factor_secret: str | None = None
    two_factor_enabled: bool = False
    preferences: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for session storage."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "is_admin": self.is_admin,
            "is_active": self.is_active,
            "two_factor_enabled": self.two_factor_enabled,
            "preferences": self.preferences,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "User":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            username=data.get("username", ""),
            email=data.get("email", ""),
            password_hash=data.get("password_hash", ""),
            is_admin=data.get("is_admin", False),
            is_active=data.get("is_active", True),
            two_factor_enabled=data.get("two_factor_enabled", False),
            preferences=data.get("preferences", {}),
        )


class PasswordHasher:
    """Secure password hashing."""

    # Number of iterations for PBKDF2
    ITERATIONS = 600000
    # Salt length
    SALT_LENGTH = 32
    # Hash length
    HASH_LENGTH = 64

    @classmethod
    def hash_password(cls, password: str) -> str:
        """
        Hash a password using PBKDF2.

        Args:
            password: Plain text password

        Returns:
            Hashed password string
        """
        salt = os.urandom(cls.SALT_LENGTH)
        key = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            cls.ITERATIONS,
            dklen=cls.HASH_LENGTH,
        )

        # Format: iterations$salt$hash
        return f"{cls.ITERATIONS}${salt.hex()}${key.hex()}"

    @classmethod
    def verify_password(cls, password: str, hashed: str) -> bool:
        """
        Verify a password against a hash.

        Args:
            password: Plain text password
            hashed: Hashed password

        Returns:
            True if password matches
        """
        try:
            parts = hashed.split("$")
            if len(parts) != 3:
                return False

            iterations = int(parts[0])
            salt = bytes.fromhex(parts[1])
            stored_hash = bytes.fromhex(parts[2])

            key = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                salt,
                iterations,
                dklen=len(stored_hash),
            )

            return hmac.compare_digest(key, stored_hash)
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False

    @classmethod
    def needs_rehash(cls, hashed: str) -> bool:
        """Check if password hash needs to be updated."""
        try:
            parts = hashed.split("$")
            if len(parts) != 3:
                return True

            iterations = int(parts[0])
            return iterations < cls.ITERATIONS
        except Exception:
            return True


class TokenManager:
    """Token generation and validation."""

    # Token expiration times
    ACCESS_TOKEN_EXPIRY = 3600  # 1 hour
    REFRESH_TOKEN_EXPIRY = 604800  # 7 days
    RESET_TOKEN_EXPIRY = 3600  # 1 hour

    def __init__(self, secret_key: str) -> None:
        """Initialize token manager."""
        self._secret_key = secret_key
        self._tokens: dict[str, dict[str, Any]] = {}

    def generate_access_token(self, user_id: str) -> str:
        """Generate an access token."""
        token = secrets.token_urlsafe(32)
        expiry = time.time() + self.ACCESS_TOKEN_EXPIRY

        self._tokens[token] = {
            "type": "access",
            "user_id": user_id,
            "expiry": expiry,
            "created": time.time(),
        }

        return token

    def generate_refresh_token(self, user_id: str) -> str:
        """Generate a refresh token."""
        token = secrets.token_urlsafe(48)
        expiry = time.time() + self.REFRESH_TOKEN_EXPIRY

        self._tokens[token] = {
            "type": "refresh",
            "user_id": user_id,
            "expiry": expiry,
            "created": time.time(),
        }

        return token

    def generate_reset_token(self, user_id: str) -> str:
        """Generate a password reset token."""
        token = secrets.token_urlsafe(32)
        expiry = time.time() + self.RESET_TOKEN_EXPIRY

        self._tokens[token] = {
            "type": "reset",
            "user_id": user_id,
            "expiry": expiry,
            "created": time.time(),
        }

        return token

    def validate_token(self, token: str, token_type: str = "access") -> str | None:
        """
        Validate a token.

        Args:
            token: Token string
            token_type: Expected token type

        Returns:
            User ID if valid, None otherwise
        """
        token_data = self._tokens.get(token)
        if not token_data:
            return None

        if token_data["type"] != token_type:
            return None

        if time.time() > token_data["expiry"]:
            del self._tokens[token]
            return None

        return token_data["user_id"]

    def revoke_token(self, token: str) -> bool:
        """Revoke a token."""
        if token in self._tokens:
            del self._tokens[token]
            return True
        return False

    def revoke_user_tokens(self, user_id: str) -> int:
        """Revoke all tokens for a user."""
        tokens_to_remove = [
            token for token, data in self._tokens.items()
            if data["user_id"] == user_id
        ]

        for token in tokens_to_remove:
            del self._tokens[token]

        return len(tokens_to_remove)

    def cleanup_expired(self) -> int:
        """Remove expired tokens."""
        now = time.time()
        expired = [
            token for token, data in self._tokens.items()
            if now > data["expiry"]
        ]

        for token in expired:
            del self._tokens[token]

        return len(expired)


class TOTPManager:
    """Time-based One-Time Password manager."""

    # TOTP settings
    DIGITS = 6
    INTERVAL = 30
    ALGORITHM = "sha1"

    @classmethod
    def generate_secret(cls) -> str:
        """Generate a new TOTP secret."""
        return secrets.token_hex(20)

    @classmethod
    def generate_code(cls, secret: str, timestamp: int | None = None) -> str:
        """
        Generate TOTP code.

        Args:
            secret: TOTP secret
            timestamp: Optional timestamp (defaults to current time)

        Returns:
            TOTP code
        """
        if timestamp is None:
            timestamp = int(time.time())

        counter = timestamp // cls.INTERVAL
        counter_bytes = counter.to_bytes(8, byteorder="big")

        secret_bytes = bytes.fromhex(secret)
        hmac_hash = hmac.new(secret_bytes, counter_bytes, cls.ALGORITHM).digest()

        offset = hmac_hash[-1] & 0x0F
        truncated = hmac_hash[offset:offset + 4]
        code = int.from_bytes(truncated, byteorder="big") & 0x7FFFFFFF

        return str(code % (10 ** cls.DIGITS)).zfill(cls.DIGITS)

    @classmethod
    def verify_code(
        cls,
        secret: str,
        code: str,
        window: int = 1,
    ) -> bool:
        """
        Verify a TOTP code.

        Args:
            secret: TOTP secret
            code: Code to verify
            window: Number of intervals to check before/after

        Returns:
            True if code is valid
        """
        timestamp = int(time.time())

        for offset in range(-window, window + 1):
            check_time = timestamp + (offset * cls.INTERVAL)
            expected = cls.generate_code(secret, check_time)

            if hmac.compare_digest(code, expected):
                return True

        return False

    @classmethod
    def get_provisioning_uri(
        cls,
        secret: str,
        username: str,
        issuer: str = "TradingBot",
    ) -> str:
        """
        Get provisioning URI for authenticator apps.

        Args:
            secret: TOTP secret
            username: User identifier
            issuer: Service name

        Returns:
            otpauth:// URI
        """
        import base64
        import urllib.parse

        # Convert hex secret to base32
        secret_bytes = bytes.fromhex(secret)
        secret_b32 = base64.b32encode(secret_bytes).decode("utf-8").rstrip("=")

        params = {
            "secret": secret_b32,
            "issuer": issuer,
            "algorithm": cls.ALGORITHM.upper(),
            "digits": str(cls.DIGITS),
            "period": str(cls.INTERVAL),
        }

        label = urllib.parse.quote(f"{issuer}:{username}")
        query = urllib.parse.urlencode(params)

        return f"otpauth://totp/{label}?{query}"


class LoginAttemptTracker:
    """Track and limit login attempts."""

    def __init__(
        self,
        max_attempts: int = 5,
        lockout_duration: int = 900,  # 15 minutes
    ) -> None:
        """Initialize login attempt tracker."""
        self._max_attempts = max_attempts
        self._lockout_duration = lockout_duration
        self._attempts: dict[str, list[float]] = {}
        self._lockouts: dict[str, float] = {}

    def is_locked_out(self, identifier: str) -> bool:
        """Check if identifier is locked out."""
        lockout_time = self._lockouts.get(identifier)
        if lockout_time is None:
            return False

        if time.time() > lockout_time + self._lockout_duration:
            del self._lockouts[identifier]
            return False

        return True

    def get_lockout_remaining(self, identifier: str) -> int:
        """Get remaining lockout time in seconds."""
        lockout_time = self._lockouts.get(identifier)
        if lockout_time is None:
            return 0

        remaining = (lockout_time + self._lockout_duration) - time.time()
        return max(0, int(remaining))

    def record_attempt(self, identifier: str, success: bool) -> None:
        """Record a login attempt."""
        if success:
            # Clear attempts on success
            if identifier in self._attempts:
                del self._attempts[identifier]
            return

        # Record failed attempt
        now = time.time()

        if identifier not in self._attempts:
            self._attempts[identifier] = []

        # Remove old attempts (older than lockout duration)
        self._attempts[identifier] = [
            t for t in self._attempts[identifier]
            if now - t < self._lockout_duration
        ]

        self._attempts[identifier].append(now)

        # Check if should lock out
        if len(self._attempts[identifier]) >= self._max_attempts:
            self._lockouts[identifier] = now
            logger.warning(f"Account locked out: {identifier}")

    def get_attempt_count(self, identifier: str) -> int:
        """Get number of recent failed attempts."""
        if identifier not in self._attempts:
            return 0

        now = time.time()
        return len([
            t for t in self._attempts[identifier]
            if now - t < self._lockout_duration
        ])


class AuthManager:
    """
    Authentication manager.

    Coordinates all authentication functionality.
    """

    def __init__(
        self,
        secret_key: str,
        max_login_attempts: int = 5,
        lockout_duration: int = 900,
    ) -> None:
        """Initialize auth manager."""
        self._password_hasher = PasswordHasher()
        self._token_manager = TokenManager(secret_key)
        self._totp_manager = TOTPManager()
        self._attempt_tracker = LoginAttemptTracker(max_login_attempts, lockout_duration)

        # User storage (in production, use database)
        self._users: dict[str, User] = {}

        logger.info("AuthManager initialized")

    def register_user(
        self,
        username: str,
        email: str,
        password: str,
        is_admin: bool = False,
    ) -> User:
        """
        Register a new user.

        Args:
            username: Username
            email: Email address
            password: Plain text password
            is_admin: Admin flag

        Returns:
            Created user
        """
        # Generate user ID
        user_id = secrets.token_hex(16)

        # Hash password
        password_hash = PasswordHasher.hash_password(password)

        # Create user
        user = User(
            id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            is_admin=is_admin,
        )

        self._users[user_id] = user
        logger.info(f"User registered: {username}")

        return user

    def authenticate(
        self,
        username: str,
        password: str,
        totp_code: str | None = None,
    ) -> tuple[User | None, str | None]:
        """
        Authenticate a user.

        Args:
            username: Username
            password: Password
            totp_code: Optional TOTP code

        Returns:
            Tuple of (user, error_message)
        """
        # Check lockout
        if self._attempt_tracker.is_locked_out(username):
            remaining = self._attempt_tracker.get_lockout_remaining(username)
            return None, f"Account locked. Try again in {remaining} seconds."

        # Find user
        user = self.get_user_by_username(username)
        if user is None:
            self._attempt_tracker.record_attempt(username, False)
            return None, "Invalid username or password"

        # Verify password
        if not PasswordHasher.verify_password(password, user.password_hash):
            self._attempt_tracker.record_attempt(username, False)
            return None, "Invalid username or password"

        # Check if account is active
        if not user.is_active:
            return None, "Account is disabled"

        # Check 2FA if enabled
        if user.two_factor_enabled:
            if not totp_code:
                return None, "Two-factor code required"

            if not user.two_factor_secret:
                return None, "Two-factor not configured"

            if not TOTPManager.verify_code(user.two_factor_secret, totp_code):
                self._attempt_tracker.record_attempt(username, False)
                return None, "Invalid two-factor code"

        # Success
        self._attempt_tracker.record_attempt(username, True)
        user.last_login = datetime.now()

        # Rehash password if needed
        if PasswordHasher.needs_rehash(user.password_hash):
            user.password_hash = PasswordHasher.hash_password(password)

        logger.info(f"User authenticated: {username}")
        return user, None

    def login_user(self, user: User) -> dict[str, str]:
        """
        Log in a user and create session.

        Args:
            user: User to log in

        Returns:
            Dictionary with access and refresh tokens
        """
        # Store user in session
        session["user"] = user.to_dict()
        session["login_time"] = datetime.now().isoformat()
        session.permanent = True

        # Generate tokens
        access_token = self._token_manager.generate_access_token(user.id)
        refresh_token = self._token_manager.generate_refresh_token(user.id)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
        }

    def logout_user(self) -> None:
        """Log out current user."""
        user_data = session.get("user")
        if user_data:
            self._token_manager.revoke_user_tokens(user_data["id"])

        session.clear()

    def get_current_user(self) -> User | None:
        """Get currently logged in user."""
        user_data = session.get("user")
        if not user_data:
            return None

        return self._users.get(user_data["id"])

    def get_user_by_username(self, username: str) -> User | None:
        """Find user by username."""
        for user in self._users.values():
            if user.username == username:
                return user
        return None

    def get_user_by_email(self, email: str) -> User | None:
        """Find user by email."""
        for user in self._users.values():
            if user.email == email:
                return user
        return None

    def get_user_by_id(self, user_id: str) -> User | None:
        """Find user by ID."""
        return self._users.get(user_id)

    def change_password(
        self,
        user_id: str,
        old_password: str,
        new_password: str,
    ) -> tuple[bool, str | None]:
        """
        Change user password.

        Args:
            user_id: User ID
            old_password: Current password
            new_password: New password

        Returns:
            Tuple of (success, error_message)
        """
        user = self._users.get(user_id)
        if not user:
            return False, "User not found"

        if not PasswordHasher.verify_password(old_password, user.password_hash):
            return False, "Current password is incorrect"

        user.password_hash = PasswordHasher.hash_password(new_password)

        # Revoke all tokens
        self._token_manager.revoke_user_tokens(user_id)

        return True, None

    def enable_two_factor(self, user_id: str) -> tuple[str | None, str | None]:
        """
        Enable two-factor authentication.

        Args:
            user_id: User ID

        Returns:
            Tuple of (provisioning_uri, error_message)
        """
        user = self._users.get(user_id)
        if not user:
            return None, "User not found"

        # Generate new secret
        secret = TOTPManager.generate_secret()
        user.two_factor_secret = secret

        # Generate provisioning URI
        uri = TOTPManager.get_provisioning_uri(
            secret,
            user.username,
        )

        return uri, None

    def confirm_two_factor(
        self,
        user_id: str,
        code: str,
    ) -> tuple[bool, str | None]:
        """
        Confirm two-factor setup with code.

        Args:
            user_id: User ID
            code: TOTP code

        Returns:
            Tuple of (success, error_message)
        """
        user = self._users.get(user_id)
        if not user:
            return False, "User not found"

        if not user.two_factor_secret:
            return False, "Two-factor not initiated"

        if not TOTPManager.verify_code(user.two_factor_secret, code):
            return False, "Invalid code"

        user.two_factor_enabled = True
        return True, None

    def disable_two_factor(
        self,
        user_id: str,
        password: str,
    ) -> tuple[bool, str | None]:
        """
        Disable two-factor authentication.

        Args:
            user_id: User ID
            password: User password

        Returns:
            Tuple of (success, error_message)
        """
        user = self._users.get(user_id)
        if not user:
            return False, "User not found"

        if not PasswordHasher.verify_password(password, user.password_hash):
            return False, "Invalid password"

        user.two_factor_enabled = False
        user.two_factor_secret = None

        return True, None

    def generate_password_reset_token(self, email: str) -> str | None:
        """Generate password reset token."""
        user = self.get_user_by_email(email)
        if not user:
            return None

        return self._token_manager.generate_reset_token(user.id)

    def reset_password(
        self,
        token: str,
        new_password: str,
    ) -> tuple[bool, str | None]:
        """
        Reset password using token.

        Args:
            token: Reset token
            new_password: New password

        Returns:
            Tuple of (success, error_message)
        """
        user_id = self._token_manager.validate_token(token, "reset")
        if not user_id:
            return False, "Invalid or expired token"

        user = self._users.get(user_id)
        if not user:
            return False, "User not found"

        user.password_hash = PasswordHasher.hash_password(new_password)

        # Revoke token and all user sessions
        self._token_manager.revoke_token(token)
        self._token_manager.revoke_user_tokens(user_id)

        return True, None


def create_auth_manager(secret_key: str, **kwargs: Any) -> AuthManager:
    """
    Create auth manager instance.

    Args:
        secret_key: Application secret key
        **kwargs: Additional configuration

    Returns:
        AuthManager instance
    """
    return AuthManager(secret_key, **kwargs)


# Module version
__version__ = "2.2.0"
