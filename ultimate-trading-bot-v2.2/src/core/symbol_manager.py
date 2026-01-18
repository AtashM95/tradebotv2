"""
Symbol Manager Module for Ultimate Trading Bot v2.2.

This module manages trading symbols, including asset information,
tradability checks, symbol validation, and universe management.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field, field_validator

from src.utils.exceptions import (
    APIError,
    SymbolNotTradableError,
    ValidationError,
)
from src.utils.helpers import generate_uuid, is_valid_symbol
from src.utils.date_utils import now_utc
from src.utils.decorators import async_retry, cache, singleton


logger = logging.getLogger(__name__)


class AssetClass(str, Enum):
    """Asset class enumeration."""

    US_EQUITY = "us_equity"
    CRYPTO = "crypto"
    OPTIONS = "options"
    FUTURES = "futures"
    FOREX = "forex"


class AssetStatus(str, Enum):
    """Asset status enumeration."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    DELISTED = "delisted"
    HALTED = "halted"


class AssetExchange(str, Enum):
    """Exchange enumeration."""

    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    AMEX = "AMEX"
    ARCA = "ARCA"
    BATS = "BATS"
    IEX = "IEX"
    OTC = "OTC"
    CRYPTO = "CRYPTO"


class SymbolManagerConfig(BaseModel):
    """Configuration for symbol manager."""

    cache_ttl_seconds: int = Field(default=3600, ge=60, le=86400)
    refresh_interval_seconds: int = Field(default=3600, ge=300, le=86400)
    max_symbols_per_request: int = Field(default=100, ge=10, le=500)
    enable_auto_refresh: bool = Field(default=True)
    validate_on_add: bool = Field(default=True)
    default_asset_class: AssetClass = Field(default=AssetClass.US_EQUITY)


class AssetInfo(BaseModel):
    """Asset information model."""

    symbol: str
    name: str = Field(default="")
    asset_class: AssetClass = Field(default=AssetClass.US_EQUITY)
    exchange: Optional[AssetExchange] = None
    status: AssetStatus = Field(default=AssetStatus.ACTIVE)

    tradable: bool = Field(default=True)
    marginable: bool = Field(default=False)
    shortable: bool = Field(default=False)
    easy_to_borrow: bool = Field(default=False)
    fractionable: bool = Field(default=False)

    min_order_size: float = Field(default=1.0)
    min_trade_increment: float = Field(default=1.0)
    price_increment: float = Field(default=0.01)
    maintenance_margin_requirement: float = Field(default=0.25)

    sector: Optional[str] = None
    industry: Optional[str] = None

    last_updated: datetime = Field(default_factory=now_utc)

    @property
    def is_active(self) -> bool:
        """Check if asset is active."""
        return self.status == AssetStatus.ACTIVE

    @property
    def is_tradable(self) -> bool:
        """Check if asset can be traded."""
        return self.tradable and self.is_active

    @property
    def is_equity(self) -> bool:
        """Check if asset is equity."""
        return self.asset_class == AssetClass.US_EQUITY

    @property
    def is_crypto(self) -> bool:
        """Check if asset is crypto."""
        return self.asset_class == AssetClass.CRYPTO

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "name": self.name,
            "asset_class": self.asset_class.value,
            "exchange": self.exchange.value if self.exchange else None,
            "status": self.status.value,
            "tradable": self.tradable,
            "marginable": self.marginable,
            "shortable": self.shortable,
            "fractionable": self.fractionable,
            "sector": self.sector,
            "industry": self.industry,
        }


class SymbolUniverse(BaseModel):
    """Trading symbol universe."""

    universe_id: str = Field(default_factory=generate_uuid)
    name: str = Field(default="default")
    description: str = Field(default="")

    symbols: list[str] = Field(default_factory=list)
    filters: dict = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=now_utc)
    updated_at: datetime = Field(default_factory=now_utc)

    @property
    def symbol_count(self) -> int:
        """Get symbol count."""
        return len(self.symbols)

    def add_symbol(self, symbol: str) -> bool:
        """Add a symbol to universe."""
        symbol = symbol.upper()
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            self.updated_at = now_utc()
            return True
        return False

    def remove_symbol(self, symbol: str) -> bool:
        """Remove a symbol from universe."""
        symbol = symbol.upper()
        if symbol in self.symbols:
            self.symbols.remove(symbol)
            self.updated_at = now_utc()
            return True
        return False

    def contains(self, symbol: str) -> bool:
        """Check if symbol is in universe."""
        return symbol.upper() in self.symbols


@singleton
class SymbolManager:
    """
    Manages trading symbols and asset information.

    This class provides:
    - Symbol validation and lookup
    - Asset information caching
    - Universe management
    - Tradability checks
    """

    def __init__(
        self,
        config: Optional[SymbolManagerConfig] = None,
        broker_client: Optional[Any] = None,
    ) -> None:
        """
        Initialize SymbolManager.

        Args:
            config: Symbol manager configuration
            broker_client: Broker API client
        """
        self._config = config or SymbolManagerConfig()
        self._broker_client = broker_client

        self._assets: dict[str, AssetInfo] = {}
        self._universes: dict[str, SymbolUniverse] = {}

        self._active_universe: Optional[str] = None

        self._update_callbacks: list[Callable[[str, AssetInfo], None]] = []

        self._refresh_task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = asyncio.Lock()

        self._universes["default"] = SymbolUniverse(name="default")
        self._active_universe = "default"

        logger.info("SymbolManager initialized")

    @property
    def assets(self) -> dict[str, AssetInfo]:
        """Get all cached assets."""
        return self._assets.copy()

    @property
    def universes(self) -> dict[str, SymbolUniverse]:
        """Get all universes."""
        return self._universes.copy()

    @property
    def active_universe(self) -> Optional[SymbolUniverse]:
        """Get active universe."""
        if self._active_universe:
            return self._universes.get(self._active_universe)
        return None

    @property
    def active_symbols(self) -> list[str]:
        """Get symbols in active universe."""
        universe = self.active_universe
        return universe.symbols if universe else []

    def set_broker_client(self, broker_client: Any) -> None:
        """Set the broker client."""
        self._broker_client = broker_client
        logger.debug("Broker client set")

    def register_update_callback(
        self,
        callback: Callable[[str, AssetInfo], None]
    ) -> None:
        """Register callback for asset updates."""
        self._update_callbacks.append(callback)

    async def start(self) -> None:
        """Start the symbol manager."""
        if self._running:
            return

        self._running = True

        if self._config.enable_auto_refresh:
            self._refresh_task = asyncio.create_task(
                self._refresh_loop(),
                name="symbol_refresh_loop"
            )

        logger.info("SymbolManager started")

    async def stop(self) -> None:
        """Stop the symbol manager."""
        self._running = False

        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
            self._refresh_task = None

        logger.info("SymbolManager stopped")

    async def _refresh_loop(self) -> None:
        """Periodic refresh loop."""
        while self._running:
            try:
                await asyncio.sleep(self._config.refresh_interval_seconds)

                if self.active_symbols:
                    await self.refresh_symbols(self.active_symbols)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in symbol refresh loop: {e}")
                await asyncio.sleep(60)

    @async_retry(max_attempts=3, delay=1.0)
    async def get_asset(
        self,
        symbol: str,
        refresh: bool = False
    ) -> Optional[AssetInfo]:
        """
        Get asset information for a symbol.

        Args:
            symbol: Trading symbol
            refresh: Force refresh from broker

        Returns:
            Asset information or None
        """
        symbol = symbol.upper()

        if not refresh and symbol in self._assets:
            cached = self._assets[symbol]
            age = (now_utc() - cached.last_updated).total_seconds()
            if age < self._config.cache_ttl_seconds:
                return cached

        async with self._lock:
            asset = await self._fetch_asset(symbol)
            if asset:
                self._assets[symbol] = asset
                await self._notify_update(symbol, asset)
            return asset

    async def _fetch_asset(self, symbol: str) -> Optional[AssetInfo]:
        """Fetch asset from broker."""
        if not self._broker_client:
            logger.warning("No broker client configured")
            return None

        try:
            if hasattr(self._broker_client, "get_asset"):
                data = await self._broker_client.get_asset(symbol)
            elif hasattr(self._broker_client, "fetch_asset"):
                data = await self._broker_client.fetch_asset(symbol)
            else:
                logger.error("Broker client has no asset method")
                return None

            return self._parse_asset_data(symbol, data)

        except Exception as e:
            logger.error(f"Failed to fetch asset {symbol}: {e}")
            return None

    def _parse_asset_data(self, symbol: str, data: dict) -> AssetInfo:
        """Parse raw asset data into AssetInfo model."""
        exchange_map = {
            "NYSE": AssetExchange.NYSE,
            "NASDAQ": AssetExchange.NASDAQ,
            "AMEX": AssetExchange.AMEX,
            "ARCA": AssetExchange.ARCA,
            "BATS": AssetExchange.BATS,
            "IEX": AssetExchange.IEX,
            "OTC": AssetExchange.OTC,
        }

        status_map = {
            "active": AssetStatus.ACTIVE,
            "inactive": AssetStatus.INACTIVE,
            "delisted": AssetStatus.DELISTED,
            "halted": AssetStatus.HALTED,
        }

        asset_class = AssetClass.US_EQUITY
        if data.get("class", "").lower() == "crypto":
            asset_class = AssetClass.CRYPTO

        return AssetInfo(
            symbol=symbol,
            name=data.get("name", ""),
            asset_class=asset_class,
            exchange=exchange_map.get(data.get("exchange", "").upper()),
            status=status_map.get(data.get("status", "active").lower(), AssetStatus.ACTIVE),
            tradable=data.get("tradable", True),
            marginable=data.get("marginable", False),
            shortable=data.get("shortable", False),
            easy_to_borrow=data.get("easy_to_borrow", False),
            fractionable=data.get("fractionable", False),
            min_order_size=float(data.get("min_order_size", 1)),
            min_trade_increment=float(data.get("min_trade_increment", 1)),
            price_increment=float(data.get("price_increment", 0.01)),
            maintenance_margin_requirement=float(
                data.get("maintenance_margin_requirement", 0.25)
            ),
            last_updated=now_utc(),
        )

    async def refresh_symbols(
        self,
        symbols: list[str]
    ) -> dict[str, Optional[AssetInfo]]:
        """
        Refresh asset information for multiple symbols.

        Args:
            symbols: List of symbols to refresh

        Returns:
            Dictionary of symbol to asset info
        """
        results = {}

        for i in range(0, len(symbols), self._config.max_symbols_per_request):
            batch = symbols[i:i + self._config.max_symbols_per_request]

            tasks = [
                self.get_asset(symbol, refresh=True)
                for symbol in batch
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for symbol, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error refreshing {symbol}: {result}")
                    results[symbol] = None
                else:
                    results[symbol] = result

        return results

    async def _notify_update(self, symbol: str, asset: AssetInfo) -> None:
        """Notify callbacks of asset update."""
        for callback in self._update_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(symbol, asset)
                else:
                    callback(symbol, asset)
            except Exception as e:
                logger.error(f"Error in update callback: {e}")

    def validate_symbol(self, symbol: str) -> tuple[bool, str]:
        """
        Validate a trading symbol.

        Args:
            symbol: Symbol to validate

        Returns:
            Tuple of (is_valid, message)
        """
        if not symbol:
            return False, "Symbol cannot be empty"

        symbol = symbol.upper()

        if not is_valid_symbol(symbol):
            return False, f"Invalid symbol format: {symbol}"

        return True, "Valid symbol"

    async def is_tradable(self, symbol: str) -> bool:
        """
        Check if a symbol is tradable.

        Args:
            symbol: Symbol to check

        Returns:
            True if tradable
        """
        asset = await self.get_asset(symbol)
        return asset is not None and asset.is_tradable

    async def ensure_tradable(self, symbol: str) -> AssetInfo:
        """
        Ensure a symbol is tradable, raising exception if not.

        Args:
            symbol: Symbol to check

        Returns:
            Asset info if tradable

        Raises:
            SymbolNotTradableError: If symbol is not tradable
        """
        asset = await self.get_asset(symbol)

        if not asset:
            raise SymbolNotTradableError(f"Symbol not found: {symbol}")

        if not asset.is_tradable:
            raise SymbolNotTradableError(
                f"Symbol {symbol} is not tradable (status={asset.status.value})"
            )

        return asset

    def create_universe(
        self,
        name: str,
        symbols: Optional[list[str]] = None,
        description: str = ""
    ) -> SymbolUniverse:
        """
        Create a new symbol universe.

        Args:
            name: Universe name
            symbols: Initial symbols
            description: Universe description

        Returns:
            Created universe
        """
        if name in self._universes:
            raise ValidationError(f"Universe already exists: {name}")

        universe = SymbolUniverse(
            name=name,
            description=description,
            symbols=[s.upper() for s in (symbols or [])],
        )

        self._universes[name] = universe
        logger.info(f"Created universe: {name} with {len(universe.symbols)} symbols")

        return universe

    def get_universe(self, name: str) -> Optional[SymbolUniverse]:
        """Get a universe by name."""
        return self._universes.get(name)

    def set_active_universe(self, name: str) -> bool:
        """
        Set the active trading universe.

        Args:
            name: Universe name

        Returns:
            True if successful
        """
        if name not in self._universes:
            logger.error(f"Universe not found: {name}")
            return False

        self._active_universe = name
        logger.info(f"Set active universe: {name}")
        return True

    def add_to_universe(
        self,
        symbol: str,
        universe_name: Optional[str] = None
    ) -> bool:
        """
        Add a symbol to a universe.

        Args:
            symbol: Symbol to add
            universe_name: Universe name (default: active)

        Returns:
            True if added
        """
        universe_name = universe_name or self._active_universe
        if not universe_name:
            return False

        universe = self._universes.get(universe_name)
        if not universe:
            return False

        return universe.add_symbol(symbol)

    def remove_from_universe(
        self,
        symbol: str,
        universe_name: Optional[str] = None
    ) -> bool:
        """
        Remove a symbol from a universe.

        Args:
            symbol: Symbol to remove
            universe_name: Universe name (default: active)

        Returns:
            True if removed
        """
        universe_name = universe_name or self._active_universe
        if not universe_name:
            return False

        universe = self._universes.get(universe_name)
        if not universe:
            return False

        return universe.remove_symbol(symbol)

    def delete_universe(self, name: str) -> bool:
        """
        Delete a universe.

        Args:
            name: Universe name

        Returns:
            True if deleted
        """
        if name not in self._universes:
            return False

        if name == "default":
            raise ValidationError("Cannot delete default universe")

        del self._universes[name]

        if self._active_universe == name:
            self._active_universe = "default"

        logger.info(f"Deleted universe: {name}")
        return True

    def filter_by_exchange(
        self,
        exchange: AssetExchange
    ) -> list[str]:
        """Filter cached assets by exchange."""
        return [
            symbol for symbol, asset in self._assets.items()
            if asset.exchange == exchange
        ]

    def filter_by_sector(self, sector: str) -> list[str]:
        """Filter cached assets by sector."""
        return [
            symbol for symbol, asset in self._assets.items()
            if asset.sector and asset.sector.lower() == sector.lower()
        ]

    def filter_tradable(self) -> list[str]:
        """Get all tradable symbols from cache."""
        return [
            symbol for symbol, asset in self._assets.items()
            if asset.is_tradable
        ]

    def filter_shortable(self) -> list[str]:
        """Get all shortable symbols from cache."""
        return [
            symbol for symbol, asset in self._assets.items()
            if asset.shortable
        ]

    def filter_marginable(self) -> list[str]:
        """Get all marginable symbols from cache."""
        return [
            symbol for symbol, asset in self._assets.items()
            if asset.marginable
        ]

    def get_cached(self, symbol: str) -> Optional[AssetInfo]:
        """Get cached asset without fetching."""
        return self._assets.get(symbol.upper())

    def clear_cache(self) -> int:
        """Clear the asset cache."""
        count = len(self._assets)
        self._assets.clear()
        logger.info(f"Cleared {count} cached assets")
        return count

    def get_statistics(self) -> dict:
        """Get symbol manager statistics."""
        return {
            "total_cached_assets": len(self._assets),
            "total_universes": len(self._universes),
            "active_universe": self._active_universe,
            "active_symbols_count": len(self.active_symbols),
            "tradable_count": len(self.filter_tradable()),
            "shortable_count": len(self.filter_shortable()),
            "marginable_count": len(self.filter_marginable()),
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SymbolManager(cached={len(self._assets)}, "
            f"universes={len(self._universes)}, "
            f"active={self._active_universe})"
        )
