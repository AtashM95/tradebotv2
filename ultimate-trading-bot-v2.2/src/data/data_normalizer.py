"""
Data Normalizer Module for Ultimate Trading Bot v2.2.

This module provides data normalization and transformation
functionality for market data from various sources.
"""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

from src.data.base_provider import Quote, Bar, Trade, DataProviderType
from src.utils.date_utils import now_utc, to_utc


logger = logging.getLogger(__name__)


class NormalizerConfig(BaseModel):
    """Configuration for data normalizer."""

    default_timezone: str = Field(default="UTC")
    price_decimals: int = Field(default=4, ge=0, le=8)
    volume_as_integer: bool = Field(default=True)
    remove_outliers: bool = Field(default=True)
    outlier_std_threshold: float = Field(default=3.0, ge=1.0, le=10.0)
    fill_missing_volume: bool = Field(default=True)
    handle_splits: bool = Field(default=True)


class DataNormalizer:
    """
    Normalizes market data from various sources.

    Handles:
    - Timestamp normalization to UTC
    - Price and volume formatting
    - Provider-specific data transformations
    - Outlier detection and handling
    """

    def __init__(
        self,
        config: Optional[NormalizerConfig] = None,
    ) -> None:
        """
        Initialize DataNormalizer.

        Args:
            config: Normalizer configuration
        """
        self._config = config or NormalizerConfig()

        self._normalized_count = 0
        self._error_count = 0

        logger.info("DataNormalizer initialized")

    def normalize_quote(
        self,
        raw_data: dict,
        provider: DataProviderType = DataProviderType.ALPACA,
    ) -> Optional[Quote]:
        """
        Normalize raw quote data.

        Args:
            raw_data: Raw quote data from provider
            provider: Data provider type

        Returns:
            Normalized Quote or None
        """
        try:
            if provider == DataProviderType.ALPACA:
                return self._normalize_alpaca_quote(raw_data)
            elif provider == DataProviderType.YAHOO:
                return self._normalize_yahoo_quote(raw_data)
            elif provider == DataProviderType.POLYGON:
                return self._normalize_polygon_quote(raw_data)
            else:
                return self._normalize_generic_quote(raw_data)

        except Exception as e:
            logger.error(f"Error normalizing quote: {e}")
            self._error_count += 1
            return None

    def _normalize_alpaca_quote(self, data: dict) -> Quote:
        """Normalize Alpaca quote data."""
        timestamp = self._parse_timestamp(data.get("t") or data.get("timestamp"))

        return Quote(
            symbol=data.get("S") or data.get("symbol", ""),
            bid_price=self._normalize_price(data.get("bp") or data.get("bid_price", 0)),
            bid_size=self._normalize_volume(data.get("bs") or data.get("bid_size", 0)),
            ask_price=self._normalize_price(data.get("ap") or data.get("ask_price", 0)),
            ask_size=self._normalize_volume(data.get("as") or data.get("ask_size", 0)),
            last_price=self._normalize_price(data.get("lp") or data.get("last_price", 0)),
            volume=self._normalize_volume(data.get("v") or data.get("volume", 0)),
            timestamp=timestamp,
            exchange=data.get("x") or data.get("exchange"),
        )

    def _normalize_yahoo_quote(self, data: dict) -> Quote:
        """Normalize Yahoo Finance quote data."""
        timestamp = self._parse_timestamp(data.get("regularMarketTime"))

        return Quote(
            symbol=data.get("symbol", ""),
            bid_price=self._normalize_price(data.get("bid", 0)),
            bid_size=self._normalize_volume(data.get("bidSize", 0)),
            ask_price=self._normalize_price(data.get("ask", 0)),
            ask_size=self._normalize_volume(data.get("askSize", 0)),
            last_price=self._normalize_price(data.get("regularMarketPrice", 0)),
            volume=self._normalize_volume(data.get("regularMarketVolume", 0)),
            timestamp=timestamp,
        )

    def _normalize_polygon_quote(self, data: dict) -> Quote:
        """Normalize Polygon quote data."""
        timestamp = self._parse_timestamp(data.get("t"))

        return Quote(
            symbol=data.get("sym") or data.get("T", ""),
            bid_price=self._normalize_price(data.get("bp", 0)),
            bid_size=self._normalize_volume(data.get("bs", 0)),
            ask_price=self._normalize_price(data.get("ap", 0)),
            ask_size=self._normalize_volume(data.get("as", 0)),
            last_price=self._normalize_price(data.get("p", 0)),
            volume=self._normalize_volume(data.get("s", 0)),
            timestamp=timestamp,
        )

    def _normalize_generic_quote(self, data: dict) -> Quote:
        """Normalize generic quote data."""
        timestamp = self._parse_timestamp(
            data.get("timestamp") or data.get("time") or data.get("t")
        )

        return Quote(
            symbol=data.get("symbol") or data.get("sym") or data.get("S", ""),
            bid_price=self._normalize_price(data.get("bid_price") or data.get("bid", 0)),
            bid_size=self._normalize_volume(data.get("bid_size") or data.get("bidSize", 0)),
            ask_price=self._normalize_price(data.get("ask_price") or data.get("ask", 0)),
            ask_size=self._normalize_volume(data.get("ask_size") or data.get("askSize", 0)),
            last_price=self._normalize_price(data.get("last_price") or data.get("last", 0)),
            volume=self._normalize_volume(data.get("volume") or data.get("vol", 0)),
            timestamp=timestamp,
        )

    def normalize_bar(
        self,
        raw_data: dict,
        symbol: str = "",
        provider: DataProviderType = DataProviderType.ALPACA,
    ) -> Optional[Bar]:
        """
        Normalize raw bar data.

        Args:
            raw_data: Raw bar data from provider
            symbol: Trading symbol
            provider: Data provider type

        Returns:
            Normalized Bar or None
        """
        try:
            if provider == DataProviderType.ALPACA:
                return self._normalize_alpaca_bar(raw_data, symbol)
            elif provider == DataProviderType.YAHOO:
                return self._normalize_yahoo_bar(raw_data, symbol)
            else:
                return self._normalize_generic_bar(raw_data, symbol)

        except Exception as e:
            logger.error(f"Error normalizing bar: {e}")
            self._error_count += 1
            return None

    def _normalize_alpaca_bar(self, data: dict, symbol: str) -> Bar:
        """Normalize Alpaca bar data."""
        timestamp = self._parse_timestamp(data.get("t") or data.get("timestamp"))

        return Bar(
            symbol=symbol or data.get("S", ""),
            timestamp=timestamp,
            open=self._normalize_price(data.get("o") or data.get("open", 0)),
            high=self._normalize_price(data.get("h") or data.get("high", 0)),
            low=self._normalize_price(data.get("l") or data.get("low", 0)),
            close=self._normalize_price(data.get("c") or data.get("close", 0)),
            volume=self._normalize_volume(data.get("v") or data.get("volume", 0)),
            vwap=self._normalize_price(data.get("vw") or data.get("vwap")),
            trade_count=data.get("n") or data.get("trade_count"),
        )

    def _normalize_yahoo_bar(self, data: dict, symbol: str) -> Bar:
        """Normalize Yahoo Finance bar data."""
        timestamp = self._parse_timestamp(data.get("timestamp") or data.get("date"))

        return Bar(
            symbol=symbol,
            timestamp=timestamp,
            open=self._normalize_price(data.get("open", 0)),
            high=self._normalize_price(data.get("high", 0)),
            low=self._normalize_price(data.get("low", 0)),
            close=self._normalize_price(data.get("close", 0)),
            volume=self._normalize_volume(data.get("volume", 0)),
        )

    def _normalize_generic_bar(self, data: dict, symbol: str) -> Bar:
        """Normalize generic bar data."""
        timestamp = self._parse_timestamp(
            data.get("timestamp") or data.get("time") or data.get("t") or data.get("date")
        )

        return Bar(
            symbol=symbol or data.get("symbol", ""),
            timestamp=timestamp,
            open=self._normalize_price(data.get("open") or data.get("o", 0)),
            high=self._normalize_price(data.get("high") or data.get("h", 0)),
            low=self._normalize_price(data.get("low") or data.get("l", 0)),
            close=self._normalize_price(data.get("close") or data.get("c", 0)),
            volume=self._normalize_volume(data.get("volume") or data.get("v", 0)),
        )

    def normalize_trade(
        self,
        raw_data: dict,
        provider: DataProviderType = DataProviderType.ALPACA,
    ) -> Optional[Trade]:
        """
        Normalize raw trade data.

        Args:
            raw_data: Raw trade data from provider
            provider: Data provider type

        Returns:
            Normalized Trade or None
        """
        try:
            timestamp = self._parse_timestamp(
                raw_data.get("t") or raw_data.get("timestamp")
            )

            return Trade(
                symbol=raw_data.get("S") or raw_data.get("symbol", ""),
                price=self._normalize_price(raw_data.get("p") or raw_data.get("price", 0)),
                size=self._normalize_volume(raw_data.get("s") or raw_data.get("size", 0)),
                timestamp=timestamp,
                exchange=raw_data.get("x") or raw_data.get("exchange"),
                conditions=raw_data.get("c") or raw_data.get("conditions"),
                trade_id=raw_data.get("i") or raw_data.get("id"),
            )

        except Exception as e:
            logger.error(f"Error normalizing trade: {e}")
            self._error_count += 1
            return None

    def normalize_bars_df(
        self,
        df: pd.DataFrame,
        symbol: str = "",
    ) -> pd.DataFrame:
        """
        Normalize a DataFrame of bar data.

        Args:
            df: Input DataFrame
            symbol: Trading symbol

        Returns:
            Normalized DataFrame
        """
        if df.empty:
            return df

        df = df.copy()

        column_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adj_close",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }

        df.rename(columns={k: v for k, v in column_map.items() if k in df.columns}, inplace=True)

        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = df[col].apply(self._normalize_price)

        if "volume" in df.columns:
            df["volume"] = df["volume"].apply(self._normalize_volume)

        if df.index.dtype == "int64":
            df.index = pd.to_datetime(df.index, unit="s", utc=True)
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)

        if self._config.remove_outliers:
            df = self._remove_outliers(df)

        if self._config.fill_missing_volume:
            if "volume" in df.columns:
                df["volume"] = df["volume"].fillna(0).astype(int)

        df.sort_index(inplace=True)

        self._normalized_count += len(df)

        return df

    def _normalize_price(self, value: Any) -> float:
        """Normalize a price value."""
        if value is None:
            return 0.0

        try:
            price = float(value)
            return round(price, self._config.price_decimals)
        except (ValueError, TypeError):
            return 0.0

    def _normalize_volume(self, value: Any) -> int:
        """Normalize a volume value."""
        if value is None:
            return 0

        try:
            vol = float(value)
            if self._config.volume_as_integer:
                return int(vol)
            return vol
        except (ValueError, TypeError):
            return 0

    def _parse_timestamp(self, value: Any) -> datetime:
        """Parse and normalize timestamp to UTC."""
        if value is None:
            return now_utc()

        if isinstance(value, datetime):
            return to_utc(value)

        if isinstance(value, (int, float)):
            if value > 1e12:
                value = value / 1000

            return datetime.fromtimestamp(value, tz=timezone.utc)

        if isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return to_utc(dt)
            except ValueError:
                pass

            try:
                return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            except ValueError:
                pass

            try:
                return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                pass

        return now_utc()

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from DataFrame."""
        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                continue

            mean = df[col].mean()
            std = df[col].std()

            if std > 0:
                lower = mean - (self._config.outlier_std_threshold * std)
                upper = mean + (self._config.outlier_std_threshold * std)

                outliers = (df[col] < lower) | (df[col] > upper)
                outlier_count = outliers.sum()

                if outlier_count > 0:
                    logger.warning(f"Removed {outlier_count} outliers from {col}")
                    df = df[~outliers]

        return df

    def validate_bar(self, bar: Bar) -> tuple[bool, str]:
        """
        Validate a bar.

        Args:
            bar: Bar to validate

        Returns:
            Tuple of (is_valid, message)
        """
        if bar.open <= 0:
            return False, "Invalid open price"

        if bar.high <= 0:
            return False, "Invalid high price"

        if bar.low <= 0:
            return False, "Invalid low price"

        if bar.close <= 0:
            return False, "Invalid close price"

        if bar.high < bar.low:
            return False, "High is less than low"

        if bar.high < bar.open or bar.high < bar.close:
            return False, "High is not the highest price"

        if bar.low > bar.open or bar.low > bar.close:
            return False, "Low is not the lowest price"

        if bar.volume < 0:
            return False, "Negative volume"

        return True, "Valid"

    def validate_quote(self, quote: Quote) -> tuple[bool, str]:
        """
        Validate a quote.

        Args:
            quote: Quote to validate

        Returns:
            Tuple of (is_valid, message)
        """
        if quote.bid_price < 0:
            return False, "Negative bid price"

        if quote.ask_price < 0:
            return False, "Negative ask price"

        if quote.bid_price > 0 and quote.ask_price > 0:
            if quote.bid_price > quote.ask_price:
                return False, "Bid price greater than ask price"

        return True, "Valid"

    def get_statistics(self) -> dict:
        """Get normalizer statistics."""
        return {
            "normalized_count": self._normalized_count,
            "error_count": self._error_count,
            "price_decimals": self._config.price_decimals,
            "remove_outliers": self._config.remove_outliers,
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"DataNormalizer(normalized={self._normalized_count}, errors={self._error_count})"
