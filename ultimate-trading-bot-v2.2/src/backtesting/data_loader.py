"""
Historical Data Loader for Backtesting.

This module provides comprehensive data loading functionality for backtesting,
including multiple data sources, caching, preprocessing, and validation.
"""

import asyncio
import hashlib
import logging
import pickle
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DataSource(str, Enum):
    """Supported data sources."""

    CSV = "csv"
    PARQUET = "parquet"
    DATABASE = "database"
    API = "api"
    PICKLE = "pickle"
    HDF5 = "hdf5"


class DataFrequency(str, Enum):
    """Data frequency/timeframe."""

    TICK = "tick"
    SECOND = "1s"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1M"


class DataType(str, Enum):
    """Types of market data."""

    OHLCV = "ohlcv"
    TRADES = "trades"
    QUOTES = "quotes"
    ORDER_BOOK = "order_book"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"


class DataLoaderConfig(BaseModel):
    """Configuration for data loader."""

    cache_enabled: bool = Field(default=True, description="Enable data caching")
    cache_dir: str = Field(default="./cache/data", description="Cache directory")
    cache_ttl_hours: int = Field(default=24, description="Cache time-to-live in hours")
    validate_data: bool = Field(default=True, description="Validate loaded data")
    fill_missing: bool = Field(default=True, description="Fill missing values")
    fill_method: str = Field(default="ffill", description="Method for filling missing values")
    adjust_splits: bool = Field(default=True, description="Adjust for stock splits")
    adjust_dividends: bool = Field(default=True, description="Adjust for dividends")
    min_data_points: int = Field(default=100, description="Minimum required data points")
    max_gap_days: int = Field(default=5, description="Maximum allowed gap in days")
    timezone: str = Field(default="UTC", description="Data timezone")
    parallel_loading: bool = Field(default=True, description="Enable parallel loading")
    max_workers: int = Field(default=4, description="Maximum parallel workers")


class DataRequest(BaseModel):
    """Request for loading data."""

    symbols: list[str] = Field(description="List of symbols to load")
    start_date: datetime = Field(description="Start date for data")
    end_date: datetime = Field(description="End date for data")
    frequency: DataFrequency = Field(default=DataFrequency.DAILY, description="Data frequency")
    data_type: DataType = Field(default=DataType.OHLCV, description="Type of data")
    source: DataSource = Field(default=DataSource.CSV, description="Data source")
    source_path: str | None = Field(default=None, description="Path for file-based sources")
    columns: list[str] | None = Field(default=None, description="Specific columns to load")
    filters: dict[str, Any] | None = Field(default=None, description="Data filters")


class DataValidationResult(BaseModel):
    """Result of data validation."""

    is_valid: bool = Field(description="Whether data is valid")
    symbol: str = Field(description="Symbol validated")
    total_rows: int = Field(description="Total number of rows")
    missing_values: dict[str, int] = Field(default_factory=dict, description="Missing values per column")
    gaps: list[tuple[datetime, datetime]] = Field(default_factory=list, description="Data gaps")
    outliers: dict[str, int] = Field(default_factory=dict, description="Outliers per column")
    issues: list[str] = Field(default_factory=list, description="Validation issues")
    warnings: list[str] = Field(default_factory=list, description="Validation warnings")


class LoadedData(BaseModel):
    """Container for loaded data."""

    model_config = {"arbitrary_types_allowed": True}

    symbol: str = Field(description="Symbol")
    data: Any = Field(description="DataFrame with loaded data")
    frequency: DataFrequency = Field(description="Data frequency")
    data_type: DataType = Field(description="Data type")
    start_date: datetime = Field(description="Actual start date")
    end_date: datetime = Field(description="Actual end date")
    row_count: int = Field(description="Number of rows")
    columns: list[str] = Field(description="Column names")
    validation: DataValidationResult | None = Field(default=None, description="Validation result")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BaseDataProvider(ABC):
    """Abstract base class for data providers."""

    @abstractmethod
    async def load(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        frequency: DataFrequency,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load data for a symbol."""
        pass

    @abstractmethod
    async def get_available_symbols(self) -> list[str]:
        """Get list of available symbols."""
        pass

    @abstractmethod
    async def get_date_range(self, symbol: str) -> tuple[datetime, datetime]:
        """Get available date range for a symbol."""
        pass


class CSVDataProvider(BaseDataProvider):
    """Data provider for CSV files."""

    def __init__(
        self,
        data_dir: str,
        date_column: str = "date",
        date_format: str | None = None,
    ) -> None:
        """
        Initialize CSV data provider.

        Args:
            data_dir: Directory containing CSV files
            date_column: Name of date column
            date_format: Date format string
        """
        self.data_dir = Path(data_dir)
        self.date_column = date_column
        self.date_format = date_format
        self._symbol_cache: dict[str, Path] = {}

    async def load(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        frequency: DataFrequency,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            symbol: Symbol to load
            start_date: Start date
            end_date: End date
            frequency: Data frequency
            **kwargs: Additional arguments

        Returns:
            DataFrame with loaded data
        """
        file_path = self._get_file_path(symbol, frequency)

        if not file_path.exists():
            logger.error(f"CSV file not found: {file_path}")
            raise FileNotFoundError(f"Data file not found for {symbol}")

        try:
            df = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: pd.read_csv(
                    file_path,
                    parse_dates=[self.date_column],
                    date_format=self.date_format,
                ),
            )

            df = df.set_index(self.date_column)
            df.index = pd.to_datetime(df.index)

            mask = (df.index >= start_date) & (df.index <= end_date)
            df = df.loc[mask]

            return df.sort_index()

        except Exception as e:
            logger.error(f"Error loading CSV for {symbol}: {e}")
            raise

    async def get_available_symbols(self) -> list[str]:
        """Get list of available symbols from CSV files."""
        symbols = []

        for file_path in self.data_dir.glob("*.csv"):
            symbol = file_path.stem.upper()
            symbols.append(symbol)
            self._symbol_cache[symbol] = file_path

        return sorted(symbols)

    async def get_date_range(self, symbol: str) -> tuple[datetime, datetime]:
        """Get available date range for a symbol."""
        file_path = self._get_file_path(symbol, DataFrequency.DAILY)

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found for {symbol}")

        df = pd.read_csv(
            file_path,
            parse_dates=[self.date_column],
            usecols=[self.date_column],
        )

        dates = pd.to_datetime(df[self.date_column])
        return dates.min().to_pydatetime(), dates.max().to_pydatetime()

    def _get_file_path(self, symbol: str, frequency: DataFrequency) -> Path:
        """Get file path for a symbol."""
        if symbol in self._symbol_cache:
            return self._symbol_cache[symbol]

        patterns = [
            f"{symbol}.csv",
            f"{symbol.lower()}.csv",
            f"{symbol}_{frequency.value}.csv",
            f"{symbol.lower()}_{frequency.value}.csv",
        ]

        for pattern in patterns:
            file_path = self.data_dir / pattern
            if file_path.exists():
                self._symbol_cache[symbol] = file_path
                return file_path

        return self.data_dir / f"{symbol}.csv"


class ParquetDataProvider(BaseDataProvider):
    """Data provider for Parquet files."""

    def __init__(
        self,
        data_dir: str,
        date_column: str = "date",
    ) -> None:
        """
        Initialize Parquet data provider.

        Args:
            data_dir: Directory containing Parquet files
            date_column: Name of date column
        """
        self.data_dir = Path(data_dir)
        self.date_column = date_column

    async def load(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        frequency: DataFrequency,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Load data from Parquet file.

        Args:
            symbol: Symbol to load
            start_date: Start date
            end_date: End date
            frequency: Data frequency
            **kwargs: Additional arguments

        Returns:
            DataFrame with loaded data
        """
        file_path = self._get_file_path(symbol, frequency)

        if not file_path.exists():
            logger.error(f"Parquet file not found: {file_path}")
            raise FileNotFoundError(f"Data file not found for {symbol}")

        try:
            df = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: pd.read_parquet(
                    file_path,
                    filters=[
                        (self.date_column, ">=", start_date),
                        (self.date_column, "<=", end_date),
                    ],
                ),
            )

            if self.date_column in df.columns:
                df = df.set_index(self.date_column)

            df.index = pd.to_datetime(df.index)
            return df.sort_index()

        except Exception as e:
            logger.error(f"Error loading Parquet for {symbol}: {e}")
            raise

    async def get_available_symbols(self) -> list[str]:
        """Get list of available symbols from Parquet files."""
        symbols = []

        for file_path in self.data_dir.glob("*.parquet"):
            symbol = file_path.stem.upper()
            symbols.append(symbol)

        return sorted(symbols)

    async def get_date_range(self, symbol: str) -> tuple[datetime, datetime]:
        """Get available date range for a symbol."""
        file_path = self._get_file_path(symbol, DataFrequency.DAILY)

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found for {symbol}")

        df = pd.read_parquet(file_path, columns=[self.date_column])
        dates = pd.to_datetime(df[self.date_column])
        return dates.min().to_pydatetime(), dates.max().to_pydatetime()

    def _get_file_path(self, symbol: str, frequency: DataFrequency) -> Path:
        """Get file path for a symbol."""
        patterns = [
            f"{symbol}.parquet",
            f"{symbol.lower()}.parquet",
            f"{symbol}_{frequency.value}.parquet",
        ]

        for pattern in patterns:
            file_path = self.data_dir / pattern
            if file_path.exists():
                return file_path

        return self.data_dir / f"{symbol}.parquet"


class DataCache:
    """Cache manager for loaded data."""

    def __init__(
        self,
        cache_dir: str,
        ttl_hours: int = 24,
    ) -> None:
        """
        Initialize data cache.

        Args:
            cache_dir: Directory for cache files
            ttl_hours: Time-to-live in hours
        """
        self.cache_dir = Path(cache_dir)
        self.ttl_hours = ttl_hours
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, request: DataRequest) -> str:
        """Generate cache key for a request."""
        key_data = f"{request.symbols}_{request.start_date}_{request.end_date}_{request.frequency}_{request.data_type}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{cache_key}.pkl"

    def get(self, request: DataRequest) -> dict[str, pd.DataFrame] | None:
        """
        Get cached data if available and valid.

        Args:
            request: Data request

        Returns:
            Cached data or None if not available
        """
        cache_key = self._get_cache_key(request)
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if cache_age > timedelta(hours=self.ttl_hours):
            logger.debug(f"Cache expired for key {cache_key}")
            cache_path.unlink()
            return None

        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            logger.debug(f"Cache hit for key {cache_key}")
            return data
        except Exception as e:
            logger.warning(f"Error reading cache: {e}")
            return None

    def set(self, request: DataRequest, data: dict[str, pd.DataFrame]) -> None:
        """
        Cache data for a request.

        Args:
            request: Data request
            data: Data to cache
        """
        cache_key = self._get_cache_key(request)
        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            logger.debug(f"Cached data for key {cache_key}")
        except Exception as e:
            logger.warning(f"Error writing cache: {e}")

    def clear(self) -> int:
        """
        Clear all cached data.

        Returns:
            Number of cache files removed
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Error removing cache file: {e}")

        logger.info(f"Cleared {count} cache files")
        return count

    def cleanup_expired(self) -> int:
        """
        Remove expired cache files.

        Returns:
            Number of expired files removed
        """
        count = 0
        cutoff = datetime.now() - timedelta(hours=self.ttl_hours)

        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_time < cutoff:
                    cache_file.unlink()
                    count += 1
            except Exception as e:
                logger.warning(f"Error checking cache file: {e}")

        logger.info(f"Removed {count} expired cache files")
        return count


class DataValidator:
    """Validator for loaded market data."""

    def __init__(
        self,
        min_data_points: int = 100,
        max_gap_days: int = 5,
        outlier_std_threshold: float = 5.0,
    ) -> None:
        """
        Initialize data validator.

        Args:
            min_data_points: Minimum required data points
            max_gap_days: Maximum allowed gap in days
            outlier_std_threshold: Standard deviations for outlier detection
        """
        self.min_data_points = min_data_points
        self.max_gap_days = max_gap_days
        self.outlier_std_threshold = outlier_std_threshold

    def validate(
        self,
        df: pd.DataFrame,
        symbol: str,
        frequency: DataFrequency,
    ) -> DataValidationResult:
        """
        Validate loaded data.

        Args:
            df: DataFrame to validate
            symbol: Symbol being validated
            frequency: Data frequency

        Returns:
            Validation result
        """
        issues: list[str] = []
        warnings: list[str] = []
        missing_values: dict[str, int] = {}
        outliers: dict[str, int] = {}
        gaps: list[tuple[datetime, datetime]] = []

        if len(df) < self.min_data_points:
            issues.append(
                f"Insufficient data points: {len(df)} < {self.min_data_points}"
            )

        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                missing_values[col] = int(null_count)
                pct = null_count / len(df) * 100
                if pct > 5:
                    warnings.append(f"Column {col} has {pct:.1f}% missing values")

        if isinstance(df.index, pd.DatetimeIndex):
            expected_freq = self._get_expected_frequency(frequency)
            if expected_freq:
                date_diffs = df.index.to_series().diff()
                large_gaps = date_diffs[date_diffs > timedelta(days=self.max_gap_days)]

                for idx in large_gaps.index:
                    gap_start = df.index[df.index.get_loc(idx) - 1]
                    gap_end = idx
                    gaps.append((gap_start.to_pydatetime(), gap_end.to_pydatetime()))
                    warnings.append(
                        f"Gap detected: {gap_start.date()} to {gap_end.date()}"
                    )

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ["volume", "Volume"]:
                continue

            series = df[col].dropna()
            if len(series) > 0:
                mean = series.mean()
                std = series.std()
                if std > 0:
                    z_scores = np.abs((series - mean) / std)
                    outlier_count = int((z_scores > self.outlier_std_threshold).sum())
                    if outlier_count > 0:
                        outliers[col] = outlier_count
                        if outlier_count > len(series) * 0.01:
                            warnings.append(
                                f"Column {col} has {outlier_count} outliers"
                            )

        if "close" in df.columns or "Close" in df.columns:
            close_col = "close" if "close" in df.columns else "Close"
            if (df[close_col] <= 0).any():
                issues.append("Negative or zero prices detected")

        if "volume" in df.columns or "Volume" in df.columns:
            vol_col = "volume" if "volume" in df.columns else "Volume"
            if (df[vol_col] < 0).any():
                issues.append("Negative volume detected")

        ohlc_cols = ["open", "high", "low", "close"]
        ohlc_present = all(
            col in df.columns.str.lower() for col in ohlc_cols
        )

        if ohlc_present:
            df_lower = df.copy()
            df_lower.columns = df_lower.columns.str.lower()

            invalid_hl = df_lower["high"] < df_lower["low"]
            if invalid_hl.any():
                issues.append(f"High < Low detected in {invalid_hl.sum()} rows")

            invalid_oh = df_lower["open"] > df_lower["high"]
            if invalid_oh.any():
                warnings.append(f"Open > High detected in {invalid_oh.sum()} rows")

            invalid_ol = df_lower["open"] < df_lower["low"]
            if invalid_ol.any():
                warnings.append(f"Open < Low detected in {invalid_ol.sum()} rows")

        is_valid = len(issues) == 0

        return DataValidationResult(
            is_valid=is_valid,
            symbol=symbol,
            total_rows=len(df),
            missing_values=missing_values,
            gaps=gaps,
            outliers=outliers,
            issues=issues,
            warnings=warnings,
        )

    def _get_expected_frequency(self, frequency: DataFrequency) -> timedelta | None:
        """Get expected timedelta for frequency."""
        freq_map = {
            DataFrequency.MINUTE_1: timedelta(minutes=1),
            DataFrequency.MINUTE_5: timedelta(minutes=5),
            DataFrequency.MINUTE_15: timedelta(minutes=15),
            DataFrequency.MINUTE_30: timedelta(minutes=30),
            DataFrequency.HOUR_1: timedelta(hours=1),
            DataFrequency.HOUR_4: timedelta(hours=4),
            DataFrequency.DAILY: timedelta(days=1),
            DataFrequency.WEEKLY: timedelta(weeks=1),
        }
        return freq_map.get(frequency)


class DataPreprocessor:
    """Preprocessor for market data."""

    def __init__(
        self,
        fill_method: str = "ffill",
        adjust_splits: bool = True,
        adjust_dividends: bool = True,
    ) -> None:
        """
        Initialize data preprocessor.

        Args:
            fill_method: Method for filling missing values
            adjust_splits: Adjust for stock splits
            adjust_dividends: Adjust for dividends
        """
        self.fill_method = fill_method
        self.adjust_splits = adjust_splits
        self.adjust_dividends = adjust_dividends

    def preprocess(
        self,
        df: pd.DataFrame,
        split_data: pd.DataFrame | None = None,
        dividend_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Preprocess market data.

        Args:
            df: DataFrame to preprocess
            split_data: Optional split adjustment data
            dividend_data: Optional dividend adjustment data

        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()

        df.columns = df.columns.str.lower()

        df = self._fill_missing(df)

        if self.adjust_splits and split_data is not None:
            df = self._adjust_for_splits(df, split_data)

        if self.adjust_dividends and dividend_data is not None:
            df = self._adjust_for_dividends(df, dividend_data)

        df = df.sort_index()

        return df

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values."""
        if self.fill_method == "ffill":
            df = df.ffill()
        elif self.fill_method == "bfill":
            df = df.bfill()
        elif self.fill_method == "interpolate":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method="linear")
        elif self.fill_method == "drop":
            df = df.dropna()

        return df

    def _adjust_for_splits(
        self,
        df: pd.DataFrame,
        split_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Adjust prices for stock splits."""
        price_cols = ["open", "high", "low", "close", "adj_close"]

        for _, split in split_data.iterrows():
            split_date = pd.to_datetime(split.get("date", split.name))
            split_ratio = split.get("ratio", split.get("split_ratio", 1.0))

            if split_ratio != 1.0:
                mask = df.index < split_date
                for col in price_cols:
                    if col in df.columns:
                        df.loc[mask, col] = df.loc[mask, col] / split_ratio

                if "volume" in df.columns:
                    df.loc[mask, "volume"] = df.loc[mask, "volume"] * split_ratio

        return df

    def _adjust_for_dividends(
        self,
        df: pd.DataFrame,
        dividend_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Adjust prices for dividends."""
        price_cols = ["open", "high", "low", "close"]

        for _, div in dividend_data.iterrows():
            div_date = pd.to_datetime(div.get("date", div.name))
            div_amount = div.get("amount", div.get("dividend", 0.0))

            if div_amount > 0:
                mask = df.index < div_date
                close_before = df.loc[df.index < div_date, "close"].iloc[-1] if mask.any() else None

                if close_before is not None and close_before > div_amount:
                    adj_factor = (close_before - div_amount) / close_before
                    for col in price_cols:
                        if col in df.columns:
                            df.loc[mask, col] = df.loc[mask, col] * adj_factor

        return df

    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return columns to DataFrame."""
        df = df.copy()

        if "close" in df.columns:
            df["returns"] = df["close"].pct_change()
            df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
            df["cumulative_returns"] = (1 + df["returns"]).cumprod() - 1

        return df

    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical features."""
        df = df.copy()

        if "close" in df.columns:
            df["sma_20"] = df["close"].rolling(window=20).mean()
            df["sma_50"] = df["close"].rolling(window=50).mean()
            df["volatility_20"] = df["close"].pct_change().rolling(window=20).std()

        if all(col in df.columns for col in ["high", "low", "close"]):
            df["true_range"] = np.maximum(
                df["high"] - df["low"],
                np.maximum(
                    np.abs(df["high"] - df["close"].shift(1)),
                    np.abs(df["low"] - df["close"].shift(1)),
                ),
            )
            df["atr_14"] = df["true_range"].rolling(window=14).mean()

        return df


class DataLoader:
    """Main data loader for backtesting."""

    def __init__(
        self,
        config: DataLoaderConfig | None = None,
    ) -> None:
        """
        Initialize data loader.

        Args:
            config: Loader configuration
        """
        self.config = config or DataLoaderConfig()
        self.providers: dict[DataSource, BaseDataProvider] = {}
        self.cache = DataCache(
            cache_dir=self.config.cache_dir,
            ttl_hours=self.config.cache_ttl_hours,
        ) if self.config.cache_enabled else None
        self.validator = DataValidator(
            min_data_points=self.config.min_data_points,
            max_gap_days=self.config.max_gap_days,
        )
        self.preprocessor = DataPreprocessor(
            fill_method=self.config.fill_method,
            adjust_splits=self.config.adjust_splits,
            adjust_dividends=self.config.adjust_dividends,
        )

        logger.info("DataLoader initialized")

    def register_provider(
        self,
        source: DataSource,
        provider: BaseDataProvider,
    ) -> None:
        """
        Register a data provider.

        Args:
            source: Data source type
            provider: Provider instance
        """
        self.providers[source] = provider
        logger.info(f"Registered provider for {source.value}")

    def register_csv_provider(
        self,
        data_dir: str,
        date_column: str = "date",
        date_format: str | None = None,
    ) -> None:
        """
        Register CSV data provider.

        Args:
            data_dir: Directory containing CSV files
            date_column: Name of date column
            date_format: Date format string
        """
        provider = CSVDataProvider(
            data_dir=data_dir,
            date_column=date_column,
            date_format=date_format,
        )
        self.register_provider(DataSource.CSV, provider)

    def register_parquet_provider(
        self,
        data_dir: str,
        date_column: str = "date",
    ) -> None:
        """
        Register Parquet data provider.

        Args:
            data_dir: Directory containing Parquet files
            date_column: Name of date column
        """
        provider = ParquetDataProvider(
            data_dir=data_dir,
            date_column=date_column,
        )
        self.register_provider(DataSource.PARQUET, provider)

    async def load(self, request: DataRequest) -> dict[str, LoadedData]:
        """
        Load data for multiple symbols.

        Args:
            request: Data request

        Returns:
            Dictionary mapping symbols to loaded data
        """
        if self.cache:
            cached = self.cache.get(request)
            if cached:
                logger.info(f"Using cached data for {len(request.symbols)} symbols")
                return self._convert_cached_data(cached, request)

        provider = self.providers.get(request.source)
        if not provider:
            raise ValueError(f"No provider registered for {request.source.value}")

        if self.config.parallel_loading and len(request.symbols) > 1:
            results = await self._load_parallel(request, provider)
        else:
            results = await self._load_sequential(request, provider)

        if self.cache:
            cache_data = {
                symbol: result.data for symbol, result in results.items()
            }
            self.cache.set(request, cache_data)

        return results

    async def _load_parallel(
        self,
        request: DataRequest,
        provider: BaseDataProvider,
    ) -> dict[str, LoadedData]:
        """Load data for multiple symbols in parallel."""
        semaphore = asyncio.Semaphore(self.config.max_workers)

        async def load_with_semaphore(symbol: str) -> tuple[str, LoadedData | None]:
            async with semaphore:
                try:
                    return symbol, await self._load_single(
                        symbol, request, provider
                    )
                except Exception as e:
                    logger.error(f"Error loading {symbol}: {e}")
                    return symbol, None

        tasks = [load_with_semaphore(symbol) for symbol in request.symbols]
        results = await asyncio.gather(*tasks)

        return {
            symbol: data for symbol, data in results if data is not None
        }

    async def _load_sequential(
        self,
        request: DataRequest,
        provider: BaseDataProvider,
    ) -> dict[str, LoadedData]:
        """Load data for multiple symbols sequentially."""
        results: dict[str, LoadedData] = {}

        for symbol in request.symbols:
            try:
                results[symbol] = await self._load_single(
                    symbol, request, provider
                )
            except Exception as e:
                logger.error(f"Error loading {symbol}: {e}")

        return results

    async def _load_single(
        self,
        symbol: str,
        request: DataRequest,
        provider: BaseDataProvider,
    ) -> LoadedData:
        """Load data for a single symbol."""
        df = await provider.load(
            symbol=symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            frequency=request.frequency,
        )

        if request.columns:
            available_cols = [col for col in request.columns if col in df.columns]
            df = df[available_cols]

        if self.config.fill_missing:
            df = self.preprocessor.preprocess(df)

        validation = None
        if self.config.validate_data:
            validation = self.validator.validate(df, symbol, request.frequency)
            if not validation.is_valid:
                logger.warning(
                    f"Validation issues for {symbol}: {validation.issues}"
                )

        return LoadedData(
            symbol=symbol,
            data=df,
            frequency=request.frequency,
            data_type=request.data_type,
            start_date=df.index.min().to_pydatetime() if len(df) > 0 else request.start_date,
            end_date=df.index.max().to_pydatetime() if len(df) > 0 else request.end_date,
            row_count=len(df),
            columns=list(df.columns),
            validation=validation,
        )

    def _convert_cached_data(
        self,
        cached: dict[str, pd.DataFrame],
        request: DataRequest,
    ) -> dict[str, LoadedData]:
        """Convert cached DataFrames to LoadedData objects."""
        results: dict[str, LoadedData] = {}

        for symbol, df in cached.items():
            validation = None
            if self.config.validate_data:
                validation = self.validator.validate(df, symbol, request.frequency)

            results[symbol] = LoadedData(
                symbol=symbol,
                data=df,
                frequency=request.frequency,
                data_type=request.data_type,
                start_date=df.index.min().to_pydatetime() if len(df) > 0 else request.start_date,
                end_date=df.index.max().to_pydatetime() if len(df) > 0 else request.end_date,
                row_count=len(df),
                columns=list(df.columns),
                validation=validation,
            )

        return results

    async def load_multi_timeframe(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        frequencies: list[DataFrequency],
        source: DataSource = DataSource.CSV,
    ) -> dict[str, dict[DataFrequency, LoadedData]]:
        """
        Load data for multiple timeframes.

        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            frequencies: List of frequencies to load
            source: Data source

        Returns:
            Nested dictionary of symbol -> frequency -> data
        """
        results: dict[str, dict[DataFrequency, LoadedData]] = {}

        for symbol in symbols:
            results[symbol] = {}

            for freq in frequencies:
                request = DataRequest(
                    symbols=[symbol],
                    start_date=start_date,
                    end_date=end_date,
                    frequency=freq,
                    source=source,
                )

                loaded = await self.load(request)
                if symbol in loaded:
                    results[symbol][freq] = loaded[symbol]

        return results

    def clear_cache(self) -> int:
        """Clear all cached data."""
        if self.cache:
            return self.cache.clear()
        return 0

    def cleanup_expired_cache(self) -> int:
        """Remove expired cache entries."""
        if self.cache:
            return self.cache.cleanup_expired()
        return 0


def create_data_loader(
    csv_dir: str | None = None,
    parquet_dir: str | None = None,
    config: dict | None = None,
) -> DataLoader:
    """
    Create and configure a data loader.

    Args:
        csv_dir: Directory for CSV files
        parquet_dir: Directory for Parquet files
        config: Optional configuration dictionary

    Returns:
        Configured DataLoader instance
    """
    loader_config = DataLoaderConfig(**(config or {}))
    loader = DataLoader(config=loader_config)

    if csv_dir:
        loader.register_csv_provider(csv_dir)

    if parquet_dir:
        loader.register_parquet_provider(parquet_dir)

    return loader
