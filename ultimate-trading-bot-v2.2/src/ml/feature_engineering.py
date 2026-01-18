"""
Feature Engineering for Machine Learning.

This module provides comprehensive feature engineering capabilities
for trading data, including technical indicators and transformations.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FeatureType(str, Enum):
    """Types of features."""

    PRICE = "price"
    VOLUME = "volume"
    RETURNS = "returns"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    TREND = "trend"
    PATTERN = "pattern"
    TIME = "time"
    STATISTICAL = "statistical"
    CUSTOM = "custom"


class FeatureTransform(str, Enum):
    """Feature transformation types."""

    NONE = "none"
    LOG = "log"
    SQRT = "sqrt"
    DIFF = "diff"
    PCT_CHANGE = "pct_change"
    Z_SCORE = "z_score"
    RANK = "rank"
    WINSORIZE = "winsorize"


class FeatureEngineeringConfig(BaseModel):
    """Configuration for feature engineering."""

    include_price_features: bool = Field(default=True, description="Include price features")
    include_volume_features: bool = Field(default=True, description="Include volume features")
    include_returns_features: bool = Field(default=True, description="Include return features")
    include_momentum_features: bool = Field(default=True, description="Include momentum features")
    include_volatility_features: bool = Field(default=True, description="Include volatility features")
    include_trend_features: bool = Field(default=True, description="Include trend features")
    include_time_features: bool = Field(default=True, description="Include time features")
    include_statistical_features: bool = Field(default=True, description="Include statistical features")
    lookback_periods: list[int] = Field(
        default=[5, 10, 20, 50, 100],
        description="Lookback periods for features",
    )
    drop_na: bool = Field(default=True, description="Drop rows with NaN")
    fill_method: str | None = Field(default="ffill", description="Fill method for NaN")


@dataclass
class FeatureInfo:
    """Information about a generated feature."""

    name: str
    feature_type: FeatureType
    description: str
    lookback: int = 0
    transform: FeatureTransform = FeatureTransform.NONE
    importance: float = 0.0


@dataclass
class FeatureSet:
    """Collection of generated features."""

    features: pd.DataFrame
    feature_info: list[FeatureInfo]
    target: pd.Series | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def feature_names(self) -> list[str]:
        """Get feature names."""
        return list(self.features.columns)

    @property
    def n_features(self) -> int:
        """Get number of features."""
        return len(self.features.columns)


class TechnicalFeatureGenerator:
    """Generator for technical analysis features."""

    def __init__(self, periods: list[int] | None = None) -> None:
        """
        Initialize technical feature generator.

        Args:
            periods: Lookback periods
        """
        self.periods = periods or [5, 10, 20, 50, 100]

    def generate_sma(self, data: pd.DataFrame, column: str = "close") -> pd.DataFrame:
        """Generate Simple Moving Average features."""
        features = pd.DataFrame(index=data.index)

        for period in self.periods:
            features[f"sma_{period}"] = data[column].rolling(window=period).mean()
            features[f"sma_{period}_ratio"] = data[column] / features[f"sma_{period}"]

        return features

    def generate_ema(self, data: pd.DataFrame, column: str = "close") -> pd.DataFrame:
        """Generate Exponential Moving Average features."""
        features = pd.DataFrame(index=data.index)

        for period in self.periods:
            features[f"ema_{period}"] = data[column].ewm(span=period, adjust=False).mean()
            features[f"ema_{period}_ratio"] = data[column] / features[f"ema_{period}"]

        return features

    def generate_rsi(
        self,
        data: pd.DataFrame,
        column: str = "close",
        periods: list[int] | None = None,
    ) -> pd.DataFrame:
        """Generate RSI features."""
        features = pd.DataFrame(index=data.index)
        periods = periods or [14, 21]

        delta = data[column].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        for period in periods:
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()

            rs = avg_gain / (avg_loss + 1e-10)
            features[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        return features

    def generate_macd(
        self,
        data: pd.DataFrame,
        column: str = "close",
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """Generate MACD features."""
        features = pd.DataFrame(index=data.index)

        ema_fast = data[column].ewm(span=fast, adjust=False).mean()
        ema_slow = data[column].ewm(span=slow, adjust=False).mean()

        features["macd_line"] = ema_fast - ema_slow
        features["macd_signal"] = features["macd_line"].ewm(span=signal, adjust=False).mean()
        features["macd_histogram"] = features["macd_line"] - features["macd_signal"]
        features["macd_crossover"] = (
            (features["macd_line"] > features["macd_signal"]).astype(int)
            - (features["macd_line"] < features["macd_signal"]).astype(int)
        )

        return features

    def generate_bollinger_bands(
        self,
        data: pd.DataFrame,
        column: str = "close",
        period: int = 20,
        num_std: float = 2.0,
    ) -> pd.DataFrame:
        """Generate Bollinger Bands features."""
        features = pd.DataFrame(index=data.index)

        sma = data[column].rolling(window=period).mean()
        std = data[column].rolling(window=period).std()

        features["bb_upper"] = sma + num_std * std
        features["bb_middle"] = sma
        features["bb_lower"] = sma - num_std * std
        features["bb_width"] = (features["bb_upper"] - features["bb_lower"]) / features["bb_middle"]
        features["bb_position"] = (data[column] - features["bb_lower"]) / (
            features["bb_upper"] - features["bb_lower"] + 1e-10
        )

        return features

    def generate_atr(
        self,
        data: pd.DataFrame,
        periods: list[int] | None = None,
    ) -> pd.DataFrame:
        """Generate Average True Range features."""
        features = pd.DataFrame(index=data.index)
        periods = periods or [14, 21]

        high = data["high"]
        low = data["low"]
        close = data["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        for period in periods:
            features[f"atr_{period}"] = true_range.rolling(window=period).mean()
            features[f"atr_{period}_pct"] = features[f"atr_{period}"] / close

        return features

    def generate_adx(
        self,
        data: pd.DataFrame,
        period: int = 14,
    ) -> pd.DataFrame:
        """Generate Average Directional Index features."""
        features = pd.DataFrame(index=data.index)

        high = data["high"]
        low = data["low"]
        close = data["close"]

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = true_range.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 1e-10))

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)

        features[f"adx_{period}"] = dx.rolling(window=period).mean()
        features[f"plus_di_{period}"] = plus_di
        features[f"minus_di_{period}"] = minus_di

        return features

    def generate_stochastic(
        self,
        data: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
    ) -> pd.DataFrame:
        """Generate Stochastic Oscillator features."""
        features = pd.DataFrame(index=data.index)

        low_min = data["low"].rolling(window=k_period).min()
        high_max = data["high"].rolling(window=k_period).max()

        features["stoch_k"] = 100 * (data["close"] - low_min) / (high_max - low_min + 1e-10)
        features["stoch_d"] = features["stoch_k"].rolling(window=d_period).mean()
        features["stoch_crossover"] = (
            (features["stoch_k"] > features["stoch_d"]).astype(int)
            - (features["stoch_k"] < features["stoch_d"]).astype(int)
        )

        return features

    def generate_williams_r(
        self,
        data: pd.DataFrame,
        period: int = 14,
    ) -> pd.DataFrame:
        """Generate Williams %R feature."""
        features = pd.DataFrame(index=data.index)

        high_max = data["high"].rolling(window=period).max()
        low_min = data["low"].rolling(window=period).min()

        features[f"williams_r_{period}"] = -100 * (high_max - data["close"]) / (
            high_max - low_min + 1e-10
        )

        return features

    def generate_obv(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate On-Balance Volume features."""
        features = pd.DataFrame(index=data.index)

        direction = np.sign(data["close"].diff())
        obv = (direction * data["volume"]).cumsum()

        features["obv"] = obv
        features["obv_sma_10"] = obv.rolling(window=10).mean()
        features["obv_sma_20"] = obv.rolling(window=20).mean()

        return features

    def generate_vwap(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate VWAP features."""
        features = pd.DataFrame(index=data.index)

        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        cumulative_tp_volume = (typical_price * data["volume"]).cumsum()
        cumulative_volume = data["volume"].cumsum()

        features["vwap"] = cumulative_tp_volume / (cumulative_volume + 1e-10)
        features["vwap_ratio"] = data["close"] / features["vwap"]

        return features


class StatisticalFeatureGenerator:
    """Generator for statistical features."""

    def __init__(self, periods: list[int] | None = None) -> None:
        """
        Initialize statistical feature generator.

        Args:
            periods: Lookback periods
        """
        self.periods = periods or [5, 10, 20, 50]

    def generate_returns(
        self,
        data: pd.DataFrame,
        column: str = "close",
    ) -> pd.DataFrame:
        """Generate return features."""
        features = pd.DataFrame(index=data.index)

        features["return_1d"] = data[column].pct_change()
        features["log_return_1d"] = np.log(data[column] / data[column].shift(1))

        for period in self.periods:
            features[f"return_{period}d"] = data[column].pct_change(period)
            features[f"log_return_{period}d"] = np.log(data[column] / data[column].shift(period))

        return features

    def generate_volatility(
        self,
        data: pd.DataFrame,
        column: str = "close",
    ) -> pd.DataFrame:
        """Generate volatility features."""
        features = pd.DataFrame(index=data.index)

        returns = data[column].pct_change()

        for period in self.periods:
            features[f"volatility_{period}d"] = returns.rolling(window=period).std()
            features[f"volatility_{period}d_ann"] = features[f"volatility_{period}d"] * np.sqrt(252)

        features["parkinson_vol"] = np.sqrt(
            (np.log(data["high"] / data["low"]) ** 2).rolling(window=20).mean() / (4 * np.log(2))
        )

        if "open" in data.columns:
            features["garman_klass_vol"] = np.sqrt(
                0.5 * (np.log(data["high"] / data["low"]) ** 2).rolling(window=20).mean()
                - (2 * np.log(2) - 1)
                * (np.log(data["close"] / data["open"]) ** 2).rolling(window=20).mean()
            )

        return features

    def generate_momentum(
        self,
        data: pd.DataFrame,
        column: str = "close",
    ) -> pd.DataFrame:
        """Generate momentum features."""
        features = pd.DataFrame(index=data.index)

        for period in self.periods:
            features[f"momentum_{period}d"] = data[column] / data[column].shift(period) - 1
            features[f"roc_{period}d"] = (data[column] - data[column].shift(period)) / data[column].shift(period) * 100

        return features

    def generate_statistical_moments(
        self,
        data: pd.DataFrame,
        column: str = "close",
    ) -> pd.DataFrame:
        """Generate statistical moment features."""
        features = pd.DataFrame(index=data.index)

        returns = data[column].pct_change()

        for period in self.periods:
            rolling = returns.rolling(window=period)

            features[f"skewness_{period}d"] = rolling.skew()
            features[f"kurtosis_{period}d"] = rolling.kurt()

            features[f"zscore_{period}d"] = (
                data[column] - data[column].rolling(window=period).mean()
            ) / (data[column].rolling(window=period).std() + 1e-10)

        return features

    def generate_range_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate price range features."""
        features = pd.DataFrame(index=data.index)

        features["daily_range"] = (data["high"] - data["low"]) / data["close"]
        features["daily_range_pct"] = (data["high"] - data["low"]) / data["open"]

        if "open" in data.columns:
            features["gap"] = (data["open"] - data["close"].shift(1)) / data["close"].shift(1)
            features["gap_fill"] = (
                (data["low"] <= data["close"].shift(1)) & (data["gap"] > 0)
            ).astype(int) + (
                (data["high"] >= data["close"].shift(1)) & (data["gap"] < 0)
            ).astype(int)

        return features


class TimeFeatureGenerator:
    """Generator for time-based features."""

    def generate_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate time-based features."""
        features = pd.DataFrame(index=data.index)

        if isinstance(data.index, pd.DatetimeIndex):
            dt = data.index
        else:
            dt = pd.to_datetime(data.index)

        features["day_of_week"] = dt.dayofweek
        features["day_of_month"] = dt.day
        features["week_of_year"] = dt.isocalendar().week.astype(int)
        features["month"] = dt.month
        features["quarter"] = dt.quarter

        features["is_monday"] = (dt.dayofweek == 0).astype(int)
        features["is_friday"] = (dt.dayofweek == 4).astype(int)
        features["is_month_start"] = dt.is_month_start.astype(int)
        features["is_month_end"] = dt.is_month_end.astype(int)
        features["is_quarter_start"] = dt.is_quarter_start.astype(int)
        features["is_quarter_end"] = dt.is_quarter_end.astype(int)

        features["sin_day_of_week"] = np.sin(2 * np.pi * features["day_of_week"] / 7)
        features["cos_day_of_week"] = np.cos(2 * np.pi * features["day_of_week"] / 7)
        features["sin_day_of_month"] = np.sin(2 * np.pi * features["day_of_month"] / 31)
        features["cos_day_of_month"] = np.cos(2 * np.pi * features["day_of_month"] / 31)
        features["sin_month"] = np.sin(2 * np.pi * features["month"] / 12)
        features["cos_month"] = np.cos(2 * np.pi * features["month"] / 12)

        return features


class FeatureEngineer:
    """Main feature engineering class."""

    def __init__(
        self,
        config: FeatureEngineeringConfig | None = None,
    ) -> None:
        """
        Initialize feature engineer.

        Args:
            config: Feature engineering configuration
        """
        self.config = config or FeatureEngineeringConfig()

        self.technical_gen = TechnicalFeatureGenerator(self.config.lookback_periods)
        self.statistical_gen = StatisticalFeatureGenerator(self.config.lookback_periods)
        self.time_gen = TimeFeatureGenerator()

        self._feature_info: list[FeatureInfo] = []

        logger.info("FeatureEngineer initialized")

    def generate_features(
        self,
        data: pd.DataFrame,
        target_column: str | None = None,
        target_shift: int = 1,
    ) -> FeatureSet:
        """
        Generate comprehensive feature set.

        Args:
            data: OHLCV data
            target_column: Target column for prediction
            target_shift: Periods to shift target (for prediction)

        Returns:
            FeatureSet with all generated features
        """
        all_features = pd.DataFrame(index=data.index)
        self._feature_info = []

        if self.config.include_price_features:
            price_features = self._generate_price_features(data)
            all_features = pd.concat([all_features, price_features], axis=1)

        if self.config.include_volume_features and "volume" in data.columns:
            volume_features = self._generate_volume_features(data)
            all_features = pd.concat([all_features, volume_features], axis=1)

        if self.config.include_returns_features:
            return_features = self.statistical_gen.generate_returns(data)
            all_features = pd.concat([all_features, return_features], axis=1)

        if self.config.include_momentum_features:
            momentum_features = self.statistical_gen.generate_momentum(data)
            all_features = pd.concat([all_features, momentum_features], axis=1)

        if self.config.include_volatility_features:
            vol_features = self.statistical_gen.generate_volatility(data)
            all_features = pd.concat([all_features, vol_features], axis=1)

        if self.config.include_trend_features:
            trend_features = self._generate_trend_features(data)
            all_features = pd.concat([all_features, trend_features], axis=1)

        if self.config.include_time_features:
            time_features = self.time_gen.generate_time_features(data)
            all_features = pd.concat([all_features, time_features], axis=1)

        if self.config.include_statistical_features:
            stat_features = self.statistical_gen.generate_statistical_moments(data)
            range_features = self.statistical_gen.generate_range_features(data)
            all_features = pd.concat([all_features, stat_features, range_features], axis=1)

        target = None
        if target_column and target_column in data.columns:
            target = data[target_column].shift(-target_shift)
            target.name = "target"

        if self.config.fill_method:
            all_features = all_features.ffill().bfill()

        if self.config.drop_na:
            valid_idx = all_features.dropna().index
            all_features = all_features.loc[valid_idx]
            if target is not None:
                target = target.loc[valid_idx]

        logger.info(f"Generated {len(all_features.columns)} features")

        return FeatureSet(
            features=all_features,
            feature_info=self._feature_info,
            target=target,
        )

    def _generate_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate price-based features."""
        features = pd.DataFrame(index=data.index)

        sma = self.technical_gen.generate_sma(data)
        ema = self.technical_gen.generate_ema(data)
        bb = self.technical_gen.generate_bollinger_bands(data)

        features = pd.concat([features, sma, ema, bb], axis=1)

        return features

    def _generate_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volume-based features."""
        features = pd.DataFrame(index=data.index)

        obv = self.technical_gen.generate_obv(data)
        vwap = self.technical_gen.generate_vwap(data)

        for period in self.config.lookback_periods:
            features[f"volume_sma_{period}"] = data["volume"].rolling(window=period).mean()
            features[f"volume_ratio_{period}"] = data["volume"] / features[f"volume_sma_{period}"]

        features = pd.concat([features, obv, vwap], axis=1)

        return features

    def _generate_trend_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trend-based features."""
        features = pd.DataFrame(index=data.index)

        rsi = self.technical_gen.generate_rsi(data)
        macd = self.technical_gen.generate_macd(data)
        stoch = self.technical_gen.generate_stochastic(data)

        if all(col in data.columns for col in ["high", "low", "close"]):
            atr = self.technical_gen.generate_atr(data)
            adx = self.technical_gen.generate_adx(data)
            features = pd.concat([features, atr, adx], axis=1)

        features = pd.concat([features, rsi, macd, stoch], axis=1)

        return features

    def add_custom_feature(
        self,
        data: pd.DataFrame,
        name: str,
        func: Callable[[pd.DataFrame], pd.Series],
        feature_type: FeatureType = FeatureType.CUSTOM,
    ) -> pd.Series:
        """
        Add a custom feature.

        Args:
            data: Input data
            name: Feature name
            func: Function to compute feature
            feature_type: Type of feature

        Returns:
            Computed feature series
        """
        feature = func(data)
        feature.name = name

        self._feature_info.append(FeatureInfo(
            name=name,
            feature_type=feature_type,
            description=f"Custom feature: {name}",
        ))

        return feature


def create_feature_engineer(
    periods: list[int] | None = None,
    config: dict | None = None,
) -> FeatureEngineer:
    """
    Create a feature engineer.

    Args:
        periods: Lookback periods
        config: Additional configuration

    Returns:
        Configured FeatureEngineer
    """
    fe_config = FeatureEngineeringConfig(
        lookback_periods=periods or [5, 10, 20, 50, 100],
        **(config or {}),
    )
    return FeatureEngineer(fe_config)
