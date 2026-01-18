"""
Signal Generator Module for Ultimate Trading Bot v2.2

Generates trading signals from ML model predictions with confidence filtering,
signal combination, and risk-adjusted signal generation.

Author: AI Assistant
Version: 2.2.0
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of trading signals."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


class SignalStrength(Enum):
    """Signal strength levels."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class SignalCombinationMethod(Enum):
    """Methods for combining signals."""
    VOTING = "voting"
    WEIGHTED_VOTING = "weighted_voting"
    UNANIMOUS = "unanimous"
    MAJORITY = "majority"
    THRESHOLD = "threshold"


@dataclass
class TradingSignal:
    """Trading signal with metadata."""
    signal_type: SignalType
    strength: SignalStrength
    confidence: float
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    expiry: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "signal_type": self.signal_type.value,
            "strength": self.strength.value,
            "confidence": self.confidence,
            "price_target": self.price_target,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "position_size": self.position_size,
            "timestamp": self.timestamp.isoformat(),
            "expiry": self.expiry.isoformat() if self.expiry else None,
            "metadata": self.metadata
        }

    @property
    def is_bullish(self) -> bool:
        """Check if signal is bullish."""
        return self.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]

    @property
    def is_bearish(self) -> bool:
        """Check if signal is bearish."""
        return self.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]


@dataclass
class SignalGeneratorConfig:
    """Configuration for signal generator."""
    min_confidence: float = 0.5
    strong_signal_threshold: float = 0.8
    neutral_zone: tuple[float, float] = (-0.1, 0.1)
    position_sizing_method: str = "kelly"
    max_position_size: float = 0.1
    stop_loss_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 3.0
    signal_expiry_bars: int = 5

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_confidence": self.min_confidence,
            "strong_signal_threshold": self.strong_signal_threshold,
            "neutral_zone": self.neutral_zone,
            "position_sizing_method": self.position_sizing_method,
            "max_position_size": self.max_position_size,
            "stop_loss_atr_multiplier": self.stop_loss_atr_multiplier,
            "take_profit_atr_multiplier": self.take_profit_atr_multiplier,
            "signal_expiry_bars": self.signal_expiry_bars
        }


@dataclass
class SignalMetrics:
    """Metrics for signal performance."""
    total_signals: int
    buy_signals: int
    sell_signals: int
    hold_signals: int
    avg_confidence: float
    signal_accuracy: float
    profitable_signals: int
    avg_profit_per_signal: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_signals": self.total_signals,
            "buy_signals": self.buy_signals,
            "sell_signals": self.sell_signals,
            "hold_signals": self.hold_signals,
            "avg_confidence": self.avg_confidence,
            "signal_accuracy": self.signal_accuracy,
            "profitable_signals": self.profitable_signals,
            "avg_profit_per_signal": self.avg_profit_per_signal
        }


class BaseSignalGenerator(ABC):
    """Base class for signal generators."""

    def __init__(self, name: str, config: Optional[SignalGeneratorConfig] = None):
        """
        Initialize signal generator.

        Args:
            name: Generator name
            config: Configuration
        """
        self.name = name
        self.config = config or SignalGeneratorConfig()
        self._signal_history: list[TradingSignal] = []

        logger.info(f"Initialized {self.__class__.__name__}: {name}")

    @abstractmethod
    async def generate(
        self,
        prediction: np.ndarray,
        confidence: np.ndarray,
        current_price: float,
        **kwargs: Any
    ) -> TradingSignal:
        """Generate trading signal."""
        pass

    def get_signal_history(self, n_recent: int = 100) -> list[TradingSignal]:
        """Get recent signal history."""
        return self._signal_history[-n_recent:]

    def clear_history(self) -> None:
        """Clear signal history."""
        self._signal_history.clear()


class DirectionSignalGenerator(BaseSignalGenerator):
    """Generates signals from directional predictions."""

    def __init__(
        self,
        name: str = "DirectionSignal",
        config: Optional[SignalGeneratorConfig] = None
    ):
        """
        Initialize direction signal generator.

        Args:
            name: Generator name
            config: Configuration
        """
        super().__init__(name, config)

    async def generate(
        self,
        prediction: np.ndarray,
        confidence: np.ndarray,
        current_price: float,
        volatility: Optional[float] = None,
        **kwargs: Any
    ) -> TradingSignal:
        """
        Generate signal from direction prediction.

        Args:
            prediction: Predicted direction/returns
            confidence: Prediction confidence
            current_price: Current market price
            volatility: Current volatility (ATR)
            **kwargs: Additional arguments

        Returns:
            TradingSignal object
        """
        try:
            pred_value = float(prediction[-1]) if len(prediction) > 0 else 0.0
            conf_value = float(np.mean(confidence))

            signal_type, strength = self._determine_signal(pred_value, conf_value)

            if volatility:
                stop_loss = self._calculate_stop_loss(
                    current_price, signal_type, volatility
                )
                take_profit = self._calculate_take_profit(
                    current_price, signal_type, volatility
                )
            else:
                stop_loss = None
                take_profit = None

            position_size = self._calculate_position_size(
                conf_value, pred_value, volatility
            )

            price_target = self._calculate_price_target(
                current_price, pred_value
            )

            signal = TradingSignal(
                signal_type=signal_type,
                strength=strength,
                confidence=conf_value,
                price_target=price_target,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                metadata={
                    "prediction": pred_value,
                    "current_price": current_price,
                    "volatility": volatility
                }
            )

            self._signal_history.append(signal)

            logger.debug(
                f"Generated {signal_type.value} signal with "
                f"confidence={conf_value:.2f}, strength={strength.value}"
            )

            return signal

        except Exception as e:
            logger.error(f"Error generating direction signal: {e}")
            return TradingSignal(
                signal_type=SignalType.HOLD,
                strength=SignalStrength.WEAK,
                confidence=0.0,
                metadata={"error": str(e)}
            )

    def _determine_signal(
        self,
        prediction: float,
        confidence: float
    ) -> tuple[SignalType, SignalStrength]:
        """Determine signal type and strength."""
        neutral_low, neutral_high = self.config.neutral_zone

        if confidence < self.config.min_confidence:
            return SignalType.HOLD, SignalStrength.WEAK

        if prediction > neutral_high:
            if confidence >= self.config.strong_signal_threshold:
                return SignalType.STRONG_BUY, SignalStrength.VERY_STRONG
            elif confidence >= 0.7:
                return SignalType.BUY, SignalStrength.STRONG
            elif confidence >= 0.6:
                return SignalType.BUY, SignalStrength.MODERATE
            else:
                return SignalType.BUY, SignalStrength.WEAK

        elif prediction < neutral_low:
            if confidence >= self.config.strong_signal_threshold:
                return SignalType.STRONG_SELL, SignalStrength.VERY_STRONG
            elif confidence >= 0.7:
                return SignalType.SELL, SignalStrength.STRONG
            elif confidence >= 0.6:
                return SignalType.SELL, SignalStrength.MODERATE
            else:
                return SignalType.SELL, SignalStrength.WEAK

        else:
            return SignalType.HOLD, SignalStrength.WEAK

    def _calculate_stop_loss(
        self,
        current_price: float,
        signal_type: SignalType,
        volatility: float
    ) -> float:
        """Calculate stop loss level."""
        atr_distance = volatility * self.config.stop_loss_atr_multiplier

        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            return current_price - atr_distance
        elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            return current_price + atr_distance
        else:
            return current_price

    def _calculate_take_profit(
        self,
        current_price: float,
        signal_type: SignalType,
        volatility: float
    ) -> float:
        """Calculate take profit level."""
        atr_distance = volatility * self.config.take_profit_atr_multiplier

        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            return current_price + atr_distance
        elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            return current_price - atr_distance
        else:
            return current_price

    def _calculate_position_size(
        self,
        confidence: float,
        prediction: float,
        volatility: Optional[float]
    ) -> float:
        """Calculate position size."""
        if self.config.position_sizing_method == "kelly":
            win_prob = confidence
            win_loss_ratio = abs(prediction) * 10 if prediction != 0 else 1.0

            kelly = win_prob - (1 - win_prob) / win_loss_ratio
            kelly = max(0, min(kelly, 0.25))

            if volatility and volatility > 0:
                vol_adjustment = 0.02 / volatility
                kelly *= min(1.0, vol_adjustment)

            return min(kelly, self.config.max_position_size)

        elif self.config.position_sizing_method == "fixed":
            return self.config.max_position_size * confidence

        else:
            return self.config.max_position_size * 0.5

    def _calculate_price_target(
        self,
        current_price: float,
        prediction: float
    ) -> float:
        """Calculate price target."""
        return current_price * (1 + prediction)


class ProbabilitySignalGenerator(BaseSignalGenerator):
    """Generates signals from probability predictions."""

    def __init__(
        self,
        name: str = "ProbabilitySignal",
        config: Optional[SignalGeneratorConfig] = None,
        buy_threshold: float = 0.6,
        sell_threshold: float = 0.4
    ):
        """
        Initialize probability signal generator.

        Args:
            name: Generator name
            config: Configuration
            buy_threshold: Probability threshold for buy
            sell_threshold: Probability threshold for sell
        """
        super().__init__(name, config)
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    async def generate(
        self,
        prediction: np.ndarray,
        confidence: np.ndarray,
        current_price: float,
        **kwargs: Any
    ) -> TradingSignal:
        """
        Generate signal from probability prediction.

        Args:
            prediction: Predicted probabilities
            confidence: Prediction confidence
            current_price: Current market price
            **kwargs: Additional arguments

        Returns:
            TradingSignal object
        """
        try:
            prob_value = float(prediction[-1]) if len(prediction) > 0 else 0.5
            conf_value = float(np.mean(confidence))

            if prob_value > self.buy_threshold:
                signal_strength = (prob_value - self.buy_threshold) / (1 - self.buy_threshold)

                if signal_strength > 0.7 and conf_value > 0.7:
                    signal_type = SignalType.STRONG_BUY
                    strength = SignalStrength.VERY_STRONG
                elif signal_strength > 0.5:
                    signal_type = SignalType.BUY
                    strength = SignalStrength.STRONG
                else:
                    signal_type = SignalType.BUY
                    strength = SignalStrength.MODERATE

            elif prob_value < self.sell_threshold:
                signal_strength = (self.sell_threshold - prob_value) / self.sell_threshold

                if signal_strength > 0.7 and conf_value > 0.7:
                    signal_type = SignalType.STRONG_SELL
                    strength = SignalStrength.VERY_STRONG
                elif signal_strength > 0.5:
                    signal_type = SignalType.SELL
                    strength = SignalStrength.STRONG
                else:
                    signal_type = SignalType.SELL
                    strength = SignalStrength.MODERATE

            else:
                signal_type = SignalType.HOLD
                strength = SignalStrength.WEAK

            final_confidence = conf_value * abs(prob_value - 0.5) * 2

            signal = TradingSignal(
                signal_type=signal_type,
                strength=strength,
                confidence=final_confidence,
                metadata={
                    "probability": prob_value,
                    "model_confidence": conf_value,
                    "current_price": current_price
                }
            )

            self._signal_history.append(signal)

            return signal

        except Exception as e:
            logger.error(f"Error generating probability signal: {e}")
            return TradingSignal(
                signal_type=SignalType.HOLD,
                strength=SignalStrength.WEAK,
                confidence=0.0,
                metadata={"error": str(e)}
            )


class MultiModelSignalGenerator(BaseSignalGenerator):
    """Combines signals from multiple models."""

    def __init__(
        self,
        name: str = "MultiModelSignal",
        config: Optional[SignalGeneratorConfig] = None,
        combination_method: SignalCombinationMethod = SignalCombinationMethod.WEIGHTED_VOTING
    ):
        """
        Initialize multi-model signal generator.

        Args:
            name: Generator name
            config: Configuration
            combination_method: Method for combining signals
        """
        super().__init__(name, config)
        self.combination_method = combination_method
        self._sub_generators: list[tuple[BaseSignalGenerator, float]] = []

    def add_generator(
        self,
        generator: BaseSignalGenerator,
        weight: float = 1.0
    ) -> None:
        """
        Add a signal generator.

        Args:
            generator: Signal generator
            weight: Generator weight
        """
        self._sub_generators.append((generator, weight))
        logger.info(f"Added generator {generator.name} with weight {weight}")

    async def generate(
        self,
        prediction: np.ndarray,
        confidence: np.ndarray,
        current_price: float,
        predictions_dict: Optional[dict[str, np.ndarray]] = None,
        confidences_dict: Optional[dict[str, np.ndarray]] = None,
        **kwargs: Any
    ) -> TradingSignal:
        """
        Generate combined signal from multiple models.

        Args:
            prediction: Default prediction
            confidence: Default confidence
            current_price: Current market price
            predictions_dict: Dictionary of model predictions
            confidences_dict: Dictionary of model confidences
            **kwargs: Additional arguments

        Returns:
            TradingSignal object
        """
        try:
            if predictions_dict is None or confidences_dict is None:
                return await super().generate(
                    prediction, confidence, current_price, **kwargs
                )

            sub_signals = []

            for generator, weight in self._sub_generators:
                gen_name = generator.name

                if gen_name in predictions_dict:
                    pred = predictions_dict[gen_name]
                    conf = confidences_dict.get(gen_name, confidence)

                    signal = await generator.generate(
                        pred, conf, current_price, **kwargs
                    )
                    sub_signals.append((signal, weight))

            if not sub_signals:
                return TradingSignal(
                    signal_type=SignalType.HOLD,
                    strength=SignalStrength.WEAK,
                    confidence=0.0
                )

            combined_signal = self._combine_signals(sub_signals)

            self._signal_history.append(combined_signal)

            return combined_signal

        except Exception as e:
            logger.error(f"Error generating multi-model signal: {e}")
            return TradingSignal(
                signal_type=SignalType.HOLD,
                strength=SignalStrength.WEAK,
                confidence=0.0,
                metadata={"error": str(e)}
            )

    def _combine_signals(
        self,
        signals_with_weights: list[tuple[TradingSignal, float]]
    ) -> TradingSignal:
        """Combine multiple signals into one."""
        if not signals_with_weights:
            return TradingSignal(
                signal_type=SignalType.HOLD,
                strength=SignalStrength.WEAK,
                confidence=0.0
            )

        if self.combination_method == SignalCombinationMethod.VOTING:
            return self._voting_combination(signals_with_weights)

        elif self.combination_method == SignalCombinationMethod.WEIGHTED_VOTING:
            return self._weighted_voting_combination(signals_with_weights)

        elif self.combination_method == SignalCombinationMethod.UNANIMOUS:
            return self._unanimous_combination(signals_with_weights)

        elif self.combination_method == SignalCombinationMethod.MAJORITY:
            return self._majority_combination(signals_with_weights)

        else:
            return self._weighted_voting_combination(signals_with_weights)

    def _voting_combination(
        self,
        signals_with_weights: list[tuple[TradingSignal, float]]
    ) -> TradingSignal:
        """Simple voting combination."""
        buy_count = sum(1 for s, _ in signals_with_weights if s.is_bullish)
        sell_count = sum(1 for s, _ in signals_with_weights if s.is_bearish)
        total = len(signals_with_weights)

        avg_confidence = np.mean([s.confidence for s, _ in signals_with_weights])

        if buy_count > sell_count:
            signal_type = SignalType.BUY if buy_count > total / 2 else SignalType.HOLD
        elif sell_count > buy_count:
            signal_type = SignalType.SELL if sell_count > total / 2 else SignalType.HOLD
        else:
            signal_type = SignalType.HOLD

        agreement_ratio = max(buy_count, sell_count) / total
        strength = self._determine_strength(agreement_ratio, avg_confidence)

        return TradingSignal(
            signal_type=signal_type,
            strength=strength,
            confidence=avg_confidence * agreement_ratio,
            metadata={
                "buy_votes": buy_count,
                "sell_votes": sell_count,
                "total_votes": total
            }
        )

    def _weighted_voting_combination(
        self,
        signals_with_weights: list[tuple[TradingSignal, float]]
    ) -> TradingSignal:
        """Weighted voting combination."""
        total_weight = sum(w for _, w in signals_with_weights)

        if total_weight == 0:
            return TradingSignal(
                signal_type=SignalType.HOLD,
                strength=SignalStrength.WEAK,
                confidence=0.0
            )

        score = 0.0
        weighted_confidence = 0.0

        for signal, weight in signals_with_weights:
            if signal.is_bullish:
                score += weight
            elif signal.is_bearish:
                score -= weight

            weighted_confidence += signal.confidence * weight

        score /= total_weight
        weighted_confidence /= total_weight

        if score > 0.3:
            signal_type = SignalType.STRONG_BUY if score > 0.7 else SignalType.BUY
        elif score < -0.3:
            signal_type = SignalType.STRONG_SELL if score < -0.7 else SignalType.SELL
        else:
            signal_type = SignalType.HOLD

        strength = self._determine_strength(abs(score), weighted_confidence)

        return TradingSignal(
            signal_type=signal_type,
            strength=strength,
            confidence=weighted_confidence * abs(score),
            metadata={
                "weighted_score": score,
                "total_weight": total_weight
            }
        )

    def _unanimous_combination(
        self,
        signals_with_weights: list[tuple[TradingSignal, float]]
    ) -> TradingSignal:
        """Unanimous agreement combination."""
        signals = [s for s, _ in signals_with_weights]

        all_bullish = all(s.is_bullish for s in signals)
        all_bearish = all(s.is_bearish for s in signals)

        avg_confidence = np.mean([s.confidence for s in signals])

        if all_bullish:
            return TradingSignal(
                signal_type=SignalType.STRONG_BUY,
                strength=SignalStrength.VERY_STRONG,
                confidence=avg_confidence,
                metadata={"unanimous": True, "direction": "bullish"}
            )
        elif all_bearish:
            return TradingSignal(
                signal_type=SignalType.STRONG_SELL,
                strength=SignalStrength.VERY_STRONG,
                confidence=avg_confidence,
                metadata={"unanimous": True, "direction": "bearish"}
            )
        else:
            return TradingSignal(
                signal_type=SignalType.HOLD,
                strength=SignalStrength.WEAK,
                confidence=0.0,
                metadata={"unanimous": False}
            )

    def _majority_combination(
        self,
        signals_with_weights: list[tuple[TradingSignal, float]]
    ) -> TradingSignal:
        """Majority vote combination."""
        signals = [s for s, _ in signals_with_weights]

        buy_count = sum(1 for s in signals if s.is_bullish)
        sell_count = sum(1 for s in signals if s.is_bearish)
        total = len(signals)

        avg_confidence = np.mean([s.confidence for s in signals])

        if buy_count > total / 2:
            majority_ratio = buy_count / total
            signal_type = SignalType.BUY
        elif sell_count > total / 2:
            majority_ratio = sell_count / total
            signal_type = SignalType.SELL
        else:
            majority_ratio = 0.0
            signal_type = SignalType.HOLD

        strength = self._determine_strength(majority_ratio, avg_confidence)

        return TradingSignal(
            signal_type=signal_type,
            strength=strength,
            confidence=avg_confidence * majority_ratio,
            metadata={
                "majority_ratio": majority_ratio,
                "buy_count": buy_count,
                "sell_count": sell_count
            }
        )

    def _determine_strength(
        self,
        agreement: float,
        confidence: float
    ) -> SignalStrength:
        """Determine signal strength."""
        combined = (agreement + confidence) / 2

        if combined >= 0.8:
            return SignalStrength.VERY_STRONG
        elif combined >= 0.6:
            return SignalStrength.STRONG
        elif combined >= 0.4:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK


class RiskAdjustedSignalGenerator(BaseSignalGenerator):
    """Generates risk-adjusted trading signals."""

    def __init__(
        self,
        base_generator: BaseSignalGenerator,
        name: str = "RiskAdjustedSignal",
        config: Optional[SignalGeneratorConfig] = None,
        max_risk_per_trade: float = 0.02,
        max_portfolio_risk: float = 0.1
    ):
        """
        Initialize risk-adjusted signal generator.

        Args:
            base_generator: Base signal generator
            name: Generator name
            config: Configuration
            max_risk_per_trade: Maximum risk per trade
            max_portfolio_risk: Maximum portfolio risk
        """
        super().__init__(name, config)
        self.base_generator = base_generator
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk

    async def generate(
        self,
        prediction: np.ndarray,
        confidence: np.ndarray,
        current_price: float,
        portfolio_value: float = 100000.0,
        current_exposure: float = 0.0,
        volatility: Optional[float] = None,
        **kwargs: Any
    ) -> TradingSignal:
        """
        Generate risk-adjusted signal.

        Args:
            prediction: Model prediction
            confidence: Prediction confidence
            current_price: Current market price
            portfolio_value: Total portfolio value
            current_exposure: Current market exposure
            volatility: Current volatility
            **kwargs: Additional arguments

        Returns:
            TradingSignal object
        """
        try:
            base_signal = await self.base_generator.generate(
                prediction, confidence, current_price,
                volatility=volatility, **kwargs
            )

            if base_signal.signal_type == SignalType.HOLD:
                return base_signal

            remaining_risk = self.max_portfolio_risk - abs(current_exposure)
            if remaining_risk <= 0:
                logger.warning("Maximum portfolio risk reached, generating HOLD signal")
                return TradingSignal(
                    signal_type=SignalType.HOLD,
                    strength=SignalStrength.WEAK,
                    confidence=0.0,
                    metadata={"reason": "max_portfolio_risk_reached"}
                )

            if base_signal.stop_loss is not None:
                risk_per_share = abs(current_price - base_signal.stop_loss)
            elif volatility:
                risk_per_share = volatility * 2
            else:
                risk_per_share = current_price * 0.02

            max_position_risk = portfolio_value * self.max_risk_per_trade
            risk_adjusted_shares = max_position_risk / risk_per_share

            position_value = risk_adjusted_shares * current_price
            max_position_value = portfolio_value * remaining_risk
            position_value = min(position_value, max_position_value)

            adjusted_position_size = position_value / portfolio_value

            if base_signal.position_size:
                adjusted_position_size = min(
                    adjusted_position_size,
                    base_signal.position_size
                )

            adjusted_confidence = base_signal.confidence
            if volatility and volatility > 0.03:
                vol_penalty = (volatility - 0.02) * 5
                adjusted_confidence *= (1 - min(vol_penalty, 0.3))

            adjusted_signal = TradingSignal(
                signal_type=base_signal.signal_type,
                strength=base_signal.strength,
                confidence=adjusted_confidence,
                price_target=base_signal.price_target,
                stop_loss=base_signal.stop_loss,
                take_profit=base_signal.take_profit,
                position_size=adjusted_position_size,
                expiry=base_signal.expiry,
                metadata={
                    **base_signal.metadata,
                    "risk_adjusted": True,
                    "original_position_size": base_signal.position_size,
                    "remaining_risk_capacity": remaining_risk
                }
            )

            self._signal_history.append(adjusted_signal)

            return adjusted_signal

        except Exception as e:
            logger.error(f"Error generating risk-adjusted signal: {e}")
            return TradingSignal(
                signal_type=SignalType.HOLD,
                strength=SignalStrength.WEAK,
                confidence=0.0,
                metadata={"error": str(e)}
            )


class SignalFilter:
    """Filters trading signals based on various criteria."""

    def __init__(
        self,
        min_confidence: float = 0.5,
        min_strength: SignalStrength = SignalStrength.MODERATE,
        allowed_types: Optional[list[SignalType]] = None
    ):
        """
        Initialize signal filter.

        Args:
            min_confidence: Minimum confidence threshold
            min_strength: Minimum signal strength
            allowed_types: Allowed signal types
        """
        self.min_confidence = min_confidence
        self.min_strength = min_strength
        self.allowed_types = allowed_types

        self._strength_order = {
            SignalStrength.WEAK: 0,
            SignalStrength.MODERATE: 1,
            SignalStrength.STRONG: 2,
            SignalStrength.VERY_STRONG: 3
        }

        logger.info(
            f"Initialized SignalFilter: min_confidence={min_confidence}, "
            f"min_strength={min_strength.value}"
        )

    def filter(self, signal: TradingSignal) -> bool:
        """
        Check if signal passes filter criteria.

        Args:
            signal: Trading signal to filter

        Returns:
            True if signal passes filter
        """
        if signal.confidence < self.min_confidence:
            return False

        if self._strength_order[signal.strength] < self._strength_order[self.min_strength]:
            return False

        if self.allowed_types is not None:
            if signal.signal_type not in self.allowed_types:
                return False

        return True

    def filter_batch(
        self,
        signals: list[TradingSignal]
    ) -> list[TradingSignal]:
        """
        Filter a batch of signals.

        Args:
            signals: List of signals to filter

        Returns:
            Filtered list of signals
        """
        return [s for s in signals if self.filter(s)]


def create_direction_signal_generator(
    min_confidence: float = 0.5,
    strong_threshold: float = 0.8,
    name: str = "DirectionSignal"
) -> DirectionSignalGenerator:
    """
    Factory function to create direction signal generator.

    Args:
        min_confidence: Minimum confidence threshold
        strong_threshold: Strong signal threshold
        name: Generator name

    Returns:
        DirectionSignalGenerator instance
    """
    config = SignalGeneratorConfig(
        min_confidence=min_confidence,
        strong_signal_threshold=strong_threshold
    )
    return DirectionSignalGenerator(name=name, config=config)


def create_probability_signal_generator(
    buy_threshold: float = 0.6,
    sell_threshold: float = 0.4,
    name: str = "ProbabilitySignal"
) -> ProbabilitySignalGenerator:
    """
    Factory function to create probability signal generator.

    Args:
        buy_threshold: Buy probability threshold
        sell_threshold: Sell probability threshold
        name: Generator name

    Returns:
        ProbabilitySignalGenerator instance
    """
    return ProbabilitySignalGenerator(
        name=name,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold
    )


def create_multi_model_signal_generator(
    combination_method: str = "weighted_voting",
    name: str = "MultiModelSignal"
) -> MultiModelSignalGenerator:
    """
    Factory function to create multi-model signal generator.

    Args:
        combination_method: Signal combination method
        name: Generator name

    Returns:
        MultiModelSignalGenerator instance
    """
    return MultiModelSignalGenerator(
        name=name,
        combination_method=SignalCombinationMethod(combination_method)
    )
