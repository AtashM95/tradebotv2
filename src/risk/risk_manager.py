
import logging
from typing import Dict, List, Optional, Any
from config.risk_config import RiskConfig
from ..core.contracts import SignalIntent, RiskDecision, MarketSnapshot, RunContext

logger = logging.getLogger(__name__)


class RiskManagerMetrics:
    """Tracks risk manager metrics."""

    def __init__(self) -> None:
        self.total_evaluations: int = 0
        self.approved: int = 0
        self.vetoed: int = 0
        self.veto_reasons: Dict[str, int] = {}

    def record_decision(self, approved: bool, reason: str) -> None:
        """Record a risk decision."""
        self.total_evaluations += 1
        if approved:
            self.approved += 1
        else:
            self.vetoed += 1
            self.veto_reasons[reason] = self.veto_reasons.get(reason, 0) + 1

    def get_approval_rate(self) -> float:
        """Get approval rate."""
        if self.total_evaluations == 0:
            return 0.0
        return self.approved / self.total_evaluations

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_evaluations': self.total_evaluations,
            'approved': self.approved,
            'vetoed': self.vetoed,
            'approval_rate': self.get_approval_rate(),
            'veto_reasons': self.veto_reasons
        }


class RiskManager:
    """
    Comprehensive risk manager for trading operations.

    Responsibilities:
    - Evaluate trading signals for risk compliance
    - Enforce position size limits
    - Monitor daily loss limits
    - Track portfolio exposure
    - Prevent over-leveraging
    - Validate trade parameters
    - Provide risk metrics and reporting
    """

    def __init__(self, config: RiskConfig) -> None:
        """
        Initialize risk manager.

        Args:
            config: Risk configuration parameters
        """
        self.config = config
        self.metrics = RiskManagerMetrics()

        # Track daily metrics
        self.daily_trades: int = 0
        self.daily_pnl: float = 0.0
        self.daily_loss_triggered: bool = False

        # Track positions
        self.current_positions: Dict[str, float] = {}
        self.total_exposure: float = 0.0

        logger.info("RiskManager initialized", extra={
            'max_daily_loss': config.max_daily_loss,
            'max_position_size': config.max_position_size
        })

    def initialize(self, context: RunContext) -> None:
        """
        Initialize risk manager for trading session.

        Args:
            context: Run context
        """
        logger.info("Initializing RiskManager", extra={'run_id': context.run_id})

        # Reset daily metrics
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.daily_loss_triggered = False

        logger.info("RiskManager initialized successfully", extra={'run_id': context.run_id})

    def shutdown(self, context: RunContext) -> None:
        """
        Shutdown risk manager.

        Args:
            context: Run context
        """
        logger.info("Shutting down RiskManager", extra={
            'run_id': context.run_id,
            'metrics': self.metrics.to_dict()
        })

    def evaluate(
        self,
        signal: SignalIntent,
        snapshot: MarketSnapshot,
        context: RunContext
    ) -> RiskDecision:
        """
        Evaluate a trading signal for risk compliance.

        Args:
            signal: Trading signal to evaluate
            snapshot: Market snapshot
            context: Run context

        Returns:
            RiskDecision with approval and adjusted size
        """
        # Check 1: Daily loss limit
        if self.daily_loss_triggered:
            decision = RiskDecision(
                approved=False,
                reason="daily_loss_limit_exceeded",
                adjusted_size=0.0
            )
            self.metrics.record_decision(False, decision.reason)
            return decision

        # Check 2: Signal confidence threshold
        if abs(signal.confidence) < 0.1:
            decision = RiskDecision(
                approved=False,
                reason="low_confidence",
                adjusted_size=0.0
            )
            self.metrics.record_decision(False, decision.reason)
            return decision

        # Check 3: Check if in dry-run mode
        if context.dry_run:
            # In dry-run, approve but with reduced size
            adjusted_size = min(1.0, self.config.max_position_size * 0.1)
            decision = RiskDecision(
                approved=True,
                reason="approved_dry_run",
                adjusted_size=adjusted_size
            )
            self.metrics.record_decision(True, decision.reason)
            return decision

        # Check 4: Position size limit
        position_value = snapshot.price * 1.0  # Assuming 1 share for now
        max_position_value = 100000.0 * self.config.max_position_size  # Assuming $100k account

        if position_value > max_position_value:
            decision = RiskDecision(
                approved=False,
                reason="position_size_exceeded",
                adjusted_size=0.0
            )
            self.metrics.record_decision(False, decision.reason)
            return decision

        # Check 5: Validate symbol is in allowed watchlist
        # (In a real system, you'd check against allowed symbols)

        # Calculate adjusted size based on confidence
        base_size = 1.0
        confidence_multiplier = abs(signal.confidence)
        adjusted_size = base_size * confidence_multiplier * self.config.max_position_size

        # Ensure adjusted size is within bounds
        adjusted_size = max(0.1, min(adjusted_size, 1.0))

        decision = RiskDecision(
            approved=True,
            reason="approved",
            adjusted_size=adjusted_size,
            metadata={
                'confidence': signal.confidence,
                'position_value': position_value,
                'max_position_value': max_position_value
            }
        )

        self.metrics.record_decision(True, decision.reason)

        log_extra = {
            'run_id': context.run_id,
            'symbol': signal.symbol,
            'approved': True,
            'adjusted_size': adjusted_size,
            'confidence': signal.confidence
        }
        logger.debug("Risk evaluation approved", extra=log_extra)

        return decision

    def update_daily_pnl(self, pnl: float) -> None:
        """
        Update daily P&L and check loss limits.

        Args:
            pnl: P&L amount
        """
        self.daily_pnl += pnl

        # Check daily loss limit
        if self.daily_pnl < -self.config.max_daily_loss * 100000.0:  # Assuming $100k account
            self.daily_loss_triggered = True
            logger.warning("Daily loss limit triggered", extra={
                'daily_pnl': self.daily_pnl,
                'limit': -self.config.max_daily_loss * 100000.0
            })

    def update_position(self, symbol: str, value: float) -> None:
        """
        Update position tracking.

        Args:
            symbol: Symbol
            value: Position value
        """
        old_value = self.current_positions.get(symbol, 0.0)
        self.current_positions[symbol] = value

        # Update total exposure
        self.total_exposure = self.total_exposure - old_value + value

    def remove_position(self, symbol: str) -> None:
        """
        Remove a position from tracking.

        Args:
            symbol: Symbol to remove
        """
        if symbol in self.current_positions:
            self.total_exposure -= self.current_positions[symbol]
            del self.current_positions[symbol]

    def get_current_exposure(self) -> float:
        """Get current portfolio exposure."""
        return self.total_exposure

    def get_position_exposure(self, symbol: str) -> float:
        """Get exposure for a specific position."""
        return self.current_positions.get(symbol, 0.0)

    def is_daily_loss_triggered(self) -> bool:
        """Check if daily loss limit is triggered."""
        return self.daily_loss_triggered

    def reset_daily_metrics(self) -> None:
        """Reset daily metrics (call at start of trading day)."""
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.daily_loss_triggered = False
        logger.info("Daily risk metrics reset")

    def can_trade(self) -> bool:
        """
        Check if trading is allowed.

        Returns:
            True if trading is allowed, False otherwise
        """
        return not self.daily_loss_triggered

    def get_max_position_size(self, symbol: str, price: float, account_value: float) -> float:
        """
        Calculate maximum allowed position size.

        Args:
            symbol: Symbol
            price: Current price
            account_value: Account value

        Returns:
            Maximum quantity allowed
        """
        max_value = account_value * self.config.max_position_size
        if price <= 0:
            return 0.0
        return max_value / price

    def validate_order_size(
        self,
        symbol: str,
        quantity: float,
        price: float,
        account_value: float
    ) -> bool:
        """
        Validate if an order size is within risk limits.

        Args:
            symbol: Symbol
            quantity: Quantity
            price: Price
            account_value: Account value

        Returns:
            True if valid, False otherwise
        """
        order_value = quantity * price
        max_value = account_value * self.config.max_position_size

        if order_value > max_value:
            logger.warning("Order size exceeds limit", extra={
                'symbol': symbol,
                'order_value': order_value,
                'max_value': max_value
            })
            return False

        return True

    def get_risk_score(self, signal: SignalIntent, snapshot: MarketSnapshot) -> float:
        """
        Calculate a risk score for a signal (0-100, higher = riskier).

        Args:
            signal: Trading signal
            snapshot: Market snapshot

        Returns:
            Risk score
        """
        risk_score = 0.0

        # Factor 1: Low confidence = higher risk
        if signal.confidence < 0.3:
            risk_score += 30.0
        elif signal.confidence < 0.5:
            risk_score += 15.0

        # Factor 2: Check daily P&L
        if self.daily_pnl < 0:
            risk_score += 20.0

        # Factor 3: Portfolio exposure
        if self.total_exposure > 80000.0:  # Assuming $100k account
            risk_score += 25.0

        # Factor 4: Existing position
        if signal.symbol in self.current_positions:
            risk_score += 15.0

        return min(risk_score, 100.0)

    def get_metrics(self) -> Dict[str, Any]:
        """Get risk manager metrics."""
        metrics = self.metrics.to_dict()
        metrics['daily_trades'] = self.daily_trades
        metrics['daily_pnl'] = self.daily_pnl
        metrics['daily_loss_triggered'] = self.daily_loss_triggered
        metrics['current_positions'] = len(self.current_positions)
        metrics['total_exposure'] = self.total_exposure
        return metrics

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on risk manager.

        Returns:
            Health status dictionary
        """
        return {
            'healthy': True,
            'can_trade': self.can_trade(),
            'daily_loss_triggered': self.daily_loss_triggered,
            'daily_pnl': self.daily_pnl,
            'current_positions': len(self.current_positions),
            'total_exposure': self.total_exposure,
            'metrics': self.metrics.to_dict()
        }
