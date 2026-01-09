"""
Correlation Risk Module for Ultimate Trading Bot v2.2.

This module analyzes and manages portfolio correlation risk
including concentration and diversification metrics.
"""

import logging
from datetime import datetime
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field

from src.risk.base_risk import (
    BaseRiskManager,
    RiskConfig,
    RiskLevel,
    RiskType,
    RiskMetric,
    RiskAlert,
    RiskAssessment,
    RiskContext,
)
from src.utils.helpers import generate_uuid
from src.utils.date_utils import now_utc


logger = logging.getLogger(__name__)


class CorrelationPair(BaseModel):
    """Model for correlation between two assets."""

    symbol1: str
    symbol2: str
    correlation: float = Field(ge=-1.0, le=1.0)
    rolling_correlation: float = Field(default=0.0, ge=-1.0, le=1.0)
    correlation_change: float = Field(default=0.0)
    is_significant: bool = Field(default=False)


class CorrelationCluster(BaseModel):
    """Model for a group of correlated assets."""

    cluster_id: str = Field(default_factory=generate_uuid)
    symbols: list[str] = Field(default_factory=list)
    avg_correlation: float = Field(default=0.0)
    total_exposure: float = Field(default=0.0)
    exposure_pct: float = Field(default=0.0)


class DiversificationMetrics(BaseModel):
    """Model for portfolio diversification metrics."""

    timestamp: datetime
    diversification_ratio: float = Field(default=1.0)
    effective_n: float = Field(default=1.0)
    herfindahl_index: float = Field(default=0.0)
    concentration_ratio: float = Field(default=0.0)
    avg_correlation: float = Field(default=0.0)
    max_correlation: float = Field(default=0.0)
    correlation_clusters: int = Field(default=0)


class CorrelationRiskConfig(RiskConfig):
    """Configuration for correlation risk manager."""

    correlation_threshold: float = Field(default=0.7, ge=0.3, le=0.95)
    high_correlation_threshold: float = Field(default=0.85, ge=0.5, le=0.99)

    max_cluster_exposure_pct: float = Field(default=0.40, ge=0.1, le=0.7)
    max_correlated_positions: int = Field(default=5, ge=2, le=20)

    lookback_days: int = Field(default=60, ge=20, le=252)
    rolling_window: int = Field(default=20, ge=5, le=60)
    min_data_points: int = Field(default=30, ge=10, le=100)

    correlation_decay_factor: float = Field(default=0.97, ge=0.9, le=0.99)

    alert_on_cluster_breach: bool = Field(default=True)
    alert_on_correlation_spike: bool = Field(default=True)


class CorrelationRiskManager(BaseRiskManager):
    """
    Correlation risk analysis and management.

    Features:
    - Correlation matrix calculation
    - Cluster identification
    - Diversification metrics
    - Correlation regime detection
    - Risk-based allocation suggestions
    """

    def __init__(
        self,
        config: Optional[CorrelationRiskConfig] = None,
    ) -> None:
        """
        Initialize CorrelationRiskManager.

        Args:
            config: Correlation risk configuration
        """
        config = config or CorrelationRiskConfig()
        super().__init__(config)

        self._corr_config = config

        self._correlation_matrix: dict[str, dict[str, float]] = {}
        self._rolling_correlations: dict[str, dict[str, list[float]]] = {}
        self._returns_data: dict[str, list[float]] = {}
        self._clusters: list[CorrelationCluster] = []
        self._diversification_history: list[DiversificationMetrics] = []

        logger.info("CorrelationRiskManager initialized")

    async def assess_risk(
        self,
        context: dict[str, Any],
    ) -> RiskAssessment:
        """
        Assess correlation risk.

        Args:
            context: Risk assessment context

        Returns:
            Risk assessment result
        """
        risk_context = self._build_risk_context(context)

        returns_data = context.get("returns_data", {})
        if returns_data:
            self._returns_data = returns_data
            self._update_correlation_matrix()

        self._identify_clusters(risk_context)
        metrics = self._calculate_diversification_metrics(risk_context)

        self._diversification_history.append(metrics)
        if len(self._diversification_history) > 1000:
            self._diversification_history = self._diversification_history[-1000:]

        risk_score = self._calculate_correlation_risk_score(metrics, risk_context)
        risk_level = self.calculate_risk_level(risk_score)

        risk_metrics = self._create_risk_metrics(metrics)
        for metric in risk_metrics:
            self.record_metric(metric)

        recommendations = self._generate_correlation_recommendations(
            metrics, risk_context
        )

        return RiskAssessment(
            timestamp=now_utc(),
            overall_risk_level=risk_level,
            overall_risk_score=risk_score,
            portfolio_risk_score=risk_score,
            metrics=risk_metrics,
            recommendations=recommendations,
            metadata={
                "diversification_ratio": metrics.diversification_ratio,
                "effective_n": metrics.effective_n,
                "avg_correlation": metrics.avg_correlation,
                "clusters": metrics.correlation_clusters,
            },
        )

    async def check_limits(
        self,
        context: dict[str, Any],
    ) -> list[RiskAlert]:
        """
        Check correlation limits.

        Args:
            context: Context for limit checking

        Returns:
            List of risk alerts
        """
        risk_context = self._build_risk_context(context)
        alerts: list[RiskAlert] = []

        for cluster in self._clusters:
            if cluster.exposure_pct > self._corr_config.max_cluster_exposure_pct:
                alerts.append(self.create_alert(
                    risk_type=RiskType.CORRELATION,
                    risk_level=RiskLevel.HIGH,
                    title=f"Correlation Cluster Exposure High",
                    message=f"Cluster with {len(cluster.symbols)} positions has {cluster.exposure_pct:.1%} exposure",
                    current_value=cluster.exposure_pct,
                    threshold_value=self._corr_config.max_cluster_exposure_pct,
                    recommended_action="Reduce exposure to correlated positions",
                    metadata={"symbols": cluster.symbols},
                ))

        high_corr_pairs = self._find_high_correlations()
        for pair in high_corr_pairs[:3]:
            if pair.correlation >= self._corr_config.high_correlation_threshold:
                alerts.append(self.create_alert(
                    risk_type=RiskType.CORRELATION,
                    risk_level=RiskLevel.MEDIUM,
                    title=f"High Correlation: {pair.symbol1}/{pair.symbol2}",
                    message=f"Correlation of {pair.correlation:.2f} exceeds threshold",
                    current_value=pair.correlation,
                    threshold_value=self._corr_config.high_correlation_threshold,
                    recommended_action=f"Consider reducing one of {pair.symbol1} or {pair.symbol2}",
                ))

        metrics = self._calculate_diversification_metrics(risk_context)
        if metrics.effective_n < 3 and risk_context.position_count >= 5:
            alerts.append(self.create_alert(
                risk_type=RiskType.CONCENTRATION,
                risk_level=RiskLevel.MEDIUM,
                title="Low Effective Diversification",
                message=f"Effective N of {metrics.effective_n:.1f} suggests concentrated risk",
                current_value=metrics.effective_n,
                threshold_value=3.0,
                recommended_action="Add uncorrelated positions to improve diversification",
            ))

        return alerts

    def _build_risk_context(self, context: dict[str, Any]) -> RiskContext:
        """Build RiskContext from generic context."""
        return RiskContext(
            timestamp=context.get("timestamp", now_utc()),
            account_value=context.get("account_value", 0),
            cash_balance=context.get("cash_balance", 0),
            positions=context.get("positions", []),
            open_orders=context.get("open_orders", []),
            daily_pnl=context.get("daily_pnl", 0),
            unrealized_pnl=context.get("unrealized_pnl", 0),
        )

    def _update_correlation_matrix(self) -> None:
        """Update correlation matrix from returns data."""
        symbols = list(self._returns_data.keys())
        n = len(symbols)

        if n < 2:
            return

        min_len = min(len(self._returns_data[s]) for s in symbols)
        if min_len < self._corr_config.min_data_points:
            return

        for i, sym1 in enumerate(symbols):
            if sym1 not in self._correlation_matrix:
                self._correlation_matrix[sym1] = {}

            for sym2 in symbols[i:]:
                if sym1 == sym2:
                    self._correlation_matrix[sym1][sym2] = 1.0
                    continue

                returns1 = self._returns_data[sym1][-min_len:]
                returns2 = self._returns_data[sym2][-min_len:]

                correlation = self._calculate_correlation(returns1, returns2)
                self._correlation_matrix[sym1][sym2] = correlation

                if sym2 not in self._correlation_matrix:
                    self._correlation_matrix[sym2] = {}
                self._correlation_matrix[sym2][sym1] = correlation

    def _calculate_correlation(
        self,
        returns1: list[float],
        returns2: list[float],
    ) -> float:
        """Calculate correlation between two return series."""
        n = min(len(returns1), len(returns2))
        if n < 2:
            return 0.0

        r1 = np.array(returns1[-n:])
        r2 = np.array(returns2[-n:])

        mean1, mean2 = np.mean(r1), np.mean(r2)
        std1, std2 = np.std(r1), np.std(r2)

        if std1 == 0 or std2 == 0:
            return 0.0

        covariance = np.mean((r1 - mean1) * (r2 - mean2))
        correlation = covariance / (std1 * std2)

        return max(-1.0, min(1.0, correlation))

    def _identify_clusters(self, context: RiskContext) -> None:
        """Identify clusters of correlated positions."""
        positions = context.positions
        if not positions:
            self._clusters = []
            return

        symbols = [p.get("symbol", "") for p in positions]
        threshold = self._corr_config.correlation_threshold

        visited = set()
        clusters: list[CorrelationCluster] = []

        for symbol in symbols:
            if symbol in visited:
                continue

            cluster_symbols = self._find_cluster_members(
                symbol, symbols, threshold, visited
            )

            if len(cluster_symbols) > 1:
                cluster_exposure = sum(
                    abs(p.get("market_value", 0))
                    for p in positions
                    if p.get("symbol") in cluster_symbols
                )

                avg_corr = self._calculate_cluster_avg_correlation(cluster_symbols)

                exposure_pct = cluster_exposure / context.account_value if context.account_value > 0 else 0

                clusters.append(CorrelationCluster(
                    symbols=list(cluster_symbols),
                    avg_correlation=avg_corr,
                    total_exposure=cluster_exposure,
                    exposure_pct=exposure_pct,
                ))

            visited.update(cluster_symbols)

        self._clusters = clusters

    def _find_cluster_members(
        self,
        start_symbol: str,
        all_symbols: list[str],
        threshold: float,
        visited: set,
    ) -> set[str]:
        """Find all symbols in the same correlation cluster."""
        cluster = {start_symbol}
        queue = [start_symbol]

        while queue:
            current = queue.pop(0)

            for symbol in all_symbols:
                if symbol in cluster or symbol in visited:
                    continue

                correlation = self.get_correlation(current, symbol)

                if abs(correlation) >= threshold:
                    cluster.add(symbol)
                    queue.append(symbol)

        return cluster

    def _calculate_cluster_avg_correlation(
        self,
        symbols: list[str],
    ) -> float:
        """Calculate average correlation within a cluster."""
        if len(symbols) < 2:
            return 1.0

        correlations = []
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i + 1:]:
                corr = self.get_correlation(sym1, sym2)
                correlations.append(abs(corr))

        return np.mean(correlations) if correlations else 0.0

    def _calculate_diversification_metrics(
        self,
        context: RiskContext,
    ) -> DiversificationMetrics:
        """Calculate portfolio diversification metrics."""
        positions = context.positions
        account_value = context.account_value

        if not positions or account_value <= 0:
            return DiversificationMetrics(timestamp=now_utc())

        weights = [
            abs(p.get("market_value", 0)) / account_value
            for p in positions
        ]

        herfindahl = sum(w ** 2 for w in weights)

        effective_n = 1 / herfindahl if herfindahl > 0 else len(positions)

        sorted_weights = sorted(weights, reverse=True)
        concentration_ratio = sum(sorted_weights[:5]) / sum(weights) if weights else 0

        correlations = []
        symbols = [p.get("symbol", "") for p in positions]
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i + 1:]:
                corr = self.get_correlation(sym1, sym2)
                correlations.append(corr)

        avg_correlation = np.mean(correlations) if correlations else 0
        max_correlation = max(correlations) if correlations else 0

        sum_weights = sum(weights)
        sum_weighted_vol_squared = sum(w ** 2 for w in weights)
        cross_term = 0

        for i, w1 in enumerate(weights):
            for j, w2 in enumerate(weights):
                if i != j:
                    corr = self.get_correlation(symbols[i], symbols[j])
                    cross_term += w1 * w2 * corr

        portfolio_vol_squared = sum_weighted_vol_squared + cross_term
        weighted_avg_vol_squared = sum_weights ** 2

        if portfolio_vol_squared > 0:
            diversification_ratio = np.sqrt(weighted_avg_vol_squared / portfolio_vol_squared)
        else:
            diversification_ratio = 1.0

        return DiversificationMetrics(
            timestamp=now_utc(),
            diversification_ratio=diversification_ratio,
            effective_n=effective_n,
            herfindahl_index=herfindahl,
            concentration_ratio=concentration_ratio,
            avg_correlation=avg_correlation,
            max_correlation=max_correlation,
            correlation_clusters=len(self._clusters),
        )

    def _calculate_correlation_risk_score(
        self,
        metrics: DiversificationMetrics,
        context: RiskContext,
    ) -> float:
        """Calculate correlation risk score."""
        score = 0.0

        if metrics.avg_correlation > 0.5:
            score += 30
        elif metrics.avg_correlation > 0.3:
            score += 15

        if metrics.effective_n < 3:
            score += 25
        elif metrics.effective_n < 5:
            score += 10

        if metrics.concentration_ratio > 0.7:
            score += 20
        elif metrics.concentration_ratio > 0.5:
            score += 10

        for cluster in self._clusters:
            if cluster.exposure_pct > self._corr_config.max_cluster_exposure_pct:
                score += 15
            elif cluster.exposure_pct > self._corr_config.max_cluster_exposure_pct * 0.8:
                score += 8

        return min(100, score)

    def _create_risk_metrics(
        self,
        metrics: DiversificationMetrics,
    ) -> list[RiskMetric]:
        """Create risk metrics from diversification metrics."""
        return [
            RiskMetric(
                name="diversification_ratio",
                value=metrics.diversification_ratio,
                unit="ratio",
                timestamp=metrics.timestamp,
                risk_level=RiskLevel.LOW if metrics.diversification_ratio > 1.5 else RiskLevel.MEDIUM,
            ),
            RiskMetric(
                name="effective_n",
                value=metrics.effective_n,
                unit="positions",
                timestamp=metrics.timestamp,
                risk_level=RiskLevel.LOW if metrics.effective_n > 5 else RiskLevel.MEDIUM,
            ),
            RiskMetric(
                name="avg_correlation",
                value=metrics.avg_correlation,
                unit="correlation",
                timestamp=metrics.timestamp,
                risk_level=RiskLevel.HIGH if metrics.avg_correlation > 0.5 else RiskLevel.LOW,
            ),
            RiskMetric(
                name="concentration_ratio",
                value=metrics.concentration_ratio,
                unit="percent",
                timestamp=metrics.timestamp,
                risk_level=RiskLevel.HIGH if metrics.concentration_ratio > 0.7 else RiskLevel.MEDIUM,
            ),
        ]

    def _generate_correlation_recommendations(
        self,
        metrics: DiversificationMetrics,
        context: RiskContext,
    ) -> list[str]:
        """Generate correlation-based recommendations."""
        recommendations: list[str] = []

        if metrics.avg_correlation > 0.5:
            recommendations.append(
                f"Portfolio average correlation of {metrics.avg_correlation:.2f} is high - "
                "consider adding uncorrelated assets"
            )

        if metrics.effective_n < context.position_count * 0.5:
            recommendations.append(
                f"Effective diversification ({metrics.effective_n:.1f}) is much lower "
                f"than position count ({context.position_count}) - rebalance weights"
            )

        for cluster in self._clusters:
            if cluster.exposure_pct > self._corr_config.max_cluster_exposure_pct * 0.8:
                recommendations.append(
                    f"High exposure to correlated group: {', '.join(cluster.symbols[:3])}"
                )

        return recommendations

    def _find_high_correlations(self) -> list[CorrelationPair]:
        """Find pairs with high correlation."""
        pairs: list[CorrelationPair] = []

        symbols = list(self._correlation_matrix.keys())
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i + 1:]:
                corr = self.get_correlation(sym1, sym2)

                if abs(corr) >= self._corr_config.correlation_threshold:
                    pairs.append(CorrelationPair(
                        symbol1=sym1,
                        symbol2=sym2,
                        correlation=corr,
                        is_significant=abs(corr) >= self._corr_config.high_correlation_threshold,
                    ))

        pairs.sort(key=lambda p: abs(p.correlation), reverse=True)

        return pairs

    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols."""
        if symbol1 == symbol2:
            return 1.0

        if symbol1 in self._correlation_matrix:
            if symbol2 in self._correlation_matrix[symbol1]:
                return self._correlation_matrix[symbol1][symbol2]

        if symbol2 in self._correlation_matrix:
            if symbol1 in self._correlation_matrix[symbol2]:
                return self._correlation_matrix[symbol2][symbol1]

        return 0.0

    def get_correlation_matrix(self) -> dict[str, dict[str, float]]:
        """Get full correlation matrix."""
        return self._correlation_matrix.copy()

    def get_clusters(self) -> list[CorrelationCluster]:
        """Get identified correlation clusters."""
        return self._clusters.copy()

    def get_diversification_history(
        self,
        limit: int = 100,
    ) -> list[DiversificationMetrics]:
        """Get diversification metrics history."""
        return self._diversification_history[-limit:]

    def update_returns(
        self,
        symbol: str,
        returns: list[float],
    ) -> None:
        """Update returns data for a symbol."""
        self._returns_data[symbol] = returns

    def get_correlation_summary(self) -> dict:
        """Get correlation risk summary."""
        metrics = self._diversification_history[-1] if self._diversification_history else None

        return {
            "symbols_tracked": len(self._correlation_matrix),
            "clusters": len(self._clusters),
            "avg_correlation": metrics.avg_correlation if metrics else 0,
            "effective_n": metrics.effective_n if metrics else 0,
            "diversification_ratio": metrics.diversification_ratio if metrics else 1,
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"CorrelationRiskManager(symbols={len(self._correlation_matrix)}, clusters={len(self._clusters)})"
