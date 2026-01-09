"""
Exposure Manager Module for Ultimate Trading Bot v2.2.

This module manages and monitors portfolio exposure across
various dimensions including sectors, asset classes, and risk factors.
"""

import logging
from datetime import datetime
from typing import Any, Optional

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


class ExposureBreakdown(BaseModel):
    """Model for exposure breakdown by category."""

    category: str
    name: str
    gross_exposure: float = Field(default=0.0)
    net_exposure: float = Field(default=0.0)
    long_exposure: float = Field(default=0.0)
    short_exposure: float = Field(default=0.0)
    position_count: int = Field(default=0)
    pct_of_portfolio: float = Field(default=0.0)
    limit: Optional[float] = None
    is_over_limit: bool = Field(default=False)


class ExposureSnapshot(BaseModel):
    """Model for portfolio exposure snapshot."""

    snapshot_id: str = Field(default_factory=generate_uuid)
    timestamp: datetime

    total_gross_exposure: float = Field(default=0.0)
    total_net_exposure: float = Field(default=0.0)
    total_long_exposure: float = Field(default=0.0)
    total_short_exposure: float = Field(default=0.0)

    leverage_ratio: float = Field(default=0.0)
    beta_adjusted_exposure: float = Field(default=0.0)

    sector_exposures: list[ExposureBreakdown] = Field(default_factory=list)
    asset_class_exposures: list[ExposureBreakdown] = Field(default_factory=list)
    geography_exposures: list[ExposureBreakdown] = Field(default_factory=list)
    factor_exposures: list[ExposureBreakdown] = Field(default_factory=list)

    largest_positions: list[dict] = Field(default_factory=list)
    concentration_score: float = Field(default=0.0)


class ExposureManagerConfig(RiskConfig):
    """Configuration for exposure manager."""

    max_gross_exposure: float = Field(default=2.0, ge=0.5, le=5.0)
    max_net_exposure: float = Field(default=1.5, ge=0.0, le=3.0)
    max_long_exposure: float = Field(default=1.5, ge=0.5, le=3.0)
    max_short_exposure: float = Field(default=1.0, ge=0.0, le=2.0)

    max_sector_exposure: float = Field(default=0.30, ge=0.1, le=0.6)
    max_single_position: float = Field(default=0.10, ge=0.01, le=0.3)
    max_top_5_concentration: float = Field(default=0.50, ge=0.2, le=0.8)

    target_gross_exposure: float = Field(default=1.0, ge=0.2, le=2.0)
    target_net_exposure: float = Field(default=0.8, ge=-1.0, le=2.0)

    use_beta_adjustment: bool = Field(default=True)
    portfolio_beta_target: float = Field(default=1.0, ge=0.0, le=2.0)
    max_portfolio_beta: float = Field(default=1.5, ge=0.5, le=3.0)

    rebalance_threshold: float = Field(default=0.05, ge=0.01, le=0.2)
    auto_rebalance: bool = Field(default=False)


class ExposureManager(BaseRiskManager):
    """
    Portfolio exposure monitoring and management.

    Features:
    - Multi-dimensional exposure tracking
    - Sector and factor exposure limits
    - Concentration monitoring
    - Beta-adjusted exposure
    - Automated rebalancing suggestions
    """

    def __init__(
        self,
        config: Optional[ExposureManagerConfig] = None,
    ) -> None:
        """
        Initialize ExposureManager.

        Args:
            config: Exposure manager configuration
        """
        config = config or ExposureManagerConfig()
        super().__init__(config)

        self._exp_config = config

        self._exposure_history: list[ExposureSnapshot] = []
        self._sector_mappings: dict[str, str] = {}
        self._asset_class_mappings: dict[str, str] = {}
        self._geography_mappings: dict[str, str] = {}
        self._beta_values: dict[str, float] = {}

        logger.info("ExposureManager initialized")

    async def assess_risk(
        self,
        context: dict[str, Any],
    ) -> RiskAssessment:
        """
        Assess exposure risk.

        Args:
            context: Risk assessment context

        Returns:
            Risk assessment result
        """
        risk_context = self._build_risk_context(context)

        snapshot = self._calculate_exposure_snapshot(risk_context)
        self._exposure_history.append(snapshot)

        if len(self._exposure_history) > 1000:
            self._exposure_history = self._exposure_history[-1000:]

        risk_score = self._calculate_exposure_risk_score(snapshot)
        risk_level = self.calculate_risk_level(risk_score)

        metrics = self._create_exposure_metrics(snapshot)
        for metric in metrics:
            self.record_metric(metric)

        recommendations = self._generate_exposure_recommendations(snapshot)

        return RiskAssessment(
            timestamp=now_utc(),
            overall_risk_level=risk_level,
            overall_risk_score=risk_score,
            portfolio_risk_score=risk_score,
            metrics=metrics,
            recommendations=recommendations,
            metadata={
                "gross_exposure": snapshot.total_gross_exposure,
                "net_exposure": snapshot.total_net_exposure,
                "leverage": snapshot.leverage_ratio,
                "concentration": snapshot.concentration_score,
            },
        )

    async def check_limits(
        self,
        context: dict[str, Any],
    ) -> list[RiskAlert]:
        """
        Check exposure limits.

        Args:
            context: Context for limit checking

        Returns:
            List of risk alerts
        """
        risk_context = self._build_risk_context(context)
        snapshot = self._calculate_exposure_snapshot(risk_context)
        alerts: list[RiskAlert] = []

        if snapshot.total_gross_exposure > self._exp_config.max_gross_exposure:
            alerts.append(self.create_alert(
                risk_type=RiskType.LEVERAGE,
                risk_level=RiskLevel.HIGH,
                title="Gross Exposure Limit Exceeded",
                message=f"Gross exposure {snapshot.total_gross_exposure:.2f}x exceeds limit",
                current_value=snapshot.total_gross_exposure,
                threshold_value=self._exp_config.max_gross_exposure,
                recommended_action="Reduce positions to bring gross exposure within limits",
            ))

        if abs(snapshot.total_net_exposure) > self._exp_config.max_net_exposure:
            alerts.append(self.create_alert(
                risk_type=RiskType.LEVERAGE,
                risk_level=RiskLevel.HIGH,
                title="Net Exposure Limit Exceeded",
                message=f"Net exposure {snapshot.total_net_exposure:.2f}x exceeds limit",
                current_value=snapshot.total_net_exposure,
                threshold_value=self._exp_config.max_net_exposure,
                recommended_action="Adjust long/short balance",
            ))

        if snapshot.total_long_exposure > self._exp_config.max_long_exposure:
            alerts.append(self.create_alert(
                risk_type=RiskType.LEVERAGE,
                risk_level=RiskLevel.MEDIUM,
                title="Long Exposure High",
                message=f"Long exposure {snapshot.total_long_exposure:.2f}x exceeds limit",
                current_value=snapshot.total_long_exposure,
                threshold_value=self._exp_config.max_long_exposure,
                recommended_action="Reduce long positions or add hedges",
            ))

        if snapshot.total_short_exposure > self._exp_config.max_short_exposure:
            alerts.append(self.create_alert(
                risk_type=RiskType.LEVERAGE,
                risk_level=RiskLevel.MEDIUM,
                title="Short Exposure High",
                message=f"Short exposure {snapshot.total_short_exposure:.2f}x exceeds limit",
                current_value=snapshot.total_short_exposure,
                threshold_value=self._exp_config.max_short_exposure,
                recommended_action="Cover short positions",
            ))

        for sector_exp in snapshot.sector_exposures:
            if sector_exp.pct_of_portfolio > self._exp_config.max_sector_exposure:
                alerts.append(self.create_alert(
                    risk_type=RiskType.CONCENTRATION,
                    risk_level=RiskLevel.MEDIUM,
                    title=f"Sector Concentration: {sector_exp.name}",
                    message=f"{sector_exp.name} exposure {sector_exp.pct_of_portfolio:.1%} exceeds limit",
                    current_value=sector_exp.pct_of_portfolio,
                    threshold_value=self._exp_config.max_sector_exposure,
                    recommended_action=f"Reduce {sector_exp.name} exposure",
                ))

        if snapshot.concentration_score > self._exp_config.max_top_5_concentration:
            alerts.append(self.create_alert(
                risk_type=RiskType.CONCENTRATION,
                risk_level=RiskLevel.MEDIUM,
                title="Portfolio Concentration High",
                message=f"Top 5 positions represent {snapshot.concentration_score:.1%}",
                current_value=snapshot.concentration_score,
                threshold_value=self._exp_config.max_top_5_concentration,
                recommended_action="Diversify holdings",
            ))

        if self._exp_config.use_beta_adjustment:
            if snapshot.beta_adjusted_exposure > self._exp_config.max_portfolio_beta:
                alerts.append(self.create_alert(
                    risk_type=RiskType.MARKET,
                    risk_level=RiskLevel.MEDIUM,
                    title="Portfolio Beta High",
                    message=f"Beta-adjusted exposure {snapshot.beta_adjusted_exposure:.2f} exceeds limit",
                    current_value=snapshot.beta_adjusted_exposure,
                    threshold_value=self._exp_config.max_portfolio_beta,
                    recommended_action="Add low-beta or negative-beta positions",
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
            market_data=context.get("market_data", {}),
        )

    def _calculate_exposure_snapshot(
        self,
        context: RiskContext,
    ) -> ExposureSnapshot:
        """Calculate current exposure snapshot."""
        account_value = context.account_value
        positions = context.positions

        if account_value <= 0:
            return ExposureSnapshot(timestamp=now_utc())

        total_long = 0.0
        total_short = 0.0
        beta_weighted_exposure = 0.0

        sector_totals: dict[str, dict] = {}
        asset_class_totals: dict[str, dict] = {}
        geography_totals: dict[str, dict] = {}

        position_values: list[tuple[str, float]] = []

        for pos in positions:
            symbol = pos.get("symbol", "")
            market_value = pos.get("market_value", 0)
            quantity = pos.get("quantity", 0)

            position_values.append((symbol, abs(market_value)))

            if quantity > 0 or market_value > 0:
                total_long += abs(market_value)
            else:
                total_short += abs(market_value)

            beta = self._beta_values.get(symbol, 1.0)
            beta_weighted_exposure += market_value * beta / account_value

            sector = self._sector_mappings.get(symbol, "Other")
            if sector not in sector_totals:
                sector_totals[sector] = {"long": 0, "short": 0, "count": 0}

            if market_value > 0:
                sector_totals[sector]["long"] += market_value
            else:
                sector_totals[sector]["short"] += abs(market_value)
            sector_totals[sector]["count"] += 1

            asset_class = self._asset_class_mappings.get(symbol, "Equity")
            if asset_class not in asset_class_totals:
                asset_class_totals[asset_class] = {"long": 0, "short": 0, "count": 0}

            if market_value > 0:
                asset_class_totals[asset_class]["long"] += market_value
            else:
                asset_class_totals[asset_class]["short"] += abs(market_value)
            asset_class_totals[asset_class]["count"] += 1

            geography = self._geography_mappings.get(symbol, "US")
            if geography not in geography_totals:
                geography_totals[geography] = {"long": 0, "short": 0, "count": 0}

            if market_value > 0:
                geography_totals[geography]["long"] += market_value
            else:
                geography_totals[geography]["short"] += abs(market_value)
            geography_totals[geography]["count"] += 1

        total_gross = total_long + total_short
        total_net = total_long - total_short

        sector_exposures = [
            ExposureBreakdown(
                category="sector",
                name=sector,
                gross_exposure=(data["long"] + data["short"]) / account_value,
                net_exposure=(data["long"] - data["short"]) / account_value,
                long_exposure=data["long"] / account_value,
                short_exposure=data["short"] / account_value,
                position_count=data["count"],
                pct_of_portfolio=(data["long"] + data["short"]) / account_value,
                limit=self._exp_config.max_sector_exposure,
                is_over_limit=(data["long"] + data["short"]) / account_value > self._exp_config.max_sector_exposure,
            )
            for sector, data in sector_totals.items()
        ]

        asset_class_exposures = [
            ExposureBreakdown(
                category="asset_class",
                name=ac,
                gross_exposure=(data["long"] + data["short"]) / account_value,
                net_exposure=(data["long"] - data["short"]) / account_value,
                long_exposure=data["long"] / account_value,
                short_exposure=data["short"] / account_value,
                position_count=data["count"],
                pct_of_portfolio=(data["long"] + data["short"]) / account_value,
            )
            for ac, data in asset_class_totals.items()
        ]

        geography_exposures = [
            ExposureBreakdown(
                category="geography",
                name=geo,
                gross_exposure=(data["long"] + data["short"]) / account_value,
                net_exposure=(data["long"] - data["short"]) / account_value,
                long_exposure=data["long"] / account_value,
                short_exposure=data["short"] / account_value,
                position_count=data["count"],
                pct_of_portfolio=(data["long"] + data["short"]) / account_value,
            )
            for geo, data in geography_totals.items()
        ]

        position_values.sort(key=lambda x: x[1], reverse=True)
        top_5_value = sum(v for _, v in position_values[:5])
        concentration_score = top_5_value / account_value if account_value > 0 else 0

        largest_positions = [
            {"symbol": symbol, "value": value, "pct": value / account_value}
            for symbol, value in position_values[:10]
        ]

        return ExposureSnapshot(
            timestamp=now_utc(),
            total_gross_exposure=total_gross / account_value,
            total_net_exposure=total_net / account_value,
            total_long_exposure=total_long / account_value,
            total_short_exposure=total_short / account_value,
            leverage_ratio=total_gross / account_value,
            beta_adjusted_exposure=beta_weighted_exposure,
            sector_exposures=sector_exposures,
            asset_class_exposures=asset_class_exposures,
            geography_exposures=geography_exposures,
            largest_positions=largest_positions,
            concentration_score=concentration_score,
        )

    def _calculate_exposure_risk_score(self, snapshot: ExposureSnapshot) -> float:
        """Calculate risk score from exposure snapshot."""
        score = 0.0

        gross_ratio = snapshot.total_gross_exposure / self._exp_config.max_gross_exposure
        score += min(30, gross_ratio * 30)

        net_ratio = abs(snapshot.total_net_exposure) / self._exp_config.max_net_exposure
        score += min(20, net_ratio * 20)

        conc_ratio = snapshot.concentration_score / self._exp_config.max_top_5_concentration
        score += min(25, conc_ratio * 25)

        over_limit_sectors = sum(1 for s in snapshot.sector_exposures if s.is_over_limit)
        score += min(15, over_limit_sectors * 5)

        if self._exp_config.use_beta_adjustment:
            beta_ratio = snapshot.beta_adjusted_exposure / self._exp_config.max_portfolio_beta
            score += min(10, beta_ratio * 10)

        return min(100, score)

    def _create_exposure_metrics(self, snapshot: ExposureSnapshot) -> list[RiskMetric]:
        """Create risk metrics from exposure snapshot."""
        return [
            RiskMetric(
                name="gross_exposure",
                value=snapshot.total_gross_exposure,
                unit="ratio",
                timestamp=snapshot.timestamp,
                risk_level=self._get_exposure_level(
                    snapshot.total_gross_exposure,
                    self._exp_config.max_gross_exposure,
                ),
                threshold_warning=self._exp_config.max_gross_exposure * 0.8,
                threshold_critical=self._exp_config.max_gross_exposure,
            ),
            RiskMetric(
                name="net_exposure",
                value=snapshot.total_net_exposure,
                unit="ratio",
                timestamp=snapshot.timestamp,
                risk_level=self._get_exposure_level(
                    abs(snapshot.total_net_exposure),
                    self._exp_config.max_net_exposure,
                ),
            ),
            RiskMetric(
                name="leverage_ratio",
                value=snapshot.leverage_ratio,
                unit="ratio",
                timestamp=snapshot.timestamp,
                risk_level=self._get_exposure_level(
                    snapshot.leverage_ratio,
                    self._exp_config.max_gross_exposure,
                ),
            ),
            RiskMetric(
                name="concentration_score",
                value=snapshot.concentration_score,
                unit="percent",
                timestamp=snapshot.timestamp,
                risk_level=self._get_exposure_level(
                    snapshot.concentration_score,
                    self._exp_config.max_top_5_concentration,
                ),
            ),
            RiskMetric(
                name="beta_exposure",
                value=snapshot.beta_adjusted_exposure,
                unit="beta",
                timestamp=snapshot.timestamp,
                risk_level=self._get_exposure_level(
                    snapshot.beta_adjusted_exposure,
                    self._exp_config.max_portfolio_beta,
                ),
            ),
        ]

    def _get_exposure_level(self, value: float, limit: float) -> RiskLevel:
        """Get risk level for exposure value."""
        ratio = value / limit if limit > 0 else 0

        if ratio >= 1.0:
            return RiskLevel.HIGH
        elif ratio >= 0.8:
            return RiskLevel.MEDIUM
        elif ratio >= 0.5:
            return RiskLevel.LOW
        return RiskLevel.MINIMAL

    def _generate_exposure_recommendations(
        self,
        snapshot: ExposureSnapshot,
    ) -> list[str]:
        """Generate exposure-based recommendations."""
        recommendations: list[str] = []

        if snapshot.total_gross_exposure > self._exp_config.target_gross_exposure * 1.2:
            recommendations.append(
                f"Consider reducing gross exposure from {snapshot.total_gross_exposure:.2f}x "
                f"toward target {self._exp_config.target_gross_exposure:.2f}x"
            )

        if snapshot.concentration_score > self._exp_config.max_top_5_concentration * 0.8:
            recommendations.append(
                f"Portfolio concentration at {snapshot.concentration_score:.1%} - consider diversifying"
            )

        over_limit_sectors = [s for s in snapshot.sector_exposures if s.is_over_limit]
        for sector in over_limit_sectors:
            recommendations.append(
                f"Reduce {sector.name} sector exposure from {sector.pct_of_portfolio:.1%}"
            )

        if abs(snapshot.total_net_exposure - self._exp_config.target_net_exposure) > 0.2:
            recommendations.append(
                f"Net exposure {snapshot.total_net_exposure:.2f}x differs from "
                f"target {self._exp_config.target_net_exposure:.2f}x"
            )

        return recommendations

    def set_sector_mapping(self, symbol: str, sector: str) -> None:
        """Set sector mapping for a symbol."""
        self._sector_mappings[symbol] = sector

    def set_asset_class_mapping(self, symbol: str, asset_class: str) -> None:
        """Set asset class mapping for a symbol."""
        self._asset_class_mappings[symbol] = asset_class

    def set_geography_mapping(self, symbol: str, geography: str) -> None:
        """Set geography mapping for a symbol."""
        self._geography_mappings[symbol] = geography

    def set_beta(self, symbol: str, beta: float) -> None:
        """Set beta value for a symbol."""
        self._beta_values[symbol] = beta

    def get_current_snapshot(self) -> Optional[ExposureSnapshot]:
        """Get most recent exposure snapshot."""
        if self._exposure_history:
            return self._exposure_history[-1]
        return None

    def get_exposure_history(self, limit: int = 100) -> list[ExposureSnapshot]:
        """Get exposure snapshot history."""
        return self._exposure_history[-limit:]

    def get_exposure_summary(self) -> dict:
        """Get exposure management summary."""
        snapshot = self.get_current_snapshot()

        if not snapshot:
            return {"status": "no_data"}

        return {
            "gross_exposure": snapshot.total_gross_exposure,
            "net_exposure": snapshot.total_net_exposure,
            "long_exposure": snapshot.total_long_exposure,
            "short_exposure": snapshot.total_short_exposure,
            "leverage": snapshot.leverage_ratio,
            "beta_adjusted": snapshot.beta_adjusted_exposure,
            "concentration": snapshot.concentration_score,
            "sector_count": len(snapshot.sector_exposures),
            "positions_count": sum(s.position_count for s in snapshot.sector_exposures),
        }

    def __repr__(self) -> str:
        """String representation."""
        snapshot = self.get_current_snapshot()
        if snapshot:
            return f"ExposureManager(gross={snapshot.total_gross_exposure:.2f}x, net={snapshot.total_net_exposure:.2f}x)"
        return "ExposureManager(no_data)"
