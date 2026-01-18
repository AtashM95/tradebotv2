"""
Risk Reporting for Ultimate Trading Bot v2.2.

This module provides comprehensive risk reporting including
dashboards, alerts, and automated report generation.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.risk.base_risk import RiskAlert, RiskLevel


logger = logging.getLogger(__name__)


class ReportFormat(str, Enum):
    """Report output formats."""

    JSON = "json"
    HTML = "html"
    TEXT = "text"
    MARKDOWN = "markdown"
    CSV = "csv"


class ReportFrequency(str, Enum):
    """Report generation frequency."""

    REAL_TIME = "real_time"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class ReportType(str, Enum):
    """Types of risk reports."""

    EXECUTIVE_SUMMARY = "executive_summary"
    DETAILED_RISK = "detailed_risk"
    POSITION_RISK = "position_risk"
    EXPOSURE_BREAKDOWN = "exposure_breakdown"
    LIMIT_STATUS = "limit_status"
    PERFORMANCE_ATTRIBUTION = "performance_attribution"
    STRESS_TEST = "stress_test"
    VAR_REPORT = "var_report"
    COMPLIANCE = "compliance"
    ALERT_HISTORY = "alert_history"


class RiskReporterConfig(BaseModel):
    """Configuration for risk reporting."""

    model_config = {"arbitrary_types_allowed": True}

    output_dir: str = Field(default="./reports/risk", description="Report output directory")
    default_format: ReportFormat = Field(default=ReportFormat.JSON)
    generate_html: bool = Field(default=True, description="Generate HTML reports")
    archive_days: int = Field(default=90, description="Days to keep archived reports")
    include_charts: bool = Field(default=True, description="Include charts in HTML")
    email_reports: bool = Field(default=False, description="Email reports")
    email_recipients: list[str] = Field(default_factory=list)
    scheduled_reports: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Scheduled report configurations"
    )


class ReportSection(BaseModel):
    """Section of a risk report."""

    title: str
    content: dict[str, Any] = Field(default_factory=dict)
    tables: list[dict[str, Any]] = Field(default_factory=list)
    charts: list[dict[str, Any]] = Field(default_factory=list)
    alerts: list[RiskAlert] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class RiskReport(BaseModel):
    """Complete risk report."""

    report_id: str
    report_type: ReportType
    generated_at: datetime = Field(default_factory=datetime.now)
    period_start: datetime | None = None
    period_end: datetime | None = None

    title: str = ""
    summary: str = ""
    overall_risk_level: RiskLevel = Field(default=RiskLevel.MEDIUM)
    risk_score: float = Field(default=50.0)

    sections: list[ReportSection] = Field(default_factory=list)
    key_metrics: dict[str, Any] = Field(default_factory=dict)
    alerts: list[RiskAlert] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class ReportSchedule:
    """Scheduled report configuration."""

    report_type: ReportType
    frequency: ReportFrequency
    recipients: list[str] = field(default_factory=list)
    last_run: datetime | None = None
    next_run: datetime | None = None
    is_enabled: bool = True


class RiskReporter:
    """
    Generates comprehensive risk reports.

    Provides automated report generation, scheduling, and distribution.
    """

    def __init__(self, config: RiskReporterConfig | None = None):
        """
        Initialize risk reporter.

        Args:
            config: Reporter configuration
        """
        self.config = config or RiskReporterConfig()
        self._report_history: list[RiskReport] = []
        self._schedules: dict[str, ReportSchedule] = {}
        self._lock = asyncio.Lock()

        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("RiskReporter initialized")

    async def generate_executive_summary(
        self,
        portfolio_data: dict[str, Any],
        risk_metrics: dict[str, Any],
        alerts: list[RiskAlert],
    ) -> RiskReport:
        """
        Generate executive summary report.

        Args:
            portfolio_data: Portfolio information
            risk_metrics: Current risk metrics
            alerts: Active alerts

        Returns:
            RiskReport object
        """
        report_id = f"exec_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        overall_level, risk_score = self._assess_overall_risk(risk_metrics, alerts)

        key_metrics = {
            "portfolio_value": portfolio_data.get("equity", 0),
            "daily_pnl": portfolio_data.get("daily_pnl", 0),
            "daily_pnl_pct": portfolio_data.get("daily_pnl_pct", 0),
            "total_return": risk_metrics.get("total_return", 0),
            "sharpe_ratio": risk_metrics.get("sharpe_ratio", 0),
            "max_drawdown": risk_metrics.get("max_drawdown", 0),
            "var_95": risk_metrics.get("var_95", 0),
            "current_leverage": portfolio_data.get("leverage", 1.0),
            "position_count": portfolio_data.get("position_count", 0),
        }

        overview_section = ReportSection(
            title="Portfolio Overview",
            content={
                "equity": portfolio_data.get("equity", 0),
                "cash": portfolio_data.get("cash", 0),
                "buying_power": portfolio_data.get("buying_power", 0),
                "positions": portfolio_data.get("position_count", 0),
            },
        )

        risk_section = ReportSection(
            title="Risk Summary",
            content={
                "risk_level": overall_level.value,
                "risk_score": risk_score,
                "var_95": risk_metrics.get("var_95", 0),
                "volatility": risk_metrics.get("volatility", 0),
                "beta": risk_metrics.get("beta", 0),
                "max_drawdown": risk_metrics.get("max_drawdown", 0),
            },
            alerts=[a for a in alerts if a.level in [RiskLevel.HIGH, RiskLevel.CRITICAL]],
        )

        performance_section = ReportSection(
            title="Performance",
            content={
                "total_return": risk_metrics.get("total_return", 0),
                "sharpe_ratio": risk_metrics.get("sharpe_ratio", 0),
                "sortino_ratio": risk_metrics.get("sortino_ratio", 0),
                "calmar_ratio": risk_metrics.get("calmar_ratio", 0),
                "win_rate": risk_metrics.get("win_rate", 0),
            },
        )

        recommendations = self._generate_recommendations(risk_metrics, alerts)

        report = RiskReport(
            report_id=report_id,
            report_type=ReportType.EXECUTIVE_SUMMARY,
            title="Executive Risk Summary",
            summary=self._generate_summary_text(key_metrics, overall_level),
            overall_risk_level=overall_level,
            risk_score=risk_score,
            sections=[overview_section, risk_section, performance_section],
            key_metrics=key_metrics,
            alerts=alerts,
            recommendations=recommendations,
        )

        await self._save_report(report)

        return report

    async def generate_detailed_risk_report(
        self,
        portfolio_data: dict[str, Any],
        positions: list[dict[str, Any]],
        risk_metrics: dict[str, Any],
        exposure_data: dict[str, Any],
        limit_status: dict[str, Any],
    ) -> RiskReport:
        """
        Generate detailed risk report.

        Args:
            portfolio_data: Portfolio information
            positions: Current positions
            risk_metrics: Risk metrics
            exposure_data: Exposure breakdown
            limit_status: Limit status

        Returns:
            RiskReport object
        """
        report_id = f"detailed_risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        overall_level, risk_score = self._assess_overall_risk(risk_metrics, [])

        position_section = ReportSection(
            title="Position Risk Analysis",
            content={"position_count": len(positions)},
            tables=[
                {
                    "name": "positions",
                    "headers": ["Symbol", "Value", "Weight", "Beta", "VaR", "Risk Contrib"],
                    "rows": [
                        [
                            p.get("symbol", ""),
                            f"${p.get('market_value', 0):,.2f}",
                            f"{p.get('weight', 0):.1%}",
                            f"{p.get('beta', 1.0):.2f}",
                            f"{p.get('var', 0):.2%}",
                            f"{p.get('risk_contribution', 0):.1%}",
                        ]
                        for p in positions[:20]
                    ],
                }
            ],
        )

        exposure_section = ReportSection(
            title="Exposure Breakdown",
            content=exposure_data,
            tables=[
                {
                    "name": "sector_exposure",
                    "headers": ["Sector", "Long", "Short", "Net", "Gross"],
                    "rows": [
                        [
                            sector,
                            f"{data.get('long', 0):.1%}",
                            f"{data.get('short', 0):.1%}",
                            f"{data.get('net', 0):.1%}",
                            f"{data.get('gross', 0):.1%}",
                        ]
                        for sector, data in exposure_data.get("by_sector", {}).items()
                    ],
                }
            ],
        )

        limits_section = ReportSection(
            title="Limit Utilization",
            content=limit_status,
            tables=[
                {
                    "name": "limits",
                    "headers": ["Limit", "Current", "Soft", "Hard", "Utilization", "Status"],
                    "rows": [
                        [
                            name,
                            f"{data.get('current', 0):.2f}",
                            f"{data.get('soft', 0):.2f}",
                            f"{data.get('hard', 0):.2f}",
                            f"{data.get('utilization', 0):.1%}",
                            data.get("status", "ok"),
                        ]
                        for name, data in limit_status.get("limits", {}).items()
                    ],
                }
            ],
        )

        var_section = ReportSection(
            title="Value at Risk Analysis",
            content={
                "var_95": risk_metrics.get("var_95", 0),
                "var_99": risk_metrics.get("var_99", 0),
                "cvar_95": risk_metrics.get("cvar_95", 0),
                "cvar_99": risk_metrics.get("cvar_99", 0),
                "method": risk_metrics.get("var_method", "historical"),
            },
        )

        report = RiskReport(
            report_id=report_id,
            report_type=ReportType.DETAILED_RISK,
            title="Detailed Risk Report",
            summary=f"Comprehensive risk analysis as of {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            overall_risk_level=overall_level,
            risk_score=risk_score,
            sections=[position_section, exposure_section, limits_section, var_section],
            key_metrics=risk_metrics,
        )

        await self._save_report(report)

        return report

    async def generate_stress_test_report(
        self,
        stress_results: list[dict[str, Any]],
        monte_carlo: dict[str, Any] | None = None,
    ) -> RiskReport:
        """
        Generate stress test report.

        Args:
            stress_results: Results from stress tests
            monte_carlo: Monte Carlo simulation results

        Returns:
            RiskReport object
        """
        report_id = f"stress_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        historical_section = ReportSection(
            title="Historical Scenario Analysis",
            tables=[
                {
                    "name": "scenarios",
                    "headers": ["Scenario", "Loss %", "Loss $", "Passed", "Breached Limits"],
                    "rows": [
                        [
                            r.get("scenario_name", ""),
                            f"{r.get('loss_pct', 0):.1%}",
                            f"${r.get('loss_amount', 0):,.0f}",
                            "Yes" if r.get("passed", True) else "No",
                            ", ".join(r.get("breached_limits", [])),
                        ]
                        for r in stress_results
                    ],
                }
            ],
        )

        sections = [historical_section]

        if monte_carlo:
            mc_section = ReportSection(
                title="Monte Carlo Simulation",
                content={
                    "simulations": monte_carlo.get("num_simulations", 0),
                    "mean_loss": monte_carlo.get("mean_loss", 0),
                    "var_95": monte_carlo.get("var_95", 0),
                    "var_99": monte_carlo.get("var_99", 0),
                    "worst_case": monte_carlo.get("worst_case", 0),
                    "probability_of_ruin": monte_carlo.get("probability_of_ruin", 0),
                },
            )
            sections.append(mc_section)

        worst_scenario = min(stress_results, key=lambda x: x.get("loss_pct", 0))
        failed_scenarios = [r for r in stress_results if not r.get("passed", True)]

        risk_level = RiskLevel.LOW
        if len(failed_scenarios) > 0:
            risk_level = RiskLevel.HIGH
        elif worst_scenario.get("loss_pct", 0) < -0.20:
            risk_level = RiskLevel.MEDIUM

        report = RiskReport(
            report_id=report_id,
            report_type=ReportType.STRESS_TEST,
            title="Stress Test Report",
            summary=f"Analyzed {len(stress_results)} scenarios. "
            f"Worst case: {worst_scenario.get('scenario_name', 'Unknown')} "
            f"({worst_scenario.get('loss_pct', 0):.1%})",
            overall_risk_level=risk_level,
            sections=sections,
            key_metrics={
                "scenarios_tested": len(stress_results),
                "scenarios_passed": len(stress_results) - len(failed_scenarios),
                "worst_loss_pct": worst_scenario.get("loss_pct", 0),
            },
        )

        await self._save_report(report)

        return report

    async def generate_alert_report(
        self,
        alerts: list[RiskAlert],
        period_hours: int = 24,
    ) -> RiskReport:
        """
        Generate alert history report.

        Args:
            alerts: Alert history
            period_hours: Period to report on

        Returns:
            RiskReport object
        """
        report_id = f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        cutoff = datetime.now() - timedelta(hours=period_hours)
        period_alerts = [a for a in alerts if a.timestamp >= cutoff]

        by_level: dict[str, int] = {}
        by_type: dict[str, int] = {}

        for alert in period_alerts:
            level = alert.level.value
            by_level[level] = by_level.get(level, 0) + 1

            alert_type = alert.alert_type.value
            by_type[alert_type] = by_type.get(alert_type, 0) + 1

        summary_section = ReportSection(
            title="Alert Summary",
            content={
                "total_alerts": len(period_alerts),
                "by_level": by_level,
                "by_type": by_type,
                "period_hours": period_hours,
            },
        )

        details_section = ReportSection(
            title="Alert Details",
            tables=[
                {
                    "name": "alerts",
                    "headers": ["Time", "Level", "Type", "Message", "Action Required"],
                    "rows": [
                        [
                            a.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                            a.level.value,
                            a.alert_type.value,
                            a.message[:50],
                            "Yes" if a.requires_action else "No",
                        ]
                        for a in sorted(period_alerts, key=lambda x: x.timestamp, reverse=True)[:50]
                    ],
                }
            ],
            alerts=period_alerts,
        )

        critical_count = by_level.get("critical", 0)
        high_count = by_level.get("high", 0)

        if critical_count > 0:
            risk_level = RiskLevel.CRITICAL
        elif high_count > 5:
            risk_level = RiskLevel.HIGH
        elif high_count > 0:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        report = RiskReport(
            report_id=report_id,
            report_type=ReportType.ALERT_HISTORY,
            title=f"Alert Report - Last {period_hours} Hours",
            summary=f"{len(period_alerts)} alerts in period. "
            f"Critical: {critical_count}, High: {high_count}",
            overall_risk_level=risk_level,
            sections=[summary_section, details_section],
            key_metrics={
                "total_alerts": len(period_alerts),
                "critical_alerts": critical_count,
                "high_alerts": high_count,
            },
            alerts=period_alerts,
        )

        await self._save_report(report)

        return report

    def _assess_overall_risk(
        self,
        metrics: dict[str, Any],
        alerts: list[RiskAlert],
    ) -> tuple[RiskLevel, float]:
        """Assess overall risk level and score."""
        score = 50.0

        if abs(metrics.get("max_drawdown", 0)) > 0.15:
            score += 20
        elif abs(metrics.get("max_drawdown", 0)) > 0.10:
            score += 10

        if metrics.get("var_95", 0) and abs(metrics["var_95"]) > 0.03:
            score += 15
        elif metrics.get("var_95", 0) and abs(metrics["var_95"]) > 0.02:
            score += 8

        if metrics.get("volatility", 0) > 0.25:
            score += 10

        if metrics.get("leverage", 1.0) > 2.0:
            score += 15
        elif metrics.get("leverage", 1.0) > 1.5:
            score += 8

        critical_alerts = sum(1 for a in alerts if a.level == RiskLevel.CRITICAL)
        high_alerts = sum(1 for a in alerts if a.level == RiskLevel.HIGH)

        score += critical_alerts * 10
        score += high_alerts * 5

        score = max(0, min(100, score))

        if score >= 80:
            level = RiskLevel.CRITICAL
        elif score >= 60:
            level = RiskLevel.HIGH
        elif score >= 40:
            level = RiskLevel.MEDIUM
        elif score >= 20:
            level = RiskLevel.LOW
        else:
            level = RiskLevel.MINIMAL

        return level, score

    def _generate_summary_text(
        self,
        metrics: dict[str, Any],
        risk_level: RiskLevel,
    ) -> str:
        """Generate summary text."""
        portfolio_value = metrics.get("portfolio_value", 0)
        daily_pnl_pct = metrics.get("daily_pnl_pct", 0)
        sharpe = metrics.get("sharpe_ratio", 0)

        pnl_direction = "up" if daily_pnl_pct >= 0 else "down"

        summary = (
            f"Portfolio value: ${portfolio_value:,.2f}. "
            f"Today: {pnl_direction} {abs(daily_pnl_pct):.2%}. "
            f"Sharpe ratio: {sharpe:.2f}. "
            f"Overall risk level: {risk_level.value}."
        )

        return summary

    def _generate_recommendations(
        self,
        metrics: dict[str, Any],
        alerts: list[RiskAlert],
    ) -> list[str]:
        """Generate recommendations based on analysis."""
        recommendations: list[str] = []

        if abs(metrics.get("max_drawdown", 0)) > 0.15:
            recommendations.append(
                "Consider reducing position sizes to limit drawdown exposure"
            )

        if metrics.get("leverage", 1.0) > 1.5:
            recommendations.append(
                "Current leverage is elevated - consider delevering"
            )

        if metrics.get("volatility", 0) > 0.25:
            recommendations.append(
                "Portfolio volatility is high - review position correlations"
            )

        concentration = metrics.get("concentration", 0)
        if concentration > 0.2:
            recommendations.append(
                "High concentration detected - consider diversifying positions"
            )

        high_alerts = [a for a in alerts if a.level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
        if len(high_alerts) > 3:
            recommendations.append(
                f"Multiple high-priority alerts ({len(high_alerts)}) require attention"
            )

        if not recommendations:
            recommendations.append("Portfolio risk metrics are within acceptable ranges")

        return recommendations

    async def _save_report(self, report: RiskReport) -> str:
        """
        Save report to file.

        Args:
            report: Report to save

        Returns:
            File path
        """
        async with self._lock:
            self._report_history.append(report)

        output_dir = Path(self.config.output_dir)
        date_dir = output_dir / datetime.now().strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)

        if self.config.default_format == ReportFormat.JSON:
            file_path = date_dir / f"{report.report_id}.json"
            with open(file_path, "w") as f:
                json.dump(report.model_dump(mode="json", exclude_none=True), f, indent=2, default=str)
        else:
            file_path = date_dir / f"{report.report_id}.txt"
            with open(file_path, "w") as f:
                f.write(self._format_as_text(report))

        logger.info(f"Saved report: {file_path}")

        return str(file_path)

    def _format_as_text(self, report: RiskReport) -> str:
        """Format report as text."""
        lines = [
            "=" * 60,
            report.title,
            "=" * 60,
            f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Risk Level: {report.overall_risk_level.value}",
            f"Risk Score: {report.risk_score:.1f}/100",
            "",
            "SUMMARY",
            "-" * 40,
            report.summary,
            "",
        ]

        for section in report.sections:
            lines.extend([
                section.title.upper(),
                "-" * 40,
            ])

            for key, value in section.content.items():
                lines.append(f"  {key}: {value}")

            lines.append("")

        if report.recommendations:
            lines.extend([
                "RECOMMENDATIONS",
                "-" * 40,
            ])
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"  {i}. {rec}")

        return "\n".join(lines)

    def get_report_history(
        self,
        report_type: ReportType | None = None,
        limit: int = 100,
    ) -> list[RiskReport]:
        """Get report history."""
        reports = self._report_history

        if report_type:
            reports = [r for r in reports if r.report_type == report_type]

        return reports[-limit:]

    async def get_reporting_summary(self) -> dict[str, Any]:
        """
        Get reporting summary.

        Returns:
            Summary dictionary
        """
        by_type: dict[str, int] = {}
        for report in self._report_history:
            t = report.report_type.value
            by_type[t] = by_type.get(t, 0) + 1

        return {
            "timestamp": datetime.now().isoformat(),
            "total_reports_generated": len(self._report_history),
            "reports_by_type": by_type,
            "output_directory": self.config.output_dir,
            "default_format": self.config.default_format.value,
            "scheduled_reports": len(self._schedules),
        }
