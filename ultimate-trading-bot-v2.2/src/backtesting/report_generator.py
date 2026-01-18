"""
Report Generator for Backtesting.

This module provides comprehensive report generation capabilities for
backtesting results, including HTML, PDF, and JSON report formats.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ReportFormat(str, Enum):
    """Supported report formats."""

    HTML = "html"
    JSON = "json"
    MARKDOWN = "markdown"
    TEXT = "text"
    CSV = "csv"


class ReportSection(str, Enum):
    """Report sections."""

    SUMMARY = "summary"
    PERFORMANCE = "performance"
    RISK = "risk"
    TRADES = "trades"
    DRAWDOWN = "drawdown"
    MONTHLY_RETURNS = "monthly_returns"
    ROLLING_METRICS = "rolling_metrics"
    COMPARISON = "comparison"
    POSITIONS = "positions"
    CUSTOM = "custom"


class ReportConfig(BaseModel):
    """Configuration for report generation."""

    title: str = Field(default="Backtest Report", description="Report title")
    output_dir: str = Field(default="./reports", description="Output directory")
    formats: list[ReportFormat] = Field(
        default=[ReportFormat.HTML, ReportFormat.JSON],
        description="Output formats",
    )
    include_sections: list[ReportSection] = Field(
        default=[
            ReportSection.SUMMARY,
            ReportSection.PERFORMANCE,
            ReportSection.RISK,
            ReportSection.TRADES,
            ReportSection.DRAWDOWN,
            ReportSection.MONTHLY_RETURNS,
        ],
        description="Sections to include",
    )
    include_charts: bool = Field(default=True, description="Include charts in HTML")
    include_trade_list: bool = Field(default=True, description="Include trade list")
    max_trades_display: int = Field(default=100, description="Max trades to display")
    decimal_places: int = Field(default=4, description="Decimal places for numbers")
    date_format: str = Field(default="%Y-%m-%d", description="Date format")
    currency_symbol: str = Field(default="$", description="Currency symbol")
    benchmark_name: str = Field(default="Benchmark", description="Benchmark name")


@dataclass
class ReportMetrics:
    """Collection of metrics for report."""

    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_trade_return: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_holding_period: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    recovery_factor: float = 0.0
    ulcer_index: float = 0.0

    start_date: datetime | None = None
    end_date: datetime | None = None
    initial_capital: float = 0.0
    final_capital: float = 0.0


@dataclass
class TradeRecord:
    """Single trade record."""

    trade_id: int
    symbol: str
    side: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_percent: float
    holding_period: int
    fees: float = 0.0


@dataclass
class ReportData:
    """Complete data for report generation."""

    metrics: ReportMetrics
    equity_curve: pd.Series | None = None
    returns: pd.Series | None = None
    drawdown_series: pd.Series | None = None
    trades: list[TradeRecord] = field(default_factory=list)
    monthly_returns: pd.DataFrame | None = None
    rolling_sharpe: pd.Series | None = None
    benchmark_returns: pd.Series | None = None
    positions_history: pd.DataFrame | None = None
    custom_data: dict[str, Any] = field(default_factory=dict)


class HTMLReportBuilder:
    """Builder for HTML reports."""

    def __init__(self, config: ReportConfig) -> None:
        """
        Initialize HTML report builder.

        Args:
            config: Report configuration
        """
        self.config = config

    def build(self, data: ReportData) -> str:
        """
        Build HTML report.

        Args:
            data: Report data

        Returns:
            HTML string
        """
        sections = []

        sections.append(self._build_header())

        for section in self.config.include_sections:
            if section == ReportSection.SUMMARY:
                sections.append(self._build_summary_section(data))
            elif section == ReportSection.PERFORMANCE:
                sections.append(self._build_performance_section(data))
            elif section == ReportSection.RISK:
                sections.append(self._build_risk_section(data))
            elif section == ReportSection.TRADES:
                sections.append(self._build_trades_section(data))
            elif section == ReportSection.DRAWDOWN:
                sections.append(self._build_drawdown_section(data))
            elif section == ReportSection.MONTHLY_RETURNS:
                sections.append(self._build_monthly_returns_section(data))

        sections.append(self._build_footer())

        return "\n".join(sections)

    def _build_header(self) -> str:
        """Build HTML header."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config.title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 30px;
            margin-bottom: 20px;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 5px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-label {{
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
        }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        th {{
            background-color: #34495e;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .chart-container {{
            width: 100%;
            height: 400px;
            margin: 20px 0;
        }}
        .timestamp {{
            color: #95a5a6;
            font-size: 12px;
            text-align: right;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{self.config.title}</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>"""

    def _build_summary_section(self, data: ReportData) -> str:
        """Build summary section."""
        m = data.metrics

        return_class = "positive" if m.total_return >= 0 else "negative"
        dd_class = "negative" if m.max_drawdown < -0.1 else ""

        return f"""
    <div class="container">
        <h2>Executive Summary</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value {return_class}">{m.total_return:.2%}</div>
                <div class="metric-label">Total Return</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {return_class}">{m.annual_return:.2%}</div>
                <div class="metric-label">Annual Return</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{m.sharpe_ratio:.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {dd_class}">{m.max_drawdown:.2%}</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{m.total_trades}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{m.win_rate:.1%}</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{self.config.currency_symbol}{m.initial_capital:,.0f}</div>
                <div class="metric-label">Initial Capital</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{self.config.currency_symbol}{m.final_capital:,.0f}</div>
                <div class="metric-label">Final Capital</div>
            </div>
        </div>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Start Date</td><td>{m.start_date.strftime(self.config.date_format) if m.start_date else 'N/A'}</td></tr>
            <tr><td>End Date</td><td>{m.end_date.strftime(self.config.date_format) if m.end_date else 'N/A'}</td></tr>
            <tr><td>Volatility (Annual)</td><td>{m.volatility:.2%}</td></tr>
            <tr><td>Profit Factor</td><td>{m.profit_factor:.2f}</td></tr>
        </table>
    </div>"""

    def _build_performance_section(self, data: ReportData) -> str:
        """Build performance section."""
        m = data.metrics

        return f"""
    <div class="container">
        <h2>Performance Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{m.sortino_ratio:.2f}</div>
                <div class="metric-label">Sortino Ratio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{m.calmar_ratio:.2f}</div>
                <div class="metric-label">Calmar Ratio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{m.recovery_factor:.2f}</div>
                <div class="metric-label">Recovery Factor</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{m.ulcer_index:.4f}</div>
                <div class="metric-label">Ulcer Index</div>
            </div>
        </div>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Best Trade</td><td class="positive">{m.best_trade:.2%}</td></tr>
            <tr><td>Worst Trade</td><td class="negative">{m.worst_trade:.2%}</td></tr>
            <tr><td>Average Trade</td><td>{m.avg_trade_return:.2%}</td></tr>
            <tr><td>Average Holding Period</td><td>{m.avg_holding_period:.1f} days</td></tr>
        </table>
    </div>"""

    def _build_risk_section(self, data: ReportData) -> str:
        """Build risk section."""
        m = data.metrics

        return f"""
    <div class="container">
        <h2>Risk Analysis</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value negative">{m.max_drawdown:.2%}</div>
                <div class="metric-label">Maximum Drawdown</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{m.volatility:.2%}</div>
                <div class="metric-label">Annual Volatility</div>
            </div>
        </div>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Max Consecutive Wins</td><td>{m.max_consecutive_wins}</td></tr>
            <tr><td>Max Consecutive Losses</td><td>{m.max_consecutive_losses}</td></tr>
            <tr><td>Average Win</td><td class="positive">{m.avg_win:.2%}</td></tr>
            <tr><td>Average Loss</td><td class="negative">{m.avg_loss:.2%}</td></tr>
        </table>
    </div>"""

    def _build_trades_section(self, data: ReportData) -> str:
        """Build trades section."""
        m = data.metrics

        trades_html = ""
        if self.config.include_trade_list and data.trades:
            trades_to_show = data.trades[: self.config.max_trades_display]

            rows = []
            for trade in trades_to_show:
                pnl_class = "positive" if trade.pnl >= 0 else "negative"
                rows.append(f"""
                <tr>
                    <td>{trade.trade_id}</td>
                    <td>{trade.symbol}</td>
                    <td>{trade.side}</td>
                    <td>{trade.entry_date.strftime(self.config.date_format)}</td>
                    <td>{trade.exit_date.strftime(self.config.date_format)}</td>
                    <td>{self.config.currency_symbol}{trade.entry_price:.2f}</td>
                    <td>{self.config.currency_symbol}{trade.exit_price:.2f}</td>
                    <td class="{pnl_class}">{self.config.currency_symbol}{trade.pnl:.2f}</td>
                    <td class="{pnl_class}">{trade.pnl_percent:.2%}</td>
                </tr>""")

            trades_html = f"""
        <h3>Trade History</h3>
        <table>
            <tr>
                <th>ID</th>
                <th>Symbol</th>
                <th>Side</th>
                <th>Entry Date</th>
                <th>Exit Date</th>
                <th>Entry Price</th>
                <th>Exit Price</th>
                <th>P&L</th>
                <th>P&L %</th>
            </tr>
            {"".join(rows)}
        </table>"""

            if len(data.trades) > self.config.max_trades_display:
                trades_html += f"<p><em>Showing {self.config.max_trades_display} of {len(data.trades)} trades</em></p>"

        return f"""
    <div class="container">
        <h2>Trade Analysis</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{m.total_trades}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric-card">
                <div class="metric-value positive">{m.winning_trades}</div>
                <div class="metric-label">Winning Trades</div>
            </div>
            <div class="metric-card">
                <div class="metric-value negative">{m.losing_trades}</div>
                <div class="metric-label">Losing Trades</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{m.profit_factor:.2f}</div>
                <div class="metric-label">Profit Factor</div>
            </div>
        </div>
        {trades_html}
    </div>"""

    def _build_drawdown_section(self, data: ReportData) -> str:
        """Build drawdown section."""
        m = data.metrics

        underwater_table = ""
        if data.drawdown_series is not None and len(data.drawdown_series) > 0:
            worst_dd = data.drawdown_series.nsmallest(5)
            rows = []
            for date, dd in worst_dd.items():
                rows.append(f"<tr><td>{date.strftime(self.config.date_format)}</td><td class='negative'>{dd:.2%}</td></tr>")

            underwater_table = f"""
        <h3>Worst Drawdown Periods</h3>
        <table>
            <tr><th>Date</th><th>Drawdown</th></tr>
            {"".join(rows)}
        </table>"""

        return f"""
    <div class="container">
        <h2>Drawdown Analysis</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value negative">{m.max_drawdown:.2%}</div>
                <div class="metric-label">Maximum Drawdown</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{m.recovery_factor:.2f}</div>
                <div class="metric-label">Recovery Factor</div>
            </div>
        </div>
        {underwater_table}
    </div>"""

    def _build_monthly_returns_section(self, data: ReportData) -> str:
        """Build monthly returns section."""
        if data.monthly_returns is None or data.monthly_returns.empty:
            return ""

        table_rows = []
        for year in data.monthly_returns.index:
            row = f"<tr><td><strong>{year}</strong></td>"
            for month in range(1, 13):
                if month in data.monthly_returns.columns:
                    value = data.monthly_returns.loc[year, month]
                    if pd.notna(value):
                        color_class = "positive" if value >= 0 else "negative"
                        row += f"<td class='{color_class}'>{value:.1%}</td>"
                    else:
                        row += "<td>-</td>"
                else:
                    row += "<td>-</td>"

            if "Year" in data.monthly_returns.columns:
                year_total = data.monthly_returns.loc[year, "Year"]
                color_class = "positive" if year_total >= 0 else "negative"
                row += f"<td class='{color_class}'><strong>{year_total:.1%}</strong></td>"
            else:
                row += "<td>-</td>"

            row += "</tr>"
            table_rows.append(row)

        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Year"]
        header = "<tr><th>Year</th>" + "".join(f"<th>{m}</th>" for m in months) + "</tr>"

        return f"""
    <div class="container">
        <h2>Monthly Returns</h2>
        <table>
            {header}
            {"".join(table_rows)}
        </table>
    </div>"""

    def _build_footer(self) -> str:
        """Build HTML footer."""
        return """
    <div class="container">
        <p class="timestamp">Report generated by Ultimate Trading Bot v2.2</p>
    </div>
</body>
</html>"""


class JSONReportBuilder:
    """Builder for JSON reports."""

    def __init__(self, config: ReportConfig) -> None:
        """
        Initialize JSON report builder.

        Args:
            config: Report configuration
        """
        self.config = config

    def build(self, data: ReportData) -> str:
        """
        Build JSON report.

        Args:
            data: Report data

        Returns:
            JSON string
        """
        report = {
            "title": self.config.title,
            "generated_at": datetime.now().isoformat(),
            "metrics": self._metrics_to_dict(data.metrics),
            "trades_summary": {
                "total": len(data.trades),
                "trades": [self._trade_to_dict(t) for t in data.trades[: self.config.max_trades_display]],
            },
        }

        if data.equity_curve is not None:
            report["equity_curve"] = {
                "dates": [d.isoformat() for d in data.equity_curve.index],
                "values": data.equity_curve.tolist(),
            }

        if data.returns is not None:
            report["returns"] = {
                "dates": [d.isoformat() for d in data.returns.index],
                "values": data.returns.tolist(),
            }

        if data.monthly_returns is not None:
            report["monthly_returns"] = data.monthly_returns.to_dict()

        if data.custom_data:
            report["custom_data"] = data.custom_data

        return json.dumps(report, indent=2, default=str)

    def _metrics_to_dict(self, metrics: ReportMetrics) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_return": metrics.total_return,
            "annual_return": metrics.annual_return,
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "calmar_ratio": metrics.calmar_ratio,
            "max_drawdown": metrics.max_drawdown,
            "volatility": metrics.volatility,
            "win_rate": metrics.win_rate,
            "profit_factor": metrics.profit_factor,
            "total_trades": metrics.total_trades,
            "winning_trades": metrics.winning_trades,
            "losing_trades": metrics.losing_trades,
            "avg_trade_return": metrics.avg_trade_return,
            "avg_win": metrics.avg_win,
            "avg_loss": metrics.avg_loss,
            "best_trade": metrics.best_trade,
            "worst_trade": metrics.worst_trade,
            "avg_holding_period": metrics.avg_holding_period,
            "max_consecutive_wins": metrics.max_consecutive_wins,
            "max_consecutive_losses": metrics.max_consecutive_losses,
            "recovery_factor": metrics.recovery_factor,
            "ulcer_index": metrics.ulcer_index,
            "start_date": metrics.start_date.isoformat() if metrics.start_date else None,
            "end_date": metrics.end_date.isoformat() if metrics.end_date else None,
            "initial_capital": metrics.initial_capital,
            "final_capital": metrics.final_capital,
        }

    def _trade_to_dict(self, trade: TradeRecord) -> dict[str, Any]:
        """Convert trade to dictionary."""
        return {
            "trade_id": trade.trade_id,
            "symbol": trade.symbol,
            "side": trade.side,
            "entry_date": trade.entry_date.isoformat(),
            "exit_date": trade.exit_date.isoformat(),
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "quantity": trade.quantity,
            "pnl": trade.pnl,
            "pnl_percent": trade.pnl_percent,
            "holding_period": trade.holding_period,
            "fees": trade.fees,
        }


class MarkdownReportBuilder:
    """Builder for Markdown reports."""

    def __init__(self, config: ReportConfig) -> None:
        """
        Initialize Markdown report builder.

        Args:
            config: Report configuration
        """
        self.config = config

    def build(self, data: ReportData) -> str:
        """
        Build Markdown report.

        Args:
            data: Report data

        Returns:
            Markdown string
        """
        sections = []

        sections.append(f"# {self.config.title}\n")
        sections.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

        sections.append(self._build_summary(data))
        sections.append(self._build_performance(data))
        sections.append(self._build_risk(data))
        sections.append(self._build_trades(data))

        return "\n".join(sections)

    def _build_summary(self, data: ReportData) -> str:
        """Build summary section."""
        m = data.metrics

        return f"""
## Executive Summary

| Metric | Value |
|--------|-------|
| Total Return | {m.total_return:.2%} |
| Annual Return | {m.annual_return:.2%} |
| Sharpe Ratio | {m.sharpe_ratio:.2f} |
| Max Drawdown | {m.max_drawdown:.2%} |
| Win Rate | {m.win_rate:.1%} |
| Total Trades | {m.total_trades} |
| Initial Capital | {self.config.currency_symbol}{m.initial_capital:,.0f} |
| Final Capital | {self.config.currency_symbol}{m.final_capital:,.0f} |
"""

    def _build_performance(self, data: ReportData) -> str:
        """Build performance section."""
        m = data.metrics

        return f"""
## Performance Metrics

| Metric | Value |
|--------|-------|
| Sortino Ratio | {m.sortino_ratio:.2f} |
| Calmar Ratio | {m.calmar_ratio:.2f} |
| Recovery Factor | {m.recovery_factor:.2f} |
| Profit Factor | {m.profit_factor:.2f} |
| Best Trade | {m.best_trade:.2%} |
| Worst Trade | {m.worst_trade:.2%} |
| Average Trade | {m.avg_trade_return:.2%} |
"""

    def _build_risk(self, data: ReportData) -> str:
        """Build risk section."""
        m = data.metrics

        return f"""
## Risk Analysis

| Metric | Value |
|--------|-------|
| Maximum Drawdown | {m.max_drawdown:.2%} |
| Annual Volatility | {m.volatility:.2%} |
| Ulcer Index | {m.ulcer_index:.4f} |
| Max Consecutive Wins | {m.max_consecutive_wins} |
| Max Consecutive Losses | {m.max_consecutive_losses} |
"""

    def _build_trades(self, data: ReportData) -> str:
        """Build trades section."""
        m = data.metrics

        return f"""
## Trade Statistics

| Metric | Value |
|--------|-------|
| Total Trades | {m.total_trades} |
| Winning Trades | {m.winning_trades} |
| Losing Trades | {m.losing_trades} |
| Average Win | {m.avg_win:.2%} |
| Average Loss | {m.avg_loss:.2%} |
| Average Holding Period | {m.avg_holding_period:.1f} days |
"""


class ReportGenerator:
    """Main report generator."""

    def __init__(
        self,
        config: ReportConfig | None = None,
    ) -> None:
        """
        Initialize report generator.

        Args:
            config: Report configuration
        """
        self.config = config or ReportConfig()

        self.builders = {
            ReportFormat.HTML: HTMLReportBuilder(self.config),
            ReportFormat.JSON: JSONReportBuilder(self.config),
            ReportFormat.MARKDOWN: MarkdownReportBuilder(self.config),
        }

        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"ReportGenerator initialized with output dir: {self.config.output_dir}")

    def generate(
        self,
        data: ReportData,
        filename_base: str | None = None,
    ) -> dict[ReportFormat, str]:
        """
        Generate reports in all configured formats.

        Args:
            data: Report data
            filename_base: Base filename for output files

        Returns:
            Dictionary mapping format to file path
        """
        if filename_base is None:
            filename_base = f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        output_files: dict[ReportFormat, str] = {}

        for format_type in self.config.formats:
            if format_type in self.builders:
                builder = self.builders[format_type]
                content = builder.build(data)

                extension = format_type.value
                if format_type == ReportFormat.MARKDOWN:
                    extension = "md"

                filepath = Path(self.config.output_dir) / f"{filename_base}.{extension}"

                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)

                output_files[format_type] = str(filepath)
                logger.info(f"Generated {format_type.value} report: {filepath}")

        return output_files

    def generate_comparison_report(
        self,
        results: dict[str, ReportData],
        filename_base: str | None = None,
    ) -> dict[ReportFormat, str]:
        """
        Generate comparison report for multiple strategies.

        Args:
            results: Dictionary mapping strategy name to report data
            filename_base: Base filename for output

        Returns:
            Dictionary mapping format to file path
        """
        if filename_base is None:
            filename_base = f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        comparison_data = self._build_comparison_data(results)

        combined = ReportData(
            metrics=ReportMetrics(),
            custom_data={"comparison": comparison_data},
        )

        return self.generate(combined, filename_base)

    def _build_comparison_data(
        self,
        results: dict[str, ReportData],
    ) -> dict[str, Any]:
        """Build comparison data structure."""
        comparison = {
            "strategies": list(results.keys()),
            "metrics_comparison": {},
        }

        metric_names = [
            "total_return",
            "annual_return",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "volatility",
            "win_rate",
            "profit_factor",
        ]

        for metric in metric_names:
            comparison["metrics_comparison"][metric] = {
                name: getattr(data.metrics, metric, None)
                for name, data in results.items()
            }

        return comparison


def create_report_generator(
    output_dir: str = "./reports",
    formats: list[str] | None = None,
    config: dict | None = None,
) -> ReportGenerator:
    """
    Create a report generator.

    Args:
        output_dir: Output directory for reports
        formats: List of output formats
        config: Additional configuration

    Returns:
        Configured ReportGenerator
    """
    format_list = [ReportFormat(f) for f in (formats or ["html", "json"])]

    report_config = ReportConfig(
        output_dir=output_dir,
        formats=format_list,
        **(config or {}),
    )
    return ReportGenerator(config=report_config)
