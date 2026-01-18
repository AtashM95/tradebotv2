"""
Visualization Module for Backtesting.

This module provides comprehensive visualization capabilities for
backtesting results, including equity curves, drawdowns, and performance charts.
"""

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


class ChartType(str, Enum):
    """Types of charts."""

    EQUITY_CURVE = "equity_curve"
    DRAWDOWN = "drawdown"
    RETURNS_DISTRIBUTION = "returns_distribution"
    MONTHLY_HEATMAP = "monthly_heatmap"
    ROLLING_SHARPE = "rolling_sharpe"
    ROLLING_VOLATILITY = "rolling_volatility"
    TRADE_ANALYSIS = "trade_analysis"
    UNDERWATER = "underwater"
    COMPARISON = "comparison"
    CORRELATION = "correlation"
    POSITION_SIZE = "position_size"


class ColorScheme(str, Enum):
    """Color schemes for charts."""

    DEFAULT = "default"
    DARK = "dark"
    LIGHT = "light"
    COLORBLIND = "colorblind"


class ChartConfig(BaseModel):
    """Configuration for chart generation."""

    width: int = Field(default=1200, description="Chart width in pixels")
    height: int = Field(default=600, description="Chart height in pixels")
    color_scheme: ColorScheme = Field(default=ColorScheme.DEFAULT, description="Color scheme")
    title_font_size: int = Field(default=16, description="Title font size")
    label_font_size: int = Field(default=12, description="Label font size")
    show_grid: bool = Field(default=True, description="Show grid lines")
    show_legend: bool = Field(default=True, description="Show legend")
    output_format: str = Field(default="png", description="Output format")
    dpi: int = Field(default=100, description="DPI for raster formats")
    transparent: bool = Field(default=False, description="Transparent background")


class VisualizationConfig(BaseModel):
    """Configuration for visualization module."""

    output_dir: str = Field(default="./charts", description="Output directory")
    chart_config: ChartConfig = Field(default_factory=ChartConfig, description="Chart configuration")
    save_charts: bool = Field(default=True, description="Save charts to files")
    interactive: bool = Field(default=False, description="Generate interactive charts")


@dataclass
class ChartData:
    """Data for a single chart."""

    chart_type: ChartType
    title: str
    data: dict[str, Any]
    config: ChartConfig | None = None


@dataclass
class ChartResult:
    """Result of chart generation."""

    chart_type: ChartType
    file_path: str | None = None
    html_content: str | None = None
    svg_content: str | None = None
    success: bool = True
    error_message: str | None = None


class ColorPalette:
    """Color palette for charts."""

    PALETTES = {
        ColorScheme.DEFAULT: {
            "primary": "#3498db",
            "secondary": "#2ecc71",
            "negative": "#e74c3c",
            "positive": "#27ae60",
            "neutral": "#95a5a6",
            "background": "#ffffff",
            "grid": "#ecf0f1",
            "text": "#2c3e50",
        },
        ColorScheme.DARK: {
            "primary": "#5dade2",
            "secondary": "#58d68d",
            "negative": "#ec7063",
            "positive": "#52be80",
            "neutral": "#aab7b8",
            "background": "#1a1a2e",
            "grid": "#2d2d44",
            "text": "#eaecee",
        },
        ColorScheme.LIGHT: {
            "primary": "#2980b9",
            "secondary": "#27ae60",
            "negative": "#c0392b",
            "positive": "#229954",
            "neutral": "#7f8c8d",
            "background": "#fdfefe",
            "grid": "#f4f6f7",
            "text": "#17202a",
        },
        ColorScheme.COLORBLIND: {
            "primary": "#0072B2",
            "secondary": "#009E73",
            "negative": "#D55E00",
            "positive": "#009E73",
            "neutral": "#999999",
            "background": "#ffffff",
            "grid": "#e5e5e5",
            "text": "#000000",
        },
    }

    @classmethod
    def get_palette(cls, scheme: ColorScheme) -> dict[str, str]:
        """Get color palette for scheme."""
        return cls.PALETTES.get(scheme, cls.PALETTES[ColorScheme.DEFAULT])


class SVGChartGenerator:
    """Generate charts as SVG."""

    def __init__(self, config: ChartConfig) -> None:
        """
        Initialize SVG chart generator.

        Args:
            config: Chart configuration
        """
        self.config = config
        self.palette = ColorPalette.get_palette(config.color_scheme)

    def generate_equity_curve(
        self,
        equity: pd.Series,
        benchmark: pd.Series | None = None,
        title: str = "Equity Curve",
    ) -> str:
        """
        Generate equity curve SVG.

        Args:
            equity: Equity time series
            benchmark: Optional benchmark series
            title: Chart title

        Returns:
            SVG string
        """
        width = self.config.width
        height = self.config.height
        margin = {"top": 50, "right": 50, "bottom": 50, "left": 80}

        inner_width = width - margin["left"] - margin["right"]
        inner_height = height - margin["top"] - margin["bottom"]

        normalized_equity = equity / equity.iloc[0] * 100
        y_min = float(normalized_equity.min()) * 0.95
        y_max = float(normalized_equity.max()) * 1.05

        if benchmark is not None:
            normalized_benchmark = benchmark / benchmark.iloc[0] * 100
            y_min = min(y_min, float(normalized_benchmark.min()) * 0.95)
            y_max = max(y_max, float(normalized_benchmark.max()) * 1.05)

        def scale_x(idx: int) -> float:
            return margin["left"] + (idx / (len(equity) - 1)) * inner_width

        def scale_y(val: float) -> float:
            return margin["top"] + inner_height - ((val - y_min) / (y_max - y_min)) * inner_height

        equity_points = " ".join(
            f"{scale_x(i):.1f},{scale_y(float(v)):.1f}"
            for i, v in enumerate(normalized_equity)
        )

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
            f'<rect width="{width}" height="{height}" fill="{self.palette["background"]}"/>',
        ]

        if self.config.show_grid:
            for i in range(5):
                y = margin["top"] + (i / 4) * inner_height
                svg_parts.append(
                    f'<line x1="{margin["left"]}" y1="{y:.1f}" '
                    f'x2="{width - margin["right"]}" y2="{y:.1f}" '
                    f'stroke="{self.palette["grid"]}" stroke-width="1"/>'
                )

        svg_parts.append(
            f'<polyline points="{equity_points}" fill="none" '
            f'stroke="{self.palette["primary"]}" stroke-width="2"/>'
        )

        if benchmark is not None:
            benchmark_points = " ".join(
                f"{scale_x(i):.1f},{scale_y(float(v)):.1f}"
                for i, v in enumerate(normalized_benchmark)
            )
            svg_parts.append(
                f'<polyline points="{benchmark_points}" fill="none" '
                f'stroke="{self.palette["neutral"]}" stroke-width="1.5" stroke-dasharray="5,5"/>'
            )

        svg_parts.append(
            f'<text x="{width / 2}" y="{margin["top"] / 2}" '
            f'text-anchor="middle" fill="{self.palette["text"]}" '
            f'font-size="{self.config.title_font_size}">{title}</text>'
        )

        svg_parts.append(
            f'<text x="{margin["left"] - 10}" y="{margin["top"]}" '
            f'text-anchor="end" fill="{self.palette["text"]}" '
            f'font-size="{self.config.label_font_size}">{y_max:.0f}</text>'
        )
        svg_parts.append(
            f'<text x="{margin["left"] - 10}" y="{height - margin["bottom"]}" '
            f'text-anchor="end" fill="{self.palette["text"]}" '
            f'font-size="{self.config.label_font_size}">{y_min:.0f}</text>'
        )

        svg_parts.append("</svg>")

        return "\n".join(svg_parts)

    def generate_drawdown_chart(
        self,
        drawdown: pd.Series,
        title: str = "Drawdown",
    ) -> str:
        """
        Generate drawdown chart SVG.

        Args:
            drawdown: Drawdown time series
            title: Chart title

        Returns:
            SVG string
        """
        width = self.config.width
        height = self.config.height
        margin = {"top": 50, "right": 50, "bottom": 50, "left": 80}

        inner_width = width - margin["left"] - margin["right"]
        inner_height = height - margin["top"] - margin["bottom"]

        y_min = float(drawdown.min()) * 1.1
        y_max = 0.0

        def scale_x(idx: int) -> float:
            return margin["left"] + (idx / (len(drawdown) - 1)) * inner_width

        def scale_y(val: float) -> float:
            return margin["top"] + ((0 - val) / (0 - y_min)) * inner_height

        area_points = f"{margin['left']},{margin['top']} "
        area_points += " ".join(
            f"{scale_x(i):.1f},{scale_y(float(v)):.1f}"
            for i, v in enumerate(drawdown)
        )
        area_points += f" {width - margin['right']},{margin['top']}"

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
            f'<rect width="{width}" height="{height}" fill="{self.palette["background"]}"/>',
        ]

        svg_parts.append(
            f'<polygon points="{area_points}" '
            f'fill="{self.palette["negative"]}" fill-opacity="0.3"/>'
        )

        line_points = " ".join(
            f"{scale_x(i):.1f},{scale_y(float(v)):.1f}"
            for i, v in enumerate(drawdown)
        )
        svg_parts.append(
            f'<polyline points="{line_points}" fill="none" '
            f'stroke="{self.palette["negative"]}" stroke-width="1.5"/>'
        )

        svg_parts.append(
            f'<line x1="{margin["left"]}" y1="{margin["top"]}" '
            f'x2="{width - margin["right"]}" y2="{margin["top"]}" '
            f'stroke="{self.palette["neutral"]}" stroke-width="1"/>'
        )

        svg_parts.append(
            f'<text x="{width / 2}" y="{margin["top"] / 2}" '
            f'text-anchor="middle" fill="{self.palette["text"]}" '
            f'font-size="{self.config.title_font_size}">{title}</text>'
        )

        svg_parts.append(
            f'<text x="{margin["left"] - 10}" y="{margin["top"]}" '
            f'text-anchor="end" fill="{self.palette["text"]}" '
            f'font-size="{self.config.label_font_size}">0%</text>'
        )
        svg_parts.append(
            f'<text x="{margin["left"] - 10}" y="{height - margin["bottom"]}" '
            f'text-anchor="end" fill="{self.palette["text"]}" '
            f'font-size="{self.config.label_font_size}">{y_min:.0%}</text>'
        )

        svg_parts.append("</svg>")

        return "\n".join(svg_parts)

    def generate_returns_histogram(
        self,
        returns: pd.Series,
        bins: int = 50,
        title: str = "Returns Distribution",
    ) -> str:
        """
        Generate returns histogram SVG.

        Args:
            returns: Return series
            bins: Number of histogram bins
            title: Chart title

        Returns:
            SVG string
        """
        width = self.config.width
        height = self.config.height
        margin = {"top": 50, "right": 50, "bottom": 50, "left": 80}

        inner_width = width - margin["left"] - margin["right"]
        inner_height = height - margin["top"] - margin["bottom"]

        hist, bin_edges = np.histogram(returns.dropna(), bins=bins)
        max_count = max(hist)

        bar_width = inner_width / bins

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
            f'<rect width="{width}" height="{height}" fill="{self.palette["background"]}"/>',
        ]

        for i, count in enumerate(hist):
            bar_height = (count / max_count) * inner_height if max_count > 0 else 0
            x = margin["left"] + i * bar_width
            y = margin["top"] + inner_height - bar_height

            bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
            color = self.palette["positive"] if bin_center >= 0 else self.palette["negative"]

            svg_parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 1:.1f}" '
                f'height="{bar_height:.1f}" fill="{color}" opacity="0.8"/>'
            )

        zero_x = margin["left"] + (-bin_edges[0] / (bin_edges[-1] - bin_edges[0])) * inner_width
        if margin["left"] < zero_x < width - margin["right"]:
            svg_parts.append(
                f'<line x1="{zero_x:.1f}" y1="{margin["top"]}" '
                f'x2="{zero_x:.1f}" y2="{height - margin["bottom"]}" '
                f'stroke="{self.palette["text"]}" stroke-width="2" stroke-dasharray="5,5"/>'
            )

        svg_parts.append(
            f'<text x="{width / 2}" y="{margin["top"] / 2}" '
            f'text-anchor="middle" fill="{self.palette["text"]}" '
            f'font-size="{self.config.title_font_size}">{title}</text>'
        )

        svg_parts.append("</svg>")

        return "\n".join(svg_parts)

    def generate_monthly_heatmap(
        self,
        monthly_returns: pd.DataFrame,
        title: str = "Monthly Returns Heatmap",
    ) -> str:
        """
        Generate monthly returns heatmap SVG.

        Args:
            monthly_returns: DataFrame with years as index, months as columns
            title: Chart title

        Returns:
            SVG string
        """
        width = self.config.width
        height = max(400, len(monthly_returns) * 40 + 100)
        margin = {"top": 60, "right": 50, "bottom": 30, "left": 80}

        inner_width = width - margin["left"] - margin["right"]
        inner_height = height - margin["top"] - margin["bottom"]

        years = list(monthly_returns.index)
        months = list(range(1, 13))

        cell_width = inner_width / 12
        cell_height = inner_height / len(years) if years else inner_height

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
            f'<rect width="{width}" height="{height}" fill="{self.palette["background"]}"/>',
        ]

        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        for i, name in enumerate(month_names):
            x = margin["left"] + i * cell_width + cell_width / 2
            svg_parts.append(
                f'<text x="{x:.1f}" y="{margin["top"] - 10}" '
                f'text-anchor="middle" fill="{self.palette["text"]}" '
                f'font-size="{self.config.label_font_size - 2}">{name}</text>'
            )

        for row_idx, year in enumerate(years):
            y = margin["top"] + row_idx * cell_height

            svg_parts.append(
                f'<text x="{margin["left"] - 10}" y="{y + cell_height / 2 + 4}" '
                f'text-anchor="end" fill="{self.palette["text"]}" '
                f'font-size="{self.config.label_font_size}">{year}</text>'
            )

            for col_idx, month in enumerate(months):
                x = margin["left"] + col_idx * cell_width

                if month in monthly_returns.columns:
                    value = monthly_returns.loc[year, month]
                    if pd.notna(value):
                        intensity = min(1.0, abs(value) / 0.10)

                        if value >= 0:
                            r, g, b = 39, 174, 96
                        else:
                            r, g, b = 231, 76, 60

                        r = int(255 - (255 - r) * intensity)
                        g = int(255 - (255 - g) * intensity)
                        b = int(255 - (255 - b) * intensity)

                        color = f"rgb({r},{g},{b})"
                        text_color = self.palette["text"] if intensity < 0.5 else "#ffffff"

                        svg_parts.append(
                            f'<rect x="{x:.1f}" y="{y:.1f}" '
                            f'width="{cell_width - 2:.1f}" height="{cell_height - 2:.1f}" '
                            f'fill="{color}" rx="2"/>'
                        )

                        svg_parts.append(
                            f'<text x="{x + cell_width / 2:.1f}" y="{y + cell_height / 2 + 4:.1f}" '
                            f'text-anchor="middle" fill="{text_color}" '
                            f'font-size="{self.config.label_font_size - 2}">{value:.1%}</text>'
                        )

        svg_parts.append(
            f'<text x="{width / 2}" y="25" '
            f'text-anchor="middle" fill="{self.palette["text"]}" '
            f'font-size="{self.config.title_font_size}">{title}</text>'
        )

        svg_parts.append("</svg>")

        return "\n".join(svg_parts)


class ChartVisualizer:
    """Main visualization class."""

    def __init__(
        self,
        config: VisualizationConfig | None = None,
    ) -> None:
        """
        Initialize chart visualizer.

        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        self.svg_generator = SVGChartGenerator(self.config.chart_config)

        if self.config.save_charts:
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"ChartVisualizer initialized with output dir: {self.config.output_dir}")

    def generate_equity_chart(
        self,
        equity: pd.Series,
        benchmark: pd.Series | None = None,
        title: str = "Equity Curve",
        filename: str | None = None,
    ) -> ChartResult:
        """
        Generate equity curve chart.

        Args:
            equity: Equity time series
            benchmark: Optional benchmark
            title: Chart title
            filename: Output filename

        Returns:
            Chart result
        """
        try:
            svg_content = self.svg_generator.generate_equity_curve(
                equity, benchmark, title
            )

            file_path = None
            if self.config.save_charts and filename:
                file_path = str(Path(self.config.output_dir) / f"{filename}.svg")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(svg_content)

            return ChartResult(
                chart_type=ChartType.EQUITY_CURVE,
                file_path=file_path,
                svg_content=svg_content,
                success=True,
            )

        except Exception as e:
            logger.error(f"Error generating equity chart: {e}")
            return ChartResult(
                chart_type=ChartType.EQUITY_CURVE,
                success=False,
                error_message=str(e),
            )

    def generate_drawdown_chart(
        self,
        equity: pd.Series,
        title: str = "Drawdown",
        filename: str | None = None,
    ) -> ChartResult:
        """
        Generate drawdown chart.

        Args:
            equity: Equity time series
            title: Chart title
            filename: Output filename

        Returns:
            Chart result
        """
        try:
            running_max = equity.expanding().max()
            drawdown = (equity - running_max) / running_max

            svg_content = self.svg_generator.generate_drawdown_chart(
                drawdown, title
            )

            file_path = None
            if self.config.save_charts and filename:
                file_path = str(Path(self.config.output_dir) / f"{filename}.svg")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(svg_content)

            return ChartResult(
                chart_type=ChartType.DRAWDOWN,
                file_path=file_path,
                svg_content=svg_content,
                success=True,
            )

        except Exception as e:
            logger.error(f"Error generating drawdown chart: {e}")
            return ChartResult(
                chart_type=ChartType.DRAWDOWN,
                success=False,
                error_message=str(e),
            )

    def generate_returns_histogram(
        self,
        returns: pd.Series,
        bins: int = 50,
        title: str = "Returns Distribution",
        filename: str | None = None,
    ) -> ChartResult:
        """
        Generate returns histogram.

        Args:
            returns: Return series
            bins: Number of bins
            title: Chart title
            filename: Output filename

        Returns:
            Chart result
        """
        try:
            svg_content = self.svg_generator.generate_returns_histogram(
                returns, bins, title
            )

            file_path = None
            if self.config.save_charts and filename:
                file_path = str(Path(self.config.output_dir) / f"{filename}.svg")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(svg_content)

            return ChartResult(
                chart_type=ChartType.RETURNS_DISTRIBUTION,
                file_path=file_path,
                svg_content=svg_content,
                success=True,
            )

        except Exception as e:
            logger.error(f"Error generating returns histogram: {e}")
            return ChartResult(
                chart_type=ChartType.RETURNS_DISTRIBUTION,
                success=False,
                error_message=str(e),
            )

    def generate_monthly_heatmap(
        self,
        returns: pd.Series,
        title: str = "Monthly Returns",
        filename: str | None = None,
    ) -> ChartResult:
        """
        Generate monthly returns heatmap.

        Args:
            returns: Daily return series
            title: Chart title
            filename: Output filename

        Returns:
            Chart result
        """
        try:
            monthly = returns.resample("M").apply(lambda x: (1 + x).prod() - 1)

            monthly_df = pd.DataFrame({
                "year": monthly.index.year,
                "month": monthly.index.month,
                "return": monthly.values,
            })

            pivot = monthly_df.pivot(index="year", columns="month", values="return")

            svg_content = self.svg_generator.generate_monthly_heatmap(
                pivot, title
            )

            file_path = None
            if self.config.save_charts and filename:
                file_path = str(Path(self.config.output_dir) / f"{filename}.svg")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(svg_content)

            return ChartResult(
                chart_type=ChartType.MONTHLY_HEATMAP,
                file_path=file_path,
                svg_content=svg_content,
                success=True,
            )

        except Exception as e:
            logger.error(f"Error generating monthly heatmap: {e}")
            return ChartResult(
                chart_type=ChartType.MONTHLY_HEATMAP,
                success=False,
                error_message=str(e),
            )

    def generate_all_charts(
        self,
        equity: pd.Series,
        returns: pd.Series,
        benchmark: pd.Series | None = None,
        prefix: str = "backtest",
    ) -> list[ChartResult]:
        """
        Generate all standard charts.

        Args:
            equity: Equity time series
            returns: Return series
            benchmark: Optional benchmark
            prefix: Filename prefix

        Returns:
            List of chart results
        """
        results = []

        results.append(
            self.generate_equity_chart(
                equity, benchmark, "Equity Curve", f"{prefix}_equity"
            )
        )

        results.append(
            self.generate_drawdown_chart(
                equity, "Drawdown Analysis", f"{prefix}_drawdown"
            )
        )

        results.append(
            self.generate_returns_histogram(
                returns, 50, "Returns Distribution", f"{prefix}_returns_hist"
            )
        )

        results.append(
            self.generate_monthly_heatmap(
                returns, "Monthly Returns", f"{prefix}_monthly"
            )
        )

        success_count = sum(1 for r in results if r.success)
        logger.info(f"Generated {success_count}/{len(results)} charts successfully")

        return results


def create_visualizer(
    output_dir: str = "./charts",
    color_scheme: str = "default",
    config: dict | None = None,
) -> ChartVisualizer:
    """
    Create a chart visualizer.

    Args:
        output_dir: Output directory
        color_scheme: Color scheme name
        config: Additional configuration

    Returns:
        Configured ChartVisualizer
    """
    chart_config = ChartConfig(
        color_scheme=ColorScheme(color_scheme),
        **(config or {}),
    )

    vis_config = VisualizationConfig(
        output_dir=output_dir,
        chart_config=chart_config,
    )

    return ChartVisualizer(config=vis_config)
