"""
Prompt Manager Module for Ultimate Trading Bot v2.2.

This module manages AI prompt templates and generation
for various trading analysis tasks.
"""

import logging
from datetime import datetime
from enum import Enum
from string import Template
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.utils.helpers import generate_uuid
from src.utils.date_utils import now_utc, format_datetime


logger = logging.getLogger(__name__)


class PromptCategory(str, Enum):
    """Prompt category enumeration."""

    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    NEWS = "news"
    SIGNAL = "signal"
    RISK = "risk"
    MARKET = "market"
    STRATEGY = "strategy"
    GENERAL = "general"


class PromptTemplate(BaseModel):
    """Prompt template model."""

    template_id: str = Field(default_factory=generate_uuid)
    name: str
    category: PromptCategory
    template: str
    system_prompt: Optional[str] = None
    description: str = Field(default="")
    variables: list[str] = Field(default_factory=list)
    version: str = Field(default="1.0")
    created_at: datetime = Field(default_factory=now_utc)

    def render(self, **kwargs: Any) -> str:
        """Render template with variables."""
        try:
            return Template(self.template).safe_substitute(**kwargs)
        except Exception as e:
            logger.error(f"Error rendering template: {e}")
            return self.template


SYSTEM_PROMPTS = {
    "trading_analyst": """You are an expert financial analyst and trading advisor with deep knowledge of:
- Technical analysis indicators (RSI, MACD, Bollinger Bands, etc.)
- Fundamental analysis (earnings, P/E ratios, revenue growth)
- Market sentiment analysis
- Risk management strategies
- Portfolio optimization

Provide clear, actionable insights based on the data provided. Be objective and data-driven.
Always consider risk factors and market conditions in your analysis.""",

    "sentiment_analyzer": """You are a sentiment analysis expert specializing in financial markets.
Your task is to analyze text and determine market sentiment.
Provide sentiment scores from -1.0 (very bearish) to 1.0 (very bullish).
Consider context, tone, and market implications in your analysis.
Be precise and consistent in your scoring methodology.""",

    "signal_generator": """You are a quantitative trading signal generator.
Analyze the provided market data and technical indicators to generate trading signals.
Consider multiple timeframes and confirmation signals.
Provide clear entry/exit points and confidence levels.
Always include risk assessment and stop-loss recommendations.""",

    "risk_analyst": """You are a risk management specialist for trading portfolios.
Analyze positions, market conditions, and potential risks.
Provide specific risk metrics and recommendations.
Consider correlation, volatility, and tail risks.
Suggest position sizing and hedging strategies when appropriate.""",

    "market_commentator": """You are a financial market commentator providing insights on market conditions.
Analyze current market trends, sector movements, and economic factors.
Provide balanced perspectives considering bulls and bears arguments.
Highlight key levels, potential catalysts, and risks to watch.""",
}

DEFAULT_TEMPLATES = {
    "sentiment_analysis": PromptTemplate(
        name="sentiment_analysis",
        category=PromptCategory.SENTIMENT,
        template="""Analyze the sentiment of the following text related to ${symbol}:

Text: "${text}"

Provide a JSON response with:
{
    "sentiment": "bullish" | "bearish" | "neutral",
    "score": <float from -1.0 to 1.0>,
    "confidence": <float from 0.0 to 1.0>,
    "key_factors": [<list of key sentiment factors>],
    "summary": "<brief summary>"
}""",
        system_prompt=SYSTEM_PROMPTS["sentiment_analyzer"],
        description="Analyze sentiment of financial text",
        variables=["symbol", "text"],
    ),

    "news_sentiment": PromptTemplate(
        name="news_sentiment",
        category=PromptCategory.NEWS,
        template="""Analyze the following news headlines for ${symbol}:

Headlines:
${headlines}

Provide a JSON response with:
{
    "overall_sentiment": "bullish" | "bearish" | "neutral",
    "sentiment_score": <float from -1.0 to 1.0>,
    "headline_analysis": [
        {
            "headline": "<headline text>",
            "sentiment": "bullish" | "bearish" | "neutral",
            "impact": "high" | "medium" | "low"
        }
    ],
    "key_themes": [<list of key themes>],
    "trading_implication": "<how this news might affect the stock>"
}""",
        system_prompt=SYSTEM_PROMPTS["sentiment_analyzer"],
        description="Analyze sentiment of news headlines",
        variables=["symbol", "headlines"],
    ),

    "technical_analysis": PromptTemplate(
        name="technical_analysis",
        category=PromptCategory.TECHNICAL,
        template="""Perform technical analysis for ${symbol} based on the following data:

Current Price: $${price}
Daily Change: ${change}%
Volume: ${volume}

Technical Indicators:
${indicators}

Price Levels:
- 52-week High: $${high_52w}
- 52-week Low: $${low_52w}
- Support: $${support}
- Resistance: $${resistance}

Provide a JSON response with:
{
    "trend": "bullish" | "bearish" | "neutral",
    "trend_strength": <float from 0.0 to 1.0>,
    "key_signals": [<list of key technical signals>],
    "support_levels": [<price levels>],
    "resistance_levels": [<price levels>],
    "recommendation": "strong_buy" | "buy" | "hold" | "sell" | "strong_sell",
    "confidence": <float from 0.0 to 1.0>,
    "analysis_summary": "<detailed analysis>"
}""",
        system_prompt=SYSTEM_PROMPTS["trading_analyst"],
        description="Comprehensive technical analysis",
        variables=["symbol", "price", "change", "volume", "indicators", "high_52w", "low_52w", "support", "resistance"],
    ),

    "trading_signal": PromptTemplate(
        name="trading_signal",
        category=PromptCategory.SIGNAL,
        template="""Generate a trading signal for ${symbol} based on:

Market Data:
- Current Price: $${price}
- Bid: $${bid} / Ask: $${ask}
- Volume: ${volume}

Technical Indicators:
${indicators}

Recent Price Action:
${price_action}

Market Context:
${market_context}

Provide a JSON response with:
{
    "signal": "buy" | "sell" | "hold",
    "strength": <float from 0.0 to 1.0>,
    "entry_price": <recommended entry price>,
    "stop_loss": <stop loss price>,
    "take_profit": [<take profit levels>],
    "risk_reward_ratio": <ratio>,
    "timeframe": "<expected holding period>",
    "reasoning": [<list of reasons for signal>],
    "risks": [<list of potential risks>],
    "confidence": <float from 0.0 to 1.0>
}""",
        system_prompt=SYSTEM_PROMPTS["signal_generator"],
        description="Generate trading signal with entry/exit points",
        variables=["symbol", "price", "bid", "ask", "volume", "indicators", "price_action", "market_context"],
    ),

    "risk_assessment": PromptTemplate(
        name="risk_assessment",
        category=PromptCategory.RISK,
        template="""Assess the risk for the following position:

Symbol: ${symbol}
Position Size: ${quantity} shares
Entry Price: $${entry_price}
Current Price: $${current_price}
P&L: $${pnl} (${pnl_pct}%)

Portfolio Context:
- Total Portfolio Value: $${portfolio_value}
- Position Weight: ${position_weight}%
- Other Positions: ${other_positions}

Market Conditions:
${market_conditions}

Provide a JSON response with:
{
    "risk_level": "low" | "medium" | "high" | "critical",
    "risk_score": <float from 0.0 to 1.0>,
    "risk_factors": [
        {
            "factor": "<risk factor>",
            "severity": "low" | "medium" | "high",
            "description": "<explanation>"
        }
    ],
    "recommendations": [<list of risk mitigation actions>],
    "suggested_stop_loss": <price>,
    "max_position_size": <shares>,
    "correlation_risk": "<correlation with other positions>",
    "summary": "<risk assessment summary>"
}""",
        system_prompt=SYSTEM_PROMPTS["risk_analyst"],
        description="Comprehensive risk assessment for positions",
        variables=["symbol", "quantity", "entry_price", "current_price", "pnl", "pnl_pct", "portfolio_value", "position_weight", "other_positions", "market_conditions"],
    ),

    "market_overview": PromptTemplate(
        name="market_overview",
        category=PromptCategory.MARKET,
        template="""Provide a market overview based on the following data:

Major Indices:
${indices}

Sector Performance:
${sectors}

Market Breadth:
- Advancing: ${advancing}
- Declining: ${declining}
- Unchanged: ${unchanged}

VIX: ${vix}
Date: ${date}

Recent News/Events:
${news}

Provide a JSON response with:
{
    "market_sentiment": "bullish" | "bearish" | "neutral",
    "risk_level": "low" | "medium" | "high",
    "key_observations": [<list of key market observations>],
    "sector_leaders": [<top performing sectors>],
    "sector_laggards": [<worst performing sectors>],
    "opportunities": [<potential trading opportunities>],
    "risks_to_watch": [<market risks>],
    "outlook": "<short-term market outlook>",
    "trading_recommendations": "<general trading approach>"
}""",
        system_prompt=SYSTEM_PROMPTS["market_commentator"],
        description="Generate market overview and outlook",
        variables=["indices", "sectors", "advancing", "declining", "unchanged", "vix", "date", "news"],
    ),

    "strategy_evaluation": PromptTemplate(
        name="strategy_evaluation",
        category=PromptCategory.STRATEGY,
        template="""Evaluate the following trading strategy performance:

Strategy: ${strategy_name}
Period: ${period}

Performance Metrics:
- Total Return: ${total_return}%
- Win Rate: ${win_rate}%
- Profit Factor: ${profit_factor}
- Sharpe Ratio: ${sharpe_ratio}
- Max Drawdown: ${max_drawdown}%
- Total Trades: ${total_trades}

Recent Trades:
${recent_trades}

Market Conditions During Period:
${market_conditions}

Provide a JSON response with:
{
    "overall_rating": "excellent" | "good" | "average" | "poor",
    "score": <float from 0.0 to 10.0>,
    "strengths": [<list of strategy strengths>],
    "weaknesses": [<list of strategy weaknesses>],
    "improvement_suggestions": [<list of improvement suggestions>],
    "market_fit": "<how well strategy fits current market>",
    "risk_assessment": "<risk level of the strategy>",
    "recommendation": "<continue | modify | suspend strategy>"
}""",
        system_prompt=SYSTEM_PROMPTS["trading_analyst"],
        description="Evaluate trading strategy performance",
        variables=["strategy_name", "period", "total_return", "win_rate", "profit_factor", "sharpe_ratio", "max_drawdown", "total_trades", "recent_trades", "market_conditions"],
    ),
}


class PromptManager:
    """
    Manages prompt templates for AI analysis.

    Provides functionality for:
    - Template registration and retrieval
    - Prompt rendering with variables
    - System prompt management
    - Template versioning
    """

    def __init__(self) -> None:
        """Initialize PromptManager."""
        self._templates: dict[str, PromptTemplate] = {}
        self._system_prompts: dict[str, str] = SYSTEM_PROMPTS.copy()

        for name, template in DEFAULT_TEMPLATES.items():
            self._templates[name] = template

        logger.info(f"PromptManager initialized with {len(self._templates)} templates")

    @property
    def template_names(self) -> list[str]:
        """Get list of template names."""
        return list(self._templates.keys())

    @property
    def system_prompt_names(self) -> list[str]:
        """Get list of system prompt names."""
        return list(self._system_prompts.keys())

    def register_template(
        self,
        name: str,
        template: str,
        category: PromptCategory,
        system_prompt: Optional[str] = None,
        description: str = "",
        variables: Optional[list[str]] = None,
    ) -> PromptTemplate:
        """
        Register a new prompt template.

        Args:
            name: Template name
            template: Template string
            category: Template category
            system_prompt: System prompt to use
            description: Template description
            variables: Required variables

        Returns:
            Registered template
        """
        prompt_template = PromptTemplate(
            name=name,
            category=category,
            template=template,
            system_prompt=system_prompt,
            description=description,
            variables=variables or [],
        )

        self._templates[name] = prompt_template
        logger.info(f"Registered template: {name}")

        return prompt_template

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name."""
        return self._templates.get(name)

    def get_templates_by_category(
        self,
        category: PromptCategory
    ) -> list[PromptTemplate]:
        """Get all templates in a category."""
        return [
            t for t in self._templates.values()
            if t.category == category
        ]

    def render_prompt(
        self,
        template_name: str,
        **kwargs: Any
    ) -> tuple[str, Optional[str]]:
        """
        Render a prompt from template.

        Args:
            template_name: Template name
            **kwargs: Template variables

        Returns:
            Tuple of (rendered_prompt, system_prompt)
        """
        template = self._templates.get(template_name)

        if not template:
            logger.error(f"Template not found: {template_name}")
            return "", None

        rendered = template.render(**kwargs)
        return rendered, template.system_prompt

    def register_system_prompt(
        self,
        name: str,
        prompt: str,
    ) -> None:
        """Register a system prompt."""
        self._system_prompts[name] = prompt
        logger.info(f"Registered system prompt: {name}")

    def get_system_prompt(self, name: str) -> Optional[str]:
        """Get a system prompt by name."""
        return self._system_prompts.get(name)

    def format_indicators(
        self,
        indicators: dict[str, Any]
    ) -> str:
        """Format technical indicators for prompt."""
        lines = []
        for name, value in indicators.items():
            if isinstance(value, float):
                lines.append(f"- {name}: {value:.2f}")
            else:
                lines.append(f"- {name}: {value}")
        return "\n".join(lines)

    def format_price_action(
        self,
        bars: list[dict],
        limit: int = 5
    ) -> str:
        """Format recent price action for prompt."""
        lines = []
        for bar in bars[-limit:]:
            date = bar.get("timestamp", bar.get("date", ""))
            if hasattr(date, "strftime"):
                date = date.strftime("%Y-%m-%d")
            o = bar.get("open", 0)
            h = bar.get("high", 0)
            l = bar.get("low", 0)
            c = bar.get("close", 0)
            v = bar.get("volume", 0)
            lines.append(f"- {date}: O=${o:.2f} H=${h:.2f} L=${l:.2f} C=${c:.2f} V={v:,}")
        return "\n".join(lines)

    def format_headlines(
        self,
        headlines: list[str],
        limit: int = 10
    ) -> str:
        """Format news headlines for prompt."""
        return "\n".join([f"- {h}" for h in headlines[:limit]])

    def format_positions(
        self,
        positions: list[dict],
    ) -> str:
        """Format positions for prompt."""
        if not positions:
            return "No other positions"

        lines = []
        for pos in positions:
            symbol = pos.get("symbol", "")
            qty = pos.get("quantity", 0)
            pnl = pos.get("unrealized_pnl", 0)
            lines.append(f"- {symbol}: {qty} shares, P&L: ${pnl:.2f}")
        return "\n".join(lines)

    def get_statistics(self) -> dict:
        """Get manager statistics."""
        category_counts = {}
        for template in self._templates.values():
            cat = template.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1

        return {
            "total_templates": len(self._templates),
            "total_system_prompts": len(self._system_prompts),
            "templates_by_category": category_counts,
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"PromptManager(templates={len(self._templates)})"
