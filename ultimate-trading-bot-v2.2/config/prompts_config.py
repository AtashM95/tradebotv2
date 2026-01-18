"""
Prompts Configuration Module for Ultimate Trading Bot v2.2.

This module provides all prompt templates for AI interactions including
sentiment analysis, chart analysis, trading signals, and more.
"""

from typing import Any, Dict, List, Optional, Union
from enum import Enum
from dataclasses import dataclass, field
from string import Template
import logging

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class PromptCategory(str, Enum):
    """Prompt category enumeration."""
    SENTIMENT = "sentiment"
    CHART = "chart"
    NEWS = "news"
    TRADING = "trading"
    RISK = "risk"
    MARKET = "market"
    STRATEGY = "strategy"
    CHAT = "chat"


@dataclass
class PromptTemplate:
    """Template for AI prompts with variable substitution."""

    name: str
    category: PromptCategory
    system_prompt: str
    user_prompt_template: str
    description: str = ""
    required_variables: List[str] = field(default_factory=list)
    optional_variables: List[str] = field(default_factory=list)
    output_format: str = "text"
    example_response: str = ""

    def format_user_prompt(self, **kwargs: Any) -> str:
        """
        Format the user prompt with provided variables.

        Args:
            **kwargs: Variables to substitute in the template

        Returns:
            Formatted prompt string
        """
        # Check required variables
        missing = [var for var in self.required_variables if var not in kwargs]
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        # Use Template for safe substitution
        template = Template(self.user_prompt_template)
        return template.safe_substitute(**kwargs)


# =============================================================================
# SENTIMENT ANALYSIS PROMPTS
# =============================================================================

SENTIMENT_SYSTEM_PROMPT = """You are an expert financial sentiment analyst with deep knowledge of market psychology and news interpretation. Your task is to analyze text and determine the sentiment towards specific financial instruments or the market in general.

Guidelines:
1. Provide sentiment scores from -1.0 (extremely bearish) to 1.0 (extremely bullish)
2. Consider both explicit and implicit sentiment signals
3. Account for the source credibility and potential biases
4. Identify key phrases that influenced your assessment
5. Consider the temporal relevance of the information

Always respond in valid JSON format with the following structure:
{
    "sentiment_score": <float between -1.0 and 1.0>,
    "sentiment_label": "<very_bearish|bearish|neutral|bullish|very_bullish>",
    "confidence": <float between 0.0 and 1.0>,
    "key_factors": [<list of key influencing factors>],
    "affected_symbols": [<list of affected stock symbols>],
    "summary": "<brief summary of sentiment>"
}"""

SENTIMENT_NEWS_PROMPT = PromptTemplate(
    name="news_sentiment",
    category=PromptCategory.SENTIMENT,
    system_prompt=SENTIMENT_SYSTEM_PROMPT,
    user_prompt_template="""Analyze the sentiment of the following news article(s) regarding $symbol:

Title: $title
Source: $source
Published: $published_date

Content:
$content

Focus on:
1. Direct mentions of $symbol and its products/services
2. Industry-wide implications
3. Competitive landscape mentions
4. Financial metrics or outlook changes
5. Management commentary if present

Provide your sentiment analysis in the specified JSON format.""",
    description="Sentiment analysis for news articles",
    required_variables=["symbol", "title", "content"],
    optional_variables=["source", "published_date"],
    output_format="json"
)

SENTIMENT_SOCIAL_PROMPT = PromptTemplate(
    name="social_sentiment",
    category=PromptCategory.SENTIMENT,
    system_prompt=SENTIMENT_SYSTEM_PROMPT,
    user_prompt_template="""Analyze the sentiment from the following social media posts/discussions about $symbol:

Posts:
$posts

Consider:
1. Overall crowd sentiment
2. Presence of institutional vs retail perspectives
3. Any specific catalysts mentioned
4. Quality and credibility of sources
5. Potential manipulation or coordinated activity

Provide sentiment analysis in JSON format.""",
    description="Sentiment analysis for social media",
    required_variables=["symbol", "posts"],
    output_format="json"
)

SENTIMENT_AGGREGATE_PROMPT = PromptTemplate(
    name="aggregate_sentiment",
    category=PromptCategory.SENTIMENT,
    system_prompt=SENTIMENT_SYSTEM_PROMPT,
    user_prompt_template="""Aggregate and analyze the overall sentiment for $symbol based on multiple sources:

News Sentiment: $news_sentiment
Social Sentiment: $social_sentiment
Analyst Ratings: $analyst_ratings
Recent Price Action: $price_action

Provide a comprehensive sentiment assessment considering:
1. Weighted importance of each source
2. Consistency across sources
3. Any divergence that might signal opportunities
4. Time-sensitive factors

Output in JSON format with an overall recommendation.""",
    description="Aggregate sentiment from multiple sources",
    required_variables=["symbol"],
    optional_variables=["news_sentiment", "social_sentiment", "analyst_ratings", "price_action"],
    output_format="json"
)


# =============================================================================
# CHART ANALYSIS PROMPTS
# =============================================================================

CHART_SYSTEM_PROMPT = """You are an expert technical analyst with extensive experience in chart pattern recognition, support/resistance identification, and price action analysis. Your task is to analyze chart images and provide actionable trading insights.

Guidelines:
1. Identify the primary trend direction (bullish, bearish, or sideways)
2. Recognize chart patterns (head and shoulders, triangles, flags, etc.)
3. Identify key support and resistance levels
4. Analyze volume patterns if visible
5. Consider multiple timeframes if shown
6. Provide specific price levels for entries, stops, and targets

Always respond in valid JSON format with the following structure:
{
    "trend": "<bullish|bearish|sideways>",
    "trend_strength": "<weak|moderate|strong>",
    "patterns": [{"pattern": "<name>", "completion": <percent>, "target": <price>}],
    "support_levels": [<list of support prices>],
    "resistance_levels": [<list of resistance prices>],
    "indicators": {"<indicator>": "<reading>"},
    "signal": "<buy|sell|hold>",
    "entry_price": <price or null>,
    "stop_loss": <price or null>,
    "take_profit": [<list of target prices>],
    "confidence": <float 0-1>,
    "analysis": "<detailed text analysis>"
}"""

CHART_ANALYSIS_PROMPT = PromptTemplate(
    name="chart_analysis",
    category=PromptCategory.CHART,
    system_prompt=CHART_SYSTEM_PROMPT,
    user_prompt_template="""Analyze the following chart for $symbol:

[Chart Image]

Timeframe: $timeframe
Current Price: $current_price
Additional Context: $context

Provide comprehensive technical analysis including:
1. Trend identification and strength
2. Key pattern recognition
3. Support and resistance levels
4. Trading signal with specific price levels
5. Risk/reward assessment""",
    description="Technical analysis of chart images",
    required_variables=["symbol"],
    optional_variables=["timeframe", "current_price", "context"],
    output_format="json"
)

CHART_PATTERN_PROMPT = PromptTemplate(
    name="pattern_recognition",
    category=PromptCategory.CHART,
    system_prompt=CHART_SYSTEM_PROMPT,
    user_prompt_template="""Identify and analyze chart patterns for $symbol in the provided image:

[Chart Image]

Focus specifically on:
1. Classic chart patterns (H&S, double top/bottom, triangles, etc.)
2. Candlestick patterns (if visible)
3. Pattern completion percentage
4. Expected price targets based on pattern measurements
5. Pattern failure scenarios

Provide pattern analysis in JSON format.""",
    description="Pattern recognition in charts",
    required_variables=["symbol"],
    output_format="json"
)


# =============================================================================
# TRADING SIGNAL PROMPTS
# =============================================================================

TRADING_SYSTEM_PROMPT = """You are an experienced quantitative trader and analyst. Your task is to evaluate market data and generate trading signals with specific entry, exit, and risk management levels.

Guidelines:
1. Base decisions on data provided, not assumptions
2. Always include stop-loss and take-profit levels
3. Consider risk/reward ratios (aim for minimum 2:1)
4. Account for market conditions and volatility
5. Provide clear reasoning for each signal
6. Express confidence levels honestly

Always respond in valid JSON format with the following structure:
{
    "signal": "<strong_buy|buy|hold|sell|strong_sell|no_signal>",
    "direction": "<long|short|none>",
    "entry_price": <price>,
    "stop_loss": <price>,
    "take_profit_1": <price>,
    "take_profit_2": <price>,
    "take_profit_3": <price>,
    "position_size_percent": <recommended size 1-10>,
    "risk_reward_ratio": <float>,
    "confidence": <float 0-1>,
    "timeframe": "<expected holding period>",
    "key_levels": {"support": [<prices>], "resistance": [<prices>]},
    "reasoning": "<detailed explanation>",
    "risks": [<list of risk factors>],
    "catalysts": [<list of potential catalysts>]
}"""

TRADING_SIGNAL_PROMPT = PromptTemplate(
    name="trading_signal",
    category=PromptCategory.TRADING,
    system_prompt=TRADING_SYSTEM_PROMPT,
    user_prompt_template="""Generate a trading signal for $symbol based on the following data:

Current Price: $current_price
Technical Indicators:
$technical_indicators

Recent Price Action:
$price_action

Sentiment Analysis:
$sentiment

Market Conditions:
$market_conditions

Risk Parameters:
- Max Risk Per Trade: $max_risk%
- Account Size: $account_size

Provide a comprehensive trading signal with all required levels.""",
    description="Generate trading signals",
    required_variables=["symbol", "current_price"],
    optional_variables=["technical_indicators", "price_action", "sentiment", "market_conditions", "max_risk", "account_size"],
    output_format="json"
)

TRADING_ENTRY_PROMPT = PromptTemplate(
    name="entry_analysis",
    category=PromptCategory.TRADING,
    system_prompt=TRADING_SYSTEM_PROMPT,
    user_prompt_template="""Analyze the entry opportunity for $symbol:

Current Price: $current_price
Proposed Entry: $entry_price
Direction: $direction

Technical Context:
$technical_context

Evaluate:
1. Is this an optimal entry point?
2. Should we wait for a better price?
3. What are the immediate support/resistance levels?
4. What's the ideal stop-loss placement?
5. What are realistic profit targets?

Provide entry analysis in JSON format.""",
    description="Entry point analysis",
    required_variables=["symbol", "current_price", "entry_price", "direction"],
    optional_variables=["technical_context"],
    output_format="json"
)


# =============================================================================
# RISK ASSESSMENT PROMPTS
# =============================================================================

RISK_SYSTEM_PROMPT = """You are a risk management expert specializing in trading and portfolio risk. Your task is to evaluate trading opportunities and portfolio positions for potential risks and provide mitigation strategies.

Guidelines:
1. Identify all relevant risk factors
2. Quantify risks where possible
3. Consider correlation and concentration risks
4. Evaluate worst-case scenarios
5. Provide specific risk mitigation recommendations
6. Consider both systematic and idiosyncratic risks

Always respond in valid JSON format with the following structure:
{
    "overall_risk_level": "<very_low|low|medium|high|very_high>",
    "risk_score": <float 0-10>,
    "risks": [
        {
            "type": "<risk type>",
            "severity": "<low|medium|high|critical>",
            "probability": <float 0-1>,
            "impact": "<description>",
            "mitigation": "<recommended action>"
        }
    ],
    "position_recommendation": "<proceed|reduce_size|avoid|close>",
    "suggested_adjustments": [<list of suggestions>],
    "max_recommended_size_percent": <float>,
    "warnings": [<critical warnings>],
    "summary": "<brief risk summary>"
}"""

RISK_ASSESSMENT_PROMPT = PromptTemplate(
    name="risk_assessment",
    category=PromptCategory.RISK,
    system_prompt=RISK_SYSTEM_PROMPT,
    user_prompt_template="""Assess the risks for the following trading opportunity:

Symbol: $symbol
Direction: $direction
Entry Price: $entry_price
Position Size: $position_size%
Stop Loss: $stop_loss

Current Portfolio:
$portfolio_positions

Market Conditions:
$market_conditions

Historical Volatility: $volatility
Beta: $beta
Sector: $sector

Evaluate all risk factors and provide recommendations.""",
    description="Comprehensive risk assessment",
    required_variables=["symbol", "direction", "entry_price"],
    optional_variables=["position_size", "stop_loss", "portfolio_positions", "market_conditions", "volatility", "beta", "sector"],
    output_format="json"
)

PORTFOLIO_RISK_PROMPT = PromptTemplate(
    name="portfolio_risk",
    category=PromptCategory.RISK,
    system_prompt=RISK_SYSTEM_PROMPT,
    user_prompt_template="""Analyze the risk profile of the following portfolio:

Positions:
$positions

Portfolio Metrics:
- Total Value: $total_value
- Cash: $cash_percent%
- Long Exposure: $long_exposure%
- Short Exposure: $short_exposure%

Correlation Matrix:
$correlation_matrix

Evaluate:
1. Concentration risk
2. Sector exposure
3. Correlation risks
4. Tail risks
5. Liquidity risks
6. Overall portfolio VaR estimate

Provide portfolio risk analysis in JSON format.""",
    description="Portfolio-level risk analysis",
    required_variables=["positions", "total_value"],
    optional_variables=["cash_percent", "long_exposure", "short_exposure", "correlation_matrix"],
    output_format="json"
)


# =============================================================================
# MARKET ANALYSIS PROMPTS
# =============================================================================

MARKET_SYSTEM_PROMPT = """You are a market strategist with expertise in macroeconomic analysis, sector rotation, and market regime identification. Your task is to provide comprehensive market analysis and outlook.

Guidelines:
1. Analyze current market conditions objectively
2. Identify key market drivers and themes
3. Assess market regime (trending, ranging, volatile)
4. Consider intermarket relationships
5. Provide sector-level insights
6. Balance short-term and long-term perspectives

Always respond in valid JSON format with the following structure:
{
    "market_regime": "<bull|bear|neutral|volatile>",
    "trend_direction": "<up|down|sideways>",
    "volatility_level": "<low|medium|high|extreme>",
    "key_themes": [<list of current market themes>],
    "sector_outlook": {
        "<sector>": {"rating": "<overweight|neutral|underweight>", "reason": "<explanation>"}
    },
    "risk_factors": [<current risk factors>],
    "opportunities": [<identified opportunities>],
    "key_levels": {"SPY": {"support": <price>, "resistance": <price>}},
    "outlook": {
        "short_term": "<1-2 weeks outlook>",
        "medium_term": "<1-3 months outlook>"
    },
    "recommended_positioning": "<description>",
    "confidence": <float 0-1>
}"""

MARKET_OVERVIEW_PROMPT = PromptTemplate(
    name="market_overview",
    category=PromptCategory.MARKET,
    system_prompt=MARKET_SYSTEM_PROMPT,
    user_prompt_template="""Provide a comprehensive market overview based on the following data:

Major Indices:
$indices_data

Sector Performance:
$sector_performance

Market Breadth:
- Advancing: $advancing
- Declining: $declining
- New Highs: $new_highs
- New Lows: $new_lows

Volatility (VIX): $vix

Recent Economic Data:
$economic_data

Provide market analysis and outlook in JSON format.""",
    description="Comprehensive market overview",
    required_variables=[],
    optional_variables=["indices_data", "sector_performance", "advancing", "declining", "new_highs", "new_lows", "vix", "economic_data"],
    output_format="json"
)

SECTOR_ANALYSIS_PROMPT = PromptTemplate(
    name="sector_analysis",
    category=PromptCategory.MARKET,
    system_prompt=MARKET_SYSTEM_PROMPT,
    user_prompt_template="""Analyze the $sector sector:

Sector ETF Performance:
$sector_etf_data

Top Holdings Performance:
$top_holdings

Sector News:
$sector_news

Technical Indicators:
$technicals

Relative Strength vs SPY: $relative_strength

Provide sector analysis and stock recommendations in JSON format.""",
    description="Sector-specific analysis",
    required_variables=["sector"],
    optional_variables=["sector_etf_data", "top_holdings", "sector_news", "technicals", "relative_strength"],
    output_format="json"
)


# =============================================================================
# STRATEGY ADVICE PROMPTS
# =============================================================================

STRATEGY_SYSTEM_PROMPT = """You are a trading strategy consultant with expertise in systematic trading, risk management, and portfolio construction. Your task is to provide strategic advice tailored to the trader's goals and constraints.

Guidelines:
1. Consider the trader's risk tolerance and goals
2. Provide actionable, specific recommendations
3. Balance risk and reward appropriately
4. Consider current market conditions
5. Suggest concrete next steps
6. Be realistic about expected outcomes

Always respond in valid JSON format with the following structure:
{
    "strategy_recommendation": "<primary strategy recommendation>",
    "risk_adjusted_allocation": {
        "<strategy/asset>": <percent allocation>
    },
    "position_sizing_advice": "<sizing recommendation>",
    "timing_advice": "<when to enter/exit>",
    "risk_management": {
        "max_position_size": <percent>,
        "stop_loss_approach": "<description>",
        "diversification_advice": "<description>"
    },
    "specific_actions": [<ordered list of actions to take>],
    "avoid": [<things to avoid>],
    "expected_outcomes": {
        "best_case": "<description>",
        "base_case": "<description>",
        "worst_case": "<description>"
    },
    "review_triggers": [<when to review/adjust strategy>],
    "summary": "<brief strategy summary>"
}"""

STRATEGY_ADVICE_PROMPT = PromptTemplate(
    name="strategy_advice",
    category=PromptCategory.STRATEGY,
    system_prompt=STRATEGY_SYSTEM_PROMPT,
    user_prompt_template="""Provide strategic advice for the following situation:

Trader Profile:
- Risk Tolerance: $risk_tolerance
- Investment Horizon: $investment_horizon
- Account Size: $account_size
- Experience Level: $experience_level

Current Portfolio:
$current_portfolio

Trading Goals:
$trading_goals

Market Conditions:
$market_conditions

Constraints:
$constraints

Provide comprehensive strategy advice in JSON format.""",
    description="Personalized strategy advice",
    required_variables=["risk_tolerance"],
    optional_variables=["investment_horizon", "account_size", "experience_level", "current_portfolio", "trading_goals", "market_conditions", "constraints"],
    output_format="json"
)


# =============================================================================
# NEWS ANALYSIS PROMPTS
# =============================================================================

NEWS_SYSTEM_PROMPT = """You are a financial news analyst specializing in market-moving news identification and impact assessment. Your task is to analyze news and determine its trading relevance.

Guidelines:
1. Identify the key facts and claims
2. Assess the credibility and source quality
3. Determine affected securities and sectors
4. Estimate the magnitude of potential impact
5. Consider both immediate and delayed effects
6. Watch for potential misinformation

Always respond in valid JSON format with the following structure:
{
    "headline_summary": "<concise summary>",
    "impact_level": "<none|low|medium|high|critical>",
    "affected_symbols": [{"symbol": "<ticker>", "impact": "<positive|negative|neutral>", "magnitude": <1-10>}],
    "affected_sectors": [<list of sectors>],
    "key_facts": [<list of key facts>],
    "market_implications": "<description>",
    "trading_relevance": "<immediate|short_term|long_term|none>",
    "recommended_action": "<description>",
    "confidence": <float 0-1>,
    "caveats": [<things to watch out for>]
}"""

NEWS_ANALYSIS_PROMPT = PromptTemplate(
    name="news_analysis",
    category=PromptCategory.NEWS,
    system_prompt=NEWS_SYSTEM_PROMPT,
    user_prompt_template="""Analyze the following news for trading relevance:

Headline: $headline
Source: $source
Published: $published_date

Article:
$article_content

Related Symbols: $related_symbols

Provide news analysis and trading implications in JSON format.""",
    description="News analysis for trading",
    required_variables=["headline", "article_content"],
    optional_variables=["source", "published_date", "related_symbols"],
    output_format="json"
)

NEWS_SUMMARY_PROMPT = PromptTemplate(
    name="news_summary",
    category=PromptCategory.NEWS,
    system_prompt=NEWS_SYSTEM_PROMPT,
    user_prompt_template="""Summarize the following news articles for $symbol:

Articles:
$articles

Provide a consolidated summary including:
1. Key themes across articles
2. Overall sentiment
3. Most important developments
4. Trading implications

Output in JSON format.""",
    description="Multi-article news summary",
    required_variables=["symbol", "articles"],
    output_format="json"
)


# =============================================================================
# PROMPT REGISTRY
# =============================================================================

class PromptRegistry:
    """Registry for managing all prompt templates."""

    def __init__(self) -> None:
        """Initialize the prompt registry."""
        self._prompts: Dict[str, PromptTemplate] = {}
        self._register_default_prompts()

    def _register_default_prompts(self) -> None:
        """Register all default prompts."""
        default_prompts = [
            SENTIMENT_NEWS_PROMPT,
            SENTIMENT_SOCIAL_PROMPT,
            SENTIMENT_AGGREGATE_PROMPT,
            CHART_ANALYSIS_PROMPT,
            CHART_PATTERN_PROMPT,
            TRADING_SIGNAL_PROMPT,
            TRADING_ENTRY_PROMPT,
            RISK_ASSESSMENT_PROMPT,
            PORTFOLIO_RISK_PROMPT,
            MARKET_OVERVIEW_PROMPT,
            SECTOR_ANALYSIS_PROMPT,
            STRATEGY_ADVICE_PROMPT,
            NEWS_ANALYSIS_PROMPT,
            NEWS_SUMMARY_PROMPT,
        ]
        for prompt in default_prompts:
            self.register(prompt)

    def register(self, prompt: PromptTemplate) -> None:
        """
        Register a prompt template.

        Args:
            prompt: Prompt template to register
        """
        self._prompts[prompt.name] = prompt
        logger.debug(f"Registered prompt: {prompt.name}")

    def get(self, name: str) -> Optional[PromptTemplate]:
        """
        Get a prompt template by name.

        Args:
            name: Prompt name

        Returns:
            PromptTemplate or None
        """
        return self._prompts.get(name)

    def get_by_category(self, category: PromptCategory) -> List[PromptTemplate]:
        """
        Get all prompts in a category.

        Args:
            category: Prompt category

        Returns:
            List of prompts in the category
        """
        return [p for p in self._prompts.values() if p.category == category]

    def list_prompts(self) -> List[str]:
        """
        List all registered prompt names.

        Returns:
            List of prompt names
        """
        return list(self._prompts.keys())

    def format_prompt(self, name: str, **kwargs: Any) -> Optional[str]:
        """
        Format a prompt by name with variables.

        Args:
            name: Prompt name
            **kwargs: Variables for substitution

        Returns:
            Formatted prompt or None
        """
        prompt = self.get(name)
        if prompt:
            return prompt.format_user_prompt(**kwargs)
        return None


# Global prompt registry instance
prompt_registry = PromptRegistry()


def get_prompt(name: str) -> Optional[PromptTemplate]:
    """
    Get a prompt template by name.

    Args:
        name: Prompt name

    Returns:
        PromptTemplate or None
    """
    return prompt_registry.get(name)


def format_prompt(name: str, **kwargs: Any) -> Optional[str]:
    """
    Format a prompt by name with variables.

    Args:
        name: Prompt name
        **kwargs: Variables for substitution

    Returns:
        Formatted prompt or None
    """
    return prompt_registry.format_prompt(name, **kwargs)
