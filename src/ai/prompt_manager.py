"""
Prompt Manager for centralized AI prompt template management.
~300 lines
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from config.prompts_config import PROMPTS

logger = logging.getLogger(__name__)


class PromptManager:
    """
    Centralized prompt template management system.

    Features:
    - Template storage and retrieval
    - Variable substitution
    - Prompt versioning
    - Usage tracking
    - Custom prompts support
    """

    DEFAULT_PROMPTS = {
        # System prompts
        "sentiment_system": """You are an expert financial sentiment analyzer. Analyze market sentiment from news and social media content.
Always respond in valid JSON format with sentiment (bullish/bearish/neutral/mixed), confidence (0-1), score (-1 to 1), reasoning, entities, key_phrases, and impact_level (low/medium/high).""",

        "chart_system": """You are an expert technical analyst with chart pattern recognition capabilities.
Analyze chart images and provide structured insights about trends, patterns, support/resistance levels, and trading signals.
Always respond in valid JSON format.""",

        "trading_agent_system": """You are a READ-ONLY trading analysis assistant. You can analyze market data, provide insights, and suggest strategies, but you CANNOT execute trades.
Your role is purely analytical and advisory. Use the available read-only tools to gather information and provide comprehensive analysis.""",

        "risk_system": """You are a risk management specialist. Analyze trading scenarios and identify potential risks, position sizing issues, and risk mitigation strategies.
Always respond in valid JSON format with risk_level, concerns, recommendations, and position_adjustment suggestions.""",

        "strategy_system": """You are a multi-strategy trading advisor. Analyze different trading strategies and provide consensus recommendations based on various approaches.
Consider momentum, mean reversion, technical, and fundamental factors. Respond in valid JSON format.""",

        "news_system": """You are a financial news analyst. Prioritize and summarize news articles based on market impact and relevance.
Extract key information and provide structured analysis in JSON format.""",

        "narrator_system": """You are a financial market narrator. Generate clear, concise commentary about market conditions and trading activity.
Write in a professional but accessible style suitable for traders.""",

        # User prompts
        "sentiment_analysis": """Analyze the sentiment of the following {content_type} regarding {symbol}:

Content: {content}

Provide detailed sentiment analysis including:
- Overall sentiment (bullish/bearish/neutral/mixed)
- Confidence level
- Sentiment score (-1 to 1)
- Key reasoning
- Mentioned entities
- Important phrases
- Impact level""",

        "chart_analysis": """Analyze this trading chart for {symbol} on {timeframe} timeframe.

Identify:
- Current trend direction and strength
- Chart patterns (if any)
- Support and resistance levels
- Key technical indicators
- Potential entry/exit points
- Risk/reward assessment

Provide analysis in JSON format.""",

        "risk_assessment": """Assess the risk for the following trading scenario:

Symbol: {symbol}
Action: {action}
Quantity: {quantity}
Entry Price: {price}
Account Balance: {balance}
Existing Positions: {positions}

Market Conditions: {conditions}

Evaluate:
- Position size appropriateness
- Risk/reward ratio
- Portfolio impact
- Potential concerns
- Risk mitigation recommendations""",

        "strategy_consensus": """Analyze the following strategies for {symbol} and provide a consensus recommendation:

Strategies:
{strategies}

Market Data:
{market_data}

Provide:
- Consensus action (buy/sell/hold)
- Confidence level
- Agreement level among strategies
- Key factors supporting the consensus
- Dissenting views (if any)
- Overall recommendation""",

        "news_prioritization": """Prioritize and summarize the following news articles for {symbol}:

{news_items}

For each article provide:
- Priority score (0-10)
- Impact level (low/medium/high)
- Sentiment (positive/negative/neutral)
- Key summary (1-2 sentences)
- Trading implications

Sort by priority.""",

        "market_commentary": """Generate market commentary for the current trading session:

Market Overview:
{market_overview}

Notable Movements:
{movements}

Key Events:
{events}

Portfolio Activity:
{portfolio_activity}

Write a concise 2-3 paragraph summary suitable for a trading journal.""",

        # Multi-turn prompts
        "followup_question": """Based on our previous discussion, {question}

Previous context: {context}""",

        "clarification": """I need clarification on: {topic}

Specific question: {question}"""
    }

    def __init__(self):
        """Initialize prompt manager with default and config prompts."""
        # Start with defaults
        self.prompts = self.DEFAULT_PROMPTS.copy()

        # Override/extend with config prompts
        self.prompts.update(PROMPTS)

        # Custom prompts added at runtime
        self.custom_prompts: Dict[str, str] = {}

        # Usage tracking
        self.usage_count: Dict[str, int] = {}

        logger.info(f"PromptManager initialized with {len(self.prompts)} prompts")

    def get_prompt(self, key: str, default: str = "") -> str:
        """
        Get prompt template by key.

        Args:
            key: Prompt key
            default: Default value if key not found

        Returns:
            Prompt template string
        """
        # Track usage
        self.usage_count[key] = self.usage_count.get(key, 0) + 1

        # Check custom prompts first
        if key in self.custom_prompts:
            return self.custom_prompts[key]

        # Then regular prompts
        if key in self.prompts:
            return self.prompts[key]

        # Return default
        if default:
            logger.warning(f"Prompt key '{key}' not found, using default")
            return default

        logger.warning(f"Prompt key '{key}' not found and no default provided")
        return ""

    def get(self, key: str) -> str:
        """Alias for get_prompt for backward compatibility."""
        return self.get_prompt(key)

    def format_prompt(self, key: str, **kwargs) -> str:
        """
        Get and format prompt with variables.

        Args:
            key: Prompt key
            **kwargs: Variables to substitute in template

        Returns:
            Formatted prompt string
        """
        template = self.get_prompt(key)

        if not template:
            return ""

        try:
            # Perform variable substitution
            formatted = template.format(**kwargs)
            return formatted
        except KeyError as e:
            logger.error(f"Missing variable {e} for prompt '{key}'")
            # Return template with unsubstituted variables
            return template
        except Exception as e:
            logger.error(f"Error formatting prompt '{key}': {e}")
            return template

    def add_custom_prompt(self, key: str, template: str, overwrite: bool = False) -> bool:
        """
        Add a custom prompt template.

        Args:
            key: Prompt key
            template: Prompt template
            overwrite: Whether to overwrite existing prompt

        Returns:
            True if added successfully
        """
        if key in self.custom_prompts and not overwrite:
            logger.warning(f"Custom prompt '{key}' already exists, not overwriting")
            return False

        self.custom_prompts[key] = template
        logger.info(f"Added custom prompt: {key}")
        return True

    def remove_custom_prompt(self, key: str) -> bool:
        """
        Remove a custom prompt.

        Args:
            key: Prompt key to remove

        Returns:
            True if removed successfully
        """
        if key in self.custom_prompts:
            del self.custom_prompts[key]
            logger.info(f"Removed custom prompt: {key}")
            return True

        logger.warning(f"Custom prompt '{key}' not found")
        return False

    def list_prompts(self, include_custom: bool = True) -> list:
        """
        List all available prompt keys.

        Args:
            include_custom: Whether to include custom prompts

        Returns:
            List of prompt keys
        """
        keys = list(self.prompts.keys())

        if include_custom:
            keys.extend(self.custom_prompts.keys())

        return sorted(keys)

    def get_usage_stats(self) -> Dict[str, int]:
        """
        Get prompt usage statistics.

        Returns:
            Dictionary of prompt keys and usage counts
        """
        return dict(sorted(
            self.usage_count.items(),
            key=lambda x: x[1],
            reverse=True
        ))

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "total_prompts": len(self.prompts),
            "custom_prompts": len(self.custom_prompts),
            "total_usage": sum(self.usage_count.values()),
            "most_used": list(self.get_usage_stats().items())[:5],
            "timestamp": datetime.now().isoformat()
        }

    def export_prompts(self) -> Dict[str, str]:
        """
        Export all prompts for backup/sharing.

        Returns:
            Dictionary of all prompts
        """
        all_prompts = self.prompts.copy()
        all_prompts.update(self.custom_prompts)
        return all_prompts

    def import_prompts(self, prompts: Dict[str, str], overwrite: bool = False) -> int:
        """
        Import prompts from dictionary.

        Args:
            prompts: Dictionary of prompts to import
            overwrite: Whether to overwrite existing prompts

        Returns:
            Number of prompts imported
        """
        imported = 0

        for key, template in prompts.items():
            if overwrite or key not in self.custom_prompts:
                self.custom_prompts[key] = template
                imported += 1

        logger.info(f"Imported {imported} prompts")
        return imported

    def clear_usage_stats(self):
        """Clear usage statistics."""
        self.usage_count.clear()
        logger.info("Usage statistics cleared")

    def validate_prompt(self, key: str) -> bool:
        """
        Validate that a prompt exists and is not empty.

        Args:
            key: Prompt key

        Returns:
            True if valid
        """
        prompt = self.get_prompt(key)
        return bool(prompt and prompt.strip())
