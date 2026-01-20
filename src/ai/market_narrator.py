"""
Market Narrator for generating human-readable market commentary.
~350 lines
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from .openai_client import OpenAIClient
from .prompt_manager import PromptManager
from config.openai_config import OpenAIConfig

logger = logging.getLogger(__name__)


class MarketNarrator:
    """
    AI-powered market narrator for generating commentary and explanations.

    Features:
    - Market commentary generation
    - Trade explanation
    - Performance narration
    - Insight generation
    """

    def __init__(self, config: Optional[OpenAIConfig] = None):
        """Initialize market narrator."""
        self.client = OpenAIClient(config)
        self.prompt_manager = PromptManager()
        self.stats = {"narratives_generated": 0}

    def generate_market_commentary(
        self,
        market_data: Dict[str, Any],
        timeframe: str = "daily"
    ) -> str:
        """
        Generate market commentary.

        Args:
            market_data: Market data and movements
            timeframe: Timeframe for commentary

        Returns:
            Human-readable market commentary
        """
        if not self.client.is_available():
            return "Market commentary unavailable - AI not available"

        try:
            self.stats["narratives_generated"] += 1

            prompt = f"""Generate a brief market commentary for {timeframe} timeframe.

Market Data:
{json.dumps(market_data, indent=2)}

Provide a concise 2-3 sentence commentary covering:
- Overall market movement
- Key drivers
- Notable trends
"""

            messages = [
                {"role": "system", "content": "You are a market commentator providing clear, concise insights."},
                {"role": "user", "content": prompt}
            ]

            response = self.client.chat_completion(messages, temperature=0.5, max_tokens=300)
            return response if response else "Commentary unavailable"

        except Exception as e:
            logger.error(f"Commentary generation failed: {e}")
            return f"Commentary unavailable: {str(e)}"

    def explain_trade(
        self,
        trade: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """
        Generate explanation for a trade.

        Args:
            trade: Trade details
            context: Trade context

        Returns:
            Human-readable explanation
        """
        if not self.client.is_available():
            return "Trade explanation unavailable"

        try:
            prompt = f"""Explain this trade in simple language:

Trade: {json.dumps(trade)}
Context: {json.dumps(context)}

Provide a clear 1-2 sentence explanation of:
- What action was taken
- Why it was taken
- Expected outcome
"""

            messages = [
                {"role": "system", "content": "You explain trading decisions clearly."},
                {"role": "user", "content": prompt}
            ]

            response = self.client.chat_completion(messages, temperature=0.5, max_tokens=200)
            return response if response else "Explanation unavailable"

        except Exception as e:
            logger.error(f"Trade explanation failed: {e}")
            return "Explanation unavailable"

    def narrate_performance(
        self,
        performance_data: Dict[str, Any],
        period: str = "day"
    ) -> str:
        """
        Narrate trading performance.

        Args:
            performance_data: Performance metrics
            period: Time period

        Returns:
            Performance narrative
        """
        if not self.client.is_available():
            return "Performance narration unavailable"

        try:
            prompt = f"""Narrate this {period} trading performance:

{json.dumps(performance_data, indent=2)}

Provide a brief narrative highlighting:
- Overall performance
- Best/worst performers
- Key statistics
"""

            messages = [
                {"role": "system", "content": "You narrate trading performance."},
                {"role": "user", "content": prompt}
            ]

            response = self.client.chat_completion(messages, temperature=0.5, max_tokens=300)
            return response if response else "Narration unavailable"

        except Exception as e:
            logger.error(f"Performance narration failed: {e}")
            return "Narration unavailable"

    def run(self, text: str) -> dict:
        """Legacy method."""
        result = self.client.analyze(text)
        return {'status': 'ok', 'result': result}

    def get_stats(self) -> Dict[str, Any]:
        """Get narrator statistics."""
        return {**self.stats, "client_stats": self.client.get_usage_stats()}
