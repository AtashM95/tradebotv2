"""
Strategy Advisor for multi-strategy consensus and recommendations.
~500 lines
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from .openai_client import OpenAIClient
from .prompt_manager import PromptManager
from .response_validator import ResponseValidator
from config.openai_config import OpenAIConfig

logger = logging.getLogger(__name__)


class StrategyAdvisor:
    """
    AI-powered strategy advisor for analyzing and recommending trading strategies.

    Features:
    - Multi-strategy consensus
    - Strategy performance analysis
    - Adaptive strategy recommendations
    - Market condition matching
    - Risk-adjusted suggestions
    """

    def __init__(self, config: Optional[OpenAIConfig] = None):
        """Initialize strategy advisor."""
        self.client = OpenAIClient(config)
        self.prompt_manager = PromptManager()
        self.validator = ResponseValidator()

        # Strategy performance history
        self.strategy_history: Dict[str, List[Dict[str, Any]]] = {}

        # Statistics
        self.stats = {
            "recommendations_made": 0,
            "consensus_analyses": 0
        }

    def recommend_strategies(
        self,
        market_conditions: Dict[str, Any],
        available_strategies: List[str],
        risk_profile: str = "moderate"
    ) -> Dict[str, Any]:
        """
        Recommend optimal strategies for current market conditions.

        Args:
            market_conditions: Current market analysis
            available_strategies: List of available strategy names
            risk_profile: Risk profile (conservative/moderate/aggressive)

        Returns:
            Strategy recommendations with reasoning
        """
        if not self.client.is_available():
            return self._get_fallback_recommendation(available_strategies)

        try:
            self.stats["recommendations_made"] += 1

            prompt = f"""Analyze current market conditions and recommend the best trading strategies.

Market Conditions:
{json.dumps(market_conditions, indent=2)}

Available Strategies:
{json.dumps(available_strategies, indent=2)}

Risk Profile: {risk_profile}

Provide recommendations in JSON format with:
- recommended_strategies: list of strategy names (top 3-5)
- reasoning: why these strategies are suitable
- market_regime: detected market regime
- confidence: 0.0 to 1.0
- risk_assessment: risk level for each strategy
- allocation_suggestion: suggested allocation percentages
- conditions_to_watch: what to monitor
"""

            messages = [
                {"role": "system", "content": self.prompt_manager.get_prompt("strategy_advisor_system")},
                {"role": "user", "content": prompt}
            ]

            response = self.client.chat_completion(messages, temperature=0.4, max_tokens=1500)

            if response:
                result = self._parse_json_response(response)
                result["analyzed_at"] = datetime.now().isoformat()
                result["risk_profile"] = risk_profile
                return result

            return self._get_fallback_recommendation(available_strategies)

        except Exception as e:
            logger.error(f"Strategy recommendation failed: {e}")
            return self._get_fallback_recommendation(available_strategies)

    def analyze_strategy_consensus(
        self,
        strategy_signals: List[Dict[str, Any]],
        market_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze consensus among multiple strategy signals.

        Args:
            strategy_signals: Signals from different strategies
            market_context: Current market context

        Returns:
            Consensus analysis
        """
        if not self.client.is_available():
            return {"consensus": "neutral", "confidence": 0.0}

        try:
            self.stats["consensus_analyses"] += 1

            # Count signal types
            signal_counts = {"buy": 0, "sell": 0, "hold": 0, "neutral": 0}
            for signal in strategy_signals:
                action = signal.get("action", "neutral").lower()
                signal_counts[action] = signal_counts.get(action, 0) + 1

            prompt = f"""Analyze these trading strategy signals and provide consensus.

Strategy Signals:
{json.dumps(strategy_signals, indent=2)}

Signal Distribution:
- Buy signals: {signal_counts['buy']}
- Sell signals: {signal_counts['sell']}
- Hold signals: {signal_counts['hold']}
- Neutral signals: {signal_counts['neutral']}

Market Context:
{json.dumps(market_context, indent=2)}

Provide analysis in JSON format with:
- consensus: overall recommendation (buy/sell/hold/neutral)
- confidence: 0.0 to 1.0
- agreement_level: how much strategies agree (low/medium/high)
- conflicting_signals: description of conflicts
- strongest_signal: which strategy has strongest case
- recommended_action: specific action to take
- reasoning: detailed explanation
- risk_level: low/medium/high
"""

            messages = [
                {"role": "system", "content": "You are an expert at analyzing multiple trading strategies."},
                {"role": "user", "content": prompt}
            ]

            response = self.client.chat_completion(messages, temperature=0.3, max_tokens=1000)

            if response:
                result = self._parse_json_response(response)
                result["analyzed_at"] = datetime.now().isoformat()
                result["num_strategies"] = len(strategy_signals)
                result["signal_distribution"] = signal_counts
                return result

            return {"consensus": "neutral", "confidence": 0.0}

        except Exception as e:
            logger.error(f"Strategy consensus analysis failed: {e}")
            return {"consensus": "neutral", "confidence": 0.0, "error": str(e)}

    def run(self, text: str) -> dict:
        """Legacy method for backward compatibility."""
        result = self.client.analyze(text)
        return {'status': 'ok', 'result': result}

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response."""
        result = self.validator.extract_json_from_response(response)

        if result:
            return result

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response")
            return {"error": "parse_failed", "raw": response[:200]}

    def _get_fallback_recommendation(self, strategies: List[str]) -> Dict[str, Any]:
        """Get fallback recommendation when AI unavailable."""
        return {
            "recommended_strategies": strategies[:3] if len(strategies) >= 3 else strategies,
            "reasoning": "AI unavailable - defaulting to first available strategies",
            "market_regime": "unknown",
            "confidence": 0.0,
            "fallback": True,
            "analyzed_at": datetime.now().isoformat()
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get advisor statistics."""
        return {
            **self.stats,
            "strategies_tracked": len(self.strategy_history),
            "client_stats": self.client.get_usage_stats()
        }
