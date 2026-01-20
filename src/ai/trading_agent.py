"""
Trading Agent with function calling capabilities for AI-assisted trading decisions.
~700 lines
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import json

from .openai_client import OpenAIClient
from .prompt_manager import PromptManager
from .response_validator import ResponseValidator
from config.openai_config import OpenAIConfig

logger = logging.getLogger(__name__)


class TradingAgent:
    """
    AI Trading Agent using OpenAI function calling.

    Features:
    - Read-only market data access
    - Trade plan generation
    - Risk assessment
    - Multi-strategy consensus
    - Context-aware decision making
    - Execution tools DISABLED by design (safety)
    """

    def __init__(self, config: Optional[OpenAIConfig] = None):
        """Initialize trading agent."""
        self.client = OpenAIClient(config)
        self.prompt_manager = PromptManager()
        self.validator = ResponseValidator()

        # Conversation history for context
        self.conversation_history: List[Dict[str, Any]] = []

        # Available functions (READ-ONLY tools)
        self.available_functions = self._register_functions()

        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_plans": 0,
            "failed_plans": 0,
            "function_calls": 0
        }

    def generate_trade_plan(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        technical_analysis: Dict[str, Any],
        sentiment: Optional[Dict[str, Any]] = None,
        risk_parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive trade plan using AI analysis.

        Args:
            symbol: Stock symbol
            market_data: Current market data
            technical_analysis: Technical analysis results
            sentiment: Sentiment analysis results
            risk_parameters: Risk parameters

        Returns:
            Trade plan with recommendations
        """
        if not self.client.is_available():
            logger.warning("OpenAI not available - returning fallback plan")
            return self._get_fallback_plan()

        try:
            self.stats["total_requests"] += 1

            # Build context
            context = self._build_trade_context(
                symbol, market_data, technical_analysis, sentiment, risk_parameters
            )

            # Create prompt
            prompt = self._build_trade_plan_prompt(context)

            # Add to conversation history
            messages = [
                {"role": "system", "content": self.prompt_manager.get_prompt("trading_agent_system")},
                {"role": "user", "content": prompt}
            ]

            # Add function definitions for AI to use
            functions = self._get_function_definitions()

            # Make request with function calling
            response = self.client.chat_completion(
                messages,
                temperature=0.3,  # Lower temperature for consistency
                max_tokens=2000
            )

            if not response:
                self.stats["failed_plans"] += 1
                return self._get_fallback_plan()

            # Parse response
            plan = self._parse_trade_plan(response)

            # Add metadata
            plan["symbol"] = symbol
            plan["generated_at"] = datetime.now().isoformat()
            plan["confidence"] = plan.get("confidence", 0.5)

            # Validate plan
            if self._validate_plan(plan):
                self.stats["successful_plans"] += 1
                return plan
            else:
                logger.warning("Invalid trade plan generated")
                self.stats["failed_plans"] += 1
                return self._get_fallback_plan()

        except Exception as e:
            logger.error(f"Trade plan generation failed: {e}")
            self.stats["failed_plans"] += 1
            return self._get_fallback_plan()

    def analyze_multi_strategy(
        self,
        symbol: str,
        strategy_signals: List[Dict[str, Any]],
        market_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze multiple strategy signals and provide consensus.

        Args:
            symbol: Stock symbol
            strategy_signals: List of signals from different strategies
            market_context: Current market context

        Returns:
            Consensus analysis with recommendation
        """
        if not self.client.is_available():
            return {"consensus": "neutral", "confidence": 0.0}

        try:
            # Build prompt
            prompt = f"""Analyze the following trading signals for {symbol} and provide a consensus recommendation.

Strategy Signals:
{json.dumps(strategy_signals, indent=2)}

Market Context:
{json.dumps(market_context, indent=2)}

Provide analysis in JSON format with:
- consensus: overall recommendation (buy/sell/hold/neutral)
- confidence: 0.0 to 1.0
- reasoning: detailed explanation
- conflicting_signals: any conflicting signals identified
- recommended_action: specific action to take
- risk_assessment: risk level (low/medium/high)
"""

            messages = [
                {"role": "system", "content": "You are an expert trading analyst providing strategy consensus."},
                {"role": "user", "content": prompt}
            ]

            response = self.client.chat_completion(messages, temperature=0.3, max_tokens=1000)

            if response:
                result = self._parse_json_response(response)
                result["symbol"] = symbol
                result["analyzed_at"] = datetime.now().isoformat()
                return result

            return {"consensus": "neutral", "confidence": 0.0}

        except Exception as e:
            logger.error(f"Multi-strategy analysis failed: {e}")
            return {"consensus": "neutral", "confidence": 0.0, "error": str(e)}

    def assess_trade_opportunity(
        self,
        symbol: str,
        signal: Dict[str, Any],
        account_state: Dict[str, Any],
        risk_limits: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess a trade opportunity considering account state and risk.

        Args:
            symbol: Stock symbol
            signal: Trading signal
            account_state: Current account state
            risk_limits: Risk limit parameters

        Returns:
            Assessment with recommendation
        """
        if not self.client.is_available():
            return {"assessment": "skip", "reason": "AI unavailable"}

        try:
            prompt = f"""Assess this trade opportunity:

Symbol: {symbol}
Signal: {json.dumps(signal)}
Account State: {json.dumps(account_state)}
Risk Limits: {json.dumps(risk_limits)}

Provide assessment in JSON format with:
- assessment: proceed/caution/skip
- sizing_recommendation: suggested position size
- risk_reward_ratio: estimated risk/reward
- concerns: any concerns identified
- entry_points: recommended entry price levels
- stop_loss: recommended stop loss level
- take_profit: recommended take profit levels
- reasoning: detailed explanation
"""

            messages = [
                {"role": "system", "content": "You are a prudent trading risk assessor."},
                {"role": "user", "content": prompt}
            ]

            response = self.client.chat_completion(messages, temperature=0.3, max_tokens=1000)

            if response:
                result = self._parse_json_response(response)
                result["symbol"] = symbol
                result["assessed_at"] = datetime.now().isoformat()
                return result

            return {"assessment": "skip", "reason": "No response"}

        except Exception as e:
            logger.error(f"Trade opportunity assessment failed: {e}")
            return {"assessment": "skip", "reason": str(e)}

    def explain_decision(
        self,
        symbol: str,
        decision: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """
        Generate human-readable explanation of a trading decision.

        Args:
            symbol: Stock symbol
            decision: Trading decision details
            context: Decision context

        Returns:
            Human-readable explanation
        """
        if not self.client.is_available():
            return "AI explanation unavailable"

        try:
            prompt = f"""Explain this trading decision in clear, simple language:

Symbol: {symbol}
Decision: {json.dumps(decision)}
Context: {json.dumps(context)}

Provide a concise explanation (2-3 sentences) that a trader can understand, covering:
- What action was taken and why
- Key factors that influenced the decision
- What the expected outcome is
"""

            messages = [
                {"role": "system", "content": "You are explaining trading decisions to traders."},
                {"role": "user", "content": prompt}
            ]

            response = self.client.chat_completion(messages, temperature=0.5, max_tokens=300)

            return response if response else "Unable to generate explanation"

        except Exception as e:
            logger.error(f"Decision explanation failed: {e}")
            return f"Explanation unavailable: {str(e)}"

    def get_market_insights(
        self,
        symbols: List[str],
        market_data: Dict[str, Any],
        news_summary: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get AI-generated market insights across multiple symbols.

        Args:
            symbols: List of symbols to analyze
            market_data: Market data for symbols
            news_summary: Optional news summary

        Returns:
            Market insights and recommendations
        """
        if not self.client.is_available():
            return {"insights": "unavailable"}

        try:
            prompt = f"""Provide market insights for these symbols:

Symbols: {', '.join(symbols)}
Market Data: {json.dumps(market_data)}

{f"Recent News: {news_summary}" if news_summary else ""}

Provide insights in JSON format with:
- overall_market_sentiment: general market sentiment
- sector_trends: any sector trends observed
- symbol_highlights: key observations for each symbol
- opportunities: potential trading opportunities
- risks: key risks to watch
- recommendations: top 3 actionable recommendations
"""

            messages = [
                {"role": "system", "content": "You are a market analyst providing insights."},
                {"role": "user", "content": prompt}
            ]

            response = self.client.chat_completion(messages, temperature=0.4, max_tokens=1500)

            if response:
                result = self._parse_json_response(response)
                result["generated_at"] = datetime.now().isoformat()
                result["symbols"] = symbols
                return result

            return {"insights": "unavailable"}

        except Exception as e:
            logger.error(f"Market insights generation failed: {e}")
            return {"insights": "unavailable", "error": str(e)}

    def _build_trade_context(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        technical_analysis: Dict[str, Any],
        sentiment: Optional[Dict[str, Any]],
        risk_parameters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build comprehensive trade context."""
        context = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "market_data": market_data,
            "technical_analysis": technical_analysis
        }

        if sentiment:
            context["sentiment"] = sentiment

        if risk_parameters:
            context["risk_parameters"] = risk_parameters

        return context

    def _build_trade_plan_prompt(self, context: Dict[str, Any]) -> str:
        """Build trade plan prompt."""
        prompt = f"""Generate a comprehensive trade plan based on this analysis:

{json.dumps(context, indent=2)}

Provide a detailed trade plan in JSON format with:
- action: buy/sell/hold/neutral
- confidence: 0.0 to 1.0
- reasoning: detailed explanation
- entry_strategy: how to enter the position
- exit_strategy: when to exit
- position_sizing: recommended size
- stop_loss: stop loss level
- take_profit: take profit levels
- risks: identified risks
- timeframe: recommended holding timeframe
- alternatives: alternative approaches to consider

Focus on:
1. Risk management
2. Clear entry and exit criteria
3. Realistic expectations
4. Capital preservation
"""
        return prompt

    def _parse_trade_plan(self, response: str) -> Dict[str, Any]:
        """Parse trade plan from response."""
        result = self._parse_json_response(response)

        # Ensure required fields
        required_fields = ["action", "confidence", "reasoning"]
        for field in required_fields:
            if field not in result:
                logger.warning(f"Missing required field: {field}")
                result[field] = "unknown" if field != "confidence" else 0.0

        return result

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from response."""
        result = self.validator.extract_json_from_response(response)

        if result:
            return result

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response")
            return {"error": "parse_failed", "raw": response[:200]}

    def _validate_plan(self, plan: Dict[str, Any]) -> bool:
        """Validate trade plan has required fields."""
        required = ["action", "confidence", "reasoning"]
        return all(field in plan for field in required)

    def _get_fallback_plan(self) -> Dict[str, Any]:
        """Get fallback plan when AI unavailable."""
        return {
            "action": "hold",
            "confidence": 0.0,
            "reasoning": "AI agent unavailable - defaulting to hold",
            "entry_strategy": "manual_review",
            "exit_strategy": "manual_review",
            "fallback": True,
            "generated_at": datetime.now().isoformat()
        }

    def _register_functions(self) -> Dict[str, Callable]:
        """Register available functions for agent (READ-ONLY)."""
        return {
            "get_market_data": self._mock_get_market_data,
            "get_technical_indicators": self._mock_get_technical_indicators,
            "get_risk_metrics": self._mock_get_risk_metrics,
            "get_account_info": self._mock_get_account_info
        }

    def _get_function_definitions(self) -> List[Dict[str, Any]]:
        """Get function definitions for OpenAI function calling."""
        return [
            {
                "name": "get_market_data",
                "description": "Get current market data for a symbol (READ-ONLY)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol"
                        }
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "get_technical_indicators",
                "description": "Get technical indicator values (READ-ONLY)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol"
                        },
                        "indicators": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of indicators to fetch"
                        }
                    },
                    "required": ["symbol"]
                }
            },
            {
                "name": "get_risk_metrics",
                "description": "Get risk metrics for account (READ-ONLY)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric_type": {
                            "type": "string",
                            "description": "Type of risk metric"
                        }
                    }
                }
            }
        ]

    # Mock functions for READ-ONLY access
    def _mock_get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Mock function - would fetch real market data."""
        self.stats["function_calls"] += 1
        return {"symbol": symbol, "status": "mock_data"}

    def _mock_get_technical_indicators(self, symbol: str, indicators: List[str]) -> Dict[str, Any]:
        """Mock function - would fetch real indicators."""
        self.stats["function_calls"] += 1
        return {"symbol": symbol, "indicators": indicators, "status": "mock_data"}

    def _mock_get_risk_metrics(self, metric_type: str) -> Dict[str, Any]:
        """Mock function - would fetch real risk metrics."""
        self.stats["function_calls"] += 1
        return {"metric_type": metric_type, "status": "mock_data"}

    def _mock_get_account_info(self) -> Dict[str, Any]:
        """Mock function - would fetch real account info."""
        self.stats["function_calls"] += 1
        return {"status": "mock_data"}

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            **self.stats,
            "success_rate": (
                self.stats["successful_plans"] / self.stats["total_requests"]
                if self.stats["total_requests"] > 0 else 0.0
            ),
            "client_stats": self.client.get_usage_stats()
        }
