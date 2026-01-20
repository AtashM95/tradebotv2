"""
AI Risk Assessor for trade risk evaluation.
~450 lines
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from .openai_client import OpenAIClient
from .prompt_manager import PromptManager
from config.openai_config import OpenAIConfig

logger = logging.getLogger(__name__)


class RiskAssessor:
    """
    AI-powered risk assessor for evaluating trading risks.

    Features:
    - Trade risk evaluation
    - Portfolio risk analysis
    - Scenario analysis
    - Risk mitigation suggestions
    """

    def __init__(self, config: Optional[OpenAIConfig] = None):
        """Initialize risk assessor."""
        self.client = OpenAIClient(config)
        self.prompt_manager = PromptManager()
        self.stats = {"assessments_made": 0}

    def assess_trade_risk(
        self,
        trade_plan: Dict[str, Any],
        market_conditions: Dict[str, Any],
        account_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess risk of a proposed trade.

        Args:
            trade_plan: Proposed trade details
            market_conditions: Current market conditions
            account_state: Current account state

        Returns:
            Risk assessment with recommendations
        """
        if not self.client.is_available():
            return self._fallback_assessment()

        try:
            self.stats["assessments_made"] += 1

            prompt = f"""Assess the risk of this trade:

Trade Plan:
{json.dumps(trade_plan, indent=2)}

Market Conditions:
{json.dumps(market_conditions, indent=2)}

Account State:
{json.dumps(account_state, indent=2)}

Provide risk assessment in JSON format with:
- overall_risk: low/medium/high/extreme
- risk_score: 0-100
- key_risks: list of identified risks
- risk_factors:
  - market_risk: assessment
  - liquidity_risk: assessment
  - volatility_risk: assessment
  - concentration_risk: assessment
- mitigation_strategies: list of risk mitigation suggestions
- recommended_action: proceed/modify/reject
- confidence: 0.0 to 1.0
"""

            messages = [
                {"role": "system", "content": self.prompt_manager.get_prompt("risk_system")},
                {"role": "user", "content": prompt}
            ]

            response = self.client.chat_completion(messages, temperature=0.3, max_tokens=1000)

            if response:
                result = self._parse_json(response)
                result["assessed_at"] = datetime.now().isoformat()
                return result

            return self._fallback_assessment()

        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return self._fallback_assessment()

    def assess_portfolio_risk(
        self,
        positions: List[Dict[str, Any]],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess overall portfolio risk.

        Args:
            positions: Current positions
            market_data: Market data

        Returns:
            Portfolio risk assessment
        """
        if not self.client.is_available():
            return {"risk_level": "unknown", "score": 0}

        try:
            prompt = f"""Assess portfolio risk:

Positions:
{json.dumps(positions[:20], indent=2)}

Market Data:
{json.dumps(market_data, indent=2)}

Provide assessment in JSON format with:
- risk_level: low/medium/high
- risk_score: 0-100
- diversification_score: 0-100
- concentration_risks: list of concentrations
- correlation_risks: correlated positions
- recommendations: risk reduction suggestions
"""

            messages = [
                {"role": "system", "content": "You are a portfolio risk analyst."},
                {"role": "user", "content": prompt}
            ]

            response = self.client.chat_completion(messages, temperature=0.3, max_tokens=800)

            if response:
                result = self._parse_json(response)
                result["assessed_at"] = datetime.now().isoformat()
                return result

            return {"risk_level": "unknown", "score": 0}

        except Exception as e:
            logger.error(f"Portfolio risk assessment failed: {e}")
            return {"risk_level": "unknown", "score": 0, "error": str(e)}

    def run(self, text: str) -> dict:
        """Legacy method."""
        result = self.client.analyze(text)
        return {'status': 'ok', 'result': result}

    def _parse_json(self, response: str) -> Dict[str, Any]:
        """Parse JSON from response."""
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
            return json.loads(json_str)
        except:
            return {"error": "parse_failed"}

    def _fallback_assessment(self) -> Dict[str, Any]:
        """Fallback assessment."""
        return {
            "overall_risk": "medium",
            "risk_score": 50,
            "key_risks": ["AI assessment unavailable"],
            "recommended_action": "review",
            "confidence": 0.0,
            "fallback": True
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get assessor statistics."""
        return {**self.stats, "client_stats": self.client.get_usage_stats()}
