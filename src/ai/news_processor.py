"""
News Processor for prioritizing and analyzing news articles.
~400 lines  
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from .openai_client import OpenAIClient
from .prompt_manager import PromptManager
from config.openai_config import OpenAIConfig

logger = logging.getLogger(__name__)


class NewsProcessor:
    """
    AI-powered news processor for prioritizing and analyzing market news.

    Features:
    - News relevance scoring
    - Priority ranking
    - Impact assessment
    - Entity extraction
    - Summary generation
    """

    def __init__(self, config: Optional[OpenAIConfig] = None):
        """Initialize news processor."""
        self.client = OpenAIClient(config)
        self.prompt_manager = PromptManager()
        self.stats = {"articles_processed": 0, "summaries_generated": 0}

    def prioritize_news(
        self,
        articles: List[Dict[str, str]],
        symbols: List[str],
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Prioritize news articles by relevance to symbols.

        Args:
            articles: List of news articles
            symbols: Symbols to prioritize for
            max_results: Maximum articles to return

        Returns:
            Prioritized list of articles with scores
        """
        if not self.client.is_available():
            return self._fallback_prioritize(articles, max_results)

        try:
            prioritized = []
            for article in articles[:20]:  # Limit batch size
                score = self._score_relevance(article, symbols)
                article["relevance_score"] = score
                prioritized.append(article)

            # Sort by relevance
            prioritized.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            return prioritized[:max_results]

        except Exception as e:
            logger.error(f"News prioritization failed: {e}")
            return self._fallback_prioritize(articles, max_results)

    def assess_impact(self, article: Dict[str, str], symbol: str) -> Dict[str, Any]:
        """
        Assess the potential market impact of a news article.

        Args:
            article: News article
            symbol: Stock symbol

        Returns:
            Impact assessment
        """
        if not self.client.is_available():
            return {"impact": "unknown", "confidence": 0.0}

        try:
            prompt = f"""Assess the market impact of this news for {symbol}.

Headline: {article.get('headline', '')}
Summary: {article.get('summary', article.get('body', ''))[:500]}

Provide assessment in JSON format with:
- impact_level: low/medium/high
- direction: positive/negative/neutral/mixed
- confidence: 0.0 to 1.0
- reasoning: explanation
- affected_sectors: list of affected sectors
- time_horizon: immediate/short-term/long-term
- key_factors: important factors mentioned
"""

            messages = [
                {"role": "system", "content": "You are a market news analyst."},
                {"role": "user", "content": prompt}
            ]

            response = self.client.chat_completion(messages, temperature=0.3, max_tokens=500)

            if response:
                try:
                    result = json.loads(response) if response.startswith('{') else {"impact_level": "low"}
                    result["assessed_at"] = datetime.now().isoformat()
                    return result
                except:
                    return {"impact": "unknown", "confidence": 0.0}

            return {"impact": "unknown", "confidence": 0.0}

        except Exception as e:
            logger.error(f"Impact assessment failed: {e}")
            return {"impact": "unknown", "confidence": 0.0, "error": str(e)}

    def generate_summary(self, articles: List[Dict[str, str]], max_length: int = 500) -> str:
        """
        Generate a summary of multiple news articles.

        Args:
            articles: List of articles to summarize
            max_length: Maximum summary length

        Returns:
            Summary text
        """
        if not self.client.is_available():
            return "News summary unavailable - AI not available"

        try:
            self.stats["summaries_generated"] += 1

            combined = "\n\n".join([
                f"- {a.get('headline', '')} ({a.get('source', 'unknown')})"
                for a in articles[:10]
            ])

            prompt = f"""Summarize these news articles concisely ({max_length} chars max):

{combined}

Provide a brief summary highlighting:
- Main themes
- Key developments
- Market implications
"""

            messages = [
                {"role": "system", "content": "You are summarizing market news."},
                {"role": "user", "content": prompt}
            ]

            response = self.client.chat_completion(messages, temperature=0.4, max_tokens=300)
            return response if response else "Summary unavailable"

        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return f"Summary error: {str(e)}"

    def run(self, text: str) -> dict:
        """Legacy method for backward compatibility."""
        result = self.client.analyze(text)
        return {'status': 'ok', 'result': result}

    def _score_relevance(self, article: Dict[str, str], symbols: List[str]) -> float:
        """Score article relevance to symbols."""
        text = (article.get('headline', '') + ' ' + article.get('summary', '')).lower()
        score = 0.0
        for symbol in symbols:
            if symbol.lower() in text:
                score += 0.5
        return min(score, 1.0)

    def _fallback_prioritize(self, articles: List[Dict[str, Any]], max_results: int) -> List[Dict[str, Any]]:
        """Fallback prioritization."""
        return articles[:max_results]

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {**self.stats, "client_stats": self.client.get_usage_stats()}
