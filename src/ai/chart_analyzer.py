"""
Chart Analyzer using GPT-4 Vision for chart pattern recognition.
~450 lines
"""

import logging
from typing import Dict, Any, Optional, Union
from datetime import datetime
import base64
import json
from pathlib import Path

try:
    from PIL import Image
    import io
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from .openai_client import OpenAIClient
from .prompt_manager import PromptManager
from .response_validator import ResponseValidator
from config.openai_config import OpenAIConfig

logger = logging.getLogger(__name__)


class ChartAnalyzer:
    """
    Analyze trading charts using GPT-4 Vision API.

    Features:
    - Image-based chart analysis
    - Pattern recognition (head & shoulders, triangles, etc.)
    - Support/resistance level identification
    - Trend analysis
    - Entry/exit point suggestions
    - Multiple image format support
    """

    CHART_SCHEMA = {
        "type": "object",
        "properties": {
            "trend": {
                "type": "string",
                "enum": ["uptrend", "downtrend", "sideways", "unknown"]
            },
            "trend_strength": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0
            },
            "patterns": {
                "type": "array",
                "items": {"type": "string"}
            },
            "support_levels": {
                "type": "array",
                "items": {"type": "number"}
            },
            "resistance_levels": {
                "type": "array",
                "items": {"type": "number"}
            },
            "signals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "confidence": {"type": "number"}
                    }
                }
            }
        },
        "required": ["trend", "trend_strength"]
    }

    def __init__(self, config: Optional[OpenAIConfig] = None):
        """Initialize chart analyzer."""
        self.client = OpenAIClient(config)
        self.prompt_manager = PromptManager()
        self.validator = ResponseValidator()

        # Use GPT-4 Vision model
        self.vision_model = "gpt-4o"  # GPT-4o has vision capabilities

        if not PIL_AVAILABLE:
            logger.warning("PIL not available - some image processing features disabled")

    def analyze_chart(
        self,
        image_input: Union[str, bytes, Path],
        symbol: str = "",
        timeframe: str = "1d",
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a trading chart image.

        Args:
            image_input: Path to image file, image bytes, or Path object
            symbol: Stock symbol
            timeframe: Chart timeframe
            additional_context: Additional context for analysis

        Returns:
            Chart analysis results
        """
        if not self.client.is_available():
            logger.warning("OpenAI not available - returning fallback analysis")
            return self._get_fallback_analysis()

        try:
            # Encode image to base64
            image_base64 = self._encode_image(image_input)

            if not image_base64:
                logger.error("Failed to encode image")
                return self._get_fallback_analysis()

            # Build prompt
            prompt = self._build_chart_prompt(symbol, timeframe, additional_context)

            # Make Vision API request
            response = self._make_vision_request(image_base64, prompt)

            if not response:
                return self._get_fallback_analysis()

            # Parse and validate response
            result = self._parse_chart_response(response)

            # Validate against schema
            if not self.validator.validate(result, self.CHART_SCHEMA):
                logger.warning("Invalid chart analysis response")
                result = self._get_fallback_analysis()

            # Add metadata
            result["analyzed_at"] = datetime.now().isoformat()
            result["symbol"] = symbol
            result["timeframe"] = timeframe
            result["model"] = self.vision_model

            return result

        except Exception as e:
            logger.error(f"Chart analysis failed: {e}")
            return self._get_fallback_analysis()

    def analyze_multiple_charts(
        self,
        images: list[Union[str, bytes, Path]],
        symbol: str = "",
        timeframes: Optional[list[str]] = None
    ) -> list[Dict[str, Any]]:
        """
        Analyze multiple charts (e.g., different timeframes).

        Args:
            images: List of image inputs
            symbol: Stock symbol
            timeframes: List of timeframes (should match images length)

        Returns:
            List of analysis results
        """
        if timeframes is None:
            timeframes = ["unknown"] * len(images)

        results = []
        for i, image in enumerate(images):
            timeframe = timeframes[i] if i < len(timeframes) else "unknown"
            result = self.analyze_chart(image, symbol, timeframe)
            results.append(result)

        return results

    def compare_charts(
        self,
        image1: Union[str, bytes, Path],
        image2: Union[str, bytes, Path],
        symbol: str = "",
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare two charts (e.g., before/after, different timeframes).

        Args:
            image1: First chart image
            image2: Second chart image
            symbol: Stock symbol
            context: Comparison context

        Returns:
            Comparison analysis
        """
        if not self.client.is_available():
            return {"comparison": "unavailable", "reason": "OpenAI not available"}

        try:
            # Encode both images
            image1_base64 = self._encode_image(image1)
            image2_base64 = self._encode_image(image2)

            if not image1_base64 or not image2_base64:
                return {"comparison": "unavailable", "reason": "Image encoding failed"}

            # Build comparison prompt
            prompt = f"""Compare these two charts for {symbol}.

Context: {context or 'General comparison'}

Identify:
- Key differences in trend
- Pattern changes
- Support/resistance changes
- Which chart shows stronger momentum
- Trading implications of the differences

Provide analysis in JSON format with:
- overall_comparison: summary
- key_differences: array of differences
- trend_change: description
- recommendation: trading implication"""

            # Make request with both images
            messages = [
                {"role": "system", "content": self.prompt_manager.get_prompt("chart_system")},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image1_base64}"}
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image2_base64}"}
                        }
                    ]
                }
            ]

            response = self.client.chat_completion(
                messages,
                model=self.vision_model,
                temperature=0.3,
                max_tokens=1000
            )

            if response:
                result = self._parse_json_response(response)
                result["analyzed_at"] = datetime.now().isoformat()
                result["symbol"] = symbol
                return result

            return {"comparison": "unavailable", "reason": "No response"}

        except Exception as e:
            logger.error(f"Chart comparison failed: {e}")
            return {"comparison": "unavailable", "reason": str(e)}

    def identify_patterns(
        self,
        image_input: Union[str, bytes, Path],
        symbol: str = ""
    ) -> Dict[str, Any]:
        """
        Focus specifically on identifying chart patterns.

        Args:
            image_input: Chart image
            symbol: Stock symbol

        Returns:
            Pattern identification results
        """
        if not self.client.is_available():
            return {"patterns": [], "confidence": 0.0}

        try:
            image_base64 = self._encode_image(image_input)

            if not image_base64:
                return {"patterns": [], "confidence": 0.0}

            prompt = f"""Analyze this chart for {symbol} and identify technical patterns.

Focus on:
- Classic chart patterns (head & shoulders, double top/bottom, triangles, flags, wedges)
- Candlestick patterns (if visible)
- Pattern completion level
- Potential price targets
- Pattern reliability

Return JSON with:
- patterns: array of identified patterns with confidence
- most_significant: the most important pattern
- potential_targets: price targets based on patterns
- reliability: overall pattern reliability assessment"""

            response = self._make_vision_request(image_base64, prompt)

            if response:
                result = self._parse_json_response(response)
                result["analyzed_at"] = datetime.now().isoformat()
                result["symbol"] = symbol
                return result

            return {"patterns": [], "confidence": 0.0}

        except Exception as e:
            logger.error(f"Pattern identification failed: {e}")
            return {"patterns": [], "confidence": 0.0, "error": str(e)}

    def _encode_image(self, image_input: Union[str, bytes, Path]) -> Optional[str]:
        """
        Encode image to base64 string.

        Args:
            image_input: Image path, bytes, or Path object

        Returns:
            Base64 encoded string or None
        """
        try:
            # If already bytes
            if isinstance(image_input, bytes):
                return base64.b64encode(image_input).decode('utf-8')

            # Convert Path to str
            if isinstance(image_input, Path):
                image_input = str(image_input)

            # Read from file path
            if isinstance(image_input, str):
                with open(image_input, 'rb') as image_file:
                    image_bytes = image_file.read()
                    return base64.b64encode(image_bytes).decode('utf-8')

            logger.error(f"Unsupported image input type: {type(image_input)}")
            return None

        except Exception as e:
            logger.error(f"Image encoding failed: {e}")
            return None

    def _build_chart_prompt(
        self,
        symbol: str,
        timeframe: str,
        additional_context: Optional[str]
    ) -> str:
        """Build chart analysis prompt."""
        prompt = f"Analyze this trading chart"

        if symbol:
            prompt += f" for {symbol}"

        if timeframe:
            prompt += f" on {timeframe} timeframe"

        prompt += ".\n\n"
        prompt += "Provide detailed technical analysis including:\n"
        prompt += "- Trend direction (uptrend/downtrend/sideways) and strength (0-1)\n"
        prompt += "- Identified chart patterns\n"
        prompt += "- Support and resistance levels (specific price levels)\n"
        prompt += "- Key technical indicators visible\n"
        prompt += "- Trading signals (buy/sell/neutral) with confidence\n"
        prompt += "- Risk/reward assessment\n\n"

        if additional_context:
            prompt += f"Additional context: {additional_context}\n\n"

        prompt += "Return analysis in JSON format with all fields."

        return prompt

    def _make_vision_request(
        self,
        image_base64: str,
        prompt: str
    ) -> Optional[str]:
        """Make Vision API request."""
        try:
            messages = [
                {"role": "system", "content": self.prompt_manager.get_prompt("chart_system")},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high"  # High detail for better analysis
                            }
                        }
                    ]
                }
            ]

            response = self.client.chat_completion(
                messages,
                model=self.vision_model,
                temperature=0.3,
                max_tokens=1500
            )

            return response

        except Exception as e:
            logger.error(f"Vision API request failed: {e}")
            return None

    def _parse_chart_response(self, response: str) -> Dict[str, Any]:
        """Parse chart analysis response."""
        # Try to extract JSON
        result = self.validator.extract_json_from_response(response)

        if result:
            return result

        # Fallback parsing if JSON extraction failed
        logger.warning("Failed to parse JSON from chart analysis")
        return self._get_fallback_analysis()

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse generic JSON response."""
        result = self.validator.extract_json_from_response(response)

        if result:
            return result

        try:
            return json.loads(response)
        except Exception:
            logger.warning("Failed to parse JSON response")
            return {"error": "parse_failed", "raw_response": response[:200]}

    def _get_fallback_analysis(self) -> Dict[str, Any]:
        """Return fallback analysis when Vision API unavailable."""
        return {
            "trend": "unknown",
            "trend_strength": 0.0,
            "patterns": [],
            "support_levels": [],
            "resistance_levels": [],
            "signals": [],
            "analysis": "Chart analysis unavailable - OpenAI Vision API not accessible",
            "fallback": True
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "vision_model": self.vision_model,
            "pil_available": PIL_AVAILABLE,
            "client_stats": self.client.get_usage_stats()
        }
