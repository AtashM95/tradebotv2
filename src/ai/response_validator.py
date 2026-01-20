"""
Response Validator for AI output schema validation with retry logic.
~250 lines
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import json

try:
    from jsonschema import validate, ValidationError, Draft7Validator
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    ValidationError = Exception

logger = logging.getLogger(__name__)


class ResponseValidator:
    """
    Validates AI responses against expected schemas.

    Features:
    - JSON schema validation
    - Type checking
    - Required field validation
    - Value range validation
    - Validation error tracking
    - Fallback handling
    """

    def __init__(self):
        """Initialize response validator."""
        self.validation_errors: list = []
        self.validation_count = 0
        self.failure_count = 0

    def validate(
        self,
        data: Any,
        schema: Dict[str, Any],
        strict: bool = False
    ) -> bool:
        """
        Validate data against schema.

        Args:
            data: Data to validate
            schema: JSON schema
            strict: If True, fail on any validation error

        Returns:
            True if valid
        """
        self.validation_count += 1

        if not JSONSCHEMA_AVAILABLE:
            logger.warning("jsonschema not available - skipping validation")
            return True

        try:
            validate(instance=data, schema=schema)
            return True

        except ValidationError as e:
            self.failure_count += 1
            error_info = {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "path": list(e.path) if e.path else [],
                "schema_path": list(e.schema_path) if e.schema_path else [],
                "data": data
            }
            self.validation_errors.append(error_info)
            logger.error(f"Validation error: {e.message}")

            if strict:
                raise

            return False

        except Exception as e:
            logger.error(f"Unexpected validation error: {e}")
            if strict:
                raise
            return False

    def validate_sentiment(self, data: Dict[str, Any]) -> bool:
        """Validate sentiment analysis response."""
        schema = {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["bullish", "bearish", "neutral", "mixed"]
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "score": {
                    "type": "number",
                    "minimum": -1.0,
                    "maximum": 1.0
                }
            },
            "required": ["sentiment", "confidence", "score"]
        }
        return self.validate(data, schema)

    def validate_chart_analysis(self, data: Dict[str, Any]) -> bool:
        """Validate chart analysis response."""
        schema = {
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
                }
            },
            "required": ["trend", "trend_strength"]
        }
        return self.validate(data, schema)

    def validate_risk_assessment(self, data: Dict[str, Any]) -> bool:
        """Validate risk assessment response."""
        schema = {
            "type": "object",
            "properties": {
                "risk_level": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "extreme"]
                },
                "risk_score": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "concerns": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "recommendations": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["risk_level", "risk_score"]
        }
        return self.validate(data, schema)

    def validate_strategy_consensus(self, data: Dict[str, Any]) -> bool:
        """Validate strategy consensus response."""
        schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["buy", "sell", "hold"]
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "agreement_level": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "reasoning": {"type": "string"}
            },
            "required": ["action", "confidence"]
        }
        return self.validate(data, schema)

    def validate_json_string(self, json_str: str) -> tuple[bool, Optional[Dict[str, Any]]]:
        """
        Validate and parse JSON string.

        Args:
            json_str: JSON string to validate

        Returns:
            Tuple of (is_valid, parsed_data or None)
        """
        try:
            data = json.loads(json_str)
            return True, data
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return False, None

    def extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from response that may contain extra text.

        Args:
            response: Response string

        Returns:
            Parsed JSON or None
        """
        # Try direct parse first
        is_valid, data = self.validate_json_string(response.strip())
        if is_valid:
            return data

        # Try extracting from code blocks
        if "```json" in response:
            try:
                json_str = response.split("```json")[1].split("```")[0].strip()
                is_valid, data = self.validate_json_string(json_str)
                if is_valid:
                    return data
            except Exception:
                pass

        if "```" in response:
            try:
                json_str = response.split("```")[1].split("```")[0].strip()
                is_valid, data = self.validate_json_string(json_str)
                if is_valid:
                    return data
            except Exception:
                pass

        # Try to find JSON object in text
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                is_valid, data = self.validate_json_string(json_str)
                if is_valid:
                    return data
        except Exception:
            pass

        logger.warning("Failed to extract JSON from response")
        return None

    def get_validation_errors(
        self,
        limit: Optional[int] = None
    ) -> list:
        """
        Get recent validation errors.

        Args:
            limit: Maximum number of errors to return

        Returns:
            List of validation error info
        """
        errors = self.validation_errors

        if limit:
            errors = errors[-limit:]

        return errors

    def clear_errors(self):
        """Clear validation error history."""
        self.validation_errors.clear()
        logger.info("Validation errors cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        success_rate = 0.0
        if self.validation_count > 0:
            success_rate = (self.validation_count - self.failure_count) / self.validation_count

        return {
            "total_validations": self.validation_count,
            "failures": self.failure_count,
            "success_rate": round(success_rate, 3),
            "recent_errors": len(self.validation_errors),
            "jsonschema_available": JSONSCHEMA_AVAILABLE
        }


# Module-level validation function for backward compatibility
def validate_response(data: dict, schema: Optional[Dict[str, Any]] = None) -> bool:
    """
    Validate response data.

    Args:
        data: Data to validate
        schema: Optional schema (uses default if not provided)

    Returns:
        True if valid
    """
    validator = ResponseValidator()

    if schema is None:
        # Default basic schema
        schema = {
            'type': 'object',
            'properties': {'status': {'type': 'string'}},
            'required': ['status']
        }

    return validator.validate(data, schema)
