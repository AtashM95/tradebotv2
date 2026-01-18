"""
AI Utilities Module for Ultimate Trading Bot v2.2.

This module provides utilities for AI/LLM operations including
token counting, response parsing, prompt formatting, and more.
"""

import json
import re
import base64
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging


logger = logging.getLogger(__name__)


# =============================================================================
# TOKEN ESTIMATION
# =============================================================================

# Approximate characters per token for different models
CHARS_PER_TOKEN: Dict[str, float] = {
    "gpt-4": 4.0,
    "gpt-4o": 4.0,
    "gpt-4o-mini": 4.0,
    "gpt-4-turbo": 4.0,
    "gpt-3.5-turbo": 4.0,
    "claude": 3.5,
    "default": 4.0,
}


def estimate_tokens(text: str, model: str = "default") -> int:
    """
    Estimate the number of tokens in text.

    Args:
        text: Text to estimate tokens for
        model: Model name for calibration

    Returns:
        Estimated token count
    """
    if not text:
        return 0

    chars_per_token = CHARS_PER_TOKEN.get(model, CHARS_PER_TOKEN["default"])
    return max(1, int(len(text) / chars_per_token))


def estimate_tokens_messages(
    messages: List[Dict[str, Any]],
    model: str = "default"
) -> int:
    """
    Estimate tokens for a list of messages.

    Args:
        messages: List of message dicts
        model: Model name

    Returns:
        Estimated token count
    """
    total = 0

    for message in messages:
        # Message overhead
        total += 4  # Every message has role, content separators

        # Role
        role = message.get("role", "")
        total += estimate_tokens(role, model)

        # Content
        content = message.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content, model)
        elif isinstance(content, list):
            # Multimodal content
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        total += estimate_tokens(item.get("text", ""), model)
                    elif item.get("type") == "image_url":
                        # Images use ~85 tokens for low detail, ~765 for high
                        total += 765

    # Reply priming
    total += 3

    return total


def count_tokens_tiktoken(text: str, model: str = "gpt-4o") -> int:
    """
    Count tokens using tiktoken library.

    Args:
        text: Text to count tokens for
        model: Model name

    Returns:
        Token count
    """
    try:
        import tiktoken

        # Get encoding for model
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))
    except ImportError:
        logger.warning("tiktoken not installed, using estimation")
        return estimate_tokens(text, model)


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    model: str = "default",
    suffix: str = "..."
) -> str:
    """
    Truncate text to fit within token limit.

    Args:
        text: Text to truncate
        max_tokens: Maximum tokens
        model: Model name
        suffix: Suffix to add when truncated

    Returns:
        Truncated text
    """
    if not text:
        return text

    current_tokens = estimate_tokens(text, model)
    if current_tokens <= max_tokens:
        return text

    # Binary search for optimal truncation point
    chars_per_token = CHARS_PER_TOKEN.get(model, CHARS_PER_TOKEN["default"])
    target_chars = int(max_tokens * chars_per_token) - len(suffix)

    return text[:target_chars] + suffix


# =============================================================================
# RESPONSE PARSING
# =============================================================================

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from text, handling various formats.

    Args:
        text: Text containing JSON

    Returns:
        Parsed JSON dict or None
    """
    if not text:
        return None

    # Try direct parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in markdown code blocks
    patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
        r'\{[\s\S]*\}',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    logger.warning("Could not extract JSON from text")
    return None


def extract_code_blocks(text: str, language: Optional[str] = None) -> List[str]:
    """
    Extract code blocks from markdown text.

    Args:
        text: Text containing code blocks
        language: Optional language to filter by

    Returns:
        List of code block contents
    """
    if language:
        pattern = rf'```{language}\s*([\s\S]*?)\s*```'
    else:
        pattern = r'```(?:\w+)?\s*([\s\S]*?)\s*```'

    return re.findall(pattern, text)


def parse_structured_response(
    text: str,
    expected_fields: List[str],
    strict: bool = False
) -> Dict[str, Any]:
    """
    Parse a structured response with expected fields.

    Args:
        text: Response text
        expected_fields: List of expected field names
        strict: Whether to require all fields

    Returns:
        Parsed response dict
    """
    result: Dict[str, Any] = {}

    # Try JSON first
    json_data = extract_json(text)
    if json_data:
        for field in expected_fields:
            if field in json_data:
                result[field] = json_data[field]
            elif strict:
                logger.warning(f"Missing required field: {field}")
        return result

    # Try key-value parsing
    for field in expected_fields:
        # Look for patterns like "field: value" or "field = value"
        patterns = [
            rf'{field}[:\s]+([^\n]+)',
            rf'\*\*{field}\*\*[:\s]+([^\n]+)',
            rf'{field}[:\s]*\n([^\n]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result[field] = match.group(1).strip()
                break

    return result


def parse_sentiment_score(text: str) -> Optional[float]:
    """
    Parse a sentiment score from text.

    Args:
        text: Text containing sentiment score

    Returns:
        Sentiment score (-1 to 1) or None
    """
    # Try JSON extraction
    json_data = extract_json(text)
    if json_data:
        for key in ["sentiment_score", "sentiment", "score"]:
            if key in json_data:
                try:
                    score = float(json_data[key])
                    return max(-1.0, min(1.0, score))
                except (TypeError, ValueError):
                    pass

    # Try to find numeric score in text
    patterns = [
        r'sentiment[:\s]*(-?\d+\.?\d*)',
        r'score[:\s]*(-?\d+\.?\d*)',
        r'(-?\d+\.?\d*)\s*(?:out of|\/)?\s*(?:1|10|100)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                # Normalize to -1 to 1 range
                if score > 1:
                    if score <= 10:
                        score = (score - 5) / 5
                    elif score <= 100:
                        score = (score - 50) / 50
                return max(-1.0, min(1.0, score))
            except ValueError:
                continue

    return None


def parse_trading_signal(text: str) -> Dict[str, Any]:
    """
    Parse a trading signal from text.

    Args:
        text: Text containing trading signal

    Returns:
        Parsed signal dict
    """
    result: Dict[str, Any] = {
        "signal": None,
        "direction": None,
        "entry_price": None,
        "stop_loss": None,
        "take_profit": None,
        "confidence": None,
    }

    # Try JSON first
    json_data = extract_json(text)
    if json_data:
        for key in result.keys():
            if key in json_data:
                result[key] = json_data[key]
        return result

    # Parse signal type
    signal_patterns = [
        (r'\b(strong\s+buy|buy|hold|sell|strong\s+sell)\b', "signal"),
        (r'\b(long|short|flat|neutral)\b', "direction"),
    ]

    for pattern, field in signal_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result[field] = match.group(1).lower()

    # Parse prices
    price_patterns = [
        (r'entry[:\s]*\$?(\d+\.?\d*)', "entry_price"),
        (r'stop[:\s]*loss[:\s]*\$?(\d+\.?\d*)', "stop_loss"),
        (r'take[:\s]*profit[:\s]*\$?(\d+\.?\d*)', "take_profit"),
        (r'target[:\s]*\$?(\d+\.?\d*)', "take_profit"),
    ]

    for pattern, field in price_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                result[field] = float(match.group(1))
            except ValueError:
                pass

    # Parse confidence
    conf_match = re.search(r'confidence[:\s]*(\d+\.?\d*)%?', text, re.IGNORECASE)
    if conf_match:
        try:
            conf = float(conf_match.group(1))
            if conf > 1:
                conf = conf / 100
            result["confidence"] = conf
        except ValueError:
            pass

    return result


# =============================================================================
# PROMPT UTILITIES
# =============================================================================

def format_messages(
    system: Optional[str] = None,
    user: Optional[str] = None,
    assistant: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    """
    Format messages for chat completion.

    Args:
        system: System message
        user: User message
        assistant: Assistant message
        history: Previous conversation history

    Returns:
        List of formatted messages
    """
    messages: List[Dict[str, str]] = []

    if system:
        messages.append({"role": "system", "content": system})

    if history:
        messages.extend(history)

    if user:
        messages.append({"role": "user", "content": user})

    if assistant:
        messages.append({"role": "assistant", "content": assistant})

    return messages


def format_context(
    data: Dict[str, Any],
    template: str = "{key}: {value}",
    separator: str = "\n"
) -> str:
    """
    Format context data as a string.

    Args:
        data: Context data
        template: Format template
        separator: Line separator

    Returns:
        Formatted context string
    """
    lines = []
    for key, value in data.items():
        if value is not None:
            formatted_value = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            lines.append(template.format(key=key, value=formatted_value))
    return separator.join(lines)


def create_few_shot_prompt(
    examples: List[Dict[str, str]],
    query: str,
    instruction: Optional[str] = None
) -> str:
    """
    Create a few-shot learning prompt.

    Args:
        examples: List of example dicts with 'input' and 'output'
        query: Query to answer
        instruction: Optional instruction

    Returns:
        Formatted few-shot prompt
    """
    parts = []

    if instruction:
        parts.append(instruction)
        parts.append("")

    for i, example in enumerate(examples, 1):
        parts.append(f"Example {i}:")
        parts.append(f"Input: {example.get('input', '')}")
        parts.append(f"Output: {example.get('output', '')}")
        parts.append("")

    parts.append("Now answer:")
    parts.append(f"Input: {query}")
    parts.append("Output:")

    return "\n".join(parts)


# =============================================================================
# IMAGE UTILITIES
# =============================================================================

def encode_image_base64(image_path: Union[str, Path]) -> str:
    """
    Encode an image file to base64.

    Args:
        image_path: Path to image file

    Returns:
        Base64 encoded string
    """
    path = Path(image_path)
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_image_bytes(image_bytes: bytes) -> str:
    """
    Encode image bytes to base64.

    Args:
        image_bytes: Image bytes

    Returns:
        Base64 encoded string
    """
    return base64.b64encode(image_bytes).decode("utf-8")


def get_image_mime_type(image_path: Union[str, Path]) -> str:
    """
    Get MIME type for an image file.

    Args:
        image_path: Path to image file

    Returns:
        MIME type string
    """
    path = Path(image_path)
    suffix = path.suffix.lower()

    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }

    return mime_types.get(suffix, "image/png")


def create_image_content(
    image_source: Union[str, Path, bytes],
    detail: str = "auto"
) -> Dict[str, Any]:
    """
    Create image content for vision API.

    Args:
        image_source: Image path, URL, or bytes
        detail: Detail level (low, high, auto)

    Returns:
        Image content dict for API
    """
    if isinstance(image_source, bytes):
        # Bytes - encode to base64
        base64_image = encode_image_bytes(image_source)
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}",
                "detail": detail
            }
        }
    elif isinstance(image_source, (str, Path)):
        path = Path(image_source) if isinstance(image_source, str) else image_source
        if path.exists():
            # Local file
            base64_image = encode_image_base64(path)
            mime_type = get_image_mime_type(path)
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_image}",
                    "detail": detail
                }
            }
        else:
            # Assume it's a URL
            return {
                "type": "image_url",
                "image_url": {
                    "url": str(image_source),
                    "detail": detail
                }
            }
    else:
        raise ValueError(f"Unsupported image source type: {type(image_source)}")


# =============================================================================
# COST TRACKING
# =============================================================================

@dataclass
class TokenUsage:
    """Token usage tracking."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: 'TokenUsage') -> 'TokenUsage':
        """Add two TokenUsage objects."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens
        )


@dataclass
class CostEstimate:
    """Cost estimate for API usage."""
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    currency: str = "USD"

    def __add__(self, other: 'CostEstimate') -> 'CostEstimate':
        """Add two CostEstimate objects."""
        return CostEstimate(
            input_cost=self.input_cost + other.input_cost,
            output_cost=self.output_cost + other.output_cost,
            total_cost=self.total_cost + other.total_cost,
            currency=self.currency
        )


# Model pricing per 1M tokens (input, output)
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
}


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int
) -> CostEstimate:
    """
    Estimate cost for API usage.

    Args:
        model: Model name
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens

    Returns:
        CostEstimate object
    """
    pricing = MODEL_PRICING.get(model, (10.0, 30.0))
    input_price, output_price = pricing

    input_cost = (prompt_tokens / 1_000_000) * input_price
    output_cost = (completion_tokens / 1_000_000) * output_price
    total_cost = input_cost + output_cost

    return CostEstimate(
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=total_cost
    )


# =============================================================================
# CACHING UTILITIES
# =============================================================================

def generate_cache_key(
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float = 0.0
) -> str:
    """
    Generate a cache key for API responses.

    Args:
        messages: Messages list
        model: Model name
        temperature: Temperature setting

    Returns:
        Cache key string
    """
    # Only cache deterministic requests (temperature 0)
    if temperature != 0:
        return ""

    key_data = {
        "messages": messages,
        "model": model,
    }

    key_json = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.sha256(key_json.encode()).hexdigest()


# =============================================================================
# SAFETY UTILITIES
# =============================================================================

def sanitize_input(text: str, max_length: int = 10000) -> str:
    """
    Sanitize input text for safety.

    Args:
        text: Input text
        max_length: Maximum allowed length

    Returns:
        Sanitized text
    """
    if not text:
        return ""

    # Truncate to max length
    text = text[:max_length]

    # Remove potential injection patterns
    # (This is basic - real production would need more)
    dangerous_patterns = [
        r'<\|.*?\|>',  # Special tokens
        r'\[INST\].*?\[/INST\]',  # Instruction markers
        r'###.*?###',  # Section markers
    ]

    for pattern in dangerous_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

    return text.strip()


def validate_response(
    response: str,
    expected_format: Optional[str] = None,
    max_length: int = 50000
) -> Tuple[bool, str]:
    """
    Validate an AI response.

    Args:
        response: Response text
        expected_format: Expected format (json, text)
        max_length: Maximum allowed length

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not response:
        return False, "Empty response"

    if len(response) > max_length:
        return False, f"Response exceeds maximum length of {max_length}"

    if expected_format == "json":
        json_data = extract_json(response)
        if json_data is None:
            return False, "Response is not valid JSON"

    return True, ""


def mask_sensitive_data(text: str) -> str:
    """
    Mask sensitive data in text for logging.

    Args:
        text: Text containing sensitive data

    Returns:
        Text with masked sensitive data
    """
    # Mask API keys
    text = re.sub(r'sk-[a-zA-Z0-9]{20,}', 'sk-***MASKED***', text)

    # Mask potential passwords
    text = re.sub(r'password[=:]\s*\S+', 'password=***MASKED***', text, flags=re.IGNORECASE)

    # Mask potential tokens
    text = re.sub(r'token[=:]\s*\S+', 'token=***MASKED***', text, flags=re.IGNORECASE)

    return text
