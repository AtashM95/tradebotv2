"""
Entity Extractor for Ultimate Trading Bot v2.2.

Extracts named entities, stock tickers, and financial entities from text.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    """Types of extracted entities."""
    TICKER = "ticker"
    COMPANY = "company"
    PERSON = "person"
    MONEY = "money"
    PERCENTAGE = "percentage"
    DATE = "date"
    QUANTITY = "quantity"
    ORGANIZATION = "organization"
    PRODUCT = "product"
    METRIC = "metric"
    INDUSTRY = "industry"
    LOCATION = "location"


@dataclass
class Entity:
    """Represents an extracted entity."""

    text: str
    entity_type: EntityType
    start: int
    end: int
    confidence: float = 1.0
    normalized: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "type": self.entity_type.value,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "normalized": self.normalized,
            "metadata": self.metadata,
        }


@dataclass
class ExtractionResult:
    """Result of entity extraction."""

    text: str
    entities: list[Entity]
    tickers: list[str]
    companies: list[str]
    people: list[str]
    money_values: list[tuple[str, float]]
    percentages: list[tuple[str, float]]
    dates: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def entity_count(self) -> int:
        """Get total entity count."""
        return len(self.entities)

    def get_entities_by_type(self, entity_type: EntityType) -> list[Entity]:
        """Get entities of a specific type."""
        return [e for e in self.entities if e.entity_type == entity_type]


class EntityExtractor:
    """
    Extracts named entities from financial text.

    Uses pattern matching and heuristics optimized for financial content.
    """

    # US stock ticker pattern (1-5 uppercase letters, optionally with . for classes)
    TICKER_PATTERN = re.compile(r'\$([A-Z]{1,5}(?:\.[A-Z])?)\b')
    TICKER_NO_DOLLAR = re.compile(r'\b([A-Z]{2,5})\b')

    # Money patterns
    MONEY_PATTERNS = [
        re.compile(r'\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(billion|million|thousand|B|M|K)?', re.I),
        re.compile(r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(billion|million|thousand)?\s*(?:dollars|USD)', re.I),
    ]

    # Percentage pattern
    PERCENTAGE_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*%')

    # Date patterns
    DATE_PATTERNS = [
        re.compile(r'\b(Q[1-4])\s*(\d{4})\b'),  # Q1 2024
        re.compile(r'\b(FY)\s*(\d{4})\b'),  # FY2024
        re.compile(r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2}),?\s*(\d{4})?\b', re.I),
        re.compile(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b'),
    ]

    # Common company suffixes
    COMPANY_SUFFIXES = [
        "Inc", "Corp", "Corporation", "Ltd", "Limited", "LLC", "LP",
        "Company", "Co", "Group", "Holdings", "Partners", "Associates",
        "Technologies", "Tech", "Systems", "Solutions", "Services",
        "International", "Global", "Worldwide", "Industries", "Enterprises",
    ]

    # Financial metrics
    FINANCIAL_METRICS = [
        "EPS", "P/E", "PE ratio", "P/B", "PB ratio", "ROE", "ROA", "ROI",
        "EBITDA", "revenue", "profit", "margin", "earnings", "dividend",
        "yield", "market cap", "volume", "price target", "guidance",
        "forecast", "estimate", "beat", "miss", "growth", "decline",
    ]

    # Common ticker/company mappings
    TICKER_COMPANY_MAP = {
        "AAPL": "Apple Inc",
        "GOOGL": "Alphabet Inc",
        "GOOG": "Alphabet Inc",
        "MSFT": "Microsoft Corporation",
        "AMZN": "Amazon.com Inc",
        "META": "Meta Platforms Inc",
        "TSLA": "Tesla Inc",
        "NVDA": "NVIDIA Corporation",
        "JPM": "JPMorgan Chase & Co",
        "V": "Visa Inc",
        "WMT": "Walmart Inc",
        "JNJ": "Johnson & Johnson",
        "PG": "Procter & Gamble Co",
        "UNH": "UnitedHealth Group Inc",
        "HD": "The Home Depot Inc",
        "BAC": "Bank of America Corp",
        "MA": "Mastercard Inc",
        "DIS": "The Walt Disney Co",
        "ADBE": "Adobe Inc",
        "NFLX": "Netflix Inc",
        "CRM": "Salesforce Inc",
        "PYPL": "PayPal Holdings Inc",
        "INTC": "Intel Corporation",
        "AMD": "Advanced Micro Devices Inc",
        "CSCO": "Cisco Systems Inc",
        "PEP": "PepsiCo Inc",
        "KO": "The Coca-Cola Co",
        "NKE": "Nike Inc",
        "MRK": "Merck & Co Inc",
        "ABBV": "AbbVie Inc",
        "TMO": "Thermo Fisher Scientific Inc",
        "COST": "Costco Wholesale Corp",
        "AVGO": "Broadcom Inc",
        "ACN": "Accenture plc",
        "TXN": "Texas Instruments Inc",
        "QCOM": "QUALCOMM Inc",
        "LLY": "Eli Lilly and Co",
        "MCD": "McDonald's Corp",
        "DHR": "Danaher Corporation",
        "BMY": "Bristol-Myers Squibb Co",
        "UPS": "United Parcel Service Inc",
        "NEE": "NextEra Energy Inc",
        "HON": "Honeywell International Inc",
        "UNP": "Union Pacific Corp",
        "IBM": "International Business Machines",
        "SBUX": "Starbucks Corp",
        "GS": "The Goldman Sachs Group Inc",
        "BA": "The Boeing Co",
        "CAT": "Caterpillar Inc",
        "GE": "General Electric Co",
        "MMM": "3M Company",
    }

    # Words that look like tickers but aren't
    TICKER_BLACKLIST = {
        "CEO", "CFO", "COO", "CTO", "IPO", "ETF", "NYSE", "NASDAQ",
        "SEC", "GDP", "FBI", "USA", "USD", "EUR", "GBP", "JPY",
        "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL",
        "CAN", "HAD", "HER", "WAS", "ONE", "OUR", "OUT", "HAS",
        "HIS", "HOW", "ITS", "MAY", "NEW", "NOW", "OLD", "SEE",
        "WAY", "WHO", "BOY", "DID", "GET", "LET", "PUT", "SAY",
        "SHE", "TOO", "USE", "AI", "PM", "AM", "UK", "EU", "UN",
        "IT", "TV", "PC", "PR", "HR", "IR", "OS", "ID", "IP",
        "API", "CEO", "CIO", "CMO", "CSO", "CRO", "EVP", "SVP",
        "VP", "GM", "MD", "ED", "REV", "DIV", "EST", "ACT",
        "BUY", "SELL", "HOLD", "LONG", "SHORT", "CALL", "PUT",
        "YTD", "QTD", "MTD", "YOY", "QOQ", "MOM", "WSJ", "CNN",
    }

    def __init__(
        self,
        extract_tickers: bool = True,
        extract_companies: bool = True,
        extract_money: bool = True,
        extract_percentages: bool = True,
        extract_dates: bool = True,
        min_ticker_confidence: float = 0.5,
    ) -> None:
        """
        Initialize entity extractor.

        Args:
            extract_tickers: Extract stock tickers
            extract_companies: Extract company names
            extract_money: Extract monetary values
            extract_percentages: Extract percentages
            extract_dates: Extract dates
            min_ticker_confidence: Minimum confidence for ticker extraction
        """
        self.extract_tickers = extract_tickers
        self.extract_companies = extract_companies
        self.extract_money = extract_money
        self.extract_percentages = extract_percentages
        self.extract_dates = extract_dates
        self.min_ticker_confidence = min_ticker_confidence

        # Build company name pattern
        suffix_pattern = "|".join(re.escape(s) for s in self.COMPANY_SUFFIXES)
        self._company_pattern = re.compile(
            rf'\b([A-Z][a-zA-Z&\s]+)\s+({suffix_pattern})\.?\b'
        )

        logger.info("Initialized EntityExtractor")

    def extract(self, text: str) -> ExtractionResult:
        """
        Extract all entities from text.

        Args:
            text: Input text

        Returns:
            Extraction result
        """
        entities: list[Entity] = []
        tickers: list[str] = []
        companies: list[str] = []
        people: list[str] = []
        money_values: list[tuple[str, float]] = []
        percentages: list[tuple[str, float]] = []
        dates: list[str] = []

        # Extract tickers
        if self.extract_tickers:
            ticker_entities = self._extract_tickers(text)
            entities.extend(ticker_entities)
            tickers = [e.normalized or e.text for e in ticker_entities]

        # Extract companies
        if self.extract_companies:
            company_entities = self._extract_companies(text)
            entities.extend(company_entities)
            companies = [e.text for e in company_entities]

        # Extract money
        if self.extract_money:
            money_entities = self._extract_money(text)
            entities.extend(money_entities)
            money_values = [
                (e.text, e.metadata.get("value", 0.0))
                for e in money_entities
            ]

        # Extract percentages
        if self.extract_percentages:
            pct_entities = self._extract_percentages(text)
            entities.extend(pct_entities)
            percentages = [
                (e.text, e.metadata.get("value", 0.0))
                for e in pct_entities
            ]

        # Extract dates
        if self.extract_dates:
            date_entities = self._extract_dates(text)
            entities.extend(date_entities)
            dates = [e.text for e in date_entities]

        # Sort entities by position
        entities.sort(key=lambda e: e.start)

        return ExtractionResult(
            text=text,
            entities=entities,
            tickers=tickers,
            companies=companies,
            people=people,
            money_values=money_values,
            percentages=percentages,
            dates=dates,
        )

    def _extract_tickers(self, text: str) -> list[Entity]:
        """Extract stock tickers from text."""
        entities = []

        # Extract tickers with $ prefix (high confidence)
        for match in self.TICKER_PATTERN.finditer(text):
            ticker = match.group(1)
            if ticker not in self.TICKER_BLACKLIST:
                entities.append(Entity(
                    text=f"${ticker}",
                    entity_type=EntityType.TICKER,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95,
                    normalized=ticker,
                    metadata={
                        "company": self.TICKER_COMPANY_MAP.get(ticker),
                    },
                ))

        # Extract tickers without $ prefix (lower confidence)
        for match in self.TICKER_NO_DOLLAR.finditer(text):
            ticker = match.group(1)

            # Skip if already found with $
            if any(e.normalized == ticker for e in entities):
                continue

            # Skip blacklisted words
            if ticker in self.TICKER_BLACKLIST:
                continue

            # Calculate confidence based on context
            confidence = self._calculate_ticker_confidence(text, match)

            if confidence >= self.min_ticker_confidence:
                entities.append(Entity(
                    text=ticker,
                    entity_type=EntityType.TICKER,
                    start=match.start(),
                    end=match.end(),
                    confidence=confidence,
                    normalized=ticker,
                    metadata={
                        "company": self.TICKER_COMPANY_MAP.get(ticker),
                    },
                ))

        return entities

    def _calculate_ticker_confidence(
        self,
        text: str,
        match: re.Match,
    ) -> float:
        """Calculate confidence that a match is a ticker."""
        ticker = match.group(1)
        confidence = 0.4  # Base confidence

        # Known ticker
        if ticker in self.TICKER_COMPANY_MAP:
            confidence += 0.4

        # Check surrounding context
        start = max(0, match.start() - 50)
        end = min(len(text), match.end() + 50)
        context = text[start:end].lower()

        # Financial context indicators
        context_indicators = [
            "stock", "share", "price", "trade", "buy", "sell",
            "market", "trading", "investor", "analyst",
        ]

        for indicator in context_indicators:
            if indicator in context:
                confidence += 0.1
                break

        # Check if it's at the start of a sentence (less likely ticker)
        if match.start() == 0 or text[match.start()-1] in '.!?\n':
            confidence -= 0.1

        return min(0.95, confidence)

    def _extract_companies(self, text: str) -> list[Entity]:
        """Extract company names from text."""
        entities = []

        for match in self._company_pattern.finditer(text):
            company_name = match.group(0).strip()

            entities.append(Entity(
                text=company_name,
                entity_type=EntityType.COMPANY,
                start=match.start(),
                end=match.end(),
                confidence=0.85,
            ))

        # Also extract companies from known ticker mappings
        for ticker, company in self.TICKER_COMPANY_MAP.items():
            if company in text:
                start = text.find(company)
                if start >= 0:
                    entities.append(Entity(
                        text=company,
                        entity_type=EntityType.COMPANY,
                        start=start,
                        end=start + len(company),
                        confidence=0.95,
                        metadata={"ticker": ticker},
                    ))

        return entities

    def _extract_money(self, text: str) -> list[Entity]:
        """Extract monetary values from text."""
        entities = []

        for pattern in self.MONEY_PATTERNS:
            for match in pattern.finditer(text):
                value_str = match.group(0)
                amount_str = match.group(1).replace(',', '')
                multiplier_str = match.group(2) if len(match.groups()) > 1 else None

                try:
                    amount = float(amount_str)
                except ValueError:
                    continue

                # Apply multiplier
                if multiplier_str:
                    mult_lower = multiplier_str.lower()
                    if mult_lower in ['billion', 'b']:
                        amount *= 1_000_000_000
                    elif mult_lower in ['million', 'm']:
                        amount *= 1_000_000
                    elif mult_lower in ['thousand', 'k']:
                        amount *= 1_000

                entities.append(Entity(
                    text=value_str,
                    entity_type=EntityType.MONEY,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9,
                    metadata={"value": amount},
                ))

        return entities

    def _extract_percentages(self, text: str) -> list[Entity]:
        """Extract percentages from text."""
        entities = []

        for match in self.PERCENTAGE_PATTERN.finditer(text):
            pct_str = match.group(0)
            try:
                value = float(match.group(1))
            except ValueError:
                continue

            entities.append(Entity(
                text=pct_str,
                entity_type=EntityType.PERCENTAGE,
                start=match.start(),
                end=match.end(),
                confidence=0.95,
                metadata={"value": value},
            ))

        return entities

    def _extract_dates(self, text: str) -> list[Entity]:
        """Extract dates from text."""
        entities = []

        for pattern in self.DATE_PATTERNS:
            for match in pattern.finditer(text):
                date_str = match.group(0)

                entities.append(Entity(
                    text=date_str,
                    entity_type=EntityType.DATE,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.85,
                ))

        return entities

    def extract_tickers_only(self, text: str) -> list[str]:
        """
        Extract only stock tickers from text.

        Args:
            text: Input text

        Returns:
            List of ticker symbols
        """
        entities = self._extract_tickers(text)
        return [e.normalized or e.text.lstrip('$') for e in entities]

    def get_ticker_company(self, ticker: str) -> str | None:
        """
        Get company name for a ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            Company name or None
        """
        return self.TICKER_COMPANY_MAP.get(ticker.upper())

    def add_ticker_mapping(self, ticker: str, company: str) -> None:
        """
        Add a ticker to company mapping.

        Args:
            ticker: Ticker symbol
            company: Company name
        """
        self.TICKER_COMPANY_MAP[ticker.upper()] = company


class FinancialEntityExtractor(EntityExtractor):
    """
    Enhanced entity extractor for financial text.

    Includes extraction of financial metrics and specialized terms.
    """

    # Financial metric patterns
    METRIC_PATTERNS = [
        (re.compile(r'EPS\s+of\s+\$?(\d+(?:\.\d+)?)', re.I), "EPS"),
        (re.compile(r'P/?E\s+(?:ratio\s+)?(?:of\s+)?(\d+(?:\.\d+)?)', re.I), "PE"),
        (re.compile(r'revenue\s+of\s+\$?(\d+(?:\.\d+)?)\s*(billion|million)?', re.I), "revenue"),
        (re.compile(r'profit\s+of\s+\$?(\d+(?:\.\d+)?)\s*(billion|million)?', re.I), "profit"),
        (re.compile(r'margin\s+of\s+(\d+(?:\.\d+)?)\s*%', re.I), "margin"),
        (re.compile(r'dividend\s+(?:yield\s+)?of\s+(\d+(?:\.\d+)?)\s*%', re.I), "dividend"),
        (re.compile(r'market\s+cap\s+of\s+\$?(\d+(?:\.\d+)?)\s*(billion|million|trillion)?', re.I), "market_cap"),
    ]

    def __init__(self, **kwargs: Any) -> None:
        """Initialize financial entity extractor."""
        super().__init__(**kwargs)
        logger.info("Initialized FinancialEntityExtractor")

    def extract(self, text: str) -> ExtractionResult:
        """Extract entities including financial metrics."""
        result = super().extract(text)

        # Extract financial metrics
        metrics = self._extract_metrics(text)
        result.entities.extend(metrics)
        result.metadata["metrics"] = [
            {"name": e.metadata.get("metric_name"), "value": e.metadata.get("value")}
            for e in metrics
        ]

        return result

    def _extract_metrics(self, text: str) -> list[Entity]:
        """Extract financial metrics."""
        entities = []

        for pattern, metric_name in self.METRIC_PATTERNS:
            for match in pattern.finditer(text):
                try:
                    value = float(match.group(1))

                    # Apply multiplier if present
                    if len(match.groups()) > 1 and match.group(2):
                        mult = match.group(2).lower()
                        if mult == 'trillion':
                            value *= 1_000_000_000_000
                        elif mult == 'billion':
                            value *= 1_000_000_000
                        elif mult == 'million':
                            value *= 1_000_000

                    entities.append(Entity(
                        text=match.group(0),
                        entity_type=EntityType.METRIC,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.85,
                        metadata={
                            "metric_name": metric_name,
                            "value": value,
                        },
                    ))
                except (ValueError, IndexError):
                    continue

        return entities


def create_entity_extractor(
    extractor_type: str = "standard",
    **kwargs: Any,
) -> EntityExtractor:
    """
    Factory function to create entity extractors.

    Args:
        extractor_type: Type of extractor
        **kwargs: Additional arguments

    Returns:
        Entity extractor instance
    """
    extractors = {
        "standard": EntityExtractor,
        "financial": FinancialEntityExtractor,
    }

    extractor_class = extractors.get(extractor_type, EntityExtractor)
    return extractor_class(**kwargs)
