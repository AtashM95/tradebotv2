"""
Text Processor for Ultimate Trading Bot v2.2.

Provides NLP text preprocessing and cleaning utilities for sentiment analysis.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for text processing."""

    # Basic cleaning
    lowercase: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_html: bool = True
    remove_mentions: bool = True
    remove_hashtags: bool = False
    expand_hashtags: bool = True

    # Character handling
    remove_punctuation: bool = False
    remove_numbers: bool = False
    remove_emojis: bool = False
    normalize_unicode: bool = True

    # Tokenization
    min_word_length: int = 2
    max_word_length: int = 50
    min_sentence_length: int = 3

    # Financial specific
    preserve_tickers: bool = True
    preserve_numbers_in_context: bool = True
    expand_contractions: bool = True

    # Stop words
    remove_stopwords: bool = False
    custom_stopwords: set[str] = field(default_factory=set)


@dataclass
class ProcessedText:
    """Result of text processing."""

    original: str
    cleaned: str
    tokens: list[str]
    sentences: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    # Extracted entities
    urls: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    hashtags: list[str] = field(default_factory=list)
    tickers: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)
    emojis: list[str] = field(default_factory=list)


class TextProcessor:
    """
    Comprehensive text processor for sentiment analysis.

    Handles cleaning, normalization, and tokenization of text.
    """

    # Common contractions
    CONTRACTIONS = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "i'd": "i would",
        "i'll": "i will",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it's": "it is",
        "let's": "let us",
        "mightn't": "might not",
        "mustn't": "must not",
        "shan't": "shall not",
        "she'd": "she would",
        "she'll": "she will",
        "she's": "she is",
        "shouldn't": "should not",
        "that's": "that is",
        "there's": "there is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'll": "we will",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what's": "what is",
        "where's": "where is",
        "who's": "who is",
        "won't": "will not",
        "wouldn't": "would not",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have",
    }

    # Basic English stop words
    STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "or", "that",
        "the", "to", "was", "were", "will", "with", "the", "this", "but",
        "they", "have", "had", "what", "when", "where", "who", "which",
        "why", "how", "all", "each", "every", "both", "few", "more",
        "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "just", "should", "now",
    }

    # Regex patterns
    URL_PATTERN = re.compile(
        r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*|'
        r'www\.[-\w.]+[^\s]*'
    )
    EMAIL_PATTERN = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')
    HTML_PATTERN = re.compile(r'<[^>]+>')
    MENTION_PATTERN = re.compile(r'@\w+')
    HASHTAG_PATTERN = re.compile(r'#\w+')
    TICKER_PATTERN = re.compile(r'\$[A-Z]{1,5}(?:\.[A-Z])?')
    NUMBER_PATTERN = re.compile(r'\b\d+(?:,\d{3})*(?:\.\d+)?%?\b')
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    WHITESPACE_PATTERN = re.compile(r'\s+')
    PUNCTUATION_PATTERN = re.compile(r'[^\w\s$#@]')
    REPEATED_CHARS_PATTERN = re.compile(r'(.)\1{2,}')

    def __init__(self, config: ProcessingConfig | None = None) -> None:
        """
        Initialize text processor.

        Args:
            config: Processing configuration
        """
        self.config = config or ProcessingConfig()
        self._stop_words = self.STOP_WORDS.union(self.config.custom_stopwords)

        logger.info("Initialized TextProcessor")

    def process(self, text: str) -> ProcessedText:
        """
        Process text with full pipeline.

        Args:
            text: Raw text input

        Returns:
            Processed text result
        """
        original = text

        # Extract entities before cleaning
        urls = self.URL_PATTERN.findall(text)
        emails = self.EMAIL_PATTERN.findall(text)
        mentions = self.MENTION_PATTERN.findall(text)
        hashtags = self.HASHTAG_PATTERN.findall(text)
        tickers = self.TICKER_PATTERN.findall(text)
        numbers = self.NUMBER_PATTERN.findall(text)
        emojis = self.EMOJI_PATTERN.findall(text)

        # Clean text
        cleaned = self.clean(text)

        # Tokenize
        tokens = self.tokenize(cleaned)

        # Split into sentences
        sentences = self.split_sentences(cleaned)

        return ProcessedText(
            original=original,
            cleaned=cleaned,
            tokens=tokens,
            sentences=sentences,
            urls=urls,
            mentions=[m.lstrip('@') for m in mentions],
            hashtags=[h.lstrip('#') for h in hashtags],
            tickers=[t.lstrip('$') for t in tickers],
            numbers=numbers,
            emojis=emojis,
            metadata={
                "url_count": len(urls),
                "mention_count": len(mentions),
                "hashtag_count": len(hashtags),
                "ticker_count": len(tickers),
                "word_count": len(tokens),
                "sentence_count": len(sentences),
            },
        )

    def clean(self, text: str) -> str:
        """
        Clean text by removing noise.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Normalize unicode
        if self.config.normalize_unicode:
            text = unicodedata.normalize('NFKD', text)
            text = text.encode('ascii', 'ignore').decode('utf-8')

        # Remove HTML
        if self.config.remove_html:
            text = self.HTML_PATTERN.sub(' ', text)

        # Remove URLs
        if self.config.remove_urls:
            text = self.URL_PATTERN.sub(' ', text)

        # Remove emails
        if self.config.remove_emails:
            text = self.EMAIL_PATTERN.sub(' ', text)

        # Handle hashtags
        if self.config.remove_hashtags:
            text = self.HASHTAG_PATTERN.sub(' ', text)
        elif self.config.expand_hashtags:
            text = self._expand_hashtags(text)

        # Handle mentions
        if self.config.remove_mentions:
            text = self.MENTION_PATTERN.sub(' ', text)

        # Handle emojis
        if self.config.remove_emojis:
            text = self.EMOJI_PATTERN.sub(' ', text)

        # Expand contractions
        if self.config.expand_contractions:
            text = self._expand_contractions(text)

        # Handle numbers
        if self.config.remove_numbers and not self.config.preserve_numbers_in_context:
            text = self.NUMBER_PATTERN.sub(' ', text)

        # Handle punctuation
        if self.config.remove_punctuation:
            # Preserve tickers and hashtags
            if self.config.preserve_tickers:
                tickers = self.TICKER_PATTERN.findall(text)
                text = self.PUNCTUATION_PATTERN.sub(' ', text)
                # Restore tickers
                for ticker in tickers:
                    text = text.replace(ticker.replace('$', ''), ticker)
            else:
                text = self.PUNCTUATION_PATTERN.sub(' ', text)

        # Normalize repeated characters (e.g., "gooood" -> "good")
        text = self.REPEATED_CHARS_PATTERN.sub(r'\1\1', text)

        # Normalize whitespace
        text = self.WHITESPACE_PATTERN.sub(' ', text)

        # Lowercase
        if self.config.lowercase:
            text = text.lower()

        return text.strip()

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text into words.

        Args:
            text: Cleaned text

        Returns:
            List of tokens
        """
        if not text:
            return []

        # Split on whitespace
        tokens = text.split()

        # Filter tokens
        filtered = []
        for token in tokens:
            # Check length
            if len(token) < self.config.min_word_length:
                continue
            if len(token) > self.config.max_word_length:
                continue

            # Check stop words
            if self.config.remove_stopwords and token.lower() in self._stop_words:
                continue

            filtered.append(token)

        return filtered

    def split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences.

        Args:
            text: Cleaned text

        Returns:
            List of sentences
        """
        if not text:
            return []

        # Split on sentence boundaries
        sentence_endings = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_endings.split(text)

        # Filter short sentences
        filtered = [
            s.strip()
            for s in sentences
            if len(s.split()) >= self.config.min_sentence_length
        ]

        return filtered

    def _expand_hashtags(self, text: str) -> str:
        """
        Expand hashtags to separate words.

        Args:
            text: Text with hashtags

        Returns:
            Text with expanded hashtags
        """
        def expand(match: re.Match) -> str:
            hashtag = match.group(0)[1:]  # Remove #

            # Try to split camelCase
            words = re.sub(r'([a-z])([A-Z])', r'\1 \2', hashtag)

            # Try to split on underscores
            words = words.replace('_', ' ')

            return words

        return self.HASHTAG_PATTERN.sub(expand, text)

    def _expand_contractions(self, text: str) -> str:
        """
        Expand contractions to full forms.

        Args:
            text: Text with contractions

        Returns:
            Text with expanded contractions
        """
        words = text.split()
        expanded = []

        for word in words:
            lower = word.lower()
            if lower in self.CONTRACTIONS:
                # Preserve capitalization
                expanded_word = self.CONTRACTIONS[lower]
                if word[0].isupper():
                    expanded_word = expanded_word.capitalize()
                expanded.append(expanded_word)
            else:
                expanded.append(word)

        return ' '.join(expanded)

    def normalize(self, text: str) -> str:
        """
        Normalize text for comparison.

        Args:
            text: Raw text

        Returns:
            Normalized text
        """
        # Basic normalization
        text = text.lower().strip()
        text = self.WHITESPACE_PATTERN.sub(' ', text)
        return text

    def extract_tickers(self, text: str) -> list[str]:
        """
        Extract stock tickers from text.

        Args:
            text: Raw text

        Returns:
            List of tickers
        """
        tickers = self.TICKER_PATTERN.findall(text)
        return [t.lstrip('$').upper() for t in tickers]

    def extract_numbers(self, text: str) -> list[tuple[str, float | None]]:
        """
        Extract numbers with context from text.

        Args:
            text: Raw text

        Returns:
            List of (context, value) tuples
        """
        results = []

        for match in self.NUMBER_PATTERN.finditer(text):
            number_str = match.group()
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            context = text[start:end].strip()

            # Parse number
            try:
                clean_num = number_str.replace(',', '').rstrip('%')
                value = float(clean_num)
                if number_str.endswith('%'):
                    value /= 100
            except ValueError:
                value = None

            results.append((context, value))

        return results

    def get_word_frequencies(self, tokens: list[str]) -> dict[str, int]:
        """
        Get word frequency counts.

        Args:
            tokens: List of tokens

        Returns:
            Dictionary of word frequencies
        """
        frequencies: dict[str, int] = {}
        for token in tokens:
            lower = token.lower()
            frequencies[lower] = frequencies.get(lower, 0) + 1
        return frequencies

    def get_ngrams(
        self,
        tokens: list[str],
        n: int = 2,
    ) -> list[tuple[str, ...]]:
        """
        Extract n-grams from tokens.

        Args:
            tokens: List of tokens
            n: N-gram size

        Returns:
            List of n-gram tuples
        """
        if len(tokens) < n:
            return []

        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


class FinancialTextProcessor(TextProcessor):
    """
    Specialized text processor for financial content.

    Handles finance-specific terminology and patterns.
    """

    # Financial abbreviations
    FINANCIAL_ABBREVIATIONS = {
        "eps": "earnings per share",
        "pe": "price to earnings",
        "pb": "price to book",
        "roi": "return on investment",
        "roe": "return on equity",
        "roa": "return on assets",
        "ebitda": "earnings before interest taxes depreciation amortization",
        "yoy": "year over year",
        "qoq": "quarter over quarter",
        "mom": "month over month",
        "ytd": "year to date",
        "mtd": "month to date",
        "ipo": "initial public offering",
        "m&a": "mergers and acquisitions",
        "ceo": "chief executive officer",
        "cfo": "chief financial officer",
        "sec": "securities exchange commission",
        "etf": "exchange traded fund",
        "nav": "net asset value",
        "aum": "assets under management",
    }

    # Units
    UNIT_PATTERNS = {
        r'\$(\d+(?:\.\d+)?)\s*[Bb](?:illion)?': r'\1 billion dollars',
        r'\$(\d+(?:\.\d+)?)\s*[Mm](?:illion)?': r'\1 million dollars',
        r'\$(\d+(?:\.\d+)?)\s*[Kk]': r'\1 thousand dollars',
        r'(\d+(?:\.\d+)?)\s*[Bb]ps': r'\1 basis points',
    }

    def __init__(self, config: ProcessingConfig | None = None) -> None:
        """
        Initialize financial text processor.

        Args:
            config: Processing configuration
        """
        if config is None:
            config = ProcessingConfig(
                preserve_tickers=True,
                preserve_numbers_in_context=True,
                remove_stopwords=False,
            )
        super().__init__(config)

        # Compile unit patterns
        self._unit_patterns = [
            (re.compile(pattern), replacement)
            for pattern, replacement in self.UNIT_PATTERNS.items()
        ]

        logger.info("Initialized FinancialTextProcessor")

    def process(self, text: str) -> ProcessedText:
        """
        Process financial text with specialized handling.

        Args:
            text: Raw financial text

        Returns:
            Processed text result
        """
        # Pre-process financial specific patterns
        text = self._preprocess_financial(text)

        # Use base processing
        result = super().process(text)

        # Extract financial entities
        result.metadata["financial_terms"] = self._extract_financial_terms(
            result.original
        )

        return result

    def _preprocess_financial(self, text: str) -> str:
        """
        Pre-process financial specific patterns.

        Args:
            text: Raw text

        Returns:
            Pre-processed text
        """
        # Expand unit abbreviations
        for pattern, replacement in self._unit_patterns:
            text = pattern.sub(replacement, text)

        return text

    def _extract_financial_terms(self, text: str) -> list[str]:
        """
        Extract financial terms from text.

        Args:
            text: Raw text

        Returns:
            List of financial terms found
        """
        terms = []
        text_lower = text.lower()

        for abbrev, full in self.FINANCIAL_ABBREVIATIONS.items():
            if abbrev in text_lower or full in text_lower:
                terms.append(abbrev.upper())

        return terms

    def expand_abbreviations(self, text: str) -> str:
        """
        Expand financial abbreviations.

        Args:
            text: Text with abbreviations

        Returns:
            Text with expanded abbreviations
        """
        words = text.split()
        expanded = []

        for word in words:
            lower = word.lower().strip('.,;:!?')
            if lower in self.FINANCIAL_ABBREVIATIONS:
                expanded.append(self.FINANCIAL_ABBREVIATIONS[lower])
            else:
                expanded.append(word)

        return ' '.join(expanded)

    def extract_financial_numbers(
        self,
        text: str,
    ) -> list[dict[str, Any]]:
        """
        Extract financial numbers with context.

        Args:
            text: Raw text

        Returns:
            List of financial number info
        """
        results = []

        # Pattern for financial numbers
        patterns = [
            # Currency amounts
            (r'\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(billion|million|thousand)?',
             'currency'),
            # Percentages
            (r'(\d+(?:\.\d+)?)\s*%', 'percentage'),
            # Basis points
            (r'(\d+(?:\.\d+)?)\s*(?:bps|basis points)', 'basis_points'),
            # Multiples
            (r'(\d+(?:\.\d+)?)\s*[xX]', 'multiple'),
        ]

        for pattern, num_type in patterns:
            for match in re.finditer(pattern, text):
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end].strip()

                results.append({
                    'value': match.group(0),
                    'type': num_type,
                    'context': context,
                    'position': match.start(),
                })

        return results


class BatchTextProcessor:
    """
    Batch text processor for efficient processing of multiple texts.
    """

    def __init__(
        self,
        processor: TextProcessor | None = None,
        batch_size: int = 100,
    ) -> None:
        """
        Initialize batch processor.

        Args:
            processor: Text processor to use
            batch_size: Batch size for processing
        """
        self.processor = processor or TextProcessor()
        self.batch_size = batch_size

        logger.info(f"Initialized BatchTextProcessor with batch_size={batch_size}")

    def process_batch(
        self,
        texts: list[str],
    ) -> list[ProcessedText]:
        """
        Process a batch of texts.

        Args:
            texts: List of texts to process

        Returns:
            List of processed text results
        """
        results = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            for text in batch:
                try:
                    result = self.processor.process(text)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing text: {e}")
                    # Return empty result on error
                    results.append(ProcessedText(
                        original=text,
                        cleaned="",
                        tokens=[],
                        sentences=[],
                        metadata={"error": str(e)},
                    ))

        return results

    def clean_batch(self, texts: list[str]) -> list[str]:
        """
        Clean a batch of texts.

        Args:
            texts: List of texts to clean

        Returns:
            List of cleaned texts
        """
        return [self.processor.clean(text) for text in texts]

    def tokenize_batch(self, texts: list[str]) -> list[list[str]]:
        """
        Tokenize a batch of texts.

        Args:
            texts: List of texts to tokenize

        Returns:
            List of token lists
        """
        results = []
        for text in texts:
            cleaned = self.processor.clean(text)
            tokens = self.processor.tokenize(cleaned)
            results.append(tokens)
        return results


def create_text_processor(
    processor_type: str = "standard",
    config: ProcessingConfig | None = None,
) -> TextProcessor:
    """
    Factory function to create text processors.

    Args:
        processor_type: Type of processor
        config: Processing configuration

    Returns:
        Text processor instance
    """
    processors = {
        "standard": TextProcessor,
        "financial": FinancialTextProcessor,
    }

    if processor_type not in processors:
        raise ValueError(f"Unknown processor type: {processor_type}")

    return processors[processor_type](config)
