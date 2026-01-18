"""
AI Package for Ultimate Trading Bot v2.2.

This package provides AI-powered analysis, signal generation,
and trading assistance using OpenAI's GPT models.

Modules:
    openai_client: OpenAI API client with rate limiting
    prompt_manager: Trading-specific prompt templates
    ai_analyzer: Market and sentiment analysis
    ai_signal_generator: Trading signal generation
    chat_assistant: Conversational trading assistant
    ai_risk_assessor: AI-powered risk assessment
    ai_strategy_advisor: Strategy recommendations
    ai_model_manager: Model selection and management
    ai_embeddings: Text embeddings and similarity
    ai_context: Conversation context management
    ai_cache: Response caching for optimization
"""

from src.ai.openai_client import (
    OpenAIClient,
    OpenAIClientConfig,
    OpenAIModel,
    ChatMessage,
    ChatResponse,
    FunctionCall,
    TokenUsage,
)

from src.ai.prompt_manager import (
    PromptManager,
    PromptTemplate,
    PromptCategory,
)

from src.ai.ai_analyzer import (
    AIAnalyzer,
    AIAnalyzerConfig,
    SentimentResult,
    TechnicalResult,
    SignalResult,
    RiskResult,
    MarketOverviewResult,
)

from src.ai.ai_signal_generator import (
    AISignalGenerator,
    AISignalGeneratorConfig,
    AISignal,
    SignalDirection,
    SignalStrength,
    SignalTimeframe,
)

from src.ai.chat_assistant import (
    ChatAssistant,
    ChatAssistantConfig,
    ChatRole,
    ChatMessage as AssistantMessage,
    Conversation,
)

from src.ai.ai_risk_assessor import (
    AIRiskAssessor,
    AIRiskAssessorConfig,
    RiskLevel,
    RiskCategory,
    RiskFactor,
    TradeRiskAssessment,
    PositionRiskAssessment,
    PortfolioRiskAssessment,
)

from src.ai.ai_strategy_advisor import (
    AIStrategyAdvisor,
    AIStrategyAdvisorConfig,
    MarketRegime,
    StrategyType,
    TimeHorizon,
    StrategyRecommendation,
    MarketAnalysis,
    StrategyOptimization,
    SymbolRecommendation,
)

from src.ai.ai_model_manager import (
    AIModelManager,
    AIModelManagerConfig,
    ModelCapability,
    ModelTier,
    TaskComplexity,
    ModelStatus,
    ModelInfo,
    ModelPerformance,
    ModelSelection,
)

from src.ai.ai_embeddings import (
    AIEmbeddings,
    AIEmbeddingsConfig,
    EmbeddingModel,
    EmbeddingResult,
    SimilarityResult,
    EmbeddingDocument,
)

from src.ai.ai_context import (
    AIContextManager,
    AIContextConfig,
    ContextType,
    MessageRole,
    ContextMessage,
    ConversationContext,
    MarketContext,
    TradingContext,
    UserContext,
)

from src.ai.ai_cache import (
    AICache,
    AICacheConfig,
    CacheStrategy,
    CachePriority,
    CacheEntry,
    CacheStats,
    SemanticCache,
)


__all__ = [
    # OpenAI Client
    "OpenAIClient",
    "OpenAIClientConfig",
    "OpenAIModel",
    "ChatMessage",
    "ChatResponse",
    "FunctionCall",
    "TokenUsage",
    # Prompt Manager
    "PromptManager",
    "PromptTemplate",
    "PromptCategory",
    # AI Analyzer
    "AIAnalyzer",
    "AIAnalyzerConfig",
    "SentimentResult",
    "TechnicalResult",
    "SignalResult",
    "RiskResult",
    "MarketOverviewResult",
    # AI Signal Generator
    "AISignalGenerator",
    "AISignalGeneratorConfig",
    "AISignal",
    "SignalDirection",
    "SignalStrength",
    "SignalTimeframe",
    # Chat Assistant
    "ChatAssistant",
    "ChatAssistantConfig",
    "ChatRole",
    "AssistantMessage",
    "Conversation",
    # AI Risk Assessor
    "AIRiskAssessor",
    "AIRiskAssessorConfig",
    "RiskLevel",
    "RiskCategory",
    "RiskFactor",
    "TradeRiskAssessment",
    "PositionRiskAssessment",
    "PortfolioRiskAssessment",
    # AI Strategy Advisor
    "AIStrategyAdvisor",
    "AIStrategyAdvisorConfig",
    "MarketRegime",
    "StrategyType",
    "TimeHorizon",
    "StrategyRecommendation",
    "MarketAnalysis",
    "StrategyOptimization",
    "SymbolRecommendation",
    # AI Model Manager
    "AIModelManager",
    "AIModelManagerConfig",
    "ModelCapability",
    "ModelTier",
    "TaskComplexity",
    "ModelStatus",
    "ModelInfo",
    "ModelPerformance",
    "ModelSelection",
    # AI Embeddings
    "AIEmbeddings",
    "AIEmbeddingsConfig",
    "EmbeddingModel",
    "EmbeddingResult",
    "SimilarityResult",
    "EmbeddingDocument",
    # AI Context
    "AIContextManager",
    "AIContextConfig",
    "ContextType",
    "MessageRole",
    "ContextMessage",
    "ConversationContext",
    "MarketContext",
    "TradingContext",
    "UserContext",
    # AI Cache
    "AICache",
    "AICacheConfig",
    "CacheStrategy",
    "CachePriority",
    "CacheEntry",
    "CacheStats",
    "SemanticCache",
]
