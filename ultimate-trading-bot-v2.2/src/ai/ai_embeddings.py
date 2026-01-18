"""
AI Embeddings Module for Ultimate Trading Bot v2.2.

This module provides text embedding generation and similarity
search functionality for trading-related content.
"""

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field

from src.ai.openai_client import OpenAIClient
from src.utils.helpers import generate_uuid
from src.utils.date_utils import now_utc


logger = logging.getLogger(__name__)


class EmbeddingModel(str, Enum):
    """Embedding model enumeration."""

    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"


class EmbeddingDimension(int, Enum):
    """Embedding dimension options."""

    SMALL_256 = 256
    SMALL_512 = 512
    SMALL_1536 = 1536
    LARGE_256 = 256
    LARGE_1024 = 1024
    LARGE_3072 = 3072


class EmbeddingResult(BaseModel):
    """Embedding result model."""

    embedding_id: str = Field(default_factory=generate_uuid)
    text: str
    embedding: list[float] = Field(default_factory=list)
    model: EmbeddingModel
    dimensions: int = Field(default=1536)
    tokens_used: int = Field(default=0)
    created_at: datetime = Field(default_factory=now_utc)


class SimilarityResult(BaseModel):
    """Similarity search result."""

    text: str
    score: float = Field(ge=0.0, le=1.0)
    metadata: dict = Field(default_factory=dict)
    embedding_id: str = Field(default="")


class EmbeddingDocument(BaseModel):
    """Document with embedding for storage."""

    doc_id: str = Field(default_factory=generate_uuid)
    text: str
    embedding: list[float] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    category: str = Field(default="general")
    created_at: datetime = Field(default_factory=now_utc)


class AIEmbeddingsConfig(BaseModel):
    """Configuration for AI embeddings."""

    default_model: EmbeddingModel = Field(default=EmbeddingModel.TEXT_EMBEDDING_3_SMALL)
    default_dimensions: int = Field(default=1536, ge=256, le=3072)
    batch_size: int = Field(default=100, ge=1, le=2048)
    cache_embeddings: bool = Field(default=True)
    cache_ttl_hours: int = Field(default=168, ge=1, le=720)
    normalize_embeddings: bool = Field(default=True)
    max_text_length: int = Field(default=8000, ge=100, le=32000)


class AIEmbeddings:
    """
    AI Embeddings manager for text similarity.

    Provides:
    - Text embedding generation
    - Similarity search
    - Document storage and retrieval
    - Batch embedding operations
    """

    def __init__(
        self,
        config: Optional[AIEmbeddingsConfig] = None,
        openai_client: Optional[OpenAIClient] = None,
    ) -> None:
        """
        Initialize AIEmbeddings.

        Args:
            config: Embeddings configuration
            openai_client: OpenAI client instance
        """
        self._config = config or AIEmbeddingsConfig()
        self._client = openai_client

        self._embedding_cache: dict[str, tuple[list[float], datetime]] = {}
        self._document_store: dict[str, EmbeddingDocument] = {}
        self._category_index: dict[str, list[str]] = {}

        self._total_embeddings = 0
        self._cache_hits = 0
        self._total_tokens = 0

        logger.info("AIEmbeddings initialized")

    def set_client(self, client: OpenAIClient) -> None:
        """Set the OpenAI client."""
        self._client = client

    async def embed_text(
        self,
        text: str,
        model: Optional[EmbeddingModel] = None,
        dimensions: Optional[int] = None,
    ) -> EmbeddingResult:
        """
        Generate embedding for text.

        Args:
            text: Text to embed
            model: Embedding model to use
            dimensions: Output dimensions

        Returns:
            EmbeddingResult with embedding vector
        """
        model = model or self._config.default_model
        dimensions = dimensions or self._config.default_dimensions

        text = self._preprocess_text(text)

        cache_key = self._get_cache_key(text, model, dimensions)
        if self._config.cache_embeddings:
            cached = self._get_cached_embedding(cache_key)
            if cached is not None:
                self._cache_hits += 1
                return EmbeddingResult(
                    text=text,
                    embedding=cached,
                    model=model,
                    dimensions=dimensions,
                )

        embedding, tokens = await self._generate_embedding(
            text=text,
            model=model,
            dimensions=dimensions,
        )

        self._total_embeddings += 1
        self._total_tokens += tokens

        if self._config.normalize_embeddings:
            embedding = self._normalize(embedding)

        if self._config.cache_embeddings:
            self._cache_embedding(cache_key, embedding)

        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model=model,
            dimensions=len(embedding),
            tokens_used=tokens,
        )

    async def embed_batch(
        self,
        texts: list[str],
        model: Optional[EmbeddingModel] = None,
        dimensions: Optional[int] = None,
    ) -> list[EmbeddingResult]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            model: Embedding model to use
            dimensions: Output dimensions

        Returns:
            List of EmbeddingResult objects
        """
        model = model or self._config.default_model
        dimensions = dimensions or self._config.default_dimensions

        results: list[EmbeddingResult] = []
        texts_to_embed: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            text = self._preprocess_text(text)
            cache_key = self._get_cache_key(text, model, dimensions)

            if self._config.cache_embeddings:
                cached = self._get_cached_embedding(cache_key)
                if cached is not None:
                    self._cache_hits += 1
                    results.append(EmbeddingResult(
                        text=text,
                        embedding=cached,
                        model=model,
                        dimensions=dimensions,
                    ))
                    continue

            texts_to_embed.append((i, text))

        if texts_to_embed:
            batch_texts = [t[1] for t in texts_to_embed]
            embeddings, total_tokens = await self._generate_batch_embeddings(
                texts=batch_texts,
                model=model,
                dimensions=dimensions,
            )

            self._total_embeddings += len(embeddings)
            self._total_tokens += total_tokens

            for (idx, text), embedding in zip(texts_to_embed, embeddings):
                if self._config.normalize_embeddings:
                    embedding = self._normalize(embedding)

                if self._config.cache_embeddings:
                    cache_key = self._get_cache_key(text, model, dimensions)
                    self._cache_embedding(cache_key, embedding)

                results.append(EmbeddingResult(
                    text=text,
                    embedding=embedding,
                    model=model,
                    dimensions=len(embedding),
                ))

        return results

    async def add_document(
        self,
        text: str,
        metadata: Optional[dict] = None,
        category: str = "general",
        doc_id: Optional[str] = None,
    ) -> EmbeddingDocument:
        """
        Add a document to the store with embedding.

        Args:
            text: Document text
            metadata: Document metadata
            category: Document category
            doc_id: Optional document ID

        Returns:
            EmbeddingDocument with generated embedding
        """
        embedding_result = await self.embed_text(text)

        doc = EmbeddingDocument(
            doc_id=doc_id or generate_uuid(),
            text=text,
            embedding=embedding_result.embedding,
            metadata=metadata or {},
            category=category,
        )

        self._document_store[doc.doc_id] = doc

        if category not in self._category_index:
            self._category_index[category] = []
        self._category_index[category].append(doc.doc_id)

        logger.debug(f"Added document {doc.doc_id} to category {category}")
        return doc

    async def search_similar(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        min_score: float = 0.0,
    ) -> list[SimilarityResult]:
        """
        Search for similar documents.

        Args:
            query: Query text
            top_k: Number of results to return
            category: Filter by category
            min_score: Minimum similarity score

        Returns:
            List of SimilarityResult objects
        """
        query_embedding = await self.embed_text(query)
        query_vector = np.array(query_embedding.embedding)

        if category:
            doc_ids = self._category_index.get(category, [])
        else:
            doc_ids = list(self._document_store.keys())

        scored_results: list[tuple[str, float]] = []

        for doc_id in doc_ids:
            doc = self._document_store.get(doc_id)
            if not doc:
                continue

            doc_vector = np.array(doc.embedding)
            score = self._cosine_similarity(query_vector, doc_vector)

            if score >= min_score:
                scored_results.append((doc_id, score))

        scored_results.sort(key=lambda x: x[1], reverse=True)
        top_results = scored_results[:top_k]

        results: list[SimilarityResult] = []
        for doc_id, score in top_results:
            doc = self._document_store[doc_id]
            results.append(SimilarityResult(
                text=doc.text,
                score=score,
                metadata=doc.metadata,
                embedding_id=doc_id,
            ))

        return results

    async def find_most_similar(
        self,
        text: str,
        candidates: list[str],
    ) -> tuple[str, float]:
        """
        Find the most similar candidate to a text.

        Args:
            text: Source text
            candidates: List of candidate texts

        Returns:
            Tuple of (most similar text, similarity score)
        """
        text_embedding = await self.embed_text(text)
        candidate_embeddings = await self.embed_batch(candidates)

        text_vector = np.array(text_embedding.embedding)

        best_candidate = ""
        best_score = -1.0

        for candidate, emb_result in zip(candidates, candidate_embeddings):
            candidate_vector = np.array(emb_result.embedding)
            score = self._cosine_similarity(text_vector, candidate_vector)

            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate, best_score

    async def compute_similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """
        Compute similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        embeddings = await self.embed_batch([text1, text2])

        if len(embeddings) != 2:
            return 0.0

        vec1 = np.array(embeddings[0].embedding)
        vec2 = np.array(embeddings[1].embedding)

        return self._cosine_similarity(vec1, vec2)

    async def cluster_texts(
        self,
        texts: list[str],
        n_clusters: int = 3,
    ) -> dict[int, list[str]]:
        """
        Cluster texts by semantic similarity.

        Args:
            texts: List of texts to cluster
            n_clusters: Number of clusters

        Returns:
            Dictionary mapping cluster ID to texts
        """
        if len(texts) < n_clusters:
            return {0: texts}

        embeddings = await self.embed_batch(texts)
        vectors = [np.array(e.embedding) for e in embeddings]

        centroids = [vectors[i] for i in range(n_clusters)]

        for iteration in range(10):
            clusters: dict[int, list[int]] = {i: [] for i in range(n_clusters)}

            for idx, vec in enumerate(vectors):
                best_cluster = 0
                best_distance = float("inf")

                for c_idx, centroid in enumerate(centroids):
                    distance = np.linalg.norm(vec - centroid)
                    if distance < best_distance:
                        best_distance = distance
                        best_cluster = c_idx

                clusters[best_cluster].append(idx)

            new_centroids = []
            for c_idx in range(n_clusters):
                if clusters[c_idx]:
                    cluster_vectors = [vectors[i] for i in clusters[c_idx]]
                    new_centroid = np.mean(cluster_vectors, axis=0)
                    new_centroids.append(new_centroid)
                else:
                    new_centroids.append(centroids[c_idx])

            centroids = new_centroids

        result: dict[int, list[str]] = {}
        for c_idx, text_indices in clusters.items():
            result[c_idx] = [texts[i] for i in text_indices]

        return result

    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the store.

        Args:
            doc_id: Document ID to remove

        Returns:
            True if removed, False if not found
        """
        doc = self._document_store.get(doc_id)
        if not doc:
            return False

        del self._document_store[doc_id]

        if doc.category in self._category_index:
            if doc_id in self._category_index[doc.category]:
                self._category_index[doc.category].remove(doc_id)

        logger.debug(f"Removed document {doc_id}")
        return True

    def get_document(self, doc_id: str) -> Optional[EmbeddingDocument]:
        """Get a document by ID."""
        return self._document_store.get(doc_id)

    def list_documents(
        self,
        category: Optional[str] = None,
    ) -> list[EmbeddingDocument]:
        """List all documents, optionally filtered by category."""
        if category:
            doc_ids = self._category_index.get(category, [])
            return [
                self._document_store[doc_id]
                for doc_id in doc_ids
                if doc_id in self._document_store
            ]

        return list(self._document_store.values())

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before embedding."""
        text = text.strip()

        text = " ".join(text.split())

        if len(text) > self._config.max_text_length:
            text = text[:self._config.max_text_length]

        return text

    def _get_cache_key(
        self,
        text: str,
        model: EmbeddingModel,
        dimensions: int,
    ) -> str:
        """Generate cache key for embedding."""
        content = f"{model.value}:{dimensions}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def _get_cached_embedding(
        self,
        cache_key: str,
    ) -> Optional[list[float]]:
        """Get embedding from cache if valid."""
        cached = self._embedding_cache.get(cache_key)
        if not cached:
            return None

        embedding, timestamp = cached
        ttl = timedelta(hours=self._config.cache_ttl_hours)

        if now_utc() - timestamp > ttl:
            del self._embedding_cache[cache_key]
            return None

        return embedding

    def _cache_embedding(
        self,
        cache_key: str,
        embedding: list[float],
    ) -> None:
        """Cache an embedding."""
        self._embedding_cache[cache_key] = (embedding, now_utc())

    async def _generate_embedding(
        self,
        text: str,
        model: EmbeddingModel,
        dimensions: int,
    ) -> tuple[list[float], int]:
        """Generate embedding using OpenAI API."""
        if not self._client:
            raise RuntimeError("OpenAI client not configured")

        try:
            response = await self._client._client.embeddings.create(
                input=text,
                model=model.value,
                dimensions=dimensions if model != EmbeddingModel.TEXT_EMBEDDING_ADA_002 else None,
            )

            embedding = response.data[0].embedding
            tokens = response.usage.total_tokens

            return embedding, tokens

        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            raise

    async def _generate_batch_embeddings(
        self,
        texts: list[str],
        model: EmbeddingModel,
        dimensions: int,
    ) -> tuple[list[list[float]], int]:
        """Generate embeddings for a batch of texts."""
        if not self._client:
            raise RuntimeError("OpenAI client not configured")

        all_embeddings: list[list[float]] = []
        total_tokens = 0

        for i in range(0, len(texts), self._config.batch_size):
            batch = texts[i:i + self._config.batch_size]

            try:
                response = await self._client._client.embeddings.create(
                    input=batch,
                    model=model.value,
                    dimensions=dimensions if model != EmbeddingModel.TEXT_EMBEDDING_ADA_002 else None,
                )

                for data in response.data:
                    all_embeddings.append(data.embedding)

                total_tokens += response.usage.total_tokens

            except Exception as e:
                logger.error(f"Batch embedding error: {e}")
                raise

        return all_embeddings, total_tokens

    def _normalize(self, embedding: list[float]) -> list[float]:
        """Normalize embedding vector to unit length."""
        vec = np.array(embedding)
        norm = np.linalg.norm(vec)

        if norm > 0:
            vec = vec / norm

        return vec.tolist()

    def _cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
    ) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def clear_cache(self) -> int:
        """Clear the embedding cache."""
        count = len(self._embedding_cache)
        self._embedding_cache.clear()
        logger.info(f"Cleared {count} cached embeddings")
        return count

    def get_statistics(self) -> dict:
        """Get embedding statistics."""
        return {
            "total_embeddings": self._total_embeddings,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._embedding_cache),
            "documents_stored": len(self._document_store),
            "categories": list(self._category_index.keys()),
            "total_tokens_used": self._total_tokens,
            "cache_hit_rate": (
                self._cache_hits / (self._total_embeddings + self._cache_hits) * 100
                if (self._total_embeddings + self._cache_hits) > 0 else 0
            ),
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"AIEmbeddings(documents={len(self._document_store)}, "
            f"embeddings={self._total_embeddings})"
        )
