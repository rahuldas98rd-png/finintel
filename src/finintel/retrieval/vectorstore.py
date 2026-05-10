"""Qdrant vector store wrapper for chunk indexing and semantic search.

Connection is configurable via environment variables so the same code path
works against local Docker Qdrant (development) and Qdrant Cloud (HF Spaces
deployment):

    QDRANT_URL=http://localhost:6333         # local Docker (default)
    QDRANT_URL=https://xxx.cloud.qdrant.io   # Qdrant Cloud
    QDRANT_API_KEY=...                       # required for cloud, unset for local
"""
from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Sequence

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from finintel.retrieval.chunker import Chunk

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "finintel_chunks"
DEFAULT_QDRANT_URL = "http://localhost:6333"


class VectorStore:
    """Thin wrapper around Qdrant for chunk indexing + retrieval."""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection: str = DEFAULT_COLLECTION,
        vector_dim: int = 768,
    ) -> None:
        # Env vars take precedence over hardcoded defaults; explicit constructor
        # args override env vars (useful for tests and the migration script).
        url = url or os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL)
        api_key = api_key or os.getenv("QDRANT_API_KEY")  # None for local

        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection = collection
        self.vector_dim = vector_dim
        self._url = url  # store for diagnostic logging

        scheme = "cloud" if api_key else "local"
        logger.info("VectorStore connected to %s Qdrant at %s", scheme, url)

    def reset_collection(self) -> None:
        """Delete and recreate the collection. Used when re-indexing from scratch."""
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
            logger.info("Deleted existing collection: %s", self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=self.vector_dim,
                distance=Distance.COSINE,
            ),
        )
        logger.info(
            "Created collection %s (dim=%d, distance=COSINE)",
            self.collection, self.vector_dim,
        )

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: np.ndarray,
        batch_size: int = 100,
    ) -> int:
        """Upsert chunks + their embeddings as Qdrant points.

        Uses deterministic UUIDv5 keys derived from chunk_id, so re-indexing
        replaces existing records rather than duplicating.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Length mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )

        points: list[PointStruct] = []
        for chunk, vec in zip(chunks, embeddings, strict=True):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_OID, chunk.chunk_id))
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vec.tolist(),
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "ticker": chunk.ticker,
                        "filing_type": chunk.filing_type,
                        "accession": chunk.accession,
                        "section": chunk.section,
                        "chunk_index": chunk.chunk_index,
                        "total_chunks": chunk.total_chunks,
                        "n_tokens": chunk.n_tokens,
                        "text": chunk.text,
                    },
                )
            )

        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection,
                points=points[i : i + batch_size],
            )
        logger.info("Upserted %d points into %s", len(points), self.collection)
        return len(points)

    def search(
        self,
        query_vector: np.ndarray,
        limit: int = 5,
        ticker: str | None = None,
        section: str | None = None,
    ) -> list[dict]:
        """Semantic search with optional metadata filters.

        Returns up to `limit` matches, ranked by cosine similarity.
        """
        conditions = []
        if ticker:
            conditions.append(FieldCondition(key="ticker", match=MatchValue(value=ticker)))
        if section:
            conditions.append(FieldCondition(key="section", match=MatchValue(value=section)))
        query_filter = Filter(must=conditions) if conditions else None

        # qdrant-client deprecated .search() in 1.7; removed/wrapped in 1.10+.
        # New API: query_points() returns a QueryResponse; points are on .points.
        hits = self.client.query_points(
            collection_name=self.collection,
            query=query_vector.tolist(),
            limit=limit,
            query_filter=query_filter,
        ).points

        return [
            {
                "score": h.score,
                "chunk_id": h.payload["chunk_id"],
                "ticker": h.payload["ticker"],
                "section": h.payload["section"],
                "text": h.payload["text"],
            }
            for h in hits
        ]
