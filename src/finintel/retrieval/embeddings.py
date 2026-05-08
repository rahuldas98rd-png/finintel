"""Embedding generation for chunk text.

Uses sentence-transformers with BGE-base-en-v1.5 by default:
  - 768 dimensions
  - Top-10 on the MTEB retrieval benchmark
  - ~440 MB model, runs comfortably on CPU
  - Free, no API costs
"""
from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_BATCH_SIZE = 32


class Embedder:
    """Wraps a sentence-transformers model for batch encoding."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        logger.info("Loading embedding model: %s", model_name)
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_embedding_dimension()
        logger.info("Model loaded. Vector dimension: %d", self.dim)

    def encode(self, texts: Iterable[str], show_progress: bool = True) -> np.ndarray:
        """Encode texts into L2-normalized embeddings (so dot product == cosine)."""
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )