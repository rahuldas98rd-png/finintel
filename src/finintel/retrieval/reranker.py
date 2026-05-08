"""Cross-encoder reranker for RAG retrieval.

After bi-encoder retrieval surfaces a candidate pool (e.g., top-20),
rerank with a cross-encoder for higher precision (e.g., top-6).
Default: ms-marco-MiniLM-L-6-v2 — ~80MB, fast on CPU, MS MARCO-tuned
for query-document reranking. (Switch to BAAI/bge-reranker-base later
if you want maximum quality and have bandwidth to spare.)
"""
from __future__ import annotations

import logging
from copy import copy

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Wraps a cross-encoder model that scores (query, document) pairs."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        logger.info("Loading reranker: %s", model_name)
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 6,
    ) -> list[dict]:
        """Score (query, candidate.text) pairs and return top-k by score.

        Each returned dict gets a new 'rerank_score' field. Original
        bi-encoder 'score' is preserved for diagnostics.
        """
        if not candidates:
            return []
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)
        ranked = []
        for cand, s in zip(candidates, scores, strict=True):
            new_cand = copy(cand)
            new_cand["rerank_score"] = float(s)
            ranked.append(new_cand)
        ranked.sort(key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:top_k]