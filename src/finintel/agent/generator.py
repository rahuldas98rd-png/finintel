"""Baseline RAG pipeline: query -> embed -> retrieve -> generate.

Provider-agnostic: uses any OpenAI-compatible endpoint (Groq, OpenAI,
OpenRouter, Together, vLLM/Ollama with their OpenAI shim, etc.). Defaults
to Groq + Llama 3.3 70B (free tier, fast, GPT-4o-class quality).

Override via env if needed:
  LLM_BASE_URL  (default: https://api.groq.com/openai/v1)
  LLM_MODEL     (default: llama-3.3-70b-versatile)
  LLM_API_KEY_ENV  (default: GROQ_API_KEY — name of env var to read)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

from finintel.agent.prompts import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE, format_context
from finintel.retrieval.embeddings import Embedder
from finintel.retrieval.vectorstore import VectorStore
from finintel.retrieval.reranker import Reranker

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
DEFAULT_API_KEY_ENV = os.getenv("LLM_API_KEY_ENV", "GROQ_API_KEY")
DEFAULT_TOP_K = 4         # was 6 — fits Llama 3.1 8B free tier (6K TPM)
DEFAULT_RETRIEVE_K = 12   # was 20 — reranker still has 3x candidates to pick from
DEFAULT_MAX_TOKENS = 1024


@dataclass(frozen=True)
class RAGAnswer:
    question: str
    answer: str
    sources: list[dict]
    model: str
    input_tokens: int
    output_tokens: int


class RAGPipeline:
    """Baseline retrieve-then-generate RAG."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        reranker: Reranker | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        top_k: int = DEFAULT_TOP_K,
        retrieve_k: int = DEFAULT_RETRIEVE_K,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.store = store or VectorStore(vector_dim=self.embedder.dim)
        self.reranker = reranker  # None disables reranking
        self.model = model
        self.top_k = top_k
        self.retrieve_k = retrieve_k

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{api_key_env} not set in .env. "
                f"Get a free Groq key at https://console.groq.com"
            )
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info(
            "Using model %s via %s | reranker=%s",
            self.model, base_url, "on" if self.reranker else "off",
        )

    def answer(
        self,
        question: str,
        ticker: str | None = None,
        section: str | None = None,
    ) -> RAGAnswer:
        # Retrieve a wider pool when reranking is enabled
        k = self.retrieve_k if self.reranker else self.top_k
        query_vec = self.embedder.encode([question], show_progress=False)[0]
        hits = self.store.search(query_vec, limit=k, ticker=ticker, section=section)

        # Optionally rerank to a tighter top_k
        if self.reranker and hits:
            hits = self.reranker.rerank(question, hits, top_k=self.top_k)

        if not hits:
            return RAGAnswer(
                question=question,
                answer="No relevant chunks for this query.",
                sources=[],
                model=self.model,
                input_tokens=0,
                output_tokens=0,
            )

        user_message = RAG_USER_TEMPLATE.format(
            question=question,
            context=format_context(hits),
        )
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=DEFAULT_MAX_TOKENS,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        return RAGAnswer(
            question=question,
            answer=response.choices[0].message.content,
            sources=hits,
            model=self.model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )