"""LangGraph-based agent pipeline.

Replaces single-shot RAG with planner → retriever → synthesizer → critic.

The planner decomposes the user question into per-company sub-queries when
needed; the retriever runs metadata-filtered search for each and accumulates
chunks; the synthesizer generates a single grounded answer; the critic runs
once at the end as an advisory check (it does NOT loop back to re-plan — see
the README's engineering decisions section for why).

Solves the multi-company retrieval problem where naive top-K returns chunks
biased toward whichever company's language embeds closer to the query.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from finintel.agent.prompts import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE, format_context
from finintel.retrieval.embeddings import Embedder
from finintel.retrieval.vectorstore import VectorStore

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
DEFAULT_API_KEY_ENV = os.getenv("LLM_API_KEY_ENV", "GROQ_API_KEY")


# ---------------------------------------------------------------------------
# Planner prompt — decomposes the user question into structured sub-queries
# ---------------------------------------------------------------------------

PLANNER_SYSTEM = """You are a query planner for a RAG system over SEC 10-K filings.

Available companies (tickers): AAPL, GOOGL, JPM, MSFT, TSLA
Available sections: risk_factors, mda

Decompose the user's question into one or more sub-queries. Each sub-query targets
ONE company and optionally one section. Use multiple sub-queries when the question
mentions or implies multiple companies — this gives balanced retrieval across them.
Use a single sub-query when the question is about one company or generic.

Output JSON only (no markdown, no commentary), in this exact format:
{"sub_queries": [{"question": "...", "ticker": "...", "section": "..."}]}

Use null (not omitted) when not specified. Examples:

Q: "How does Google describe AI risks?"
{"sub_queries": [{"question": "How does Google describe AI risks?", "ticker": "GOOGL", "section": "risk_factors"}]}

Q: "Compare AAPL and MSFT cloud strategies"
{"sub_queries": [
  {"question": "Apple's cloud strategy", "ticker": "AAPL", "section": "mda"},
  {"question": "Microsoft's cloud strategy", "ticker": "MSFT", "section": "mda"}
]}

Q: "Which tech companies discuss AI risk most extensively?"
{"sub_queries": [
  {"question": "AI risks discussion at Apple", "ticker": "AAPL", "section": "risk_factors"},
  {"question": "AI risks discussion at Google", "ticker": "GOOGL", "section": "risk_factors"},
  {"question": "AI risks discussion at Microsoft", "ticker": "MSFT", "section": "risk_factors"}
]}
"""

PLANNER_USER_TEMPLATE = "Question: {question}\n\nOutput JSON only:"


# ---------------------------------------------------------------------------
# Critic prompt — advisory groundedness check, runs once after synthesizer
# ---------------------------------------------------------------------------

CRITIC_SYSTEM = """You are evaluating a RAG-generated answer about SEC filings.

You will see the user's question, the planner's sub-queries, the synthesized answer,
and the chunk IDs that were cited. Evaluate the answer along three dimensions:

1. **Coverage**: Does the answer address every entity (company, section) implied by the sub-queries?
2. **Grounding**: Are factual claims attributed via chunk IDs in [BRACKETS]?
3. **Hallucinations**: Are there specific claims (numbers, dates, named entities) that wouldn't plausibly come from a 10-K filing?

Be strict but fair. If the answer honestly says "the filings do not contain X," that is GOOD
behavior, not a hallucination. Refusal grounded in the corpus is a feature, not a bug.

Output JSON only, in this exact shape:
{
  "covers_all_entities": true|false,
  "well_grounded": true|false,
  "no_hallucinations": true|false,
  "issues": ["short specific issue", ...]
}

Use an empty issues list when the answer is clean."""

CRITIC_USER_TEMPLATE = """Question: {question}

Sub-queries planned:
{sub_queries}

Answer:
{answer}

Chunk IDs cited as sources: {chunk_ids}

Output JSON only:"""


# ---------------------------------------------------------------------------
# Graph state — what flows between nodes
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    question: str
    sub_queries: list[dict]
    retrieved_chunks: list[dict]
    answer: str
    critique: dict
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class AgentAnswer:
    """Result of an agent run, including planner output and critic notes."""
    question: str
    answer: str
    sub_queries: list[dict]
    sources: list[dict]
    critique: dict
    input_tokens: int
    output_tokens: int


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class AgentPipeline:
    """Multi-step agentic RAG via LangGraph: plan → retrieve → synthesize → critique.

    The critic runs once as an advisory pass and never re-loops. A looping
    critic was prototyped but rejected: on free-tier infrastructure it doubles
    per-query token cost and adds unbounded latency without measurable answer
    improvement on this corpus.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        per_query_k: int = 2,
        max_context_chunks: int = 4,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.store = store or VectorStore(vector_dim=self.embedder.dim)
        self.per_query_k = per_query_k
        self.max_context_chunks = max_context_chunks

        api_key = os.getenv(DEFAULT_API_KEY_ENV)
        if not api_key:
            raise RuntimeError(
                f"{DEFAULT_API_KEY_ENV} not set in .env. "
                f"Get a free Groq key at https://console.groq.com"
            )
        self.client = OpenAI(api_key=api_key, base_url=DEFAULT_BASE_URL)
        self.model = DEFAULT_MODEL

        self._graph = self._build_graph()
        logger.info("AgentPipeline ready | model=%s", self.model)

    # ----- Graph wiring -----

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("planner", self._plan)
        g.add_node("retriever", self._retrieve)
        g.add_node("synthesizer", self._synthesize)
        g.add_node("critic", self._critique)
        g.add_edge(START, "planner")
        g.add_edge("planner", "retriever")
        g.add_edge("retriever", "synthesizer")
        g.add_edge("synthesizer", "critic")
        g.add_edge("critic", END)
        return g.compile()

    # ----- Nodes -----

    def _plan(self, state: AgentState) -> dict:
        """Use the LLM to decompose the question into sub-queries."""
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=512,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": PLANNER_USER_TEMPLATE.format(question=state["question"])},
            ],
        )
        raw = response.choices[0].message.content
        try:
            parsed = json.loads(raw)
            sub_queries = parsed.get("sub_queries", [])
        except json.JSONDecodeError:
            logger.warning("Planner returned invalid JSON; falling back to a single query")
            sub_queries = []

        # Defensive fallback: never proceed with zero sub-queries
        if not sub_queries:
            sub_queries = [{"question": state["question"], "ticker": None, "section": None}]

        logger.info("Planner produced %d sub-queries", len(sub_queries))
        return {
            "sub_queries": sub_queries,
            "input_tokens": state.get("input_tokens", 0) + response.usage.prompt_tokens,
            "output_tokens": state.get("output_tokens", 0) + response.usage.completion_tokens,
        }

    def _retrieve(self, state: AgentState) -> dict:
        """Run filtered retrieval per sub-query, then round-robin merge to preserve
        coverage across all sub-queries.

        A naive concat-then-truncate strategy drops the last sub-queries entirely
        when total candidate chunks exceed max_context_chunks. Round-robin guarantees
        every sub-query contributes its top chunk before any contributes its second.
        """
        # Step 1: gather per-sub-query hits
        per_sq_hits: list[list[dict]] = []
        for sq in state["sub_queries"]:
            query_vec = self.embedder.encode([sq["question"]], show_progress=False)[0]
            hits = self.store.search(
                query_vec,
                limit=self.per_query_k,
                ticker=sq.get("ticker"),
                section=sq.get("section"),
            )
            per_sq_hits.append(list(hits))

        # Step 2: round-robin merge — round k pulls the k-th best chunk from each sub-query
        merged: list[dict] = []
        seen: set[str] = set()
        for round_idx in range(self.per_query_k):
            if len(merged) >= self.max_context_chunks:
                break
            for hits in per_sq_hits:
                if round_idx >= len(hits):
                    continue
                if len(merged) >= self.max_context_chunks:
                    break
                hit = hits[round_idx]
                if hit["chunk_id"] not in seen:
                    seen.add(hit["chunk_id"])
                    merged.append(hit)

        logger.info(
            "Retriever (round-robin) | sub-queries=%d, kept=%d/%d",
            len(state["sub_queries"]),
            len(merged),
            self.max_context_chunks,
        )
        return {"retrieved_chunks": merged}

    def _synthesize(self, state: AgentState) -> dict:
        """Generate the final answer from accumulated chunks."""
        chunks = state["retrieved_chunks"]
        if not chunks:
            return {"answer": "No relevant chunks were retrieved for this query."}

        user_message = RAG_USER_TEMPLATE.format(
            question=state["question"],
            context=format_context(chunks),
        )
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        return {
            "answer": response.choices[0].message.content,
            "input_tokens": state.get("input_tokens", 0) + response.usage.prompt_tokens,
            "output_tokens": state.get("output_tokens", 0) + response.usage.completion_tokens,
        }

    def _critique(self, state: AgentState) -> dict:
        """Single-pass advisory critic: validates answer, does not re-loop."""
        # Skip cleanly if synthesizer produced no answer
        if not state.get("answer") or state["answer"].startswith("No relevant chunks"):
            return {
                "critique": {
                    "covers_all_entities": False,
                    "well_grounded": False,
                    "no_hallucinations": True,
                    "issues": ["No answer was generated"],
                }
            }

        sub_query_lines = "\n".join(
            f"  - [{sq.get('ticker') or 'any'}/{sq.get('section') or 'any'}] {sq['question']}"
            for sq in state["sub_queries"]
        )
        chunk_ids = [c["chunk_id"] for c in state["retrieved_chunks"]]

        user_message = CRITIC_USER_TEMPLATE.format(
            question=state["question"],
            sub_queries=sub_query_lines,
            answer=state["answer"],
            chunk_ids=chunk_ids,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=384,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": CRITIC_SYSTEM},
                    {"role": "user", "content": user_message},
                ],
            )
            critique = json.loads(response.choices[0].message.content)
            # Defensive defaults — never trust an LLM to produce all keys
            critique.setdefault("covers_all_entities", True)
            critique.setdefault("well_grounded", True)
            critique.setdefault("no_hallucinations", True)
            critique.setdefault("issues", [])

            logger.info(
                "Critic | coverage=%s grounded=%s no_halluc=%s issues=%d",
                critique["covers_all_entities"],
                critique["well_grounded"],
                critique["no_hallucinations"],
                len(critique["issues"]),
            )

            return {
                "critique": critique,
                "input_tokens": state.get("input_tokens", 0) + response.usage.prompt_tokens,
                "output_tokens": state.get("output_tokens", 0) + response.usage.completion_tokens,
            }
        except Exception as exc:
            logger.warning("Critic failed: %s", exc)
            # Critic failure must NOT break the user experience
            return {
                "critique": {
                    "covers_all_entities": True,
                    "well_grounded": True,
                    "no_hallucinations": True,
                    "issues": [f"Critic unavailable: {type(exc).__name__}"],
                }
            }

    # ----- Public API -----

    def answer(self, question: str) -> AgentAnswer:
        initial: AgentState = {
            "question": question,
            "sub_queries": [],
            "retrieved_chunks": [],
            "answer": "",
            "critique": {},
            "input_tokens": 0,
            "output_tokens": 0,
        }
        final = self._graph.invoke(initial)
        return AgentAnswer(
            question=question,
            answer=final["answer"],
            sub_queries=final["sub_queries"],
            sources=final["retrieved_chunks"],
            critique=final.get("critique", {}),
            input_tokens=final["input_tokens"],
            output_tokens=final["output_tokens"],
        )
