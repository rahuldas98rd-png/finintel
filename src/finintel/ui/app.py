"""FinIntel Streamlit UI.

Three pipeline modes selectable from the sidebar:
  - Single-shot RAG (default, fastest)
  - Single-shot + cross-encoder reranker (slower, no measurable gain on this corpus)
  - Agent (multi-step planner → retriever → synthesizer; best for multi-entity queries)
"""
from __future__ import annotations

import time
from collections import Counter

import streamlit as st

from finintel.agent.generator import RAGPipeline
from finintel.agent.graph import AgentPipeline
from finintel.retrieval.reranker import Reranker

st.set_page_config(
    page_title="FinIntel — Ask the SEC",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached pipelines — each loads once per session
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading single-shot pipeline…")
def get_pipeline_basic() -> RAGPipeline:
    return RAGPipeline()


@st.cache_resource(show_spinner="Loading reranker pipeline…")
def get_pipeline_rerank() -> RAGPipeline:
    return RAGPipeline(reranker=Reranker())


@st.cache_resource(show_spinner="Loading agent pipeline…")
def get_agent() -> AgentPipeline:
    return AgentPipeline()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📊 FinIntel")
    st.caption("Agentic RAG over SEC filings")

    st.subheader("Mode")
    mode = st.radio(
        "Pipeline",
        options=["Single-shot", "Single-shot + Reranker", "Agent (planner → retrieve → synthesize)"],
        index=0,
        captions=[
            "Fastest. Top-4 chunks straight to the LLM.",
            "Cross-encoder reranks top-12 to top-4. No measurable gain on eval set.",
            "Decomposes multi-entity questions into per-company sub-queries. Best for comparisons.",
        ],
    )

    is_agent = mode.startswith("Agent")

    st.subheader("Filters")
    if is_agent:
        st.caption("⚙️ Filters are managed by the planner in Agent mode.")
        ticker, section = None, None
    else:
        ticker = st.selectbox(
            "Ticker (optional)",
            options=[None, "AAPL", "GOOGL", "JPM", "MSFT", "TSLA"],
            format_func=lambda x: "All companies" if x is None else x,
        )
        section = st.selectbox(
            "Section (optional)",
            options=[None, "risk_factors", "mda"],
            format_func=lambda x: "All sections" if x is None else x.replace("_", " ").title(),
        )

    st.subheader("About")
    st.caption(
        "Ask analyst-grade questions about Apple, Google, JPMorgan, Microsoft, "
        "or Tesla. Answers are grounded in 10-K filings with chunk-level citations."
    )
    st.caption("📂 [GitHub](https://github.com/rahuldas98rd-png/finintel)")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("Ask the SEC filings")
st.caption(
    "Try: *How does Google describe AI risks?* · *What drove Apple's services revenue?* · "
    "*Compare cloud competition for Microsoft and Google.* (Use Agent mode for comparisons.)"
)

question = st.text_area(
    "Your question",
    placeholder="e.g., How do Google and Microsoft each frame cloud competition?",
    height=80,
)

ask_clicked = st.button("Ask", type="primary", disabled=not question.strip())

if ask_clicked and question.strip():
    if is_agent:
        agent = get_agent()
        with st.spinner("Planning · retrieving · synthesizing…"):
            t0 = time.time()
            result = agent.answer(question.strip())
            elapsed = time.time() - t0

        # ---- Plan reveal (the agentic differentiator) ----
        with st.expander(
            f"🧭 Agent's plan ({len(result.sub_queries)} sub-quer"
            f"{'y' if len(result.sub_queries) == 1 else 'ies'})",
            expanded=True,
        ):
            for i, sq in enumerate(result.sub_queries, 1):
                t = sq.get("ticker") or "any"
                s = sq.get("section") or "any"
                st.markdown(f"**{i}.** `[{t} · {s}]` — {sq['question']}")

        st.subheader("Answer")
        st.markdown(result.answer)

        # ---- Metrics including source distribution ----
        cols = st.columns(4)
        cols[0].metric("Latency", f"{elapsed:.1f}s")
        cols[1].metric("Sources", len(result.sources))
        cols[2].metric("Input tokens", f"{result.input_tokens:,}")
        cols[3].metric("Output tokens", f"{result.output_tokens:,}")

        ticker_counts = Counter(s["ticker"] for s in result.sources)
        if len(ticker_counts) > 1:
            st.caption(
                "**Source coverage:** "
                + " · ".join(f"{t}: {n}" for t, n in sorted(ticker_counts.items()))
            )

        sources = result.sources

    else:
        # Single-shot path
        rag = get_pipeline_rerank() if mode == "Single-shot + Reranker" else get_pipeline_basic()
        with st.spinner("Retrieving and synthesizing…"):
            t0 = time.time()
            result = rag.answer(question.strip(), ticker=ticker, section=section)
            elapsed = time.time() - t0

        st.subheader("Answer")
        st.markdown(result.answer)

        cols = st.columns(4)
        cols[0].metric("Latency", f"{elapsed:.1f}s")
        cols[1].metric("Sources", len(result.sources))
        cols[2].metric("Input tokens", f"{result.input_tokens:,}")
        cols[3].metric("Output tokens", f"{result.output_tokens:,}")

        sources = result.sources

    # ---- Sources (shared by all modes) ----
    st.subheader(f"Sources ({len(sources)})")
    for i, src in enumerate(sources, 1):
        score_label = f"{src['score']:.3f}"
        rerank_part = f" → rerank {src['rerank_score']:.3f}" if "rerank_score" in src else ""
        with st.expander(
            f"{i}. {src['ticker']} · {src['section'].replace('_', ' ')} · "
            f"score {score_label}{rerank_part} · `{src['chunk_id']}`"
        ):
            st.caption(f"Chunk ID: `{src['chunk_id']}`")
            st.write(src["text"])

else:
    st.info(
        "Pick a mode in the sidebar, enter a question, and hit Ask. "
        "**Try Agent mode for any question that names two or more companies.**"
    )