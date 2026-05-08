"""FinIntel Streamlit UI.

A demo-ready interface for the RAG pipeline:
  - Free-form question input
  - Optional ticker / section filters
  - Toggle for reranking
  - Expandable source chunks with similarity scores
  - Token usage and latency display
"""
from __future__ import annotations

import time

import streamlit as st

from finintel.agent.generator import RAGPipeline
from finintel.retrieval.reranker import Reranker

st.set_page_config(
    page_title="FinIntel — Ask the SEC",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached resources — build once per session
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model…")
def get_pipeline_no_rerank() -> RAGPipeline:
    return RAGPipeline()


@st.cache_resource(show_spinner="Loading reranker…")
def get_pipeline_with_rerank() -> RAGPipeline:
    return RAGPipeline(reranker=Reranker())


# ---------------------------------------------------------------------------
# Sidebar — filters and config
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📊 FinIntel")
    st.caption("Agentic RAG over SEC filings")

    st.subheader("Filters")
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

    st.subheader("Retrieval")
    use_rerank = st.toggle("Use cross-encoder reranker", value=False)
    st.caption(
        "Default off — eval shows no measurable gain on this corpus. "
        "Toggle on to compare."
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
    "Try: *How does Google describe AI risks?* — *What drove Apple's services revenue?* — "
    "*Compare cloud competition for Microsoft and Google.*"
)

question = st.text_area(
    "Your question",
    placeholder="e.g., What does Tesla disclose about Full Self-Driving risks?",
    height=80,
)

ask_clicked = st.button("Ask", type="primary", disabled=not question.strip())

if ask_clicked and question.strip():
    rag = get_pipeline_with_rerank() if use_rerank else get_pipeline_no_rerank()

    with st.spinner("Retrieving and synthesizing…"):
        t0 = time.time()
        result = rag.answer(
            question.strip(),
            ticker=ticker,
            section=section,
        )
        elapsed = time.time() - t0

    # ---- Answer ----
    st.subheader("Answer")
    st.markdown(result.answer)

    # ---- Metadata strip ----
    cols = st.columns(4)
    cols[0].metric("Latency", f"{elapsed:.1f}s")
    cols[1].metric("Sources", len(result.sources))
    cols[2].metric("Input tokens", f"{result.input_tokens:,}")
    cols[3].metric("Output tokens", f"{result.output_tokens:,}")

    # ---- Sources ----
    st.subheader(f"Sources ({len(result.sources)})")
    for i, src in enumerate(result.sources, 1):
        score_label = f"{src['score']:.3f}"
        rerank_part = (
            f" → rerank {src['rerank_score']:.3f}"
            if "rerank_score" in src
            else ""
        )
        with st.expander(
            f"{i}. {src['ticker']} · {src['section'].replace('_', ' ')} · "
            f"score {score_label}{rerank_part} · `{src['chunk_id']}`"
        ):
            st.caption(f"Chunk ID: `{src['chunk_id']}`")
            st.write(src["text"])

else:
    st.info("Pick filters in the sidebar (or leave them off), enter a question, and hit Ask.")