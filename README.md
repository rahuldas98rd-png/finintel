# FinIntel � Agentic RAG for Financial Document Intelligence

> Analyst-grade question answering over SEC filings, earnings calls, and news, with citations and multi-hop reasoning.

## Status
?? In active development. 6-week build plan in progress.

## Why this exists
Naive RAG fails on questions like *"How has Tesla''s gross margin commentary evolved over the last 4 quarters?"* � they require planning, multi-document retrieval, and synthesis. FinIntel uses a LangGraph agent to handle these.

## Architecture
_(diagram coming in Week 3)_

## Quick start
_(coming in Week 2 once the baseline RAG runs)_

## Evaluation
_(RAGAS scorecard coming in Week 4)_

## Tech stack
- LangGraph (agent orchestration)
- Qdrant (vector store)
- BGE / OpenAI embeddings
- FastAPI + Streamlit
- Docker, GitHub Actions

## Roadmap
- [x] Week 1: Repo scaffold + SEC EDGAR ingestion
- [ ] Week 2: Baseline RAG + Streamlit MVP
- [ ] Week 3: LangGraph agent + citations
- [ ] Week 4: RAGAS evaluation harness
- [ ] Week 5: Multi-company comparison + temporal queries
- [ ] Week 6: Deploy + demo video

```bash
finintel/
├── .gitignore
├── .env.example
├── .python-version
├── README.md
├── pyproject.toml
├── uv.lock
├── data/
│   └── raw/
├── evals/
├── notebooks/
├── src/
│   └── finintel/
│       ├── __init__.py
│       ├── agent/__init__.py
│       ├── api/__init__.py
│       ├── ingestion/__init__.py
│       ├── retrieval/__init__.py
│       └── ui/__init__.py
└── tests/
```

## Evaluation results

| Configuration | Recall | Citations | Tokens | Notes |
|---|---|---|---|---|
| BGE-base + top-4 (baseline) | 0.79 | 3.6 | 34K | Default |
| + ms-marco-MiniLM reranker | 0.78 | 3.1 | 37K | Disabled — no measurable gain |

The cross-encoder reranker did not improve keyword recall on a 10-question eval set, while adding 10% to token cost and ~50% to wall-clock time. Kept as an optional component (toggleable via `--rerank`) since it may help on harder corpora or with LLM-based grading.

Eval methodology: keyword matching against curated `must_mention` terms.
Future work: add LLM-as-judge grading (RAGAS) to catch semantic equivalences.