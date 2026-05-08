"""Run the eval set through the RAG pipeline and compute baseline metrics.

v0: keyword-based grading + structural metrics (citation count, source
diversity, refusal correctness). LLM-based faithfulness/relevancy comes later.
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

from finintel.agent.generator import RAGAnswer, RAGPipeline

import argparse
from finintel.retrieval.reranker import Reranker

EVAL_DIR = Path(__file__).parent
EVAL_SET = EVAL_DIR / "eval_set.jsonl"
RESULTS_DIR = EVAL_DIR / "results"

# Matches our chunk_id format: TICKER_FORM_ACCESSION_SECTION_NNN
_CITATION_RE = re.compile(r"\[[A-Z]+_10-[KQ]_[0-9-]+_[a-z_]+_\d+\]")


def load_eval_set() -> list[dict]:
    with EVAL_SET.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_answer(meta: dict, rag: RAGAnswer) -> dict:
    """Compute per-question metrics."""
    text = rag.answer.lower()

    # Keyword recall: fraction of must_mention terms that appear
    must = [t.lower() for t in meta.get("must_mention", [])]
    if must:
        hits = sum(1 for term in must if term in text)
        keyword_recall = hits / len(must)
    else:
        keyword_recall = None  # for refusal questions

    # Did the model refuse? Match our system-prompt refusal phrase
    refused = "do not contain information" in text or "do not contain" in text

    # Source diversity
    tickers = sorted({s["ticker"] for s in rag.sources})

    # Citation count in the answer text
    n_citations = len(_CITATION_RE.findall(rag.answer))

    return {
        "id": meta["id"],
        "question": meta["question"],
        "answer": rag.answer,
        "keyword_recall": keyword_recall,
        "refused": refused,
        "expected_refusal": meta.get("expected_refusal", False),
        "n_sources": len(rag.sources),
        "tickers_in_sources": tickers,
        "n_citations": n_citations,
        "input_tokens": rag.input_tokens,
        "output_tokens": rag.output_tokens,
        "answer_length": len(rag.answer),
    }


def aggregate(scores: list[dict]) -> dict:
    """Corpus-level summary stats."""
    factual = [s for s in scores if not s["expected_refusal"]]
    refusal = [s for s in scores if s["expected_refusal"]]

    recalls = [s["keyword_recall"] for s in factual if s["keyword_recall"] is not None]
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0

    refusal_correct = sum(1 for s in refusal if s["refused"])
    refusal_total = len(refusal)

    return {
        "n_questions": len(scores),
        "n_factual": len(factual),
        "n_refusal": refusal_total,
        "avg_keyword_recall": round(avg_recall, 3),
        "refusal_accuracy": f"{refusal_correct}/{refusal_total}" if refusal_total else "N/A",
        "avg_citations_per_factual_answer": round(
            sum(s["n_citations"] for s in factual) / max(len(factual), 1), 2
        ),
        "total_input_tokens": sum(s["input_tokens"] for s in scores),
        "total_output_tokens": sum(s["output_tokens"] for s in scores),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the FinIntel eval set.")
    parser.add_argument(
        "--rerank", action="store_true",
        help="Enable cross-encoder reranking (over-retrieve to 20, rerank to 6).",
    )
    parser.add_argument(
        "--label", type=str, default="baseline",
        help="Tag for the results file (e.g., 'baseline', 'rerank').",
    )
    args = parser.parse_args()

    eval_set = load_eval_set()
    print(f"Loaded {len(eval_set)} eval questions")

    reranker = Reranker() if args.rerank else None
    rag = RAGPipeline(reranker=reranker)
    print(f"Model: {rag.model} | reranker={'on' if reranker else 'off'}\n")

    # ... rest of main() unchanged, but change the file naming to include label:
    ts = time.strftime("%Y%m%d_%H%M%S")
    detail_path = RESULTS_DIR / f"{args.label}_{ts}_detailed.json"
    summary_path = RESULTS_DIR / f"{args.label}_{ts}_summary.json"
    # ... continue with existing detail/summary saving


if __name__ == "__main__":
    sys.exit(main())