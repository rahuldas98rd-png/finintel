"""Run the eval set through one of three pipelines and compute metrics.

Modes:
  baseline   — single-shot RAG, top-4 retrieval, ticker/section filters from question metadata
  rerank     — single-shot RAG with cross-encoder reranker (top-4 from 12)
  agent      — LangGraph agent (planner → retriever → synthesizer → critic);
               filters are managed by the planner, NOT by question metadata

Metrics:
  keyword_recall                   — fraction of must_mention terms in the answer (factual Qs only)
  ticker_coverage                  — fraction of expected_tickers present in retrieved sources
                                     (the metric where the agent is expected to win)
  refusal_correct                  — for refusal questions only
  n_citations                      — count of [CHUNK_ID] references
  tokens / latency                 — usual

Agent-only metrics (saved to JSON for inspection):
  n_sub_queries                    — how the planner decomposed the question
  critic_clean                     — did the critic flag any issues
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Union

from finintel.agent.generator import RAGAnswer, RAGPipeline
from finintel.agent.graph import AgentAnswer, AgentPipeline
from finintel.retrieval.reranker import Reranker

EVAL_DIR = Path(__file__).parent
EVAL_SET = EVAL_DIR / "eval_set.jsonl"
RESULTS_DIR = EVAL_DIR / "results"

# Matches our chunk_id format: TICKER_FORM_ACCESSION_SECTION_NNN
_CITATION_RE = re.compile(r"\[[A-Z]+_10-[KQ]_[0-9-]+_[a-z_]+_\d+\]")


def load_eval_set() -> list[dict]:
    with EVAL_SET.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_answer(meta: dict, rag: Union[RAGAnswer, AgentAnswer]) -> dict:
    """Compute per-question metrics for either single-shot or agent answers."""
    text = rag.answer.lower()

    # Keyword recall over must_mention terms
    must = [t.lower() for t in meta.get("must_mention", [])]
    keyword_recall = sum(1 for t in must if t in text) / len(must) if must else None

    # Refusal detection (matches our system-prompt refusal phrasing)
    refused = "do not contain information" in text or "do not contain" in text

    # Source diversity / ticker coverage
    tickers_in_sources = sorted({s["ticker"] for s in rag.sources})
    expected_tickers = meta.get("expected_tickers") or (
        [meta["ticker"]] if meta.get("ticker") and not meta.get("expected_refusal") else []
    )
    if expected_tickers:
        hit = sum(1 for t in expected_tickers if t in tickers_in_sources)
        ticker_coverage = hit / len(expected_tickers)
    else:
        ticker_coverage = None  # not applicable (e.g., refusal questions)

    n_citations = len(_CITATION_RE.findall(rag.answer))

    record = {
        "id": meta["id"],
        "question": meta["question"],
        "answer": rag.answer,
        "keyword_recall": keyword_recall,
        "refused": refused,
        "expected_refusal": meta.get("expected_refusal", False),
        "n_sources": len(rag.sources),
        "tickers_in_sources": tickers_in_sources,
        "expected_tickers": expected_tickers,
        "ticker_coverage": ticker_coverage,
        "n_citations": n_citations,
        "input_tokens": rag.input_tokens,
        "output_tokens": rag.output_tokens,
        "answer_length": len(rag.answer),
    }

    # Agent-only fields
    if isinstance(rag, AgentAnswer):
        record["n_sub_queries"] = len(rag.sub_queries)
        record["sub_queries"] = rag.sub_queries
        crit = rag.critique or {}
        record["critic_clean"] = (
            crit.get("covers_all_entities", True)
            and crit.get("well_grounded", True)
            and crit.get("no_hallucinations", True)
            and not crit.get("issues")
        )
        record["critic_issues"] = crit.get("issues", [])

    return record


def aggregate(scores: list[dict]) -> dict:
    factual = [s for s in scores if not s["expected_refusal"]]
    refusal = [s for s in scores if s["expected_refusal"]]

    recalls = [s["keyword_recall"] for s in factual if s["keyword_recall"] is not None]
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0

    multi_co = [
        s for s in factual
        if s["ticker_coverage"] is not None and len(s["expected_tickers"]) > 1
    ]
    coverages = [s["ticker_coverage"] for s in multi_co]
    avg_multi_co_coverage = sum(coverages) / len(coverages) if coverages else 0.0

    refusal_correct = sum(1 for s in refusal if s["refused"])
    refusal_total = len(refusal)

    summary = {
        "n_questions": len(scores),
        "n_factual": len(factual),
        "n_multi_company": len(multi_co),
        "n_refusal": refusal_total,
        "avg_keyword_recall": round(avg_recall, 3),
        "avg_multi_company_ticker_coverage": round(avg_multi_co_coverage, 3),
        "refusal_accuracy": f"{refusal_correct}/{refusal_total}" if refusal_total else "N/A",
        "avg_citations_per_factual_answer": round(
            sum(s["n_citations"] for s in factual) / max(len(factual), 1), 2
        ),
        "avg_latency_sec": round(
            sum(s.get("elapsed_sec", 0) for s in scores) / max(len(scores), 1), 2
        ),
        "total_input_tokens": sum(s["input_tokens"] for s in scores),
        "total_output_tokens": sum(s["output_tokens"] for s in scores),
    }

    # Agent-only summary stats
    agent_scores = [s for s in scores if "critic_clean" in s]
    if agent_scores:
        n_clean = sum(1 for s in agent_scores if s["critic_clean"])
        summary["critic_clean_rate"] = f"{n_clean}/{len(agent_scores)}"
        summary["avg_sub_queries_per_question"] = round(
            sum(s["n_sub_queries"] for s in agent_scores) / len(agent_scores), 2
        )

    return summary


def _build_pipeline(mode: str):
    if mode == "agent":
        return AgentPipeline()
    if mode == "rerank":
        return RAGPipeline(reranker=Reranker())
    return RAGPipeline()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the FinIntel eval set in one of three modes."
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "rerank", "agent"],
        default="baseline",
        help="Pipeline mode to evaluate. baseline=top-4, rerank=top-4-of-12, agent=LangGraph.",
    )
    parser.add_argument(
        "--rerank", action="store_true",
        help="DEPRECATED: prefer --mode rerank. Sets mode to rerank for backward compat.",
    )
    parser.add_argument(
        "--label", type=str, default=None,
        help="Tag for the results file. Defaults to the mode name.",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.0,
        help="Seconds to sleep between questions. Useful for staying under TPM caps in agent mode.",
    )
    args = parser.parse_args()

    # Backward-compat: --rerank promotes mode to rerank if mode wasn't explicitly set
    mode = "rerank" if args.rerank and args.mode == "baseline" else args.mode
    label = args.label or mode

    eval_set = load_eval_set()
    print(f"Loaded {len(eval_set)} eval questions | mode={mode}")

    pipeline = _build_pipeline(mode)
    print()

    scores: list[dict] = []
    for i, q in enumerate(eval_set, 1):
        short_q = q["question"][:78] + ("…" if len(q["question"]) > 80 else "")
        print(f"[{i:>2}/{len(eval_set)}] {short_q}")
        t0 = time.time()
        try:
            if mode == "agent":
                # Agent's planner does its own filtering; ignore ticker/section_hint metadata
                rag_answer = pipeline.answer(q["question"])
            else:
                rag_answer = pipeline.answer(
                    q["question"],
                    ticker=q.get("ticker"),
                    section=q.get("section_hint"),
                )
            elapsed = time.time() - t0
        except Exception as e:
            print(f"        ERROR: {type(e).__name__}: {e}")
            continue

        s = score_answer(q, rag_answer)
        s["elapsed_sec"] = round(elapsed, 2)
        scores.append(s)

        kr = s["keyword_recall"]
        kr_str = f"{kr:.2f}" if kr is not None else "n/a"
        cov = s["ticker_coverage"]
        cov_str = f"{cov:.2f}" if cov is not None else "n/a"
        ok = (kr is not None and kr >= 0.5) or (s["expected_refusal"] and s["refused"])
        marker = "PASS" if ok else "FAIL"
        print(
            f"        {marker}  recall={kr_str}  coverage={cov_str}  "
            f"citations={s['n_citations']}  sources={s['tickers_in_sources']}  "
            f"{s['elapsed_sec']}s"
        )

        if args.sleep > 0:
            time.sleep(args.sleep)

    summary = aggregate(scores)

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    detail_path = RESULTS_DIR / f"{label}_{ts}_detailed.json"
    summary_path = RESULTS_DIR / f"{label}_{ts}_summary.json"
    detail_path.write_text(json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"SUMMARY ({mode})")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:<40}  {v}")
    print(f"\nSaved details: evals/results/{detail_path.name}")
    print(f"Saved summary: evals/results/{summary_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
