"""Chunking pipeline for SEC filing sections.

Splits cleaned filing text (Risk Factors, MD&A) into overlapping token-sized
chunks suitable for embedding and retrieval. Every chunk carries enough
provenance metadata that the downstream agent can cite the exact source.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Tuned for: ~3-5 chunks fit in any context window; preserves paragraph
# coherence; 150-token overlap keeps cross-boundary context.
DEFAULT_CHUNK_TOKENS = 800
DEFAULT_OVERLAP_TOKENS = 150

# cl100k_base is OpenAI's tokenizer used by GPT-4/3.5/embeddings; close enough
# to most LLM tokenizers that chunk sizes generalize well.
_ENCODING_NAME = "cl100k_base"


@dataclass(frozen=True)
class Chunk:
    """A single text chunk ready for embedding, with full provenance."""

    chunk_id: str        # unique: f"{ticker}_{filing_type}_{accession}_{section}_{i:03d}"
    ticker: str          # AAPL, MSFT, ...
    filing_type: str     # 10-K, 10-Q
    accession: str       # 0000320193-25-000079
    section: str         # risk_factors, mda
    chunk_index: int     # 0-indexed within section
    total_chunks: int    # how many chunks the section produced
    n_tokens: int        # token count under cl100k_base
    text: str            # the chunk content

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _make_splitter(
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> RecursiveCharacterTextSplitter:
    """Token-aware splitter that prefers paragraph -> sentence -> word boundaries."""
    encoding = tiktoken.get_encoding(_ENCODING_NAME)
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_tokens,
        chunk_overlap=overlap_tokens,
        length_function=lambda txt: len(encoding.encode(txt)),
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_section(
    text: str,
    *,
    ticker: str,
    filing_type: str,
    accession: str,
    section: str,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Split one section's text into Chunks with full provenance metadata."""
    splitter = _make_splitter(chunk_tokens, overlap_tokens)
    encoding = tiktoken.get_encoding(_ENCODING_NAME)

    pieces = splitter.split_text(text)
    return [
        Chunk(
            chunk_id=f"{ticker}_{filing_type}_{accession}_{section}_{i:03d}",
            ticker=ticker,
            filing_type=filing_type,
            accession=accession,
            section=section,
            chunk_index=i,
            total_chunks=len(pieces),
            n_tokens=len(encoding.encode(piece)),
            text=piece,
        )
        for i, piece in enumerate(pieces)
    ]


def chunk_processed_corpus(
    processed_dir: Path,
    *,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> Iterator[Chunk]:
    """Walk data/processed/<TICKER>/<FORM>/<ACCESSION>/<section>.txt
    and yield chunks for every section file found.
    """
    for ticker_dir in sorted(p for p in processed_dir.iterdir() if p.is_dir()):
        for form_dir in sorted(p for p in ticker_dir.iterdir() if p.is_dir()):
            for accession_dir in sorted(p for p in form_dir.iterdir() if p.is_dir()):
                for section_file in sorted(accession_dir.glob("*.txt")):
                    text = section_file.read_text(encoding="utf-8")
                    yield from chunk_section(
                        text,
                        ticker=ticker_dir.name,
                        filing_type=form_dir.name,
                        accession=accession_dir.name,
                        section=section_file.stem,
                        chunk_tokens=chunk_tokens,
                        overlap_tokens=overlap_tokens,
                    )


def write_chunks_jsonl(chunks: Iterable[Chunk], output_path: Path) -> int:
    """Serialize chunks to JSONL. Returns the number of records written."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.to_jsonl() + "\n")
            n += 1
    return n