"""SEC filing parser.

Extracts the primary HTML document from SEC EDGAR's multi-document SGML
"full-submission.txt" files, then surgically extracts named sections
(Risk Factors, MD&A, etc.) as clean text.

Strategy developed in notebooks/01_parsing_exploration.ipynb:
  - SEC section headers use 'Item N.' (period+space) format
  - Cross-references use 'Item N of' or 'Item N,' (no period)
  - The TOC uses periods too, but appears earlier in the document
  - Therefore: 'last period match' for an item = the body section header
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SGML document extraction
# ---------------------------------------------------------------------------

# Each <DOCUMENT> in full-submission.txt has TYPE, SEQUENCE, FILENAME,
# optional DESCRIPTION, then a <TEXT>...</TEXT> payload.
_DOC_PATTERN = re.compile(
    r"<DOCUMENT>\s*"
    r"<TYPE>([^\n<]+)\s*"
    r"<SEQUENCE>([^\n<]+)\s*"
    r"<FILENAME>([^\n<]+)\s*"
    r"(?:<DESCRIPTION>[^\n<]*\s*)?"
    r"<TEXT>(.*?)</TEXT>\s*"
    r"</DOCUMENT>",
    re.DOTALL,
)


def extract_primary_html(submission_text: str, filing_type: str) -> str | None:
    """Extract the primary HTML document matching `filing_type` from raw SGML."""
    for doc_type, _seq, _fname, html in _DOC_PATTERN.findall(submission_text):
        if doc_type.strip() == filing_type:
            return html
    return None


# ---------------------------------------------------------------------------
# HTML -> clean text
# ---------------------------------------------------------------------------

def html_to_clean_text(html: str) -> str:
    """Convert iXBRL HTML into clean human-readable text.

    Removes display:none iXBRL metadata blocks, collapses whitespace runs.
    """
    soup = BeautifulSoup(html, "lxml")
    for hidden in soup.find_all(style=lambda s: s and "display:none" in s.lower()):
        hidden.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------

def find_section_body(text: str, item_id: str) -> int | None:
    """Find the start position of an Item's body section header.

    SEC section headers always use 'Item N.' (period+space). The TOC also
    uses this format but appears earlier in the document, so the LAST match
    is the body.
    """
    pattern = re.compile(rf"\bItem\s+{re.escape(item_id)}\.\s", re.IGNORECASE)
    matches = list(pattern.finditer(text))
    return matches[-1].start() if matches else None


def extract_section(
    text: str,
    item_id: str,
    end_candidates: list[str],
) -> str | None:
    """Extract the body of an Item, ending at the earliest end_candidate that
    appears after the section's start. Falls back to end-of-text if none found.
    """
    start = find_section_body(text, item_id)
    if start is None:
        return None
    end = len(text)
    for next_id in end_candidates:
        next_start = find_section_body(text, next_id)
        if next_start is not None and next_start > start:
            end = min(end, next_start)
    return text[start:end].strip()


# ---------------------------------------------------------------------------
# Per-filing-type configuration
# ---------------------------------------------------------------------------

# Maps section names to (item_id, end_candidates).
# Note: 10-K Item 7 is MD&A; 10-Q Part I Item 2 is MD&A. The numbering schemes
# of these two filing types are NOT interchangeable.
SECTION_SPECS: dict[str, dict[str, tuple[str, list[str]]]] = {
    "10-K": {
        "risk_factors": ("1A", ["1B", "1C", "2"]),
        "mda":          ("7",  ["7A", "8"]),
    },
    "10-Q": {
        "mda":          ("2",  ["3", "4"]),
        "risk_factors": ("1A", ["2", "3", "4", "5", "6"]),
    },
}

# Filing types where, if MD&A is suspiciously short, we should look in
# common exhibit documents that often contain the actual MD&A by reference.
# EX-13 = Annual Report to Shareholders (used by JPMorgan, Berkshire, etc.)
# EX-99 / EX-99.1 = various supplementary materials
_MDA_FALLBACK_EXHIBITS = ("EX-13", "EX-99.1", "EX-99")

# Below this length, an extracted section is almost certainly a "see Annual
# Report" pointer rather than actual content.
_SUSPICIOUSLY_SHORT_THRESHOLD = 5_000

# Heading patterns for finding MD&A when Item-number extraction fails
# (banks/conglomerates that embed annual-report content directly).
_MDA_HEADING_PATTERN = re.compile(
    r"Management.{1,3}s\s+[Dd]iscussion\s+and\s+[Aa]nalysis",
    re.IGNORECASE,
)
_MDA_END_PATTERNS = (
    r"Quantitative\s+and\s+Qualitative\s+Disclosures\s+About\s+Market\s+Risk",
    r"Report\s+of\s+Independent\s+Registered\s+Public\s+Accounting\s+Firm",
    r"Consolidated\s+Statements\s+of\s+Income",
    r"Audited\s+Financial\s+Statements",
)


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedFiling:
    """Extracted sections from a single SEC filing."""

    filing_type: str           # "10-K", "10-Q"
    accession: str             # e.g., "0000320193-25-000079"
    sections: dict[str, str]   # section_name -> clean text


def parse_filing(submission_path: Path, filing_type: str) -> ParsedFiling | None:
    """Parse a full-submission.txt file into a ParsedFiling.

    For 10-Ks where MD&A in the primary document is suspiciously short
    (indicating incorporation-by-reference, common for financial holding
    companies), falls back to looking for MD&A inside EX-13 / EX-99 exhibits.
    """
    if filing_type not in SECTION_SPECS:
        raise ValueError(f"No section spec for filing type {filing_type!r}")

    accession = submission_path.parent.name
    raw = submission_path.read_text(encoding="utf-8", errors="replace")

    html = extract_primary_html(raw, filing_type)
    if html is None:
        logger.warning("No primary %s document found in %s", filing_type, accession)
        return None

    text = html_to_clean_text(html)

    sections: dict[str, str] = {}
    for section_name, (item_id, ends) in SECTION_SPECS[filing_type].items():
        extracted = extract_section(text, item_id, ends)
        if extracted is None:
            logger.warning(
                "Could not extract %s (Item %s) from %s/%s",
                section_name, item_id, filing_type, accession,
            )
            continue
        sections[section_name] = extracted

    # Fallback: 10-K MD&A may be (a) in an EX-13/EX-99 exhibit (incorporation
    # by reference, e.g., Berkshire), or (b) embedded directly in the primary
    # 10-K HTML using annual-report-style headings (e.g., JPMorgan, BAC).
    # Try both, pick the longest match.
    if (
        filing_type == "10-K"
        and "mda" in sections
        and len(sections["mda"]) < _SUSPICIOUSLY_SHORT_THRESHOLD
    ):
        candidates: list[str] = []

        exhibit_match = _find_mda_in_exhibits(raw)
        if exhibit_match:
            candidates.append(exhibit_match)

        primary_match = _extract_mda_by_heading(text)
        if primary_match:
            candidates.append(primary_match)

        if candidates:
            fallback = max(candidates, key=len)
            if len(fallback) > len(sections["mda"]):
                logger.info(
                    "Fallback MD&A used for %s/%s (%d -> %d chars)",
                    filing_type, accession, len(sections["mda"]), len(fallback),
                )
                sections["mda"] = fallback

    return ParsedFiling(filing_type=filing_type, accession=accession, sections=sections)


def _find_mda_in_exhibits(raw_submission: str) -> str | None:
    """Search common exhibits for MD&A content. Returns the longest plausible match."""
    candidates: list[str] = []

    for exhibit_type in _MDA_FALLBACK_EXHIBITS:
        html = extract_primary_html(raw_submission, exhibit_type)
        if html is None:
            continue
        text = html_to_clean_text(html)

        # Try the standard 10-K item structure first
        mda = extract_section(text, "7", ["7A", "8"])
        if mda and len(mda) > _SUSPICIOUSLY_SHORT_THRESHOLD:
            candidates.append(mda)
            continue

        # If the exhibit isn't structured by Item numbers (annual reports
        # often aren't), find the section by heading text
        mda = _extract_mda_by_heading(text)
        if mda:
            candidates.append(mda)

    return max(candidates, key=len) if candidates else None


def _extract_mda_by_heading(text: str) -> str | None:
    """Find MD&A by heading text rather than Item number.

    When multiple occurrences exist (cross-references, footers, body header),
    picks the (start, end) pair with the LONGEST content between them — that
    is, by definition, the actual section body. This is more robust than
    'first match' or 'last match' heuristics.
    """
    start_positions = [m.start() for m in _MDA_HEADING_PATTERN.finditer(text)]
    if not start_positions:
        return None

    end_positions: list[int] = []
    for p in _MDA_END_PATTERNS:
        end_positions.extend(m.start() for m in re.finditer(p, text, re.IGNORECASE))
    if not end_positions:
        return None

    best_length = 0
    best_span: tuple[int, int] | None = None
    for start in start_positions:
        ends_after = [e for e in end_positions if e > start]
        if not ends_after:
            continue
        end = min(ends_after)  # nearest end-marker after this start
        length = end - start
        if length > best_length:
            best_length = length
            best_span = (start, end)

    if best_span is None or best_length < _SUSPICIOUSLY_SHORT_THRESHOLD:
        return None
    return text[best_span[0] : best_span[1]].strip()