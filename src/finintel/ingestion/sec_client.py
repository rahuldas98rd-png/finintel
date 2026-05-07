"""SEC EDGAR filing downloader.

Thin wrapper around `sec-edgar-downloader` that:
- Reads SEC_USER_AGENT from .env (required by SEC ToS)
- Writes filings to data/raw/sec-edgar-filings/<TICKER>/<FORM>/...
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from sec_edgar_downloader import Downloader

load_dotenv()

DATA_DIR = Path("data/raw")


def _get_user_agent() -> tuple[str, str]:
    """Parse 'Company Name email@example.com' from SEC_USER_AGENT."""
    raw = os.getenv("SEC_USER_AGENT")
    if not raw:
        raise RuntimeError(
            "SEC_USER_AGENT not set. Copy .env.example to .env and add your email."
        )
    name, _, email = raw.rpartition(" ")
    if "@" not in email:
        raise RuntimeError(
            f"SEC_USER_AGENT must end with an email. Got: {raw!r}"
        )
    return name.strip().strip('"'), email.strip().strip('"')


def download_filings(
    ticker: str,
    filing_type: str = "10-K",
    limit: int = 1,
) -> Path:
    """Download `limit` filings of `filing_type` for `ticker`.

    Returns the path where filings were written.
    """
    company, email = _get_user_agent()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dl = Downloader(company, email, str(DATA_DIR))
    dl.get(filing_type, ticker, limit=limit)

    out = DATA_DIR / "sec-edgar-filings" / ticker / filing_type
    print(f"✓ Downloaded {limit} {filing_type} filing(s) for {ticker} → {out}")
    return out


if __name__ == "__main__":
    # Smoke test: grab Apple's most recent 10-K
    download_filings("AAPL", filing_type="10-K", limit=1)