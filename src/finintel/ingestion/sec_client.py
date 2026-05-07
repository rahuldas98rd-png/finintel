"""SEC EDGAR filing downloader.

Downloads filings (10-K, 10-Q, etc.) into data/raw/sec-edgar-filings/<TICKER>/<FORM>/
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from sec_edgar_downloader import Downloader

load_dotenv()
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw")

# Phase 1 target universe: 5 mega-caps across tech and finance.
# Big enough for interesting comparisons, small enough to iterate fast on.
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "JPM"]
DEFAULT_FORMS = ["10-K", "10-Q"]


@dataclass(frozen=True)
class DownloadResult:
    """Outcome of a single (ticker, form) download attempt."""

    ticker: str
    form: str
    count: int
    error: str | None = None


def _get_user_agent() -> tuple[str, str]:
    """Parse 'Company Name email@example.com' from SEC_USER_AGENT env var."""
    raw = os.getenv("SEC_USER_AGENT")
    if not raw:
        raise RuntimeError(
            "SEC_USER_AGENT not set. Copy .env.example to .env and add your email."
        )
    name, _, email = raw.rpartition(" ")
    if "@" not in email:
        raise RuntimeError(f"SEC_USER_AGENT must end with an email. Got: {raw!r}")
    return name.strip().strip('"'), email.strip().strip('"')


def _make_downloader() -> Downloader:
    company, email = _get_user_agent()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return Downloader(company, email, str(DATA_DIR))


def download_filings(ticker: str, form: str = "10-K", limit: int = 1) -> int:
    """Download `limit` most recent filings of `form` type for `ticker`.

    Returns the number of new filings downloaded (already-cached filings are skipped).
    """
    dl = _make_downloader()
    return dl.get(form, ticker, limit=limit)


def download_batch(
    tickers: list[str] | None = None,
    forms: list[str] | None = None,
    limit_per_form: int = 4,
) -> list[DownloadResult]:
    """Download multiple (ticker, form) combinations.

    Default: 5 tickers × {10-K, 10-Q} × 4 most recent = 40 filings.
    """
    tickers = tickers or DEFAULT_TICKERS
    forms = forms or DEFAULT_FORMS
    results: list[DownloadResult] = []

    for ticker in tickers:
        for form in forms:
            logger.info("Downloading %d %s filings for %s...", limit_per_form, form, ticker)
            try:
                count = download_filings(ticker, form=form, limit=limit_per_form)
                results.append(DownloadResult(ticker, form, count))
                logger.info("  done: %s %s -> %d filing(s)", ticker, form, count)
            except Exception as e:  # noqa: BLE001 — broad catch is intentional here
                logger.exception("  failed: %s %s", ticker, form)
                results.append(DownloadResult(ticker, form, 0, str(e)))

    return results


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _print_summary(results: list[DownloadResult]) -> None:
    total = sum(r.count for r in results)
    failed = [r for r in results if r.error]
    print("\n" + "=" * 50)
    print(f"Downloaded {total} new filings across {len(results)} (ticker, form) pairs")
    if failed:
        print(f"WARN: {len(failed)} failed:")
        for r in failed:
            print(f"  - {r.ticker} {r.form}: {r.error}")
    else:
        print("All downloads succeeded.")


if __name__ == "__main__":
    _setup_logging()
    results = download_batch(limit_per_form=4)
    _print_summary(results)