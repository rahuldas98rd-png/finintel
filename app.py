"""Entry point for Hugging Face Spaces deployment.

HF Spaces looks for ``app.py`` at the repo root. The actual Streamlit
UI lives under ``src/finintel/ui/app.py`` (proper src/ layout). This
shim adds ``src/`` to sys.path and imports the real app module —
importing triggers all module-level Streamlit code (st.set_page_config,
sidebar, query handling, etc.).

For local development, you can still run either:

    uv run streamlit run src/finintel/ui/app.py    # canonical path
    uv run streamlit run app.py                    # via this shim

Both behave identically.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable so the `finintel` package resolves
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Importing the UI module triggers all module-level Streamlit code,
# which is how Streamlit apps render. The noqa below silences flake8's
# "imported but unused" — the import IS the side effect.
import finintel.ui.app  # noqa: F401
