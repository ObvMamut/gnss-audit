"""Data and results roots, outside git (D-15; docs/6 section 5)."""

from __future__ import annotations

import os
from pathlib import Path


def _root(var: str, name: str) -> Path:
    value = os.environ.get(var)
    return Path(value) if value else Path.home() / "all" / "private" / name


def data_root() -> Path:
    """$GNSSAUDIT_DATA, by default ~/all/private/GNSS-data."""
    return _root("GNSSAUDIT_DATA", "GNSS-data")


def results_root() -> Path:
    """$GNSSAUDIT_RESULTS, by default ~/all/private/GNSS-results."""
    return _root("GNSSAUDIT_RESULTS", "GNSS-results")
