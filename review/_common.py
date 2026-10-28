"""Shared helpers for the review studies: result files with provenance."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, float) and not np.isfinite(x):
        return str(x)
    return x


def save(study: str, payload: dict[str, Any], meta_extra: dict[str, Any] | None = None) -> Path:
    """Write review/results/<study>.json: the provenance meta (with meta_extra's keys added, e.g. pinned inputs) and
    the payload."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    try:
        uv = subprocess.run(["uv", "pip", "freeze"], capture_output=True, text=True, check=False).stdout
    except OSError:
        uv = ""
    meta = {
        "study": study,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "packages": [ln for ln in uv.splitlines() if ln.split("==")[0] in {"numpy", "scipy"}],
        **(meta_extra or {}),
    }
    path = RESULTS / f"{study}.json"
    path.write_text(json.dumps(_jsonable({"meta": meta, **payload}), indent=2) + "\n")
    print(f"wrote {path}")
    return path
