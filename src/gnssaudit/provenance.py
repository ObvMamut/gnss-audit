"""Run IDs, write-once run directories and run manifests (docs/6 sections 7.1 and 7.3)."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def run_id(spec: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON of the expanded run specification, first 12 hex digits."""
    return hashlib.sha256(canonical_json(spec).encode()).hexdigest()[:12]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True, timeout=30).stdout


def git_describe(repo: Path) -> str:
    return _git(repo, "describe", "--always", "--dirty", "--abbrev=12").strip()


def is_dirty(repo: Path) -> bool:
    """Tracked files differ from HEAD (the same rule as `git describe --dirty`)."""
    return bool(_git(repo, "status", "--porcelain", "--untracked-files=no").strip())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def create_run_dir(results: Path, experiment: str, rid: str) -> Path:
    """results/<experiment>/<run_id>/, created fresh; an existing one raises FileExistsError (write-once)."""
    path = results / experiment / rid
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()
    return path


def write_manifest(
    run_dir: Path,
    spec: dict[str, Any],
    describe: str,
    inputs: list[Path],
    timings_s: dict[str, float],
    counts: dict[str, int],
    started_utc: str,
    summary: dict[str, Any],
) -> Path:
    manifest = {
        "run_id": run_dir.name,
        "spec": spec,
        "git_describe": describe,
        "inputs": {str(p): sha256_file(p) for p in inputs},
        "host": {"node": platform.node(), "python": platform.python_version(), "machine": platform.machine()},
        "timings_s": timings_s,
        "counts": counts,
        "started_utc": started_utc,
        "summary": summary,
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path
