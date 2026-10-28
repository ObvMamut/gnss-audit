"""Experiment runners (docs/10). This is the only layer that imports truth, uq and eval together."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Outcome:
    inputs: list[Path]  # every file read, hashed into the manifest
    counts: dict[str, int] = field(default_factory=dict)  # warnings and exclusions
    summary: dict[str, Any] = field(default_factory=dict)  # gate verdicts and headline numbers


Runner = Callable[[dict[str, Any], Path], Outcome]
REGISTRY: dict[str, Runner] = {}


def _register() -> None:
    from gnssaudit.experiments.admit import run_admit
    from gnssaudit.experiments.e0a import run_e0a
    from gnssaudit.experiments.e0b import run_e0b
    from gnssaudit.experiments.e1a import run_e1a
    from gnssaudit.experiments.e1c import run_e1c_scout
    from gnssaudit.experiments.e2 import run_e2
    from gnssaudit.experiments.e2k import run_e2k
    from gnssaudit.experiments.e2post import run_e2post

    REGISTRY["Admit"] = run_admit
    REGISTRY["E0a"] = run_e0a
    REGISTRY["E0b"] = run_e0b
    REGISTRY["E1a"] = run_e1a
    REGISTRY["E1c-scout"] = run_e1c_scout
    REGISTRY["E2"] = run_e2
    REGISTRY["E2K"] = run_e2k
    REGISTRY["E2post"] = run_e2post


_register()
