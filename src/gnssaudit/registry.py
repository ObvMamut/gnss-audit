"""Dataset and interval registries in registry/ (docs/7 sections 3.1 and 4.1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from gnssaudit.paths import data_root

REGISTRY = Path(__file__).resolve().parents[2] / "registry"


def load_intervals(path: Path = REGISTRY / "intervals.yaml") -> dict[str, dict[str, Any]]:
    """Interval entries keyed by interval_id."""
    out: dict[str, dict[str, Any]] = {}
    for entry in yaml.safe_load(path.read_text())["intervals"] or []:
        if entry["interval_id"] in out:
            raise ValueError(f"duplicate interval_id {entry['interval_id']}")
        out[entry["interval_id"]] = entry
    return out


def dataset_file(dataset_id: str, name: str, path: Path = REGISTRY / "datasets.yaml") -> Path:
    """Absolute path of a registered file, under the data root."""
    for entry in yaml.safe_load(path.read_text())["datasets"]:
        if entry["id"] == dataset_id:
            for f in entry["files"]:
                if f["name"] == name:
                    return data_root() / f["local_path"]
    raise KeyError(f"{dataset_id}/{name} is not registered")


@dataclass(frozen=True)
class PublishedFile:
    """A file of a dataset not yet fetched, as its record publishes it (`published_files`)."""

    dataset_id: str
    metax_id: str  # Fairdata Metax dataset identifier, for the download authorisation
    record_path: str
    name: str
    size_bytes: int
    sha256: str  # lower-case hex


def published_file(dataset_id: str, name: str, path: Path = REGISTRY / "datasets.yaml") -> PublishedFile:
    """The record's entry for one published file, with its dataset's Metax identifier."""
    for entry in yaml.safe_load(path.read_text())["datasets"]:
        if entry["id"] == dataset_id:
            for f in entry.get("published_files") or []:
                if f["name"] == name:
                    algo, _, digest = str(f["published_checksum"]).partition(":")
                    if algo != "sha256":
                        raise ValueError(f"{dataset_id}/{name}: checksum {algo!r} is not sha256")
                    return PublishedFile(dataset_id, str(entry["metax_id"]), str(f["record_path"]), name,
                                         int(f["size_bytes"]), digest.lower())  # fmt: skip
    raise KeyError(f"{dataset_id}/{name} is not a published file of the registry")
