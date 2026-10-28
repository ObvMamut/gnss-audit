"""The registered FGI prefixes are the bytes that were verified, and the admitted intervals lie inside them (P2b task 6;
docs/7 sections 3.1 and 4.1)."""

import hashlib
import json

import pytest
import yaml

from gnssaudit.paths import data_root
from gnssaudit.registry import REGISTRY, load_intervals

pytestmark = [pytest.mark.data, pytest.mark.slow]
BYTES_PER_SAMPLE = 2  # interleaved int8 I and Q
FGI = [d for d in yaml.safe_load((REGISTRY / "datasets.yaml").read_text())["datasets"]
       if d["id"] in ("fgi-jt2023", "fgi-spoofrepo")]  # fmt: skip
FILES = [(d["id"], f) for d in FGI for f in d["files"]]


@pytest.mark.parametrize(("dataset", "entry"), FILES, ids=[f["name"] for _, f in FILES])
def test_a_registered_prefix_hashes_as_registered_and_its_file_was_verified(dataset: str, entry: dict) -> None:
    path = data_root() / entry["local_path"]
    if not path.exists():
        pytest.skip(f"{entry['name']} is not fetched")
    assert path.stat().st_size == entry["size_bytes"] == entry["extracted_ranges"][0][1]
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    assert h.hexdigest() == entry["sha256"]
    side = json.loads(path.with_name(path.name + ".prefix.json").read_text())
    assert side["verified"] and side["prefix_sha256"] == entry["sha256"]
    assert f"sha256:{side['full_sha256']}" == entry["published_checksum"]


def test_every_fgi_interval_lies_inside_its_registered_prefix() -> None:
    sizes = {(d, f["name"]): f["size_bytes"] for d, f in FILES}
    rows = [r for r in load_intervals().values() if r["dataset_id"] in ("fgi-jt2023", "fgi-spoofrepo")]
    if not rows:
        pytest.skip("no FGI interval is registered yet")
    for r in rows:
        assert 0 <= r["sample_start"] < r["sample_end"]
        assert r["sample_end"] * BYTES_PER_SAMPLE <= sizes[(r["dataset_id"], r["file"])], r["interval_id"]
