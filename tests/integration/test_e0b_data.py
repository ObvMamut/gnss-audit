"""E0b end to end (P2a task 10; P2b task 10): the twin pipeline on a synthetic recording and on the CTTC sample (the
`recording` variant on a registered interval), through the patched GNSS-SDR image. Development runs, not G3. Both need
Docker; the CTTC run also needs the registered recording."""

import json
from pathlib import Path

import pytest
import yaml

from gnssaudit.cli.main import main
from gnssaudit.registry import dataset_file

pytestmark = [pytest.mark.container, pytest.mark.data, pytest.mark.slow]
REPO = Path(__file__).resolve().parents[2]


def run(spec_name: str, results: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.setenv("GNSSAUDIT_RESULTS", str(results))
    assert main(["run", str(REPO / "experiments" / spec_name)]) == 0
    (run_dir,) = (results / "E0b").iterdir()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["spec"] == yaml.safe_load((REPO / "experiments" / spec_name).read_text())
    summary = manifest["summary"]
    receivers = summary["receivers"]  # every receiver run: image, patch, configuration hash and its GNSS-SDR log
    assert len(receivers["runs"]) == summary["receiver_runs"] and set(receivers["logs"]) == set(receivers["runs"])
    assert all((run_dir / rel).is_file() for rel in receivers["logs"].values())
    return summary


def gates(summary: dict[str, object]) -> dict[str, dict[str, object]]:
    g = summary["gates"]
    assert isinstance(g, dict)
    return g


def test_synthetic_twins_pass_the_gates_and_the_ghost_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = run("E0b-synthetic.yaml", tmp_path, monkeypatch)
    assert all(v["passed"] for v in gates(s).values()), gates(s)
    assert s["v6_pass"]
    ghost = s["ghost"]
    assert isinstance(ghost, dict) and ghost["tracked"] and ghost["v7_pass"] and ghost["sanity_pass"]


def test_cttc_twins_pass_the_gates_linearity_and_the_ghost_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    try:
        dataset_file("cttc-2013-sample", "2013_04_04_GNSS_SIGNAL_at_CTTC_SPAIN.dat").stat()
    except FileNotFoundError:
        pytest.skip("the CTTC recording is not fetched")
    s = run("E0b-cttc.yaml", tmp_path, monkeypatch)
    assert all(v["passed"] for v in gates(s).values()), gates(s)
    assert s["linearity_pass"]  # V6 is recorded, not asserted: this floor's lines confound it (docs/8 section 2.1)
    ghost = s["ghost"]
    # ghost sanity is recorded, not asserted, here too: against the C/A-weighted floor the ghost's error SD reads 0.78
    # x B3 (PRN 22 at 730 Hz), about 30 sampling errors off, because the ghost sees this structured floor at its own
    # Doppler (docs/3 section 7)
    assert isinstance(ghost, dict) and ghost["v7_pass"] and "sanity_pass" in ghost
    ghz = s["ghz"]
    assert isinstance(ghz, dict) and ghz["default"]["identified"] and ghz["default"]["passed"]
