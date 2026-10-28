"""E0(a) end to end on the fetched UGE data (docs/10 section 2.1)."""

import json
from pathlib import Path

import pytest
import yaml

from gnssaudit.experiments.e0a import run_e0a
from gnssaudit.experiments.uge import SCENARIO_DIR

pytestmark = pytest.mark.data
if not SCENARIO_DIR.is_dir():
    pytest.skip("UGE data not fetched", allow_module_level=True)

SPEC = yaml.safe_load((Path(__file__).resolve().parents[2] / "experiments" / "E0a.yaml").read_text())


def test_e0a_writes_coverage_per_model_interval_and_band(tmp_path: Path) -> None:
    spec = {**SPEC, "bootstrap": {"n_boot": 50, "seed": 1}}
    outcome = run_e0a(spec, tmp_path)
    result = json.loads((tmp_path / "result.json").read_text())
    assert {r["model"] for r in result["coverage"]} == {"B0", "B1"}
    assert {r["interval"] for r in result["coverage"]} == {"uge-1.6.4-clean-pre", "uge-1.6.4-clean-post"}
    assert {"G1", "G2", "E1", "E7"} <= {r["band"] for r in result["coverage"]}
    assert set(result["b1_c_i_m2hz"]) == {"uge-1.6.4-clean-pre", "uge-1.6.4-clean-post"}
    gate = result["gate"]
    assert gate["name"] == "G-clean-cal" and isinstance(gate["passed"], bool)
    assert outcome.summary["gate_passed"] == gate["passed"]
    assert all(p.exists() for p in outcome.inputs)
    assert outcome.counts["out_of_scope_rows"] > 0
