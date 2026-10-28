"""E1 arm a end to end on the fetched UGE data (docs/10 section 2.2)."""

import json
from pathlib import Path

import pytest
import yaml

from gnssaudit.experiments.e1a import run_e1a
from gnssaudit.experiments.uge import SCENARIO_DIR

pytestmark = pytest.mark.data
if not SCENARIO_DIR.is_dir():
    pytest.skip("UGE data not fetched", allow_module_level=True)

SPEC = yaml.safe_load((Path(__file__).resolve().parents[2] / "experiments" / "E1a.yaml").read_text())


def test_e1a_writes_audit_t3_sensitivity_and_g1(tmp_path: Path) -> None:
    spec = {**SPEC, "bootstrap": {"n_boot": 20, "seed": 1}}
    outcome = run_e1a(spec, tmp_path)
    result = json.loads((tmp_path / "result.json").read_text())
    models = {r["model"] for r in result["coverage"]}
    assert models == {"B0", "B1", "B2", "B4-censored-at-bound", "B4-uncensored-only", "B6"}
    assert any(r["class"] == "all" for r in result["coverage"])
    clean = [r for r in result["coverage"] if r["class"] == "clean"]  # H1a compares with the clean intervals
    assert {r["model"] for r in clean} == models and all(r["n"] > 0 for r in clean)
    assert len({r["class"] for r in result["coverage"]}) >= 5
    t3 = result["t3_cpstdev"]
    assert all(sum(row["histogram"]) == row["n"] for row in t3)
    assert {"rate_code14", "rate_code15"} <= set(t3[0])
    sens = result["sensitivity"]
    assert len(sens["shifts"]) == 6 and "withdrawn" in sens and "extra_position" in sens
    g1 = result["g1"]
    assert set(g1["criteria"]) == {"coordinates", "jam_duration", "route_d_sats", "kp"}
    assert g1["criteria"]["route_d_sats"]["min_fraction"] == SPEC["g1"]["min_epoch_fraction"]
    assert isinstance(g1["passed"], bool) and outcome.summary["g1_passed"] == g1["passed"]
