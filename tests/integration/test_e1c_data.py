"""E1 arm c scout: the month result written by `gnssaudit run experiments/E1c-scout.yaml` is self-consistent."""

import json
from pathlib import Path

import pytest
import yaml

from gnssaudit.paths import results_root

pytestmark = pytest.mark.data
SPEC = yaml.safe_load((Path(__file__).resolve().parents[2] / "experiments" / "E1c-scout.yaml").read_text())
RUNS = sorted(
    p for p in (results_root() / "E1c-scout").glob("*/result.json") if "month" in json.loads(p.read_text())
) if (results_root() / "E1c-scout").is_dir() else []  # fmt: skip
if not RUNS:
    pytest.skip("no E1c-scout month run", allow_module_level=True)


@pytest.mark.parametrize("path", RUNS, ids=lambda p: p.parent.name)
def test_month_totals_are_consistent(path: Path) -> None:
    result = json.loads(path.read_text())
    month, g = result["month"], result["month"]["g1c"]
    assert g["label"].startswith("provisional") and g["decision"] == "D-32"
    assert sum(g["events_by_partner_state"].values()) == len(g["events"])
    admitted = [e for e in g["events"] if e["admitted"]]
    assert all(e["partner_state"] == "clear" and e["kp_max"] <= g["kp_max_allowed"] for e in admitted)
    assert g["admitted"]["events"] == len(admitted) == sum(c["events"] for c in g["classes"].values())
    total = sum(e["flagged_min"] for e in admitted)
    assert g["admitted"]["flagged_min"] == pytest.approx(total)
    assert sum(c["flagged_min"] for c in g["classes"].values()) == pytest.approx(total)
    excluded = [e for e in g["events"] if e["partner_state"] == "clear" and not e["admitted"]]
    assert g["clear_kp_excluded"]["events"] == len(excluded)
    assert g["passed_events"] == (total >= g["min_clear_flagged_min"])
    assert g["passed"] == (g["passed_events"] and g["coordinates"]["passed"])
    assert all(e["flagged_min"] * 60.0 <= e["duration_s"] / month["epoch_s"] + 1.0 + 1e-9 for e in g["events"])
    assert result["screen"]["pick"]["station"] == month["station"]
