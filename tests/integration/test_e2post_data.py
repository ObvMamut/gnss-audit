"""E2post on the finished E2 run of segment A (P2b part 2, tasks 1 and 3): structure only. Every cell and ghost is
present with finite numbers and gate A's two booleans are written; B1's r is E2's to the spec's tolerance and its
prediction rebuilt from the re-rendered interference is E2's up to the twins' int16 rounding; its post hoc keys are
present and finite; the source run is left as it was (a stat-only check here; the unit test hashes every file). Needs
the E2 run under the results root and the registered FGI recording (only stat)."""

import math
from pathlib import Path

import pytest
import yaml

from gnssaudit.experiments.e2post import run_e2post
from gnssaudit.paths import results_root
from gnssaudit.registry import dataset_file

pytestmark = pytest.mark.data
REPO = Path(__file__).resolve().parents[2]
SPEC = yaml.safe_load((REPO / "experiments" / "E2post-A.yaml").read_text())


def stats(root: Path) -> dict[str, tuple[int, int]]:
    return {str(p.relative_to(root)): (p.stat().st_mtime_ns, p.stat().st_size) for p in [root, *root.rglob("*")]}


def test_e2post_on_segment_a_writes_every_cell(tmp_path: Path) -> None:
    src = results_root() / SPEC["source_run"]
    if not (src / "manifest.json").is_file():
        pytest.skip(f"the E2 run {SPEC['source_run']} is not under {results_root()}")
    try:
        dataset_file("fgi-jt2023", "JT23_4.1.5_L1_E1.iq").stat()
    except FileNotFoundError:
        pytest.skip("the FGI recording JT23_4.1.5 is not fetched")
    before = stats(src)
    run_dir = tmp_path / "E2post"
    run_dir.mkdir()
    out = run_e2post(SPEC, run_dir)
    assert stats(src) == before
    s = out.summary
    assert (run_dir / "summary.json").is_file() and set(s["cells"]) == set(SPEC["families"])
    for fam in SPEC["families"]:
        assert set(s["cells"][fam]) == {f"{float(lv):g}" for lv in SPEC["levels_db_hz"]}
        for c in s["cells"][fam].values():
            assert set(c["ghosts"]) == {str(ch) for ch in SPEC["ghost_channels"]}
            assert math.isfinite(c["ratio"]) and math.isfinite(c["ratio_se"]) and c["ratio_se"] > 0
            assert c["verdict"] in {"pass", "fail", "unresolved"}
            for g in c["ghosts"].values():
                assert g["n"] > 1000 and all(math.isfinite(g[k]) for k in ("slope", "excess_a", "excess_b"))
    assert isinstance(s["gate_A"]["A1"], bool) and isinstance(s["gate_A"]["A2"], bool)
    b1 = SPEC["b1"]
    for fam in b1["families"]:
        assert set(s["b1"][fam]) == {f"{float(lv):g}" for lv in b1["levels_db_hz"]}
        for c in s["b1"][fam].values():
            assert set(c) == {str(ch) for ch in SPEC["ghost_channels"]}
            for g in c.values():
                assert g["r_diff"] <= b1["r_tol"] and 0 < g["var_n_iota_rad2"] < 1
                # the same draw as E2's projection: rounding alone separates them (a few % of the prediction's SD)
                assert g["pred_rms_diff_cyc"] < 0.1 * g["pred_sd_cyc"]
                ph = g["post_hoc"]  # reported only: structure here, the numbers are task 3's report
                ls, sp, dc = ph["lag_scan"], ph["spectrum"], ph["decomposition"]
                assert (
                    ls["lags"] == list(range(-5, 6)) and len(ls["r"]) == 11 and all(math.isfinite(v) for v in ls["r"])
                )
                assert (
                    isinstance(ls["argmin_lag"], int)
                    and -5 <= ls["argmin_lag"] <= 5
                    and ls["corr_sign_lag0"] in (-1, 1)
                )
                assert all(math.isfinite(v) for v in [*sp.values(), *dc.values(), ls["corr_lag0"]])
                assert (
                    0 <= sp["share_below_50_hz"] <= 1 and 0 <= dc["cross_share"] <= 1 and 0 <= dc["second_share"] <= 1
                )
    assert "post hoc" in s["b1_post_hoc_note"]
