"""E2 end to end in miniature (P2b task 11) on GNSS-SDR's CTTC sample: a survey, the clean twins, one noise cell in
modes A and B with two ghosts, B1's first-order prediction for the ghosts, and one repeat. It pins the prediction's
sign and timing on the binary: dPhi_ghost follows H n_iota with correlation near +1 (docs/3 section 2.2). Under (a')
(P2b part 2, task 5): the clean twins keep their ghosts' absolute errors, and the summary carries the front-end cell,
B1's record with Var(n_iota), and the decisions. Under ghost truth (P2c task 3, the second parametrization): the ghost
ceiling, check C-1 on the binary (ghost truth on the jammed track equals e_c + dPhi up to a constant, within R-1's
1e-3 cycle), the Route B arm and G3's clauses."""

import json
from pathlib import Path

import pytest
import yaml

from gnssaudit.cli.main import main
from gnssaudit.registry import dataset_file

pytestmark = [pytest.mark.container, pytest.mark.data, pytest.mark.slow]

SPEC = {
    "id": "E2",
    "segments": {"C": {"interval": "cttc-2013-full", "item_type": "ishort", "fs_hz": 4.0e6}},
    "template": "1C",
    "frontend_bw_hz": 2.0e6,
    "quant": {"max_level": 32767, "zero_level": True},
    "seed": 5,
    "max_channels": 7,
    "survey_channels": 8,
    # a false track on the +580 kHz line reads 35.1 to 35.4 dB-Hz in 14 of 48 stock surveys (P2b finding (b); R-8, P2c)
    "min_cn0_dbhz": 36,
    "warmup_s": 30.0,
    "pre_window_s": 20.0,
    "onset_s": 55.0,
    "ghosts": {"cn0_db_hz": 45.0, "doppler_hz": [1500.0, -2500.0]},
    "families": {"noise": {"bw_hz": 2.0e6}},
    "levels_db_hz": [62.0],
    "mode_b": {"tau_s": 0.01, "side_tau_s": [], "side_level_db_hz": 62.0, "side_families": []},
    "repeat_fraction": 0.2,
    "rules": {
        "jmax_tol": 0.10,
        "slope_tol": 0.10,
        "excess_se_max": 0.05,
        "residual_max": 0.05,
        "lost_truth_max_s": 1.0,
        "k_small": 51,
        "bootstrap": {"block_s": 1.0, "n_boot": 400, "seed": 7},
    },
    "pipelines": 2,
    "workers_a": 1,
    "timeout_s": 1800,
    "abort_on_gate_failure": False,
    "receiver_overrides": {"Tracking_1C.max_carrier_lock_fail": "1000000", "Tracking_1C.enable_fll_pull_in": "true"},
}


@pytest.mark.parametrize("truth", ["route_b", "ghost"])
def test_a_miniature_e2_writes_its_units_and_pins_b1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, truth: str
) -> None:
    try:
        dataset_file("cttc-2013-sample", "2013_04_04_GNSS_SIGNAL_at_CTTC_SPAIN.dat").stat()
    except FileNotFoundError:
        pytest.skip("the CTTC recording is not fetched")
    spec = tmp_path / "e2.yaml"
    spec.write_text(yaml.safe_dump(SPEC if truth == "route_b" else {**SPEC, "truth": "ghost"}))
    monkeypatch.setenv("GNSSAUDIT_RESULTS", str(tmp_path / "results"))
    assert main(["run", str(spec)]) == 0
    (run_dir,) = (tmp_path / "results" / "E2").iterdir()
    s = json.loads((run_dir / "manifest.json").read_text())["summary"]
    units = {p.parent.name for p in (run_dir / "units").glob("*/done.json")}
    assert {"survey-C", "C-clean-A", "C-clean-A2", "C-clean-B10", "C-grid-noise-62-A", "C-grid-noise-62-B10"} <= units
    assert any(u.startswith("repeat-") for u in units)
    for clean in ("C-clean-A", "C-clean-B10"):  # e_c of the paired excess, hashed with the unit
        done = json.loads((run_dir / "units" / clean / "done.json").read_text())["files"]
        assert len([f for f in done if f.startswith("ghost_abs_ch")]) == 2
    seg = s["segments"]["C"]
    assert seg["var_b_over_a"]["noise"]["62.0"] > 0 and len(seg["ghost_prn"]) == 2
    resid = seg["residual"]["noise"]["62.0"]
    assert len(resid) == 2
    for prn, v in resid.items():  # the sign and timing pin: the first-order prediction is the ghost's dPhi
        assert v["corr"] > 0.9 and v["r"] < 0.2, (prn, v)
        assert 0 < v["var_n_iota_rad2"] < 1 and v["floor_plus_var"] == v["floor"] + v["var_n_iota_rad2"]
    cell = seg["front_end"]["noise"]["62.0"]
    assert cell["verdict"] in {"pass", "fail", "unresolved"} and cell["ratio_se"] > 0, cell
    assert len(cell["slopes"]) == 2 and all(g["n"] > 10_000 for g in cell["excess"].values())
    d = s["decisions"]
    assert set(d) >= {"j_max_db_hz", "ceiling_db_hz", "k_level_db_hz", "k", "option_a", "residual_pass"}
    assert d["j_max_db_hz"]["noise"] == seg["j_max_db_hz"]["noise"] and d["k"] is None  # one segment, no K-study
    gates = s["gates"]
    assert {g.split("[")[0] for g in gates} >= {"G-identity", "G-null", "G-repeat"}
    assert gates["G-repeat[repeat]"]["bytes_differ"] == 0
    if truth == "route_b":  # E2-v2's summary: none of ghost truth's keys
        assert not {"ghost_ceiling_db_hz", "route_b_arm"} & set(d) and "g3_clauses" not in s
        assert not {"ghost_ceiling_db_hz", "ghost_check"} & set(seg)
        return
    assert d["ghost_ceiling_db_hz"]["noise"] == seg["ghost_ceiling_db_hz"]["noise"]
    assert d["k_level_db_hz"] == {"noise": None}  # the miniature has no K-study
    checks = seg["ghost_check"]["noise"]["62.0"]
    assert len(checks) == 2
    for ch, c in checks.items():  # C-1 on the binary
        assert c.get("computed", True) and c["n"] > 10_000 and c["rms_cyc"] <= 1e-3, (ch, c)
    arm = d["route_b_arm"]
    assert arm["ceiling_db_hz"] == d["ceiling_db_hz"] and arm["k_study"]["k"] is None
    assert arm["j_max_ta"] == {"computed": False, "reason": "the spec pins no s29_result"}  # segment C has no s29
    g3 = s["g3_clauses"]
    assert g3["gates"]["count"] == len(gates) and g3["k"] == {"k": None, "chosen": False} and g3["passed"] is False
