"""S34's own logic (P2c task 5): the front-end acceptance rule, the conditional range, the order check and the pins.
Synthetic mappings only; no pinned input is read."""

import copy
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml
from scipy.stats import chi2

import review.s34_front_end_acceptance as s34
from gnssaudit.provenance import run_id
from review.s30_k_rule_null import SourceRefused
from review.s34_front_end_acceptance import accept, check_order, combined_range, conditional_range, pinned_json

FAMILIES = ("noise", "swept_cw")
RUN3_LEVELS = (50.0, 55.0, 60.0, 65.0, 70.0)
S33_LEVELS = (*RUN3_LEVELS, 75.0, 80.0, 85.0)
TAUS = ("0.001", "0.01", "0.1")


def lk(level: float) -> str:
    return f"{level:.1f}"


def make_s33(*, ratio_b: float = 1.0, ta: Mapping[str, float] | None = None, se: float = 0.02) -> dict[str, Any]:
    """s33's JSON shape: cells[segment][family][level] with ratio_B, ratio_TA[tau] and slope_TA[tau]. `ta` gives each
    tau's true-front-end ratio (default 1 at every tau); the same cell for every level, family and segment."""
    ta = ta or dict.fromkeys(TAUS, 1.0)
    cell = {
        "ratio_B": ratio_b,
        "ratio_B_se": se,
        "ratio_TA": dict(ta),
        "ratio_TA_se": dict.fromkeys(TAUS, 0.01),
        "slope_TA": dict.fromkeys(TAUS, 1.0),
    }
    return {"cells": {s: {f: {lk(lv): copy.deepcopy(cell) for lv in S33_LEVELS} for f in FAMILIES} for s in "AB"}}


def make_run3(*, ratio: float = 1.0, se: float = 0.02, kept: bool = True) -> dict[str, Any]:
    fe = {f: {lk(lv): {"ratio": ratio, "ratio_se": se} for lv in RUN3_LEVELS} for f in FAMILIES}
    tk = {f: dict.fromkeys(map(lk, RUN3_LEVELS), kept) for f in FAMILIES}
    return {"segments": {s: {"front_end": copy.deepcopy(fe), "truth_kept": copy.deepcopy(tk)} for s in "AB"}}


def cell_ids() -> list[tuple[str, str]]:
    return [(f, lk(lv)) for f in FAMILIES for lv in RUN3_LEVELS]


def test_identical_inputs_are_accepted() -> None:
    r = accept(make_run3(), make_s33(), "A")
    assert r["accepted"] is True
    assert r["n"] == 10
    assert r["chi2"] == 0.0
    assert r["max_abs_z"] == 0.0
    assert set(r) == {"z", "chi2", "n", "max_abs_z", "accepted"}
    assert len(r["z"]) == 10


def test_one_cell_at_z_3_1_is_rejected() -> None:
    run3 = make_run3()
    se = math.hypot(0.02, 0.02)
    run3["segments"]["A"]["front_end"]["noise"]["60.0"]["ratio"] = 1.0 + 3.1 * se
    r = accept(run3, make_s33(), "A")
    assert r["max_abs_z"] == pytest.approx(3.1)
    assert r["chi2"] == pytest.approx(3.1**2)
    assert r["chi2"] < chi2.ppf(0.95, 10)  # chi2 alone would pass; the max |z| < 3 clause rejects
    assert r["accepted"] is False


def test_chi2_18_4_over_ten_cells_is_rejected() -> None:
    assert chi2.ppf(0.95, 10) == pytest.approx(18.307, abs=1e-3)
    run3 = make_run3()
    z = math.sqrt(18.4 / 10)  # every |z| is 1.36, far below 3
    se = math.hypot(0.02, 0.02)
    for f, lv in cell_ids():
        run3["segments"]["A"]["front_end"][f][lv]["ratio"] = 1.0 + z * se
    r = accept(run3, make_s33(), "A")
    assert r["chi2"] == pytest.approx(18.4)
    assert r["max_abs_z"] < 3
    assert r["accepted"] is False


def test_chi2_just_inside_the_bound_is_accepted() -> None:
    run3 = make_run3()
    z = math.sqrt(18.0 / 10)
    se = math.hypot(0.02, 0.02)
    for f, lv in cell_ids():
        run3["segments"]["A"]["front_end"][f][lv]["ratio"] = 1.0 - z * se
    r = accept(run3, make_s33(), "A")
    assert r["chi2"] == pytest.approx(18.0)
    assert r["accepted"] is True


def test_seven_cells_present_are_rejected() -> None:
    run3 = make_run3()
    for f, lv in cell_ids()[:3]:
        del run3["segments"]["A"]["front_end"][f][lv]
    r = accept(run3, make_s33(), "A")
    assert r["n"] == 7
    assert r["chi2"] == 0.0
    assert r["accepted"] is False


def test_eight_cells_present_are_accepted() -> None:
    run3 = make_run3()
    for f, lv in cell_ids()[:2]:
        del run3["segments"]["A"]["front_end"][f][lv]
    r = accept(run3, make_s33(), "A")
    assert r["n"] == 8
    assert r["accepted"] is True


@pytest.mark.parametrize("bad", [None, math.nan, math.inf])
def test_a_missing_or_non_finite_cell_is_not_present(bad: float | None) -> None:
    run3 = make_run3()
    run3["segments"]["A"]["front_end"]["noise"]["50.0"]["ratio_se"] = bad
    s33 = make_s33()
    s33["cells"]["A"]["swept_cw"]["70.0"]["ratio_B"] = bad
    r = accept(run3, s33, "A")
    assert r["n"] == 8
    assert "noise/50.0" not in r["z"] and "swept_cw/70.0" not in r["z"]


def test_cells_above_70_do_not_enter_the_chi2() -> None:
    run3 = make_run3()
    s33 = make_s33()
    s33["cells"]["A"]["noise"]["75.0"]["ratio_B"] = 5.0
    assert accept(run3, s33, "A")["n"] == 10


def test_a_segment_is_judged_on_its_own_cells() -> None:
    run3 = make_run3()
    s33 = make_s33()
    for f, lv in cell_ids():
        s33["cells"]["B"][f][lv]["ratio_B"] = 2.0
    assert accept(run3, s33, "A")["accepted"] is True
    assert accept(run3, s33, "B")["accepted"] is False


# ---------------------------------------------------------------------------------------------- conditional range
def test_conditional_range_is_the_top_of_the_grid_when_everything_holds() -> None:
    # tau-ratios stay at 1 and truth is kept everywhere: J_max_TA is 85, capped at run 3's 70
    assert conditional_range(make_s33(), make_run3(), "A", "noise") == 70.0


def test_conditional_range_is_capped_at_70_even_when_jmax_is_higher() -> None:
    s33 = make_s33()
    assert s34.jmax_ta(s33, "A", "noise") == 85.0
    assert conditional_range(s33, make_run3(), "A", "noise") == 70.0


def test_conditional_range_stops_at_the_s33_jmax() -> None:
    s33 = make_s33()
    s33["cells"]["A"]["noise"]["65.0"]["ratio_TA"] = dict.fromkeys(TAUS, 0.85)  # a fail at 65 caps J_max at 60
    assert s34.jmax_ta(s33, "A", "noise") == 60.0
    assert conditional_range(s33, make_run3(), "A", "noise") == 60.0


def test_conditional_range_takes_the_tau_with_the_largest_departure() -> None:
    s33 = make_s33()
    # at 60 dB-Hz only the 100 ms tau departs (1.2 against 1.0, 1.0): the cell must be judged on it and fail
    s33["cells"]["A"]["noise"]["60.0"]["ratio_TA"] = {"0.001": 1.0, "0.01": 1.0, "0.1": 1.2}
    assert s34.jmax_ta(s33, "A", "noise") == 55.0
    # a departure below 1 counts the same way
    s33["cells"]["A"]["noise"]["60.0"]["ratio_TA"] = {"0.001": 0.88, "0.01": 1.04, "0.1": 1.05}
    assert s34.jmax_ta(s33, "A", "noise") == 55.0


def test_conditional_range_judges_the_worst_tau_slope_too() -> None:
    s33 = make_s33()  # ratios fine; the worst tau's slope is 1.2 (slope_tol 0.10): a fail, as front_end_jmax judges it
    s33["cells"]["A"]["noise"]["60.0"]["ratio_TA"] = {"0.001": 1.0, "0.01": 1.0, "0.1": 1.09}
    s33["cells"]["A"]["noise"]["60.0"]["slope_TA"] = {"0.001": 1.0, "0.01": 1.0, "0.1": 1.2}
    assert s34.jmax_ta(s33, "A", "noise") == 55.0


def test_an_unresolved_tau_cell_does_not_pass() -> None:
    s33 = make_s33()
    s33["cells"]["A"]["noise"]["50.0"]["ratio_TA_se"] = dict.fromkeys(TAUS, 0.06)  # SE > 0.05: unresolved at 50
    assert s34.jmax_ta(s33, "A", "noise") == 85.0  # unresolved is neither pass nor fail
    for lv in S33_LEVELS:
        s33["cells"]["A"]["noise"][lk(lv)]["ratio_TA_se"] = dict.fromkeys(TAUS, 0.06)
    assert s34.jmax_ta(s33, "A", "noise") is None
    assert conditional_range(s33, make_run3(), "A", "noise") is None


def test_conditional_range_needs_truth_kept_at_every_grid_level_up_to_it() -> None:
    run3 = make_run3()
    run3["segments"]["A"]["truth_kept"]["noise"]["65.0"] = False
    assert conditional_range(make_s33(), run3, "A", "noise") == 60.0
    run3["segments"]["A"]["truth_kept"]["noise"]["55.0"] = False  # an earlier loss gates everything above it
    assert conditional_range(make_s33(), run3, "A", "noise") == 50.0
    run3["segments"]["A"]["truth_kept"]["noise"]["50.0"] = False
    assert conditional_range(make_s33(), run3, "A", "noise") is None


def test_a_truth_kept_hole_after_a_loss_does_not_reopen_the_range() -> None:
    run3 = make_run3()
    run3["segments"]["A"]["truth_kept"]["noise"]["60.0"] = False  # kept again at 65 and 70, which must not count
    assert conditional_range(make_s33(), run3, "A", "noise") == 55.0


def test_truth_none_or_missing_is_not_kept() -> None:
    run3 = make_run3()
    run3["segments"]["A"]["truth_kept"]["noise"]["60.0"] = None
    assert conditional_range(make_s33(), run3, "A", "noise") == 55.0
    del run3["segments"]["A"]["truth_kept"]["swept_cw"]
    assert conditional_range(make_s33(), run3, "A", "swept_cw") is None


def test_a_missing_mid_grid_key_stops_the_range() -> None:
    run3 = make_run3()
    del run3["segments"]["A"]["truth_kept"]["noise"]["55.0"]  # the level is on run 3's grid: absent is not kept
    assert conditional_range(make_s33(), run3, "A", "noise") == 50.0
    del run3["segments"]["A"]["truth_kept"]["noise"]["50.0"]
    assert conditional_range(make_s33(), run3, "A", "noise") is None
    run3 = make_run3()
    del run3["segments"]["A"]["truth_kept"]["noise"]["70.0"]  # the top level missing: the range stops at 65
    assert conditional_range(make_s33(), run3, "A", "noise") == 65.0


def test_truth_kept_gates_only_up_to_s33s_jmax() -> None:
    run3 = make_run3()
    run3["segments"]["A"]["truth_kept"]["noise"]["70.0"] = False
    s33 = make_s33()
    s33["cells"]["A"]["noise"]["60.0"]["ratio_TA"] = dict.fromkeys(TAUS, 0.8)  # J_max_TA is 55; the loss at 70 is moot
    assert conditional_range(s33, run3, "A", "noise") == 55.0


def test_a_rejected_segment_has_no_conditional_range() -> None:
    run3 = make_run3()
    s33 = make_s33()
    for f, lv in cell_ids():
        s33["cells"]["A"][f][lv]["ratio_B"] = 2.0
    assert accept(run3, s33, "A")["accepted"] is False
    assert conditional_range(s33, run3, "A", "noise") is None
    assert conditional_range(s33, run3, "B", "noise") == 70.0


def test_one_rejected_segment_gives_a_combined_none() -> None:
    run3 = make_run3()
    s33 = make_s33()
    for f, lv in cell_ids():
        s33["cells"]["A"][f][lv]["ratio_B"] = 2.0
    out = combined_range(s33, run3)
    assert out == {"noise": None, "swept_cw": None}


def test_combined_range_is_the_lower_of_the_segments() -> None:
    run3 = make_run3()
    run3["segments"]["B"]["truth_kept"]["noise"]["65.0"] = False
    out = combined_range(make_s33(), run3)
    assert out == {"noise": 60.0, "swept_cw": 70.0}


# ---------------------------------------------------------------------------------------------- order and pins
def test_the_order_check_accepts_an_earlier_commit() -> None:
    check_order("2026-10-26T12:00:00+02:00", "2026-10-26T11:00:00+00:00")  # 10:00 UTC < 11:00 UTC


def test_the_order_check_refuses_a_later_commit() -> None:
    with pytest.raises(SourceRefused, match="later"):
        check_order("2026-10-26T14:00:00+02:00", "2026-10-26T11:00:00+00:00")  # 12:00 UTC > 11:00 UTC


def test_the_order_check_refuses_a_commit_at_the_same_instant() -> None:
    with pytest.raises(SourceRefused):
        check_order("2026-10-26T13:00:00+02:00", "2026-10-26T11:00:00+00:00")


def test_the_order_check_refuses_a_time_without_an_offset() -> None:
    with pytest.raises(ValueError, match="offset"):
        check_order("2026-10-26T10:00:00", "2026-10-26T11:00:00+00:00")


def test_another_sha256_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"segments": {}}))
    import hashlib

    good = hashlib.sha256(path.read_bytes()).hexdigest()
    assert pinned_json(path, good, "run 3 summary") == {"segments": {}}
    with pytest.raises(SourceRefused, match="not the pinned"):
        pinned_json(path, "0" * 64, "run 3 summary")


def test_main_refuses_while_a_pin_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(s34, "RUN3_SUMMARY_SHA256", None)
    with pytest.raises(SourceRefused, match="RUN3_SUMMARY_SHA256"):
        s34.main()
    monkeypatch.setattr(s34, "RUN3_SUMMARY_SHA256", "0" * 64)
    monkeypatch.setattr(s34, "S33_SHA256", None)
    with pytest.raises(SourceRefused, match="S33_SHA256"):
        s34.main()


def test_run3_id_is_the_run_id_of_e2_v3() -> None:
    """The run s34 judges is the run of experiments/E2-v3.yaml, the spec run 3 ran (a link that drifted once, when
    workers_b was added; the header's comments do not enter the run id)."""
    spec = yaml.safe_load((Path(__file__).resolve().parents[2] / "experiments" / "E2-v3.yaml").read_text())
    assert run_id(spec) == s34.RUN3_ID == "9650dc00e766"


def test_judge_reports_both_segments_and_the_combined_range() -> None:
    out = s34.judge(make_s33(), make_run3())
    assert out["combined_range"] == {"noise": 70.0, "swept_cw": 70.0}
    assert out["segments"]["A"]["accepted"] is True
    assert out["segments"]["A"]["chi2_bound"] == pytest.approx(chi2.ppf(0.95, 10))
    assert out["segments"]["B"]["j_max_ta_s33"] == {"noise": 85.0, "swept_cw": 85.0}
    json.dumps(out)  # the study's payload must be serialisable as it stands
