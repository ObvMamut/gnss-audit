"""E2 helpers (docs/10 section 2.3; P2b task 11): the unit plan, the decision rules stated in the spec (J_max, K,
the residual), the paired-difference stream B1's prediction despreads, and finished-unit validation for --resume. Under
(a') (P2b part 2, task 5): the spec's pre-registered rules, seeds that do not follow mapping order, the K-study at the
Route B ceiling with its noise fallback, and the decisions on a fake two-segment run, whose front-end cells must be
E2post's."""

import dataclasses
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from gnssaudit.cli.main import receiver_patch_refusal
from gnssaudit.experiments import e2, e2post
from gnssaudit.experiments.e0b import SLIP_CYC
from gnssaudit.experiments.e2_frontend import FrontEndCell
from gnssaudit.experiments.twin import SHM_MARGIN, Twin, jsonable
from gnssaudit.gates import GateReport
from gnssaudit.inject import GhostSpec, InjectionReport, IQSource, disturbance_seed, ghost_phase_rad
from gnssaudit.io.fairdata import StorageGuardError
from gnssaudit.io.schemas import EVENTS, TRACK_STREAM, TRUTH_EPOCH_SAT, build
from gnssaudit.sdr import RunArtifacts, SourceSpec
from gnssaudit.sdr.images import PATCH_ID
from gnssaudit.truth.ghost import NoAmbiguityWindow, PhasePrecisionError, ghost_truth
from gnssaudit.truth.oracle import k_convergence
from gnssaudit.truth.rxtime import TwinClock
from gnssaudit.uq.loop import CarrierLoop

SPEC = {
    "seed": 23,
    "families": {"noise": {"bw_hz": 2.0e6}, "swept_cw": {"span_hz": 1.0e6, "period_s": 0.01},
                 "chirp": {"span_hz": 1.0e7, "period_s": 1.0e-5}},
    "levels_db_hz": [60.0, 70.0, 80.0, 90.0],
    "mode_b": {"tau_s": 0.01, "side_tau_s": [0.001, 0.1], "side_level_db_hz": 90.0,
               "side_families": ["noise", "swept_cw"]},
    "kstudy": {"segment": "A", "level_cap_db_hz": 80.0, "offset_s": 0.0037,
               "blocks": {"cond": {"family": "swept_cw", "K": 101, "conditional": True},
                          "marg": {"family": "swept_cw", "K": 51, "conditional": False},
                          "noise": {"family": "noise", "K": 51, "conditional": False}}},
    "thermal": {"segment": "A", "family": "swept_cw", "K": 51},
    "repeat_fraction": 0.05,
}  # fmt: skip


def test_mode_a_injects_on_workers_a_and_mode_b_on_workers_b_else_one_core() -> None:
    """A spec without `workers_b` (E2-v2) keeps mode B on one core, as its runs were made."""
    assert e2.inject_workers({"workers_a": 6}, "A") == 6
    assert e2.inject_workers({"workers_a": 6}, "B") == 1
    assert e2.inject_workers({"workers_a": 6, "workers_b": 4}, "B") == 4
    assert e2.inject_workers({}, "A") == 1


def test_the_grid_pairs_each_mode_b_twin_with_its_mode_a_twin() -> None:
    """Per segment: 3 families x 4 levels x {A, B at 10 ms} jammed twins; A and B of a cell share their seed (paired
    seeds: Var_B / Var_A compares the front end alone); a null pair of mode A clean twins and one clean twin per
    mode B time constant."""
    units = e2.plan_grid(SPEC, "A")
    jam = [u for u in units if u.block == "grid"]
    assert len(jam) == 24 and {u.clean for u in jam} == {"A-clean-A", "A-clean-B10"}
    a = {(u.family, u.level): u for u in jam if u.mode == "A"}
    b = {(u.family, u.level): u for u in jam if u.mode == "B"}
    assert a.keys() == b.keys() and all(a[k].seed == b[k].seed for k in a)
    assert all(u.project for u in a.values()) and not any(u.project for u in b.values())
    cleans = {u.name: u for u in units if u.block == "clean"}
    assert set(cleans) == {"A-clean-A", "A-clean-A2", "A-clean-B10", "A-clean-B1", "A-clean-B100"}
    assert cleans["A-clean-A"].keep_bytes and not cleans["A-clean-A2"].keep_bytes
    side = [u for u in units if u.block == "tau"]
    assert len(side) == 4 and {u.clean for u in side} == {"A-clean-B1", "A-clean-B100"}
    assert {u.level for u in side} == {90.0} and {u.family for u in side} == {"noise", "swept_cw"}
    assert len({u.name for u in units}) == len(units)


def test_the_k_study_and_the_thermal_block_follow_the_oracle_forms() -> None:
    """Conditional realizations fix the sweep timing and draw the rest; marginal ones draw the timing too (F8). Each
    thermal pair shares its own common floor: the clean twin of realization r carries the floor of the jammed one."""
    units = e2.plan_oracle(SPEC, "A", {"swept_cw": 80.0, "noise": 70.0, "chirp": None})
    cond = [u for u in units if u.block == "k-cond"]
    marg = [u for u in units if u.block == "k-marg"]
    noise = [u for u in units if u.block == "k-noise"]
    assert (len(cond), len(marg), len(noise)) == (101, 51, 51)
    assert {u.offset_s for u in cond} == {0.0037} and {u.offset_s for u in marg} == {None}
    assert (
        len({u.seed for u in cond}) == 101 and {u.level for u in cond} == {80.0} and {u.level for u in noise} == {70.0}
    )
    jam = [u for u in units if u.block == "thermal"]
    clean = {u.name: u for u in units if u.block == "thermal-clean"}
    assert len(jam) == 51 and len(clean) == 51
    assert all(clean[u.clean or ""].floor_seed == u.floor_seed is not None for u in jam)
    assert len({u.floor_seed for u in jam}) == 51 and {u.offset_s for u in jam} == {0.0037}


def test_k_study_level_is_capped_at_80_db_hz_and_at_j_max() -> None:
    assert e2.k_level(80.0, 90.0) == 80.0 and e2.k_level(80.0, 70.0) == 70.0
    assert e2.k_level(80.0, None) is None  # no level passes: the block is not run, and that is reported


def test_j_max_is_the_largest_level_up_to_which_every_level_passes() -> None:
    """docs/8 section 5: |Var_B / Var_A - 1| < 0.10; a level that passes above a failing one does not count."""
    assert e2.j_max({60.0: 1.02, 70.0: 1.05, 80.0: 1.08, 90.0: 1.4}, tol=0.10) == 80.0
    assert e2.j_max({60.0: 1.02, 70.0: 1.3, 80.0: 1.05, 90.0: 1.0}, tol=0.10) == 60.0
    assert e2.j_max({60.0: 1.2, 70.0: 1.0}, tol=0.10) is None
    assert e2.j_max({60.0: 0.95, 70.0: float("nan")}, tol=0.10) == 60.0  # an unmeasured level ends the run


def test_k_is_51_when_the_median_relative_change_is_at_most_0_2() -> None:
    """docs/5 section 4.4 as pre-registered: K = 51 if the median |sigma^2_101 / sigma^2_51 - 1| <= 0.2, else 101.
    k_convergence's own bar sqrt(2 / (K_small - 1)) is that 0.2 exactly at K_small = 51."""
    assert math.sqrt(2.0 / (51 - 1)) == 0.2 == e2.K_CHANGE_MAX
    assert e2.choose_k({"median_rel_change": 0.2}) == 51 and e2.choose_k({"median_rel_change": 0.0}) == 51
    assert e2.choose_k({"median_rel_change": 0.2000001}) == 101
    assert e2.choose_k({"median_rel_change": float("nan")}) == 101  # no comparison, no K = 51


def test_the_residual_ratio_of_a_perfect_prediction_is_zero() -> None:
    rng = np.random.default_rng(3)
    d = rng.standard_normal(5000)
    assert e2.residual_ratio(d, d) == pytest.approx(0.0, abs=1e-15)
    assert e2.residual_ratio(d, np.zeros_like(d)) == pytest.approx(1.0)
    assert e2.residual_ratio(d, -d) == pytest.approx(4.0)  # a sign error shows as r = 4


def test_the_paired_difference_stream_is_the_interference_in_input_counts(tmp_path: Path) -> None:
    """Jammed minus clean int16 twin (interleaved I, Q), over the twins' scale: the decimated interference, exact up to
    the rounding of each twin; read in blocks with their first output index."""
    rng = np.random.default_rng(4)
    n, scale = 10_001, 50.0
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    i = 0.3 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))

    def write(path: Path, z: np.ndarray) -> Path:
        iq = np.empty(2 * n, dtype="<i2")
        iq[0::2], iq[1::2] = np.round(scale * np.real(z)), np.round(scale * np.imag(z))
        iq.tofile(path)
        return path

    chunks = list(e2.pair_difference(write(tmp_path / "j.bin", x + i), write(tmp_path / "c.bin", x), scale, block=4096))
    assert [m0 for m0, _ in chunks] == [0, 4096, 8192]
    got = np.concatenate([y for _, y in chunks])
    err = got - i
    assert max(np.max(np.abs(np.real(err))), np.max(np.abs(np.imag(err)))) <= 1.0 / scale + 1e-12  # a count each


def test_a_finished_unit_is_one_whose_files_still_hash_as_recorded(tmp_path: Path) -> None:
    """--resume redoes a unit unless its done.json names the same code version and every file hashes as recorded."""
    u = tmp_path / "A-grid-noise-60-A"
    u.mkdir()
    (u / "truth.parquet").write_bytes(b"abc")
    e2.mark_done(u, "v1", ["truth.parquet"])
    assert e2.is_done(u, "v1") and not e2.is_done(u, "v2")
    (u / "truth.parquet").write_bytes(b"abd")
    assert not e2.is_done(u, "v1")
    assert not e2.is_done(tmp_path / "missing", "v1")
    assert json.loads((u / "done.json").read_text())["describe"] == "v1"


def test_repeats_are_a_seeded_five_percent_of_the_units() -> None:
    names = [f"u{i}" for i in range(390)]
    pick = e2.pick_repeats(names, fraction=0.05, seed=23)
    assert len(pick) == 20 and set(pick) <= set(names) and pick == e2.pick_repeats(names, fraction=0.05, seed=23)


def test_b1_has_a_thermal_floor_of_one_over_cn0_t() -> None:
    """An atan discriminator normalises by the noisy prompt: r cannot fall below 1 / (C/N0 T), 3.2 % at 45 dB-Hz
    and 1 ms."""
    assert e2.thermal_floor(45.0, 1e-3) == pytest.approx(1 / 31.6227766, rel=1e-6)
    assert e2.thermal_floor(55.0, 1e-3) == pytest.approx(0.1 * e2.thermal_floor(45.0, 1e-3))


# ------------------------------------------------------------------------------------ (a') (P2b part 2, task 5)
REPO = Path(__file__).resolve().parents[2]
V2 = yaml.safe_load((REPO / "experiments" / "E2-v2.yaml").read_text())


def test_the_v2_spec_holds_the_pre_registered_rules() -> None:
    """experiments/E2-v2.yaml as pre-registered on 16 Oct 2026 (commit 15ef977; docs/8 section 5, docs/3 section 2.2,
    docs/5 section 4.4): thresholds, bootstrap, grid, tau side check and K-study, value for value."""
    assert V2["rules"] == {"jmax_tol": 0.10, "slope_tol": 0.10, "excess_se_max": 0.05, "residual_max": 0.05,
                           "lost_truth_max_s": 1.0, "k_small": 51,
                           "bootstrap": {"block_s": 1.0, "n_boot": 400, "seed": 7}}  # fmt: skip
    assert V2["levels_db_hz"] == [50.0, 55.0, 60.0, 65.0, 70.0]
    assert V2["mode_b"] == {"tau_s": 0.01, "side_tau_s": [0.001, 0.1], "side_level_db_hz": 65.0,
                            "side_families": ["noise", "swept_cw"]}  # fmt: skip
    assert V2["kstudy"] == {"segment": "A", "level_cap_db_hz": 80.0, "fallback_family": "noise", "offset_s": 0.0037,
                            "blocks": {"cond": {"family": "swept_cw", "K": 101, "conditional": True},
                                       "marg": {"family": "swept_cw", "K": 51, "conditional": False},
                                       "noise": {"family": "noise", "K": 51, "conditional": False}}}  # fmt: skip
    assert V2["thermal"] == {"segment": "A", "family": "swept_cw", "K": 51}
    assert list(V2["segments"]) == ["A", "B"] and list(V2["families"]) == ["noise", "swept_cw", "chirp"]
    assert V2["ghosts"] == {"cn0_db_hz": 45.0, "doppler_hz": [1500.0, -2500.0]} and V2["confirmatory"] is True
    assert V2["receiver_overrides"] == {"Tracking_1C.max_carrier_lock_fail": "1000000",
                                        "Tracking_1C.enable_fll_pull_in": "true"}  # D-34  # fmt: skip


def test_the_v2_grid_runs_five_levels_and_the_tau_side_check_at_65_on_segment_a() -> None:
    """Per segment 3 families x 5 levels x {A, B10}, B1 projected in mode A only; the tau side check (segment A's,
    docs/8 section 5) at 65 dB-Hz, noise and swept CW at tau = 1 and 100 ms, each on the jammer draw of the mode A
    grid unit it is paired with."""
    a, b = e2.plan_grid(V2, "A"), e2.plan_grid(V2, "B")
    grid = [u for u in a if u.block == "grid"]
    assert sorted({float(u.level or 0.0) for u in grid}) == [50.0, 55.0, 60.0, 65.0, 70.0] and len(grid) == 30
    assert all(u.project == (u.mode == "A") for u in grid)  # no B1 in mode B under (a')
    a65 = {u.family: u for u in grid if u.level == 65.0 and u.mode == "A"}
    tau = [u for u in a if u.block == "tau"]
    assert sorted((u.family, u.tau_s) for u in tau) == [("noise", 0.001), ("noise", 0.1), ("swept_cw", 0.001),
                                                        ("swept_cw", 0.1)]  # fmt: skip
    assert all(u.level == 65.0 and u.mode == "B" and not u.project and u.seed == a65[u.family].seed for u in tau)
    assert {u.clean for u in tau} == {"A-clean-B1", "A-clean-B100"}
    assert {u.name for u in b if u.block == "clean"} == {"B-clean-A", "B-clean-A2", "B-clean-B10"}
    assert not [u for u in b if u.block == "tau"] and len([u for u in b if u.block == "grid"]) == 30


def reordered(spec: dict[str, Any]) -> dict[str, Any]:
    ks = {**spec["kstudy"], "blocks": dict(reversed(list(spec["kstudy"]["blocks"].items())))}
    return {**spec, "families": dict(reversed(list(spec["families"].items()))), "kstudy": ks}


def drawn(spec: dict[str, Any], ceiling: dict[str, float | None]) -> dict[str, tuple[Any, ...]]:
    units = [*e2.plan_grid(spec, "A"), *e2.plan_oracle(spec, "A", ceiling)]
    return {u.name: (u.seed, u.floor_seed, u.family, u.level) for u in units}


def test_seeds_do_not_depend_on_the_order_of_the_families_or_the_k_study_blocks() -> None:
    """The manifest keeps the spec as sorted JSON and two YAMLs that only reorder share a run id: either way every
    planned unit must draw the same jammer (on E2/4b3e282ef77d a plan rebuilt from the manifest gave every grid cell
    another cell's seed)."""
    for ceiling in ({"noise": 60.0, "swept_cw": 55.0, "chirp": None}, {"noise": 60.0, "swept_cw": None, "chirp": None}):
        want = drawn(V2, ceiling)
        assert drawn(reordered(V2), ceiling) == want
        assert drawn(json.loads(json.dumps(V2, sort_keys=True)), ceiling) == want
        own = [u for u in [*e2.plan_grid(V2, "A"), *e2.plan_oracle(V2, "A", ceiling)]
               if u.mode == "A" and u.block not in ("clean", "thermal-clean")]  # fmt: skip
        assert len({u.seed for u in own}) == len(own)  # every cell and realization its own draw


def test_without_a_swept_cw_ceiling_the_k_101_and_thermal_blocks_run_on_noise() -> None:
    """docs/5 section 4.4: L_B(swept CW) None moves the K = 101 block and the thermal block to noise at L_K(noise),
    without a fixed sweep timing (noise has none); swept CW's marginal block is not run; every K-study block is
    differenced against the segment's existing mode A clean twin. With no noise ceiling either, nothing runs."""
    units = e2.plan_oracle(V2, "A", {"noise": 60.0, "swept_cw": None, "chirp": None})
    by: dict[str, list[e2.Unit]] = {}
    for u in units:
        by.setdefault(u.block, []).append(u)
    assert set(by) == {"k-cond", "k-noise", "thermal", "thermal-clean"}
    assert len(by["k-cond"]) == 101
    assert {(u.family, u.level, u.offset_s, u.clean) for u in by["k-cond"]} == {("noise", 60.0, None, "A-clean-A")}
    assert len(by["k-noise"]) == 51 and {(u.family, u.level, u.clean) for u in by["k-noise"]} == {
        ("noise", 60.0, "A-clean-A")
    }
    assert len(by["thermal"]) == 51 and {(u.family, u.level, u.offset_s) for u in by["thermal"]} == {
        ("noise", 60.0, None)
    }
    assert not {u.seed for u in by["k-cond"]} & {u.seed for u in by["k-noise"]}  # the moved block draws its own
    assert e2.plan_oracle(V2, "A", {"noise": None, "swept_cw": None, "chirp": None}) == []  # K is not chosen
    assert e2.plan_oracle(V2, "B", {"noise": 60.0, "swept_cw": 60.0, "chirp": None}) == []  # segment A's study


def test_the_k_study_runs_at_l_k_the_lesser_of_80_and_the_ceiling() -> None:
    units = e2.plan_oracle(V2, "A", {"noise": 65.0, "swept_cw": 55.0, "chirp": None})
    assert {(u.block, u.family, u.level, u.offset_s) for u in units if u.block != "thermal-clean"} == {
        ("k-cond", "swept_cw", 55.0, 0.0037), ("k-marg", "swept_cw", 55.0, None), ("k-noise", "noise", 65.0, None),
        ("thermal", "swept_cw", 55.0, 0.0037)}  # fmt: skip


def test_b1s_prediction_is_the_loop_response_less_its_pre_onset_mean() -> None:
    """The one prediction e2.reduce_unit and e2post share: H n_iota in cycles (a record without n_iota counts as 0) at
    the truth's record times, less its mean over [onset - pre, onset)."""
    t_rec = np.asarray(10.0 + np.arange(20_000) * 1e-3, dtype=np.float64)
    n = np.random.default_rng(9).normal(0.0, 0.05, t_rec.size)
    n[50] = np.nan
    want = CarrierLoop.gnsssdr(35.0, 1e-3).filter(np.nan_to_num(n)) / (2 * np.pi)
    ref = (t_rec >= 15.0) & (t_rec < 20.0)
    np.testing.assert_array_equal(e2.b1_prediction(n, t_rec, t_rec, 20.0, 5.0), want - np.mean(want[ref]))
    assert not hasattr(e2post, "b1_prediction")  # one copy only


def test_b1s_record_keeps_var_n_iota_on_rs_records_beside_r() -> None:
    """Var(n_iota) (rad^2) on the records r is read on, joined on the record time (a record without n_iota left
    out), and the formula floor + Var(n_iota), reported beside r and the floor (docs/3 section 2.2)."""
    rng = np.random.default_rng(8)
    t_rec = np.asarray(30.0 + np.arange(6000) * 1e-3, dtype=np.float64)
    n_iota = rng.normal(0.0, 0.05, t_rec.size)
    n_iota[[100, 4000]] = np.nan
    t = np.asarray(t_rec[2000:5000] + 1e-12, dtype=np.float64)  # r's epochs, float noise under 1 ns
    pred = rng.normal(size=t.size)
    e = pred + rng.normal(0.0, 0.3, t.size)
    rec = e2.b1_residual(e, pred, n_iota, t_rec, t, cn0_clean_dbhz=45.0)
    assert rec["var_n_iota_rad2"] == pytest.approx(np.var(np.delete(n_iota[2000:5000], 2000)), rel=1e-12)
    assert rec["n_iota_records"] == 2999 and rec["n"] == 3000 and rec["cn0_clean_dbhz"] == 45.0
    assert rec["floor"] == e2.thermal_floor(45.0, 1e-3) and rec["r"] == e2.residual_ratio(e, pred)
    assert rec["floor_plus_var"] == rec["floor"] + rec["var_n_iota_rad2"]
    assert rec["corr"] == pytest.approx(np.corrcoef(e, pred)[0, 1])


def cell(ratio: float, se: float, slopes: tuple[float, ...] = (1.0, 1.0)) -> FrontEndCell:
    return FrontEndCell(slopes, ratio, se)


def test_j_max_stops_below_a_level_without_a_verdict_and_keeps_under_the_tau_cap() -> None:
    """A level whose cell could not be formed (a ghost lost, too few epochs, a non-finite number) has no verdict, so
    "no fail at or below" cannot be shown above it. The tau cap bounds the levels J_max is chosen from, so J_max stays
    a level that passes."""
    p, f, u = cell(1.0, 0.01), cell(1.3, 0.01), cell(1.0, 0.2)
    rules = V2["rules"]
    assert e2.segment_jmax({50.0: p, 55.0: p, 60.0: p}, rules=rules) == 60.0
    assert e2.segment_jmax({50.0: p, 55.0: None, 60.0: p}, rules=rules) == 50.0
    assert e2.segment_jmax({50.0: p, 55.0: f, 60.0: p}, rules=rules) == 50.0
    assert e2.segment_jmax({50.0: f, 55.0: p, 60.0: p}, rules=rules) is None
    assert e2.segment_jmax({50.0: p, 55.0: p, 60.0: p}, rules=rules, cap=55.0) == 55.0
    assert e2.segment_jmax({50.0: p, 55.0: u, 60.0: p}, rules=rules, cap=55.0) == 50.0
    assert e2.segment_jmax({50.0: p, 55.0: p}, rules=rules, cap=-math.inf) is None
    assert e2.side_cap([50.0, 55.0, 60.0, 65.0, 70.0], 65.0) == 60.0 and e2.side_cap([65.0, 70.0], 65.0) == -math.inf


# A fake finished E2 run of two segments with tiny parquet files. Per segment, family and level, mode B's jammer
# slope, r - floor of both ghosts, and the first ghost's lost truth (s) in the mode A unit:
FS, DECIM = 40_000.0, 4  # 10 kHz out: a 1 ms record is 10 output samples
CLOCK = TwinClock(fs_native_hz=FS, sample_start=0, decim=DECIM, delay=0.0)
DESCRIBE = "fake01234567"
N_REC = 25_000  # 25 s of 1 ms records; usable from onset + settle = 15 s
SHIFTED = np.arange(16_000, 22_000, 200)  # records the mode B clean twin stamps one sample later (first ghost)
VAR_C, VAR_U, VAR_J = 0.001, 0.002, 0.008
LEVELS = (50.0, 55.0, 60.0)
SURVEY = {"sats": [4, 9], "ghosts": [{"prn": 32, "doppler_hz": 1510.0}, {"prn": 31, "doppler_hz": -2440.0}],
          "in_view": {"4": -2002.5, "9": -112.1}, "cn0": {"4": 44.5, "9": 46.3}, "n0": 1.77e-7,
          "line_excess_db": 2.08, "agc_rms": 2.16, "analog_sd": 1.50, "scale": 76.6}  # fmt: skip
FAKE = {**V2, "warmup_s": 2.0, "onset_s": 13.0, "pre_window_s": 5.0, "decim": DECIM, "fir_taps": 1,
        "segments": {k: {"interval": f"fake-{k}", "item_type": "ibyte", "fs_hz": FS} for k in "AB"},
        "levels_db_hz": list(LEVELS), "mode_b": {**V2["mode_b"], "side_level_db_hz": 60.0},
        "rules": {**V2["rules"], "bootstrap": {"block_s": 1.0, "n_boot": 60, "seed": 7}}}  # fmt: skip
CELLS = {  # r - floor at 0.04 passes B1's bar while r itself (floor 0.032 at 45 dB-Hz) is above 0.05
    ("A", "noise"): {"slope_b": (1.0, 1.0, 1.0), "rmf": (0.04, 0.04, 0.04), "lost": (0.0, 0.0, 1.5)},
    ("A", "swept_cw"): {"slope_b": (1.0, 1.0, 1.0), "rmf": (0.04, 0.04, 0.2), "lost": (1.0, 0.0, 0.0)},
    ("B", "noise"): {"slope_b": (0.8, 0.8, 0.8), "rmf": (0.04, 0.04, 0.04), "lost": (0.0, 0.0, 0.0)},
    ("B", "swept_cw"): {"slope_b": (1.0, 1.0, 1.0), "rmf": (0.04, 0.2, 0.04), "lost": (0.0, 0.0, 0.0)},
}  # the chirp has no unit at all
TAU_SLOPE = {("noise", 0.001): 1.0, ("noise", 0.1): 1.0, ("swept_cw", 0.001): 1.0, ("swept_cw", 0.1): 0.8}
POST = {"id": "E2post", "source_run": "E2/fake", "segment": "A", "families": ["noise", "swept_cw"],
        "levels_db_hz": list(LEVELS), "ghost_channels": [2, 3], "bootstrap": FAKE["rules"]["bootstrap"],
        "rules": {"tol": 0.10, "slope_tol": 0.10, "se_max": 0.05},
        "gate_a": {"families": ["noise", "swept_cw"], "a1": {"levels_db_hz": [50.0, 60.0], "slope_min": 0.9,
                                                             "slope_max": 1.1},
                   "a2": {"level_db_hz": 50.0, "cancel_share": 0.5}}}  # fmt: skip
COMMON = {"unlocked": [], "lock_ok": True, "identity": {}, "stock_lock_verdict": {}, "clip_count": 0}
C1_RECORD = {"2": {"rms_cyc": 2e-4, "n": 9000}, "3": {"computed": False, "reason": "no window"}}


def ghost_spec(i: int) -> GhostSpec:
    g = SURVEY["ghosts"][i]
    return GhostSpec(g["prn"], g["doppler_hz"], 0.0, 45.0, 17.3 + 101.0 * i, disturbance_seed(23, 0, 0, i))


def track(stamps: np.ndarray, acc_rad: np.ndarray, prn: int) -> pa.Table:
    n = stamps.size
    return build(TRACK_STREAM, {
        "run_id": ["r"] * n, "channel": np.zeros(n), "sys": ["G"] * n, "prn": np.full(n, prn), "signal": ["1C"] * n,
        "k": np.arange(n), "sample_count": stamps, "t_rx_s": CLOCK.file_time(stamps), "abs_ve": None, "abs_e": None,
        "abs_p": None, "abs_l": None, "abs_vl": None, "prompt_i": None, "prompt_q": None, "acc_phase_rad_f32": None,
        "acc_phase_rad": acc_rad, "dop_hz": None, "dop_rate_hzps": None, "code_freq_chips": None,
        "code_rate_chipsps2": None, "carr_err": None, "carr_err_filt": None, "code_err_chips": None,
        "code_err_filt": None, "cn0_dbhz": np.full(n, 45.0), "lock_test": None, "tow_ms": None, "wn": None,
        "t_int_s": np.full(n, 1e-3),
    }, n)  # fmt: skip


def truth_at(t: np.ndarray, e_cyc: np.ndarray, prn: int, valid: np.ndarray | None = None) -> pa.Table:
    n = t.size
    return build(TRUTH_EPOCH_SAT, {
        "truth_id": ["g"] * n, "kind": ["twin"] * n, "sys": ["G"] * n, "prn": np.full(n, prn), "signal": ["1C"] * n,
        "t_rx_s": t, "e_cyc": e_cyc, "slip_count": np.zeros(n), "arc_id": np.zeros(n),
        "transient": np.zeros(n, bool), "valid": np.ones(n, bool) if valid is None else valid, "regime": np.zeros(n),
    }, n)  # fmt: skip


def twin_truth(stamps: np.ndarray, e_cyc: np.ndarray, prn: int, valid: np.ndarray | None = None) -> pa.Table:
    return truth_at(CLOCK.file_time(stamps), e_cyc, prn, valid)


def clean_track(g: GhostSpec, stamps: np.ndarray, e_cyc: np.ndarray) -> pa.Table:
    theta = ghost_phase_rad(g, np.rint(CLOCK.native_index(stamps)).astype(np.int64), FS)
    return track(stamps, -(theta + 2 * np.pi * e_cyc), g.prn)


def write_unit(run: Path, name: str, tables: dict[str, pa.Table], rec: dict[str, Any]) -> None:
    udir = run / "units" / name
    udir.mkdir(parents=True)
    for f, t in tables.items():
        pq.write_table(t, udir / f)
    (udir / "unit.json").write_text(json.dumps(jsonable(rec)))
    e2.mark_done(udir, DESCRIBE, [*tables, "unit.json"])


def fake_context(run: Path, monkeypatch: pytest.MonkeyPatch, spec: dict[str, Any] = FAKE) -> e2.Context:
    """The run's surveys (both segments share the ghosts and the clock), and E2's Context on them (FAKE or `spec`)."""
    run.mkdir(parents=True)
    (run.parent / "recording.iq").write_bytes(bytes(2000))
    src = IQSource(path=run.parent / "recording.iq", item_type="ibyte", fs_hz=FS, sample_start=0, sample_end=1000)
    monkeypatch.setattr(e2, "recording_source", lambda seg_spec: (src, CLOCK))
    for key in spec["segments"]:
        udir = run / "units" / f"survey-{key}"
        udir.mkdir(parents=True)
        (udir / "survey.json").write_text(json.dumps(SURVEY))
        e2.mark_done(udir, DESCRIBE, ["survey.json"])
    ctx = e2.Context(spec, run, run / "work", DESCRIBE, None, {}, {}, {}, {})
    for key in spec["segments"]:
        ctx.segments[key] = e2._segment_survey(ctx, key)
    return ctx


def jammed_record(u: e2.Unit, var: float, residual: dict[str, Any], lost: float) -> dict[str, Any]:
    return {**COMMON, "unit": dataclasses.asdict(u), "var_cyc2": {"ghost32": var, "ghost31": var}, "residual": residual,
            "lost_ghost_s": {"2": lost, "3": 0.0}}  # fmt: skip


def build_run(
    run: Path, monkeypatch: pytest.MonkeyPatch, drop: tuple[str, str] | None = None, check: bool = False
) -> e2.Context:
    """Per ghost: the clean twins' errors share c and add their own u; mode A's twin difference is the jammer effect
    j, mode B's slope_b j plus a little white noise; the clean twins' ghost errors are written as E2's clean units
    write them. Mode A units carry B1's residual record and the lost ghost truth (and with `check` a C-1 record).
    `drop` = (unit, file) is not written (a ghost lost in that twin)."""
    ctx = fake_context(run, monkeypatch)
    rng = np.random.default_rng(11)
    k = np.arange(1, N_REC + 1)
    floor = e2.thermal_floor(45.0, 1e-3)
    side = float(FAKE["mode_b"]["side_level_db_hz"])
    for key, seg in ctx.segments.items():
        units = {u.name: u for u in e2.plan_grid(FAKE, key)}
        tables: dict[str, dict[str, pa.Table]] = {}
        for i, ch in enumerate(seg.ghost_ch):
            g = ghost_spec(i)
            stamps_a = (10 * k + 2 * i).astype(np.uint64)  # the two ghost channels never share stamps
            stamps_b = stamps_a + (np.isin(k, SHIFTED) & (i == 0)).astype(np.uint64)
            c = rng.normal(0.0, np.sqrt(VAR_C), k.size)
            for u in units.values():
                if u.block == "clean" and not u.name.endswith("A2"):
                    st = stamps_b if u.tau_s == FAKE["mode_b"]["tau_s"] else stamps_a
                    e_c = 0.3 + c + rng.normal(0.0, np.sqrt(VAR_U), k.size)
                    tables.setdefault(u.name, {})[f"track_ch{ch}.parquet"] = clean_track(g, st, e_c)
            for fam in ("noise", "swept_cw"):
                for n, lv in enumerate(LEVELS):
                    j = rng.normal(0.0, np.sqrt(VAR_J), k.size)
                    small = 0.1 * np.sqrt(VAR_J) * rng.standard_normal(k.size)
                    tables.setdefault(f"{key}-grid-{fam}-{lv:g}-A", {})[f"ghost_ch{ch}.parquet"] = twin_truth(
                        stamps_a, j, g.prn)  # fmt: skip
                    tables.setdefault(f"{key}-grid-{fam}-{lv:g}-B10", {})[f"ghost_ch{ch}.parquet"] = twin_truth(
                        stamps_b, CELLS[key, fam]["slope_b"][n] * j + small, g.prn)  # fmt: skip
                    for u in units.values():
                        if u.block == "tau" and u.family == fam and u.level == lv == side:
                            e = TAU_SLOPE[fam, float(u.tau_s or 0.0)] * j + 0.1 * np.sqrt(VAR_J) * rng.standard_normal(
                                k.size
                            )
                            tables.setdefault(u.name, {})[f"ghost_ch{ch}.parquet"] = twin_truth(stamps_a, e, g.prn)
        for name, tabs in tables.items():
            u = units[name]
            tabs = {f: t for f, t in tabs.items() if (name, f) != drop}
            if u.block == "clean":
                errs = e2.ghost_errors(seg, {int(f[8:-8]): t for f, t in tabs.items()}, converge_s=ctx.warm / 2,
                                       name=name)  # fmt: skip
                tabs = {**tabs, **{f"ghost_abs_ch{ch}.parquet": t for ch, t in errs.items()}}
                write_unit(run, name, tabs, dict(COMMON))
            elif u.mode == "A":
                cfg = CELLS[key, u.family or ""]
                n = LEVELS.index(u.level or 0.0)
                resid = {str(p): {"r": floor + cfg["rmf"][n], "floor": floor, "cn0_clean_dbhz": 45.0, "corr": 0.9,
                                  "n": 10_000, "var_n_iota_rad2": 0.01, "floor_plus_var": floor + 0.01}
                         for p in (32, 31)}  # fmt: skip
                c1 = {"ghost_check": C1_RECORD} if check else {}
                write_unit(run, name, tabs, {**jammed_record(u, VAR_J, resid, cfg["lost"][n]), **c1})
            else:
                write_unit(run, name, tabs, jammed_record(u, 1.3 * VAR_J, {}, 0.0))
    summary = {"gates": {"G-identity[A]": {"passed": True}}}  # E2 always records its gates; E2post reads them
    (run / "manifest.json").write_text(json.dumps({"spec": FAKE, "git_describe": DESCRIBE, "summary": summary}))
    return ctx


@pytest.fixture(scope="module")
def fake_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], dict[str, Any]]:
    """(E2's aggregate, E2post's summary) on one fake run."""
    root = tmp_path_factory.mktemp("e2")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GNSSAUDIT_RESULTS", str(root / "results"))
        ctx = build_run(root / "results" / "E2" / "fake", mp)
        summary, _ = e2.aggregate(ctx)
        post_dir = root / "results" / "E2post" / "run"
        post_dir.mkdir(parents=True)
        post = e2post.run_e2post(POST, post_dir).summary
    return summary, post


def test_a_segment_without_a_ceiling_leaves_the_family_without_one(fake_run: tuple[dict[str, Any], ...]) -> None:
    """B's noise fails the front-end rule at every level: J_max(B, noise) and L_B(B, noise) are None, and so are the
    combined J_max, L_B and L_K, whatever segment A reads (combine_segments: a None propagates)."""
    s, _ = fake_run
    a, b, d = s["segments"]["A"], s["segments"]["B"], s["decisions"]
    assert {lv: c["verdict"] for lv, c in b["front_end"]["noise"].items()} == dict.fromkeys(LEVELS, "fail")
    assert b["j_max_db_hz"]["noise"] is None and b["ceiling_db_hz"]["noise"] is None
    assert {lv: c["verdict"] for lv, c in a["front_end"]["noise"].items()} == dict.fromkeys(LEVELS, "pass")
    assert a["j_max_db_hz"]["noise"] == 60.0 and a["ceiling_db_hz"]["noise"] == 55.0  # ghost truth lost at 60
    assert d["j_max_db_hz"]["noise"] is None and d["ceiling_db_hz"]["noise"] is None
    assert d["k_level_db_hz"]["noise"] is None and d["k"] is None


def test_j_max_and_the_ceiling_combine_as_minima_under_segment_as_tau_cap(fake_run: tuple[dict[str, Any], ...]) -> None:
    """Swept CW passes every grid cell in both segments; segment A's tau = 100 ms side cell fails, so J_max(A, swept
    CW) is capped at the grid level below the side level; noise's side cells pass. B1 fails at 60 in A and at 55 in
    B; the combined J_max and L_B are the minima, L_K = min(80, L_B)."""
    s, _ = fake_run
    a, b, d = s["segments"]["A"], s["segments"]["B"], s["decisions"]
    assert all(c["verdict"] == "pass" for seg in (a, b) for c in seg["front_end"]["swept_cw"].values())
    assert {t: c["verdict"] for t, c in a["tau_side"]["swept_cw"].items()} == {0.001: "pass", 0.1: "fail"}
    assert {t: c["verdict"] for t, c in a["tau_side"]["noise"].items()} == {0.001: "pass", 0.1: "pass"}
    assert a["tau_cap_db_hz"] == {"noise": None, "swept_cw": 55.0}
    assert a["j_max_db_hz"]["swept_cw"] == 55.0 and a["ceiling_db_hz"]["swept_cw"] == 55.0
    assert b["j_max_db_hz"]["swept_cw"] == 60.0 and b["ceiling_db_hz"]["swept_cw"] == 50.0 and not b["tau_side"]
    assert d["j_max_db_hz"] == {"noise": None, "swept_cw": 55.0, "chirp": None}
    assert d["ceiling_db_hz"] == {"noise": None, "swept_cw": 50.0, "chirp": None}
    assert d["k_level_db_hz"] == {"noise": None, "swept_cw": 50.0, "chirp": None}
    # option (a), reported alongside: Var_B / Var_A = 1.3 everywhere, so its J_max is None
    assert a["var_b_over_a"]["swept_cw"][50.0] == pytest.approx(1.3)
    assert d["option_a"]["j_max_db_hz"] == {"noise": None, "swept_cw": None, "chirp": None}
    assert d["option_a"]["var_b_over_a"]["B"]["noise"][60.0] == pytest.approx(1.3)


def test_residual_pass_is_per_level_and_none_for_a_family_with_no_projected_level(
    fake_run: tuple[dict[str, Any], ...],
) -> None:
    """E2-A's summary read B1 as passed for every family because with no J_max there was no level to check (all([])
    is True). B1's bar r - 1 / (C/N0 T) < residual_max is now kept per level, and a family with no projected level
    reads None, in each segment and, combined over segments, in the decisions."""
    s, _ = fake_run
    a, b, d = s["segments"]["A"], s["segments"]["B"], s["decisions"]
    assert a["residual_pass"]["chirp"] is None and b["residual_pass"]["chirp"] is None
    assert d["residual_pass"]["chirp"] is None
    assert a["residual_pass"]["swept_cw"] == {50.0: True, 55.0: True, 60.0: False}
    assert b["residual_pass"]["swept_cw"] == {50.0: True, 55.0: False, 60.0: True}
    assert d["residual_pass"]["swept_cw"] == {50.0: True, 55.0: False, 60.0: False}
    assert d["residual_pass"]["noise"] == dict.fromkeys(LEVELS, True)
    assert a["truth_kept"]["noise"] == {50.0: True, 55.0: True, 60.0: False}


def test_e2s_front_end_cells_are_e2posts_on_the_same_run(fake_run: tuple[dict[str, Any], ...]) -> None:
    """E2post (P2b part 2, task 1) is the reference construction of a cell: on the same units E2's aggregate gives
    the same numbers, ghost by ghost, with the same records dropped."""
    s, post = fake_run
    for fam in ("noise", "swept_cw"):
        for lv in LEVELS:
            mine, ref = s["segments"]["A"]["front_end"][fam][lv], post["cells"][fam][f"{lv:g}"]
            assert (mine["ratio"], mine["ratio_se"], mine["verdict"]) == (ref["ratio"], ref["ratio_se"], ref["verdict"])
            assert {str(ch): g for ch, g in mine["excess"].items()} == ref["ghosts"]
            assert {str(ch): v for ch, v in mine["slopes"].items()} == {c: g["slope"] for c, g in ref["ghosts"].items()}
            assert mine["dropped_aligning"] == ref["dropped_aligning"]
    assert post["cells"]["noise"]["50"]["ghosts"]["2"]["dropped"] == 4 * SHIFTED.size  # the exact join did drop


def test_a_clean_twin_keeps_each_ghosts_absolute_error_as_a_done_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """e_c of the paired excess: ghost truth on the clean twin's own track (converged after half the warm-up, as
    E2post forms it), written beside the track and hashed into done.json, so --resume redoes the unit if it no longer
    hashes as recorded."""
    ctx = fake_context(tmp_path / "run", monkeypatch)
    seg = ctx.segments["A"]
    rng = np.random.default_rng(3)
    k = np.arange(1, N_REC + 1)
    tracks = {0: clean_track(GhostSpec(4, -2002.5, 0.0, 45.0, 3.0, 9), (10 * k).astype(np.uint64), 0.01 * k / k)}
    for i, ch in enumerate(seg.ghost_ch):
        tracks[ch] = clean_track(ghost_spec(i), (10 * k + 2 * i).astype(np.uint64), rng.normal(0.3, 0.05, k.size))

    def inject_receive(ctx: e2.Context, u: e2.Unit, *, keep: Path | None = None) -> tuple[Twin, None]:
        art = RunArtifacts(tmp_path, "cfg", tmp_path / "log", 0, {}, None, 1.0)
        return Twin(u.name, InjectionReport(seg.n0, {}, 0, "sha", "", 0, (), {}), art, dict(tracks), None), None

    monkeypatch.setattr(e2, "_inject_receive", inject_receive)
    e2.run_clean(ctx, e2.Unit("A-clean-A2", "A", "clean", "A", None, None, None, 0, None))
    udir = ctx.unit_dir("A-clean-A2")
    files = set(json.loads((udir / "done.json").read_text())["files"])
    assert {f"ghost_abs_ch{ch}.parquet" for ch in seg.ghost_ch} <= files and "ghost_abs_ch0.parquet" not in files
    for i, ch in enumerate(seg.ghost_ch):
        want = ghost_truth(tracks[ch], ghost_spec(i), clock=CLOCK, converge_s=ctx.warm / 2, truth_id="x")
        got = pq.read_table(udir / f"ghost_abs_ch{ch}.parquet")
        np.testing.assert_array_equal(got.column("e_cyc").to_numpy(), want.column("e_cyc").to_numpy())
        np.testing.assert_array_equal(got.column("valid").to_numpy(), want.column("valid").to_numpy())
    assert e2.is_done(udir, DESCRIBE)
    pq.write_table(pq.read_table(udir / f"ghost_abs_ch{seg.ghost_ch[1]}.parquet").slice(1),
                   udir / f"ghost_abs_ch{seg.ghost_ch[1]}.parquet")  # fmt: skip
    assert not e2.is_done(udir, DESCRIBE)


def test_run_e2_plans_the_k_study_from_the_ceiling_combined_over_both_segments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both segments' grids run before any decision; the oracle units are then planned at L_K of the ceiling combined
    over the segments (their minimum), not at segment A's own."""
    monkeypatch.setenv("GNSSAUDIT_RESULTS", str(tmp_path / "results"))
    monkeypatch.setattr(e2, "SHM_ROOT", tmp_path / "shm")  # the run clears its /dev/shm directory: not the host's
    calls: list[tuple[str, list[e2.Unit]]] = []
    monkeypatch.setattr(e2, "_segment_survey", lambda ctx, key: SimpleNamespace(source=SimpleNamespace(path=key)))
    monkeypatch.setattr(e2, "_parallel", lambda ctx, fn, units: calls.append((fn.__name__, list(units))))
    ceilings = {"A": {"noise": 65.0, "swept_cw": 60.0, "chirp": None},
                "B": {"noise": 55.0, "swept_cw": 70.0, "chirp": None}}  # fmt: skip

    def front(ctx: e2.Context) -> dict[str, dict[str, Any]]:
        jammed = {u.segment for name, units in calls if name == "run_jammed" for u in units if u.block == "grid"}
        assert jammed == {"A", "B"}  # both grids ran before the decisions
        return {key: {"ceiling_db_hz": c} for key, c in ceilings.items()}

    monkeypatch.setattr(e2, "front_end_segments", front)
    monkeypatch.setattr(e2, "aggregate", lambda ctx, front=None: ({"segments": {}, "decisions": {}}, []))
    run_dir = tmp_path / "results" / "E2" / "x"
    run_dir.mkdir(parents=True)
    e2.run_e2(dict(V2), run_dir)
    oracle = [u for _, units in calls for u in units if u.block.startswith(("k-", "thermal"))]
    assert {(u.block, u.family, u.level) for u in oracle if u.block != "thermal-clean"} == {
        ("k-cond", "swept_cw", 60.0), ("k-marg", "swept_cw", 60.0), ("k-noise", "noise", 55.0),
        ("thermal", "swept_cw", 60.0)}  # fmt: skip


# ------------------------------------------------------------------------------------- fix round 1 (task 5 review)
SHAPE = {"units", "slopes", "ratio", "ratio_se", "verdict", "excess", "dropped_aligning", "no_verdict"}


def test_a_tau_cell_without_a_verdict_caps_j_max_like_a_failed_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ruling (b): a tau side cell that could not be formed (here a ghost lost in the tau = 1 ms twin) cannot be shown
    not to fail, so it caps J_max(A, f) at the grid level below the side level, as a failed cell does. It keeps the
    full shape of a cell, with no verdict and the reason."""
    ctx = build_run(tmp_path / "run", monkeypatch, drop=("A-tau-noise-60-B1", "ghost_ch3.parquet"))
    a = e2.front_end_segment(ctx, "A")
    cell = a["tau_side"]["noise"][0.001]
    assert set(cell) == SHAPE and cell["verdict"] is None and cell["slopes"] is None and cell["ratio"] is None
    assert "A-tau-noise-60-B1/ghost_ch3.parquet" in cell["no_verdict"]
    assert a["tau_side"]["noise"][0.1]["verdict"] == "pass"
    assert a["tau_cap_db_hz"]["noise"] == 55.0 and a["j_max_db_hz"]["noise"] == 55.0  # 60 without the cap


def cell_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dphi: Any) -> tuple[e2.Context, e2.Unit, e2.Unit]:
    """One cell (segment A, noise at 50 dB-Hz) on disk: per ghost index i, dphi(i, k) gives (dPhi_A, dPhi_B, dPhi_B's
    validity); the clean twins' errors are white."""
    ctx = fake_context(tmp_path / "run", monkeypatch)
    units = {u.name: u for u in e2.plan_grid(FAKE, "A")}
    ua, ub = units["A-grid-noise-50-A"], units["A-grid-noise-50-B10"]
    rng = np.random.default_rng(4)
    k = np.arange(1, N_REC + 1)
    tables: dict[str, dict[str, pa.Table]] = {}
    for i, ch in enumerate(ctx.segments["A"].ghost_ch):
        stamps, prn = (10 * k + 2 * i).astype(np.uint64), ghost_spec(i).prn
        da, db, valid_b = dphi(i, k)
        for clean in (ua.clean or "", ub.clean or ""):
            e_c = rng.normal(0.0, np.sqrt(VAR_C), k.size)
            tables.setdefault(clean, {})[e2.GHOST_ABS.format(ch)] = twin_truth(stamps, e_c, prn)
        tables.setdefault(ua.name, {})[f"ghost_ch{ch}.parquet"] = twin_truth(stamps, da, prn)
        tables.setdefault(ub.name, {})[f"ghost_ch{ch}.parquet"] = twin_truth(stamps, db, prn, valid_b)
    for name, tabs in tables.items():
        write_unit(ctx.run_dir, name, tabs, dict(COMMON))
    return ctx, ua, ub


def jammed(rng: np.random.Generator, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    j = rng.normal(0.0, np.sqrt(VAR_J), k.size)
    return j, j + 0.1 * np.sqrt(VAR_J) * rng.standard_normal(k.size)


def test_a_cell_with_too_few_paired_epochs_has_no_verdict_and_the_full_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ghost whose mode B twin difference is valid on 1.5 s after onset + settle has fewer paired epochs than two
    1 s bootstrap blocks: no verdict, every key of a cell present."""
    rng = np.random.default_rng(5)

    def dphi(i: int, k: np.ndarray) -> tuple[np.ndarray, ...]:
        return (*jammed(rng, k), k < 16_500 if i == 1 else np.ones(k.size, bool))

    _, rec = e2.front_end_record(*cell_run(tmp_path, monkeypatch, dphi))
    assert set(rec) == SHAPE and rec["verdict"] is None and rec["excess"] is None and rec["ratio_se"] is None
    assert "paired epochs" in rec["no_verdict"]


def test_a_cell_whose_ghosts_share_too_few_epochs_has_no_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each ghost has 6 s of paired epochs, but the two overlap on 1999 records, one short of two 1 s blocks once
    aligned (SE(R) draws its blocks on epochs every ghost holds): no verdict; the per-ghost excess stays reported."""
    rng = np.random.default_rng(6)

    def dphi(i: int, k: np.ndarray) -> tuple[np.ndarray, ...]:
        return (*jammed(rng, k), k < 21_000 if i == 0 else k >= 19_001)

    _, rec = e2.front_end_record(*cell_run(tmp_path, monkeypatch, dphi))
    assert set(rec) == SHAPE and rec["verdict"] is None and rec["ratio"] is None
    assert set(rec["excess"]) == {2, 3} and "aligned" in rec["no_verdict"]


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:divide by zero encountered:RuntimeWarning")
def test_a_cell_with_a_non_finite_slope_or_ratio_has_no_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ghost the jammer leaves untouched in mode A (dPhi_A = 0) has no slope (0 / 0); with both so, R has no finite
    value either. Neither may reach a verdict, nor stop the aggregation."""
    rng = np.random.default_rng(7)
    for untouched in ({0}, {0, 1}):

        def dphi(i: int, k: np.ndarray, untouched: set[int] = untouched) -> tuple[np.ndarray, ...]:
            da, db = jammed(rng, k)
            return (np.zeros(k.size) if i in untouched else da), db, np.ones(k.size, bool)

        c, rec = e2.front_end_record(*cell_run(tmp_path / str(len(untouched)), monkeypatch, dphi))
        assert c is None and set(rec) == SHAPE and rec["verdict"] is None, rec
        assert rec["no_verdict"] == "a non-finite slope, R or SE(R)" and math.isnan(rec["slopes"][2])
        assert math.isfinite(rec["ratio"]) == (len(untouched) == 1)


def test_the_seed_scheme_refuses_what_would_let_two_blocks_draw_alike() -> None:
    """A seed's configuration is 100 x block + the family's sorted index (grid 1, K-study 2 on, thermal 9): a family
    index of 100 would reach the next block, and a K-study block number of 9 the thermal block."""
    many = {**V2, "families": {f"f{i:03d}": {} for i in range(101)}}
    with pytest.raises(ValueError, match="100 families"):
        e2.plan_grid(many, "B")
    e2.plan_grid({**V2, "families": {f"f{i:03d}": {} for i in range(100)}}, "B")  # 100 still fit
    blocks = {f"b{i}": {"family": "noise", "K": 2, "conditional": False} for i in range(8)}
    ceiling = {"noise": 60.0, "swept_cw": 60.0, "chirp": None}
    with pytest.raises(ValueError, match="K-study blocks"):
        e2.plan_oracle({**V2, "kstudy": {**V2["kstudy"], "blocks": blocks}}, "A", ceiling)
    seven = dict(list(blocks.items())[:7])
    assert len(e2.plan_oracle({**V2, "kstudy": {**V2["kstudy"], "blocks": seven}}, "A", ceiling)) == 7 * 2 + 2 * 51


def test_a_finished_unit_that_is_not_the_planned_one_is_refused_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Oracle unit names omit the level, which follows from the front-end decisions: a finished unit recorded for
    another level must not stand in for the planned one. The planned unit itself is reused as it is."""
    ctx = fake_context(tmp_path / "run", monkeypatch)
    u = next(x for x in e2.plan_oracle(FAKE, "A", {"noise": 55.0, "swept_cw": 50.0}) if x.name == "A-k-noise-r0")

    def inject_receive(*args: Any, **kwargs: Any) -> None:
        pytest.fail("a finished unit was redone")

    monkeypatch.setattr(e2, "_inject_receive", inject_receive)
    write_unit(ctx.run_dir, u.name, {}, {"unit": dataclasses.asdict(dataclasses.replace(u, level=60.0))})
    with pytest.raises(RuntimeError, match="A-k-noise-r0.*not the planned unit"):
        e2.run_jammed(ctx, u)
    (ctx.unit_dir(u.name) / "unit.json").write_text(json.dumps(jsonable({"unit": dataclasses.asdict(u)})))
    e2.mark_done(ctx.unit_dir(u.name), DESCRIBE, ["unit.json"])
    e2.run_jammed(ctx, u)


# ------------------------------------------------------------------------------------- task 13 review (Important 1)
def test_g_warmup_lock_covers_the_thermal_pairs_clean_twins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """twin_gates on one thermal pair: its clean twin (A-thermal-clean-r0) is held to its lock policy like every other
    twin, so one that did not meet it fails G-warmup-lock (the name matched none of the lock records' prefixes before
    the task 13 review). The gates keep their names and order, and the records their keys (jammed units only)."""
    ceiling = {"noise": 55.0, "swept_cw": None, "chirp": None}
    u = next(x for x in e2.plan_oracle(FAKE, "A", ceiling) if x.name == "A-thermal-r0")
    k = np.arange(1, N_REC + 1)
    for ok in (True, False):
        ctx = fake_context(tmp_path / f"run-{ok}", monkeypatch)
        clean = {**COMMON, "lock_ok": ok, "unlocked": [] if ok else [1], "config_sha256": "cfg", "rx_wall_s": 1.0}
        write_unit(ctx.run_dir, u.clean or "", {"track_ch0.parquet": track((10 * k).astype(np.uint64), 0.0 * k, 4)},
                   clean)  # fmt: skip
        write_unit(ctx.run_dir, u.name, {}, {**jammed_record(u, VAR_J, {}, 0.0), "identity": {"0": {"passed": True}}})
        ctx.cleans[u.clean or ""] = e2._load_twin(ctx.unit_dir(u.clean or ""), u.clean or "")
        records, gates = e2.twin_gates(ctx, "A")
        # no null pair here: G-null fails for want of one (final review R1 M2)
        assert [(sfx, g.gate) for sfx, g in gates] == [("[A]", "G-warmup-lock"), ("[A]", "G-phase-precision"),
                                                       ("[A]", "G-identity"), ("[A]", "G-null")]  # fmt: skip
        lock = gates[0][1]
        assert lock.passed is ok, lock.details
        assert lock.details == {"units_failed": [] if ok else [u.clean],
                                "excluded_channels": {} if ok else {u.clean: [1]}}  # fmt: skip
        assert gates[1][1].passed and gates[2][1].passed and gates[2][1].details == {"pairs": 1, "failed": 0}
        assert not gates[3][1].passed
        assert set(records) == {"lost_ghost_s", "stock_lock_drops", "clips", "ghost_null"}
        assert list(records["lost_ghost_s"]) == [u.name]


# ------------------------------------------------------------------------------- final review, fix wave 1 (R1)
def test_a_torn_done_file_counts_as_not_done(tmp_path: Path) -> None:
    """R1 I1: after a crash, a renamed but unflushed done.json can come back empty or torn. The unit is then redone,
    not left to stop every resume with a decoding error."""
    u = tmp_path / "A-grid-noise-60-A"
    u.mkdir()
    (u / "truth.parquet").write_bytes(b"abc")
    e2.mark_done(u, "v1", ["truth.parquet"])
    assert e2.is_done(u, "v1")
    for torn in (b"", b"\0" * 64, b'{"describe": "v1", "fi', b"\xff\xfe\xfd"):
        (u / "done.json").write_bytes(torn)
        assert not e2.is_done(u, "v1")


def test_done_json_is_fsynced_before_its_rename_and_its_directory_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1 I1: done.json is made durable as every committed file is (io.fetch.commit_part): the file is fsynced under
    its .part name, renamed, then its directory is fsynced."""
    synced: list[str] = []
    real = os.fsync

    def fsync(fd: int) -> None:
        synced.append(os.readlink(f"/proc/self/fd/{fd}"))
        real(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    u = (tmp_path / "unit").resolve()
    u.mkdir()
    (u / "a.parquet").write_bytes(b"abc")
    e2.mark_done(u, "v1", ["a.parquet"])
    assert synced == [str(u / "done.json.part"), str(u)]
    assert e2.is_done(u, "v1") and not (u / "done.json.part").exists()


def test_an_oracle_block_reads_its_realizations_in_realization_order(tmp_path: Path) -> None:
    """R1 M5: the first k_small realizations of the K = 101 block are r0, r1, ... in number order (r2 before r10), and a
    realization without truth is not among them."""
    for r in range(12):
        d = tmp_path / "units" / f"A-k-cond-r{r}"
        d.mkdir(parents=True)
        if r != 4:
            (d / "truth_obs.parquet").write_bytes(b"")
    ctx = e2.Context({}, tmp_path, tmp_path / "work", DESCRIBE, None, {}, {}, {}, {})
    got = [d.name for d in e2._realizations(ctx, "A", "k-cond")]
    assert got == [f"A-k-cond-r{r}" for r in range(12) if r != 4]


def shm_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[e2.Context, e2.Unit, int]:
    """The fake run's context, a mode A grid unit of segment A, and the bytes of one of its twins in /dev/shm."""
    ctx = fake_context(tmp_path / "run", monkeypatch)
    u = next(x for x in e2.plan_grid(FAKE, "A") if x.name == "A-grid-noise-50-A")
    seg = ctx.segments["A"]
    return ctx, u, 4 * ((seg.source.sample_end - seg.source.sample_start) // seg.clock.decim)


def free_space(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shm: int, disk: int = 10**15) -> list[Path]:
    """shutil.disk_usage replaced: `shm` bytes free in /dev/shm, `disk` elsewhere; the paths asked are recorded."""
    usage, asked = shutil.disk_usage(tmp_path), []

    def disk_usage(p: Any) -> Any:
        asked.append(Path(p))
        return usage._replace(free=shm if Path(p) == Path("/dev/shm") else disk)

    monkeypatch.setattr(shutil, "disk_usage", disk_usage)
    return asked


def test_the_shm_guard_refuses_a_twin_before_it_is_injected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R1 I4: with less than one twin and the margin free in /dev/shm, the unit stops with StorageGuardError before
    inject_twin writes anything."""
    ctx, u, twin = shm_context(tmp_path, monkeypatch)
    asked = free_space(tmp_path, monkeypatch, shm=SHM_MARGIN + twin - 1)

    def inject_twin(*args: Any, **kwargs: Any) -> None:
        pytest.fail("a twin was injected past the guard")

    monkeypatch.setattr(e2, "inject_twin", inject_twin)
    with pytest.raises(StorageGuardError, match="/dev/shm"):
        e2._inject_receive(ctx, u)
    assert asked == [Path("/dev/shm")]


class Written(Exception):
    """A fake inject_twin's way out: the twin counts as written."""


def test_pipelines_in_flight_cannot_all_pass_a_guard_with_room_for_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1 I4: a twin being written is not yet in /dev/shm's usage, so the guard counts the twins other pipelines are
    still writing. With room for one and a half twins, a second unit that asks while the first is written is refused;
    once the first is written (or has failed), the second may go."""
    ctx, u, twin = shm_context(tmp_path, monkeypatch)
    free_space(tmp_path, monkeypatch, shm=SHM_MARGIN + twin + twin // 2)
    second = next(x for x in e2.plan_grid(FAKE, "A") if x.name == "A-grid-noise-55-A")
    calls: list[str] = []

    def inject_twin(name: str, *args: Any, **kwargs: Any) -> None:
        calls.append(name)
        if name == u.name:  # the first twin is being written: another pipeline asks for room
            with pytest.raises(StorageGuardError):
                e2._inject_receive(ctx, second)
        raise Written(name)

    monkeypatch.setattr(e2, "inject_twin", inject_twin)
    with pytest.raises(Written):
        e2._inject_receive(ctx, u)
    assert calls == [u.name]
    with pytest.raises(Written):
        e2._inject_receive(ctx, second)
    assert calls == [u.name, second.name]


@pytest.mark.parametrize("fails", ["receive", "keep"])
def test_a_unit_that_fails_after_injection_leaves_no_twin_in_shm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fails: str
) -> None:
    """R1 I4: a twin whose receiver run raises, or whose bytes cannot be kept on disk, is deleted from /dev/shm, so the
    next units' guards do not count it (one failure must not cascade)."""
    ctx, u, _ = shm_context(tmp_path, monkeypatch)
    free_space(tmp_path, monkeypatch, shm=10**15, disk=0 if fails == "keep" else 10**15)
    shm = tmp_path / "shm" / f"{u.name}.bin"
    report = InjectionReport(1.0, {}, 0, "sha", "", 0, (), {})

    def inject_twin(*args: Any, **kwargs: Any) -> tuple[SourceSpec, InjectionReport]:
        shm.parent.mkdir(parents=True, exist_ok=True)
        shm.write_bytes(bytes(64))
        return SourceSpec(path=shm, item_type="ishort", fs_hz=FS / DECIM), report

    def receive(name: str, *args: Any, **kwargs: Any) -> Twin:
        if fails == "receive":
            raise RuntimeError("GNSS-SDR exited with status 1")
        return Twin(name, None, RunArtifacts(tmp_path, "cfg", tmp_path / "log", 0, {}, None, 1.0), {}, None)

    monkeypatch.setattr(e2, "inject_twin", inject_twin)
    monkeypatch.setattr(e2, "receive", receive)
    with pytest.raises(RuntimeError if fails == "receive" else StorageGuardError):
        e2._inject_receive(ctx, u, keep=tmp_path / "work" / f"{u.name}.bin" if fails == "keep" else None)
    assert not shm.exists()


@pytest.mark.parametrize("position", [0, 3])
def test_the_first_failure_stops_a_batch_without_running_its_queued_units(tmp_path: Path, position: int) -> None:
    """R1 I4: with several pipelines, a failed unit cancels the units still queued (those running finish) and its
    error surfaces then, not after every queued unit has run."""
    ran: list[str] = []

    def fn(ctx: e2.Context, u: e2.Unit) -> None:
        if u.name == "bad":
            raise RuntimeError("bad failed")
        time.sleep(0.05)
        ran.append(u.name)

    names = [f"u{i}" for i in range(40)]
    names.insert(position, "bad")
    units = [e2.Unit(n, "A", "grid", "A", None, None, None, 0, None) for n in names]
    ctx = e2.Context({"pipelines": 2}, tmp_path, tmp_path / "work", DESCRIBE, None, {}, {}, {}, {})
    with pytest.raises(RuntimeError, match="bad failed"):
        e2._parallel(ctx, fn, units)
    assert len(ran) <= 10


def test_run_e2_clears_its_shm_directory_before_any_unit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R1 I4: twins a killed run left in /dev/shm (a SIGKILL skips the final cleanup) are removed when the run starts
    again, before the first guard reads the free space."""
    monkeypatch.setenv("GNSSAUDIT_RESULTS", str(tmp_path / "results"))
    monkeypatch.setattr(e2, "SHM_ROOT", tmp_path / "shm")
    run_dir = tmp_path / "results" / "E2" / "x"
    run_dir.mkdir(parents=True)
    stale = tmp_path / "shm" / "x" / "A-grid-noise-50-A.bin"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(bytes(8))
    seen: list[bool] = []

    def survey(ctx: e2.Context, key: str) -> None:
        seen.append(stale.exists())
        raise RuntimeError("stop")

    monkeypatch.setattr(e2, "_segment_survey", survey)
    with pytest.raises(RuntimeError, match="stop"):
        e2.run_e2(dict(V2), run_dir)
    assert seen == [False]


def test_g_phase_precision_reads_only_the_segments_own_clean_twins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1 M1: each segment's G-phase-precision is read on that segment's clean twins, not on every segment's (run 2
    showed one value under both [A] and [B])."""
    ctx = fake_context(tmp_path / "run", monkeypatch)
    k = np.arange(1, 101)
    phase = {"A": 10.0, "B": 2.5e6}
    for key, p in phase.items():
        name = f"{key}-clean-A"
        tracks = {"track_ch0.parquet": track((10 * k).astype(np.uint64), np.full(k.size, p), 4)}
        write_unit(ctx.run_dir, name, tracks, {**COMMON, "config_sha256": "cfg", "rx_wall_s": 1.0})
        ctx.cleans[name] = e2._load_twin(ctx.unit_dir(name), name)
    for key, p in phase.items():
        _, gates = e2.twin_gates(ctx, key)
        (precision,) = [g for _, g in gates if g.gate == "G-phase-precision"]
        assert precision.details == {"max_abs_phase_rad": p}


def test_g_null_uses_the_slip_size_of_the_specs_signal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R1 M7: the null gate's twin difference takes the slip size of the spec's template, as reduce_unit does."""
    ctx = fake_context(tmp_path / "run", monkeypatch)
    ctx.spec = {**ctx.spec, "template": "L5"}
    k = np.arange(1, 101)
    for name in ("A-clean-A", "A-clean-A2"):
        tracks = {f"track_ch{ch}.parquet": track((10 * k).astype(np.uint64), 0.0 * k, prn)
                  for ch, prn in ((0, 4), (1, 9))}  # fmt: skip
        write_unit(ctx.run_dir, name, tracks, {**COMMON, "config_sha256": "cfg", "rx_wall_s": 1.0})
        ctx.cleans[name] = e2._load_twin(ctx.unit_dir(name), name)
    sizes: list[float] = []

    def twin_difference(a: Any, b: Any, **kwargs: Any) -> tuple[None, None, None]:
        sizes.append(kwargs["slip_size_cyc"])
        return None, None, None

    monkeypatch.setattr(e2, "twin_difference", twin_difference)
    monkeypatch.setattr(e2, "null_gate", lambda *a, **k: GateReport("G-null", True, {}))
    e2.twin_gates(ctx, "A")
    assert sizes == [SLIP_CYC["L5"]] * 2


def test_kept_clean_bytes_that_no_longer_hash_as_recorded_are_made_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1 M4: on resume, a finished mode A clean twin's kept bytes are reused only if they hash as its unit.json
    records; a file a crash left short is made again, and must then hash as recorded."""
    ctx = fake_context(tmp_path / "run", monkeypatch)
    u = next(x for x in e2.plan_grid(FAKE, "A") if x.name == "A-clean-A")
    assert u.keep_bytes
    good = bytes(range(64))
    k = np.arange(1, 101)
    tracks = {"track_ch0.parquet": track((10 * k).astype(np.uint64), 0.0 * k, 4)}
    rec = {**COMMON, "config_sha256": "cfg", "rx_wall_s": 1.0, "output_sha256": hashlib.sha256(good).hexdigest()}
    write_unit(ctx.run_dir, u.name, tracks, rec)
    kept = ctx.work / f"{u.name}.bin"
    kept.parent.mkdir(parents=True)
    calls: list[str] = []

    def inject_receive(ctx: e2.Context, u: e2.Unit, *, keep: Path | None = None) -> tuple[Twin, None]:
        calls.append(u.name)
        assert keep is not None and keep == kept and not keep.exists()
        keep.write_bytes(good)
        report = InjectionReport(1.0, {}, 0, hashlib.sha256(good).hexdigest(), "", 0, (), {})
        return Twin(u.name, report, RunArtifacts(tmp_path, "cfg", tmp_path / "log", 0, {}, None, 1.0), {}, None), None

    monkeypatch.setattr(e2, "_inject_receive", inject_receive)
    kept.write_bytes(good)
    e2.run_clean(ctx, u)
    assert calls == []  # intact: reused as it is
    kept.write_bytes(good[:10])  # left short by a crash
    e2.run_clean(ctx, u)
    assert calls == [u.name] and kept.read_bytes() == good


# ------------------------------------------------------------------------------- final review, fix wave 2 (R1)
def null_twins(ctx: e2.Context, tracks: dict[str, dict[int, np.ndarray]], seconds: int = 16) -> None:
    """Segment A's null pair (A-clean-A, A-clean-A2) on 1 ms records: per twin, the channels given, with that phase
    (cycles) on the fake survey's PRNs (channels 0 and 1 real, 2 and 3 the ghosts)."""
    k = np.arange(1, 1000 * seconds + 1)
    prns = [*SURVEY["sats"], *(g["prn"] for g in SURVEY["ghosts"])]
    for name, chans in tracks.items():
        tabs = {f"track_ch{ch}.parquet": track((10 * k).astype(np.uint64), 2 * np.pi * phase, prns[ch])
                for ch, phase in chans.items()}  # fmt: skip
        write_unit(ctx.run_dir, name, tabs, {**COMMON, "config_sha256": "cfg", "rx_wall_s": 1.0})
        ctx.cleans[name] = e2._load_twin(ctx.unit_dir(name), name)


def test_g_null_fails_without_a_null_twin_or_a_channel_in_both(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R1 M2 (ruled 22 Oct 2026): G-null wrote no entry when a null twin was missing or no real channel was in both
    twins, so it could not fail. Either case now writes one failing G-null for the segment, naming what is missing;
    with a real channel in both twins the gate is per channel, as before."""
    k = np.zeros(16_000)
    ctx = fake_context(tmp_path / "missing", monkeypatch)
    null_twins(ctx, {"A-clean-A": {0: k, 1: k}})
    _, gates = e2.twin_gates(ctx, "A")
    nulls = [(sfx, g) for sfx, g in gates if g.gate == "G-null"]
    assert [(sfx, g.passed, g.details) for sfx, g in nulls] == [
        ("[A]", False, {"pairs": 0, "missing_twins": ["A-clean-A2"]})]  # fmt: skip
    ctx = fake_context(tmp_path / "unpaired", monkeypatch)
    null_twins(ctx, {"A-clean-A": {0: k}, "A-clean-A2": {1: k, 2: k}})  # no real channel in both
    _, gates = e2.twin_gates(ctx, "A")
    nulls = [(sfx, g) for sfx, g in gates if g.gate == "G-null"]
    assert [(sfx, g.passed, g.details) for sfx, g in nulls] == [("[A]", False, {"pairs": 0, "missing_twins": []})]
    ctx = fake_context(tmp_path / "paired", monkeypatch)
    null_twins(ctx, {"A-clean-A": {0: k, 1: k}, "A-clean-A2": {0: k, 1: k}})
    _, gates = e2.twin_gates(ctx, "A")
    assert [(sfx, g.passed) for sfx, g in gates if g.gate == "G-null"] == [("[A][ch0]", True), ("[A][ch1]", True)]


def test_the_ghosts_null_difference_is_reported_beside_g_null_and_gates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1 M2 (ruled 22 Oct 2026): the ghosts, whose dPhi every front-end and B1 rule rests on, had no null check. Their
    null twin difference is now computed by G-null's own test and reported in the segment's records; it gates nothing
    (adding them to the pre-registered gate after the data would change its scope). Here ghost PRN 31 (channel 3)
    differs between the two twins far beyond G-null's tolerance, and every gate still passes."""
    rng = np.random.default_rng(3)
    k = np.zeros(16_000)
    noisy = rng.normal(0.0, 0.01, k.size)
    ctx = fake_context(tmp_path / "run", monkeypatch)
    null_twins(ctx, {"A-clean-A": {0: k, 1: k, 2: k, 3: k}, "A-clean-A2": {0: k, 1: k, 2: k, 3: noisy}})
    records, gates = e2.twin_gates(ctx, "A")
    assert all(g.passed for sfx, g in gates if g.gate == "G-null")
    assert [sfx for sfx, g in gates if g.gate == "G-null"] == ["[A][ch0]", "[A][ch1]"]  # the ghosts gate nothing
    ghost = records["ghost_null"]
    assert ghost["gating"] is False and set(ghost["channels"]) == {2, 3}
    same, apart = ghost["channels"][2], ghost["channels"][3]
    assert (same["prn"], apart["prn"]) == (32, 31) and same["computed"] and apart["computed"]
    assert same["rms_over_sigma"] == 0.0 and same["meets_g_null"] is True and same["epochs"] > 10_000
    assert apart["rms_over_sigma"] > apart["tol"] and apart["meets_g_null"] is False
    # without the second twin nothing is computed, and the record says why
    ctx = fake_context(tmp_path / "alone", monkeypatch)
    null_twins(ctx, {"A-clean-A": {0: k, 1: k, 2: k, 3: k}})
    records, _ = e2.twin_gates(ctx, "A")
    reason = "null twin missing: A-clean-A2"
    assert records["ghost_null"] == {"gating": False, "channels": {
        ch: {"prn": prn, "computed": False, "reason": reason} for ch, prn in ((2, 32), (3, 31))}}  # fmt: skip


def test_g_repeat_fails_with_no_receiver_check(tmp_path: Path) -> None:
    """R1 M3 (ruled 22 Oct 2026): a jammed unit's repeat records only its truth agreement, which nothing gates, so a
    draw of jammed units alone passed G-repeat with zero receiver checks. G-repeat now needs at least one receiver
    check (a clean twin's repeat); identical bytes alone do not pass it."""
    ctx = e2.Context({}, tmp_path, tmp_path / "work", DESCRIBE, None, {}, {}, {}, {})
    jam = {**COMMON, "of": "A-k-cond-r0", "same_bytes": True, "truth_rms_over_sigma": {"G04": 0.01}}
    write_unit(tmp_path, "repeat-A-k-cond-r0", {}, jam)
    g = e2.repeat_report(ctx)
    assert (g.gate, g.passed) == ("G-repeat", False)
    assert g.details == {"units": 1, "bytes_differ": 0, "receiver_checks": 0, "receiver_failed": 0}
    clean = {**COMMON, "of": "A-thermal-clean-r0", "same_bytes": True, "repeat": {"0": {"passed": True}}}
    write_unit(tmp_path, "repeat-A-thermal-clean-r0", {}, clean)
    g = e2.repeat_report(ctx)
    assert g.passed and g.details == {"units": 2, "bytes_differ": 0, "receiver_checks": 1, "receiver_failed": 0}


# ------------------------------------------------------------------------ P2c task 3: E2 under ghost truth (R-1 to R-3)
E2K_SPEC = yaml.safe_load((REPO / "experiments" / "E2K.yaml").read_text())
GHOST = {**FAKE, "truth": "ghost", "s29_result": E2K_SPEC["s29_result"], "s29_sha256": E2K_SPEC["s29_sha256"]}


def load_v3() -> dict[str, Any]:
    return yaml.safe_load((REPO / "experiments" / "E2-v3.yaml").read_text())


def test_e2_v3_is_e2_v2_with_the_registered_changes() -> None:
    """R-3: run 3's spec is E2-v2's with a fresh seed, ghost truth and the receiver build pinned, plus E2K's s29 pin,
    which only the reported Route B arm reads (controller ruling P2). Everything else, the rules, the K-study and the
    thermal block included, is E2-v2's, value for value. `workers_b` (ruling R3a, 27 Oct 2026) is an execution
    parameter only: mode B on six cores, every injected byte as on one."""
    v3 = load_v3()
    changed = {"seed", "truth", "receiver_patch", "s29_result", "s29_sha256", "workers_b"}
    assert {k: v for k, v in v3.items() if k not in changed} == {k: v for k, v in V2.items() if k not in changed}
    assert set(v3) ^ set(V2) == changed - {"seed"}
    assert (v3["seed"], v3["truth"], v3["receiver_patch"]) == (20261024, "ghost", PATCH_ID)
    assert (e2.inject_workers(v3, "A"), e2.inject_workers(v3, "B")) == (6, 6)
    assert (v3["s29_result"], v3["s29_sha256"]) == (E2K_SPEC["s29_result"], E2K_SPEC["s29_sha256"])
    assert receiver_patch_refusal(v3) is None  # it runs on this tree's build
    assert e2.truth_mode(v3) == "ghost" and e2.truth_mode(V2) == "route_b"  # E2-v2 has no key: Route B, as before


def test_an_unknown_truth_is_refused_before_any_unit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A misspelt `truth` must not run as E2-v2."""
    with pytest.raises(ValueError, match="truth 'ghosts'"):
        e2.truth_mode({**V2, "truth": "ghosts"})
    monkeypatch.setenv("GNSSAUDIT_RESULTS", str(tmp_path / "results"))
    monkeypatch.setattr(e2, "SHM_ROOT", tmp_path / "shm")
    monkeypatch.setattr(e2, "_segment_survey", lambda ctx, key: pytest.fail("a unit ran"))
    run_dir = tmp_path / "results" / "E2" / "x"
    run_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="truth 'ghosts'"):
        e2.run_e2({**V2, "truth": "ghosts"}, run_dir)


def test_a_wrong_s29_pin_stops_run_3_before_any_unit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The Route B arm reads s29 by SHA-256 only at the end of the run; a wrong pin is refused at its start, not
    after nine hours of units."""
    monkeypatch.setenv("GNSSAUDIT_RESULTS", str(tmp_path / "results"))
    monkeypatch.setattr(e2, "SHM_ROOT", tmp_path / "shm")
    monkeypatch.setattr(e2, "_segment_survey", lambda ctx, key: pytest.fail("a unit ran"))
    run_dir = tmp_path / "results" / "E2" / "x"
    run_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="s29_sha256"):
        e2.run_e2({**load_v3(), "s29_sha256": "0" * 64}, run_dir)


def drawn_seeds(spec: dict[str, Any], run: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, set[int]]:
    """Every seed a run of `spec` draws: its jammed units' (both segments' grids, the tau side check, every K-study
    block and the thermal pairs, all planned), the thermal floors, the mode B dither and the ghosts."""
    ctx = fake_context(run, monkeypatch, {**FAKE, "seed": spec["seed"]})
    every = {"noise": 60.0, "swept_cw": 60.0, "chirp": 60.0}
    units = [u for key in spec["segments"] for u in (*e2.plan_grid(spec, key), *e2.plan_oracle(spec, key, every))]
    mode_b = next(u for u in units if u.mode == "B")
    return {"unit": {u.seed for u in units if u.family is not None},
            "floor": {u.floor_seed for u in units if u.floor_seed is not None},
            "dither": {e2.front_end(ctx, mode_b).dither_seed or 0},
            "ghost": {g.seed for seg in ctx.segments.values() for g in seg.ghosts}}  # fmt: skip


def test_no_seed_of_run_3_repeats_one_of_run_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Review Focus 4: run 3's seed 20261024 gives every draw afresh; no unit, floor, dither or ghost seed of its plan
    is one of E2-v2's (a clean twin's seed 0 draws nothing: its jammer is at zero gain)."""
    v2, v3 = drawn_seeds(V2, tmp_path / "v2", monkeypatch), drawn_seeds(load_v3(), tmp_path / "v3", monkeypatch)
    assert {k: len(v) for k, v in v3.items()} == {k: len(v) for k, v in v2.items()}
    assert all(v3[k] for k in v3) and len(v3["unit"]) > 200 and len(v3["floor"]) == 51
    assert not set().union(*v3.values()) & set().union(*v2.values())


def oracle_plan(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch, spec: dict[str, Any], front: dict[str, dict[str, Any]]
) -> tuple[set[tuple[Any, ...]], dict[str, Any]]:
    """The oracle units run_e2 plans (block, family, level, timing) once the grid has given `front`, and the summary it
    writes (aggregate replaced, with the decisions g3_clauses_ghost reads)."""
    monkeypatch.setenv("GNSSAUDIT_RESULTS", str(run_dir.parent.parent))
    monkeypatch.setattr(e2, "SHM_ROOT", run_dir.parent.parent / "shm")
    calls: list[tuple[str, list[e2.Unit]]] = []
    monkeypatch.setattr(e2, "_segment_survey", lambda ctx, key: SimpleNamespace(source=SimpleNamespace(path=key)))
    monkeypatch.setattr(e2, "_parallel", lambda ctx, fn, units: calls.append((fn.__name__, list(units))))
    monkeypatch.setattr(e2, "front_end_segments", lambda ctx: front)
    d = {"j_max_db_hz": {"noise": 60.0, "swept_cw": 65.0}, "ghost_ceiling_db_hz": {"noise": 60.0, "swept_cw": 65.0},
         "k": None}  # fmt: skip
    monkeypatch.setattr(e2, "aggregate", lambda ctx, front=None: ({"segments": {}, "decisions": d}, []))
    run_dir.mkdir(parents=True)
    e2.run_e2(spec, run_dir)
    oracle = [u for _, units in calls for u in units if u.block.startswith(("k-", "thermal"))]
    planned = {(u.block, u.family, u.level, u.offset_s) for u in oracle if u.block != "thermal-clean"}
    return planned, json.loads((run_dir / "summary.json").read_text())


def test_under_ghost_truth_run_e2_plans_the_k_study_at_the_ghost_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R-2, clause 4: under truth: ghost the K-study is planned at L_K = min(80, L_G), the ghost ceiling combined over
    the segments. Here L_G is {noise: 60, swept CW: 65} and L_B {noise: 60, swept CW: None}: the swept CW blocks run at
    65, where under route_b the missing L_B(swept CW) moves the K = 101 and thermal blocks to noise. Under ghost truth
    run_e2 also writes g3_clauses."""
    front = {"A": {"ceiling_db_hz": {"noise": 60.0, "swept_cw": None, "chirp": None},
                   "ghost_ceiling_db_hz": {"noise": 60.0, "swept_cw": 65.0, "chirp": None}},
             "B": {"ceiling_db_hz": {"noise": 65.0, "swept_cw": 60.0, "chirp": None},
                   "ghost_ceiling_db_hz": {"noise": 65.0, "swept_cw": 70.0, "chirp": None}}}  # fmt: skip
    ghost, s = oracle_plan(tmp_path / "g" / "E2" / "x", monkeypatch, load_v3(), front)
    assert ghost == {("k-cond", "swept_cw", 65.0, 0.0037), ("k-marg", "swept_cw", 65.0, None),
                     ("k-noise", "noise", 60.0, None), ("thermal", "swept_cw", 65.0, 0.0037)}  # fmt: skip
    assert s["g3_clauses"] == jsonable(e2.g3_clauses_ghost({}, s["decisions"], e2.G3_HZ_SOURCE))
    route_b, s = oracle_plan(tmp_path / "b" / "E2" / "x", monkeypatch, {**load_v3(), "truth": "route_b"}, front)
    assert route_b == {("k-cond", "noise", 60.0, None), ("k-noise", "noise", 60.0, None),
                       ("thermal", "noise", 60.0, None)}  # fmt: skip
    assert "g3_clauses" not in s


GATES = {g: {"passed": True} for g in ("G-warmup-lock[A]", "G-phase-precision[A]", "G-identity[A]", "G-null[A][ch0]",
                                       "G-warmup-lock[B]", "G-phase-precision[B]", "G-identity[B]", "G-null[B][ch3]",
                                       "G-repeat[repeat]")}  # fmt: skip


def ghost_decisions(**over: Any) -> dict[str, Any]:
    d = {"j_max_db_hz": {"noise": 60.0, "swept_cw": 65.0, "chirp": None}, "ceiling_db_hz": dict.fromkeys(LEVELS),
         "ghost_ceiling_db_hz": {"noise": 60.0, "swept_cw": 65.0, "chirp": None}, "k": 51}  # fmt: skip
    return {**d, **over}


def test_g3_under_ghost_truth_passes_when_every_clause_does() -> None:
    """R-2: clause 1 every gate of the run, clause 2 G-hz as passed on E0 (recorded, not re-run), clause 3 a ghost
    ceiling for noise and for swept CW, clause 4 K chosen. The function reads nothing but its arguments."""
    gates, d = json.loads(json.dumps(GATES)), ghost_decisions()
    c = e2.g3_clauses_ghost(gates, d, e2.G3_HZ_SOURCE)
    assert c == {"gates": {"passed": True, "count": 9, "failed": []},
                 "hz": {"source": ["E0b/fee767ba3538", "E0b/46b76d4b47d9"]},
                 "clause3": {"passed": True, "narrowed": [], "j_max_db_hz": d["j_max_db_hz"],
                             "ghost_ceiling_db_hz": d["ghost_ceiling_db_hz"]},
                 "k": {"k": 51, "chosen": True}, "passed": True}  # fmt: skip
    assert gates == GATES and d == ghost_decisions()  # its arguments are left as they were


def test_one_family_with_a_ghost_ceiling_does_not_pass_clause_3() -> None:
    """Review Focus 1: the registered clause names noise and swept CW. A ceiling for one of them is a narrowed result,
    reported beside the verdict, never the verdict; the chirp's ceiling counts for neither."""
    for lg, narrowed in (({"noise": 60.0, "swept_cw": None, "chirp": None}, ["noise"]),
                         ({"noise": None, "swept_cw": 55.0, "chirp": 70.0}, ["swept_cw"]),
                         ({"noise": None, "swept_cw": None, "chirp": 70.0}, [])):  # fmt: skip
        c = e2.g3_clauses_ghost(GATES, ghost_decisions(ghost_ceiling_db_hz=lg), e2.G3_HZ_SOURCE)
        assert c["clause3"]["passed"] is False and c["clause3"]["narrowed"] == narrowed and c["passed"] is False
        assert c["clause3"]["ghost_ceiling_db_hz"] == lg and c["gates"]["passed"] and c["k"]["chosen"]


def test_one_failed_gate_fails_clause_1() -> None:
    """Review Focus 6: clause 1 is every gate the run writes, on both segments; one failed G-null channel of segment B,
    or segment A's G-warmup-lock, fails it. A run that wrote no gate does not pass it either."""
    for name in ("G-null[B][ch3]", "G-warmup-lock[A]"):
        gates = {**GATES, name: {"passed": False, "units_failed": ["A-thermal-clean-r3"]}}
        c = e2.g3_clauses_ghost(gates, ghost_decisions(), e2.G3_HZ_SOURCE)
        assert c["gates"] == {"passed": False, "count": 9, "failed": [name]} and c["passed"] is False
        assert c["clause3"]["passed"] and c["k"]["chosen"]
    c = e2.g3_clauses_ghost({}, ghost_decisions(), e2.G3_HZ_SOURCE)
    assert c["gates"] == {"passed": False, "count": 0, "failed": []} and c["passed"] is False


def test_k_not_chosen_fails_clause_4() -> None:
    c = e2.g3_clauses_ghost(GATES, ghost_decisions(k=None), e2.G3_HZ_SOURCE)
    assert c["k"] == {"k": None, "chosen": False} and c["passed"] is False
    assert c["gates"]["passed"] and c["clause3"]["passed"]


def no_slips() -> pa.Table:
    return build(EVENTS, {c: [] for c in EVENTS.names}, 0)


def ghost_realization(
    ctx: e2.Context, name: str, off: float, value: Any, drop: tuple[int, int] | None = None, chans: Any = None
) -> None:
    """One oracle realization on disk as reduce_unit keeps it under ghost truth: per ghost, 1 ms records from
    onset - pre (8 s) to the onset (13 s), then one record per 20 ms grid point, every record `off` s from the grid. The
    record nearest a grid point carries value(grid index, prn); every other record 100 cycles, so that a mispaired
    record would show. drop = (prn, grid index) is not written; `chans` limits the ghost channels written."""
    seg = ctx.segments["A"]
    t = np.r_[8.01 + 1e-3 * np.arange(4980), 13.0 + 0.02 * np.arange(350)] + off
    idx = np.rint(t / 0.02).astype(np.int64)
    nearest = np.abs(t - idx * 0.02) < 0.5e-3
    tables = {}
    for ch in seg.ghost_ch if chans is None else chans:
        prn = int(seg.channels[ch][2])
        e = np.where(nearest, value(idx, prn), 100.0)
        keep = ~(nearest & (idx == drop[1])) if drop and drop[0] == prn else np.ones(t.size, bool)
        tables[f"ghost_ch{ch}.parquet"] = truth_at(t[keep], e[keep], prn)
        tables[f"ghost_slips_ch{ch}.parquet"] = no_slips()
    write_unit(ctx.run_dir, name, tables, {})


def test_the_ghost_oracle_pairs_realizations_on_the_snapped_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review Focus 3: thermal pairs each have their own clean twin, so their ghost record times differ across
    realizations by fractions of a millisecond. Two realizations 0.3 ms apart (0.1 ms after and 0.2 ms before the grid)
    share every snapped key round(t / 0.02) 0.02; before the onset, where every 1 ms record is kept, one record per grid
    point enters (Q2 ruling); a grid point one realization has no record at lowers its n_valid to 1 instead of pairing
    another record. A unit without every ghost's truth is not a realization."""
    ctx = fake_context(tmp_path / "run", monkeypatch, GHOST)
    ghost_realization(ctx, "A-k-cond-r0", 1e-4, lambda i, p: 1e-3 * i + p)
    ghost_realization(ctx, "A-k-cond-r1", -2e-4, lambda i, p: 1e-3 * i + p + 0.01, drop=(32, 750))
    ghost_realization(ctx, "A-k-cond-r2", 0.0, lambda i, p: 0.0 * i, chans=[2])  # ghost channel 3 missing
    o = e2._ghost_oracle(ctx, "A", "k-cond", "conditional")
    assert o is not None and set(o.column("K").to_pylist()) == {2} and set(o.column("form").to_pylist()) == {
        "conditional"}  # fmt: skip
    t, prn = o.column("t_rx_s").to_numpy(), o.column("prn").to_numpy()
    idx = np.rint(t / 0.02).astype(np.int64)
    assert np.array_equal(t, np.rint(t / 0.02) * 0.02)  # every key is a grid time
    for p in (32, 31):
        assert np.array_equal(np.sort(idx[prn == p]), np.arange(401, 1000))  # 8.02 s to 19.98 s, each once
    n, mu, var = (o.column(c).to_numpy() for c in ("n_valid", "mu_cyc", "var_cyc2"))
    missing = (prn == 32) & (idx == 750)
    assert missing.sum() == 1 and n[missing][0] == 1 and np.isnan(var[missing][0])
    assert (n[~missing] == 2).all()
    np.testing.assert_allclose(mu[~missing], 1e-3 * idx[~missing] + prn[~missing] + 0.005, rtol=0, atol=1e-9)
    np.testing.assert_allclose(var[~missing], 5e-5, rtol=1e-6)
    assert e2._ghost_oracle(ctx, "A", "k-cond", "conditional", k=1) is None  # one realization is no ensemble


def test_an_oracle_unit_keeps_the_ghost_record_nearest_every_grid_point() -> None:
    """Q1 ruling: E2-v2's oracle-unit filter kept, after the onset, only records at or up to T_S / 2 after a 20 ms grid
    point, so a ghost whose records fall just before it kept none (E2K's first ghost on segment A, 0.05 to 0.12 ms
    before the grid). Under ghost truth the record nearest each grid point is kept, the nearer of two within T_S / 2;
    before the onset every record from onset - pre is kept, as before. Under route_b the selection is E2-v2's, unchanged
    for any record phase."""
    onset, pre = 55.0, 20.0
    t = np.asarray(30.0 + 1e-3 * np.arange(90_000) - 0.08e-3, dtype=np.float64)  # each 0.08 ms before the ms
    keep = e2.oracle_records(t, onset, pre, truth="ghost")
    post = t[keep & (t >= onset)]
    assert np.array_equal(np.rint(post / 0.02).astype(np.int64), np.arange(2751, 6000))  # 55.02 s to 119.98 s
    np.testing.assert_allclose(post - np.rint(post / 0.02) * 0.02, -0.08e-3, atol=1e-9)
    assert np.array_equal(keep[t < onset], (t >= onset - pre)[t < onset])
    assert not e2.oracle_records(t, onset, pre, truth="route_b")[t >= onset].any()  # the defect, kept for route_b
    for off in (-0.4e-3, -0.08e-3, 0.0, 0.07e-3, 0.3e-3, 0.5e-3):
        u = np.asarray(30.0 + 1e-3 * np.arange(90_000) + off, dtype=np.float64)
        legacy = (np.isclose(np.mod(u + 1e-9, 0.02), 0.0, atol=0.5e-3) | (u < onset)) & (u >= onset - pre)
        assert np.array_equal(e2.oracle_records(u, onset, pre, truth="route_b"), legacy)
    two = np.array([59.98 - 0.49996e-3, 59.98 + 0.49997e-3, 60.0 + 0.49997e-3, 60.02 + 0.6e-3])
    assert e2.nearest_on_grid(two).tolist() == [True, False, True, False]  # the nearer of two; none beyond T_S / 2


def reduce_ghosts(
    ctx: e2.Context, u: e2.Unit, stamps: np.ndarray, e_clean: np.ndarray, effect: np.ndarray, t: Any = None
) -> tuple[list[str], dict[str, Any]]:
    """reduce_unit on unit u of segment A, ghost channels only: per ghost i, the clean twin's track carries e_clean and
    the jammed twin's e_clean + effect + 0.25 cycle (a constant every truth must remove), at stamps + 2 i; `t` replaces
    the record times (ghost 1's 0.2 ms later). The clean twin's ghost_abs files are written as run_clean writes them."""
    seg = ctx.segments["A"]
    clean_tracks, jam_tracks = {}, {}
    for i, ch in enumerate(seg.ghost_ch):
        g, st = ghost_spec(i), stamps + np.uint64(2 * i)
        c, j = clean_track(g, st, e_clean), clean_track(g, st, e_clean + effect + 0.25)
        if t is not None:
            col = c.schema.get_field_index("t_rx_s")
            c, j = (x.set_column(col, "t_rx_s", pa.array(t + 2e-4 * i, pa.float64())) for x in (c, j))
        clean_tracks[ch], jam_tracks[ch] = c, j
    errs = e2.ghost_errors(seg, clean_tracks, converge_s=ctx.warm / 2, name=u.clean or "")
    write_unit(ctx.run_dir, u.clean or "", {e2.GHOST_ABS.format(ch): x for ch, x in errs.items()}, dict(COMMON))
    art = RunArtifacts(ctx.run_dir, "cfg", ctx.run_dir / "log", 0, {}, None, 1.0)
    ctx.unit_dir(u.name).mkdir(parents=True)
    return e2.reduce_unit(ctx, u, Twin(u.name, None, art, jam_tracks, None),
                          Twin(u.clean or "", None, art, clean_tracks, None), None)  # fmt: skip


def test_an_oracle_unit_under_ghost_truth_stores_a_ghost_record_at_every_grid_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q1 ruling, through reduce_unit: a K-study unit whose first ghost's records fall 0.08 ms before the grid (the
    second's 0.12 ms after it) stores, under ghost truth, one record of each ghost at every grid point after the onset;
    under route_b the first ghost keeps none after the onset, as E2-v2 stored it. No C-1 on an oracle unit."""
    k = np.arange(1, N_REC + 1)
    stamps, t = (10 * k).astype(np.uint64), 1e-3 * k - 0.08e-3
    ceiling = {"noise": 55.0, "swept_cw": 50.0, "chirp": None}
    for spec in (GHOST, FAKE):
        ctx = fake_context(tmp_path / str(spec.get("truth")), monkeypatch, spec)
        u = next(x for x in e2.plan_oracle(spec, "A", ceiling) if x.name == "A-k-noise-r0")
        _, rec = reduce_ghosts(ctx, u, stamps, np.zeros(k.size), np.zeros(k.size), t)
        assert "ghost_check" not in rec
        for i, ch in enumerate(ctx.segments["A"].ghost_ch):
            kept = pq.read_table(ctx.unit_dir(u.name) / f"ghost_ch{ch}.parquet").column("t_rx_s").to_numpy()
            post = np.rint(kept[kept >= 13.0] / 0.02).astype(np.int64)
            if spec is GHOST or i == 1:  # from 13.02 s (ghost 0, at 12.99992 s before the onset) or 13.00 s, to 25.00 s
                assert np.array_equal(post, np.arange(651 - i, 1251)), (spec.get("truth"), ch)
            else:
                assert post.size == 0
            assert np.sum((kept >= 8.0) & (kept < 13.0)) == 5000  # every record before the onset, from onset - pre


def test_check_c1_on_a_synthetic_jammed_track_equals_e_c_plus_dphi_up_to_a_constant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R-1, check C-1 on a mode A grid unit: ghost truth on the jammed twin's own track less (e_c + dPhi), on valid,
    non-transient, slip-free epochs from onset + 2 s, is a constant (here the jammed track's 0.25 cycle and the
    references), though the jamming adds 0.05 cycle RMS. Recorded per ghost channel; under route_b nothing."""
    k = np.arange(1, N_REC + 1)
    stamps = (10 * k).astype(np.uint64)
    t = CLOCK.file_time(stamps)
    rng = np.random.default_rng(12)
    e_clean = 0.01 * rng.standard_normal(k.size)
    effect = np.where(t >= 13.0, 0.05 * rng.standard_normal(k.size), 0.0)
    assert np.std(effect[t >= 15.0]) > 0.04
    ctx = fake_context(tmp_path / "ghost", monkeypatch, GHOST)
    u = next(x for x in e2.plan_grid(GHOST, "A") if x.name == "A-grid-noise-50-A")
    _, rec = reduce_ghosts(ctx, u, stamps, e_clean, effect)
    assert set(rec["ghost_check"]) == set(ctx.segments["A"].ghost_ch)
    for c in rec["ghost_check"].values():
        assert set(c) == {"rms_cyc", "n"} and c["rms_cyc"] < 1e-9 and c["n"] == 10_001  # 15 s to 25 s
    ctx = fake_context(tmp_path / "route_b", monkeypatch)
    _, rec = reduce_ghosts(ctx, u, stamps, e_clean, effect)
    assert "ghost_check" not in rec


def test_check_c1_records_only_a_missing_ambiguity_window_as_not_computed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Controller ruling: C-1 records {computed: false, reason} for NoAmbiguityWindow alone; any other error of ghost
    truth on the jammed track stops the unit (no broad catch)."""
    k = np.arange(1, N_REC + 1)
    stamps = (10 * k).astype(np.uint64)
    u = next(x for x in e2.plan_grid(GHOST, "A") if x.name == "A-grid-noise-50-A")
    real = e2.ghost_truth

    def failing(error: type[ValueError]) -> Any:
        def ghost_truth(track: pa.Table, ghost: GhostSpec, **kw: Any) -> pa.Table:
            if kw["truth_id"].startswith(u.name):
                raise error(f"{kw['truth_id']}: no arc holds a full ambiguity window after 1.0 s of convergence")
            return real(track, ghost, **kw)

        return ghost_truth

    monkeypatch.setattr(e2, "ghost_truth", failing(NoAmbiguityWindow))
    ctx = fake_context(tmp_path / "window", monkeypatch, GHOST)
    _, rec = reduce_ghosts(ctx, u, stamps, np.zeros(k.size), np.zeros(k.size))
    assert rec["ghost_check"] == {ch: {"computed": False, "reason": f"{u.name}-g{ch}: no arc holds a full ambiguity "
                                       f"window after 1.0 s of convergence"} for ch in (2, 3)}  # fmt: skip
    monkeypatch.setattr(e2, "ghost_truth", failing(PhasePrecisionError))
    ctx = fake_context(tmp_path / "other", monkeypatch, GHOST)
    with pytest.raises(PhasePrecisionError):
        reduce_ghosts(ctx, u, stamps, np.zeros(k.size), np.zeros(k.size))


COND4 = {"cond": {"family": "swept_cw", "K": 4, "conditional": True}}
SMALL_K = {**GHOST, "thermal": {"segment": "A", "family": "swept_cw", "K": 3}, "rules": {**FAKE["rules"], "k_small": 2},
           "kstudy": {**V2["kstudy"], "blocks": COND4}}  # fmt: skip


def oracle_units(ctx: e2.Context, rng: np.random.Generator) -> None:
    """A K = 4 conditional block and a thermal block of 3 on disk: each realization's ghosts (ghost_realization, its
    records 0 to 0.3 ms from the grid) and its real satellites' truth on the 20 ms grid, drawn independently."""
    grid = 13.0 + 0.02 * np.arange(350)
    for block, n in (("k-cond", 4), ("thermal", 3)):
        for r in range(n):
            name = f"A-{block}-r{r}"
            ghost_realization(ctx, name, 1e-4 * (r % 4), lambda i, p: rng.normal(0.0, 0.02, i.size))
            truth = pa.concat_tables([truth_at(grid, rng.normal(0.0, 0.01, grid.size), p) for p in SURVEY["sats"]])
            pq.write_table(truth, ctx.unit_dir(name) / "truth_obs.parquet")
            pq.write_table(no_slips(), ctx.unit_dir(name) / "slips_obs.parquet")


def test_under_ghost_truth_k_is_decided_on_the_ghosts_oracle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R-2, clause 4: K and the thermal comparison are read on the ghosts' oracle, written as oracle_ghost_*.parquet;
    the Route B arm's K-study (truth="route_b") reads the real satellites' truth on the same units, beside it. A
    realization that lost a ghost's truth makes realized K differ from planned K: K is not chosen."""
    ctx = fake_context(tmp_path / "run", monkeypatch, SMALL_K)
    oracle_units(ctx, np.random.default_rng(21))
    ceiling = {"noise": 55.0, "swept_cw": 50.0, "chirp": None}
    out = e2.k_study_decision(ctx, ceiling)
    small, full = (e2._ghost_oracle(ctx, "A", "k-cond", "conditional", k=k) for k in (2, None))
    assert small is not None and full is not None
    assert out["k_convergence"] == k_convergence(small, full) and out["k"] == e2.choose_k(out["k_convergence"])
    assert out["k_study"]["realized_k"] == {"cond": {"planned": 4, "realized": 4}}
    assert out["k_study"]["thermal_realized_k"] == {"planned": 3, "realized": 3}
    base, th = (e2._ghost_oracle(ctx, "A", b, "conditional", k=k) for b, k in (("k-cond", 3), ("thermal", None)))
    assert base is not None and th is not None and out["thermal_vs_fixed"] == k_convergence(base, th)
    files = {p.name for p in ctx.run_dir.glob("oracle_*.parquet")}
    assert files == {"oracle_ghost_A_k-cond.parquet", "oracle_ghost_A_thermal.parquet"}
    rb = e2.k_study_decision(ctx, ceiling, truth="route_b")
    real_small, real_full = (e2._oracle(ctx, "A", "k-cond", "conditional", k=k) for k in (2, None))
    assert real_small is not None and real_full is not None
    assert rb["k_convergence"] == k_convergence(real_small, real_full) != out["k_convergence"]
    assert {"oracle_A_k-cond.parquet", "oracle_A_thermal.parquet"} <= {p.name for p in ctx.run_dir.glob("*.parquet")}
    (ctx.unit_dir("A-k-cond-r3") / "ghost_slips_ch3.parquet").unlink()
    out = e2.k_study_decision(ctx, ceiling)
    assert out["k"] is None and out["k_study"]["realized_k"]["cond"] == {"planned": 4, "realized": 3}
    assert "3 realizations hold truth, 4 planned" in out["k_not_chosen"]


@pytest.fixture(scope="module")
def ghost_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """E2's aggregate under ghost truth on the fake run, its mode A grid units carrying a C-1 record."""
    root = tmp_path_factory.mktemp("e2-ghost")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GNSSAUDIT_RESULTS", str(root / "results"))
        ctx = build_run(root / "results" / "E2" / "ghost", mp, check=True)
        ctx.spec = GHOST
        summary, _ = e2.aggregate(ctx)
    return summary


def test_under_ghost_truth_the_ceiling_is_l_g_and_route_b_is_reported_beside_it(ghost_run: dict[str, Any]) -> None:
    """R-2, clause 3, on the fake run (Review Focus 2 at E2's level): segment B's swept CW fails B1 at 55 but keeps its
    truth, so L_G(B, swept CW) is J_max (60) where L_B is 50; segment A's noise loses truth at 60, so L_G stops at 55.
    The K-study is planned at L_K = min(80, L_G); L_B, unchanged, and the Route B arm (R-6) are reported beside it:
    L_B under s29's J_max_TA (70 for every family) by E2K's method, and the real satellites' K-study."""
    s = ghost_run
    a, b, d = s["segments"]["A"], s["segments"]["B"], s["decisions"]
    assert a["ghost_ceiling_db_hz"] == {"noise": 55.0, "swept_cw": 55.0, "chirp": None}
    assert b["ghost_ceiling_db_hz"] == {"noise": None, "swept_cw": 60.0, "chirp": None}
    assert b["ceiling_db_hz"] == {"noise": None, "swept_cw": 50.0, "chirp": None}
    assert d["ghost_ceiling_db_hz"] == {"noise": None, "swept_cw": 55.0, "chirp": None}
    assert d["ceiling_db_hz"] == {"noise": None, "swept_cw": 50.0, "chirp": None}  # L_B as under route_b
    assert d["k_level_db_hz"] == {"noise": None, "swept_cw": 55.0, "chirp": None}
    assert {tag: (blk["family"], blk["level"]) for tag, blk in d["k_study"]["blocks"].items()} == {
        "cond": ("swept_cw", 55.0), "marg": ("swept_cw", 55.0)}  # fmt: skip
    assert d["k"] is None and "0 realizations hold truth" in d["k_not_chosen"]  # no oracle unit in the fake run
    # fix round 1: the deciding ceiling is named, and the Route B arm says it is exploratory and decides nothing (R-6)
    assert (d["truth"], d["deciding_ceiling"]) == ("ghost", "ghost_ceiling_db_hz")
    assert d[d["deciding_ceiling"]] == d["ghost_ceiling_db_hz"]
    arm = d["route_b_arm"]
    assert arm["exploratory"] is True and arm["decides"] is False
    assert arm["ceiling_db_hz"] == d["ceiling_db_hz"] and arm["k_study"]["k"] is None  # L_B, inside the labelled arm
    assert arm["k_study"]["k_study"]["blocks"] == d["k_study"]["blocks"]  # the same units, the real satellites' truth
    ta = arm["j_max_ta"]
    assert (ta["s29_result"], ta["s29_sha256"]) == (GHOST["s29_result"], GHOST["s29_sha256"])
    assert "decides" not in ta and ta["s29_decides"] == {"noise": True, "swept_cw": True, "chirp": False}  # s29's flag
    assert ta["j_max_ta_db_hz"] == {"noise": 70.0, "swept_cw": 70.0, "chirp": 70.0}
    assert ta["segments"] == {"A": {"noise": 55.0, "swept_cw": 55.0, "chirp": None},
                              "B": {"noise": 60.0, "swept_cw": 50.0, "chirp": None}}  # fmt: skip
    assert ta["ceiling_db_hz"] == {"noise": 55.0, "swept_cw": 50.0, "chirp": None}
    assert a["ghost_check"]["noise"][50.0] == {"2": {"rms_cyc": 2e-4, "n": 9000},
                                               "3": {"computed": False, "reason": "no window"}}  # fmt: skip
    assert set(b["ghost_check"]) == {"noise", "swept_cw"} and set(b["ghost_check"]["noise"]) == set(LEVELS)


def test_the_route_b_arm_needs_no_s29_pin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A spec without s29 keys (the CTTC miniature) reports L_B under J_max_TA as not computed, with the reason."""
    ctx = build_run(tmp_path / "run", monkeypatch)
    ctx.spec = {**FAKE, "truth": "ghost"}
    front = e2.front_end_segments(ctx)
    arm = e2.route_b_arm(ctx, front, e2.combined(front, "ghost_ceiling_db_hz", list(FAKE["families"])))
    assert arm["j_max_ta"] == {"computed": False, "reason": "the spec pins no s29_result"}


def test_route_b_summaries_carry_none_of_ghost_truths_keys(fake_run: tuple[dict[str, Any], ...]) -> None:
    """E2-v2's behaviour is unchanged: without `truth`, no L_G, no Route B arm and no C-1 in the summary."""
    s, _ = fake_run
    assert not {"ghost_ceiling_db_hz", "route_b_arm", "truth", "deciding_ceiling"} & set(s["decisions"])
    assert not any({"ghost_ceiling_db_hz", "ghost_check"} & set(seg) for seg in s["segments"].values())
