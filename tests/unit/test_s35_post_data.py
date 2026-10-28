"""S35's own logic (P2c task 9): run 2 in s33's schema, the comparisons through s34's accept, the verdict flips, the
rule's rejection and its exact values, and B1 near its bar. Synthetic mappings only; no pinned input is read."""

import copy
import math
from typing import Any

import numpy as np
import pytest
from scipy.stats import chi2

from review.s30_k_rule_null import SourceRefused
from review.s34_front_end_acceptance import accept
from review.s35_run3_post_data import (
    as_model,
    b1_values,
    compare,
    eigen_check,
    exact_reject_full_correlation,
    k_equal_tail,
    margins,
    near_bar,
    require_cells,
    rule_rejects,
    size_and_tails,
    spread,
    verdicts,
)

FAMILIES = ("noise", "swept_cw")
LEVELS = ("50.0", "55.0", "60.0", "65.0", "70.0")
HALF = 1.0 / math.sqrt(2.0)  # an SE in each source that makes z = R_x - R_y


def make_run(*, ratio: float = 1.0, se: float = HALF, verdict: str = "pass", b1: float = 0.04) -> dict[str, Any]:
    """An E2 summary's shape: front_end and residual (two ghosts, PRN 1 and 2, with r - floor = b1 and b1 - 0.01)."""
    fe = {f: {lv: {"ratio": ratio, "ratio_se": se, "verdict": verdict} for lv in LEVELS} for f in FAMILIES}
    res = {f: {lv: {"1": {"r": 0.05 + b1, "floor": 0.05}, "2": {"r": 0.04 + b1, "floor": 0.05}} for lv in LEVELS}
           for f in (*FAMILIES, "chirp")}  # fmt: skip
    return {"segments": {s: {"front_end": copy.deepcopy(fe), "residual": copy.deepcopy(res)} for s in "AB"}}


def test_as_model_carries_ratio_and_se_into_s33s_schema() -> None:
    run = make_run(ratio=0.9, se=0.03)
    m = as_model(run)
    assert m["cells"]["B"]["swept_cw"]["65.0"] == {"ratio_B": 0.9, "ratio_B_se": 0.03}
    assert sum(len(m["cells"][s][f]) for s in "AB" for f in FAMILIES) == 20
    r = compare(run, m, "A")
    assert r["chi2"] == 0.0 and r["n"] == 10 and r["within_s34_bounds"] is True


def test_compare_is_s34s_accept_with_its_tails() -> None:
    run3, run2 = make_run(), make_run()
    run3["segments"]["A"]["front_end"]["noise"]["55.0"]["ratio"] = 1.0 + 2.0  # z = +2 (combined SE 1)
    run3["segments"]["A"]["front_end"]["swept_cw"]["60.0"]["ratio"] = 1.0 - 1.0  # z = -1
    r = compare(run3, as_model(run2), "A")
    a = accept(run3, as_model(run2), "A")
    assert r["z"] == a["z"] and r["chi2"] == pytest.approx(5.0) and r["max_abs_z"] == pytest.approx(2.0)
    assert r["max_abs_z_cell"] == "noise/55.0"
    assert r["positive"] == 1  # zeros are not positive
    assert r["p_independent"] == pytest.approx(chi2.sf(5.0, 10))
    assert r["p_full_correlation"] == pytest.approx(chi2.sf(0.5, 1))


def test_compare_refuses_a_segment_with_a_missing_cell() -> None:
    run3 = make_run()
    del run3["segments"]["B"]["front_end"]["noise"]["70.0"]
    with pytest.raises(SourceRefused, match="9 cells"):
        compare(run3, as_model(make_run()), "B")


@pytest.mark.parametrize("bad", [None, float("nan"), 0.0, True])
def test_require_cells_refuses_a_cell_without_a_finite_positive_se(bad: object) -> None:
    m = as_model(make_run())
    require_cells(m, "fine")
    m["cells"]["A"]["noise"]["50.0"]["ratio_B_se"] = bad
    with pytest.raises(SourceRefused, match="A/noise/50.0"):
        require_cells(m, "bad")


def test_require_cells_refuses_a_missing_cell() -> None:
    m = as_model(make_run())
    del m["cells"]["B"]["swept_cw"]["55.0"]
    with pytest.raises(SourceRefused, match="B/swept_cw/55.0"):
        require_cells(m, "short")


def test_verdict_flips_are_listed_with_both_runs() -> None:
    run2, run3 = make_run(), make_run()
    run2["segments"]["A"]["front_end"]["noise"]["55.0"]["verdict"] = "fail"
    run3["segments"]["B"]["front_end"]["swept_cw"]["60.0"]["verdict"] = "unresolved"
    table, flips = verdicts(run2, run3)
    assert table["A"]["noise"]["55.0"] == {"run2": "fail", "run3": "pass"}
    assert [(f["segment"], f["family"], f["level"], f["run2"], f["run3"]) for f in flips] == [
        ("A", "noise", "55.0", "fail", "pass"),
        ("B", "swept_cw", "60.0", "pass", "unresolved"),
    ]


def test_rule_rejects_agrees_with_s34s_accept() -> None:
    rng = np.random.default_rng(5)
    z = rng.standard_normal((300, 10)) * 1.4
    z[0] = 0.0
    z[1] = 0.0
    z[1, 3] = 3.0  # |z| = 3 rejects (accept needs max |z| < 3), with sum z^2 = 9 under the bound
    z[2] = math.sqrt(chi2.ppf(0.95, 10) / 10) * 0.999  # sum z^2 just under the bound
    rej = rule_rejects(z)
    model = as_model(make_run(se=0.0))  # with the run's SE 1, z = R_run3 - 1 exactly
    for row, r in zip(z, rej, strict=True):
        run3 = make_run(se=1.0)
        for i, (f, lv) in enumerate((f, lv) for f in FAMILIES for lv in LEVELS):
            run3["segments"]["A"]["front_end"][f][lv]["ratio"] = 1.0 + float(row[i])
        assert bool(r) is not accept(run3, model, "A")["accepted"]
    assert not rej[0] and rej[1] and not rej[2]
    assert 0 < int(rej.sum()) < 300


def test_exact_size_at_full_correlation() -> None:
    p = exact_reject_full_correlation()
    assert p == pytest.approx(0.1760, abs=1e-4)  # 2 Phi(-sqrt(18.307 / 10))
    z = np.full((1, 10), 1.36)  # 10 * 1.36^2 = 18.5 > 18.307
    assert rule_rejects(z)[0] and not rule_rejects(np.full((1, 10), 1.35))[0]


def test_simulated_size_matches_the_exact_values() -> None:
    out = size_and_tails({"t": 64.92}, rng=np.random.default_rng(1), n_draws=100_000, rhos=(0.0, 1.0))
    assert out["1"]["p_reject"] == pytest.approx(exact_reject_full_correlation(), abs=0.006)
    assert out["0"]["p_reject"] == pytest.approx(0.061, abs=0.004)  # above 5 %: the max |z| condition adds size
    assert out["1"]["p_sum_z2_at_least"]["t"] == pytest.approx(k_equal_tail(64.92, 1), abs=0.0015)
    assert out["0"]["p_sum_z2_at_least"]["t"] < 1e-4


def test_k_equal_family_ends() -> None:
    assert k_equal_tail(42.79, 1) == pytest.approx(chi2.sf(4.279, 1))
    assert k_equal_tail(42.79, 10) == pytest.approx(chi2.sf(42.79, 10))
    assert k_equal_tail(42.79, 1) > k_equal_tail(42.79, 2) > k_equal_tail(42.79, 10)


def test_eigen_check_finds_one_eigenvalue_worst_at_a_large_threshold() -> None:
    e = eigen_check({"t": 50.0}, rng=np.random.default_rng(2), n_draws=50_000, n_dirichlet=200)["t"]
    assert e["k_equal_worst"]["k"] == 1
    assert e["two_point_worst"]["a"] == 1.0
    assert e["dirichlet_mc_worst"]["p"] < e["k_equal_worst"]["p"]


def test_b1_near_bar_spread_and_margins() -> None:
    run2, run3 = make_run(b1=0.040), make_run(b1=0.048)  # PRN 1 at b1, PRN 2 at b1 - 0.01
    run3["segments"]["A"]["residual"]["noise"]["50.0"]["1"]["r"] += 0.004  # 0.052: a fail near the bar
    b2, b3 = b1_values(run2), b1_values(run3)
    assert b3[("A", "noise", "50.0", "1")] == pytest.approx(0.052)
    near = near_bar(b2, b3)
    assert all(abs(r["run3"] - 0.05) <= 0.005 or abs(r["run2"] - 0.05) <= 0.005 for r in near)
    assert len(near) == 2 * 3 * 5 * 1  # PRN 1 in every cell (0.048, or 0.052 once); PRN 2 (0.038, 0.030) never
    assert all(abs(r["run3_margin"]) == pytest.approx(0.002) for r in near)
    one = next(r for r in near if r["segment"] == "A" and r["family"] == "noise" and r["level"] == "50.0")
    assert one["run3_passes"] is False and one["run2_passes"] is True
    sp = spread(b2, b3)
    assert sp["cells"] == 2 * 2 * 3 * 2  # noise and swept CW, 50-60 dB-Hz, both segments, both ghosts
    d = np.full(24, 0.008)
    d[list(sp["differences"]).index("A/noise/50.0/1")] = 0.012
    assert sp["sd"] == pytest.approx(float(np.std(d, ddof=1)))
    assert sp["per_draw_sd"] == pytest.approx(sp["sd"] / math.sqrt(2.0))
    assert sp["max"] == pytest.approx(0.012) and sp["min"] == pytest.approx(0.008)
    m = margins(b3)
    assert m["A"]["noise"] == {"prn": "1", "value": pytest.approx(0.052), "margin": pytest.approx(-0.002)}
