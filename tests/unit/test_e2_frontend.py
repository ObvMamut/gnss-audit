"""The front-end rules of (a') (P2b part 2, task 1; the plan's pre-registered rules): the paired excess, the jammer
slope, the block bootstrap, the cell verdict, J_max, the route B ceiling, the combination across segments, and the
exact epoch pairing they rest on."""

from typing import TypedDict

import numpy as np
import pytest

from gnssaudit.experiments import e2_frontend as fe
from gnssaudit.experiments.e2_frontend import FloatArray, FrontEndCell


class Rules(TypedDict):
    tol: float
    slope_tol: float
    se_max: float


RULES: Rules = {"tol": 0.10, "slope_tol": 0.10, "se_max": 0.05}


def f64(x: object) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


def test_the_paired_excess_cancels_what_only_reshuffles_the_clean_twins_noise() -> None:
    """e_c = t + q_c and dPhi = j + (q_j - q_c), q_j correlated 0.5 with q_c and of the same variance: the twin
    difference shows Var(j) + Var(q_j - q_c), while what jamming adds to the receiver's error is Var(j)."""
    n, var_t, var_q, var_j = 200_000, 2.0, 0.5, 1.0
    rng = np.random.default_rng(1)
    t = rng.normal(0.0, np.sqrt(var_t), n)
    q_c = rng.normal(0.0, np.sqrt(var_q), n)
    q_j = 0.5 * q_c + np.sqrt(1 - 0.25) * rng.normal(0.0, np.sqrt(var_q), n)
    j = rng.normal(0.0, np.sqrt(var_j), n)
    e_c, dphi = t + q_c, j + (q_j - q_c)
    p = fe.paired_excess(dphi, e_c)
    x, y = e_c + dphi, e_c  # excess = mean of the paired difference of squared deviations; its SE by the CLT
    z = (x - x.mean()) ** 2 - (y - y.mean()) ** 2
    se = float(z.std() / np.sqrt(n))
    assert abs(p.excess - var_j) < 3 * se
    assert p.excess == pytest.approx(p.var_dphi + p.cov2)
    assert p.var_dphi - var_j == pytest.approx(var_q, rel=0.05)  # Var(q_j - q_c) = 2 var_q - 2 (0.5 var_q) = var_q
    assert p.cov2 == pytest.approx(-var_q, rel=0.05)  # and it cancels against the clean twin's own error


def test_the_paired_excess_needs_paired_epochs() -> None:
    with pytest.raises(ValueError, match="paired"):
        fe.paired_excess(np.zeros(5), np.zeros(4))


def test_the_jammer_slope_is_the_regression_of_mode_b_on_mode_a() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(0.0, 1.0, 100_000)
    b = 0.93 * a + rng.normal(0.0, 0.5, a.size)
    assert fe.jammer_slope(a, b) == pytest.approx(0.93, abs=0.01)


def test_the_block_bootstrap_se_of_a_white_mean_is_sigma_over_root_n() -> None:
    sigma, n = 2.0, 20_000
    x = np.random.default_rng(3).normal(0.0, sigma, n)
    se = fe.block_bootstrap_se(lambda a: float(a.mean()), [x], block=50)
    assert se == pytest.approx(sigma / np.sqrt(n), rel=0.15)


def test_the_block_bootstrap_draws_the_same_blocks_for_every_array() -> None:
    """Two copies of one series: every replicate resamples both alike, so their difference never varies."""
    x = np.random.default_rng(4).normal(size=5_000)
    assert fe.block_bootstrap_se(lambda a, b: float(a.mean() - b.mean()), [x, x.copy()], block=100) == 0.0


def test_the_block_bootstrap_refuses_unpaired_or_short_arrays() -> None:
    with pytest.raises(ValueError, match="two blocks"):
        fe.block_bootstrap_se(lambda a, b: 0.0, [np.zeros(300), np.zeros(299)], block=100)
    with pytest.raises(ValueError, match="two blocks"):
        fe.block_bootstrap_se(lambda a: 0.0, [np.zeros(150)], block=100)


def cell(ratio: float, se: float, slopes: tuple[float, ...] = (1.0, 1.0)) -> FrontEndCell:
    return FrontEndCell(slopes=slopes, ratio=ratio, ratio_se=se)


def test_the_cell_verdict_follows_the_pre_registered_rule() -> None:
    assert fe.cell_verdict(cell(1.0, 0.02, (1.0, 0.85)), **RULES) == "fail"  # a ghost's |slope - 1| >= 0.10
    assert fe.cell_verdict(cell(1.0, 0.08), **RULES) == "unresolved"  # slopes pass, SE(R) > 0.05
    assert fe.cell_verdict(cell(1.12, 0.02), **RULES) == "fail"  # resolved, |R - 1| >= 0.10
    assert fe.cell_verdict(cell(0.95, 0.02), **RULES) == "pass"
    assert fe.cell_verdict(cell(1.5, 0.08, (0.85, 1.0)), **RULES) == "fail"  # a slope fails whatever the SE


def verdicts(**by_level: str) -> dict[float, FrontEndCell]:
    make = {"pass": cell(1.0, 0.02), "fail": cell(1.3, 0.02), "unresolved": cell(1.0, 0.2)}
    return {float(k.removeprefix("l")): make[v] for k, v in by_level.items()}


def test_front_end_jmax_is_the_largest_pass_with_no_fail_at_or_below_it() -> None:
    """An unresolved level below a resolved pass does not stop the scan; unresolved levels alone never pass."""
    assert fe.front_end_jmax(verdicts(l50="unresolved", l55="pass", l60="pass", l65="fail"), **RULES) == 60.0
    assert fe.front_end_jmax(verdicts(l50="unresolved", l55="unresolved", l60="unresolved"), **RULES) is None
    assert fe.front_end_jmax(verdicts(l50="fail", l55="pass", l60="pass"), **RULES) is None
    assert fe.front_end_jmax(verdicts(l50="pass", l55="unresolved", l60="pass", l65="unresolved"), **RULES) == 60.0


def test_the_route_b_ceiling_stops_at_the_first_level_that_fails_b1_or_loses_truth() -> None:
    levels = [50.0, 55.0, 60.0, 65.0]
    ok = dict.fromkeys(levels, True)
    assert fe.route_b_ceiling(levels, 65.0, {**ok, 60.0: False}, ok) == 55.0
    assert fe.route_b_ceiling(levels, 65.0, ok, {**ok, 55.0: False}) == 50.0
    assert fe.route_b_ceiling(levels, 60.0, ok, ok) == 60.0  # never above J_max
    assert fe.route_b_ceiling(levels, 65.0, {**ok, 50.0: False}, ok) is None
    assert fe.route_b_ceiling(levels, None, ok, ok) is None
    assert fe.route_b_ceiling(levels, 65.0, {50.0: True}, ok) == 50.0  # an unmeasured level is no pass


def test_the_ghost_ceiling_ignores_b1_and_stops_at_the_first_level_with_truth_lost() -> None:
    """R-2 (P2c, Review Focus 2): L_G(s, f) is the largest level up to J_max at and below which both ghosts keep their
    truth in mode A; B1 is not an argument at all, so a level that fails B1 but keeps truth cannot stop it, while a
    level that loses truth does."""
    levels = [50.0, 55.0, 60.0, 65.0]
    ok = dict.fromkeys(levels, True)
    b1_fails_at_55 = {**ok, 55.0: False}
    assert fe.route_b_ceiling(levels, 65.0, b1_fails_at_55, ok) == 50.0  # Route B stops at B1
    assert fe.ghost_ceiling(levels, 65.0, ok) == 65.0  # the ghost ceiling does not see it
    assert fe.ghost_ceiling(levels, 65.0, {**ok, 60.0: False}) == 55.0  # truth lost at 60: stops below it
    assert fe.ghost_ceiling(levels, 65.0, {**ok, 50.0: False}) is None
    assert fe.ghost_ceiling(levels, 60.0, ok) == 60.0  # never above J_max
    assert fe.ghost_ceiling(levels, None, ok) is None
    assert fe.ghost_ceiling(levels, 65.0, {50.0: True, 60.0: True}) == 50.0  # an unmeasured level is no pass


def test_combining_segments_keeps_a_none() -> None:
    assert fe.combine_segments([60.0, None]) is None
    assert fe.combine_segments([60.0, 55.0]) == 55.0
    assert fe.combine_segments([]) is None


def test_pairing_joins_on_exact_record_times_and_counts_what_it_drops() -> None:
    t = f64(30.0 + np.arange(10) * 1e-3)
    a, b = f64(np.arange(10.0)), f64(100.0 + np.arange(10.0))
    tt, (va, vb), dropped = fe.pair_on_time((t, a), (f64(t + 1e-12), b))  # float noise under 1 ns still matches
    assert dropped == 0 and np.array_equal(tt, t) and np.array_equal(va, a) and np.array_equal(vb, b)
    tt, (va, vb), dropped = fe.pair_on_time((t, a), (f64(t + 1e-3), b))  # b one record late: a different pairing
    assert dropped == 2 and tt.size == 9
    assert np.array_equal(va, a[1:]) and np.array_equal(vb, b[:-1])
    # one sample of 7.67 MS/s off: no match, and no interpolation
    tt, _, dropped = fe.pair_on_time((t, a), (f64(t + 130e-9), b))
    assert tt.size == 0 and dropped == 20


def test_pairing_sorts_by_time_and_refuses_repeated_records() -> None:
    t = np.array([0.003, 0.001, 0.002])
    tt, (v,), dropped = fe.pair_on_time((t, np.array([3.0, 1.0, 2.0])))
    assert np.array_equal(tt, np.sort(t)) and np.array_equal(v, [1.0, 2.0, 3.0]) and dropped == 0
    with pytest.raises(ValueError, match="repeat"):
        fe.pair_on_time((np.array([0.001, 0.001]), np.zeros(2)))


def test_records_of_different_channels_align_on_the_nearest_record_within_the_tolerance() -> None:
    """Two ghost channels never share stamps (each integrates over its own code period); 23.6 us apart on real E2
    data. A record the other channel has no counterpart for is dropped and counted."""
    t1 = f64(57.0 + np.arange(20) * 1e-3)
    t2 = f64(np.delete(t1 + 23.6e-6, [4, 11]))
    (i1, i2), dropped = fe.align_records([t1, t2], tol_s=0.5e-3)
    assert np.array_equal(i1, np.delete(np.arange(20), [4, 11])) and np.array_equal(i2, np.arange(18))
    assert dropped == 2


def ghost(rng: np.random.Generator, t: np.ndarray, slope: float) -> tuple[np.ndarray, ...]:
    j = rng.normal(0.0, 1.0, t.size)
    e_a, e_b = rng.normal(0.0, 0.3, t.size), rng.normal(0.0, 0.3, t.size)
    return t, j, slope * j + rng.normal(0.0, 0.2, t.size), e_a, e_b


def test_a_front_end_cell_pools_the_ghosts_and_bootstraps_on_shared_blocks() -> None:
    rng = np.random.default_rng(5)
    t = 57.0 + np.arange(6_000) * 1e-3
    g1, g2 = ghost(rng, t, 1.0), ghost(rng, np.delete(t + 23.6e-6, [10, 20, 30]), 0.95)
    c, dropped = fe.front_end_cell([g1, g2], block=1000, tol_s=0.5e-3)
    assert c.slopes[0] == pytest.approx(1.0, abs=0.02) and c.slopes[1] == pytest.approx(0.95, abs=0.02)
    excess = [(fe.paired_excess(g[1], g[3]).excess, fe.paired_excess(g[2], g[4]).excess) for g in (g1, g2)]
    assert c.ratio == pytest.approx(sum(b for _, b in excess) / sum(a for a, _ in excess))
    assert c.ratio == pytest.approx(fe.pooled_ratio([g1[1:], g2[1:]]))
    assert 0.0 < c.ratio_se < 0.1 and dropped == 3


def test_a_verdict_refuses_a_non_finite_slope_ratio_or_se() -> None:
    """A NaN slope would pass the slope test (|NaN - 1| >= 0.10 is false) and a NaN SE(R) would read as resolved:
    neither may give a verdict, nor may J_max rest on one."""
    nan = float("nan")
    for c in (cell(1.0, 0.02, (nan, 1.0)), cell(1.0, nan), cell(nan, 0.02), cell(float("inf"), 0.02)):
        with pytest.raises(ValueError, match="non-finite"):
            fe.cell_verdict(c, **RULES)
    with pytest.raises(ValueError, match="non-finite"):
        fe.front_end_jmax({50.0: cell(1.0, 0.02), 55.0: cell(1.0, 0.02, (1.0, nan))}, **RULES)


def test_aligning_records_refuses_times_out_of_order() -> None:
    """The nearest-record search assumes each epoch set in increasing time; out of order it would pair wrong records."""
    t1 = f64(57.0 + np.arange(20) * 1e-3)
    t2 = t1 + 23.6e-6
    with pytest.raises(ValueError, match="increasing"):
        fe.align_records([t1, f64(t2[::-1])], tol_s=0.5e-3)
    with pytest.raises(ValueError, match="increasing"):
        fe.align_records([f64(np.r_[t1[1:], t1[:1]]), t2], tol_s=0.5e-3)
    with pytest.raises(ValueError, match="increasing"):
        fe.align_records([t1, f64(np.r_[t2[:5], t2[4:]])], tol_s=0.5e-3)  # a repeated record


def test_a_refused_verdict_names_its_cell() -> None:
    nan = float("nan")
    with pytest.raises(ValueError, match="segment A, noise at 60 dB-Hz: .*non-finite"):
        fe.cell_verdict(cell(1.0, nan), **RULES, cell="segment A, noise at 60 dB-Hz")
    with pytest.raises(ValueError, match="segment B, swept_cw at 55 dB-Hz: .*non-finite"):
        fe.front_end_jmax({50.0: cell(1.0, 0.02), 55.0: cell(nan, 0.02)}, **RULES, cell="segment B, swept_cw")
