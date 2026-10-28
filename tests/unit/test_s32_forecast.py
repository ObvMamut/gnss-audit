"""S32's own logic (P2c task 1): rule (a') without slopes, the J_max events on drawn cells, and the thermal block's
chance of a clean re-run. Synthetic arrays only; the study's pinned inputs are not read here."""

import numpy as np
import pytest

from review.s32_g3_rerun_forecast import (
    cell_verdicts,
    p_jmax_at_least,
    p_jmax_not_none,
    p_thermal_clean,
)

ONES = np.ones(5)
SE = np.full(5, 0.01)


def test_rule_a_prime_without_slopes() -> None:
    r = np.array([[1.0, 0.89, 1.11, 0.91, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]])
    se = np.array([0.01, 0.01, 0.01, 0.01, 0.06])
    v = cell_verdicts(r, se)
    assert v[0].tolist() == ["p", "f", "f", "p", "u"]  # |R - 1| >= 0.10 fails; SE above 0.05 is unresolved
    assert v[1].tolist() == ["p", "p", "p", "p", "u"]


def test_the_boundary_cells_follow_cell_verdict() -> None:
    v = cell_verdicts(np.array([1.1, 1.0]), np.array([0.01, 0.05]))
    assert v.tolist() == [
        "f",
        "p",
    ]  # |R - 1| = 0.1 fails (a pass needs < 0.1); SE = 0.05 is resolved (unresolved is SE > 0.05)


def test_an_unresolved_cell_is_not_a_failure() -> None:
    assert cell_verdicts(np.array([0.5]), np.array([0.06])).tolist() == ["u"]


@pytest.mark.parametrize("i", range(5))
def test_cells_at_one_always_reach_every_level(i: int) -> None:
    p = p_jmax_at_least(ONES, SE, SE, i, rng=np.random.default_rng(1), n=2000)
    assert p == 1.0


def test_a_55_cell_at_080_blocks_every_level_from_55_up() -> None:
    mu = np.array([1.0, 0.80, 1.0, 1.0, 1.0])
    for i in (1, 2, 3, 4):
        assert p_jmax_at_least(mu, SE, SE, i, rng=np.random.default_rng(2), n=20_000) < 1e-6
    assert p_jmax_at_least(mu, SE, SE, 0, rng=np.random.default_rng(2), n=2000) == 1.0


def test_not_none_needs_a_pass_with_no_failure_at_or_below() -> None:
    se_unres_first = np.array([0.06, 0.01, 0.01, 0.01, 0.01])
    assert p_jmax_not_none(ONES, se_unres_first, se_unres_first, rng=np.random.default_rng(3), n=500) == 1.0
    all_unres = np.full(5, 0.06)
    assert p_jmax_not_none(ONES, all_unres, all_unres, rng=np.random.default_rng(3), n=500) == 0.0
    fail_first = np.array([0.5, 1.0, 1.0, 1.0, 1.0])  # a fail at the lowest level blocks everything above it
    assert p_jmax_not_none(fail_first, SE, SE, rng=np.random.default_rng(3), n=500) == 0.0


def test_p_thermal_clean_is_the_plug_in_and_the_predictive() -> None:
    plug_in, predictive = p_thermal_clean()
    assert plug_in == pytest.approx(0.000709, abs=1e-6)
    assert predictive == pytest.approx(0.003529, abs=1e-6)


def test_p_thermal_clean_with_no_failures_is_one_for_the_plug_in() -> None:
    plug_in, predictive = p_thermal_clean(failed=0, units=10)
    assert plug_in == 1.0
    assert predictive == pytest.approx(11.0 / 21.0, rel=1e-9)  # B(1, 21) / B(1, 11) = (1 / 21) / (1 / 11)
