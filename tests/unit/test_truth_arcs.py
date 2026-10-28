"""U-dd-operator and arc bookkeeping (docs/3 sections 4.1 and 4.2, docs/11 section 4)."""

import numpy as np
import pytest

from gnssaudit.truth.arcs import arc_ids, dd_operator, remove_arc_mean


@pytest.mark.parametrize(("n_sat", "ref"), [(2, 0), (5, 0), (8, 3), (12, 11)])
def test_dd_operator_rank_and_common_mode(n_sat: int, ref: int) -> None:
    d = dd_operator(n_sat, ref)
    assert d.shape == (n_sat - 1, n_sat)
    assert np.linalg.matrix_rank(d) == n_sat - 1
    assert np.allclose(d @ np.full(n_sat, 3.7), 0.0)
    x = np.arange(n_sat, dtype=float)
    assert np.allclose(d @ x, np.delete(x, ref) - x[ref])


def test_arc_ids_break_on_lli_unusable_and_gaps() -> None:
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 7.0, 8.0])
    lli = np.array([0, 0, 1, 0, 0, 0, 0], dtype=np.uint8)
    usable = np.array([True, True, True, False, True, True, True])
    assert arc_ids(t, lli, usable, max_gap_s=1.5).tolist() == [0, 0, 1, -1, 2, 3, 3]


def test_remove_arc_mean() -> None:
    v = np.array([1.0, 3.0, 10.0, 20.0, 30.0, 5.0])
    arc = np.array([0, 0, 1, 1, 1, -1])
    out, inflation = remove_arc_mean(v, arc)
    assert np.allclose(out[:5], [-1.0, 1.0, -10.0, 0.0, 10.0])
    assert np.allclose(inflation[:5], [2.0, 2.0, 1.5, 1.5, 1.5])
    assert np.isnan(out[5]) and np.isnan(inflation[5])


def test_only_the_loss_of_lock_bit_breaks_an_arc() -> None:
    # RINEX LLI: bit 0 loss of lock; bit 1 half-cycle ambiguity; bit 2 anti-spoofing / BOC tracking
    t = np.arange(4.0)
    lli = np.array([0, 4, 2, 5], dtype=np.uint8)
    assert arc_ids(t, lli, np.ones(4, dtype=bool), max_gap_s=1.5).tolist() == [0, 0, 0, 1]
