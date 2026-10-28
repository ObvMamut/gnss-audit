"""Oracle moments over K jammed twins of one recording and configuration (docs/3 section 2.4; P2b task 11): per epoch
and satellite the mean and variance of the twin error and the slip probability, in a conditional (jammer timing fixed)
or marginal (timing drawn) form (F8); and the K check of docs/5 section 4.4."""

import numpy as np
import pyarrow as pa
import pytest

from gnssaudit.io.schemas import EVENTS, TRUTH_EPOCH_SAT, build
from gnssaudit.truth.oracle import NoCommonEpochs, k_convergence, oracle_moments

T = np.arange(0.0, 1.0, 0.02)


def truth(
    e: np.ndarray, *, prn: int = 5, valid: np.ndarray | None = None, transient: np.ndarray | None = None
) -> pa.Table:
    n = e.size
    return build(TRUTH_EPOCH_SAT, {
        "truth_id": ["r"] * n, "kind": ["twin"] * n, "sys": ["G"] * n, "prn": np.full(n, prn, np.int16),
        "signal": ["1C"] * n, "t_rx_s": T[:n], "e_cyc": e, "slip_count": np.zeros(n, np.int16),
        "arc_id": np.zeros(n, np.int32), "transient": np.zeros(n, bool) if transient is None else transient,
        "valid": np.ones(n, bool) if valid is None else valid, "regime": np.zeros(n, np.uint8)}, n)  # fmt: skip


def slips(times: list[float], prn: int = 5) -> pa.Table:
    m = len(times)
    return build(EVENTS, {
        "event_id": [f"s{i}" for i in range(m)], "source_id": ["r"] * m, "kind": ["slip"] * m, "sys": ["G"] * m,
        "prn": np.full(m, prn, np.int16), "signal": ["1C"] * m, "t_start_s": times, "t_end_s": times,
        "size_cyc": np.full(m, 0.5, np.float32), "confidence": np.ones(m, np.float32),
        "origin": ["truth"] * m}, m)  # fmt: skip


def test_identical_realizations_have_no_variance() -> None:
    e = np.sin(T)
    m = oracle_moments([truth(e)] * 4, [slips([])] * 4, config_id="c", form="conditional", slip_window_s=0.1)
    assert m.num_rows == T.size and set(m.column("K").to_pylist()) == {4}
    np.testing.assert_allclose(m.column("var_cyc2").to_numpy(), 0.0, atol=1e-15)
    np.testing.assert_allclose(m.column("mu_cyc").to_numpy(), e)
    assert set(m.column("form").to_pylist()) == {"conditional"} and set(m.column("config_id").to_pylist()) == {"c"}


def test_known_moments_come_back() -> None:
    rng = np.random.default_rng(1)
    k = 400
    draws = [0.3 + 0.1 * rng.standard_normal(T.size) for _ in range(k)]
    m = oracle_moments([truth(d) for d in draws], [slips([])] * k, config_id="c", form="marginal", slip_window_s=0.1)
    assert np.median(m.column("mu_cyc").to_numpy()) == pytest.approx(0.3, abs=0.01)
    assert np.median(m.column("var_cyc2").to_numpy()) == pytest.approx(0.01, rel=0.1)


def test_invalid_and_transient_epochs_leave_the_ensemble() -> None:
    v = np.ones(T.size, bool)
    v[3] = False
    tr = np.zeros(T.size, bool)
    tr[4] = True
    m = oracle_moments([truth(np.zeros(T.size), valid=v, transient=tr), truth(np.ones(T.size)), truth(np.ones(T.size))],
                       [slips([])] * 3, config_id="c", form="conditional", slip_window_s=0.1)  # fmt: skip
    n_valid = dict(zip(m.column("t_rx_s").to_pylist(), m.column("n_valid").to_pylist(), strict=True))
    assert n_valid[T[3]] == 2 and n_valid[T[4]] == 2 and n_valid[T[5]] == 3


def test_slip_probability_counts_realizations_slipping_within_the_window() -> None:
    sl = [slips([0.5]), slips([]), slips([0.55]), slips([0.9])]
    m = oracle_moments([truth(np.zeros(T.size))] * 4, sl, config_id="c", form="conditional", slip_window_s=0.1)
    p = dict(zip(np.round(m.column("t_rx_s").to_numpy(), 2), m.column("p_slip").to_numpy(), strict=True))
    assert p[0.46] == pytest.approx(0.5)  # [0.46, 0.56) holds 0.5 and 0.55: two of four
    assert p[0.5] == pytest.approx(0.5) and p[0.56] == 0.0 and p[0.82] == pytest.approx(0.25)
    np.testing.assert_allclose(m.column("slip_window_s").to_numpy(), 0.1, rtol=1e-6)


def test_doubling_k_is_judged_against_the_standard_error() -> None:
    rng = np.random.default_rng(2)
    small = oracle_moments([truth(rng.standard_normal(T.size)) for _ in range(51)], [slips([])] * 51,
                           config_id="c", form="conditional", slip_window_s=0.1)  # fmt: skip
    large = oracle_moments([truth(rng.standard_normal(T.size)) for _ in range(101)], [slips([])] * 101,
                           config_id="c", form="conditional", slip_window_s=0.1)  # fmt: skip
    ok = k_convergence(small, large)
    assert ok["se"] == pytest.approx(np.sqrt(2 / 50)) and ok["converged"]
    doubled = oracle_moments([truth(2 * rng.standard_normal(T.size)) for _ in range(101)], [slips([])] * 101,
                             config_id="c", form="conditional", slip_window_s=0.1)  # fmt: skip
    assert not k_convergence(small, doubled)["converged"]


def test_ensembles_without_a_common_epoch_raise_their_own_error() -> None:
    """Final review R1, I3: no common epoch is its own error (a ValueError), so that a caller can record a reported-only
    comparison as not computed without catching any other refusal."""
    rng = np.random.default_rng(3)
    a = oracle_moments([truth(rng.standard_normal(10)) for _ in range(3)], [slips([])] * 3, config_id="a",
                       form="conditional", slip_window_s=0.1)  # fmt: skip
    b = oracle_moments([truth(rng.standard_normal(20)).slice(10) for _ in range(3)], [slips([])] * 3, config_id="b",
                       form="conditional", slip_window_s=0.1)  # fmt: skip
    with pytest.raises(NoCommonEpochs, match="no epoch with a variance in both ensembles"):
        k_convergence(a, b)
    assert issubclass(NoCommonEpochs, ValueError)
