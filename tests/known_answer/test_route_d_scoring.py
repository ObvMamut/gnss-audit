"""KA-route-d-score: Route D residuals standardised under a model's predictions (docs/3 section 4.5, docs/5 3.1).

Synthetic truth: per-epoch increments with variance sigma^2(t) + sigma^2(t + 1), a common clock increment, and the
model-free projection r = y - mean(y). Under the true sigma the w statistics are N(0, 1).
"""

import numpy as np
import pyarrow as pa
import pytest
from scipy import stats

from gnssaudit.eval.route_d import coverage_table, score_route_d
from gnssaudit.io.schemas import PREDICTION, TRUTH_EPOCH_SAT, build
from gnssaudit.signals import wavelength_m

LAM = wavelength_m("G", "1C")
N_EPOCH, N_SAT = 4000, 7


def _truth_and_sigma(seed: int) -> tuple[pa.Table, np.ndarray]:
    rng = np.random.default_rng(seed)
    sigma_m = rng.uniform(0.001, 0.006, size=(N_EPOCH + 1, N_SAT))  # sigma(t) per satellite, metres
    sd_inc = np.sqrt(sigma_m[:-1] ** 2 + sigma_m[1:] ** 2)
    y = sd_inc * rng.standard_normal((N_EPOCH, N_SAT)) + rng.normal(0, 0.3, size=(N_EPOCH, 1))  # + clock
    r = y - y.mean(axis=1, keepdims=True)
    t = np.repeat(np.arange(N_EPOCH, dtype=float), N_SAT)
    prn = np.tile(np.arange(1, N_SAT + 1), N_EPOCH)
    n = t.size
    truth = build(
        TRUTH_EPOCH_SAT,
        {
            "truth_id": ["tdcp"] * n, "kind": ["tdcp"] * n, "sys": ["G"] * n, "prn": prn, "signal": ["1C"] * n,
            "t_rx_s": t, "e_cyc": r.ravel() / LAM, "slip_count": np.zeros(n), "arc_id": np.zeros(n),
            "transient": np.zeros(n, bool), "valid": np.ones(n, bool), "regime": np.zeros(n),
        },
        n,
    )  # fmt: skip
    return truth, sigma_m


def _pred(sigma_m: np.ndarray, factor: float = 1.0, drop: tuple[int, int] | None = None) -> pa.Table:
    t = np.repeat(np.arange(N_EPOCH + 1, dtype=float), N_SAT)
    prn = np.tile(np.arange(1, N_SAT + 1), N_EPOCH + 1)
    scale = factor * sigma_m.ravel() / LAM
    if drop is not None:
        scale[drop[0] * N_SAT + drop[1] - 1] = np.nan
    n = t.size
    return build(
        PREDICTION,
        {
            "model_id": ["M"] * n, "model_version": ["1"] * n, "run_id": ["r"] * n, "sys": ["G"] * n,
            "prn": prn, "signal": ["1C"] * n, "t_rx_s": t, "scale_cyc": scale, "nu": np.zeros(n),
            "p_slip": None, "valid": np.isfinite(scale), "flags": np.zeros(n),
        },
        n,
    )  # fmt: skip


def _mc_se(p: float, n: int) -> float:
    return float(np.sqrt(p * (1 - p) / n))


def test_true_sigma_gives_nominal_coverage() -> None:
    truth, sigma = _truth_and_sigma(1)
    scored, counts = score_route_d(truth, _pred(sigma))
    assert scored.num_rows == N_EPOCH * N_SAT and counts == {}
    w = scored.column("w").to_numpy()
    # w is marginally N(0, 1), but the projection correlates the 7 values of an epoch: allow 4 SE of an epoch count
    for p in (0.68, 0.95, 0.99):
        assert np.mean(np.abs(w) <= stats.norm.ppf((1 + p) / 2)) == pytest.approx(p, abs=4 * _mc_se(p, N_EPOCH))
    assert np.sqrt(np.mean(w**2)) == pytest.approx(1.0, abs=0.02)


def test_halved_sigma_collapses_coverage_to_closed_form() -> None:
    truth, sigma = _truth_and_sigma(2)
    scored, _ = score_route_d(truth, _pred(sigma, factor=0.5))
    c95 = coverage_table(scored, levels=(0.95,), by=())[0]["c0.95"]
    assert c95 == pytest.approx(2 * stats.norm.cdf(1.959964 / 2) - 1, abs=4 * _mc_se(0.67, N_EPOCH))


def test_member_without_prediction_is_recentred_out() -> None:
    truth, sigma = _truth_and_sigma(3)
    scored, counts = score_route_d(truth, _pred(sigma, drop=(10, 3)))  # PRN 3 has no prediction at t = 10
    assert counts == {"no_prediction": 2}  # the increments starting at t = 9 and at t = 10
    at9 = scored.filter(scored.column("t_rx_s").to_numpy() == 9.0)
    assert at9.num_rows == N_SAT - 1 and 3 not in at9.column("prn").to_pylist()
    assert np.sqrt(np.mean(scored.column("w").to_numpy() ** 2)) == pytest.approx(1.0, abs=0.02)


def test_epoch_with_too_few_predicted_satellites_is_not_scored() -> None:
    truth, sigma = _truth_and_sigma(4)
    scored, counts = score_route_d(truth, _pred(sigma), min_sats=8)
    assert scored.num_rows == 0 and counts == {"too_few_predicted": N_EPOCH * N_SAT}


def test_coverage_table_groups_by_column() -> None:
    truth, sigma = _truth_and_sigma(5)
    scored, _ = score_route_d(truth, _pred(sigma))
    rows = coverage_table(scored, levels=(0.68, 0.95), by=("prn",))
    assert [r["prn"] for r in rows] == list(range(1, N_SAT + 1))
    assert all(r["n"] == N_EPOCH for r in rows)
    assert all(0.9 < r["c0.95"] < 1.0 for r in rows)
