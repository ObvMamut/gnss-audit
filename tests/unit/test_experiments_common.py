"""Shared experiment steps: scope filter, 1 Hz decimation, Route D fit of B1, B6 innovations, coverage summary."""

import numpy as np
import pyarrow as pa
import pytest

from gnssaudit.experiments.common import (
    b6_innovations,
    decimate_1hz,
    fit_b1_route_d,
    in_scope,
    summarize_coverage,
)
from gnssaudit.io.schemas import OBS_EPOCH_SAT, TRUTH_EPOCH_SAT, build
from gnssaudit.signals import wavelength_m


def _obs(t: np.ndarray, sys: list[str], prn: np.ndarray, sig: list[str], cn0: np.ndarray) -> pa.Table:
    n = t.size
    return build(
        OBS_EPOCH_SAT,
        {
            "source_id": ["s"] * n, "receiver_id": ["r"] * n, "t_gps_s": t, "sys": sys, "prn": prn, "signal": sig,
            "pr_m": None, "cp_cyc": None, "dop_hz": None, "cn0_dbhz": cn0, "lock_time_s": None, "lli": None,
            "trk_stat": None, "cp_std_code": None, "cp_std_cyc": None, "censored": None, "pr_std_m": None,
            "elev_deg": None, "azim_deg": None, "valid": np.ones(n, bool),
        },
        n,
    )  # fmt: skip


def _truth(t: np.ndarray, sys: list[str], prn: np.ndarray, sig: list[str], e_cyc: np.ndarray) -> pa.Table:
    n = t.size
    return build(
        TRUTH_EPOCH_SAT,
        {
            "truth_id": ["tdcp"] * n, "kind": ["tdcp"] * n, "sys": sys, "prn": prn, "signal": sig, "t_rx_s": t,
            "e_cyc": e_cyc, "slip_count": np.zeros(n), "arc_id": np.zeros(n), "transient": np.zeros(n, bool),
            "valid": np.ones(n, bool), "regime": np.zeros(n),
        },
        n,
    )  # fmt: skip


def test_in_scope_keeps_mapped_gps_and_galileo_signals() -> None:
    t = np.zeros(4)
    obs = _obs(t, ["G", "E", "C", "G"], np.array([1, 2, 3, 4]), ["1C", "7Q", "u0", "u4"], np.full(4, 40.0))
    kept, dropped = in_scope(obs)
    assert kept.column("prn").to_pylist() == [1, 2] and dropped == 2


def test_decimate_keeps_whole_seconds_from_the_first_epoch() -> None:
    t = 100.59 + 0.2 * np.arange(11)  # 100.59 ... 102.59
    obs = _obs(t, ["G"] * 11, np.ones(11, int), ["1C"] * 11, np.full(11, 40.0))
    kept, dropped = decimate_1hz(obs)
    assert kept.column("t_gps_s").to_numpy() == pytest.approx([100.59, 101.59, 102.59])
    assert dropped == 8


def _route_d_world(seed: int, c_true: dict[str, float], quiet_frac: float = 0.0) -> tuple[pa.Table, pa.Table]:
    """Epochs of 20 GPS L1 and 20 Galileo E5b signals (n = 40, as on the UGE receiver); increments drawn from B1
    with c_true, then projected.

    quiet_frac of the increments carry no error while C/N0 = 25 dB-Hz is reported at both ends (a C/N0 that lags
    a recovered signal). C/N0 at one time is shared by two increments, so about 3 quiet_frac of them see an
    inflated u: a model violation that may move the fit by about that much, not by its leverage.
    """
    rng = np.random.default_rng(seed)
    n_ep, sats = 600, [("G", k, "1C") for k in range(1, 21)] + [("E", k, "7Q") for k in range(1, 21)]  # n = 40
    cn0 = rng.uniform(30.0, 50.0, size=(n_ep + 1, len(sats)))
    c = np.array([c_true[s + g[:1]] for s, _, g in sats])
    var = c * 10.0 ** (-cn0 / 10.0)
    y = np.sqrt(var[:-1] + var[1:]) * rng.standard_normal((n_ep, len(sats)))
    quiet = rng.random((n_ep, len(sats))) < quiet_frac
    y[quiet] = 0.0
    cn0[:-1][quiet] = 25.0
    cn0[1:][quiet] = 25.0
    r = y - y.mean(axis=1, keepdims=True)
    lam = np.array([wavelength_m(s, g) for s, _, g in sats])
    tt = np.repeat(np.arange(n_ep + 1, dtype=float), len(sats))
    obs = _obs(tt, [s for _ in range(n_ep + 1) for s, _, _ in sats],
               np.tile([p for _, p, _ in sats], n_ep + 1), [g for _ in range(n_ep + 1) for _, _, g in sats],
               cn0.ravel())  # fmt: skip
    truth = _truth(tt[: n_ep * len(sats)], [s for _ in range(n_ep) for s, _, _ in sats],
                   np.tile([p for _, p, _ in sats], n_ep), [g for _ in range(n_ep) for _, _, g in sats],
                   (r / lam).ravel())  # fmt: skip
    return truth, obs


def test_b1_route_d_fit_recovers_both_bands() -> None:
    c_true = {"G1": 2.0, "E7": 0.5}
    truth, obs = _route_d_world(7, c_true)
    got = fit_b1_route_d(truth, obs)
    assert got["G1"] == pytest.approx(2.0, rel=0.05)
    assert got["E7"] == pytest.approx(0.5, rel=0.05)


def test_b1_route_d_fit_is_not_dominated_by_low_cn0_rows() -> None:
    c_true = {"G1": 2.0, "E7": 0.5}
    truth, obs = _route_d_world(8, c_true, quiet_frac=0.02)  # C/N0 is shared by two increments: ~6 % affected
    got = fit_b1_route_d(truth, obs)
    assert got["G1"] == pytest.approx(2.0, rel=0.12)  # unweighted least squares on r^2 gives about half
    assert got["E7"] == pytest.approx(0.5, rel=0.12)


def test_b6_innovations_rescale_the_projected_increment_to_one_epoch() -> None:
    rng = np.random.default_rng(3)
    n_ep, n_sat, sigma_cyc = 4000, 6, 0.01
    y = np.sqrt(2.0) * sigma_cyc * rng.standard_normal((n_ep, n_sat))
    r = y - y.mean(axis=1, keepdims=True)
    t = np.repeat(np.arange(n_ep, dtype=float), n_sat)
    truth = _truth(t, ["G"] * t.size, np.tile(np.arange(1, n_sat + 1), n_ep), ["1C"] * t.size, r.ravel())
    innov = b6_innovations(truth)
    assert np.sqrt(np.mean(innov.column("innov_cyc").to_numpy() ** 2)) == pytest.approx(sigma_cyc, rel=0.02)


def test_summarize_coverage_reports_dependence_and_label() -> None:
    rng = np.random.default_rng(5)
    n_ep, n_sat = 3000, 6
    scored = pa.table(
        {
            "sys": ["G"] * (n_ep * n_sat),
            "prn": pa.array(np.tile(np.arange(1, n_sat + 1), n_ep), pa.int16()),
            "signal": ["1C"] * (n_ep * n_sat),
            "t_rx_s": np.repeat(np.arange(n_ep, dtype=float), n_sat),
            "w": rng.standard_normal(n_ep * n_sat),
        }
    )
    s = summarize_coverage(scored, levels=(0.68, 0.95, 0.99), n_boot=200, seed=1)
    assert s["n"] == n_ep * n_sat
    assert s["c0.95"] == pytest.approx(0.95, abs=0.01)
    assert s["ci95_c0.95"][0] < s["c0.95"] < s["ci95_c0.95"][1]
    assert s["tau_int_s"] == pytest.approx(1.0, abs=0.2)
    assert s["n_eff"] > 10_000 and s["label"] == "audited"
