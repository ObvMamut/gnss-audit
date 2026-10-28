"""Baselines B0 to B5 against their defining formulas (docs/4 section 2)."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from gnssaudit.io.schemas import OBS_EPOCH_SAT, PREDICTION, build, validate
from gnssaudit.signals import wavelength_m
from gnssaudit.uq.baselines import (
    B0,
    B1,
    B2,
    B3,
    B4,
    B6,
    M2M4_BIAS_DB,
    M2M4_TRUE_DBHZ,
    Flag,
    b0_sigma_m,
    b2_sigma_m,
    b3_sigma_rad,
    b4_sigma_cyc,
    b5_sigma_rad,
    b6_sigma,
    fit_b1,
    m2m4_true_cn0_dbhz,
)

REPO = Path(__file__).resolve().parents[2]


def obs_table(elev: list[float | None], cn0: list[float | None], code: list[int | None]) -> pa.Table:
    n = len(elev)
    return build(
        OBS_EPOCH_SAT,
        {
            "source_id": ["s"] * n,
            "receiver_id": ["r"] * n,
            "t_gps_s": np.arange(n, dtype=float),
            "sys": ["G"] * (n - 1) + ["E"],
            "prn": np.arange(1, n + 1),
            "signal": ["1C"] * (n - 1) + ["5Q"],
            "pr_m": None,
            "cp_cyc": None,
            "dop_hz": None,
            "cn0_dbhz": cn0,
            "lock_time_s": None,
            "lli": None,
            "trk_stat": None,
            "cp_std_code": code,
            "cp_std_cyc": None,
            "censored": None,
            "pr_std_m": None,
            "elev_deg": elev,
            "azim_deg": None,
            "valid": np.ones(n, dtype=bool),
        },  # fmt: skip
        n,
    )


def test_b0_b2_values() -> None:
    assert b0_sigma_m(np.array([90.0]))[0] == pytest.approx(np.sqrt(2) * 0.003)
    assert b0_sigma_m(np.array([30.0]))[0] == pytest.approx(np.sqrt(0.003**2 + 0.006**2))
    # at or above SNR_max the penalty is e_sigma^2
    assert b2_sigma_m(np.array([90.0]), np.array([55.0]))[0] == pytest.approx(np.sqrt(2 * 0.003**2 + 0.005**2))
    assert b2_sigma_m(np.array([90.0]), np.array([42.0]))[0] == pytest.approx(np.sqrt(2 * 0.003**2 + 0.005**2 * 10))


def test_b1_fit_recovers_c_i() -> None:
    rng = np.random.default_rng(0)
    cn0 = rng.uniform(30, 50, 200_000)
    c_i = 2.5e-1
    e = rng.standard_normal(cn0.size) * np.sqrt(c_i * 10 ** (-cn0 / 10))
    assert fit_b1(e, cn0) == pytest.approx(c_i, rel=0.01)


def test_b3_matches_analysis_worked_example() -> None:
    # docs/1 section 2.4 and KA-analysis: 1.02 / 5.80 / 10.58 deg at 45 / 30 / 25 dB-Hz, B_n = 10 Hz, T = 20 ms
    sigma = np.degrees(b3_sigma_rad(np.array([45.0, 30.0, 25.0]), 10.0, 20e-3))
    assert sigma == pytest.approx([1.02, 5.80, 10.58], rel=0.005)


def test_m2m4_table_matches_review_result_and_inverts() -> None:
    rows = json.loads((REPO / "review/results/s11_cn0_bias.json").read_text())["rows"]
    assert [r["cn0_dbhz"] for r in rows] == M2M4_TRUE_DBHZ.tolist()
    assert np.allclose([r["iid_mean_db_bias"] for r in rows], M2M4_BIAS_DB, atol=1e-4)
    reported = M2M4_TRUE_DBHZ + M2M4_BIAS_DB
    assert m2m4_true_cn0_dbhz(reported) == pytest.approx(M2M4_TRUE_DBHZ)


def test_b4_codes() -> None:
    sigma, censored, invalid = b4_sigma_cyc(np.array([0, 3, 13, 14, 15], dtype=np.uint8))
    assert sigma[:4] == pytest.approx([0.0, 0.012, 0.052, 0.056])
    assert np.isnan(sigma[4])
    assert censored.tolist() == [False, False, False, True, False]
    assert invalid.tolist() == [False, False, False, False, True]


def test_b5_white_noise_factor() -> None:
    assert b5_sigma_rad(np.array([1.0]), 5.0, 1e-3)[0] ** 2 == pytest.approx(0.01 / 1.01)


def test_model_tables() -> None:
    obs = obs_table([90.0, None, 30.0], [45.0, 40.0, 35.0], [3, 14, None])
    for model in (B0(), B1({"G1": 0.1, "E5": 0.2}), B2(), B3(bn_eff_hz=30.3, t_s=1e-3), B4()):
        pred = model.predict(obs)
        assert validate(pred, PREDICTION).num_rows == 3
        assert set(pred.column("model_id").to_pylist()) == {model.model_id}
    b0 = B0().predict(obs).to_pylist()
    assert b0[0]["scale_cyc"] == pytest.approx(np.sqrt(2) * 0.003 / wavelength_m("G", "1C"))
    assert b0[2]["scale_cyc"] == pytest.approx(np.sqrt(0.003**2 + 0.006**2) / wavelength_m("E", "5Q"))
    assert b0[1]["valid"] is False and b0[1]["flags"] == Flag.MISSING_INPUT
    b1 = B1({"G1": 0.1}).predict(obs).to_pylist()
    assert b1[2]["valid"] is False  # no C_i for Galileo E5
    b4 = B4().predict(obs).to_pylist()
    assert b4[1]["scale_cyc"] == pytest.approx(0.056) and b4[1]["flags"] == Flag.CENSORED | Flag.SATURATION
    assert b4[2]["valid"] is False
    assert B3(bn_eff_hz=30.3, t_s=1e-3, bias_corrected=True).model_id == "B3c"


def test_b3c_refuses_reported_cn0_below_its_table() -> None:
    # below reported 25 + 4.70 dB-Hz the inversion would clamp and raise C/N0 (review finding 8)
    assert np.isnan(m2m4_true_cn0_dbhz(np.array([20.0, 29.0]))).all()
    obs = obs_table([90.0, 90.0, 90.0], [20.0, 45.0, 40.0], [None, None, None])
    rows = B3(bn_eff_hz=30.3, t_s=1e-3, bias_corrected=True).predict(obs).to_pylist()
    assert rows[0]["valid"] is False and rows[0]["flags"] == Flag.ASSUMPTION
    assert rows[1]["valid"] is True and rows[1]["flags"] == 0


def test_b6_is_causal_and_lagged() -> None:
    t_innov = np.arange(10, dtype=float)
    innov = np.zeros(10)
    innov[5] = 100.0  # the increment that starts at t = 5 ends at t = 6
    t_pred = np.array([5.5, 6.0, 6.5])
    sigma = b6_sigma(t_pred, t_innov, innov, window_s=10.0, lag_s=1.0, min_count=1)
    assert sigma[0] == 0.0  # t = 5.5: the increment at 5 is not finished
    assert sigma[1] > 0.0 and sigma[2] > 0.0  # from t = 6 on it is known


def test_b6_rms_over_the_window_skipping_nan() -> None:
    t_innov = np.arange(6, dtype=float)
    innov = np.array([3.0, 4.0, np.nan, 0.0, 0.0, 0.0])
    sigma = b6_sigma(np.array([3.0]), t_innov, innov, window_s=3.0, lag_s=1.0, min_count=2)
    assert sigma[0] == pytest.approx(np.sqrt((9.0 + 16.0) / 2))  # increments at t = 0, 1 (2 is NaN)


def test_b6_nan_until_enough_samples() -> None:
    t_innov = np.arange(5, dtype=float)
    sigma = b6_sigma(np.array([1.0, 2.0, 4.0]), t_innov, np.ones(5), window_s=30.0, lag_s=1.0, min_count=3)
    assert np.isnan(sigma[0]) and np.isnan(sigma[1]) and sigma[2] == pytest.approx(1.0)


def test_b6_model_keys_innovations_by_series() -> None:
    obs = obs_table([45.0] * 4, [45.0] * 4, [1] * 4)  # rows: G1..G3 at t = 0..2, E4 at t = 3
    innov = pa.table(
        {
            "sys": ["G", "G", "E"],
            "prn": pa.array([1, 1, 4], pa.int16()),
            "signal": ["1C", "1C", "5Q"],
            "t_rx_s": [-3.0, -2.0, 1.0],
            "innov_cyc": [0.01, 0.03, 0.02],
        }
    )
    pred = B6(innov=innov, window_s=10.0, lag_s=1.0, min_count=1).predict(obs)
    validate(pred, PREDICTION)
    scale = pred.column("scale_cyc").to_pylist()
    assert scale[0] == pytest.approx(np.sqrt((0.01**2 + 0.03**2) / 2))
    assert scale[1] is None and scale[2] is None  # G2, G3 have no innovations
    assert scale[3] == pytest.approx(0.02)
    assert pred.column("model_id")[0].as_py() == "B6"
