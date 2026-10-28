"""Baseline uncertainty models B0 to B6 (docs/4 section 2).

Kernels work on numpy arrays in their natural units (metres or radians); `predict` turns an obs_epoch_sat table
into a prediction table in cycles. B6 receives its innovations from the caller, so nothing here imports
gnssaudit.truth (docs/6 section 4.3). B7 (LS-VCE classes) needs Route C and is not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntFlag
from typing import Protocol

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from gnssaudit.io.schemas import OBS_EPOCH_SAT, PREDICTION, build, validate
from gnssaudit.signals import wavelength_m

FloatArray = NDArray[np.float64]


class Flag(IntFlag):
    """Reason codes of the prediction `flags` column (docs/4 section 1)."""

    SATURATION = 1
    CENSORED = 2
    ASSUMPTION = 4
    BITWIPE = 8
    CIRCULARITY = 16
    MISSING_INPUT = 32


# M2M4 mean-dB bias of GNSS-SDR's 20-prompt C/N0 estimate at T = 1 ms (review/results/s11_cn0_bias.json, F21)
M2M4_TRUE_DBHZ = np.array([25.0, 28.0, 30.0, 33.0, 35.0, 40.0, 45.0, 50.0])
M2M4_BIAS_DB = np.array([4.6953, 2.5263, 1.4272, 0.6331, 0.5587, 0.5040, 0.4617, 0.4767])


def b0_sigma_m(elev_deg: FloatArray, a_m: float = 0.003, b_m: float = 0.003) -> FloatArray:
    """B0: sigma^2 = a^2 + b^2 / sin^2(El)."""
    return np.sqrt(a_m**2 + b_m**2 / np.sin(np.radians(elev_deg)) ** 2)


def b1_sigma_m(cn0_dbhz: FloatArray, c_i_m2hz: float) -> FloatArray:
    """B1 (SIGMA-epsilon): sigma^2 = C_i 10^(-C/N0 / 10)."""
    return np.sqrt(c_i_m2hz * 10.0 ** (-cn0_dbhz / 10.0))


def fit_b1(e_m: FloatArray, cn0_dbhz: FloatArray) -> float:
    """Gaussian maximum-likelihood C_i from clean-interval errors (calibration split only)."""
    return float(np.mean(e_m**2 * 10.0 ** (cn0_dbhz / 10.0)))


def b2_sigma_m(
    elev_deg: FloatArray, cn0_dbhz: FloatArray, e_sigma_m: float = 0.005, snr_max_dbhz: float = 52.0
) -> FloatArray:
    """B2 (GNSS-SDR SNR penalty): sigma^2 = B0 + e_sigma^2 10^(0.1 max(SNR_max - C/N0, 0))."""
    penalty = e_sigma_m**2 * 10.0 ** (0.1 * np.maximum(snr_max_dbhz - cn0_dbhz, 0.0))
    return np.sqrt(b0_sigma_m(elev_deg) ** 2 + penalty)


def b3_sigma_rad(cn0_dbhz: FloatArray, bn_eff_hz: float, t_s: float) -> FloatArray:
    """B3: thermal PLL jitter, sigma^2 = (B_n,eff / (C/N0)) (1 + 1 / (2 T C/N0)) rad^2, on the effective bandwidth."""
    cn0 = 10.0 ** (cn0_dbhz / 10.0)
    return np.sqrt(bn_eff_hz / cn0 * (1.0 + 1.0 / (2.0 * t_s * cn0)))


def m2m4_true_cn0_dbhz(reported_dbhz: FloatArray) -> FloatArray:
    """Invert reported = true + bias(true) on the s11 grid. NaN below the grid, where the bias is unknown and
    clamping would raise C/N0; clamped above it, where the bias is flat (about 0.47 dB)."""
    return np.interp(reported_dbhz, M2M4_TRUE_DBHZ + M2M4_BIAS_DB, M2M4_TRUE_DBHZ, left=np.nan)


def b3c_sigma_rad(reported_dbhz: FloatArray, bn_eff_hz: float, t_s: float) -> FloatArray:
    """B3c: B3 on the bias-corrected C/N0. The bias table holds for T = 1 ms only."""
    if not np.isclose(t_s, 1e-3):
        raise ValueError("the M2M4 bias table was computed for T = 1 ms")
    return b3_sigma_rad(m2m4_true_cn0_dbhz(reported_dbhz), bn_eff_hz, t_s)


def b4_sigma_cyc(code: NDArray[np.uint8]) -> tuple[FloatArray, NDArray[np.bool_], NDArray[np.bool_]]:
    """B4: u-blox cpStdev. Returns sigma (code 14 at its bound 0.056 cycles, code 15 NaN), censored, invalid."""
    code = np.asarray(code)
    invalid = code == 15
    return np.where(invalid, np.nan, 0.004 * code), code == 14, invalid


def b5_sigma_rad(disc_var_rad2: FloatArray, bn_eff_hz: float, t_s: float) -> FloatArray:
    """B5: white-noise reading of the windowed discriminator variance, 2 B T / (1 + 2 B T) Var_W(d)."""
    g = 2.0 * bn_eff_hz * t_s
    return np.sqrt(g / (1.0 + g) * disc_var_rad2)


class UncertaintyModel(Protocol):
    model_id: str
    model_version: str

    def predict(self, obs: pa.Table) -> pa.Table: ...


def _wavelengths(obs: pa.Table) -> FloatArray:
    pairs = zip(obs.column("sys").to_pylist(), obs.column("signal").to_pylist(), strict=True)
    cache: dict[tuple[str, str], float] = {}
    return np.array([cache.setdefault(p, wavelength_m(*p)) for p in pairs])


def prediction_table(
    obs: pa.Table, model_id: str, model_version: str, scale_cyc: FloatArray, flags: NDArray[np.uint32]
) -> pa.Table:
    """prediction rows keyed like obs (run_id = source_id, t_rx_s = t_gps_s); valid where the scale is finite."""
    n = obs.num_rows
    return build(
        PREDICTION,
        {
            "model_id": [model_id] * n,
            "model_version": [model_version] * n,
            "run_id": obs.column("source_id"),
            "sys": obs.column("sys"),
            "prn": obs.column("prn"),
            "signal": obs.column("signal"),
            "t_rx_s": obs.column("t_gps_s"),
            "scale_cyc": scale_cyc,
            "nu": np.zeros(n, dtype=np.float32),
            "p_slip": None,
            "valid": np.isfinite(scale_cyc),
            "flags": flags,
        },
        n,
    )


def _numeric(obs: pa.Table, name: str) -> FloatArray:
    return obs.column(name).to_numpy(zero_copy_only=False).astype(float)


def _missing(*arrays: FloatArray) -> NDArray[np.uint32]:
    bad = np.zeros(arrays[0].size, dtype=bool)
    for a in arrays:
        bad |= ~np.isfinite(a)
    return np.where(bad, np.uint32(Flag.MISSING_INPUT), np.uint32(0)).astype(np.uint32)


@dataclass(frozen=True)
class B0:
    a_m: float = 0.003
    b_m: float = 0.003
    model_id: str = "B0"
    model_version: str = "1"

    def predict(self, obs: pa.Table) -> pa.Table:
        obs = validate(obs, OBS_EPOCH_SAT)
        elev = _numeric(obs, "elev_deg")
        return prediction_table(obs, self.model_id, self.model_version,
                                b0_sigma_m(elev, self.a_m, self.b_m) / _wavelengths(obs), _missing(elev))  # fmt: skip


@dataclass(frozen=True)
class B1:
    c_i_m2hz: dict[str, float] = field(default_factory=dict)  # keyed by "<sys><band>", for example "G1"
    model_id: str = "B1"
    model_version: str = "1"

    def predict(self, obs: pa.Table) -> pa.Table:
        obs = validate(obs, OBS_EPOCH_SAT)
        cn0 = _numeric(obs, "cn0_dbhz")
        keys = [s + g[:1] for s, g in zip(obs.column("sys").to_pylist(), obs.column("signal").to_pylist(), strict=True)]
        c_i = np.array([self.c_i_m2hz.get(k, np.nan) for k in keys])
        sigma = np.sqrt(c_i * 10.0 ** (-cn0 / 10.0))
        return prediction_table(obs, self.model_id, self.model_version, sigma / _wavelengths(obs), _missing(cn0, c_i))


@dataclass(frozen=True)
class B2:
    e_sigma_m: float = 0.005
    snr_max_dbhz: float = 52.0
    model_id: str = "B2"
    model_version: str = "1"

    def predict(self, obs: pa.Table) -> pa.Table:
        obs = validate(obs, OBS_EPOCH_SAT)
        elev, cn0 = _numeric(obs, "elev_deg"), _numeric(obs, "cn0_dbhz")
        sigma = b2_sigma_m(elev, cn0, self.e_sigma_m, self.snr_max_dbhz)
        return prediction_table(obs, self.model_id, self.model_version, sigma / _wavelengths(obs), _missing(elev, cn0))


@dataclass(frozen=True)
class B3:
    bn_eff_hz: float
    t_s: float
    bias_corrected: bool = False
    model_version: str = "1"

    @property
    def model_id(self) -> str:
        return "B3c" if self.bias_corrected else "B3"

    def predict(self, obs: pa.Table) -> pa.Table:
        obs = validate(obs, OBS_EPOCH_SAT)
        cn0 = _numeric(obs, "cn0_dbhz")
        kernel = b3c_sigma_rad if self.bias_corrected else b3_sigma_rad
        sigma_cyc = kernel(cn0, self.bn_eff_hz, self.t_s) / (2.0 * np.pi)
        flags = _missing(cn0)
        outside = np.isfinite(cn0) & ~np.isfinite(sigma_cyc)
        flags = np.where(outside, np.uint32(Flag.ASSUMPTION), flags).astype(np.uint32)
        return prediction_table(obs, self.model_id, self.model_version, sigma_cyc, flags)


@dataclass(frozen=True)
class B4:
    model_id: str = "B4"
    model_version: str = "1"

    def predict(self, obs: pa.Table) -> pa.Table:
        obs = validate(obs, OBS_EPOCH_SAT)
        code = obs.column("cp_std_code").fill_null(15).to_numpy()
        sigma, censored, invalid = b4_sigma_cyc(code)
        flags = np.where(censored, np.uint32(Flag.CENSORED | Flag.SATURATION), np.uint32(0))
        flags = np.where(invalid, np.uint32(Flag.MISSING_INPUT), flags).astype(np.uint32)
        return prediction_table(obs, self.model_id, self.model_version, sigma, flags)


def b6_sigma(
    t_pred: FloatArray, t_innov: FloatArray, innov: FloatArray, window_s: float, lag_s: float, min_count: int
) -> FloatArray:
    """B6 (residual-adaptive, Mehra-style [27]): RMS of the innovations finished by each prediction time.

    An innovation tagged ti covers [ti, ti + lag_s], so the prediction at t uses ti in (t - lag_s - window_s,
    t - lag_s]; it never sees its own epoch. NaN innovations are skipped; fewer than min_count finite ones give NaN.
    t_innov must be sorted.
    """
    ok = np.isfinite(innov)
    ti, sq = t_innov[ok], innov[ok] ** 2
    csum = np.concatenate([[0.0], np.cumsum(sq)])
    hi = np.searchsorted(ti, t_pred - lag_s, side="right")
    lo = np.searchsorted(ti, t_pred - lag_s - window_s, side="right")
    n = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma = np.sqrt((csum[hi] - csum[lo]) / n)
    return np.where(n >= min_count, sigma, np.nan)


@dataclass(frozen=True, eq=False)
class B6:
    """B6 on supplied innovations: a table of sys, prn, signal, t_rx_s and innov_cyc, an error proxy per epoch.

    On real data the caller derives them from Route D residuals, so B6 absorbs any fault that the residuals see
    (the trap of docs/1 section 4.3): it is evaluated, never used as truth (docs/4 section 2).
    """

    innov: pa.Table
    window_s: float = 30.0
    lag_s: float = 1.0
    min_count: int = 15
    model_id: str = "B6"
    model_version: str = "1"

    def predict(self, obs: pa.Table) -> pa.Table:
        obs = validate(obs, OBS_EPOCH_SAT)
        series: dict[tuple[str, int, str], tuple[FloatArray, FloatArray]] = {}
        keys = zip(self.innov.column("sys").to_pylist(), self.innov.column("prn").to_pylist(),
                   self.innov.column("signal").to_pylist(), strict=True)  # fmt: skip
        t_all = self.innov.column("t_rx_s").to_numpy()
        v_all = self.innov.column("innov_cyc").to_numpy(zero_copy_only=False).astype(float)
        rows: dict[tuple[str, int, str], list[int]] = {}
        for i, k in enumerate(keys):
            rows.setdefault(k, []).append(i)
        for k, idx in rows.items():
            order = np.array(idx)[np.argsort(t_all[idx], kind="stable")]
            series[k] = (t_all[order], v_all[order])
        t = obs.column("t_gps_s").to_numpy()
        scale = np.full(obs.num_rows, np.nan)
        obs_rows: dict[tuple[str, int, str], list[int]] = {}
        okeys = zip(obs.column("sys").to_pylist(), obs.column("prn").to_pylist(), obs.column("signal").to_pylist(),
                    strict=True)  # fmt: skip
        for i, k in enumerate(okeys):
            obs_rows.setdefault(k, []).append(i)
        for k, idx in obs_rows.items():
            if k in series:
                ti, vi = series[k]
                scale[idx] = b6_sigma(t[idx], ti, vi, self.window_s, self.lag_s, self.min_count)
        flags = np.where(np.isfinite(scale), np.uint32(0), np.uint32(Flag.MISSING_INPUT)).astype(np.uint32)
        return prediction_table(obs, self.model_id, self.model_version, scale, flags)
