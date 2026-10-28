"""Steps shared by the real-data experiments E0(a) and E1(a) (docs/10 sections 2.1 and 2.2)."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray
from scipy import optimize, stats

from gnssaudit.eval.bootstrap import Hierarchy, bootstrap_ci
from gnssaudit.eval.dependence import audit_label, tau_int_s
from gnssaudit.io.schemas import OBS_EPOCH_SAT, TRUTH_EPOCH_SAT, validate
from gnssaudit.signals import wavelength_m

FloatArray = NDArray[np.float64]
SYSTEMS = ("G", "E")  # D-11
IRLS_ITERATIONS = 8


def in_scope(obs: pa.Table) -> tuple[pa.Table, int]:
    """GPS and Galileo rows with a mapped RINEX signal code; the count of the other rows (D-11)."""
    obs = validate(obs, OBS_EPOCH_SAT)
    sys = np.array(obs.column("sys").to_pylist())
    sig = obs.column("signal").to_pylist()
    keep = np.isin(sys, SYSTEMS) & np.array([not s.startswith("u") for s in sig], dtype=bool)
    return obs.filter(keep), int((~keep).sum())


def decimate_1hz(obs: pa.Table, tol_s: float = 0.05) -> tuple[pa.Table, int]:
    """Epochs a whole number of seconds after the first one, so Route D increments do not overlap (T_s = 1 s)."""
    t = obs.column("t_gps_s").to_numpy()
    frac = (t - t.min()) % 1.0
    keep = (frac < tol_s) | (frac > 1.0 - tol_s)
    return obs.filter(keep), int((~keep).sum())


def slice_time(table: pa.Table, column: str, t0: float, t1: float) -> pa.Table:
    t = table.column(column).to_numpy()
    return table.filter((t >= t0) & (t <= t1))


def _key(sys: str, prn: int, sig: str, t: float) -> tuple[str, int, str, float]:
    return sys, prn, sig, round(t, 3)


def band(sys: str, signal: str) -> str:
    """B1's key: system letter and band digit, for example "G1" or "E7"."""
    return sys + signal[:1]


def _epochs(truth: pa.Table) -> list[NDArray[np.int64]]:
    t = truth.column("t_rx_s").to_numpy()
    order = np.argsort(t, kind="stable")
    return np.split(order, np.flatnonzero(np.diff(t[order])) + 1)


def b1_route_d_design(truth: pa.Table, obs: pa.Table, dt_s: float = 1.0) -> tuple[FloatArray, FloatArray, list[str]]:
    """Design A and response y of the Route D moment equations E[r_i^2] = sum_b A_ib C_b, with the band order.

    With increment variances V_i = C_b u_i, u_i = 10^(-C/N0(t)/10) + 10^(-C/N0(t+dt)/10), and r = P y over the n
    members of an epoch, E[r_i^2] = V_i (1 - 2/n) + sum_j V_j / n^2. Epochs with a member lacking C/N0 at either
    end are skipped.
    """
    truth = validate(truth, TRUTH_EPOCH_SAT)
    obs = validate(obs, OBS_EPOCH_SAT)
    cn0 = dict(zip(
        (_key(*k) for k in zip(obs.column("sys").to_pylist(), obs.column("prn").to_pylist(),
                               obs.column("signal").to_pylist(), obs.column("t_gps_s").to_pylist(), strict=True)),
        obs.column("cn0_dbhz").to_pylist(), strict=True,
    ))  # fmt: skip
    sys, prn, sig = truth.column("sys").to_pylist(), truth.column("prn").to_pylist(), truth.column("signal").to_pylist()
    t, e = truth.column("t_rx_s").to_numpy(), truth.column("e_cyc").to_numpy()
    bands = sorted({band(s, g) for s, g in zip(sys, sig, strict=True)})
    col = {b: j for j, b in enumerate(bands)}
    rows_a, rows_y = [], []
    for rows in _epochs(truth):
        c0 = [cn0.get(_key(sys[i], prn[i], sig[i], t[i])) for i in rows]
        c1 = [cn0.get(_key(sys[i], prn[i], sig[i], t[i] + dt_s)) for i in rows]
        if any(v is None for v in c0 + c1):
            continue
        n = rows.size
        u = 10.0 ** (-np.array(c0, dtype=float) / 10.0) + 10.0 ** (-np.array(c1, dtype=float) / 10.0)
        b_idx = np.array([col[band(sys[i], sig[i])] for i in rows])
        totals = np.bincount(b_idx, weights=u, minlength=len(bands)) / n**2
        a = np.tile(totals, (n, 1))
        a[np.arange(n), b_idx] += u * (1.0 - 2.0 / n)
        r_m = np.array([e[i] * wavelength_m(sys[i], sig[i]) for i in rows])
        rows_a.append(a)
        rows_y.append(r_m**2)
    if not rows_a:
        raise ValueError("no epoch with C/N0 at both ends of every increment")
    return np.vstack(rows_a), np.concatenate(rows_y), bands


def fit_b1_route_d(truth: pa.Table, obs: pa.Table, dt_s: float = 1.0) -> dict[str, float]:
    """B1's C_i per band (m^2 Hz) from the projected Route D residuals (calibration split only).

    Non-negative least squares on the moment equations of b1_route_d_design. Var(r_i^2) = 2 E[r_i^2]^2 under a
    Gaussian, so rows are weighted by 1 / E[r_i^2] and the weights iterated (quasi-likelihood, starting from about
    1 / u_i). Without weights, a few low-C/N0 rows, whose u is up to 100 times the rest, dominate the fit.
    """
    a_all, y_all, bands = b1_route_d_design(truth, obs, dt_s)
    scale = a_all.max(axis=1)
    c = np.zeros(a_all.shape[1])
    for _ in range(IRLS_ITERATIONS):
        c, _ = optimize.nnls(a_all / scale[:, None], y_all / scale)
        predicted = a_all @ c
        scale = np.where(predicted > 0, predicted, scale)
    return {b: float(c[j]) for j, b in enumerate(bands)}


def b6_innovations(truth: pa.Table) -> pa.Table:
    """B6's innovations: each projected increment rescaled to one epoch's error, r sqrt(n / (2 (n - 1))) cycles.

    For equal variances E[r^2] = 2 sigma^2 (n - 1) / n, so the rescaled value has the per-epoch variance sigma^2.
    """
    truth = validate(truth, TRUTH_EPOCH_SAT)
    e = truth.column("e_cyc").to_numpy()
    factor = np.empty(truth.num_rows)
    for rows in _epochs(truth):
        n = rows.size
        factor[rows] = math.sqrt(n / (2.0 * (n - 1))) if n > 1 else np.nan
    return pa.table(
        {
            "sys": truth.column("sys"),
            "prn": truth.column("prn"),
            "signal": truth.column("signal"),
            "t_rx_s": truth.column("t_rx_s"),
            "innov_cyc": pa.array(e * factor, pa.float64()),
        }
    )


def summarize_coverage(
    scored: pa.Table, levels: Sequence[float] = (0.68, 0.95, 0.99), n_boot: int = 1000, seed: int = 0, t_s: float = 1.0
) -> dict[str, Any]:
    """c(p), the measured tau_int of the 95 % exceedance indicators, n_eff, the power-rule label (docs/5 4.2), and a
    hierarchical block-bootstrap 95 % interval of c(0.95) (series resampled, moving blocks of 3 tau_int inside)."""
    n = scored.num_rows
    if n == 0:
        return {"n": 0}
    w = scored.column("w").to_numpy()
    series = [f"{s}{p:02d}{g}" for s, p, g in zip(scored.column("sys").to_pylist(), scored.column("prn").to_pylist(),
                                                     scored.column("signal").to_pylist(), strict=True)]  # fmt: skip
    t = scored.column("t_rx_s").to_numpy()
    codes = np.unique(np.array(series), return_inverse=True)[1]
    order = np.lexsort((t, codes))
    w, codes = w[order], codes[order]
    out: dict[str, Any] = {"n": int(n)}
    for p in levels:
        out[f"c{p:g}"] = float(np.mean(np.abs(w) <= stats.norm.ppf((1 + p) / 2)))
    inside = (np.abs(w) <= stats.norm.ppf(0.975)).astype(float)
    groups: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(codes):
        groups[int(c)].append(i)
    taus, n_eff = [], 0.0
    for idx in groups.values():
        tau = tau_int_s(inside[idx], t_s) if len(idx) > 1 else t_s
        tau = max(tau, t_s)
        taus.append(tau)
        n_eff += len(idx) * t_s / tau
    tau_med = float(np.median(taus))
    out["tau_int_s"] = tau_med
    out["n_eff"] = float(n_eff)
    out["label"] = audit_label(out["c0.95"], n_eff)
    hierarchy = Hierarchy(levels=[codes.astype(np.int64)], block_len=max(1, math.ceil(3 * tau_med / t_s)))
    ci = bootstrap_ci(lambda idx: float(inside[idx].mean()), hierarchy, n_boot=n_boot, seed=seed)
    out["ci95_c0.95"] = [ci.lo, ci.hi]
    return out
