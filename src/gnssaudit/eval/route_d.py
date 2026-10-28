"""Standardising Route D residuals under a model's variances (docs/3 section 4.5, docs/5 section 3.1)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from gnssaudit.eval.scores import coverage
from gnssaudit.io.schemas import PREDICTION, TRUTH_EPOCH_SAT, validate
from gnssaudit.signals import wavelength_m

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RouteDEpoch:
    w: FloatArray  # per-satellite statistic, N(0, 1) under the model
    chi2: float  # chi-square with dof degrees of freedom under the model
    dof: int


def standardize_epoch(r_m: FloatArray, var_t_m2: FloatArray, var_t_next_m2: FloatArray) -> RouteDEpoch:
    """r_m: projected residuals of one epoch (metres). Under the model the increments are independent with variance
    var(t) + var(t + Delta), so Cov(r) = P Sigma P with P = I - 1 1^T / n."""
    n = r_m.size
    if n < 2:
        raise ValueError("Route D needs at least two satellites per epoch")
    p = np.eye(n) - 1.0 / n
    q = p @ np.diag(var_t_m2 + var_t_next_m2) @ p
    w = r_m / np.sqrt(np.diag(q))
    chi2 = float(r_m @ np.linalg.pinv(q) @ r_m)
    return RouteDEpoch(w=w, chi2=chi2, dof=n - 1)


def _key(sys: str, prn: int, signal: str, t: float) -> tuple[str, int, str, float]:
    return sys, prn, signal, round(t, 3)


def score_route_d(
    truth: pa.Table, pred: pa.Table, dt_s: float = 1.0, min_sats: int = 5
) -> tuple[pa.Table, dict[str, int]]:
    """w statistics of every Route D residual under a model's Gaussian predictions, and the unscored row counts.

    Each epoch keeps the residuals whose prediction is valid at both ends of the increment, then re-centres them on
    that subset: r_S - mean(r_S) = y_S - mean(y_S), so the result is the exact Route D residual of the subset and
    the clock projector applies unchanged. Reasons: no_prediction (a member without a valid prediction at t or
    t + dt_s), too_few_predicted (fewer than min_sats distinct satellites left in the epoch).
    """
    truth = validate(truth, TRUTH_EPOCH_SAT)
    pred = validate(pred, PREDICTION)
    if np.any(pred.column("nu").to_numpy(zero_copy_only=False) > 0):
        raise ValueError("Route D scoring covers Gaussian predictions (nu = 0) only")
    p_valid = pred.column("valid").to_numpy(zero_copy_only=False)
    p_scale = pred.column("scale_cyc").to_numpy(zero_copy_only=False)
    scale: dict[tuple[str, int, str, float], float] = {}
    for i, k in enumerate(
        zip(
            pred.column("sys").to_pylist(),
            pred.column("prn").to_pylist(),
            pred.column("signal").to_pylist(),
            pred.column("t_rx_s").to_pylist(),
            strict=True,
        )  # fmt: skip
    ):
        if p_valid[i] and np.isfinite(p_scale[i]):
            scale[_key(*k)] = float(p_scale[i])

    sys = truth.column("sys").to_pylist()
    prn = truth.column("prn").to_pylist()
    sig = truth.column("signal").to_pylist()
    t = truth.column("t_rx_s").to_numpy()
    e = truth.column("e_cyc").to_numpy()
    counts: dict[str, int] = defaultdict(int)
    out: dict[str, list] = {k: [] for k in ("sys", "prn", "signal", "t_rx_s", "w", "sd_inc_m")}
    order = np.argsort(t, kind="stable")
    bounds = np.flatnonzero(np.diff(t[order])) + 1
    for rows in np.split(order, bounds):
        keep, lam, v0, v1 = [], [], [], []
        for i in rows:
            s0 = scale.get(_key(sys[i], prn[i], sig[i], t[i]))
            s1 = scale.get(_key(sys[i], prn[i], sig[i], t[i] + dt_s))
            if s0 is None or s1 is None:
                counts["no_prediction"] += 1
                continue
            lam_i = wavelength_m(sys[i], sig[i])
            keep.append(i)
            lam.append(lam_i)
            v0.append((s0 * lam_i) ** 2)
            v1.append((s1 * lam_i) ** 2)
        if len({(sys[i], prn[i]) for i in keep}) < min_sats:
            counts["too_few_predicted"] += len(keep)
            continue
        r = e[keep] * np.array(lam)
        epoch = standardize_epoch(r - r.mean(), np.array(v0), np.array(v1))
        for j, i in enumerate(keep):
            out["sys"].append(sys[i])
            out["prn"].append(prn[i])
            out["signal"].append(sig[i])
            out["t_rx_s"].append(float(t[i]))
            out["w"].append(float(epoch.w[j]))
            out["sd_inc_m"].append(float(np.sqrt(v0[j] + v1[j])))
    scored = pa.table(
        {
            "sys": pa.array(out["sys"], pa.string()),
            "prn": pa.array(out["prn"], pa.int16()),
            "signal": pa.array(out["signal"], pa.string()),
            "t_rx_s": pa.array(out["t_rx_s"], pa.float64()),
            "w": pa.array(out["w"], pa.float64()),
            "sd_inc_m": pa.array(out["sd_inc_m"], pa.float64()),
        }
    )
    return scored, dict(counts)


def coverage_table(
    scored: pa.Table, levels: Sequence[float] = (0.68, 0.95, 0.99), by: Sequence[str] = ()
) -> list[dict[str, Any]]:
    """c(p) of the w statistics per group of the `by` columns (all rows when `by` is empty), with n."""
    w = scored.column("w").to_numpy()
    keys = list(zip(*(scored.column(c).to_pylist() for c in by), strict=True)) if by else [()] * scored.num_rows
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)
    rows: list[dict[str, Any]] = []
    for k in sorted(groups):
        idx = np.array(groups[k])
        c = coverage(w[idx], np.ones(idx.size), np.zeros(idx.size), levels)
        row: dict[str, Any] = dict(zip(by, k, strict=True))
        row["n"] = int(idx.size)
        row.update({f"c{p:g}": float(v) for p, v in zip(levels, c, strict=True)})
        rows.append(row)
    return rows
