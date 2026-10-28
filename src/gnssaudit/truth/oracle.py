"""Oracle moments over an ensemble of K jammed twins (docs/3 section 2.4; P2b task 11).

K jammed twins share one clean twin, one recording and one configuration, and differ in the interference's random
draw. Per epoch and satellite: mu = the mean twin error, var = its sample variance (ddof 1), and p_slip = the share of
realizations with a slip in [t, t + slip_window). Invalid and transient epochs leave the ensemble (n_valid counts the
rest). The form says what the draw covered (F8): `conditional` fixes the jammer's timing and draws only carrier phase
and noise, the target of per-epoch calibration (H3); `marginal` also draws the timing, for climatological statements.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pyarrow as pa

from gnssaudit.io.schemas import EVENTS, ORACLE_MOMENTS, TRUTH_EPOCH_SAT, build, validate

KEY = ("sys", "prn", "signal", "t_rx_s")


class NoCommonEpochs(ValueError):
    """Two ensembles share no epoch with a variance in both: k_convergence has nothing to compare."""


def oracle_moments(
    truths: Sequence[pa.Table],
    slips: Sequence[pa.Table],
    *,
    config_id: str,
    form: Literal["conditional", "marginal"],
    slip_window_s: float,
) -> pa.Table:
    """ORACLE_MOMENTS from K twin-truth tables (TRUTH_EPOCH_SAT) and their slip events (EVENTS), one per realization."""
    if len(truths) != len(slips) or not truths:
        raise ValueError(f"{len(truths)} truth tables against {len(slips)} slip tables")
    k = len(truths)
    rows: dict[tuple[str, int, str, float], list[float]] = {}
    for tr in truths:
        tr = validate(tr, TRUTH_EPOCH_SAT)
        valid = tr.column("valid").to_numpy(zero_copy_only=False)
        ok = valid & ~tr.column("transient").to_numpy(zero_copy_only=False)
        keys = zip(*(tr.column(c).to_pylist() for c in KEY), strict=True)
        for key, e, good in zip(keys, tr.column("e_cyc").to_pylist(), ok, strict=True):
            rows.setdefault(key, [])
            if good:
                rows[key].append(e)
    ev = [validate(s, EVENTS) for s in slips]
    slip_t: list[dict[tuple[str, int, str], np.ndarray]] = []
    for s in ev:
        s = s.filter(np.asarray(s.column("kind").to_pylist()) == "slip") if s.num_rows else s
        by: dict[tuple[str, int, str], list[float]] = {}
        for sys, prn, sig, t in zip(*(s.column(c).to_pylist() for c in ("sys", "prn", "signal", "t_start_s")),
                                    strict=True):  # fmt: skip
            by.setdefault((sys, prn, sig), []).append(t)
        slip_t.append({key: np.asarray(v) for key, v in by.items()})
    out: dict[str, list] = {c: [] for c in ORACLE_MOMENTS.names}
    for (sys, prn, sig, t), vals in sorted(rows.items()):
        v = np.asarray(vals, dtype=np.float64)
        at = [st.get((sys, prn, sig), np.empty(0)) for st in slip_t]
        hits = sum(1 for a in at if np.any((a >= t) & (a < t + slip_window_s)))
        out["config_id"].append(config_id)
        out["form"].append(form)
        out["sys"].append(sys)
        out["prn"].append(prn)
        out["signal"].append(sig)
        out["t_rx_s"].append(t)
        out["K"].append(k)
        out["n_valid"].append(v.size)
        out["mu_cyc"].append(float(v.mean()) if v.size else float("nan"))
        out["var_cyc2"].append(float(v.var(ddof=1)) if v.size > 1 else float("nan"))
        out["p_slip"].append(hits / k)
        out["slip_window_s"].append(slip_window_s)
    return build(ORACLE_MOMENTS, out, len(out["t_rx_s"]))


def k_convergence(small: pa.Table, large: pa.Table) -> dict[str, float | bool]:
    """docs/5 section 4.4: doubling K must change sigma^2_oracle by less than its standard error
    sqrt(2 / (K_small - 1)), judged on the median over common epochs of |var_large / var_small - 1|."""
    a = {k: v for k, v in zip(zip(*(small.column(c).to_pylist() for c in KEY), strict=True),
                              small.column("var_cyc2").to_pylist(), strict=True)}  # fmt: skip
    b = {k: v for k, v in zip(zip(*(large.column(c).to_pylist() for c in KEY), strict=True),
                              large.column("var_cyc2").to_pylist(), strict=True)}  # fmt: skip
    common = [key for key in a if key in b and a[key] and b[key] and np.isfinite(a[key]) and np.isfinite(b[key])]
    if not common:
        raise NoCommonEpochs("no epoch with a variance in both ensembles")
    change = float(np.median([abs(b[key] / a[key] - 1.0) for key in common]))
    k_small = int(small.column("K")[0].as_py())
    se = float(np.sqrt(2.0 / (k_small - 1)))
    return {"median_rel_change": change, "se": se, "converged": change <= se, "epochs": float(len(common))}
