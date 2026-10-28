"""Route D: time-differenced carrier phase at a known position (docs/3 section 4.5).

The truth table holds the projected residual r = P y per satellite, with P = I - 1 1^T / n removing the common
receiver-clock increment and y = lambda dPhi - drho + c dt_sat. P does not depend on any model, so the truth
stays model-free (docs/3 section 6); standardising r under a model's variances is eval's job
(gnssaudit.eval.route_d). What Route D can and cannot see is in docs/3 section 4.5 (F3).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from gnssaudit.geometry import Ephemerides, elev_azim_deg, geometric_range, sat_pos_clock
from gnssaudit.io.schemas import EXCLUSIONS, OBS_EPOCH_SAT, TRUTH_EPOCH_SAT, build, validate
from gnssaudit.signals import C_M_S, wavelength_m
from gnssaudit.truth.arcs import LLI_HALF_CYCLE, arc_ids

FloatArray = NDArray[np.float64]
STAGE = "truth.route_d"


def _column(table: pa.Table, name: str, fill: object) -> np.ndarray:
    return table.column(name).fill_null(fill).to_numpy()


def receiver_clock_s(
    t: FloatArray, sys: list[str], prn: NDArray[np.int16], pr_m: FloatArray, eph: Ephemerides, rx_ecef: FloatArray
) -> dict[float, float]:
    """Receiver clock offset per time tag, median of (P - rho + c dt_sat) / c over the satellites at that tag.

    With the position known this needs no solution for position; the tropospheric and ionospheric delays left in P
    bias it by tens of nanoseconds, far below what matters for evaluating the geometry (review finding 1).
    """
    per_tag: dict[float, list[float]] = defaultdict(list)
    for i in np.flatnonzero(np.isfinite(pr_m)):
        try:
            record = eph.select(sys[i], int(prn[i]), float(t[i]))
        except KeyError:
            continue
        rho, clock = geometric_range(record, rx_ecef, float(t[i]))
        per_tag[round(float(t[i]), 6)].append((pr_m[i] - rho) / C_M_S + clock)
    return {tag: float(np.median(v)) for tag, v in per_tag.items()}


def tdcp_residuals(
    obs: pa.Table,
    eph: Ephemerides,
    rx_ecef: FloatArray,
    dt_s: float = 1.0,
    elev_mask_deg: float = 10.0,
    min_sats: int = 5,
    truth_id: str = "tdcp",
) -> tuple[pa.Table, pa.Table]:
    """truth_epoch_sat rows (kind 'tdcp', t_rx_s = time tag at the start of the increment, e_cyc = projected residual
    in cycles of each signal) and the exclusions table. One receiver per call; systems and bands share one clock
    increment; min_sats counts distinct satellites, not signals.

    Time tags carry the receiver clock offset b; the geometry is evaluated at tag - b, with b from the pseudoranges
    (receiver_clock_s). Increments pair each epoch with the epoch dt_s later in the same series, so logs faster than
    1 / dt_s give overlapping increments. Reasons: no_clock, no_partner (data continue but the epoch dt_s later is
    missing), half_cycle (LLI bit 1 at either end), arc_break, no_ephemeris, below_mask, too_few_sats.
    """
    obs = validate(obs, OBS_EPOCH_SAT)
    t = obs.column("t_gps_s").to_numpy()
    sys = obs.column("sys").to_pylist()
    prn = obs.column("prn").to_numpy()
    sig = obs.column("signal").to_pylist()
    cp = obs.column("cp_cyc").to_numpy(zero_copy_only=False)
    pr = obs.column("pr_m").to_numpy(zero_copy_only=False)
    lli = _column(obs, "lli", 0)
    half_cycle = (lli & LLI_HALF_CYCLE) != 0
    valid = _column(obs, "valid", False) & np.isfinite(cp) & ~half_cycle
    source = obs.column("source_id")[0].as_py() if obs.num_rows else ""
    clock_s = receiver_clock_s(t, sys, prn, pr, eph, rx_ecef)

    series: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for i in range(obs.num_rows):
        series[(sys[i], int(prn[i]), sig[i])].append(i)

    excluded: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    increments: dict[float, list[tuple[tuple[str, int, str], float, int]]] = defaultdict(list)
    for key, rows in series.items():
        idx = np.array(rows)[np.argsort(t[rows])]
        ts = t[idx]
        arcs = arc_ids(ts, lli[idx], valid[idx], max_gap_s=1.5 * dt_s)
        lam = wavelength_m(key[0], key[2])
        partner = np.searchsorted(ts, ts + dt_s - 1e-3)
        for a in range(idx.size):
            b = int(partner[a])
            if b >= idx.size or abs(ts[b] - ts[a] - dt_s) > 1e-3:
                if ts[-1] > ts[a] + dt_s - 1e-3:
                    excluded[("no_partner", *key)].append(float(ts[a]))
                continue
            i, j = idx[a], idx[b]
            if half_cycle[i] or half_cycle[j]:
                excluded[("half_cycle", *key)].append(float(t[i]))
                continue
            if arcs[a] < 0 or arcs[a] != arcs[b]:
                excluded[("arc_break", *key)].append(float(t[i]))
                continue
            b0, b1 = clock_s.get(round(float(t[i]), 6)), clock_s.get(round(float(t[j]), 6))
            if b0 is None or b1 is None:
                excluded[("no_clock", *key)].append(float(t[i]))
                continue
            t0, t1 = float(t[i]) - b0, float(t[j]) - b1
            try:
                record = eph.select(key[0], key[1], t0)
            except KeyError:
                excluded[("no_ephemeris", *key)].append(float(t[i]))
                continue
            pos, _ = sat_pos_clock(record, t0)
            if elev_azim_deg(rx_ecef, pos)[0] < elev_mask_deg:
                excluded[("below_mask", *key)].append(float(t[i]))
                continue
            rho0, clk0 = geometric_range(record, rx_ecef, t0)
            rho1, clk1 = geometric_range(record, rx_ecef, t1)
            y = lam * (cp[j] - cp[i]) - (rho1 - rho0) + C_M_S * (clk1 - clk0)
            increments[round(float(t[i]), 6)].append((key, y, int(arcs[a])))

    out: dict[str, list] = {k: [] for k in ("sys", "prn", "signal", "t", "e", "arc")}
    for t0, members in sorted(increments.items()):
        if len({(m[0][0], m[0][1]) for m in members}) < min_sats:
            for key, _, _ in members:
                excluded[("too_few_sats", *key)].append(t0)
            continue
        y = np.array([m[1] for m in members])
        r = y - y.mean()
        for (key, _, arc), r_m in zip(members, r, strict=True):
            out["sys"].append(key[0])
            out["prn"].append(key[1])
            out["signal"].append(key[2])
            out["t"].append(t0)
            out["e"].append(r_m / wavelength_m(key[0], key[2]))
            out["arc"].append(arc)

    n = len(out["t"])
    truth = build(
        TRUTH_EPOCH_SAT,
        {
            "truth_id": [truth_id] * n,
            "kind": ["tdcp"] * n,
            "sys": out["sys"],
            "prn": np.array(out["prn"], dtype=np.int16),
            "signal": out["signal"],
            "t_rx_s": np.array(out["t"], dtype=float),
            "e_cyc": np.array(out["e"], dtype=float),
            "slip_count": np.zeros(n, dtype=np.int16),
            "arc_id": np.array(out["arc"], dtype=np.int32),
            "transient": np.zeros(n, dtype=bool),
            "valid": np.ones(n, dtype=bool),
            "regime": np.zeros(n, dtype=np.uint8),
        },
        n,
    )
    return truth, exclusions_table(STAGE, source, excluded)


def exclusions_table(stage: str, source_id: str, excluded: dict[tuple[str, str, int, str], list[float]]) -> pa.Table:
    """One exclusions row per reason and satellite-signal, with the count and the time span it covers."""
    keys = sorted(excluded)
    n = len(keys)
    return build(
        EXCLUSIONS,
        {
            "stage": [stage] * n,
            "source_id": [source_id] * n,
            "sys": [k[1] for k in keys],
            "prn": np.array([k[2] for k in keys], dtype=np.int16),
            "signal": [k[3] for k in keys],
            "t_start_s": np.array([min(excluded[k]) for k in keys], dtype=float),
            "t_end_s": np.array([max(excluded[k]) for k in keys], dtype=float),
            "reason_code": [k[0] for k in keys],
            "count": np.array([len(excluded[k]) for k in keys], dtype=np.int64),
            "detail": [""] * n,
        },
        n,
    )
