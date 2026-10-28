"""Route B twin truth: the twin difference, its slips and the label it yields (docs/3 sections 2.1 to 2.3).

Twins are paired on their observables epochs: the receiver-time label `t_gps_s` (docs/7 section 5.1), which with the
receiver time held fixed (docs/9 section 3, item 5) names the same sample instant in both twins. GNSS-SDR starts
tracking at a different sample in every run (F24), so the accumulated phases differ by a constant per satellite, and
each twin pulls in from its own start: epochs before the warm-up end are not paired, and the median of the
slip-corrected difference over a pre-onset window is removed. Slips are steps of the difference by a multiple of the
slip size (docs/3 section 2.3), searched on a 50 Hz grid (block medians when the series is faster, as tracking
records at 1 kHz are, where a real slip ramps over tens of epochs). A step that is not one, or one that noise makes
uncertain, ends that satellite's series: without a sure integer the rest has an unknown offset. Nothing is dropped
silently.
"""

from __future__ import annotations

from collections import defaultdict
from typing import cast

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from gnssaudit.io.schemas import EVENTS, OBS_EPOCH_SAT, TRUTH_EPOCH_SAT, build, validate
from gnssaudit.truth.route_d import exclusions_table

FloatArray = NDArray[np.float64]
STAGE = "truth.twin"
OVERLAP_MIN = 0.95
STEP_THRESHOLD_CYC = 0.25
MULTIPLE_TOL_CYC = 0.1
MARGIN_SD = 5.0
MEDIAN_LEN = 5
LOCAL_LEN = 50  # epochs on each side for the local robust spread (1 s at 50 Hz)
SLIP_GRID_S = 0.02  # slips are searched on a 50 Hz grid: at 1 kHz a real slip ramps over tens of epochs (p2a review)
REACQ_GAP_EPOCHS = 12  # a longer gap in one twin's records is a re-acquisition: a new phase origin (p2a review)
MAX_SLIP_CYC = 4.0  # a larger step is no slip; it is ambiguous
PAIR_TOL_STEPS = 0.25  # twins' labels pair within this fraction of a step (one sample apart at 1 kHz tracking records)

Sat = tuple[str, int, str]


class TwinAlignmentError(ValueError):
    """The twins do not share enough observable epochs to be paired."""


def _by_satellite(obs: pa.Table) -> dict[Sat, tuple[FloatArray, FloatArray]]:
    obs = validate(obs, OBS_EPOCH_SAT)
    ok = obs.column("valid").to_numpy(zero_copy_only=False) & obs.column("cp_cyc").is_valid().to_numpy(
        zero_copy_only=False
    )
    obs = obs.filter(ok)
    sys, prn, sig = obs.column("sys").to_pylist(), obs.column("prn").to_pylist(), obs.column("signal").to_pylist()
    t, cp = obs.column("t_gps_s").to_numpy(), obs.column("cp_cyc").to_numpy()
    rows: dict[Sat, list[int]] = defaultdict(list)
    for i, key in enumerate(zip(sys, prn, sig, strict=True)):
        rows[key].append(i)
    out = {}
    for key, idx in rows.items():
        ii = np.array(idx)
        order = np.argsort(t[ii], kind="stable")
        out[key] = (t[ii][order], cp[ii][order])
    return out


def _break_at(t: FloatArray, step: float) -> int:
    """Index of the first epoch whose label does not advance by a whole number of steps, within a tenth of a step (a
    repeated or backward label: the receiver time moved), or len(t) if there is none. The tolerance is relative
    because labels taken from tracking stamps jitter by a sample with the code Doppler (4000 +-1 samples per 1 ms)."""
    d = np.diff(t)
    k = np.round(d / step)
    bad = (k < 1) | (np.abs(d - k * step) > 0.1 * step)
    return int(np.flatnonzero(bad)[0]) + 1 if bad.any() else t.size


def _pair(tc: FloatArray, tj: FloatArray, tol: float) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Indices (ic, ij) of the clean and jammed labels that pair one to one: the nearest jammed label within tol of
    each clean one. Tracking-record labels of the two twins can differ by a sample (code-loop jitter)."""
    if tc.size == 0 or tj.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    idx = np.searchsorted(tj, tc)
    lo, hi = np.clip(idx - 1, 0, tj.size - 1), np.clip(idx, 0, tj.size - 1)
    best = np.where(np.abs(tj[hi] - tc) < np.abs(tj[lo] - tc), hi, lo)
    ok = np.abs(tj[best] - tc) <= tol
    ic, ij = np.flatnonzero(ok), best[ok]
    _, first = np.unique(ij, return_index=True)  # one to one
    return ic[np.sort(first)], ij[np.sort(first)]


def _after_gap(t: FloatArray, step: float) -> float:
    """Label of the first epoch after a gap longer than REACQ_GAP_EPOCHS steps, or inf: GNSS-SDR restarts the
    accumulated phase when a channel re-acquires, so nothing after such a gap shares the earlier phase origin."""
    gaps = np.flatnonzero(np.diff(t) > REACQ_GAP_EPOCHS * step)
    return float(t[gaps[0] + 1]) if gaps.size else float("inf")


def _running_median(x: FloatArray, n: int) -> FloatArray:
    pad = n // 2
    xp = np.pad(x, pad, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(xp, n), axis=1)


def _robust_sd(x: FloatArray) -> float:
    if x.size < 3:
        return float("inf")
    d = np.diff(x)
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))


def twin_difference(
    clean_obs: pa.Table,
    jammed_obs: pa.Table,
    *,
    onset_s: float,
    pre_window_s: float,
    warmup_s: float,
    slip_size_cyc: float,
    tau_loop_s: float,
    truth_id: str,
) -> tuple[pa.Table, pa.Table, pa.Table]:
    """Twin truth (kind 'twin', e_cyc = slip-corrected jammed minus clean phase, offset removed), slip events, and
    exclusions. Epochs labelled before warmup_s are excluded ('warmup'). Raises TwinAlignmentError when fewer than
    95 % of the clean twin's valid epochs between the warm-up and the onset are in both twins: the rule tests the
    epoch grid; epochs lost after the onset are exclusions (p2a review, important 2). Labels pair within a quarter step,
    and the jammed phase is carried to the clean label with its own rate."""
    clean, jammed = _by_satellite(clean_obs), _by_satellite(jammed_obs)
    excluded: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    n_clean = common_total = 0
    # the epoch step after the warm-up: GNSS-SDR logs 1 ms records until bit sync, then one per extended integration
    # (20 ms), so the whole record's median spacing would read every later epoch as a gap (P2b task 10)
    steps = [np.median(np.diff(t[t >= warmup_s])) for t, _ in clean.values() if np.sum(t >= warmup_s) > 1]
    steps = steps or [np.median(np.diff(t)) for t, _ in clean.values() if t.size > 1]
    step = float(np.median(steps)) if steps else 0.02
    tol = PAIR_TOL_STEPS * step
    cols: dict[str, list[object]] = defaultdict(list)
    ev: dict[str, list[object]] = defaultdict(list)
    for key in sorted(set(clean) | set(jammed)):
        sys, prn, sig = key
        tc, cc = clean.get(key, (np.zeros(0), np.zeros(0)))
        tj, cj = jammed.get(key, (np.zeros(0), np.zeros(0)))
        grid = (tc >= warmup_s) & (tc < onset_s)  # label alignment before the onset, before any break is cut away
        n_clean += int(grid.sum())
        common_total += int(grid[_pair(tc, tj, tol)[0]].sum())
        early = np.union1d(tc[tc < warmup_s], tj[tj < warmup_s])  # each twin's own pull-in: no common phase yet
        if early.size:
            excluded[("warmup", sys, prn, sig)].extend(early.tolist())
            tc, cc, tj, cj = tc[tc >= warmup_s], cc[tc >= warmup_s], tj[tj >= warmup_s], cj[tj >= warmup_s]
        # breaks are sought after the warm-up only: the pull-in is never analysed, and its records come at another
        # spacing (1 ms until bit sync, then one per 20 ms integration; P2b task 10)
        for label, (tt, _) in (("clean", (tc, cc)), ("jammed", (tj, cj))):
            cut = _break_at(tt, step)
            if cut < tt.size:
                excluded[("rx_time_break", sys, prn, sig)].extend(tt[cut:].tolist())
                if label == "clean":
                    tc, cc = tc[:cut], cc[:cut]
                else:
                    tj, cj = tj[:cut], cj[:cut]
        t_cut = min(_after_gap(tc, step), _after_gap(tj, step))  # a re-acquisition after the warm-up ends the pair
        if np.isfinite(t_cut):
            excluded[("reacquisition", sys, prn, sig)].extend(np.union1d(tc[tc >= t_cut], tj[tj >= t_cut]).tolist())
            tc, cc, tj, cj = tc[tc < t_cut], cc[tc < t_cut], tj[tj < t_cut], cj[tj < t_cut]
        ic, ij = _pair(tc, tj, tol)
        only = np.concatenate([np.delete(tc, ic), np.delete(tj, ij)])
        if only.size:
            excluded[("one_twin_epoch", sys, prn, sig)].extend(np.sort(only).tolist())
        if ic.size == 0:
            continue
        both = tc[ic]
        rate = np.gradient(cj, tj) if tj.size > 1 else np.zeros(tj.size)
        d = cj[ij] + rate[ij] * (both - tj[ij]) - cc[ic]
        pre = (both >= onset_s - pre_window_s) & (both < onset_s)
        if not pre.any():
            excluded[("no_pre_onset_reference", sys, prn, sig)].extend(both.tolist())
            continue
        d = d - np.median(d[pre])
        e, slip_count, transient, keep = _slips_on_grid(both, d, slip_size_cyc, tau_loop_s, key, truth_id, ev,
                                                        excluded)  # fmt: skip
        ref = pre & keep
        if not ref.any():
            excluded[("no_pre_onset_reference", sys, prn, sig)].extend(both[keep].tolist())
            continue
        e = e - np.median(e[ref])  # a slip before or inside the window must not shift the reference level
        n = int(keep.sum())
        cols["sys"] += [sys] * n
        cols["prn"] += [prn] * n
        cols["signal"] += [sig] * n
        cols["t_rx_s"] += both[keep].tolist()
        cols["e_cyc"] += e[keep].tolist()
        cols["slip_count"] += slip_count[keep].tolist()
        cols["transient"] += transient[keep].tolist()
    if n_clean and common_total / n_clean < OVERLAP_MIN:
        raise TwinAlignmentError(
            f"{truth_id}: only {100 * common_total / n_clean:.1f} % of the clean twin's epochs are in both twins "
            f"(at least {100 * OVERLAP_MIN:.0f} % required)"
        )
    n = len(cols["t_rx_s"])
    truth = build(TRUTH_EPOCH_SAT, {
        "truth_id": [truth_id] * n, "kind": ["twin"] * n, "sys": cols["sys"],
        "prn": np.array(cols["prn"], dtype=np.int16), "signal": cols["signal"],
        "t_rx_s": np.array(cols["t_rx_s"], dtype=float), "e_cyc": np.array(cols["e_cyc"], dtype=float),
        "slip_count": np.array(cols["slip_count"], dtype=np.int16), "arc_id": np.zeros(n, dtype=np.int32),
        "transient": np.array(cols["transient"], dtype=bool), "valid": np.ones(n, dtype=bool),
        "regime": np.zeros(n, dtype=np.uint8),
    }, n)  # fmt: skip
    m = len(ev["event_id"])
    events = build(EVENTS, {
        "event_id": ev["event_id"], "source_id": [truth_id] * m, "kind": ["slip"] * m, "sys": ev["sys"],
        "prn": np.array(ev["prn"], dtype=np.int16), "signal": ev["signal"], "t_start_s": np.array(ev["t"], dtype=float),
        "t_end_s": np.array(ev["t"], dtype=float), "size_cyc": np.array(ev["size"], dtype=float),
        "confidence": np.array(ev["confidence"], dtype=float), "origin": ["truth"] * m,
    }, m)  # fmt: skip
    return truth, events, exclusions_table(STAGE, truth_id, excluded)


def _slips_on_grid(
    t: FloatArray,
    d: FloatArray,
    size: float,
    tau_s: float,
    key: Sat,
    truth_id: str,
    ev: dict[str, list[object]],
    excluded: dict[tuple[str, str, int, str], list[float]],
) -> tuple[FloatArray, NDArray[np.int64], NDArray[np.bool_], NDArray[np.bool_]]:
    """_slips on a series sampled at SLIP_GRID_S or slower; a faster one (tracking records, 1 kHz) is searched on the
    medians of consecutive blocks of SLIP_GRID_S, and what is found is applied to every epoch from its block on."""
    if t.size < 2:
        return _slips(t, d, size, tau_s, key, truth_id, ev, excluded)
    m = int(round(SLIP_GRID_S / float(np.median(np.diff(t)))))
    if m <= 1:
        return _slips(t, d, size, tau_s, key, truth_id, ev, excluded)
    starts = np.arange(0, t.size, m)
    t_blk = t[starts]
    d_blk = np.array([np.median(d[a : a + m]) for a in starts])
    first_event = len(ev["event_id"])
    found: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    _slips(t_blk, d_blk, size, tau_s, key, truth_id, ev, found)
    corr = np.zeros(d.size)
    count = np.zeros(d.size, dtype=np.int64)
    times = [cast(float, x) for x in ev["t"][first_event:]]
    sizes = [cast(float, x) for x in ev["size"][first_event:]]
    for ts, k in zip(times, sizes, strict=True):
        corr[t >= ts] += k
        count[t >= ts] += 1
    keep = np.ones(d.size, dtype=bool)
    sys, prn, sig = key
    amb = found.get(("ambiguous_slip", sys, prn, sig))
    if amb:
        t_cut, t_last = min(amb), max(amb)
        last_blk = int(np.searchsorted(t_blk, t_last, side="right"))
        t_end = t_blk[last_blk] if last_blk < t_blk.size else np.inf  # the ambiguous blocks end where the next starts
        keep[t >= t_cut] = False
        excluded[("ambiguous_slip", sys, prn, sig)].extend(t[(t >= t_cut) & (t < t_end)].tolist())
        excluded[("after_ambiguous_step", sys, prn, sig)].extend(t[t >= t_end].tolist())
    transient = np.zeros(d.size, dtype=bool)
    for ts in times:
        transient |= np.abs(t - ts) <= 3 * tau_s + 1e-9
    return d - corr, count, transient, keep


def _slips(
    t: FloatArray,
    d: FloatArray,
    size: float,
    tau_s: float,
    key: Sat,
    truth_id: str,
    ev: dict[str, list[object]],
    excluded: dict[tuple[str, str, int, str], list[float]],
) -> tuple[FloatArray, NDArray[np.int64], NDArray[np.bool_], NDArray[np.bool_]]:
    """Find slips in one satellite's difference d; return the corrected series, the running slip count, the
    transient flags and the epochs kept."""
    sys, prn, sig = key
    keep = np.ones(d.size, dtype=bool)
    corr = np.zeros(d.size)
    count = np.zeros(d.size, dtype=np.int64)
    slip_times: list[float] = []
    after = 0  # steps are searched for after the last accepted one, on the series corrected so far
    while True:
        y = d - corr
        cand = np.flatnonzero(np.abs(np.diff(_running_median(y, MEDIAN_LEN))) > STEP_THRESHOLD_CYC) + 1
        cand = cand[cand > after]
        if cand.size == 0:
            break
        i = int(cand[0])
        jump = float(np.median(y[i : i + MEDIAN_LEN]) - np.median(y[max(0, i - MEDIAN_LEN) : i]))
        k = round(jump / size)
        local = np.concatenate([y[max(0, i - LOCAL_LEN) : max(0, i - 2)], y[i + 2 : i + LOCAL_LEN] - jump])
        if (k == 0 or abs(k * size) > MAX_SLIP_CYC or abs(jump - k * size) > MULTIPLE_TOL_CYC
                or STEP_THRESHOLD_CYC < MARGIN_SD * _robust_sd(local)):  # fmt: skip
            lo = max(0, i - 2)  # the running median can place a noisy step up to two epochs early
            excluded[("ambiguous_slip", sys, prn, sig)].extend(t[lo : i + 1].tolist())
            if i + 1 < d.size:
                excluded[("after_ambiguous_step", sys, prn, sig)].extend(t[i + 1 :].tolist())
            keep[lo:] = False
            break
        corr[i:] += k * size
        count[i:] += 1
        after = i
        slip_times.append(float(t[i]))
        ev["event_id"].append(f"{truth_id}:{sys}{prn:02d}{sig}:{t[i]:.3f}")
        ev["sys"].append(sys)
        ev["prn"].append(prn)
        ev["signal"].append(sig)
        ev["t"].append(float(t[i]))
        ev["size"].append(k * size)
        ev["confidence"].append(1.0 - abs(jump - k * size) / MULTIPLE_TOL_CYC)
    transient = np.zeros(d.size, dtype=bool)
    for ts in slip_times:
        transient |= np.abs(t - ts) <= 3 * tau_s + 1e-9
    return d - corr, count, transient, keep
