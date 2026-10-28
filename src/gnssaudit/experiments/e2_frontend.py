"""The front-end rules of (a') (P2b part 2; the plan's pre-registered rules, docs/8 section 5).

Mode B's twin difference carries a term that only reshuffles the clean twin's own error, so Var_B / Var_A overstates
what the 3-bit front end adds. What jamming adds to the receiver's error is the paired excess,
Var(e_jammed) - Var(e_clean) = Var(dPhi) + 2 Cov(e_clean, dPhi), on epochs paired at identical record times. Per cell
(segment, family, level): each ghost's jammer slope of mode B on mode A, and R = sum_g excess_B / sum_g excess_A with a
moving-block bootstrap SE (the same blocks for every series). The verdicts, J_max, the route B ceiling and the
combination across segments follow the plan's rules as written.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Verdict = Literal["pass", "fail", "unresolved"]
NS = 1e9  # record times are paired at 1 ns


@dataclass(frozen=True)
class PairedExcess:
    var_dphi: float  # Var(dPhi): what the twin difference shows
    cov2: float  # 2 Cov(e_clean, dPhi): what cancels against the clean twin's own error
    excess: float  # var_dphi + cov2 = Var(e_jammed) - Var(e_clean)


@dataclass(frozen=True)
class FrontEndCell:
    slopes: tuple[float, ...]  # per ghost
    ratio: float  # sum_g excess_B / sum_g excess_A
    ratio_se: float


def paired_excess(dphi: FloatArray, e_clean: FloatArray) -> PairedExcess:
    """Var(e_c + dPhi) - Var(e_c) = Var(dPhi) + 2 Cov(e_c, dPhi) on paired epochs: what jamming adds to the receiver's
    error. A term of dPhi that only reshuffles the clean twin's own noise cancels in it (docs/8 section 5,
    2026-10-16)."""
    if dphi.shape != e_clean.shape or dphi.size < 2:
        raise ValueError("dPhi and the clean error must be paired epoch by epoch")
    d, c = dphi - dphi.mean(), e_clean - e_clean.mean()
    v, c2 = float(np.mean(d * d)), 2.0 * float(np.mean(c * d))
    return PairedExcess(v, c2, v + c2)


def jammer_slope(dphi_a: FloatArray, dphi_b: FloatArray) -> float:
    """Cov(dPhi_A, dPhi_B) / Var(dPhi_A) on one jammer draw: 1 when the front end passes the jammer's effect."""
    a, b = dphi_a - dphi_a.mean(), dphi_b - dphi_b.mean()
    return float(np.mean(a * b) / np.mean(a * a))


def block_bootstrap_se(
    stat: Callable[..., float], arrays: Sequence[FloatArray], *, block: int, n_boot: int = 400, seed: int = 7
) -> float:
    """SE of stat(*arrays) by a moving-block bootstrap over epochs: the same blocks for every array, one replicate at a
    time (memory stays at one resample)."""
    n = arrays[0].size
    if any(a.size != n for a in arrays) or n < 2 * block:
        raise ValueError("arrays must share a length of at least two blocks")
    rng = np.random.default_rng(seed)
    k = -(-n // block)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = (rng.integers(0, n - block + 1, size=k)[:, None] + np.arange(block)).ravel()[:n]
        vals[i] = stat(*(a[idx] for a in arrays))
    return float(np.std(vals, ddof=1))


def finite_cell(c: FrontEndCell) -> bool:
    """Every slope, R and SE(R) finite: the verdict's comparisons mean what they say only then."""
    return all(math.isfinite(v) for v in (*c.slopes, c.ratio, c.ratio_se))


def cell_verdict(
    c: FrontEndCell, *, tol: float, slope_tol: float, se_max: float, cell: str = "a front-end cell"
) -> Verdict:
    """The pre-registered verdict. A cell with a non-finite slope, R or SE(R) is refused, naming `cell`: |NaN - 1| >=
    tol is false, so a NaN slope would pass and a NaN SE(R) would read as resolved."""
    if not finite_cell(c):
        raise ValueError(f"{cell}: a non-finite slope, R or SE(R) has no verdict ({c})")
    if any(abs(s - 1.0) >= slope_tol for s in c.slopes):
        return "fail"
    if c.ratio_se > se_max:
        return "unresolved"
    return "pass" if abs(c.ratio - 1.0) < tol else "fail"


def front_end_jmax(
    cells: Mapping[float, FrontEndCell], *, tol: float, slope_tol: float, se_max: float, cell: str = "front-end"
) -> float | None:
    """The largest level that passes with no fail at or below it; None if none passes. `cell` names the series (a
    refused verdict names its level too)."""
    best = None
    for lv in sorted(cells):
        v = cell_verdict(cells[lv], tol=tol, slope_tol=slope_tol, se_max=se_max, cell=f"{cell} at {lv:g} dB-Hz")
        if v == "fail":
            break
        if v == "pass":
            best = lv
    return best


def route_b_ceiling(
    levels: Sequence[float], jmax: float | None, b1_ok: Mapping[float, bool], truth_ok: Mapping[float, bool]
) -> float | None:
    if jmax is None:
        return None
    best = None
    for lv in sorted(lv for lv in levels if lv <= jmax):
        if not (b1_ok.get(lv, False) and truth_ok.get(lv, False)):
            break
        best = lv
    return best


def ghost_ceiling(levels: Sequence[float], jmax: float | None, truth_ok: Mapping[float, bool]) -> float | None:
    """The ghost ceiling L_G(s, f) (P2c, R-2): the largest level up to J_max such that at every grid level up to it both
    ghosts keep their truth in mode A. B1 does not enter (R-2's Judgment: ghost truth's target needs no linear model of
    dPhi): it is the Route B ceiling with B1's bar passed at every level."""
    return route_b_ceiling(levels, jmax, dict.fromkeys(levels, True), truth_ok)


def combine_segments(values: Sequence[float | None]) -> float | None:
    vals = list(values)
    return None if not vals or any(v is None for v in vals) else min(v for v in vals if v is not None)


# ------------------------------------------------------------------------------------------------ pairing
def pair_on_time(*series: tuple[FloatArray, FloatArray]) -> tuple[FloatArray, list[FloatArray], int]:
    """(common t, values per series, records dropped) of (t, value) series joined on identical record times (t rounded
    to 1 ns), in time order. No interpolation: one record of shift is another pairing, so a record whose time is not
    in every series is dropped and counted."""
    keys = []
    for t, v in series:
        k = np.rint(np.asarray(t, dtype=np.float64) * NS).astype(np.int64)
        if k.shape != np.shape(v):
            raise ValueError("each series needs one value per record time")
        if np.unique(k).size != k.size:
            raise ValueError("a series repeats a record time")
        keys.append(k)
    common = keys[0]
    for k in keys[1:]:
        common = np.intersect1d(common, k, assume_unique=True)
    common = np.unique(common)  # sorted, for a single series too
    pos = []
    for k in keys:
        order = np.argsort(k, kind="stable")
        pos.append(order[np.searchsorted(k, common, sorter=order)])
    t0 = np.asarray(series[0][0], dtype=np.float64)
    values = [np.asarray(v, dtype=np.float64)[p] for (_, v), p in zip(series, pos, strict=True)]
    dropped = sum(k.size for k in keys) - len(keys) * common.size
    return t0[pos[0]], values, int(dropped)


def align_records(times: Sequence[FloatArray], *, tol_s: float) -> tuple[list[NDArray[np.intp]], int]:
    """(indices into each sorted epoch set, records dropped) of the records of different channels that stand for the
    same epoch: each record of the first set paired with the nearest record of every other set within tol_s (under
    half a record period, so the pairing is one to one). Channels never share stamps, so the pooled bootstrap draws
    its blocks on these; a record without a counterpart in every set is dropped and counted."""
    if any(np.any(np.diff(np.asarray(t, dtype=np.float64)) <= 0) for t in times):
        raise ValueError("align_records needs each epoch set in strictly increasing time (the nearest-record search)")
    ref = np.asarray(times[0], dtype=np.float64)
    ok = np.ones(ref.size, dtype=bool)
    idx: list[NDArray[np.intp]] = [np.arange(ref.size)]
    for t in times[1:]:
        t = np.asarray(t, dtype=np.float64)
        if t.size == 0:
            ok[:] = False
            idx.append(np.zeros(ref.size, dtype=np.intp))
            continue
        j = np.searchsorted(t, ref)
        left, right = np.clip(j - 1, 0, t.size - 1), np.clip(j, 0, t.size - 1)
        near = np.where(np.abs(t[left] - ref) <= np.abs(t[right] - ref), left, right)
        ok &= np.abs(t[near] - ref) < tol_s
        idx.append(near)
    out = [i[ok] for i in idx]
    if any(np.unique(i).size != i.size for i in out):
        raise ValueError(f"records closer than 2 x {tol_s} s: the alignment is not one to one")
    dropped = sum(np.size(t) for t in times) - len(times) * int(ok.sum())
    return out, int(dropped)


def pooled_ratio(ghosts: Sequence[Sequence[FloatArray]]) -> float:
    """R = sum_g excess_B,g / sum_g excess_A,g, each ghost given as (dPhi_A, dPhi_B, e_clean_A, e_clean_B) on its
    paired epochs."""
    a = sum(paired_excess(g[0], g[2]).excess for g in ghosts)
    b = sum(paired_excess(g[1], g[3]).excess for g in ghosts)
    with np.errstate(divide="ignore", invalid="ignore"):  # no mode A excess: R is inf or NaN, refused as non-finite
        return float(np.float64(b) / np.float64(a))


def front_end_cell(
    ghosts: Sequence[Sequence[FloatArray]], *, block: int, tol_s: float, n_boot: int = 400, seed: int = 7
) -> tuple[FrontEndCell, int]:
    """One cell from each ghost's (t, dPhi_A, dPhi_B, e_clean_A, e_clean_B), paired on its own record times: the
    slopes and R on each ghost's epochs; SE(R) by the block bootstrap on the epochs every ghost holds (align_records),
    so every series shares the blocks. Also returns the records that alignment dropped (for the SE only)."""
    slopes = tuple(jammer_slope(g[1], g[2]) for g in ghosts)
    ratio = pooled_ratio([g[1:] for g in ghosts])
    idx, dropped = align_records([g[0] for g in ghosts], tol_s=tol_s)
    arrays = [np.asarray(s)[i] for g, i in zip(ghosts, idx, strict=True) for s in g[1:]]

    def stat(*a: FloatArray) -> float:
        return pooled_ratio([a[i : i + 4] for i in range(0, len(a), 4)])

    se = block_bootstrap_se(stat, arrays, block=block, n_boot=n_boot, seed=seed)
    return FrontEndCell(slopes, ratio, se), dropped
