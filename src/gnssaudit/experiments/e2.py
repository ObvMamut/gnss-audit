"""E2: truth fidelity and design constants on real twins (docs/10 section 2.3; P2b task 11).

Per segment, after a stock survey (the strongest real satellites, two ghosts at 45 dB-Hz on PRNs not in view):
- the grid: three families x J/N0 levels x front-end mode {A, B at the spec's AGC time constant}, A and B on paired
  seeds, against a null pair of mode A clean twins and a mode B clean twin; B1's first-order prediction for the ghosts
  on the mode A twins (the residual ratio r against J/N0);
- the AGC side check: mode B at two more time constants, at the top level, for two families;
- on the K-study segment: conditional (timing fixed) and marginal (timing drawn) swept CW, and noise, at
  L_K = min(cap, J_max of the family), for oracle moments and the K choice (docs/5 section 4.4);
- on the thermal segment: pairs sharing a common white floor at N0 (docs/3 section 2.4), to see whether sigma^2_oracle
  depends on the thermal draw;
- a seeded 5 % of the twins again (G-repeat).

The decision rules are the spec's, pre-registered for (a') (docs/8 section 5, docs/3 section 2.2, docs/5 section 4.4;
P2b part 2, task 5):
- the front-end J_max(s, f) by the paired excess (e2_frontend): per cell, the grid's mode A and mode B units against
  the clean twins' absolute ghost errors (ghost_abs_ch*.parquet of the clean units), on epochs joined on identical
  record times; capped on segment A by the tau side check;
- the Route B ceiling L_B(s, f): the largest level up to J_max at and below which B1's bar r - 1/(C/N0 T) < residual_max
  holds and no more than lost_truth_max_s of ghost truth is lost, in mode A for both ghosts;
- across segments the minimum, None if any segment's is None (combine_segments), for J_max, L_B and residual_pass;
- the K-study at L_K = min(cap, L_B) of the combined ceiling, on noise when swept CW has no ceiling; K = 51 if the
  median |sigma^2_101 / sigma^2_51 - 1| <= 0.2, else 101 (0.2 is sqrt(2 / (51 - 1)), sigma^2_oracle's relative SE
  at K = 51).
Option (a)'s Var_B / Var_A and its J_max are reported alongside, not deciding. Every seed keys on sorted names, never on
the order of a mapping in the spec (the manifest keeps the spec as sorted JSON).

The spec key `truth` (P2c; the rules R-1 to R-3 of docs/12 section 1.2) is `route_b`, the default and everything above,
or `ghost`, G3's fallback (run 3, experiments/E2-v3.yaml):
- the ghost ceiling L_G(s, f) (e2_frontend.ghost_ceiling: L_B without B1) plans and decides the K-study, at
  L_K = min(cap, L_G), on the ghosts' oracle (_ghost_oracle: their twin differences on the 20 ms grid);
- oracle units keep the ghost record nearest each 20 ms grid point after the onset (oracle_records);
- check C-1 on every mode A grid unit (ghost_check; reported, deciding nothing);
- the Route B arm reported beside it (route_b_arm: L_B, L_B under s29's J_max_TA, the real satellites' K-study);
- G3's clauses as R-2 states them (g3_clauses_ghost), in the summary.

Every twin is a unit in run_dir/units/<name>/ whose done.json records the code version and its files' hashes;
`gnssaudit run --resume` redoes only units that are not done. Twins live in /dev/shm (D-10), at most `pipelines` in
flight; the mode A clean twin of each segment is kept on disk while the grid needs it, because B1's interference is
read as jammed minus clean bytes. Real satellites' truth is formed on observable epochs with the onset placed by the
clock map (docs/3 section 2.6); ghosts' truth is the twin difference at every record (they have no observables).
"""

from __future__ import annotations

import json
import math
import shutil
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from gnssaudit.experiments import Outcome
from gnssaudit.experiments.e0b import (
    DEFAULT_BW_HZ,
    OBS_INTERVAL_S,
    SLIP_CYC,
    T_S,
    choose_ghost,
    epoch_label,
    lock_report,
    recording_source,
    strongest,
    survey_satellites,
)
from gnssaudit.experiments.e2_frontend import (
    NS,
    FrontEndCell,
    align_records,
    cell_verdict,
    combine_segments,
    finite_cell,
    front_end_cell,
    front_end_jmax,
    ghost_ceiling,
    jammer_slope,
    pair_on_time,
    paired_excess,
    route_b_ceiling,
)
from gnssaudit.experiments.twin import (
    SHM_MARGIN,
    SHM_ROOT,
    LockPolicy,
    Twin,
    common_obs,
    inject_twin,
    jsonable,
    receive,
    receiver_spec,
    run_twin,
    runs_of,
    sigma_disc_cyc,
    sigma_reported_cyc,
    track_as_obs,
)
from gnssaudit.gates import GateFailure, GateReport, identity_gate, null_gate, repeat_gate
from gnssaudit.inject import (
    Disturbance,
    FrontEndSpec,
    GhostSpec,
    InjectionReport,
    IQSource,
    QuantSpec,
    clean_twin,
    disturbance_seed,
    estimate_analog_sd,
    estimate_n0,
    gain,
    int16_scale,
    iter_iq,
)
from gnssaudit.inject.families import Kind
from gnssaudit.inject.pipeline import FULL_SCALE
from gnssaudit.io.fairdata import require_free
from gnssaudit.io.fetch import commit_part
from gnssaudit.io.gnsssdr import Channel
from gnssaudit.io.schemas import TRUTH_EPOCH_SAT
from gnssaudit.paths import results_root
from gnssaudit.provenance import git_describe, sha256_file
from gnssaudit.sdr import RunArtifacts, SourceSpec
from gnssaudit.truth.ghost import NoAmbiguityWindow, ghost_projection, ghost_truth, phase_precision_ok
from gnssaudit.truth.oracle import NoCommonEpochs, k_convergence, oracle_moments
from gnssaudit.truth.rxtime import TwinClock
from gnssaudit.truth.twin import twin_difference
from gnssaudit.uq.loop import CarrierLoop

REPO = Path(__file__).resolve().parents[3]
FloatArray = NDArray[np.float64]
SETTLE_S = 2.0  # after the onset, before variances and residuals are read
PEAK_SHARE = 0.5  # of int16 full scale, for the recording's largest component
FLOOR_BW_FRACTION = 0.9  # the thermal block's common floor is white over this share of the native band
JAMMERS = ("noise", "cw", "swept_cw", "chirp")
BREAKS = ("reacquisition", "ambiguous_slip")  # exclusions after which a ghost has no truth (lost, not error)
TAU_SIDE_SEGMENT = "A"  # the tau side check is segment A's (docs/8 section 5, rule (a'))
K_CHANGE_MAX = 0.2  # K = 51 if the median |sigma^2_101 / sigma^2_51 - 1| is at most this (docs/5 section 4.4)
# A jammer seed's configuration is 100 x block + the family's sorted index; the ghosts use 0, the mode B dither 800 and
# the thermal floors 900 (realization r, index 2; a thermal jammer's index is its level, at least 50 dB-Hz).
GRID_BLOCK, K_BLOCK0, THERMAL_BLOCK = 1, 2, 9
GHOST_ABS = "ghost_abs_ch{}.parquet"  # a clean twin's absolute ghost error, e_c of the paired excess
Truth = Literal["route_b", "ghost"]
TRUTHS: tuple[Truth, ...] = ("route_b", "ghost")  # the spec's `truth`; route_b, the default, is E2-v2's
G3_FAMILIES = ("noise", "swept_cw")  # clause 3's families under ghost truth (R-2); the chirp is reported
G3_HZ_SOURCE = ("E0b/fee767ba3538", "E0b/46b76d4b47d9")  # clause 2: G-hz as passed on E0, not re-run (R-2, ruling P3)


# ------------------------------------------------------------------------------------------------- the plan
@dataclass(frozen=True)
class Unit:
    """One twin: what is injected (block, family, level, seeds, timing, floor), through which front end, and the clean
    twin it is differenced against (None for a clean twin)."""

    name: str
    segment: str
    block: str  # clean | grid | tau | k-cond | k-marg | k-noise | thermal | thermal-clean
    mode: str  # A | B
    tau_s: float | None  # mode B's AGC time constant
    family: str | None
    level: float | None
    seed: int  # the jammer's seed (paired across modes)
    clean: str | None
    offset_s: float | None = None  # fixed sweep timing (the conditional oracle, F8)
    floor_seed: int | None = None  # the common white floor of a thermal pair
    keep_bytes: bool = False  # kept on disk for B1's paired difference
    project: bool = False  # B1's first-order prediction for the ghosts
    full_ghost: bool = True  # ghost truth at every record (else at observable epochs)


def _tau_tag(tau_s: float | None) -> str:
    return "" if tau_s is None else f"{tau_s * 1000:g}"


def _clean(segment: str, mode: str, tau_s: float | None, suffix: str = "") -> Unit:
    name = f"{segment}-clean-{mode}{_tau_tag(tau_s)}{suffix}"
    return Unit(name, segment, "clean", mode, tau_s, None, None, 0, None, keep_bytes=mode == "A" and not suffix)


def _cell(spec: Mapping[str, Any], family: str, level: float, realization: int, block: int) -> int:
    """A jammer's seed, keyed on the family's place among the sorted family names: the manifest keeps the spec as
    sorted JSON, and two specs that only reorder their families share a run id, so no seed may follow mapping order
    (E2's first run keyed on the YAML's order; e2post reads its seeds from the unit records). A family index of 100
    would reach the next block's configurations: refused."""
    idx = sorted(spec["families"]).index(family)
    if idx >= 100:
        raise ValueError(f"{len(spec['families'])} families: the seed scheme holds at most 100 families per block")
    return disturbance_seed(int(spec["seed"]), 100 * block + idx, realization, int(level))


def plan_grid(spec: Mapping[str, Any], segment: str) -> list[Unit]:
    """The clean twins and the grid of one segment and, on segment A, the tau side check (docs/8 section 5): mode B at
    the side time constants at side_level_db_hz, each on the jammer draw of the mode A grid unit it is paired with."""
    mb = spec["mode_b"]
    tau = float(mb["tau_s"])
    side_tau = [float(t) for t in mb.get("side_tau_s", [])] if segment == TAU_SIDE_SEGMENT else []
    side_fams = list(mb.get("side_families", [])) if side_tau else []
    if side_fams and float(mb["side_level_db_hz"]) not in {float(v) for v in spec["levels_db_hz"]}:
        raise ValueError(
            f"the tau side level {mb['side_level_db_hz']} is not a grid level: no mode A unit to pair with"
        )
    out = [_clean(segment, "A", None), _clean(segment, "A", None, "2"), _clean(segment, "B", tau)]
    out += [_clean(segment, "B", t) for t in side_tau]
    for fam in spec["families"]:
        for lv in spec["levels_db_hz"]:
            seed = _cell(spec, fam, float(lv), 0, GRID_BLOCK)
            for mode, t in (("A", None), ("B", tau)):
                name = f"{segment}-grid-{fam}-{float(lv):g}-{mode}{_tau_tag(t)}"
                out.append(Unit(name, segment, "grid", mode, t, fam, float(lv), seed,
                                f"{segment}-clean-{mode}{_tau_tag(t)}", project=mode == "A"))  # fmt: skip
    for fam in side_fams:
        side_lv = float(mb["side_level_db_hz"])
        seed = _cell(spec, fam, side_lv, 0, GRID_BLOCK)  # the grid's draw at that level: the side check moves tau
        for t in side_tau:
            out.append(Unit(f"{segment}-tau-{fam}-{side_lv:g}-B{_tau_tag(t)}", segment, "tau", "B", t, fam, side_lv,
                            seed, f"{segment}-clean-B{_tau_tag(t)}"))  # fmt: skip
    return out


def k_level(cap_db_hz: float, ceiling_db_hz: float | None) -> float | None:
    """L_K = min(cap, the family's ceiling: L_B, or L_G under ghost truth); None when the family has none."""
    return None if ceiling_db_hz is None else min(cap_db_hz, ceiling_db_hz)


@dataclass(frozen=True)
class OracleBlock:
    """Where a K-study block or the thermal block runs: its family at L_K, K realizations, a conditional block's fixed
    sweep timing (None: drawn), and whether it moved to the fallback family."""

    family: str
    level: float
    k: int
    offset_s: float | None
    fallback: bool


def oracle_blocks(
    spec: Mapping[str, Any], ceiling: Mapping[str, float | None]
) -> tuple[dict[str, OracleBlock], OracleBlock | None]:
    """(the K-study's blocks by tag, the thermal block) at L_K(f) = min(cap, L_B(f)) of the ceiling combined over the
    segments (docs/5 section 4.4). A block whose family has no L_K is not run, except that the conditional (K = 101)
    block and the thermal block move to kstudy.fallback_family at its L_K, with no fixed timing, when theirs has none;
    with no L_K there either, they are not run."""
    ks, th = spec.get("kstudy"), spec.get("thermal")
    cap = float(ks["level_cap_db_hz"]) if ks else math.inf
    fallback = str(ks["fallback_family"]) if ks and ks.get("fallback_family") else None
    offset = float(ks["offset_s"]) if ks else None

    def place(family: str, k: int, off: float | None, movable: bool) -> OracleBlock | None:
        lv = k_level(cap, ceiling.get(family))
        if lv is not None:
            return OracleBlock(family, lv, k, off, False)
        lv = k_level(cap, ceiling.get(fallback)) if movable and fallback else None
        return None if lv is None or fallback is None else OracleBlock(fallback, lv, k, None, True)

    blocks: dict[str, OracleBlock] = {}
    for tag, b in sorted(ks["blocks"].items()) if ks else []:
        cond = bool(b.get("conditional"))
        if (blk := place(str(b["family"]), int(b["K"]), offset if cond else None, cond)) is not None:
            blocks[tag] = blk
    thermal = place(str(th["family"]), int(th["K"]), offset, True) if th else None
    return blocks, thermal


def plan_oracle(spec: Mapping[str, Any], segment: str, ceiling: Mapping[str, float | None]) -> list[Unit]:
    """The K-study and thermal blocks of one segment where oracle_blocks puts them (blocks it leaves out are not
    planned; the summary says so). Every K-study block is differenced against the segment's mode A clean twin; a
    block's seeds key on its tag's place among the sorted tags, below the thermal block's number (more K-study blocks
    than fit are refused)."""
    out: list[Unit] = []
    ks, th = spec.get("kstudy"), spec.get("thermal")
    if ks and K_BLOCK0 + len(ks["blocks"]) > THERMAL_BLOCK:
        raise ValueError(f"{len(ks['blocks'])} K-study blocks: the seed scheme holds {THERMAL_BLOCK - K_BLOCK0}")
    blocks, thermal = oracle_blocks(spec, ceiling)
    if ks and ks["segment"] == segment:
        tags = sorted(ks["blocks"])
        for tag, b in blocks.items():
            for r in range(b.k):
                out.append(Unit(f"{segment}-k-{tag}-r{r}", segment, f"k-{tag}", "A", None, b.family, b.level,
                                _cell(spec, b.family, b.level, r + 1, K_BLOCK0 + tags.index(tag)), f"{segment}-clean-A",
                                offset_s=b.offset_s, full_ghost=False))  # fmt: skip
    if th and th["segment"] == segment and thermal is not None:
        for r in range(thermal.k):
            floor = disturbance_seed(int(spec["seed"]), 900, r, 2)
            clean = Unit(f"{segment}-thermal-clean-r{r}", segment, "thermal-clean", "A", None, None, None, 0, None,
                         floor_seed=floor)  # fmt: skip
            out.append(clean)
            out.append(Unit(f"{segment}-thermal-r{r}", segment, "thermal", "A", None, thermal.family, thermal.level,
                            _cell(spec, thermal.family, thermal.level, r + 1, THERMAL_BLOCK), clean.name,
                            offset_s=thermal.offset_s,
                            floor_seed=floor, full_ghost=False))  # fmt: skip
    return out


def pick_repeats(names: Sequence[str], *, fraction: float, seed: int) -> list[str]:
    """A seeded draw of ceil(fraction x n) units, re-run for G-repeat."""
    k = math.ceil(fraction * len(names))
    return sorted(np.random.default_rng(seed).choice(sorted(names), size=k, replace=False).tolist())


# --------------------------------------------------------------------------------------------- the rules
def truth_mode(spec: Mapping[str, Any]) -> Truth:
    """The spec's `truth` (P2c): `route_b` when absent (E2-v2's rules, unchanged) or `ghost` (R-1 to R-3). Anything
    else is refused, so that a misspelt key does not run as E2-v2."""
    truth = spec.get("truth", "route_b")
    if truth not in TRUTHS:
        raise ValueError(f"truth {truth!r} is not one of {', '.join(TRUTHS)}")
    return cast(Truth, truth)


def ceiling_field(spec: Mapping[str, Any]) -> str:
    """The per-segment ceiling the K-study is planned and decided from: L_G under ghost truth (R-2), else L_B."""
    return "ghost_ceiling_db_hz" if truth_mode(spec) == "ghost" else "ceiling_db_hz"


def nearest_on_grid(t: FloatArray) -> NDArray[np.bool_]:
    """Per record of one channel: whether it is the record nearest its 20 ms grid point round(t / 0.02) 0.02, within
    T_S / 2 of it (of two equally near, the first). At most one record per grid point; a grid point may have none."""
    grid = np.rint(t / OBS_INTERVAL_S)
    off = np.abs(t - grid * OBS_INTERVAL_S)
    idx = np.flatnonzero(off <= T_S / 2)
    idx = idx[np.lexsort((idx, off[idx], grid[idx]))]  # by grid point, then distance, then record order
    first = np.ones(idx.size, dtype=bool)
    first[1:] = grid[idx][1:] != grid[idx][:-1]
    out = np.zeros(t.size, dtype=bool)
    out[idx[first]] = True
    return out


def oracle_records(t: FloatArray, onset: float, pre: float, *, truth: Truth) -> NDArray[np.bool_]:
    """Which records of one ghost's twin difference an oracle unit (K-study, thermal) keeps: every record from
    onset - pre to the onset, and after it one record per 20 ms grid point. Under ghost truth that is the record nearest
    the grid point (nearest_on_grid; the controller's ruling Q1 of P2c task 3). Under route_b it stays E2-v2's
    selection, the records at or up to T_S / 2 after a grid point, so that E2-v2's outputs do not change: a ghost whose
    records fall just before the grid keeps none after the onset (E2K's first ghost on segment A, 0.05 to 0.12 ms
    before it). Only those stored files were affected; no recorded result reads them."""
    if truth == "ghost":
        on_grid = nearest_on_grid(t)
    else:
        on_grid = np.isclose(np.mod(t + 1e-9, OBS_INTERVAL_S), 0.0, atol=T_S / 2)
    return (on_grid | (t < onset)) & (t >= onset - pre)


def j_max(ratio: Mapping[float, float], *, tol: float) -> float | None:
    """The largest level up to which every level has |Var_B / Var_A - 1| < tol (docs/8 section 5)."""
    best = None
    for lv in sorted(ratio):
        r = ratio[lv]
        if not (np.isfinite(r) and abs(r - 1.0) < tol):
            break
        best = lv
    return best


def choose_k(convergence: Mapping[str, Any]) -> int:
    """docs/5 section 4.4, as pre-registered: K = 51 if the median |sigma^2_101 / sigma^2_51 - 1| <= 0.2, else 101.
    k_convergence's own bar, sqrt(2 / (K_small - 1)), is that 0.2 exactly at K_small = 51; the rule's 0.2 decides."""
    return 51 if float(convergence["median_rel_change"]) <= K_CHANGE_MAX else 101


def residual_ratio(dphi: FloatArray, prediction: FloatArray) -> float:
    """r = Var(dPhi - prediction) / Var(dPhi): what B1's first order leaves (docs/3 section 2.2)."""
    return float(np.var(dphi - prediction) / np.var(dphi))


def thermal_floor(cn0_dbhz: float, t_s: float) -> float:
    """B1's residual floor for an atan (Costas) discriminator: 1 / (C/N0 T), the rule's J-independent thermal term. The
    discriminator normalises by the noisy in-phase prompt, so thermal noise times interference is first order in the
    interference. P2b task 11's two-loop simulation (review/s26_b1_floor.py) gave r = 1 / (C/N0 T) + var(n_iota) at
    any loop bandwidth for white n_iota, and the CTTC binary agreed. On FGI that sum does not hold for any family
    (P2b part 2, task 3, run a22677d7f2ad): the cross term is white, so the loop passes it at the white gain
    while n_iota is weighted by its own spectrum (docs/3 section 2.2; the derivation is in the docstring of
    e2post.decomposition). The value here is the rule's floor, r - 1 / (C/N0 T) < 0.05, not a prediction of r."""
    return 1.0 / (10 ** (cn0_dbhz / 10) * t_s)


def b1_prediction(n_iota: FloatArray, t_rec: FloatArray, t: FloatArray, onset: float, pre: float) -> FloatArray:
    """B1's first-order prediction of a ghost's dPhi in cycles (docs/3 section 2.2), the one reduce_unit saves and
    e2post rebuilds: H n_iota through the default loop (a record without n_iota counts as 0), n_iota per record of the
    clean twin (times t_rec), at the truth's record times t, less its mean over [onset - pre, onset)."""
    pred = CarrierLoop.gnsssdr(DEFAULT_BW_HZ, T_S).filter(np.nan_to_num(n_iota)) / (2 * np.pi)
    at = np.interp(t, t_rec, pred)
    ref = (t >= onset - pre) & (t < onset)
    return at - (float(np.mean(at[ref])) if ref.any() else 0.0)


def b1_residual(
    e: FloatArray, pred: FloatArray, n_iota: FloatArray, t_rec: FloatArray, t: FloatArray, *, cn0_clean_dbhz: float
) -> dict[str, Any]:
    """B1's record for one ghost on r's epochs (e, pred and their times t): r, the floor 1 / (C/N0 T) at the clean
    ghost's median C/N0 (the rule's bar is r - floor), and, reported alongside, Var(n_iota) (rad^2) on the records r
    is read on (n_iota per clean record at t_rec, joined on the record time to 1 ns, records without n_iota left out)
    and the formula floor + Var(n_iota) (docs/3 section 2.2)."""
    floor = thermal_floor(cn0_clean_dbhz, T_S)
    on = np.isin(np.rint(t_rec * NS).astype(np.int64), np.rint(t * NS).astype(np.int64)) & np.isfinite(n_iota)
    var_n = float(np.var(n_iota[on])) if on.any() else float("nan")
    return {"r": residual_ratio(e, pred), "floor": floor, "cn0_clean_dbhz": cn0_clean_dbhz,
            "corr": float(np.corrcoef(e, pred)[0, 1]), "n": int(e.size), "var_n_iota_rad2": var_n,
            "floor_plus_var": floor + var_n, "n_iota_records": int(on.sum())}  # fmt: skip


def segment_jmax(
    cells: Mapping[float, FrontEndCell | None], *, rules: Mapping[str, Any], cap: float = math.inf, name: str = ""
) -> float | None:
    """The front-end J_max(s, f) (docs/8 section 5): the pre-registered scan (front_end_jmax) over the grid levels at
    or below `cap` (the tau side check's) and below the first level without a verdict (None: a cell that could not be
    formed). Above such a level "no fail at any level at or below" cannot be shown; and J_max stays a level that
    passes, also under the cap."""
    scan: dict[float, FrontEndCell] = {}
    for lv in sorted(cells):
        c = cells[lv]
        if lv > cap or c is None:
            break
        scan[lv] = c
    return front_end_jmax(scan, tol=float(rules["jmax_tol"]), slope_tol=float(rules["slope_tol"]),
                          se_max=float(rules["excess_se_max"]), cell=name or "front-end")  # fmt: skip


def side_cap(levels: Sequence[float], side_level: float) -> float:
    """What a failed tau side cell caps J_max(A, f) at: the largest grid level below the side level (-inf: none)."""
    return max((float(v) for v in levels if float(v) < side_level), default=-math.inf)


def pair_difference(jammed: Path, clean: Path, scale: float, *, block: int = 1 << 22) -> Iterator[tuple[int, Any]]:
    """(first output index, (jammed - clean) / scale) of two int16 twins: the decimated interference in input counts
    (recording and ghosts cancel; each twin's rounding leaves at most one count per component)."""
    with jammed.open("rb") as fj, clean.open("rb") as fc:
        m0 = 0
        while True:
            a = np.frombuffer(fj.read(4 * block), dtype="<i2").astype(np.float64)
            b = np.frombuffer(fc.read(4 * block), dtype="<i2").astype(np.float64)
            if a.size == 0:
                return
            if a.size != b.size:
                raise ValueError("the twins differ in length")
            d = (a - b) / scale
            yield m0, d[0::2] + 1j * d[1::2]
            m0 += d.size // 2


def mark_done(unit_dir: Path, describe: str, files: Sequence[str]) -> None:
    """done.json: the code version and every file's SHA-256, written last (resume validates it) and made durable
    (fsync, rename, fsync of the directory), so that a crash cannot leave it empty under its name."""
    rec = {"describe": describe, "files": {f: sha256_file(unit_dir / f) for f in files}}
    tmp = unit_dir / "done.json.part"
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    commit_part(tmp, unit_dir / "done.json")


def is_done(unit_dir: Path, describe: str) -> bool:
    """The unit is finished under `describe`: its done.json names that code version and every file still hashes as
    recorded. A done.json that does not decode (torn by a crash before this was fsynced) is not done: the unit is
    redone."""
    try:
        rec = json.loads((unit_dir / "done.json").read_text())
    except FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError:
        return False
    if rec["describe"] != describe:
        return False
    return all((unit_dir / f).exists() and sha256_file(unit_dir / f) == h for f, h in rec["files"].items())


# ------------------------------------------------------------------------------------------- the segments
@dataclass
class Segment:
    """A registered interval with what every twin of it shares: clock, floor, channels, ghosts, scale and fronts."""

    key: str
    source: IQSource
    clock: TwinClock
    n0: float
    line_excess_db: float
    agc_rms: float
    analog_sd: float  # SD of the Gaussian noise the recorder quantised (mode B's reconstruction)
    channels: dict[int, Channel]
    ghost_ch: list[int]
    ghosts: list[Disturbance]
    scale: float
    t0: float  # the segment's start in recording time
    expected_doppler_hz: dict[int, float]  # per PRN at the end of the warm-up: the survey's, or the ghost's set one

    @property
    def real(self) -> list[int]:
        return [ch for ch in sorted(self.channels) if ch not in self.ghost_ch]


@dataclass
class Context:
    spec: dict[str, Any]
    run_dir: Path
    work: Path
    describe: str
    rx: Any
    segments: dict[str, Segment]
    cleans: dict[str, Twin]  # clean units' twins, reloaded on resume
    runs: dict[str, RunArtifacts]
    reports: dict[str, InjectionReport]

    def unit_dir(self, name: str) -> Path:
        return self.run_dir / "units" / name

    @property
    def warm(self) -> float:
        return float(self.spec["warmup_s"])


def _segment_survey(ctx: Context, key: str) -> Segment:
    """The stock survey of one segment (cached in units/survey-<key>/survey.json): channels, ghosts, floor, AGC RMS."""
    spec = ctx.spec
    seg_spec = {"recording": spec["segments"][key], "decim": spec.get("decim", 1), "fir_taps": spec.get("fir_taps", 1)}
    source, clock = recording_source(seg_spec)
    udir = ctx.unit_dir(f"survey-{key}")
    if is_done(udir, ctx.describe):
        s = json.loads((udir / "survey.json").read_text())
    else:
        shutil.rmtree(udir, ignore_errors=True)
        udir.mkdir(parents=True)
        floor = estimate_n0(source, frontend_bw_hz=float(spec["frontend_bw_hz"]))
        power, peak = 0.0, 0.0
        for _, x in iter_iq(source, 1 << 22):
            power += float(np.sum(np.abs(x) ** 2))
            peak = max(peak, float(np.max(np.abs(x.real))), float(np.max(np.abs(x.imag))))
        agc_rms = math.sqrt(power / (source.sample_end - source.sample_start))
        q = spec["quant"]
        analog_sd = estimate_analog_sd(source, QuantSpec(int(q["max_level"]), bool(q["zero_level"])))
        # the floor and the top J/N0 set the scale (docs/8 section 5); the recording's largest component (a start-up
        # transient on CTTC) must also stay at half of full scale
        scale = min(int16_scale(floor.n0_per_hz, source.fs_hz, max(float(v) for v in spec["levels_db_hz"])),
                    PEAK_SHARE * FULL_SCALE / peak)  # fmt: skip
        fe = FrontEndSpec(clock.decim, int(spec.get("fir_taps", 1)), scale)
        template = str(spec.get("template", "1C"))
        free: dict[int, Channel] = {ch: ("G", template, 0) for ch in range(int(spec.get("survey_channels", 12)))}
        rx = receiver_spec({"stock_receiver": True}, template=template)
        tw = run_twin(f"survey-{key}", source, [], fe, free, ctx.work, floor.n0_per_hz, float(spec["timeout_s"]), rx,
                      clock=clock, warmup_s=ctx.warm, workers=inject_workers(spec, "A"))  # fmt: skip
        ctx.runs[f"survey-{key}"] = tw.art
        _keep_unit_logs(ctx, f"survey-{key}", udir)
        in_view, cn0 = survey_satellites(tw, clock.sample_start / source.fs_hz + ctx.warm)
        g = spec["ghosts"]
        sats = strongest(cn0, min_cn0_dbhz=float(spec.get("min_cn0_dbhz", 35.0)),
                         n=int(spec["max_channels"]) - len(g["doppler_hz"]))  # fmt: skip
        ghosts, taken = [], dict(in_view)
        for dop in g["doppler_hz"]:
            prn, d = choose_ghost(taken, doppler_hz=float(dop))
            ghosts.append({"prn": prn, "doppler_hz": d})
            taken[prn] = d  # the next ghost stays clear of this one too
        s = {"sats": sats, "ghosts": ghosts, "in_view": in_view, "cn0": cn0, "n0": floor.n0_per_hz,
             "line_excess_db": floor.line_excess_db, "agc_rms": agc_rms, "analog_sd": analog_sd,
             "scale": scale}  # fmt: skip
        (udir / "survey.json").write_text(json.dumps(jsonable(s), indent=1, sort_keys=True) + "\n")
        mark_done(udir, ctx.describe, ["survey.json"])
    g = spec["ghosts"]
    seed = int(spec["seed"])
    gd = []
    for i, x in enumerate(s["ghosts"]):
        params = {"prn": int(x["prn"]), "doppler_hz": float(x["doppler_hz"]), "doppler_rate_hzps": 0.0,
                  "chip0": 17.3 + 101.0 * i}  # fmt: skip
        gd.append(Disturbance("ghost", params, 0.0, 0.0, float(g["cn0_db_hz"]), disturbance_seed(seed, 0, 0, i)))
    template = str(spec.get("template", "1C"))
    sats = [int(p) for p in s["sats"]]
    channels: dict[int, Channel] = {
        i: ("G", template, p) for i, p in enumerate([*sats, *(int(d.params["prn"]) for d in gd)])
    }
    ghost_ch = list(range(len(sats), len(channels)))
    t0 = clock.sample_start / source.fs_hz
    expected = {**{int(p): float(d) for p, d in s["in_view"].items() if int(p) in sats},
                **{int(x["prn"]): float(x["doppler_hz"]) for x in s["ghosts"]}}  # fmt: skip
    return Segment(key, source, clock, float(s["n0"]), float(s["line_excess_db"]), float(s["agc_rms"]),
                   float(s["analog_sd"]), channels, ghost_ch, gd, float(s["scale"]), t0, expected)  # fmt: skip


def disturbances(ctx: Context, u: Unit) -> list[Disturbance]:
    """The unit's disturbance list: the segment's ghosts, its jammer (zero gain on a clean twin: D-10), and a common
    floor for a thermal pair. Onsets are recording time."""
    spec, seg = ctx.spec, ctx.segments[u.segment]
    onset = seg.t0 + float(spec["onset_s"])
    out = list(seg.ghosts)
    if u.floor_seed is not None:
        bw = FLOOR_BW_FRACTION * seg.source.fs_hz
        out.append(Disturbance("noise", {"bw_hz": bw, "f_hz": 0.0, "common": 1.0}, 0.0, 0.0, 10 * math.log10(bw),
                               u.floor_seed))  # fmt: skip
    if u.family is None:  # a clean twin: the reference jammer at zero gain
        fam = next(iter(spec["families"]))
        out.append(Disturbance(jammer_kind(fam), dict(spec["families"][fam]), onset, 0.01, -math.inf, 1))
        return clean_twin(out)
    params = {k: float(v) for k, v in spec["families"][u.family].items()}
    if u.offset_s is not None:
        params["offset_s"] = u.offset_s
    out.append(Disturbance(jammer_kind(u.family), params, onset, 0.01, float(u.level or 0.0), u.seed))
    return out


def jammer_kind(family: str) -> Kind:
    """A family name of the spec as an additive jammer kind (docs/8 section 3); anything else is refused."""
    if family not in JAMMERS:
        raise ValueError(f"{family!r} is not an additive jammer family ({', '.join(JAMMERS)})")
    return cast(Kind, family)


def inject_workers(spec: Mapping[str, Any], mode: str) -> int:
    """Processes per injection: `workers_a` for a mode A twin, `workers_b` for a mode B twin, 1 when the key is absent.
    The output is bitwise the same for any count (`inject`)."""
    return int(spec.get("workers_a" if mode == "A" else "workers_b", 1))


def front_end(ctx: Context, u: Unit) -> FrontEndSpec:
    seg = ctx.segments[u.segment]
    taps = int(ctx.spec.get("fir_taps", 1))
    if u.mode == "A":
        return FrontEndSpec(seg.clock.decim, taps, seg.scale)
    q = ctx.spec["quant"]
    seed = disturbance_seed(int(ctx.spec["seed"]), 800, 0, 0)  # the analog reconstruction: one per segment's twins
    return FrontEndSpec(seg.clock.decim, taps, seg.scale, mode="B", agc_tau_s=u.tau_s,
                        quant=QuantSpec(int(q["max_level"]), bool(q["zero_level"])), dither_seed=seed,
                        analog_sd=seg.analog_sd)  # fmt: skip


# ------------------------------------------------------------------------------------------------ one unit
def _write(unit_dir: Path, name: str, table: pa.Table) -> str:
    pq.write_table(table, unit_dir / name, compression="zstd")
    return name


def _load_twin(unit_dir: Path, name: str) -> Twin:
    """A clean twin's tracks and observables, as saved (full per-integration streams, D-18)."""
    tracks = {}
    for p in sorted(unit_dir.glob("track_ch*.parquet")):
        tracks[int(p.stem.removeprefix("track_ch"))] = pq.read_table(p)
    obs = pq.read_table(unit_dir / "obs.parquet") if (unit_dir / "obs.parquet").exists() else None
    rep = json.loads((unit_dir / "unit.json").read_text())
    art = RunArtifacts(unit_dir, rep["config_sha256"], unit_dir / "gnss-sdr.log", 0, {}, None, float(rep["rx_wall_s"]))
    return Twin(name, None, art, tracks, obs)


SHM = Path("/dev/shm")


class ShmBudget:
    """The /dev/shm guard of the twins this process writes, `pipelines` at a time (final review R1, I4). A twin being
    written is not yet all in the file system's usage, so each check also counts the bytes held by the twins other
    pipelines are still writing: units that ask at once cannot all pass with room for one. A hold ends when its twin is
    written or its writing fails. It over-counts what those twins have already written, so the guard errs towards
    refusing."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._held = 0

    @contextmanager
    def hold(self, nbytes: int) -> Iterator[None]:
        with self._lock:
            require_free(SHM, self._held + nbytes, reserve=SHM_MARGIN)
            self._held += nbytes
        try:
            yield
        finally:
            with self._lock:
                self._held -= nbytes


SHM_BUDGET = ShmBudget()


def _inject_receive(ctx: Context, u: Unit, *, keep: Path | None = None) -> tuple[Twin, SourceSpec | None]:
    """Inject and receive one twin. With `keep` its bytes are moved there (disk); otherwise the caller gets the shm
    source and deletes it once done (B1's projection reads it first). The twin is written under SHM_BUDGET's guard,
    and a unit that fails after the injection deletes it."""
    seg = ctx.segments[u.segment]
    fe = front_end(ctx, u)
    n_out = (seg.source.sample_end - seg.source.sample_start) // seg.clock.decim
    workers = inject_workers(ctx.spec, u.mode)
    with SHM_BUDGET.hold(4 * n_out):
        shutil.rmtree(ctx.work / u.name, ignore_errors=True)
        for k in range(1, 4):
            shutil.rmtree(ctx.work / f"{u.name}~{k}", ignore_errors=True)
        src, report = inject_twin(u.name, seg.source, disturbances(ctx, u), fe, ctx.work, seg.n0, workers=workers)
    try:
        tw = receive(u.name, src, seg.channels, ctx.work, float(ctx.spec["timeout_s"]), ctx.rx, clock=seg.clock,
                     warmup_s=ctx.warm, t_int_s=T_S, expected_doppler_hz=seg.expected_doppler_hz,
                     lock_policy=LockPolicy(frozenset(seg.ghost_ch),
                                            int(ctx.spec.get("min_locked_real", 4))))  # fmt: skip
        tw.report = report
        for ch in tw.unlocked:  # still unlocked after the last attempt: no pair is formed (G-warmup-lock reports it)
            tw.tracks.pop(ch, None)
        if keep is not None:
            require_free(keep.parent, src.path.stat().st_size)
            shutil.move(src.path, keep)
            return tw, None
    except BaseException:
        src.path.unlink(missing_ok=True)  # re-raised below; no later guard counts a failed unit's twin
        raise
    return tw, src


def _unit_record(tw: Twin, report: InjectionReport) -> dict[str, Any]:
    return {"config_sha256": tw.art.config_sha256, "rx_wall_s": tw.art.wall_s, "attempts": tw.attempts,
            "unlocked": tw.unlocked, "lock_ok": tw.lock_ok, "output_sha256": report.output_sha256,
            "clip_count": report.clip_count,
            "achieved_db_hz": report.achieved_db_hz, "inject_s": report.timings_s.get("total")}  # fmt: skip


def ghost_errors(seg: Segment, tracks: Mapping[int, pa.Table], *, converge_s: float, name: str) -> dict[int, pa.Table]:
    """e_c per ghost channel of a clean twin: ghost truth on its own track, the absolute error the paired excess
    pairs with dPhi (docs/8 section 5, rule (a')), converged after converge_s as E2post forms it."""
    return {ch: ghost_truth(tracks[ch], GhostSpec.of(gd), clock=seg.clock, converge_s=converge_s,
                            truth_id=f"{name}-g{ch}")
            for ch, gd in zip(seg.ghost_ch, seg.ghosts, strict=True) if ch in tracks}  # fmt: skip


def run_clean(ctx: Context, u: Unit) -> None:
    """A clean twin: its tracks and observables and, for the grid's clean twins, each ghost's absolute error
    (ghost_abs_ch*.parquet, a done-file like the rest). Thermal pairs' clean twins need none."""
    udir = ctx.unit_dir(u.name)
    if is_done(udir, ctx.describe):
        ctx.cleans[u.name] = _load_twin(udir, u.name)
        kept = ctx.work / f"{u.name}.bin"
        if u.keep_bytes:  # a resume: the kept bytes are reused only as recorded, else made again
            recorded = json.loads((udir / "unit.json").read_text())["output_sha256"]
            if kept.exists() and sha256_file(kept) != recorded:  # left short or torn by a crash
                kept.unlink()
            if not kept.exists():
                tw, _ = _inject_receive(ctx, u, keep=kept)
                if tw.report is None or tw.report.output_sha256 != recorded:
                    raise RuntimeError(f"{u.name}: re-injected bytes differ from the recorded twin (G-repeat)")
        return
    shutil.rmtree(udir, ignore_errors=True)
    udir.mkdir(parents=True)
    tw, src = _inject_receive(ctx, u, keep=ctx.work / f"{u.name}.bin" if u.keep_bytes else None)
    if src is not None:
        src.path.unlink(missing_ok=True)
    assert tw.report is not None
    files = [_write(udir, f"track_ch{ch}.parquet", t) for ch, t in sorted(tw.tracks.items())]
    if u.block == "clean":
        errors = ghost_errors(ctx.segments[u.segment], tw.tracks, converge_s=ctx.warm / 2, name=u.name)
        files += [_write(udir, GHOST_ABS.format(ch), t) for ch, t in sorted(errors.items())]
    if tw.obs is not None:
        files.append(_write(udir, "obs.parquet", tw.obs))
    rec = _unit_record(tw, tw.report)
    (udir / "unit.json").write_text(json.dumps(jsonable(rec), indent=1, sort_keys=True) + "\n")
    _keep_unit_logs(ctx, u.name, udir)
    mark_done(udir, ctx.describe, [*files, "unit.json"])
    ctx.cleans[u.name] = tw
    ctx.runs.update(runs_of([tw]))


def _keep_unit_logs(ctx: Context, name: str, udir: Path) -> None:
    for run in [ctx.work / name, *sorted(ctx.work.glob(f"{name}~*"))]:
        if (run / "gnss-sdr.log").exists():
            shutil.copy(run / "gnss-sdr.log", udir / f"{run.name}.gnss-sdr.log")
        shutil.rmtree(run, ignore_errors=True)


def _post(t: FloatArray, onset: float) -> NDArray[np.bool_]:
    return t >= onset + SETTLE_S


def reduce_unit(
    ctx: Context, u: Unit, tw: Twin, clean: Twin, src: SourceSpec | None
) -> tuple[list[str], dict[str, Any]]:
    """A jammed twin against its clean twin: identity per channel, real satellites' truth on observable epochs (onset
    by the clock map), ghosts' twin difference per record (and B1's prediction on projected units), post-onset
    variances, the stock lock verdict, lost ghost truth; under ghost truth, check C-1 per ghost of a mode A grid unit
    (ghost_check), here while the jammed tracks exist (the dumps go after the unit, D-18)."""
    spec, seg = ctx.spec, ctx.segments[u.segment]
    truth_kind = truth_mode(spec)
    c1 = truth_kind == "ghost" and u.block == "grid" and u.mode == "A"
    udir = ctx.unit_dir(u.name)
    onset = seg.t0 + float(spec["onset_s"])
    warm_abs = seg.t0 + ctx.warm
    pre = float(spec["pre_window_s"])
    slip = SLIP_CYC[str(spec.get("template", "1C"))]
    rec: dict[str, Any] = {"identity": {}, "var_cyc2": {}, "lost_ghost_s": {}, "residual": {}}
    files: list[str] = []
    for ch in sorted(seg.channels):
        if ch not in tw.tracks or ch not in clean.tracks:
            continue
        cn0 = float(np.median(clean.tracks[ch].column("cn0_dbhz").to_numpy()))
        g = identity_gate(clean.tracks[ch], tw.tracks[ch], onset_s=onset, warmup_s=warm_abs,
                          sigma_th_cyc=sigma_disc_cyc(cn0, t_s=T_S))  # fmt: skip
        rec["identity"][ch] = {"passed": g.passed, **jsonable(g.details)}
    # real satellites: on observable epochs, the onset placed by the clock map (docs/3 section 2.6)
    if tw.obs is not None and clean.obs is not None:
        n_ch = len(seg.channels)
        rx_onset = epoch_label(clean.obs, n_ch, seg.clock, onset)
        rx_warm = epoch_label(clean.obs, n_ch, seg.clock, warm_abs)
        if np.isfinite(rx_onset) and np.isfinite(rx_warm):
            oa, ob = common_obs(clean, tw, seg.channels)
            assert oa is not None and ob is not None
            truth, events, excl = twin_difference(oa, ob, onset_s=rx_onset, pre_window_s=pre,
                                                  warmup_s=rx_warm, slip_size_cyc=slip, tau_loop_s=0.03,
                                                  truth_id=u.name)  # fmt: skip
            files += [_write(udir, "truth_obs.parquet", truth), _write(udir, "slips_obs.parquet", events),
                      _write(udir, "excl_obs.parquet", excl)]  # fmt: skip
            t = truth.column("t_rx_s").to_numpy()
            ok = _post(t, rx_onset) & ~truth.column("transient").to_numpy(zero_copy_only=False)
            ok &= truth.column("valid").to_numpy(zero_copy_only=False)
            prn = truth.column("prn").to_numpy()
            e = truth.column("e_cyc").to_numpy()
            for p in np.unique(prn[ok]):
                rec["var_cyc2"][f"G{int(p):02d}"] = float(np.var(e[ok & (prn == p)]))
            rec["obs_onset_label"] = rx_onset
    # ghosts: the twin difference at every record, B1's prediction where projected
    projections: dict[int, FloatArray] = {}
    if u.project and src is not None:
        cpath = ctx.work / f"{u.clean}.bin"
        for ch, gd in zip(seg.ghost_ch, seg.ghosts, strict=True):
            if ch in tw.tracks:
                stamps = clean.tracks[ch].column("sample_count").to_numpy()
                projections[ch] = ghost_projection(pair_difference(src.path, cpath, seg.scale), GhostSpec.of(gd),
                                                   stamps, seg.clock, gain(gd.level_db_hz, seg.n0))  # fmt: skip
    for ch in seg.ghost_ch:
        if ch not in tw.tracks or ch not in clean.tracks:
            rec["lost_ghost_s"][ch] = None
            if c1:
                rec.setdefault("ghost_check", {})[ch] = {"computed": False, "reason": "no track in both twins"}
            continue
        a = track_as_obs(clean.tracks[ch], source_id="e2", receiver_id="c")
        b = track_as_obs(tw.tracks[ch], source_id="e2", receiver_id="j")
        truth, events, excl = twin_difference(a, b, onset_s=onset, pre_window_s=pre, warmup_s=warm_abs,
                                              slip_size_cyc=slip, tau_loop_s=0.03,
                                              truth_id=f"{u.name}-g{ch}")  # fmt: skip
        if c1:
            rec.setdefault("ghost_check", {})[ch] = _ghost_check(ctx, u, ch, tw.tracks[ch], truth, onset)
        t = truth.column("t_rx_s").to_numpy()
        valid = truth.column("valid").to_numpy(zero_copy_only=False) & ~truth.column("transient").to_numpy(
            zero_copy_only=False)  # fmt: skip
        post = _post(t, onset)
        end = float(seg.t0 + (seg.source.sample_end - seg.source.sample_start) / seg.source.fs_hz)
        rec["lost_ghost_s"][ch] = float(max(0.0, (end - onset - SETTLE_S) - post[valid].sum() * T_S))
        prn = int(seg.channels[ch][2])
        e = truth.column("e_cyc").to_numpy()
        if (post & valid).sum() > 100:
            rec["var_cyc2"][f"ghost{prn:02d}"] = float(np.var(e[post & valid]))
        if ch in projections:
            stamps = clean.tracks[ch].column("sample_count").to_numpy()
            t_rec = seg.clock.file_time(stamps)
            pred_at = b1_prediction(projections[ch], t_rec, t, onset, pre)  # cycles; dPhi = H n_iota
            sel = post & valid
            if sel.sum() > 100:
                cn0_c = float(np.median(clean.tracks[ch].column("cn0_dbhz").to_numpy()))
                rec["residual"][prn] = b1_residual(e[sel], pred_at[sel], projections[ch], t_rec, t[sel],
                                                   cn0_clean_dbhz=cn0_c)  # fmt: skip
            truth = truth.append_column("pred_cyc", pa.array(pred_at, pa.float64()))
        if not u.full_ghost:  # observable-epoch resolution is enough for the oracle
            truth = truth.filter(pa.array(oracle_records(t, onset, pre, truth=truth_kind)))
        files += [_write(udir, f"ghost_ch{ch}.parquet", truth), _write(udir, f"ghost_slips_ch{ch}.parquet", events)]
        reasons = zip(excl.column("reason_code").to_pylist(), excl.column("count").to_pylist(), strict=True)
        rec.setdefault("ghost_breaks", {})[ch] = int(sum(c for r, c in reasons if r in BREAKS))
    rec["stock_lock_verdict"] = lock_report(tw.tracks, str(spec.get("template", "1C")), warm_abs)
    return files, rec


def ghost_check(e_jammed: pa.Table, e_clean: pa.Table, dphi: pa.Table, onset: float) -> dict[str, Any]:
    """Check C-1 (R-1; reported, deciding nothing) on one ghost: its truth on the jammed twin's own track (e_jammed)
    less e_c + dPhi, on epochs at or after onset + SETTLE_S that are valid, not transient and slip-free in all three,
    joined on identical record times (pair_on_time). Ghost truth's own constant leaves a constant, so its RMS about
    its median (cycles; R-1's bar is 1e-3) is reported with the epochs it is read on."""
    series = []
    for tab in (e_jammed, e_clean, dphi):
        t = tab.column("t_rx_s").to_numpy()
        ok = _post(t, onset) & tab.column("valid").to_numpy(zero_copy_only=False)
        ok &= ~tab.column("transient").to_numpy(zero_copy_only=False) & (tab.column("slip_count").to_numpy() == 0)
        series.append((t[ok], tab.column("e_cyc").to_numpy()[ok]))
    _, (eu, ec, d), _ = pair_on_time(*series)
    if eu.size == 0:
        return {"computed": False, "reason": "no epoch in all three series"}
    r = eu - (ec + d)
    return {"rms_cyc": float(np.sqrt(np.mean((r - np.median(r)) ** 2))), "n": int(r.size)}


def _ghost_check(ctx: Context, u: Unit, ch: int, track: pa.Table, dphi: pa.Table, onset: float) -> dict[str, Any]:
    """C-1 on ghost channel ch of jammed unit u: ghost truth on its track, converged as the clean twins' (half the
    warm-up), against e_c (the clean twin's ghost_abs file, the paired excess's own) and dPhi. A track with no full
    ambiguity window is recorded as not computed; any other error stops the unit (no broad catch)."""
    seg = ctx.segments[u.segment]
    ghost = GhostSpec.of(seg.ghosts[seg.ghost_ch.index(ch)])
    try:
        e_u = ghost_truth(track, ghost, clock=seg.clock, converge_s=ctx.warm / 2, truth_id=f"{u.name}-g{ch}")
    except NoAmbiguityWindow as exc:
        return {"computed": False, "reason": str(exc)}
    return ghost_check(e_u, pq.read_table(ctx.unit_dir(u.clean or "") / GHOST_ABS.format(ch)), dphi, onset)


def check_recorded(udir: Path, u: Unit) -> None:
    """A finished jammed unit stands in for the planned one only if its unit.json records that very unit: oracle unit
    names omit the level, which follows from the front-end decisions."""
    rec = json.loads((udir / "unit.json").read_text()).get("unit")
    if rec is None or Unit(**rec) != u:
        raise RuntimeError(f"{u.name}: the finished unit on disk ({rec}) is not the planned unit ({u}); resume refused")


def run_jammed(ctx: Context, u: Unit) -> None:
    udir = ctx.unit_dir(u.name)
    if is_done(udir, ctx.describe):
        check_recorded(udir, u)
        return
    shutil.rmtree(udir, ignore_errors=True)
    udir.mkdir(parents=True)
    clean = ctx.cleans[u.clean or ""]
    tw, src = _inject_receive(ctx, u)
    try:
        assert tw.report is not None
        files, rec = reduce_unit(ctx, u, tw, clean, src)
    finally:
        if src is not None:
            src.path.unlink(missing_ok=True)
    rec.update(_unit_record(tw, tw.report))
    rec["unit"] = asdict(u)
    (udir / "unit.json").write_text(json.dumps(jsonable(rec), indent=1, sort_keys=True) + "\n")
    _keep_unit_logs(ctx, u.name, udir)
    mark_done(udir, ctx.describe, [*files, "unit.json"])


def run_repeat(ctx: Context, u: Unit) -> None:
    """G-repeat: inject the unit again (byte-identical) and receive it. A clean twin's tracks must agree with the
    first run's within the identity tolerance; a jammed twin is reduced again, and the RMS difference of its truth
    from the first run's is recorded over the reported-phase thermal SD."""
    name = f"repeat-{u.name}"
    udir = ctx.unit_dir(name)
    if is_done(udir, ctx.describe):
        return
    shutil.rmtree(udir, ignore_errors=True)
    udir.mkdir(parents=True)
    r = replace(u, name=name, keep_bytes=False, project=False)
    first_dir = ctx.unit_dir(u.name)
    first = json.loads((first_dir / "unit.json").read_text())
    seg = ctx.segments[u.segment]
    tw, src = _inject_receive(ctx, r)
    files: list[str] = []
    try:
        assert tw.report is not None
        rec: dict[str, Any] = {"of": u.name, "same_bytes": tw.report.output_sha256 == first["output_sha256"],
                               **_unit_record(tw, tw.report)}  # fmt: skip
        if u.clean is None:
            base = _load_twin(first_dir, u.name)
            for ch in seg.real:
                if ch in tw.tracks and ch in base.tracks:
                    cn0 = float(np.median(base.tracks[ch].column("cn0_dbhz").to_numpy()))
                    rep0 = InjectionReport(seg.n0, {}, 0, first["output_sha256"], "", 0, (), {})
                    g = repeat_gate([rep0, tw.report], [base.tracks[ch], tw.tracks[ch]], warmup_s=seg.t0 + ctx.warm,
                                    sigma_th_cyc=sigma_disc_cyc(cn0, t_s=T_S))  # fmt: skip
                    rec.setdefault("repeat", {})[ch] = {"passed": g.passed, **jsonable(g.details)}
        else:
            files, again = reduce_unit(ctx, r, tw, ctx.cleans[u.clean], src)
            rec["truth_rms_over_sigma"] = _truth_agreement(first_dir / "truth_obs.parquet",
                                                           udir / "truth_obs.parquet", first)  # fmt: skip
            rec["var_cyc2"] = again["var_cyc2"]
    finally:
        if src is not None:
            src.path.unlink(missing_ok=True)
    (udir / "unit.json").write_text(json.dumps(jsonable(rec), indent=1, sort_keys=True) + "\n")
    _keep_unit_logs(ctx, name, udir)
    mark_done(udir, ctx.describe, [*files, "unit.json"])


def _truth_agreement(a: Path, b: Path, first: Mapping[str, Any]) -> dict[str, float]:
    """Per PRN: RMS of the difference of two runs' truth at common epochs, over the reported-phase thermal SD at the
    PRN's post-onset dPhi (a jammed repeat; recorded, not gated)."""
    if not (a.exists() and b.exists()):
        return {}
    ta, tb = pq.read_table(a), pq.read_table(b)
    out = {}
    for p in np.unique(ta.column("prn").to_numpy()):
        fa = ta.filter(pa.array(ta.column("prn").to_numpy() == p))
        fb = tb.filter(pa.array(tb.column("prn").to_numpy() == p))
        common, ia, ib = np.intersect1d(fa.column("t_rx_s").to_numpy(), fb.column("t_rx_s").to_numpy(),
                                        return_indices=True)  # fmt: skip
        if common.size < 100:
            continue
        d = fa.column("e_cyc").to_numpy()[ia] - fb.column("e_cyc").to_numpy()[ib]
        sd = math.sqrt(first["var_cyc2"].get(f"G{int(p):02d}", float("nan")))
        out[f"G{int(p):02d}"] = float(np.sqrt(np.nanmean(d**2)) / sd) if sd > 0 else float("nan")
    return out


def _parallel(ctx: Context, fn: Callable[[Context, Unit], None], units: Sequence[Unit]) -> None:
    """fn on every unit, `pipelines` at a time. The first failure cancels the units still queued; those running finish,
    and then the failure is raised."""
    n = int(ctx.spec.get("pipelines", 1))
    if n <= 1:
        for u in units:
            fn(ctx, u)
        return
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(fn, ctx, u) for u in units]
        wait(futures, return_when=FIRST_EXCEPTION)
        pool.shutdown(cancel_futures=True)
        for f in futures:
            if not f.cancelled():
                f.result()


# ------------------------------------------------------------------------------------------- aggregation
def _units(ctx: Context, prefix: str) -> dict[str, dict[str, Any]]:
    out = {}
    for d in sorted((ctx.run_dir / "units").glob(f"{prefix}*")):
        if (d / "unit.json").exists():
            out[d.name] = json.loads((d / "unit.json").read_text())
    return out


def var_ratios(ctx: Context, seg: str) -> dict[str, dict[float, float]]:
    """Per family and level: the median over channels (real and ghost) of Var_B / Var_A of the post-onset dPhi."""
    spec = ctx.spec
    tau = _tau_tag(float(spec["mode_b"]["tau_s"]))
    out: dict[str, dict[float, float]] = {}
    for fam in spec["families"]:
        out[fam] = {}
        for lv in spec["levels_db_hz"]:
            a = _units(ctx, f"{seg}-grid-{fam}-{float(lv):g}-A").get(f"{seg}-grid-{fam}-{float(lv):g}-A")
            b = _units(ctx, f"{seg}-grid-{fam}-{float(lv):g}-B{tau}").get(f"{seg}-grid-{fam}-{float(lv):g}-B{tau}")
            if not a or not b:
                out[fam][float(lv)] = float("nan")
                continue
            ratios = [b["var_cyc2"][k] / v for k, v in a["var_cyc2"].items() if k in b["var_cyc2"] and v > 0]
            out[fam][float(lv)] = float(np.median(ratios)) if ratios else float("nan")
    return out


def _block_dirs(ctx: Context, seg: str, block: str) -> list[Path]:
    """A block's unit directories in realization order (r2 before r10)."""
    return sorted((ctx.run_dir / "units").glob(f"{seg}-{block}-r*"), key=lambda p: int(p.name.rsplit("-r", 1)[1]))


def _realizations(ctx: Context, seg: str, block: str) -> list[Path]:
    """A block's unit directories that hold truth, in realization order (r2 before r10): the ensemble its oracle
    moments average. A unit without truth (reduce_unit found no observables, or no onset label) is not in it."""
    return [d for d in _block_dirs(ctx, seg, block) if (d / "truth_obs.parquet").exists()]


def _ghost_files(ctx: Context, seg: str) -> list[tuple[str, str]]:
    return [(f"ghost_ch{ch}.parquet", f"ghost_slips_ch{ch}.parquet") for ch in ctx.segments[seg].ghost_ch]


def _ghost_realizations(ctx: Context, seg: str, block: str) -> list[Path]:
    """A block's unit directories that hold every ghost's truth (each of the segment's ghost channels' twin difference
    and slip events), in realization order: the ensemble the ghosts' oracle averages, and its realized K."""
    files = [f for pair in _ghost_files(ctx, seg) for f in pair]
    return [d for d in _block_dirs(ctx, seg, block) if all((d / f).exists() for f in files)]


def _snapped(truth: pa.Table) -> pa.Table:
    """One ghost's twin difference on the 20 ms grid (R-2, ruling Q2): the record nearest each grid point
    (nearest_on_grid; before the onset an oracle unit keeps every record), its time replaced by the grid time
    round(t / 0.02) 0.02, so that realizations whose records differ by fractions of a millisecond share their keys."""
    t = truth.column("t_rx_s").to_numpy()
    keep = nearest_on_grid(t)
    out = truth.select(TRUTH_EPOCH_SAT.names).filter(pa.array(keep))
    grid = np.rint(t[keep] / OBS_INTERVAL_S) * OBS_INTERVAL_S
    return out.set_column(out.schema.get_field_index("t_rx_s"), "t_rx_s", pa.array(grid, pa.float64()))


def _ghost_oracle(ctx: Context, seg: str, block: str, form: str, k: int | None = None) -> pa.Table | None:
    """The ghosts' oracle moments of a block (R-2, clause 4): per realization (_ghost_realizations, the first k), every
    ghost's twin difference on the 20 ms grid (_snapped) with its slip events, through oracle_moments as the real
    satellites' are. A grid point a realization holds no record at lowers that key's n_valid; no record of one
    realization stands in for another's."""
    dirs = _ghost_realizations(ctx, seg, block)[:k]
    if len(dirs) < 2:
        return None
    files = _ghost_files(ctx, seg)
    truths = [pa.concat_tables([_snapped(pq.read_table(d / f)) for f, _ in files]) for d in dirs]
    slips = [pa.concat_tables([pq.read_table(d / f) for _, f in files]) for d in dirs]
    return oracle_moments(truths, slips, config_id=f"{seg}-{block}-ghosts", form=form, slip_window_s=1.0)  # type: ignore[arg-type]


def _oracle(ctx: Context, seg: str, block: str, form: str, k: int | None = None) -> pa.Table | None:
    dirs = _realizations(ctx, seg, block)[:k]
    if len(dirs) < 2:
        return None
    truths = [pq.read_table(d / "truth_obs.parquet") for d in dirs]
    slips = [pq.read_table(d / "slips_obs.parquet") for d in dirs]
    return oracle_moments(truths, slips, config_id=f"{seg}-{block}", form=form, slip_window_s=1.0)  # type: ignore[arg-type]


def _precision(track: pa.Table) -> float:
    return float(np.max(np.abs(track.column("acc_phase_rad").to_numpy()))) if track.num_rows else 0.0


def _record(ctx: Context, name: str) -> dict[str, Any] | None:
    path = ctx.unit_dir(name) / "unit.json"
    return json.loads(path.read_text()) if path.is_file() else None


def _truth_series(path: Path, onset: float) -> tuple[FloatArray, FloatArray]:
    """(t_rx_s, e_cyc) of a truth table on its usable epochs: at or after onset + SETTLE_S, valid and not transient."""
    tr = pq.read_table(path, columns=["t_rx_s", "e_cyc", "valid", "transient"])
    t = tr.column("t_rx_s").to_numpy()
    ok = _post(t, onset) & tr.column("valid").to_numpy(zero_copy_only=False)
    ok &= ~tr.column("transient").to_numpy(zero_copy_only=False)
    return t[ok], tr.column("e_cyc").to_numpy()[ok]


def front_end_record(ctx: Context, a: Unit, b: Unit) -> tuple[FrontEndCell | None, dict[str, Any]]:
    """One front-end cell (docs/8 section 5, rule (a')): mode A unit `a` against mode B unit `b` on one jammer draw.
    Per ghost, dPhi_A, dPhi_B and the clean twins' absolute errors e_c(A), e_c(B) on usable epochs joined on identical
    record times (pair_on_time; no interpolation), the paired excess per mode and the slope; then R, SE(R) and the
    verdict as E2post forms them (front_end_cell). A cell an input is missing for, with fewer epochs than two bootstrap
    blocks, or with a non-finite slope, R or SE(R) gets no verdict (None) and says why."""
    seg, rules = ctx.segments[a.segment], ctx.spec["rules"]
    boot = rules["bootstrap"]
    block = round(float(boot["block_s"]) / T_S)
    onset = seg.t0 + float(ctx.spec["onset_s"])
    rec: dict[str, Any] = {"units": [a.name, b.name], "slopes": None, "ratio": None, "ratio_se": None,
                           "verdict": None, "excess": None, "dropped_aligning": None, "no_verdict": None}  # fmt: skip
    jam_a, jam_b, clean_a, clean_b = (ctx.unit_dir(n) for n in (a.name, b.name, a.clean or "", b.clean or ""))
    paths = {ch: [jam_a / f"ghost_ch{ch}.parquet", jam_b / f"ghost_ch{ch}.parquet", clean_a / GHOST_ABS.format(ch),
                  clean_b / GHOST_ABS.format(ch)] for ch in seg.ghost_ch}  # fmt: skip
    missing = [str(p.relative_to(ctx.run_dir)) for ps in paths.values() for p in ps if not p.is_file()]
    if missing:
        return None, {**rec, "no_verdict": f"missing {', '.join(missing)}"}
    excess: dict[int, dict[str, float | int]] = {}
    paired = []
    for ch, ps in paths.items():
        t, (da, db, ca, cb), dropped = pair_on_time(*(_truth_series(p, onset) for p in ps))
        if t.size < 2 * block:
            return None, {**rec, "no_verdict": f"ghost channel {ch}: {t.size} paired epochs, under two blocks"}
        xa, xb = paired_excess(da, ca), paired_excess(db, cb)
        excess[ch] = {"n": int(t.size), "dropped": dropped, "var_a": xa.var_dphi, "var_b": xb.var_dphi,
                      "cov2_a": xa.cov2, "cov2_b": xb.cov2, "excess_a": xa.excess, "excess_b": xb.excess,
                      "slope": jammer_slope(da, db)}  # fmt: skip
        paired.append((t, da, db, ca, cb))
    rec["excess"] = excess
    aligned, _ = align_records([p[0] for p in paired], tol_s=T_S / 2)
    if aligned[0].size < 2 * block:
        return None, {**rec, "no_verdict": f"{aligned[0].size} epochs aligned across the ghosts, under two blocks"}
    c, dropped = front_end_cell(paired, block=block, tol_s=T_S / 2, n_boot=int(boot["n_boot"]), seed=int(boot["seed"]))
    rec.update(slopes=dict(zip(paths, c.slopes, strict=True)), ratio=c.ratio, ratio_se=c.ratio_se,
               dropped_aligning=dropped)  # fmt: skip
    if not finite_cell(c):
        return None, {**rec, "no_verdict": "a non-finite slope, R or SE(R)"}
    rec["verdict"] = cell_verdict(c, tol=float(rules["jmax_tol"]), slope_tol=float(rules["slope_tol"]),
                                  se_max=float(rules["excess_se_max"]), cell=f"{a.name} against {b.name}")  # fmt: skip
    return c, rec


def b1_ok(rec: Mapping[str, Any], prns: Sequence[int], residual_max: float) -> bool:
    """B1's bar at one level, mode A, both ghosts (docs/3 section 2.2, D-36): r - 1/(C/N0 T) < residual_max with each
    ghost's floor at its clean median C/N0. A ghost without a residual (or with a NaN) does not pass."""
    res = rec.get("residual") or {}
    return all(str(p) in res and float(res[str(p)]["r"]) - float(res[str(p)]["floor"]) < residual_max for p in prns)


def truth_ok(rec: Mapping[str, Any], channels: Sequence[int], lost_max_s: float) -> bool:
    """Ghost truth kept at one level, mode A, both ghosts: at most lost_max_s lost after onset + settle."""
    lost = rec.get("lost_ghost_s") or {}
    return all(lost.get(str(ch)) is not None and float(lost[str(ch)]) <= lost_max_s for ch in channels)


def route_b_checks(
    recs: Mapping[float, Mapping[str, Any]], prns: Sequence[int], channels: Sequence[int], rules: Mapping[str, Any]
) -> tuple[dict[float, bool], dict[float, bool]]:
    """(B1's bar, the ghost truth kept) per level, what route_b_ceiling reads, from each level's mode A unit record
    (its `residual` and `lost_ghost_s`): the unit.json files on disk for E2, their copies in E2's summary for E2K."""
    b1 = {lv: b1_ok(rec, prns, float(rules["residual_max"])) for lv, rec in recs.items()}
    kept = {lv: truth_ok(rec, channels, float(rules["lost_truth_max_s"])) for lv, rec in recs.items()}
    return b1, kept


def front_end_segment(ctx: Context, key: str) -> dict[str, Any]:
    """Segment s's part of the rules of (a'): the front-end cells per family and level, segment A's tau side cells
    (each tau unit against the mode A grid unit of its cell) and the cap a failed or verdict-less one puts on J_max,
    J_max(s, f); B1's bar (`residual_pass`) and the truth kept per projected level, None for a family with no projected
    level; the Route B ceiling L_B(s, f); and under ghost truth the ghost ceiling L_G(s, f) (ghost_ceiling_db_hz)."""
    spec, rules = ctx.spec, ctx.spec["rules"]
    seg = ctx.segments[key]
    units = plan_grid(spec, key)
    grid = {(u.family, u.level, u.mode): u for u in units if u.block == "grid"}
    levels = sorted(float(v) for v in spec["levels_db_hz"])
    prns = [int(seg.channels[ch][2]) for ch in seg.ghost_ch]
    ghost = truth_mode(spec) == "ghost"
    fields = ["front_end", "tau_side", "tau_cap_db_hz", "j_max_db_hz", "residual_pass", "truth_kept", "ceiling_db_hz"]
    out: dict[str, Any] = {k: {} for k in [*fields, *(["ghost_ceiling_db_hz"] if ghost else [])]}
    for fam in spec["families"]:
        cells: dict[float, FrontEndCell | None] = {}
        for lv in levels:
            cells[lv], out["front_end"].setdefault(fam, {})[lv] = front_end_record(ctx, grid[fam, lv, "A"],
                                                                                  grid[fam, lv, "B"])  # fmt: skip
        cap = math.inf
        side = [u for u in units if u.block == "tau" and u.family == fam]
        if side:
            taus = {u.tau_s: front_end_record(ctx, grid[fam, u.level, "A"], u)[1] for u in side}
            out["tau_side"][fam] = taus
            capped = any(r["verdict"] in ("fail", None) for r in taus.values())  # no verdict: no pass shown
            cap = side_cap(levels, float(spec["mode_b"]["side_level_db_hz"])) if capped else math.inf
            out["tau_cap_db_hz"][fam] = cap if capped else None
        jm = segment_jmax(cells, rules=rules, cap=cap, name=f"segment {key}, {fam}")
        mode_a = {lv: grid[fam, lv, "A"] for lv in levels}
        recs = {lv: rec for lv, u in mode_a.items() if u.project and (rec := _record(ctx, u.name)) is not None}
        b1, kept = route_b_checks(recs, prns, seg.ghost_ch, rules)
        out["j_max_db_hz"][fam] = jm
        out["residual_pass"][fam] = b1 or None
        out["truth_kept"][fam] = kept or None
        out["ceiling_db_hz"][fam] = route_b_ceiling(levels, jm, b1, kept)
        if ghost:
            out["ghost_ceiling_db_hz"][fam] = ghost_ceiling(levels, jm, kept)
    return out


def front_end_segments(ctx: Context) -> dict[str, dict[str, Any]]:
    return {key: front_end_segment(ctx, key) for key in ctx.segments}


def combined(front: Mapping[str, Mapping[str, Any]], field: str, families: Sequence[str]) -> dict[str, float | None]:
    """Per family, the minimum over segments of a per-segment level, None if any segment's is None."""
    return {fam: combine_segments([f[field][fam] for f in front.values()]) for fam in families}


def combined_levels(
    front: Mapping[str, Mapping[str, Any]], field: str, families: Sequence[str]
) -> dict[str, dict[float, bool | None] | None]:
    """Per family and level, a per-level verdict combined over segments as combine_segments does (False if any is
    False, None if any is missing); None for a family that some segment has no level for."""
    out: dict[str, dict[float, bool | None] | None] = {}
    for fam in families:
        per = [f[field][fam] for f in front.values()]
        if not per or any(p is None for p in per):
            out[fam] = None
            continue
        both = [combine_segments([p.get(lv) for p in per]) for lv in sorted(set().union(*per))]
        out[fam] = {lv: None if v is None else bool(v) for lv, v in zip(sorted(set().union(*per)), both, strict=True)}
    return out


def _form(b: OracleBlock) -> Literal["conditional", "marginal"]:
    return "conditional" if b.offset_s is not None else "marginal"  # F8: the timing fixed, or drawn too


def k_study_decision(
    ctx: Context, ceiling: Mapping[str, float | None], *, truth: Truth | None = None
) -> dict[str, Any]:
    """The K-study from the oracle units on disk (docs/5 section 4.4): each block's oracle moments (written to
    oracle_<segment>_k-<tag>.parquet), K by the K = 101 block against its first k_small realizations (None when no
    K-study ran, so K is not chosen), the thermal block against the first K of that block, and where each block ran
    (oracle_blocks: on the fallback family when swept CW has no ceiling, the conditional-against-marginal comparison
    then not done). Each block's realized K (its realizations with truth) is recorded beside its planned K. K's rule
    compares the full K = 101 block with its first k_small: when that block's realized K is not its planned K, K is not
    chosen and k_not_chosen says why. The thermal comparison is reported only: ensembles with no common epoch record
    it as not computed, with the reason. Under ghost truth (`truth`, by default the spec's) the oracle is the ghosts'
    (_ghost_oracle; R-2): realized K counts the realizations holding every ghost's truth, and the tables are written to
    oracle_ghost_<segment>_<block>.parquet."""
    spec, rules = ctx.spec, ctx.spec["rules"]
    ks, th = spec.get("kstudy"), spec.get("thermal")
    blocks, thermal = oracle_blocks(spec, ceiling)
    ghost = (truth or truth_mode(spec)) == "ghost"
    oracle = _ghost_oracle if ghost else _oracle
    held = _ghost_realizations if ghost else _realizations
    prefix = "oracle_ghost" if ghost else "oracle"
    out: dict[str, Any] = {"k": None}
    study: dict[str, Any] = {"blocks": {tag: asdict(b) for tag, b in blocks.items()},
                             "thermal": None if thermal is None else asdict(thermal)}  # fmt: skip
    out["k_study"] = study
    if not ks:
        return out
    seg = ks["segment"]
    realized = {tag: len(held(ctx, seg, f"k-{tag}")) for tag in blocks}
    study["realized_k"] = {tag: {"planned": b.k, "realized": realized[tag]} for tag, b in blocks.items()}
    study["thermal_realized_k"] = None if not th or thermal is None else {
        "planned": thermal.k, "realized": len(held(ctx, th["segment"], "thermal"))}  # fmt: skip
    tabs = {}
    for tag, b in blocks.items():
        t = oracle(ctx, seg, f"k-{tag}", _form(b))
        if t is not None:
            pq.write_table(t, ctx.run_dir / f"{prefix}_{seg}_k-{tag}.parquet", compression="zstd")
            tabs[tag] = t
    cond = next((tag for tag in sorted(ks["blocks"]) if ks["blocks"][tag].get("conditional")), None)
    ran = cond if cond is not None and cond in tabs else None  # the K = 101 block, if it ran
    if cond is not None and cond in blocks and realized[cond] != blocks[cond].k:
        out["k_not_chosen"] = (f"block {cond}: {realized[cond]} realizations hold truth, {blocks[cond].k} planned; "
                               f"K's rule compares sigma^2 over the full block with sigma^2 over its first "
                               f"{int(rules['k_small'])} (docs/5 section 4.4)")  # fmt: skip
    elif ran is not None:
        small = oracle(ctx, seg, f"k-{ran}", _form(blocks[ran]), k=int(rules["k_small"]))
        if small is not None:
            conv = k_convergence(small, tabs[ran])
            out["k_convergence"] = conv
            out["k"] = choose_k(conv)
    fam = ks["blocks"][cond]["family"] if cond is not None else None
    marg = [tag for tag, b in sorted(ks["blocks"].items()) if not b.get("conditional") and b["family"] == fam]
    study["cond_vs_marg_done"] = ran is not None and not blocks[ran].fallback and any(m in tabs for m in marg)
    same = ran is not None and (thermal is not None) and (blocks[ran].family, blocks[ran].level) == (
        thermal.family, thermal.level)  # fmt: skip
    if th and thermal is not None and ran is not None and same:  # both moved to the fallback, or neither did
        t = oracle(ctx, th["segment"], "thermal", _form(thermal))
        base = oracle(ctx, th["segment"], f"k-{ran}", _form(blocks[ran]), k=thermal.k)
        if t is not None and base is not None:
            pq.write_table(t, ctx.run_dir / f"{prefix}_{th['segment']}_thermal.parquet", compression="zstd")
            try:
                out["thermal_vs_fixed"] = k_convergence(base, t)  # same K: the SE of a ratio of two
            except NoCommonEpochs as exc:  # reported only: K stands and the summary is written
                out["thermal_vs_fixed"] = {"computed": False, "reason": str(exc), "epochs": 0}
    return out


def _read_s29(spec: Mapping[str, Any]) -> tuple[Any, ...]:
    """e2k.read_s29 on the spec's segments and families: J_max_TA from the s29 result the spec pins by SHA-256."""
    from gnssaudit.experiments.e2k import read_s29  # e2k imports this module: imported when called

    return read_s29(spec, list(spec["segments"]), list(spec["families"]))


def route_b_arm(
    ctx: Context, front: Mapping[str, Mapping[str, Any]], study: Mapping[str, float | None]
) -> dict[str, Any]:
    """The Route B arm under ghost truth (R-6; exploratory, reported, deciding nothing, and labelled so in the record:
    exploratory True, decides False): L_B under the registered J_max (B1 per level is decisions.residual_pass); L_B
    under J_max_TA as E2K forms it, per segment the largest grid level at or below s29's combined J_max_TA(f) with B1's
    bar passed and the truth kept at every level up to it, then combine_segments (not computed when the spec pins no
    s29 result; s29's own per-family flag kept as s29_decides); and the real satellites' K-study on the units planned
    at `study`, the ghost ceiling."""
    spec = ctx.spec
    fams = list(spec["families"])
    levels = sorted(float(v) for v in spec["levels_db_hz"])
    arm: dict[str, Any] = {
        "exploratory": True,
        "decides": False,
        "ceiling_db_hz": combined(front, "ceiling_db_hz", fams),
    }
    if "s29_result" in spec:
        _, jmax_ta, decides, _ = _read_s29(spec)
        per = {key: {fam: route_b_ceiling(levels, jmax_ta[fam], f["residual_pass"][fam] or {},
                                          f["truth_kept"][fam] or {}) for fam in fams}
               for key, f in front.items()}  # fmt: skip
        both = {fam: combine_segments([p[fam] for p in per.values()]) for fam in fams}
        arm["j_max_ta"] = {"s29_result": str(spec["s29_result"]), "s29_sha256": str(spec["s29_sha256"]),
                           "j_max_ta_db_hz": jmax_ta, "s29_decides": decides, "segments": per,
                           "ceiling_db_hz": both}  # fmt: skip
    else:
        arm["j_max_ta"] = {"computed": False, "reason": "the spec pins no s29_result"}
    arm["k_study"] = k_study_decision(ctx, study, truth="route_b")
    return arm


def g3_clauses_ghost(
    gates: Mapping[str, Mapping[str, Any]], decisions: Mapping[str, Any], hz_source: Sequence[str]
) -> dict[str, Any]:
    """G3's clauses under ghost truth (R-2), from the run's gates and decisions alone. Clause 1: every gate the run
    wrote passed (none written: not passed). Clause 2: G-hz as passed on E0 (`hz_source`), recorded, not re-run.
    Clause 3: a ghost ceiling L_G for noise and for swept CW; with one of them only, the result is narrowed, reported
    beside the verdict and not the verdict; the chirp's counts for neither. Clause 4: K chosen (the K-study ran and its
    realized K is its planned K). G3 passes iff clauses 1, 3 and 4 do (clause 2 is carried)."""
    failed = sorted(name for name, g in gates.items() if not g["passed"])
    lg = dict(decisions["ghost_ceiling_db_hz"])
    held = [fam for fam in G3_FAMILIES if lg.get(fam) is not None]
    c1, c3, k = bool(gates) and not failed, len(held) == len(G3_FAMILIES), decisions["k"]
    return {"gates": {"passed": c1, "count": len(gates), "failed": failed},
            "hz": {"source": list(hz_source)},
            "clause3": {"passed": c3, "narrowed": [] if c3 else held, "j_max_db_hz": dict(decisions["j_max_db_hz"]),
                        "ghost_ceiling_db_hz": lg},
            "k": {"k": k, "chosen": k is not None},
            "passed": c1 and c3 and k is not None}  # fmt: skip


def twin_gates(ctx: Context, key: str) -> tuple[dict[str, Any], list[tuple[str, GateReport]]]:
    """Segment `key`'s twin records and gates, from the units in the run directory and the twins in ctx.cleans: per
    jammed unit the lost ghost truth, the stock lock drops and the clips; G-warmup-lock (the jammed units and the clean
    twins, the thermal pairs' included; not the repeats), G-phase-precision (the segment's clean twins), G-identity
    and G-null (null_checks: the mode A clean twins <key>-clean-A and -A2, per real channel; the ghost channels'
    null difference reported in the records as ghost_null, not gating)."""
    s: dict[str, Any] = {}
    gates: list[tuple[str, GateReport]] = []
    jam = {**_units(ctx, f"{key}-grid-"), **_units(ctx, f"{key}-tau-"), **_units(ctx, f"{key}-k-"),
           **_units(ctx, f"{key}-thermal-r")}  # fmt: skip
    # every twin's lock: the jammed units and the clean twins, the thermal pairs' too (not in jam: no lost_ghost_s)
    recs = {**jam, **_units(ctx, f"{key}-clean-"), **_units(ctx, f"{key}-thermal-clean-")}
    unlocked = {n: r["unlocked"] for n, r in recs.items() if r["unlocked"]}
    failed = sorted(n for n, r in recs.items() if not r.get("lock_ok", True))
    lock_details = {"units_failed": failed, "excluded_channels": unlocked}
    gates.append((f"[{key}]", GateReport("G-warmup-lock", not failed, lock_details)))
    own = [c for name, c in ctx.cleans.items() if name.startswith(f"{key}-")]  # this segment's clean twins
    precise = [_precision(t) for c in own for t in c.tracks.values()]
    if precise:
        worst = max(precise)
        sd = sigma_reported_cyc(35.0, bn_eff_hz=CarrierLoop.gnsssdr(DEFAULT_BW_HZ, T_S).bn_eff_hz(), t_s=T_S)
        ok = phase_precision_ok(worst, np.dtype(np.float64), sd * 2 * np.pi)
        gates.append((f"[{key}]", GateReport("G-phase-precision", ok, {"max_abs_phase_rad": worst})))
    ident = [v["passed"] for r in jam.values() for v in r.get("identity", {}).values()]
    gates.append((f"[{key}]", GateReport("G-identity", bool(ident) and all(ident),
                                         {"pairs": len(ident), "failed": len(ident) - sum(ident)})))  # fmt: skip
    s["lost_ghost_s"] = {n: r["lost_ghost_s"] for n, r in jam.items()}
    s["stock_lock_drops"] = {n: sum(v["drops"] for v in r["stock_lock_verdict"].values()) for n, r in jam.items()}
    s["clips"] = {n: r["clip_count"] for n, r in jam.items() if r["clip_count"]}
    nulls, s["ghost_null"] = null_checks(ctx, key)
    gates += nulls
    return s, gates


def _null_pair(ctx: Context, key: str, a0: Twin, a2: Twin, ch: int) -> GateReport:
    """G-null's test on channel ch of segment `key`'s null pair: the twin difference of the two mode A clean twins, at
    the spec's signal's slip size, against the reported-phase thermal SD at a0's median C/N0."""
    spec, seg = ctx.spec, ctx.segments[key]
    cn0 = float(np.median(a0.tracks[ch].column("cn0_dbhz").to_numpy()))
    a = track_as_obs(a0.tracks[ch], source_id="e2", receiver_id="a0")
    b = track_as_obs(a2.tracks[ch], source_id="e2", receiver_id="a2")
    null, _, ex = twin_difference(a, b, onset_s=seg.t0 + float(spec["onset_s"]),
                                  pre_window_s=float(spec["pre_window_s"]), warmup_s=seg.t0 + ctx.warm,
                                  slip_size_cyc=SLIP_CYC[str(spec.get("template", "1C"))], tau_loop_s=0.03,
                                  truth_id=f"{key}-null-ch{ch}")  # fmt: skip
    sd = sigma_reported_cyc(cn0, bn_eff_hz=CarrierLoop.gnsssdr(DEFAULT_BW_HZ, T_S).bn_eff_hz(), t_s=T_S)
    return null_gate(null, ex, sigma_th_cyc=sd)


def null_checks(ctx: Context, key: str) -> tuple[list[tuple[str, GateReport]], dict[str, Any]]:
    """G-null on segment `key`'s null pair, the mode A clean twins <key>-clean-A and -A2: per real channel in both
    twins, gating. A missing twin, or no real channel in both, is one failing G-null[<key>] naming the missing twins:
    the gate may not pass for want of anything to test (final review R1 M2, ruled 22 Oct 2026). The ghost channels'
    null twin difference, by G-null's own test, is reported beside it and gates nothing: adding the ghosts to a
    pre-registered gate after the data would change its scope (the same ruling)."""
    seg = ctx.segments[key]
    names = (f"{key}-clean-A", f"{key}-clean-A2")
    a0, a2 = ctx.cleans.get(names[0]), ctx.cleans.get(names[1])
    missing = [n for n in names if n not in ctx.cleans]
    gates: list[tuple[str, GateReport]] = []
    ghosts: dict[int, dict[str, Any]] = {}
    for ch in seg.ghost_ch:
        prn = int(seg.channels[ch][2])
        if a0 is None or a2 is None:
            ghosts[ch] = {"prn": prn, "computed": False, "reason": f"null twin missing: {', '.join(missing)}"}
        elif ch not in a0.tracks or ch not in a2.tracks:
            ghosts[ch] = {"prn": prn, "computed": False, "reason": "no track in both null twins"}
        else:
            g = _null_pair(ctx, key, a0, a2, ch)
            ghosts[ch] = {"prn": prn, "computed": True, "meets_g_null": g.passed, **jsonable(g.details)}
    if a0 is not None and a2 is not None:
        gates = [(f"[{key}][ch{ch}]", _null_pair(ctx, key, a0, a2, ch)) for ch in seg.real
                 if ch in a0.tracks and ch in a2.tracks]  # fmt: skip
    if not gates:
        gates = [(f"[{key}]", GateReport("G-null", False, {"pairs": 0, "missing_twins": missing}))]
    return gates, {"gating": False, "channels": ghosts}


def repeat_report(ctx: Context) -> GateReport:
    """G-repeat over the run's repeat- units: every repeat's bytes as the first run's, and at least one receiver check
    (a clean twin's repeat), every one passed. A jammed unit's repeat records its truth agreement, ungated, so without
    a receiver check the gate would pass on bytes alone (final review R1 M3, ruled 22 Oct 2026)."""
    reps = _units(ctx, "repeat-")
    same = [r["same_bytes"] for r in reps.values()]
    rx_ok = [v["passed"] for r in reps.values() for v in r.get("repeat", {}).values()]
    details = {"units": len(same), "bytes_differ": len(same) - sum(same), "receiver_checks": len(rx_ok),
               "receiver_failed": len(rx_ok) - sum(rx_ok)}  # fmt: skip
    return GateReport("G-repeat", bool(same) and all(same) and bool(rx_ok) and all(rx_ok), details)


def aggregate(
    ctx: Context, front: Mapping[str, Mapping[str, Any]] | None = None
) -> tuple[dict[str, Any], list[tuple[str, GateReport]]]:
    """The summary and the gates. `front` is front_end_segments(ctx), passed by run_e2 so that the K-study and the
    summary rest on one computation; computed here when not given. Under ghost truth the K-study is decided at the
    ghost ceiling on the ghosts' oracle, L_G and the Route B arm are in the decisions, and each segment has C-1 per
    family, level and ghost channel (ghost_check)."""
    spec, rules = ctx.spec, ctx.spec["rules"]
    ghost = truth_mode(spec) == "ghost"
    front = front_end_segments(ctx) if front is None else front
    summary: dict[str, Any] = {"segments": {}, "decisions": {}}
    gates: list[tuple[str, GateReport]] = []
    for key, seg in ctx.segments.items():
        s: dict[str, Any] = {
            "channels": seg.channels,
            "ghost_prn": [int(seg.channels[c][2]) for c in seg.ghost_ch],
            "n0_per_hz": seg.n0,
            "line_excess_db": seg.line_excess_db,
            "agc_rms": seg.agc_rms,
            "analog_sd": seg.analog_sd,
        }
        ratios = var_ratios(ctx, key)  # option (a), reported alongside
        s["var_b_over_a"] = ratios
        s["option_a_j_max_db_hz"] = {fam: j_max(r, tol=float(rules["jmax_tol"])) for fam, r in ratios.items()}
        # (a'): front_end, tau_side, tau_cap_db_hz, j_max_db_hz, residual_pass, truth_kept, ceiling_db_hz
        s.update(front[key])
        resid: dict[str, dict[float, dict[str, Any]]] = {}
        checks: dict[str, dict[float, dict[str, Any]]] = {}
        for rec in _units(ctx, f"{key}-grid-").values():
            if rec["unit"]["mode"] == "A" and rec.get("residual"):
                resid.setdefault(rec["unit"]["family"], {})[rec["unit"]["level"]] = rec["residual"]
            if rec["unit"]["mode"] == "A" and "ghost_check" in rec:
                checks.setdefault(rec["unit"]["family"], {})[rec["unit"]["level"]] = rec["ghost_check"]
        s["residual"] = resid  # r, floor, Var(n_iota) and floor + Var(n_iota) per level and ghost
        if ghost:
            s["ghost_check"] = checks  # C-1 (R-1; reported, deciding nothing)
        records, seg_gates = twin_gates(ctx, key)
        s.update(records)
        gates += seg_gates
        summary["segments"][key] = s
    fams = list(spec["families"])
    d = summary["decisions"]
    ceiling = combined(front, "ceiling_db_hz", fams)
    study = combined(front, ceiling_field(spec), fams)  # L_G under ghost truth (R-2), else L_B
    d.update(k_study_decision(ctx, study))
    gates.append(("[repeat]", repeat_report(ctx)))
    ks = spec.get("kstudy")
    cap = float(ks["level_cap_db_hz"]) if ks else None
    d["j_max_db_hz"] = combined(front, "j_max_db_hz", fams)
    d["ceiling_db_hz"] = ceiling
    d["k_level_db_hz"] = {fam: None if cap is None else k_level(cap, study[fam]) for fam in fams}
    if ghost:  # the deciding ceiling named in the record (fix round 1); L_B stays at the top level, as under route_b
        d["truth"], d["deciding_ceiling"] = "ghost", "ghost_ceiling_db_hz"
        d["ghost_ceiling_db_hz"] = study
        d["route_b_arm"] = route_b_arm(ctx, front, study)
    d["residual_pass"] = combined_levels(front, "residual_pass", fams)
    d["option_a"] = {"var_b_over_a": {k: s["var_b_over_a"] for k, s in summary["segments"].items()},
                     "j_max_db_hz": combined(summary["segments"], "option_a_j_max_db_hz", fams)}  # fmt: skip
    return summary, gates


# ------------------------------------------------------------------------------------------------ the runner
def run_e2(spec: dict[str, Any], run_dir: Path) -> Outcome:
    started = run_dir / "started.json"
    describe = json.loads(started.read_text())["git_describe"] if started.exists() else git_describe(REPO)
    work = results_root() / "E2-work" / run_dir.name
    work.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(SHM_ROOT / work.name, ignore_errors=True)  # twins a killed start left (a SIGKILL skips `finally`)
    template = str(spec.get("template", "1C"))
    for fam in spec["families"]:
        jammer_kind(fam)
    truth = truth_mode(spec)
    if truth == "ghost" and "s29_result" in spec:
        _read_s29(spec)  # the Route B arm's pin, read after the units: a wrong one is refused before them
    ctx = Context(spec, run_dir, work, describe, receiver_spec(spec, template=template), {}, {}, {}, {})
    try:
        for key in spec["segments"]:
            ctx.segments[key] = _segment_survey(ctx, key)
        grid = [u for key in spec["segments"] for u in plan_grid(spec, key)]
        _parallel(ctx, run_clean, [u for u in grid if u.block == "clean"])
        _parallel(ctx, run_jammed, [u for u in grid if u.block != "clean"])
        front = front_end_segments(ctx)  # every segment's grid is in: J_max and L_B are decided once, on all of them
        ceiling = combined(front, ceiling_field(spec), list(spec["families"]))  # L_G under ghost truth, else L_B
        oracle = [u for key in spec["segments"] for u in plan_oracle(spec, key, ceiling)]
        _parallel(ctx, run_clean, [u for u in oracle if u.block == "thermal-clean"])
        _parallel(ctx, run_jammed, [u for u in oracle if u.block != "thermal-clean"])
        everything = {u.name: u for u in [*grid, *oracle]}
        reps = pick_repeats(list(everything), fraction=float(spec["repeat_fraction"]), seed=int(spec["seed"]))
        _parallel(ctx, run_repeat, [everything[n] for n in reps])
        summary, gates = aggregate(ctx, front)
    finally:
        shutil.rmtree(SHM_ROOT / work.name, ignore_errors=True)
    summary["units"] = len(list((run_dir / "units").glob("*/done.json")))
    summary["receivers"] = {"overrides": dict(ctx.rx.overrides), "image": ctx.rx.image, "patch_id": ctx.rx.patch_id}
    summary["gates"] = {g.gate + sfx: {"passed": g.passed, **jsonable(g.details)} for sfx, g in gates}
    if truth == "ghost":
        summary["g3_clauses"] = g3_clauses_ghost(summary["gates"], summary["decisions"], G3_HZ_SOURCE)
    (run_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n")
    shutil.rmtree(work, ignore_errors=True)  # dumps and kept clean bytes go (D-18); units stay
    failed = [g for _, g in gates if not g.passed]
    if failed and spec.get("abort_on_gate_failure", True):
        raise GateFailure(failed[0])
    inputs = [seg.source.path for seg in ctx.segments.values()]
    return Outcome(inputs=inputs, counts={"units": int(summary["units"]), "gates_failed": len(failed)},
                   summary=jsonable(summary))  # fmt: skip
