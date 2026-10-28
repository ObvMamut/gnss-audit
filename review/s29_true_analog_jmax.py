"""S29: J_max against a true analog 3-bit front end (P2b part 2 addendum, task 12; a deviation, owner 19 Oct 2026).

Requirements: docs/superpowers/plans/2026-10-19-p2b-g3-true-analog-jmax.md (committed in ddbc26c before any of the data
below were computed). E2 run 2 (6214cf52489f) gave no J_max under the pre-registered (a') rule; that result stands and
this one is reported beside it, never in its place.

Construction: exactly s28's E2 structure (review.s28_twin_pair_quantisation.e2s_chains, imported, not copied). The
recording r0 = requantise(x0) holds no ghost; the ghost g (s28's GHOST) is injected after the jammer into every twin,
in mode B after the reconstruction. Pairs, each against its own clean twin: mode A r0 + j + g vs r0 + g; the true front
end requantise(G (x0 + j + g)) vs requantise(x0 + g), G = sqrt(P_x / P_{x+j}) the stationary AGC gain; mode B
Q(AGC(x^ + j + g)) vs Q(AGC(x^ + g)). FGI's 3-bit alphabet, analog noise flat over +-10 MHz at 30.69 MS/s, E2-A's
jammer families (noise, swept CW, chirp) with s28's make_cell seeds, s28's dither seed, windows and bootstrap.

Per segment s in {A, B}: the analog's SD per component is E2 run 2's estimate (summary.json segments[s].analog_sd,
read here); N0 and mode B's analog SD are estimated on the synthetic recording as s28 does (recording_floor). Levels
50 to 70 dB-Hz. Each cell runs 20, 40 or 60 s of 1 ms windows and stops at the first checkpoint where SE(R_TA) <= 0.02,
otherwise at 60 s. R_TA = excess_truth / excess_A with s28's moving-block bootstrap SE (1 s blocks, 400 replicates,
seed 7); beta_TA = Cov(dPhi_A, dPhi_truth) / Var(dPhi_A); the verdict is cell_verdict(FrontEndCell((beta_TA,), R_TA,
SE), tol 0.10, slope_tol 0.10, se_max 0.05). J_max_TA(s, f) = front_end_jmax over the five levels; J_max_TA(f) =
combine_segments over A and B (None if either is None, else the minimum). Noise and swept CW decide; the chirp is
reported.

Reported, not deciding: mode B's ratios against A and against the truth per cell; H-B (segment A, noise, 55 dB-Hz:
mode B's excess / the truth's <= 0.92); H-A (that cell: |R_TA - 1| < 0.10); E2 run 2's tau side check; the chirp.

Known answer, checked first per segment (the study stops if it fails): with the jammer at zero, every jammed twin is its
clean twin sample for sample on the first chunk, and over three chunks of windows every pair's correlations equal and
its excess is exactly 0.

Usage: uv run python -m review.s29_true_analog_jmax (reads E2 run 2 read-only; about 1.5 h with 5 processes).
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import shutil
import tempfile
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from gnssaudit.experiments.e2_frontend import (
    FrontEndCell,
    Verdict,
    cell_verdict,
    combine_segments,
    front_end_jmax,
    paired_excess,
)
from gnssaudit.inject import GhostSpec, gain, ghost_modulation
from gnssaudit.inject.pipeline import _state  # mode B's AGC state, as s28 builds it
from review._common import save
from review.s28_twin_pair_quantisation import (
    BAND_HZ,
    BOOT_BLOCK,
    BOOT_N,
    BOOT_SEED,
    CHUNK,
    DITHER_SEED,
    E2S_PAIRS,
    FAMILIES,
    FGI,
    FIR_TAPS,
    FS,
    GHOST,
    LEVELS,
    ROOT,
    TAU_S,
    WIN,
    SelfCheckError,
    analog_noise,
    e2s_chains,
    e2s_stats,
    e2s_windows,
    errors,
    make_cell,
    recording_floor,
)

ADDENDUM = "docs/superpowers/plans/2026-10-19-p2b-g3-true-analog-jmax.md"
ADDENDUM_COMMIT = "ddbc26c"
RUN2_ID = "6214cf52489f"
RUN2_DIR = Path("/srv/gnss/results/E2") / RUN2_ID  # read only
SEGMENTS = ("A", "B")
DECIDING = ("noise", "swept_cw")
CHECKPOINTS_S = (20.0, 40.0, 60.0)
CHECKPOINT_WINDOWS = tuple(round(s * 1000) for s in CHECKPOINTS_S)
SE_STOP = 0.02  # a cell stops at the first checkpoint with SE(R_TA) at most this
TOL, SLOPE_TOL, SE_MAX = 0.10, 0.10, 0.05  # cell_verdict's
H_SEGMENT, H_FAMILY, H_LEVEL = "A", "noise", 55.0
HB_MAX, HA_TOL = 0.92, 0.10
WORKERS = 5  # about 2.5 GB per cell process; the machine is shared (E0 runs beside it)
REPORTED = "reported, not deciding"


# ------------------------------------------------------------------------------------------------ E2 run 2 (read only)
def tau_side(side: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    """Run 2's tau side cells (segment A, 65 dB-Hz; none on B): R(B/A), its SE and the verdict per family and tau."""
    return {
        f: {tau: {k: c[k] for k in ("ratio", "ratio_se", "verdict")} for tau, c in by.items()} for f, by in side.items()
    }


def read_run2(run_dir: Path = RUN2_DIR, run_id: str = RUN2_ID) -> dict[str, Any]:
    """What s29 takes from E2 run 2: each segment's analog SD (it sets the simulation), and, reported, its N0, the
    pre-registered J_max and the tau side check. Refuses any other run."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("run_id") != run_id:
        raise ValueError(f"{run_dir} holds run {manifest.get('run_id')!r}, not E2 run 2 ({run_id})")
    raw = (run_dir / "summary.json").read_bytes()
    summary = json.loads(raw)
    segs = summary["segments"]
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "git_describe": manifest.get("git_describe"),
        "summary_sha256": hashlib.sha256(raw).hexdigest(),
        "analog_sd": {s: float(segs[s]["analog_sd"]) for s in SEGMENTS},
        "n0_per_hz": {s: float(segs[s]["n0_per_hz"]) for s in SEGMENTS},
        "j_max_db_hz_pre_registered": {
            "segments": {s: segs[s]["j_max_db_hz"] for s in SEGMENTS},
            "combined": summary["decisions"]["j_max_db_hz"],
        },
        "tau_cap_db_hz": {s: segs[s]["tau_cap_db_hz"] for s in SEGMENTS},
        "tau_side": {s: tau_side(segs[s]["tau_side"]) for s in SEGMENTS},
    }


# ------------------------------------------------------------------------------------------------ one cell
def adaptive[Z](
    stream: Iterable[tuple[int, Z]],
    stats: Callable[[Z, int], dict[str, Any]],
    checkpoints: Sequence[int],
    se_stop: float = SE_STOP,
) -> dict[str, Any]:
    """The addendum's length: at each checkpoint n (windows) in turn, the statistics on the first n windows; the cell
    stops at the first whose SE(R_TA) is at most se_stop, otherwise at the last. `stream` gives (windows complete,
    state) after each chunk and is read no further than the chunk that completes the stopping checkpoint."""
    cps = sorted(checkpoints)
    trail: list[dict[str, Any]] = []
    for done, z in stream:
        while cps and done >= cps[0]:
            n = cps.pop(0)
            st = stats(z, n)
            se = st["ratio_truth_se"]
            trail.append({"n_windows": n, "ratio_truth": st["ratio_truth"], "ratio_truth_se": se})
            if se <= se_stop or not cps:
                return {**st, "se_stop_reached": bool(se <= se_stop), "checkpoints": trail}
    raise ValueError(f"the stream ended before the last checkpoint ({checkpoints[-1]} windows)")


def ta_cell(
    segment: str, family: str, level: float, sigma: float, n0: float, sd_b: float, checkpoints: Sequence[int]
) -> dict[str, Any]:
    """One cell of s28's E2 structure at a segment's analog SD, with the adaptive length (runs in a worker)."""
    t0, c0 = time.time(), time.process_time()
    c = make_cell(family, level, n0, sd_b, 0, sigma)
    z_ref = gain(GHOST.level_db_hz, n0) * WIN  # the noiseless correlation, as e2s_cell
    st = adaptive(e2s_windows(c, n0, sd_b, max(checkpoints), sigma), lambda z, n: e2s_stats(z, z_ref, n), checkpoints)
    wall, cpu = time.time() - t0, time.process_time() - c0
    print(f"{segment} {family}-{level:g}: {st['length_s']:.0f} s, R_TA {st['ratio_truth']:.3f}+-"
          f"{st['ratio_truth_se']:.3f}, B/truth {st['b_over_truth']:.3f}+-{st['b_over_truth_se']:.3f} "
          f"({wall:.0f} s)", flush=True)  # fmt: skip
    return {"segment": segment, "family": family, "level_db_hz": level,
            "role": "decides" if family in DECIDING else REPORTED, "agc_gain_truth": c.agc_gain, **st,
            "wall_s": wall, "cpu_s": cpu}  # fmt: skip


def front_end(st: Mapping[str, Any]) -> FrontEndCell:
    """The cell cell_verdict judges: the truth pair against mode A."""
    return FrontEndCell(slopes=(float(st["slope_truth"]),), ratio=float(st["ratio_truth"]),
                        ratio_se=float(st["ratio_truth_se"]))  # fmt: skip


def verdict(st: Mapping[str, Any], name: str) -> Verdict:
    return cell_verdict(front_end(st), tol=TOL, slope_tol=SLOPE_TOL, se_max=SE_MAX, cell=name)


# ------------------------------------------------------------------------------------------------ decisions
def decide(fe: Mapping[str, Mapping[str, Mapping[float, FrontEndCell]]]) -> dict[str, Any]:
    """Per family: J_max_TA(s, f) = front_end_jmax over the levels of each segment, J_max_TA(f) = combine_segments over
    the segments. Noise and swept CW decide; any other family is reported."""
    families = list(next(iter(fe.values())))
    out: dict[str, Any] = {}
    for f in families:
        per = {s: front_end_jmax(fe[s][f], tol=TOL, slope_tol=SLOPE_TOL, se_max=SE_MAX, cell=f"segment {s}, {f}")
               for s in fe}  # fmt: skip
        out[f] = {"decides": f in DECIDING, "label": "decides" if f in DECIDING else REPORTED,
                  "j_max_ta_db_hz": per, "combined_db_hz": combine_segments(list(per.values()))}  # fmt: skip
    return out


def hypotheses(cells: Mapping[str, Mapping[str, Mapping[float, Mapping[str, Any]]]]) -> dict[str, Any]:
    """H-B and H-A at segment A, noise, 55 dB-Hz (reported, not deciding)."""
    c = cells[H_SEGMENT][H_FAMILY][H_LEVEL]
    return {
        "label": REPORTED,
        "segment": H_SEGMENT,
        "family": H_FAMILY,
        "level_db_hz": H_LEVEL,
        "n_windows": c["n_windows"],
        "H_B": {"statement": f"mode B's excess / the true front end's <= {HB_MAX}: the yardstick reads low at weak "
                "jamming", "b_over_truth": c["b_over_truth"], "b_over_truth_se": c["b_over_truth_se"],
                "max": HB_MAX, "holds": bool(c["b_over_truth"] <= HB_MAX)},
        "H_A": {"statement": f"|R_TA - 1| < {HA_TOL}", "R_TA": c["ratio_truth"], "R_TA_se": c["ratio_truth_se"],
                "tol": HA_TOL, "holds": bool(abs(c["ratio_truth"] - 1) < HA_TOL)},
        "mode_b_over_a": {"ratio": c["ratio_B"], "se": c["ratio_B_se"]},
    }  # fmt: skip


# ------------------------------------------------------------------------------------------------ known answer
def require(ok: bool, what: str) -> None:
    if not ok:
        raise SelfCheckError(f"s29 self-check failed: {what}")


def zero_jammer_check(sigma: float, n0: float, sd_b: float) -> dict[str, Any]:
    """The jammer at zero, E2's structure at this segment's SD: on the first chunk every jammed twin is its clean twin
    sample for sample (and the true front end's AGC gain is exactly 1); over three chunks of windows every pair's
    correlations equal its clean twin's and its excess is exactly 0."""
    x0 = next(analog_noise(CHUNK, sigma))[1]
    g = gain(GHOST.level_db_hz, n0) * ghost_modulation(GhostSpec.of(GHOST), np.arange(x0.size, dtype=np.int64), FS)
    n, z_ref = 3 * CHUNK // WIN, gain(GHOST.level_db_hz, n0) * WIN
    out: dict[str, Any] = {}
    for fam in FAMILIES:
        c = make_cell(fam, -math.inf, n0, sd_b, 0, sigma)
        require(c.agc_gain == 1.0, f"{fam}: the true front end's AGC gain at j = 0 is {c.agc_gain!r}, not 1")
        ys = e2s_chains(c, _state(c.fe), 0, x0, g, n0, sd_b)
        for p, (cl, jam) in E2S_PAIRS.items():
            require(bool(np.array_equal(ys[jam], ys[cl])), f"{fam}: at j = 0 the {p} twin is not its clean twin")
        _, z = deque(e2s_windows(make_cell(fam, -math.inf, n0, sd_b, 0, sigma), n0, sd_b, n, sigma), maxlen=1)[0]
        excess = {}
        for p, (cl, jam) in E2S_PAIRS.items():
            require(bool(np.array_equal(z[jam][:n], z[cl][:n])), f"{fam}: at j = 0 the {p} pair's correlations differ")
            e_c, e_j = errors(z[cl][:n], z_ref)[0], errors(z[jam][:n], z_ref)[0]
            excess[p] = paired_excess(e_j - e_c, e_c).excess
            require(excess[p] == 0.0, f"{fam}: the {p} pair's excess at j = 0 is {excess[p]!r}, not 0")
        out[fam] = {"samples_equal_first_chunk": int(x0.size), "n_windows": n, "z_equal_clean": True,
                    **{f"excess_{p}": x for p, x in excess.items()}}  # fmt: skip
    return out


# ------------------------------------------------------------------------------------------------ the study
MODEL_LIMITS = {
    "addendum": ["Gaussian analog noise", "a stationary AGC", "one ghost",
                 "no real front-end filter beyond the flat +-10 MHz band"],
    "noted_by_the_implementer": [
        "segments A and B differ only in the analog SD (and so in N0 and every level's power): the white noise, the "
        "ghost, the jammer waveforms and the dither are the same draws (s28's seeds), so the two segments are not "
        "independent replications",
        "N0 is the synthetic recording's (estimate_n0 on a flat band), not run 2's measured N0 (reported beside it); "
        "every level is set against its own N0",
    ],
}  # fmt: skip


REPORTED_ONLY = {
    "label": REPORTED,
    "items": [
        "per cell: mode B's pair (ratio_B, ratio_B_se, b_over_truth, b_over_truth_se, var_dphi_B, cov2_B, excess_B, "
        "var_ratio_B, cancel_B, slope_B) and the variance and cancellation fields of the truth pair",
        "the chirp: its cells (role) and J_max_TA(chirp) (decisions.families.chirp)",
        "hypotheses: H-A and H-B",
        "e2_run2: everything but analog_sd (n0_per_hz, the pre-registered J_max, the tau side check and tau cap)",
        "segments.*.floor beyond n0_per_hz and analog_sd (which set the levels and mode B's reconstruction)",
    ],
}


def main(levels: Sequence[float] = LEVELS, checkpoints: Sequence[int] = CHECKPOINT_WINDOWS) -> None:
    t0 = time.time()
    run2 = read_run2()
    segs: dict[str, dict[str, Any]] = {}
    tmp = Path(tempfile.mkdtemp(prefix="s29-", dir="/dev/shm"))
    try:
        for s in SEGMENTS:
            sigma = run2["analog_sd"][s]
            segs[s] = {"analog_sd_run2": sigma, "floor": recording_floor(tmp, sigma)}
            print(f"segment {s}: analog SD {sigma:.4f} (run 2), N0 {segs[s]['floor']['n0_per_hz']:.4e} counts^2/Hz "
                  f"(run 2's {run2['n0_per_hz'][s]:.4e}), mode B's SD {segs[s]['floor']['analog_sd']:.4f}")  # fmt: skip
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for seg in segs.values():
        seg["self_check"] = zero_jammer_check(
            seg["analog_sd_run2"], seg["floor"]["n0_per_hz"], seg["floor"]["analog_sd"]
        )
    print(f"self-check passed ({time.time() - t0:.0f} s): at j = 0 on both segments every jammed twin is its clean twin "
          "and every pair's excess is exactly 0", flush=True)  # fmt: skip
    jobs = [(s, f, lv) for lv in levels for f in FAMILIES for s in SEGMENTS]  # the weakest (longest) cells first
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as pool:
        futures = {(s, f, lv): pool.submit(ta_cell, s, f, lv, segs[s]["analog_sd_run2"], segs[s]["floor"]["n0_per_hz"],
                                           segs[s]["floor"]["analog_sd"], checkpoints) for s, f, lv in jobs}  # fmt: skip
        done = {job: fut.result() for job, fut in futures.items()}
    cells = {s: {f: {lv: done[(s, f, lv)] for lv in levels} for f in FAMILIES} for s in SEGMENTS}
    for s, by in cells.items():
        for f, lvs in by.items():
            for lv, c in lvs.items():
                c["verdict"] = verdict(c, f"segment {s}, {f} at {lv:g} dB-Hz")
    dec = decide(
        {s: {f: {lv: front_end(c) for lv, c in lvs.items()} for f, lvs in by.items()} for s, by in cells.items()}
    )
    hyp = hypotheses(cells)
    wall = time.time() - t0
    cpu = sum(c["cpu_s"] for by in cells.values() for lvs in by.values() for c in lvs.values())
    print(f"{'seg':3s} {'family':9s} {'J/N0':>5s} {'len':>4s} {'R_TA':>15s} {'beta_TA':>7s} {'verdict':>10s} "
          f"{'R(B/A)':>15s} {'B/truth':>15s}")  # fmt: skip
    for s, by in cells.items():
        for f, lvs in by.items():
            for lv, c in lvs.items():
                print(f"{s:3s} {f:9s} {lv:5.0f} {c['length_s']:4.0f} {c['ratio_truth']:7.3f}+-{c['ratio_truth_se']:.3f} "
                      f"{c['slope_truth']:7.3f} {c['verdict']:>10s} {c['ratio_B']:7.3f}+-{c['ratio_B_se']:.3f} "
                      f"{c['b_over_truth']:7.3f}+-{c['b_over_truth_se']:.3f}")  # fmt: skip
    for f, d in dec.items():
        print(f"J_max_TA({f}) [{d['label']}]: A {d['j_max_ta_db_hz']['A']}, B {d['j_max_ta_db_hz']['B']}, combined "
              f"{d['combined_db_hz']}")  # fmt: skip
    print(f"H-B {hyp['H_B']['holds']} (B/truth {hyp['H_B']['b_over_truth']:.3f}+-{hyp['H_B']['b_over_truth_se']:.3f}); "
          f"H-A {hyp['H_A']['holds']} (R_TA {hyp['H_A']['R_TA']:.3f}+-{hyp['H_A']['R_TA_se']:.3f}); "
          f"{wall:.0f} s, {cpu / 3600:.2f} h of cell CPU")  # fmt: skip
    design = {"construction": "s28's E2 structure (review.s28_twin_pair_quantisation.e2s_chains, e2s_windows, "
              "e2s_stats, make_cell, analog_noise, recording_floor), at each segment's analog SD",
              "fs_hz": FS, "window_samples": WIN, "quant": {"max_level": FGI.max_level, "zero_level": FGI.zero_level},
              "band_hz": BAND_HZ, "fir_taps": FIR_TAPS, "root_seed": ROOT,
              "ghost": {"prn": 32, "doppler_hz": 1510.0, "cn0_db_hz": GHOST.level_db_hz, "seed": GHOST.seed},
              "families": FAMILIES, "levels_db_hz": levels, "mode_b_tau_s": TAU_S, "dither_seed": DITHER_SEED,
              "chunk_samples": CHUNK, "checkpoints_s": [n / 1000 for n in checkpoints], "se_stop": SE_STOP,
              "bootstrap": {"block_windows": BOOT_BLOCK, "n_boot": BOOT_N, "seed": BOOT_SEED},
              "verdict": {"tol": TOL, "slope_tol": SLOPE_TOL, "se_max": SE_MAX},
              "deciding_families": DECIDING}  # fmt: skip
    save("s29_true_analog_jmax", {
        "addendum": {"path": ADDENDUM, "commit": ADDENDUM_COMMIT, "status": "Judgment, owner's decision of 19 Oct "
                     "2026; a deviation, reported beside the pre-registered result (J_max None for every family), "
                     "never in its place"},
        "e2_run2": {"label": f"analog_sd sets the simulation; the rest is {REPORTED}", **run2},
        "design": design,
        "segments": segs,
        "cells": cells,
        "decisions": {"rule": "J_max_TA(s, f) = front_end_jmax over the five levels (the largest pass with no fail at "
                      "or below it; unresolved does not block; None if none passes); J_max_TA(f) = combine_segments "
                      "over A and B (None if either is None, else the minimum). Noise and swept CW decide; the "
                      "chirp is reported. No tau cap (stationary AGC; run 2's tau side check gave none).",
                      "families": dec},
        "hypotheses": hyp,
        "reported_only": REPORTED_ONLY,
        "model_limits": MODEL_LIMITS,
        "run_time": {"wall_s": wall, "cell_cpu_s": cpu, "workers": WORKERS},
    })  # fmt: skip


if __name__ == "__main__":
    main()
