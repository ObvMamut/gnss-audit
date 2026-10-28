"""S28: mode B's twin-pair quantisation artefact, with a true analog (P2b part 2, task 2; gate A3 of the plan's rules).

On segment A, mode B's twin difference had 20-40 % more variance than mode A's, yet the error jamming adds to the
receiver (the paired excess Var(dPhi) + 2 Cov(e_clean, dPhi) = Var(e_jammed) - Var(e_clean)) agreed within 3-8 %: the
extra term cancelled against the clean twin's own error. Mode B reconstructs each sample's analog value independently
within its quantisation cell. Here the analog is known, band-limited and correlated from sample to sample, and the
3-bit front end requantises it after the jammer is added, so what this shows is not an artefact of the reconstruction.

The analog x: complex Gaussian noise, white through a firwin low-pass flat over +-10 MHz at 30.69 MS/s (FGI's front
end), SIGMA per component (segment A's estimate), plus a BPSK C/A ghost (PRN 32, 1510 Hz) at 45 dB-Hz. The recording
is r = requantise(x, FGI). Levels are J/N0 (C/N0 for the ghost) above N0 estimated on r, as E2's survey does
(estimate_n0, inject.gain). Per jammer draw j (E2-A's noise, swept CW and chirp; 50 to 70 dB-Hz), four chains:
  - clean: r (also every chain's clean twin: at j = 0 each of them is r);
  - mode A: r + j;
  - real (3-bit plus AGC, true analog): requantise(G (x + j)), G = sqrt(P_x / P_{x+j}) (a stationary AGC);
  - mode B model: production mode B on r (reconstruct_analog, the pipeline's AGC at E2-A's tau, requantise).
Per 1 ms window: z = sum y conj(s), with s the ghost's exact unit waveform (ghost_modulation); z_ref = A x 30690, the
noiseless correlation; e = Im(z / z_ref) / g with g = mean Re(z / z_ref) over the run, so each chain is normalised by
its own Bussgang and AGC gain. dPhi = e_chain - e_clean. The paired excess, the jammer slopes and the block-bootstrap
SEs (1 s blocks, 400 replicates, seed 7) are e2_frontend's. Each level runs 5, 10 or 20 s of signal, stopping at the
first whose SE of the excess ratio real / A is at most 0.02; a level still above it at 20 s is unresolved. The analog,
the recording and the reconstruction are one stream shared by every cell, as E2's cells share one recording.

Known answers, checked first (the study stops if any fails): with j = 0, every chain is r sample for sample, every
chain's excess is exactly 0 and mode A equals clean; with a jammer, the mode B model chain equals inject() in mode B
bit for bit; and, for the exploratory section, both of its mode B twins equal inject()'s bit for bit (temporary
recordings in /dev/shm, deleted after).

Pre-stated outcomes (the task brief): P1, the real chain's excess ratio within 0.10 of 1 at every resolved level for
noise and swept CW; P2 = A3, at 60 dB-Hz for noise and swept CW the twin-difference variance ratio real / A exceeds
1.10 and A2's cancellation holds, -(2Cov_real - 2Cov_A) >= 0.5 (Var_real - Var_A); P3, the mode B model's excess
within 0.10 of the true analog's at the same levels.

Result (review/results/s28_twin_pair_quantisation.json): A3 = P2 and P1 hold, P3 fails. What P3's failure means: with
the satellite inside the recording, mode B's per-sample reconstruction adds within-cell noise that the clean twin (the
recording itself) does not share, so it does not cancel, and mode B overstates the jamming-induced error of real
satellites by 16-30 % in this simulation (the mode B model's excess over the true analog's, noise and swept CW at the
resolved levels). For injected ghosts (the exploratory section) mode B's excess is within 8 % of the truth's: 3-8 %
low at 60 dB-Hz, 0-1 % low at 70 dB-Hz.

Note (added after the task review; not pre-registered; the pre-registered verdicts stand as computed): the
pre-registered mode A pair r + j carries a Bussgang confound. The ghost inside r has the recording's gain beta = 1.0224,
but the jammer added in mode A passes no quantiser, so normalising mode A by its own gain scales its jammer term by
1 / beta; in the real chain ghost and jammer both pass the quantiser. A3's variance ratio and P1's R(real/A) carry a
factor of about beta^2 = 1.045 (pre-registered against the confound-free E2 structure: variance ratio 1.246 / 1.186
noise-60, 1.171 / 1.126 swept-60; R(real/A) 1.062 / 1.014, 1.018 / 0.994, 1.049 / 1.005, 1.074 / 1.034). In the
confound-free construction A3's conditions still hold (variance ratios 1.186 and 1.126 > 1.10, cancellation shares
0.93 and 1.04 >= 0.5) and the truth's R(real/A) is 0.994-1.034. The same factor means mode A understates the
jamming-induced error of satellites inside a 3-bit recording by about 1 - 1 / beta^2 = 4.3 %; injected ghosts carry no
such factor. JSON: confound_note, computed from the two sections.

Exploratory section (added after the pre-registered results were seen, because P3 failed; not pre-registered; does not
enter gate A): E2's structure. The recording r0 = requantise(x0) holds no ghost; the ghost is injected after the jammer
into every twin, and in mode B after the reconstruction, as E2 does, so both mode B twins requantise reconstruction
plus ghost (checked bit for bit against inject()). Pairs, each against its own clean twin: mode A, truth (3-bit plus
AGC on x0 + j + g) and mode B, noise and swept CW at 60 and 70 dB-Hz, 20 s each. Mode B's excess is 0.92-1.00 times the
truth's there, and its excess ratio against mode A (0.93-1.03) is close to what E2-A measured on segment A's ghosts.

Usage: uv run python -m review.s28_twin_pair_quantisation
"""

from __future__ import annotations

import math
import multiprocessing
import shutil
import tempfile
import time
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from gnssaudit.experiments.e2 import jammer_kind
from gnssaudit.experiments.e2_frontend import PairedExcess, block_bootstrap_se, jammer_slope, paired_excess
from gnssaudit.inject import (
    Disturbance,
    FrontEndSpec,
    GhostSpec,
    IQSource,
    QuantSpec,
    SinkSpec,
    clean_twin,
    disturbance_seed,
    estimate_analog_sd,
    estimate_n0,
    gain,
    ghost_modulation,
    inject,
    reconstruct_analog,
    render,
    requantise,
)
from gnssaudit.inject.pipeline import _front_end, _state  # mode B's AGC and requantisation, as inject() runs them
from review._common import save

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

FS = 30.69e6
WIN = 30690  # samples per 1 ms window
FGI = QuantSpec(max_level=4, zero_level=False)
SIGMA = 1.504  # the analog's SD per component, in counts (segment A's estimate_analog_sd)
BAND_HZ = 20e6  # flat over +-10 MHz (docs/7 section 2.3)
FIR_TAPS = 129
ROOT = 28
GHOST = Disturbance("ghost", {"prn": 32.0, "doppler_hz": 1510.0, "doppler_rate_hzps": 0.0, "chip0": 17.3}, 0.0, 0.0,
                    45.0, disturbance_seed(ROOT, 0, 0, 0))  # fmt: skip
FAMILIES = {"noise": {"bw_hz": 2e6}, "swept_cw": {"span_hz": 1e6, "period_s": 0.01},
            "chirp": {"span_hz": 1e7, "period_s": 1e-5}}  # experiments/E2-A.yaml  # fmt: skip
LEVELS = (50.0, 55.0, 60.0, 65.0, 70.0)
TAU_S = 0.01  # E2-A's mode B AGC time constant
DITHER_SEED = disturbance_seed(ROOT, 800, 0, 0)
CHUNK = 1 << 23  # samples per chunk: whole RNG blocks, so the AGC's sub-blocks fall where inject() puts them
CHECKPOINTS_S = (5.0, 10.0, 20.0)
SE_MAX = 0.02
BOOT_BLOCK, BOOT_N, BOOT_SEED = 1000, 400, 7  # 1 s blocks of 1 ms windows, as the pre-registered rules
GATE_FAMILIES, GATE_LEVEL = ("noise", "swept_cw"), 60.0
VAR_RATIO_MIN, CANCEL_SHARE, TOL = 1.10, 0.5, 0.10


class SelfCheckError(RuntimeError):
    """A known answer failed: nothing after it means anything."""


def require(ok: bool, what: str) -> None:
    if not ok:
        raise SelfCheckError(f"s28 self-check failed: {what}")


# ------------------------------------------------------------------------------------------------ the stream
def analog_noise(n_total: int, sigma: float = SIGMA) -> Iterator[tuple[int, ComplexArray]]:
    """(lo, samples) of the band-limited analog noise in CHUNK pieces: white noise through the firwin low-pass, scaled
    to sigma per component, with the filter's history carried so that the stream is one stationary process."""
    h = signal.firwin(FIR_TAPS, BAND_HZ / 2, fs=FS)
    h = h * (sigma / np.sqrt(np.sum(h**2)))

    def white(key: int) -> ComplexArray:
        rng = np.random.default_rng([ROOT, 1, key])
        return rng.standard_normal(CHUNK) + 1j * rng.standard_normal(CHUNK)

    hist = white(0)[CHUNK - (FIR_TAPS - 1) :]
    for k, lo in enumerate(range(0, n_total, CHUNK)):
        w = white(k + 1)
        x = signal.oaconvolve(np.concatenate([hist, w]), h, mode="valid")
        hist = w[CHUNK - (FIR_TAPS - 1) :]
        yield lo, np.asarray(x[: min(CHUNK, n_total - lo)], dtype=np.complex128)


@dataclass(frozen=True)
class Frame:
    """One chunk of what every cell shares."""

    lo: int
    x: ComplexArray  # the analog: noise plus the ghost
    r: ComplexArray  # the recording
    recon: ComplexArray  # mode B's reconstruction of the recording
    sc: ComplexArray  # conj of the ghost's unit waveform


def frames(n_total: int, n0: float, sd_b: float) -> Iterator[Frame]:
    g, amp = GhostSpec.of(GHOST), gain(GHOST.level_db_hz, n0)
    for lo, noise in analog_noise(n_total):
        s = ghost_modulation(g, np.arange(lo, lo + noise.size, dtype=np.int64), FS)
        x = noise + amp * s
        r = requantise(x, FGI)
        yield Frame(lo, x, r, reconstruct_analog(r, FGI, sd_b, DITHER_SEED, lo), np.conj(s))


def window_sums(y: ComplexArray, sc: ComplexArray, lo: int) -> tuple[NDArray[np.int64], ComplexArray]:
    """(window index, sum of y conj(s) over the window's samples in this chunk) on the absolute 1 ms grid."""
    edges = np.unique(np.concatenate([[0], np.arange(-lo % WIN, y.size, WIN)]))
    return (lo + edges) // WIN, np.add.reduceat(y * sc, edges)


# ------------------------------------------------------------------------------------------------ one cell
@dataclass
class Cell:
    family: str
    level: float
    jammer: Disturbance
    agc_gain: float  # G of the real chain: sqrt(P_x / P_{x+j})
    fe: FrontEndSpec  # the mode B model's front end
    st: Any  # the pipeline's block state (the mode B AGC's smoothed powers)
    z: dict[str, ComplexArray] = field(default_factory=dict)


def make_cell(family: str, level: float, n0: float, sd_b: float, n_win: int, sigma: float = SIGMA) -> Cell:
    seed = disturbance_seed(ROOT, 1 + list(FAMILIES).index(family), 0 if math.isinf(level) else int(level), 1)
    jam = Disturbance(jammer_kind(family), dict(FAMILIES[family]), 0.0, 0.0, level, seed)
    p_x = 2 * sigma**2 + gain(GHOST.level_db_hz, n0) ** 2
    g = math.sqrt(p_x / (p_x + gain(level, n0) ** 2))  # unit-power waveforms: J = 10^(L/10) N0; exactly 1 at j = 0
    fe = FrontEndSpec(1, 1, 1.0, mode="B", agc_tau_s=TAU_S, quant=FGI, dither_seed=DITHER_SEED, analog_sd=sd_b)
    z = {k: np.zeros(n_win + 1, dtype=np.complex128) for k in ("A", "real", "B")}
    return Cell(family, level, jam, g, fe, _state(fe), z)


def chains(c: Cell, fr: Frame, n0: float) -> dict[str, ComplexArray]:
    j = render(c.jammer, fr.lo, fr.r.size, FS, n0)
    return {
        "A": fr.r + j,
        "real": requantise(c.agc_gain * (fr.x + j), FGI),
        "B": _front_end(fr.recon + j, fr.lo, 0, c.fe, FS, c.st, fr.recon),
    }


def errors(z: ComplexArray, z_ref: float) -> tuple[FloatArray, float]:
    """Im(z / z_ref) over the chain's own gain g = mean Re(z / z_ref), and g."""
    u = z / z_ref
    g = float(np.mean(u.real))
    return u.imag / g, g


@dataclass(frozen=True)
class Errors:
    e_c: FloatArray  # the clean twin's error per window
    gains: dict[str, float]  # per chain (clean included): mean Re(z / z_ref)
    dphi: dict[str, FloatArray]  # per jammed chain: e_chain - e_clean
    excess: dict[str, PairedExcess]


def chain_errors(zc: ComplexArray, c: Cell, n: int, z_ref: float) -> Errors:
    e_c, g_c = errors(zc[:n], z_ref)
    gains, dphi, excess = {"clean": g_c}, {}, {}
    for k, z in c.z.items():
        e, gains[k] = errors(z[:n], z_ref)
        dphi[k] = e - e_c
        excess[k] = paired_excess(dphi[k], e_c)
    return Errors(e_c, gains, dphi, excess)


def ratio_se(num: FloatArray, den: FloatArray, e_c: FloatArray) -> float:
    def stat(a: FloatArray, b: FloatArray, c: FloatArray) -> float:
        return paired_excess(a, c).excess / paired_excess(b, c).excess

    return block_bootstrap_se(stat, [num, den, e_c], block=BOOT_BLOCK, n_boot=BOOT_N, seed=BOOT_SEED)


def cell_stats(zc: ComplexArray, c: Cell, n: int, z_ref: float) -> dict[str, Any]:
    """Everything reported for one cell on its first n windows."""
    er = chain_errors(zc, c, n, z_ref)
    d, x = er.dphi, er.excess
    out: dict[str, Any] = {"n_windows": n, "length_s": n / 1000, "agc_gain_real": c.agc_gain,
                           "var_e_clean": float(np.var(er.e_c)), "gains": er.gains}  # fmt: skip
    for k in ("A", "real", "B"):
        out[f"var_dphi_{k}"], out[f"cov2_{k}"], out[f"excess_{k}"] = x[k].var_dphi, x[k].cov2, x[k].excess
    for k in ("real", "B"):
        cancel = -(x[k].cov2 - x["A"].cov2)
        extra = x[k].var_dphi - x["A"].var_dphi
        out[f"var_ratio_{k}"] = x[k].var_dphi / x["A"].var_dphi
        out[f"cancel_{k}"] = {"cancel": cancel, "extra_var": extra, "share": cancel / extra if extra else math.nan,
                              "holds": bool(cancel >= CANCEL_SHARE * extra)}  # fmt: skip
        out[f"slope_{k}"] = jammer_slope(d["A"], d[k])
        out[f"ratio_{k}"] = x[k].excess / x["A"].excess
        out[f"ratio_{k}_se"] = ratio_se(d[k], d["A"], er.e_c)
    out["b_over_real"] = x["B"].excess / x["real"].excess
    out["b_over_real_se"] = ratio_se(d["B"], d["real"], er.e_c)
    out["resolved"] = bool(out["ratio_real_se"] <= SE_MAX)
    return out


Stop = Callable[[ComplexArray, Cell, int, float, bool], dict[str, Any] | None]


def adaptive(zc: ComplexArray, c: Cell, n: int, z_ref: float, last: bool) -> dict[str, Any] | None:
    """A cell stops at the first checkpoint where SE(real / A) <= SE_MAX, or at the last, resolved or not."""
    st = cell_stats(zc, c, n, z_ref)
    return st if st["resolved"] or last else None


def run_cells(cells: list[Cell], n0: float, sd_b: float, checkpoints: tuple[float, ...], stop: Stop) -> list[Any]:
    """Feed every cell the shared stream; at each checkpoint (seconds of windows) a cell whose `stop` gives a result
    leaves. Results in the cells' order."""
    n_win = round(checkpoints[-1] * 1000)
    zc = np.zeros(n_win + 1, dtype=np.complex128)
    z_ref = gain(GHOST.level_db_hz, n0) * WIN  # the noiseless correlation: A |s|^2 summed over a window
    out: dict[int, Any] = {}
    cps = list(checkpoints)
    for fr in frames(n_win * WIN, n0, sd_b):
        idx, s = window_sums(fr.r, fr.sc, fr.lo)
        zc[idx] += s
        for i, c in enumerate(cells):
            if i not in out:
                for k, y in chains(c, fr, n0).items():
                    idx, s = window_sums(y, fr.sc, fr.lo)
                    c.z[k][idx] += s
        done = fr.lo + fr.r.size
        while cps and done >= round(cps[0] * 1000) * WIN:
            n = round(cps.pop(0) * 1000)
            for i, c in enumerate(cells):
                if i not in out and (res := stop(zc, c, n, z_ref, not cps)) is not None:
                    out[i] = res
        if len(out) == len(cells):
            break
    return [out[i] for i in range(len(cells))]


def run_family(family: str, n0: float, sd_b: float) -> dict[float, Any]:
    t0 = time.time()
    n_win = round(CHECKPOINTS_S[-1] * 1000)
    res = run_cells([make_cell(family, lv, n0, sd_b, n_win) for lv in LEVELS], n0, sd_b, CHECKPOINTS_S, adaptive)
    print(f"{family}: done in {time.time() - t0:.0f} s", flush=True)
    return dict(zip(LEVELS, res, strict=True))


# ------------------------------------------------------------------------------------------------ known answers
def write_ibyte(r: ComplexArray, path: Path) -> IQSource:
    """The recording as interleaved int8 I/Q (its values are FGI levels), as a source inject() and the survey read."""
    iq = np.empty(2 * r.size, dtype=np.int8)
    iq[0::2], iq[1::2] = r.real, r.imag
    iq.tofile(path)
    return IQSource(path=path, item_type="ibyte", fs_hz=FS, sample_start=0, sample_end=r.size)


def recording_floor(tmp: Path, sigma: float = SIGMA) -> dict[str, float]:
    """N0 and the analog SD as E2's survey estimates them (estimate_n0, estimate_analog_sd) on the first 2^24 samples
    of the simulated recording, noise only: the ghost's level is set from this N0 (at -28 dB in band it would not move
    a median PSD)."""
    r = np.concatenate([requantise(x, FGI) for _, x in analog_noise(2 * CHUNK, sigma)])
    src = write_ibyte(r, tmp / "floor.bin")
    floor = estimate_n0(src, frontend_bw_hz=BAND_HZ)
    sd = estimate_analog_sd(src, FGI)
    src.path.unlink()
    return {"n0_per_hz": floor.n0_per_hz, "n0_spread_db": floor.spread_db, "analog_sd": sd,
            "n0_analog_per_hz": 2 * sigma**2 / BAND_HZ, "recording_power": float(np.mean(np.abs(r) ** 2))}  # fmt: skip


def zero_answer(zc: ComplexArray, c: Cell, n: int, z_ref: float, last: bool) -> dict[str, Any]:
    er = chain_errors(zc, c, n, z_ref)
    for k, x in er.excess.items():
        require(x.excess == 0.0, f"{c.family}: the {k} chain's excess at j = 0 is {x.excess!r}, not 0")
        require(bool(np.array_equal(c.z[k][:n], zc[:n])), f"{c.family}: the {k} chain's correlations at j = 0 differ "
                "from clean's")  # fmt: skip
    return {"n_windows": n, **{f"excess_{k}": x.excess for k, x in er.excess.items()}, "z_equal_clean": True}


def zero_jammer_check(n0: float, sd_b: float) -> dict[str, Any]:
    """j = 0: every chain is the recording sample for sample on the first chunk, and over three chunks of windows
    every chain's excess is exactly 0 and its correlations are clean's (so mode A equals clean)."""
    fr = next(frames(CHUNK, n0, sd_b))
    for fam in FAMILIES:
        c = make_cell(fam, -math.inf, n0, sd_b, 0)
        require(c.agc_gain == 1.0, f"{fam}: the real chain's AGC gain at j = 0 is {c.agc_gain!r}, not 1")
        for k, y in chains(c, fr, n0).items():
            require(bool(np.array_equal(y, fr.r)), f"{fam}: the {k} chain at j = 0 is not the recording")
    n = 3 * CHUNK // WIN
    cells = [make_cell(f, -math.inf, n0, sd_b, n) for f in FAMILIES]
    return dict(zip(FAMILIES, run_cells(cells, n0, sd_b, (n / 1000,), zero_answer), strict=True))


def inject_check(n0: float, sd_b: float, tmp: Path) -> dict[str, Any]:
    """The mode B model chain against inject() in mode B (scale 1, no decimation) on the first chunk of the recording,
    with E2-A's noise jammer at 70 dB-Hz: bit for bit."""
    fr = next(frames(CHUNK, n0, sd_b))
    src, out = write_ibyte(fr.r, tmp / "rec.bin"), tmp / "modeb.bin"
    c = make_cell("noise", 70.0, n0, sd_b, 0)
    y = chains(c, fr, n0)["B"]
    rep = inject(src, [c.jammer], c.fe, SinkSpec("file", out), n0_per_hz=n0)
    got = np.fromfile(out, dtype="<i2").astype(np.float64)
    want = np.empty(2 * y.size)
    want[0::2], want[1::2] = y.real, y.imag
    require(rep.clip_count == 0 and bool(np.array_equal(got, want)), "the mode B model chain is not inject()'s mode B")
    changed = float(np.mean(y != fr.r))
    require(changed > 0.01, f"the 70 dB-Hz jammer changed only {changed:.4f} of the mode B samples")
    src.path.unlink()
    out.unlink()
    return {"samples": int(y.size), "family": "noise", "level_db_hz": 70.0, "bitwise_equal": True,
            "share_of_samples_changed": changed, "output_sha256": rep.output_sha256}  # fmt: skip


# ------------------------------------------------------------------------------------------------ verdicts
def verdicts(cells: dict[str, dict[float, Any]]) -> dict[str, Any]:
    gate = {f: {"var_ratio_real": cells[f][GATE_LEVEL]["var_ratio_real"], **cells[f][GATE_LEVEL]["cancel_real"],
                "resolved": cells[f][GATE_LEVEL]["resolved"]} for f in GATE_FAMILIES}  # fmt: skip
    a3 = all(g["var_ratio_real"] > VAR_RATIO_MIN and g["holds"] for g in gate.values())
    resolved = {f: [lv for lv, c in sorted(cells[f].items()) if c["resolved"] and lv <= 70.0] for f in cells}

    def within(key: str) -> bool:  # at every resolved level of each gate family, and at least one level resolved
        return all(resolved[f] and all(abs(cells[f][lv][key] - 1) < TOL for lv in resolved[f]) for f in GATE_FAMILIES)

    p1, p3 = within("ratio_real"), within("b_over_real")
    return {"A3": a3, "P1": p1, "P2": a3, "P3": p3, "a3_cells": gate, "resolved_levels": resolved,
            "rules": {"gate_families": GATE_FAMILIES, "gate_level_db_hz": GATE_LEVEL, "var_ratio_min": VAR_RATIO_MIN,
                      "cancel_share": CANCEL_SHARE, "tol": TOL, "se_max": SE_MAX}}  # fmt: skip


# ------------------------------------------------------------------------- exploratory: E2's structure (not gate A)
EXPLORATORY = ("exploratory: added after the pre-registered results were seen (P3 failed); not pre-registered; does not "
               "enter gate A")  # fmt: skip
E2S_CELLS = (("noise", 60.0), ("swept_cw", 60.0), ("noise", 70.0), ("swept_cw", 70.0))
E2S_SECONDS = 20.0
E2S_PAIRS = {"A": ("A_clean", "A"), "truth": ("truth_clean", "truth"), "B": ("B_clean", "B")}  # (clean, jammed)


def e2s_chains(
    c: Cell, st_clean: Any, lo: int, x0: ComplexArray, g: ComplexArray, n0: float, sd_b: float
) -> dict[str, ComplexArray]:
    """E2's structure: the recording r0 = requantise(x0) holds no ghost; the ghost g is injected into every twin after
    the jammer, in inject()'s order, and in mode B after the reconstruction, so both mode B twins requantise
    reconstruction plus ghost. Each pair has its own clean twin (the jammer at zero gain)."""
    r0 = requantise(x0, FGI)
    rec = reconstruct_analog(r0, FGI, sd_b, DITHER_SEED, lo)
    j = render(c.jammer, lo, x0.size, FS, n0)
    return {
        "A_clean": r0 + g,
        "A": r0 + j + g,
        "truth_clean": requantise(x0 + g, FGI),
        "truth": requantise(c.agc_gain * (x0 + j + g), FGI),
        "B_clean": _front_end(rec + g, lo, 0, c.fe, FS, st_clean, rec),
        "B": _front_end(rec + j + g, lo, 0, c.fe, FS, c.st, rec),
    }


def e2s_stats(z: dict[str, ComplexArray], z_ref: float, n: int) -> dict[str, Any]:
    """Per pair against its own clean twin: e, dPhi and the paired excess as in the pre-registered section; the
    bootstrap draws the same blocks for all six series."""
    e, gains = {}, {}
    for k, zk in z.items():
        e[k], gains[k] = errors(zk[:n], z_ref)
    d = {p: e[jam] - e[cl] for p, (cl, jam) in E2S_PAIRS.items()}
    x = {p: paired_excess(d[p], e[cl]) for p, (cl, _) in E2S_PAIRS.items()}
    order = list(E2S_PAIRS)
    arrays = [a for p, (cl, _) in E2S_PAIRS.items() for a in (d[p], e[cl])]

    def se(num: str, den: str) -> float:
        def stat(*a: FloatArray) -> float:
            ex = {p: paired_excess(a[2 * i], a[2 * i + 1]).excess for i, p in enumerate(order)}
            return ex[num] / ex[den]

        return block_bootstrap_se(stat, arrays, block=BOOT_BLOCK, n_boot=BOOT_N, seed=BOOT_SEED)

    out: dict[str, Any] = {"n_windows": n, "length_s": n / 1000, "gains": gains}
    for p in E2S_PAIRS:
        out[f"var_dphi_{p}"], out[f"cov2_{p}"], out[f"excess_{p}"] = x[p].var_dphi, x[p].cov2, x[p].excess
    for p in ("truth", "B"):
        cancel = -(x[p].cov2 - x["A"].cov2)
        extra = x[p].var_dphi - x["A"].var_dphi
        out[f"var_ratio_{p}"] = x[p].var_dphi / x["A"].var_dphi
        out[f"cancel_{p}"] = {"cancel": cancel, "extra_var": extra, "share": cancel / extra if extra else math.nan,
                              "holds": bool(cancel >= CANCEL_SHARE * extra)}  # fmt: skip
        out[f"slope_{p}"] = jammer_slope(d["A"], d[p])
        out[f"ratio_{p}"] = x[p].excess / x["A"].excess
        out[f"ratio_{p}_se"] = se(p, "A")
    out["b_over_truth"] = x["B"].excess / x["truth"].excess
    out["b_over_truth_se"] = se("B", "truth")
    return out


def e2s_windows(
    c: Cell, n0: float, sd_b: float, n: int, sigma: float = SIGMA
) -> Iterator[tuple[int, dict[str, ComplexArray]]]:
    """E2's structure for one cell over the first n windows of the analog stream: after each chunk, (windows complete
    so far, every twin's correlation per window). The arrays are updated in place from chunk to chunk."""
    st_clean = _state(c.fe)
    gs, amp = GhostSpec.of(GHOST), gain(GHOST.level_db_hz, n0)
    z = {k: np.zeros(n + 1, dtype=np.complex128) for pair in E2S_PAIRS.values() for k in pair}
    for lo, x0 in analog_noise(n * WIN, sigma):
        s = ghost_modulation(gs, np.arange(lo, lo + x0.size, dtype=np.int64), FS)
        sc = np.conj(s)
        for k, y in e2s_chains(c, st_clean, lo, x0, amp * s, n0, sd_b).items():
            idx, v = window_sums(y, sc, lo)
            z[k][idx] += v
        yield (lo + x0.size) // WIN, z


def e2s_cell(family: str, level: float, n0: float, sd_b: float) -> dict[str, Any]:
    """One cell of E2's structure over E2S_SECONDS of the same analog noise stream as the pre-registered section."""
    t0 = time.time()
    n = round(E2S_SECONDS * 1000)
    _, z = deque(e2s_windows(make_cell(family, level, n0, sd_b, 0), n0, sd_b, n), maxlen=1)[0]  # the whole stream
    out = e2s_stats(z, gain(GHOST.level_db_hz, n0) * WIN, n)
    print(f"exploratory {family}-{level:g}: done in {time.time() - t0:.0f} s", flush=True)
    return {"family": family, "level_db_hz": level, **out}


def e2s_inject_check(n0: float, sd_b: float, tmp: Path) -> dict[str, Any]:
    """E2's structure, mode B: the jammed and the clean twin of e2s_chains equal inject() in mode B on the ghost-free
    recording with the jammer and the ghost as disturbances (clean: clean_twin of them), bit for bit, first chunk."""
    x0 = next(analog_noise(CHUNK))[1]
    src = write_ibyte(requantise(x0, FGI), tmp / "rec0.bin")
    c = make_cell("noise", 70.0, n0, sd_b, 0)
    s = ghost_modulation(GhostSpec.of(GHOST), np.arange(0, x0.size, dtype=np.int64), FS)
    ys = e2s_chains(c, _state(c.fe), 0, x0, gain(GHOST.level_db_hz, n0) * s, n0, sd_b)
    for name, ds in (("B", [c.jammer, GHOST]), ("B_clean", clean_twin([c.jammer, GHOST]))):
        path = tmp / f"{name}.bin"
        rep = inject(src, ds, c.fe, SinkSpec("file", path), n0_per_hz=n0)
        got = np.fromfile(path, dtype="<i2").astype(np.float64)
        want = np.empty(2 * x0.size)
        want[0::2], want[1::2] = ys[name].real, ys[name].imag
        require(rep.clip_count == 0 and bool(np.array_equal(got, want)), f"E2's structure: {name} is not inject()'s")
        path.unlink()
    src.path.unlink()
    return {"samples": int(x0.size), "family": "noise", "level_db_hz": 70.0, "twins": ["B", "B_clean"],
            "bitwise_equal": True}  # fmt: skip


def _against(vals: list[float], what: str) -> str:
    """Ratios to a reference in words that follow their sign."""
    lo, hi = min(vals), max(vals)
    if lo > 1:
        return f"overstates {what} by {100 * (lo - 1):.1f}-{100 * (hi - 1):.1f} %"
    if hi < 1:
        return f"understates {what} by {100 * (1 - hi):.1f}-{100 * (1 - lo):.1f} %"
    return f"is {lo:.3f}-{hi:.3f} times {what}"


def p3_meaning(cells: dict[str, dict[float, Any]], v: dict[str, Any], e2s: list[dict[str, Any]]) -> str:
    """What P3's verdict means, in words that follow the verdict and the signs of the measured ratios."""
    at = [(f, lv) for f in GATE_FAMILIES for lv in v["resolved_levels"][f]]
    over = [cells[f][lv]["b_over_real"] for f, lv in at]
    bt = [c["b_over_truth"] for c in e2s]
    vs_real, vs_truth = _against(over, "the true front end's"), _against(bt, "the truth's")
    text = (f"P3 {'holds' if v['P3'] else 'fails'}. With the satellite inside the recording, mode B's jamming-induced "
            f"error {vs_real} in this simulation (the mode B model's excess over the "
            f"true analog's, {min(over):.3f}-{max(over):.3f}, noise and swept CW at the resolved levels).")  # fmt: skip
    if not v["P3"] and min(over) > 1:
        sb = [cells[f][lv]["cancel_B"]["share"] for f, lv in at]
        sr = [cells[f][lv]["cancel_real"]["share"] for f, lv in at]
        text += (f" Its per-sample reconstruction adds within-cell noise that the clean twin (the recording itself) "
                 f"does not share, so it does not cancel: mode B's cancellation share is {min(sb):.2f}-{max(sb):.2f} "
                 f"at those levels, the true analog's {min(sr):.2f}-{max(sr):.2f}.")  # fmt: skip
    return text + (f" For injected ghosts (E2's structure, the exploratory section), both mode B twins requantise "
                   f"reconstruction plus ghost, and mode B's excess {vs_truth} ({min(bt):.3f}-"
                   f"{max(bt):.3f}, noise and swept CW at 60 and 70 dB-Hz).")  # fmt: skip


def confound_note(cells: dict[str, dict[float, Any]], e2s: list[dict[str, Any]]) -> dict[str, Any]:
    """The pre-registered mode A pair's Bussgang factor, from the two sections' numbers (added after the task review;
    not pre-registered; the pre-registered verdicts stand as computed)."""
    gains = [c["gains"]["clean"] for by in cells.values() for c in by.values()]
    beta = float(np.mean(gains))
    gain_a = float(np.mean([c["gains"]["A_clean"] for c in e2s]))
    paired = []
    for c in e2s:
        p = cells[c["family"]][c["level_db_hz"]]
        paired.append({"family": c["family"], "level_db_hz": c["level_db_hz"],
                       **{k: {"pre_registered": p[a], "e2_structure_truth": c[b], "quotient": p[a] / c[b]}
                          for k, a, b in (("var_ratio", "var_ratio_real", "var_ratio_truth"),
                                          ("excess_ratio_real_over_a", "ratio_real", "ratio_truth"))},
                       "cancel_share": {"pre_registered": p["cancel_real"]["share"],
                                        "e2_structure_truth": c["cancel_truth"]["share"]}})  # fmt: skip
    gate = {c["family"]: {"var_ratio": c["var_ratio_truth"], "var_ratio_above_min": c["var_ratio_truth"] > VAR_RATIO_MIN,
                          "cancel_share": c["cancel_truth"]["share"], "cancel_holds": c["cancel_truth"]["holds"]}
            for c in e2s if c["level_db_hz"] == GATE_LEVEL and c["family"] in GATE_FAMILIES}  # fmt: skip
    hold = len(gate) == len(GATE_FAMILIES) and all(g["var_ratio_above_min"] and g["cancel_holds"] for g in gate.values())  # fmt: skip
    rt = [c["ratio_truth"] for c in e2s]
    bias = 1 / beta**2 - 1

    def pairs(key: str) -> str:
        return ", ".join(f"{q['family']}-{q['level_db_hz']:g} {q[key]['pre_registered']:.3f}/"
                         f"{q[key]['e2_structure_truth']:.3f}" for q in paired)  # fmt: skip

    conds = "; ".join(f"{f} variance ratio {g['var_ratio']:.3f} ({'>' if g['var_ratio_above_min'] else '<='} "
                      f"{VAR_RATIO_MIN}), cancellation share {g['cancel_share']:.2f} "
                      f"({'>=' if g['cancel_holds'] else '<'} {CANCEL_SHARE})" for f, g in gate.items())  # fmt: skip
    statement = (
        f"Not pre-registered (added after the task review; the pre-registered verdicts stand as computed). The "
        f"pre-registered mode A pair is r + j: the ghost inside r carries the recording's Bussgang gain beta = "
        f"{beta:.4f} (the clean twin's mean Re(z / z_ref)), but the jammer added in mode A passes no quantiser, so "
        f"normalising mode A by its own gain scales its jammer term by 1 / beta; in the real chain ghost and jammer "
        f"both pass the quantiser. A3's variance ratio and P1's R(real/A) therefore carry a factor of about beta^2 = "
        f"{beta**2:.3f}. Pre-registered / confound-free (the exploratory E2 structure, ghost and jammer both injected; "
        f"mode A's gain {gain_a:.4f}): variance ratio {pairs('var_ratio')}; R(real/A) "
        f"{pairs('excess_ratio_real_over_a')}; cancellation share {pairs('cancel_share')}. In the confound-free "
        f"construction A3's conditions {'still hold' if hold else 'do not hold'}: {conds}; the truth's R(real/A) is "
        f"{min(rt):.3f}-{max(rt):.3f}. The same factor means mode A {'understates' if bias < 0 else 'overstates'} the "
        f"jamming-induced error of satellites inside a 3-bit recording by about {100 * abs(bias):.1f} % "
        f"(1 / beta^2 - 1 = {bias:+.4f}); injected ghosts, added digitally like the jammer, carry no such factor."
    )  # fmt: skip
    return {"beta": beta, "beta_range": [min(gains), max(gains)], "beta_squared": beta**2,
            "mode_a_gain_e2_structure": gain_a, "paired": paired,
            "a3_conditions_confound_free": {"cells": gate, "hold": hold},
            "truth_excess_ratio_real_over_a_confound_free": [min(rt), max(rt)],
            "mode_a_error_bias_satellites_in_recording": bias, "statement": statement}  # fmt: skip


def derived(cells: dict[str, dict[float, Any]], v: dict[str, Any], e2s: list[dict[str, Any]]) -> dict[str, Any]:
    """The keys computed at save time from the two sections (neither enters gate A)."""
    return {"p3_meaning": p3_meaning(cells, v, e2s), "confound_note": confound_note(cells, e2s)}


def main() -> None:
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="s28-", dir="/dev/shm"))
    try:
        floor = recording_floor(tmp)
        n0, sd_b = floor["n0_per_hz"], floor["analog_sd"]
        print(f"N0 {n0:.4e} counts^2/Hz (analog {floor['n0_analog_per_hz']:.4e}), analog SD estimate {sd_b:.4f}")
        zero = zero_jammer_check(n0, sd_b)
        modeb = inject_check(n0, sd_b, tmp)
        print(f"self-check passed ({time.time() - t0:.0f} s): at j = 0 every chain is the recording, every excess is "
              "exactly 0 and mode A equals clean; the mode B model is inject()'s mode B bit for bit", flush=True)  # fmt: skip
        e2s_check = e2s_inject_check(n0, sd_b, tmp)
        print("exploratory check passed: E2's structure, both mode B twins are inject()'s bit for bit", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(FAMILIES) + len(E2S_CELLS), mp_context=ctx) as pool:
        futures = {f: pool.submit(run_family, f, n0, sd_b) for f in FAMILIES}
        e2s_futures = [pool.submit(e2s_cell, f, lv, n0, sd_b) for f, lv in E2S_CELLS]
        cells = {f: fut.result() for f, fut in futures.items()}
        e2s = [fut.result() for fut in e2s_futures]
    v = verdicts(cells)
    print(f"{'family':9s} {'J/N0':>5s} {'len':>4s} {'VarR/VarA':>9s} {'cancel/extra':>12s} {'R(real/A)':>15s} "
          f"{'slope':>6s} {'R(B/A)':>15s} {'B/real':>15s} resolved")  # fmt: skip
    for f, by in cells.items():
        for lv, c in sorted(by.items()):
            print(f"{f:9s} {lv:5.0f} {c['length_s']:4.0f} {c['var_ratio_real']:9.3f} "
                  f"{c['cancel_real']['share']:8.2f} ({'y' if c['cancel_real']['holds'] else 'n'}) "
                  f"{c['ratio_real']:7.3f}+-{c['ratio_real_se']:.3f} {c['slope_real']:6.3f} "
                  f"{c['ratio_B']:7.3f}+-{c['ratio_B_se']:.3f} {c['b_over_real']:7.3f}+-{c['b_over_real_se']:.3f} "
                  f"{c['resolved']}")  # fmt: skip
    print(f"A3 {v['A3']}  P1 {v['P1']}  P2 {v['P2']}  P3 {v['P3']}  ({time.time() - t0:.0f} s)")
    extra = derived(cells, v, e2s)
    print(extra["p3_meaning"])
    print(extra["confound_note"]["statement"])
    print(f"{EXPLORATORY}:")
    print(f"{'family':9s} {'J/N0':>5s} {'len':>4s} {'Var B/A':>7s} {'cancel B':>9s} {'R(B/A)':>15s} {'Var T/A':>7s} "
          f"{'cancel T':>9s} {'R(T/A)':>15s} {'B/truth':>15s}")  # fmt: skip
    for c in e2s:
        print(f"{c['family']:9s} {c['level_db_hz']:5.0f} {c['length_s']:4.0f} {c['var_ratio_B']:7.3f} "
              f"{c['cancel_B']['share']:9.2f} {c['ratio_B']:7.3f}+-{c['ratio_B_se']:.3f} {c['var_ratio_truth']:7.3f} "
              f"{c['cancel_truth']['share']:9.2f} {c['ratio_truth']:7.3f}+-{c['ratio_truth_se']:.3f} "
              f"{c['b_over_truth']:7.3f}+-{c['b_over_truth_se']:.3f}")  # fmt: skip
    design = {"fs_hz": FS, "window_samples": WIN, "quant": {"max_level": FGI.max_level, "zero_level": FGI.zero_level},
              "sigma_per_component": SIGMA, "band_hz": BAND_HZ, "fir_taps": FIR_TAPS, "root_seed": ROOT,
              "ghost": {"prn": 32, "doppler_hz": 1510.0, "cn0_db_hz": GHOST.level_db_hz, "seed": GHOST.seed},
              "families": FAMILIES, "levels_db_hz": LEVELS, "mode_b_tau_s": TAU_S, "dither_seed": DITHER_SEED,
              "chunk_samples": CHUNK, "checkpoints_s": CHECKPOINTS_S,
              "bootstrap": {"block_windows": BOOT_BLOCK, "n_boot": BOOT_N, "seed": BOOT_SEED}}  # fmt: skip
    exploratory = {
        "label": EXPLORATORY,
        "construction": "the recording r0 = requantise(x0) holds no ghost; the ghost is injected after the jammer into "
        "every twin, and in mode B after the reconstruction (as E2 does), so both mode B twins requantise "
        "reconstruction plus ghost. Pairs, each against its own clean twin: mode A r0 + j + g vs r0 + g; truth "
        "requantise(G (x0 + j + g)) vs requantise(x0 + g); mode B Q(AGC(x^ + j + g)) vs Q(AGC(x^ + g)). Same analog "
        "noise, ghost, jammer seeds, N0, analog SD and statistics as the pre-registered section; fixed length.",
        "seconds": E2S_SECONDS,
        "mode_b_vs_inject": e2s_check,
        "cells": e2s,
    }
    save("s28_twin_pair_quantisation", {"design": design, "floor": floor, "self_check": {"zero_jammer": zero,
         "mode_b_vs_inject": modeb}, "cells": cells, "verdicts": v, **extra,
         "exploratory_e2_structure": exploratory})  # fmt: skip


if __name__ == "__main__":
    main()
