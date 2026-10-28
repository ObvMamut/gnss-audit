"""S33: the front-end prediction for E2 run 3, uncensored to 85 dB-Hz (P2c task 4; R-7 of the plan of 24 Oct 2026).

Written before run 3, and committed before its started.json. It decides only the R9 re-scope's conditional front-end
range, through s34, and never G3.

Question. Mode B, the registered yardstick, reads low at weak jamming for a reconstruction reason (s28's P3, s29's H-B).
s29 modelled a true 3-bit front end, but it was adopted after run 2's data and could hardly fail. What mode B / mode A
ratios R_B will run 3 measure, per segment, family and level, if s29's construction, revised as R-7 says, is right? And
how far from mode A does the true front end's error excess move (R_TA = excess_truth / excess_A) up to 85 dB-Hz? After
run 3, s34 accepts or rejects this model per segment by a chi^2 test of run 3's R against R_B (noise and swept CW,
50-70 dB-Hz), and if both segments are accepted it reads the conditional range from R_TA.

Method. s29's construction (s28's E2 structure: the recording r0 = requantise(x0) holds no ghost; the ghost g is injected
after the jammer into every twin, in mode B after the reconstruction; each pair against its own clean twin, per 1 ms
window z = sum y conj(s) on the ghost's exact waveform, e = Im(z / z_ref) / g, the paired excess and s28's block
bootstrap (1 s blocks, 400 replicates, seed 7, the same blocks for every series)), with R-7's six changes:
  1. Independent draws per segment: every SeedSequence is [33, segment, category, key], segment A = 0 and B = 1;
     categories 0 the ghost (its data bits and carrier phase), 1 the analog noise (key = chunk), 2-4 the jammer of each
     family (key = level), 800 mode B's dither. All tuples have four words, since SeedSequence pads entropy with zeros to
     four words. The bootstrap keeps E2's seed 7 on both segments: it is the registered statistic, not a simulated draw.
  2. Analog noise shaped to each segment's measured PSD. Welch (2^16, Hann, 50 %, scipy's defaults: constant detrend,
     two-sided) on the first 10 s of the registered interval, streamed, and the SHA-256 of the bytes read recorded. The
     PSD is averaged into the FIR's 4 097 frequency bins (fs / 4 097 = 7.49 kHz). A complex FIR of 4 097 taps is then
     built by frequency sampling of sqrt(PSD), centred and Hann-windowed, and scaled to run 2's analog_sd per component
     (A 1.5042, B 1.5272).
  3. Levels set on run 2's measured N0 (A 1.7735e-7, B 1.7882e-7): the jammer's J/N0 and the ghost's C/N0 (45 dB-Hz).
     The synthetic recording's own estimate_n0 (first 2^24 samples, as s28's recording_floor) is reported beside them as
     n0_synthetic, and its C/A-weighted line excess beside run 2's. Neither sets anything.
  4. The true front end's AGC is mode B's recursive one at tau in {1, 10, 100} ms (ruling T4b). It calls
     gnssaudit.inject.pipeline._agc itself, so there is no re-implementation to check:
     - first-order smoothing of 256-sample sub-block powers on the absolute grid, alpha = 1 - exp(-256 / (tau fs)),
       pre-charged, with the state carried through chunks;
     - it holds the clean analog stream's power (reference x0 + g, input x0 + j + g, gain sqrt(P_ref / P_in)), and its
       output is requantised to FGI's alphabet;
     - the clean true twin is requantise(x0 + g), the gain being exactly 1 at j = 0;
     - s29's stationary AGC, G = sqrt(P_x / P_{x+j}), is reported beside it (ratio_TA_stationary).
  5. Levels 50, 55, ..., 85 dB-Hz for noise, swept CW and the chirp (E2-v2's families), on both segments: 48 cells.
  6. Mode B: the production mode B chain at tau = 10 ms, as run 3 runs it (ruling T4d). It reconstructs the synthetic
     recording with reconstruct_analog at the recording's own estimate_analog_sd and the segment's dither seed, then
     applies the AGC and requantises. It is checked bit for bit against inject() in mode B.
  Length as s29: checkpoints at 20, 40 and 60 s. A cell stops at the first checkpoint where the SEs of R_B and of R_TA
  at all three tau are all <= 0.02, otherwise at 60 s (controller ruling of 25 Oct 2026). s33's SE enters s34's z
  denominator, so stopping on the largest SE avoids an easier acceptance. The stationary R_TA is reported and stops
  nothing.

Self-checks, run first (any failure stops the study):
  - the study runs from a commit. At the start it hashes every first-party module loaded (src/gnssaudit and review),
    together with the numpy and scipy versions, and git must show none of those files changed. Each worker checks the
    hash, and that it loads no first-party module outside the set, before and after its cell. The save checks the hash
    and git again. The hash also keys the cell cache (--cells-dir), from which a resumed run skips finished cells;
  - no seed is shared between A and B, comparing SeedSequence states, so zero padding cannot hide a match;
  - the shaped noise's Welch PSD, over the first 10 s of the stream the cells use, matches its target (the measured PSD
    scaled to 2 analog_sd^2) within 0.5 dB over |f| <= 8 MHz, both averaged into cells of 4 FIR bins (29.96 kHz).
    Ruling T4e, correcting P5: a frequency-sampled FIR with a Hann window convolves its target with the window's
    transform, whose main lobe is 4 bins wide (+-2 bins, 4 / N). The FIR's resolution is therefore 4 bins, a width that
    follows from R-7's design and not from the measured PSD. The 1-bin mismatch is reported beside it;
  - with the jammer at zero, on each segment and family:
    - on the first chunk every jammed twin (mode A, the true front end at each tau and stationary, mode B) is its clean
      twin sample for sample;
    - over three chunks of windows every pair's correlations are equal and its excess is exactly 0;
  - mode B's two twins equal inject()'s in mode B, bit for bit (first chunk, noise at 70 dB-Hz).

Reading, stated before this study ran:
  - R_TA lies near 1 at weak jamming (within 0.10 at 50-60 dB-Hz at every tau) and departs from it as J/N0 grows toward
    85 dB-Hz, where the AGC and the 3-bit quantiser act on a stream that the jammer dominates. The direction and the
    level of departure are not predicted: s29 stopped at 70.
  - R_B is low at weak jamming (below 1 at 50-55 dB-Hz), as s28 and s29 found for mode B's reconstruction.
  - s33 reads from run 2 only analog_sd, n0_per_hz and line_excess_db. While reading run 2's summary layout, the
    implementer saw two front-end ratios (noise, 55 dB-Hz); no design choice depends on them.
  - The two bullets above were written before any cell ran. Before the code's first commit (47dbe06) the implementer
    then saw the following, and changed neither those bullets nor any design choice after them:
    - two timing probes of one cell (segment A, noise, 50 dB-Hz, 20 s): R_B 0.491 +- 0.067; R_TA 1.001, 1.000 and 1.000
      at tau = 1, 10 and 100 ms;
    - four diagnostic cells that checked the machinery against s29 (segment A, noise, 60 dB-Hz, 20 s). R_B: D1, flat
      noise at s29's N0, 0.950 +- 0.009; D4, flat at run 2's N0, 0.927 +- 0.014; D3, shaped at the synthetic N0,
      0.898 +- 0.012; D2, shaped at run 2's N0 (s33's own cell), 0.887 +- 0.012. R_TA was 0.991-0.995 in all four.
    The same note is in the JSON (meta.disclosure).

Result (review/results/s33_front_end_prediction.json): filled after the run.

Model limits (also in the JSON):
  - Gaussian analog noise from a 4 097-tap Hann-windowed FIR. Lines narrower than about 30 kHz are smeared: on segment
    B, those at -5.42, -5.59 and +4.58 MHz. s34's acceptance judges whether that matters.
  - The target is the 3-bit recording's PSD, which already holds quantisation noise and the real satellites, so the
    requantised synthetic recording is flatter than the real one (line_excess_db_synthetic against run 2's).
  - The jammer and the ghost are added to the analog digitally, as E2 adds them to the recording; they do not pass the
    front end's filter.
  - One ghost per segment (s28's PRN 32, 1 510 Hz, 45 dB-Hz), correlated at the native rate on its exact waveform. Run 3
    has two ghosts per segment, chosen by its survey, tracked by GNSS-SDR after a 4x decimation.
  - The true front end's AGC is mode B's recursion holding the clean analog power (T4b), not a measured AGC law.

Usage: uv run python -m review.s33_front_end_prediction --workers N [--probe] [--cells-dir DIR]. It reads run 2's
summary and the first 10 s of each registered interval read-only. --probe times one 20 s cell and saves nothing.
Finished cells are kept in DIR (default /srv/gnss/results/dev/s33-cells), so a re-run resumes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from numpy.typing import NDArray
from scipy import signal
from scipy.signal import windows

from gnssaudit.experiments.e2_frontend import paired_excess
from gnssaudit.inject import (
    Disturbance,
    FrontEndSpec,
    GhostSpec,
    IQSource,
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
from gnssaudit.inject.pipeline import _agc, _front_end, _state  # mode B's AGC, as inject() runs it
from gnssaudit.inject.source import read_iq
from gnssaudit.registry import dataset_file, load_intervals
from review._common import _jsonable, save
from review.s28_twin_pair_quantisation import (
    BAND_HZ,
    BOOT_BLOCK,
    BOOT_N,
    BOOT_SEED,
    CHUNK,
    FAMILIES,
    FGI,
    FS,
    GHOST,
    TAU_S,
    WIN,
    Cell,
    SelfCheckError,
    e2s_stats,
    errors,
    make_cell,
    window_sums,
    write_ibyte,
)

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

ROOT = 33
SEGMENTS = ("A", "B")
SEG_INDEX = {"A": 0, "B": 1}
INTERVALS = {"A": "fgi-jt2023-JT23_4.1.5_L1_E1-clean", "B": "fgi-jt2023-JT23_16.1.1_L1_E1-clean"}
GHOST_CAT, NOISE_CAT, JAMMER_CAT, DITHER_CAT = 0, 1, 2, 800  # seed categories; the jammer's is 2 + family index
RUN2_ID = "6214cf52489f"
RUN2_DIR = Path("/srv/gnss/results/E2") / RUN2_ID  # read only
RUN2_SHA256 = "4419fd70188279102a2ab862fda3ef17acdbb81018b8793918d165b11e962c99"
PSD_SECONDS = 10.0
PSD_PIECE = 1 << 22  # samples read at a time
NPERSEG = 1 << 16
SHAPE_TAPS = 4097
CHECK_BINS = 4  # the Hann window's main lobe in FIR bins (ruling T4e)
CHECK_HALF_BAND_HZ = 8e6  # the central 80 % of the +-10 MHz band
PSD_TOL_DB = 0.5
TAUS = (0.001, 0.01, 0.1)
LEVELS = (50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0)
DECIDING = ("noise", "swept_cw")
ACCEPTANCE_MAX_DB_HZ = 70.0  # s34's chi^2 runs over noise and swept CW at 50-70 dB-Hz
CHECKPOINTS_S = (20.0, 40.0, 60.0)
CHECKPOINT_WINDOWS = tuple(round(s * 1000) for s in CHECKPOINTS_S)
SE_STOP = 0.02
CELLS_DIR = Path("/srv/gnss/results/dev/s33-cells")  # finished cells, for a resumed run (not results)
WORKERS = 9  # T4a: min(10, floor((available GiB - 4) / peak GiB per cell)), from the probe; --workers overrides
CHAINS = ("A_clean", "A", "truth_clean", "truth_stationary", *(f"truth_{t}" for t in TAUS), "B_clean", "B")
PAIRS = {"A": ("A_clean", "A"), **{f"TA_{t}": ("truth_clean", f"truth_{t}") for t in TAUS},
         "TA_stationary": ("truth_clean", "truth_stationary"), "B": ("B_clean", "B")}  # fmt: skip


def require(ok: bool, what: str) -> None:
    if not ok:
        raise SelfCheckError(f"s33 self-check failed: {what}")


def tau_key(tau: float) -> str:
    return str(tau)


def level_key(level: float) -> str:
    return f"{level:.1f}"


# ------------------------------------------------------------------------------------------------ seeds
def seed_tuple(segment: str, category: int, key: int) -> tuple[int, int, int, int]:
    return (ROOT, SEG_INDEX[segment], category, key)


def _level_seed_key(level: float) -> int:
    return 0 if math.isinf(level) else int(level)


def ghost_seed(segment: str) -> int:
    return disturbance_seed(*seed_tuple(segment, GHOST_CAT, 0))


def jammer_seed(segment: str, family: str, level: float) -> int:
    return disturbance_seed(*seed_tuple(segment, JAMMER_CAT + list(FAMILIES).index(family), _level_seed_key(level)))


def dither_seed(segment: str) -> int:
    return disturbance_seed(*seed_tuple(segment, DITHER_CAT, 0))


def noise_rng(segment: str, key: int) -> np.random.Generator:
    return np.random.default_rng(list(seed_tuple(segment, NOISE_CAT, key)))


def ghost(segment: str) -> Disturbance:
    """s28's ghost (PRN 32, 1 510 Hz, chip 17.3, 45 dB-Hz) with the segment's seed."""
    return replace(GHOST, seed=ghost_seed(segment))


def seed_tuples(segment: str, n_noise_keys: int) -> list[tuple[int, int, int, int]]:
    """Every seed tuple the segment draws from: the ghost, the dither, the noise's first n_noise_keys chunk keys, and
    each family's jammer at every level and at zero (the self-check's)."""
    out = [seed_tuple(segment, GHOST_CAT, 0), seed_tuple(segment, DITHER_CAT, 0)]
    out += [seed_tuple(segment, NOISE_CAT, k) for k in range(n_noise_keys)]
    out += [seed_tuple(segment, JAMMER_CAT + fi, k) for fi in range(len(FAMILIES))
            for k in (0, *(int(lv) for lv in LEVELS))]  # fmt: skip
    return out


def seed_independence(a: Iterable[Sequence[int]], b: Iterable[Sequence[int]]) -> dict[str, Any]:
    """No seed shared between the segments, compared as SeedSequence states (entropy padded to four words, so a
    padded duplicate is caught too)."""

    def states(ts: Iterable[Sequence[int]]) -> set[tuple[int, ...]]:
        return {tuple(np.random.SeedSequence(list(t)).generate_state(4, dtype=np.uint32).tolist()) for t in ts}

    ta, tb = [tuple(t) for t in a], [tuple(t) for t in b]
    shared = len(states(ta) & states(tb))  # an equal tuple has an equal state
    require(shared == 0, f"{shared} seed(s) shared between segments A and B")
    return {"tuples": {"A": len(ta), "B": len(tb)}, "shared": 0,
            "compared": "SeedSequence states (entropy padded to four words)"}  # fmt: skip


# ------------------------------------------------------------------------------------------------ PSD and the FIR
class Welch:
    """scipy.signal.welch (Hann, 50 % overlap, constant detrend, two-sided) over a stream fed in pieces: the same
    segments as one call on the whole stream, averaged; result() is fftshifted."""

    def __init__(self, nperseg: int = NPERSEG, fs: float = FS) -> None:
        self.nperseg, self.hop, self.fs = nperseg, nperseg // 2, fs
        self._buf = np.zeros(0, dtype=np.complex128)
        self._sum: FloatArray | None = None
        self._f: FloatArray | None = None
        self.count = 0

    def add(self, x: NDArray[Any]) -> None:
        w = np.concatenate([self._buf, np.asarray(x, dtype=np.complex128)])
        m = (w.size - self.nperseg) // self.hop + 1 if w.size >= self.nperseg else 0
        if m:
            f, p = signal.welch(w[: (m - 1) * self.hop + self.nperseg], fs=self.fs, window="hann",
                                nperseg=self.nperseg, return_onesided=False)  # fmt: skip
            self._sum = p * m if self._sum is None else self._sum + p * m
            self._f, self.count = f, self.count + m
            w = w[m * self.hop :]
        self._buf = w

    def result(self) -> tuple[FloatArray, FloatArray, int]:
        if self._sum is None or self._f is None:
            raise ValueError("no complete Welch segment yet")
        return np.fft.fftshift(self._f), np.fft.fftshift(self._sum / self.count), self.count


def fir_bin_average(f: FloatArray, p: FloatArray, width: int = 1) -> tuple[FloatArray, FloatArray]:
    """(mean frequency, mean PSD) of the Welch bins in each cell of `width` consecutive FIR bins (fs / SHAPE_TAPS each,
    centred on k fs / SHAPE_TAPS, k = -2048..2048), from the lowest bin up; the last cell may be narrower."""
    half = SHAPE_TAPS // 2
    k = np.clip(np.rint(f * SHAPE_TAPS / FS).astype(np.int64), -half, half) + half
    c = k // width
    n = (SHAPE_TAPS - 1) // width + 1
    cnt = np.bincount(c, minlength=n)
    return np.bincount(c, weights=f, minlength=n) / cnt, np.bincount(c, weights=p, minlength=n) / cnt


def shaping_fir(f: FloatArray, p: FloatArray, sigma: float) -> ComplexArray:
    """R-7's FIR: frequency sampling of sqrt(PSD) on the FIR's own bins, centred, Hann-windowed, scaled so that unit
    white noise per component comes out with SD sigma per component."""
    _, pb = fir_bin_average(f, p, 1)
    hd = np.fft.ifft(np.fft.ifftshift(np.sqrt(pb)))
    h = np.roll(hd, SHAPE_TAPS // 2) * windows.hann(SHAPE_TAPS)
    return np.asarray(h * (sigma / math.sqrt(float(np.sum(np.abs(h) ** 2)))), dtype=np.complex128)


def psd_match(
    f: FloatArray,
    measured: FloatArray,
    shaped: FloatArray,
    sigma: float,
    width: int = CHECK_BINS,
    band: float = CHECK_HALF_BAND_HZ,
    tol: float = PSD_TOL_DB,
) -> dict[str, Any]:
    """The shaped noise's PSD against its target, the measured PSD scaled to 2 sigma^2 in all, both averaged into cells
    of `width` FIR bins (same Welch grid f), over |f| <= band: the largest |difference| in dB and where."""
    target = measured * (2 * sigma**2 / (float(np.mean(measured)) * FS))
    fc, tc = fir_bin_average(f, target, width)
    _, sc = fir_bin_average(f, shaped, width)
    d = 10 * np.log10(sc / tc)
    inside = np.abs(fc) <= band
    fi, di = fc[inside], d[inside]
    i = int(np.argmax(np.abs(di)))
    order = np.argsort(-np.abs(di))
    over = [{"f_hz": float(fi[j]), "delta_db": float(di[j])} for j in order if abs(di[j]) > tol]
    return {"max_abs_db": float(abs(di[i])), "at_hz": float(fi[i]), "rms_db": float(np.sqrt(np.mean(di**2))),
            "cells": int(inside.sum()), "width_bins": width, "cell_hz": width * FS / SHAPE_TAPS, "band_hz": band,
            "tol_db": tol, "passes": bool(abs(di[i]) <= tol), "over_tol": over}  # fmt: skip


def shaped_noise(n_total: int, h: ComplexArray, segment: str, chunk: int = CHUNK) -> Iterator[tuple[int, ComplexArray]]:
    """(lo, samples) of the segment's analog noise in `chunk` pieces: unit white noise per component from the
    segment's seeds through h, with the filter's history carried so that the stream is one stationary process."""
    taps = h.size

    def white(key: int) -> ComplexArray:
        rng = noise_rng(segment, key)
        return rng.standard_normal(chunk) + 1j * rng.standard_normal(chunk)

    hist = white(0)[chunk - (taps - 1) :]
    for k, lo in enumerate(range(0, n_total, chunk)):
        w = white(k + 1)
        x = signal.oaconvolve(np.concatenate([hist, w]), h, mode="valid")
        hist = w[chunk - (taps - 1) :]
        yield lo, np.asarray(x[: min(chunk, n_total - lo)], dtype=np.complex128)


def noise_keys(n_samples: int) -> int:
    """How many noise chunk keys a stream of n_samples draws (the history's key 0 included)."""
    return -(-n_samples // CHUNK) + 1


# ------------------------------------------------------------------------------------------------ the chains
@dataclass
class TrueCell:
    """s28's cell with the segment's jammer and dither seeds (the jammer, s29's stationary AGC gain, mode B's front end
    and its state), mode B's clean twin's AGC state, and the true front end's recursive AGC per tau (spec, state)."""

    cell: Cell
    st_clean: Any
    agc: dict[float, tuple[FrontEndSpec, Any]]


def make_true_cell(segment: str, family: str, level: float, n0: float, sd_b: float, sigma: float) -> TrueCell:
    base = make_cell(family, level, n0, sd_b, 0, sigma)  # its stationary gain uses run 2's N0 and SD here
    fe = replace(base.fe, dither_seed=dither_seed(segment))
    cell = replace(base, jammer=replace(base.jammer, seed=jammer_seed(segment, family, level)), fe=fe, st=_state(fe))
    return TrueCell(cell, _state(fe), {t: (replace(fe, agc_tau_s=t), _state(fe)) for t in TAUS})


def chains(tc: TrueCell, lo: int, x0: ComplexArray, g: ComplexArray, n0: float) -> Iterator[tuple[str, ComplexArray]]:
    """Every twin of the cell on samples [lo, lo + x0.size), in CHAINS order (consume them all: the AGC states advance
    as they are drawn). s28's E2 structure with the true front end's recursive AGC beside its stationary one."""
    c = tc.cell
    assert c.fe.dither_seed is not None and c.fe.analog_sd is not None
    r0 = requantise(x0, FGI)
    rec = reconstruct_analog(r0, FGI, c.fe.analog_sd, c.fe.dither_seed, lo)
    j = render(c.jammer, lo, x0.size, FS, n0)
    yield "A_clean", r0 + g
    yield "A", r0 + j + g
    clean, jammed = x0 + g, x0 + j + g
    yield "truth_clean", requantise(clean, FGI)
    yield "truth_stationary", requantise(c.agc_gain * jammed, FGI)
    for t in TAUS:
        fe, st = tc.agc[t]
        yield f"truth_{t}", requantise(_agc(jammed, clean, lo, fe, FS, st), FGI)
    yield "B_clean", _front_end(rec + g, lo, 0, c.fe, FS, tc.st_clean, rec)
    yield "B", _front_end(rec + j + g, lo, 0, c.fe, FS, c.st, rec)


def true_windows(
    tc: TrueCell, h: ComplexArray, segment: str, n0: float, n: int
) -> Iterator[tuple[int, dict[str, ComplexArray]]]:
    """The cell over the first n windows of the segment's stream: after each chunk, (windows complete so far, every
    twin's correlation per window), updated in place."""
    gs, amp = GhostSpec.of(ghost(segment)), gain(GHOST.level_db_hz, n0)
    z = {k: np.zeros(n + 1, dtype=np.complex128) for k in CHAINS}
    for lo, x0 in shaped_noise(n * WIN, h, segment):
        s = ghost_modulation(gs, np.arange(lo, lo + x0.size, dtype=np.int64), FS)
        sc = np.conj(s)
        for k, y in chains(tc, lo, x0, amp * s, n0):
            idx, v = window_sums(y, sc, lo)
            z[k][idx] += v
        yield (lo + x0.size) // WIN, z


# ------------------------------------------------------------------------------------------------ statistics
TRUTHS = {**{tau_key(t): f"truth_{t}" for t in TAUS}, "stationary": "truth_stationary"}


def cell_stats(z: Mapping[str, ComplexArray], z_ref: float, n: int) -> dict[str, Any]:
    """s28's e2s_stats once per true front end (each tau, and stationary), on the first n windows; mode B's pair is
    the same in each."""
    common = {k: z[k] for k in ("A_clean", "A", "truth_clean", "B_clean", "B")}
    per = {k: e2s_stats({**common, "truth": z[chain]}, z_ref, n) for k, chain in TRUTHS.items()}
    taus = [tau_key(t) for t in TAUS]
    b = per[taus[0]]

    def by_tau(key: str) -> dict[str, float]:
        return {k: per[k][key] for k in taus}

    keys = ["A", "B", *(f"TA_{k}" for k in TRUTHS)]
    src = {"A": (b, "A"), "B": (b, "B"), **{f"TA_{k}": (per[k], "truth") for k in TRUTHS}}
    return {
        "n_windows": n, "length_s": n / 1000,
        "ratio_B": b["ratio_B"], "ratio_B_se": b["ratio_B_se"], "slope_B": b["slope_B"],
        "ratio_TA": by_tau("ratio_truth"), "ratio_TA_se": by_tau("ratio_truth_se"), "slope_TA": by_tau("slope_truth"),
        "b_over_truth": by_tau("b_over_truth"), "b_over_truth_se": by_tau("b_over_truth_se"),
        "ratio_TA_stationary": per["stationary"]["ratio_truth"],
        "ratio_TA_stationary_se": per["stationary"]["ratio_truth_se"],
        "slope_TA_stationary": per["stationary"]["slope_truth"],
        "b_over_truth_stationary": per["stationary"]["b_over_truth"],
        "b_over_truth_stationary_se": per["stationary"]["b_over_truth_se"],
        **{f: {k: src[k][0][f"{f}_{src[k][1]}"] for k in keys} for f in ("excess", "var_dphi", "cov2")},
        "gains": {**{k: v for k, v in b["gains"].items() if k != "truth"},
                  **{TRUTHS[k]: per[k]["gains"]["truth"] for k in TRUTHS}},
    }  # fmt: skip


def stop_ses(st: Mapping[str, Any]) -> dict[str, float]:
    """The SEs a cell's length answers to: R_B's and R_TA's at each tau (what s34 reads)."""
    return {"ratio_B": float(st["ratio_B_se"]), **{f"ratio_TA[{k}]": float(v) for k, v in st["ratio_TA_se"].items()}}


def adaptive[Z](
    stream: Iterable[tuple[int, Z]],
    stats: Callable[[Z, int], dict[str, Any]],
    checkpoints: Sequence[int],
    se_stop: float = SE_STOP,
) -> dict[str, Any]:
    """At each checkpoint n (windows) in turn, the statistics on the first n windows; the cell stops at the first
    where every SE of stop_ses is at most se_stop (a non-finite one never is), otherwise at the last. The stream is
    read no further than the chunk that completes the stopping checkpoint."""
    cps = sorted(checkpoints)
    trail: list[dict[str, Any]] = []
    for done, z in stream:
        while cps and done >= cps[0]:
            n = cps.pop(0)
            st = stats(z, n)
            ses = stop_ses(st)
            worst = max(ses.values()) if all(math.isfinite(v) for v in ses.values()) else math.inf
            trail.append({"n_windows": n, "ses": ses, "max_se": worst})
            if worst <= se_stop or not cps:
                return {**st, "se_stop_reached": bool(worst <= se_stop), "checkpoints": trail}
    raise ValueError(f"the stream ended before the last checkpoint ({max(checkpoints)} windows)")


def workers_for(available_gib: float, peak_gib: float) -> int:
    """Ruling T4a: min(10, floor((available - 4) / peak per cell)); refused below one worker."""
    w = min(10, math.floor((available_gib - 4.0) / peak_gib))
    if w < 1:
        raise ValueError(f"not enough memory: {available_gib:.1f} GiB available, {peak_gib:.1f} GiB per cell")
    return w


REPO = Path(__file__).resolve().parent.parent


def first_party_paths() -> list[str]:
    """Every loaded module of this repository (src/gnssaudit and review), as sorted absolute paths: what the cells run."""
    roots = (str(REPO / "src" / "gnssaudit") + os.sep, str(REPO / "review") + os.sep)
    found = {str(Path(f).resolve()) for m in list(sys.modules.values()) if (f := getattr(m, "__file__", None))}
    return sorted(p for p in found if p.startswith(roots))


def code_hash(paths: Iterable[str]) -> str:
    """SHA-256 over each file (its path and bytes, in sorted order) and the numpy and scipy versions."""
    d = hashlib.sha256()
    for p in sorted(paths):
        d.update(p.encode() + b"\0" + Path(p).read_bytes() + b"\0")
    d.update(f"numpy {np.__version__} scipy {scipy.__version__}".encode())
    return d.hexdigest()


def require_code(paths: Iterable[str], expected: str) -> None:
    require(code_hash(paths) == expected, "the code (or numpy or scipy) changed since the study started")


def require_loaded_within(paths: Iterable[str]) -> None:
    extra = sorted(set(first_party_paths()) - set(paths))
    require(not extra, f"loaded first-party modules not in the hashed set: {extra}")


def git_state(paths: Iterable[str], root: Path = REPO) -> dict[str, Any]:
    """HEAD, and git's porcelain status lines for the given paths (empty when they are as committed)."""

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout

    return {"head": git("rev-parse", "HEAD").strip(),
            "dirty": [ln for ln in git("status", "--porcelain", "--", *paths).splitlines() if ln]}  # fmt: skip


def mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("MemAvailable not in /proc/meminfo")


# ------------------------------------------------------------------------------------------------ inputs (read only)
def read_run2(run_dir: Path = RUN2_DIR, run_id: str = RUN2_ID, sha256: str = RUN2_SHA256) -> dict[str, Any]:
    """What s33 takes from E2 run 2, and nothing else: each segment's analog_sd, n0_per_hz and line_excess_db. Refuses
    another run or another summary."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("run_id") != run_id:
        raise ValueError(f"{run_dir} holds run {manifest.get('run_id')!r}, not E2 run 2 ({run_id})")
    raw = (run_dir / "summary.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != sha256:
        raise ValueError(f"{run_dir / 'summary.json'} has SHA-256 {digest}, not run 2's pinned {sha256}")
    segs = json.loads(raw)["segments"]
    return {"run_id": run_id, "summary_path": str(run_dir / "summary.json"), "summary_sha256": digest,
            **{k: {s: float(segs[s][k]) for s in SEGMENTS} for k in ("analog_sd", "n0_per_hz", "line_excess_db")}}  # fmt: skip


def interval_source(segment: str) -> IQSource:
    iv = load_intervals()[INTERVALS[segment]]
    return IQSource(path=dataset_file(iv["dataset_id"], iv["file"]), item_type="ibyte", fs_hz=FS,
                    sample_start=int(iv["sample_start"]), sample_end=int(iv["sample_end"]))  # fmt: skip


def measured_psd(segment: str) -> tuple[FloatArray, FloatArray, dict[str, Any]]:
    """Welch over the first PSD_SECONDS of the segment's registered interval, streamed, and the SHA-256 of the bytes
    read (each byte once, in order)."""
    src = interval_source(segment)
    n = round(PSD_SECONDS * FS)
    acc, digest = Welch(), hashlib.sha256()
    raw = np.memmap(src.path, dtype=np.int8, mode="r")
    for a in range(src.sample_start, src.sample_start + n, PSD_PIECE):
        m = min(PSD_PIECE, src.sample_start + n - a)
        acc.add(read_iq(src, a, m))
        digest.update(raw[2 * a : 2 * (a + m)].tobytes())
    f, p, k = acc.result()
    return f, p, {"interval_id": INTERVALS[segment], "file": str(src.path), "item_type": src.item_type,
                  "sample_start": src.sample_start, "samples": n, "bytes": 2 * n, "sha256": digest.hexdigest(),
                  "welch_segments": k}  # fmt: skip


def shaped_psd(h: ComplexArray, segment: str) -> tuple[FloatArray, FloatArray, int]:
    acc = Welch()
    for _, x in shaped_noise(round(PSD_SECONDS * FS), h, segment):
        acc.add(x)
    return acc.result()


def synthetic_floor(h: ComplexArray, segment: str, tmp: Path) -> dict[str, float]:
    """N0, the C/A-weighted line excess and the analog SD as E2's survey estimates them (estimate_n0,
    estimate_analog_sd) on the first 2^24 samples of the synthetic recording requantise(x0), as s28's recording_floor."""
    r = np.concatenate([requantise(x, FGI) for _, x in shaped_noise(2 * CHUNK, h, segment)])
    src = write_ibyte(r, tmp / f"floor-{segment}.bin")
    floor = estimate_n0(src, frontend_bw_hz=BAND_HZ)
    sd = estimate_analog_sd(src, FGI)
    src.path.unlink()
    return {"n0_per_hz": floor.n0_per_hz, "n0_spread_db": floor.spread_db, "ca_weighted_per_hz": floor.ca_weighted_per_hz,
            "line_excess_db": floor.line_excess_db, "analog_sd": sd,
            "recording_power": float(np.mean(np.abs(r) ** 2))}  # fmt: skip


def segment_entry(
    *, n0_run2: float, line_excess_run2: float, floor: Mapping[str, float], match: Mapping[str, Any],
    match_1bin: Mapping[str, Any],
) -> dict[str, Any]:  # fmt: skip
    """The segment's keys s34 and the report read."""
    return {"n0_run2": n0_run2, "n0_synthetic": floor["n0_per_hz"], "psd_match_db": match["max_abs_db"],
            "psd_match_db_1bin": match_1bin["max_abs_db"], "line_excess_db_synthetic": floor["line_excess_db"],
            "line_excess_db_run2": line_excess_run2, "analog_sd_synthetic": floor["analog_sd"]}  # fmt: skip


# ------------------------------------------------------------------------------------------------ known answers
def zero_jammer_check(segment: str, h: ComplexArray, sigma: float, n0: float, sd_b: float) -> dict[str, Any]:
    """The jammer at zero: on the first chunk every jammed twin is its clean twin sample for sample (s29's stationary
    gain exactly 1); over three chunks of windows every pair's correlations equal and its excess is exactly 0."""
    x0 = next(shaped_noise(CHUNK, h, segment))[1]
    amp = gain(GHOST.level_db_hz, n0)
    g = amp * ghost_modulation(GhostSpec.of(ghost(segment)), np.arange(x0.size, dtype=np.int64), FS)
    n, z_ref = 3 * CHUNK // WIN, amp * WIN
    out: dict[str, Any] = {}
    for fam in FAMILIES:
        tc = make_true_cell(segment, fam, -math.inf, n0, sd_b, sigma)
        require(tc.cell.agc_gain == 1.0, f"{segment} {fam}: the stationary AGC gain at j = 0 is {tc.cell.agc_gain!r}")
        ys = dict(chains(tc, 0, x0, g, n0))
        for p, (cl, jam) in PAIRS.items():
            require(
                bool(np.array_equal(ys[jam], ys[cl])), f"{segment} {fam}: at j = 0 the {p} twin is not its clean twin"
            )
        _, z = deque(true_windows(make_true_cell(segment, fam, -math.inf, n0, sd_b, sigma), h, segment, n0, n),
                     maxlen=1)[0]  # fmt: skip
        excess = {}
        for p, (cl, jam) in PAIRS.items():
            require(
                bool(np.array_equal(z[jam][:n], z[cl][:n])), f"{segment} {fam}: at j = 0 the {p} correlations differ"
            )
            e_c, e_j = errors(z[cl][:n], z_ref)[0], errors(z[jam][:n], z_ref)[0]
            excess[p] = paired_excess(e_j - e_c, e_c).excess
            require(excess[p] == 0.0, f"{segment} {fam}: the {p} pair's excess at j = 0 is {excess[p]!r}, not 0")
        out[fam] = {"samples_equal_first_chunk": int(x0.size), "n_windows": n, "z_equal_clean": True,
                    **{f"excess_{p}": x for p, x in excess.items()}}  # fmt: skip
    return out


def inject_check(segment: str, h: ComplexArray, sigma: float, n0: float, sd_b: float, tmp: Path) -> dict[str, Any]:
    """Mode B's two twins against inject() in mode B on the ghost-free synthetic recording, with the jammer and the
    ghost as disturbances (clean: clean_twin of them), first chunk, noise at 70 dB-Hz: bit for bit."""
    x0 = next(shaped_noise(CHUNK, h, segment))[1]
    src = write_ibyte(requantise(x0, FGI), tmp / f"rec0-{segment}.bin")
    tc = make_true_cell(segment, "noise", 70.0, n0, sd_b, sigma)
    gh = ghost(segment)
    s = ghost_modulation(GhostSpec.of(gh), np.arange(x0.size, dtype=np.int64), FS)
    ys = dict(chains(tc, 0, x0, gain(gh.level_db_hz, n0) * s, n0))
    for name, ds in (("B", [tc.cell.jammer, gh]), ("B_clean", clean_twin([tc.cell.jammer, gh]))):
        path = tmp / f"{name}-{segment}.bin"
        rep = inject(src, ds, tc.cell.fe, SinkSpec("file", path), n0_per_hz=n0)
        got = np.fromfile(path, dtype="<i2").astype(np.float64)
        want = np.empty(2 * x0.size)
        want[0::2], want[1::2] = ys[name].real, ys[name].imag
        require(
            rep.clip_count == 0 and bool(np.array_equal(got, want)), f"{segment}: mode B's {name} is not inject()'s"
        )
        path.unlink()
    src.path.unlink()
    changed = float(np.mean(ys["B"] != ys["B_clean"]))
    require(changed > 0.01, f"{segment}: the 70 dB-Hz jammer changed only {changed:.4f} of mode B's samples")
    return {"samples": int(x0.size), "family": "noise", "level_db_hz": 70.0, "twins": ["B", "B_clean"],
            "bitwise_equal": True, "share_of_samples_changed": changed}  # fmt: skip


# ------------------------------------------------------------------------------------------------ one cell
def role(family: str, level: float) -> str:
    if family not in DECIDING:
        return "reported"
    if level <= ACCEPTANCE_MAX_DB_HZ:
        return "s34: R_B enters the acceptance; R_TA the conditional range"
    return "s34: R_TA enters the conditional range"


def cell_job(
    segment: str, family: str, level: float, h: ComplexArray, sigma: float, n0: float, sd_b: float,
    checkpoints: Sequence[int], code_paths: Sequence[str], code: str, store: Path | None = None,
    fingerprint: str = "",
) -> dict[str, Any]:  # fmt: skip
    """One cell with the adaptive length (runs in a worker process). It refuses to run, or to return, if the code
    differs from the hash taken at the study's start or a first-party module outside that set is loaded. With `store`,
    it is also written there under `fingerprint` as soon as it is done, so that a resumed run skips it."""
    require_code(code_paths, code)
    require_loaded_within(code_paths)
    t0, c0 = time.time(), time.process_time()
    tc = make_true_cell(segment, family, level, n0, sd_b, sigma)
    z_ref = gain(GHOST.level_db_hz, n0) * WIN  # the noiseless correlation
    st = adaptive(true_windows(tc, h, segment, n0, max(checkpoints)), lambda z, n: cell_stats(z, z_ref, n), checkpoints)
    wall, cpu = time.time() - t0, time.process_time() - c0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2  # KiB on Linux
    ta = " ".join(f"{st['ratio_TA'][tau_key(t)]:.3f}" for t in TAUS)
    print(f"{segment} {family}-{level:g}: {st['length_s']:.0f} s, R_B {st['ratio_B']:.3f}+-{st['ratio_B_se']:.3f}, "
          f"R_TA {ta} ({wall:.0f} s, {rss:.2f} GiB)", flush=True)  # fmt: skip
    out = {"segment": segment, "family": family, "level_db_hz": level, "role": role(family, level),
           "agc_gain_stationary": tc.cell.agc_gain, **st, "wall_s": wall, "cpu_s": cpu, "peak_rss_gib": rss}  # fmt: skip
    require_code(code_paths, code)
    require_loaded_within(code_paths)
    if store is not None:
        store_cell(store, fingerprint, out)
    return out


def cell_fingerprint(
    h: ComplexArray, sigma: float, n0: float, sd_b: float, checkpoints: Sequence[int], code: str
) -> str:
    """What a stored cell was computed from: the code hash (every first-party module the cells run, numpy and scipy),
    the segment's FIR and its levels' inputs, and the checkpoints. A stored cell with another fingerprint is
    recomputed."""
    d = hashlib.sha256(code.encode())
    d.update(np.ascontiguousarray(h, dtype=np.complex128).tobytes())
    d.update(repr((float(sigma), float(n0), float(sd_b), tuple(int(c) for c in checkpoints))).encode())
    return d.hexdigest()


def cell_path(cells_dir: Path, segment: str, family: str, level: float) -> Path:
    return cells_dir / f"{segment}-{family}-{level_key(level)}.json"


def store_cell(path: Path, fingerprint: str, result: Mapping[str, Any]) -> None:
    """A finished cell, written atomically, so that a resumed run can skip it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as fh:
        fh.write(json.dumps({"fingerprint": fingerprint, "result": _jsonable(dict(result))}, indent=1) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)  # the rename itself
    finally:
        os.close(dfd)


def load_cell(path: Path, fingerprint: str) -> dict[str, Any] | None:
    """The stored cell at path if it was computed under this fingerprint, else None."""
    if not path.exists():
        return None
    stored = json.loads(path.read_text())
    return dict(stored["result"]) if stored["fingerprint"] == fingerprint else None


def nest_cells(results: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """cells[segment][family][level key as run 2's, "50.0"...] in SEGMENTS, FAMILIES and level order."""
    by = {(r["segment"], r["family"], float(r["level_db_hz"])): dict(r) for r in results}
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for s in SEGMENTS:
        for f in FAMILIES:
            lvs = sorted(lv for (s2, f2, lv) in by if (s2, f2) == (s, f))
            if lvs:
                out.setdefault(s, {})[f] = {level_key(lv): by[(s, f, lv)] for lv in lvs}
    return out


# ------------------------------------------------------------------------------------------------ the study
MODEL_LIMITS = [
    "Gaussian analog noise from a 4 097-tap Hann-windowed FIR: lines narrower than about 30 kHz (the window's 4-bin "
    "main lobe) are smeared by the model, on segment B those at -5.42, -5.59 and +4.58 MHz (segments.*.psd_match_1bin); "
    "s34's acceptance judges whether that matters",
    "the target is the 3-bit recording's PSD, which already holds quantisation noise and the real satellites, so the "
    "requantised synthetic recording is flatter than the real one (segments.*.line_excess_db_synthetic against "
    "line_excess_db_run2)",
    "the jammer and the ghost are added to the analog digitally, as E2 adds them to the recording; they do not pass the "
    "front end's filter",
    "one ghost per segment (s28's PRN 32, 1 510 Hz, 45 dB-Hz), correlated at the native rate on its exact waveform; run "
    "3 has two ghosts per segment, chosen by its survey, tracked by GNSS-SDR after a 4x decimation",
    "the true front end's AGC is mode B's recursion holding the clean analog power (ruling T4b), not a measured AGC law",
    "the bootstrap's seed 7 is shared by both segments: it is the registered statistic, not a simulated draw",
]


DISCLOSURE = (
    "The Reading in the docstring was written before any cell ran and was not changed after the implementer had seen, "
    "before the code's first commit (47dbe06): two timing probes of one cell (segment A, noise, 50 dB-Hz, 20 s: R_B "
    "0.491 +- 0.067; R_TA 1.001, 1.000, 1.000 at tau = 1, 10, 100 ms) and four diagnostic cells checking the machinery "
    "against s29 (segment A, noise, 60 dB-Hz, 20 s; R_B: D1 flat noise at s29's N0 0.950 +- 0.009, D4 flat at run 2's "
    "N0 0.927 +- 0.014, D3 shaped at the synthetic N0 0.898 +- 0.012, D2 shaped at run 2's N0, s33's own cell, 0.887 "
    "+- 0.012; R_TA 0.991-0.995 in all four). No design choice changed after them. While reading run 2's summary "
    "layout the implementer saw two of its front-end ratios (noise, 55 dB-Hz); no design choice depends on them."
)


def code_record(paths: Sequence[str], code: str, git0: Mapping[str, Any]) -> dict[str, Any]:
    """The code the cells ran, checked again at save time: the hash and git state must be those of the start."""
    require_code(paths, code)
    git1 = git_state(paths)
    require(not git1["dirty"], f"the hashed modules changed during the study: {git1['dirty']}")
    return {"hash": code, "numpy": np.__version__, "scipy": scipy.__version__,
            "git_head_at_start": git0["head"], "git_head_at_save": git1["head"], "clean": True,
            "paths": [str(Path(p).relative_to(REPO)) for p in paths]}  # fmt: skip


def setup_segment(segment: str, run2: Mapping[str, Any], tmp: Path, check_psd: bool) -> dict[str, Any]:
    """The segment's measured PSD, its FIR, the synthetic recording's floor and (check_psd) the PSD-match check."""
    sigma, n0 = run2["analog_sd"][segment], run2["n0_per_hz"][segment]
    t0 = time.time()
    f, p, psd = measured_psd(segment)
    h = shaping_fir(f, p, sigma)
    floor = synthetic_floor(h, segment, tmp)
    out: dict[str, Any] = {"h": h, "sigma": sigma, "n0": n0, "sd_b": floor["analog_sd"], "psd": psd, "floor": floor}
    if check_psd:
        fs_, ps_, k = shaped_psd(h, segment)
        require(bool(np.array_equal(fs_, f)), f"{segment}: the shaped and measured Welch grids differ")
        out["match"] = psd_match(f, p, ps_, sigma)
        out["match_1bin"] = psd_match(f, p, ps_, sigma, width=1)
        out["shaped_welch_segments"] = k
        m, m1 = out["match"], out["match_1bin"]
        print(f"segment {segment}: PSD match {m['max_abs_db']:.3f} dB at {m['at_hz'] / 1e3:.1f} kHz over 4-bin cells "
              f"(1 bin: {m1['max_abs_db']:.3f} dB at {m1['at_hz'] / 1e3:.1f} kHz)", flush=True)  # fmt: skip
        require(m["passes"], f"{segment}: the shaped PSD is {m['max_abs_db']:.3f} dB from its target at "
                f"{m['at_hz']:.0f} Hz (4-bin cells, |f| <= 8 MHz), above {PSD_TOL_DB} dB")  # fmt: skip
    print(f"segment {segment}: analog SD {sigma:.4f} (run 2), N0 {n0:.4e} (run 2), synthetic N0 {floor['n0_per_hz']:.4e},"
          f" line excess {floor['line_excess_db']:.2f} dB (run 2 {run2['line_excess_db'][segment]:.2f}), mode B's SD "
          f"{floor['analog_sd']:.4f} ({time.time() - t0:.0f} s)", flush=True)  # fmt: skip
    return out


def probe(run2: Mapping[str, Any], seg: Mapping[str, Any], available: float, paths: Sequence[str], code: str) -> None:
    """T4a's probe: one cell (A, noise, 50 dB-Hz) for one 20 s checkpoint in a fresh process; its wall, CPU and peak
    RSS, and the worst case of the full study (every cell at 60 s)."""
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        r = pool.submit(cell_job, "A", "noise", 50.0, seg["h"], seg["sigma"], seg["n0"], seg["sd_b"],
                        (CHECKPOINT_WINDOWS[0],), paths, code).result()  # fmt: skip
    per_s = r["cpu_s"] / r["length_s"]
    w = workers_for(available, r["peak_rss_gib"])
    worst = len(SEGMENTS) * len(FAMILIES) * len(LEVELS) * CHECKPOINTS_S[-1] * per_s
    print(f"probe: {r['length_s']:.0f} s of signal in {r['wall_s']:.0f} s wall, {r['cpu_s']:.0f} s CPU "
          f"({per_s:.1f} CPU s per signal s), peak RSS {r['peak_rss_gib']:.2f} GiB; available {available:.1f} GiB -> "
          f"T4a workers {w}; worst case (48 cells x 60 s) {worst / 3600:.1f} h CPU, {worst / w / 3600:.1f} h wall at "
          f"{w} workers", flush=True)  # fmt: skip


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="review.s33_front_end_prediction")
    ap.add_argument("--workers", type=int, default=WORKERS, help="cell processes (ruling T4a)")
    ap.add_argument("--probe", action="store_true", help="time one 20 s cell, save nothing")
    ap.add_argument("--cells-dir", type=Path, default=CELLS_DIR, help="finished cells, skipped on a resumed run")
    args = ap.parse_args([] if argv is None else list(argv))
    t0 = time.time()
    paths = first_party_paths()
    code = code_hash(paths)
    git0 = git_state(paths)
    if not args.probe:
        require(not git0["dirty"], f"the study runs from a commit: uncommitted changes to {git0['dirty']}")
    available = mem_available_gib()
    run2 = read_run2()
    segs: dict[str, dict[str, Any]] = {}
    tmp = Path(tempfile.mkdtemp(prefix="s33-", dir="/dev/shm"))
    try:
        if args.probe:
            probe(run2, setup_segment("A", run2, tmp, check_psd=False), available, paths, code)
            return
        n_keys = noise_keys(max(CHECKPOINT_WINDOWS) * WIN)
        seeds = seed_independence(seed_tuples("A", n_keys), seed_tuples("B", n_keys))
        print(f"self-check: no seed shared between A and B ({seeds['tuples']['A']} per segment)", flush=True)
        for s in SEGMENTS:
            segs[s] = setup_segment(s, run2, tmp, check_psd=True)
        for s, seg in segs.items():
            seg["zero_jammer"] = zero_jammer_check(s, seg["h"], seg["sigma"], seg["n0"], seg["sd_b"])
            seg["mode_b_vs_inject"] = inject_check(s, seg["h"], seg["sigma"], seg["n0"], seg["sd_b"], tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"self-checks passed ({time.time() - t0:.0f} s): no shared seed; the shaped PSD within {PSD_TOL_DB} dB of its "
          "target on both segments; at j = 0 every jammed twin is its clean twin and every excess is exactly 0; mode B "
          "is inject()'s bit for bit", flush=True)  # fmt: skip
    jobs = [(s, f, lv) for lv in LEVELS for f in FAMILIES for s in SEGMENTS]  # the weakest (longest) cells first
    fps = {
        s: cell_fingerprint(seg["h"], seg["sigma"], seg["n0"], seg["sd_b"], CHECKPOINT_WINDOWS, code)
        for s, seg in segs.items()
    }
    done = {job: r for job in jobs if (r := load_cell(cell_path(args.cells_dir, *job), fps[job[0]])) is not None}
    if done:
        print(f"resumed: {len(done)} finished cells from {args.cells_dir} under this code and these inputs", flush=True)
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, max_tasks_per_child=1) as pool:
        futures = {(s, f, lv): pool.submit(cell_job, s, f, lv, segs[s]["h"], segs[s]["sigma"], segs[s]["n0"],
                                           segs[s]["sd_b"], CHECKPOINT_WINDOWS, paths, code,
                                           cell_path(args.cells_dir, s, f, lv), fps[s])
                   for s, f, lv in jobs if (s, f, lv) not in done}  # fmt: skip
        results = [done[job] if job in done else futures[job].result() for job in jobs]
    cells = nest_cells(results)
    wall = time.time() - t0
    cpu = sum(r["cpu_s"] for r in results)
    print(f"{'seg':3s} {'family':9s} {'J/N0':>5s} {'len':>4s} {'R_B':>13s} "
          + " ".join(f"{'R_TA ' + tau_key(t):>13s}" for t in TAUS) + f" {'R_TA stat':>13s}")  # fmt: skip
    for s, by in cells.items():
        for f, lvs in by.items():
            for lv, c in lvs.items():
                ta = " ".join(f"{c['ratio_TA'][tau_key(t)]:6.3f}+-{c['ratio_TA_se'][tau_key(t)]:.3f}" for t in TAUS)
                print(f"{s:3s} {f:9s} {lv:>5s} {c['length_s']:4.0f} {c['ratio_B']:6.3f}+-{c['ratio_B_se']:.3f} {ta} "
                      f"{c['ratio_TA_stationary']:6.3f}+-{c['ratio_TA_stationary_se']:.3f}")  # fmt: skip
    print(f"{wall:.0f} s wall (this invocation), {cpu / 3600:.2f} h of cell CPU, {args.workers} workers")
    segments = {
        s: {**segment_entry(n0_run2=seg["n0"], line_excess_run2=run2["line_excess_db"][s], floor=seg["floor"],
                            match=seg["match"], match_1bin=seg["match_1bin"]),
            "analog_sd_run2": seg["sigma"], "floor": seg["floor"], "psd": seg["psd"],
            "psd_match": seg["match"], "psd_match_1bin": {"label": "reported, not the check (ruling T4e)",
                                                          **seg["match_1bin"]},
            "shaped_welch_segments": seg["shaped_welch_segments"],
            "self_check": {"zero_jammer": seg["zero_jammer"], "mode_b_vs_inject": seg["mode_b_vs_inject"]}}
        for s, seg in segs.items()
    }  # fmt: skip
    design = {
        "construction": "s28's E2 structure (review.s28_twin_pair_quantisation: make_cell, window_sums, errors, "
        "e2s_stats, write_ibyte) with R-7's six changes; the true front end's recursive AGC is "
        "gnssaudit.inject.pipeline._agc itself",
        "fs_hz": FS, "window_samples": WIN, "quant": {"max_level": FGI.max_level, "zero_level": FGI.zero_level},
        "band_hz": BAND_HZ, "chunk_samples": CHUNK,
        "seeds": {"root": ROOT, "layout": "SeedSequence [33, segment (A 0, B 1), category, key]",
                  "categories": {"ghost": GHOST_CAT, "analog_noise (key = chunk)": NOISE_CAT,
                                 "jammer (2 + family index; key = level)": JAMMER_CAT, "dither": DITHER_CAT},
                  "ghost": {s: ghost_seed(s) for s in SEGMENTS}, "dither": {s: dither_seed(s) for s in SEGMENTS},
                  "bootstrap": f"E2's seed {BOOT_SEED} on both segments (the registered statistic, not a draw)"},
        "psd": {"seconds": PSD_SECONDS, "nperseg": NPERSEG, "window": "hann", "overlap": 0.5,
                "detrend": "constant (scipy's default, as estimate_n0)", "sides": "two-sided, fftshifted"},
        "fir": {"taps": SHAPE_TAPS, "design": "frequency sampling of sqrt(PSD) averaged into the FIR's bins "
                "(fs / 4097), centred, Hann-windowed (scipy.signal.windows.hann, symmetric), scaled to run 2's "
                "analog_sd per component"},
        "psd_check": {"width_bins": CHECK_BINS, "cell_hz": CHECK_BINS * FS / SHAPE_TAPS, "band_hz": CHECK_HALF_BAND_HZ,
                      "tol_db": PSD_TOL_DB, "ruling": "T4e (25 Oct 2026), correcting P5: the Hann window's main lobe "
                      "is 4 FIR bins (4 / N), so the FIR's resolution is 4 bins; the width follows from R-7's design, "
                      "not from the measured PSD; the 1-bin mismatch is reported beside it"},
        "ghost": {"prn": int(GHOST.params["prn"]), "doppler_hz": GHOST.params["doppler_hz"],
                  "chip0": GHOST.params["chip0"], "cn0_db_hz": GHOST.level_db_hz},
        "families": FAMILIES, "levels_db_hz": LEVELS, "true_agc_tau_s": TAUS, "mode_b_tau_s": TAU_S,
        "levels_n0": "run 2's measured n0_per_hz (ruling T4c)",
        "checkpoints_s": CHECKPOINTS_S, "se_stop": SE_STOP,
        "stop_rule": "a cell stops at the first checkpoint where SE(R_B) and SE(R_TA) at every tau are all <= 0.02, "
        "otherwise at 60 s (controller ruling, 25 Oct 2026); the stationary R_TA stops nothing",
        "bootstrap": {"block_windows": BOOT_BLOCK, "n_boot": BOOT_N, "seed": BOOT_SEED},
    }  # fmt: skip
    inputs = {"e2_run2": {k: run2[k] for k in ("run_id", "summary_path", "summary_sha256")},
              "intervals": {s: seg["psd"] for s, seg in segs.items()},
              "code": code_record(paths, code, git0)}  # fmt: skip
    save("s33_front_end_prediction", {
        "status": "written before E2 run 3 (P2c task 4, R-7); it decides only the R9 re-scope's conditional front-end "
                  "range, through s34, never G3",
        "e2_run2": {"label": "analog_sd, n0_per_hz and line_excess_db only; run 2's front-end ratios are not read",
                    **run2},
        "design": design,
        "self_check": {"seeds": seeds, "psd_match": {s: segs[s]["match"]["passes"] for s in segs},
                       "zero_jammer": "segments.*.self_check.zero_jammer", "mode_b_vs_inject":
                       "segments.*.self_check.mode_b_vs_inject",
                       "agc": "the true front end calls gnssaudit.inject.pipeline._agc itself: no re-implementation"},
        "segments": segments,
        "cells": cells,
        "model_limits": MODEL_LIMITS,
        "run_time": {"wall_s": wall, "wall_label": "this invocation's, from its start to the save", "cell_cpu_s": cpu,
                     "workers": args.workers, "cells_dir": str(args.cells_dir),
                     "resumed_cells": [f"{s}-{f}-{level_key(lv)}" for s, f, lv in done],
                     "available_gib_at_start": available,
                     "peak_rss_gib_max": max(r["peak_rss_gib"] for r in results),
                     "rule": "T4a: workers = min(10, floor((available GiB - 4) / peak GiB per cell)), the peak "
                             "measured by the probe; --workers overrides"},
    }, meta_extra={"inputs": inputs, "disclosure": DISCLOSURE})  # fmt: skip


if __name__ == "__main__":
    main(sys.argv[1:])
