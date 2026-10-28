"""The twin pipeline: inject at the native rate, decimate, scale to int16 and write the sink (docs/8 sections 1, 5, 6).

Every step runs on the fixed RNG_BLOCK grid of the absolute sample index, with the decimation filter's history carried
from block to block and zero before the interval starts, so the output and its SHA-256 are bitwise identical for any
chunk size (V5). The clean twin is the same call with every interference gain at zero (`clean_twin`, D-10). Order:
additive interference, then the common-mode phase process, then ghosts (docs/8 section 1).

Front-end modes (docs/8 section 5):
- A: float arithmetic, then int16 at a fixed scale.
- B: each stored sample's analog value is reconstructed as a seeded draw from the recorder's Gaussian noise
  (`analog_sd`), truncated to that sample's quantisation cell (the same draw for every twin of a segment:
  `dither_seed`), the disturbances are added, an AGC holds
  the recording's own smoothed power (first-order recursions on the power of 256-sample sub-blocks on the absolute grid,
  of the reconstruction and of the stream; each sub-block's gain comes from the sub-blocks before it; pre-charged with
  the first), and the stream is requantised to the recording's own alphabet at the native rate, before decimation, as
  a real front end does (docs/8 sections 5 and 6, amended P2b). With nothing added the gain is exactly 1 and every
  sample requantises to itself; a small addition passes with the quantiser's Bussgang gain, as through the recorder,
  whose own noise dithered it. Without the reconstruction, requantising an already quantised recording erased every
  addition below half a step.
In both modes an int16 clip fails the twin (in mode B the quantiser saturates by design; int16 must not).

Both modes can run on several cores (`workers`): the interval is cut into pieces on the block grid, each piece rebuilds
the decimation filter's history by injecting the block before it, and writes its bytes at its own offset, so the output
is bitwise the sequential one. Mode B's AGC carries state through the whole stream, but that state is a recursion on the
sub-block powers of the stream before the AGC: a first pass on all cores computes them, the parent runs the recursion
block by block exactly as the sequential pass does, and each piece starts from its state.
"""

from __future__ import annotations

import functools
import hashlib
import multiprocessing
import os
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import signal, special

from gnssaudit.inject.families import RNG_BLOCK, Disturbance, phase_process, render
from gnssaudit.inject.sink import SinkSpec, open_sink
from gnssaudit.inject.source import IQSource, read_iq
from gnssaudit.io.fetch import commit_part

FULL_SCALE = 32767
AGC_SUB = 256  # samples per AGC sub-block; divides RNG_BLOCK, so sub-blocks never straddle a block
FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class QuantSpec:
    """A recording's own quantiser: levels -max_level..max_level, with or without a zero level (FGI: 4, False)."""

    max_level: int
    zero_level: bool


@dataclass(frozen=True)
class FrontEndSpec:
    decim: int  # output rate is fs / decim; must divide RNG_BLOCK
    fir_taps: int  # anti-alias FIR length, (fir_taps - 1) a multiple of decim; unused when decim == 1
    scale: float  # int16 counts per input count (per requantised count in mode B), one value per twin pair
    mode: Literal["A", "B"] = "A"
    agc_tau_s: float | None = None  # mode B: AGC time constant
    quant: QuantSpec | None = None  # mode B: the alphabet requantised to
    dither_seed: int | None = None  # mode B: the analog reconstruction within each cell, common to a segment's twins
    analog_sd: float | None = None  # mode B: SD of the Gaussian noise the recorder quantised, in counts

    def __post_init__(self) -> None:
        if self.mode == "B" and not (self.agc_tau_s and self.quant and self.dither_seed is not None and self.analog_sd):
            raise ValueError("mode B needs agc_tau_s, quant, dither_seed and analog_sd")


@dataclass(frozen=True)
class InjectionReport:
    n0_per_hz: float
    achieved_db_hz: dict[int, float]  # by index into the disturbance list: J/N0, or C/N0 for a ghost
    clip_count: int
    output_sha256: str
    fir_sha256: str
    samples_out: int
    order: tuple[str, ...]
    timings_s: dict[str, float]


class ClipError(RuntimeError):
    """An int16 value clipped; the output was not committed."""

    def __init__(self, count: int, first_index: int) -> None:
        super().__init__(f"{count} int16 values clipped, first at output sample {first_index}")
        self.count = count
        self.first_index = first_index


def clean_twin(disturbances: Sequence[Disturbance]) -> list[Disturbance]:
    """The same disturbances with every interference gain at zero. Ghosts stay, as part of the recording, and so does
    a noise floor marked `common` (thermal randomisation, docs/3 section 2.4) (D-10)."""
    out = []
    for d in disturbances:
        if d.kind == "ghost" or d.params.get("common", 0.0):
            out.append(d)
        elif d.kind == "phase":
            out.append(replace(d, params={**d.params, "a_rad": 0.0}))
        else:
            out.append(replace(d, level_db_hz=-np.inf))
    return out


def int16_scale(n0_per_hz: float, fs_hz: float, max_level_db_hz: float, headroom_db: float = 30.0) -> float:
    """Counts per input count that put the RMS of noise plus the strongest planned interference headroom_db below
    full scale (docs/8 section 5, mode A)."""
    rms = np.sqrt(n0_per_hz * fs_hz + 10 ** (max_level_db_hz / 10) * n0_per_hz)
    return float(FULL_SCALE / (10 ** (headroom_db / 20) * rms))


def decimation_taps(fe: FrontEndSpec) -> FloatArray:
    """Low-pass FIR passing 90 % of the output Nyquist band, unit gain at DC; [1.0] when there is no decimation."""
    if fe.decim == 1:
        return np.ones(1)
    if (fe.fir_taps - 1) % fe.decim:
        raise ValueError(f"fir_taps - 1 = {fe.fir_taps - 1} must be a multiple of decim = {fe.decim}")
    return np.asarray(signal.firwin(fe.fir_taps, 0.9 / fe.decim), dtype=np.float64)


def requantise(x: ComplexArray, q: QuantSpec) -> ComplexArray:
    """I and Q each to the nearest level of the alphabet (thresholds midway between levels); without a zero level the
    sign decides below the first threshold."""

    def one(v: FloatArray) -> FloatArray:
        mag = np.minimum(np.floor(np.abs(v) + 0.5), q.max_level)
        if not q.zero_level:
            return np.where(v < 0, -1.0, 1.0) * np.maximum(mag, 1.0)
        return np.sign(v) * mag

    return one(x.real) + 1j * one(x.imag)


# ------------------------------------------------------------------------------------------- the block engine
@dataclass
class _State:
    """What one pass over consecutive blocks carries: the decimation history and the AGC's smoothed power."""

    taps: FloatArray
    hist: ComplexArray
    agc_power: float | None = None  # mode B: the stream's smoothed power
    ref_power: float | None = None  # mode B: the reconstructed recording's smoothed power


def _state(fe: FrontEndSpec) -> _State:
    taps = decimation_taps(fe)
    return _State(taps, np.zeros(taps.size - 1, dtype=np.complex128))


def _groups(ds: Sequence[Disturbance]) -> tuple[list[int], list[int], list[int]]:
    additive = [i for i, d in enumerate(ds) if d.kind not in ("phase", "ghost")]
    phases = [i for i, d in enumerate(ds) if d.kind == "phase"]
    ghosts = [i for i, d in enumerate(ds) if d.kind == "ghost"]
    return additive, phases, ghosts


def _power_init(ds: Sequence[Disturbance]) -> dict[int, list[float]]:
    additive, _, ghosts = _groups(ds)
    return {i: [0.0, 0] for i in additive + ghosts}


def _accumulate(acc: list[float], d: Disturbance, c: ComplexArray, t: FloatArray) -> None:
    """The component's power over the samples past its onset ramp (the achieved level, docs/8 2.2)."""
    on = t >= d.onset_s + d.ramp_s
    acc[0] += float(np.sum(np.abs(c[on]) ** 2))
    acc[1] += int(on.sum())


def _cells(v: FloatArray, q: QuantSpec) -> tuple[FloatArray, FloatArray]:
    """The analog interval each stored level stands for (thresholds midway between levels; without a zero level the
    sign decides, so +-1 cover (0, 1.5) and (-1.5, 0); the outer levels reach to infinity)."""
    k = np.abs(v)
    lo = np.where(k >= q.max_level, q.max_level - 0.5, k - 0.5)
    hi = np.where(k >= q.max_level, np.inf, k + 0.5)
    if not q.zero_level:
        lo = np.where(k <= 1, 0.0, lo)
    neg = v < 0
    lo_s, hi_s = np.where(neg, -hi, lo), np.where(neg, -lo, hi)
    if q.zero_level:
        zero = k == 0
        lo_s, hi_s = np.where(zero, -0.5, lo_s), np.where(zero, 0.5, hi_s)
    return lo_s, hi_s


@functools.lru_cache(maxsize=8)
def _cell_table(q: QuantSpec, sd: float) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Per level -max_level..max_level (index level + max_level): the lower bound from `_cells` itself, the ndtr of
    each bound over sd and the largest float below the upper bound, the very floats a per-sample call returns. Cached
    and shared by every call with the same (q, sd), so read-only."""
    levels = np.arange(-q.max_level, q.max_level + 1, dtype=np.float64)
    c_lo, c_hi = _cells(levels, q)
    table = (c_lo, special.ndtr(c_lo / sd), special.ndtr(c_hi / sd), np.nextafter(c_hi, -np.inf))
    for arr in table:
        arr.setflags(write=False)
    return table


def reconstruct_analog(x: ComplexArray, q: QuantSpec, sd: float, seed: int, lo: int) -> ComplexArray:
    """Mode B: each component's analog value, drawn from N(0, sd^2) truncated to its level's cell, with uniforms on the
    fixed block grid (the same draw for every twin of a segment). complex128 input only: the floats are those of the
    per-sample formula for it (P2b part 2, task 6), not for complex64."""
    if x.dtype != np.complex128:
        raise TypeError(f"mode B reconstructs complex128 samples, not {x.dtype}")
    n = x.size
    u = np.empty(2 * n)
    for b in range(lo // RNG_BLOCK, (lo + n - 1) // RNG_BLOCK + 1):
        a, e = max(lo, b * RNG_BLOCK), min(lo + n, (b + 1) * RNG_BLOCK)
        blk = np.random.default_rng([seed, 0xD17, b]).random(2 * RNG_BLOCK)
        u[2 * (a - lo) : 2 * (e - lo)] = blk[2 * (a - b * RNG_BLOCK) : 2 * (e - b * RNG_BLOCK)]

    c_lo_t, p_lo_t, p_hi_t, c_top_t = _cell_table(q, sd)

    def one(v: FloatArray, w: FloatArray) -> FloatArray:
        if not np.all(v == np.rint(v)):  # NaN fails too
            raise ValueError("mode B: stored levels must be integer-valued (a recording on an integer alphabet)")
        i = np.clip(v, -q.max_level, q.max_level).astype(np.intp) + q.max_level  # beyond the outer level: its cell
        c_lo, c_top, p_lo, p_hi = c_lo_t[i], c_top_t[i], p_lo_t[i], p_hi_t[i]
        r = sd * special.ndtri(p_lo + w * (p_hi - p_lo))
        return np.clip(r, c_lo, c_top)  # within the cell, so it requantises to itself

    return one(x.real, u[0::2]) + 1j * one(x.imag, u[1::2])


def estimate_analog_sd(source: IQSource, q: QuantSpec, max_samples: int = 1 << 24) -> float:
    """SD of the Gaussian noise a recorder quantised to the alphabet q, from the recording's first samples: without a
    zero level from the share of +-1 (|v| < 1.5), with one and few levels from the share of 0 (|v| < 0.5), and for a
    fine alphabet from the mean square less the quantisation noise (Sheppard's 1/12)."""
    x = read_iq(source, source.sample_start, min(max_samples, source.sample_end - source.sample_start))
    v = np.concatenate([x.real, x.imag]).astype(np.float64)
    if not q.zero_level:
        return float(1.5 / special.ndtri((1.0 + np.mean(np.abs(v) == 1)) / 2.0))
    if q.max_level <= 8:
        return float(0.5 / special.ndtri((1.0 + np.mean(v == 0)) / 2.0))
    return float(np.sqrt(np.mean(v**2) - 1.0 / 12.0))


def _injected(
    source: IQSource | None,
    ds: Sequence[Disturbance],
    lo: int,
    n: int,
    fs: float,
    n0_per_hz: float,
    power: dict[int, list[float]] | None,
    analog: tuple[QuantSpec, float, int] | None = None,
) -> tuple[ComplexArray, ComplexArray | None]:
    """Samples [lo, lo + n) at the native rate after injection, and (mode B: analog = quantiser, noise SD, seed) the
    reconstructed analog recording they were added to; source None gives the disturbances on zeros."""
    additive, phases, ghosts = _groups(ds)
    x = np.zeros(n, dtype=np.complex128) if source is None else read_iq(source, lo, n).astype(np.complex128)
    ref = None
    if analog is not None and source is not None:
        q, sd, seed = analog
        x = reconstruct_analog(x, q, sd, seed, lo)
        ref = x.copy()
    t = np.arange(lo, lo + n, dtype=np.int64) / fs
    for i in additive:
        c = render(ds[i], lo, n, fs, n0_per_hz)
        if power is not None:
            _accumulate(power[i], ds[i], c, t)
        x += c
    if phases:
        phi = np.sum([phase_process(ds[i], lo, n, fs) for i in phases], axis=0)
        if np.any(phi):
            x *= np.exp(1j * phi)
    for i in ghosts:
        c = render(ds[i], lo, n, fs, n0_per_hz)
        if power is not None:
            _accumulate(power[i], ds[i], c, t)
        x += c
    return x, ref


def _analog(fe: FrontEndSpec) -> tuple[QuantSpec, float, int] | None:
    if fe.mode != "B":
        return None
    assert fe.quant is not None and fe.analog_sd is not None and fe.dither_seed is not None
    return fe.quant, fe.analog_sd, fe.dither_seed


def _alpha(fe: FrontEndSpec, fs: float) -> float:
    """The AGC recursion's weight per sub-block."""
    assert fe.agc_tau_s is not None
    return 1.0 - np.exp(-AGC_SUB / (fe.agc_tau_s * fs))


def _sub_blocks(n: int, lo: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Starts (and the end) of the AGC sub-blocks of samples [lo, lo + n) on the absolute grid, and their widths."""
    edges = np.unique(np.concatenate([[0], np.arange(-lo % AGC_SUB, n, AGC_SUB), [n]]))
    return edges, np.diff(edges)


def _sub_powers(x: ComplexArray, ref: ComplexArray, lo: int) -> tuple[FloatArray, FloatArray]:
    """Mean power per AGC sub-block of the stream before the AGC and of the reconstructed recording."""
    edges, width = _sub_blocks(x.size, lo)
    return np.add.reduceat(np.abs(x) ** 2, edges[:-1]) / width, np.add.reduceat(np.abs(ref) ** 2, edges[:-1]) / width


def _smoothed_before(p: FloatArray, alpha: float, state: float | None) -> tuple[FloatArray, float]:
    """A first-order recursion on sub-block powers: the value before each sub-block, and the state after the last
    (pre-charged with the first sub-block when there is no state yet)."""
    s0 = float(p[0]) if state is None else state
    smoothed, _ = signal.lfilter([alpha], [1.0, -(1.0 - alpha)], p, zi=[(1.0 - alpha) * s0])
    return np.concatenate([[s0], smoothed[:-1]]), float(smoothed[-1])


def _agc(x: ComplexArray, ref: ComplexArray, lo: int, fe: FrontEndSpec, fs: float, st: _State) -> ComplexArray:
    """Mode B: each AGC sub-block (absolute grid) scaled by the root of the recording's smoothed power over the
    stream's, both before it: the AGC holds the power the recording had."""
    alpha = _alpha(fe, fs)
    p, q = _sub_powers(x, ref, lo)
    before_p, st.agc_power = _smoothed_before(p, alpha, st.agc_power)
    before_q, st.ref_power = _smoothed_before(q, alpha, st.ref_power)
    return x * np.repeat(np.sqrt(before_q / before_p), _sub_blocks(x.size, lo)[1])


def _front_end(
    x: ComplexArray, lo: int, start: int, fe: FrontEndSpec, fs: float, st: _State, ref: ComplexArray | None = None
) -> ComplexArray:
    """Mode B's AGC and requantisation, then decimation (the output grid is start, start + decim, ...)."""
    if fe.mode == "B":
        assert fe.quant is not None and ref is not None
        x = requantise(_agc(x, ref, lo, fe, fs, st), fe.quant)
    dd = fe.decim
    if dd == 1:
        return x
    w = np.concatenate([st.hist, x])
    st.hist = w[w.size - st.hist.size :]
    i0 = (start - lo) % dd  # first sample of this block on the output grid
    s = (st.taps.size - 1 + i0) % dd
    full = _fir_decimate(st.taps, w[s:], dd)
    j0 = (st.taps.size - 1 + i0 - s) // dd
    return full[j0 : j0 + len(range(i0, x.size, dd))]


def _fir_decimate(taps: FloatArray, w: ComplexArray, dd: int) -> ComplexArray:
    """`signal.upfirdn(taps, w, down=dd)` on the real and the imaginary part apart: scipy would cast the real taps to
    complex, and the real filter forms the same products in the same order at about half the cost."""
    re = signal.upfirdn(taps, w.real, up=1, down=dd)
    out = np.empty(re.size, dtype=np.complex128)
    out.real, out.imag = re, signal.upfirdn(taps, w.imag, up=1, down=dd)
    return out


def _blocks(lo: int, hi: int) -> Iterator[tuple[int, int]]:
    """(lo, n) of the grid blocks covering [lo, hi)."""
    for b0 in range(lo - lo % RNG_BLOCK, hi, RNG_BLOCK):
        a, b = max(lo, b0), min(hi, b0 + RNG_BLOCK)
        if b > a:
            yield a, b - a


def _to_int16(y: ComplexArray, scale: float) -> tuple[bytes, int, int]:
    """(little-endian bytes, clip count, first clipped index within y or -1)."""
    iq = np.empty(2 * y.size)
    iq[0::2], iq[1::2] = np.rint(scale * y.real), np.rint(scale * y.imag)
    bad = (iq > FULL_SCALE) | (iq < -FULL_SCALE - 1)
    first = int(np.flatnonzero(bad)[0]) // 2 if bad.any() else -1
    return np.clip(iq, -FULL_SCALE - 1, FULL_SCALE).astype("<i2").tobytes(), int(bad.sum()), first


def _out_index(m: int, start: int, dd: int) -> int:
    """Output index of the first output sample at or after input index m."""
    return -(-(m - start) // dd)


# ----------------------------------------------------------------------------------------------- the pipeline
def inject(
    source: IQSource,
    disturbances: Sequence[Disturbance],
    frontend: FrontEndSpec,
    sink: SinkSpec,
    *,
    n0_per_hz: float,
    chunk: int = 1 << 20,
    workers: int = 1,
) -> InjectionReport:
    """Write one twin of `source` with `disturbances` to `sink` and report what was injected. `chunk` is I/O
    granularity only; `workers` > 1 runs on that many processes with a bitwise-identical output."""
    if RNG_BLOCK % frontend.decim:
        raise ValueError(f"decim {frontend.decim} must divide the block of {RNG_BLOCK} samples")
    t0 = time.perf_counter()
    ds = list(disturbances)
    additive, phases, ghosts = _groups(ds)
    taps = decimation_taps(frontend)
    n_out = _out_index(source.sample_end, source.sample_start, frontend.decim)
    if workers > 1:
        sha, clips, power = _parallel(source, ds, frontend, sink, n0_per_hz, workers, n_out)
    else:
        sha, clips, power = _sequential(source, ds, frontend, sink, n0_per_hz, chunk)
    achieved = {i: (10 * np.log10(p / k / n0_per_hz) if k and p > 0 else float("-inf")) for i, (p, k) in power.items()}
    return InjectionReport(
        n0_per_hz=n0_per_hz,
        achieved_db_hz={i: float(v) for i, v in achieved.items()},
        clip_count=clips,
        output_sha256=sha,
        fir_sha256=hashlib.sha256(taps.astype("<f8").tobytes()).hexdigest(),
        samples_out=n_out,
        order=tuple(ds[i].kind for i in additive + phases + ghosts),
        timings_s={"total": round(time.perf_counter() - t0, 3), "workers": float(workers)},
    )


def _sequential(
    source: IQSource, ds: Sequence[Disturbance], fe: FrontEndSpec, sink: SinkSpec, n0_per_hz: float, chunk: int
) -> tuple[str, int, dict[int, list[float]]]:
    start, end, fs = source.sample_start, source.sample_end, source.fs_hz
    st = _state(fe)
    power = _power_init(ds)
    sha = hashlib.sha256()
    clips, first_clip, n_out = 0, -1, 0
    with open_sink(sink) as fh:
        for c0 in range(start - start % RNG_BLOCK, end, max(chunk, RNG_BLOCK)):
            for lo, n in _blocks(max(start, c0), min(end, c0 + max(chunk, RNG_BLOCK))):
                x, ref = _injected(source, ds, lo, n, fs, n0_per_hz, power, _analog(fe))
                y = _front_end(x, lo, start, fe, fs, st, ref)
                data, k, first = _to_int16(y, fe.scale)
                if k:
                    clips += k
                    first_clip = n_out + first if first_clip < 0 else first_clip
                sha.update(data)
                fh.write(data)
                n_out += y.size
        if clips:
            raise ClipError(clips, first_clip)
    return sha.hexdigest(), clips, power


@dataclass(frozen=True)
class _AgcPass:
    """Mode B's first pass over one piece, block by block: the AGC sub-block powers (`_sub_powers`, concatenated),
    each block's count of sub-blocks, and each block's power sum per disturbance and the piece's sample counts."""

    p: FloatArray
    q: FloatArray
    sizes: list[int]
    sums: dict[int, list[float]]
    counts: dict[int, int]


def _agc_pass(
    source: IQSource, ds: list[Disturbance], fe: FrontEndSpec, n0_per_hz: float, lo: int, hi: int
) -> _AgcPass:
    fs = source.fs_hz
    ps, qs, sizes = [], [], []
    sums: dict[int, list[float]] = {i: [] for i in _power_init(ds)}
    counts = dict.fromkeys(sums, 0)
    for b_lo, n in _blocks(lo, hi):
        power = _power_init(ds)
        x, ref = _injected(source, ds, b_lo, n, fs, n0_per_hz, power, _analog(fe))
        assert ref is not None
        p, q = _sub_powers(x, ref, b_lo)
        ps.append(p)
        qs.append(q)
        sizes.append(p.size)
        for i, (s, k) in power.items():
            sums[i].append(s)
            counts[i] += int(k)
    return _AgcPass(np.concatenate(ps), np.concatenate(qs), sizes, sums, counts)


def _agc_states(passes: Sequence[_AgcPass], alpha: float) -> list[tuple[float | None, float | None]]:
    """For each piece, the AGC's (stream, recording) state before the last block of the piece before it ((None, None)
    for the first piece): the same recursion, block by block on the same powers, as the sequential pass runs."""
    state: tuple[float | None, float | None] = (None, None)
    out: list[tuple[float | None, float | None]] = [state]
    for a in passes[:-1]:
        off = 0
        for k, m in enumerate(a.sizes):
            if k == len(a.sizes) - 1:
                out.append(state)
            sp = _smoothed_before(a.p[off : off + m], alpha, state[0])[1]
            sq = _smoothed_before(a.q[off : off + m], alpha, state[1])[1]
            state, off = (sp, sq), off + m
    return out


def _piece(
    source: IQSource,
    ds: list[Disturbance],
    fe: FrontEndSpec,
    n0_per_hz: float,
    lo: int,
    hi: int,
    part: str,
    agc: tuple[float | None, float | None] = (None, None),
) -> tuple[int, int, dict[int, list[float]] | None]:
    """Inject [lo, hi) and write its output at its offset in the part file: (clips, first clipped output index or -1,
    power sums; mode B: none, its first pass has them). The decimation history is rebuilt by injecting the block before
    lo, as the sequential pass holds it. Mode B: `agc` is the AGC's state before that block; the block runs through
    the AGC (which leaves the state at lo) and the requantisation, as the sequential pass ran it."""
    start, fs = source.sample_start, source.fs_hz
    analog = _analog(fe)
    st = _state(fe)
    if lo > start and (st.taps.size > 1 or analog is not None):
        p_lo = max(start, lo - RNG_BLOCK)
        prev, ref = _injected(source, ds, p_lo, lo - p_lo, fs, n0_per_hz, None, analog)
        if analog is not None:
            assert ref is not None
            st.agc_power, st.ref_power = agc
            prev = requantise(_agc(prev, ref, p_lo, fe, fs, st), analog[0])
        if st.taps.size > 1:
            st.hist = np.concatenate([st.hist, prev])[-(st.taps.size - 1) :]
    power = _power_init(ds) if analog is None else None
    clips, first_clip, n_out = 0, -1, 0
    out0 = _out_index(lo, start, fe.decim)
    fd = os.open(part, os.O_WRONLY)
    try:
        for b_lo, n in _blocks(lo, hi):
            x, ref = _injected(source, ds, b_lo, n, fs, n0_per_hz, power, analog)
            y = _front_end(x, b_lo, start, fe, fs, st, ref)
            data, k, first = _to_int16(y, fe.scale)
            if k:
                clips += k
                first_clip = out0 + n_out + first if first_clip < 0 else first_clip
            os.pwrite(fd, data, 4 * (out0 + n_out))
            n_out += y.size
        os.fsync(fd)
    finally:
        os.close(fd)
    return clips, first_clip, power


def _parallel(
    source: IQSource,
    ds: Sequence[Disturbance],
    fe: FrontEndSpec,
    sink: SinkSpec,
    n0_per_hz: float,
    workers: int,
    n_out: int,
) -> tuple[str, int, dict[int, list[float]]]:
    start, end = source.sample_start, source.sample_end
    first_block = start - start % RNG_BLOCK
    n_blocks = -(-(end - first_block) // RNG_BLOCK)
    cuts = sorted({first_block + RNG_BLOCK * round(k * n_blocks / workers) for k in range(1, workers)})
    bounds = [start, *[c for c in cuts if start < c < end], end]
    sink.path.parent.mkdir(parents=True, exist_ok=True)
    part = sink.path.with_name(sink.path.name + ".part")
    with part.open("wb") as fh:
        fh.truncate(4 * n_out)
    try:
        ctx = multiprocessing.get_context("spawn")
        pieces = list(zip(bounds[:-1], bounds[1:], strict=True))
        power = _power_init(ds)
        agc: list[tuple[float | None, float | None]] = [(None, None)] * len(pieces)
        with ProcessPoolExecutor(max_workers=len(pieces), mp_context=ctx) as pool:
            if fe.mode == "B":  # the first pass: each piece's AGC state, and the power sums added in block order
                passes = [f.result() for f in [pool.submit(_agc_pass, source, list(ds), fe, n0_per_hz, a, b)
                                               for a, b in pieces]]  # fmt: skip
                agc = _agc_states(passes, _alpha(fe, source.fs_hz))
                for a in passes:
                    for i, sums in a.sums.items():
                        for s in sums:
                            power[i][0] += s
                        power[i][1] += a.counts[i]
            futures = [pool.submit(_piece, source, list(ds), fe, n0_per_hz, a, b, str(part), agc[k])
                       for k, (a, b) in enumerate(pieces)]  # fmt: skip
            results = [f.result() for f in futures]
        clips = sum(r[0] for r in results)
        if clips:
            raise ClipError(clips, min(r[1] for r in results if r[1] >= 0))
        for _, _, pw in results:
            for i, (p, k) in (pw or {}).items():
                power[i][0] += p
                power[i][1] += k
        sha = hashlib.sha256()
        with part.open("rb") as fh:
            while data := fh.read(1 << 24):
                sha.update(data)
        commit_part(part, sink.path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return sha.hexdigest(), clips, power


def interference(
    source: IQSource, disturbances: Sequence[Disturbance], frontend: FrontEndSpec, *, n0_per_hz: float, block: int
) -> Iterator[tuple[int, ComplexArray]]:
    """(first output index, decimated samples) of the interference alone over the source's interval, in float and in
    chunks of `block` input samples: the additive disturbances without ghosts or a common floor, through the same
    decimation as the twins (mode A's path; the int16 scale is not applied). The first-order B1 prediction, P2b task 11.
    The recording is not read: its samples do not enter."""
    ds = [d for d in disturbances if d.kind not in ("phase", "ghost") and not d.params.get("common", 0.0)]
    start, end, fs = source.sample_start, source.sample_end, source.fs_hz
    fe = replace(frontend, mode="A", agc_tau_s=None, quant=None, dither_seed=None, analog_sd=None)
    st = _state(fe)
    n_out = 0
    for c0 in range(start - start % RNG_BLOCK, end, max(block, RNG_BLOCK)):
        ys = [
            _front_end(_injected(None, ds, lo, n, fs, n0_per_hz, None)[0], lo, start, fe, fs, st)
            for lo, n in _blocks(max(start, c0), min(end, c0 + max(block, RNG_BLOCK)))
        ]
        y = np.concatenate(ys) if ys else np.zeros(0, dtype=np.complex128)
        yield n_out, y
        n_out += y.size
