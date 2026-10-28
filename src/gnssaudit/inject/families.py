"""Disturbance families on the recording's sample grid (docs/8 sections 1, 3 and 4).

Every waveform is a deterministic function of the absolute sample index and the disturbance seed. It is computed in
fixed blocks of RNG_BLOCK samples and then sliced, so its values are bitwise identical however a caller chunks the
stream (V5): the random draws, the FIR filtering and the vectorized maths always see the same arrays of the same size.
Additive waveforms have unit power; `render` scales them to their level above N0 (D-12).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import fft as sp_fft
from scipy import signal

from gnssaudit.inject.calibrate import gain
from gnssaudit.signals import carrier_hz
from gnssaudit.sim.despread import ca_code, cw, swept_cw

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
RNG_BLOCK = 1 << 16
_BIT_BLOCK = 4096  # ghost data bits are drawn in blocks of this many bits
Kind = Literal["noise", "cw", "swept_cw", "chirp", "phase", "ghost"]


@dataclass(frozen=True)
class Disturbance:
    """One disturbance (docs/6 section 4.2). `phase` is a common-mode phase process, the others are additive."""

    kind: Kind
    params: Mapping[str, float]
    onset_s: float
    ramp_s: float
    level_db_hz: float  # J/N0 for additive kinds, C/N0 for a ghost; unused for a phase process
    seed: int


@dataclass(frozen=True)
class GhostSpec:
    """A synthetic GPS L1 C/A satellite (docs/8 section 4): Doppler with a linear rate, random 50 bit/s data."""

    prn: int
    doppler_hz: float
    doppler_rate_hzps: float
    cn0_db_hz: float
    chip0: float  # code phase in chips at sample 0
    seed: int

    @classmethod
    def of(cls, d: Disturbance) -> GhostSpec:
        p = d.params
        return cls(int(p["prn"]), float(p["doppler_hz"]), float(p.get("doppler_rate_hzps", 0.0)), d.level_db_hz,
                   float(p.get("chip0", 0.0)), d.seed)  # fmt: skip


def disturbance_seed(root: int, config: int, realization: int, index: int) -> int:
    """The seed of disturbance `index` in realization `realization` of configuration `config` (docs/6 section 7.4)."""
    return int(np.random.SeedSequence([root, config, realization, index]).generate_state(1, dtype=np.uint64)[0])


def _rng(seed: int, *keys: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, *keys])))


def _envelope(t: FloatArray, onset_s: float, ramp_s: float) -> FloatArray:
    e = np.where(t >= onset_s, 1.0, 0.0)
    if ramp_s > 0:
        m = (t >= onset_s) & (t < onset_s + ramp_s)
        e[m] = 0.5 * (1 - np.cos(np.pi * (t[m] - onset_s) / ramp_s))
    return e


@lru_cache(maxsize=32)
def _noise_taps(bw_hz: float, f_hz: float, fs_hz: float) -> ComplexArray:
    """Kaiser band-pass of width bw_hz centred at f_hz, transition 0.5 % of bw_hz, 60 dB; unit noise gain."""
    numtaps, beta = signal.kaiserord(60.0, 0.005 * bw_hz / (0.5 * fs_hz))
    numtaps |= 1
    if numtaps >= RNG_BLOCK:
        raise ValueError(f"a {bw_hz} Hz band at {fs_hz} Hz needs {numtaps} taps, more than one block")
    k = np.arange(numtaps) - (numtaps - 1) / 2
    h = np.sinc(bw_hz / fs_hz * k) * np.kaiser(numtaps, beta)  # windowed ideal low-pass, cut-off bw/2
    hc = h * np.exp(2j * np.pi * f_hz * k / fs_hz)
    return hc / np.sqrt(np.sum(np.abs(hc) ** 2))


@lru_cache(maxsize=8)
def _white(seed: int, block: int) -> ComplexArray:
    rng = _rng(seed, 0, block + 1)  # stream 0, offset so that the block before the first one has a key too
    w = (rng.standard_normal(RNG_BLOCK) + 1j * rng.standard_normal(RNG_BLOCK)) / np.sqrt(2)
    w.flags.writeable = False
    return w


@lru_cache(maxsize=32)
def _noise_fft(bw_hz: float, f_hz: float, fs_hz: float) -> tuple[ComplexArray, int, int]:
    """The taps' FFT at the overlap-save size for one block plus the filter history, the size and the tap count."""
    h = _noise_taps(bw_hz, f_hz, fs_hz)
    nfft = sp_fft.next_fast_len(RNG_BLOCK + h.size - 1)
    assert nfft is not None  # scipy returns None only for a non-positive target
    return np.asarray(sp_fft.fft(h, nfft), dtype=np.complex128), nfft, h.size


def _noise_block(d: Disturbance, b: int, fs: float) -> ComplexArray:
    """Overlap-save on the fixed grid: block b's white noise after the last L - 1 samples of block b - 1."""
    hf, nfft, taps = _noise_fft(float(d.params["bw_hz"]), float(d.params.get("f_hz", 0.0)), fs)
    x = np.concatenate([_white(d.seed, b - 1)[RNG_BLOCK - (taps - 1) :], _white(d.seed, b)])
    y = sp_fft.ifft(np.asarray(sp_fft.fft(x, nfft)) * hf)
    return np.asarray(y[taps - 1 : taps - 1 + RNG_BLOCK], dtype=np.complex128)


def _sweep_block(d: Disturbance, b: int, fs: float) -> ComplexArray:
    p = d.params
    span, period = float(p["span_hz"]), float(p["period_s"])
    rng = _rng(d.seed, 1)
    psi, offset = rng.uniform(0, 2 * np.pi), rng.uniform(0, period)
    if "offset_s" in p:  # fixed timing, drawn phase: the conditional oracle (F8, docs/3 section 2.4)
        offset = float(p["offset_s"])
    return swept_cw(RNG_BLOCK, fs, float(p.get("f_hz", 0.0)) - span / 2, span, period, 1.0, psi, offset,
                    b * RNG_BLOCK, bool(p.get("triangle", 0.0)))  # fmt: skip


def _cw_block(d: Disturbance, b: int, fs: float) -> ComplexArray:
    psi = _rng(d.seed, 1).uniform(0, 2 * np.pi)
    return cw(RNG_BLOCK, fs, float(d.params["f_hz"]), 1.0, psi, b * RNG_BLOCK)


def ghost_phase_rad(g: GhostSpec, sample_index: NDArray[np.int64], fs_hz: float) -> FloatArray:
    """theta_g at the given samples: 2 pi (f_D t + f_dot t^2 / 2) + psi_0, in float64 (docs/3 section 3.1)."""
    return _ghost_phase(g, sample_index / fs_hz)


def _ghost_phase(g: GhostSpec, t: FloatArray) -> FloatArray:
    psi0 = _rng(g.seed, 1).uniform(0, 2 * np.pi)
    return 2 * np.pi * (g.doppler_hz * t + 0.5 * g.doppler_rate_hzps * t**2) + psi0


def ghost_code_chips(g: GhostSpec, sample_index: NDArray[np.int64], fs_hz: float) -> FloatArray:
    """Code phase in chips: the code clock follows the carrier Doppler, d(chips)/dt = 1.023 MHz (1 + f_D / f_L1)."""
    return _ghost_chips(g, sample_index / fs_hz)


def _ghost_chips(g: GhostSpec, t: FloatArray) -> FloatArray:
    return g.chip0 + 1.023e6 * (t + (g.doppler_hz * t + 0.5 * g.doppler_rate_hzps * t**2) / carrier_hz("G", "1C"))


@lru_cache(maxsize=64)
def _code(prn: int) -> FloatArray:
    code = ca_code(prn)
    code.flags.writeable = False
    return code


@lru_cache(maxsize=8)
def _bit_block(seed: int, block: int) -> FloatArray:
    bits = np.where(_rng(seed, 2, block).random(_BIT_BLOCK) < 0.5, -1.0, 1.0)
    bits.flags.writeable = False
    return bits


def _ghost_bits(seed: int, bit_index: NDArray[np.int64]) -> FloatArray:
    """The data bits at the given bit indices, drawn in fixed blocks of bits (indices rise with the sample index)."""
    b0, b1 = int(bit_index.min()) // _BIT_BLOCK, int(bit_index.max()) // _BIT_BLOCK
    table = np.concatenate([_bit_block(seed, blk) for blk in range(b0, b1 + 1)])
    return table[bit_index - b0 * _BIT_BLOCK]


def ghost_modulation(g: GhostSpec, sample_index: NDArray[np.int64], fs_hz: float) -> ComplexArray:
    """The ghost's unit-amplitude signal at the given native samples: data bits x C/A code x carrier (docs/3 section
    3.1). The injector draws exactly this; the first-order B1 prediction despreads by it (P2b task 11)."""
    t = sample_index / fs_hz
    whole = np.floor(_ghost_chips(g, t)).astype(np.int64)  # chips // 1, without float floor division's cost
    code = _code(g.prn)[whole % 1023]
    bits = _ghost_bits(g.seed, whole // (1023 * 20))  # chips // 20460, as floor(x / n) = floor(floor(x) / n)
    return np.asarray(bits * code * np.exp(1j * _ghost_phase(g, t)), dtype=np.complex128)


def _ghost_block(d: Disturbance, b: int, fs: float) -> ComplexArray:
    return ghost_modulation(GhostSpec.of(d), np.arange(b * RNG_BLOCK, (b + 1) * RNG_BLOCK, dtype=np.int64), fs)


_BLOCKS: dict[str, Callable[[Disturbance, int, float], ComplexArray]] = {
    "noise": _noise_block,
    "cw": _cw_block,
    "swept_cw": _sweep_block,
    "chirp": _sweep_block,  # a PPD-like chirp is a fast sawtooth sweep (docs/8 section 3)
    "ghost": _ghost_block,
}


def _on_grid(n0: int, n: int, fn: Callable[[int], NDArray[np.generic]], dtype: type) -> NDArray[np.generic]:
    out = np.empty(n, dtype=dtype)
    if n == 0:
        return out
    for b in range(n0 // RNG_BLOCK, (n0 + n - 1) // RNG_BLOCK + 1):
        lo, hi = max(n0, b * RNG_BLOCK), min(n0 + n, (b + 1) * RNG_BLOCK)
        out[lo - n0 : hi - n0] = fn(b)[lo - b * RNG_BLOCK : hi - b * RNG_BLOCK]
    return out


def _times(b: int, fs: float) -> FloatArray:
    return np.arange(b * RNG_BLOCK, (b + 1) * RNG_BLOCK, dtype=np.int64) / fs


def unit_waveform(d: Disturbance, n0: int, n: int, fs_hz: float) -> ComplexArray:
    """Samples [n0, n0 + n) of the unit-power additive waveform, with its raised-cosine onset applied."""
    if d.kind not in _BLOCKS:
        raise ValueError(f"{d.kind!r} is not an additive disturbance (a phase process multiplies the stream)")
    make = _BLOCKS[d.kind]

    def block(b: int) -> ComplexArray:
        # the block's times rise, so its first and last decide where the envelope is constant (the same divisions
        # as `_times`); there the waveform is not drawn (before the onset) or not multiplied (past the ramp)
        if ((b + 1) * RNG_BLOCK - 1) / fs_hz < d.onset_s:
            return np.zeros(RNG_BLOCK, dtype=np.complex128)
        if b * RNG_BLOCK / fs_hz >= d.onset_s + d.ramp_s:
            return make(d, b, fs_hz)
        return make(d, b, fs_hz) * _envelope(_times(b, fs_hz), d.onset_s, d.ramp_s)

    return _on_grid(n0, n, block, np.complex128)  # type: ignore[return-value]


def render(d: Disturbance, n0: int, n: int, fs_hz: float, n0_per_hz: float) -> ComplexArray:
    """The injected component: the unit waveform times sqrt(10^(level/10) N0). A zero gain (the clean twin's
    interference, D-10) gives exact zeros without generating the waveform."""
    g = np.float64(gain(d.level_db_hz, n0_per_hz))
    if g == 0.0:
        return np.zeros(n, dtype=np.complex128)
    return g * unit_waveform(d, n0, n, fs_hz)


def tone_set(d: Disturbance) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Frequencies (log-spaced), amplitudes and random phases of a phase-tone process (docs/8 section 3)."""
    p = d.params
    k = int(p["n_tones"])
    freqs = np.geomspace(float(p["f_min_hz"]), float(p["f_max_hz"]), k)
    return freqs, np.full(k, float(p["a_rad"])), _rng(d.seed, 3).uniform(0, 2 * np.pi, k)


def phase_process(d: Disturbance, n0: int, n: int, fs_hz: float) -> FloatArray:
    """Samples [n0, n0 + n) of the common-mode phase phi(t) = sum_k a_k sin(2 pi f_k t + psi_k), in radians."""
    if d.kind != "phase":
        raise ValueError(f"{d.kind!r} is not a phase process")
    freqs, amps, phases = tone_set(d)

    def block(b: int) -> FloatArray:
        t = _times(b, fs_hz)
        phi = np.zeros(RNG_BLOCK)
        for f, a, p in zip(freqs, amps, phases, strict=True):
            phi += a * np.sin(2 * np.pi * f * t + p)
        return phi * _envelope(t, d.onset_s, d.ramp_s)

    return _on_grid(n0, n, block, np.float64)  # type: ignore[return-value]
