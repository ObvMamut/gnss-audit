"""The injector's speed-ups (branch perf-inject) against frozen copies of the code each one replaced: the same values on
the same inputs, so every twin keeps its bytes (the full-unit check re-injects E2 run 2's units against their recorded
SHA-256). Where an exact zero may change sign, the values are compared, since no output byte keeps a zero's sign."""

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import signal

from gnssaudit.inject import RNG_BLOCK, Disturbance, GhostSpec, ghost_modulation, unit_waveform
from gnssaudit.inject import families as F
from gnssaudit.inject.pipeline import _fir_decimate
from gnssaudit.signals import carrier_hz

FGI_FS = 30.69e6


def _ref_ghost_modulation(g: GhostSpec, sample_index: NDArray[np.int64], fs_hz: float) -> NDArray[np.complex128]:
    """Frozen copy of ghost_modulation before the speed-up: time computed twice, chips // 1 and chips // 20460."""
    t = sample_index / fs_hz
    chips = g.chip0 + 1.023e6 * (t + (g.doppler_hz * t + 0.5 * g.doppler_rate_hzps * t**2) / carrier_hz("G", "1C"))
    t = sample_index / fs_hz
    psi0 = F._rng(g.seed, 1).uniform(0, 2 * np.pi)
    phase = 2 * np.pi * (g.doppler_hz * t + 0.5 * g.doppler_rate_hzps * t**2) + psi0
    code = F._code(g.prn)[(chips // 1).astype(np.int64) % 1023]
    bits = F._ghost_bits(g.seed, (chips // (1023 * 20)).astype(np.int64))
    return np.asarray(bits * code * np.exp(1j * phase), dtype=np.complex128)


@pytest.mark.parametrize("fs", [FGI_FS, 1.0e6])
@pytest.mark.parametrize(("doppler", "rate"), [(1500.0, 0.0), (-2500.0, 0.0), (-2500.0, 0.8)])
def test_ghost_modulation_is_bitwise_the_formula_it_replaced(fs: float, doppler: float, rate: float) -> None:
    """Blocks at the start, near 4 s and near 120 s of a 30.69 MS/s recording, where a data bit and the code roll over
    inside the block; the floor of the chips and the integer bit index give the very indices // gave."""
    g = GhostSpec(5, doppler, rate, 45.0, 118.3, 12345)
    for b in (0, 2000, 56000):
        idx = np.arange(b * RNG_BLOCK, (b + 1) * RNG_BLOCK, dtype=np.int64)
        np.testing.assert_array_equal(ghost_modulation(g, idx, fs).view(np.int64),
                                      _ref_ghost_modulation(g, idx, fs).view(np.int64))  # fmt: skip


def _ref_unit_waveform(d: Disturbance, n0: int, n: int, fs: float) -> NDArray[np.complex128]:
    """Frozen copy of unit_waveform before the speed-up: every block times its envelope."""
    make = F._BLOCKS[d.kind]
    return F._on_grid(n0, n, lambda b: make(d, b, fs) * F._envelope(F._times(b, fs), d.onset_s, d.ramp_s),
                      np.complex128)  # type: ignore[return-value]  # fmt: skip


@pytest.mark.parametrize("kind", ["noise", "cw", "swept_cw", "ghost"])
def test_unit_waveform_skips_the_envelope_where_it_is_constant_with_the_same_values(kind: str) -> None:
    """Blocks wholly before the onset are zeros without drawing the waveform, and blocks wholly past the ramp are the
    waveform itself; the blocks around the onset and its ramp (1.0 to 1.01 s at 1 MS/s, which spans a block edge)
    keep the product. Equal values over 0 to 2 s, starting off the block grid."""
    params = {"noise": {"bw_hz": 2e5}, "cw": {"f_hz": 1234.0}, "swept_cw": {"span_hz": 1e5, "period_s": 0.01},
              "ghost": {"prn": 5, "doppler_hz": 1500.0}}[kind]  # fmt: skip
    for onset, ramp in ((1.0, 0.01), (1.0, 0.0), (0.0, 0.0), (65536 / 1e6, 0.01)):
        d = Disturbance(kind, params, onset, ramp, 60.0, 4)  # type: ignore[arg-type]
        np.testing.assert_array_equal(unit_waveform(d, 777, 2_000_000, 1e6), _ref_unit_waveform(d, 777, 2_000_000, 1e6))


@pytest.mark.parametrize(("decim", "taps"), [(4, 129), (2, 65), (4, 1)])
def test_decimation_on_real_and_imaginary_parts_is_the_complex_filter(decim: int, taps: int) -> None:
    """scipy casts real taps to complex for a complex input; filtering the two parts with the real taps forms the same
    products in the same order (taps 1: the plain decimation)."""
    h = np.asarray(signal.firwin(taps, 0.9 / decim), dtype=np.float64) if taps > 1 else np.ones(1)
    rng = np.random.default_rng(3)
    w = rng.standard_normal(RNG_BLOCK + taps - 1) + 1j * rng.standard_normal(RNG_BLOCK + taps - 1)
    np.testing.assert_array_equal(_fir_decimate(h, w, decim), signal.upfirdn(h, w, up=1, down=decim))
