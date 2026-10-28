"""P-inject-lin and P-inject-det (V5): injection is linear in its gain and independent of chunking (docs/8 6)."""

import numpy as np
import pytest

from gnssaudit.inject import Disturbance, disturbance_seed, gain, phase_process, render, unit_waveform

FS = 8.0e6
SWEEP = {"f_hz": 0.0, "span_hz": 2.0e6, "period_s": 1e-4}
CHIRP = {"f_hz": 0.0, "span_hz": 6.0e6, "period_s": 8e-6}
GHOST = {"prn": 21, "doppler_hz": 900.0, "doppler_rate_hzps": -0.5, "chip0": 3.0}
FAMILIES = [
    Disturbance("noise", {"bw_hz": 1.0e6, "f_hz": 5.0e5}, 0.01, 0.002, 60.0, disturbance_seed(1, 2, 3, 0)),
    Disturbance("cw", {"f_hz": 3.3e5}, 0.0, 0.0, 60.0, disturbance_seed(1, 2, 3, 1)),
    Disturbance("swept_cw", SWEEP, 0.0, 0.0, 60.0, disturbance_seed(1, 2, 3, 2)),
    Disturbance("chirp", CHIRP, 0.0, 0.0, 60.0, disturbance_seed(1, 2, 3, 3)),
    Disturbance("ghost", GHOST, 0.0, 0.0, 45.0, disturbance_seed(1, 2, 3, 4)),
]
N0 = 1e-4


@pytest.mark.parametrize("d", FAMILIES, ids=lambda d: d.kind)
def test_output_does_not_depend_on_chunking(d: Disturbance) -> None:
    n0, n = 12_345, 1 << 22
    whole = render(d, n0, n, FS, N0)
    for c in (1 << 18, 1 << 20, 1 << 22):
        parts = np.concatenate([render(d, n0 + s, min(c, n - s), FS, N0) for s in range(0, n, c)])
        assert np.array_equal(parts, whole), c


@pytest.mark.parametrize("d", FAMILIES, ids=lambda d: d.kind)
def test_injected_component_scales_exactly_with_the_gain(d: Disturbance) -> None:
    unit = unit_waveform(d, 777, 1 << 16, FS)
    assert np.array_equal(render(d, 777, 1 << 16, FS, N0), np.float64(gain(d.level_db_hz, N0)) * unit)


def test_seeds_separate_realizations_and_repeat_exactly() -> None:
    a = Disturbance("noise", {"bw_hz": 1.0e6, "f_hz": 0.0}, 0.0, 0.0, 60.0, disturbance_seed(1, 0, 0, 0))
    b = Disturbance("noise", {"bw_hz": 1.0e6, "f_hz": 0.0}, 0.0, 0.0, 60.0, disturbance_seed(1, 0, 1, 0))
    xa, xb = unit_waveform(a, 0, 1 << 17, FS), unit_waveform(b, 0, 1 << 17, FS)
    assert np.array_equal(xa, unit_waveform(a, 0, 1 << 17, FS))
    assert abs(np.vdot(xa, xb)) / xa.size < 0.02  # independent draws


def test_phase_process_does_not_depend_on_chunking() -> None:
    d = Disturbance("phase", {"n_tones": 12, "f_min_hz": 0.05, "f_max_hz": 50.0, "a_rad": 0.05}, 0.0, 0.0, 0.0, 9)
    whole = phase_process(d, 5, 1 << 20, FS)
    parts = np.concatenate([phase_process(d, 5 + s, 1 << 18, FS) for s in range(0, 1 << 20, 1 << 18)])
    assert np.array_equal(parts, whole)


def test_a_zero_gain_component_is_exact_zeros_and_costs_nothing() -> None:
    """D-10: the clean twin's interference has gain zero; it must add exact zeros without generating the waveform."""
    import time

    d = Disturbance("noise", {"bw_hz": 1.0e6, "f_hz": 0.0}, 0.0, 0.0, -np.inf, disturbance_seed(1, 0, 0, 0))
    t0 = time.perf_counter()
    z = render(d, 0, 1 << 22, FS, N0)
    assert not np.any(z) and z.dtype == np.complex128 and z.size == 1 << 22
    assert time.perf_counter() - t0 < 0.5  # generating this much band-limited noise takes seconds
