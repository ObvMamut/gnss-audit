"""Observable epochs on the recording's sample axis (P2b task 9; docs/9 section 10): GNSS-SDR's observables block
interpolates the tracking phase at the samples of a sample-counter clock, epoch i at (i + k0) x step. k0 is pinned by
matching the observables phase to the float64 tracking phase, up to a constant (and a sign)."""

import numpy as np
import pytest

from gnssaudit.truth.rxtime import ObsClock, RxTimeError, epoch_samples, match_clock, onset_epoch

FS = 4.0e6
STEP = 80_000  # 20 ms at 4 MS/s


def tracking(seconds: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """Stamps every 1 ms and the reported phase (cycles) of a carrier whose Doppler drifts by 30 Hz."""
    stamps = np.arange(4000, int(seconds * FS), 4000, dtype=np.int64)
    t = stamps / FS
    phase = 1500.0 * t + 0.25 * t**2 + 0.3
    return stamps, phase


def observables(stamps: np.ndarray, phase: np.ndarray, k0: int, n: int, sign: float = 1.0, c: float = 17.5):
    idx = np.arange(n)
    return idx, sign * np.interp((idx + k0) * STEP, stamps, phase) + c


def test_epoch_samples_follow_the_clock() -> None:
    np.testing.assert_array_equal(
        epoch_samples(np.array([0, 1, 5]), ObsClock(k0=3, step=STEP)), [240_000, 320_000, 640_000]
    )


@pytest.mark.parametrize(("k0", "sign"), [(1, 1.0), (7, 1.0), (12, -1.0)])
def test_the_clock_offset_is_recovered_from_the_phase(k0: int, sign: float) -> None:
    stamps, phase = tracking()
    idx, obs = observables(stamps, phase, k0, 2500, sign)
    clock = match_clock(idx, obs, stamps, phase, step=STEP)
    assert clock.k0 == k0 and clock.sign == sign and clock.residual_cyc < 1e-6


def test_no_clock_that_fits_raises() -> None:
    stamps, phase = tracking()
    idx, obs = observables(stamps, phase, 5, 2500)
    obs = obs + 0.3 * np.sin(np.arange(obs.size) / 7.0)  # not the tracking phase at any shift
    with pytest.raises(RxTimeError, match="no clock offset"):
        match_clock(idx, obs, stamps, phase, step=STEP)


def test_the_onset_lands_on_the_first_epoch_at_or_after_it() -> None:
    clock = ObsClock(k0=1, step=STEP)
    assert onset_epoch(55 * 4_000_000, clock) == 55 * 50 - 1  # epoch i sits at (i + 1) x 80 000
    assert onset_epoch(55 * 4_000_000 + 1, clock) == 55 * 50


def test_a_residual_of_a_few_samples_is_accepted_when_the_offset_stands_out() -> None:
    """On the binary the observables are interpolated at the record's code-phase time, a few samples from its stamp:
    0.001 to 0.003 cycle at 5 to 10 kHz Doppler, against 0.13 cycle at the neighbouring offsets (CTTC, P2b task 9)."""
    stamps, phase = tracking()
    idx, obs = observables(stamps, phase, 1, 2500, -1.0)
    obs = obs + 0.004 * np.sin(np.arange(obs.size) / 3.0)
    clock = match_clock(idx, obs, stamps, phase, step=STEP)
    assert clock.k0 == 1 and clock.sign == -1.0 and clock.residual_cyc < 0.01
