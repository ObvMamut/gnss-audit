"""Ghost truth (docs/3 section 3.2) and G-phase-precision (docs/3 section 2.6)."""

import numpy as np
import pyarrow as pa
import pytest

from gnssaudit.inject import GhostSpec, ghost_phase_rad
from gnssaudit.io.schemas import TRACK_STREAM, build
from gnssaudit.truth.ghost import NoAmbiguityWindow, PhasePrecisionError, ghost_truth, phase_precision_ok
from gnssaudit.truth.rxtime import TwinClock

FS = 4.0e6
CLOCK = TwinClock.identity(FS)
G = GhostSpec(prn=31, doppler_hz=1200.0, doppler_rate_hzps=-0.5, cn0_db_hz=45.0, chip0=3.0, seed=4)


def track(acc_rad: np.ndarray | None, stamps: np.ndarray) -> pa.Table:
    n = stamps.size
    f32 = None if acc_rad is not None else np.zeros(n, dtype=np.float32)
    return build(TRACK_STREAM, {
        "run_id": ["r"] * n, "channel": np.zeros(n), "sys": ["G"] * n, "prn": np.full(n, 31), "signal": ["1C"] * n,
        "k": np.arange(n), "sample_count": stamps, "t_rx_s": stamps / FS, "abs_ve": None, "abs_e": None, "abs_p": None,
        "abs_l": None, "abs_vl": None, "prompt_i": None, "prompt_q": None, "acc_phase_rad_f32": f32,
        "acc_phase_rad": acc_rad, "dop_hz": None, "dop_rate_hzps": None, "code_freq_chips": None,
        "code_rate_chipsps2": None, "carr_err": None, "carr_err_filt": None, "code_err_chips": None,
        "code_err_filt": None, "cn0_dbhz": None, "lock_test": None, "tow_ms": None, "wn": None,
        "t_int_s": np.full(n, 1e-3),
    }, n)  # fmt: skip


STAMPS = np.arange(4_000, int(60 * FS), 4_000, dtype=np.uint64)  # one record per ms for 60 s


def test_ghost_truth_follows_the_sign_convention_and_removes_the_ambiguity() -> None:
    """GNSS-SDR accumulates with `-=`, so a perfect track reads -theta_g plus a constant (here 0.3 cycle)."""
    theta = ghost_phase_rad(G, STAMPS.astype(np.int64), FS)
    t = STAMPS / FS
    e_true = np.where(t > 30.0, 0.01 * np.sin(2 * np.pi * 0.5 * t), 0.0)  # cycles, after the ambiguity window
    truth = ghost_truth(
        track(-(theta + 2 * np.pi * (0.3 + e_true)), STAMPS), G, clock=CLOCK, converge_s=5.0, truth_id="g"
    )
    valid = truth.column("valid").to_numpy()
    assert set(truth.column("kind").to_pylist()) == {"ghost"}
    assert not valid[t <= t[0] + 5.0].any() and valid[t > t[0] + 5.0].all()
    np.testing.assert_allclose(truth.column("e_cyc").to_numpy()[valid], e_true[valid], atol=1e-9)


def restarted(acc: np.ndarray, stamps: np.ndarray, restart_s: float, reset_s: float) -> tuple[np.ndarray, np.ndarray]:
    """What GNSS-SDR's dump shows when a channel loses lock: no records for 133 ms while it re-acquires, the phase
    accumulated from zero on restart, and set again (to -rem_carr_phase) when narrow tracking begins after bit sync
    (`check_carrier_phase_coherent_initialization`, dll_pll_veml_tracking.cc:1536)."""
    t = stamps / FS
    keep = (t <= restart_s) | (t > restart_s + 0.133)
    acc, stamps, t = acc[keep].copy(), stamps[keep], t[keep]
    first = int(np.flatnonzero(t > restart_s)[0])
    acc[first:] -= acc[first]
    at = int(np.flatnonzero(t >= reset_s)[0])
    acc[at:] += -0.7 - acc[at]
    return acc, stamps


def test_a_phase_reset_after_the_ambiguity_window_ends_the_valid_truth() -> None:
    theta = ghost_phase_rad(G, STAMPS.astype(np.int64), FS)
    acc, stamps = restarted(-(theta + 2 * np.pi * 0.3), STAMPS, restart_s=20.0, reset_s=26.5)
    truth = ghost_truth(track(acc, stamps), G, clock=CLOCK, converge_s=5.0, truth_id="g")
    t, valid = stamps / FS, truth.column("valid").to_numpy()
    assert np.array_equal(valid, (t > t[0] + 5.0) & (t <= 20.0))
    np.testing.assert_allclose(truth.column("e_cyc").to_numpy()[valid], 0.0, atol=1e-9)
    arcs = truth.column("arc_id").to_numpy()
    assert np.array_equal(np.unique(arcs), [0, 1, 2]) and arcs[t > 26.5].min() == 2


def test_the_ambiguity_is_taken_inside_one_arc() -> None:
    """Found in the CTTC E0b run: the ghost lost lock at 11.1 s and reached narrow tracking again at 17.6 s, inside the
    window after convergence; one constant across both resets gave an error SD of 2.4e5 times the thermal one."""
    theta = ghost_phase_rad(G, STAMPS.astype(np.int64), FS)
    t_all = STAMPS / FS
    e_true = np.where(t_all > 30.0, 0.01 * np.sin(2 * np.pi * 0.5 * t_all), 0.0)
    acc, stamps = restarted(-(theta + 2 * np.pi * (0.3 + e_true)), STAMPS, restart_s=8.0, reset_s=14.5)
    truth = ghost_truth(track(acc, stamps), G, clock=CLOCK, converge_s=5.0, truth_id="g")
    t, valid = stamps / FS, truth.column("valid").to_numpy()
    assert np.array_equal(valid, t >= 14.5)  # the restart at 8.13 s converged by 13.13 s
    keep = (t_all <= 8.0) | (t_all > 8.133)
    np.testing.assert_allclose(truth.column("e_cyc").to_numpy()[valid], e_true[keep][valid], atol=1e-9)


def test_a_track_without_an_ambiguity_window_raises_its_own_error() -> None:
    """P2c task 3: check C-1 records a jammed track whose arcs are all shorter than convergence plus the 10 s window,
    and only that, as not computed; so the case has its own ValueError, not one a broad catch would have to match."""
    theta = ghost_phase_rad(G, STAMPS.astype(np.int64), FS)
    short = STAMPS[:12_000]  # 12 s of records: no arc reaches 5 s of convergence plus the 10 s window
    with pytest.raises(NoAmbiguityWindow, match="no arc holds a full ambiguity window") as err:
        ghost_truth(track(-theta[:12_000], short), G, clock=CLOCK, converge_s=5.0, truth_id="g")
    assert isinstance(err.value, ValueError) and not isinstance(err.value, PhasePrecisionError)


def test_ghost_truth_needs_the_float64_phase_of_the_patched_build() -> None:
    with pytest.raises(PhasePrecisionError, match="float64"):
        ghost_truth(track(None, STAMPS), G, clock=CLOCK, converge_s=5.0, truth_id="g")


def test_phase_precision_gate() -> None:
    """At 3.8e5 rad float32 resolves 0.031 rad, far above 1e-3 of a 0.012 rad thermal sigma; float64 does not."""
    assert not phase_precision_ok(3.8e5, np.dtype(np.float32), sigma_th=0.012)
    assert phase_precision_ok(3.8e5, np.dtype(np.float64), sigma_th=0.012)
    assert phase_precision_ok(10.0, np.dtype(np.float32), sigma_th=0.012)  # float32 is fine near zero
