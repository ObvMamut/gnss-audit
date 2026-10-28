"""Route B twin truth (docs/3 sections 2.1 to 2.3): offset removal, slips, transients and exclusions."""

import numpy as np
import pyarrow as pa
import pytest

from gnssaudit.io.schemas import OBS_EPOCH_SAT, build
from gnssaudit.truth.twin import TwinAlignmentError, twin_difference

STEP = 0.02  # 50 Hz observable epochs
T = np.round(np.arange(100.0, 160.0, STEP), 6)  # 60 s of receiver time
ONSET = 130.0
SATS = [("G", 5), ("G", 12)]


def obs(cp: dict[int, np.ndarray], t: np.ndarray = T, receiver: str = "c") -> pa.Table:
    rows = [(prn, ti, c) for prn, series in cp.items() for ti, c in zip(t, series, strict=True)]
    n = len(rows)
    return build(OBS_EPOCH_SAT, {
        "source_id": ["s"] * n, "receiver_id": [receiver] * n, "t_gps_s": [r[1] for r in rows], "sys": ["G"] * n,
        "prn": [r[0] for r in rows], "signal": ["1C"] * n, "pr_m": np.full(n, 2.1e7), "cp_cyc": [r[2] for r in rows],
        "dop_hz": np.full(n, 1000.0), "cn0_dbhz": None, "lock_time_s": None, "lli": None, "trk_stat": None,
        "cp_std_code": None, "cp_std_cyc": None, "censored": None, "pr_std_m": None, "elev_deg": None,
        "azim_deg": None, "valid": np.ones(n, dtype=bool),
    }, n)  # fmt: skip


def base(prn: int) -> np.ndarray:
    return 1.5e6 + prn * 1e4 + 1000.0 * (T - T[0])  # the shared true phase: large and moving, in cycles


def error_after_onset() -> np.ndarray:
    return np.where(T >= ONSET, 0.04 * np.sin(2 * np.pi * 0.3 * (T - ONSET)), 0.0)


def run(clean: pa.Table, jammed: pa.Table, warmup_s: float = 100.0) -> tuple[pa.Table, pa.Table, pa.Table]:
    return twin_difference(clean, jammed, onset_s=ONSET, pre_window_s=10.0, warmup_s=warmup_s, slip_size_cyc=0.5,
                           tau_loop_s=0.1, truth_id="tw")  # fmt: skip


def totals(excl: pa.Table) -> dict[str, int]:
    """Exclusion counts summed over satellites, by reason."""
    out: dict[str, int] = {}
    for reason, count in zip(excl.column("reason_code").to_pylist(), excl.column("count").to_pylist(), strict=True):
        out[reason] = out.get(reason, 0) + count
    return out


def series(truth: pa.Table, prn: int, col: str = "e_cyc") -> np.ndarray:
    m = truth.column("prn").to_numpy() == prn
    order = np.argsort(truth.column("t_rx_s").to_numpy()[m])
    return truth.column(col).to_numpy(zero_copy_only=False)[m][order]


def test_offset_slips_and_transients_are_recovered() -> None:
    e = error_after_onset()
    slips = np.where(T >= 140.0, 0.5, 0.0) + np.where(T >= 150.0, -0.5, 0.0) + np.where(T >= 155.0, 1.0, 0.0)
    clean = obs({p: base(p) for _, p in SATS})
    jammed = obs({p: base(p) + 37.25 + e + (slips if p == 5 else 0.0) for _, p in SATS}, receiver="j")
    truth, events, excl = run(clean, jammed)
    assert set(truth.column("kind").to_pylist()) == {"twin"} and excl.num_rows == 0
    for _, p in SATS:
        np.testing.assert_allclose(series(truth, p), e, atol=1e-8)  # the 37.25-cycle start offset is gone (F24)
    starts, sizes = events.column("t_start_s").to_pylist(), events.column("size_cyc").to_pylist()
    assert sorted(zip(starts, sizes, strict=True)) == [
        (140.0, 0.5), (150.0, -0.5), (155.0, 1.0),
    ]  # fmt: skip
    assert series(truth, 5, "slip_count")[-1] == 3 and series(truth, 12, "slip_count")[-1] == 0
    transient = series(truth, 5, "transient")
    near = np.min(np.abs(T[:, None] - np.array([140.0, 150.0, 155.0])[None, :]), axis=1) <= 0.3 + 1e-9
    assert np.array_equal(transient, near) and not series(truth, 12, "transient").any()


def test_a_constant_added_to_one_twin_changes_nothing() -> None:
    e = error_after_onset()
    clean = obs({p: base(p) for _, p in SATS})
    jammed = obs({p: base(p) + e + np.where(T >= 141.0, 0.5, 0.0) for _, p in SATS}, receiver="j")
    shifted = obs({p: base(p) + e + np.where(T >= 141.0, 0.5, 0.0) + 123.75 for _, p in SATS}, receiver="j")
    (a, ea, xa), (b, eb, xb) = run(clean, jammed), run(clean, shifted)
    np.testing.assert_allclose(a.column("e_cyc").to_numpy(), b.column("e_cyc").to_numpy(), atol=1e-8)
    assert ea.column("t_start_s").to_pylist() == eb.column("t_start_s").to_pylist() and xa.num_rows == xb.num_rows == 0


def test_a_step_that_is_not_a_slip_excludes_the_rest_of_the_series() -> None:
    clean = obs({p: base(p) for _, p in SATS})
    jammed = obs({p: base(p) + (np.where(T >= 145.0, 0.37, 0.0) if p == 5 else 0.0) for _, p in SATS}, receiver="j")
    truth, events, excl = run(clean, jammed)
    assert events.num_rows == 0
    assert series(truth, 5).size == np.sum(T < 145.0 - STEP * 2)  # nothing kept at or after the ambiguous step
    assert set(totals(excl)) == {"ambiguous_slip", "after_ambiguous_step"}
    assert series(truth, 12).size == T.size


def test_a_slip_in_noise_too_large_for_the_margin_is_ambiguous() -> None:
    rng = np.random.default_rng(4)
    clean = obs({p: base(p) for _, p in SATS})
    noisy = {p: base(p) + rng.normal(0.0, 0.08, T.size) + np.where(T >= 145.0, 0.5, 0.0) for _, p in SATS}
    _, events, excl = run(clean, obs(noisy, receiver="j"))
    assert events.num_rows == 0 and "ambiguous_slip" in excl.column("reason_code").to_pylist()


def test_epochs_in_one_twin_only_are_counted_not_dropped() -> None:
    keep = np.ones(T.size, dtype=bool)
    keep[np.arange(200, 2700, 100)] = False  # 25 epochs, 0.8 %
    clean = obs({p: base(p) for _, p in SATS})
    jammed = obs({p: base(p)[keep] for _, p in SATS}, t=T[keep], receiver="j")
    truth, _, excl = run(clean, jammed)
    assert totals(excl) == {"one_twin_epoch": 50} and truth.num_rows == 2 * (T.size - 25)  # 25 per satellite


def test_twins_that_barely_overlap_are_refused() -> None:
    keep = np.arange(T.size) % 10 != 0  # 10 % of the clean epochs missing from the jammed twin
    clean = obs({p: base(p) for _, p in SATS})
    jammed = obs({p: base(p)[keep] for _, p in SATS}, t=T[keep], receiver="j")
    with pytest.raises(TwinAlignmentError, match="90.0 %"):
        run(clean, jammed)


def test_a_receiver_time_break_in_one_twin_excludes_what_follows() -> None:
    """A receiver-time reset can repeat or step back the epoch labels; nothing after it is paired (docs/9 3.5)."""
    t_j = T.copy()
    t_j[T >= 145.0] -= STEP  # labels step back once: 144.98 appears twice
    clean = obs({p: base(p) for _, p in SATS})
    jammed = obs({p: base(p) for _, p in SATS}, t=np.round(t_j, 6), receiver="j")
    truth, _, excl = run(clean, jammed)
    assert totals(excl)["rx_time_break"] == 2 * np.sum(T >= 145.0)  # the repeated label and everything after it
    assert truth.column("t_rx_s").to_numpy().max() == pytest.approx(145.0 - STEP)  # the last epoch before the reset


def test_two_slips_a_few_epochs_apart_are_both_corrected() -> None:
    """Rapid slips under strong jamming: the second one, three epochs after the first, must not be skipped."""
    e = error_after_onset()
    slips = np.where(T >= 140.0, 0.5, 0.0) + np.where(T >= 140.0 + 3 * STEP, 0.5, 0.0)
    clean = obs({p: base(p) for _, p in SATS})
    jammed = obs({p: base(p) + e + slips for _, p in SATS}, receiver="j")
    truth, events, excl = run(clean, jammed)
    assert excl.num_rows == 0
    assert sorted(events.column("t_start_s").to_pylist()) == [140.0, 140.0, 140.06, 140.06]  # two satellites
    for _, p in SATS:
        np.testing.assert_allclose(series(truth, p), e, atol=1e-8)


def test_every_paired_epoch_is_either_kept_or_excluded() -> None:
    """No silent drops (docs/6 section 8.4): kept truth rows plus excluded epochs account for every paired epoch."""
    clean = obs({p: base(p) for _, p in SATS})
    jammed = obs({p: base(p) + (np.where(T >= 145.0, 0.37, 0.0) if p == 5 else 0.0) for _, p in SATS}, receiver="j")
    truth, _, excl = run(clean, jammed)
    assert truth.num_rows + sum(excl.column("count").to_pylist()) == 2 * T.size


def test_tracking_labels_with_code_doppler_jitter_pair_fully() -> None:
    """Twins differenced on tracking records (labels = stamp / fs) step by 4000 +-1 samples: that jitter is not a
    receiver-time break (found in the first E0b run, where it excluded every epoch)."""
    rng = np.random.default_rng(0)
    stamps = 4_000_000 + np.cumsum(4000 + rng.integers(-1, 2, 60_000))
    t = stamps / 4.0e6
    cp = 1e5 + 1000.0 * t
    clean = obs({3: cp}, t=t)
    jammed = obs({3: cp + 3.25}, t=t, receiver="j")
    truth, _, excl = twin_difference(clean, jammed, onset_s=40.0, pre_window_s=10.0, warmup_s=1.0, slip_size_cyc=0.5,
                                     tau_loop_s=0.03, truth_id="trk")  # fmt: skip
    assert excl.num_rows == 0 and truth.num_rows == t.size
    np.testing.assert_allclose(truth.column("e_cyc").to_numpy(), 0.0, atol=1e-8)


def test_the_pull_in_before_warm_up_is_neither_paired_nor_a_slip() -> None:
    """Each twin pulls in from its own tracking start (F24): before warm-up their phases may differ by any amount.
    Found in the second E0b run, where a 6.5-cycle pull-in step was taken for 13 slips and offset the whole series."""
    pull_in = np.where(T < 105.0, 6.5 * np.exp(-(T - 100.0)), 0.0) + np.where(T < 101.0, 6.5, 0.0)
    clean = obs({p: base(p) for _, p in SATS})
    jammed = obs({p: base(p) + 3.0 + pull_in for _, p in SATS}, receiver="j")
    truth, events, excl = run(clean, jammed, warmup_s=110.0)
    assert events.num_rows == 0
    assert totals(excl) == {"warmup": 2 * np.sum(T < 110.0)}
    np.testing.assert_allclose(truth.column("e_cyc").to_numpy(), 0.0, atol=1e-8)


def test_the_pre_onset_window_is_centred_after_slip_correction() -> None:
    """A slip after warm-up but before the pre-onset window must not shift the reference level."""
    e = error_after_onset()
    clean = obs({p: base(p) for _, p in SATS})
    jammed = obs({p: base(p) + e + np.where(T >= 112.0, 0.5, 0.0) for _, p in SATS}, receiver="j")
    truth, events, _ = run(clean, jammed, warmup_s=105.0)
    assert sorted(set(events.column("t_start_s").to_pylist())) == [112.0]
    for _, p in SATS:
        np.testing.assert_allclose(series(truth, p), e[T >= 105.0], atol=1e-8)


@pytest.mark.parametrize("ramp_ms", [5, 30])
def test_a_ramped_slip_is_found_at_the_tracking_record_rate(ramp_ms: int) -> None:
    """Review finding (p2a, critical 2): at 1 kHz a real PLL slip takes tens of milliseconds, so the per-epoch step of
    a 5-epoch running median never reaches 0.25 cycle. Slips are searched on a 50 Hz grid of block medians instead."""
    fs_rec = 1000.0
    t = np.round(np.arange(100.0, 160.0, 1 / fs_rec), 6)
    rng = np.random.default_rng(4)
    rho = np.exp(-1 / 14.0)  # 14 ms correlation, like a 35 Hz loop's error
    w = rng.standard_normal(t.size) * 0.03 * np.sqrt(1 - rho**2)
    noise = np.zeros(t.size)
    for i in range(1, t.size):
        noise[i] = rho * noise[i - 1] + w[i]
    slip = 0.5 * np.clip((t - 145.0) / (ramp_ms / 1000), 0.0, 1.0)
    cp = 1.5e6 + 1000.0 * (t - t[0])
    clean = obs({5: cp}, t=t)
    jammed = obs({5: cp + np.where(t >= ONSET, noise, 0.0) + slip}, t=t, receiver="j")
    truth, events, excl = run(clean, jammed)
    assert events.num_rows == 1 and events.column("size_cyc")[0].as_py() == pytest.approx(0.5)
    e = series(truth, 5)
    tt = np.sort(truth.column("t_rx_s").to_numpy())
    assert abs(float(np.mean(e[tt > 145.2]))) < 0.02  # corrected: no half cycle left in the truth
    assert "ambiguous_slip" not in totals(excl)


@pytest.mark.parametrize("origin", [1234.43, 1234.05, -987.58])
def test_a_reacquisition_ends_the_series_instead_of_passing_as_a_slip(origin: float) -> None:
    """Review finding (p2a, important 1): GNSS-SDR restarts the accumulated phase when it re-acquires. A gap in one
    twin's records followed by a new phase origin ends the series ('reacquisition'); it is never taken as a slip."""
    gone = (T >= 140.0) & (T < 142.0)
    t_j = T[~gone]
    cp = {5: base(5)}
    jammed = obs({5: np.where(t_j >= 142.0, base(5)[~gone] + origin, base(5)[~gone])}, t=t_j, receiver="j")
    truth, events, excl = run(obs(cp), jammed)
    assert events.num_rows == 0
    assert totals(excl)["reacquisition"] == int(np.sum(T >= 142.0))
    assert truth.column("t_rx_s").to_numpy().max() < 140.0


def test_a_step_of_many_cycles_is_not_a_slip() -> None:
    """A slip is at most MAX_SLIP_CYC; a larger step (no gap) is ambiguous, not a correction by a large integer."""
    jammed = obs({5: base(5) + np.where(T >= 145.0, 7.5, 0.0)}, receiver="j")
    truth, events, excl = run(obs({5: base(5)}), jammed)
    assert events.num_rows == 0 and totals(excl)["after_ambiguous_step"] > 0


def test_losses_after_the_onset_do_not_count_against_the_alignment() -> None:
    """Review finding (p2a, important 2): the 95 % overlap rule tests the epoch grid, before the onset. Epochs the
    jammed twin loses under jamming are exclusions, not a misaligned grid."""
    gone = (T >= 135.0) & (T < 150.0)
    jammed = obs({5: base(5)[~gone]}, t=T[~gone], receiver="j")
    truth, _, excl = run(obs({5: base(5)}), jammed)
    assert truth.column("t_rx_s").to_numpy().max() < 135.0
    assert totals(excl)["reacquisition"] == int(np.sum(T >= 150.0))


def test_tracking_labels_one_sample_apart_pair_with_the_phase_rate() -> None:
    """Twins' integration boundaries can differ by a sample (code-loop jitter). Records closer than a quarter step
    pair, and the jammed phase is carried to the clean label with its own rate (1000 Hz x 0.25 us = 2.5e-4 cycle)."""
    rng = np.random.default_rng(1)
    fs = 4.0e6
    stamps = 4_000_000 + np.cumsum(np.full(60_000, 4000))
    shift = rng.integers(0, 2, stamps.size)  # one sample later in half of the jammed records
    t_c, t_j = stamps / fs, (stamps + shift) / fs
    clean = obs({3: 1e5 + 1000.0 * t_c}, t=t_c)
    jammed = obs({3: 1e5 + 1000.0 * t_j}, t=t_j, receiver="j")
    truth, _, excl = twin_difference(clean, jammed, onset_s=40.0, pre_window_s=10.0, warmup_s=1.0, slip_size_cyc=0.5,
                                     tau_loop_s=0.03, truth_id="trk")  # fmt: skip
    assert excl.num_rows == 0 and truth.num_rows == stamps.size
    np.testing.assert_allclose(truth.column("e_cyc").to_numpy(), 0.0, atol=1e-6)


def test_records_that_switch_to_20_ms_after_bit_sync_are_not_read_as_reacquisitions() -> None:
    """P2b task 10 (E0 FGI-A, 20 ms cells): GNSS-SDR logs 1 ms records until bit sync, then one per 20 ms
    integration. The epoch step is the spacing after the warm-up; taken over the whole record (1 ms), every 20 ms
    gap read as a re-acquisition and the whole truth was excluded."""
    early = np.round(np.arange(100.0, 108.0, 0.001), 6)  # 8 s at 1 ms before bit sync
    late = np.round(np.arange(108.0, 160.0, 0.02), 6)
    t = np.concatenate([early, late])
    phase = 1.5e6 + 1000.0 * (t - t[0])
    err = np.where(t >= ONSET, 0.04 * np.sin(2 * np.pi * 0.3 * (t - ONSET)), 0.0)
    truth, _, excl = twin_difference(obs({5: phase}, t), obs({5: phase + err}, t, "j"), onset_s=ONSET,
                                     pre_window_s=10.0, warmup_s=115.0, slip_size_cyc=0.5, tau_loop_s=0.1,
                                     truth_id="tw")  # fmt: skip
    reasons = dict(zip(excl.column("reason_code").to_pylist(), excl.column("count").to_pylist(), strict=True))
    assert "reacquisition" not in reasons
    tt = truth.column("t_rx_s").to_numpy()
    assert truth.num_rows == np.sum(late >= 115.0)
    np.testing.assert_allclose(truth.column("e_cyc").to_numpy(), np.interp(tt, t, err), atol=1e-9)
