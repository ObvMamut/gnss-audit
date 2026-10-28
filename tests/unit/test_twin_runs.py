"""Twin-run machinery shared by E0b and E2 (docs/3 section 2; docs/6 section 4.2; D-34)."""

import gzip
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from gnssaudit.experiments import twin
from gnssaudit.experiments.twin import (
    Twin,
    ghost_sanity,
    keep_logs,
    provenance,
    receiver_run_count,
    receiver_spec,
    runs_of,
    sigma_disc_cyc,
    sigma_reported_cyc,
    throughput,
    track_as_obs,
    unlocked_channels,
)
from gnssaudit.inject import FrontEndSpec, InjectionReport, IQSource
from gnssaudit.io.schemas import TRACK_STREAM, build
from gnssaudit.sdr import RunArtifacts
from gnssaudit.truth.rxtime import TwinClock

TWIN_1C = {"Tracking_1C.max_carrier_lock_fail": "1000000", "Tracking_1C.enable_fll_pull_in": "true"}  # D-34
CLOCK = TwinClock.identity(4e6)


def test_thermal_sigmas() -> None:
    """Reported phase: B3 on B_n,eff (with squaring loss); discriminator: 1 / sqrt(2 C/N0 T) per integration."""
    cn0 = 45.0
    want_rep = np.sqrt(30.0 / 10**4.5 * (1 + 1 / (2 * 1e-3 * 10**4.5))) / (2 * np.pi)
    assert sigma_reported_cyc(cn0, bn_eff_hz=30.0, t_s=1e-3) == pytest.approx(want_rep, rel=1e-9)
    assert sigma_disc_cyc(cn0, t_s=1e-3) == pytest.approx(1 / np.sqrt(2 * 10**4.5 * 1e-3) / (2 * np.pi), rel=1e-9)


def test_track_as_obs_turns_the_patched_phase_into_cycles() -> None:
    n = 4
    none = dict.fromkeys(
        [
            "abs_ve",
            "abs_e",
            "abs_p",
            "abs_l",
            "abs_vl",
            "prompt_i",
            "prompt_q",
            "acc_phase_rad_f32",
            "dop_rate_hzps",
            "code_freq_chips",
            "code_rate_chipsps2",
            "carr_err",
            "carr_err_filt",
            "code_err_chips",
            "code_err_filt",
            "lock_test",
            "tow_ms",
            "wn",
        ]
    )
    track = build(
        TRACK_STREAM,
        {
            **none,
            "run_id": ["r"] * n,
            "channel": np.zeros(n),
            "sys": ["G"] * n,
            "prn": np.full(n, 3),
            "signal": ["1C"] * n,
            "k": np.arange(n),
            "sample_count": np.arange(1, n + 1, dtype=np.uint64) * 4000,
            "t_rx_s": np.arange(1, n + 1) * 1e-3,
            "acc_phase_rad": -2 * np.pi * np.array([10.0, 10.25, 10.5, 10.75]),
            "dop_hz": np.full(n, 1000.0),
            "cn0_dbhz": np.full(n, 45.0),
            "t_int_s": np.full(n, 1e-3),
        },
        n,
    )
    obs = track_as_obs(track, source_id="syn", receiver_id="clean")
    assert isinstance(obs, pa.Table)
    assert obs.column("cp_cyc").to_pylist() == [10.0, 10.25, 10.5, 10.75]  # -acc / 2 pi (GNSS-SDR's `-=`)
    assert obs.column("t_gps_s").to_pylist() == pytest.approx([1e-3, 2e-3, 3e-3, 4e-3])
    assert all(obs.column("valid").to_pylist()) and obs.column("cn0_dbhz").to_pylist() == [45.0] * 4


def test_every_receiver_log_is_kept_beside_the_results(tmp_path: Path) -> None:
    """The work directory, dumps included, is deleted after reduction (D-18); each run's GNSS-SDR log is not."""
    work, run_dir = tmp_path / "work", tmp_path / "run"
    for name, text in (("zero0", "Tracking of GPS L1 C/A signal started\n"), ("survey", "Loss of lock in channel 5\n")):
        (work / name).mkdir(parents=True)
        (work / name / "gnss-sdr.log").write_text(text)
    run_dir.mkdir()
    assert keep_logs(work, run_dir) == {"survey": "logs/survey.log.gz", "zero0": "logs/zero0.log.gz"}
    assert gzip.decompress((run_dir / "logs" / "survey.log.gz").read_bytes()) == b"Loss of lock in channel 5\n"


def test_throughput_is_reported_per_twin_with_the_shared_memory_peak(tmp_path: Path) -> None:
    """docs/10 E0 outputs: the injector's real-time factor per family (one twin per family here) and GNSS-SDR's wall
    time per run; twins are written to /dev/shm one at a time, so the peak is the largest twin (int16 I and Q)."""

    def twin(name: str, inject_s: float, receiver_s: float, samples: int) -> Twin:
        report = InjectionReport(1.0, {}, 0, "a", "b", samples, ("ghost",), {"total": inject_s})
        art = RunArtifacts(tmp_path, "c", tmp_path / "log", 0, {}, None, receiver_s)
        return Twin(name, report, art, {}, None)

    out = throughput([twin("zero0", 50.0, 4.0, 400_000_000), twin("jam60", 80.0, 5.0, 400_000_000)], 100.0)
    assert out["injector_x_realtime"] == {"zero0": 2.0, "jam60": 1.25}
    assert out["receiver_x_realtime"] == {"zero0": 25.0, "jam60": 20.0}
    assert out["injector_x_realtime_median"] == pytest.approx(1.625) and out["shm_peak_bytes"] == 1_600_000_000


def test_spec_overrides_reach_the_receiver_and_its_record(tmp_path: Path) -> None:
    """Deviations from the v0.0.22 defaults are stated in the experiment spec, used by every receiver run and
    recorded with them."""
    lock = {**TWIN_1C, "Tracking_1C.pll_bw_hz": "15.0"}
    rx = receiver_spec({"receiver_overrides": lock})
    assert dict(rx.overrides) == lock and rx.template == "1C"
    art = RunArtifacts(tmp_path, "c", tmp_path / "log", 0, {}, None, 4.0)
    rec = provenance(rx, {"zero0": art})
    assert rec["overrides"] == lock and rec["runs"]["zero0"]["config_sha256"] == "c"


@pytest.mark.parametrize(
    ("overrides", "template"),
    [
        ({}, "1C"),
        ({"Tracking_1C.max_carrier_lock_fail": "5000"}, "1C"),  # the stock limit is not the deviation
        ({"Tracking_1C.max_carrier_lock_fail": "1000000"}, "1C"),  # the FLL pull-in is missing
        (TWIN_1C, "L5"),  # the deviation must name the run's signal
    ],
)
def test_a_twin_spec_without_the_d34_lock_deviation_is_refused(overrides: dict[str, str], template: str) -> None:
    """D-34: every twin run raises the carrier-lock limit so the rule never drops a channel, and pulls in with the
    FLL so a start-up frequency false lock is not kept for good. A spec that forgets either is refused."""
    with pytest.raises(ValueError, match="D-34"):
        receiver_spec({"receiver_overrides": overrides}, template=template)


def test_a_stock_receiver_spec_runs_the_defaults() -> None:
    """The audit of GNSS-SDR as shipped (arm b) declares itself and keeps every default."""
    rx = receiver_spec({"stock_receiver": True}, template="L5")
    assert dict(rx.overrides) == {} and rx.template == "L5"
    l5_dev = {"Tracking_L5.max_carrier_lock_fail": "1000000", "Tracking_L5.enable_fll_pull_in": "true"}
    l5 = receiver_spec({"receiver_overrides": l5_dev}, template="L5")
    assert l5.template == "L5"


REPORT = InjectionReport(1.0, {}, 0, "s", "f", 10, (), {"total": 1.0})


def lock_track(prn: int, lock: float, t0: float = 5.0, t1: float = 40.0, dop: float = 1000.0) -> pa.Table:
    """A channel's records at 1 ms with a constant smoothed lock statistic and Doppler."""
    t = np.arange(t0, t1, 1e-3)
    n = t.size
    nulls = {c: [None] * n for c in TRACK_STREAM.names}
    cols = {**nulls, "run_id": ["z"] * n, "channel": [0] * n, "sys": ["G"] * n, "signal": ["1C"] * n,
            "prn": [prn] * n, "k": list(range(n)), "sample_count": (t * 4e6).astype(np.uint64), "t_rx_s": t,
            "lock_test": np.full(n, lock), "cn0_dbhz": np.full(n, 45.0), "dop_hz": np.full(n, dop),
            "acc_phase_rad": np.zeros(n), "t_int_s": [1e-3] * n}  # fmt: skip
    return build(TRACK_STREAM, cols, n)


def test_a_channel_not_locked_at_the_end_of_the_warm_up_is_found() -> None:
    """D-34 keeps every channel; a false lock at start-up (code tracked, carrier not: the lock statistic near 0) must
    then be caught before the analysis starts. Weak but real tracking (0.36 at 30 dB-Hz, F22) is locked."""
    tracks = {0: lock_track(3, 0.95), 1: lock_track(17, 0.01), 2: lock_track(9, 0.36), 3: lock_track(5, 0.9, t1=20.0)}
    channels = {0: ("G", "1C", 3), 1: ("G", "1C", 17), 2: ("G", "1C", 9), 3: ("G", "1C", 5), 4: ("G", "1C", 0)}
    assert unlocked_channels(tracks, channels, warmup_s=30.0) == [1, 3]  # ch 3 has no records in the window


def test_a_channel_locked_at_the_wrong_frequency_is_found() -> None:
    """P2b task 10 (E0 FGI-A, cell bn5-t20, PRN 6): a Costas loop at T = 1 ms also locks 1/(2T) = 500 Hz off, where each
    integration flips sign; its lock statistic still reads 0.75, so the check also compares the Doppler at the end of
    the warm-up with the PRN's surveyed Doppler (150 Hz: true Doppler moves under 1 Hz/s). A PRN without a reference is
    judged on lock alone."""
    tracks = {
        0: lock_track(4, 0.95, dop=-2760.0),
        1: lock_track(6, 0.75, dop=-2250.0),  # 500 Hz off: the false lock of E0 FGI-A
        2: lock_track(9, 0.9, dop=-3300.0),  # 500 Hz off the other way
        3: lock_track(11, 0.9, dop=700.0),  # no reference
    }
    channels = {0: ("G", "1C", 4), 1: ("G", "1C", 6), 2: ("G", "1C", 9), 3: ("G", "1C", 11)}
    expected = {4: -2750.0, 6: -2750.0, 9: -2800.0}
    assert unlocked_channels(tracks, channels, warmup_s=30.0, expected_doppler_hz=expected) == [1, 2]
    assert unlocked_channels(tracks, channels, warmup_s=30.0) == []


def test_a_twin_whose_channel_is_not_locked_is_received_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The injected bytes are kept and only GNSS-SDR re-runs, at most three times; the attempts are recorded."""
    calls: list[int] = []
    locks = iter([0.01, 0.02, 0.9])

    def fake_parse(name: str, art: RunArtifacts, channels: object, clock: TwinClock, source_id: str, *,
                   t_int_s: float) -> Twin:  # fmt: skip
        calls.append(1)
        return Twin(name, None, art, {0: lock_track(17, next(locks))}, None)

    monkeypatch.setattr(twin, "inject", lambda *a, **k: REPORT)
    monkeypatch.setattr(twin, "render_config", lambda *a, **k: "")
    monkeypatch.setattr(
        twin, "run_receiver", lambda *a, **k: RunArtifacts(tmp_path, "c", tmp_path / "l", 0, {}, None, 1.0)
    )
    monkeypatch.setattr(twin, "parse_run", fake_parse)
    src = IQSource(path=Path(__file__), item_type="ibyte", fs_hz=4e6, sample_start=0, sample_end=10)
    rx = receiver_spec({"receiver_overrides": TWIN_1C})
    tw = twin.run_twin("zero1", src, [], FrontEndSpec(1, 1, 1.0), {0: ("G", "1C", 17)}, tmp_path / "w", 1.0, 60.0, rx,
                        clock=CLOCK, warmup_s=30.0)  # fmt: skip
    assert len(calls) == 3 and tw.attempts == 3 and tw.unlocked == []
    locks = iter([0.01, 0.01, 0.01])
    calls.clear()
    tw = twin.run_twin("zero2", src, [], FrontEndSpec(1, 1, 1.0), {0: ("G", "1C", 17)}, tmp_path / "w", 1.0, 60.0, rx,
                        clock=CLOCK, warmup_s=30.0)  # fmt: skip
    assert len(calls) == 3 and tw.unlocked == [0]


def _fake_receiver(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, runs: list[dict[int, float]]) -> list[int]:
    """Receiver runs whose channels carry the given lock statistics, one dict per attempt."""
    calls: list[int] = []
    it = iter(runs)

    def fake_parse(name: str, art: RunArtifacts, channels: object, clock: TwinClock, source_id: str, *,
                   t_int_s: float) -> Twin:  # fmt: skip
        calls.append(1)
        locks = next(it)
        return Twin(name, None, art, {ch: lock_track(10 + ch, v) for ch, v in locks.items()}, None)

    monkeypatch.setattr(twin, "inject", lambda *a, **k: REPORT)
    monkeypatch.setattr(twin, "render_config", lambda *a, **k: "")
    monkeypatch.setattr(
        twin, "run_receiver", lambda *a, **k: RunArtifacts(tmp_path, "c", tmp_path / "l", 0, {}, None, 1.0)
    )
    monkeypatch.setattr(twin, "parse_run", fake_parse)
    return calls


SIX = {ch: ("G", "1C", 10 + ch) for ch in range(6)}  # channels 0-4 real, 5 the ghost
POLICY = twin.LockPolicy(must_lock=frozenset({5}), min_real=4)


def _run(name: str, tmp_path: Path, policy: twin.LockPolicy | None) -> Twin:
    src = IQSource(path=Path(__file__), item_type="ibyte", fs_hz=4e6, sample_start=0, sample_end=10)
    rx = receiver_spec({"receiver_overrides": TWIN_1C})
    return twin.run_twin(name, src, [], FrontEndSpec(1, 1, 1.0), SIX, tmp_path / "w", 1.0, 60.0, rx, clock=CLOCK,
                         warmup_s=30.0, lock_policy=policy)  # fmt: skip


def test_a_real_channel_that_never_locks_is_excluded_without_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P2b task 10: a twin's channels start on identical bytes, so a bad start (PRN 9 on segment A) repeats in every
    run; retrying it triples the receiver time for nothing. With enough real channels and every ghost locked, the
    channel is excluded and counted, and G-warmup-lock passes."""
    calls = _fake_receiver(monkeypatch, tmp_path, [{0: 0.9, 1: 0.01, 2: 0.9, 3: 0.9, 4: 0.9, 5: 0.9}])
    tw = _run("zero0", tmp_path, POLICY)
    assert len(calls) == 1 and tw.unlocked == [1] and tw.lock_ok
    (gate,) = twin.warmup_lock_gates([tw])
    assert gate[1].passed and gate[1].details["unlocked_channels"] == [1] and 1 not in tw.tracks


@pytest.mark.parametrize(
    "runs",
    [
        [{0: 0.9, 1: 0.9, 2: 0.9, 3: 0.9, 4: 0.9, 5: 0.01}] * 3,  # the ghost never locks
        [{0: 0.9, 1: 0.01, 2: 0.01, 3: 0.9, 4: 0.9, 5: 0.9}] * 3,  # three real channels: below the minimum
    ],
)
def test_a_lost_ghost_or_too_few_real_channels_is_retried_then_fails(
    runs: list[dict[int, float]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _fake_receiver(monkeypatch, tmp_path, runs)
    tw = _run("zero0", tmp_path, POLICY)
    assert len(calls) == 3 and not tw.lock_ok
    assert not twin.warmup_lock_gates([tw])[0][1].passed


def test_a_retry_that_meets_the_policy_ends_the_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_receiver(monkeypatch, tmp_path, [{0: 0.9, 1: 0.9, 2: 0.9, 3: 0.9, 4: 0.9, 5: 0.01},
                                                   {0: 0.9, 1: 0.01, 2: 0.9, 3: 0.9, 4: 0.9, 5: 0.9}])  # fmt: skip
    tw = _run("zero0", tmp_path, POLICY)
    assert len(calls) == 2 and tw.lock_ok and tw.unlocked == [1] and tw.attempts == 2


def test_every_receiver_run_is_recorded_and_counted(tmp_path: Path) -> None:
    """Review finding (p2a, important 5): re-runs of the warm-up lock check are runs too; the count and the run
    record come from one place (the E0b data test failed on a run with a re-run)."""
    art = RunArtifacts(tmp_path, "c", tmp_path / "log", 0, {}, None, 4.0)
    retried = Twin("zero1", None, art, {}, None, attempts=3, unlocked=[], failed=[art, art])
    runs = runs_of([Twin("zero0", None, art, {}, None), retried])
    assert sorted(runs) == ["zero0", "zero1", "zero1~1", "zero1~2"]
    assert receiver_run_count({"survey": art, **runs}) == 5


def ar1(sd: float, corr_epochs: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rho = np.exp(-1 / corr_epochs)
    w = rng.standard_normal(n) * sd * np.sqrt(1 - rho**2)
    x = np.zeros(n)
    x[0] = sd * rng.standard_normal()
    for i in range(1, n):
        x[i] = rho * x[i - 1] + w[i]
    return x


def test_ghost_sanity_is_judged_against_its_sampling_error() -> None:
    """Review finding (p2a, important 6): docs/3 section 7 asks for agreement 'within its sampling error'. The
    standard error of the SD ratio comes from the integrated autocorrelation time of e^2 (loop-correlated error)."""
    good = ghost_sanity(ar1(0.01, 14.0, 60_000, 3), sigma_cyc=0.01, dt_s=1e-3)
    assert good["pass"] and 0.005 < good["se"] < 0.03 and abs(good["sd_over_b3"] - 1) < 3 * good["se"]
    low = ghost_sanity(ar1(0.00909, 14.0, 60_000, 4), sigma_cyc=0.01, dt_s=1e-3)  # the CTTC reading, 0.909
    assert not low["pass"]


def test_observables_pairs_keep_only_satellites_kept_in_both_twins(tmp_path: Path) -> None:
    """P2b task 10 (E0 FGI-A): a channel excluded by the lock policy in one twin (PRN 6 in zero1) must leave the
    observables pair too, or the pair's epoch grid fails its alignment check."""
    art = RunArtifacts(tmp_path, "c", tmp_path / "l", 0, {}, None, 1.0)
    n = 6
    obs = pa.table({"prn": pa.array([4, 6, 7] * 2, pa.int16()), "t_gps_s": np.repeat([1.0, 2.0], 3)})
    a = Twin("zero0", None, art, {0: lock_track(4, 0.9), 1: lock_track(6, 0.9), 2: lock_track(7, 0.9)}, obs)
    b = Twin("zero1", None, art, {0: lock_track(4, 0.9), 2: lock_track(7, 0.9)}, obs)
    channels = {0: ("G", "1C", 4), 1: ("G", "1C", 6), 2: ("G", "1C", 7)}
    oa, ob = twin.common_obs(a, b, channels)
    assert oa is not None and ob is not None and oa.num_rows == ob.num_rows == n - 2
    assert set(oa.column("prn").to_pylist()) == {4, 7}
    assert twin.common_obs(a, Twin("x", None, art, {}, None), channels) == (None, None)


def test_a_satellite_without_valid_observables_in_one_twin_leaves_the_observables_pair(tmp_path: Path) -> None:
    """A channel can lock at the right frequency yet miss bit sync: no telemetry, so no valid observables in that
    twin. It stays in the tracking pairs and leaves the observables pair."""
    art = RunArtifacts(tmp_path, "c", tmp_path / "l", 0, {}, None, 1.0)
    tracks = {0: lock_track(4, 0.9), 1: lock_track(6, 0.9)}
    good = pa.table({"prn": pa.array([4, 6, 4, 6], pa.int16()), "t_gps_s": [1.0, 1.0, 2.0, 2.0],
                     "valid": [True, True, True, True]})  # fmt: skip
    no6 = good.set_column(2, "valid", pa.array([True, False, True, False]))
    channels = {0: ("G", "1C", 4), 1: ("G", "1C", 6)}
    oa, ob = twin.common_obs(Twin("a", None, art, tracks, good), Twin("b", None, art, tracks, no6), channels)
    assert (
        oa is not None
        and ob is not None
        and set(oa.column("prn").to_pylist()) == {4} == set(ob.column("prn").to_pylist())
    )
