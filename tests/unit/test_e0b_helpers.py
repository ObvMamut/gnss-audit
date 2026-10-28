"""E0b helpers (docs/10 section 2.1; docs/8 sections 2.4 and 4; docs/3 section 7)."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import yaml

from gnssaudit.experiments import e0b
from gnssaudit.experiments.e0b import betz_loss_db, choose_ghost, linearity_slope, run_e0b
from gnssaudit.experiments.twin import Twin
from gnssaudit.io.schemas import TRACK_STREAM, build
from gnssaudit.sdr import RunArtifacts, loop_overrides
from gnssaudit.truth.rxtime import TwinClock


def test_ghost_prn_is_unused_and_its_doppler_avoids_code_line_beats() -> None:
    """docs/8 section 4: no acquisition peak on the clean twin, and >= 50 Hz from 1 kHz multiples of every strong
    satellite's Doppler (C/A cross-correlation is worst there)."""
    visible = {32: -1200.0, 31: 2410.0, 5: 300.0}
    prn, dop = choose_ghost(visible, doppler_hz=1220.0)
    assert prn == 30  # the highest PRN not in view
    for f in visible.values():
        beat = (dop - f) % 1000.0
        assert min(beat, 1000.0 - beat) >= 50.0
    assert abs(dop - 1220.0) < 100.0
    assert choose_ghost({}, doppler_hz=700.0) == (32, 700.0)


def test_linearity_slope_is_one_when_variance_follows_the_power() -> None:
    levels = np.array([50.0, 55.0, 60.0, 65.0, 70.0])
    assert linearity_slope(levels, 3e-7 * 10 ** (levels / 10)) == pytest.approx(1.0, abs=1e-12)
    assert linearity_slope(levels, 3e-7 * 10 ** (0.8 * levels / 10)) == pytest.approx(0.8, abs=1e-12)


def test_betz_loss_of_band_limited_noise() -> None:
    """(C/N0)_eff = C / (N0 + J kappa): the loss is 10 log10(1 + (J/N0) kappa)."""
    assert betz_loss_db(70.0, kappa_s=1 / 2e6) == pytest.approx(10 * np.log10(1 + 1e7 / 2e6))
    assert betz_loss_db(-np.inf, kappa_s=1 / 2e6) == 0.0


def test_a_reference_window_inside_the_warm_up_is_refused_before_any_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before the warm-up ends the twins share no phase, so a pre-onset window there leaves every epoch without a
    reference. The spec is refused before anything is written."""
    monkeypatch.setenv("GNSSAUDIT_RESULTS", str(tmp_path / "results"))
    spec = {"variant": "recording", "warmup_s": 30.0, "onset_s": 55.0, "pre_window_s": 20.0, "tones": {"onset_s": 30.0}}
    with pytest.raises(ValueError, match=r"\[20, 30\) s starts inside the 30 s warm-up"):
        run_e0b(spec, tmp_path / "run")
    assert not (tmp_path / "results").exists()


def test_a_crashed_run_s_work_directory_is_cleared_before_a_new_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review finding (p2a, important 7): a crash leaves E0b-work/<run id>; a re-run of the same spec starts empty."""
    monkeypatch.setenv("GNSSAUDIT_RESULTS", str(tmp_path / "results"))
    stale = tmp_path / "results" / "E0b-work" / "rid" / "zero0"
    stale.mkdir(parents=True)
    (stale / "trk_3.dat").write_bytes(b"old")
    seen: list[list[str]] = []

    def fake_synthetic(spec: object, work: Path, run_dir: Path) -> object:
        seen.append(sorted(p.name for p in work.iterdir()))
        raise RuntimeError("stop here")

    monkeypatch.setattr(e0b, "_synthetic", fake_synthetic)
    spec = {"variant": "synthetic", "warmup_s": 30.0, "onset_s": 60.0, "pre_window_s": 20.0}
    with pytest.raises(RuntimeError, match="stop here"):
        run_e0b(spec, tmp_path / "rid")
    assert seen == [[]]


def test_loop_cells_take_their_keys_from_the_inverse_map() -> None:
    """docs/10 E0 (c): `default` changes nothing, `default-o2` only the filter order, and the bn cells are the E3
    cells of `sdr.config.loop_overrides` (D-21), judged against the source-timing model at their own T."""
    assert e0b.loop_cell("default", "1C").overrides == {}
    assert e0b.loop_cell("default-o2", "L5").overrides == {"Tracking_L5.pll_filter_order": "2"}
    c1 = e0b.loop_cell("bn5-t1", "1C")
    assert c1.overrides == loop_overrides("1C", 5.0, 1e-3) and c1.t_s == 1e-3
    assert c1.loop.bn_eff_hz() == pytest.approx(5.0, rel=1e-3)
    c20 = e0b.loop_cell("bn15-t20", "1C")
    assert c20.overrides == loop_overrides("1C", 15.0, 20e-3) and c20.t_s == 20e-3
    assert c20.loop.bn_eff_hz() == pytest.approx(15.0, rel=1e-3)
    with pytest.raises(ValueError, match="unknown loop cell"):
        e0b.loop_cell("narrow", "1C")


BASE = {"variant": "recording", "warmup_s": 30.0, "onset_s": 55.0, "pre_window_s": 20.0}


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"template": "L5", "ghost": {"doppler_hz": 1500.0, "cn0_db_hz": 45.0}}, "GPS L1 C/A only"),
        ({"loop_cells": ["default", "bn5"]}, "unknown loop cell"),
    ],
)
def test_a_spec_the_run_cannot_honour_is_refused_before_any_work(
    extra: dict[str, object], match: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ghosts are C/A only (D-16): on L5, V7 and the ghost checks do not apply. A loop cell must be known."""
    monkeypatch.setenv("GNSSAUDIT_RESULTS", str(tmp_path / "results"))
    with pytest.raises(ValueError, match=match):
        run_e0b({**BASE, **extra}, tmp_path / "run")
    assert not (tmp_path / "results").exists()


def test_an_interval_resolves_to_its_samples_and_its_twins_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The twins of a registered interval start at its first sample; decimation by 4 with a 129-tap filter delays
    every output stamp by 64 native samples (p2a review, important 3)."""
    row = {"dataset_id": "d", "file": "f.iq", "sample_start": 1_000, "sample_end": 31_000}
    monkeypatch.setattr(e0b, "load_intervals", lambda: {"iv": row})
    monkeypatch.setattr(e0b, "dataset_file", lambda d, f: tmp_path / d / f)
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "f.iq").write_bytes(bytes(2 * 40_000))  # int8 I and Q
    spec = {"recording": {"interval": "iv", "item_type": "ibyte", "fs_hz": 30.69e6, "q_first": True}, "decim": 4,
            "fir_taps": 129}  # fmt: skip
    source, clock = e0b.recording_source(spec)
    assert (source.path, source.sample_start, source.sample_end, source.q_first) == (tmp_path / "d" / "f.iq", 1_000,
                                                                                    31_000, True)  # fmt: skip
    assert clock == TwinClock(fs_native_hz=30.69e6, sample_start=1_000, decim=4, delay=64.0)
    _, plain = e0b.recording_source({"recording": spec["recording"]})
    assert plain == TwinClock(fs_native_hz=30.69e6, sample_start=1_000, decim=1, delay=0.0)


def test_the_strongest_satellites_get_the_channels() -> None:
    """Free-channel surveys also hold noise tracks near 29 dB-Hz (JT23_4.1.5): only satellites at the bar or above
    are kept, the strongest first, listed in PRN order."""
    cn0 = {4: 44.6, 6: 41.0, 20: 37.7, 26: 42.4, 31: 28.9, 9: 44.4}
    assert e0b.strongest(cn0, min_cn0_dbhz=35.0, n=4) == [4, 6, 9, 26]
    assert e0b.strongest(cn0, min_cn0_dbhz=35.0, n=10) == [4, 6, 9, 20, 26]
    with pytest.raises(ValueError, match="no satellite"):
        e0b.strongest({31: 28.9}, min_cn0_dbhz=35.0, n=4)


def test_the_cttc_floor_of_36_drops_a_false_track_at_35_5() -> None:
    """P2c task 6: on the CTTC recording a false track on the +580 kHz line reads 35.1 to 35.4 dB-Hz, above a floor of
    35; at 36 it is dropped and a real satellite at 36.5 is kept."""
    cn0 = {4: 44.6, 27: 35.5, 9: 36.5}
    assert e0b.strongest(cn0, min_cn0_dbhz=35.0, n=4) == [4, 9, 27]
    assert e0b.strongest(cn0, min_cn0_dbhz=36.0, n=4) == [4, 9]


def test_the_cttc_spec_sets_the_floor_that_drops_the_false_track() -> None:
    """The CTTC fixture's own spec carries the floor: the survey's false track at 35.5 is not kept."""
    spec = yaml.safe_load((Path(__file__).resolve().parents[2] / "experiments" / "E0b-cttc.yaml").read_text())
    floor = float(spec.get("min_cn0_dbhz", 35.0))
    assert e0b.strongest({4: 44.6, 27: 35.5, 9: 36.5}, min_cn0_dbhz=floor, n=8) == [4, 9]


def test_an_onset_in_recording_time_is_placed_on_the_observable_epochs() -> None:
    """docs/3 section 2.6 (P2b task 9): epoch i sits at output sample (i + 1) x step; a decimated mid-file twin's
    output sample m is native sample start + 4 m - 64. The label is the receiver time of the first epoch at or after
    the onset, NaN while the receiver has no time yet."""
    fs, start = 30.69e6, 3_069_000  # the twin starts 0.1 s into the file
    clock = TwinClock(fs_native_hz=fs, sample_start=start, decim=4, delay=64.0)
    step = 153_450  # 20 ms at 7.6725 MS/s
    n_ep, n_ch = 3000, 2
    labels = np.where(np.arange(n_ep) < 100, 0.0, 5000.0 + 0.02 * np.arange(n_ep))
    obs = pa.table({"t_gps_s": np.repeat(labels, n_ch)})
    onset = 0.1 + 10.0  # output sample 10 x 7.6725e6 + 16 (the delay): 16 past epoch 499's, so epoch 500
    m = int(np.ceil((onset * fs - start + 64.0) / 4))
    i = -(-m // step) - 1
    assert i == 500 and e0b.epoch_label(obs, n_ch, clock, onset) == pytest.approx(5000.0 + 0.02 * i)
    assert np.isnan(e0b.epoch_label(obs, n_ch, clock, 0.1 + 1.0))  # epoch 50: no receiver time yet


def test_the_survey_counts_only_locked_records(tmp_path: Path) -> None:
    """P2b task 6: a noise track (about 29 dB-Hz, lock statistic near 0) is no satellite, and a satellite lost after
    the warm-up keeps the C/N0 of its locked records. A PRN held in lock for at least 1 s is in view, and stays out of
    the ghost's choice, however weak (it would bias ghost truth); the stock survey's noise tracks visit many PRNs and
    would otherwise exhaust the choice."""

    def tr(prn: int, cn0: list[float], lock: list[float], dop: float) -> pa.Table:
        n = len(cn0)
        t = 30.0 + 1e-3 * np.arange(n, dtype=float)
        nulls = {c: [None] * n for c in TRACK_STREAM.names}
        cols = {**nulls, "run_id": ["s"] * n, "channel": [0] * n, "sys": ["G"] * n, "signal": ["1C"] * n,
                "prn": [prn] * n, "k": list(range(n)), "sample_count": (t * 4e6).astype(np.uint64), "t_rx_s": t,
                "cn0_dbhz": cn0, "lock_test": lock, "dop_hz": [dop] * n, "acc_phase_rad": [0.0] * n,
                "t_int_s": [1e-3] * n}  # fmt: skip
        return build(TRACK_STREAM, cols, n)

    art = RunArtifacts(tmp_path, "c", tmp_path / "l", 0, {}, None, 1.0)
    n = 6000  # 6 s of 1 ms records; each episode's first 2 s are skipped (sdr.lock.in_lock)
    tracks = {
        0: tr(13, [28.9] * n, [0.01] * n, -7e4),  # noise
        1: tr(7, [41.0] * 4000 + [29.0] * 2000, [0.9] * 4000 + [0.0] * 2000, 900.0),  # lost after 4 s
        2: tr(25, [30.0] * n, [0.36] * n, -1200.0),  # weak but real (F22)
        3: tr(5, [29.0] * n, [0.0] * (n - 10) + [0.25] * 10, 4e4),  # 10 ms above the bar: still noise
    }
    in_view, cn0 = e0b.survey_satellites(Twin("survey", None, art, tracks, None), warm_abs=30.0)
    assert in_view == {7: 900.0, 25: -1200.0} and cn0 == {7: 41.0, 25: 30.0}


def test_a_pinned_ghost_keeps_its_prn_unless_that_prn_is_in_view() -> None:
    """A spec may pin the ghost's PRN, so a result does not move with the survey's PRN choice (on CTTC, V7 reads
    +0.32 dB on PRN 31 at 1500 Hz and +0.79 dB on PRN 30 at 1550 Hz: the floor's line meets each code differently).
    The Doppler is still chosen clear of the satellites in view."""
    visible = {32: -1200.0, 5: 300.0}
    prn, dop = choose_ghost(visible, doppler_hz=1220.0, prn=31)
    assert prn == 31 and min((dop + 1200.0) % 1000.0, 1000.0 - (dop + 1200.0) % 1000.0) >= 50.0
    with pytest.raises(ValueError, match="in view"):
        choose_ghost(visible, doppler_hz=1220.0, prn=5)


def test_a_channel_on_another_integration_than_its_twin_is_found(tmp_path: Path) -> None:
    """P2b task 10: in a 20 ms cell a channel that misses bit sync stays at 1 ms; its twin pair compares two loops and
    is left out of the loop identification (and listed), not paired."""
    art = RunArtifacts(tmp_path, "c", tmp_path / "l", 0, {}, None, 1.0)

    def at(step_s: float) -> pa.Table:
        n = int(10.0 / step_s)
        t = 30.0 + step_s * np.arange(n, dtype=float)
        nulls = {c: [None] * n for c in TRACK_STREAM.names}
        cols = {**nulls, "run_id": ["s"] * n, "channel": [0] * n, "sys": ["G"] * n, "signal": ["1C"] * n,
                "prn": [6] * n, "k": list(range(n)), "sample_count": (t * 4e6).astype(np.uint64), "t_rx_s": t,
                "acc_phase_rad": [0.0] * n, "t_int_s": [step_s] * n}  # fmt: skip
        return build(TRACK_STREAM, cols, n)

    a = Twin("c", None, art, {1: at(0.02), 2: at(0.02)}, None)
    b = Twin("t", None, art, {1: at(0.001), 2: at(0.02)}, None)
    assert e0b.integration_mismatch(a, b, after_s=31.0) == [1]
