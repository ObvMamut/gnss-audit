"""The Admit experiment's helpers (P2b task 5): where the prefix is, the satellites' C/N0 series across free
channels, GNSS-SDR's position output, and the proposed interval-registry row."""

import json
import shutil
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
import yaml

from gnssaudit import experiments
from gnssaudit.cli.main import main
from gnssaudit.experiments import Outcome
from gnssaudit.experiments.admit import (
    FixTimesError,
    check_windows,
    cn0_by_prn,
    dop_by_prn,
    gps_time_map,
    interval_row,
    onset_blocks,
    power_1ms_db,
    resolve_prefix,
    run_admit,
    sky_blocks,
    sky_spec,
    survey_scale,
    timed_fixes,
)
from gnssaudit.geometry import Ephemerides, ecef_to_geodetic, predicted_doppler_hz
from gnssaudit.inject import IQSource
from gnssaudit.io.gnsssdr import read_pvt_geojson, read_pvt_nmea
from gnssaudit.io.rinex import gps_seconds
from gnssaudit.io.schemas import OBS_EPOCH_SAT, TRACK_STREAM, build
from gnssaudit.provenance import run_id
from gnssaudit.screen import Admission, SkyRules
from gnssaudit.sdr import SourceSpec
from gnssaudit.sdr.run import RunArtifacts
from tests.unit.sky_geometry import ARC, EPH, RX, T0, orbit, sky_sats


def track(
    prn: list[int], t: list[float], cn0: list[float], channel: int = 0, lock: list[float] | None = None
) -> pa.Table:
    n = len(t)
    nulls = {c: [None] * n for c in TRACK_STREAM.names}
    cols = {**nulls, "run_id": ["s"] * n, "channel": [channel] * n, "sys": ["G"] * n, "signal": ["1C"] * n,
            "prn": prn, "k": list(range(n)), "sample_count": [int(x * 4e6) for x in t], "t_rx_s": t,
            "cn0_dbhz": cn0, "lock_test": lock if lock is not None else [0.95] * n, "acc_phase_rad": [0.0] * n,
            "t_int_s": [1e-3] * n}  # fmt: skip
    return build(TRACK_STREAM, cols, n)


def test_the_verified_prefix_is_preferred_to_the_unverified_one(tmp_path: Path) -> None:
    final = tmp_path / "raw" / "x" / "a.iq"
    final.parent.mkdir(parents=True)
    (final.parent / "a.iq.unverified").write_bytes(b"12")
    assert resolve_prefix(tmp_path, "raw/x/a.iq") == (final.parent / "a.iq.unverified", False)
    final.write_bytes(b"12")
    assert resolve_prefix(tmp_path, "raw/x/a.iq") == (final, True)
    with pytest.raises(FileNotFoundError):
        resolve_prefix(tmp_path, "raw/x/b.iq")


def episode(prn: int, t0: float, seconds: float, cn0: float, lock: float, channel: int = 0) -> pa.Table:
    """One tracking episode at 1 ms: settled values throughout (the opening is skipped by the 2 s settle)."""
    t = list(t0 + 1e-3 * np.arange(int(seconds * 1000)))
    return track([prn] * len(t), t, [cn0] * len(t), channel=channel, lock=[lock] * len(t))


def test_cn0_series_follow_each_prn_across_free_channels() -> None:
    """A free channel can drop one PRN and acquire another; the series are by PRN and skip each episode's first 2 s
    (no C/N0 estimate yet, P2b task 6); idle records (PRN 0) are dropped."""
    a = pa.concat_tables(
        [episode(5, 0.0, 4.0, 40.0, 0.9), episode(0, 4.0, 1.0, 0.0, 0.0), episode(9, 5.0, 4.0, 44.0, 0.9)]
    )
    b = episode(9, 0.0, 4.0, 43.0, 0.9, channel=1)
    s = cn0_by_prn([a, b])
    assert sorted(s) == [5, 9]
    t9, c9 = s[9]
    assert t9.min() == pytest.approx(2.0) and np.all(np.diff(t9) >= 0)  # both channels' settled records, in time order
    assert set(c9) == {43.0, 44.0} and np.sum(c9 == 44.0) == 2000


def test_only_locked_records_count_as_a_satellite() -> None:
    """P2b task 6: under D-34 a free channel keeps tracking noise (C/N0 near 29 dB-Hz, lock statistic near 0) for good.
    Counted as a satellite, such a track widened the C/N0 spread past the spoofing bar on any recording; the series
    keep only settled records whose smoothed lock statistic reaches the warm-up check's 0.2 (a real 30 dB-Hz track reads
    0.36, F22)."""
    noise = episode(13, 0.0, 5.0, 28.9, 0.01)
    weak = episode(25, 0.0, 5.0, 30.0, 0.36, channel=1)
    lost = pa.concat_tables([episode(7, 0.0, 4.0, 41.0, 0.9, channel=2)])
    fading = track([7] * 1000, list(4.0 + 1e-3 * np.arange(1000)), [29.0] * 1000, channel=2, lock=[0.05] * 1000)
    s = cn0_by_prn([noise, weak, pa.concat_tables([lost, fading])])
    assert sorted(s) == [7, 25]
    assert set(s[7][1]) == {41.0} and s[7][0].size == 2000


def test_gnss_sdr_geojson_positions_are_read_as_lon_lat_h(tmp_path: Path) -> None:
    p = tmp_path / "pvt.dat_1.geojson"
    p.write_text(json.dumps({"type": "Feature", "properties": {}, "geometry": {"type": "MultiPoint",
                             "coordinates": [[1.98, 41.27, 85.4], [1.99, 41.28, 79.9]]}}))  # fmt: skip
    np.testing.assert_array_equal(read_pvt_geojson(p), [[1.98, 41.27, 85.4], [1.99, 41.28, 79.9]])


def test_the_proposed_row_registers_the_admitted_samples() -> None:
    adm = Admission(window_s=(0.0, 120.0), onset_s=None, reasons={}, sats={3: 47.0}, cn0_spread_db=9.0, pvt={})
    spec = {"dataset": "fgi-jt2023", "local_path": "raw/fgi-jt2023/JT23_4.1.5_L1_E1.iq", "fs_hz": 30.69e6,
            "band": "L1 (GPS L1 C/A, Galileo E1)", "receivers": ["LabSat 3 Wideband"]}  # fmt: skip
    row = interval_row(spec, adm)
    assert row["interval_id"] == "fgi-jt2023-JT23_4.1.5_L1_E1-clean" and row["state"] == "clean"
    assert row["sample_start"] == 0 and row["sample_end"] == 3_682_800_000 and row["file"] == "JT23_4.1.5_L1_E1.iq"
    assert interval_row(spec, Admission(window_s=None, onset_s=0.0))["state"] == "excluded"


def test_the_survey_scale_maps_the_largest_value_to_half_of_full_scale(tmp_path: Path) -> None:
    raw = np.array([3, -7, 1200, 5, -2, -32000], dtype=np.int16)  # one start-up spike far above the floor
    p = tmp_path / "s.bin"
    raw.tofile(p)
    src = IQSource(path=p, item_type="ishort", fs_hz=4e6, sample_start=0, sample_end=3)
    assert survey_scale(src) * 32000 == pytest.approx(16384)


def test_doppler_series_keep_the_same_settled_records_as_the_cn0_series() -> None:
    a = episode(5, 0.0, 4.0, 40.0, 0.9)
    dop = pa.array(np.full(a.num_rows, 1500.0), pa.float32())
    a = a.set_column(a.schema.get_field_index("dop_hz"), "dop_hz", dop)
    noise = episode(13, 0.0, 4.0, 28.9, 0.01, channel=1)
    noise = noise.set_column(
        noise.schema.get_field_index("dop_hz"), "dop_hz", pa.array(np.full(4000, 9e5), pa.float32())
    )
    d, c = dop_by_prn([a, noise]), cn0_by_prn([a, noise])
    assert sorted(d) == [5] and d[5][0].size == c[5][0].size == 2000
    assert set(d[5][1]) == {1500.0}


def obs_table(n_epochs: int, n_ch: int, tow0: float, valid_from: int = 0) -> pa.Table:
    """Observables as the receiver writes them: epoch i at sample (i + 1) x step, step 0.02 s at 7.6725 MS/s."""
    n = n_epochs * n_ch
    epoch = np.arange(n) // n_ch
    cols: dict[str, object] = {c: [None] * n for c in OBS_EPOCH_SAT.names}
    cols.update(source_id=["s"] * n, receiver_id=["r"] * n, t_gps_s=tow0 + 0.02 * (epoch + 1) + 0.003,
                sys=["G"] * n, prn=[1] * n, signal=["1C"] * n, valid=epoch >= valid_from)  # fmt: skip
    return build(OBS_EPOCH_SAT, cols, n)


def test_recording_time_maps_to_gps_time_through_the_sample_counter_clock() -> None:
    """Epoch i is stamped at sample (i + 1) x step (docs/9 section 10), so recording time t has the epoch whose
    receiver time is TOW(t); the week and day come from the spec's day, the TOW must lie inside that day."""
    day = 2278 * 604_800.0 + 3 * 86_400.0  # Wednesday 20 Sep 2023 00:00 GPS
    tow0 = 3 * 86_400.0 + 9 * 3600.0 + 13 * 60.0 + 29.0
    f = gps_time_map(obs_table(1000, 3, tow0, valid_from=40), n_channels=3, fs_hz=7_672_500.0, interval_s=0.02,
                     day_start_s=day)  # fmt: skip
    assert f(10.0) == pytest.approx(2278 * 604_800.0 + tow0 + 10.0 + 0.003, abs=1e-6)
    assert f(0.0) == pytest.approx(f(1.0) - 1.0)
    with pytest.raises(ValueError, match="day"):
        gps_time_map(obs_table(1000, 3, tow0), n_channels=3, fs_hz=7_672_500.0, interval_s=0.02,
                     day_start_s=day + 2 * 86_400.0)  # fmt: skip


def test_the_summary_has_a_sky_block_per_window() -> None:
    sats = sky_sats(drift_hz=300.0)
    t = np.arange(0.0, 150.0, 0.5)
    to_gps = lambda x: T0 + x - 75.0  # noqa: E731  (recording time 75 s is T0)
    cn0 = {p: (t, np.full(t.size, c)) for p, (c, _, _) in sats.items()}
    dop = {  # each PRN's Doppler follows its orbit through the recording, plus the clock drift
        p: (t, np.array([predicted_doppler_hz(orbit(p, ARC[p - 1]), RX, to_gps(x)) + 300.0 for x in t])) for p in sats
    }
    windows = [{"name": "positive", "t_start_s": 30.0, "t_end_s": 120.0, "expect": "pass"},
               {"name": "negative", "t_start_s": 135.0, "t_end_s": 150.0, "expect": "fail"}]  # fmt: skip
    equal = {p: (t, np.where(t >= 130.0, 45.0, v)) for p, (t, v) in cn0.items()}  # a spoofer: every PRN at one power
    blocks = sky_blocks(windows, cn0=equal, dop=dop, to_gps=to_gps, eph=EPH, rx_ecef=RX, rules=SkyRules())
    assert list(blocks) == ["positive", "negative"]
    pos, neg = blocks["positive"], blocks["negative"]
    assert pos["passed"] == {"S1": True, "S2": True, "S3": True} and pos["as_expected"] is True
    assert pos["S1_healthy_only"] is True and pos["unhealthy_prns"] == []
    assert pos["gps_t_s"] == pytest.approx(T0) and pos["n_sats"] == 8 and pos["clock_drift_hz"] == pytest.approx(300.0)
    assert pos["doppler_rms_hz"] < 0.05  # a Doppler that moves by tens of Hz across the window, judged at its epochs
    assert pos["prn"]["1"]["elev_deg"] > 5.0 and pos["prn"]["1"]["cn0_dbhz"] == sats[1][0]
    assert pos["prn"]["1"]["n_records"] == 180 and pos["prn"]["1"]["t_gps_median_s"] == pytest.approx(T0, abs=0.5)
    assert neg["passed"]["S3"] is False and neg["as_expected"] is True and neg["expect"] == "fail"
    assert neg["t_start_s"] == 135.0 and neg["gps_t_s"] == pytest.approx(T0 + 142.5 - 75.0)


def test_the_sky_block_reports_both_readings_of_s1() -> None:
    """A PRN with only an unhealthy record is in view (S1 passes), is listed, and fails the old reading."""
    sick = replace(orbit(9, 0.0), health=63)
    eph = Ephemerides([*(orbit(i + 1, m) for i, m in enumerate(ARC)), sick])
    t = np.arange(0.0, 60.0, 0.5)
    to_gps = lambda x: T0 + x - 30.0  # noqa: E731
    orbits = {**{i + 1: orbit(i + 1, m) for i, m in enumerate(ARC)}, 9: sick}
    cn0 = {p: (t, np.full(t.size, 38.0 + 0.1 * p)) for p in orbits}
    dop = {p: (t, np.array([predicted_doppler_hz(o, RX, to_gps(x)) for x in t])) for p, o in orbits.items()}
    block = sky_blocks([{"name": "w", "t_start_s": 0.0, "t_end_s": 60.0, "expect": "pass"}],
                       cn0=cn0, dop=dop, to_gps=to_gps, eph=eph, rx_ecef=RX, rules=SkyRules())["w"]  # fmt: skip
    assert block["passed"]["S1"] is True and block["S1_healthy_only"] is False and block["unhealthy_prns"] == [9]
    assert block["prn"]["9"]["elev_deg"] > 80.0 and block["n_sats"] == 9


# ------------------------------------------------------------------------------- specs, time map, runner (task 7)
SPECS = sorted(Path(__file__).resolve().parents[2].joinpath("experiments").glob("*.yaml"))
SKY_SPECS = ["admit-fgi-jt2023-JT23_4.1.5-sky.yaml", "admit-fgi-jt2023-JT23_16.1.1-B.yaml"]


@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.name)
def test_every_committed_spec_has_a_run_id(path: Path) -> None:
    """A date written unquoted in a spec loads as a datetime.date, which the canonical JSON of the run id refuses
    (task 7: both sky specs crashed at start)."""
    assert len(run_id(yaml.safe_load(path.read_text()))) == 12


@pytest.mark.parametrize(("name", "day"), [(SKY_SPECS[0], (2023, 9, 18)), (SKY_SPECS[1], (2023, 9, 20))])
def test_the_sky_specs_name_their_day_and_load_through_the_runner(
    name: str, day: tuple[int, int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = yaml.safe_load((SPECS[0].parent / name).read_text())
    monkeypatch.setattr("gnssaudit.experiments.admit.read_rinex_nav", lambda path: (Ephemerides([]), {}))
    _, day_start, rules = sky_spec(spec["sky"])
    assert day_start == gps_seconds(*day, 0, 0, 0.0) and rules == SkyRules()
    assert all(w["expect"] in ("pass", "fail") for w in spec["sky"]["windows"])


@pytest.mark.parametrize("name", SKY_SPECS)
def test_the_sky_specs_get_through_the_cli_to_the_runner(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI's own loading and run_id path, with the runner replaced: the spec reaches it intact and the run's
    manifest is written."""
    seen: list[dict] = []

    def stub(spec: dict, run_dir: Path) -> Outcome:
        seen.append(spec)
        return Outcome(inputs=[], counts={}, summary={})

    monkeypatch.setitem(experiments.REGISTRY, "Admit", stub)
    monkeypatch.setenv("GNSSAUDIT_RESULTS", str(tmp_path / "results"))
    monkeypatch.setattr("gnssaudit.cli.main.is_dirty", lambda repo: False)
    assert main(["run", str(SPECS[0].parent / name), "--confirmatory"]) == 0
    assert seen[0]["sky"]["day"] in ("2023-09-18", "2023-09-20") and isinstance(seen[0]["sky"]["day"], str)
    (run_dir,) = (tmp_path / "results" / "Admit").iterdir()
    assert (run_dir / "manifest.json").exists()


def test_the_time_map_refuses_a_clock_that_is_not_the_sample_counter() -> None:
    day = 2278 * 604_800.0 + 3 * 86_400.0
    obs = obs_table(1000, 3, 3 * 86_400.0 + 3600.0)
    tow = obs.column("t_gps_s").to_numpy() - 3 * 86_400.0 - 3600.0
    slow = obs.set_column(2, "t_gps_s", pa.array(3 * 86_400.0 + 3600.0 + 1.01 * tow))  # runs 1 % fast
    with pytest.raises(ValueError, match="sample-counter clock"):
        gps_time_map(slow, n_channels=3, fs_hz=7_672_500.0, interval_s=0.02, day_start_s=day)


def test_the_time_map_needs_whole_epochs_and_a_valid_one() -> None:
    day = 2278 * 604_800.0 + 3 * 86_400.0
    obs = obs_table(10, 3, 3 * 86_400.0 + 3600.0)
    with pytest.raises(ValueError, match="whole epochs"):
        gps_time_map(obs, n_channels=4, fs_hz=7_672_500.0, interval_s=0.02, day_start_s=day)
    with pytest.raises(ValueError, match="no valid observables"):
        gps_time_map(obs_table(10, 3, 3 * 86_400.0 + 3600.0, valid_from=10), n_channels=3, fs_hz=7_672_500.0,
                     interval_s=0.02, day_start_s=day)  # fmt: skip


FAKE_FS = 1000.0  # Hz: a 150 s fake recording of 300 kB


class FakeSurvey:
    """run_admit with the receiver and the heavy screens replaced: a survey whose eight satellites follow the
    constructed geometry, written as the receiver would leave it (a log, a PVT file, tracking dumps)."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, with_observables: bool = True) -> None:
        self.results = tmp_path / "results"
        data = tmp_path / "data"
        monkeypatch.setenv("GNSSAUDIT_RESULTS", str(self.results))
        monkeypatch.setenv("GNSSAUDIT_DATA", str(data))
        (data / "raw" / "d").mkdir(parents=True)
        (data / "raw" / "d" / "x.iq").write_bytes(bytes(int(150 * FAKE_FS * 2)))
        self.with_observables = with_observables
        to_gps = lambda x: T0 + x - 75.0  # noqa: E731
        sats = sky_sats()
        t = np.arange(0.0, 150.0, 0.01)
        self.tables = {}
        for ch, (prn, (c, _, _)) in enumerate(sats.items()):
            tr = track([prn] * t.size, list(t), [c] * t.size, channel=ch)
            dop = np.array([predicted_doppler_hz(orbit(prn, ARC[prn - 1]), RX, float(g)) for g in to_gps(t)])
            self.tables[ch] = tr.set_column(tr.schema.get_field_index("dop_hz"), "dop_hz", pa.array(dop, pa.float32()))
        a = "gnssaudit.experiments.admit."
        lat, lon, h = ecef_to_geodetic(RX)
        zero = (np.arange(150.0), np.zeros(150))
        monkeypatch.setattr(a + "estimate_n0", lambda *a_, **k: SimpleNamespace(n0_per_hz=1.0, line_excess_db=0.0))
        monkeypatch.setattr(a + "psd_deviation_db", lambda *a_, **k: zero)
        monkeypatch.setattr(a + "survey_scale", lambda source: 1.0)
        monkeypatch.setattr(a + "inject", lambda *a_, **k: None)
        monkeypatch.setattr(a + "receiver_spec", lambda spec, template: object())
        monkeypatch.setattr(a + "render_config", lambda *a_, **k: "")
        monkeypatch.setattr(a + "provenance", lambda rx, runs: {})
        monkeypatch.setattr(a + "run_receiver", self.run_receiver)
        monkeypatch.setattr(a + "read_tracking_dump", lambda p, fmt, **k: self.tables[k["channel"]])
        monkeypatch.setattr(a + "read_pvt_geojson", lambda p: np.array([[np.degrees(lon), np.degrees(lat), h]]))
        monkeypatch.setattr(
            a + "read_observables_dump", lambda *a_, **k: obs_table(7500, len(sats), 345_600.0 - 75.003)
        )
        monkeypatch.setattr(a + "sky_spec", lambda sky: (EPH, T0 - 3600.0, SkyRules()))
        self.spec = {
            "id": "Admit", "dataset": "d", "local_path": "raw/d/x.iq", "item_type": "ibyte", "fs_hz": FAKE_FS,
            "frontend_bw_hz": 400.0, "decim": 1, "fir_taps": 3, "max_channels": len(sats), "acquisition": False,
            "band": "L1", "receivers": [], "sky": {"nav": "n.rnx", "day": "2023-01-01", "windows": [
                {"name": "w", "t_start_s": 30.0, "t_end_s": 119.0, "expect": "pass"}]},
        }  # fmt: skip

    def run_receiver(self, rx: object, cfg: str, src: object, workdir: Path, timeout: float) -> RunArtifacts:
        (workdir / "pvt").mkdir(parents=True)
        (workdir / "pvt" / "p.geojson").write_text("{}")
        log = workdir / "gnss-sdr.log"
        log.write_text("survey log\n")
        obs = workdir / "observables.dat" if self.with_observables else None
        return RunArtifacts(workdir, "0" * 64, log, 0, {ch: workdir / f"trk_{ch}.dat" for ch in self.tables}, obs, 1.0)


def test_run_admit_writes_the_sky_blocks_into_its_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSurvey(tmp_path, monkeypatch)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    out = run_admit(fake.spec, run_dir)
    block = out.summary["sky"]["w"]
    assert block["passed"] == {"S1": True, "S2": True, "S3": True} and block["as_expected"] is True
    assert block["n_sats"] == 8 and block["doppler_rms_hz"] < 0.05
    ctx = out.summary["sky_context"]
    assert ctx["day"] == "2023-01-01" and ctx["rx_ecef_m"] == pytest.approx(list(RX), abs=1e-3)
    assert ctx["gps_s_at_recording_start"] == pytest.approx(T0 - 75.0, abs=0.01)
    assert (run_dir / "logs" / "survey.log.gz").exists() and not (fake.results / "Admit-work" / "run").exists()
    json.dumps(out.summary)  # the manifest takes it


def test_a_sky_failure_still_keeps_the_survey_log_and_leaves_no_work_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeSurvey(tmp_path, monkeypatch, with_observables=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(ValueError, match="observables"):
        run_admit(fake.spec, run_dir)
    assert (run_dir / "logs" / "survey.log.gz").exists()
    assert not (fake.results / "Admit-work" / "run").exists()


def test_power_1ms_db_finds_a_dip_in_its_window_and_not_outside_it(tmp_path: Path) -> None:
    fs = 1e6  # 1000 samples per ms
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1000, size=(2, 3_000_000))  # 3 s
    x[:, 1_500_000:1_510_000] *= 10 ** (-3 / 20)  # a 3 dB dip over 10 ms at 1.500 s
    path = tmp_path / "x.sc16"
    path.write_bytes(np.round(x.T).astype("<i2").tobytes())
    src = IQSource(path=path, item_type="ishort", fs_hz=fs, sample_start=0, sample_end=3_000_000)
    w = [{"name": "dip", "t_start_s": 1.4, "t_end_s": 1.6}, {"name": "quiet", "t_start_s": 2.0, "t_end_s": 2.9}]
    res = power_1ms_db(src, w, [0.5, 1.4])  # the baseline lies before the dip
    out = res["windows"]
    assert -3.5 < out["dip"]["min_rel_db"] < -2.0 and 1.5 <= out["dip"]["t_min_s"] <= 1.51
    assert out["dip"]["dip"] is True and out["quiet"]["dip"] is False
    assert out["dip"]["n_ms_below_1db"] >= 8 and out["quiet"]["n_ms_below_1db"] <= 2
    assert res["baseline_s"] == [0.5, 1.4] and "quantisation" in res["note"]


def real_if_spec(fake: FakeSurvey) -> dict:
    return {**fake.spec, "item_type": "byte", "source": {"real_if": {"if_hz": 100.0, "decim": 2, "taps": 5}}}


@pytest.fixture
def shm_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A private directory under /dev/shm (the sink insists on it) that is removed afterwards, whatever the test did;
    the free-space guard is replaced so the test does not depend on the host."""
    root = Path("/dev/shm") / f"gnssaudit-test-{tmp_path.name}"
    monkeypatch.setattr("gnssaudit.experiments.admit.SHM_ROOT", root)
    monkeypatch.setattr("gnssaudit.experiments.admit.require_free", lambda *a, **k: None)
    yield root
    shutil.rmtree(root, ignore_errors=True)


def test_a_real_if_spec_is_converted_in_shm_and_removed_afterwards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shm_root: Path
) -> None:
    fake = FakeSurvey(tmp_path, monkeypatch)
    seen = {}
    real_run = fake.run_receiver

    def run_receiver(rx: object, cfg: str, src: SourceSpec, workdir: Path, timeout: float) -> RunArtifacts:
        seen["bytes"] = src.path.stat().st_size  # the survey reads the converted file
        seen["fs"] = src.fs_hz
        return real_run(rx, cfg, src, workdir, timeout)

    monkeypatch.setattr("gnssaudit.experiments.admit.run_receiver", run_receiver)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    out = run_admit(real_if_spec(fake), run_dir)
    n_in = int(150 * FAKE_FS * 2)
    assert seen == {"bytes": 4 * (n_in // 2), "fs": FAKE_FS / 2}
    assert out.summary["duration_s"] == n_in / FAKE_FS and not list(shm_root.glob("*"))
    assert len(out.summary["real_if"]["sha256"]) == 64 and out.summary["real_if"]["n_samples"] == n_in // 2


def test_a_real_if_failure_still_removes_the_converted_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shm_root: Path
) -> None:
    fake = FakeSurvey(tmp_path, monkeypatch)

    def boom(*a: object, **k: object) -> RunArtifacts:
        raise RuntimeError("receiver died")

    monkeypatch.setattr("gnssaudit.experiments.admit.run_receiver", boom)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(RuntimeError, match="receiver died"):
        run_admit(real_if_spec(fake), run_dir)
    assert not list(shm_root.glob("*"))


def test_a_real_if_source_must_be_real_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shm_root: Path) -> None:
    fake = FakeSurvey(tmp_path, monkeypatch)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(ValueError, match="item_type must be 'byte'"):
        run_admit({**real_if_spec(fake), "item_type": "ibyte"}, run_dir)


def test_windows_outside_the_prefix_are_refused_before_any_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shm_root: Path
) -> None:
    fake = FakeSurvey(tmp_path, monkeypatch)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    spec = {**real_if_spec(fake), "onset_windows": [[58.5, 61.5], [99.5, 400.0]]}
    with pytest.raises(ValueError, match="onset_windows"):
        run_admit(spec, run_dir)
    assert not shm_root.exists()  # nothing was converted
    with pytest.raises(ValueError, match="power_baseline_s"):
        check_windows({"power_windows": [{"name": "w", "t_start_s": 1.0, "t_end_s": 2.0}],
                       "power_baseline_s": [30.0, 400.0]}, 300.0)  # fmt: skip


def test_the_summary_lists_every_sign_per_window_and_keeps_values_apart_from_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeSurvey(tmp_path, monkeypatch)
    spec = {**fake.spec, "onset_windows": [[58.5, 61.5], [99.5, 102.0]], "known_position": [60.0, 24.0, 10.0]}
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    out = run_admit(spec, run_dir)
    adm = out.summary["admission"]
    assert [w["t_start_s"] for w in out.summary["onset_windows"]] == [58.5, 99.5]
    assert all(w["pvt_offset"]["observable"] is False for w in out.summary["onset_windows"])  # no NMEA in the fake
    assert "pvt_offset" in adm["whole_prefix"] and "offset_m" in adm["whole_prefix"]["pvt_offset"]
    assert "pvt_offset" not in adm["reasons"]  # no metres in a list of seconds
    json.dumps(out.summary)


def test_onset_blocks_list_every_sign_inside_each_window_without_truncation() -> None:
    cn0 = [58.4, *[59.0 + 0.005 * i for i in range(300)], 62.0]
    reasons = {"cn0_step": cn0, "psd_change": [60.0, 101.0], "second_peak": [10.0]}
    nan = "no NMEA"
    w1, w2 = onset_blocks([[58.5, 61.5], [99.5, 102.0]], reasons, nan, pre_s=(30.0, 58.5), max_offset_m=50.0,
                          known=None)  # fmt: skip
    assert len(w1["cn0_step"]) == 300 and w1["psd_change"] == [60.0] and w1["second_peak"] == []
    assert w2["psd_change"] == [101.0] and w2["cn0_step"] == []
    assert w1["pvt_offset"] == {"observable": False, "reason": nan, "sign": None} and w1["any_sign"] is True
    assert w2["any_sign"] is True


def test_the_pvt_sign_is_the_window_median_against_the_pre_window_median() -> None:
    t = np.arange(30.0, 70.0, 0.5)
    lat = np.full(t.size, 60.0)
    lon = np.full(t.size, 24.0)
    lat[(t >= 58.5) & (t <= 61.5)] += 100.0 / 111_320.0  # 100 m north inside the first window only
    blocks = onset_blocks([[58.5, 61.5], [62.0, 65.0]], {}, (t, lat, lon), pre_s=(30.0, 58.5), max_offset_m=50.0,
                          known=(60.0, 24.0, 10.0))  # fmt: skip
    first, second = (b["pvt_offset"] for b in blocks)
    assert first["sign"] is True and first["offset_vs_pre_m"] == pytest.approx(100.0, abs=1.0)
    assert first["offset_vs_known_m"] == pytest.approx(100.0, abs=1.0) and blocks[0]["any_sign"] is True
    assert second["sign"] is False and second["offset_vs_pre_m"] < 1.0 and blocks[1]["any_sign"] is False
    empty = onset_blocks([[200.0, 201.0]], {}, (t, lat, lon), pre_s=(30.0, 58.5), max_offset_m=50.0, known=None)
    assert empty[0]["pvt_offset"]["observable"] is False and empty[0]["pvt_offset"]["sign"] is None


NMEA = """$GPGGA,000115.00,6010.9200,N,02449.6800,E,1,07,1.0,47.2,M,19.0,M,,*00
$GPGSA,A,3,05,07,13,14,20,30,,,,,,,1.5,1.0,1.1*00
$GPGGA,000116.00,6010.9206,N,02449.6806,E,1,07,1.0,47.2,M,19.0,M,,*00
$GPGGA,000117.00,,,,,0,00,,,M,,M,,*00
"""


def test_the_nmea_log_gives_timed_fixes_and_maps_to_recording_time(tmp_path: Path) -> None:
    f = tmp_path / "gnss_sdr_pvt.nmea"
    f.write_text(NMEA)
    tod, lat, lon = read_pvt_nmea(f)
    assert tod.tolist() == [75.0, 76.0] and lat[0] == pytest.approx(60.182, abs=1e-6)
    assert lon[0] == pytest.approx(24.828, abs=1e-6)
    # recording time t has TOW tow0 + 0.003 + t, and a fix's TOW is its UTC time of day plus the 18 s GPS-UTC offset
    tow0 = 100.0 - 0.003 - 40.0  # so the recording starts at TOW 60 s of day 0 (in the hour after midnight)
    obs = obs_table(1000, 3, tow0 + 3 * 86_400.0)
    t, _, _ = timed_fixes([f], obs, n_channels=3, fs_hz=7_672_500.0, duration_s=100.0)
    assert t.tolist() == pytest.approx([75.0 + 18.0 - 60.0, 76.0 + 18.0 - 60.0], abs=1e-6)


def test_fixes_that_cannot_be_timed_are_reported_not_guessed(tmp_path: Path) -> None:
    f = tmp_path / "n.nmea"
    f.write_text(NMEA)
    obs = obs_table(1000, 3, 3 * 86_400.0)
    with pytest.raises(FixTimesError, match="no NMEA"):
        timed_fixes([], obs, n_channels=3, fs_hz=7_672_500.0, duration_s=100.0)
    with pytest.raises(FixTimesError, match="no observables"):
        timed_fixes([f], None, n_channels=3, fs_hz=7_672_500.0, duration_s=100.0)
    with pytest.raises(FixTimesError, match="outside"):  # the mapping puts the fixes hours from the recording
        timed_fixes([f], obs_table(1000, 3, 3 * 86_400.0 + 20_000.0), n_channels=3, fs_hz=7_672_500.0, duration_s=20.0)


@pytest.mark.parametrize("verified", [True, False])
def test_the_manifest_hashes_the_navigation_file_the_sky_check_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verified: bool
) -> None:
    """Final review R2, I3: the ephemerides that decide S1-S3 are an input of the run, hashed into its manifest beside
    the prefix (which is hashed only once its file is verified)."""
    fake = FakeSurvey(tmp_path, monkeypatch)
    # the survey's injection is faked, so its /dev/shm sink is never opened: a stand-in keeps the test off /dev/shm
    monkeypatch.setattr(
        "gnssaudit.experiments.admit.SinkSpec", lambda kind, path: SimpleNamespace(path=tmp_path / "shm" / path.name)
    )
    data = tmp_path / "data"
    (data / "n.rnx").write_text("a navigation file")
    if not verified:
        (data / "raw" / "d" / "x.iq").rename(data / "raw" / "d" / "x.iq.unverified")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    out = run_admit(fake.spec, run_dir)
    prefix = [data / "raw" / "d" / "x.iq"] if verified else []
    assert out.inputs == [*prefix, data / "n.rnx"] and out.summary["verified"] is verified
