"""E2K (P2b part 2 addendum of 19 Oct 2026, task 13) on a fake finished E2 run 2 and a fake s29 result: run 2 is read
and never written; an s29 result that does not hash as the spec says is refused; L_B per segment and combined comes
out as hand-computed (a None in one segment leaves the family without a ceiling, J_max_TA caps it, swept CW without
one moves the K = 101 and thermal blocks onto noise) and the oracle units are e2.plan_oracle's for those ceilings; with
L_B(noise) None nothing runs and the result says G3 fails on K; the oracle units pair with run 2's clean twin, read in
place, and E2's gates, K choice and thermal comparison come out in the summary with G3's clauses."""

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from gnssaudit.experiments import REGISTRY, e2, e2k
from gnssaudit.experiments.twin import jsonable
from gnssaudit.inject import IQSource
from gnssaudit.io.schemas import EVENTS, TRACK_STREAM, TRUTH_EPOCH_SAT, build
from gnssaudit.sdr.images import PATCH_ID
from gnssaudit.truth.rxtime import TwinClock

REPO = Path(__file__).resolve().parents[2]
REAL_RUNNERS = {name: getattr(e2, name) for name in ("run_clean", "run_jammed", "run_repeat")}  # before any fake
V2 = yaml.safe_load((REPO / "experiments" / "E2-v2.yaml").read_text())
FS, DECIM = 40_000.0, 4  # 10 kHz out: a 1 ms record is 10 output samples
CLOCK = TwinClock(fs_native_hz=FS, sample_start=0, decim=DECIM, delay=0.0)
DESCRIBE = "fake01234567"  # run 2's code version
N_REC = 25_000  # 25 s of 1 ms records
LEVELS = (50.0, 55.0, 60.0, 65.0, 70.0)
FAMILIES = ("noise", "swept_cw", "chirp")
SURVEY = {"sats": [4, 9], "ghosts": [{"prn": 32, "doppler_hz": 1510.0}, {"prn": 31, "doppler_hz": -2440.0}],
          "in_view": {"4": -2002.5, "9": -112.1}, "cn0": {"4": 44.5, "9": 46.3}, "n0": 1.77e-7,
          "line_excess_db": 2.08, "agc_rms": 2.16, "analog_sd": 1.50, "scale": 76.6}  # fmt: skip
# run 2's spec as its manifest keeps it: E2-v2's, on a fake recording and with small K-study blocks
SRC_SPEC = {**V2, "warmup_s": 2.0, "onset_s": 13.0, "pre_window_s": 5.0, "decim": DECIM, "fir_taps": 1, "pipelines": 1,
            "segments": {k: {"interval": f"fake-{k}", "item_type": "ibyte", "fs_hz": FS} for k in "AB"},
            "kstudy": {**V2["kstudy"], "blocks": {"cond": {"family": "swept_cw", "K": 3, "conditional": True},
                                                  "marg": {"family": "swept_cw", "K": 2, "conditional": False},
                                                  "noise": {"family": "noise", "K": 2, "conditional": False}}},
            "thermal": {**V2["thermal"], "K": 2}, "rules": {**V2["rules"], "k_small": 2}}  # fmt: skip
P1 = {"image": "sha256:1a63", "patch_id": "p1:0001+0002", "overrides": V2["receiver_overrides"]}
# Run 2's mode A records per segment and family, level by level (50 to 70 dB-Hz): r - floor of both ghosts, and the
# first ghost's lost truth (s). By hand, with J_max_TA 70 (chirp 65, B's): noise A 60 (B1 fails at 65), B 65 (truth
# lost at 70; 1.0 s at 65 is kept), combined 60; swept CW A None (B1 fails at 50), B 60, combined None; chirp A 65
# (passes every level, capped by J_max_TA), B None, combined None.
RUN2 = {
    ("A", "noise"): {"rmf": (0.04, 0.03, 0.03, 0.056, 0.13), "lost": (0.0, 0.0, 0.0, 0.0, 0.0)},
    ("B", "noise"): {"rmf": (0.04, 0.04, 0.03, 0.048, 0.04), "lost": (0.0, 0.0, 0.0, 1.0, 1.5)},
    ("A", "swept_cw"): {"rmf": (0.055, 0.01, 0.02, 0.02, 0.3), "lost": (0.0, 0.0, 0.0, 0.0, 0.0)},
    ("B", "swept_cw"): {"rmf": (0.02, 0.02, 0.03, 0.08, 0.2), "lost": (0.0, 0.0, 0.0, 0.0, 0.0)},
    ("A", "chirp"): {"rmf": (0.01, 0.01, 0.01, 0.01, 0.01), "lost": (0.0, 0.0, 0.0, 0.0, 0.0)},
    ("B", "chirp"): {"rmf": (0.35, 0.35, 0.35, 0.35, 0.35), "lost": (0.0, 0.0, 0.0, 0.0, 0.0)},
}
S29 = {"noise": {"A": 70.0, "B": 70.0}, "swept_cw": {"A": 70.0, "B": 70.0}, "chirp": {"A": 70.0, "B": 65.0}}
WANT_SEGMENT = {"A": {"noise": 60.0, "swept_cw": None, "chirp": 65.0},
                "B": {"noise": 65.0, "swept_cw": 60.0, "chirp": None}}  # fmt: skip
WANT = {"noise": 60.0, "swept_cw": None, "chirp": None}


def track(stamps: np.ndarray, acc_rad: np.ndarray, prn: int) -> pa.Table:
    n = stamps.size
    return build(TRACK_STREAM, {
        "run_id": ["r"] * n, "channel": np.zeros(n), "sys": ["G"] * n, "prn": np.full(n, prn), "signal": ["1C"] * n,
        "k": np.arange(n), "sample_count": stamps, "t_rx_s": CLOCK.file_time(stamps), "abs_ve": None, "abs_e": None,
        "abs_p": None, "abs_l": None, "abs_vl": None, "prompt_i": None, "prompt_q": None, "acc_phase_rad_f32": None,
        "acc_phase_rad": acc_rad, "dop_hz": None, "dop_rate_hzps": None, "code_freq_chips": None,
        "code_rate_chipsps2": None, "carr_err": None, "carr_err_filt": None, "code_err_chips": None,
        "code_err_filt": None, "cn0_dbhz": np.full(n, 45.0), "lock_test": None, "tow_ms": None, "wn": None,
        "t_int_s": np.full(n, 1e-3),
    }, n)  # fmt: skip


def clean_tracks() -> dict[str, pa.Table]:
    """Every channel of a clean twin (two real satellites, two ghosts): both mode A clean twins hold the same ones."""
    k = np.arange(1, N_REC + 1)
    prns = [*SURVEY["sats"], *(g["prn"] for g in SURVEY["ghosts"])]
    phase = np.random.default_rng(2).normal(0.0, 0.01, (len(prns), k.size))
    return {f"track_ch{ch}.parquet": track((10 * k + 2 * ch).astype(np.uint64), 2 * np.pi * (0.3 + phase[ch]), prn)
            for ch, prn in enumerate(prns)}  # fmt: skip


def truth_obs(rng: np.random.Generator, sd: float) -> pa.Table:
    """A jammed twin's truth for the real satellites at 10 Hz after onset + settle (what the oracle moments read)."""
    t = np.tile(15.0 + 0.1 * np.arange(100), 2)
    n = t.size
    return build(TRUTH_EPOCH_SAT, {
        "truth_id": ["g"] * n, "kind": ["twin"] * n, "sys": ["G"] * n, "prn": np.repeat(SURVEY["sats"], 100),
        "signal": ["1C"] * n, "t_rx_s": t, "e_cyc": rng.normal(0.0, sd, n), "slip_count": np.zeros(n),
        "arc_id": np.zeros(n), "transient": np.zeros(n, bool), "valid": np.ones(n, bool), "regime": np.zeros(n),
    }, n)  # fmt: skip


def write_unit(udir: Path, tables: dict[str, pa.Table], rec: dict[str, Any], describe: str) -> None:
    udir.mkdir(parents=True)
    for f, t in tables.items():
        pq.write_table(t, udir / f)
    (udir / "unit.json").write_text(json.dumps(jsonable(rec)))
    e2.mark_done(udir, describe, [*tables, "unit.json"])


def run2_summary(table: dict[tuple[str, str], dict[str, tuple[float, ...]]]) -> dict[str, Any]:
    """Run 2's summary as E2 wrote it: per segment the mode A records' residual (keyed by level, then ghost PRN) and
    lost ghost truth (keyed by unit, then channel), with the residual_pass and truth_kept E2 decided from them."""
    floor = e2.thermal_floor(45.0, 1e-3)
    segments: dict[str, Any] = {}
    for (key, fam), cfg in table.items():
        s = segments.setdefault(key, {"ghost_prn": [32, 31], "residual": {}, "lost_ghost_s": {}, "residual_pass": {},
                                      "truth_kept": {}})  # fmt: skip
        for lv, rmf, lost in zip(LEVELS, cfg["rmf"], cfg["lost"], strict=True):
            s["residual"].setdefault(fam, {})[str(lv)] = {p: {"r": floor + rmf, "floor": floor} for p in ("32", "31")}
            s["lost_ghost_s"][f"{key}-grid-{fam}-{lv:g}-A"] = {"2": lost, "3": 0.0}
            s["lost_ghost_s"][f"{key}-grid-{fam}-{lv:g}-B10"] = {"2": 0.0, "3": 0.0}
            s["residual_pass"].setdefault(fam, {})[str(lv)] = rmf < 0.05
            s["truth_kept"].setdefault(fam, {})[str(lv)] = lost <= 1.0
    gates = {"G-identity[A]": {"passed": True}, "G-identity[B]": {"passed": True}, "G-null[A][ch0]": {"passed": True},
             "G-repeat[repeat]": {"passed": True}}  # fmt: skip
    return {
        "segments": segments,
        "gates": gates,
        "receivers": P1,
        "decisions": {"j_max_db_hz": dict.fromkeys(FAMILIES), "ceiling_db_hz": dict.fromkeys(FAMILIES)},
    }


def write_s29(path: Path, jmax: Mapping[str, Mapping[str, float | None]]) -> str:
    fams = {}
    for f, per in jmax.items():
        both = None if None in per.values() else min(v for v in per.values() if v is not None)
        fams[f] = {"decides": f != "chirp", "j_max_ta_db_hz": per, "combined_db_hz": both}
    path.write_text(json.dumps({"decisions": {"families": fams}}, indent=1))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def shm_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """E2K clears its /dev/shm directory at its start and end: here a private one, never the host's."""
    monkeypatch.setattr(e2, "SHM_ROOT", tmp_path / "shm")
    return tmp_path / "shm"


@pytest.fixture
def source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """results/E2/run2: the manifest and summary, both segments' surveys, and the two mode A clean twins of segment A
    (A-clean-A, which the oracle units pair with, and A-clean-A2, its null partner), every unit done under run 2's
    code version."""
    monkeypatch.setenv("GNSSAUDIT_RESULTS", str(tmp_path / "results"))
    rec = tmp_path / "recording.iq"
    rec.write_bytes(bytes(2000))
    fake = IQSource(path=rec, item_type="ibyte", fs_hz=FS, sample_start=0, sample_end=1000)
    monkeypatch.setattr(e2, "recording_source", lambda spec: (fake, CLOCK))
    src = tmp_path / "results" / "E2" / "run2"
    src.mkdir(parents=True)
    summary = run2_summary(RUN2)
    (src / "manifest.json").write_text(json.dumps({"spec": SRC_SPEC, "git_describe": DESCRIBE, "summary": summary}))
    (src / "summary.json").write_text(json.dumps(summary))
    for key in "AB":
        udir = src / "units" / f"survey-{key}"
        udir.mkdir(parents=True)
        (udir / "survey.json").write_text(json.dumps(SURVEY))
        e2.mark_done(udir, DESCRIBE, ["survey.json"])
    base = {"config_sha256": "cfg", "rx_wall_s": 1.0, "unlocked": [], "lock_ok": True, "clip_count": 0}
    for name in ("A-clean-A", "A-clean-A2"):
        write_unit(src / "units" / name, clean_tracks(), {**base, "output_sha256": "clean"}, DESCRIBE)
    return src


def rewrite_summary(src: Path, summary: dict[str, Any]) -> None:
    """Another run 2 summary, written as E2 writes it: summary.json and the manifest's copy alike."""
    manifest = json.loads((src / "manifest.json").read_text())
    (src / "manifest.json").write_text(json.dumps({**manifest, "summary": summary}))
    (src / "summary.json").write_text(json.dumps(summary))


def e2k_spec(tmp_path: Path, jmax: Mapping[str, Mapping[str, float | None]] = S29) -> dict[str, Any]:
    path = tmp_path / "s29.json"
    digest = write_s29(path, jmax)
    return {"id": "E2K", "source_run": "E2/run2", "s29_result": str(path), "s29_sha256": digest,
            "receiver_patch": PATCH_ID, "kstudy": SRC_SPEC["kstudy"], "thermal": SRC_SPEC["thermal"],
            "rules": SRC_SPEC["rules"], "confirmatory": True}  # fmt: skip


def new_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "results" / "E2K" / "e2k-test-run"
    run_dir.mkdir(parents=True)
    (run_dir / "started.json").write_text(json.dumps({"git_describe": "e2k01234567"}))
    return run_dir


def snapshot(root: Path) -> dict[str, tuple[int, str]]:
    """Every path under root with its mtime and, for a file, its SHA-256."""
    out = {".": (root.stat().st_mtime_ns, "dir")}
    for p in sorted(root.rglob("*")):
        digest = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "dir"
        out[str(p.relative_to(root))] = (p.stat().st_mtime_ns, digest)
    return out


@pytest.fixture
def runners(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, Any]]:
    """E2's unit runners, faked: each writes what E2's would for the aggregation (unit.json, the real satellites' truth
    and slips of a jammed twin, a clean twin's tracks) under the run's units/ and records (runner, unit, the clean twin
    it was handed)."""
    calls: list[tuple[str, str, Any]] = []
    rng = np.random.default_rng(9)
    rec = {"config_sha256": "cfg", "rx_wall_s": 1.0, "unlocked": [], "lock_ok": True, "clip_count": 0,
           "output_sha256": "x", "identity": {"0": {"passed": True}, "1": {"passed": True}},
           "lost_ghost_s": {"2": 0.0, "3": 0.0},
           "stock_lock_verdict": {"0": {"first_drop_s": None, "drops": 0}}}  # fmt: skip

    def run_clean(ctx: e2.Context, u: e2.Unit) -> None:
        calls.append(("run_clean", u.name, None))
        write_unit(ctx.unit_dir(u.name), clean_tracks(), dict(rec), ctx.describe)
        ctx.cleans[u.name] = e2._load_twin(ctx.unit_dir(u.name), u.name)

    def run_jammed(ctx: e2.Context, u: e2.Unit) -> None:
        calls.append(("run_jammed", u.name, ctx.cleans[u.clean or ""]))
        tables = {"truth_obs.parquet": truth_obs(rng, 0.01), "slips_obs.parquet": EVENTS.empty_table()}
        write_unit(ctx.unit_dir(u.name), tables, {**rec, "unit": dataclasses.asdict(u)}, ctx.describe)

    def run_repeat(ctx: e2.Context, u: e2.Unit) -> None:
        calls.append(("run_repeat", u.name, None))
        # one passed receiver check per repeat, as a clean twin's repeat records: G-repeat needs at least one (final
        # review R1 M3), and this plan's seeded draw is a single jammed unit
        again = {**rec, "of": u.name, "same_bytes": True, "repeat": {"0": {"passed": True}}}
        write_unit(ctx.unit_dir(f"repeat-{u.name}"), {}, again, ctx.describe)

    for fn in (run_clean, run_jammed, run_repeat):
        monkeypatch.setattr(e2, fn.__name__, fn)
    return calls


def nothing_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(ctx: e2.Context, u: e2.Unit) -> None:
        pytest.fail(f"{u.name} was run")

    for name in ("run_clean", "run_jammed", "run_repeat"):
        monkeypatch.setattr(e2, name, refuse)


# ------------------------------------------------------------------------------------------------- the tests
def test_e2k_reads_run_2_and_never_writes_it(source: Path, tmp_path: Path, runners: list[Any]) -> None:
    before = snapshot(source)
    run_dir = new_run(tmp_path)
    out = e2k.run_e2k(e2k_spec(tmp_path), run_dir)
    assert snapshot(source) == before
    assert json.loads((run_dir / "summary.json").read_text())["decisions"] == out.summary["decisions"]
    units = source / "units"
    read = {source / "manifest.json", source / "summary.json", units / "survey-B" / "survey.json",
            units / "A-clean-A" / "track_ch2.parquet", units / "A-clean-A2" / "unit.json"}  # fmt: skip
    assert read <= set(out.inputs)  # hashed into the manifest
    assert all(p.is_file() and (source in p.parents or p in {tmp_path / "s29.json", tmp_path / "recording.iq"})
               for p in out.inputs)  # fmt: skip


def test_an_s29_result_that_does_not_hash_as_the_spec_says_is_refused(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nothing_runs(monkeypatch)
    spec = e2k_spec(tmp_path)
    with pytest.raises(ValueError, match="SHA-256"):
        e2k.run_e2k({**spec, "s29_sha256": "0" * 64}, new_run(tmp_path))
    (tmp_path / "s29.json").write_text((tmp_path / "s29.json").read_text().replace("70.0", "75.0", 1))
    with pytest.raises(ValueError, match="SHA-256"):
        e2k.run_e2k(spec, new_run(tmp_path / "again"))


def test_the_ceilings_come_out_as_hand_computed_per_segment_and_combined(source: Path, tmp_path: Path) -> None:
    """L_B(s, f) at most J_max_TA(f), combined over the segments: one segment without a ceiling leaves the family
    without one, and J_max_TA (65 for the chirp, segment B's) caps segment A's ceiling below its last passing level."""
    plan = e2k.decide(e2k_spec(tmp_path), new_run(tmp_path))
    assert {key: f["ceiling_db_hz"] for key, f in plan.front.items()} == WANT_SEGMENT
    assert plan.ceiling == WANT
    assert plan.j_max_ta == {"noise": 70.0, "swept_cw": 70.0, "chirp": 65.0}
    assert {key: f["j_max_ta_db_hz"] for key, f in plan.front.items()} == {
        key: {f: S29[f][key] for f in FAMILIES} for key in "AB"}  # fmt: skip
    assert plan.decides == {"noise": True, "swept_cw": True, "chirp": False}
    # E2's own decisions, reproduced from its records
    assert plan.front["A"]["residual_pass"]["noise"] == {50.0: True, 55.0: True, 60.0: True, 65.0: False, 70.0: False}
    assert plan.front["B"]["truth_kept"]["noise"] == {50.0: True, 55.0: True, 60.0: True, 65.0: True, 70.0: False}
    # a lower J_max_TA caps the noise ceiling
    capped = e2k.decide(e2k_spec(tmp_path, {**S29, "noise": {"A": 70.0, "B": 55.0}}), new_run(tmp_path / "capped"))
    assert capped.ceiling["noise"] == 55.0 and capped.front["B"]["ceiling_db_hz"]["noise"] == 55.0


def test_swept_cw_without_a_ceiling_moves_the_k_101_and_thermal_blocks_onto_noise(source: Path, tmp_path: Path) -> None:
    """The planned units are e2.plan_oracle's for the combined ceilings, which by hand are: the K = 101 block (here
    K = 3) and the thermal block on noise at L_K(noise) = 60 with no fixed timing, the noise block at 60, and no swept
    CW marginal block, all on segment A."""
    plan = e2k.decide(e2k_spec(tmp_path), new_run(tmp_path))
    assert plan.units == [u for key in "AB" for u in e2.plan_oracle(SRC_SPEC, key, WANT)]
    got: dict[str, list[e2.Unit]] = {}
    for u in plan.units:
        got.setdefault(u.block, []).append(u)
    assert {b: len(us) for b, us in got.items()} == {"k-cond": 3, "k-noise": 2, "thermal-clean": 2, "thermal": 2}
    assert {(u.family, u.level, u.offset_s, u.segment) for b in ("k-cond", "k-noise", "thermal") for u in got[b]} == {
        ("noise", 60.0, None, "A")}  # fmt: skip
    assert {u.clean for u in got["k-cond"] + got["k-noise"]} == {"A-clean-A"}


def test_without_a_noise_ceiling_nothing_runs_and_g3_fails_on_k(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The addendum: if L_B(noise) is None, no K-study is run and G3 fails on K, even with a swept CW ceiling (here
    swept CW passes B1 on A up to 65 and on B up to 60). A result, not a crash."""
    table = {**RUN2, ("B", "noise"): {**RUN2["B", "noise"], "rmf": (0.06, 0.04, 0.03, 0.048, 0.04)},
             ("A", "swept_cw"): {**RUN2["A", "swept_cw"], "rmf": (0.02, 0.01, 0.02, 0.02, 0.3)}}  # fmt: skip
    rewrite_summary(source, run2_summary(table))
    nothing_runs(monkeypatch)
    run_dir = new_run(tmp_path)
    out = e2k.run_e2k(e2k_spec(tmp_path), run_dir)
    d = out.summary["decisions"]
    assert d["ceiling_db_hz"] == {"noise": None, "swept_cw": 60.0, "chirp": None}
    assert d["k"] is None and d["k_study"]["blocks"] == {} and d["k_study"]["thermal"] is None
    assert d["k_study"]["note"] == "no K-study; G3 fails on K"
    g3 = out.summary["g3_clauses"]
    assert g3["k"] == {"k": None, "chosen": False, "text": "no K-study; G3 fails on K"}
    assert out.summary["gates"] == {} and g3["gates"]["e2k"]["passed"] is None
    # the owner excluded swept CW and the chirp; no noise ceiling is G3 failing on K, not an exclusion
    assert g3["exclusion"]["families"] == {"swept_cw": 60.0, "chirp": None}
    assert out.counts == {"units": 0, "gates_failed": 0}
    assert not (run_dir / "units").exists() or not list((run_dir / "units").glob("*/done.json"))


def test_the_oracle_units_pair_with_run_2s_clean_twin_and_the_summary_reports_k(
    source: Path, tmp_path: Path, runners: list[tuple[str, str, Any]]
) -> None:
    """E2's runners get the planned units: the thermal pairs' clean twins first, then the jammed units, then a seeded
    5 % again; every K-study unit is handed run 2's A-clean-A as it is on disk. The summary carries E2's gates on these
    twins (the null gate on run 2's two mode A clean twins), K by e2.choose_k and the thermal comparison, both
    receivers, and G3's clauses."""
    run_dir = new_run(tmp_path)
    s = e2k.run_e2k(e2k_spec(tmp_path), run_dir).summary
    planned = e2k.decide(e2k_spec(tmp_path), new_run(tmp_path / "plan")).units
    order = [name for fn, name, _ in runners if fn != "run_repeat"]
    assert order == [u.name for u in planned if u.block == "thermal-clean"] + [
        u.name for u in planned if u.block != "thermal-clean"]  # fmt: skip
    assert [name for fn, name, _ in runners if fn == "run_repeat"] == e2.pick_repeats(
        [u.name for u in planned], fraction=0.05, seed=23)  # fmt: skip
    on_disk = {int(p.stem.removeprefix("track_ch")): pq.read_table(p)
               for p in sorted((source / "units" / "A-clean-A").glob("track_ch*.parquet"))}  # fmt: skip
    for fn, name, clean in runners:
        if fn == "run_jammed" and name.startswith("A-k-"):
            assert clean.name == "A-clean-A" and clean.tracks.keys() == on_disk.keys()
            assert all(clean.tracks[ch].equals(t) for ch, t in on_disk.items())
    d = s["decisions"]
    assert d["j_max_ta_db_hz"] == {"noise": 70.0, "swept_cw": 70.0, "chirp": 65.0}
    assert d["ceiling_db_hz"] == WANT and d["k_level_db_hz"] == WANT
    assert {key: seg["ceiling_db_hz"] for key, seg in s["segments"].items()} == WANT_SEGMENT
    assert d["k_study"]["blocks"] == {"cond": {"family": "noise", "level": 60.0, "k": 3, "offset_s": None,
                                               "fallback": True},
                                      "noise": {"family": "noise", "level": 60.0, "k": 2, "offset_s": None,
                                                "fallback": False}}  # fmt: skip
    assert d["k_study"]["cond_vs_marg_done"] is False and "note" not in d["k_study"]
    assert d["k"] in (51, 101) and d["k"] == e2.choose_k(d["k_convergence"]) and "thermal_vs_fixed" in d
    gates = s["gates"]
    assert set(gates) == {"G-warmup-lock[A]", "G-phase-precision[A]", "G-identity[A]", "G-null[A][ch0]",
                          "G-null[A][ch1]", "G-repeat[repeat]"}  # fmt: skip
    assert all(g["passed"] for g in gates.values()) and gates["G-identity[A]"]["pairs"] == 2 * 7
    assert s["receivers"]["patch_id"] == PATCH_ID and s["source_receivers"] == P1
    assert s["source_clean_twins"] == dict.fromkeys(("A-clean-A", "A-clean-A2"), {"lock_ok": True, "unlocked": []})
    g3 = s["g3_clauses"]
    assert g3["gates"]["e2_run2"] == {"passed": True, "gates": 4, "failed": []}
    assert g3["gates"]["e2k"] == {"passed": True, "gates": 6, "failed": []}
    assert "E0: reported in the G3 report" in g3["gates"]["text"]
    assert g3["j_max"]["noise"] == {"pre_registered_db_hz": None, "j_max_ta_db_hz": 70.0, "decides": True}
    assert g3["ceiling_db_hz"] == WANT and g3["k"]["k"] == d["k"] and g3["k"]["chosen"] is True
    assert g3["exclusion"] == {
        "families": {"swept_cw": None, "chirp": None},
        "text": "The pre-registered B1 bar stands. A family without a ceiling is excluded from Route B and reported: "
        "swept CW and the chirp (risk R12). No correction of the B1 floor and no segment-only ceiling.",
    }


def test_e2k_refuses_k_study_rules_that_are_not_run_2s(source: Path, tmp_path: Path) -> None:
    spec = e2k_spec(tmp_path)
    changed = {"kstudy": {**spec["kstudy"], "level_cap_db_hz": 75.0}, "rules": {**spec["rules"], "residual_max": 0.06},
               "thermal": {**spec["thermal"], "K": 3}}  # fmt: skip
    for key, other in changed.items():
        with pytest.raises(ValueError, match=key):
            e2k.decide({**spec, key: other}, new_run(tmp_path / key))


def test_e2k_stops_where_run_2s_records_do_not_give_the_b1_verdict_it_decided(source: Path, tmp_path: Path) -> None:
    summary = json.loads((source / "summary.json").read_text())
    summary["segments"]["B"]["residual_pass"]["swept_cw"]["65.0"] = True  # r - floor is 0.08 there
    rewrite_summary(source, summary)
    with pytest.raises(ValueError, match="segment B, swept_cw: residual_pass"):
        e2k.decide(e2k_spec(tmp_path), new_run(tmp_path))


def test_the_e2k_spec_copies_the_pre_registered_k_study_and_names_its_inputs() -> None:
    """experiments/E2K.yaml: run 2 by its run id, s29 by path and the SHA-256 of the committed file, the current
    receiver patch, and the K-study's keys (kstudy, thermal, rules) copied unchanged from E2-v2.yaml."""
    spec = yaml.safe_load((REPO / "experiments" / "E2K.yaml").read_text())
    assert set(spec) == {"id", "source_run", "s29_result", "s29_sha256", "receiver_patch", "kstudy", "thermal",
                         "rules", "confirmatory"}  # fmt: skip
    assert spec["id"] == "E2K" and spec["source_run"] == "E2/6214cf52489f" and spec["confirmatory"] is True
    assert all(spec[k] == V2[k] for k in ("kstudy", "thermal", "rules"))
    assert spec["receiver_patch"] == PATCH_ID
    s29 = REPO / spec["s29_result"]
    assert spec["s29_sha256"] == hashlib.sha256(s29.read_bytes()).hexdigest()
    assert spec["s29_sha256"] == "efd04c8d0cc39d2d373fb7fae6454793f80c2f8d88648226b99d7562a70e6fc2"
    assert REGISTRY["E2K"] is e2k.run_e2k


# ------------------------------------------------------------------------------------ task 13 review, fix round 1
def test_e2k_refuses_a_source_clean_twin_that_did_not_meet_its_lock_policy(
    source: Path, tmp_path: Path, runners: list[Any]
) -> None:
    """Run 2's clean twins are not under E2K's own G-warmup-lock (that gate reads E2K's run directory): one whose
    unit.json does not record lock_ok true is refused before any twin runs, even when it still hashes as recorded."""
    udir = source / "units" / "A-clean-A2"
    rec = json.loads((udir / "unit.json").read_text())
    (udir / "unit.json").write_text(json.dumps({**rec, "lock_ok": False, "unlocked": [1]}))
    e2.mark_done(udir, DESCRIBE, [*sorted(p.name for p in udir.glob("track_ch*.parquet")), "unit.json"])
    with pytest.raises(ValueError, match="A-clean-A2: .*lock_ok"):
        e2k.run_e2k(e2k_spec(tmp_path), new_run(tmp_path))
    assert runners == []


def test_e2k_refuses_a_run_2_summary_json_that_is_not_the_summary_its_manifest_records(
    source: Path, tmp_path: Path
) -> None:
    """summary.json is where B1 and the truth kept come from. E2's manifest keeps the summary itself (not a hash of the
    file), so the file must equal manifest.json["summary"]. Here it is altered alone, consistently (B1 passing at 65 on
    segment A for noise): read unchecked, it would raise L_B(A, noise) from 60 to 65."""
    summary = json.loads((source / "summary.json").read_text())
    seg = summary["segments"]["A"]
    for p in ("32", "31"):
        seg["residual"]["noise"]["65.0"][p]["r"] = seg["residual"]["noise"]["65.0"][p]["floor"] + 0.04
    seg["residual_pass"]["noise"]["65.0"] = True
    (source / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="summary.json is not the summary its manifest.json records"):
        e2k.decide(e2k_spec(tmp_path), new_run(tmp_path))


def test_an_s29_result_whose_segments_do_not_combine_to_its_combined_value_is_refused(
    source: Path, tmp_path: Path
) -> None:
    """combine_segments over s29's per-segment J_max_TA must give s29's own combined value: here the chirp's segments
    (A 70, B 65) combine to 65 while the file says 70 (hashed as the spec says, so only this check can catch it)."""
    spec = e2k_spec(tmp_path)
    path = Path(spec["s29_result"])
    doc = json.loads(path.read_text())
    doc["decisions"]["families"]["chirp"]["combined_db_hz"] = 70.0
    path.write_text(json.dumps(doc))
    spec = {**spec, "s29_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match="s29, chirp: the segments' J_max_TA combine to 65.0, s29 says 70.0"):
        e2k.decide(spec, new_run(tmp_path))


def test_a_second_start_redoes_no_finished_unit_rebuilds_source_and_refuses_another_plan(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runners: list[Any]
) -> None:
    """--resume on E2K's run directory. The first start runs every unit (fake runners); the second goes through E2's
    own runners, which must find every unit finished under the run's code version and inject nothing, and give the
    same decisions and gates, with source/ made afresh. A finished unit recorded for another plan is refused."""
    spec, run_dir = e2k_spec(tmp_path), new_run(tmp_path)
    first = e2k.run_e2k(spec, run_dir).summary
    ran = len(runners)
    stale = run_dir / "source" / "stale"
    stale.write_text("an earlier start's")
    for name, fn in REAL_RUNNERS.items():
        monkeypatch.setattr(e2, name, fn)

    def inject_receive(ctx: e2.Context, u: e2.Unit, *, keep: Path | None = None) -> None:
        pytest.fail(f"{u.name} was injected again")

    monkeypatch.setattr(e2, "_inject_receive", inject_receive)
    second = e2k.run_e2k(spec, run_dir).summary
    assert len(runners) == ran and second["decisions"] == first["decisions"] and second["gates"] == first["gates"]
    assert second["units"] == first["units"] == 3 + 2 + 2 + 2 + 1
    assert not stale.exists() and (run_dir / "source" / "units" / "survey-B" / "survey.json").is_file()
    udir = run_dir / "units" / "A-k-noise-r1"
    rec = json.loads((udir / "unit.json").read_text())
    (udir / "unit.json").write_text(json.dumps({**rec, "unit": {**rec["unit"], "level": 55.0}}))
    e2.mark_done(udir, "e2k01234567", list(json.loads((udir / "done.json").read_text())["files"]))
    with pytest.raises(RuntimeError, match="A-k-noise-r1: the finished unit on disk .* is not the planned unit"):
        e2k.run_e2k(spec, run_dir)


# ------------------------------------------------------------------------------- final review, fix wave 1 (R1)
def rewrite_truth(udir: Path, shift_s: float, describe: str) -> None:
    """Move a finished jammed unit's truth by shift_s in time and mark it done again (its epochs then pair with no
    other unit's)."""
    tab = pq.read_table(udir / "truth_obs.parquet")
    t = tab.column("t_rx_s").to_numpy() + shift_s
    pq.write_table(
        tab.set_column(tab.schema.get_field_index("t_rx_s"), "t_rx_s", pa.array(t)), udir / "truth_obs.parquet"
    )
    e2.mark_done(udir, describe, list(json.loads((udir / "done.json").read_text())["files"]))


def test_a_k_block_short_of_its_planned_realizations_chooses_no_k(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runners: list[Any]
) -> None:
    """R1 I2: K's rule compares sigma^2 over the full K = 101 block (K = 3 here) with sigma^2 over its first k_small. A
    realization without truth (no observables, or no onset label) would silently leave a smaller block; K is then not
    chosen, with the reason, and every block's realized K is recorded beside its planned K."""
    fake = e2.run_jammed

    def run_jammed(ctx: e2.Context, u: e2.Unit) -> None:
        fake(ctx, u)
        if u.name == "A-k-cond-r1":  # reduce_unit found no truth for this realization
            udir = ctx.unit_dir(u.name)
            (udir / "truth_obs.parquet").unlink()
            (udir / "slips_obs.parquet").unlink()
            e2.mark_done(udir, ctx.describe, ["unit.json"])

    monkeypatch.setattr(e2, "run_jammed", run_jammed)
    s = e2k.run_e2k(e2k_spec(tmp_path), new_run(tmp_path)).summary
    d = s["decisions"]
    assert d["k_study"]["realized_k"] == {"cond": {"planned": 3, "realized": 2}, "noise": {"planned": 2, "realized": 2}}
    assert d["k_study"]["thermal_realized_k"] == {"planned": 2, "realized": 2}
    assert d["k"] is None and "k_convergence" not in d and "cond" in d["k_not_chosen"]
    assert s["g3_clauses"]["k"] == {"k": None, "chosen": False, "text": "K not chosen; G3 fails on K"}


def test_a_thermal_comparison_with_no_common_epoch_is_recorded_not_raised(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runners: list[Any]
) -> None:
    """R1 I3: the thermal comparison is reported only. If its ensembles share no epoch, it is recorded as not computed,
    with the reason; K, chosen before it, and the summary are still written."""
    fake = e2.run_jammed

    def run_jammed(ctx: e2.Context, u: e2.Unit) -> None:
        fake(ctx, u)
        if u.block == "thermal":  # labelled by its own clean twin: here on epochs no K-study twin has
            rewrite_truth(ctx.unit_dir(u.name), 0.05, ctx.describe)

    monkeypatch.setattr(e2, "run_jammed", run_jammed)
    run_dir = new_run(tmp_path)
    d = e2k.run_e2k(e2k_spec(tmp_path), run_dir).summary["decisions"]
    assert d["thermal_vs_fixed"] == {"computed": False, "reason": "no epoch with a variance in both ensembles",
                                     "epochs": 0}  # fmt: skip
    assert d["k"] in (51, 101) and d["k"] == e2.choose_k(d["k_convergence"])
    assert json.loads((run_dir / "summary.json").read_text())["decisions"]["k"] == d["k"]


def test_run_e2k_clears_its_shm_directory_before_any_unit(
    source: Path, tmp_path: Path, shm_root: Path, monkeypatch: pytest.MonkeyPatch, runners: list[Any]
) -> None:
    """R1 I4: twins a killed E2K left in /dev/shm are removed when it starts again, before the first unit."""
    stale = shm_root / "e2k-test-run" / "A-thermal-clean-r0.bin"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(bytes(8))
    seen: list[bool] = []
    fake = e2.run_clean

    def run_clean(ctx: e2.Context, u: e2.Unit) -> None:
        seen.append(stale.exists())
        fake(ctx, u)

    monkeypatch.setattr(e2, "run_clean", run_clean)
    e2k.run_e2k(e2k_spec(tmp_path), new_run(tmp_path))
    assert seen and not any(seen)


def test_e2k_refuses_a_run_2_whose_gates_failed(source: Path, tmp_path: Path, runners: list[Any]) -> None:
    """R1 M10: run 2 read with a failed gate (possible only with abort_on_gate_failure false) is refused before any
    twin runs."""
    summary = json.loads((source / "summary.json").read_text())
    summary["gates"]["G-repeat[repeat]"] = {"passed": False}
    rewrite_summary(source, summary)
    with pytest.raises(ValueError, match=r"G-repeat\[repeat\]"):
        e2k.run_e2k(e2k_spec(tmp_path), new_run(tmp_path))
    assert runners == []
