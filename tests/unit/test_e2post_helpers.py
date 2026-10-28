"""E2post (P2b part 2, task 1) on a fake finished E2 run with tiny parquet files: the source run is read and never
written, the paired decomposition comes out as built (e_c by ghost truth on the clean twins' tracks), the exact pairing
drops and counts the records the two clean twins stamp differently, and gate A's booleans follow from the per-cell
numbers."""

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from numpy.typing import NDArray
from scipy import special

from gnssaudit.experiments import e2, e2post
from gnssaudit.inject import (
    Disturbance,
    FrontEndSpec,
    GhostSpec,
    IQSource,
    disturbance_seed,
    gain,
    ghost_modulation,
    ghost_phase_rad,
    pipeline,
)
from gnssaudit.io.schemas import TRACK_STREAM, TRUTH_EPOCH_SAT, build
from gnssaudit.truth.ghost import ghost_projection
from gnssaudit.truth.rxtime import TwinClock
from gnssaudit.uq.loop import CarrierLoop

FS, DECIM = 40_000.0, 4  # 10 kHz out: a 1 ms record is 10 output samples, a sample of shift is 0.1 ms
CLOCK = TwinClock(fs_native_hz=FS, sample_start=0, decim=DECIM, delay=0.0)
DESCRIBE = "fake01234567"
N_REC = 35_000  # 35 s of 1 ms records; post-onset (13 s + 2 s settle) leaves 20 s
SHIFTED = np.arange(16_000, 22_000, 200)  # post-onset records the mode B clean twin stamps one sample later (ch 2)
VAR_C, VAR_U, VAR_J = 0.001, 0.002, 0.008  # clean error shared by both clean twins, each twin's own part, jammer effect
SURVEY = {"sats": [4, 9], "ghosts": [{"prn": 32, "doppler_hz": 1510.0}, {"prn": 31, "doppler_hz": -2440.0}],
          "in_view": {"4": -2002.5, "9": -112.1}, "cn0": {"4": 44.5, "9": 46.3}, "n0": 1.77e-7,
          "line_excess_db": 2.08, "agc_rms": 2.16, "analog_sd": 1.50, "scale": 76.6}  # fmt: skip
SRC_SPEC = {"id": "E2", "seed": 23, "template": "1C", "warmup_s": 2.0, "onset_s": 13.0, "pre_window_s": 5.0,
            "segments": {"A": {"interval": "fake", "item_type": "ibyte", "fs_hz": FS}}, "decim": DECIM, "fir_taps": 1,
            "ghosts": {"cn0_db_hz": 45.0, "doppler_hz": [1500.0, -2500.0]},
            "families": {"noise": {"bw_hz": 2.0e6}, "swept_cw": {"span_hz": 1.0e6, "period_s": 0.01}},
            "levels_db_hz": [60.0, 70.0], "mode_b": {"tau_s": 0.01}}  # fmt: skip
GATE = {"families": ["noise", "swept_cw"], "a1": {"levels_db_hz": [60.0, 70.0], "slope_min": 0.90, "slope_max": 1.10},
        "a2": {"level_db_hz": 60.0, "cancel_share": 0.5}}  # fmt: skip
SPEC = {"id": "E2post", "source_run": "E2/fake", "segment": "A", "families": ["noise", "swept_cw"],
        "levels_db_hz": [60.0, 70.0], "ghost_channels": [2, 3], "bootstrap": {"block_s": 1.0, "n_boot": 100, "seed": 7},
        "rules": {"tol": 0.10, "slope_tol": 0.10, "se_max": 0.05}, "gate_a": GATE}  # fmt: skip


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


def twin_truth(t: np.ndarray, e_cyc: np.ndarray, prn: int) -> pa.Table:
    n = t.size
    return build(TRUTH_EPOCH_SAT, {
        "truth_id": ["g"] * n, "kind": ["twin"] * n, "sys": ["G"] * n, "prn": np.full(n, prn), "signal": ["1C"] * n,
        "t_rx_s": t, "e_cyc": e_cyc, "slip_count": np.zeros(n), "arc_id": np.zeros(n), "transient": np.zeros(n, bool),
        "valid": np.ones(n, bool), "regime": np.zeros(n),
    }, n)  # fmt: skip


def unit(src: Path, name: str, tables: dict[str, pa.Table]) -> None:
    udir = src / "units" / name
    udir.mkdir(parents=True)
    for f, t in tables.items():
        pq.write_table(t, udir / f)
    e2.mark_done(udir, DESCRIBE, list(tables))


@pytest.fixture
def source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """results/E2/fake: per ghost a clean error shared by the clean twins plus each twin's own part; mode A's twin
    difference is the jammer effect j alone, mode B's adds w_B - u_B, where w_B (the jammed twin's own part) is
    correlated 0.5 with u_B (the clean twin's): Var_B exceeds Var_A by VAR_U and all of it cancels in the excess."""
    monkeypatch.setenv("GNSSAUDIT_RESULTS", str(tmp_path / "results"))
    rec = tmp_path / "recording.iq"
    rec.write_bytes(bytes(2000))
    fake_source = IQSource(path=rec, item_type="ibyte", fs_hz=FS, sample_start=0, sample_end=1000)
    monkeypatch.setattr(e2, "recording_source", lambda spec: (fake_source, CLOCK))
    src = tmp_path / "results" / "E2" / "fake"
    src.mkdir(parents=True)
    summary = {"gates": {"G-identity[A]": {"passed": True}}}  # E2 always records its gates; E2post reads them
    (src / "manifest.json").write_text(json.dumps({"spec": SRC_SPEC, "git_describe": DESCRIBE, "summary": summary}))
    (src / "units" / "survey-A").mkdir(parents=True)
    (src / "units" / "survey-A" / "survey.json").write_text(json.dumps(SURVEY))
    e2.mark_done(src / "units" / "survey-A", DESCRIBE, ["survey.json"])
    rng = np.random.default_rng(11)
    k = np.arange(1, N_REC + 1)
    clean: dict[str, dict[str, pa.Table]] = {"A-clean-A": {}, "A-clean-B10": {}}
    jammed: dict[str, dict[str, pa.Table]] = {}
    for i, ch in enumerate((2, 3)):
        g = SURVEY["ghosts"][i]
        spec = GhostSpec(g["prn"], g["doppler_hz"], 0.0, 45.0, 17.3 + 101.0 * i, disturbance_seed(23, 0, 0, i))
        stamps_a = (10 * k + 2 * i).astype(np.uint64)  # the two ghost channels never share stamps
        stamps_b = stamps_a + np.isin(k, SHIFTED).astype(np.uint64) * (ch == 2)
        c = rng.normal(0.0, np.sqrt(VAR_C), k.size)
        u = {m: rng.normal(0.0, np.sqrt(VAR_U), k.size) for m in "AB"}
        for name, m, stamps in (("A-clean-A", "A", stamps_a), ("A-clean-B10", "B", stamps_b)):
            theta = ghost_phase_rad(spec, np.rint(CLOCK.native_index(stamps)).astype(np.int64), FS)
            clean[name][f"track_ch{ch}.parquet"] = track(stamps, -(theta + 2 * np.pi * (0.3 + c + u[m])), g["prn"])
        for fam in SRC_SPEC["families"]:
            for lv in (60, 70):
                j = rng.normal(0.0, np.sqrt(VAR_J), k.size)
                w = 0.5 * u["B"] + np.sqrt(0.75) * rng.normal(0.0, np.sqrt(VAR_U), k.size)
                ta, tb = CLOCK.file_time(stamps_a), CLOCK.file_time(stamps_b)
                jammed.setdefault(f"A-grid-{fam}-{lv}-A", {})[f"ghost_ch{ch}.parquet"] = twin_truth(ta, j, g["prn"])
                jammed.setdefault(f"A-grid-{fam}-{lv}-B10", {})[f"ghost_ch{ch}.parquet"] = twin_truth(
                    tb, j + w - u["B"], g["prn"]
                )
    for name, tables in {**clean, **jammed}.items():
        unit(src, name, tables)
    return src


def snapshot(root: Path) -> dict[str, tuple[int, str]]:
    """Every path under root with its mtime and, for a file, its SHA-256."""
    out = {".": (root.stat().st_mtime_ns, "dir")}
    for p in sorted(root.rglob("*")):
        digest = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "dir"
        out[str(p.relative_to(root))] = (p.stat().st_mtime_ns, digest)
    return out


def new_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "results" / "E2post" / "run"
    run_dir.mkdir(parents=True)
    return run_dir


def test_e2post_reads_the_source_run_and_never_writes_it(source: Path, tmp_path: Path) -> None:
    before = snapshot(source)
    run_dir = new_run(tmp_path)
    out = e2post.run_e2post(SPEC, run_dir)
    assert snapshot(source) == before
    assert json.loads((run_dir / "summary.json").read_text())["gate_A"] == out.summary["gate_A"]
    assert out.inputs and all(source in p.parents and p.is_file() for p in out.inputs)  # hashed into the manifest


def test_the_paired_decomposition_comes_out_as_built(source: Path, tmp_path: Path) -> None:
    s = e2post.run_e2post(SPEC, new_run(tmp_path)).summary
    for fam in ("noise", "swept_cw"):
        for lv in ("60", "70"):
            c = s["cells"][fam][lv]
            g2, g3 = c["ghosts"]["2"], c["ghosts"]["3"]
            # a shifted record leaves one record unmatched in each of the four series
            assert g2["dropped"] == 4 * SHIFTED.size and g3["dropped"] == 0
            assert g3["n"] == N_REC - 15_000 + 1 and g2["n"] == g3["n"] - SHIFTED.size  # t >= 15 s
            for g in (g2, g3):
                assert g["slope"] == pytest.approx(1.0, abs=0.02)
                assert g["var_b"] - g["var_a"] == pytest.approx(VAR_U, rel=0.15)  # option (a) sees the extra
                assert g["cov2_b"] == pytest.approx(-VAR_U, rel=0.2)  # and all of it cancels
                assert abs(g["cov2_a"]) < 0.2 * VAR_U
                assert g["excess_a"] == pytest.approx(g["var_a"] + g["cov2_a"])
                assert g["excess_b"] == pytest.approx(g["excess_a"], rel=0.08)
            assert c["ratio"] == pytest.approx(1.0, abs=0.06) and 0.0 < c["ratio_se"] < 0.05
            # ghost channel 2 lacks the shifted records once paired, so aligning it with channel 3 drops exactly those
            assert c["verdict"] == "pass" and c["dropped_aligning"] == SHIFTED.size
    assert s["gate_A"] == e2post.gate_a(s["cells"], GATE) and s["gate_A"]["A1"] and s["gate_A"]["A2"]


def test_e2post_refuses_a_source_unit_that_no_longer_hashes_as_recorded(source: Path, tmp_path: Path) -> None:
    with (source / "units" / "A-clean-B10" / "track_ch3.parquet").open("ab") as f:
        f.write(b"\0")
    with pytest.raises(ValueError, match="A-clean-B10"):
        e2post.run_e2post(SPEC, new_run(tmp_path))


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_e2post_names_the_cell_whose_verdict_it_refuses(source: Path, tmp_path: Path) -> None:
    """A ghost the jammer leaves untouched in mode A has no slope (0 / 0): the refusal says which cell it is."""
    udir = source / "units" / "A-grid-swept_cw-70-A"
    tab = pq.read_table(udir / "ghost_ch3.parquet")
    tab = tab.set_column(tab.schema.get_field_index("e_cyc"), "e_cyc", pa.array(np.zeros(tab.num_rows)))
    pq.write_table(tab, udir / "ghost_ch3.parquet")
    e2.mark_done(udir, DESCRIBE, ["ghost_ch2.parquet", "ghost_ch3.parquet"])
    with pytest.raises(ValueError, match="segment A, swept_cw at 70 dB-Hz: .*non-finite"):
        e2post.run_e2post(SPEC, new_run(tmp_path))


def test_e2post_refuses_an_unfinished_source_run(source: Path, tmp_path: Path) -> None:
    (source / "manifest.json").unlink()
    with pytest.raises(ValueError, match="finished"):
        e2post.run_e2post(SPEC, new_run(tmp_path))


# ------------------------------------------------------------------------------- final review, fix wave 1 (R1)
@pytest.mark.parametrize("where", ["equal", "inside", "above"])
def test_e2post_never_writes_into_the_run_it_reads(source: Path, tmp_path: Path, where: str) -> None:
    """R1 M9: a run directory equal to the source run, inside it or above it is refused before anything is written
    (E2K opens its source through the same guard)."""
    run_dir = {"equal": source, "inside": source / "post", "above": source.parent}[where]
    before = snapshot(source)
    with pytest.raises(ValueError, match="never writes into the run it reads"):
        e2post.run_e2post(SPEC, run_dir)
    assert snapshot(source) == before and not (source / "post").exists()


def test_a_second_start_makes_the_survey_copy_afresh(source: Path, tmp_path: Path) -> None:
    """R1 M6: an interrupted E2post (or E2K) starts again in its own run directory: the survey copy of the first
    start is replaced, not refused as existing."""
    run_dir = new_run(tmp_path)
    e2post.open_source(source, run_dir, "A")
    copy = run_dir / "units" / "survey-A" / "survey.json"
    copy.write_text("a torn copy")
    ctx, _ = e2post.open_source(source, run_dir, "A")
    assert copy.read_bytes() == (source / "units" / "survey-A" / "survey.json").read_bytes()
    assert ctx.segments["A"].ghost_ch == [2, 3]


def test_a_source_run_whose_gates_failed_is_refused(source: Path, tmp_path: Path) -> None:
    """R1 M10: with abort_on_gate_failure false, a run whose gates failed still ends with a manifest. E2post (and E2K,
    which opens its source the same way) refuse such a source, and one whose summary records no gates."""
    manifest = json.loads((source / "manifest.json").read_text())
    failed = {
        **manifest,
        "summary": {"gates": {"G-identity[A]": {"passed": True}, "G-null[A][ch0]": {"passed": False}}},
    }
    (source / "manifest.json").write_text(json.dumps(failed))
    with pytest.raises(ValueError, match=r"G-null\[A\]\[ch0\]"):
        e2post.run_e2post(SPEC, new_run(tmp_path))
    (source / "manifest.json").write_text(json.dumps({**manifest, "summary": {}}))
    with pytest.raises(ValueError, match="no gates"):
        e2post.run_e2post(SPEC, new_run(tmp_path / "again"))


def numbers(slope: float = 1.0, extra: float = 0.25, cancel: float = 0.25) -> dict[str, float]:  # exact in binary
    return {"slope": slope, "var_a": 1.0, "var_b": 1.0 + extra, "cov2_a": 0.0, "cov2_b": -cancel}


def cells(**over: dict[str, float]) -> dict[str, Any]:
    """Every gate cell at nominal numbers; `over` replaces one ghost, keyed family_level_channel."""
    out: dict[str, Any] = {}
    for fam in ("noise", "swept_cw", "chirp"):
        for lv in ("60", "70"):
            out.setdefault(fam, {})[lv] = {"ghosts": {ch: over.get(f"{fam}_{lv}_{ch}", numbers()) for ch in "78"}}
    return out


def test_gate_a1_asks_every_gate_ghosts_slope_within_the_closed_range() -> None:
    assert e2post.gate_a(cells(), GATE)["A1"]
    assert not e2post.gate_a(cells(swept_cw_70_8=numbers(slope=1.12)), GATE)["A1"]
    assert not e2post.gate_a(cells(noise_60_7=numbers(slope=0.89)), GATE)["A1"]
    assert e2post.gate_a(cells(noise_60_7=numbers(slope=0.90), noise_70_8=numbers(slope=1.10)), GATE)["A1"]
    assert e2post.gate_a(cells(chirp_60_7=numbers(slope=0.5)), GATE)["A1"]  # the chirp is reported, not gated


def test_gate_a2_asks_half_the_extra_variance_to_cancel_pooled_over_ghosts_at_60() -> None:
    g = e2post.gate_a(cells(noise_60_7=numbers(cancel=0.0625), noise_60_8=numbers(cancel=0.0625)), GATE)
    assert not g["A2"] and g["a2"]["noise"]["cancel"] == pytest.approx(0.125)
    assert g["a2"]["noise"]["extra_var"] == pytest.approx(0.5) and g["a2"]["swept_cw"]["holds"]
    # pooled: one ghost cancels it all, the other none, and half of the pooled extra cancels (the bound is >=)
    assert e2post.gate_a(cells(swept_cw_60_7=numbers(cancel=0.25), swept_cw_60_8=numbers(cancel=0.0)), GATE)["A2"]
    # A2 reads 60 dB-Hz only
    assert e2post.gate_a(cells(noise_70_7=numbers(cancel=0.0), noise_70_8=numbers(cancel=0.0)), GATE)["A2"]
    assert e2post.gate_a(cells(), GATE)["A1"] and e2post.gate_a(cells(), GATE)["A2"]


# ------------------------------------------------------------------------------------------------ B1 (task 3)
# The interferer B1 is checked against: EPS A s_g e^(j phi) for ghost KNOWN_CH, phi = PHI_RAD sin(2 pi PHI_HZ t), with
# A the ghost's amplitude. Despread by s_g it leaves EPS A e^(j phi), so n_iota,k = EPS mean(sin phi) over record k, in
# rad (tests/known_answer/test_ghost_projection.py). VAR_W is the white part of the mode A twin difference, in cycles^2.
EPS, PHI_RAD, PHI_HZ, VAR_W = 0.05, 0.3, 7.0, 2.0e-6
KNOWN_CH, KNOWN_I = 3, 1  # ghost channel 3 is the survey's second ghost
B1 = {"families": ["noise", "swept_cw"], "levels_db_hz": [60.0, 70.0], "block": 1 << 16, "workers": 1, "r_tol": 1e-12}
# The spec as E2's first run ran it: its YAML listed the families in another order than the manifest's sorted JSON
# keeps, and that run keyed a cell's seed on that order (e2._cell until P2b part 2, task 5, which keys seeds on sorted
# names), so the manifest's spec alone does not give the seeds it drew.
RAN = {**SRC_SPEC, "families": {"swept_cw": SRC_SPEC["families"]["swept_cw"], "noise": SRC_SPEC["families"]["noise"]}}


def ran_grid() -> list[e2.Unit]:
    """The units as E2's first run planned them: each grid cell's seed by its family's place in RAN's order."""
    order = list(RAN["families"])
    return [dataclasses.replace(u, seed=disturbance_seed(23, 100 + order.index(u.family or ""), 0, int(u.level or 0)))
            if u.block == "grid" else u for u in e2.plan_grid(RAN, "A")]  # fmt: skip


N_OUT = 10 * (N_REC + 1) + 10  # output samples: past the last stamp's integration


def ghost(i: int) -> GhostSpec:
    g = SURVEY["ghosts"][i]
    return GhostSpec(g["prn"], g["doppler_hz"], 0.0, 45.0, 17.3 + 101.0 * i, disturbance_seed(23, 0, 0, i))


def known_stream() -> np.ndarray:
    m = np.arange(N_OUT, dtype=np.int64)
    phi = PHI_RAD * np.sin(2 * np.pi * PHI_HZ * CLOCK.file_time(m))
    native = np.rint(CLOCK.native_index(m)).astype(np.int64)
    return EPS * gain(45.0, SURVEY["n0"]) * ghost_modulation(ghost(KNOWN_I), native, FS) * np.exp(1j * phi)


def clean_stamps(src: Path, ch: int) -> np.ndarray:
    return pq.read_table(src / "units" / "A-clean-A" / f"track_ch{ch}.parquet").column("sample_count").to_numpy()


@pytest.fixture
def b1_source(source: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, list[tuple[Any, ...]]]:
    """The fake run with what E2 saved on its projected (mode A grid) units: per ghost, pred_cyc = E2's prediction from
    the known interferer's n_iota (e2.reduce_unit's steps) and e_cyc = pred_cyc + w, w white with VAR_W; unit.json with
    E2's residual and its unit record (seeds as drawn under RAN's family order). e2post's `interference` yields the
    known interferer and records its calls."""
    y = known_stream()
    calls: list[tuple[Any, ...]] = []

    def interference(iq: IQSource, ds: list[Disturbance], fe: FrontEndSpec, *, n0_per_hz: float, block: int):
        calls.append((iq, ds, fe, n0_per_hz))
        for m0 in range(0, y.size, block):
            yield m0, y[m0 : m0 + block]

    monkeypatch.setattr(e2post, "interference", interference)
    loop = CarrierLoop.gnsssdr(35.0, 1e-3)
    rng = np.random.default_rng(5)
    cdir = source / "units" / "A-clean-A"
    for ch in (2, 3):  # the clean twin's prompts: the ghost at A^ with the data bits, and white noise
        tab = pq.read_table(cdir / f"track_ch{ch}.parquet")
        bits = np.repeat(np.where(rng.random(tab.num_rows // 20 + 1) < 0.5, -1.0, 1.0), 20)[: tab.num_rows]
        prompt = cplx(55.0 * bits * (1 + circular(rng, 0.0245, bits.size)))
        for col, v in (("prompt_i", prompt.real), ("prompt_q", prompt.imag)):
            tab = tab.set_column(tab.schema.get_field_index(col), col, pa.array(v, pa.float32()))
        pq.write_table(tab, cdir / f"track_ch{ch}.parquet")
    e2.mark_done(cdir, DESCRIBE, ["track_ch2.parquet", "track_ch3.parquet"])
    onset, pre = SRC_SPEC["onset_s"], SRC_SPEC["pre_window_s"]
    ran = {u.name: u for u in ran_grid()}
    for fam in SRC_SPEC["families"]:
        for lv in (60, 70):
            udir = source / "units" / f"A-grid-{fam}-{lv}-A"
            residual = {}
            for i, ch in enumerate((2, 3)):
                stamps = clean_stamps(source, ch)
                # the stream in B1's blocks, as e2post's `interference` yields it: a projection's sums are bitwise
                # the same only on the same chunking (one chunk differs by ~1e-15 cycle on some SIMD paths)
                blocks = ((m0, y[m0 : m0 + B1["block"]]) for m0 in range(0, y.size, B1["block"]))
                n_iota = ghost_projection(blocks, ghost(i), stamps, CLOCK, gain(45.0, SURVEY["n0"]))
                pred = loop.filter(np.nan_to_num(n_iota)) / (2 * np.pi)
                tab = pq.read_table(udir / f"ghost_ch{ch}.parquet")
                t = tab.column("t_rx_s").to_numpy()
                p = np.interp(t, CLOCK.file_time(stamps), pred)
                p -= np.mean(p[(t >= onset - pre) & (t < onset)])
                e = p + rng.normal(0.0, np.sqrt(VAR_W), t.size)
                tab = tab.set_column(tab.schema.get_field_index("e_cyc"), "e_cyc", pa.array(e))
                pq.write_table(tab.append_column("pred_cyc", pa.array(p)), udir / f"ghost_ch{ch}.parquet")
                sel = e2._post(t, onset)
                residual[str(SURVEY["ghosts"][i]["prn"])] = {
                    "r": e2.residual_ratio(e[sel], p[sel]), "floor": e2.thermal_floor(45.0, 1e-3),
                    "cn0_clean_dbhz": 45.0, "corr": float(np.corrcoef(e[sel], p[sel])[0, 1]), "n": int(sel.sum()),
                }  # fmt: skip
            unit_rec = dataclasses.asdict(ran[udir.name])
            (udir / "unit.json").write_text(json.dumps({"residual": residual, "unit": unit_rec}))
            e2.mark_done(udir, DESCRIBE, ["ghost_ch2.parquet", "ghost_ch3.parquet", "unit.json"])
    return source, calls


def b1_run(tmp_path: Path, **b1: Any) -> dict[str, Any]:
    return e2post.run_e2post({**SPEC, "b1": {**B1, **b1}}, new_run(tmp_path)).summary


def saved(src: Path, fam: str, lv: int, ch: int) -> tuple[np.ndarray, np.ndarray]:
    tab = pq.read_table(src / "units" / f"A-grid-{fam}-{lv}-A" / f"ghost_ch{ch}.parquet")
    sel = e2._post(tab.column("t_rx_s").to_numpy(), SRC_SPEC["onset_s"])
    return tab.column("e_cyc").to_numpy()[sel], tab.column("pred_cyc").to_numpy()[sel]


def test_b1_recomputes_r_from_the_saved_series_and_r_is_the_white_parts_share(
    b1_source: tuple[Path, list[tuple[Any, ...]]], tmp_path: Path
) -> None:
    """dPhi = H n_iota + w with E2's prediction H n_iota: r = Var(w) / Var(dPhi), and e2post's r from the saved e_cyc
    and pred_cyc is E2's saved r to 1e-12."""
    src, _ = b1_source
    s = b1_run(tmp_path)
    for fam in ("noise", "swept_cw"):
        for lv in (60, 70):
            for i, ch in enumerate((2, 3)):
                g = s["b1"][fam][f"{lv}"][str(ch)]
                e, p = saved(src, fam, lv, ch)
                assert g["prn"] == SURVEY["ghosts"][i]["prn"] and g["n"] == e.size
                assert abs(g["r"] - g["r_saved"]) <= 1e-12 and g["r_diff"] <= 1e-12
                assert g["r"] == pytest.approx(np.var(e - p) / np.var(e), rel=1e-12)
                assert 0.2 < g["r"] < 0.9  # Var(H n_iota) and VAR_W are of one order here
                assert g["floor"] == pytest.approx(1.0 / (10**4.5 * 1e-3), rel=1e-12)  # 1 / (C/N0 T) at 45 dB-Hz
                assert g["formula"] == pytest.approx(g["floor"] + g["var_n_iota_rad2"], rel=1e-12)
                assert g["r_minus_formula"] == pytest.approx(g["r"] - g["formula"], rel=1e-12)
                assert g["corr"] == pytest.approx(np.corrcoef(e, p)[0, 1], rel=1e-12)
                ph = g["post_hoc"]  # reported only: the lag scan, n_iota's spectrum, the record-level decomposition
                assert ph["lag_scan"]["argmin_lag"] == 0 and ph["lag_scan"]["corr_sign_lag0"] == 1
                assert ph["spectrum"]["eta_w"] == pytest.approx(LOOP.noise_gain())
                assert ph["decomposition"]["g"] == pytest.approx(gain(45.0, SURVEY["n0"]) * 10 * SURVEY["scale"] / 55.0,
                                                                 rel=0.01)  # fmt: skip
                assert all(math.isfinite(v) for part in ph.values() for v in part.values() if not isinstance(v, list))
    assert "post hoc" in s["b1_post_hoc_note"]


def test_b1_var_n_iota_is_in_rad2(b1_source: tuple[Path, list[tuple[Any, ...]]], tmp_path: Path) -> None:
    """The known interferer projects to n_iota,k = EPS mean(sin phi) over record k, in rad: its variance over the
    post-onset records whose integration the stream covers (all but the last) is the reported one."""
    src, _ = b1_source
    s = b1_run(tmp_path)
    stamps = clean_stamps(src, KNOWN_CH).astype(np.int64)
    m = np.arange(N_OUT)
    sums = np.add.reduceat(np.sin(PHI_RAD * np.sin(2 * np.pi * PHI_HZ * CLOCK.file_time(m))), stamps)[:-1]
    n_iota = EPS * sums / np.diff(stamps)  # records 0 .. K-2; the last runs past the stream
    post = e2._post(CLOCK.file_time(stamps[:-1]), SRC_SPEC["onset_s"])
    want = float(np.var(n_iota[post]))
    assert want == pytest.approx(EPS**2 * 0.5 * (1 - special.j0(2 * PHI_RAD)), rel=0.02)  # E sin^2(a sin x)
    for fam in ("noise", "swept_cw"):
        for lv in ("60", "70"):
            g = s["b1"][fam][lv][str(KNOWN_CH)]
            assert g["var_n_iota_rad2"] == pytest.approx(want, rel=1e-9)
            assert g["n_iota_records"] == int(post.sum())


def test_b1_rebuilds_each_units_interference_and_e2s_prediction(
    b1_source: tuple[Path, list[tuple[Any, ...]]], tmp_path: Path
) -> None:
    """n_iota comes from the unit's own disturbances (the jammer draw E2 recorded for the unit, not one rebuilt from the
    manifest's reordered spec) through mode A's front end at the segment's N0; E2's prediction from it is the saved
    pred_cyc (here exactly: the same stream as E2's projection)."""
    src, calls = b1_source
    s = b1_run(tmp_path)
    want = {(u.family, u.level, u.seed) for u in ran_grid() if u.block == "grid" and u.mode == "A"}
    assert want != {(u.family, u.level, u.seed) for u in e2.plan_grid(SRC_SPEC, "A") if u.block == "grid"
                    and u.mode == "A"}  # fmt: skip
    got = set()
    for iq, ds, fe, n0 in calls:
        assert fe.mode == "A" and fe.decim == DECIM and fe.scale == SURVEY["scale"] and n0 == SURVEY["n0"]
        assert iq.fs_hz == FS
        jam = [d for d in ds if d.kind != "ghost"]
        assert len(jam) == 1 and jam[0].onset_s == SRC_SPEC["onset_s"]
        got.add((jam[0].kind, jam[0].level_db_hz, jam[0].seed))
    assert got == want
    for fam in ("noise", "swept_cw"):
        for lv in ("60", "70"):
            for ch in ("2", "3"):
                g = s["b1"][fam][lv][ch]
                assert g["pred_max_abs_diff_cyc"] == 0.0 and g["pred_rms_diff_cyc"] == 0.0 and g["pred_sd_cyc"] > 0


def test_b1_refuses_a_saved_r_it_does_not_recompute(
    b1_source: tuple[Path, list[tuple[Any, ...]]], tmp_path: Path
) -> None:
    src, _ = b1_source
    udir = src / "units" / "A-grid-swept_cw-70-A"
    rec = json.loads((udir / "unit.json").read_text())
    rec["residual"]["31"]["r"] += 1e-9
    (udir / "unit.json").write_text(json.dumps(rec))
    e2.mark_done(udir, DESCRIBE, ["ghost_ch2.parquet", "ghost_ch3.parquet", "unit.json"])
    with pytest.raises(ValueError, match="A-grid-swept_cw-70-A.*r"):
        b1_run(tmp_path)


def test_b1_refuses_a_unit_record_that_is_not_the_planned_unit(
    b1_source: tuple[Path, list[tuple[Any, ...]]], tmp_path: Path
) -> None:
    """Only the seed may differ from the plan rebuilt from the manifest (the families' order); any other field means
    the record is not the unit the plan names."""
    src, _ = b1_source
    udir = src / "units" / "A-grid-noise-60-A"
    rec = json.loads((udir / "unit.json").read_text())
    rec["unit"]["level"] = 70.0
    (udir / "unit.json").write_text(json.dumps(rec))
    e2.mark_done(udir, DESCRIBE, ["ghost_ch2.parquet", "ghost_ch3.parquet", "unit.json"])
    with pytest.raises(ValueError, match="A-grid-noise-60-A.*not the planned unit"):
        b1_run(tmp_path)


def test_b1_in_parallel_equals_b1_in_sequence(
    b1_source: tuple[Path, list[tuple[Any, ...]]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Worker processes give the same numbers as one process, on the injector's own interference (the recording is not
    read, so a source of the right length suffices), and b1 leaves the source run as it was, in either path."""
    monkeypatch.setattr(e2post, "interference", pipeline.interference)
    with (tmp_path / "long.iq").open("wb") as f:
        f.truncate(2 * DECIM * N_OUT)  # sparse: one I and one Q byte per sample, never read
    big = IQSource(path=tmp_path / "long.iq", item_type="ibyte", fs_hz=FS, sample_start=0, sample_end=DECIM * N_OUT)
    monkeypatch.setattr(e2, "recording_source", lambda spec: (big, CLOCK))
    before = snapshot(b1_source[0])
    one = b1_run(tmp_path / "one", workers=1)["b1"]
    two = b1_run(tmp_path / "two", workers=3)["b1"]
    assert snapshot(b1_source[0]) == before
    assert one == two
    assert all(math.isfinite(g["var_n_iota_rad2"]) and g["var_n_iota_rad2"] > 0
               for f in one.values() for c in f.values() for g in c.values())  # fmt: skip


# ------------------------------------------------------------- B1 post hoc: the record-level decomposition (task 3)
LOOP = CarrierLoop.gnsssdr(35.0, 1e-3)  # E2's default loop (e0b.DEFAULT_BW_HZ, T_S)
K = 60_000  # records
Cplx = NDArray[np.complex128]


def cplx(x: Any) -> Cplx:
    return np.asarray(x, dtype=np.complex128)


def tone(f_hz: float, amp: float, phase: float) -> Cplx:
    return cplx(amp * np.exp(1j * (2 * np.pi * f_hz * 1e-3 * np.arange(K) + phase)))


def costas(z: Cplx) -> np.ndarray:
    return np.arctan(z.imag / z.real)


def predict(x: np.ndarray) -> np.ndarray:
    return LOOP.filter(np.nan_to_num(x)) / (2 * np.pi)


def circular(rng: np.random.Generator, var: float, size: int) -> Cplx:
    """Complex white Gaussian noise with `var` per component."""
    return cplx(np.sqrt(var) * (rng.standard_normal(size) + 1j * rng.standard_normal(size)))


def test_the_lag_scan_finds_an_injected_shift() -> None:
    """e = the prediction delayed by k records (+ a little noise): the scan's minimum is at k, and the sign of the
    correlation at lag 0 follows the sign of e."""
    rng = np.random.default_rng(3)
    p = predict(rng.standard_normal(20_000))
    for k in (-5, -2, 0, 1, 4):
        e = np.roll(p, k) + rng.normal(0.0, 0.1 * p.std(), p.size)
        s = e2post.lag_scan(e, p, np.arange(p.size), max_lag=5)
        assert s["lags"] == list(range(-5, 6)) and len(s["r"]) == 11
        assert s["argmin_lag"] == k and s["r"][k + 5] == min(s["r"])
    assert e2post.lag_scan(p, p, np.arange(p.size), max_lag=5)["corr_sign_lag0"] == 1
    assert e2post.lag_scan(-p, p, np.arange(p.size), max_lag=5)["corr_sign_lag0"] == -1


def test_the_lag_scan_refuses_epochs_that_skip_a_record() -> None:
    """A lag is a shift by records: on epochs with a gap in record index, a shift of the selected series would not be
    one, so the scan refuses them (r's epochs are contiguous on E2-A, checked in task 3's fix round 1)."""
    p = predict(np.random.default_rng(3).standard_normal(5_000))
    records = np.r_[np.arange(2_000), np.arange(2_001, 5_001)]
    with pytest.raises(ValueError, match="record"):
        e2post.lag_scan(p, p, records, max_lag=5)


def test_eta_n_is_the_white_gain_for_white_n_and_the_loop_gain_at_a_tone() -> None:
    """eta_n = Var(H n_Q) / Var(n_Q): noise_gain for white n_Q (to sampling), |H(f)|^2 for a tone at f (the transient
    is skipped); the share below 50 Hz is 0.1 for white n and nil for a 433 Hz tone, whose complex spectrum peaks at
    433 Hz."""
    rng = np.random.default_rng(4)
    on = np.ones(K, dtype=bool)
    s = e2post.n_iota_spectrum(circular(rng, 0.01, K), on, LOOP)
    assert s["eta_w"] == LOOP.noise_gain()
    assert s["eta_n"] == pytest.approx(LOOP.noise_gain(), rel=0.06)
    assert s["share_below_50_hz"] == pytest.approx(0.1, abs=0.01)
    f = 433.0
    s = e2post.n_iota_spectrum(tone(f, 0.05, 0.7), on, LOOP)
    assert s["eta_n"] == pytest.approx(abs(LOOP.freq_response(np.array([2 * np.pi * f * 1e-3]))[0]) ** 2, rel=0.01)
    assert abs(s["peak_hz"] - f) < 0.25 and s["share_below_50_hz"] < 1e-4


def synthetic(rng: np.random.Generator, n: Cplx, var_w: float, g: float) -> tuple[Cplx, np.ndarray]:
    """(P~, dPhi): a normalised clean prompt 1 + w, w circular with var_w per component, and the twin difference the
    linear loop makes of the Costas discriminator's every-order response to g n."""
    pt = 1.0 + circular(rng, var_w, n.size)
    return pt, predict(costas(pt + g * n) - costas(pt))


def test_the_decomposition_of_a_synthetic_discriminator() -> None:
    """dPhi = H [atan2q(P~ + g n) - atan2q(P~)]: the every-order prediction leaves nothing; the first-order one in n
    (lin) leaves the second order; B1's H n_Q leaves the thermal x interference products as well, whose measured share
    is the closed form X / (1 + X), X = Var w (1 + Var n_I / Var n_Q) eta_w / eta_n, to first order in Var w. A 450 Hz
    tone (eta_n << eta_w) gives a large X, white n X ~ 2 Var w. At FGI's Var w (0.0245 here) the next order shows: X
    rises by 1 + 4 Var w (e2post.decomposition derives it), so the share X' / (1 + X') rises by
    (1 + 4 Var w) (1 + X) / (1 + X (1 + 4 Var w)): 5 % for the tone (X ~ 0.8), 9 % for white n (X ~ 0.05)."""
    rng = np.random.default_rng(5)
    g = 1.03
    on = np.ones(K, dtype=bool)
    for var_w in (0.002, 0.0245):
        for n in (tone(450.0, 0.03, 0.4), circular(rng, 2e-4, K)):
            pt, e = synthetic(rng, n, var_w, g)
            cn0 = 10 * np.log10(0.5 / (var_w * 1e-3))  # 1 / (2 C/N0 T) = var_w
            d = e2post.decomposition(e, predict, n, pt, g, on, cn0_dbhz=cn0, loop=LOOP)
            assert d["r_exact"] == 0.0 and d["g"] == g
            assert d["var_w_i"] == pytest.approx(var_w, rel=0.02) and d["var_w_i_over_thermal"] == pytest.approx(
                1, rel=0.02
            )
            assert d["second_share"] < 0.01 and d["r_lin"] < 0.2 * d["r_b1"]
            x2 = d["x"] * (1 + 4 * d["var_w_i"])
            if var_w < 0.01:
                assert d["cross_share"] == pytest.approx(d["x_over_1px"], rel=0.05)
            else:
                assert d["cross_share"] == pytest.approx(x2 / (1 + x2), rel=0.05) and d["cross_share"] > d["x_over_1px"]
            # B1 at gain 1 leaves the products, the second order and the gain error (1 - 1/g) of the rest
            rest = d["cross_share"] + d["second_share"]
            assert d["r_b1"] == pytest.approx(rest + (1 - 1 / g) ** 2 * (1 - rest), rel=0.02)


def test_the_prompt_dumped_with_the_next_record_is_this_records_integration() -> None:
    """GNSS-SDR dumps record k's integration [stamp_k, stamp_k+1) with record k+1, as P = sum x conj(replica) (read
    off E2-A, task 3): from prompts dumped that way (with the data bits), lin explains the thermal x interference
    products; shifted by one record more, or conjugated, the prompt's noise is unrelated and lin does worse than B1."""
    rng = np.random.default_rng(6)
    var_w, g, a_hat = 0.0245, 1.03, 42_000.0
    n = circular(rng, 2e-3, K)
    pt, e = synthetic(rng, n, var_w, g)
    bits = np.repeat(np.where(rng.random(K // 20 + 1) < 0.5, -1.0, 1.0), 20)[:K]
    dumped = cplx(np.r_[a_hat * (1 + 0.1j), a_hat * bits[:-1] * pt[:-1]])  # record k+1 carries record k's integration
    on = np.ones(K, dtype=bool)
    on[-1] = False  # the last record's integration is dumped with a record that does not exist

    def r_lin(prompts: Cplx) -> float:
        p, a = e2post.normalised_prompts(e2post.record_prompts(prompts.real, prompts.imag), on)
        assert a == pytest.approx(a_hat, rel=0.01)
        return e2post.decomposition(e, predict, n, p, g, on, cn0_dbhz=45.0, loop=LOOP)["r_lin"]

    r_b1 = e2post.decomposition(e, predict, n, pt, g, on, cn0_dbhz=45.0, loop=LOOP)["r_b1"]
    assert r_lin(dumped) < 0.1 * r_b1
    assert r_lin(cplx(np.roll(dumped, 1))) > r_b1 and r_lin(cplx(np.roll(dumped, -1))) > r_b1
    assert r_lin(cplx(np.conj(dumped))) > r_b1
