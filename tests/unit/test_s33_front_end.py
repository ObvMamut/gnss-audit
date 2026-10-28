"""S33's own pieces (P2c task 4; R-7, rulings P5/T4e and T4a-T4d): the streamed Welch, the PSD-shaping FIR and its
match check at the FIR's resolution, the segment-indexed seeds, the true front end's recursive AGC (`_agc` itself, its
state carried through chunks), the stop rule on every SE s34 reads, reading run 2, and the JSON's keys."""

import hashlib
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import signal

from gnssaudit.inject import disturbance_seed, render, requantise
from gnssaudit.inject.pipeline import _agc, _state
from review import _common
from review import s33_front_end_prediction as s33
from review.s28_twin_pair_quantisation import FAMILIES, FGI, FS, SelfCheckError

SIGMA = 1.5
FREQS = np.fft.fftshift(np.fft.fftfreq(s33.NPERSEG, 1 / FS))  # the Welch grid of a 2^16 segment at 30.69 MS/s


# ------------------------------------------------------------------------------------------------ streamed Welch
def test_the_streamed_welch_is_scipys_welch_over_the_whole_stream() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(5000) + 1j * rng.standard_normal(5000)
    acc = s33.Welch(nperseg=256, fs=2.0)
    for a, b in [(0, 100), (100, 1377), (1377, 1500), (1500, 4321), (4321, 5000)]:
        acc.add(x[a:b])
    f, p, k = acc.result()
    fw, pw = signal.welch(x, fs=2.0, window="hann", nperseg=256, return_onesided=False)
    assert k == (5000 - 256) // 128 + 1
    np.testing.assert_allclose(f, np.fft.fftshift(fw))
    np.testing.assert_allclose(p, np.fft.fftshift(pw), rtol=1e-12)


# ------------------------------------------------------------------------------------------------ the FIR's bins
def test_bins_are_the_firs_frequency_bins_of_fs_over_4097() -> None:
    fc, pc = s33.fir_bin_average(FREQS, FREQS.copy(), 1)
    assert fc.size == s33.SHAPE_TAPS
    k = np.arange(s33.SHAPE_TAPS) - s33.SHAPE_TAPS // 2
    assert np.all(np.abs(fc - k * FS / s33.SHAPE_TAPS) <= FS / (2 * s33.SHAPE_TAPS))  # each centre inside its bin
    np.testing.assert_allclose(pc, fc)  # a linear PSD averages to the bin's mean frequency


def test_four_bin_cells_merge_consecutive_fir_bins() -> None:
    fc1, _ = s33.fir_bin_average(FREQS, np.ones_like(FREQS), 1)
    fc4, pc4 = s33.fir_bin_average(FREQS, np.ones_like(FREQS), 4)
    assert fc4.size == s33.SHAPE_TAPS // 4 + 1  # 1 024 cells of four bins, and the last bin alone
    np.testing.assert_allclose(pc4, 1.0)
    width = np.diff(fc4[:-1])
    np.testing.assert_allclose(width, 4 * FS / s33.SHAPE_TAPS, rtol=0.02)
    assert fc4[0] == pytest.approx(np.mean(fc1[:4]), rel=0.01)


# ------------------------------------------------------------------------------------------------ the shaping FIR
def test_a_flat_psd_gives_sigma_times_a_centred_impulse() -> None:
    h = s33.shaping_fir(FREQS, np.full(FREQS.size, 3e-7), SIGMA)
    want = np.zeros(s33.SHAPE_TAPS, dtype=np.complex128)
    want[s33.SHAPE_TAPS // 2] = SIGMA
    np.testing.assert_allclose(h, want, atol=1e-12)


def bump(f: np.ndarray) -> np.ndarray:
    """A smooth, asymmetric PSD: a 4 MHz-wide bump at +3 MHz on a floor."""
    return 1e-7 * (0.2 + np.exp(-(((f - 3e6) / 4e6) ** 2)))


def test_the_fir_has_4097_taps_and_sigma_per_component() -> None:
    h = s33.shaping_fir(FREQS, bump(FREQS), SIGMA)
    assert h.shape == (s33.SHAPE_TAPS,)
    assert np.sum(np.abs(h) ** 2) == pytest.approx(SIGMA**2, rel=1e-12)


def test_the_firs_response_follows_a_smooth_asymmetric_target() -> None:
    h = s33.shaping_fir(FREQS, bump(FREQS), SIGMA)
    response = np.fft.fftshift(np.abs(np.fft.fft(h, s33.NPERSEG)) ** 2) * 2 / FS  # the PSD of h * unit white noise
    m = s33.psd_match(FREQS, bump(FREQS), response, SIGMA)
    assert m["max_abs_db"] < 0.05
    assert m["passes"] is True
    i_pos, i_neg = np.argmin(np.abs(FREQS - 3e6)), np.argmin(np.abs(FREQS + 3e6))
    assert 10 * np.log10(response[i_pos] / response[i_neg]) > 3.0  # complex taps: +f and -f differ


def test_the_hann_window_smears_a_one_bin_line_over_its_main_lobe() -> None:
    """Why the check runs at four bins (ruling T4e): a line one FIR bin wide is lowered at one bin, not at four."""
    p = np.full(FREQS.size, 1e-7)
    fc = s33.fir_bin_average(FREQS, FREQS.copy(), 1)[0]
    k = int(np.argmin(np.abs(fc - 2e6)))
    in_bin = np.abs(FREQS - fc[k]) < FS / (2 * s33.SHAPE_TAPS)
    p[in_bin] *= 10 ** (1.5 / 10)  # +1.5 dB in one bin, as on segment B
    h = s33.shaping_fir(FREQS, p, SIGMA)
    response = np.fft.fftshift(np.abs(np.fft.fft(h, s33.NPERSEG)) ** 2) * 2 / FS
    assert s33.psd_match(FREQS, p, response, SIGMA, width=1)["max_abs_db"] > 0.5
    assert s33.psd_match(FREQS, p, response, SIGMA, width=4)["max_abs_db"] < 0.5


# ------------------------------------------------------------------------------------------------ the PSD check
def scaled(p: np.ndarray, sigma: float = SIGMA) -> np.ndarray:
    """p at the shaped noise's power: 2 sigma^2 in all."""
    return p * 2 * sigma**2 / (np.mean(p) * FS)


def test_the_check_compares_with_the_target_scaled_to_2_sigma_squared() -> None:
    p = bump(FREQS)
    m = s33.psd_match(FREQS, 7.0 * p, scaled(p), SIGMA)
    assert m["max_abs_db"] == pytest.approx(0.0, abs=1e-9)
    assert m["passes"] is True
    assert m["width_bins"] == 4
    assert m["cell_hz"] == pytest.approx(4 * FS / s33.SHAPE_TAPS)
    assert s33.psd_match(FREQS, p, 1.2 * scaled(p), SIGMA)["passes"] is False  # +0.79 dB everywhere


def raise_cell(p: np.ndarray, f0: float, db: float) -> np.ndarray:
    fc = s33.fir_bin_average(FREQS, FREQS.copy(), 4)[0]
    c = int(np.argmin(np.abs(fc - f0)))
    k = np.clip(np.rint(FREQS * s33.SHAPE_TAPS / FS).astype(int), -2048, 2048) + 2048
    out = p.copy()
    out[k // 4 == c] *= 10 ** (db / 10)
    return out


def test_a_cell_off_by_more_than_half_a_db_inside_8_mhz_fails() -> None:
    p = bump(FREQS)
    m = s33.psd_match(FREQS, p, raise_cell(scaled(p), 5e6, 0.6), SIGMA)
    assert m["passes"] is False
    assert m["max_abs_db"] == pytest.approx(0.6, abs=0.01)
    assert abs(m["at_hz"] - 5e6) < 4 * FS / s33.SHAPE_TAPS
    assert [round(x["f_hz"] / 1e3) for x in m["over_tol"]] == [round(m["at_hz"] / 1e3)]


def test_half_a_db_passes_and_cells_beyond_8_mhz_do_not_count() -> None:
    p = bump(FREQS)
    assert s33.psd_match(FREQS, p, raise_cell(scaled(p), 5e6, 0.499), SIGMA)["passes"] is True
    m = s33.psd_match(FREQS, p, raise_cell(scaled(p), 9e6, 3.0), SIGMA)
    assert m["passes"] is True
    assert m["band_hz"] == 8e6


# ------------------------------------------------------------------------------------------------ seeds
def test_every_seed_is_root_33_with_the_segment_index() -> None:
    for seg, idx in (("A", 0), ("B", 1)):
        tuples = s33.seed_tuples(seg, n_noise_keys=5)
        assert all(len(t) == 4 and t[0] == 33 and t[1] == idx for t in tuples)
        assert s33.seed_tuple(seg, s33.GHOST_CAT, 0) in tuples
        assert s33.seed_tuple(seg, s33.DITHER_CAT, 0) in tuples
        assert all(s33.seed_tuple(seg, s33.NOISE_CAT, k) in tuples for k in range(5))
        for fi, fam in enumerate(FAMILIES):
            assert all(s33.seed_tuple(seg, s33.JAMMER_CAT + fi, int(lv)) in tuples for lv in s33.LEVELS)
            assert s33.seed_tuple(seg, s33.JAMMER_CAT + fi, 0) in tuples  # the zero jammer of the self-check
            assert s33.jammer_seed(seg, fam, 60.0) == disturbance_seed(33, idx, s33.JAMMER_CAT + fi, 60)
        assert s33.ghost_seed(seg) == disturbance_seed(33, idx, s33.GHOST_CAT, 0)
        assert s33.dither_seed(seg) == disturbance_seed(33, idx, s33.DITHER_CAT, 0)


def test_no_seed_is_shared_between_segments() -> None:
    out = s33.seed_independence(s33.seed_tuples("A", 300), s33.seed_tuples("B", 300))
    assert out["shared"] == 0
    assert out["tuples"]["A"] == out["tuples"]["B"] > 300


def test_a_shared_seed_stops_the_study_even_through_seedsequence_padding() -> None:
    a = s33.seed_tuples("A", 3)
    with pytest.raises(SelfCheckError, match="shared"):
        s33.seed_independence(a, [*s33.seed_tuples("B", 3), a[0]])
    with pytest.raises(SelfCheckError, match="shared"):  # SeedSequence pads to four words: (33, 0) is (33, 0, 0, 0)
        s33.seed_independence([(33, 0, 0, 0)], [(33, 0)])


def test_the_noise_stream_draws_from_the_segments_seeds() -> None:
    h = np.zeros(s33.SHAPE_TAPS, dtype=np.complex128)
    h[s33.SHAPE_TAPS // 2] = SIGMA  # a pure delay of 2 048 samples
    chunk = 1 << 14
    for seg in ("A", "B"):
        (lo, x), *_ = list(s33.shaped_noise(2 * chunk, h, seg, chunk=chunk))
        rng = s33.noise_rng(seg, 1)
        w1 = rng.standard_normal(chunk) + 1j * rng.standard_normal(chunk)
        assert lo == 0 and x.size == chunk
        np.testing.assert_allclose(x[2048:2058], SIGMA * w1[:10], rtol=0, atol=1e-12)
    xa = next(s33.shaped_noise(chunk, h, "A", chunk=chunk))[1]
    xb = next(s33.shaped_noise(chunk, h, "B", chunk=chunk))[1]
    assert not np.allclose(xa, xb)


def test_the_cell_carries_the_segments_jammer_and_dither_seeds() -> None:
    tc = s33.make_true_cell("B", "swept_cw", 65.0, 1.8e-7, 1.52, 1.53)
    assert tc.cell.jammer.seed == s33.jammer_seed("B", "swept_cw", 65.0)
    assert tc.cell.fe.dither_seed == s33.dither_seed("B")
    assert tc.cell.fe.agc_tau_s == 0.01  # mode B as run 3 runs it
    assert tc.cell.fe.analog_sd == 1.52  # the synthetic recording's own estimate
    assert sorted(tc.agc) == list(s33.TAUS)
    assert all(fe.agc_tau_s == t for t, (fe, _) in tc.agc.items())
    assert s33.ghost("B").seed == s33.ghost_seed("B")


# ------------------------------------------------------------------------------------------------ the chains
def piece(lo: int, n: int, n0: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng([5, lo])
    x0 = SIGMA * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    t = np.arange(lo, lo + n) / FS
    return x0, math.sqrt(10**4.5 * n0) * np.exp(2j * np.pi * 1510.0 * t)


N0 = 1.8e-7


def test_at_zero_jammer_every_jammed_twin_is_its_clean_twin() -> None:
    tc = s33.make_true_cell("A", "noise", -math.inf, N0, 1.5, SIGMA)
    x0, g = piece(0, 1 << 16, N0)
    ys = dict(s33.chains(tc, 0, x0, g, N0))
    assert set(ys) == set(s33.CHAINS)
    for clean, jammed in s33.PAIRS.values():
        np.testing.assert_array_equal(ys[jammed], ys[clean])
    np.testing.assert_array_equal(ys["truth_clean"], requantise(x0 + g, FGI))


def test_the_true_front_end_is_agc_then_requantise_with_the_state_carried_through_chunks() -> None:
    n = 1 << 16
    x0a, ga = piece(0, n, N0)
    x0b, gb = piece(n, n, N0)
    split = s33.make_true_cell("A", "noise", 70.0, N0, 1.5, SIGMA)
    one = s33.make_true_cell("A", "noise", 70.0, N0, 1.5, SIGMA)
    first, second = dict(s33.chains(split, 0, x0a, ga, N0)), dict(s33.chains(split, n, x0b, gb, N0))
    whole = dict(s33.chains(one, 0, np.concatenate([x0a, x0b]), np.concatenate([ga, gb]), N0))
    for k in s33.CHAINS:
        np.testing.assert_array_equal(np.concatenate([first[k], second[k]]), whole[k])
    x0, g = np.concatenate([x0a, x0b]), np.concatenate([ga, gb])
    j = render(one.cell.jammer, 0, x0.size, FS, N0)
    for t in s33.TAUS:
        fe = one.agc[t][0]
        want = requantise(_agc(x0 + j + g, x0 + g, 0, fe, FS, _state(fe)), FGI)  # one pass over the whole stream
        np.testing.assert_array_equal(whole[f"truth_{t}"], want)
    assert not np.array_equal(whole["truth_0.001"], whole["truth_0.1"])


# ------------------------------------------------------------------------------------------------ statistics
def fake_z(n: int, seed: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z_ref = 1000.0
    base = z_ref * (1 + 0.01 * rng.standard_normal(n + 1)) + 1j * z_ref * 0.01 * rng.standard_normal(n + 1)
    z = {}
    for i, k in enumerate(s33.CHAINS):
        jam = 0.0 if k.endswith("clean") else 0.005 * (1 + 0.1 * i)
        z[k] = base + 1j * z_ref * jam * rng.standard_normal(n + 1)
    return z


def test_cell_stats_hold_every_ratio_s34_reads_keyed_by_tau() -> None:
    n = 2000
    st = s33.cell_stats(fake_z(n), 1000.0, n)
    taus = ["0.001", "0.01", "0.1"]
    for key in ("ratio_TA", "ratio_TA_se", "slope_TA", "b_over_truth", "b_over_truth_se"):
        assert list(st[key]) == taus
    for key in ("ratio_B", "ratio_B_se", "slope_B", "ratio_TA_stationary", "ratio_TA_stationary_se",
                "slope_TA_stationary", "n_windows", "length_s"):  # fmt: skip
        assert key in st
    assert st["n_windows"] == n
    assert all(math.isfinite(v) for v in st["ratio_TA"].values())


def test_cell_stats_take_each_tau_from_its_own_true_chain() -> None:
    from review.s28_twin_pair_quantisation import e2s_stats

    n, z = 2000, fake_z(2000)
    st = s33.cell_stats(z, 1000.0, n)
    pair = {k: z[k] for k in ("A_clean", "A", "truth_clean", "B_clean", "B")}
    want = e2s_stats({**pair, "truth": z["truth_0.1"]}, 1000.0, n)
    assert st["ratio_TA"]["0.1"] == want["ratio_truth"]
    assert st["ratio_TA_se"]["0.1"] == want["ratio_truth_se"]
    assert st["slope_TA"]["0.1"] == want["slope_truth"]
    assert st["ratio_B"] == want["ratio_B"] and st["ratio_B_se"] == want["ratio_B_se"]
    stat = e2s_stats({**pair, "truth": z["truth_stationary"]}, 1000.0, n)
    assert st["ratio_TA_stationary"] == stat["ratio_truth"]


# ------------------------------------------------------------------------------------------------ the stop rule
def stream(chunk: int, last: int, seen: list[int]) -> Iterator[tuple[int, str]]:
    done = 0
    while done < last:
        done = min(done + chunk, last)
        seen.append(done)
        yield done, f"state@{done}"


def stats_with(ses: dict[int, tuple[float, ...]], calls: list[int]) -> Any:
    """SE(R_B) and SE(R_TA) at 1, 10 and 100 ms per checkpoint."""

    def stats(z: str, n: int) -> dict[str, Any]:
        calls.append(n)
        b, *ta = ses[n]
        return {"n_windows": n, "ratio_B_se": b, "ratio_TA_se": dict(zip(["0.001", "0.01", "0.1"], ta, strict=True)),
                "ratio_TA_stationary_se": 0.5}  # fmt: skip

    return stats


def test_checkpoints_are_20_40_60_s() -> None:
    assert s33.CHECKPOINT_WINDOWS == (20_000, 40_000, 60_000)
    assert s33.SE_STOP == 0.02


def test_a_cell_stops_only_when_every_se_s34_reads_is_within_the_bar() -> None:
    seen: list[int] = []
    calls: list[int] = []
    ses = {20_000: (0.01, 0.01, 0.025, 0.01), 40_000: (0.02, 0.019, 0.02, 0.01), 60_000: (0.01, 0.01, 0.01, 0.01)}
    st = s33.adaptive(stream(5_000, 60_000, seen), stats_with(ses, calls), s33.CHECKPOINT_WINDOWS)
    assert st["n_windows"] == 40_000
    assert st["se_stop_reached"] is True
    assert seen[-1] == 40_000 and calls == [20_000, 40_000]
    assert [c["max_se"] for c in st["checkpoints"]] == [0.025, 0.02]
    assert st["checkpoints"][0]["ses"] == {"ratio_B": 0.01, "ratio_TA[0.001]": 0.01, "ratio_TA[0.01]": 0.025,
                                           "ratio_TA[0.1]": 0.01}  # fmt: skip


def test_the_stationary_se_does_not_hold_a_cell_and_mode_bs_does() -> None:
    st = s33.adaptive(stream(5_000, 60_000, []), stats_with({20_000: (0.01,) * 4}, []), s33.CHECKPOINT_WINDOWS)
    assert st["n_windows"] == 20_000  # SE(R_TA stationary) = 0.5 is reported, not a stop condition
    ses = {20_000: (0.03, 0.01, 0.01, 0.01), 40_000: (0.025, 0.01, 0.01, 0.01), 60_000: (0.021, 0.01, 0.01, 0.01)}
    st = s33.adaptive(stream(5_000, 60_000, []), stats_with(ses, []), s33.CHECKPOINT_WINDOWS)
    assert st["n_windows"] == 60_000
    assert st["se_stop_reached"] is False
    assert st["ratio_B_se"] == 0.021


def test_a_non_finite_se_never_stops_a_cell() -> None:
    ses = {20_000: (math.nan, 0.01, 0.01, 0.01), 40_000: (0.01,) * 4, 60_000: (0.01,) * 4}
    st = s33.adaptive(stream(5_000, 60_000, []), stats_with(ses, []), s33.CHECKPOINT_WINDOWS)
    assert st["n_windows"] == 40_000
    assert st["checkpoints"][0]["max_se"] == math.inf


def test_a_stream_shorter_than_the_last_checkpoint_is_refused() -> None:
    with pytest.raises(ValueError, match="before the last checkpoint"):
        s33.adaptive(stream(5_000, 45_000, []), stats_with({n: (0.05,) * 4 for n in (20_000, 40_000)}, []),
                     s33.CHECKPOINT_WINDOWS)  # fmt: skip


# ------------------------------------------------------------------------------------------------ T4a
def test_workers_follow_the_measured_peak_memory() -> None:
    assert s33.workers_for(27.0, 2.5) == 9
    assert s33.workers_for(60.0, 1.0) == 10
    with pytest.raises(ValueError, match="memory"):
        s33.workers_for(5.0, 3.0)


# ------------------------------------------------------------------------------------------------ run 2 (read only)
def write_run2(run: Path, run_id: str) -> bytes:
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"run_id": run_id}))
    segs = {s: {"analog_sd": 1.5 + i / 10, "n0_per_hz": 1.7e-7 + i * 1e-9, "line_excess_db": 2.0 + i,
                "front_end": {"noise": {"55.0": {"ratio": 0.8}}}} for i, s in enumerate("AB")}  # fmt: skip
    raw = json.dumps({"segments": segs}).encode()
    (run / "summary.json").write_bytes(raw)
    return raw


def test_run_2_gives_only_analog_sd_n0_and_the_line_excess(tmp_path: Path) -> None:
    raw = write_run2(tmp_path / "r", s33.RUN2_ID)
    got = s33.read_run2(tmp_path / "r", sha256=hashlib.sha256(raw).hexdigest())
    assert got["analog_sd"] == {"A": 1.5, "B": 1.6}
    assert got["n0_per_hz"] == {"A": 1.7e-7, "B": 1.7e-7 + 1e-9}
    assert got["line_excess_db"] == {"A": 2.0, "B": 3.0}
    assert "front_end" not in json.dumps(got) and "ratio" not in json.dumps(got)  # never its front-end ratios


def test_another_run_or_another_summary_is_refused(tmp_path: Path) -> None:
    raw = write_run2(tmp_path / "r", "000000000000")
    with pytest.raises(ValueError, match="not E2 run 2"):
        s33.read_run2(tmp_path / "r", sha256=hashlib.sha256(raw).hexdigest())
    write_run2(tmp_path / "s", s33.RUN2_ID)
    with pytest.raises(ValueError, match="SHA-256"):
        s33.read_run2(tmp_path / "s", sha256="0" * 64)


def test_run_2_is_pinned_by_its_summary_sha256() -> None:
    assert s33.RUN2_ID == "6214cf52489f"
    assert s33.RUN2_SHA256 == "4419fd70188279102a2ab862fda3ef17acdbb81018b8793918d165b11e962c99"


# ------------------------------------------------------------------------------------------------ the JSON
def test_cells_nest_by_segment_family_and_run_2s_level_keys() -> None:
    res = [{"segment": s, "family": f, "level_db_hz": lv, "ratio_B": 0.9} for s in "AB" for f in FAMILIES
           for lv in s33.LEVELS]  # fmt: skip
    cells = s33.nest_cells(res)
    assert list(cells) == ["A", "B"]
    assert list(cells["A"]) == list(FAMILIES)
    assert list(cells["B"]["chirp"]) == ["50.0", "55.0", "60.0", "65.0", "70.0", "75.0", "80.0", "85.0"]
    assert cells["B"]["noise"]["85.0"]["level_db_hz"] == 85.0


def test_a_segment_entry_holds_what_s34_reads() -> None:
    e = s33.segment_entry(n0_run2=1.77e-7, line_excess_run2=2.08, floor={"n0_per_hz": 1.8e-7, "line_excess_db": 1.0,
                          "analog_sd": 1.5}, match={"max_abs_db": 0.2}, match_1bin={"max_abs_db": 0.8})  # fmt: skip
    assert e["n0_run2"] == 1.77e-7 and e["n0_synthetic"] == 1.8e-7
    assert e["psd_match_db"] == 0.2 and e["psd_match_db_1bin"] == 0.8
    assert e["line_excess_db_synthetic"] == 1.0 and e["line_excess_db_run2"] == 2.08


def test_meta_inputs_go_into_the_results_meta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_common, "RESULTS", tmp_path)
    path = _common.save("sxx", {"cells": {}}, meta_extra={"inputs": {"run2": {"summary_sha256": "ab"}}})
    meta = json.loads(path.read_text())["meta"]
    assert meta["study"] == "sxx" and meta["inputs"] == {"run2": {"summary_sha256": "ab"}}
    path = _common.save("syy", {"cells": {}})  # every other study's call: the meta it always had, nothing added
    meta = json.loads(path.read_text())["meta"]
    assert set(meta) == {"study", "generated", "python", "platform", "numpy", "packages"}


# ------------------------------------------------------------------------------------------------ resuming
def test_a_stored_cell_is_reloaded_only_under_its_fingerprint(tmp_path: Path) -> None:
    path = s33.cell_path(tmp_path, "A", "noise", 50.0)
    assert path.name == "A-noise-50.0.json"
    assert s33.load_cell(path, "fp1") is None  # nothing stored yet
    s33.store_cell(path, "fp1", {"segment": "A", "ratio_B": 0.9, "checkpoints": [{"n_windows": 20_000}]})
    assert s33.load_cell(path, "fp1") == {"segment": "A", "ratio_B": 0.9, "checkpoints": [{"n_windows": 20_000}]}
    assert s33.load_cell(path, "fp2") is None  # another code or input: recomputed
    assert not list(tmp_path.glob("*.tmp"))  # written atomically


def test_the_fingerprint_follows_every_input_of_a_cell() -> None:
    h = np.arange(5, dtype=np.complex128)
    base = s33.cell_fingerprint(h, 1.5, 1.8e-7, 1.5046, (20_000, 40_000, 60_000), code="c")
    assert base == s33.cell_fingerprint(h.copy(), 1.5, 1.8e-7, 1.5046, (20_000, 40_000, 60_000), code="c")
    assert base != s33.cell_fingerprint(
        np.asarray(h + 1e-12, dtype=np.complex128), 1.5, 1.8e-7, 1.5046, (20_000, 40_000, 60_000), code="c"
    )
    assert base != s33.cell_fingerprint(h, 1.6, 1.8e-7, 1.5046, (20_000, 40_000, 60_000), code="c")
    assert base != s33.cell_fingerprint(h, 1.5, 1.9e-7, 1.5046, (20_000, 40_000, 60_000), code="c")
    assert base != s33.cell_fingerprint(h, 1.5, 1.8e-7, 1.5047, (20_000, 40_000, 60_000), code="c")
    assert base != s33.cell_fingerprint(h, 1.5, 1.8e-7, 1.5046, (20_000,), code="c")


# ------------------------------------------------------------------------------------------ provenance (fix round 1)
def test_the_code_hash_covers_every_file_and_the_numpy_and_scipy_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_text("x = 1\n")
    b.write_text("y = 2\n")
    paths = [str(a), str(b)]
    base = s33.code_hash(paths)
    assert base == s33.code_hash(list(reversed(paths)))  # order-free
    b.write_text("y = 3\n")
    assert s33.code_hash(paths) != base
    b.write_text("y = 2\n")
    assert s33.code_hash(paths) == base
    monkeypatch.setattr(np, "__version__", "0.0.0")
    assert s33.code_hash(paths) != base


def test_the_first_party_modules_are_every_loaded_module_of_the_repository() -> None:
    paths = s33.first_party_paths()
    for want in ("review/s33_front_end_prediction.py", "review/s28_twin_pair_quantisation.py", "review/_common.py",
                 "src/gnssaudit/inject/pipeline.py", "src/gnssaudit/inject/families.py",
                 "src/gnssaudit/experiments/e2_frontend.py", "src/gnssaudit/experiments/e2.py"):  # fmt: skip
        assert any(p.endswith(want) for p in paths), want
    assert all(("/src/gnssaudit/" in p or "/review/" in p) and ".venv" not in p for p in paths)
    assert paths == sorted(paths)


def test_a_worker_refuses_code_that_changed_since_the_start(tmp_path: Path) -> None:
    a = tmp_path / "a.py"
    a.write_text("x = 1\n")
    start = s33.code_hash([str(a)])
    s33.require_code([str(a)], start)
    a.write_text("x = 2\n")
    with pytest.raises(SelfCheckError, match="changed"):
        s33.require_code([str(a)], start)


def test_a_worker_refuses_a_loaded_module_outside_the_hashed_set() -> None:
    paths = s33.first_party_paths()
    s33.require_loaded_within(paths)
    with pytest.raises(SelfCheckError, match="not in the hashed set"):
        s33.require_loaded_within([p for p in paths if not p.endswith("pipeline.py")])


def test_the_cell_fingerprint_follows_the_code_hash() -> None:
    h = np.arange(5, dtype=np.complex128)
    args = (h, 1.5, 1.8e-7, 1.5046, (20_000, 40_000, 60_000))
    assert s33.cell_fingerprint(*args, code="c1") == s33.cell_fingerprint(*args, code="c1")
    assert s33.cell_fingerprint(*args, code="c1") != s33.cell_fingerprint(*args, code="c2")


def git(root: Path, *args: str) -> str:
    import subprocess

    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args], cwd=root,
                          capture_output=True, text=True, check=True).stdout  # fmt: skip


def test_the_git_state_names_the_head_and_any_change_to_the_hashed_paths(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    (tmp_path / "m.py").write_text("x = 1\n")
    (tmp_path / "other.txt").write_text("notes\n")
    git(tmp_path, "add", "m.py", "other.txt")
    git(tmp_path, "commit", "-q", "-m", "c")
    head = git(tmp_path, "rev-parse", "HEAD").strip()
    st = s33.git_state([str(tmp_path / "m.py")], root=tmp_path)
    assert st == {"head": head, "dirty": []}
    (tmp_path / "other.txt").write_text("changed\n")  # outside the hashed paths: not counted
    assert s33.git_state([str(tmp_path / "m.py")], root=tmp_path)["dirty"] == []
    (tmp_path / "m.py").write_text("x = 2\n")
    assert s33.git_state([str(tmp_path / "m.py")], root=tmp_path)["dirty"] == [" M m.py"]


def test_a_stored_cell_is_synced_to_disk_with_its_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    synced: list[int] = []
    real = os.fsync

    def fsync(fd: int) -> None:
        synced.append(fd)
        real(fd)

    monkeypatch.setattr(s33.os, "fsync", fsync)
    s33.store_cell(tmp_path / "c" / "A-noise-50.0.json", "fp", {"ratio_B": 0.9})
    assert len(synced) == 2  # the file before the rename, the directory after it
    assert s33.load_cell(tmp_path / "c" / "A-noise-50.0.json", "fp") == {"ratio_B": 0.9}
