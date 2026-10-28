"""P2b task 7: front-end mode B (an AGC, then requantisation at the native rate, before decimation; docs/8 section 5),
injection on several cores with a bitwise-identical output, the interference-only stream for the first-order B1
prediction, the common noise floor both twins keep, and fixed jammer timing for the conditional oracle."""

import hashlib
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import special

from gnssaudit.inject import (
    RNG_BLOCK,
    Disturbance,
    FrontEndSpec,
    IQSource,
    QuantSpec,
    SinkSpec,
    clean_twin,
    estimate_analog_sd,
    inject,
    interference,
    reconstruct_analog,
    requantise,
    unit_waveform,
)

FS = 1.0e6
FGI = QuantSpec(max_level=4, zero_level=False)  # FGI JT2023: 3 bits per component, values -4..-1 and 1..4
RNG = np.random.default_rng(11)


def three_bit_file(path: Path, n: int, seed: int = 1) -> IQSource:
    """Interleaved int8 I/Q on the FGI alphabet: Gaussian noise of SD 1.53 counts per component, quantised."""
    rng = np.random.default_rng(seed)
    v = 1.53 * rng.standard_normal(2 * n)
    q = (np.sign(v) * np.clip(np.floor(np.abs(v) + 0.5), 1, 4)).astype(np.int8)
    q[q == 0] = 1
    q.tofile(path)
    return IQSource(path=path, item_type="ibyte", fs_hz=FS, sample_start=0, sample_end=n)


def noise(level: float, onset_s: float = 0.5, seed: int = 4, **extra: float) -> Disturbance:
    return Disturbance("noise", {"bw_hz": 2e5, "f_hz": 0.0, **extra}, onset_s, 0.01, level, seed)


def mode_b(tau_s: float, decim: int = 1, taps: int = 1, scale: float = 1000.0, seed: int = 3) -> FrontEndSpec:
    return FrontEndSpec(decim, taps, scale, mode="B", agc_tau_s=tau_s, quant=FGI, dither_seed=seed, analog_sd=1.53)


def bussgang_fgi(sd: float) -> float:
    """Small-signal gain of the FGI quantiser on Gaussian input of SD sd: the sum of its jumps times the input density
    at each threshold (2 at 0, 1 at +-1.5, +-2.5, +-3.5)."""
    pdf = lambda t: np.exp(-0.5 * (t / sd) ** 2) / (sd * np.sqrt(2 * np.pi))  # noqa: E731
    return float(2 * pdf(0.0) + 2 * (pdf(1.5) + pdf(2.5) + pdf(3.5)))


def run(src: IQSource, ds: list[Disturbance], fe: FrontEndSpec, out: Path, **kw: int) -> tuple[str, np.ndarray]:
    r = inject(src, ds, fe, SinkSpec("file", out), n0_per_hz=1.0 / FS, **kw)
    return r.output_sha256, np.fromfile(out, dtype="<i2")


def test_requantisation_maps_to_the_nearest_level_of_the_alphabet() -> None:
    v = np.array([0.2, -0.2, 1.4, 1.6, 2.4, -3.7, 9.0, -0.01])
    got = requantise(v + 1j * v[::-1], FGI)
    np.testing.assert_array_equal(got.real, [1, -1, 1, 2, 2, -4, 4, -1])  # no zero level: the sign decides
    np.testing.assert_array_equal(got.imag, [-1, 4, -4, 2, 2, 1, -1, 1])
    eight = requantise(np.array([0.4, -0.6, 200.0]) + 0j, QuantSpec(max_level=127, zero_level=True))
    np.testing.assert_array_equal(eight.real, [0, -1, 127])


@pytest.mark.parametrize("tau_s", [1e-3, 1e-2, 1e-1])
def test_a_zero_gain_mode_b_twin_is_the_recording(tmp_path: Path, tau_s: float) -> None:
    """The AGC holds the recording's own smoothed power, so with nothing added its gain is exactly 1; the analog value
    reconstructed within each sample's quantisation cell requantises to the stored level."""
    src = three_bit_file(tmp_path / "s.bin", 400_000)
    raw = np.fromfile(src.path, dtype=np.int8).astype(np.float64)
    _, a = run(src, clean_twin([noise(70.0)]), FrontEndSpec(1, 1, 1000.0), tmp_path / "a.bin")
    _, b = run(src, clean_twin([noise(70.0)]), mode_b(tau_s), tmp_path / "b.bin")
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, 1000 * raw)


def eight_bit_file(path: Path, n: int, sd: float = 20.0) -> IQSource:
    """Interleaved int8 I/Q: Gaussian noise of SD sd counts, fine enough that requantisation noise is negligible."""
    v = np.clip(np.round(sd * np.random.default_rng(2).standard_normal(2 * n)), -127, 127).astype(np.int8)
    v.tofile(path)
    return IQSource(path=path, item_type="ibyte", fs_hz=FS, sample_start=0, sample_end=n)


def test_the_agc_brings_a_power_step_back_to_the_recording_s_power(tmp_path: Path) -> None:
    """About +20 dB of injected noise: after 5 tau the output RMS is back within 1 % of the recording's own RMS (the
    level the recorder's AGC held), on an 8-bit alphabet that does not clip."""
    src = eight_bit_file(tmp_path / "s.bin", 600_000)  # complex power 800 counts^2
    raw = np.fromfile(src.path, dtype=np.int8).astype(float)
    fe = FrontEndSpec(1, 1, 1.0, mode="B", agc_tau_s=0.01, quant=QuantSpec(127, True), dither_seed=5, analog_sd=20.0)
    _, out = run(src, [noise(110.0, onset_s=0.2)], fe, tmp_path / "o.bin")  # 10^11 x 1e-6 = 1e5 counts^2: +21 dB
    x = out[0::2].astype(float) + 1j * out[1::2]
    late = x[int((0.2 + 0.01 + 5 * 0.01) * FS) :]
    want = np.sqrt(np.mean(raw[0::2] ** 2 + raw[1::2] ** 2))
    assert np.sqrt(np.mean(np.abs(late) ** 2)) == pytest.approx(want, rel=0.01)


def test_a_signal_below_one_quantisation_step_survives_mode_b(tmp_path: Path) -> None:
    """P2b task 11: requantising a recording that is already quantised erased every addition below half a step (a
    45 dB-Hz ghost changed no sample of FGI segment A). With each sample's analog value drawn from the Gaussian noise,
    truncated to its cell, a small addition survives in the mean with the quantiser's Bussgang gain (1.019 at SD 1.53):
    what the real 3-bit front end applied to the satellites the recording holds."""
    src = three_bit_file(tmp_path / "s.bin", 400_000)
    raw = np.fromfile(src.path, dtype=np.int8).astype(float)
    tone = Disturbance("cw", {"f_hz": 1234.0}, 0.0, 0.0, 47.0, 9)  # 10^4.7 x 1e-6: amplitude 0.22 counts
    _, out = run(src, [tone], mode_b(0.01, scale=1.0), tmp_path / "o.bin")
    y = (out[0::2] - raw[0::2]) + 1j * (out[1::2] - raw[1::2])
    w = unit_waveform(tone, 0, 400_000, FS)
    gain = np.vdot(w, y) / np.vdot(w, w) / np.sqrt(10**4.7 / FS)
    assert abs(gain) == pytest.approx(bussgang_fgi(1.53), abs=0.03) and abs(np.angle(gain)) < 0.03


def test_mode_b_output_does_not_depend_on_the_chunk_size(tmp_path: Path) -> None:
    src = three_bit_file(tmp_path / "s.bin", 5_000_000)
    fe = mode_b(0.01, decim=4, taps=129, scale=3000.0)
    shas = {run(src, [noise(75.0)], fe, tmp_path / f"c{c}.bin", chunk=c)[0] for c in (1 << 18, 1 << 20, 1 << 22)}
    assert len(shas) == 1


def test_injection_on_several_cores_is_bitwise_the_same(tmp_path: Path) -> None:
    """Pieces start on the block grid; each rebuilds the decimation FIR's history from the block before it. The onset
    ramp (1.0 to 1.01 s) and piece boundaries fall anywhere."""
    src = three_bit_file(tmp_path / "s.bin", 5_000_000)
    fe = FrontEndSpec(4, 129, 300.0)
    ds = [noise(75.0, onset_s=1.0)]
    shas = {w: run(src, ds, fe, tmp_path / f"w{w}.bin", workers=w)[0] for w in (1, 2, 5)}
    assert len(set(shas.values())) == 1


def test_mode_b_on_several_cores_is_bitwise_the_same(tmp_path: Path) -> None:
    """Mode B's AGC state is a recursion on the sub-block powers of the stream before the AGC: a first pass on several
    cores gives each piece its starting state, a second writes the pieces. The bytes and the achieved levels equal the
    one-core run's, from a start off the block grid, with the onset ramp (1.0 to 1.01 s) inside a piece."""
    three_bit_file(tmp_path / "s.bin", 2_000_000)
    src = IQSource(path=tmp_path / "s.bin", item_type="ibyte", fs_hz=FS, sample_start=12_345, sample_end=2_000_000)
    ds = [noise(75.0, onset_s=1.0), Disturbance("ghost", {"prn": 5, "doppler_hz": 1500.0}, 0.0, 0.0, 45.0, 7)]
    fe = mode_b(0.01, decim=4, taps=129, scale=3000.0)
    reports = [inject(src, ds, fe, SinkSpec("file", tmp_path / f"w{w}.bin"), n0_per_hz=1.0 / FS, workers=w)
               for w in (1, 2, 5)]  # fmt: skip
    assert len({r.output_sha256 for r in reports}) == 1
    assert all(r.achieved_db_hz == reports[0].achieved_db_hz for r in reports)


class _InProcess:
    """ProcessPoolExecutor's stand-in: each call runs when it is submitted, in this process, so a probe sees it."""

    def __init__(self, *_: object, **__: object) -> None:
        pass

    def __enter__(self) -> _InProcess:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def submit(self, fn: Callable[..., Any], /, *args: Any) -> Future[Any]:
        done: Future[Any] = Future()
        done.set_result(fn(*args))
        return done


def test_each_mode_b_piece_starts_from_the_one_core_pass_s_agc_state_and_fir_history_bit_for_bit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review of perf-inject, Important 2: the bytes do not show the AGC state a piece is handed (with every piece's
    starting state shifted by 10^6 ulps, the test above still passed). A probe on `_front_end` records the AGC's stream
    and recording powers (float.hex) and the FIR history (as uint64) before every block of the one-core pass. The same
    probe then watches `inject` on 2, 3, 5 and 7 pieces, run in this process (the stand-in executor runs each
    `_agc_pass` and `_piece` as it is submitted, so `_piece`'s own prologue runs, not a copy of it). Each piece must be
    handed the one-core pass's state at the start of the block before its cut, and must start its first block, after
    the prologue, with the one-core pass's AGC powers and FIR history at that sample. tau = 1 ms moves the state fast,
    from sub-block to sub-block; it also forgets the handed state within the block before the cut, so the handed state
    is compared too. The cuts include one right after the first, partial block (handed no state: the pre-charge) and
    later ones."""
    from gnssaudit.inject import pipeline

    three_bit_file(tmp_path / "s.bin", 8 * RNG_BLOCK)
    src = IQSource(path=tmp_path / "s.bin", item_type="ibyte", fs_hz=FS, sample_start=12_345,
                   sample_end=8 * RNG_BLOCK - 777)  # fmt: skip
    ds = [noise(75.0, onset_s=0.2), Disturbance("ghost", {"prn": 5, "doppler_hz": 1500.0}, 0.0, 0.0, 45.0, 7)]
    fe = mode_b(0.001, decim=4, taps=129, scale=3000.0)

    def hx(v: float | None) -> str:
        return "None" if v is None else float(v).hex()

    seen: dict[int, tuple[str, str, bytes]] = {}
    handed: list[tuple[int, tuple[float | None, float | None]]] = []
    front_end, piece = pipeline._front_end, pipeline._piece

    def probe(x: Any, lo: int, start: int, f: FrontEndSpec, fs: float, st: Any, ref: Any = None) -> Any:
        seen[lo] = (hx(st.agc_power), hx(st.ref_power), st.hist.view(np.uint64).tobytes())
        return front_end(x, lo, start, f, fs, st, ref)

    def watched(s: IQSource, d: Any, f: FrontEndSpec, n0: float, lo: int, hi: int, part: str, agc: Any) -> Any:
        handed.append((lo, agc))
        return piece(s, d, f, n0, lo, hi, part, agc)

    monkeypatch.setattr(pipeline, "_front_end", probe)
    monkeypatch.setattr(pipeline, "_piece", watched)
    monkeypatch.setattr(pipeline, "ProcessPoolExecutor", _InProcess)
    inject(src, ds, fe, SinkSpec("file", tmp_path / "w1.bin"), n0_per_hz=1.0 / FS)
    one_core = dict(seen)
    assert sorted(one_core) == [lo for lo, _ in pipeline._blocks(src.sample_start, src.sample_end)]
    handed_none = handed_float = 0
    for w in (2, 3, 5, 7):
        seen.clear()
        handed.clear()
        inject(src, ds, fe, SinkSpec("file", tmp_path / f"w{w}.bin"), n0_per_hz=1.0 / FS, workers=w)
        assert len(handed) == w and handed[0] == (src.sample_start, (None, None))
        for lo, agc in handed[1:]:
            before = one_core[max(src.sample_start, lo - RNG_BLOCK)]
            assert (hx(agc[0]), hx(agc[1])) == before[:2], f"{w} pieces: the AGC state handed to the piece at {lo}"
            assert seen[lo][:2] == one_core[lo][:2], f"{w} pieces: the AGC state at {lo}, after the prologue"
            assert seen[lo][2] == one_core[lo][2], f"{w} pieces: the FIR history at {lo}, after the prologue"
            handed_none, handed_float = handed_none + (agc[0] is None), handed_float + (agc[0] is not None)
    assert handed_none >= 1 and handed_float >= 3


def test_the_interference_stream_is_what_injection_adds(tmp_path: Path) -> None:
    """On an all-zero recording the injected twin is the interference alone; the stream gives it in float, decimated
    the same way, ghosts and the common floor left out."""
    zeros = tmp_path / "z.bin"
    np.zeros(2 * 1_000_000, dtype=np.int8).tofile(zeros)
    src = IQSource(path=zeros, item_type="ibyte", fs_hz=FS, sample_start=0, sample_end=1_000_000)
    fe = FrontEndSpec(4, 129, 300.0)
    ds = [noise(75.0), noise(60.0, seed=9, common=1.0)]
    _, out = run(src, ds[:1], fe, tmp_path / "o.bin")
    stream = np.concatenate([b for _, b in interference(src, ds, fe, n0_per_hz=1.0 / FS, block=1 << 18)])
    expect = np.empty(2 * stream.size)
    expect[0::2], expect[1::2] = np.rint(300.0 * stream.real), np.rint(300.0 * stream.imag)
    np.testing.assert_array_equal(out, expect)


def test_the_clean_twin_keeps_the_common_floor() -> None:
    kept, silenced = clean_twin([noise(60.0, common=1.0), noise(70.0)])
    assert kept.level_db_hz == 60.0 and silenced.level_db_hz == -np.inf


def test_a_fixed_offset_fixes_the_sweep_timing_but_not_its_phase() -> None:
    """Conditional oracle (F8): realizations share the sweep's timing and differ in carrier phase only."""
    p = {"span_hz": 2e5, "period_s": 1e-3, "f_hz": 0.0}
    a = unit_waveform(Disturbance("swept_cw", {**p, "offset_s": 3e-4}, 0.0, 0.0, 0.0, 1), 0, 4096, FS)
    b = unit_waveform(Disturbance("swept_cw", {**p, "offset_s": 3e-4}, 0.0, 0.0, 0.0, 2), 0, 4096, FS)
    ratio = a / b
    assert np.allclose(ratio, ratio[0], atol=1e-9) and abs(ratio[0] - 1) > 1e-3
    c = unit_waveform(Disturbance("swept_cw", p, 0.0, 0.0, 0.0, 2), 0, 4096, FS)  # offset drawn from the seed
    assert not np.allclose(a / c, (a / c)[0], atol=1e-3)


def test_sha_is_of_the_bytes_written(tmp_path: Path) -> None:
    src = three_bit_file(tmp_path / "s.bin", 300_000)
    sha, _ = run(src, [noise(70.0)], FrontEndSpec(4, 129, 300.0), tmp_path / "o.bin", workers=3)
    assert sha == hashlib.sha256((tmp_path / "o.bin").read_bytes()).hexdigest()


def test_the_analog_noise_sd_is_estimated_from_the_level_fractions(tmp_path: Path) -> None:
    """FGI's levels +-1 cover |v| < 1.5: their share gives the SD of the Gaussian noise the recorder quantised."""
    src = three_bit_file(tmp_path / "s.bin", 400_000)
    assert estimate_analog_sd(src, FGI) == pytest.approx(1.53, rel=0.02)
    eight = eight_bit_file(tmp_path / "e.bin", 400_000, sd=20.0)
    assert estimate_analog_sd(eight, QuantSpec(127, True)) == pytest.approx(20.0, rel=0.02)


def test_reconstruct_analog_is_mode_b_s_reconstruction_renamed() -> None:
    """P2b part 2, task 2: the reconstruction is public as reconstruct_analog, so s28 can compare a true analog with
    mode B's model. A rename keeps the draws: the values below are the former _reconstruct's output on samples that
    straddle a block of the uniform grid, and each requantises to its stored level."""
    v = np.array([1, -1, 2, -4, 4, -3, 3, -2], dtype=float)
    x = v + 1j * v[::-1]
    got = reconstruct_analog(x, FGI, 1.504, 5, RNG_BLOCK - 4)
    old = np.array([1.0141571216140048 - 1.526140174357058j, -0.12086667410785876 + 2.7077340303682647j,
                    1.6417843594640238 - 2.951200828516j, -3.9541352348663015 + 4.12298329392401j,
                    4.476717545497036 - 4.1537058423876525j, -3.061894063725707 + 1.7811385328695863j,
                    2.520305924197005 - 0.20848442615010346j, -2.061561143699332 + 0.025644190248872534j])  # fmt: skip
    np.testing.assert_allclose(got, old, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(requantise(got, FGI), x)


# ------------------------------------------------------------------ P2b part 2, task 6: tabulated cell probabilities
def _ref_cells(v: np.ndarray, q: QuantSpec) -> tuple[np.ndarray, np.ndarray]:
    """Frozen copy of the per-sample cell bounds before task 6."""
    k = np.abs(v)
    lo = np.where(k >= q.max_level, q.max_level - 0.5, k - 0.5)
    hi = np.where(k >= q.max_level, np.inf, k + 0.5)
    if not q.zero_level:
        lo = np.where(k <= 1, 0.0, lo)
    neg = v < 0
    lo_s, hi_s = np.where(neg, -hi, lo), np.where(neg, -lo, hi)
    if q.zero_level:
        zero = k == 0
        lo_s, hi_s = np.where(zero, -0.5, lo_s), np.where(zero, 0.5, hi_s)
    return lo_s, hi_s


def _ref_reconstruct(x: NDArray[np.complex128], q: QuantSpec, sd: float, seed: int, lo: int) -> NDArray[np.complex128]:
    """Frozen copy of reconstruct_analog before task 6 (two ndtr calls per sample)."""
    from scipy import special

    n = x.size
    u = np.empty(2 * n)
    for b in range(lo // RNG_BLOCK, (lo + n - 1) // RNG_BLOCK + 1):
        a, e = max(lo, b * RNG_BLOCK), min(lo + n, (b + 1) * RNG_BLOCK)
        blk = np.random.default_rng([seed, 0xD17, b]).random(2 * RNG_BLOCK)
        u[2 * (a - lo) : 2 * (e - lo)] = blk[2 * (a - b * RNG_BLOCK) : 2 * (e - b * RNG_BLOCK)]

    def one(v: np.ndarray, w: np.ndarray) -> np.ndarray:
        c_lo, c_hi = _ref_cells(v, q)
        p_lo, p_hi = special.ndtr(c_lo / sd), special.ndtr(c_hi / sd)
        r = sd * special.ndtri(p_lo + w * (p_hi - p_lo))
        return np.clip(r, c_lo, np.nextafter(c_hi, -np.inf))

    return one(x.real, u[0::2]) + 1j * one(x.imag, u[1::2])


INT8_ZERO = QuantSpec(max_level=7, zero_level=True)


def _levels(q: QuantSpec, n: int, seed: int) -> NDArray[np.complex128]:
    rng = np.random.default_rng(seed)
    lv = np.arange(-q.max_level, q.max_level + 1)
    if not q.zero_level:
        lv = lv[lv != 0]
    edge = np.tile(lv, 4)  # every level appears, including the outer cells, +-1 and 0 where it exists
    re = np.concatenate([edge, rng.choice(lv, n - edge.size)]).astype(np.float64)
    im = np.concatenate([edge[::-1], rng.choice(lv, n - edge.size)]).astype(np.float64)
    return re + 1j * im


@pytest.mark.parametrize(("q", "sd"), [(FGI, 1.53), (FGI, 0.9), (INT8_ZERO, 2.7)])
def test_reconstruct_analog_matches_the_per_sample_reference_bitwise(q: QuantSpec, sd: float) -> None:
    x = _levels(q, 1 << 20, 5)
    for lo in (0, RNG_BLOCK - 7):
        assert reconstruct_analog(x, q, sd, 9, lo).tobytes() == _ref_reconstruct(x, q, sd, 9, lo).tobytes()


def test_reconstruct_analog_tabulates_the_cells_once_per_alphabet_and_sd() -> None:
    from gnssaudit.inject import pipeline

    t = pipeline._cell_table(FGI, 1.53)
    assert t is pipeline._cell_table(FGI, 1.53)
    c_lo, p_lo, p_hi, c_top = t
    c_lo_ref, c_hi_ref = _ref_cells(np.arange(-4.0, 5.0), FGI)
    assert c_lo.size == 2 * FGI.max_level + 1
    assert np.array_equal(c_lo, c_lo_ref) and np.array_equal(c_top, np.nextafter(c_hi_ref, -np.inf))
    assert np.array_equal(p_lo, special.ndtr(c_lo_ref / 1.53)) and np.array_equal(p_hi, special.ndtr(c_hi_ref / 1.53))


def test_reconstruct_analog_levels_beyond_the_alphabet_clamp_like_the_formula_and_fractions_refuse() -> None:
    far = np.array([-9.0 + 5.0j, 6.0 - 12.0j, 4.0 + 4.0j])
    assert reconstruct_analog(far, FGI, 1.53, 2, 0).tobytes() == _ref_reconstruct(far, FGI, 1.53, 2, 0).tobytes()
    zero = np.array([0.0 + 1.0j, 2.0 + 0.0j])  # 0 on an alphabet without a zero level: the formula's cell (0, 0.5)
    assert reconstruct_analog(zero, FGI, 1.53, 2, 0).tobytes() == _ref_reconstruct(zero, FGI, 1.53, 2, 0).tobytes()
    for bad in (np.array([1.5 + 1.0j]), np.array([np.nan + 1.0j])):
        with pytest.raises(ValueError, match="integer"):
            reconstruct_analog(bad, FGI, 1.53, 2, 0)


# ------------------------------------------------------------------------------- final review, fix wave 1 (R2)
def test_the_cached_cell_tables_cannot_be_written() -> None:
    """Final review R2, minor 4: the four arrays are shared by every later call with the same alphabet and SD, so a
    caller cannot change them."""
    from gnssaudit.inject import pipeline

    for arr in pipeline._cell_table(FGI, 1.53):
        assert not arr.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            arr[0] = 0.0


def test_reconstruct_analog_takes_complex128_only() -> None:
    """Final review R2, minor 5: the bitwise equality with the per-sample formula holds for complex128 input (the
    pipeline's); complex64 input would give other floats, so it is refused."""
    x = _levels(FGI, 64, 3)
    single: Any = x.astype(np.complex64)  # what the annotation already rules out, checked at run time
    with pytest.raises(TypeError, match="complex128"):
        reconstruct_analog(single, FGI, 1.53, 9, 0)
    assert reconstruct_analog(x, FGI, 1.53, 9, 0).tobytes() == _ref_reconstruct(x, FGI, 1.53, 9, 0).tobytes()
