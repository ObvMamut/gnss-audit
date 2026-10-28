"""Real-IF to complex baseband (P2b part 2, task 8; docs/7 section 4.2): the sign of the mix, chunk independence, DC."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from gnssaudit.inject.realif import GRID, OUT_SCALE, _taps, to_baseband

FS = 26e6
IF = 6.39e6
DECIM = 4
TAPS = 129


def write_real(path: Path, x: np.ndarray) -> None:
    path.write_bytes(np.clip(np.round(x), -127, 127).astype(np.int8).tobytes())


def read_iq16(path: Path) -> NDArray[np.complex128]:
    v = np.fromfile(path, dtype="<i2").astype(np.float64)
    return v[0::2] + 1j * v[1::2]


def run(src: Path, dst: Path, n: int, chunk: int = 1 << 22, **kw: float) -> str:
    return to_baseband(src, dst, fs_hz=FS, if_hz=IF, decim=DECIM, taps=TAPS, sample_end=n, chunk=chunk, **kw)


def test_a_tone_above_the_if_comes_out_at_plus_its_offset_with_the_image_suppressed(tmp_path: Path) -> None:
    n = 1 << 21
    t = np.arange(n) / FS
    write_real(tmp_path / "x.bin", 40 * np.cos(2 * np.pi * (IF + 1e3) * t + 0.3))
    run(tmp_path / "x.bin", tmp_path / "y.iq", n, scale=100.0)
    y = read_iq16(tmp_path / "y.iq")[200:]  # past the filter's start-up
    assert y.size > 100_000
    # correlate with the two complex exponentials at +1 kHz and -1 kHz: the first is the signal, the second its mirror
    fs_out = FS / DECIM
    tt = np.arange(y.size) / fs_out
    plus = abs(np.sum(y * np.exp(-2j * np.pi * 1e3 * tt)))
    minus = abs(np.sum(y * np.exp(+2j * np.pi * 1e3 * tt)))
    assert (
        20 * np.log10(plus / minus) > 40.0
    )  # a real tone has no energy at the mirror frequency once the mix has the right sign
    assert abs(y).mean() > 100  # the tone is well above the integer rounding of the output


def test_the_output_is_the_same_bytes_for_any_chunk_size(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    n = (1 << 20) + 12_345  # not a multiple of the chunk or the block
    write_real(tmp_path / "x.bin", rng.choice([-3, -1, 1, 3], size=n))
    a = run(tmp_path / "x.bin", tmp_path / "a.iq", n, chunk=1 << 18)
    b = run(tmp_path / "x.bin", tmp_path / "b.iq", n, chunk=1 << 22)
    assert a == b == hashlib.sha256((tmp_path / "a.iq").read_bytes()).hexdigest()
    assert (tmp_path / "a.iq").read_bytes() == (tmp_path / "b.iq").read_bytes()
    assert (tmp_path / "a.iq").stat().st_size == 4 * -(-n // DECIM)


def test_dc_in_the_input_does_not_reach_the_output_band(tmp_path: Path) -> None:
    n = 1 << 20
    write_real(tmp_path / "x.bin", np.full(n, 100.0))
    run(tmp_path / "x.bin", tmp_path / "y.iq", n, scale=100.0)
    y = read_iq16(tmp_path / "y.iq")[200:]
    ref = 100.0 * 100.0  # counts the DC would have had if it had passed
    assert np.abs(y).max() < ref * 1e-2  # below -40 dB of itself


def test_2_bit_samples_neither_underflow_nor_clip(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 1 << 20
    write_real(tmp_path / "x.bin", rng.choice([-3, -1, 1, 3], size=n))
    run(tmp_path / "x.bin", tmp_path / "y.iq", n)
    y = read_iq16(tmp_path / "y.iq")[200:]
    assert 1000 < y.real.std() < 8000 and np.abs(y).max() < 32767


def test_a_clipping_output_is_refused(tmp_path: Path) -> None:
    n = 1 << 16
    write_real(tmp_path / "x.bin", np.full(n, 127.0) * np.cos(2 * np.pi * IF * np.arange(n) / FS))
    with pytest.raises(ValueError, match="clip"):
        run(tmp_path / "x.bin", tmp_path / "y.iq", n)


def test_a_tone_in_the_stop_band_is_removed(tmp_path: Path) -> None:
    n = 1 << 20
    t = np.arange(n) / FS
    write_real(tmp_path / "pass.bin", 40 * np.cos(2 * np.pi * (IF + 1e6) * t))
    write_real(
        tmp_path / "stop.bin", 40 * np.cos(2 * np.pi * (IF + 5e6) * t)
    )  # 5 MHz from the carrier: beyond 3.25 MHz
    run(tmp_path / "pass.bin", tmp_path / "p.iq", n, scale=100.0)
    run(tmp_path / "stop.bin", tmp_path / "s.iq", n, scale=100.0)
    p, s = read_iq16(tmp_path / "p.iq")[200:], read_iq16(tmp_path / "s.iq")[200:]
    assert 20 * np.log10(np.abs(p).mean() / np.abs(s).mean()) > 40.0


def reference(x: NDArray[np.int8], scale: float) -> NDArray[np.int16]:
    """An independent one-shot conversion: mix the whole array, convolve with the taps (zero history), keep every
    4th sample from index 0, scale, round."""
    n = np.arange(x.size)
    mixed = x.astype(np.float64) * np.exp(-2j * np.pi * IF * n / FS)
    y = np.convolve(mixed, _taps(TAPS, DECIM, FS))[: x.size][::DECIM] * scale
    out = np.empty(2 * y.size, dtype=np.int16)
    out[0::2], out[1::2] = np.round(y.real), np.round(y.imag)
    return out


def test_odd_chunks_a_single_read_and_an_independent_reference_agree(tmp_path: Path) -> None:
    """chunk 100_003 is a multiple of neither the filter block nor the decimation, and the input spans several blocks:
    a filter history that is lost between blocks changes the first outputs of every block (the mutation `prev` reset
    to zeros each block fails this test)."""
    rng = np.random.default_rng(11)
    n = 3 * GRID + 12_345
    x = rng.choice([-3, -1, 1, 3], size=n).astype(np.int8)
    (tmp_path / "x.bin").write_bytes(x.tobytes())
    ref = reference(x, OUT_SCALE)
    a = run(tmp_path / "x.bin", tmp_path / "a.iq", n, chunk=100_003)
    b = run(tmp_path / "x.bin", tmp_path / "b.iq", n, chunk=n)
    assert a == b
    got = np.fromfile(tmp_path / "a.iq", dtype="<i2")
    assert got.size == ref.size
    assert np.abs(got.astype(np.int64) - ref).max() <= 1  # the same values up to a rounding tie
    assert np.mean(got != ref) < 1e-3


def test_a_decimation_that_does_not_divide_the_block_is_refused(tmp_path: Path) -> None:
    (tmp_path / "x.bin").write_bytes(bytes(1000))
    with pytest.raises(ValueError, match="divide"):
        to_baseband(tmp_path / "x.bin", tmp_path / "y.iq", fs_hz=FS, if_hz=IF, decim=3, taps=127, sample_end=1000)
