"""The interference projected on the ghost, record by record (B1's n_iota, docs/3 section 2.2; P2b task 11):
n_iota,k = Im(iota_k) / (A N_k), with iota_k the interference despread by the ghost's whole modulation (data bits, code,
carrier) over the samples of record k's integration, [stamp_k, stamp_k+1). That window is the loop timing G-hz
identified on the binary (the record's response is H sinc(f T) e^(j 2 pi f T / 2), docs/9 section 5.4)."""

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from gnssaudit.inject import GhostSpec, ghost_modulation
from gnssaudit.truth.ghost import ghost_despread, ghost_projection
from gnssaudit.truth.rxtime import TwinClock

FS = 2.0e6
G = GhostSpec(prn=9, doppler_hz=1234.0, doppler_rate_hzps=0.2, cn0_db_hz=45.0, chip0=17.3, seed=4)


def test_a_copy_of_the_ghost_projects_as_its_phase_wobble() -> None:
    """x = eps s_g e^(j phi): despread by s_g it leaves eps e^(j phi) at every sample, so n_iota,k = (eps / A) mean(sin
    phi) over record k's integration (exact: bits and code cancel in s_g conj(s_g)). The last record's integration
    runs past the stream, and stays NaN; chunk boundaries do not matter."""
    n = np.arange(0, int(0.2 * FS), dtype=np.int64)
    phi = 0.3 * np.sin(2 * np.pi * 7.0 * n / FS)
    eps, amp = 0.05, 2.0
    x = eps * ghost_modulation(G, n, FS) * np.exp(1j * phi)
    stamps = np.arange(1000, n.size, 2000, dtype=np.uint64)  # 1 ms records from an odd start
    cut = 123_457
    chunks = iter([(0, x[:cut]), (cut, x[cut:])])
    got = ghost_projection(chunks, G, stamps, TwinClock.identity(FS), amp)
    s = stamps.astype(np.int64)
    want = [eps / amp * np.mean(np.sin(phi[a:b])) for a, b in zip(s[:-1], s[1:], strict=True)]
    np.testing.assert_allclose(got[:-1], want, rtol=1e-9, atol=1e-13)
    assert np.isnan(got[-1])


def test_a_decimated_mid_file_stream_is_despread_on_the_native_grid() -> None:
    """Output sample m of a decimated twin stands for native sample start + decim m - delay (truth.rxtime.TwinClock):
    the replica follows that index."""
    clock = TwinClock(fs_native_hz=FS, sample_start=40_003, decim=4, delay=64.0)
    m = np.arange(0, 50_000, dtype=np.int64)
    native = (40_003 + 4 * m - 64).astype(np.int64)
    y = 0.05 * ghost_modulation(G, native, FS) * np.exp(1j * 0.2)
    stamps = np.arange(0, m.size, 500, dtype=np.uint64)
    got = ghost_projection(iter([(0, y)]), G, stamps, clock, 1.0)
    np.testing.assert_allclose(got[:-1], 0.05 * np.sin(0.2), rtol=1e-9)


def test_a_record_the_stream_does_not_cover_is_not_projected() -> None:
    """A stream that starts or stops inside an integration gives no value there (NaN), never a partial one."""
    n = np.arange(0, 20_000, dtype=np.int64)
    x = 0.05 * ghost_modulation(G, n, FS)
    stamps = np.arange(0, 20_000, 2000, dtype=np.uint64)
    got = ghost_projection(iter([(500, x[500:15_000])]), G, stamps, TwinClock.identity(FS), 1.0)
    assert np.isnan(got[0]) and np.isnan(got[7]) and np.isnan(got[-1])
    np.testing.assert_allclose(got[1:7], 0.0, atol=1e-12)


def test_the_whole_despread_record_holds_both_parts() -> None:
    """ghost_despread keeps iota_k / (A N_k) whole: for the copy of the ghost, (eps / A) mean(e^(j phi)) over record k,
    so its real part is (eps / A) mean(cos phi); its imaginary part is ghost_projection's, to the bit."""
    n = np.arange(0, int(0.1 * FS), dtype=np.int64)
    phi = 0.3 * np.sin(2 * np.pi * 7.0 * n / FS) + 0.2
    eps, amp = 0.05, 2.0
    x = eps * ghost_modulation(G, n, FS) * np.exp(1j * phi)
    stamps = np.arange(1000, n.size, 2000, dtype=np.uint64)
    got = ghost_despread(iter([(0, x)]), G, stamps, TwinClock.identity(FS), amp)
    s = stamps.astype(np.int64)
    want = [eps / amp * np.mean(np.exp(1j * phi[a:b])) for a, b in zip(s[:-1], s[1:], strict=True)]
    np.testing.assert_allclose(got[:-1], want, rtol=1e-9, atol=1e-13)
    assert np.isnan(got[-1])
    im = ghost_projection(iter([(0, x)]), G, stamps, TwinClock.identity(FS), amp)
    np.testing.assert_array_equal(got.imag[:-1], im[:-1])


def _ghost_projection_e8cbc82(
    interference: Iterable[tuple[int, NDArray[np.complex128]]],
    ghost: GhostSpec,
    stamps: NDArray[np.integer],
    clock: TwinClock,
    amplitude: float,
) -> NDArray[np.float64]:
    """truth.ghost.ghost_projection as E2 run 1 used it (frozen at e8cbc82, before ghost_despread split it)."""
    s = np.asarray(stamps, dtype=np.int64)
    k = s.size
    acc = np.zeros(k, dtype=np.complex128)
    cnt = np.zeros(k, dtype=np.int64)
    for m0, y in interference:
        m = np.arange(m0, m0 + y.size, dtype=np.int64)
        idx = np.searchsorted(s, m, side="right") - 1
        ok = (idx >= 0) & (idx < k - 1)
        if not ok.any():
            continue
        native = np.rint(clock.native_index(m[ok])).astype(np.int64)
        p = y[ok] * np.conj(ghost_modulation(ghost, native, clock.fs_native_hz))
        acc += np.bincount(idx[ok], weights=p.real, minlength=k) + 1j * np.bincount(
            idx[ok], weights=p.imag, minlength=k
        )
        cnt += np.bincount(idx[ok], minlength=k)
    out = np.full(k, np.nan)
    whole = np.zeros(k, dtype=bool)
    whole[:-1] = (cnt[:-1] == np.diff(s)) & (cnt[:-1] > 0)
    out[whole] = acc[whole].imag / (amplitude * cnt[whole])
    return out


def test_ghost_projection_is_bitwise_the_one_e2_run_1_used() -> None:
    """E2 run 2 calls ghost_projection: on a fixed noise-like stream through a decimated mid-file clock, in uneven
    chunks, with records the stream does not cover, it is the frozen pre-split function to the bit (NaNs included)."""
    rng = np.random.default_rng(20261017)
    clock = TwinClock(fs_native_hz=FS, sample_start=40_003, decim=4, delay=64.0)
    m_out = 120_000
    y = 0.7 * (rng.standard_normal(m_out) + 1j * rng.standard_normal(m_out))
    stamps = np.cumsum(rng.integers(480, 521, size=260)).astype(np.uint64)  # uneven records, some past the stream
    cuts = [0, 7, 33_331, 65_536, 90_001, m_out]

    def chunks() -> Iterable[tuple[int, NDArray[np.complex128]]]:
        return iter([(a, y[a:b]) for a, b in zip(cuts[:-1], cuts[1:], strict=True)])

    got = ghost_projection(chunks(), G, stamps, clock, 0.37)
    want = _ghost_projection_e8cbc82(chunks(), G, stamps, clock, 0.37)
    assert np.isnan(want).any() and np.isfinite(want).sum() > 200
    np.testing.assert_array_equal(got, want)
    assert got.tobytes() == want.tobytes()
