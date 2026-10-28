"""S04: second-order and distributional structure of despread interference (F4, N5, N6; docs/4 A3, P4).

Each class is synthesised at 2.046 Msps, despread against GPS C/A PRN 5 at 1500 Hz Doppler, and analysed in the
nominal signal frame (theta = 0, A = 1): m = Re(eta), n = Im(eta). The open-loop error is e = H_phi n (linear
regime). For each class we report:
* propriety: Var(m)/Var(n) and |E eta^2| / E|eta|^2;
* the loop-band coherence |S_mn|^2 / (S_m S_n) (zero iff the baseband spectrum is symmetric there; F4);
* dependence beyond correlation: corr(m^2, n^2) of the loop-filtered pair (zero for independent Gaussian);
* shape: excess kurtosis of e, and the coverage of a Gaussian 95 % interval that uses the exact variance;
* N6: the variance reduction of e achieved by the causal Wiener prediction of n from m.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any, cast

import numpy as np
from scipy import signal, stats

from gnssaudit.sim import despread as ds
from gnssaudit.uq.loop import CarrierLoop
from review._common import save

FS = 2.046e6
F_D = 1234.5  # not a multiple of 500 Hz: 1500 Hz makes the per-block rotation exactly pi
PRN = 5
SECONDS = 400
CHUNK_S = 2


def waveform(name: str, rng: np.random.Generator) -> Callable[[int, int], np.ndarray]:
    """Return gen(n, start) producing the class waveform for samples [start, start + n)."""
    ph = float(rng.uniform(0, 2 * np.pi))
    off = float(rng.uniform(0, 1))
    if name == "wideband_noise":
        return lambda n, start: ds.white_noise(rng, n, 1.0)
    if name.startswith("cw_beat_"):
        beat = float(name.removeprefix("cw_beat_").removesuffix("hz"))
        f = F_D + 3000.0 + beat
        return lambda n, start: ds.cw(n, FS, f, 1.0, ph, start)
    if name == "swept_slow":  # 100 kHz span, 10 ms period
        return lambda n, start: ds.swept_cw(n, FS, -5e4, 1e5, 1e-2, 1.0, ph, off * 1e-2, start)
    if name == "swept_fast":  # 1 MHz span, 100 us period
        return lambda n, start: ds.swept_cw(n, FS, -5e5, 1e6, 1e-4, 1.0, ph, off * 1e-4, start)
    if name == "chirp_inband":  # PPD-like: full in-band sweep every 20 us
        return lambda n, start: ds.swept_cw(n, FS, -1.0e6, 2.0e6, 2e-5, 1.0, ph, off * 2e-5, start)
    if name == "pulsed_noise":  # 30 us bursts at 1.3 kHz
        return lambda n, start: ds.pulsed(ds.white_noise(rng, n, 1.0), FS, 3e-5, 1.3e3, off, start)
    if name == "pulsed_noise_lowprf":  # 10 us bursts at 100 Hz: one block in ten is hit
        return lambda n, start: ds.pulsed(ds.white_noise(rng, n, 1.0), FS, 1e-5, 1e2, off, start)
    if name == "pulsed_cw":  # 100 us CW bursts at 0.7 kHz, beat 7 Hz
        f = F_D + 3000.0 + 7.0
        return lambda n, start: ds.pulsed(ds.cw(n, FS, f, 1.0, ph, start), FS, 1e-4, 7e2, off, start)
    raise ValueError(name)


def despread_series(name: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gen = waveform(name, rng)
    dsp = ds.Despreader(prn=PRN, fs_hz=FS, doppler_hz=F_D)
    out = []
    blocks_per_chunk = int(CHUNK_S / dsp.t_s)
    for c in range(SECONDS // CHUNK_S):
        start = c * blocks_per_chunk * dsp.ns
        x = gen(blocks_per_chunk * dsp.ns, start)
        out.append(dsp.despread(x, first_block=c * blocks_per_chunk))
    return np.concatenate(out)


def analyse(name: str, seed: int) -> dict[str, object]:
    eta = despread_series(name, seed)
    eta = eta - eta.mean()
    eta = 0.05 * eta / np.sqrt(np.mean(np.abs(eta) ** 2) / 2)  # sigma_m about 0.05 rad (linear regime)
    m, n = eta.real, eta.imag
    res: dict[str, object] = {
        "class": name,
        "var_m_over_var_n": float(np.var(m) / np.var(n)),
        "pseudo_cov_ratio": float(np.abs(np.mean(eta**2)) / np.mean(np.abs(eta) ** 2)),
    }
    per_loop = {}
    for bn in (5.0, 35.0):
        loop = CarrierLoop.gnsssdr(bn, 1e-3)
        f, smn = signal.csd(m, n, fs=1000.0, nperseg=8192)
        _, sm = signal.welch(m, fs=1000.0, nperseg=8192)
        _, sn = signal.welch(n, fs=1000.0, nperseg=8192)
        h2 = np.abs(loop.freq_response(2 * np.pi * f / 1000.0)) ** 2
        coh = np.abs(smn) ** 2 / (sm * sn)
        coh_loop = float(np.sum(h2 * sn * coh) / np.sum(h2 * sn))
        hm = loop.filter(m)[5000:]
        e = loop.filter(n)[5000:]
        sig = e.std()
        z = stats.norm.ppf(0.975)
        # N6: causal Wiener prediction of e from the past and present of hm (FIR, least squares)
        lags = 200
        xm = np.lib.stride_tricks.sliding_window_view(hm, lags)[:, ::-1]
        y = e[lags - 1 :]
        half = y.size // 2
        coef, *_ = np.linalg.lstsq(xm[:half], y[:half], rcond=None)
        resid = y[half:] - xm[half:] @ coef
        per_loop[f"bn{int(bn)}"] = {
            "loop_band_coherence_mn": coh_loop,
            "corr_sq_hm_e": float(np.corrcoef(hm**2, e**2)[0, 1]),
            "iw_bias_var_hm_over_var_e": float(np.var(hm) / np.var(e)),
            "excess_kurtosis_e": float(stats.kurtosis(e)),
            "gauss95_coverage_exact_var": float(np.mean(np.abs(e) <= z * sig)),
            "gauss99_coverage_exact_var": float(np.mean(np.abs(e) <= stats.norm.ppf(0.995) * sig)),
            "gauss999_coverage_exact_var": float(np.mean(np.abs(e) <= stats.norm.ppf(0.9995) * sig)),
            "wiener_var_reduction": float(1 - np.var(resid) / np.var(y[half:])),
            "ks_hm_vs_e": float(cast(Any, stats.ks_2samp(hm / hm.std(), e / sig)).statistic),
        }
    res["loops"] = per_loop
    return res


def main() -> None:
    classes = ["wideband_noise", "cw_beat_7hz", "cw_beat_-7hz", "cw_beat_60hz", "cw_beat_0hz",
               "swept_slow", "swept_fast", "chirp_inband", "pulsed_noise", "pulsed_noise_lowprf", "pulsed_cw"]  # fmt: skip
    with ProcessPoolExecutor(max_workers=len(classes)) as ex:
        rows = list(ex.map(analyse, classes, range(500, 500 + len(classes))))
    for r in rows:
        print(r["class"], "vm/vn=%.3f pc=%.3f" % (r["var_m_over_var_n"], r["pseudo_cov_ratio"]))
        loops = r["loops"]
        assert isinstance(loops, dict)
        for k, v in loops.items():
            print("   ", k, {kk: round(vv, 3) for kk, vv in v.items()})
    save("s04_structured_interference", {"seconds": SECONDS, "prn": PRN, "doppler_hz": F_D, "rows": rows})


if __name__ == "__main__":
    main()
