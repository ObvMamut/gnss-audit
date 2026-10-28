"""S07: the pilot-data differential witness (N1).

For signals with a pilot and a data component on one carrier, the data prompt is correlated with the pilot's
replica (GNSS-SDR track_pilot = true, dll_pll_veml_tracking.cc:767-776; dumped as Prompt I/Q, lines 1869-1876).
In the linear regime both discriminators see the same carrier error, so

    Delta d = d_data - d_pilot = n_data - n_pilot     at every frequency,

with no loop high-pass and no amplitude. If the two components carry equal power and their despread
disturbances are uncorrelated with equal spectra, sigma_PDW^2 = (1/2) mean_W[(H_phi Delta d)^2] is unbiased for
Var(H_phi n_pilot), the noise-driven error, with no window-mean projection because Delta d has zero mean.

Part A (assumption check): coherence and power ratio of the two codes' despread interference, per class.
Part B (estimator): identifiable fraction of IW versus PDW on stochastic coloured disturbances, nonlinear closed loop.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import signal

from gnssaudit.sim import despread as ds
from gnssaudit.sim.correlator import TrackingConfig, thermal_noise, track
from gnssaudit.uq import iw
from gnssaudit.uq.loop import CarrierLoop
from review._common import save
from review.s02_window_and_p5 import ar1
from review.s04_structured_interference import CHUNK_S, F_D, FS, waveform

T = 1e-3
PILOT_PRN, DATA_PRN = 5, 17
SECONDS_A = 120


def part_a(name: str, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    gen = waveform(name, rng)
    dp = ds.Despreader(prn=PILOT_PRN, fs_hz=FS, doppler_hz=F_D)
    dd = ds.Despreader(prn=DATA_PRN, fs_hz=FS, doppler_hz=F_D)
    bpc = int(CHUNK_S / dp.t_s)
    ep, ed = [], []
    for c in range(SECONDS_A // CHUNK_S):
        x = gen(bpc * dp.ns, c * bpc * dp.ns)
        ep.append(dp.despread(x, c * bpc))
        ed.append(dd.despread(x, c * bpc))
    p = np.concatenate(ep)
    d = np.concatenate(ed)
    p, d = p - p.mean(), d - d.mean()
    out: dict[str, object] = {"class": name}
    for bn in (5.0, 35.0):
        loop = CarrierLoop.gnsssdr(bn, T)
        hp = loop.filter(p.imag)[5000:]
        hd = loop.filter(d.imag)[5000:]
        kw = {"fs": 1000.0, "nperseg": 4096, "return_onesided": False}
        f, spd = signal.csd(p, d, **kw)
        _, spp = signal.welch(p, **kw)
        _, sdd = signal.welch(d, **kw)
        cxy = np.abs(spd) ** 2 / (spp * sdd)
        h2 = np.abs(loop.freq_response(2 * np.pi * np.abs(f) * T)) ** 2
        out[f"bn{int(bn)}"] = {
            "loop_band_coherence_pd": float(np.sum(h2 * cxy) / np.sum(h2)),
            "power_ratio_data_over_pilot_in_loop_band": float(np.var(hd) / np.var(hp)),
            "corr_of_error_components": float(np.corrcoef(hp, hd)[0, 1]),
        }
    return out


def part_b(args: tuple[float, float | None, int]) -> dict[str, object]:
    bn, fc, seed = args
    rng = np.random.default_rng(seed)
    k, r, burn = 400_000, 2, 20_000
    n_win = 2000
    cfg = TrackingConfig(bn_hz=bn, t_s=T, pilot=True)
    loop = CarrierLoop.gnsssdr(bn, T)
    th_p = thermal_noise(rng, (r, k), 10**4.5, T)
    th_d = thermal_noise(rng, (r, k), 10**4.5, T)
    sig_n = np.sqrt(1 / (2 * 10**4.5 * T))

    def colour() -> np.ndarray:
        if fc is None:
            return np.zeros((r, k), complex)
        x = np.stack([ar1(rng, k, fc) + 1j * ar1(rng, k, fc) for _ in range(r)])
        return 2.0 * sig_n * x / np.sqrt(2)  # quadrature variance 4 sigma_n^2

    ip, idd = colour(), colour()
    jam = track(cfg, np.zeros((r, k)), th_p + ip, eta_data=th_d + idd)
    rows = []
    for j in range(r):
        e_tot = 2 * np.pi * jam.phi[j]  # theta = 0: total error in rad
        assert jam.prompt_data is not None
        dd = np.angle(jam.prompt_data[j]) - np.angle(jam.prompt[j])  # pilot atan2; data is bit-free here
        dd = np.angle(np.exp(1j * dd))
        h = loop.filter(dd)
        nb = (k - burn) // n_win
        sl = slice(burn, burn + nb * n_win)
        hb = h[sl].reshape(nb, n_win)
        pdw = 0.5 * np.mean(hb**2, axis=1)
        i_prompt = np.abs(jam.prompt[j]) * np.cos(np.angle(jam.prompt[j]))  # I of the pilot prompt
        out = iw.iw_windows(i_prompt[sl], np.zeros(nb * n_win), loop, iw.IWConfig(window=n_win, gated=False))
        truth = float(np.var(e_tot[sl]))
        rows.append({"pdw_ratio": float(pdw.mean() / truth), "iw_ratio": float(out.noise.mean() / truth),
                     "pdw_rel_sd": float(pdw.std() / pdw.mean()), "iw_rel_sd": float(out.noise.std() / out.noise.mean())})  # fmt: skip
    return {"bn_cfg_hz": bn, "corner_hz": fc if fc is not None else "thermal_only",
            "pdw_ratio": float(np.mean([x["pdw_ratio"] for x in rows])),
            "iw_ratio": float(np.mean([x["iw_ratio"] for x in rows])),
            "pdw_rel_sd": float(np.mean([x["pdw_rel_sd"] for x in rows])),
            "iw_rel_sd": float(np.mean([x["iw_rel_sd"] for x in rows]))}  # fmt: skip


def main() -> None:
    classes = ["wideband_noise", "pulsed_noise", "pulsed_noise_lowprf", "swept_slow", "swept_fast",
               "chirp_inband", "cw_beat_7hz", "cw_beat_0.3hz"]  # fmt: skip
    cells_b = [
        (bn, fc, 4000 + i)
        for i, (bn, fc) in enumerate((bn, fc) for bn in (5.0, 35.0) for fc in (None, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03))
    ]
    with ProcessPoolExecutor(max_workers=19) as ex:
        fa = [ex.submit(part_a, c, 700 + i) for i, c in enumerate(classes)]
        rows_b = list(ex.map(part_b, cells_b))
        rows_a = [f.result() for f in fa]
    for r in rows_a:
        print(r)
    for r in rows_b:
        print(r)
    save("s07_pilot_data_witness", {"pilot_prn": PILOT_PRN, "data_prn": DATA_PRN, "part_a": rows_a,
                                    "part_b": rows_b})  # fmt: skip


if __name__ == "__main__":
    main()
