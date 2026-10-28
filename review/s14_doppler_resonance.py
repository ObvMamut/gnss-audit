"""S14: Doppler resonance of periodic jammers and the general two-filter law (N2 / P6 on a real despread spectrum).

A jammer repeating with period T_p has lines at multiples of 1/T_p. Despread against a 1 ms code and sampled once
per millisecond, they fold to beats (k / T_p - f_D) mod 1 kHz. For an in-band PPD-like chirp with T_p = 20 us
(lines every 50 kHz) every line folds to the single beat -f_D mod 1 kHz, so the carrier-phase error depends on the
satellite's Doppler modulo 1 kHz: a resonance ridge of width about B_n around each 1 kHz multiple.

For each Doppler: the despread series of the chirp (fixed J/C), its two-sided PSD S_eta, the P6 prediction with
S_m = S_n = [S_eta(f) + S_eta(-f)] / 4, and the nonlinear closed-loop twin-run simulation. The jammer's timing
and phase are the same in every cell. Because all lines fold onto one beat they add coherently, so the despread
power also depends on the chirp-to-code-epoch alignment (which real code Doppler makes drift slowly).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import signal

from gnssaudit.sim import despread as ds
from gnssaudit.sim.correlator import TrackingConfig, thermal_noise, track
from gnssaudit.uq.loop import CarrierLoop
from review._common import save
from review.s06_mismatch_law import CN0, N_BUF, boxcar2, sigma_th2

T = 1e-3
FS = 2.046e6
SECONDS = 40
J_OVER_C = 300.0  # jammer-to-signal power ratio (25 dB) before despreading; despread tone about 2 sigma_n^2


def despread_chirp(f_d: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ph, off = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2e-5)
    dsp = ds.Despreader(prn=5, fs_hz=FS, doppler_hz=f_d)
    out = []
    bpc = 2000
    for c in range(SECONDS * 1000 // bpc):
        x = ds.swept_cw(bpc * dsp.ns, FS, -1.0e6, 2.0e6, 2e-5, J_OVER_C, ph, off, c * bpc * dsp.ns)
        out.append(dsp.despread(x, c * bpc))
    return np.concatenate(out)


def predict(loop: CarrierLoop, eta: np.ndarray, bias_db: float) -> dict[str, float]:
    f, s = signal.welch(eta - eta.mean(), fs=1000.0, nperseg=8192, return_onesided=False, detrend="constant")
    order = np.argsort(f)
    f, s = f[order], s[order]
    s_rev = np.interp(-f, f, s)
    s_q = (s + s_rev) / 4  # per-quadrature PSD (Hz^-1), integrates to Var(m) = Var(n)
    df = f[1] - f[0]
    w = 2 * np.pi * np.abs(f) * T
    hp2 = np.abs(loop.freq_response(w)) ** 2
    hd2 = np.abs(loop.freq_response(w, "disc")) ** 2
    sig_i = float(np.sum(hp2 * s_q) * df)
    n_i = float(np.sum((1 - boxcar2(f, N_BUF)) * (s_q + hd2 * s_q)) * df)
    n_w = 1 / (CN0 * T)
    cn0_rep = 10 ** (bias_db / 10) / (T * (n_w + n_i))
    st = sigma_th2(loop, cn0_rep)
    sig_c = loop.bn_eff_hz() / CN0
    return {
        "sigma_i2": sig_i,
        "cn0_rep_dbhz": float(10 * np.log10(cn0_rep)),
        "r_corrected": (sig_c + sig_i) / st,
    }


def cell(args: tuple[float, float, int]) -> dict[str, object]:
    bn, delta, seed = args
    f_d = 2000.0 + delta
    eta = despread_chirp(f_d, 8000)  # same jammer timing and phase in every cell: only the Doppler moves
    loop = CarrierLoop.gnsssdr(bn, T)
    rng = np.random.default_rng(seed + 7)
    k = eta.size
    r = 2
    th = thermal_noise(rng, (r, k), CN0, T)
    cfg = TrackingConfig(bn_hz=bn, t_s=T, pilot=True)
    clean = track(cfg, np.zeros((r, k)), th)
    jam = track(cfg, np.zeros((r, k)), th + eta[None, :])
    burn = 10_000
    sig_i = float(np.mean(np.var(2 * np.pi * (jam.phi - clean.phi)[:, burn:], axis=1)))
    sig_c = float(np.var(2 * np.pi * clean.phi[:, burn:]))
    cn0_rep = float(np.mean(jam.cn0_dbhz[:, burn:]))
    cn0_clean = float(np.mean(clean.cn0_dbhz[:, burn:]))
    st = sigma_th2(loop, 10 ** (cn0_rep / 10))
    pred = predict(loop, eta, cn0_clean - 45.0)
    return {"bn_cfg_hz": bn, "doppler_offset_hz": delta, "despread_power": float(np.mean(np.abs(eta) ** 2)),
            "sim": {"sigma_i2": sig_i, "cn0_rep_dbhz": cn0_rep, "r_corrected": (sig_c + sig_i) / st,
                    "max_abs_err_rad": float(np.max(np.abs(2 * np.pi * jam.phi[:, burn:])))},
            "pred": pred}  # fmt: skip


def main() -> None:
    deltas = [-40, -20, -10, -5, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 5, 10, 20, 40, 100, 250]
    jobs = [(bn, float(d), 8000 + i) for i, (bn, d) in enumerate((bn, d) for bn in (5.0, 35.0) for d in deltas)]
    with ProcessPoolExecutor(max_workers=19) as ex:
        rows = list(ex.map(cell, jobs))
    for r in rows:
        s, p = r["sim"], r["pred"]
        assert isinstance(s, dict) and isinstance(p, dict)
        print(
            r["bn_cfg_hz"],
            r["doppler_offset_hz"],
            "P=%.4f" % r["despread_power"],
            "sim r=%.2f cn0=%.2f maxerr=%.2f | pred r=%.2f cn0=%.2f"
            % (
                s["r_corrected"],
                s["cn0_rep_dbhz"],
                s["max_abs_err_rad"],
                p["r_corrected"],
                p["cn0_rep_dbhz"],
            ),
        )
    save("s14_doppler_resonance", {"j_over_c": J_OVER_C, "rows": rows})


if __name__ == "__main__":
    main()
