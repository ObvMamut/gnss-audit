"""S06: the two-filter mismatch law (N2, proposed P6) and the corrected mismatch ratio (F2; docs/2 H2).

Prediction. For despread interference with quadrature and in-phase spectra S_n = S_m (circular, non-zero beat):
  interference-induced error      sigma_i^2 = int |H_phi(f)|^2 S_n(f) df
  noise counted by C/N0 (M2M4)    N_i       = int (1 - |B_N(f)|^2) [S_m(f) + |H_d(f)|^2 S_n(f)] df
where B_N is the N-prompt boxcar (C/N0 buffer, N = 20). The reported C/N0 is 1 / (T (N_w + N_i)) times the
estimator's clean-condition bias b (finding F21), and the thermal model gives sigma_th^2(C/N0_rep). Then
  r_corrected = (sigma_clean^2 + sigma_i^2) / sigma_th^2(C/N0_rep)   (total over total)
  r_docs2     = sigma_i^2 / sigma_th^2(C/N0_rep)                    (as written in docs/2 section 5.3)

Simulation. Nonlinear closed loop (pilot, faithful GNSS-SDR timing, M2M4 + smoother), thermal 45 dB-Hz plus a
despread tone at beat f_b with power a^2 = P (complex), twin-run truth with identical thermal noise.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from gnssaudit.sim.correlator import TrackingConfig, thermal_noise, track
from gnssaudit.uq.loop import CarrierLoop
from review._common import save

T = 1e-3
CN0 = 10**4.5
N_BUF = 20


def boxcar2(f: np.ndarray, n: int) -> np.ndarray:
    w = 2 * np.pi * f * T
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(w == 0, 1.0, np.sin(n * w / 2) / (n * np.sin(w / 2)))
    return b**2


def sigma_th2(loop: CarrierLoop, cn0: float) -> float:
    return loop.bn_eff_hz() / cn0 * (1 + 1 / (2 * T * cn0))


def predict_tone(loop: CarrierLoop, f_b: float, power: float, bias_db: float, cn0: float = CN0) -> dict[str, float]:
    w = np.array([2 * np.pi * abs(f_b) * T])
    hp2 = float(np.abs(loop.freq_response(w))[0] ** 2)
    hd2 = float(np.abs(loop.freq_response(w, "disc"))[0] ** 2)
    s = power / 2  # variance of each quadrature component of a complex tone of power `power`
    sig_i = hp2 * s
    n_w = 1 / (cn0 * T)
    n_i = (1 - boxcar2(np.array([f_b]), N_BUF)[0]) * (s + hd2 * s)
    cn0_rep = 10 ** (bias_db / 10) / (T * (n_w + n_i))
    st = sigma_th2(loop, cn0_rep)
    sig_clean = loop.bn_eff_hz() / cn0
    return {"sigma_i2": sig_i, "cn0_rep_dbhz": 10 * np.log10(cn0_rep),
            "r_corrected": (sig_clean + sig_i) / st, "r_docs2": sig_i / st}  # fmt: skip


def simulate_tone(args: tuple[float, float, float, int]) -> dict[str, float]:
    bn, f_b, power, seed = args
    k = 40_000
    burn = 10_000
    rng = np.random.default_rng(seed)
    r = 4
    eta_w = thermal_noise(rng, (r, k), CN0, T)
    t = np.arange(k) * T
    ph = rng.uniform(0, 2 * np.pi, size=(r, 1))
    eta_i = np.sqrt(power) * np.exp(1j * (2 * np.pi * f_b * t[None, :] + ph))
    cfg = TrackingConfig(bn_hz=bn, t_s=T, pilot=True)
    clean = track(cfg, np.zeros((r, k)), eta_w)
    jam = track(cfg, np.zeros((r, k)), eta_w + eta_i)
    dphi = 2 * np.pi * (jam.phi - clean.phi)[:, burn:]
    e_clean = 2 * np.pi * clean.phi[:, burn:]
    cn0_rep = 10 ** (np.mean(jam.cn0_dbhz[:, burn:]) / 10)
    cn0_clean = np.mean(clean.cn0_dbhz[:, burn:])
    loop = CarrierLoop.gnsssdr(bn, T)
    st = sigma_th2(loop, cn0_rep)
    sig_i = float(np.mean(np.var(dphi, axis=1)))
    sig_c = float(np.var(e_clean))
    return {"bn_cfg_hz": bn, "f_b_hz": f_b, "power": power, "sigma_i2": sig_i, "sigma_clean2": sig_c,
            "cn0_rep_dbhz": 10 * np.log10(cn0_rep), "cn0_clean_dbhz": cn0_clean,
            "r_corrected": (sig_c + sig_i) / st, "r_docs2": sig_i / st,
            "max_abs_phase_err_rad": float(np.max(np.abs(2 * np.pi * jam.phi[:, burn:])))}  # fmt: skip


def main() -> None:
    beats = [0.2, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 150, 300]
    sig2 = 1 / (2 * CN0 * T)
    jobs = []
    for bn in (5.0, 35.0):
        for f_b in beats:
            for rel in (0.5, 2.0):  # tone quadrature variance relative to thermal sigma_n^2
                jobs.append((bn, float(f_b), 2 * rel * sig2, 3000 + len(jobs)))
    # wideband-noise reference at matched power: no ridge expected
    with ProcessPoolExecutor(max_workers=19) as ex:
        sims = list(ex.map(simulate_tone, jobs))
    bias_db = float(np.mean([s["cn0_clean_dbhz"] for s in sims]) - 45.0)
    rows = []
    for s in sims:
        loop = CarrierLoop.gnsssdr(s["bn_cfg_hz"], T)
        pred = predict_tone(loop, s["f_b_hz"], s["power"], bias_db)
        rows.append({"sim": s, "pred": pred})
        print(s["bn_cfg_hz"], s["f_b_hz"], round(s["power"] / (2 * sig2), 2),
              "sim r=%.2f rdoc=%.2f cn0=%.2f | pred r=%.2f rdoc=%.2f cn0=%.2f | maxerr=%.2f" % (
                  s["r_corrected"], s["r_docs2"], s["cn0_rep_dbhz"], pred["r_corrected"], pred["r_docs2"],
                  pred["cn0_rep_dbhz"], s["max_abs_phase_err_rad"]))  # fmt: skip
    save("s06_mismatch_law", {"clean_cn0_bias_db": bias_db, "rows": rows})


if __name__ == "__main__":
    main()
