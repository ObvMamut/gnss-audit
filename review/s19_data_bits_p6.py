"""S19: the two-filter law for data-modulated signals (P6', refining P6) on the simulator and the GNSS-SDR binary.

With a Costas discriminator on a data channel, I ~ D A + m and Q ~ D A eps + n, so atan(Q/I) ~ eps + D n / A and
|P|^2 ~ A^2 + 2 D A m: the loop and the C/N0 estimator both see the interference multiplied by the data bits D.
For random +-1 bits held for N_b integrations, S_D(w) = N_b |B_{N_b}(w)|^2 (unit variance), so P6' replaces the
per-quadrature spectrum S_q by S_q * S_D (circular convolution). A pilot channel has D = 1 and P6' = P6.

Part 1: nonlinear closed-loop simulator, Costas with random bits (N_b = 20), tones at several beats, against P6'.
Part 2: the GNSS-SDR binary runs of S18 (C/A with random data) against P6 and P6'.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from gnssaudit.sim.correlator import TrackingConfig, thermal_noise, track
from gnssaudit.uq.loop import CarrierLoop
from review._common import RESULTS, save
from review.s06_mismatch_law import sigma_th2

T = 1e-3
N_BUF = 20
N_BIT = 20
M = 1 << 16  # frequency grid (per-sample, 1 kHz)


def _boxcar2(w: np.ndarray, n: int) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(w == 0, 1.0, np.sin(n * w / 2) / (n * np.sin(w / 2)))
    return b**2


def predict_tone_p6(
    loop: CarrierLoop, f_b: float, power: float, cn0: float, bias_db: float, data: bool
) -> dict[str, float]:
    """P6 (data=False) or P6' (data=True) for a despread complex tone of total power `power` at beat f_b."""
    w = 2 * np.pi * np.fft.fftfreq(M)
    hp2 = np.abs(loop.freq_response(np.abs(w))) ** 2
    hd2 = np.abs(loop.freq_response(np.abs(w), "disc")) ** 2
    s_q = np.zeros(M)  # per-quadrature PSD of the tone (cos and sin parts), integrates to power/2
    k = int(round(f_b * T * M))
    s_q[k % M] += power / 4 * M
    s_q[-k % M] += power / 4 * M
    if data:
        s_d = N_BIT * _boxcar2(w, N_BIT)  # mean 1 over the grid
        s_q = np.real(np.fft.ifft(np.fft.fft(s_q) * np.fft.fft(s_d))) / M
    mean = np.mean  # (1/2pi) int dw  ->  mean over the grid
    sig_i = float(mean(hp2 * s_q))
    n_i = float(mean((1 - _boxcar2(w, N_BUF)) * (s_q + hd2 * s_q)))
    n_w = 1 / (cn0 * T)
    cn0_rep = 10 ** (bias_db / 10) / (T * (n_w + n_i))
    sig_c = loop.bn_eff_hz() / cn0 * (1 + 1 / (2 * T * cn0))
    return {"sigma_i2": sig_i, "cn0_rep_dbhz": float(10 * np.log10(cn0_rep)),
            "r": (sig_c + sig_i) / sigma_th2(loop, cn0_rep)}  # fmt: skip


def sim_cell(args: tuple[float, float, float, int]) -> dict[str, float]:
    bn, f_b, cn0_db, seed = args
    cn0 = 10 ** (cn0_db / 10)
    rng = np.random.default_rng(seed)
    r, k, burn = 4, 60_000, 10_000
    sig_n2 = 1 / (2 * cn0 * T)
    power = 4 * sig_n2
    bits = np.repeat(np.where(rng.random((r, k // N_BIT)) < 0.5, -1.0, 1.0), N_BIT, axis=1)
    th = thermal_noise(rng, (r, k), cn0, T)
    tt = np.arange(k) * T
    tone = np.sqrt(power) * np.exp(1j * (2 * np.pi * f_b * tt[None, :] + rng.uniform(0, 2 * np.pi, (r, 1))))
    cfg = TrackingConfig(bn_hz=bn, t_s=T, pilot=False)
    clean = track(cfg, np.zeros((r, k)), th, bits=bits)
    jam = track(cfg, np.zeros((r, k)), th + tone, bits=bits)
    e = 2 * np.pi * jam.phi[:, burn:]
    loop = CarrierLoop.gnsssdr(bn, T)
    cn0_rep = 10 ** (np.mean(jam.cn0_dbhz[:, burn:]) / 10)
    bias_db = float(np.mean(clean.cn0_dbhz[:, burn:]) - cn0_db)
    p_data = predict_tone_p6(loop, f_b, power, cn0, bias_db, True)
    p_pilot = predict_tone_p6(loop, f_b, power, cn0, bias_db, False)
    return {"bn_cfg_hz": bn, "beat_hz": f_b, "sim_r": float(np.var(e) / sigma_th2(loop, cn0_rep)),
            "sim_cn0_rep": float(10 * np.log10(cn0_rep)), "p6prime_r": p_data["r"],
            "p6prime_cn0": p_data["cn0_rep_dbhz"], "p6_r": p_pilot["r"], "p6_cn0": p_pilot["cn0_rep_dbhz"]}  # fmt: skip


def binary_rows() -> list[dict[str, object]]:
    d = json.loads((RESULTS / "s18_gnsssdr_cn0_lock_p6.json").read_text())
    head, *rows = d["part_b"]
    bias = head["bias_db"]
    cn0 = 10**5.5
    out = []
    for row in rows:
        if "meas" not in row:
            continue
        bn = 35.0 if row["config"] == "bn35" else 15.0
        loop = CarrierLoop.gnsssdr(bn, T)
        power = 4 / (2 * cn0 * T)
        p = predict_tone_p6(loop, row["beat_hz"], power, cn0, bias, True)
        q = predict_tone_p6(loop, row["beat_hz"], power, cn0, bias, False)
        out.append({"beat_hz": row["beat_hz"], "config": row["config"], "binary_r": row["meas"]["r"],
                    "binary_cn0": row["meas"]["cn0_rep_dbhz"], "p6prime_r": p["r"], "p6prime_cn0": p["cn0_rep_dbhz"],
                    "p6_r": q["r"], "p6_cn0": q["cn0_rep_dbhz"]})  # fmt: skip
    return out


def resonance_cell(args: tuple[float, float, float, int]) -> dict[str, float]:
    """S14's despread chirp tone (power from S14) on a C/A data channel: simulator against P6' and P6."""
    bn, delta, power, seed = args
    cn0_db = 45.0
    cn0 = 10 ** (cn0_db / 10)
    rng = np.random.default_rng(seed)
    r, k, burn = 4, 40_000, 10_000
    bits = np.repeat(np.where(rng.random((r, k // N_BIT)) < 0.5, -1.0, 1.0), N_BIT, axis=1)
    th = thermal_noise(rng, (r, k), cn0, T)
    tt = np.arange(k) * T
    tone = np.sqrt(power) * np.exp(1j * (2 * np.pi * (-delta) * tt[None, :] + rng.uniform(0, 2 * np.pi, (r, 1))))
    cfg = TrackingConfig(bn_hz=bn, t_s=T, pilot=False)
    clean = track(cfg, np.zeros((r, k)), th, bits=bits)
    jam = track(cfg, np.zeros((r, k)), th + tone, bits=bits)
    loop = CarrierLoop.gnsssdr(bn, T)
    cn0_rep = 10 ** (np.mean(jam.cn0_dbhz[:, burn:]) / 10)
    bias_db = float(np.mean(clean.cn0_dbhz[:, burn:]) - cn0_db)
    e = 2 * np.pi * jam.phi[:, burn:]
    pd = predict_tone_p6(loop, abs(delta), power, cn0, bias_db, True)
    pp = predict_tone_p6(loop, abs(delta), power, cn0, bias_db, False)
    return {"bn_cfg_hz": bn, "doppler_offset_hz": delta, "sim_r_data": float(np.var(e) / sigma_th2(loop, cn0_rep)),
            "sim_cn0_data": float(10 * np.log10(cn0_rep)), "p6prime_r": pd["r"], "p6prime_cn0": pd["cn0_rep_dbhz"],
            "p6_pilot_r": pp["r"]}  # fmt: skip


def main() -> None:
    jobs = [
        (bn, fb, 55.0, 190 + i)
        for i, (bn, fb) in enumerate((bn, fb) for bn in (35.0, 15.0) for fb in (0.5, 2.0, 5.0, 12.0, 30.0, 80.0))
    ]
    with ProcessPoolExecutor(max_workers=12) as ex:
        sims = list(ex.map(sim_cell, jobs))
    for s in sims:
        print("sim", {k: round(v, 3) for k, v in s.items()})
    s14 = json.loads((RESULTS / "s14_doppler_resonance.json").read_text())["rows"]
    res_jobs = [(r["bn_cfg_hz"], r["doppler_offset_hz"], r["despread_power"], 400 + i) for i, r in enumerate(s14)
                if abs(r["doppler_offset_hz"]) <= 40 and r["doppler_offset_hz"] > 0]  # fmt: skip
    with ProcessPoolExecutor(max_workers=16) as ex:
        res = list(ex.map(resonance_cell, res_jobs))
    for x in res:
        print("res", {k: round(v, 2) for k, v in x.items()})
    bins = binary_rows()
    for b in bins:
        print("bin", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in b.items()})
    save("s19_data_bits_p6", {"simulator": sims, "binary": bins, "resonance_data_channel": res})


if __name__ == "__main__":
    main()
