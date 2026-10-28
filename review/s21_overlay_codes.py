"""S21: the receiver's own symbol and secondary-code wipe-off multiplies the interference (P6' for real signals).

GNSS-SDR multiplies each pilot prompt by its secondary-code chip before the discriminator, the C/N0 buffer and the
lock test (`save_correlation_results()`, dll_pll_veml_tracking.cc), and a Costas data loop effectively multiplies
by the data bits. A despread tone at beat f_b times a periodic +-1 sequence s_k of period L becomes lines at
f_b + j / (L T) with powers |c_j|^2 (c the DFT of s, normalised so that sum |c_j|^2 = 1). P6 is linear in the
spectrum, so P6' sums the line contributions. Random data bits (C/A) use their expected spectrum N_b |B_{N_b}|^2.

Predicted mismatch r versus beat for: an unmodulated pilot; C/A with data (20 ms bits, T = 1 ms); Galileo E1C
(CS25 at T = 4 ms); GPS L5Q (NH20 at T = 1 ms); each at the default 35 Hz loop and at 5 Hz. The closed-loop
simulator checks a few cells per signal with the actual secondary sequences.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from gnssaudit.sim.correlator import TrackingConfig, thermal_noise, track
from gnssaudit.uq.loop import CarrierLoop
from review._common import save
from review.s06_mismatch_law import sigma_th2

N_BUF = 20
E1C_CS25 = "0011100000001010110110010"  # Galileo_E1.h:52
L5Q_NH20 = "00000100110101001110"  # GPS_L5.h:172

SIGNALS: dict[str, dict[str, object]] = {
    "pilot_unmodulated": {"t_s": 1e-3, "seq": "0"},
    "ca_data_random_bits": {"t_s": 1e-3, "seq": None},
    "e1c_cs25": {"t_s": 4e-3, "seq": E1C_CS25},
    "l5q_nh20": {"t_s": 1e-3, "seq": L5Q_NH20},
}


def _box2(w: np.ndarray, n: int) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(np.isclose(np.mod(w + np.pi, 2 * np.pi) - np.pi, 0), 1.0, np.sin(n * w / 2) / (n * np.sin(w / 2)))
    return b**2


def lines(f_b: float, t_s: float, seq: str | None, n_rand: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Frequencies (Hz) and relative powers of a unit tone at beat f_b after multiplication by the sequence."""
    if seq is None:  # random bits held n_rand integrations: dense expected spectrum on a fine grid
        m = 4096
        f = (np.arange(m) - m // 2) / (m * t_s)
        w = 2 * np.pi * f * t_s
        p = n_rand * _box2(w, n_rand) / m
        return f_b + f, p / p.sum()
    s = np.array([1.0 if c == "0" else -1.0 for c in seq])
    c = np.fft.fft(s) / s.size
    j = np.arange(s.size)
    return f_b + j / (s.size * t_s), np.abs(c) ** 2


def predict(loop: CarrierLoop, t_s: float, f_b: float, seq: str | None, power: float, cn0: float) -> dict[str, float]:
    f, p = lines(f_b, t_s, seq)
    w = 2 * np.pi * f * t_s
    hp2 = np.abs(loop.freq_response(np.abs(np.mod(w + np.pi, 2 * np.pi) - np.pi))) ** 2
    hd2 = np.abs(loop.freq_response(np.abs(np.mod(w + np.pi, 2 * np.pi) - np.pi), "disc")) ** 2
    q = power / 2 * p  # per-quadrature variance carried by each line
    sig_i = float(np.sum(hp2 * q))
    n_i = float(np.sum((1 - _box2(w, N_BUF)) * q * (1 + hd2)))
    cn0_rep = 1 / (t_s * (1 / (cn0 * t_s) + n_i))
    sig_c = loop.bn_eff_hz() / cn0
    return {"r": (sig_c + sig_i) / sigma_th2(loop, cn0_rep), "cn0_drop_db": float(10 * np.log10(cn0 / cn0_rep))}


def sim(args: tuple[str, float, float, int]) -> dict[str, object]:
    name, bn, f_b, seed = args
    spec = SIGNALS[name]
    t_s = float(spec["t_s"])  # type: ignore[arg-type]
    seq = spec["seq"]
    cn0 = 10**4.5
    rng = np.random.default_rng(seed)
    r, k, burn = 4, 40_000, 8_000
    sig_n2 = 1 / (2 * cn0 * t_s)
    power = 4 * sig_n2
    if seq is None:
        bits = np.repeat(np.where(rng.random((r, k // 20 + 1)) < 0.5, -1.0, 1.0), 20, axis=1)[:, :k]
    else:
        s = np.array([1.0 if c == "0" else -1.0 for c in str(seq)])
        bits = np.tile(s, k // s.size + 1)[None, :k].repeat(r, axis=0)
    th = thermal_noise(rng, (r, k), cn0, t_s)
    tt = np.arange(k) * t_s
    tone = np.sqrt(power) * np.exp(1j * (2 * np.pi * f_b * tt[None, :] + rng.uniform(0, 2 * np.pi, (r, 1))))
    # the receiver wipes `bits` off the signal, so in its frame the interference is multiplied by them
    cfg = TrackingConfig(bn_hz=bn, t_s=t_s, pilot=seq is not None)
    out = track(cfg, np.zeros((r, k)), th + tone * bits)
    loop = CarrierLoop.gnsssdr(bn, t_s)
    cn0_rep = 10 ** (np.mean(out.cn0_dbhz[:, burn:]) / 10)
    clean = track(cfg, np.zeros((r, k)), th)
    bias = float(np.mean(clean.cn0_dbhz[:, burn:])) - 45.0
    pred = predict(loop, t_s, f_b, None if seq is None else str(seq), power, cn0)
    return {"signal": name, "bn_cfg_hz": bn, "beat_hz": f_b,
            "sim_r": float(np.var(2 * np.pi * out.phi[:, burn:]) / sigma_th2(loop, cn0_rep)),
            "sim_cn0_drop_db": float(45.0 + bias - 10 * np.log10(cn0_rep)), "pred_r": pred["r"],
            "pred_cn0_drop_db": pred["cn0_drop_db"]}  # fmt: skip


def main() -> None:
    cn0 = 10**4.5
    curves = []
    for name, spec in SIGNALS.items():
        t_s = float(spec["t_s"])  # type: ignore[arg-type]
        seq = spec["seq"]
        for bn in (5.0, 35.0):
            loop = CarrierLoop.gnsssdr(bn, t_s)
            power = 4 / (2 * cn0 * t_s)
            for f_b in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0):
                p = predict(loop, t_s, f_b, None if seq is None else str(seq), power, cn0)
                curves.append({"signal": name, "bn_cfg_hz": bn, "beat_hz": f_b, **p})
    peak = {}
    for row in curves:
        key = f"{row['signal']}|{row['bn_cfg_hz']:g}"
        peak[key] = max(peak.get(key, 0.0), float(row["r"]))
    for k, v in peak.items():
        print("peak r", k, round(v, 1))
    jobs = [
        (n, bn, fb, 210 + i)
        for i, (n, bn, fb) in enumerate((n, bn, fb) for n in SIGNALS for bn in (5.0, 35.0) for fb in (1.0, 5.0, 20.0))
    ]
    with ProcessPoolExecutor(max_workers=19) as ex:
        sims = list(ex.map(sim, jobs))
    for s in sims:
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()})
    save("s21_overlay_codes", {"curves": curves, "peak_r": peak, "simulator": sims})


if __name__ == "__main__":
    main()
