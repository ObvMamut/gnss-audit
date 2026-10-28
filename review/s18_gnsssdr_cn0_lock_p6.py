"""S18: the C/N0 bias (F21), the lock detector (F22) and the two-filter law P6 on the real GNSS-SDR v0.0.22 binary.

Synthetic GPS L1 C/A recordings (PRN 1, 300 Hz Doppler, 4 Msps int8, random data) are run through the pinned image.
The true carrier phase is known exactly, so the receiver's own phase error is e = phi_reported - theta (up to a
constant), with no twin needed.

Part A (F21): clean recordings at constant 30 / 33 / 36 / 40 / 45 dB-Hz. Only 40 and 45 dB-Hz acquire with a 1 ms
coherent acquisition; the weaker ones are acquired late or falsely, so F22 is tested by S20 with stepped C/N0. Reported C/N0 against the predicted M2M4 bias,
the dumped lock statistic, and whether the receiver drops the channel.
Part B (P6): 55 dB-Hz with a CW at f_D + 3 kHz + f_b, which despreads to a tone at beat f_b on code line 3. Its
power is set so the despread tone has quadrature variance 2 sigma_n^2. The measured r = Var(e) / sigma_th^2(C/N0_rep)
and the reported C/N0 are compared with P6, computed from the despread spectrum of the same samples.

Usage: uv run python -m review.s18_gnsssdr_cn0_lock_p6   (needs Docker via sg; about 15 min)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from gnssaudit.sim.correlator import m2m4_cn0_dbhz, thermal_noise
from gnssaudit.sim.despread import Despreader, ca_code
from gnssaudit.uq.loop import CarrierLoop
from review._common import save
from review.s06_mismatch_law import predict_tone, sigma_th2
from review.s17_gnsssdr_identification import DUMP_DTYPE, IMAGE, WORK, config_text, os_ids

FS = 4.0e6
DOPPLER = 300.0
PRN = 1
F_L1 = 1575.42e6
NOISE_STD = 16.0
DURATION_S = 60.0
T = 1e-3
OUT = WORK.parent / "s18"


STEP_S = 20.0  # stepped recordings: step_from_db until STEP_S, then cn0_db (acquire strong, then degrade)


def generate(
    path: Path, cn0_db: float, cw_hz: float | None, cw_power: float, seed: int = 180, step_from_db: float | None = None
) -> None:
    rng = np.random.default_rng(seed)  # same noise and bits in every recording
    code = ca_code(PRN)
    f_code = 1.023e6 * (1 + DOPPLER / F_L1)
    amp = np.sqrt(10 ** (cn0_db / 10) / FS) * NOISE_STD * np.sqrt(2)
    amp0 = amp if step_from_db is None else np.sqrt(10 ** (step_from_db / 10) / FS) * NOISE_STD * np.sqrt(2)
    n_total = int(DURATION_S * FS)
    bits = np.where(rng.random(int(DURATION_S * 50) + 2) < 0.5, -1.0, 1.0)
    chunk = 1 << 22
    with open(path, "wb") as fh:
        for start in range(0, n_total, chunk):
            n = np.arange(start, min(start + chunk, n_total))
            t = n / FS
            chips = 123.4 + f_code * t
            c = code[np.floor(chips).astype(np.int64) % 1023]
            d = bits[np.floor(chips / (1023 * 20)).astype(np.int64)]
            a_t = np.where(t < STEP_S, amp0, amp)
            x = a_t * d * c * np.exp(2j * np.pi * DOPPLER * t)
            x = x + NOISE_STD * (rng.standard_normal(n.size) + 1j * rng.standard_normal(n.size))
            if cw_hz is not None:
                x = x + np.sqrt(cw_power) * amp * np.exp(2j * np.pi * cw_hz * t + 0.7j)
            iq = np.empty(2 * n.size, dtype=np.int8)
            iq[0::2] = np.clip(np.round(x.real), -127, 127)
            iq[1::2] = np.clip(np.round(x.imag), -127, 127)
            iq.tofile(fh)


def run_receiver(name: str, recording: str, overrides: dict[str, str]) -> dict[str, object]:
    conf = config_text("default").replace("/data/sig.bin", f"/data/{recording}").replace("trk_default", f"trk_{name}")
    for k, v in overrides.items():
        conf += f"{k}={v}\n"
    (OUT / f"{name}.conf").write_text(conf)
    cmd = (f"docker run --rm --user {os_ids()} -e HOME=/data/home -v {OUT}:/data -w /data {IMAGE} "
           f"gnss-sdr --config_file=/data/{name}.conf")  # fmt: skip
    res = subprocess.run(["sg", "docker", "-c", cmd], capture_output=True, text=True, check=False)
    log = res.stdout + "\n--- stderr\n" + res.stderr
    (OUT / f"{name}.log").write_text(log)
    files = sorted(OUT.glob(f"trk_{name}*.dat"))
    rec = np.fromfile(files[0], dtype=DUMP_DTYPE) if files else np.zeros(0, dtype=DUMP_DTYPE)
    return {"log": log, "rec": rec, "exit": res.returncode}


def phase_error(rec: np.ndarray, t_min: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
    t = rec["prn_start_sample"].astype(np.float64) / FS
    sel = t > t_min
    e = -rec["acc_phase_rad"][sel].astype(np.float64) - 2 * np.pi * DOPPLER * t[sel]
    e = e - np.polyval(np.polyfit(t[sel], e, 1), t[sel])  # constant (ambiguity) and tiny frequency offset
    return t[sel], e


def part_a() -> list[dict[str, object]]:
    rows = []
    rng = np.random.default_rng(1)
    for cn0 in (30.0, 33.0, 36.0, 40.0, 45.0):
        rec_name = f"clean_{int(cn0)}.bin"
        generate(OUT / rec_name, cn0, None, 0.0)
        r = run_receiver(f"clean{int(cn0)}", rec_name, {})
        rec = r["rec"]
        assert isinstance(rec, np.ndarray)
        log = str(r["log"])
        w = thermal_noise(rng, (40_000, 20), 10 ** (cn0 / 10), T)
        pred_bias = float(np.mean(m2m4_cn0_dbhz(1 + w, T)) - cn0)
        t = rec["prn_start_sample"].astype(np.float64) / FS if rec.size else np.zeros(0)
        sel = t > 15.0
        row: dict[str, object] = {
            "cn0_true_dbhz": cn0, "records": int(rec.size), "last_record_s": float(t[-1]) if t.size else None,
            "loss_of_lock_in_log": "Loss of lock" in log or "loss of lock" in log,
            "reported_minus_true_db": float(np.mean(rec["cn0_dbhz"][sel]) - cn0) if sel.any() else None,
            "predicted_m2m4_bias_db": pred_bias,
            "dumped_lock_stat_median": float(np.median(rec["lock"][sel])) if sel.any() else None,
        }  # fmt: skip
        if sel.sum() > 100:
            _, e = phase_error(rec)
            loop = CarrierLoop.gnsssdr(35.0, T)
            c = 10 ** (cn0 / 10)
            row["phase_var_over_thermal_true_cn0"] = float(np.var(e) / sigma_th2(loop, c))
        rows.append(row)
        print(row, flush=True)
        (OUT / rec_name).unlink()
    return rows


def part_b() -> list[dict[str, object]]:
    cn0 = 55.0
    c_lin = 10 ** (cn0 / 10)
    sig_n2 = 1 / (2 * c_lin * T)
    # despread power of a unit-power CW (relative to the signal power) at each beat, from the actual waveform
    dsp = Despreader(prn=PRN, fs_hz=FS, doppler_hz=DOPPLER)
    rows = []
    generate(OUT / "clean_55.bin", cn0, None, 0.0)
    clean = run_receiver("clean55", "clean_55.bin", {})
    clean_rec = clean["rec"]
    assert isinstance(clean_rec, np.ndarray)
    sel_c = clean_rec["prn_start_sample"] / FS > 15.0
    cn0_clean_rep = float(np.mean(clean_rec["cn0_dbhz"][sel_c]))
    bias_db = cn0_clean_rep - cn0
    (OUT / "clean_55.bin").unlink()
    for f_b in (0.5, 2.0, 5.0, 12.0, 30.0, 80.0):
        f_cw = DOPPLER + 3000.0 + f_b
        n = np.arange(int(0.5 * FS))
        unit = np.exp(2j * np.pi * f_cw * n / FS)
        p_unit = float(np.mean(np.abs(dsp.despread(unit)) ** 2))
        cw_power = 2 * 2 * sig_n2 / p_unit  # despread tone power 4 sigma_n^2: quadrature variance 2 sigma_n^2
        rec_name = f"cw_{f_b:g}.bin"
        generate(OUT / rec_name, cn0, f_cw, cw_power)
        for cfg_name, overrides, bn in (("bn35", {}, 35.0), ("bn15", {"Tracking_1C.pll_bw_hz": "15.0"}, 15.0)):
            name = f"cw{f_b:g}_{cfg_name}"
            r = run_receiver(name, rec_name, overrides)
            rec = r["rec"]
            assert isinstance(rec, np.ndarray)
            sel = rec["prn_start_sample"] / FS > 15.0
            if sel.sum() < 100:
                rows.append({"beat_hz": f_b, "config": cfg_name, "error": "no steady tracking"})
                continue
            _, e = phase_error(rec)
            loop = CarrierLoop.gnsssdr(bn, T)
            cn0_rep = float(np.mean(rec["cn0_dbhz"][sel]))
            r_meas = float(np.var(e) / sigma_th2(loop, 10 ** (cn0_rep / 10)))
            pred = predict_tone(loop, f_b, 4 * sig_n2, bias_db, cn0=c_lin)
            row = {"beat_hz": f_b, "config": cfg_name, "cw_power_over_c": cw_power,
                   "meas": {"r": r_meas, "cn0_rep_dbhz": cn0_rep, "var_e_rad2": float(np.var(e))},
                   "pred": {"r": pred["r_corrected"], "cn0_rep_dbhz": pred["cn0_rep_dbhz"]},
                   "loss_of_lock_in_log": "oss of lock" in str(r["log"])}  # fmt: skip
            rows.append(row)
            print(json.dumps(row), flush=True)
        (OUT / rec_name).unlink()
    return [{"clean_cn0_rep_dbhz": cn0_clean_rep, "bias_db": bias_db}, *rows]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "home").mkdir(exist_ok=True)
    (OUT / "pvt").mkdir(exist_ok=True)
    a = part_a()
    b = part_b()
    save("s18_gnsssdr_cn0_lock_p6", {"image": IMAGE, "part_a": a, "part_b": b})


if __name__ == "__main__":
    main()
