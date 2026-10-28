"""S22: secondary-code whitening on the GNSS-SDR binary: Galileo E1 (pilot tracking) against GPS C/A under CW.

S21 predicts that a slow-beat CW, which on C/A drives the phase variance about 20x above what the reported C/N0
implies (confirmed on the binary in S18/S19), is almost harmless relative to C/N0 on Galileo E1C. GNSS-SDR wipes
the CS25 secondary code off the pilot every 4 ms integration, which whitens the interference: predicted r ~ 1, with
a C/N0 drop of about 5 dB that correctly reports the damage.

The CW starts at 20 s (after acquisition; present from the start it blocks the 4 ms acquisition).

Recording: Galileo E1 PRN 1, sinBOC(1,1) for both components (s = a (D e_B - cs e_C) boc), pilot C/N0 47 dB-Hz,
random I/NAV symbols, 300 Hz Doppler, 8 Msps int8, 60 s. Jammer: CW at f_D + 1.023 MHz + f_b, set so that its
despread power on the pilot replica is 4 sigma_n^2. Runs: clean, and CW with f_b = 2 and 5 Hz.

Usage: uv run python -m review.s22_gnsssdr_e1_pilot   (needs Docker via sg)
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np

from gnssaudit.sim.despread import Despreader
from gnssaudit.uq.loop import CarrierLoop
from review._common import save
from review.s06_mismatch_law import sigma_th2
from review.s17_gnsssdr_identification import DUMP_DTYPE, IMAGE, WORK, os_ids
from review.s21_overlay_codes import E1C_CS25, predict

SRC = WORK.parent / "src" / "gnss-sdr" / "src" / "core" / "system_parameters" / "Galileo_E1.h"
OUT = WORK.parent / "s22"
FS = 8.0e6
DOPPLER = 300.0
F_E1 = 1575.42e6
CN0_PILOT_DB = 47.0
NOISE_STD = 6.0  # leaves int8 headroom for the CW (16 clipped it at about 1800x the pilot power)
DURATION_S = 60.0
T = 4e-3
ONSET_S = 20.0


def e1_code(kind: str, prn: int = 1) -> np.ndarray:
    text = SRC.read_text()
    m = re.search(rf"GALILEO_E1_{kind}_PRIMARY_CODE\[[^\]]*\]\[1024\] = \{{(.*?)\}};", text, re.S)
    assert m is not None
    # each code is one array element made of adjacent string literals; elements are separated by commas
    elements = [e for e in m.group(1).split(",") if '"' in e]
    code_hex = "".join(re.findall(r'"([0-9A-F]+)"', elements[prn - 1]))
    bits = "".join(f"{int(h, 16):04b}" for h in code_hex)
    return np.array([1.0 if b == "0" else -1.0 for b in bits])  # hex_to_binary_converter: 0 -> +1


def boc(code: np.ndarray) -> np.ndarray:
    return np.repeat(code, 2) * np.tile([1.0, -1.0], code.size)  # galileo_e1_sinboc_11_gen_int


def generate(path: Path, cw_rel_power: float | None, f_cw: float, seed: int = 220) -> None:
    rng = np.random.default_rng(seed)
    e_b, e_c = boc(e1_code("B")), boc(e1_code("C"))  # 8184 half-chips at 2.046 MHz
    cs = np.array([1.0 if c == "0" else -1.0 for c in E1C_CS25])
    f_half = 2 * 1.023e6 * (1 + DOPPLER / F_E1)
    a = np.sqrt(10 ** (CN0_PILOT_DB / 10) / FS) * NOISE_STD * np.sqrt(2)  # per-component amplitude
    n_total = int(DURATION_S * FS)
    sym = np.where(rng.random(int(DURATION_S / T) + 2) < 0.5, -1.0, 1.0)
    chunk = 1 << 22
    with open(path, "wb") as fh:
        for start in range(0, n_total, chunk):
            n = np.arange(start, min(start + chunk, n_total))
            t = n / FS
            half = 37.3 + f_half * t
            hi = np.floor(half).astype(np.int64)
            epoch = hi // 8184
            sig = a * (sym[epoch] * e_b[hi % 8184] - cs[epoch % 25] * e_c[hi % 8184])
            x = sig * np.exp(2j * np.pi * DOPPLER * t)
            x = x + NOISE_STD * (rng.standard_normal(n.size) + 1j * rng.standard_normal(n.size))
            if cw_rel_power is not None:  # jammer onset after acquisition (CW from the start blocks acquisition)
                x = x + (t >= ONSET_S) * np.sqrt(cw_rel_power) * a * np.exp(2j * np.pi * f_cw * t + 0.4j)
            iq = np.empty(2 * n.size, dtype=np.int8)
            iq[0::2] = np.clip(np.round(x.real), -127, 127)
            iq[1::2] = np.clip(np.round(x.imag), -127, 127)
            iq.tofile(fh)


def config(name: str, recording: str) -> str:
    kv = {
        "GNSS-SDR.internal_fs_sps": str(int(FS)),
        "SignalSource.implementation": "File_Signal_Source",
        "SignalSource.filename": f"/data/{recording}",
        "SignalSource.item_type": "ibyte",
        "SignalSource.sampling_frequency": str(int(FS)),
        "SignalSource.samples": "0",
        "SignalSource.repeat": "false",
        "SignalSource.enable_throttle_control": "false",
        "SignalConditioner.implementation": "Signal_Conditioner",
        "DataTypeAdapter.implementation": "Ibyte_To_Complex",
        "InputFilter.implementation": "Pass_Through",
        "InputFilter.item_type": "gr_complex",
        "Resampler.implementation": "Pass_Through",
        "Resampler.item_type": "gr_complex",
        "Channels_1B.count": "1",
        "Channels.in_acquisition": "1",
        "Channel0.satellite": "1",
        "Channel.signal": "1B",
        "Acquisition_1B.implementation": "Galileo_E1_PCPS_Ambiguous_Acquisition",
        "Acquisition_1B.item_type": "gr_complex",
        "Acquisition_1B.coherent_integration_time_ms": "4",
        "Acquisition_1B.pfa": "0.00001",
        "Acquisition_1B.doppler_max": "2000",
        "Acquisition_1B.doppler_step": "125",
        "Acquisition_1B.cboc": "false",
        "Tracking_1B.implementation": "Galileo_E1_DLL_PLL_VEML_Tracking",
        "Tracking_1B.item_type": "gr_complex",
        "Tracking_1B.track_pilot": "true",
        "Tracking_1B.dump": "true",
        "Tracking_1B.dump_filename": f"/data/trk_{name}",
        "Tracking_1B.dump_mat": "false",
        "TelemetryDecoder_1B.implementation": "Galileo_E1B_Telemetry_Decoder",
        "Observables.implementation": "Hybrid_Observables",
        "PVT.implementation": "RTKLIB_PVT",
        "PVT.positioning_mode": "Single",
        "PVT.output_rate_ms": "100",
        "PVT.display_rate_ms": "500",
        "PVT.output_path": "/data/pvt",
    }
    return "[GNSS-SDR]\n" + "\n".join(f"{k}={v}" for k, v in kv.items()) + "\n"


def run(name: str, recording: str) -> tuple[np.ndarray, str]:
    (OUT / f"{name}.conf").write_text(config(name, recording))
    cmd = (f"docker run --rm --user {os_ids()} -e HOME=/data/home -v {OUT}:/data -w /data {IMAGE} "
           f"gnss-sdr --config_file=/data/{name}.conf")  # fmt: skip
    res = subprocess.run(["sg", "docker", "-c", cmd], capture_output=True, text=True, check=False)
    log = res.stdout + "\n--- stderr\n" + res.stderr
    (OUT / f"{name}.log").write_text(log)
    files = sorted(OUT.glob(f"trk_{name}*.dat"))
    return (np.fromfile(files[0], dtype=DUMP_DTYPE) if files else np.zeros(0, dtype=DUMP_DTYPE)), log


def steady(rec: np.ndarray, t_min: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
    t = rec["prn_start_sample"].astype(np.float64) / FS
    sel = t > t_min
    e = -rec["acc_phase_rad"][sel].astype(np.float64) - 2 * np.pi * DOPPLER * t[sel]
    e = e - np.polyval(np.polyfit(t[sel], e, 1), t[sel])
    return rec[sel], e


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "home").mkdir(exist_ok=True)
    (OUT / "pvt").mkdir(exist_ok=True)
    loop = CarrierLoop.gnsssdr(35.0, T)
    cn0_p = 10 ** (CN0_PILOT_DB / 10)
    sig_n2 = 1 / (2 * cn0_p * T)
    generate(OUT / "clean.bin", None, 0.0)
    rec, log = run("e1clean", "clean.bin")
    (OUT / "clean.bin").unlink()
    r0, e0 = steady(rec)
    clean = {"records": int(rec.size), "sessions": log.count("Tracking of Galileo E1"),
             "cn0_rep_db": float(np.mean(r0["cn0_dbhz"])), "dt_s": float(np.median(np.diff(r0["prn_start_sample"])) / FS),
             "var_e_over_thermal": float(np.var(e0) / sigma_th2(loop, cn0_p))}  # fmt: skip
    print("clean", clean, flush=True)
    bias_db = clean["cn0_rep_db"] - CN0_PILOT_DB
    pilot_half_chips = boc(e1_code("C"))
    dsp = Despreader(prn=1, fs_hz=FS, doppler_hz=DOPPLER, chip_rate_hz=2 * 1.023e6, code=pilot_half_chips)
    rows = []
    for f_b in (2.0, 5.0):
        f_cw = DOPPLER + 1.023e6 + f_b
        n = np.arange(int(0.4 * FS))
        p_unit = float(np.mean(np.abs(dsp.despread(np.exp(2j * np.pi * f_cw * n / FS))) ** 2))
        rel = 4 * sig_n2 / p_unit
        name = f"e1cw{int(f_b)}"
        generate(OUT / f"{name}.bin", rel, f_cw)
        rec, log = run(name, f"{name}.bin")
        (OUT / f"{name}.bin").unlink()
        r1, e1 = steady(rec, t_min=ONSET_S + 10.0)
        cn0_rep = float(np.mean(r1["cn0_dbhz"]))
        pred = predict(loop, T, f_b, E1C_CS25, 4 * sig_n2, cn0_p)
        row = {"beat_hz": f_b, "cw_rel_power": rel, "sessions": log.count("Tracking of Galileo E1"),
               "meas_r": float(np.var(e1) / sigma_th2(loop, 10 ** (cn0_rep / 10))),
               "meas_cn0_drop_db": float(clean["cn0_rep_db"] - cn0_rep),
               "pred_r_p6prime_cs25": pred["r"], "pred_cn0_drop_db": pred["cn0_drop_db"],
               "pred_r_if_unmodulated": predict(loop, T, f_b, "0", 4 * sig_n2, cn0_p)["r"]}  # fmt: skip
        rows.append(row)
        print(row, flush=True)
    save("s22_gnsssdr_e1_pilot", {"image": IMAGE, "clean": clean, "bias_db": bias_db, "rows": rows})


if __name__ == "__main__":
    main()
