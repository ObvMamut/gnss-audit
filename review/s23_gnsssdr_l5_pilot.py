"""S23: secondary-code whitening on the GNSS-SDR binary for GPS L5 (pilot Q5 with NH20 at 1 ms).

Same protocol as S22. Recording: GPS L5 PRN 1, s = a (D NH10 I5 + j NH20 Q5) (IS-GPS-705J section 3.3.2.2 codes, below),
random CNAV symbols at 100 sym/s, pilot C/N0 47 dB-Hz, 300 Hz Doppler, 15 Msps int8, 45 s. GNSS-SDR tracks the Q5 pilot
(`track_pilot = true`) and wipes NH20 off it each 1 ms integration. Jammer: CW at f_D + 1 MHz + f_b from 15 s, with
despread power 4 sigma_n^2 on the Q5 replica. P6'' predicts r ~ 1 (NH20) against r ~ 40 for a pilot without a secondary
code at the 35 Hz loop.

Redone 20 Oct 2026 (P2b part 2, task 11a; D-40). The first run (9 Oct 2026) copied its codes from GNSS-SDR's own
generator, which reads XB modulo 10230, and ran the stock image, whose replica has the same defect; it could not see the
defect, and its recording was not IS-GPS-705's signal. Now:
- codes: XA xor XB, with XB on its natural 8191-chip period from the Table 3-Ia advance (patch 0003's rule, IS-GPS-705J
  section 3.3.2.2); tests/unit/test_s23_l5_codes.py checks PRN 1's I5 and Q5 chip for chip against XB run from the
  table's initial states (task 9b's ICD-only generator);
- receiver: image p2 (gnssaudit.sdr.images.PATCHED), whose replica carries patch 0003, capped at 2 CPUs and run at
  nice 10 (it shared the machine with a long experiment);
- dump: p2's layout (patch 0001: float64 accumulated phase; patch 0002: a record every integration), read with
  gnssaudit.io.gnsssdr.read_tracking_dump.
The protocol is otherwise unchanged: PRN, C/N0, Doppler, sample rate, length, jammer, seed and the P6'' prediction.

Expectation, stated 20 Oct 2026 before the redo ran: the qualitative result holds. With NH20 wiped, the measured r stays
near 1 (within a factor of 2, H10 in docs/2) in each beat cell that keeps one tracking session; P6'' gives 1.08 at 2 Hz
and 1.03 at 5 Hz, against 43.3 and 37.9 for a pilot without a secondary code. The clean run reads close to 1x thermal,
with C/N0 about 0.4 dB above 47 dB-Hz (M2M4 bias), and its records come every 1 ms (first run: every 10 ms). The first
run (review/results/s23_gnsssdr_l5_pilot.json of 9 Oct) measured r = 0.835 at 5 Hz (C/N0 drop 3.6 dB, P6'' 4.8 dB) and
r = 2.9e7 at 2 Hz, where the receiver lost lock before the jammer started (3 sessions; F24), so that cell was void; its
clean run read 1.014x thermal and 47.37 dB-Hz. A cell with more than one session is void, as before. A cell that keeps
one session and measures r > 3 (H10's falsification bar) contradicts the first run's conclusion.

Result of the redo, 20 Oct 2026 (image p2, 10 min): the expectation holds. Both beat cells keep one session with NH20
locked: r = 1.78 at 2 Hz and 1.63 at 5 Hz (first run: void and 0.835), C/N0 drops 5.3 and 5.5 dB (first run: 3.6 dB at
5 Hz). That is 1.65 and 1.59 times P6'', where the first run read 0.81 times it. Clean run: 1.013x thermal, 47.40 dB-Hz,
a record every 1 ms.

Calibration fixed 22 Oct 2026 (s23 diagnosis, .superpowers/sdd/2026-10-16-p2b-jmax-k-segb-l5-g3/s23-diagnosis-report.md).
The CW was scaled on the despread power at one code-sampling alignment (Despreader: chip 0 on sample 0). GNSS-SDR cuts
its blocks at the code epoch, whose sub-sample offset drifts through a whole sample every 261 blocks, so the receiver
sees the mean over alignments: 1.56 times the design for this code (0.83 times for the first run's). The redo of 20 Oct
therefore jammed at 6.2 sigma_n^2, not 4. The CW is now scaled on `despread_power`, that mean
(tests/unit/test_s23_jammer_calibration.py checks it against a receiver-like correlator). P6'' is linear. The repository's
nonlinear closed-loop simulator (s21's `sim`) reads about 40 % higher for NH20 at this power: the atan2 discriminator's
second-order term is not whitened (s_k^2 = 1), and M2M4 reports about 1 dB more C/N0 than P6'' assumes (its clean bias,
and a constant-envelope jammer counted as less noise than Gaussian). The simulator is the quantitative reference: each
row now carries its r and C/N0 drop for this cell (`simulate`: a tone at 4 sigma_n^2 times NH20, 47 dB-Hz, 35 Hz at
1 ms, r computed as for the binary), beside P6''. Each row also records `secondary_locked`.

Expectation, stated 22 Oct 2026 before this run: one session with NH20 locked in each cell; cw_rel_power 612.0 (2 Hz) and
613.7 (5 Hz); r near the simulator's 1.45 to 1.50 (2 Hz) and 1.44 (5 Hz), driven by this recording's jammer series, and
near the diagnosis' binary run of the same fix (1.51 and 1.45), within 0.1 (F24 run-to-run scatter); C/N0 drop about
4.2 to 4.3 dB (simulator 4.27, diagnosis run 4.17 and 4.27); P6'' stays at 1.08 and 1.03 with a 4.8 dB drop. The
row's simulator, a pure tone, gives about 1.53 to 1.56 (2 Hz) and 1.40 to 1.42 (5 Hz), with drops of 4.3 to 4.4 dB
(diagnosis runs A4 and A5, other seeds).

Result of this run, 22 Oct 2026 (image p2, 13 min): the expectation holds. Both cells keep one session with NH20 locked;
cw_rel_power 612.0 and 613.7. r = 1.51 at 2 Hz and 1.45 at 5 Hz (redo of 20 Oct: 1.78 and 1.63), C/N0 drops 4.17 and
4.27 dB. The simulator gives 1.54 +- 0.01 and 1.43 +- 0.01 with drops of 4.30 and 4.34 dB: the binary is within 2 % of
it, and 1.40 and 1.42 times P6''. Clean run: 1.013x thermal, 47.40 dB-Hz, a record every 1 ms.

Usage: uv run python -m review.s23_gnsssdr_l5_pilot   (needs Docker via sg)
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from gnssaudit.io.gnsssdr import read_tracking_dump
from gnssaudit.sdr.images import PATCH_ID, PATCHED
from gnssaudit.sim.correlator import TrackingConfig, thermal_noise, track
from gnssaudit.uq.loop import CarrierLoop
from review._common import save
from review.s06_mismatch_law import sigma_th2
from review.s17_gnsssdr_identification import WORK, os_ids
from review.s21_overlay_codes import L5Q_NH20, predict

OUT = WORK.parent / "s23"
IMAGE = PATCHED  # image p2: GNSS-SDR's L5 replica with XB on its 8191-chip period (patch 0003, D-40)
FS = 15.0e6
DOPPLER = 300.0
F_L5 = 1176.45e6
CHIP_RATE = 10.23e6
N_CHIPS = 10230
CN0_PILOT_DB = 47.0
NOISE_STD = 6.0
DURATION_S = 45.0
ONSET_S = 15.0
T = 1e-3
L5I_NH10 = "0000110101"
XB_PERIOD_CHIPS = 8191  # IS-GPS-705J 3.3.2.2: XB is not short-cycled; it restarts at its natural completion
XB_ADVANCE = {"I": 266, "Q": 1701}  # IS-GPS-705J Table 3-Ia, PRN 1: XB code advance in chips beyond the all-ones state
DUMP = ("sample_count", "acc_phase_rad", "cn0_dbhz")  # the track_stream columns this study reads


def _lfsr_seq(taps: tuple[int, ...], reset_on_state: tuple[bool, ...] | None) -> np.ndarray:
    """13-stage register as in gps_l5_signal_replica.cc: output stage 13, feedback XOR of the taps."""
    reg = [True] * 13
    out = np.empty(N_CHIPS, dtype=bool)
    for i in range(N_CHIPS):
        out[i] = reg[12]
        if reset_on_state is not None and tuple(reg) == reset_on_state:
            reg = [True] * 13
            continue
        fb = False
        for t in taps:
            fb ^= reg[t]
        reg = [fb, *reg[:-1]]
    return out


def l5_code(kind: str) -> np.ndarray:
    """PRN 1's I5 ("I") or Q5 ("Q") code. XB, run from all ones, is read from its advance modulo its 8191-chip period
    (patch 0003's rule); the first run read it modulo 10230, as stock GNSS-SDR does."""
    xa = _lfsr_seq((12, 11, 9, 8), (True,) * 11 + (False, True))  # l5i_xa_shift / l5q_xa_shift
    xb = _lfsr_seq((12, 11, 7, 6, 5, 3, 2, 0), None)  # l5i_xb_shift / l5q_xb_shift
    chips = xa ^ xb[(XB_ADVANCE[kind] + np.arange(N_CHIPS)) % XB_PERIOD_CHIPS]
    return 1.0 - 2.0 * chips.astype(float)  # gps_l5*_code_gen_float: 1 - 2 c


def despread_power(code: np.ndarray, f_res: float, n_phase: int = 256) -> float:
    """Mean despread power of a unit CW at residual frequency f_res against `code`, over the code-sampling alignment.
    The receiver's 1 ms blocks start at every sub-sample offset from the code epoch in turn (a code period is
    14999.996 samples at 300 Hz Doppler), so it sees this mean, not the value at one alignment (the s25 lesson)."""
    n = np.arange(round(FS * N_CHIPS / CHIP_RATE))
    p = 0.0
    for d in np.arange(n_phase) / n_phase:
        t = (n + d) / FS
        chips = code[np.floor(CHIP_RATE * t).astype(np.int64) % N_CHIPS]
        p += abs(np.mean(chips * np.exp(2j * np.pi * f_res * t))) ** 2
    return p / n_phase


def cw_rel_power(code: np.ndarray, f_cw: float) -> float:
    """CW power, relative to the pilot's a^2, that gives despread power 4 sigma_n^2 on `code` in the receiver."""
    sig_n2 = 1 / (2 * 10 ** (CN0_PILOT_DB / 10) * T)
    return 4 * sig_n2 / despread_power(code, f_cw - DOPPLER)


def simulate(f_b: float, power: float, seed: int, r: int = 8, k: int = 30_000, burn: int = 10_000) -> dict[str, float]:
    """The repository's nonlinear closed-loop simulator (s21's `sim`) in this study's conditions: a tone at beat f_b with
    despread power `power`, times NH20 (the receiver wipes NH20 off the signal, so in its frame the jammer carries it),
    thermal noise at CN0_PILOT_DB, 35 Hz at 1 ms, r independent channels. r and the C/N0 drop are computed as for the
    binary: var(phase) / sigma_th2(reported C/N0), and the reported C/N0 against a clean twin."""
    rng = np.random.default_rng(seed)
    loop = CarrierLoop.gnsssdr(35.0, T)
    cn0 = 10 ** (CN0_PILOT_DB / 10)
    nh20 = np.array([1.0 if c == "0" else -1.0 for c in L5Q_NH20])[np.arange(k) % 20]
    th = thermal_noise(rng, (r, k), cn0, T)
    tone = np.sqrt(power) * np.exp(
        1j * (2 * np.pi * f_b * T * np.arange(k)[None, :] + rng.uniform(0, 2 * np.pi, (r, 1)))
    )
    cfg = TrackingConfig(bn_hz=35.0, t_s=T, pilot=True)
    jam = track(cfg, np.zeros((r, k)), th + tone * nh20)
    clean = track(cfg, np.zeros((r, k)), th)
    cn0_rep = np.mean(jam.cn0_dbhz[:, burn:], axis=1)
    rr = np.var(2 * np.pi * jam.phi[:, burn:], axis=1) / np.array([sigma_th2(loop, 10 ** (c / 10)) for c in cn0_rep])
    drop = np.mean(clean.cn0_dbhz[:, burn:], axis=1) - cn0_rep
    return {
        "r": float(np.mean(rr)),
        "r_se": float(np.std(rr, ddof=1) / np.sqrt(r)),
        "cn0_drop_db": float(np.mean(drop)),
    }


def generate(path: Path, cw_rel_power: float | None, f_cw: float, seed: int = 230) -> None:
    rng = np.random.default_rng(seed)
    c_i, c_q = l5_code("I"), l5_code("Q")
    nh10 = np.array([1.0 if c == "0" else -1.0 for c in L5I_NH10])
    nh20 = np.array([1.0 if c == "0" else -1.0 for c in L5Q_NH20])
    f_chip = CHIP_RATE * (1 + DOPPLER / F_L5)
    a = np.sqrt(10 ** (CN0_PILOT_DB / 10) / FS) * NOISE_STD * np.sqrt(2)
    n_total = int(DURATION_S * FS)
    sym = np.where(rng.random(int(DURATION_S * 100) + 2) < 0.5, -1.0, 1.0)
    chunk = 1 << 22
    with open(path, "wb") as fh:
        for start in range(0, n_total, chunk):
            n = np.arange(start, min(start + chunk, n_total))
            t = n / FS
            chips = 517.3 + f_chip * t
            ci = np.floor(chips).astype(np.int64)
            ms = ci // N_CHIPS
            data = sym[ms // 10] * nh10[ms % 10]
            sig = a * (data * c_i[ci % N_CHIPS] + 1j * nh20[ms % 20] * c_q[ci % N_CHIPS])
            x = sig * np.exp(2j * np.pi * DOPPLER * t)
            x = x + NOISE_STD * (rng.standard_normal(n.size) + 1j * rng.standard_normal(n.size))
            if cw_rel_power is not None:
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
        "Channels_L5.count": "1",
        "Channels.in_acquisition": "1",
        "Channel0.satellite": "1",
        "Channel.signal": "L5",
        "Acquisition_L5.implementation": "GPS_L5i_PCPS_Acquisition",
        "Acquisition_L5.item_type": "gr_complex",
        "Acquisition_L5.pfa": "0.01",
        "Acquisition_L5.doppler_max": "2000",
        "Acquisition_L5.doppler_step": "125",
        "Acquisition_L5.max_dwells": "1",
        "Tracking_L5.implementation": "GPS_L5_DLL_PLL_Tracking",
        "Tracking_L5.item_type": "gr_complex",
        "Tracking_L5.track_pilot": "true",
        "Tracking_L5.dump": "true",
        "Tracking_L5.dump_filename": f"/data/trk_{name}",
        "Tracking_L5.dump_mat": "false",
        "TelemetryDecoder_L5.implementation": "GPS_L5_Telemetry_Decoder",
        "Observables.implementation": "Hybrid_Observables",
        "PVT.implementation": "RTKLIB_PVT",
        "PVT.positioning_mode": "Single",
        "PVT.output_rate_ms": "100",
        "PVT.display_rate_ms": "500",
        "PVT.output_path": "/data/pvt",
    }
    return "[GNSS-SDR]\n" + "\n".join(f"{k}={v}" for k, v in kv.items()) + "\n"


def run(name: str, recording: str) -> tuple[dict[str, np.ndarray], str]:
    (OUT / f"{name}.conf").write_text(config(name, recording))
    cmd = (f"docker run --rm --cpus 2 --user {os_ids()} -e HOME=/data/home -v {OUT}:/data -w /data {IMAGE} "
           f"nice -n 10 gnss-sdr --config_file=/data/{name}.conf")  # fmt: skip
    res = subprocess.run(["sg", "docker", "-c", cmd], capture_output=True, text=True, check=False)
    log = res.stdout + "\n--- stderr\n" + res.stderr
    (OUT / f"{name}.log").write_text(log)
    files = sorted(OUT.glob(f"trk_{name}*.dat"))
    if not files:
        return {k: np.zeros(0) for k in DUMP}, log
    tab = read_tracking_dump(files[0], "patched112", run_id=name, channel=0, sys="G", signal="L5", fs_hz=FS, t_int_s=T)
    return {k: tab.column(k).to_numpy() for k in DUMP}, log


def steady(rec: dict[str, np.ndarray], t_min: float) -> tuple[dict[str, np.ndarray], np.ndarray]:
    t = rec["sample_count"].astype(np.float64) / FS
    sel = t > t_min
    e = -rec["acc_phase_rad"][sel].astype(np.float64) - 2 * np.pi * DOPPLER * t[sel]
    e = e - np.polyval(np.polyfit(t[sel], e, 1), t[sel])
    return {k: v[sel] for k, v in rec.items()}, e


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "home").mkdir(exist_ok=True)
    (OUT / "pvt").mkdir(exist_ok=True)
    loop = CarrierLoop.gnsssdr(35.0, T)
    cn0_p = 10 ** (CN0_PILOT_DB / 10)
    sig_n2 = 1 / (2 * cn0_p * T)
    generate(OUT / "clean.bin", None, 0.0)
    rec, log = run("l5clean", "clean.bin")
    (OUT / "clean.bin").unlink()
    r0, e0 = steady(rec, 10.0)
    clean = {"records": int(rec["sample_count"].size), "sessions": log.count("Tracking of GPS L5"),
             "secondary_locked": "secondary code locked" in log,
             "cn0_rep_db": float(np.mean(r0["cn0_dbhz"])), "dt_s": float(np.median(np.diff(r0["sample_count"])) / FS),
             "var_e_over_thermal": float(np.var(e0) / sigma_th2(loop, cn0_p))}  # fmt: skip
    print("clean", clean, flush=True)
    c_q = l5_code("Q")
    rows = []
    for f_b in (2.0, 5.0):
        f_cw = DOPPLER + 1.0e6 + f_b
        rel = cw_rel_power(c_q, f_cw)
        name = f"l5cw{int(f_b)}"
        generate(OUT / f"{name}.bin", rel, f_cw)
        rec, log = run(name, f"{name}.bin")
        (OUT / f"{name}.bin").unlink()
        r1, e1 = steady(rec, ONSET_S + 10.0)
        cn0_rep = float(np.mean(r1["cn0_dbhz"]))
        pred = predict(loop, T, f_b, L5Q_NH20, 4 * sig_n2, cn0_p)
        sim = simulate(f_b, 4 * sig_n2, seed=2300 + int(f_b))
        row = {"beat_hz": f_b, "cw_rel_power": rel, "sessions": log.count("Tracking of GPS L5"),
               "secondary_locked": "secondary code locked" in log,
               "meas_r": float(np.var(e1) / sigma_th2(loop, 10 ** (cn0_rep / 10))),
               "meas_cn0_drop_db": float(clean["cn0_rep_db"] - cn0_rep),
               "pred_r_sim_nh20": sim["r"], "pred_r_sim_se": sim["r_se"], "pred_cn0_drop_db_sim": sim["cn0_drop_db"],
               "pred_r_p6pp_nh20": pred["r"], "pred_cn0_drop_db": pred["cn0_drop_db"],
               "pred_r_if_unmodulated": predict(loop, T, f_b, "0", 4 * sig_n2, cn0_p)["r"]}  # fmt: skip
        rows.append(row)
        print(row, flush=True)
    codes = "IS-GPS-705J 3.3.2.2: XB on its 8191-chip period from the Table 3-Ia advance (patch 0003's rule)"
    cal = "CW despread power averaged over 256 code-sampling alignments (despread_power; s23 diagnosis, 22 Oct 2026)"
    save(
        "s23_gnsssdr_l5_pilot",
        {"image": IMAGE, "patch_id": PATCH_ID, "codes": codes, "cw_calibration": cal, "clean": clean, "rows": rows},
    )


if __name__ == "__main__":
    main()
