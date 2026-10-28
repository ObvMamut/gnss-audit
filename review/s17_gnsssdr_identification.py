"""S17: identify the carrier loop of the real GNSS-SDR v0.0.22 binary (G-hz on synthetic signal; F1, kappa timing).

A synthetic GPS L1 C/A recording (PRN 1, 300 Hz Doppler, 55 dB-Hz, random 50 bit/s data, int8 IQ at 4 Msps)
carries a known common-mode phase process phi(t) = sum_k a_k sin(2 pi f_k t + psi_k) on its carrier
(docs/3 section 2.5). GNSS-SDR runs it with different tracking configurations; its tracking dump gives the
discriminator d (field 14, cycles) and the accumulated NCO phase (field 9). Lock-in estimates at each tone give
the measured responses |H_d| (tone -> discriminator) and |H_phi| (tone -> reported phase), compared with the
harness models CarrierLoop.gnsssdr for the simplified (kappa = 0) and source (kappa = 1/2) timings.

Usage:
  uv run python -m review.s17_gnsssdr_identification generate
  uv run python -m review.s17_gnsssdr_identification run      # needs Docker (sg docker)
  uv run python -m review.s17_gnsssdr_identification analyse
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field

import numpy as np

from gnssaudit.gates import lockin, stamp_and_verdict
from gnssaudit.io.gnsssdr import STOCK_108
from gnssaudit.paths import results_root
from gnssaudit.sim.despread import ca_code
from gnssaudit.uq.loop import CarrierLoop
from review._common import save

WORK = results_root() / "review-work" / "s17"
IMAGE = "carlesfernandez/docker-gnsssdr:v0.0.22@sha256:befe06ed81163068e9eaf918024e18cedf1fc374925431320d57055650b48b79"  # containers/digests.md
FS = 4.0e6
DURATION_S = 100.0
DOPPLER = 300.0
CN0_DB = 55.0
PRN = 1
F_L1 = 1575.42e6
TONES_HZ = [0.3, 0.7, 1.5, 3.0, 5.0, 8.0, 12.0, 17.0, 23.0, 31.0, 43.0]
TONE_AMP_RAD = 0.04
NOISE_STD = 16.0  # int8 counts per quadrature


@dataclass(frozen=True)
class Config:
    """One GNSS-SDR tracking configuration; T and the bandwidth come from here, never from the name."""

    overrides: dict[str, str] = field(default_factory=dict[str, str])
    t_s: float = 1e-3  # loop update interval (coherent integration)
    t_code_s: float | None = None  # code period under extended integration (None: T itself)
    configured_bw_hz: float = 35.0  # the value GNSS-SDR is given (pll_bw_hz, or pll_bw_narrow_hz when extended)
    target_bn_hz: float | None = None  # the B_n,eff the inverse map (docs/9 section 5.3) aims at


def _ext20(bw_hz: float, target: float | None = None) -> Config:
    ov = {"Tracking_1C.extend_correlation_symbols": "20", "Tracking_1C.pll_bw_narrow_hz": str(bw_hz)}
    return Config(ov, t_s=20e-3, t_code_s=1e-3, configured_bw_hz=bw_hz, target_bn_hz=target)


def _bw1(bw_hz: float, target: float | None = None) -> Config:
    return Config({"Tracking_1C.pll_bw_hz": str(bw_hz)}, configured_bw_hz=bw_hz, target_bn_hz=target)


CONFIGS: dict[str, Config] = {
    "default": Config(),  # expect pll_bw_hz = 35 Hz at T = 1 ms (F1)
    "bw15_1ms": _bw1(15.0),
    "ext20_narrow15": _ext20(15.0),
    # E3 cells: configured bandwidths from the inverse map (docs/9 section 5.3), aimed at B_n,eff = 5 and 15 Hz
    "e3_5hz_1ms": _bw1(5.84, 5.0),
    "e3_15hz_1ms": _bw1(17.47, 15.0),
    "e3_5hz_20ms": _ext20(4.50, 5.0),
    "e3_15hz_20ms": _ext20(9.66, 15.0),
}


def tone_phases() -> np.ndarray:
    return np.random.default_rng(17).uniform(0, 2 * np.pi, len(TONES_HZ))


def phi_of_t(t: np.ndarray, psi: np.ndarray) -> np.ndarray:
    out = np.zeros_like(t)
    for f, p in zip(TONES_HZ, psi, strict=True):
        out += TONE_AMP_RAD * np.sin(2 * np.pi * f * t + p)
    return out


def generate() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(170)
    psi = tone_phases()
    code = ca_code(PRN)
    f_code = 1.023e6 * (1 + DOPPLER / F_L1)
    amp = np.sqrt(10 ** (CN0_DB / 10) / FS) * NOISE_STD * np.sqrt(2)  # A^2 / (N0) = C/N0, N0 = 2 sigma^2 / fs
    n_total = int(DURATION_S * FS)
    n_bits = int(DURATION_S * 50) + 2
    bits = np.where(rng.random(n_bits) < 0.5, -1.0, 1.0)
    chip0 = 123.4  # initial code phase in chips
    chunk = 1 << 22
    with open(WORK / "sig.bin", "wb") as fh:
        for start in range(0, n_total, chunk):
            n = np.arange(start, min(start + chunk, n_total))
            t = n / FS
            chips = chip0 + f_code * t
            c = code[np.floor(chips).astype(np.int64) % 1023]
            d = bits[np.floor(chips / (1023 * 20)).astype(np.int64)]
            carrier = 2 * np.pi * DOPPLER * t + phi_of_t(t, psi)
            s = amp * d * c * np.exp(1j * carrier)
            x = s + NOISE_STD * (rng.standard_normal(n.size) + 1j * rng.standard_normal(n.size))
            iq = np.empty(2 * n.size, dtype=np.int8)
            iq[0::2] = np.clip(np.round(x.real), -127, 127)
            iq[1::2] = np.clip(np.round(x.imag), -127, 127)
            iq.tofile(fh)
    meta = {"fs": FS, "duration_s": DURATION_S, "doppler_hz": DOPPLER, "cn0_db": CN0_DB, "prn": PRN,
            "tones_hz": TONES_HZ, "tone_amp_rad": TONE_AMP_RAD, "tone_phase_rad": psi.tolist(), "chip0": chip0}  # fmt: skip
    (WORK / "sig.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {WORK / 'sig.bin'}")


def config_text(name: str) -> str:
    base = {
        "GNSS-SDR.internal_fs_sps": str(int(FS)),
        "SignalSource.implementation": "File_Signal_Source",
        "SignalSource.filename": "/data/sig.bin",
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
        "Channels_1C.count": "1",
        "Channels.in_acquisition": "1",
        "Channel0.satellite": str(PRN),
        "Channel0.signal": "1C",
        "Acquisition_1C.implementation": "GPS_L1_CA_PCPS_Acquisition",
        "Acquisition_1C.item_type": "gr_complex",
        "Acquisition_1C.coherent_integration_time_ms": "1",
        "Acquisition_1C.pfa": "0.01",
        "Acquisition_1C.doppler_max": "2000",
        "Acquisition_1C.doppler_step": "100",
        "Tracking_1C.implementation": "GPS_L1_CA_DLL_PLL_Tracking",
        "Tracking_1C.item_type": "gr_complex",
        "Tracking_1C.dump": "true",
        "Tracking_1C.dump_filename": f"/data/trk_{name}",
        "Tracking_1C.dump_mat": "false",
        "TelemetryDecoder_1C.implementation": "GPS_L1_CA_Telemetry_Decoder",
        "Observables.implementation": "Hybrid_Observables",
        "PVT.implementation": "RTKLIB_PVT",
        "PVT.positioning_mode": "Single",
        "PVT.output_rate_ms": "100",
        "PVT.display_rate_ms": "500",
        "PVT.output_path": "/data/pvt",
    }
    base.update(CONFIGS[name].overrides)
    return "[GNSS-SDR]\n" + "\n".join(f"{k}={v}" for k, v in base.items()) + "\n"


def run() -> None:
    (WORK / "pvt").mkdir(exist_ok=True)
    (WORK / "home").mkdir(exist_ok=True)  # the image's HOME is not writable for a non-root UID (gnuradio cache)
    for name in CONFIGS:
        (WORK / f"{name}.conf").write_text(config_text(name))
        cmd = (f"docker run --rm --user {os_ids()} -e HOME=/data/home -v {WORK}:/data -w /data {IMAGE} "
               f"gnss-sdr --config_file=/data/{name}.conf")  # fmt: skip
        print("$", cmd, flush=True)
        res = subprocess.run(["sg", "docker", "-c", cmd], capture_output=True, text=True, check=False)
        (WORK / f"{name}.log").write_text(res.stdout + "\n--- stderr\n" + res.stderr)
        print(name, "exit", res.returncode, flush=True)


def os_ids() -> str:
    return f"{os.getuid()}:{os.getgid()}"


DUMP_DTYPE = STOCK_108  # the v0.0.22 tracking-dump record (docs/9 section 4.1); s18, s22, s23, s25 import it


def analyse() -> None:
    meta = json.loads((WORK / "sig.json").read_text())
    psi = np.array(meta["tone_phase_rad"])
    # lockin returns C with y = Re(C exp(j 2 pi f t)); a sin(2 pi f t + psi) = Re(-j a exp(j psi) exp(j 2 pi f t))
    tone_in = -1j * TONE_AMP_RAD * np.exp(1j * psi)
    rows = []
    for name in CONFIGS:
        files = sorted(WORK.glob(f"trk_{name}*.dat"))
        if not files:
            print(name, "no dump")
            continue
        rec = np.fromfile(files[0], dtype=DUMP_DTYPE)
        cfg = CONFIGS[name]
        t_s = cfg.t_s
        tt = rec["prn_start_sample"].astype(np.float64) / FS
        steady = tt > tt[0] + 15.0
        r = rec[steady]
        t = tt[steady]
        dt_med = float(np.median(np.diff(t)))
        d_cyc = r["carr_err_cyc"].astype(np.float64)
        acc = r["acc_phase_rad"].astype(np.float64)
        # reported phase: remove the Doppler ramp; GNSS-SDR accumulates with a minus sign
        acc_res = -acc - 2 * np.pi * DOPPLER * t
        ld = lockin(2 * np.pi * d_cyc, t, TONES_HZ)
        lp = lockin(acc_res - np.polyval(np.polyfit(t, acc_res, 1), t), t, TONES_HZ)
        meas_hd = ld[:-1] / tone_in
        meas_hp = lp[:-1] / tone_in
        cfg_bw = cfg.configured_bw_hz
        tc = cfg.t_code_s
        models = {
            "simplified_kappa0": CarrierLoop.gnsssdr(cfg_bw, t_s, kappa=0.0),
            "source_kappa_half": CarrierLoop.gnsssdr(cfg_bw, t_s, kappa=0.5, t_code_s=tc),
            "if_5hz_were_used": CarrierLoop.gnsssdr(5.0, t_s, kappa=0.5, t_code_s=tc),
        }
        f_arr = np.array(TONES_HZ)
        w = 2 * np.pi * f_arr * t_s
        # the loop input is the block average of the carrier phase: a tone enters scaled by sinc(f T)
        block_avg = np.abs(np.sinc(f_arr * t_s))
        out: dict[str, object] = {
            "config": name, "configured_bw_hz": cfg_bw, "records": int(r.size), "median_dt_s": dt_med,
            "mean_cn0_dbhz": float(np.mean(r["cn0_dbhz"])), "mean_lock": float(np.mean(r["lock"])),
            "tones_hz": TONES_HZ, "meas_abs_hd": np.abs(meas_hd), "meas_abs_hphi": np.abs(meas_hp),
            "lockin_resid_sd_d_rad": float(ld[-1].real),
            "amp_se_rad": float(ld[-1].real * np.sqrt(2.0 / r.size)),
        }  # fmt: skip
        for mname, loop in models.items():
            hd = np.abs(loop.freq_response(w, "disc")) * block_avg
            hp = np.abs(loop.freq_response(w)) * block_avg
            sel = np.abs(meas_hd) > 0.15  # tones where the discriminator response is well above the noise floor
            out[mname] = {"abs_hd": hd, "abs_hphi": hp, "bn_eff_hz": loop.bn_eff_hz(),
                          "rms_log_err_hd": float(np.sqrt(np.mean(np.log(np.abs(meas_hd[sel]) / hd[sel]) ** 2))),
                          "rms_log_err_hphi": float(np.sqrt(np.mean(np.log(np.abs(meas_hp) / hp) ** 2))),
                          "max_abs_log_err_hphi": float(np.max(np.abs(np.log(np.abs(meas_hp) / hp))))}  # fmt: skip
        # time-stamp convention: fit a delay to the complex responses (source-timing model, block-average sinc)
        src = models["source_kappa_half"]
        ghz_ok_by_lab: dict[str, bool] = {}
        for lab, meas, resp, sd in (
            ("hphi", meas_hp, src.freq_response(w), float(lp[-1].real)),
            ("hd", meas_hd, src.freq_response(w, "disc"), float(ld[-1].real)),
        ):
            off_s, resid_deg, ghz_rows, ghz_ok = stamp_and_verdict(
                meas, resp, t_s, f_arr, sd, r.size, tone_amp_rad=TONE_AMP_RAD
            )
            out[f"stamp_offset_T_{lab}"] = off_s / t_s
            out[f"ghz_{lab}"] = {"rows": ghz_rows, "passed": ghz_ok}
            ghz_ok_by_lab[lab] = ghz_ok
            out[f"rms_phase_resid_deg_{lab}"] = resid_deg
        out["ghz_pass"] = ghz_ok_by_lab["hphi"]  # G-hz is judged on the reported phase; H_d is reported beside it
        out["model_bn_eff_hz"] = src.bn_eff_hz()
        out["target_bn_hz"] = cfg.target_bn_hz
        rows.append(out)
        print(json.dumps({k: (v if not isinstance(v, np.ndarray) else np.round(v, 4).tolist()) for k, v in out.items()
                          if k not in models}, indent=None))  # fmt: skip
        for mname in models:
            m = out[mname]
            assert isinstance(m, dict)
            print("  ", mname, "Bn_eff=%.2f" % m["bn_eff_hz"], "rms_log_err_hd=%.4f" % m["rms_log_err_hd"],
                  "rms_log_err_hphi=%.4f" % m["rms_log_err_hphi"], "max=%.4f" % m["max_abs_log_err_hphi"])  # fmt: skip
    save("s17_gnsssdr_identification", {"image": IMAGE, "signal": meta, "rows": rows})


if __name__ == "__main__":
    {"generate": generate, "run": run, "analyse": analyse}[sys.argv[1]]()
