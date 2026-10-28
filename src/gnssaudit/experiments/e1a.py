"""E1 arm a: the audit of u-blox cpStdev and C/N0-based models under real jamming, UGE 1.6.4 (docs/10 2.2, H1a).

Route D at the provisional position (D-30). Each model is scored per jammer-power class of the up-and-down 2 dB
ramp (the two halves pooled per class), and the
verdict is withdrawn if shifting the position by 1 m in any direction moves a pooled c(0.95) by 0.005 or more.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from gnssaudit.eval.route_d import score_route_d
from gnssaudit.experiments import Outcome
from gnssaudit.experiments.common import b6_innovations, band, fit_b1_route_d, slice_time, summarize_coverage
from gnssaudit.experiments.uge import UgeData, load
from gnssaudit.geometry import annotate_elevation, ecef_to_geodetic
from gnssaudit.io.kp import kp_max, read_kp_json
from gnssaudit.io.uge import mean_position_ecef
from gnssaudit.paths import data_root
from gnssaudit.truth.route_d import tdcp_residuals
from gnssaudit.uq.baselines import B0, B1, B2, B4, B6, Flag

FloatArray = NDArray[np.float64]
KP_FILE = data_root() / "raw" / "gfz-kp" / "kp_2024-09-09.json"


def ramp_power_dbm(t: FloatArray, t_on: float, ramp: dict[str, float]) -> FloatArray:
    """Jammer power at each time of the up-and-down step ramp (2024 catalogue, group 6): `levels` levels up from
    p0_dbm, then back down, each held hold_s. NaN outside the ramp."""
    top = ramp["levels"] - 1
    k = np.floor((t - t_on) / ramp["hold_s"])
    index = top - np.abs(k - top)
    return np.where((k >= 0) & (k <= 2 * top), ramp["p0_dbm"] + ramp["step_db"] * index, np.nan)


def power_class(p_dbm: FloatArray, ramp: dict[str, float]) -> list[str]:
    """10 dB classes counted from the ramp's first level, as "[lo,hi) dBm"."""
    w, p0 = ramp["class_width_db"], ramp["p0_dbm"]
    lo = p0 + w * np.floor((p_dbm - p0) / w)
    return [f"[{a:g},{a + w:g}) dBm" if np.isfinite(a) else "outside ramp" for a in lo]


def _class_order(label: str) -> float:
    return float(label[1:].split(",")[0]) if label.startswith("[") else float("-inf")


def enu_shift(x: FloatArray, east: float, north: float, up: float) -> FloatArray:
    """x moved by (east, north, up) metres in its local frame."""
    lat, lon, _ = ecef_to_geodetic(x)
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    e = np.array([-so, co, 0.0])
    n = np.array([-sl * co, -sl * so, cl])
    u = np.array([cl * co, cl * so, sl])
    return x + east * e + north * n + up * u


def uncensored_only(pred: pa.Table) -> pa.Table:
    """B4 read on uncensored epochs only: code-14 rows (censored at 0.056 cycles) become invalid."""
    censored = (pred.column("flags").to_numpy() & int(Flag.CENSORED)) != 0
    valid = pred.column("valid").to_numpy(zero_copy_only=False) & ~censored
    pred = pred.set_column(pred.schema.get_field_index("valid"), "valid", pa.array(valid))
    ids = pa.array(["B4-uncensored-only"] * pred.num_rows, pa.string())
    return pred.set_column(pred.schema.get_field_index("model_id"), "model_id", ids)


def _truth_whole_log(data: UgeData, position: FloatArray, rd: dict[str, Any]) -> tuple[pa.Table, pa.Table]:
    obs = annotate_elevation(data.obs, data.eph, position)
    truth, _ = tdcp_residuals(obs, data.eph, position, rd["dt_s"], rd["elev_mask_deg"], rd["min_sats"])
    return obs, truth


def _within(truth: pa.Table, iv: dict[str, Any], dt_s: float) -> pa.Table:
    """Increments that start and end inside the interval."""
    return slice_time(truth, "t_rx_s", iv["t_start"], iv["t_end"] - dt_s)


def _predictions(spec: dict[str, Any], obs: pa.Table, c_i: dict[str, float], innov: pa.Table) -> list[pa.Table]:
    m = spec["models"]
    b4 = B4().predict(obs)
    b4_bound = b4.set_column(b4.schema.get_field_index("model_id"), "model_id",
                             pa.array(["B4-censored-at-bound"] * b4.num_rows, pa.string()))  # fmt: skip
    b6 = B6(innov=innov, window_s=m["B6"]["window_s"], lag_s=m["B6"]["lag_s"], min_count=m["B6"]["min_count"])
    return [
        B0(**m["B0"]).predict(obs),
        B1(c_i).predict(obs),
        B2(**m["B2"]).predict(obs),
        b4_bound,
        uncensored_only(b4),
        b6.predict(obs),
    ]


def _scored(spec: dict[str, Any], jam_truth: pa.Table, preds: list[pa.Table]) -> dict[str, pa.Table]:
    rd = spec["route_d"]
    out = {}
    for pred in preds:
        scored, _ = score_route_d(jam_truth, pred, rd["dt_s"], rd["min_sats"])
        out[pred.column("model_id")[0].as_py()] = scored
    return out


def _pooled_c95(scored: pa.Table) -> float:
    w = scored.column("w").to_numpy()
    return float(np.mean(np.abs(w) <= 1.959963984540054)) if w.size else float("nan")


def _t3(obs: pa.Table, classes: list[str], name: str) -> list[dict[str, Any]]:
    code = obs.column("cp_std_code").fill_null(15).to_numpy()
    rows = []
    for c in [*sorted(set(classes), key=_class_order), "all"]:
        sel = np.ones(code.size, bool) if c == "all" else np.array(classes) == c
        hist = np.bincount(code[sel], minlength=16).tolist()
        n = int(sel.sum())
        rates = {f"rate_code{k}": hist[k] / n if n else None for k in (14, 15)}
        rows.append({"interval": name, "class": c, "n": n, "histogram": hist, **rates})
    return rows


def _pooled_at(spec: dict[str, Any], data: UgeData, position: FloatArray, c_i: dict[str, float]) -> dict[str, float]:
    """Pooled jammed c(0.95) per model with the truth and B6's innovations recomputed at another position."""
    rd, jam = spec["route_d"], data.intervals[spec["jam_interval"]]
    obs, truth = _truth_whole_log(data, position, rd)
    jam_obs = slice_time(obs, "t_gps_s", jam["t_start"], jam["t_end"])
    preds = _predictions(spec, jam_obs, c_i, b6_innovations(truth))
    return {m: _pooled_c95(sc) for m, sc in _scored(spec, _within(truth, jam, rd["dt_s"]), preds).items()}


def run_e1a(spec: dict[str, Any], run_dir: Path) -> Outcome:
    rd, levels, boot, ramp = spec["route_d"], spec["levels"], spec["bootstrap"], spec["power_ramp"]
    data = load(spec["position_interval"])
    jam = data.intervals[spec["jam_interval"]]
    counts = dict(data.counts)

    obs, truth = _truth_whole_log(data, data.position, rd)
    clean_truth = pa.concat_tables([_within(truth, data.intervals[n], rd["dt_s"]) for n in spec["clean_intervals"]])
    c_i = fit_b1_route_d(clean_truth, obs, rd["dt_s"])
    jam_truth = _within(truth, jam, rd["dt_s"])
    jam_obs = slice_time(obs, "t_gps_s", jam["t_start"], jam["t_end"])
    innov = b6_innovations(truth)
    scored = _scored(spec, jam_truth, _predictions(spec, jam_obs, c_i, innov))

    coverage: list[dict[str, Any]] = []
    for model, sc in scored.items():
        t = sc.column("t_rx_s").to_numpy()
        classes = np.array(power_class(ramp_power_dbm(t, jam["t_start"], ramp), ramp))
        bands = np.array([band(s, g) for s, g in zip(sc.column("sys").to_pylist(), sc.column("signal").to_pylist(),
                                                     strict=True)])  # fmt: skip
        groups = [("all", "all", np.ones(t.size, bool))]
        groups += [(c, "all", classes == c) for c in sorted(set(classes.tolist()), key=_class_order)]
        groups += [("all", b, bands == b) for b in sorted(set(bands.tolist()))]
        for cls, bnd, sel in groups:
            stats = summarize_coverage(sc.filter(sel), levels, n_boot=boot["n_boot"], seed=boot["seed"])
            coverage.append({"model": model, "class": cls, "band": bnd, **stats})
        counts[f"score.{model}.jam_truth_rows_unscored"] = jam_truth.num_rows - sc.num_rows

    # Clean reference (H1a compares jammed with clean coverage). B1 is in-sample here: its C_i come from these data.
    clean_obs = pa.concat_tables([slice_time(obs, "t_gps_s", data.intervals[n]["t_start"], data.intervals[n]["t_end"])
                                  for n in spec["clean_intervals"]])  # fmt: skip
    for model, sc in _scored(spec, clean_truth, _predictions(spec, clean_obs, c_i, innov)).items():
        stats = summarize_coverage(sc, levels, n_boot=boot["n_boot"], seed=boot["seed"])
        coverage.append({"model": model, "class": "clean", "band": "all", "in_sample": model == "B1", **stats})

    t_obs = jam_obs.column("t_gps_s").to_numpy()
    t3 = _t3(jam_obs, power_class(ramp_power_dbm(t_obs, jam["t_start"], ramp), ramp), spec["jam_interval"])
    for name in spec["clean_intervals"]:
        iv_obs = slice_time(obs, "t_gps_s", data.intervals[name]["t_start"], data.intervals[name]["t_end"])
        t3 += [r for r in _t3(iv_obs, ["clean"] * iv_obs.num_rows, name) if r["class"] == "all"]

    # Sensitivity: the truth and B6 recomputed at shifted positions; B1's C_i kept, so only the truth moves.
    base = {m: _pooled_c95(sc) for m, sc in scored.items()}
    sens = spec["sensitivity"]
    shifts = []
    for axis in range(3):
        for sign in (1.0, -1.0):
            d = [0.0, 0.0, 0.0]
            d[axis] = sign * sens["shift_m"]
            shifted = _pooled_at(spec, data, enu_shift(data.position, *d), c_i)
            delta = {m: c - base[m] for m, c in shifted.items()}
            shifts.append({"enu_m": d, "delta_c95": delta, "max_abs": max(abs(v) for v in delta.values())})
    withdrawn = any(s["max_abs"] >= sens["max_abs_delta_c95"] for s in shifts)
    xiv = data.intervals[sens["extra_position_interval"]]
    x_pos, _ = mean_position_ecef(data.pvt, xiv["t_start"], xiv["t_end"])
    extra = {
        "interval": sens["extra_position_interval"],
        "offset_m": float(np.linalg.norm(x_pos - data.position)),
        "delta_c95": {m: c - base[m] for m, c in _pooled_at(spec, data, x_pos, c_i).items()},
    }

    # G1 for arm a (docs/3 section 4.7, amended by D-30).
    g = spec["g1"]
    jam_min = (jam["t_end"] - jam["t_start"]) / 60.0
    t_truth = jam_truth.column("t_rx_s").to_numpy()
    epochs = np.unique(t_truth)
    jam_epochs = np.arange(jam["t_start"], jam["t_end"] - rd["dt_s"], rd["dt_s"]).size
    kp = kp_max(read_kp_json(KP_FILE), jam["t_start"], jam["t_end"])
    criteria = {
        "coordinates": {"source": "nav-pvt-provisional (D-30)", "passed": not withdrawn,
                        "spread_m": data.position_spread_m},
        "jam_duration": {"minutes": jam_min, "class": jam["class"], "passed": jam_min >= g["min_jam_min"]},
        "route_d_sats": {"epochs_with_truth": int(epochs.size), "jam_epochs": int(jam_epochs),
                         "fraction": epochs.size / jam_epochs, "min_fraction": g["min_epoch_fraction"],
                         "passed": epochs.size / jam_epochs >= g["min_epoch_fraction"]},
        "kp": {"kp_max": kp, "passed": kp <= g["kp_max"]},
    }  # fmt: skip
    g1_passed = all(c["passed"] for c in criteria.values())
    result = {
        "experiment": "E1a",
        "position_ecef_m": data.position.tolist(),
        "b1_c_i_m2hz": c_i,
        "jam_truth_rows": jam_truth.num_rows,
        "coverage": coverage,
        "t3_cpstdev": t3,
        "sensitivity": {"base_c95": base, "shifts": shifts, "withdrawn": withdrawn, "extra_position": extra},
        "g1": {"passed": g1_passed, "criteria": criteria},
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    summary = {"g1_passed": g1_passed, "withdrawn": withdrawn, "pooled_c95": base}
    return Outcome(inputs=[*data.inputs, KP_FILE], counts=counts, summary=summary)
