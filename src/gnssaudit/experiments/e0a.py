"""E0(a): B0 and B1 on the clean UGE intervals, Route D, gate G-clean-cal (docs/10 section 2.1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from gnssaudit.eval.route_d import score_route_d
from gnssaudit.experiments import Outcome
from gnssaudit.experiments.common import band, fit_b1_route_d, slice_time, summarize_coverage
from gnssaudit.experiments.uge import load
from gnssaudit.geometry import annotate_elevation
from gnssaudit.truth.route_d import tdcp_residuals
from gnssaudit.uq.baselines import B0, B1


def coverage_rows(
    scored: pa.Table, base: dict[str, Any], levels: list[float], n_boot: int, seed: int
) -> list[dict[str, Any]]:
    """One row per band plus a pooled row ("all")."""
    bands = np.array([band(s, g) for s, g in zip(scored.column("sys").to_pylist(),
                                                 scored.column("signal").to_pylist(), strict=True)])  # fmt: skip
    rows = []
    for b in [*sorted(set(bands.tolist())), "all"]:
        part = scored if b == "all" else scored.filter(bands == b)
        rows.append({**base, "band": b, **summarize_coverage(part, levels, n_boot=n_boot, seed=seed)})
    return rows


def run_e0a(spec: dict[str, Any], run_dir: Path) -> Outcome:
    rd = spec["route_d"]
    levels = spec["levels"]
    boot = spec["bootstrap"]
    data = load(spec["position_interval"])
    counts = dict(data.counts)

    truths, obs_by = {}, {}
    for name in spec["intervals"]:
        iv = data.intervals[name]
        obs = annotate_elevation(slice_time(data.obs, "t_gps_s", iv["t_start"], iv["t_end"]), data.eph, data.position)
        truth, exc = tdcp_residuals(obs, data.eph, data.position, rd["dt_s"], rd["elev_mask_deg"], rd["min_sats"])
        for reason, n in zip(exc.column("reason_code").to_pylist(), exc.column("count").to_pylist(), strict=True):
            counts[f"truth.{reason}"] = counts.get(f"truth.{reason}", 0) + int(n)
        truths[name], obs_by[name] = truth, obs

    c_i = {name: fit_b1_route_d(truths[name], obs_by[name], rd["dt_s"]) for name in spec["intervals"]}
    coverage: list[dict[str, Any]] = []
    for name in spec["intervals"]:
        other = next(n for n in spec["intervals"] if n != name)
        models: list[tuple[B0 | B1, str]] = [
            (B0(**spec["models"]["B0"]), "defaults"),
            (B1(c_i[other]), f"fitted on {other}"),
        ]
        for model, fit_note in models:
            scored, unscored = score_route_d(truths[name], model.predict(obs_by[name]), rd["dt_s"], rd["min_sats"])
            for reason, n in unscored.items():
                key = f"score.{model.model_id}.{reason}"
                counts[key] = counts.get(key, 0) + n
            base = {"model": model.model_id, "interval": name, "fit": fit_note}
            coverage += coverage_rows(scored, base, levels, boot["n_boot"], boot["seed"])

    g = spec["gate"]
    pooled = [r for r in coverage if r["model"] == g["model"] and r["band"] == "all"]
    passed = bool(pooled) and all(g["lo"] <= r[f"c{g['level']:g}"] <= g["hi"] for r in pooled)
    gate = {**g, "passed": passed, "values": {r["interval"]: r[f"c{g['level']:g}"] for r in pooled},
            "labels": {r["interval"]: r.get("label") for r in pooled}}  # fmt: skip
    result = {
        "experiment": "E0a",
        "position_ecef_m": data.position.tolist(),
        "position_spread_m": data.position_spread_m,
        "b1_c_i_m2hz": c_i,
        "truth_rows": {n: t.num_rows for n, t in truths.items()},
        "coverage": coverage,
        "gate": gate,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    summary = {"gate_passed": passed, "gate_values": gate["values"], "gate_labels": gate["labels"]}
    return Outcome(inputs=data.inputs, counts=counts, summary=summary)
