"""Figures for docs/14 from review/results/*.json, written to figures/ (palette: dataviz reference slots 1-4,
validated light mode).

Contrast relief for slots 3-4 (below 3:1 on the surface): every series is direct-labelled, uses its own marker
shape, and the numbers are tabulated in docs/14.

`g3` draws docs/10's E2 and G3 figures (f10-*) from the runs, read-only; the sources are pinned by run id and SHA-256
(G3_*), and any other source is refused (`uv run python -I review/figures.py g3`).

`p2c` draws docs/10's run 3 figures, f10-6 to f10-8, the same way: E2 run 3, run 2, s33 and s34, each pinned (P2C_*),
and any other source refused (`uv run python -I review/figures.py p2c`).

The pinned run files are also committed in run-summaries/, laid out like the results root, so `make figures` redraws
every figure without the runs (D-48).
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from gnssaudit.experiments.e2_frontend import FrontEndCell, cell_verdict  # noqa: E402
from gnssaudit.paths import results_root  # noqa: E402
from gnssaudit.provenance import sha256_file  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
OUT = Path(__file__).resolve().parents[1] / "figures"
SURFACE = "#fcfcfb"
INK, INK2, GRID = "#1f1f1e", "#5f5e58", "#e4e3dd"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"


def _style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _load(name: str) -> dict:
    return json.loads((RES / f"{name}.json").read_text())


def fig_resonance() -> None:
    rows = _load("s14_doppler_resonance")["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE, sharey=True)
    for ax, bn in zip(axes, (5.0, 35.0), strict=True):
        sel = sorted((r for r in rows if r["bn_cfg_hz"] == bn and abs(r["doppler_offset_hz"]) <= 40),
                     key=lambda r: r["doppler_offset_hz"])  # fmt: skip
        x = [r["doppler_offset_hz"] for r in sel]
        ax.plot(x, [r["pred"]["r_corrected"] for r in sel], color=C1, linewidth=2, label="P6 prediction")
        ax.plot(x, [r["sim"]["r_corrected"] for r in sel], linestyle="none", marker="o", markersize=7,
                markerfacecolor=C2, markeredgecolor=SURFACE, markeredgewidth=1.5, label="closed-loop simulation")  # fmt: skip
        ax.set_yscale("log")
        ax.axhline(1.0, color=INK2, linewidth=1, linestyle=":")
        ax.text(39, 1.15, "calibrated (r = 1)", color=INK2, fontsize=8, ha="right")
        ax.set_xlabel("satellite Doppler minus 2 kHz (Hz)")
        ax.set_title(f"configured {bn:g} Hz loop, T = 1 ms", fontsize=10)
        _style(ax)
    axes[0].set_ylabel("variance mismatch r\n(true / implied by reported C/N₀)")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    fig.suptitle("PPD chirp (20 µs period, J/C = 25 dB): Doppler-resonant phase-error ridge at flat C/N₀",
                 color=INK, fontsize=11)  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "f14-1_doppler_resonance.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_identifiable() -> None:
    s02 = [r for r in _load("s02_window_and_p5")["rows"] if r["loop"] == "gnsssdr5" and r["W_s"] == 2.0]
    s07 = [r for r in _load("s07_pilot_data_witness")["part_b"]
           if r["bn_cfg_hz"] == 5.0 and r["corner_hz"] != "thermal_only"]  # fmt: skip
    s02 = sorted(s02, key=lambda r: r["corner_hz"])
    s07 = sorted(s07, key=lambda r: r["corner_hz"])
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    xc = [r["corner_hz"] for r in s02]
    series = [
        (xc, [r["phi_mc"] for r in s02], C1, "o", "IW, Monte Carlo"),
        (xc, [r["pred"]["docs4_P5_centred"] for r in s02], C2, "s", "P5 as in docs/4"),
        (xc, [r["pred"]["projection_once"] for r in s02], C3, "D", "P5 corrected (one projection)"),
        ([r["corner_hz"] for r in s07], [r["pdw_ratio"] for r in s07], C4, "^", "pilot–data witness (N1)"),
    ]
    for x, y, c, mk, lab in series:
        ax.plot(x, y, color=c, linewidth=2, marker=mk, markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=lab)  # fmt: skip
    # direct labels placed where the curves separate; IW Monte Carlo and the corrected P5 coincide
    i03 = xc.index(0.3)
    ax.annotate("IW Monte Carlo ≈ corrected P5", (0.3, series[0][1][i03]), textcoords="offset points",
                xytext=(-12, 14), fontsize=8, color=INK, ha="right")  # fmt: skip
    ax.annotate("P5 as in docs/4", (0.3, series[1][1][i03]), textcoords="offset points", xytext=(8, -14),
                fontsize=8, color=INK)  # fmt: skip
    ax.annotate("pilot–data witness (N1)", (0.1, series[3][1][1]), textcoords="offset points", xytext=(0, 12),
                fontsize=8, color=INK, ha="center")  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlabel("disturbance corner frequency (Hz)")
    ax.set_ylabel("recovered fraction of Var(e)")
    ax.set_ylim(0, 1.25)
    ax.axhline(1.0, color=INK2, linewidth=1, linestyle=":")
    ax.set_title("Identifiable fraction, configured 5 Hz GNSS-SDR loop, W = 2 s", fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "f14-2_identifiable_fraction.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_beat() -> None:
    rows = _load("s06_mismatch_law")["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE, sharey=True)
    for ax, bn in zip(axes, (5.0, 35.0), strict=True):
        for rel, c, mk, lab in ((0.5, C1, "o", "tone variance 0.5 σ²"), (2.0, C2, "s", "tone variance 2 σ²")):
            sel = sorted((r for r in rows if r["sim"]["bn_cfg_hz"] == bn
                          and abs(r["sim"]["power"] / (2 / (2 * 10**4.5 * 1e-3)) - rel) < 1e-6),
                         key=lambda r: r["sim"]["f_b_hz"])  # fmt: skip
            x = [r["sim"]["f_b_hz"] for r in sel]
            ax.plot(x, [r["pred"]["r_corrected"] for r in sel], color=c, linewidth=2)
            ax.plot(x, [r["sim"]["r_corrected"] for r in sel], linestyle="none", marker=mk, markersize=7,
                    markerfacecolor=c, markeredgecolor=SURFACE, markeredgewidth=1.5, label=lab)  # fmt: skip
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axhline(1.0, color=INK2, linewidth=1, linestyle=":")
        ax.set_xlabel("despread beat frequency (Hz)")
        ax.set_title(f"configured {bn:g} Hz loop (lines: P6; markers: simulation)", fontsize=10)
        _style(ax)
    axes[0].set_ylabel("variance mismatch r")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "f14-3_mismatch_vs_beat.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


G2_RUN = "b7a0a7f6d31a"
G2_OUT = Path(__file__).resolve().parents[1] / "outreach" / "figures"
# (result-file model key, label, colour, marker, linestyle); B0 is neutral grey, the other four use slots 1-4
G2_MODELS: tuple[tuple[str, str, str, str, Literal["-", ":"]], ...] = (
    ("B0", "B0 elevation", INK2, "x", ":"),
    ("B1", "B1 C/N0 law", C1, "o", "-"),
    ("B2", "B2 SNR penalty", C3, "D", "-"),
    ("B4-censored-at-bound", "B4 cpStdev", C2, "s", "-"),
    ("B6", "B6 residual-adaptive", "#b07800", "^", "-"),
)


def fig_g2_coverage() -> Path:
    """Observed against nominal coverage per jammer-power class and model (E1a run b7a0a7f6d31a, arm a).

    One panel per nominal level. Hollow markers are case-study classes (power rule, docs/5 4.2). The 95 % CI of
    c(0.95) exists in the result file only for that level, so only the middle panel has bars.
    """
    cov = json.loads((results_root() / "E1a" / G2_RUN / "result.json").read_text())["coverage"]
    classes = [r["class"] for r in cov if r["model"] == "B1" and r["band"] == "all" and r["class"] != "all"]
    levels = (0.68, 0.95, 0.99)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.0), facecolor=SURFACE, sharex=True)
    for ax, p in zip(axes, levels, strict=True):
        for k, (key, _lab, col, mk, ls) in enumerate(G2_MODELS):
            rows = {r["class"]: r for r in cov if r["model"] == key and r["band"] == "all"}
            xs = [i + (k - 2) * 0.07 for i, c in enumerate(classes) if c in rows]
            rs = [rows[c] for c in classes if c in rows]
            ys = [r[f"c{p}"] for r in rs]
            ax.plot(xs, ys, color=col, linewidth=1.6, linestyle=ls)
            for x, y, r in zip(xs, ys, rs, strict=True):
                hollow = r["label"] != "audited"
                ax.plot(x, y, marker=mk, markersize=6.5, linestyle="none", markeredgecolor=col,
                        markerfacecolor=SURFACE if hollow else col, markeredgewidth=1.4)  # fmt: skip
            if p == 0.95 and key != "B0":
                lo = [y - r["ci95_c0.95"][0] for y, r in zip(ys, rs, strict=True)]
                hi = [r["ci95_c0.95"][1] - y for y, r in zip(ys, rs, strict=True)]
                ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=col, elinewidth=1, capsize=2, alpha=0.7)
        ax.axhline(p, color=INK, linewidth=1, linestyle="--")
        ax.set_ylim(0.45, 1.06)
        ax.set_ylabel(f"observed\n(dashed: {p:g})")
        _style(ax)
    b1 = next(r for r in cov if r["model"] == "B1" and r["class"] == "[13,23) dBm" and r["band"] == "all")
    axes[1].annotate("B1: 0.85", (classes.index(b1["class"]), b1["c0.95"]), xytext=(9, -9),
                     textcoords="offset points", fontsize=8, color=INK, ha="left", va="top")  # fmt: skip
    axes[1].annotate("B4: below nominal in every class", (1.0, 0.82), xytext=(0, -22), textcoords="offset points",
                     fontsize=8, color=INK, ha="center")  # fmt: skip
    axes[2].set_xticks(range(len(classes)))
    axes[2].set_xticklabels(
        [c.replace(" dBm", "").replace(",", ", ") if c != "clean" else "clean data" for c in classes],
        fontsize=8,
        rotation=25,
    )
    axes[2].set_xlabel("jammer power class (dBm, up-and-down ramp pooled); last column: pooled clean intervals")
    handles = [Line2D([], [], color=c, marker=m, linestyle=ls, markersize=6, label=lab)
               for _k, lab, c, m, ls in G2_MODELS]  # fmt: skip
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=8, labelcolor=INK)
    fig.text(0.5, 0.925, "Hollow marker: case-study class (no calibration verdict). Bars: 95 % CI of c(0.95).",
             ha="center", fontsize=8, color=INK2)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.915))
    G2_OUT.mkdir(parents=True, exist_ok=True)
    path = G2_OUT / "g2_coverage.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------------- G3: docs/10 section 2.3 (P2b)
# Sources, pinned. E2 run 2 (confirmatory, commit 7bc707e) and its summary's SHA-256 as s29 recorded it; s29 as
# committed in 4482537; E2K's planning dry run (L_B per segment, recomputed from run 2 and s29 by E2K's code); E0's
# G-hz loop files on FGI segment A (C/A) and SpoofRepo L5.
G3_E2_RUN = "6214cf52489f"
G3_E2_SUMMARY_SHA256 = "4419fd70188279102a2ab862fda3ef17acdbb81018b8793918d165b11e962c99"
G3_S29 = RES / "s29_true_analog_jmax.json"
G3_S29_SHA256 = "efd04c8d0cc39d2d373fb7fae6454793f80c2f8d88648226b99d7562a70e6fc2"
G3_E2K_PLAN = Path("dev") / "e2k-dryrun-task13.json"
G3_E2K_PLAN_SHA256 = "3b8f7611f7f65ba499b973ad8723b9d21f3539cbf3ddf8087facb64bdc8a0af2"
G3_E0_RUNS = (("fee767ba3538", "C/A, FGI segment A"), ("46b76d4b47d9", "L5, SpoofRepo"))
G3_LEVELS = ("50.0", "55.0", "60.0", "65.0", "70.0")
G3_SEGMENTS = ("A", "B")
G3_FAMILIES = {"noise": "band-limited noise", "swept_cw": "swept CW", "chirp": "chirp"}
G3_B1_BAR = 0.05  # r - 1/((C/N0) T), pre-registered (docs/3 section 2.2)
G3_TOL = 0.10  # |R - 1| of a front-end cell, pre-registered (docs/8 section 5)
G3_GHZ = (0.3, 5.0)  # G-hz bars, dB and degrees (docs/11)
# one colour per loop cell, in the palette's fixed order (slots 1-6), on both signals
C5, C6 = "#e87ba4", "#008300"
G3_CELLS = {"default": (C1, "o"), "default-o2": (C2, "s"), "bn15-t1": (C3, "D"), "bn5-t20": (C4, "^"),
            "bn15-t20": (C5, "v"), "bn5-t1": (C6, "P")}  # fmt: skip


# E2K (confirmatory, code 3eaed0b) ended on a gate failure (G-warmup-lock) and so wrote no manifest: its summary is
# pinned by SHA-256. s30 rebuilt the K rule's tables from E2K's hash-checked unit files and simulated the rule's null.
G3_E2K_RUN = "0443516ca48c"
G3_E2K_SUMMARY_SHA256 = "802582093b0c79c0e5df1c9fcda8f7faae70ce2ebdd81c7c27c78b2b34aa51ed"
G3_S30 = RES / "s30_k_rule_null.json"
G3_S30_SHA256 = "8ea0104304f46233c63ae3f1186566564665ba5c38ecb31d9feda268d282d1ba"
G3_K_BAR = 0.2  # docs/5 section 4.4: K = 51 if the median |sigma^2_101 / sigma^2_51 - 1| is at most this


class SourceRefused(ValueError):
    """A source is not the run or file the figures name."""


def checked_run(root: Path, experiment: str, run_id: str) -> Path:
    """root/experiment/run_id, refused unless it holds a manifest that names run_id (a finished run)."""
    run = root / experiment / run_id
    manifest = run / "manifest.json"
    if not manifest.is_file():
        raise SourceRefused(f"{run}: no manifest.json, so no finished run {run_id}")
    found = json.loads(manifest.read_text()).get("run_id")
    if found != run_id:
        raise SourceRefused(f"{run}: the manifest names run {found!r}, the figures name {run_id!r}")
    return run


def checked_file(path: Path, sha256: str) -> Path:
    """path, refused unless its SHA-256 is sha256."""
    digest = sha256_file(path)
    if digest != sha256:
        raise SourceRefused(f"{path}: SHA-256 {digest}, the figures name {sha256}")
    return path


def g3_sources(root: Path) -> dict[str, Any]:
    """Every source of the G3 figures, each refused unless it is the pinned one: run 2's summary, s29 (which must
    name run 2 and its summary's SHA-256), E2K's dry run (whose residual_pass must be run 2's per segment), the E0
    loop files, E2K's summary (no manifest: the run failed a gate) and s30 (which must name E2K and that summary)."""
    run2 = checked_run(root, "E2", G3_E2_RUN)
    summary = json.loads(checked_file(run2 / "summary.json", G3_E2_SUMMARY_SHA256).read_text())
    s29 = json.loads(checked_file(G3_S29, G3_S29_SHA256).read_text())
    named = (s29["e2_run2"]["run_id"], s29["e2_run2"]["summary_sha256"])
    if named != (G3_E2_RUN, G3_E2_SUMMARY_SHA256):
        raise SourceRefused(f"s29 reads run {named[0]} with summary SHA-256 {named[1]}, not run {G3_E2_RUN}")
    plan = json.loads(checked_file(root / G3_E2K_PLAN, G3_E2K_PLAN_SHA256).read_text())
    for seg in G3_SEGMENTS:
        if plan["front"][seg]["residual_pass"] != summary["segments"][seg]["residual_pass"]:
            raise SourceRefused(f"E2K's dry run: segment {seg}'s residual_pass is not run {G3_E2_RUN}'s")
    loops = {rid: [json.loads(p.read_text()) for p in sorted((checked_run(root, "E0b", rid) / "loop").glob("*.json"))]
             for rid, _ in G3_E0_RUNS}  # fmt: skip
    return {"summary": summary, "s29": s29, "plan": plan, "loops": loops, **g3_k_sources(root)}


def g3_k_sources(root: Path) -> dict[str, Any]:
    """E2K's summary, pinned by SHA-256 (the run failed a gate, so it has no manifest), and s30, which must name E2K
    and that summary's SHA-256."""
    e2k = json.loads(checked_file(root / "E2K" / G3_E2K_RUN / "summary.json", G3_E2K_SUMMARY_SHA256).read_text())
    s30 = json.loads(checked_file(G3_S30, G3_S30_SHA256).read_text())
    named = (s30["e2k"]["run_id"], s30["e2k"]["summary_sha256"])
    if named != (G3_E2K_RUN, G3_E2K_SUMMARY_SHA256):
        raise SourceRefused(f"s30 reads run {named[0]} with summary SHA-256 {named[1]}, not E2K run {G3_E2K_RUN}")
    return {"e2k": e2k, "s30": s30}


def paired_sums(cell: Mapping[str, Any]) -> dict[str, float]:
    """A front-end cell's Var(dPhi) and paired excess per mode, summed over its ghosts (R = excess_B / excess_A)."""
    keys = ("var_a", "excess_a", "var_b", "excess_b")
    return {k: math.fsum(g[k] for g in cell["excess"].values()) for k in keys}


def _g3_save(fig: Any, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def fig_g3_paired(summary: Mapping[str, Any]) -> Path:
    """Var(dPhi) against the paired excess, modes A and B, per segment, family and level (run 2's front_end)."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9.4), facecolor=SURFACE, sharex=True, sharey=True)
    lo, hi = 1.5e-6, 9e-4
    for row, fam in zip(axes, ("noise", "swept_cw"), strict=True):
        for ax, seg in zip(row, G3_SEGMENTS, strict=True):
            ax.plot([lo, hi], [lo, hi], color=INK2, linewidth=1, linestyle=":")
            for lv in G3_LEVELS:
                s = paired_sums(summary["segments"][seg]["front_end"][fam][lv])
                ax.plot([s["excess_a"], s["excess_b"]], [s["var_a"], s["var_b"]], color=INK2, linewidth=1)
                for x, y, c, mk in ((s["excess_b"], s["var_b"], C2, "s"), (s["excess_a"], s["var_a"], C1, "o")):
                    ax.plot(x, y, linestyle="none", marker=mk, markersize=8, markerfacecolor=c,
                            markeredgecolor=SURFACE, markeredgewidth=1.5)  # fmt: skip
                ax.annotate(f"{float(lv):g}", (s["excess_b"], s["var_b"]), textcoords="offset points",
                            xytext=(-6, 6), fontsize=8, color=INK2, ha="right")  # fmt: skip
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            ax.set_title(f"segment {seg}, {G3_FAMILIES[fam]}", fontsize=10)
            _style(ax)
    axes[0][0].text(7e-4, 2.5e-6, "dotted diagonal: Var(ΔΦ) = excess,\nnothing cancels against the clean twin",
                    fontsize=8, color=INK2, ha="right", va="bottom")  # fmt: skip
    for ax in axes[1]:
        ax.set_xlabel("paired excess Σ_g [Var(ΔΦ) + 2 Cov(e_c, ΔΦ)] (cycles²)")
    for ax in axes[:, 0]:
        ax.set_ylabel("twin-difference variance Σ_g Var(ΔΦ) (cycles²)")
    handles = [Line2D([], [], color=c, marker=mk, linestyle="none", markersize=8, label=lab)
               for c, mk, lab in ((C1, "o", "mode A"), (C2, "s", "mode B"))]  # fmt: skip
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=9, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.955))  # fmt: skip
    fig.suptitle(f"Paired decomposition, E2 run {G3_E2_RUN}: one marker pair per J/N₀ level (dB-Hz, labelled)",
                 color=INK, fontsize=11)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _g3_save(fig, "f10-1_paired_decomposition.png")


def fig_g3_yardstick(summary: Mapping[str, Any], s29: Mapping[str, Any]) -> Path:
    """R(B/A) from run 2 and R_TA from s29, each +- SE, against J/N0, with the +-0.10 band."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4), facecolor=SURFACE, sharex=True, sharey=True)
    xs = [float(lv) for lv in G3_LEVELS]
    for row, fam in zip(axes, ("noise", "swept_cw"), strict=True):
        for ax, seg in zip(row, G3_SEGMENTS, strict=True):
            ax.axhspan(1 - G3_TOL, 1 + G3_TOL, color=GRID, alpha=0.6, linewidth=0)
            ax.axhline(1.0, color=INK2, linewidth=1)
            run = [summary["segments"][seg]["front_end"][fam][lv] for lv in G3_LEVELS]
            sim = [s29["cells"][seg][fam][lv] for lv in G3_LEVELS]
            series = (
                ([x - 0.4 for x in xs], [c["ratio"] for c in run], [c["ratio_se"] for c in run],
                 [c["verdict"] for c in run], C2, "s"),
                ([x + 0.4 for x in xs], [c["ratio_truth"] for c in sim], [c["ratio_truth_se"] for c in sim],
                 [c["verdict"] for c in sim], C1, "o"),
            )  # fmt: skip
            for x, y, se, verdicts, c, mk in series:
                ax.errorbar(x, y, yerr=se, fmt="none", ecolor=c, elinewidth=1.5, capsize=3)
                ax.plot(x, y, color=c, linewidth=1.2, alpha=0.5)
                for xi, yi, si, v in zip(x, y, se, verdicts, strict=True):
                    face = SURFACE if v == "unresolved" else c
                    ax.plot(xi, yi, linestyle="none", marker=mk, markersize=8, markerfacecolor=face,
                            markeredgecolor=c if v == "unresolved" else SURFACE, markeredgewidth=1.5)  # fmt: skip
                    if v == "fail":
                        ax.annotate("fail", (xi, yi - si), xytext=(0, -11), textcoords="offset points", fontsize=8,
                                    color=INK, ha="center")  # fmt: skip
            ax.set_ylim(0.68, 1.14)
            ax.set_xticks(xs)
            ax.set_title(f"segment {seg}, {G3_FAMILIES[fam]}", fontsize=10)
            _style(ax)
    axes[0][0].text(69.6, 1 + G3_TOL - 0.012, "±0.10", fontsize=8, color=INK2, ha="right", va="top")
    for ax in axes[1]:
        ax.set_xlabel("J/N₀ (dB-Hz)")
    for ax in axes[:, 0]:
        ax.set_ylabel("ratio of paired excesses")
    handles = [
        Line2D([], [], color=C2, marker="s", markersize=8, label=f"R(B/A): mode B / mode A, E2 run {G3_E2_RUN} "
               "(the pre-registered yardstick)"),
        Line2D([], [], color=C1, marker="o", markersize=8, label="R_TA: true front end / mode A, review s29 (addendum)"),
        Line2D([], [], color=INK2, marker="s", markerfacecolor=SURFACE, linestyle="none", markersize=8,
               label="hollow: unresolved (SE > 0.05)"),
    ]  # fmt: skip
    fig.legend(handles=handles, loc="upper center", ncol=1, frameon=False, fontsize=8.5, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.95))  # fmt: skip
    fig.suptitle("Front-end yardstick: ratio of paired excesses ± 1 SE (band: the ±0.10 tolerance)", color=INK,
                 fontsize=11)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    return _g3_save(fig, "f10-2_frontend_yardstick.png")


def b1_excess(summary: Mapping[str, Any], seg: str, fam: str) -> dict[str, list[float]]:
    """r - 1/((C/N0) T) per ghost PRN and level, from run 2's per-segment residual records."""
    res = summary["segments"][seg]["residual"][fam]
    prns = [str(p) for p in summary["segments"][seg]["ghost_prn"]]
    return {prn: [res[lv][prn]["r"] - res[lv][prn]["floor"] for lv in G3_LEVELS] for prn in prns}


def fig_g3_b1(summary: Mapping[str, Any], plan: Mapping[str, Any], s29: Mapping[str, Any]) -> Path:
    """r - 1/((C/N0) T) per ghost against J/N0, per family and segment (run 2's residual), with the 0.05 bar and L_B
    per segment and combined (E2K's dry run)."""
    fams = ("noise", "swept_cw", "chirp")
    fig, axes = plt.subplots(3, 2, figsize=(10, 9.6), facecolor=SURFACE, sharex=True)
    xs = [float(lv) for lv in G3_LEVELS]
    for row, fam in zip(axes, fams, strict=True):
        both = plan["ceiling"][fam]
        top = 1.15 * max(v for seg in G3_SEGMENTS for ys in b1_excess(summary, seg, fam).values() for v in ys)
        for ax, seg in zip(row, G3_SEGMENTS, strict=True):
            for (prn, y), c, mk in zip(b1_excess(summary, seg, fam).items(), (C1, C2), ("o", "s"), strict=True):
                ax.plot(xs, y, color=c, linewidth=2, marker=mk, markersize=8, markeredgecolor=SURFACE,
                        markeredgewidth=1.5, label=f"ghost PRN {prn}")  # fmt: skip
            ax.axhline(G3_B1_BAR, color=INK, linewidth=1)
            ceiling = plan["front"][seg]["ceiling_db_hz"][fam]
            if ceiling is None:
                ok = summary["segments"][seg]["residual_pass"][fam]
                first = next(float(lv) for lv in G3_LEVELS if not ok[lv])
                note = f"L_B({seg}) none: B1 fails from {first:g} dB-Hz"
            else:
                ax.axvline(ceiling, color=INK2, linewidth=1, linestyle=":")
                note = f"L_B({seg}) = {ceiling:g} dB-Hz"
            ax.text(0.02, 0.96, note, transform=ax.transAxes, fontsize=8.5, color=INK, va="top")
            ax.set_ylim(0, top)
            ax.set_xticks(xs)
            combined = "none" if both is None else f"{both:g} dB-Hz"
            ax.set_title(f"segment {seg}, {G3_FAMILIES[fam]}\naddendum's combined L_B: {combined}", fontsize=10)
            anchor = (0.0, 0.15) if fam == "chirp" else (0.0, 0.42)  # in the empty band of each row
            ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left", bbox_to_anchor=anchor)
            _style(ax)
        row[0].set_ylabel("r − 1/((C/N₀)·T)")
    axes[0][0].annotate("B1 bar 0.05", (55.3, G3_B1_BAR), xytext=(0, 4), textcoords="offset points", fontsize=8,
                        color=INK2)  # fmt: skip
    for ax in axes[2]:
        ax.set_xlabel("J/N₀ (dB-Hz)")
    grid_top = float(G3_LEVELS[-1])  # a J_max_TA at the top of the grid only says "at least this"

    def ta(fam: str) -> str:
        v = s29["decisions"]["families"][fam]["combined_db_hz"]
        return "none" if v is None else (f"≥ {v:g}" if v >= grid_top else f"{v:g}")

    jmax = ", ".join(f"{G3_FAMILIES[f]} {ta(f)}" for f in fams)
    d = summary["decisions"]
    if all(d["j_max_db_hz"][f] is None for f in fams):
        pre_text = "J_max, and so L_B, None for every family"
    else:
        pre_text = "L_B " + ", ".join(f"{G3_FAMILIES[f]} {d['ceiling_db_hz'][f]}" for f in fams)
    fig.suptitle(f"B1 residual above the thermal floor, mode A, E2 run {G3_E2_RUN}\nL_B under the addendum "
                 f"(J_max_TA, s29: {jmax} dB-Hz; {grid_top:g} is the top of the grid)\nunder the pre-registered "
                 f"rules: {pre_text}", color=INK, fontsize=11)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 1))
    return _g3_save(fig, "f10-3_b1_residual.png")


def fig_g3_identified_h(loops: Mapping[str, list[dict[str, Any]]]) -> Path:
    """Identified H_phi on real twins against the model per tone (E0's G-hz rows), per loop cell and signal."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), facecolor=SURFACE, sharex=True, sharey="row")
    for col, (rid, label) in enumerate(G3_E0_RUNS):
        for loop in loops[rid]:
            judged = [r for r in loop["rows"] if r["status"] in ("pass", "fail")]
            f = [r["f_hz"] for r in judged]
            c, mk = G3_CELLS[loop["cell"]]
            for ax, key in zip(axes[:, col], ("err_db", "err_deg"), strict=True):
                ax.plot(f, [r[key] for r in judged], color=c, linewidth=1.5, marker=mk, markersize=7,
                        markeredgecolor=SURFACE, markeredgewidth=1.2)  # fmt: skip
        axes[0][col].set_title(f"{label} (E0 run {rid})", fontsize=10)
        for ax in axes[:, col]:
            ax.axhline(0.0, color=INK2, linewidth=1)
            ax.set_xscale("log")
            _style(ax)
        axes[1][col].set_xlabel("tone frequency (Hz)")
    axes[0][0].set_ylim(-0.02, 0.02)
    axes[1][0].set_ylim(-0.12, 0.12)
    axes[0][0].set_ylabel("gain error (dB)")
    axes[1][0].set_ylabel("phase error (degrees)")
    cells = [cell for cell in G3_CELLS if any(loop["cell"] == cell for ls in loops.values() for loop in ls)]
    handles = [Line2D([], [], color=G3_CELLS[cell][0], marker=G3_CELLS[cell][1], markersize=7, label=cell)
               for cell in cells]  # fmt: skip
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=8.5, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.925), title="loop cell", title_fontsize=8.5)  # fmt: skip
    fig.suptitle(f"Identified loop response H_φ on real twins against the model: identified / model per judged "
                 f"tone\n(the G-hz bars, ±{G3_GHZ[0]:g} dB and ±{G3_GHZ[1]:g}°, lie far outside these axes)",
                 color=INK, fontsize=11)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    return _g3_save(fig, "f10-4_identified_h.png")


def spread_labels(values: Mapping[str, float], gap_decades: float) -> dict[str, float]:
    """Label heights for a log axis: each at its value unless that is within gap_decades of the label below it."""
    out: dict[str, float] = {}
    last = -math.inf
    for key in sorted(values, key=lambda k: values[k]):
        pos = max(math.log10(values[key]), last + gap_decades)
        out[key], last = 10**pos, pos
    return out


G3_K_ENSEMBLES = (("K = 51 (k-cond first 51)", "K = 51\nfirst 51 of the 101"), ("K = 101 (k-cond)", "K = 101"),
                  ("independent 51 (k-noise)", "K = 51\nindependent (k-noise)"))  # fmt: skip


def fig_g3_k(e2k: Mapping[str, Any], s30: Mapping[str, Any]) -> Path:
    """K's rule on E2K: sigma^2_oracle against K per satellite, and the per-epoch statistic against its Gaussian null
    (s30, which rebuilt E2K's tables; its K rule statistic must be E2K's own)."""
    obs, k = s30["e2k"], e2k["decisions"]["k_convergence"]
    if obs["designs"]["nested"]["median_rel_change"] != k["median_rel_change"]:
        raise SourceRefused("s30's K rule statistic is not E2K's decisions.k_convergence")
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.9), facecolor=SURFACE,
                                      gridspec_kw={"width_ratios": (1.0, 1.55)})  # fmt: skip
    # left: sigma^2_oracle against K, per satellite (median over the epochs from onset + 2 s)
    xs = range(len(G3_K_ENSEMBLES))
    ends = {}
    for prn, by in obs["per_prn"].items():
        ys = [by[key]["median_var_cyc2_post_onset"] for key, _ in G3_K_ENSEMBLES]
        # K = 51 and 101 are nested (one ensemble); the independent 51 is another ensemble: no line to it
        left.plot(xs[:2], ys[:2], color=C1, linewidth=2, marker="o", markersize=8, markeredgecolor=SURFACE,
                  markeredgewidth=1.5)  # fmt: skip
        left.plot(xs[2:], ys[2:], color=C1, linestyle="none", marker="o", markersize=8, markeredgecolor=SURFACE,
                  markeredgewidth=1.5)  # fmt: skip
        ends[prn] = ys[-1]
    for prn, y in spread_labels(ends, 0.035).items():
        left.annotate(f"PRN {prn}", (len(G3_K_ENSEMBLES) - 1, y), xytext=(10, 0), textcoords="offset points",
                      fontsize=8, color=INK, va="center")  # fmt: skip
    left.set_yscale("log")
    left.set_xticks(list(xs))
    left.set_xticklabels([lab for _, lab in G3_K_ENSEMBLES], fontsize=8)
    left.set_xlim(-0.3, len(G3_K_ENSEMBLES) - 0.4)
    left.set_ylabel("median σ²_oracle from onset + 2 s (cycles²)")
    left.set_title("σ²_oracle against K, per satellite", fontsize=10)
    _style(left)
    # right: the per-epoch statistic, stacked by the rows before and from the onset, with one Gaussian replication
    h = obs["histogram"]
    edges = h["edges"]
    lo, width = edges[:-1], [b - a for a, b in zip(edges[:-1], edges[1:], strict=True)]
    n = h["observed_nested"]["n"]
    before = [100 * c / n for c in h["observed_nested_before_onset"]["counts"]]
    after = [100 * c / n for c in h["observed_nested_from_onset"]["counts"]]
    right.bar(lo, after, width=width, align="edge", color=C1, edgecolor=SURFACE, linewidth=0.6,
              label="rows from the onset")  # fmt: skip
    right.bar(lo, before, width=width, bottom=after, align="edge", color=C2, edgecolor=SURFACE, linewidth=0.6,
              label="rows before the onset (receiver jitter)")  # fmt: skip
    g = h["gaussian_same_pattern"]
    right.stairs([100 * c / g["n"] for c in g["counts"]], edges, color=INK, linewidth=1.5,
                 label="one Gaussian replication, E2K's pattern (s30)")  # fmt: skip
    null = obs["realized_pattern"]["null"]
    right.axvspan(null["p05"], null["p95"], color=INK2, alpha=0.35, linewidth=0)
    right.axvline(G3_K_BAR, color=INK, linewidth=1.5)
    right.axvline(k["median_rel_change"], color=INK2, linewidth=1)
    top = right.get_ylim()[1]
    right.annotate(f"bar {G3_K_BAR:g}: K = 51\nif the median ≤ {G3_K_BAR:g}", (G3_K_BAR, 0.4 * top), xytext=(5, 0),
                   textcoords="offset points", fontsize=8, color=INK, va="top")  # fmt: skip
    above = (h["observed_nested"]["above"], h["observed_nested_before_onset"]["above"], g["above"])
    right.text(0.985, 0.66, f"E2K's median {k['median_rel_change']:.4f} ({k['epochs']:.0f} rows; thin line)\n"
               f"Gaussian null of the median, E2K's pattern: {null['median']:.4f}\n"
               f"its 5–95 % (shaded): {null['p05']:.4f}–{null['p95']:.4f}\n"
               f"{obs['realized_pattern']['reps_at_or_above_observed']} of {null['reps']} replications reach "
               f"{k['median_rel_change']:.4f}\n"
               f"beyond {edges[-1]:g}: {above[0]} rows ({above[1]} before the onset); Gaussian {above[2]}",
               transform=right.transAxes, fontsize=8, color=INK, ha="right", va="top")  # fmt: skip
    right.set_xlim(0, edges[-1])
    right.set_xlabel("per-epoch |σ²₁₀₁ / σ²₅₁ − 1| (K = 101 block against its first 51)")
    right.set_ylabel("share of epoch-satellite rows (%)")
    right.set_title("K's rule: the statistic per epoch, and its median", fontsize=10)
    right.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    _style(right)
    d = e2k["decisions"]
    fig.suptitle(f"K on E2K run {G3_E2K_RUN} (segment A, noise at L_K = {d['k_level_db_hz']['noise']:g} dB-Hz, "
                 f"under the addendum): K = {d['k']}\nK's statistic lies near its Gaussian null value, at about half "
                 "the bar", color=INK, fontsize=11, y=0.98)  # fmt: skip
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.15, top=0.85, wspace=0.22)
    return _g3_save(fig, "f10-5_k_rule.png")


# ---------------------------------------------------------------------------------- P2c: docs/10 section 2.3.3 (run 3)
# Sources, pinned. E2 run 3 (confirmatory, code db790aa, ghost truth) and its summary's SHA-256 as s34 pinned it; s33
# (committed in a907825, before run 3 started) and s34 (committed in 2c809bd). Run 2 is G3_E2_RUN, pinned above.
P2C_E2_RUN = "9650dc00e766"
P2C_E2_SUMMARY_SHA256 = "f38389b07f8c025114d980d4c0cef1952f6f102b2f5d493e1eb19f717be2a88b"
P2C_S33 = RES / "s33_front_end_prediction.json"
P2C_S33_SHA256 = "cee73739dcc8b2c207ab5b2b9683ddecb92412e583d6c75e80a21b141e8579ac"
P2C_S34 = RES / "s34_front_end_acceptance.json"
P2C_S34_SHA256 = "b0a194d5163c10cb66e4bd46ef1ab9b2d6a0431750c0eb140bab14643374f1e9"
P2C_TAUS = (("0.001", "τ = 1 ms", C1, "o"), ("0.01", "τ = 10 ms", C2, "s"), ("0.1", "τ = 100 ms", C3, "D"))
P2C_S33_LEVELS = tuple(f"{lv:.1f}" for lv in range(50, 90, 5))
P2C_SE_MAX = 0.05  # a front-end cell with a larger SE(R) is unresolved (docs/8 section 5, rule (a'))
P2C_SLOPE_TOL = 0.10


def p2c_sources(root: Path) -> dict[str, Any]:
    """Every source of f10-6 to f10-8, each refused unless it is the pinned one: run 3's summary (a finished run), run
    2's summary, s33 (which must name run 2 and its summary's SHA-256) and s34 (which must name run 3's summary, s33 and
    run 2's summary by these SHA-256)."""
    run3 = json.loads(
        checked_file(checked_run(root, "E2", P2C_E2_RUN) / "summary.json", P2C_E2_SUMMARY_SHA256).read_text()
    )
    run2 = json.loads(
        checked_file(checked_run(root, "E2", G3_E2_RUN) / "summary.json", G3_E2_SUMMARY_SHA256).read_text()
    )
    s33 = json.loads(checked_file(P2C_S33, P2C_S33_SHA256).read_text())
    named = (s33["e2_run2"]["run_id"], s33["e2_run2"]["summary_sha256"])
    if named != (G3_E2_RUN, G3_E2_SUMMARY_SHA256):
        raise SourceRefused(f"s33 reads run {named[0]} with summary SHA-256 {named[1]}, not run {G3_E2_RUN}")
    s34 = json.loads(checked_file(P2C_S34, P2C_S34_SHA256).read_text())
    ins = s34["inputs"]
    judged = (ins["run3_summary"]["run_id"], ins["run3_summary"]["sha256"], ins["s33"]["sha256"],
              ins["run2_summary"]["sha256"])  # fmt: skip
    pinned = (P2C_E2_RUN, P2C_E2_SUMMARY_SHA256, P2C_S33_SHA256, G3_E2_SUMMARY_SHA256)
    if judged != pinned:
        raise SourceRefused(f"s34 judged run {judged[0]} (summary {judged[1]}), s33 {judged[2]} and run 2's summary "
                            f"{judged[3]}, not the pinned {pinned}")  # fmt: skip
    return {"run3": run3, "run2": run2, "s33": s33, "s34": s34}


def check_s34_z(run3: Mapping[str, Any], s33: Mapping[str, Any], s34: Mapping[str, Any]) -> None:
    """Refuse s34 unless each of its z is (R_run3 - R_B,s33) / sqrt(SE_run3^2 + SE_s33^2) from these two sources, and
    its chi2 (the sum of z^2), n (the cells present) and max_abs_z per segment are what those z give, so that f10-6
    draws and prints the values s34 judged."""
    for seg, rec in s34["segments"].items():
        mine: list[float] = []
        for key, z in rec["z"].items():
            fam, lv = key.split("/")
            r = run3["segments"][seg]["front_end"][fam][lv]
            p = s33["cells"][seg][fam][lv]
            mine.append((r["ratio"] - p["ratio_B"]) / math.sqrt(r["ratio_se"] ** 2 + p["ratio_B_se"] ** 2))
            if not math.isclose(mine[-1], z, rel_tol=1e-12, abs_tol=1e-12):
                raise SourceRefused(f"s34's z at segment {seg}, {key} is {z}; run 3 and s33 give {mine[-1]}")
        if rec["n"] != len(mine):
            raise SourceRefused(f"s34's n for segment {seg} is {rec['n']}; it holds {len(mine)} z")
        derived = {"chi2": math.fsum(z * z for z in mine), "max_abs_z": max(abs(z) for z in mine)}
        for field, value in derived.items():
            if not math.isclose(rec[field], value, rel_tol=1e-12, abs_tol=1e-12):
                raise SourceRefused(f"s34's {field} for segment {seg} is {rec[field]}; its z give {value}")


def _point(ax: Axes, x: float, y: float, se: float, c: str, mk: str, hollow: bool) -> None:
    ax.errorbar([x], [y], yerr=[se], fmt="none", ecolor=c, elinewidth=1.5, capsize=3)
    ax.plot(x, y, linestyle="none", marker=mk, markersize=8, markerfacecolor=SURFACE if hollow else c,
            markeredgecolor=c if hollow else SURFACE, markeredgewidth=1.5)  # fmt: skip


def fig_p2c_yardstick(run3: Mapping[str, Any], run2: Mapping[str, Any], s33: Mapping[str, Any],
                      s34: Mapping[str, Any]) -> Path:  # fmt: skip
    """Run 3's R(B/A) against run 2's and s33's predicted R_B, each +- SE, per segment, noise and swept CW, 50-70
    dB-Hz, with the +-0.10 band; each column carries s34's verdict on that segment."""
    check_s34_z(run3, s33, s34)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.8), facecolor=SURFACE, sharex=True, sharey=True)
    xs = [float(lv) for lv in G3_LEVELS]
    for row, fam in zip(axes, ("noise", "swept_cw"), strict=True):
        for ax, seg in zip(row, G3_SEGMENTS, strict=True):
            ax.axhspan(1 - G3_TOL, 1 + G3_TOL, color=GRID, alpha=0.6, linewidth=0)
            ax.axhline(1.0, color=INK2, linewidth=1)
            r3 = [run3["segments"][seg]["front_end"][fam][lv] for lv in G3_LEVELS]
            r2 = [run2["segments"][seg]["front_end"][fam][lv] for lv in G3_LEVELS]
            pb = [s33["cells"][seg][fam][lv] for lv in G3_LEVELS]
            series = (
                (-0.6, [c["ratio"] for c in r2], [c["ratio_se"] for c in r2], [c["verdict"] for c in r2], C2, "s"),
                (0.0, [c["ratio_B"] for c in pb], [c["ratio_B_se"] for c in pb], [None] * len(pb), C3, "D"),
                (0.6, [c["ratio"] for c in r3], [c["ratio_se"] for c in r3], [c["verdict"] for c in r3], C1, "o"),
            )
            for dx, y, se, verdicts, c, mk in series:
                x = [v + dx for v in xs]
                ax.plot(x, y, color=c, linewidth=1.2, alpha=0.5)
                for xi, yi, si, v in zip(x, y, se, verdicts, strict=True):
                    _point(ax, xi, yi, si, c, mk, v == "unresolved")
                    if v == "fail" and c == C1:
                        ax.annotate("fail", (xi, yi), xytext=(9, 0), textcoords="offset points", fontsize=8,
                                    color=INK, ha="left", va="center")  # fmt: skip
            ax.set_ylim(0.48, 1.14)
            ax.set_xticks(xs)
            ax.set_title(f"segment {seg}, {G3_FAMILIES[fam]}", fontsize=10)
            _style(ax)
    for ax, seg in zip(axes[0], G3_SEGMENTS, strict=True):
        rec = s34["segments"][seg]
        worst = max(rec["z"], key=lambda k: abs(rec["z"][k]))
        fam, lv = worst.split("/")
        verdict = "accepted" if rec["accepted"] else "rejected"
        ax.text(0.98, 0.04, f"s34, segment {seg}: Σz² = {rec['chi2']:.2f} against {rec['chi2_bound']:.2f} "
                f"(n = {rec['n']}),\nmax |z| = {rec['max_abs_z']:.2f} ({G3_FAMILIES[fam]}, {float(lv):g} dB-Hz): "
                f"{verdict}",
                transform=ax.transAxes, fontsize=8, color=INK, ha="right", va="bottom")  # fmt: skip
    axes[0][0].text(69.6, 1 + G3_TOL - 0.012, "±0.10", fontsize=8, color=INK2, ha="right", va="top")
    for ax in axes[1]:
        ax.set_xlabel("J/N₀ (dB-Hz)")
    for ax in axes[:, 0]:
        ax.set_ylabel("ratio of paired excesses, B / A")
    handles = [
        Line2D([], [], color=C1, marker="o", markersize=8, label=f"R(B/A), E2 run 3 {P2C_E2_RUN} (confirmatory, "
               "out of sample for s33)"),
        Line2D([], [], color=C2, marker="s", markersize=8, label=f"R(B/A), E2 run 2 {G3_E2_RUN} (in sample for s33)"),
        Line2D([], [], color=C3, marker="D", markersize=8, label="R_B predicted by review s33 (committed before run 3)"),
        Line2D([], [], color=INK2, marker="o", markerfacecolor=SURFACE, linestyle="none", markersize=8,
               label=f"hollow: unresolved (SE > {P2C_SE_MAX:g}); \"fail\" marks a failing run 3 cell"),
    ]  # fmt: skip
    fig.legend(handles=handles, loc="upper center", ncol=1, frameon=False, fontsize=8.5, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.95))  # fmt: skip
    fig.suptitle("The registered yardstick in run 3 against s33's prediction, ± 1 SE (band: the ±0.10 tolerance)",
                 color=INK, fontsize=11)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return _g3_save(fig, "f10-6_run3_yardstick.png")


def ghost_excess(summary: Mapping[str, Any], seg: str, fam: str) -> dict[str, list[float]]:
    """The paired excess Var(e_u,g) - Var(e_c,g) of each ghost in mode A per level (cycles^2), keyed by ghost channel:
    segments.<s>.front_end.<family>.<level>.excess.<channel>.excess_a."""
    cells = summary["segments"][seg]["front_end"][fam]
    chans = list(cells[G3_LEVELS[0]]["excess"])
    return {ch: [cells[lv]["excess"][ch]["excess_a"] for lv in G3_LEVELS] for ch in chans}


def sci(x: float) -> str:
    """x as m × 10^e with two significant digits and a Unicode minus and superscripts, for figure labels."""
    m, e = f"{x:.1e}".split("e")
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{m.replace('-', '−')} × 10{str(int(e)).translate(sup)}"


def fig_p2c_ghost_excess(run3: Mapping[str, Any]) -> Path:
    """The ghosts' paired excess in mode A per level and family, both segments (run 3), on a log axis; a value at or
    below 0 is drawn as a hollow triangle on the bottom edge and labelled."""
    fams = (("noise", C1), ("swept_cw", C2), ("chirp", C3))
    lo, hi = 3e-9, 1e-3
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.9), facecolor=SURFACE, sharey=True)
    xs = [float(lv) for lv in G3_LEVELS]
    for ax, seg in zip(axes, G3_SEGMENTS, strict=True):
        s = run3["segments"][seg]
        prn = {ch: int(s["channels"][ch][2]) for ch in s["front_end"]["noise"][G3_LEVELS[0]]["excess"]}
        for fam, c in fams:
            per = ghost_excess(run3, seg, fam)
            mean = [math.fsum(v) / len(v) for v in zip(*per.values(), strict=True)]
            ax.plot(xs, mean, color=c, linewidth=2)
            for (ch, ys), mk in zip(per.items(), ("o", "s"), strict=True):
                for x, y in zip(xs, ys, strict=True):
                    if y > 0:
                        ax.plot(x, y, linestyle="none", marker=mk, markersize=8, markerfacecolor=c,
                                markeredgecolor=SURFACE, markeredgewidth=1.5)  # fmt: skip
                    else:
                        ax.plot(x, lo * 1.25, linestyle="none", marker="v", markersize=8, markerfacecolor=SURFACE,
                                markeredgecolor=c, markeredgewidth=1.5, clip_on=False)  # fmt: skip
                        ax.annotate(f"PRN {prn[ch]}: {sci(y)} cycles² (≤ 0)", (x, lo * 1.25), xytext=(8, 0),
                                    textcoords="offset points", fontsize=8, color=INK, va="center")  # fmt: skip
            ax.annotate(G3_FAMILIES[fam], (xs[-1], mean[-1]), xytext=(12, 0), textcoords="offset points", fontsize=8,
                        color=INK, va="center")  # fmt: skip
        lost = [
            v for name, d in s["lost_ghost_s"].items() if "-grid-" in name and name.endswith("-A") for v in d.values()
        ]
        ax.text(0.98, 0.03, f"ghosts PRN {prn[min(prn)]} (circles) and PRN {prn[max(prn)]} (squares);\nline: their mean."
                f" Truth kept at every level:\nat most {max(lost):.3f} s lost per ghost\n(mode A grid units; bar 1.0 s)",
                transform=ax.transAxes, fontsize=8, color=INK, ha="right", va="bottom")  # fmt: skip
        ax.set_yscale("log")
        ax.set_ylim(lo, hi)
        ax.set_xlim(48.5, 76.5)
        ax.set_xticks(xs)
        ax.set_xlabel("J/N₀ (dB-Hz)")
        ax.set_title(f"segment {seg}", fontsize=10)
        _style(ax)
    axes[0].set_ylabel("paired excess Var(e_jammed) − Var(e_clean) (cycles²)")
    handles = [Line2D([], [], color=c, linewidth=2, marker="o", markersize=7, label=G3_FAMILIES[f]) for f, c in fams]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=8.5, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.92))  # fmt: skip
    fig.suptitle(f"Ghost truth in E2 run {P2C_E2_RUN}: the variance interference adds to each ghost's absolute error, "
                 "mode A", color=INK, fontsize=11)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return _g3_save(fig, "f10-7_ghost_truth_excess.png")


def ta_verdict(cell: Mapping[str, Any], tau: str) -> str:
    """Rule (a') on one s33 cell at one tau, by e2_frontend.cell_verdict as s34 judges s33's levels: R_TA against 1,
    its SE and its slope (tolerances 0.10 / 0.10 / 0.05)."""
    c = FrontEndCell(slopes=(cell["slope_TA"][tau],), ratio=cell["ratio_TA"][tau], ratio_se=cell["ratio_TA_se"][tau])
    return cell_verdict(c, tol=G3_TOL, slope_tol=P2C_SLOPE_TOL, se_max=P2C_SE_MAX, cell=f"s33 cell at tau {tau} s")


def fig_p2c_true_analog(s33: Mapping[str, Any]) -> Path:
    """s33's R_TA (true front end / mode A) per tau, 50-85 dB-Hz, per segment and family, +- SE, with the +-0.10
    band and run 3's top grid level."""
    fams = ("noise", "swept_cw", "chirp")
    fig, axes = plt.subplots(3, 2, figsize=(10, 10.2), facecolor=SURFACE, sharex=True)
    xs = [float(lv) for lv in P2C_S33_LEVELS]
    for row, fam in zip(axes, fams, strict=True):
        for ax, seg in zip(row, G3_SEGMENTS, strict=True):
            ax.axhspan(1 - G3_TOL, 1 + G3_TOL, color=GRID, alpha=0.6, linewidth=0)
            ax.axhline(1.0, color=INK2, linewidth=1)
            ax.axvline(float(G3_LEVELS[-1]), color=INK2, linewidth=1, linestyle=":")
            cells = [s33["cells"][seg][fam][lv] for lv in P2C_S33_LEVELS]
            for dx, (tau, _, c, mk) in zip((-0.8, 0.0, 0.8), P2C_TAUS, strict=True):
                x = [v + dx for v in xs]
                y = [cell["ratio_TA"][tau] for cell in cells]
                ax.plot(x, y, color=c, linewidth=1.2, alpha=0.5)
                for xi, yi, cell in zip(x, y, cells, strict=True):
                    v = ta_verdict(cell, tau)
                    _point(ax, xi, yi, cell["ratio_TA_se"][tau], c, mk, v == "unresolved")
            for xi, cell in zip(xs, cells, strict=True):
                if any(ta_verdict(cell, tau) == "fail" for tau, *_ in P2C_TAUS):
                    top = max(cell["ratio_TA"][t] + cell["ratio_TA_se"][t] for t, *_ in P2C_TAUS)
                    ax.annotate("fail", (xi, top), xytext=(0, 5), textcoords="offset points", fontsize=8, color=INK,
                                ha="center")  # fmt: skip
            ax.set_ylim((0.55, 1.42) if fam == "chirp" else (0.86, 1.17))
            ax.set_xticks(xs)
            ax.set_title(f"segment {seg}, {G3_FAMILIES[fam]}", fontsize=10)
            _style(ax)
        row[0].set_ylabel("R_TA: true front end / mode A")
    axes[0][0].text(70.6, 0.875, "run 3's grid ends at 70 dB-Hz", fontsize=8, color=INK2, ha="left", va="bottom")
    for ax in axes[2]:
        ax.set_xlabel("J/N₀ (dB-Hz)")
    handles = [Line2D([], [], color=c, marker=mk, markersize=8, label=lab) for _, lab, c, mk in P2C_TAUS]
    handles.append(Line2D([], [], color=INK2, marker="o", markerfacecolor=SURFACE, linestyle="none", markersize=8,
                          label=f"hollow: unresolved (SE > {P2C_SE_MAX:g})"))  # fmt: skip
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8.5, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.955), title="time constant of the true front end's recursive AGC",
               title_fontsize=8.5)  # fmt: skip
    fig.suptitle("Review s33's simulated true front end against mode A, ± 1 SE, up to 85 dB-Hz (band: the ±0.10 "
                 "tolerance)", color=INK, fontsize=11)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return _g3_save(fig, "f10-8_s33_true_analog.png")


def p2c_figures(root: Path) -> list[Path]:
    src = p2c_sources(root)
    return [
        fig_p2c_yardstick(src["run3"], src["run2"], src["s33"], src["s34"]),
        fig_p2c_ghost_excess(src["run3"]),
        fig_p2c_true_analog(src["s33"]),
    ]


def g3_figures(root: Path) -> list[Path]:
    src = g3_sources(root)
    return [
        fig_g3_paired(src["summary"]),
        fig_g3_yardstick(src["summary"], src["s29"]),
        fig_g3_b1(src["summary"], src["plan"], src["s29"]),
        fig_g3_identified_h(src["loops"]),
        fig_g3_k(src["e2k"], src["s30"]),
    ]


def main() -> None:
    if sys.argv[1:] == ["g2"]:
        print(f"figure at {fig_g2_coverage()}")
        return
    if sys.argv[1:] == ["g3"]:
        for path in g3_figures(results_root()):
            print(f"figure at {path}")
        return
    if sys.argv[1:] == ["p2c"]:
        for path in p2c_figures(results_root()):
            print(f"figure at {path}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    fig_resonance()
    fig_identifiable()
    fig_beat()
    print(f"figures in {OUT}")


if __name__ == "__main__":
    main()
