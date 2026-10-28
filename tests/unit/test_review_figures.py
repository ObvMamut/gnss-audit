"""The G3 figures read only the sources they name (review/figures.py, P2b part 2 tasks 11b and 11c): a run whose
manifest names another run id, a file whose SHA-256 differs, or an s29 or s30 result that reads another run is
refused."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from review import figures
from review.figures import SourceRefused, checked_file, checked_run, g3_k_sources, g3_sources


def write_run(root: Path, experiment: str, dirname: str, run_id: str) -> Path:
    run = root / experiment / dirname
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"run_id": run_id}))
    return run


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_named_run_is_accepted(tmp_path: Path) -> None:
    run = write_run(tmp_path, "E2", "6214cf52489f", "6214cf52489f")
    assert checked_run(tmp_path, "E2", "6214cf52489f") == run


def test_a_run_whose_manifest_names_another_run_id_is_refused(tmp_path: Path) -> None:
    write_run(tmp_path, "E2", "6214cf52489f", "4b3e282ef77d")
    with pytest.raises(SourceRefused, match="names run '4b3e282ef77d'"):
        checked_run(tmp_path, "E2", "6214cf52489f")


def test_an_unfinished_run_without_a_manifest_is_refused(tmp_path: Path) -> None:
    (tmp_path / "E2" / "6214cf52489f").mkdir(parents=True)
    with pytest.raises(SourceRefused, match="no manifest"):
        checked_run(tmp_path, "E2", "6214cf52489f")


def test_a_file_whose_sha256_differs_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "s29.json"
    path.write_text("{}")
    assert checked_file(path, sha(path)) == path
    with pytest.raises(SourceRefused, match="SHA-256"):
        checked_file(path, "0" * 64)


def test_an_s29_result_that_reads_another_e2_run_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = write_run(tmp_path, "E2", figures.G3_E2_RUN, figures.G3_E2_RUN)
    (run / "summary.json").write_text(json.dumps({"segments": {}}))
    s29 = tmp_path / "s29.json"
    s29.write_text(json.dumps({"e2_run2": {"run_id": "4b3e282ef77d", "summary_sha256": sha(run / "summary.json")}}))
    monkeypatch.setattr(figures, "G3_E2_SUMMARY_SHA256", sha(run / "summary.json"))
    monkeypatch.setattr(figures, "G3_S29", s29)
    monkeypatch.setattr(figures, "G3_S29_SHA256", sha(s29))
    with pytest.raises(SourceRefused, match="s29 reads run 4b3e282ef77d"):
        g3_sources(tmp_path)


def test_a_run_2_summary_that_is_not_the_pinned_one_is_refused(tmp_path: Path) -> None:
    run = write_run(tmp_path, "E2", figures.G3_E2_RUN, figures.G3_E2_RUN)
    (run / "summary.json").write_text(json.dumps({"segments": {}}))
    with pytest.raises(SourceRefused, match="SHA-256"):
        g3_sources(tmp_path)


def write_e2k_and_s30(root: Path, monkeypatch: pytest.MonkeyPatch, run_id: str) -> None:
    run = root / "E2K" / figures.G3_E2K_RUN
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({"decisions": {"k": 51}}))
    s30 = root / "s30.json"
    s30.write_text(json.dumps({"e2k": {"run_id": run_id, "summary_sha256": sha(run / "summary.json")}}))
    monkeypatch.setattr(figures, "G3_E2K_SUMMARY_SHA256", sha(run / "summary.json"))
    monkeypatch.setattr(figures, "G3_S30", s30)
    monkeypatch.setattr(figures, "G3_S30_SHA256", sha(s30))


def test_e2k_and_s30_are_read_as_pinned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_e2k_and_s30(tmp_path, monkeypatch, figures.G3_E2K_RUN)
    assert g3_k_sources(tmp_path)["e2k"] == {"decisions": {"k": 51}}


def test_an_s30_result_that_reads_another_e2k_run_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_e2k_and_s30(tmp_path, monkeypatch, "0000aaaa1111")
    with pytest.raises(SourceRefused, match="s30 reads run 0000aaaa1111"):
        g3_k_sources(tmp_path)


def test_an_e2k_summary_that_is_not_the_pinned_one_is_refused(tmp_path: Path) -> None:
    run = tmp_path / "E2K" / figures.G3_E2K_RUN
    run.mkdir(parents=True)
    (run / "summary.json").write_text("{}")
    with pytest.raises(SourceRefused, match="SHA-256"):
        g3_k_sources(tmp_path)


def test_an_s30_statistic_that_is_not_e2ks_is_refused() -> None:
    e2k = {"decisions": {"k": 51, "k_convergence": {"median_rel_change": 0.0979, "epochs": 26898.0}}}
    s30 = {"e2k": {"designs": {"nested": {"median_rel_change": 0.0978}}}}
    with pytest.raises(SourceRefused, match="not E2K's decisions.k_convergence"):
        figures.fig_g3_k(e2k, s30)


# ----------------------------------------------------------------------------------------- P2c figures (f10-6 to f10-8)
def write_p2c(root: Path, monkeypatch: pytest.MonkeyPatch, *, s33_run2: str, s34_run3_sha: str | None = None) -> None:
    """Run 3 and run 2 under root, s33 and s34 beside them, every pin pointed at these files; s33 names run 2 as
    s33_run2, and s34 names run 3's summary SHA-256 as s34_run3_sha (the real one when None)."""
    run3 = write_run(root, "E2", figures.P2C_E2_RUN, figures.P2C_E2_RUN)
    (run3 / "summary.json").write_text(json.dumps({"segments": {"A": {}}}))
    run2 = write_run(root, "E2", figures.G3_E2_RUN, figures.G3_E2_RUN)
    (run2 / "summary.json").write_text(json.dumps({"segments": {}}))
    s33 = root / "s33.json"
    s33.write_text(json.dumps({"e2_run2": {"run_id": s33_run2, "summary_sha256": sha(run2 / "summary.json")}}))
    s34 = root / "s34.json"
    s34.write_text(json.dumps({"inputs": {
        "run3_summary": {"run_id": figures.P2C_E2_RUN, "sha256": s34_run3_sha or sha(run3 / "summary.json")},
        "s33": {"sha256": sha(s33)},
        "run2_summary": {"sha256": sha(run2 / "summary.json")},
    }}))  # fmt: skip
    monkeypatch.setattr(figures, "P2C_E2_SUMMARY_SHA256", sha(run3 / "summary.json"))
    monkeypatch.setattr(figures, "G3_E2_SUMMARY_SHA256", sha(run2 / "summary.json"))
    monkeypatch.setattr(figures, "P2C_S33", s33)
    monkeypatch.setattr(figures, "P2C_S33_SHA256", sha(s33))
    monkeypatch.setattr(figures, "P2C_S34", s34)
    monkeypatch.setattr(figures, "P2C_S34_SHA256", sha(s34))


def test_the_p2c_sources_are_read_as_pinned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_p2c(tmp_path, monkeypatch, s33_run2=figures.G3_E2_RUN)
    assert figures.p2c_sources(tmp_path)["run3"] == {"segments": {"A": {}}}


def test_a_run_3_summary_that_is_not_the_pinned_one_is_refused(tmp_path: Path) -> None:
    run = write_run(tmp_path, "E2", figures.P2C_E2_RUN, figures.P2C_E2_RUN)
    (run / "summary.json").write_text("{}")
    with pytest.raises(SourceRefused, match="SHA-256"):
        figures.p2c_sources(tmp_path)


def test_an_s33_result_built_on_another_run_2_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_p2c(tmp_path, monkeypatch, s33_run2="4b3e282ef77d")
    with pytest.raises(SourceRefused, match="s33 reads run 4b3e282ef77d"):
        figures.p2c_sources(tmp_path)


def test_an_s34_result_that_judged_another_run_3_summary_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_p2c(tmp_path, monkeypatch, s33_run2=figures.G3_E2_RUN, s34_run3_sha="0" * 64)
    with pytest.raises(SourceRefused, match="s34 judged"):
        figures.p2c_sources(tmp_path)


def front_end_cell(ratio: float, se: float) -> dict[str, float]:
    return {"ratio": ratio, "ratio_se": se}


def s34_segment(z: dict[str, float], **over: float) -> dict[str, dict[str, Any]]:
    """An s34 result for segment A with these z, and chi2, n and max_abs_z as s34 forms them unless overridden."""
    rec: dict[str, Any] = {"z": z, "chi2": sum(v * v for v in z.values()), "n": len(z),
                           "max_abs_z": max(abs(v) for v in z.values())}  # fmt: skip
    return {"segments": {"A": {**rec, **over}}}


RUN3_TWO = {"segments": {"A": {"front_end": {"noise": {"50.0": front_end_cell(0.97, 0.03),
                                                       "55.0": front_end_cell(0.92, 0.02)}}}}}  # fmt: skip
S33_TWO = {"cells": {"A": {"noise": {"50.0": {"ratio_B": 0.90, "ratio_B_se": 0.04},
                                     "55.0": {"ratio_B": 0.80, "ratio_B_se": 0.01}}}}}  # fmt: skip
Z_TWO = {
    "noise/50.0": (0.97 - 0.90) / (0.03**2 + 0.04**2) ** 0.5,
    "noise/55.0": (0.92 - 0.80) / (0.02**2 + 0.01**2) ** 0.5,
}


def test_f10_6_refuses_an_s34_z_that_its_inputs_do_not_give() -> None:
    """f10-6 draws run 3 and s33 beside s34's verdict; a z that run 3's and s33's values do not reproduce is refused."""
    figures.check_s34_z(RUN3_TWO, S33_TWO, s34_segment(Z_TWO))
    with pytest.raises(SourceRefused, match="noise/50.0"):
        figures.check_s34_z(RUN3_TWO, S33_TWO, s34_segment({**Z_TWO, "noise/50.0": Z_TWO["noise/50.0"] + 1e-6}))


@pytest.mark.parametrize(
    ("field", "value"), [("chi2", sum(v * v for v in Z_TWO.values()) + 1e-6), ("n", 3), ("max_abs_z", 1.0)]
)
def test_f10_6_refuses_an_s34_chi2_n_or_max_z_that_its_z_do_not_give(field: str, value: float) -> None:
    """f10-6 prints s34's chi2, n and max |z| per segment; each must be what the reproduced z give."""
    with pytest.raises(SourceRefused, match=field):
        figures.check_s34_z(RUN3_TWO, S33_TWO, s34_segment(Z_TWO, **{field: value}))
