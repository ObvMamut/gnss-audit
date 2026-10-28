"""S30's own logic (P2b part 2, task 11c): its vectorised statistic is the oracle's estimator, and it reads E2K only as
pinned (the run's directory, started.json's describe, summary.json's SHA-256, each unit file's SHA-256 in done.json,
unit.json included)."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from review import s30_k_rule_null as s30
from review.s30_k_rule_null import SourceRefused, check_against_project, e2k_provenance, unit_record, unit_table


@pytest.mark.parametrize("seed", [1, 2])
def test_the_vectorised_statistics_are_the_oracle_estimator(seed: int) -> None:
    out = check_against_project(seed=seed, rows=120)  # raises if any design differs beyond 1e-12
    assert set(out["gnssaudit"]) == set(s30.DESIGNS)
    for d in s30.DESIGNS:
        assert out["vectorised"][d] == pytest.approx(out["gnssaudit"][d], rel=1e-12)


def write_e2k(run: Path, describe: str, summary: bytes) -> None:
    run.mkdir(parents=True)
    (run / "started.json").write_text(json.dumps({"git_describe": describe}))
    (run / "summary.json").write_bytes(summary)


def test_e2k_is_read_only_as_pinned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    summary = json.dumps({"decisions": {"k": 51}}).encode()
    write_e2k(tmp_path / s30.E2K_ID, s30.E2K_DESCRIBE, summary)
    monkeypatch.setattr(s30, "E2K_SUMMARY_SHA256", hashlib.sha256(summary).hexdigest())
    prov = e2k_provenance(tmp_path / s30.E2K_ID)
    assert prov["summary"] == {"decisions": {"k": 51}}
    assert prov["describe"] == s30.E2K_DESCRIBE


def test_another_run_id_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    summary = b"{}"
    write_e2k(tmp_path / "0000aaaa1111", s30.E2K_DESCRIBE, summary)
    monkeypatch.setattr(s30, "E2K_SUMMARY_SHA256", hashlib.sha256(summary).hexdigest())
    with pytest.raises(SourceRefused, match="not E2K run 0443516ca48c"):
        e2k_provenance(tmp_path / "0000aaaa1111")


def test_e2k_started_by_other_code_is_refused(tmp_path: Path) -> None:
    write_e2k(tmp_path / s30.E2K_ID, "fdb9ccb00000", b"{}")
    with pytest.raises(SourceRefused, match="started.json names code 'fdb9ccb00000'"):
        e2k_provenance(tmp_path / s30.E2K_ID)


def test_an_e2k_summary_that_is_not_the_pinned_one_is_refused(tmp_path: Path) -> None:
    write_e2k(tmp_path / s30.E2K_ID, s30.E2K_DESCRIBE, b"{}")
    with pytest.raises(SourceRefused, match="summary.json SHA-256"):
        e2k_provenance(tmp_path / s30.E2K_ID)


def test_a_unit_file_is_read_only_if_done_json_hashed_it(tmp_path: Path) -> None:
    unit = tmp_path / "A-k-cond-r0"
    unit.mkdir()
    pq.write_table(pa.table({"e_cyc": np.arange(3.0)}), unit / "truth_obs.parquet")
    digest = hashlib.sha256((unit / "truth_obs.parquet").read_bytes()).hexdigest()
    (unit / "done.json").write_text(json.dumps({"files": {"truth_obs.parquet": digest}}))
    assert unit_table(unit, "truth_obs.parquet").column("e_cyc").to_pylist() == [0.0, 1.0, 2.0]
    (unit / "done.json").write_text(json.dumps({"files": {"truth_obs.parquet": "0" * 64}}))
    with pytest.raises(SourceRefused, match="done.json records"):
        unit_table(unit, "truth_obs.parquet")
    (unit / "done.json").write_text(json.dumps({"files": {}}))
    with pytest.raises(SourceRefused, match="done.json records None"):
        unit_table(unit, "truth_obs.parquet")


def test_unit_json_is_read_only_if_done_json_hashed_it(tmp_path: Path) -> None:
    unit = tmp_path / "A-k-cond-r0"
    unit.mkdir()
    (unit / "unit.json").write_text(json.dumps({"lock_ok": True, "obs_onset_label": 1.0}))
    digest = hashlib.sha256((unit / "unit.json").read_bytes()).hexdigest()
    (unit / "done.json").write_text(json.dumps({"files": {"unit.json": digest}}))
    assert unit_record(unit) == {"lock_ok": True, "obs_onset_label": 1.0}
    (unit / "unit.json").write_text(json.dumps({"lock_ok": False, "obs_onset_label": 1.0}))
    with pytest.raises(SourceRefused, match="done.json records"):
        unit_record(unit)
