"""Run IDs, write-once result directories, manifests and the dirty-tree gate (docs/6 section 7)."""

import json
import subprocess
from pathlib import Path

import pytest

from gnssaudit import experiments
from gnssaudit.cli.main import main
from gnssaudit.experiments import Outcome
from gnssaudit.provenance import create_run_dir, git_describe, is_dirty, run_id

GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@example.org", "-c", "commit.gpgsign=false"]


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run([*GIT, "init", "-q"], cwd=repo, check=True)
    (repo / "a.txt").write_text("a\n")
    subprocess.run([*GIT, "add", "a.txt"], cwd=repo, check=True)
    subprocess.run([*GIT, "commit", "-qm", "init"], cwd=repo, check=True)
    return repo


def test_run_id_ignores_key_order() -> None:
    a = run_id({"id": "E0a", "models": ["B0", "B1"], "levels": {"x": 1, "y": 2}})
    b = run_id({"levels": {"y": 2, "x": 1}, "models": ["B0", "B1"], "id": "E0a"})
    assert a == b and len(a) == 12 and int(a, 16) >= 0
    assert run_id({"id": "E0a", "models": ["B1", "B0"]}) != run_id({"id": "E0a", "models": ["B0", "B1"]})


def test_dirty_tree_detection(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    assert not is_dirty(repo) and not git_describe(repo).endswith("-dirty")
    (repo / "a.txt").write_text("changed\n")
    assert is_dirty(repo) and git_describe(repo).endswith("-dirty")


def test_run_directory_is_write_once(tmp_path: Path) -> None:
    first = create_run_dir(tmp_path, "E0a", "abc123def456")
    assert first == tmp_path / "E0a" / "abc123def456" and first.is_dir()
    with pytest.raises(FileExistsError):
        create_run_dir(tmp_path, "E0a", "abc123def456")


def _spec(tmp_path: Path) -> Path:
    path = tmp_path / "Edummy.yaml"
    path.write_text("id: Edummy\nconfirmatory: false\nlevels: [0.95]\n")
    return path


def _dummy(spec: dict, run_dir: Path) -> Outcome:
    data = run_dir.parents[2] / "input.bin"  # outside the results tree
    data.write_bytes(b"x")
    (run_dir / "result.json").write_text(json.dumps({"ok": True}))
    return Outcome(inputs=[data], counts={"excluded_rows": 3}, summary={"ok": True})


def test_run_writes_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(experiments.REGISTRY, "Edummy", _dummy)
    monkeypatch.setenv("GNSSAUDIT_RESULTS", str(tmp_path / "results"))
    repo = _repo(tmp_path)
    assert main(["run", str(_spec(tmp_path))], repo=repo) == 0
    (run_dir,) = (tmp_path / "results" / "Edummy").iterdir()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["run_id"] == run_dir.name
    assert manifest["spec"]["id"] == "Edummy"
    assert set(manifest) >= {"git_describe", "inputs", "host", "timings_s", "counts", "started_utc", "summary"}
    assert list(manifest["inputs"].values())[0] == "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881"
    assert manifest["counts"] == {"excluded_rows": 3}
    assert main(["run", str(_spec(tmp_path))], repo=repo) == 3  # same spec again: write-once refusal


def test_confirmatory_run_refuses_dirty_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(experiments.REGISTRY, "Edummy", _dummy)
    monkeypatch.setenv("GNSSAUDIT_RESULTS", str(tmp_path / "results"))
    repo = _repo(tmp_path)
    (repo / "a.txt").write_text("changed\n")
    assert main(["run", str(_spec(tmp_path)), "--confirmatory"], repo=repo) == 2
    assert not (tmp_path / "results").exists()
