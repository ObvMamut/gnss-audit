"""`gnssaudit run --resume` (P2b task 11): a long run that was interrupted continues in its own run directory, which
has no manifest yet; a finished run stays write-once, and a resume under another code version is refused."""

import json
import subprocess
from pathlib import Path

import pytest

from gnssaudit import experiments
from gnssaudit.cli.main import main
from gnssaudit.experiments import Outcome

GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@example.org", "-c", "commit.gpgsign=false"]


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run([*GIT, "init", "-q"], cwd=repo, check=True)
    (repo / "a.txt").write_text("a\n")
    subprocess.run([*GIT, "add", "a.txt"], cwd=repo, check=True)
    subprocess.run([*GIT, "commit", "-qm", "init"], cwd=repo, check=True)
    return repo


def _spec(tmp_path: Path) -> Path:
    path = tmp_path / "Elong.yaml"
    path.write_text("id: Elong\nunits: 2\n")
    return path


class Interrupted(RuntimeError):
    pass


def _runner(seen: list[list[str]]):  # noqa: ANN202
    def run(spec: dict, run_dir: Path) -> Outcome:
        seen.append(sorted(p.name for p in run_dir.iterdir()))
        if not (run_dir / "unit-1.done").exists():
            (run_dir / "unit-1.done").write_text("x")
            raise Interrupted("the machine slept")
        (run_dir / "unit-2.done").write_text("x")
        return Outcome(inputs=[], counts={"units": 2}, summary={"ok": True})

    return run


def test_an_interrupted_run_continues_in_its_own_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []
    monkeypatch.setitem(experiments.REGISTRY, "Elong", _runner(seen))
    monkeypatch.setenv("GNSSAUDIT_RESULTS", str(tmp_path / "results"))
    repo, spec = _repo(tmp_path), str(_spec(tmp_path))
    with pytest.raises(Interrupted):
        main(["run", spec], repo=repo)
    (run_dir,) = (tmp_path / "results" / "Elong").iterdir()
    started = json.loads((run_dir / "started.json").read_text())
    assert main(["run", spec], repo=repo) == 3  # without --resume the directory is write-once
    assert main(["run", spec, "--resume"], repo=repo) == 0
    assert seen[-1] == ["started.json", "unit-1.done"]  # the runner sees the finished unit
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["started_utc"] == started["started_utc"] and manifest["git_describe"] == started["git_describe"]
    assert main(["run", spec, "--resume"], repo=repo) == 3  # finished: write-once again


def test_a_resume_under_another_code_version_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []
    monkeypatch.setitem(experiments.REGISTRY, "Elong", _runner(seen))
    monkeypatch.setenv("GNSSAUDIT_RESULTS", str(tmp_path / "results"))
    repo, spec = _repo(tmp_path), str(_spec(tmp_path))
    with pytest.raises(Interrupted):
        main(["run", spec], repo=repo)
    (repo / "a.txt").write_text("b\n")
    subprocess.run([*GIT, "commit", "-qam", "change"], cwd=repo, check=True)
    assert main(["run", spec, "--resume"], repo=repo) == 4
    assert len(seen) == 1  # the runner never ran again


@pytest.mark.parametrize("started", [None, "", "{}"])
def test_a_resume_without_a_readable_started_file_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], started: str | None
) -> None:
    """Final review R2, minor 10: a run directory whose started.json is missing (a crash between the directory and the
    file, or a directory from before --resume) or unreadable has no known code version: refused with exit code 4, a
    line on stderr and no traceback, and the runner never runs."""
    seen: list[list[str]] = []
    monkeypatch.setitem(experiments.REGISTRY, "Elong", _runner(seen))
    monkeypatch.setenv("GNSSAUDIT_RESULTS", str(tmp_path / "results"))
    repo, spec = _repo(tmp_path), str(_spec(tmp_path))
    with pytest.raises(Interrupted):
        main(["run", spec], repo=repo)
    (run_dir,) = (tmp_path / "results" / "Elong").iterdir()
    if started is None:
        (run_dir / "started.json").unlink()
    else:
        (run_dir / "started.json").write_text(started)
    assert main(["run", spec, "--resume"], repo=repo) == 4
    assert "started.json" in capsys.readouterr().err and len(seen) == 1
