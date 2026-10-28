"""`receiver_patch` (D-40, P2b part 2 task 9b): a spec may name the patched receiver build it needs. `gnssaudit run`
refuses it, before the run directory exists, unless it equals `gnssaudit.sdr.images.PATCH_ID`; a spec without the key
runs as before. The key is part of the spec, so it gives the run its own run id."""

from pathlib import Path

import pytest

from gnssaudit import experiments
from gnssaudit.cli.main import main
from gnssaudit.experiments import Outcome
from gnssaudit.provenance import run_id
from gnssaudit.sdr.images import PATCH_ID

P1 = "p1:0001-acc-phase-float64+0002-log-every-integration"


@pytest.fixture
def seen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def stub(spec: dict, run_dir: Path) -> Outcome:
        calls.append(spec)
        return Outcome(inputs=[], counts={}, summary={})

    monkeypatch.setitem(experiments.REGISTRY, "Epatch", stub)
    monkeypatch.setenv("GNSSAUDIT_RESULTS", str(tmp_path / "results"))
    monkeypatch.setattr("gnssaudit.cli.main.is_dirty", lambda repo: False)
    return calls


def _spec(tmp_path: Path, patch: str | None) -> Path:
    path = tmp_path / "Epatch.yaml"
    path.write_text("id: Epatch\nunits: 1\n" + ("" if patch is None else f"receiver_patch: '{patch}'\n"))
    return path


def test_a_spec_naming_the_current_patch_runs(tmp_path: Path, seen: list[dict]) -> None:
    assert main(["run", str(_spec(tmp_path, PATCH_ID))]) == 0
    assert seen[0]["receiver_patch"] == PATCH_ID
    (run_dir,) = (tmp_path / "results" / "Epatch").iterdir()
    assert (run_dir / "manifest.json").exists()


@pytest.mark.parametrize("patch", [P1, PATCH_ID + " ", PATCH_ID.upper(), ""])
def test_a_spec_naming_another_patch_is_refused_before_the_run_directory_exists(
    tmp_path: Path, seen: list[dict], patch: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """A refusal like the CLI's others (final review R2, minor 1): a line on stderr and exit code 5, no traceback."""
    assert main(["run", str(_spec(tmp_path, patch))]) == 5
    assert main(["run", str(_spec(tmp_path, patch)), "--resume"]) == 5
    err = capsys.readouterr().err
    assert err.count(f"the spec needs receiver_patch {patch!r}, but this tree runs {PATCH_ID!r}") == 2
    assert not (tmp_path / "results").exists() and seen == []


def test_a_spec_without_the_key_runs_as_before(tmp_path: Path, seen: list[dict]) -> None:
    assert main(["run", str(_spec(tmp_path, None))]) == 0
    assert "receiver_patch" not in seen[0]


def test_the_key_gives_the_run_its_own_id() -> None:
    base = {"id": "Epatch", "units": 1}
    assert run_id({**base, "receiver_patch": PATCH_ID}) != run_id(base)
    assert run_id({**base, "receiver_patch": PATCH_ID}) != run_id({**base, "receiver_patch": P1})
