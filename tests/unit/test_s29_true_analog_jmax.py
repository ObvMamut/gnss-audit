"""S29's own logic (P2b part 2 addendum, task 12): the adaptive length, the cell's verdict inputs, the J_max_TA plumbing
through front_end_jmax and combine_segments, H-A and H-B, and reading E2 run 2. The E2-structure chains are s28's."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from gnssaudit.experiments.e2_frontend import FrontEndCell
from review.s29_true_analog_jmax import (
    CHECKPOINT_WINDOWS,
    adaptive,
    decide,
    front_end,
    hypotheses,
    read_run2,
    verdict,
)

PASS = FrontEndCell((1.0,), 1.0, 0.01)
FAIL = FrontEndCell((1.0,), 0.85, 0.01)
UNRESOLVED = FrontEndCell((1.0,), 1.0, 0.08)
LEVELS = (50.0, 55.0, 60.0, 65.0, 70.0)


def stream(chunk: int, last: int, seen: list[int]) -> Iterator[tuple[int, str]]:
    """(windows complete, state) after each chunk of `chunk` windows up to `last`; records what was read."""
    done = 0
    while done < last:
        done = min(done + chunk, last)
        seen.append(done)
        yield done, f"state@{done}"


def stats_with(se: dict[int, float], calls: list[tuple[str, int]]) -> Any:
    def stats(z: str, n: int) -> dict[str, Any]:
        calls.append((z, n))
        return {"ratio_truth": 1.0, "ratio_truth_se": se[n], "n_windows": n}

    return stats


def test_checkpoints_are_20_40_60_s_of_1_ms_windows() -> None:
    assert CHECKPOINT_WINDOWS == (20_000, 40_000, 60_000)


def test_a_cell_stops_at_the_first_checkpoint_within_the_se_bar_and_reads_no_further() -> None:
    seen: list[int] = []
    calls: list[tuple[str, int]] = []
    st = adaptive(stream(5_000, 60_000, seen), stats_with({20_000: 0.03, 40_000: 0.019, 60_000: 0.01}, calls),
                  CHECKPOINT_WINDOWS)  # fmt: skip
    assert st["n_windows"] == 40_000
    assert st["se_stop_reached"] is True
    assert [c["n_windows"] for c in st["checkpoints"]] == [20_000, 40_000]
    assert [c["ratio_truth_se"] for c in st["checkpoints"]] == [0.03, 0.019]
    assert seen[-1] == 40_000  # the 60 s stream was not read past the stopping checkpoint
    assert [n for _, n in calls] == [20_000, 40_000]


def test_an_se_exactly_at_the_bar_stops() -> None:
    st = adaptive(stream(5_000, 60_000, []), stats_with({20_000: 0.02}, []), CHECKPOINT_WINDOWS)
    assert st["n_windows"] == 20_000
    assert st["se_stop_reached"] is True


def test_a_cell_above_the_bar_runs_to_60_s_and_keeps_that_result() -> None:
    st = adaptive(stream(5_000, 60_000, []), stats_with({20_000: 0.05, 40_000: 0.03, 60_000: 0.025}, []),
                  CHECKPOINT_WINDOWS)  # fmt: skip
    assert st["n_windows"] == 60_000
    assert st["ratio_truth_se"] == 0.025
    assert st["se_stop_reached"] is False
    assert len(st["checkpoints"]) == 3


def test_a_checkpoint_uses_exactly_its_windows_when_a_chunk_overshoots_it() -> None:
    calls: list[tuple[str, int]] = []
    adaptive(stream(8_193, 60_000, []), stats_with({20_000: 0.03, 40_000: 0.03, 60_000: 0.03}, calls),
             CHECKPOINT_WINDOWS)  # fmt: skip
    # the chunk that completes 20 000 windows ends at 24 579: the statistics see the first 20 000 only
    assert calls == [("state@24579", 20_000), ("state@40965", 40_000), ("state@60000", 60_000)]


def test_a_stream_shorter_than_the_last_checkpoint_is_refused() -> None:
    with pytest.raises(ValueError, match="before the last checkpoint"):
        adaptive(stream(5_000, 45_000, []), stats_with({20_000: 0.05, 40_000: 0.05}, []), CHECKPOINT_WINDOWS)


def test_the_cell_is_the_truth_pair_against_mode_a() -> None:
    st = {"slope_truth": 0.97, "ratio_truth": 1.03, "ratio_truth_se": 0.012, "slope_B": 0.5, "ratio_B": 0.7,
          "ratio_B_se": 0.001}  # fmt: skip
    assert front_end(st) == FrontEndCell(slopes=(0.97,), ratio=1.03, ratio_se=0.012)


def test_the_verdict_uses_the_addendum_tolerances() -> None:
    def v(slope: float, r: float, se: float) -> str:
        return verdict({"slope_truth": slope, "ratio_truth": r, "ratio_truth_se": se}, "test cell")

    assert v(1.0, 1.0, 0.01) == "pass"
    assert v(1.0, 0.91, 0.05) == "pass"  # se_max 0.05 is inclusive
    assert v(1.0, 1.0, 0.0501) == "unresolved"
    assert v(1.0, 0.899, 0.01) == "fail"  # tol 0.10: |R - 1| < 0.10 passes (0.90 itself rounds to inside in floats)
    assert v(1.0, 1.101, 0.01) == "fail"
    assert v(0.899, 1.0, 0.01) == "fail"  # slope_tol 0.10
    assert v(0.899, 1.0, 0.2) == "fail"  # a slope fail is a fail even unresolved
    with pytest.raises(ValueError, match="test cell"):
        v(1.0, float("nan"), 0.01)


def series(*cells: FrontEndCell) -> dict[float, FrontEndCell]:
    return dict(zip(LEVELS, cells, strict=True))


def test_j_max_ta_is_front_end_jmax_per_segment_and_combine_segments_across() -> None:
    fe = {
        "A": {"noise": series(UNRESOLVED, PASS, PASS, PASS, PASS),  # unresolved does not block: 70
              "swept_cw": series(PASS, PASS, FAIL, PASS, PASS),  # a fail blocks every level above it: 55
              "chirp": series(UNRESOLVED, UNRESOLVED, PASS, PASS, PASS)},
        "B": {"noise": series(FAIL, PASS, PASS, PASS, PASS),  # a fail at the weakest level: None
              "swept_cw": series(PASS, PASS, PASS, PASS, PASS),
              "chirp": series(UNRESOLVED, UNRESOLVED, UNRESOLVED, UNRESOLVED, UNRESOLVED)},  # nothing passes: None
    }  # fmt: skip
    d = decide(fe)
    assert d["noise"]["j_max_ta_db_hz"] == {"A": 70.0, "B": None}
    assert d["noise"]["combined_db_hz"] is None  # None on either segment is None
    assert d["swept_cw"]["j_max_ta_db_hz"] == {"A": 55.0, "B": 70.0}
    assert d["swept_cw"]["combined_db_hz"] == 55.0  # otherwise the minimum
    assert d["chirp"]["j_max_ta_db_hz"] == {"A": 70.0, "B": None}
    assert d["chirp"]["combined_db_hz"] is None


def test_noise_and_swept_cw_decide_and_the_chirp_is_reported() -> None:
    fe = {s: {f: series(PASS, PASS, PASS, PASS, PASS) for f in ("noise", "swept_cw", "chirp")} for s in ("A", "B")}
    d = decide(fe)
    assert d["noise"]["decides"] is True
    assert d["swept_cw"]["decides"] is True
    assert d["chirp"]["decides"] is False
    assert "reported" in d["chirp"]["label"]


def h_cells(b_over_truth: float, r_ta: float) -> dict[str, Any]:
    cell = {"b_over_truth": b_over_truth, "b_over_truth_se": 0.02, "ratio_truth": r_ta, "ratio_truth_se": 0.02,
            "ratio_B": 0.9, "ratio_B_se": 0.03, "n_windows": 40_000}  # fmt: skip
    return {"A": {"noise": {55.0: cell}}}


def test_h_b_is_mode_b_over_truth_at_most_0_92_on_segment_a_noise_55() -> None:
    assert hypotheses(h_cells(0.92, 1.0))["H_B"]["holds"] is True
    assert hypotheses(h_cells(0.9201, 1.0))["H_B"]["holds"] is False


def test_h_a_is_r_ta_within_0_10_of_one_at_that_cell() -> None:
    assert hypotheses(h_cells(0.9, 1.0999))["H_A"]["holds"] is True
    assert hypotheses(h_cells(0.9, 0.899))["H_A"]["holds"] is False
    assert hypotheses(h_cells(0.9, 1.101))["H_A"]["holds"] is False
    h = hypotheses(h_cells(0.9, 1.0))
    assert "reported" in h["label"]
    assert (h["segment"], h["family"], h["level_db_hz"]) == ("A", "noise", 55.0)


def write_run(tmp: Path, run_id: str) -> Path:
    run = tmp / run_id
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"run_id": run_id, "git_describe": "7bc707e5a4e3"}))
    seg = {
        "analog_sd": 1.5,
        "n0_per_hz": 1.7e-7,
        "j_max_db_hz": {"noise": None},
        "tau_cap_db_hz": {"noise": None},
        "tau_side": {"noise": {"0.001": {"ratio": 0.97, "ratio_se": 0.008, "verdict": "pass", "units": ["u"]}}},
    }
    segs = {"A": seg, "B": {**seg, "analog_sd": 1.52, "tau_cap_db_hz": {}, "tau_side": {}}}
    (run / "summary.json").write_text(json.dumps({"segments": segs, "decisions": {"j_max_db_hz": {"noise": None}}}))
    return run


def test_run_2_is_read_with_its_run_id_and_segment_sds(tmp_path: Path) -> None:
    got = read_run2(write_run(tmp_path, "6214cf52489f"), "6214cf52489f")
    assert got["run_id"] == "6214cf52489f"
    assert got["analog_sd"] == {"A": 1.5, "B": 1.52}
    assert len(got["summary_sha256"]) == 64
    assert got["tau_side"]["A"]["noise"]["0.001"] == {"ratio": 0.97, "ratio_se": 0.008, "verdict": "pass"}


def test_another_run_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="6214cf52489f"):
        read_run2(write_run(tmp_path, "000000000000"), "6214cf52489f")
