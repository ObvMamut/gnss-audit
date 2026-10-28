"""S31's own logic (P2b part 2, task 11c fix round 1): the receiver-log parser, the order of a unit's logs, the pairs
left out of the thermal comparison, the per-run count of policy misses, and the refusal of another E2 run."""

import hashlib
import json
from pathlib import Path

import pytest

from review import s31_e2k_warmup_unlocks as s31
from review.s30_k_rule_null import SourceRefused
from review.s31_e2k_warmup_unlocks import affected_pairs, attempt_logs, by_run, parse_log, read_run2

LOG = """Current receiver time: 1 s
Tracking of GPS L1 C/A signal started on channel 7 for satellite GPS PRN 32 (Block IIF)
Tracking of GPS L1 C/A signal started on channel 8 for satellite GPS PRN 31 (Block IIR-M)
Current receiver time: 7 s
GPS L1 C/A tracking bit synchronization locked in channel 7 for satellite GPS PRN 32 (Block IIF)
Current receiver time: 1 min 2 s
GPS L1 C/A histogram bit synchronization locked in channel 8 for satellite GPS PRN 31 (Block IIR-M)
"""


def test_the_log_parser_reads_starts_and_bit_sync_in_receiver_time() -> None:
    p = parse_log(LOG)
    assert p[7] == {"starts": [1.0], "sync": [7.0]}
    assert p[8] == {"starts": [1.0], "sync": [62.0]}  # synced after a 30 s warm-up: a false lock by the proxy


def test_a_units_logs_come_in_receiver_run_order(tmp_path: Path) -> None:
    unit = tmp_path / "A-thermal-r7"
    unit.mkdir()
    for name in ("A-thermal-r7~10.gnss-sdr.log", "A-thermal-r7~2.gnss-sdr.log", "A-thermal-r7.gnss-sdr.log"):
        (unit / name).write_text("")
    assert [p.name for p in attempt_logs(unit)] == [
        "A-thermal-r7.gnss-sdr.log",
        "A-thermal-r7~2.gnss-sdr.log",
        "A-thermal-r7~10.gnss-sdr.log",
    ]


def test_a_pair_is_left_out_for_either_twin() -> None:
    units = {
        "A-thermal-r0": {"lock_ok": True, "unit": {"clean": "A-thermal-clean-r0"}},
        "A-thermal-clean-r0": {"lock_ok": False},
        "A-thermal-r1": {"lock_ok": True, "unit": {"clean": "A-thermal-clean-r1"}},
        "A-thermal-clean-r1": {"lock_ok": True},
        "A-thermal-r12": {"lock_ok": False, "unit": {"clean": "A-thermal-clean-r12"}},
        "A-thermal-clean-r12": {"lock_ok": True},
    }
    lock_failed, any_run = affected_pairs(units, ["A-thermal-r0", "A-thermal-r1", "A-thermal-r12"],
                                          {"A-thermal-clean-r1"})  # fmt: skip
    assert lock_failed == [0, 12]
    assert any_run == [1]


def test_policy_misses_are_counted_per_receiver_run() -> None:
    units = {
        "a": {"attempts": 1, "lock_ok": True},
        "b": {"attempts": 2, "lock_ok": True},  # missed run 1, passed run 2
        "c": {"attempts": 3, "lock_ok": False},  # missed all three
    }
    assert by_run(units, ["a", "b", "c"]) == [
        {"run": 1, "reached": 3, "missed": 2},
        {"run": 2, "reached": 2, "missed": 1},
        {"run": 3, "reached": 1, "missed": 1},
    ]


def test_another_e2_run_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"run_id": "4b3e282ef77d", "spec": {"warmup_s": 30.0}}))
    (run / "summary.json").write_text("{}")
    with pytest.raises(SourceRefused, match="names run '4b3e282ef77d'"):
        read_run2(run)
    (run / "manifest.json").write_text(json.dumps({"run_id": s31.RUN2_ID, "spec": {"warmup_s": 30.0}}))
    with pytest.raises(SourceRefused, match="not run 2's pinned summary"):
        read_run2(run)
    summary = {"segments": {"A": {"channels": {"7": ["G", "1C", 32]}, "line_excess_db": 2.08}}}
    (run / "summary.json").write_text(json.dumps(summary))
    monkeypatch.setattr(s31, "RUN2_SUMMARY_SHA256", hashlib.sha256((run / "summary.json").read_bytes()).hexdigest())
    assert read_run2(run)["channels"] == {7: 32}
