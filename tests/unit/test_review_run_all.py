"""review.run_all's selection (P2c final review, I2): the frozen records, whose JSON is pinned or cited by SHA-256,
run only when named, and s33's pre-registered JSON is never re-run without the explicit flag. Nothing is run."""

import hashlib
import importlib
from pathlib import Path

import pytest

from review import run_all
from review.run_all import NAMED_ONLY, OVERWRITE, PREREGISTERED, STUDIES, select

REPO = Path(__file__).resolve().parents[2]
DEFAULT = (
    "s01_loop_facts s02_window_and_p5 s03_tdcp_and_spectral_witness s04_structured_interference s05_predictive_shape "
    "s06_mismatch_law s07_pilot_data_witness s08_gate_bias s09_oracle_and_slope s10_statistics s11_cn0_bias "
    "s12_gate_alternatives s13_lecam_bound s14_doppler_resonance s15_slips s27_slip_offset s16_lock_detector "
    "s28_twin_pair_quantisation"
).split()


def test_the_default_run_is_s01_to_s16_s27_and_s28() -> None:
    """`make review` with no study named: s29 to s35 are frozen records and stay out (s36 is not registered)."""
    assert select([]) == DEFAULT
    assert NAMED_ONLY == {s for s in STUDIES if s not in DEFAULT}
    assert sorted(s[:3] for s in NAMED_ONLY) == ["s29", "s30", "s31", "s32", "s33", "s34", "s35"]
    assert PREREGISTERED == {"s33_front_end_prediction"}


def test_no_study_whose_json_is_cited_by_sha256_runs_by_default() -> None:
    """A study whose committed JSON's SHA-256 (its first 8 hex digits) appears in the docs, the specs, the review code
    or the tests is a frozen record: a re-run rewrites meta.generated and breaks that pin."""
    sources = [("docs", "**/*.md"), ("experiments", "*.yaml"), ("review", "*.py"), ("tests", "**/*.py")]
    corpus = "\n".join(p.read_text(errors="replace") for d, pattern in sources for p in (REPO / d).glob(pattern))
    results = REPO / "review" / "results"
    cited = {
        s
        for s in STUDIES
        if (results / f"{s}.json").is_file()
        and hashlib.sha256((results / f"{s}.json").read_bytes()).hexdigest()[:8] in corpus
    }
    assert {"s29_true_analog_jmax", "s33_front_end_prediction"} <= cited  # the scan finds the pins it must
    assert cited <= NAMED_ONLY


def test_a_frozen_record_runs_only_when_its_whole_id_names_it() -> None:
    assert select(["s29"]) == ["s29_true_analog_jmax"]
    assert select(["s35_run3"]) == ["s35_run3_post_data"]
    assert select(["s2"]) == ["s27_slip_offset", "s28_twin_pair_quantisation"]
    assert select(["s3"]) == []
    assert select(["s01", "s06"]) == ["s01_loop_facts", "s06_mismatch_law"]


def test_s33_is_refused_before_anything_runs_unless_the_flag_is_given(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib, "import_module", lambda name: pytest.fail(f"{name} was imported"))
    for args in (["s33"], ["s32", "s33", "s34"], ["s33_front_end_prediction"]):
        with pytest.raises(SystemExit, match="s33_front_end_prediction is pre-registered"):
            run_all.main(args)
    assert select(["s33", OVERWRITE]) == ["s33_front_end_prediction"]
    assert select([OVERWRITE]) == DEFAULT


def test_an_unknown_option_is_refused() -> None:
    with pytest.raises(SystemExit, match="unknown option"):
        select(["s33", "--overwrite"])
