"""Regenerate the review studies' result files: runs the registered studies in order, one process each.

Usage: OPENBLAS_NUM_THREADS=1 uv run python -m review.run_all [s01 s06 ...] [--overwrite-preregistered]
With no study named it runs every registered study except the frozen records (NAMED_ONLY). A name is a prefix of a
study's name; a frozen record runs only when the prefix holds its whole id (`s29`, not `s2`), and s33 only with
--overwrite-preregistered as well. s17 to s26 and s36 are not registered. Each study writes
review/results/<name>.json with provenance. Studies parallelise internally.
"""

from __future__ import annotations

import importlib
import sys
import time
from collections.abc import Sequence

STUDIES = [
    "s01_loop_facts",
    "s02_window_and_p5",
    "s03_tdcp_and_spectral_witness",
    "s04_structured_interference",
    "s05_predictive_shape",
    "s06_mismatch_law",
    "s07_pilot_data_witness",
    "s08_gate_bias",
    "s09_oracle_and_slope",
    "s10_statistics",
    "s11_cn0_bias",
    "s12_gate_alternatives",
    "s13_lecam_bound",
    "s14_doppler_resonance",
    "s15_slips",
    "s27_slip_offset",
    "s16_lock_detector",
    "s28_twin_pair_quantisation",
    "s29_true_analog_jmax",
    "s30_k_rule_null",
    "s31_e2k_warmup_unlocks",
    "s32_g3_rerun_forecast",
    "s33_front_end_prediction",
    "s34_front_end_acceptance",
    "s35_run3_post_data",
]

# Frozen records, re-run only when named (and then re-pinned). Each JSON's SHA-256 is pinned or cited elsewhere, and a
# re-run rewrites meta.generated (review/_common.py), so every pin after it refuses and every citation goes stale:
# - s29: experiments/E2K.yaml, experiments/E2-v3.yaml, s32, s35, review/figures.py, tests/unit/test_e2k.py;
# - s30: review/figures.py;
# - s31: s32;
# - s32: cited in docs/10 and docs/12 (R-4's correction, run 3's sources);
# - s33: s34, s35, review/figures.py; its commit time is s34's order check;
# - s34: s35, review/figures.py;
# - s35: cited in docs/2, docs/6, docs/10 and docs/12.
NAMED_ONLY = frozenset(
    {
        "s29_true_analog_jmax",
        "s30_k_rule_null",
        "s31_e2k_warmup_unlocks",
        "s32_g3_rerun_forecast",
        "s33_front_end_prediction",
        "s34_front_end_acceptance",
        "s35_run3_post_data",
    }
)
# Pre-registered evidence (R-7): s33's JSON was committed before run 3 started. A re-run takes about 2.7 h, cannot
# reproduce its code hash (perf-inject changed gnssaudit.inject after a907825) and overwrites the committed JSON, so
# even when named it is refused unless OVERWRITE is given.
PREREGISTERED = frozenset({"s33_front_end_prediction"})
OVERWRITE = "--overwrite-preregistered"


def select(args: Sequence[str]) -> list[str]:
    """The studies a run with these arguments runs, in STUDIES' order: with no study named, every study not in
    NAMED_ONLY; otherwise each study whose name starts with a named prefix, a NAMED_ONLY one only when the prefix holds
    its whole id. Refuses an unknown option, and a PREREGISTERED study unless OVERWRITE is given."""
    options = {a for a in args if a.startswith("-")}
    if unknown := sorted(options - {OVERWRITE}):
        raise SystemExit(f"run_all: unknown option {unknown}; the only option is {OVERWRITE}")
    names = [a for a in args if not a.startswith("-")]

    def named(study: str) -> bool:
        ident = study.split("_")[0]
        return any(study.startswith(a) and (study not in NAMED_ONLY or a.startswith(ident)) for a in names)

    wanted = [s for s in STUDIES if named(s)] if names else [s for s in STUDIES if s not in NAMED_ONLY]
    if refused := [s for s in wanted if s in PREREGISTERED and OVERWRITE not in options]:
        raise SystemExit(
            f"run_all: {', '.join(refused)} is pre-registered evidence, and a re-run overwrites its committed JSON; "
            f"nothing was run. Add {OVERWRITE} to run it anyway"
        )
    return wanted


def main(argv: Sequence[str]) -> None:
    for name in select(argv):
        t0 = time.time()
        print(f"== {name}", flush=True)
        importlib.import_module(f"review.{name}").main()
        print(f"== {name} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
