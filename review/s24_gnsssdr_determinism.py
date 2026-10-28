"""S24: is GNSS-SDR deterministic on identical file input? (F24; docs/9 section 3, gate G3, the B1 identity.)

S23 showed two runs of byte-identical pre-onset samples starting tracking 41 ms apart, with no bitwise-identical
tracking records. Here one C/A recording (55 dB-Hz, 60 s) is processed three times with the acquisition in its
default (non-blocking) mode and three times with `Acquisition_1C.blocking=true`. Reported: the tracking start
sample of each run and the fraction of tracking records that are bitwise identical to the first run's.

Usage: uv run python -m review.s24_gnsssdr_determinism   (needs Docker via sg)
"""

from __future__ import annotations

import numpy as np

from review._common import save
from review.s18_gnsssdr_cn0_lock_p6 import OUT, generate, run_receiver


def compare(ref: np.ndarray, other: np.ndarray) -> dict[str, object]:
    common = np.intersect1d(ref["prn_start_sample"], other["prn_start_sample"])
    ia = np.searchsorted(ref["prn_start_sample"], common)
    ib = np.searchsorted(other["prn_start_sample"], common)
    same = sum(ref[i].tobytes() == other[j].tobytes() for i, j in zip(ia, ib, strict=True))
    return {"start_sample": int(other["prn_start_sample"][0]) if other.size else None,
            "records": int(other.size), "common_stamps": int(common.size),
            "bitwise_identical_at_common_stamps": int(same)}  # fmt: skip


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "home").mkdir(exist_ok=True)
    (OUT / "pvt").mkdir(exist_ok=True)
    generate(OUT / "det.bin", 55.0, None, 0.0)
    result: dict[str, list[dict[str, object]]] = {}
    for mode, overrides in (("nonblocking", {}), ("blocking", {"Acquisition_1C.blocking": "true"})):
        recs = []
        for k in range(3):
            r = run_receiver(f"det{mode}{k}", "det.bin", overrides)
            rec = r["rec"]
            assert isinstance(rec, np.ndarray)
            recs.append(rec)
        result[mode] = [compare(recs[0], x) for x in recs]
        print(mode, result[mode], flush=True)
    (OUT / "det.bin").unlink()
    save("s24_gnsssdr_determinism", result)


if __name__ == "__main__":
    main()
