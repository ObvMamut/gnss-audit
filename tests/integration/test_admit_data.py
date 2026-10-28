"""Admit end to end (P2b task 5) on GNSS-SDR's CTTC sample: the screen, the C/A scan and the survey through the
patched image write a manifest with a decision, its reasons, the floor and a proposed interval row."""

import json
from pathlib import Path

import pytest

from gnssaudit.cli.main import main
from gnssaudit.registry import dataset_file

pytestmark = [pytest.mark.container, pytest.mark.data, pytest.mark.slow]
CTTC = "2013_04_04_GNSS_SIGNAL_at_CTTC_SPAIN.dat"


def test_admit_screens_the_cttc_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        path = dataset_file("cttc-2013-sample", CTTC)
        path.stat()
    except FileNotFoundError:
        pytest.skip("the CTTC recording is not fetched")
    from gnssaudit.paths import data_root

    spec = {
        "id": "Admit",
        "dataset": "cttc-2013-sample",
        "local_path": str(path.relative_to(data_root())),
        "item_type": "ishort",
        "fs_hz": 4.0e6,
        "frontend_bw_hz": 2.0e6,
        "template": "1C",
        "decim": 1,
        "fir_taps": 1,
        "max_channels": 8,
        "band": "L1 (GPS L1 C/A)",
        "stock_receiver": True,  # the survey is no twin (P2b task 6)
    }
    spec_path = tmp_path / "admit.yaml"
    spec_path.write_text(json.dumps(spec))  # JSON is YAML
    monkeypatch.setenv("GNSSAUDIT_RESULTS", str(tmp_path / "results"))
    assert main(["run", str(spec_path)]) == 0
    (run_dir,) = (tmp_path / "results" / "Admit").iterdir()
    s = json.loads((run_dir / "manifest.json").read_text())["summary"]
    assert s["verified"] and s["duration_s"] == pytest.approx(100.0)
    # satellites are measured from the warm-up to the first onset sign (d2cddff; docs/7 section 4.2). CTTC's first
    # sign, a second peak at 6 s, falls inside the 30 s warm-up (few_sats also starts at 30 s), so that span is
    # empty: no satellite is measured and nothing is admitted. The PVT check still reads every fix.
    a = s["admission"]
    assert a["onset_s"] <= s["rules"]["warmup_s"] and a["sats"] == {} and a["window_s"] is None
    assert a["pvt"]["n"] > 100
    assert s["line_excess_db"] > 1.0  # the sample's floor carries a strong line (docs/8 section 2.1)
    row = s["interval_row"]
    assert row["dataset_id"] == "cttc-2013-sample" and row["state"] in ("clean", "excluded")
    assert set(s["receivers"]["logs"]) == {"survey"} and s["receivers"]["overrides"] == {}
