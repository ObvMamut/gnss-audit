"""GFZ Kp series and the geomagnetic screen of docs/7 section 4.3."""

import json
from pathlib import Path

import pytest

from gnssaudit.io.kp import KP_THRESHOLD, kp_max, read_kp_json
from gnssaudit.io.rinex import gps_seconds


def _kp(tmp_path: Path) -> Path:
    path = tmp_path / "kp.json"
    times = [f"2024-09-09T{h:02d}:00:00Z" for h in range(0, 24, 3)]
    path.write_text(json.dumps({"Kp": [2.3, 1.3, 2.3, 1.3, 1.0, 4.7, 2.7, 2.3], "datetime": times}))
    return path


def test_bins_start_in_gps_time(tmp_path: Path) -> None:
    series = read_kp_json(_kp(tmp_path))
    assert series[0] == (gps_seconds(2024, 9, 9, 0, 0, 18), 2.3)
    assert len(series) == 8


def test_kp_max_takes_every_overlapping_bin(tmp_path: Path) -> None:
    series = read_kp_json(_kp(tmp_path))
    inside_12_15 = (gps_seconds(2024, 9, 9, 14, 0, 18), gps_seconds(2024, 9, 9, 14, 40, 18))
    assert kp_max(series, *inside_12_15) == 1.0
    straddling = (gps_seconds(2024, 9, 9, 14, 50, 18), gps_seconds(2024, 9, 9, 15, 10, 18))
    assert kp_max(series, *straddling) == 4.7
    assert KP_THRESHOLD == 4.0


def test_interval_outside_the_series_raises(tmp_path: Path) -> None:
    series = read_kp_json(_kp(tmp_path))
    with pytest.raises(ValueError, match="Kp"):
        kp_max(series, gps_seconds(2024, 9, 10, 1, 0, 0), gps_seconds(2024, 9, 10, 2, 0, 0))
