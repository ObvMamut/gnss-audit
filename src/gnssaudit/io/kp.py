"""GFZ Kp index [59] and the geomagnetic screen (docs/7 section 4.3)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from gnssaudit.io.rinex import gps_seconds

KP_THRESHOLD = 4.0  # 3-hour Kp above this flags an interval (Judgment, docs/7 section 4.3)
BIN_S = 10_800.0
GPS_UTC_S = 18.0  # leap seconds from 2017-01-01


def read_kp_json(path: Path) -> list[tuple[float, float]]:
    """(start of the 3-hour bin in GPS seconds, Kp) from the kp.gfz.de JSON service."""
    meta = json.loads(path.read_text())
    out = []
    for stamp, kp in zip(meta["datetime"], meta["Kp"], strict=True):
        d = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        out.append((gps_seconds(d.year, d.month, d.day, d.hour, d.minute, d.second + GPS_UTC_S), float(kp)))
    return out


def kp_max(series: list[tuple[float, float]], t0: float, t1: float) -> float:
    """Largest Kp over the 3-hour bins that overlap [t0, t1]."""
    hits = [kp for start, kp in series if start < t1 and start + BIN_S > t0]
    if not hits:
        raise ValueError("no Kp bin overlaps the interval")
    return max(hits)
