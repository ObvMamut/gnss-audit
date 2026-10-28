"""Loading UGE Jammertest 2024 scenario 1.6.4 for E0(a) and E1(a) (docs/7 section 2.1, registry/intervals.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from gnssaudit.experiments.common import decimate_1hz, in_scope
from gnssaudit.geometry import Ephemerides
from gnssaudit.io.rinex import read_rinex_nav
from gnssaudit.io.ubx import read_ubx_rawx
from gnssaudit.io.uge import Scenario, check_attack_window, mean_position_ecef, read_mon_rf, read_nav_pvt, read_scenario
from gnssaudit.paths import data_root
from gnssaudit.registry import load_intervals

RAW = data_root() / "raw"
SCENARIO_DIR = RAW / "uge-jammertest2024" / "extracted/Jamming/stationary/Very High Power (≥10W)/Bands_L1_L2_L5/1.6.4"
UBX = SCENARIO_DIR / "24DR400_240909_f9p_1-6-4.ubx"
NAV = RAW / "igs-brdc" / "extracted" / "BRDC00IGS_R_20242530000_01D_MN.rnx"


@dataclass
class UgeData:
    obs: pa.Table  # GPS and Galileo, 1 Hz
    eph: Ephemerides
    pvt: pa.Table
    rf: pa.Table
    scenario: Scenario
    position: NDArray[np.float64]  # provisional ECEF (D-30)
    position_spread_m: float
    intervals: dict[str, dict[str, Any]]
    inputs: list[Path] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)


def load(position_interval: str) -> UgeData:
    """Observations, broadcast ephemerides, side files and the provisional position from one clean interval."""
    intervals = load_intervals()
    obs, out_of_scope = in_scope(read_ubx_rawx(UBX, source_id="uge-1.6.4", receiver_id="zed-f9p"))
    obs, decimated = decimate_1hz(obs)
    eph, _ = read_rinex_nav(NAV)
    pvt = read_nav_pvt(SCENARIO_DIR / "nav_pvt.csv")
    rf = read_mon_rf(SCENARIO_DIR / "mon_rf.csv")
    scenario = read_scenario(SCENARIO_DIR / "scenario.json")
    for block in (0, 1):
        check_attack_window(scenario, rf, block)
    iv = intervals[position_interval]
    position, spread = mean_position_ecef(pvt, iv["t_start"], iv["t_end"])
    inputs = [UBX, NAV, SCENARIO_DIR / "nav_pvt.csv", SCENARIO_DIR / "mon_rf.csv", SCENARIO_DIR / "scenario.json"]
    counts = {"out_of_scope_rows": out_of_scope, "decimated_rows": decimated}
    return UgeData(obs, eph, pvt, rf, scenario, position, spread, intervals, inputs, counts)
