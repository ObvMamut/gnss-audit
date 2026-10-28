"""E1(a) helpers: jammer power from the ramp, 10 dB classes, ENU position shifts, B4's uncensored reading."""

import numpy as np
import pyarrow as pa
import pytest

from gnssaudit.experiments.e1a import enu_shift, power_class, ramp_power_dbm, uncensored_only
from gnssaudit.geometry import ecef_to_geodetic, geodetic_to_ecef
from gnssaudit.io.schemas import PREDICTION, build
from gnssaudit.uq.baselines import Flag

RAMP = {"p0_dbm": -37.0, "step_db": 2.0, "levels": 43, "hold_s": 10.0, "class_width_db": 10.0}


def test_ramp_goes_up_then_down_with_10_s_holds() -> None:
    """2024 catalogue, group 6: ramped up and down in 2 dB steps, 10 s per level (43 up, 42 down: 850 s)."""
    t_on = 1000.0
    t = t_on + np.array([0.0, 9.9, 10.0, 420.0, 429.9, 430.0, 849.9])
    assert ramp_power_dbm(t, t_on, RAMP).tolist() == [-37.0, -37.0, -35.0, 47.0, 47.0, 45.0, -37.0]


def test_ramp_is_undefined_outside_its_850_s() -> None:
    p = ramp_power_dbm(np.array([999.0, 1850.0]), 1000.0, RAMP)
    assert np.isnan(p).all()


def test_power_class_labels_are_10_db_wide_from_the_ramp_start() -> None:
    assert power_class(np.array([-37.0, -28.0, -27.0, 47.0]), RAMP) == [
        "[-37,-27) dBm", "[-37,-27) dBm", "[-27,-17) dBm", "[43,53) dBm",
    ]  # fmt: skip


def test_enu_shift_moves_the_point_by_the_requested_metres() -> None:
    lat, lon = np.radians(69.2753), np.radians(15.9681)
    x = geodetic_to_ecef(lat, lon, 40.0)
    up = enu_shift(x, 0.0, 0.0, 1.0)
    assert np.linalg.norm(up - x) == pytest.approx(1.0)
    assert ecef_to_geodetic(up)[2] == pytest.approx(41.0, abs=1e-6)
    north = enu_shift(x, 0.0, 1.0, 0.0)
    assert ecef_to_geodetic(north)[0] > lat and ecef_to_geodetic(north)[2] == pytest.approx(40.0, abs=1e-6)


def test_uncensored_only_invalidates_censored_rows() -> None:
    n = 3
    pred = build(
        PREDICTION,
        {
            "model_id": ["B4"] * n, "model_version": ["1"] * n, "run_id": ["r"] * n, "sys": ["G"] * n,
            "prn": [1, 2, 3], "signal": ["1C"] * n, "t_rx_s": [0.0] * n, "scale_cyc": [0.008, 0.056, np.nan],
            "nu": np.zeros(n), "p_slip": None, "valid": [True, True, False],
            "flags": [0, int(Flag.CENSORED | Flag.SATURATION), int(Flag.MISSING_INPUT)],
        },
        n,
    )  # fmt: skip
    out = uncensored_only(pred)
    assert out.column("valid").to_pylist() == [True, False, False]
    assert out.column("model_id")[0].as_py() == "B4-uncensored-only"
    assert isinstance(out, pa.Table)


def test_b6_as_configured_never_sees_the_increment_it_is_scored_on() -> None:
    """Route D scores the increment [t, t + 1] with B6's sigma at both t and t + 1; neither may include it."""
    from pathlib import Path

    import yaml

    from gnssaudit.uq.baselines import b6_sigma

    spec = yaml.safe_load((Path(__file__).resolve().parents[2] / "experiments" / "E1a.yaml").read_text())
    b6 = spec["models"]["B6"]
    t_innov = np.arange(100, dtype=float)
    innov = np.full(100, 0.01)
    innov[60] = 10.0  # the increment [60, 61]
    sigma = b6_sigma(np.array([60.0, 61.0]), t_innov, innov, b6["window_s"], b6["lag_s"], b6["min_count"])
    assert sigma == pytest.approx([0.01, 0.01])
