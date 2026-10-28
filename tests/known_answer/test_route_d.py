"""Route D on a synthetic static receiver: the truth equals the projected noise increments (docs/3 section 4.5)."""

from dataclasses import replace

import numpy as np
import pyarrow as pa
import pytest

from gnssaudit.geometry import OMEGA_E, WGS84_A, Ephemerides, KeplerEphemeris, geometric_range
from gnssaudit.io.schemas import OBS_EPOCH_SAT, build
from gnssaudit.signals import C_M_S, wavelength_m
from gnssaudit.truth.route_d import tdcp_residuals

WEEK, TOE_SOW = 2330, 345_600.0
T0 = WEEK * 604_800.0 + TOE_SOW
RX = np.array([WGS84_A, 0.0, 0.0])


def constellation() -> Ephemerides:
    base = KeplerEphemeris(
        sys="G", prn=1, toc_s=T0, af0=1e-5, af1=1e-11, af2=0.0, crs=0.0, delta_n=0.0, m0=0.0, cuc=0.0, e=0.01,
        cus=0.0, sqrt_a=5153.7, toe_sow=TOE_SOW, cic=0.0, omega0=OMEGA_E * TOE_SOW, cis=0.0, i0=0.0, crc=0.0,
        omega=0.0, omega_dot=0.0, idot=0.0, week=WEEK, health=0,
    )  # fmt: skip
    geometry = [(0.0, 0.0), (0.0, 0.3), (0.0, -0.3), (0.96, 0.2), (0.96, -0.25), (0.5, 0.1), (0.0, 2.5)]
    return Ephemerides(
        replace(base, prn=k + 1, i0=inc, m0=m0, af0=1e-5 * (k + 1)) for k, (inc, m0) in enumerate(geometry)
    )


def synthetic_obs(
    eph: Ephemerides,
    noise_cyc: np.ndarray,
    lli_at: tuple[int, int] | None = None,
    lli_value: int = 1,
    rate_hz: float = 1.0,
    bias_s: np.ndarray | None = None,
) -> pa.Table:
    """Satellite 7 is below the horizon. noise_cyc has shape (n_epochs, n_sat).

    Physically consistent receiver clock: the time tag is true time + b, the pseudorange and the phase carry c b.
    bias_s defaults to a small random walk (tens of nanoseconds per epoch).
    """
    n_ep, n_sat = noise_cyc.shape
    if bias_s is None:
        bias_s = np.cumsum(np.random.default_rng(9).normal(0.0, 30.0, n_ep)) / C_M_S
    lam = wavelength_m("G", "1C")
    cols: dict[str, list] = {"t": [], "prn": [], "pr": [], "cp": [], "lli": []}
    for k in range(n_ep):
        t_true = T0 + 30.0 + k / rate_hz
        for s in range(n_sat):
            rho, clock = geometric_range(eph.select("G", s + 1, t_true), RX, t_true)
            code = rho + C_M_S * (bias_s[k] - clock)
            cols["t"].append(t_true + bias_s[k])
            cols["prn"].append(s + 1)
            cols["pr"].append(code)
            cols["cp"].append(code / lam + 1e6 + noise_cyc[k, s])
            cols["lli"].append(lli_value if lli_at in ((k, s), (-1, -1)) else 0)
    n = len(cols["t"])
    return build(
        OBS_EPOCH_SAT,
        {
            "source_id": ["synthetic"] * n,
            "receiver_id": ["rx"] * n,
            "t_gps_s": np.array(cols["t"]),
            "sys": ["G"] * n,
            "prn": np.array(cols["prn"]),
            "signal": ["1C"] * n,
            "pr_m": np.array(cols["pr"]),
            "cp_cyc": np.array(cols["cp"]),
            "dop_hz": None,
            "cn0_dbhz": None,
            "lock_time_s": None,
            "lli": np.array(cols["lli"]),
            "trk_stat": None,
            "cp_std_code": None,
            "cp_std_cyc": None,
            "censored": None,
            "pr_std_m": None,
            "elev_deg": None,
            "azim_deg": None,
            "valid": np.ones(n, dtype=bool),
        },  # fmt: skip
        n,
    )


def projected(noise: np.ndarray) -> np.ndarray:
    inc = np.diff(noise[:, :6], axis=0)
    return inc - inc.mean(axis=1, keepdims=True)


def reason_counts(excl: pa.Table) -> dict[str, int]:
    out: dict[str, int] = {}
    for reason, count in zip(excl.column("reason_code").to_pylist(), excl.column("count").to_pylist(), strict=True):
        out[reason] = out.get(reason, 0) + count
    return out


def test_truth_is_the_projected_noise_increment() -> None:
    eph = constellation()
    noise = np.random.default_rng(1).normal(0.0, 0.01, (4, 7))
    truth, excl = tdcp_residuals(synthetic_obs(eph, noise), eph, RX)
    assert truth.num_rows == 3 * 6  # three increments, six visible satellites
    reasons = reason_counts(excl)
    assert reasons == {"below_mask": 3}
    inc = np.diff(noise[:, :6], axis=0)
    expected = inc - inc.mean(axis=1, keepdims=True)
    got = truth.column("e_cyc").to_numpy().reshape(3, 6)
    assert np.allclose(got, expected, atol=1e-6)


def test_loss_of_lock_and_too_few_satellites_are_excluded() -> None:
    eph = constellation()
    truth, excl = tdcp_residuals(synthetic_obs(eph, np.zeros((3, 7)), lli_at=(1, 0)), eph, RX, min_sats=6)
    reasons = reason_counts(excl)
    assert reasons["arc_break"] == 1
    assert reasons["too_few_sats"] == 5  # the first increment keeps only five satellites
    assert truth.num_rows == 6
    assert truth.column("e_cyc").to_numpy() == pytest.approx(0.0, abs=1e-6)


def test_two_bands_share_one_clock_and_keep_their_own_cycles() -> None:
    eph = constellation()
    noise = np.random.default_rng(2).normal(0.0, 0.01, (2, 7))
    l1 = synthetic_obs(eph, noise)
    lam1, lam5 = wavelength_m("G", "1C"), wavelength_m("G", "5Q")
    cp5 = pa.array((l1.column("cp_cyc").to_numpy() - noise.ravel()) * lam1 / lam5 + noise.ravel())
    l5 = l1.set_column(l1.schema.get_field_index("signal"), "signal", pa.array(["5Q"] * l1.num_rows))
    l5 = l5.set_column(l5.schema.get_field_index("cp_cyc"), "cp_cyc", cp5)
    truth, _ = tdcp_residuals(pa.concat_tables([l1, l5]), eph, RX)
    inc_m = np.concatenate([np.diff(noise[:, :6], axis=0)[0] * lam1, np.diff(noise[:, :6], axis=0)[0] * lam5])
    expected_m = inc_m - inc_m.mean()
    got = {(r["prn"], r["signal"]): r["e_cyc"] for r in truth.to_pylist()}
    for k in range(6):
        assert got[(k + 1, "1C")] * lam1 == pytest.approx(expected_m[k], abs=1e-6)
        assert got[(k + 1, "5Q")] * lam5 == pytest.approx(expected_m[6 + k], abs=1e-6)


def test_receiver_clock_offset_and_drift_in_the_time_tags() -> None:
    # u-blox tags carry the receiver clock: 0.5 ms offset drifting 1 ppm (review finding 1)
    eph = constellation()
    noise = np.random.default_rng(3).normal(0.0, 0.01, (4, 7))
    bias = 5e-4 + 1e-6 * np.arange(4)
    truth, _ = tdcp_residuals(synthetic_obs(eph, noise, bias_s=bias), eph, RX)
    assert truth.column("e_cyc").to_numpy().reshape(3, 6) == pytest.approx(projected(noise), abs=1e-5)


def test_higher_rate_log_pairs_by_lookup_and_counts_missing_partners() -> None:
    # 5 Hz log, 1 s increments (review finding 2): every epoch with a partner 1 s later starts an increment
    eph = constellation()
    obs = synthetic_obs(eph, np.zeros((11, 7)), rate_hz=5.0)
    truth, excl = tdcp_residuals(obs, eph, RX)
    assert truth.num_rows == 6 * 6
    assert "no_partner" not in reason_counts(excl)  # the last second has no partner because the data end
    keep = pa.array([not (p == 2 and abs(t - (T0 + 31.0)) < 0.01) for p, t in zip(
        obs.column("prn").to_pylist(), obs.column("t_gps_s").to_pylist(), strict=True)])  # fmt: skip
    truth_gap, excl_gap = tdcp_residuals(obs.filter(keep), eph, RX)
    assert reason_counts(excl_gap)["no_partner"] == 1
    assert truth_gap.num_rows == 6 * 6 - 2  # the epoch lost from G02 ends one increment and starts another


def test_lli_bits_other_than_loss_of_lock_keep_the_arc() -> None:
    # RINEX LLI bit 2 (anti-spoofing / BOC tracking) set on every epoch (review finding 3)
    eph = constellation()
    truth, excl = tdcp_residuals(synthetic_obs(eph, np.zeros((3, 7)), lli_at=(-1, -1), lli_value=4), eph, RX)
    assert truth.num_rows == 2 * 6
    assert "arc_break" not in reason_counts(excl)


def test_half_cycle_flag_makes_the_epoch_unusable() -> None:
    eph = constellation()
    truth, excl = tdcp_residuals(synthetic_obs(eph, np.zeros((3, 7)), lli_at=(1, 0), lli_value=2), eph, RX)
    assert reason_counts(excl)["half_cycle"] == 2  # both increments touching that epoch
    assert truth.num_rows == 2 * 5


def test_min_sats_counts_satellites_not_signals() -> None:
    # three satellites on two bands are six signals but only three satellites (review finding 9)
    eph = constellation()
    l1 = synthetic_obs(eph, np.zeros((2, 7)))
    l1 = l1.filter(pa.array([p <= 3 for p in l1.column("prn").to_pylist()]))
    l5 = l1.set_column(l1.schema.get_field_index("signal"), "signal", pa.array(["5Q"] * l1.num_rows))
    truth, excl = tdcp_residuals(pa.concat_tables([l1, l5]), eph, RX, min_sats=5)
    assert truth.num_rows == 0
    assert reason_counts(excl)["too_few_sats"] == 6
