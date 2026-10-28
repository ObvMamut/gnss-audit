"""Pipeline gates in their tolerance forms (docs/11 section 5; D-33, proposed): identity, null and repeat."""

import numpy as np
import pyarrow as pa
import pytest

from gnssaudit.gates import GateFailure, GateReport, identity_gate, null_gate, repeat_gate, require
from gnssaudit.inject import InjectionReport
from gnssaudit.io.schemas import OBS_EPOCH_SAT, TRACK_STREAM, TRUTH_EPOCH_SAT, build
from gnssaudit.truth.route_d import exclusions_table
from gnssaudit.truth.twin import twin_difference

FS = 4.0e6
SIGMA_D = 0.02  # thermal SD of the discriminator, cycles
STAMPS = np.arange(4_000, int(60 * FS), 4_000, dtype=np.uint64)  # 1 ms records for 60 s
ONSET_S = 45.0


def track(carr_err: np.ndarray, stamps: np.ndarray = STAMPS, prn: int = 7) -> pa.Table:
    n = stamps.size
    none = dict.fromkeys(["abs_ve", "abs_e", "abs_p", "abs_l", "abs_vl", "prompt_i", "prompt_q", "acc_phase_rad_f32",
                          "dop_hz", "dop_rate_hzps", "code_freq_chips", "code_rate_chipsps2", "carr_err_filt",
                          "code_err_chips", "code_err_filt", "cn0_dbhz", "lock_test", "tow_ms", "wn"])  # fmt: skip
    return build(TRACK_STREAM, {
        **none, "run_id": ["r"] * n, "channel": np.zeros(n), "sys": ["G"] * n, "prn": np.full(n, prn),
        "signal": ["1C"] * n, "k": np.arange(n), "sample_count": stamps, "t_rx_s": stamps / FS,
        "acc_phase_rad": np.zeros(n), "carr_err": carr_err, "t_int_s": np.full(n, 1e-3),
    }, n)  # fmt: skip


def noisy(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.normal(0.0, SIGMA_D, n)


@pytest.mark.parametrize(("extra_sd", "passed"), [(0.05, True), (0.15, False)])
def test_identity_gate_tolerates_a_small_pre_onset_difference(extra_sd: float, passed: bool) -> None:
    """F24: twins never match bit for bit, but after warm-up their discriminators agree to a fraction of thermal."""
    rng = np.random.default_rng(1)
    d = noisy(rng, STAMPS.size)
    jam = d + rng.normal(0.0, extra_sd * SIGMA_D, d.size)
    jam[STAMPS >= ONSET_S * FS] += 0.3  # what happens after onset is not the gate's business
    rep = identity_gate(track(d), track(jam), onset_s=ONSET_S, warmup_s=30.0, sigma_th_cyc=SIGMA_D)
    assert rep.gate == "G-identity" and rep.passed is passed
    assert rep.details["rms_over_sigma"][7] == pytest.approx(extra_sd, rel=0.15)


def test_identity_gate_reads_the_onset_on_the_tracks_time_base() -> None:
    """p2a review, important 3: a twin cut from mid-file has t_rx_s = segment start + stamp / fs, so the onset must be
    given on that time base, not recovered from stamps and times (which read a wrong rate there)."""
    rng = np.random.default_rng(3)
    d = noisy(rng, STAMPS.size)
    jam = d.copy()
    jam[STAMPS >= ONSET_S * FS] += 0.3

    def mid_file(x: np.ndarray) -> pa.Table:
        tr = track(x)
        return tr.set_column(tr.schema.get_field_index("t_rx_s"), "t_rx_s", pa.array(100.0 + STAMPS / FS))

    a, b = mid_file(d), mid_file(jam)
    assert identity_gate(a, b, onset_s=100.0 + ONSET_S, warmup_s=130.0, sigma_th_cyc=SIGMA_D).passed
    assert not identity_gate(a, b, onset_s=100.0 + ONSET_S + 5.0, warmup_s=130.0, sigma_th_cyc=SIGMA_D).passed


def test_identity_gate_fails_when_the_twins_integrate_on_different_samples() -> None:
    """Same sync state means the same integration boundaries: shifted stamps leave nothing to compare."""
    d = noisy(np.random.default_rng(2), STAMPS.size)
    rep = identity_gate(track(d), track(d, STAMPS + 1_000), onset_s=ONSET_S, warmup_s=30.0, sigma_th_cyc=SIGMA_D)
    assert not rep.passed and rep.details["common_fraction"][7] < 0.9


NONE = exclusions_table("truth.twin", "null", {})


def null_truth(e_cyc: np.ndarray, slip_count: np.ndarray | None = None) -> pa.Table:
    n = e_cyc.size
    return build(TRUTH_EPOCH_SAT, {
        "truth_id": ["n"] * n, "kind": ["twin"] * n, "sys": ["G"] * n, "prn": np.full(n, 7), "signal": ["1C"] * n,
        "t_rx_s": np.arange(n) * 0.02, "e_cyc": e_cyc,
        "slip_count": np.zeros(n) if slip_count is None else slip_count, "arc_id": np.zeros(n),
        "transient": np.zeros(n, dtype=bool), "valid": np.ones(n, dtype=bool), "regime": np.zeros(n),
    }, n)  # fmt: skip


@pytest.mark.parametrize(("sd", "passed"), [(0.05, True), (0.2, False)])
def test_null_gate_bounds_the_zero_gain_twin_difference(sd: float, passed: bool) -> None:
    e = np.random.default_rng(3).normal(0.0, sd * 0.002, 3000)
    rep = null_gate(null_truth(e), NONE, sigma_th_cyc=0.002)
    assert rep.gate == "G-null" and rep.passed is passed


def test_null_gate_fails_on_any_slip() -> None:
    slips = np.zeros(3000)
    slips[2000:] = 1
    assert not null_gate(null_truth(np.zeros(3000), slips), NONE, sigma_th_cyc=0.002).passed


def report(sha: str) -> InjectionReport:
    return InjectionReport(1e-4, {}, 0, sha, "f", 10, (), {})


def test_repeat_gate_needs_identical_injector_bytes() -> None:
    d = noisy(np.random.default_rng(4), STAMPS.size)
    same = repeat_gate([report("a"), report("a")], [track(d), track(d)], warmup_s=30.0, sigma_th_cyc=SIGMA_D)
    assert same.passed
    differ = repeat_gate([report("a"), report("b")], [track(d), track(d)], warmup_s=30.0, sigma_th_cyc=SIGMA_D)
    assert not differ.passed and differ.details["injector_identical"] is False


def test_repeat_gate_judges_receiver_runs_at_the_identity_tolerance() -> None:
    rng = np.random.default_rng(5)
    d = noisy(rng, STAMPS.size)
    far = d + rng.normal(0.0, 0.3 * SIGMA_D, d.size)
    rep = repeat_gate([report("a"), report("a")], [track(d), track(far)], warmup_s=30.0, sigma_th_cyc=SIGMA_D)
    assert not rep.passed and rep.details["injector_identical"] is True


def test_require_raises_on_a_failed_gate() -> None:
    ok = GateReport("G-x", True, {})
    assert require(ok) is ok
    with pytest.raises(GateFailure, match="G-x") as err:
        require(GateReport("G-x", False, {"why": "test"}))
    assert err.value.report.details == {"why": "test"}


def null_pair(t: np.ndarray, jammed_extra: np.ndarray) -> tuple[pa.Table, pa.Table]:
    """Twin truth and exclusions of two zero-gain twins whose phases differ by jammed_extra (cycles)."""
    n = t.size
    cp = 1e5 + 1000.0 * t

    def obs(c: np.ndarray, rx: str) -> pa.Table:
        return build(OBS_EPOCH_SAT, {
            "source_id": ["s"] * n, "receiver_id": [rx] * n, "t_gps_s": t, "sys": ["G"] * n, "prn": [7] * n,
            "signal": ["1C"] * n, "pr_m": None, "cp_cyc": c, "dop_hz": np.full(n, 1000.0), "cn0_dbhz": None,
            "lock_time_s": None, "lli": None, "trk_stat": None, "cp_std_code": None, "cp_std_cyc": None,
            "censored": None, "pr_std_m": None, "elev_deg": None, "azim_deg": None,
            "valid": np.ones(n, bool)}, n)  # fmt: skip

    truth, _, excl = twin_difference(obs(cp, "a"), obs(cp + jammed_extra, "b"), onset_s=float(t[-1]) + 1.0,
                                     pre_window_s=float(t[-1] - t[0]), warmup_s=float(t[0]), slip_size_cyc=0.5,
                                     tau_loop_s=0.03, truth_id="null")  # fmt: skip
    return truth, excl


def test_null_gate_fails_when_a_step_cut_the_pair_short() -> None:
    """Review finding (p2a, critical 1): a 0.37-cycle step, not a slip, moves the rest of the series to exclusions;
    the kept part is clean, but the engine defect G-null exists for must fail it."""
    t = np.round(np.arange(30.0, 90.0, 0.02), 6)
    truth, excl = null_pair(t, np.where(t >= 60.0, 0.37, 0.0))
    rep = null_gate(truth, excl, sigma_th_cyc=0.002)
    assert not rep.passed and rep.details["breaking_exclusions"] > 0
    clean_truth, clean_excl = null_pair(t, np.zeros(t.size))
    assert null_gate(clean_truth, clean_excl, sigma_th_cyc=0.002).passed


def test_null_gate_fails_when_too_little_of_the_pair_is_kept() -> None:
    truth = null_truth(np.zeros(1000))
    lost = exclusions_table("truth.twin", "null", {("one_twin_epoch", "G", 7, "1C"): [40.0] * 20})  # 2 % lost
    rep = null_gate(truth, lost, sigma_th_cyc=0.002)
    assert not rep.passed and rep.details["kept_share"] == pytest.approx(1000 / 1020)
