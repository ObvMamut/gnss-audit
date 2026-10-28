"""G-hz verdict (gnssaudit.gates, first used by s17): +-0.3 dB and +-5 deg at every tone with response SNR >= 20 dB."""

import numpy as np

from gnssaudit.gates import ghz_verdict, lockin, stamp_and_verdict
from gnssaudit.uq.loop import CarrierLoop
from review.s17_gnsssdr_identification import TONE_AMP_RAD, TONES_HZ

F = np.array([0.3, 1.5, 5.0, 12.0, 31.0])
MODEL = np.array([0.99 - 0.02j, 0.95 - 0.2j, 0.7 - 0.5j, 0.3 - 0.4j, 0.05 - 0.1j])
STAMP_S = 0.0005
DELAYED = MODEL * np.exp(2j * np.pi * F * STAMP_S)  # what the verdict compares against
AMP_SE = 1e-4  # response SNR of a unit response is 20 log10(0.04 / 1e-4) = 52 dB


def test_an_exact_response_passes() -> None:
    rows, passed = ghz_verdict(DELAYED, MODEL, AMP_SE, STAMP_S, F, tone_amp_rad=TONE_AMP_RAD)
    assert passed
    assert [r["status"] for r in rows] == ["pass"] * len(F)
    assert all(abs(r["err_db"]) < 1e-9 and abs(r["err_deg"]) < 1e-9 for r in rows)


def test_plus_0p4_db_fails() -> None:
    meas = DELAYED.copy()
    meas[2] *= 10 ** (0.4 / 20)
    rows, passed = ghz_verdict(meas, MODEL, AMP_SE, STAMP_S, F, tone_amp_rad=TONE_AMP_RAD)
    assert not passed
    assert [r["status"] for r in rows] == ["pass", "pass", "fail", "pass", "pass"]


def test_plus_6_deg_fails() -> None:
    meas = DELAYED * np.exp(1j * np.radians(6.0))
    rows, passed = ghz_verdict(meas, MODEL, AMP_SE, STAMP_S, F, tone_amp_rad=TONE_AMP_RAD)
    assert not passed
    assert all(r["status"] == "fail" for r in rows)


def test_inside_the_limits_passes() -> None:
    meas = DELAYED * 10 ** (0.25 / 20) * np.exp(1j * np.radians(4.5))
    assert ghz_verdict(meas, MODEL, AMP_SE, STAMP_S, F, tone_amp_rad=TONE_AMP_RAD)[1]


def test_a_low_snr_tone_is_excluded_not_failed() -> None:
    meas = DELAYED.copy()
    meas[4] *= 0.5  # 6 dB wrong, but at amp_se = 1e-3 its SNR is 20 log10(0.0559 * 0.04 / 1e-3) = 7 dB
    rows, passed = ghz_verdict(meas, MODEL, 1e-3, STAMP_S, F, tone_amp_rad=TONE_AMP_RAD)
    assert rows[4]["status"] == "excluded_low_snr"
    assert rows[4]["snr_db"] < 20.0
    assert [r["status"] for r in rows[:4]] == ["pass"] * 4
    assert passed


def test_no_judged_tone_is_not_a_pass() -> None:
    rows, passed = ghz_verdict(DELAYED, MODEL, 1.0, STAMP_S, F, tone_amp_rad=TONE_AMP_RAD)
    assert all(r["status"] == "excluded_low_snr" for r in rows)
    assert not passed


def test_tones_at_or_above_the_loop_nyquist_are_listed_not_judged() -> None:
    f = np.array([5.0, 31.0, 43.0])
    model = np.array([0.9, 0.2, 0.1], dtype=complex)
    meas = model.copy()
    meas[1:] *= 3.0  # grossly wrong, but beyond 1 / (2 * 20 ms) = 25 Hz
    rows, passed = ghz_verdict(meas, model, 1e-5, 0.0, f, t_s=20e-3, tone_amp_rad=TONE_AMP_RAD)
    assert [r["status"] for r in rows] == ["pass", "above_nyquist", "above_nyquist"]
    assert passed


def _synthetic(gain: float, sd: float = 0.02) -> tuple[bool, float, list[str]]:
    """Dump-like series at T = 1 ms: known tones through the source-timing model, stamped 0.5 T late."""
    t_s, offset = 1e-3, 0.5e-3
    loop = CarrierLoop.gnsssdr(15.0, t_s, kappa=0.5)
    f = np.array(TONES_HZ)
    psi = np.random.default_rng(1).uniform(0, 2 * np.pi, f.size)
    tone_in = -1j * TONE_AMP_RAD * np.exp(1j * psi)
    resp = loop.freq_response(2 * np.pi * f * t_s)
    h = gain * resp * np.sinc(f * t_s) * np.exp(2j * np.pi * f * offset)
    t = np.arange(30_000) * t_s
    y = sum(np.real(c * hk * np.exp(2j * np.pi * fk * t)) for c, hk, fk in zip(tone_in, h, f, strict=True))
    y = y + sd * np.random.default_rng(2).standard_normal(t.size)
    lp = lockin(y, t, TONES_HZ)
    off_s, _, rows, passed = stamp_and_verdict(
        lp[:-1] / tone_in, resp, t_s, f, float(lp[-1].real), t.size, tone_amp_rad=TONE_AMP_RAD
    )
    return passed, off_s, [r["status"] for r in rows]


def test_the_analysis_chain_passes_a_known_response_and_recovers_the_stamp() -> None:
    passed, off_s, status = _synthetic(1.0)
    assert abs(off_s - 0.5e-3) < 0.05e-3
    assert passed
    assert "pass" in status and "fail" not in status


def test_the_analysis_chain_fails_a_1_db_gain_error() -> None:
    passed, _, status = _synthetic(10 ** (1.0 / 20))
    assert not passed
    assert "fail" in status
