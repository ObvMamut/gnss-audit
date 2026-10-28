"""The patched GNSS-SDR build (D-08, D-27; containers/Dockerfile.patched) against the stock one, on synthetic
GPS L1 C/A recordings with exactly known phase (docs/9 section 1; G-phase-precision, docs/3 section 2.6)."""

from pathlib import Path

import numpy as np
import pytest

from gnssaudit.gates import lockin, stamp_and_verdict
from gnssaudit.io.gnsssdr import PATCHED_112, read_tracking_dump
from gnssaudit.sdr import ReceiverSpec, SourceSpec, loop_overrides, render_config, run_receiver
from gnssaudit.sdr.images import PATCH_ID, PATCHED, STOCK
from gnssaudit.sim.recording import write_ca_recording
from gnssaudit.uq.baselines import b3_sigma_rad
from gnssaudit.uq.loop import CarrierLoop

pytestmark = pytest.mark.container
FS = 4.0e6
DOPPLER = 1000.0
CN0 = 55.0
PRN = 3


@pytest.fixture(scope="module")
def recording(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("in") / "ca_1khz.bin"
    write_ca_recording(path, prn=PRN, doppler_hz=DOPPLER, cn0_dbhz=CN0, seconds=100.0, seed=5)
    return path


def run(
    image: str, patch_id: str | None, rec: Path, workdir: Path, overrides: dict[str, str] | None = None, prn: int = PRN
) -> Path:
    spec = ReceiverSpec(
        image=image, source_commit="3fcb502a", patch_id=patch_id, template="1C", overrides=overrides or {}
    )
    src = SourceSpec(path=rec, item_type="ibyte", fs_hz=FS)
    art = run_receiver(spec, render_config(spec, src, {0: ("G", "1C", prn)}), src, workdir, timeout_s=900)
    return art.tracking_dumps[0]


def residual_rad(phase_rad: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Reported phase minus the known carrier phase, after removing a constant and a linear trend (the ambiguity and
    a tiny frequency offset). GNSS-SDR accumulates with a minus sign (docs/3 section 3.2)."""
    e = -phase_rad - 2 * np.pi * DOPPLER * t
    return e - np.polyval(np.polyfit(t, e, 1), t)


@pytest.fixture(scope="module")
def dumps(recording: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("runs")
    return {
        "stock": run(STOCK, None, recording, base / "stock"),
        "patched": run(PATCHED, PATCH_ID, recording, base / "patched"),
    }


def table(path: Path, layout: str):  # noqa: ANN201
    return read_tracking_dump(path, layout, run_id=layout, channel=0, sys="G", signal="1C", fs_hz=FS, t_int_s=1e-3)  # type: ignore[arg-type]


def test_patched_records_are_112_bytes_and_self_consistent(dumps: dict[str, Path]) -> None:
    assert dumps["patched"].stat().st_size % 112 == 0
    t = table(dumps["patched"], "patched112")  # raises on a layout mismatch (float64 sample count against stamp)
    assert t.num_rows > 50_000 and t.column("acc_phase_rad").null_count == 0


def test_one_record_per_integration_after_bit_sync(dumps: dict[str, Path]) -> None:
    """F23 / D-27: stock dumps once per 20 ms data symbol in narrow tracking; the patched build every 1 ms."""
    for name, layout, step in (("stock", "stock108", 80_000), ("patched", "patched112", 4_000)):
        t = table(dumps[name], layout)
        late = t.column("t_rx_s").to_numpy() > 30.0
        d = np.diff(t.column("sample_count").to_numpy().astype(np.int64)[late])
        assert np.median(d) == step, name


def test_float64_phase_reaches_thermal_noise_where_float32_cannot(dumps: dict[str, Path]) -> None:
    """Beyond 3.8e5 rad of accumulated phase (about 60 s of tracking at 1 kHz) float32 steps are 0.031 rad or more."""
    stock = table(dumps["stock"], "stock108")
    patched = table(dumps["patched"], "patched112")
    sigma_th = float(b3_sigma_rad(np.array([CN0]), CarrierLoop.gnsssdr(35.0, 1e-3).bn_eff_hz(), 1e-3)[0])
    rms = {}
    for name, t, col in (("stock", stock, "acc_phase_rad_f32"), ("patched", patched, "acc_phase_rad")):
        tt = t.column("t_rx_s").to_numpy()
        phase = t.column(col).to_numpy().astype(np.float64)
        sel = np.abs(phase) > 3.8e5
        assert sel.sum() > 500
        rms[name] = float(np.std(residual_rad(phase[sel], tt[sel])))
    acc64 = patched.column("acc_phase_rad").to_numpy()
    acc64 = acc64[np.abs(acc64) > 3.8e5]
    assert np.max(np.abs(acc64 - acc64.astype(np.float32).astype(np.float64))) > 0.005  # finer than the float32 grid
    assert rms["patched"] == pytest.approx(sigma_th, rel=0.25)
    assert rms["stock"] > 1.2 * rms["patched"]


def test_extended_integration_keeps_one_record_per_loop_update(recording: Path, tmp_path: Path) -> None:
    """At T = 20 ms (extended) the loop updates every 20 ms: the patched dump must not add records between updates."""
    trk = run(PATCHED, PATCH_ID, recording, tmp_path / "ext", loop_overrides("1C", 5.0, 20e-3))
    t = table(trk, "patched112")
    late = t.column("t_rx_s").to_numpy() > 30.0
    d = np.diff(t.column("sample_count").to_numpy().astype(np.int64)[late])
    assert np.median(d) == 80_000 and np.all(d >= 79_000)


def test_patched_build_keeps_the_identified_loop(tmp_path: Path) -> None:
    """G-hz (docs/11) on the s17 tone recording: the patches change the dump, not the loop, so the default cell's
    reported-phase response still matches the source-timing model within +-0.3 dB and +-5 degrees."""
    from review import s17_gnsssdr_identification as s17

    sig = s17.WORK / "sig.bin"
    if not sig.exists():
        pytest.skip(f"{sig} not present (run review.s17_gnsssdr_identification generate)")
    rec = np.fromfile(run(PATCHED, PATCH_ID, sig, tmp_path / "ghz", prn=s17.PRN), dtype=PATCHED_112)
    tt = rec["prn_start_sample"].astype(np.float64) / FS
    steady = tt > tt[0] + 15.0
    r, t = rec[steady], tt[steady]
    tone_in = -1j * s17.TONE_AMP_RAD * np.exp(1j * s17.tone_phases())
    acc_res = -r["acc_phase_rad"] - 2 * np.pi * s17.DOPPLER * t
    lp = lockin(acc_res - np.polyval(np.polyfit(t, acc_res, 1), t), t, s17.TONES_HZ)
    f = np.array(s17.TONES_HZ)
    model = CarrierLoop.gnsssdr(35.0, 1e-3, kappa=0.5).freq_response(2 * np.pi * f * 1e-3)
    _, _, rows, passed = stamp_and_verdict(
        lp[:-1] / tone_in, model, 1e-3, f, float(lp[-1].real), r.size, tone_amp_rad=s17.TONE_AMP_RAD
    )
    assert passed, rows
    assert sum(row["status"] == "pass" for row in rows) >= 8
