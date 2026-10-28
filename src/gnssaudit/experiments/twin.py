"""Twin-run machinery shared by the Route B experiments (E0b, E2; docs/3 section 2; docs/6 section 4.2).

A twin is one injection of a registered recording, received by the patched GNSS-SDR:
- `inject_twin` writes its int16 stream to /dev/shm (D-10);
- `receive` runs the receiver on it (with the D-34 warm-up lock check, re-running on the same bytes) and reads its
  dumps, with every track put on recording time through the twin's clock (`truth.rxtime.TwinClock`: the twin's start
  in the recording, its decimation and the filter delay; p2a review, important 3);
- `run_twin` does both and deletes the stream.
A twin's bytes can be received under several receiver configurations (the loop cells of E0's phase-tone
identification) before they are deleted.
"""

from __future__ import annotations

import gzip
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from gnssaudit.eval.dependence import tau_int_s
from gnssaudit.gates import GateReport
from gnssaudit.inject import Disturbance, FrontEndSpec, InjectionReport, IQSource, SinkSpec, inject
from gnssaudit.io.gnsssdr import Channel, read_observables_dump, read_tracking_dump
from gnssaudit.io.schemas import OBS_EPOCH_SAT, TRACK_STREAM, build, validate
from gnssaudit.sdr import ReceiverSpec, RunArtifacts, SourceSpec, render_config, run_receiver
from gnssaudit.sdr.images import PATCH_ID, PATCHED
from gnssaudit.truth.rxtime import TwinClock, retime
from gnssaudit.uq.baselines import b3_sigma_rad

SHM_ROOT = Path("/dev/shm/gnssaudit")
SHM_MARGIN = 2 * 10**9  # bytes kept free in /dev/shm beyond the files in flight
# A false lock at start-up (code tracked, carrier not) reads a lock statistic near 0 for good once D-34 stops the rule
# from dropping it; real tracking reads 0.36 even at 30 dB-Hz (F22). The stock rule would drop and re-acquire such a
# channel; the pipeline does the same, before the analysis starts, by receiving the same twin again (D-34 amendment).
WARMUP_LOCK_WINDOW_S = 5.0  # the last seconds of the warm-up the check reads
WARMUP_LOCK_MIN = 0.2  # median smoothed lock statistic a channel must reach there
WARMUP_DOPPLER_TOL_HZ = 150.0  # from the PRN's expected Doppler; true Doppler moves under 1 Hz/s, a false lock 500 Hz
RX_ATTEMPTS = 3
TWIN_LOCK_DEVIATION = {  # D-34 (and its amendment of 15 Oct 2026): every twin run deviates from the defaults by
    # raising the carrier-lock limit (1000 s of failures at 1 ms), so the rule never drops a channel, and by pulling in
    # with the FLL (`enable_fll_pull_in`, the pull-in's first seconds only), so a start-up frequency false lock onto a
    # cross-correlation line is not kept for good; the steady-state loop is unchanged
    "1C": {"Tracking_1C.max_carrier_lock_fail": "1000000", "Tracking_1C.enable_fll_pull_in": "true"},
    "L5": {"Tracking_L5.max_carrier_lock_fail": "1000000", "Tracking_L5.enable_fll_pull_in": "true"},
}


# ------------------------------------------------------------------------------------------- noise models
def sigma_reported_cyc(cn0_dbhz: float, *, bn_eff_hz: float, t_s: float) -> float:
    """Thermal SD of the reported phase, B3 on B_n,eff (docs/4 section 2), in cycles."""
    return float(b3_sigma_rad(np.array([cn0_dbhz]), bn_eff_hz, t_s)[0]) / (2 * np.pi)


def sigma_disc_cyc(cn0_dbhz: float, *, t_s: float) -> float:
    """Thermal SD of one integration's phase discriminator, 1 / sqrt(2 C/N0 T) rad, in cycles."""
    return 1.0 / np.sqrt(2 * 10 ** (cn0_dbhz / 10) * t_s) / (2 * np.pi)


def ghost_sanity(e_cyc: NDArray[np.float64], *, sigma_cyc: float, dt_s: float, k_se: float = 3.0) -> dict[str, Any]:
    """docs/3 section 7: the clean twin's ghost error SD agrees with the thermal model "within its sampling error".
    The SE of the SD ratio is 1 / sqrt(2 n_eff), n_eff = N dt / tau_int(e^2) (the variance estimator of a correlated
    Gaussian series); the check passes within k_se SEs."""
    e = np.asarray(e_cyc, dtype=np.float64)
    ratio = float(np.std(e) / sigma_cyc)
    n_eff = e.size * dt_s / tau_int_s((e - e.mean()) ** 2, dt_s)
    se = float(1.0 / np.sqrt(2.0 * n_eff))
    return {"sd_over_b3": ratio, "se": se, "n_eff": float(n_eff), "pass": abs(ratio - 1.0) <= k_se * se}


# ----------------------------------------------------------------------------------------------- records
def keep_logs(work: Path, run_dir: Path) -> dict[str, str]:
    """Every receiver run's GNSS-SDR log, gzipped into run_dir/logs before the work directory and its dumps are
    deleted (D-18); returns {run name: path relative to run_dir}."""
    kept = {}
    for log in sorted(work.glob("*/gnss-sdr.log")):
        (run_dir / "logs").mkdir(exist_ok=True)
        rel = f"logs/{log.parent.name}.log.gz"
        with log.open("rb") as src, gzip.open(run_dir / rel, "wb") as dst:
            shutil.copyfileobj(src, dst)
        kept[log.parent.name] = rel
    return kept


def track_as_obs(track: pa.Table, *, source_id: str, receiver_id: str) -> pa.Table:
    """A patched tracking dump as obs_epoch_sat rows: the reported phase -acc / 2 pi (cycles) at each stamp's time.
    Lets twins without observables (a ghost has no telemetry) be differenced at every integration."""
    track = validate(track, TRACK_STREAM)
    n = track.num_rows
    nulls = dict.fromkeys(["pr_m", "lock_time_s", "lli", "trk_stat", "cp_std_code", "cp_std_cyc", "censored",
                           "pr_std_m", "elev_deg", "azim_deg"])  # fmt: skip
    return build(
        OBS_EPOCH_SAT,
        {
            **nulls,
            "source_id": [source_id] * n,
            "receiver_id": [receiver_id] * n,
            "t_gps_s": track.column("t_rx_s"),
            "sys": track.column("sys"),
            "prn": track.column("prn"),
            "signal": track.column("signal"),
            "cp_cyc": -track.column("acc_phase_rad").to_numpy() / (2 * np.pi),
            "dop_hz": track.column("dop_hz"),
            "cn0_dbhz": track.column("cn0_dbhz"),
            "valid": np.ones(n, dtype=bool),
        },
        n,
    )


def provenance(rx: ReceiverSpec, runs: Mapping[str, RunArtifacts]) -> dict[str, Any]:
    """What every receiver run records: the image, the patch, the overrides, and per run the configuration hash,
    exit and wall time."""
    return {
        "image": rx.image,
        "source_commit": rx.source_commit,
        "patch_id": rx.patch_id,
        "template": rx.template,
        "overrides": dict(rx.overrides),
        "runs": {
            name: {"config_sha256": a.config_sha256, "exit_code": a.exit_code, "wall_s": round(a.wall_s, 2)}
            for name, a in runs.items()
        },
    }


def jsonable(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return str(x)
    return x


def write_table(run_dir: Path, name: str, table: pa.Table) -> None:
    pq.write_table(table, run_dir / f"{name}.parquet", compression="zstd")


def concat_tracks(tracks: Mapping[int, pa.Table]) -> pa.Table:
    return pa.concat_tables([tracks[ch] for ch in sorted(tracks)])


def median_of(table: pa.Table, col: str, t0: float = -np.inf, t1: float = np.inf, time_col: str = "t_rx_s") -> float:
    t = table.column(time_col).to_numpy()
    v = table.column(col).to_numpy(zero_copy_only=False).astype(float)[(t >= t0) & (t < t1)]
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


# --------------------------------------------------------------------------------------------- receivers
@dataclass(frozen=True)
class LockPolicy:
    """What a twin needs at the end of the warm-up (D-34 amendment, P2b task 10): every channel in must_lock (the
    ghosts) locked, and at least min_real other assigned channels. A twin's channels start on identical bytes, so a
    bad start repeats in every run of it; a channel still unlocked after the retries is excluded and counted."""

    must_lock: frozenset[int]
    min_real: int

    def met(self, channels: Mapping[int, Channel], unlocked: Sequence[int]) -> bool:
        bad = set(unlocked)
        real = [ch for ch in channels if channels[ch][2] > 0 and ch not in self.must_lock]
        return not (bad & self.must_lock) and sum(ch not in bad for ch in real) >= self.min_real


@dataclass
class Twin:
    name: str
    report: InjectionReport | None
    art: RunArtifacts
    tracks: dict[int, pa.Table]  # by channel, t_rx_s on recording time
    obs: pa.Table | None
    attempts: int = 1  # receiver runs on the same injected bytes (warm-up lock check)
    unlocked: list[int] = field(default_factory=list[int])  # channels still not locked after the last attempt
    failed: list[RunArtifacts] = field(default_factory=list[RunArtifacts])  # earlier attempts
    lock_ok: bool = True  # the lock policy is met (strict without one: no channel unlocked)


def unlocked_channels(
    tracks: Mapping[int, pa.Table],
    channels: Mapping[int, Channel],
    warmup_s: float,
    *,
    expected_doppler_hz: Mapping[int, float] | None = None,
) -> list[int]:
    """Assigned channels (PRN > 0) whose median smoothed lock statistic over the last WARMUP_LOCK_WINDOW_S before
    warmup_s (recording time) is below WARMUP_LOCK_MIN, that hold no record there, or whose median Doppler there is
    more than WARMUP_DOPPLER_TOL_HZ from the PRN's expected Doppler (a survey's, or a ghost's set one): a Costas loop
    at T = 1 ms also locks 1/(2T) off, where its lock statistic reads high (P2b task 10)."""
    bad = []
    for ch in sorted(channels):
        if channels[ch][2] == 0:
            continue
        tr = tracks.get(ch)
        if tr is None:
            bad.append(ch)
            continue
        t = tr.column("t_rx_s").to_numpy()
        lock = tr.column("lock_test").to_numpy(zero_copy_only=False).astype(float)
        win = (t >= warmup_s - WARMUP_LOCK_WINDOW_S) & (t < warmup_s)
        w = lock[win]
        if w.size == 0 or not float(np.median(w)) >= WARMUP_LOCK_MIN:
            bad.append(ch)
            continue
        want = (expected_doppler_hz or {}).get(int(channels[ch][2]))
        if want is not None:
            dop = float(np.median(tr.column("dop_hz").to_numpy(zero_copy_only=False).astype(float)[win]))
            if not abs(dop - want) <= WARMUP_DOPPLER_TOL_HZ:
                bad.append(ch)
    return bad


def receiver_run_count(runs: Mapping[str, RunArtifacts]) -> int:
    """Every GNSS-SDR run of an experiment, re-runs included: the count is taken from the run record itself."""
    return len(runs)


def warmup_lock_gates(twins: Sequence[Twin]) -> list[tuple[str, GateReport]]:
    """One G-warmup-lock report per twin: passed when its lock policy is met. A channel still unlocked after the last
    attempt is removed from that twin's tracks, so no pairing uses it, and is listed (excluded, not hidden)."""
    out = []
    for tw in twins:
        for ch in tw.unlocked:
            tw.tracks.pop(ch, None)
        details = {"attempts": tw.attempts, "unlocked_channels": list(tw.unlocked)}
        out.append((f"[{tw.name}]", GateReport("G-warmup-lock", tw.lock_ok, details)))
    return out


def common_obs(a: Twin, b: Twin, channels: Mapping[int, Channel]) -> tuple[pa.Table | None, pa.Table | None]:
    """Both twins' observables restricted to the satellites whose channels both kept and that hold valid observables
    in both (a channel excluded by the lock policy, or one that locked but missed bit sync and so has no telemetry,
    leaves the observables pair; P2b task 10); (None, None) when either has no observables."""
    if a.obs is None or b.obs is None:
        return None, None

    def valid_prns(o: pa.Table) -> set[int]:
        ok = (
            o.column("valid").to_numpy(zero_copy_only=False) if "valid" in o.column_names else np.ones(o.num_rows, bool)
        )
        return {int(p) for p in np.unique(o.column("prn").to_numpy()[ok])}

    kept = {int(channels[ch][2]) for ch in set(a.tracks) & set(b.tracks)}
    keep = sorted(kept & valid_prns(a.obs) & valid_prns(b.obs))

    def only(o: pa.Table) -> pa.Table:
        return o.filter(pa.array(np.isin(o.column("prn").to_numpy(), keep)))

    return only(a.obs), only(b.obs)


def runs_of(twins: Sequence[Twin]) -> dict[str, RunArtifacts]:
    """Every receiver run of the twins, earlier attempts as <name>~<k>, for provenance."""
    out: dict[str, RunArtifacts] = {}
    for tw in twins:
        out[tw.name] = tw.art
        for k, art in enumerate(tw.failed, start=1):
            out[f"{tw.name}~{k}"] = art
    return out


def receiver_spec(spec: Mapping[str, Any], *, template: str = "1C",
                  extra: Mapping[str, str] | None = None) -> ReceiverSpec:  # fmt: skip
    """The patched receiver for every run of an experiment. Deviations from the v0.0.22 defaults are stated in the
    spec (`receiver_overrides`) and recorded with the runs; `extra` adds a loop cell's keys.

    D-34: a twin run must raise the carrier-lock limit of its signal and pull in with the FLL; a spec without both is
    refused unless it declares `stock_receiver: true` (the audit of GNSS-SDR as shipped).
    """
    overrides = {str(k): str(v) for k, v in (spec.get("receiver_overrides") or {}).items()}
    if not spec.get("stock_receiver", False):
        if template not in TWIN_LOCK_DEVIATION:
            raise ValueError(f"no D-34 deviation is defined for template {template!r}")
        missing = {k: v for k, v in TWIN_LOCK_DEVIATION[template].items() if overrides.get(k) != v}
        if missing:
            raise ValueError(f"a twin-run spec must set {missing} (D-34), or declare stock_receiver: true")
    overrides.update(extra or {})
    return ReceiverSpec(
        image=PATCHED, source_commit="3fcb502a", patch_id=PATCH_ID, template=template, overrides=overrides
    )


def parse_run(name: str, art: RunArtifacts, channels: Mapping[int, Channel], clock: TwinClock, source_id: str, *,
              t_int_s: float) -> Twin:  # fmt: skip
    """A receiver run's dumps: tracks on recording time (clock), observables as written."""
    fs_out = clock.fs_native_hz / clock.decim
    tracks = {
        ch: retime(read_tracking_dump(p, "patched112", run_id=name, channel=ch, sys=channels[ch][0],
                                      signal=channels[ch][1], fs_hz=fs_out, t_int_s=t_int_s), clock)
        for ch, p in art.tracking_dumps.items()
    }  # fmt: skip
    obs = None
    if art.observables_dump is not None:
        obs = read_observables_dump(art.observables_dump, channels, source_id=source_id, receiver_id=name)
    return Twin(name, None, art, tracks, obs)


def inject_twin(name: str, source: IQSource, ds: Sequence[Disturbance], fe: FrontEndSpec, work: Path, n0: float,
                *, workers: int = 1) -> tuple[SourceSpec, InjectionReport]:  # fmt: skip
    """Write one twin to /dev/shm (D-10); the caller deletes it (`SourceSpec.path`)."""
    sink = SinkSpec("shm_file", SHM_ROOT / work.name / f"{name}.bin")
    report = inject(source, ds, fe, sink, n0_per_hz=n0, workers=workers)
    return SourceSpec(path=sink.path, item_type="ishort", fs_hz=source.fs_hz / fe.decim), report


def receive(name: str, src: SourceSpec, channels: Mapping[int, Channel], work: Path, timeout_s: float,
            rx: ReceiverSpec, *, clock: TwinClock, warmup_s: float, t_int_s: float,
            expected_doppler_hz: Mapping[int, float] | None = None,
            lock_policy: LockPolicy | None = None) -> Twin:  # fmt: skip
    """Receive one twin's bytes, again (at most RX_ATTEMPTS runs in all) while its channels at the end of the warm-up
    (warmup_s after the twin's start, on recording time) do not meet the lock policy (without one: while any assigned
    channel is unlocked)."""
    warm_abs = clock.file_time(np.zeros(1, dtype=np.int64))[0] + warmup_s
    failed: list[RunArtifacts] = []

    def attempt_run(attempt: int) -> tuple[Twin, list[int]]:
        run_dir = work / (name if attempt == 1 else f"{name}~{attempt - 1}")
        art = run_receiver(rx, render_config(rx, src, channels), src, run_dir, timeout_s)
        tw = parse_run(name, art, channels, clock, work.name, t_int_s=t_int_s)
        return tw, unlocked_channels(tw.tracks, channels, warm_abs, expected_doppler_hz=expected_doppler_hz)

    def ok(bad: list[int]) -> bool:
        return lock_policy.met(channels, bad) if lock_policy is not None else not bad

    attempt = 1
    twin, bad = attempt_run(attempt)
    while not ok(bad) and attempt < RX_ATTEMPTS:
        failed.append(twin.art)
        attempt += 1
        twin, bad = attempt_run(attempt)
    twin.attempts = attempt
    twin.unlocked = bad
    twin.failed = failed
    twin.lock_ok = ok(bad)
    return twin


def run_twin(name: str, source: IQSource, ds: Sequence[Disturbance], fe: FrontEndSpec,
             channels: Mapping[int, Channel], work: Path, n0: float, timeout_s: float, rx: ReceiverSpec, *,
             clock: TwinClock, warmup_s: float, t_int_s: float = 1e-3, workers: int = 1,
             expected_doppler_hz: Mapping[int, float] | None = None,
             lock_policy: LockPolicy | None = None) -> Twin:  # fmt: skip
    """Inject one twin, receive it (with the warm-up lock check), and delete its bytes."""
    src, report = inject_twin(name, source, ds, fe, work, n0, workers=workers)
    try:
        twin = receive(name, src, channels, work, timeout_s, rx, clock=clock, warmup_s=warmup_s, t_int_s=t_int_s,
                       expected_doppler_hz=expected_doppler_hz, lock_policy=lock_policy)  # fmt: skip
    finally:
        src.path.unlink(missing_ok=True)
    twin.report = report
    return twin


def throughput(twins: Sequence[Twin], seconds: float) -> dict[str, Any]:
    """Real-time factors of the injector and of GNSS-SDR, per twin and as medians, and the /dev/shm peak: the
    largest twin's int16 I/Q stream (docs/10 E0 outputs)."""
    inj = {t.name: seconds / t.report.timings_s["total"] for t in twins if t.report and t.report.timings_s["total"]}
    rx = {t.name: seconds / t.art.wall_s for t in twins if t.art.wall_s}
    return {
        "injector_x_realtime": inj,
        "receiver_x_realtime": rx,
        "injector_x_realtime_median": float(np.median(list(inj.values()))) if inj else float("nan"),
        "receiver_x_realtime_median": float(np.median(list(rx.values()))) if rx else float("nan"),
        "shm_peak_bytes": max((4 * t.report.samples_out for t in twins if t.report), default=0),
    }
