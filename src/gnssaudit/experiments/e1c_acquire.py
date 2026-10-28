"""Network side of the arm c scout run: fetch and reduce the screen, fetch the station-month (docs/7 section 2.2).

    uv run python -m gnssaudit.experiments.e1c_acquire screen experiments/E1c-scout.yaml
    uv run python -m gnssaudit.experiments.e1c_acquire month experiments/E1c-scout.yaml

`screen` fetches the coordinate file and broadcast navigation of the screen days, draws the sample
(`e1c_scout.screen_sample`), and for each station and its nearest partner with data on the target day downloads the
target day and its history days, reduces every hourly file to GPS 1C C/N0 and elevation (`io.rgp.reduce_hourly`), writes
one parquet per station-day under reduced/rgp/YYYY/DDD/ and deletes the raw files. The SHA-256 of each deleted file
and the number of missing hours are kept in the parquet metadata; the station-to-partner choice goes to
screen-index.json. Reruns skip station-days already reduced.

`month` downloads the hourly files of `month.station` and `month.partner` (history days included) into raw/rgp/YYYY/DDD/
read-only, plus the daily coordinate and navigation files and the GFZ Kp of the month, and writes the per-file SHA-256
sidecar registry/rgp-<year>-<station>-<partner>.sha256 that registry/datasets.yaml points to.

At most 4 downloads run at once (`io.fetch` retries with backoff); a 404 is a missing hour, any other failure stops.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import urllib.error
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from gnssaudit.experiments.e1c import (
    coords_path,
    day_date,
    kp_path,
    nav_name,
    nav_path,
    raw_hour_path,
    reduce_job,
    reduced_path,
    screen_index_path,
    workers,
)
from gnssaudit.experiments.e1c_scout import screen_sample
from gnssaudit.geometry import FloatArray
from gnssaudit.io.fetch import commit_part, fetch
from gnssaudit.io.hatanaka import decompress_z
from gnssaudit.io.rgp import REDUCED_SCHEMA, coords_url, hourly_url, partners, read_rgf93_coords
from gnssaudit.paths import data_root
from gnssaudit.provenance import sha256_file
from gnssaudit.registry import REGISTRY

DOWNLOADS = 4
RAW_BUDGET_BYTES = 35 * 10**9  # docs/7 section 3.3
NAV_URL = "ftp://gssc.esa.int/gnss/data/daily/{year}/{doy:03d}/{name}.gz"
KP_URL = "https://kp.gfz.de/app/json/?start={start}T00:00:00Z&end={end}T23:59:59Z&index=Kp&status=def"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def ensure_coords(year: int, doy: int) -> Path:
    path = coords_path(year, doy)
    if not path.exists():
        fetch(coords_url(year, doy), path.parent)
    return path


def ensure_nav(year: int, doy: int) -> Path:
    """The day's IGS merged broadcast navigation from ESA GSSC, gunzipped beside its archive, both read-only."""
    path = nav_path(year, doy)
    if path.exists():
        return path
    name = nav_name(year, doy)
    archive = data_root() / "raw" / "igs-brdc" / f"{name}.gz"
    if not archive.exists():
        fetch(NAV_URL.format(year=year, doy=doy, name=name), archive.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with gzip.open(archive, "rb") as src, part.open("wb") as out:
        shutil.copyfileobj(src, out, 1 << 20)
    commit_part(part, path)
    path.chmod(0o444)
    return path


def ensure_kp(year: int, month: int) -> Path:
    path = kp_path(year, month)
    if not path.exists():
        first = f"{year}-{month:02d}-01"
        nxt = (np.datetime64(first, "M") + 1).astype("datetime64[D]") - 1
        fetch(KP_URL.format(start=first, end=str(nxt)), path.parent, name=path.name)
    return path


REDUCED_META = (b"station", b"year_doy", b"missing_hours", b"raw_sha256", b"rx_ecef_m")


def reduced_valid(path: Path) -> bool:
    """A reduced parquet counts as done only if its schema and metadata read and hold every key reduce_days writes."""
    try:
        md = pq.read_schema(path).metadata or {}
    except OSError, pa.ArrowException:
        return False
    return all(k in md for k in REDUCED_META)


def pending_days(station: str, year: int, doys: list[int]) -> list[int]:
    """The days still to reduce; an invalid file at the reduced path (a crash left it) is deleted and redone."""
    todo = []
    for d in doys:
        path = reduced_path(station, year, d)
        if path.exists() and reduced_valid(path):
            continue
        path.unlink(missing_ok=True)
        todo.append(d)
    return todo


def raw_usable(path: Path) -> bool:
    """A raw hourly file already on disk is reused only if it expands."""
    try:
        decompress_z(path)
    except OSError, ValueError:
        return False
    return True


def _get(url: str, dest: Path) -> tuple[Path, int, str] | None:
    """Download into dest (an existing file is reused only if it expands, else fetched again); None on 404."""
    target = dest / url.rsplit("/", 1)[-1]
    if target.exists():
        if raw_usable(target):
            return target, target.stat().st_size, sha256_file(target)
        target.unlink()
    try:
        got = fetch(url, dest)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise
    return got.path, got.size_bytes, got.sha256


# ---- screen ---------------------------------------------------------------------------------------------------------


def reduce_days(
    station: str, year: int, doys: list[int], rx: FloatArray, dl: ThreadPoolExecutor, pool: ProcessPoolExecutor
) -> dict[int, int]:
    """Reduce the station-days not yet reduced; raw files go to a scratch directory per station-day and are deleted.
    Returns the missing hours per day (read back from the parquet metadata for days already done)."""
    missing: dict[int, int] = {}
    todo = pending_days(station, year, doys)
    scratch = {d: data_root() / "tmp" / "rgp-screen" / station / f"{year}-{d:03d}" for d in todo}
    downloads = {(d, h): dl.submit(_get, hourly_url(station, year, d, h), scratch[d]) for d in todo for h in range(24)}
    decodes: dict[tuple[int, int], Future[pa.Table]] = {}
    hashes: dict[int, dict[str, str]] = {d: {} for d in todo}
    for (d, h), fut in downloads.items():  # submitted in order; decoding starts as each file lands
        got = fut.result()
        if got is None:
            continue
        path, _, sha = got
        hashes[d][path.name] = sha
        rx_t = (float(rx[0]), float(rx[1]), float(rx[2]))
        decodes[(d, h)] = pool.submit(reduce_job, (str(path), str(nav_path(year, d)), rx_t))
    for d in todo:
        tables = [decodes[(d, h)].result() for h in range(24) if (d, h) in decodes]
        table = pa.concat_tables(tables) if tables else REDUCED_SCHEMA.empty_table()
        meta = {
            b"station": station.encode(),
            b"year_doy": f"{year}-{d:03d}".encode(),
            b"missing_hours": str(24 - len(tables)).encode(),
            b"raw_sha256": json.dumps(hashes[d], sort_keys=True).encode(),
            b"rx_ecef_m": json.dumps([float(v) for v in rx]).encode(),
        }
        out = reduced_path(station, year, d)
        out.parent.mkdir(parents=True, exist_ok=True)
        part = out.with_name(out.name + ".part")
        pq.write_table(table.replace_schema_metadata(meta), part, compression="zstd")
        commit_part(part, out)
        if scratch[d].exists():
            shutil.rmtree(scratch[d])
    for d in doys:
        md = pq.read_schema(reduced_path(station, year, d)).metadata or {}
        missing[d] = int(md.get(b"missing_hours", b"0"))
    return missing


def screen(spec: dict[str, Any]) -> None:
    sc, pr = spec["screen"], spec["partner"]
    year, doy, hist = sc["year"], sc["doy"], sc["history_days"]
    doys = list(range(doy - hist, doy + 1))
    min_m, max_m = pr["min_km"] * 1e3, pr["max_km"] * 1e3
    coords = read_rgf93_coords(ensure_coords(year, doy))
    for d in doys:
        ensure_nav(year, d)
    sample = screen_sample(coords, sc["n_stations"], sc["seed"], sc["always"], min_m, max_m)
    _log(f"screen sample ({len(sample)}): {' '.join(sample)}")
    index: dict[str, Any] = {"year": year, "doy": doy, "seed": sc["seed"], "stations": {}}
    with ThreadPoolExecutor(DOWNLOADS) as dl, ProcessPoolExecutor(max_workers=workers()) as pool:
        for i, station in enumerate(sample, 1):
            miss = reduce_days(station, year, doys, coords[station], dl, pool)
            chosen, chosen_km, partner_miss = None, None, None
            for cand, dist in partners(station, coords, min_m, max_m):
                cm = reduce_days(cand, year, [doy], coords[cand], dl, pool)
                if cm[doy] < 24:  # the partner has data on the target day
                    chosen, chosen_km = cand, dist / 1e3
                    partner_miss = reduce_days(cand, year, doys, coords[cand], dl, pool)
                    break
            index["stations"][station] = {
                "partner": chosen,
                "partner_km": chosen_km,
                "missing_hours": {str(d): m for d, m in miss.items()},
                "partner_missing_hours": {str(d): m for d, m in (partner_miss or {}).items()},
            }
            _log(f"[{i}/{len(sample)}] {station}: missing {miss}; partner {chosen} ({chosen_km} km)")
    path = screen_index_path(year, doy)
    path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    shutil.rmtree(data_root() / "tmp" / "rgp-screen", ignore_errors=True)
    _log(f"wrote {path}")


# ---- month ----------------------------------------------------------------------------------------------------------


def _tree_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def month(spec: dict[str, Any]) -> None:
    mo, hist = spec["month"], spec["screen"]["history_days"]
    year, station, partner = mo["year"], mo["station"], mo["partner"]
    if not station or not partner:
        raise SystemExit("month.station and month.partner must be set from the screen run first")
    doys = list(range(mo["doy_first"] - hist, mo["doy_last"] + 1))
    raw = data_root() / "raw"
    expected = 2 * len(doys) * 24 * 2_100_000
    used = _tree_bytes(raw)
    free = shutil.disk_usage(data_root()).free
    _log(f"raw store {used / 1e9:.2f} GB, expected +{expected / 1e9:.2f} GB, {free / 1e9:.1f} GB free on the disk")
    if used + expected > RAW_BUDGET_BYTES or expected > free:
        raise SystemExit("the month would break the 35 GB raw budget or the free space")
    for d in doys:
        ensure_coords(year, d)
        ensure_nav(year, d)
    targets = range(mo["doy_first"], mo["doy_last"] + 1)
    for y, m in sorted({(day_date(year, d).year, day_date(year, d).month) for d in targets}):
        ensure_kp(y, m)
    lines, missing, total = [], [], 0
    with ThreadPoolExecutor(DOWNLOADS) as dl:
        futs = {
            (who, d, h): dl.submit(_get, hourly_url(who, year, d, h), raw_hour_path(who, year, d, h).parent)
            for who in (station, partner)
            for d in doys
            for h in range(24)
        }
        for (who, d, h), fut in futs.items():
            got = fut.result()
            if got is None:
                missing.append(f"{who} {year}-{d:03d} hour {h}")
                continue
            path, size, sha = got
            total += size
            lines.append(f"{sha}  {size}  {path.relative_to(data_root()).as_posix()}")
            if h == 23:
                _log(f"{who} {year}-{d:03d} done")
    sidecar = REGISTRY / f"rgp-{year}-{station}-{partner}.sha256"
    header = [
        f"# RGP 1 s hourly files, {station} and partner {partner}, {year} days {doys[0]:03d} to {doys[-1]:03d}",
        f"# URL pattern: {hourly_url('ssss', year, 1, 0).replace('/001/', '/DDD/').replace('ssss001a', 'ssssDDDh')}",
        f"# files: {len(lines)}, bytes: {total}, missing hours: {len(missing)}",
        *[f"# missing: {m}" for m in missing],
        "# sha256  size_bytes  path under $GNSSAUDIT_DATA",
    ]
    sidecar.write_text("\n".join(header + lines) + "\n")
    _log(f"{len(lines)} files, {total / 1e9:.3f} GB, {len(missing)} missing; wrote {sidecar}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="e1c_acquire")
    parser.add_argument("stage", choices=["screen", "month"])
    parser.add_argument("spec", type=Path)
    args = parser.parse_args(argv)
    spec = yaml.safe_load(args.spec.read_text())
    if args.stage == "screen":
        screen(spec)
    else:
        month(spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
