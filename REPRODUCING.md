# Reproducing

The tiers below are ordered from cheapest to most expensive. Every command runs from the repository root. The project was developed on Linux x86-64.

## Tier 0: install and test

You need [uv](https://docs.astral.sh/uv/). It installs the Python 3.14 that `.python-version` names if you don't have it, and the exact package versions from `uv.lock`. Work from a `git clone`, not a downloaded archive: every run records its commit (`git describe`), and the tests that drive the CLI need a repository.

```sh
uv sync --locked
make check
```

`make check` runs ruff, pyright and the test suite, except tests marked `slow` (long Monte Carlo), `container` (needs Docker and the patched receiver image) or `data` (needs the recordings). `make check-full` runs those too, once tier 3's prerequisites are in place.

## Tier 1: redraw every figure

```sh
make figures
```

- **f14-1 to f14-3** come from the committed review results `review/results/s02`, `s06`, `s07` and `s14`.
- **f10-1 to f10-8** come from the run files committed in `run-summaries/`.

`review/figures.py` refuses any source whose run ID or SHA-256 differs from the one it pins. The redrawn PNGs are byte-identical to the committed ones, and CI checks this on every push.

## Tier 2: re-run the review studies

```sh
make review
```

This re-runs the simulation studies s01 to s16, s27 and s28 (no data, no Docker) and rewrites their `review/results/*.json`. On the development machine (20 cores) it took about 29 minutes; s28 alone takes 12. Each result file carries a `meta` block with the run time and package versions. In the locked environment, 17 of the 18 results came out identical to the committed files apart from that block (checked 29 Oct 2026). The exception is s14:
- its closed-loop simulations are identical, but its P6 predictions differ by up to 0.05 % relative;
- the committed s14 was generated on 9 Oct 2026, before the lock file existed, and its code has not changed since;
- its `meta` block does not record the SciPy version, so the cause is not established.

No number in the README changes at the precision quoted.

The other studies are not part of `make review`:

- **s29 to s35** are frozen records. They read the original run directories, and their result files are pinned by SHA-256 elsewhere. They run only when named, for example `uv run python -m review.run_all s30`, and a re-run breaks those pins.
- **s17 to s26 and s36** are run as modules, for example `uv run python -m review.s18_gnsssdr_cn0_lock_p6`. Those that drive the GNSS-SDR binary need the patched image (tier 3): s17, s18, s20, s22, s23 and s24.

## Tier 3: re-run the experiments

### Prerequisites

- **Docker,** usable without `sudo`.
- **Disk:**
  - about 26 GB for the three recordings below;
  - 40 GB kept free on that disk while each one downloads (`stream-prefix` refuses less);
  - about 1 GB per E2 run and 4 GB for E2K.
- **Memory:** twins are streamed through `/dev/shm/gnssaudit`. The specifications use 6 workers. The original runs used a machine with 20 cores, 31 GB of RAM and a 16 GB `/dev/shm`.
- **Locations:** set `GNSSAUDIT_DATA` for the recordings and `GNSSAUDIT_RESULTS` for the run directories. Without them, the code falls back to `~/all/private/GNSS-data` and `~/all/private/GNSS-results`.

### The receiver image

```sh
make image-patched
```

This builds `gnssaudit/gnss-sdr:v0.0.22-p2` from `containers/Dockerfile.patched`, on top of `carlesfernandez/docker-gnsssdr:v0.0.22` pulled by digest, and prints the new image's ID. The build needs network access and about 8 GB of Docker space. The three patches in `containers/patches/` make the tracking dump:

- carry the accumulated carrier phase as float64 (0001);
- write one record per loop update (0002);
- use the correct XB code period for GPS L5 (0003).

The pipeline runs the patched image **by its ID**. That ID is `PATCHED` in `src/gnssaudit/sdr/images.py`, `sha256:5bacd9f5…`, the build of 19 Oct 2026, and a build on another machine gets another ID. Set `PATCHED` to the ID `make image-patched` printed and commit the change, because confirmatory runs refuse a dirty tree. Your build compiles the same source and patches, but it is not guaranteed to be bit-identical to the original.

### The recordings

The experiments use the first seconds of three files from the Finnish Geospatial Research Institute's public repositories on Fairdata (CC BY 4.0). Fairdata serves no HTTP ranges, so `gnssaudit stream-prefix` streams each file from its start and keeps the prefix. By default it then reads on to the end to check the whole file against its published SHA-256. At the roughly 4 MB/s measured, that takes:

- about 1.6 h for `JT23_4.1.5` (24.6 GB);
- about 3.6 h for `JT23_16.1.1` (55 GB);
- about 1.3 h for `TGS_L5_E5a` (20.2 GB).

```sh
uv run gnssaudit stream-prefix fgi-jt2023 JT23_4.1.5_L1_E1.iq --seconds 150 --bytes-per-second 61380000
uv run gnssaudit stream-prefix fgi-jt2023 JT23_16.1.1_L1_E1.iq --seconds 150 --bytes-per-second 61380000
uv run gnssaudit stream-prefix fgi-spoofrepo TGS_L5_E5a.dat --seconds 150 --bytes-per-second 52000000
```

The rates are 30.69 MHz × 2 bytes for the JT2023 files and 26 MHz × 2 bytes for the L5 file. Each command prints the file's registry block. Its `sha256` must equal the one in `registry/datasets.yaml`. The admitted clean segments (`registry/intervals.yaml`) are already committed, so the admission runs (`experiments/admit-*.yaml`) are not needed.

### The runs

```sh
uv run gnssaudit run experiments/E2-v3.yaml --confirmatory
```

A run's ID is the SHA-256 of its specification, so each specification writes to the directory shown below. The run is write-once. `--resume` continues an interrupted run under the same code version.

| Specification | Run directory | What it does | Original wall time | Needs |
|---|---|---|---|---|
| `E0b-fgi-A.yaml` | `E0b/fee767ba3538` | Identifies the receiver loop on segment A (GPS C/A) | 38 min | JT23_4.1.5 |
| `E0b-spoofrepo-L5.yaml` | `E0b/46b76d4b47d9` | The same on the L5 segment | 48 min | TGS_L5_E5a |
| `E2-v2.yaml` | `E2/6214cf52489f` | E2 run 2: twins, front-end cells and Route B truth on segments A and B | 8.8 h | both JT2023 prefixes |
| `E2K.yaml` | `E2K/0443516ca48c` | The K-study and thermal comparison on run 2 | about 1.5 days | run 2's directory; review s29 |
| `E2-v3.yaml` | `E2/9650dc00e766` | E2 run 3 with ghost truth | 6.5 h | both JT2023 prefixes |

Run 2 originally ran on image p1, which had patches 0001 and 0002 but not 0003. Re-running `E2-v2.yaml` today uses p2.

### Comparing with the committed results

GNSS-SDR is not bit-deterministic on identical input. After warm-up, twins of the same samples agree to about 4 % of the thermal variance (review s24). A re-run's `summary.json` will therefore not match the committed SHA-256. Compare its values with their standard errors instead: every front-end cell carries `ratio` and `ratio_se`. The injector itself is byte-reproducible (the G-repeat gate).

To draw the figures from your own runs, point `GNSSAUDIT_RESULTS` at them and update the pins (`G3_*`, `P2C_*`) in `review/figures.py`, which refuses any source it does not name.
