# Container digest log

One row per image the pipeline runs. Rows are appended, never edited (D-04, [[9 Receiver Integration#1. Version and build policy]]).

| Recorded | Image | Index digest | Image created | `gnss-sdr --version` | Role |
|---|---|---|---|---|---|
| 2026-10-10 | `carlesfernandez/docker-gnsssdr:v0.0.22` | `sha256:befe06ed81163068e9eaf918024e18cedf1fc374925431320d57055650b48b79` | 2026-10-05T07:38:11Z | 0.0.22 | primary receiver (D-04), stock build |
| 2026-10-14 | `gnssaudit/gnss-sdr:v0.0.22-p1`, local build of `containers/Dockerfile.patched` on the stock image above | image ID `sha256:1a63194c869aa1b1b1f184ec119ad1547b94bef72c2f4c4d8d59b0e795eb633a` (local, no registry digest) | 2026-10-14T00:08:26+02:00 | 0.0.22 | patched build (D-08, D-27): `0001-acc-phase-float64.patch` sha256 `6e50ad998c3c00afb06f7ef9563ddf48ab9e055cdfad8fa213fd14d2d644ea3f`, `0002-log-every-integration.patch` sha256 `4f404392b13ee03237f33f7039e57171d1f763a2dc20bfc2d964450396a63510` |
| 2026-10-19 | `gnssaudit/gnss-sdr:v0.0.22-p2`, local build of `containers/Dockerfile.patched` (`MAKE_JOBS=6`) on the stock image above | image ID `sha256:5bacd9f59966f09cb425897cb7828cec3721d75d592fdc2cec77840fc0c3694c` (local, no registry digest) | 2026-10-19T14:52:33+02:00 | 0.0.22 | patched build (D-08, D-27, D-40): `0001-acc-phase-float64.patch` sha256 `6e50ad998c3c00afb06f7ef9563ddf48ab9e055cdfad8fa213fd14d2d644ea3f`, `0002-log-every-integration.patch` sha256 `4f404392b13ee03237f33f7039e57171d1f763a2dc20bfc2d964450396a63510`, `0003-l5-xb-period.patch` sha256 `40d1a28b7f9c84e830ac507106b893f3c65bd0611b74f0f4a390b2d9e75a18c0` |

Run it by digest: `carlesfernandez/docker-gnsssdr:v0.0.22@sha256:befe06ed81163068e9eaf918024e18cedf1fc374925431320d57055650b48b79`.

Run the patched build by its image ID: p2, `sha256:5bacd9f59966f09cb425897cb7828cec3721d75d592fdc2cec77840fc0c3694c` (`make image-patched` rebuilds it; a rebuild on another machine gets another ID, to be recorded here). p1, `sha256:1a63194c869aa1b1b1f184ec119ad1547b94bef72c2f4c4d8d59b0e795eb633a`, is kept for the runs pinned to it (E2 run 2's worktree at `7bc707e`).
