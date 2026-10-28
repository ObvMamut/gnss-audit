# Run summaries

These are the small files from finished runs that the E2 figures (`figures/f10-*`) are drawn from. They are laid out like the results root (`$GNSSAUDIT_RESULTS`), so this command redraws every figure without re-running anything:

```sh
make figures
```

`review/figures.py` checks each source before drawing. A run directory must hold a manifest naming its run id. Each summary must match the SHA-256 pinned in the script, and the review results (`review/results/s29`, `s30`, `s33`, `s34`) must name the same runs and summaries.

| Path | What it is |
|---|---|
| `E2/6214cf52489f/` | E2 run 2 (`experiments/E2-v2.yaml`): manifest and summary |
| `E2/9650dc00e766/` | E2 run 3 (`experiments/E2-v3.yaml`): manifest and summary |
| `E2K/0443516ca48c/summary.json` | E2K (`experiments/E2K.yaml`); the run failed a gate, so it has no manifest |
| `dev/e2k-dryrun-task13.json` | E2K's planning dry run, read by the G3 figures |
| `E0b/fee767ba3538/`, `E0b/46b76d4b47d9/` | E0 on FGI segment A (C/A) and the SpoofRepo L5 segment: manifest and loop-identification files |

The files are copied byte for byte from the runs. Everything else the runs wrote, such as per-unit records and tracking dumps (about 1 GB per E2 run), stays outside the repository.

## Licence and attribution

The runs processed recordings from two datasets of the Finnish Geospatial Research Institute (FGI), National Land Survey of Finland, both published under CC BY 4.0:

- FGI Jammer Test 2023 Repository, [doi.org/10.23729/fd-06d27736-45cb-3ca2-aff8-725d42c6caeb](https://doi.org/10.23729/fd-06d27736-45cb-3ca2-aff8-725d42c6caeb);
- FGI's GNSS Spoofing Dataset Repository (FGI-SpoofRepo), [doi.org/10.23729/7a648509-2ca8-4a7d-8223-0b429182f857](https://doi.org/10.23729/7a648509-2ca8-4a7d-8223-0b429182f857). Its companion paper is S. Islam et al., *GPS Solutions*, 2024.

These summaries are derived from that data and are released under CC BY 4.0.
