# CMAPSS Turbofan Degradation — Dataset Card

A portable reference for the dataset behind this project. For how the data is
*modelled* (windowing, normalisation, thresholding) see README sections 4–5.

## Overview

NASA's **C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation)
Turbofan Engine Degradation Simulation Data Set contains run-to-failure
simulations of a fleet of jet engines. Each engine starts with unknown initial
wear, operates normally, then develops a fault that grows until failure. The
data are multivariate sensor time series contaminated with noise — the original
benchmark task is to predict each unit's Remaining Useful Life (RUL).

This project uses the **FD001** sub-dataset and reframes the task as
**reconstruction-based anomaly detection** (flag windows close to failure).

## Provenance & citation

- **Origin:** NASA Prognostics Center of Excellence (PCoE) data repository.
- **Reference:** A. Saxena, K. Goebel, D. Simon, and N. Eklund, *"Damage
  Propagation Modeling for Aircraft Engine Run-to-Failure Simulation"*,
  Proceedings of the 1st International Conference on Prognostics and Health
  Management (PHM08), Denver, CO, Oct 2008.

## License

Produced and released by NASA; as a U.S. Government work it carries no domestic
copyright and is in the **public domain**, distributed for open research use via
the NASA PCoE repository. Please cite the reference above when using it.

## Structure

The full archive ships **four sub-datasets**, each a different mix of operating
conditions and fault modes:

| Sub-dataset | Train engines | Test engines | Conditions | Fault modes |
|---|---|---|---|---|
| FD001 | 100 | 100 | 1 (sea level) | 1 (HPC degradation) |
| FD002 | 260 | 259 | 6 | 1 (HPC degradation) |
| FD003 | 100 | 100 | 1 (sea level) | 2 (HPC + fan degradation) |
| FD004 | 248 | 249 | 6 | 2 (HPC + fan degradation) |

Each sub-dataset comes as three space-separated text files (no header):

- `train_FDxxx.txt` — engines run all the way to failure.
- `test_FDxxx.txt` — engines truncated some cycles **before** failure.
- `RUL_FDxxx.txt` — published ground-truth RUL at the last observed cycle of
  each test engine.

Every row in `train`/`test` is one operational cycle of one engine and has
**26 columns**:

| Columns | Meaning |
|---|---|
| 1 | `unit` — engine id |
| 2 | `cycle` — time index, in operational cycles |
| 3–5 | `op_setting_1..3` — operating conditions |
| 6–26 | `sensor_1..21` — 21 sensor measurements |

RUL is implicit in `train` (last cycle = 0, counting up backwards) and supplied
by `RUL_FDxxx.txt` for `test`.

## How this project uses it

- **FD001 only** — single operating condition and single fault mode, the
  cleanest variant for an autoencoder baseline.
- **14 informative sensors.** Seven sensors are constant or near-constant in
  FD001 and carry no signal (`1, 5, 6, 10, 16, 18, 19`); they are dropped,
  matching common practice in the degradation literature.
- **Engine-disjoint splits** (no engine appears in two splits):

  | Split | Engine ids | Use |
  |---|---|---|
  | train | 1–70 | healthy windows only (`RUL > 100`) — fits the autoencoder |
  | validation | 71–80 | healthy windows only (`RUL > 100`) — calibrates the threshold |
  | test | 81–100 | every window, labelled anomaly = (`RUL ≤ 30`) |

- **Windows:** length 30 cycles, stride 1; per-sensor z-score statistics fitted
  on healthy training rows only (to avoid leaking the degradation signal).
- **Resulting counts** (reproducible exactly from `src.data.build_dataset`):
  5 032 train / 708 validation / 3 913 test windows, of which 620 (15.84 %) are
  labelled anomalies.

## How to obtain

The raw files are **not redistributed here** (`data/` is gitignored). They are
fetched automatically on the first pipeline run:

```bash
python main.py                       # downloads CMAPSS into data/raw on first call
# or, just the download step:
python -c "from src.data import download_cmapss; download_cmapss()"
```

Source archive:
`https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip`
