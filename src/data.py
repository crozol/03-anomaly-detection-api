"""CMAPSS FD001 loading, preprocessing, and window construction.

Public protocol exposed by this module:

* ``download_cmapss(raw_dir)`` — fetch the NASA PCOE CMAPSS archive (only on
  first call; subsequent calls are no-ops if the text files are already on
  disk).
* ``load_split(name, split, raw_dir)`` — return the raw DataFrame for one
  of the four sub-datasets (FD001…FD004) and one of the two splits
  (``"train"`` / ``"test"``).
* ``build_dataset(...)`` — end-to-end builder that returns the three
  ``(windows, labels, meta)`` tuples used downstream by the autoencoder.

Defaults mirror the most common CMAPSS-anomaly-detection protocol in the
literature: window length 30 cycles, stride 1, FD001 only, sensors with
non-degenerate variance only, per-sensor z-score statistics fitted on the
training healthy windows.
"""

from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
#  paths and constants
# --------------------------------------------------------------------------- #

CMAPSS_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)

COLUMN_NAMES: list[str] = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in (1, 2, 3)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Sensors that are constant or near-constant in FD001 and contribute no
# information — discarded across the literature (e.g. Heimes, 2008).
LOW_VARIANCE_SENSORS_FD001: tuple[int, ...] = (1, 5, 6, 10, 16, 18, 19)
INFORMATIVE_SENSORS_FD001: tuple[int, ...] = tuple(
    i for i in range(1, 22) if i not in LOW_VARIANCE_SENSORS_FD001
)

DEFAULT_SEQ_LEN = 30
DEFAULT_HEALTHY_RUL_CUTOFF = 100   # train/val: keep windows with RUL > this
DEFAULT_ANOMALY_RUL_CUTOFF = 30    # test: label window anomaly if RUL <= this


# --------------------------------------------------------------------------- #
#  download
# --------------------------------------------------------------------------- #

def download_cmapss(raw_dir: str | os.PathLike = "data/raw") -> Path:
    """Download CMAPSS into ``raw_dir`` if the text files are not already there.

    Returns the directory where the .txt files live.
    """
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    expected = raw / "train_FD001.txt"
    if expected.exists():
        return raw

    print(f"[data] downloading CMAPSS archive into {raw} …")
    with urllib.request.urlopen(CMAPSS_URL) as resp:
        outer = zipfile.ZipFile(io.BytesIO(resp.read()))
    inner_name = next(n for n in outer.namelist() if n.endswith("CMAPSSData.zip"))
    with outer.open(inner_name) as fh:
        inner = zipfile.ZipFile(io.BytesIO(fh.read()))
    inner.extractall(raw)
    return raw


# --------------------------------------------------------------------------- #
#  raw split loading
# --------------------------------------------------------------------------- #

def load_split(
    name: str = "FD001",
    split: str = "train",
    raw_dir: str | os.PathLike = "data/raw",
) -> pd.DataFrame:
    """Load one CMAPSS .txt file as a DataFrame with named columns."""
    if split not in ("train", "test"):
        raise ValueError(f"unknown split {split!r}, must be 'train' or 'test'")
    path = Path(raw_dir) / f"{split}_{name}.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    return df


def load_rul(
    name: str = "FD001",
    raw_dir: str | os.PathLike = "data/raw",
) -> np.ndarray:
    """Load the published RUL ground truth for the test engines of ``name``."""
    return np.loadtxt(Path(raw_dir) / f"RUL_{name}.txt")


# --------------------------------------------------------------------------- #
#  RUL annotation
# --------------------------------------------------------------------------- #

def annotate_rul_train(df: pd.DataFrame) -> pd.DataFrame:
    """Add a per-row ``rul`` column to a *training* DataFrame.

    Each engine in the training set runs to failure — its last cycle
    has RUL = 0, and earlier cycles count up linearly.
    """
    last_cycle = df.groupby("unit")["cycle"].transform("max")
    return df.assign(rul=last_cycle - df["cycle"])


def annotate_rul_test(df: pd.DataFrame, rul_at_end: np.ndarray) -> pd.DataFrame:
    """Add a per-row ``rul`` column to a *test* DataFrame.

    Each test engine ends some cycles before failure; ``rul_at_end[i]``
    is the published RUL at that last observed cycle for unit ``i+1``.
    """
    n_units = df["unit"].nunique()
    if rul_at_end.shape[0] != n_units:
        raise ValueError(
            f"RUL ground truth has {rul_at_end.shape[0]} entries but the test "
            f"DataFrame contains {n_units} engines"
        )
    last_cycle = df.groupby("unit")["cycle"].transform("max")
    rul_end_per_row = df["unit"].map(
        {u: rul_at_end[u - 1] for u in range(1, n_units + 1)}
    )
    return df.assign(rul=last_cycle - df["cycle"] + rul_end_per_row.astype(float))


# --------------------------------------------------------------------------- #
#  window construction
# --------------------------------------------------------------------------- #

@dataclass
class WindowSet:
    """A bundle of fixed-length windows extracted from one or more engines."""

    x: np.ndarray              # (N, seq_len, n_features)
    rul: np.ndarray            # (N,) RUL at the *last* cycle of each window
    unit: np.ndarray           # (N,) engine id
    cycle_end: np.ndarray      # (N,) last cycle of each window

    @property
    def n_features(self) -> int:
        return self.x.shape[2]


def _windows_for_engine(
    engine_df: pd.DataFrame,
    sensor_cols: list[str],
    seq_len: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = engine_df[sensor_cols].to_numpy(dtype=np.float32)
    rul = engine_df["rul"].to_numpy(dtype=np.float32)
    cycle = engine_df["cycle"].to_numpy(dtype=np.int32)
    n = arr.shape[0]
    if n < seq_len:
        return (
            np.empty((0, seq_len, arr.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )
    starts = np.arange(0, n - seq_len + 1, stride, dtype=np.int32)
    xs = np.stack([arr[s : s + seq_len] for s in starts], axis=0)
    end_idx = starts + seq_len - 1
    return xs, rul[end_idx], cycle[end_idx]


def build_windows(
    df: pd.DataFrame,
    sensor_cols: list[str],
    seq_len: int = DEFAULT_SEQ_LEN,
    stride: int = 1,
    engines: Iterable[int] | None = None,
) -> WindowSet:
    """Stack fixed-length sliding windows for every engine in ``df``."""
    if engines is None:
        engines = sorted(df["unit"].unique())
    xs, ruls, units, cycles = [], [], [], []
    for u in engines:
        sub = df[df["unit"] == u].sort_values("cycle")
        x_u, rul_u, c_u = _windows_for_engine(sub, sensor_cols, seq_len, stride)
        if x_u.shape[0] == 0:
            continue
        xs.append(x_u)
        ruls.append(rul_u)
        units.append(np.full(x_u.shape[0], u, dtype=np.int32))
        cycles.append(c_u)
    if not xs:
        raise RuntimeError("no windows were produced — check seq_len / engines")
    return WindowSet(
        x=np.concatenate(xs, axis=0),
        rul=np.concatenate(ruls, axis=0),
        unit=np.concatenate(units, axis=0),
        cycle_end=np.concatenate(cycles, axis=0),
    )


# --------------------------------------------------------------------------- #
#  end-to-end dataset builder
# --------------------------------------------------------------------------- #

@dataclass
class Dataset:
    """Train / val / test bundles plus the normalisation statistics."""

    train: WindowSet           # healthy windows from training engines
    val: WindowSet             # healthy windows held out for threshold calibration
    test: WindowSet            # all windows from test engines, with binary labels
    test_label: np.ndarray     # (N_test,) — anomaly = 1 if RUL <= anomaly_cutoff
    sensor_cols: list[str]
    mean: np.ndarray
    std: np.ndarray
    seq_len: int
    healthy_cutoff: int
    anomaly_cutoff: int


def build_dataset(
    name: str = "FD001",
    raw_dir: str | os.PathLike = "data/raw",
    seq_len: int = DEFAULT_SEQ_LEN,
    healthy_cutoff: int = DEFAULT_HEALTHY_RUL_CUTOFF,
    anomaly_cutoff: int = DEFAULT_ANOMALY_RUL_CUTOFF,
    train_engines: tuple[int, int] = (1, 70),
    val_engines: tuple[int, int] = (71, 80),
    test_engines: tuple[int, int] = (81, 100),
    sensors: Iterable[int] | None = None,
    stride: int = 1,
    seed: int = 0,
) -> Dataset:
    """Construct the train / val / test windows from CMAPSS train_FDxxx.

    The protocol is engine-disjoint: distinct engine ids feed train, val and
    test. Train and val use only windows with ``rul > healthy_cutoff``; test
    uses every window from its engines and labels them by the threshold
    ``rul <= anomaly_cutoff``.
    """
    download_cmapss(raw_dir)
    df = load_split(name=name, split="train", raw_dir=raw_dir)
    df = annotate_rul_train(df)

    sensor_ids = tuple(sensors) if sensors is not None else INFORMATIVE_SENSORS_FD001
    sensor_cols = [f"sensor_{i}" for i in sensor_ids]

    # 1. Fit z-score statistics on the *training* healthy windows only.
    train_units = list(range(train_engines[0], train_engines[1] + 1))
    val_units = list(range(val_engines[0], val_engines[1] + 1))
    test_units = list(range(test_engines[0], test_engines[1] + 1))

    train_df_raw = df[df["unit"].isin(train_units)]
    healthy_train_rows = train_df_raw[train_df_raw["rul"] > healthy_cutoff]
    mean = healthy_train_rows[sensor_cols].mean().to_numpy(dtype=np.float32)
    std = healthy_train_rows[sensor_cols].std().to_numpy(dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    # 2. Apply the same normalisation to every row in df.
    df_norm = df.copy()
    df_norm[sensor_cols] = (df_norm[sensor_cols].to_numpy(dtype=np.float32) - mean) / std

    # 3. Build windows.
    train_ws = build_windows(
        df_norm[df_norm["unit"].isin(train_units)],
        sensor_cols, seq_len, stride, engines=train_units,
    )
    val_ws = build_windows(
        df_norm[df_norm["unit"].isin(val_units)],
        sensor_cols, seq_len, stride, engines=val_units,
    )
    test_ws = build_windows(
        df_norm[df_norm["unit"].isin(test_units)],
        sensor_cols, seq_len, stride, engines=test_units,
    )

    # 4. Filter train / val to the healthy regime.
    healthy_train_mask = train_ws.rul > healthy_cutoff
    healthy_val_mask = val_ws.rul > healthy_cutoff
    train_ws = WindowSet(
        x=train_ws.x[healthy_train_mask],
        rul=train_ws.rul[healthy_train_mask],
        unit=train_ws.unit[healthy_train_mask],
        cycle_end=train_ws.cycle_end[healthy_train_mask],
    )
    val_ws = WindowSet(
        x=val_ws.x[healthy_val_mask],
        rul=val_ws.rul[healthy_val_mask],
        unit=val_ws.unit[healthy_val_mask],
        cycle_end=val_ws.cycle_end[healthy_val_mask],
    )

    # 5. Test: keep every window, derive a binary anomaly label from RUL.
    test_label = (test_ws.rul <= anomaly_cutoff).astype(np.int32)

    return Dataset(
        train=train_ws, val=val_ws, test=test_ws, test_label=test_label,
        sensor_cols=sensor_cols, mean=mean, std=std,
        seq_len=seq_len,
        healthy_cutoff=healthy_cutoff,
        anomaly_cutoff=anomaly_cutoff,
    )


def summarise_dataset(ds: Dataset) -> dict:
    """Compact summary used by main.py and the metrics export."""
    return {
        "name": "FD001",
        "seq_len": ds.seq_len,
        "n_features": ds.train.n_features,
        "sensors": ds.sensor_cols,
        "healthy_cutoff": ds.healthy_cutoff,
        "anomaly_cutoff": ds.anomaly_cutoff,
        "n_train_windows": int(ds.train.x.shape[0]),
        "n_val_windows": int(ds.val.x.shape[0]),
        "n_test_windows": int(ds.test.x.shape[0]),
        "n_test_anomalies": int(ds.test_label.sum()),
        "anomaly_fraction": float(ds.test_label.mean()),
    }


__all__ = [
    "CMAPSS_URL",
    "COLUMN_NAMES",
    "INFORMATIVE_SENSORS_FD001",
    "Dataset",
    "WindowSet",
    "annotate_rul_test",
    "annotate_rul_train",
    "build_dataset",
    "build_windows",
    "download_cmapss",
    "load_rul",
    "load_split",
    "summarise_dataset",
]
