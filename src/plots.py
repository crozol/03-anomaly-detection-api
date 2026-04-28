"""Publication-style static figures for the README and the portfolio page.

The visual style mirrors ``02-neural-odes/src/plots.py``: dark background,
muted grids, monospace numerics, sans-serif prose. All figures are
written as PNGs into the directory passed by the caller.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# --------------------------- portfolio palette ------------------------------ #

BG_PANEL = "#0d1220"
BG_AXES = "#0c101c"
FG_0 = "#e7ecf5"
FG_1 = "#9aa3b8"
GRID = (1.0, 1.0, 1.0, 0.05)
SPINE = (1.0, 1.0, 1.0, 0.20)

PURPLE = "#7c5cff"   # anomaly score / model output
CYAN = "#22d3ee"     # ground truth / healthy reference
PINK = "#f472b6"     # threshold / decision boundary
AMBER = "#fbbf24"    # warning region / anomalous samples
GREEN = "#34d399"    # nominal / true negatives

SANS = ["DejaVu Sans", "Inter", "Segoe UI", "Arial"]
MONO = ["DejaVu Sans Mono", "JetBrains Mono", "Consolas", "monospace"]


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": BG_PANEL,
        "savefig.facecolor": BG_PANEL,
        "axes.facecolor": BG_AXES,
        "axes.edgecolor": SPINE,
        "axes.labelcolor": FG_0,
        "axes.titlecolor": FG_0,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 11.5,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "xtick.color": FG_1,
        "ytick.color": FG_1,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "text.color": FG_0,
        "legend.frameon": True,
        "legend.facecolor": BG_PANEL,
        "legend.edgecolor": SPINE,
        "legend.labelcolor": FG_0,
        "legend.fontsize": 10,
        "font.family": SANS,
        "mathtext.fontset": "cm",
        "savefig.dpi": 170,
        "savefig.bbox": "tight",
    })


def _save(fig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  1. raw vs normalised sensor curves (data overview)
# --------------------------------------------------------------------------- #

def plot_sensor_overview(
    raw_df, norm_df, sensor_cols: list[str], unit: int, out_path: str | Path,
    n_panels: int = 4,
) -> None:
    """Top: raw sensor traces along one engine's life.
    Bottom: same traces after per-sensor z-score normalisation.
    """
    import matplotlib.pyplot as plt
    _style()

    chosen = sensor_cols[:n_panels]
    raw = raw_df[raw_df["unit"] == unit].sort_values("cycle")
    norm = norm_df[norm_df["unit"] == unit].sort_values("cycle")

    fig, axes = plt.subplots(2, 1, figsize=(9.4, 5.2), sharex=True)
    palette = [PURPLE, CYAN, PINK, AMBER, GREEN]

    ax = axes[0]
    for col, color in zip(chosen, palette):
        ax.plot(raw["cycle"], raw[col], color=color, lw=1.2, alpha=0.95, label=col)
    ax.set_ylabel("raw reading")
    ax.set_title(f"engine #{unit} · raw sensor traces ({len(chosen)} of {len(sensor_cols)})")
    ax.legend(ncol=len(chosen), loc="upper center", bbox_to_anchor=(0.5, -0.04), fontsize=9)

    ax = axes[1]
    for col, color in zip(chosen, palette):
        ax.plot(norm["cycle"], norm[col], color=color, lw=1.2, alpha=0.95)
    ax.axhline(0, color=FG_1, lw=0.6, ls="--", alpha=0.5)
    ax.set_xlabel("operational cycle")
    ax.set_ylabel("z-score")
    ax.set_title("after per-sensor normalisation (statistics fit on healthy training rows)")

    fig.tight_layout()
    _save(fig, out_path)


# --------------------------------------------------------------------------- #
#  2. training and validation loss
# --------------------------------------------------------------------------- #

def plot_training_loss(
    train_loss: list[float], val_loss: list[float], out_path: str | Path,
) -> None:
    import matplotlib.pyplot as plt
    _style()

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    epochs = np.arange(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, color=CYAN, lw=1.6, label="train")
    ax.plot(epochs, val_loss, color=PURPLE, lw=1.6, label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE on healthy windows")
    ax.set_title("autoencoder training · CMAPSS FD001 healthy windows")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, out_path)


# --------------------------------------------------------------------------- #
#  3. error distributions, healthy vs anomalous, with threshold
# --------------------------------------------------------------------------- #

def plot_error_distributions(
    err_test_healthy: np.ndarray, err_test_anomalous: np.ndarray,
    threshold: float, out_path: str | Path,
) -> None:
    import matplotlib.pyplot as plt
    _style()

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    upper = float(np.quantile(err_test_anomalous, 0.97))
    bins = np.linspace(0, upper, 70)
    ax.hist(err_test_healthy, bins=bins, color=CYAN, alpha=0.55,
            edgecolor="none", label=f"healthy (RUL > 30) · n={len(err_test_healthy)}")
    ax.hist(err_test_anomalous, bins=bins, color=AMBER, alpha=0.7,
            edgecolor="none", label=f"anomalous (RUL ≤ 30) · n={len(err_test_anomalous)}")
    ax.axvline(threshold, color=PINK, lw=1.6, ls="--",
               label=f"threshold = {threshold:.3f}")
    ax.set_yscale("log")
    ax.set_xlabel("reconstruction MSE per window")
    ax.set_ylabel("count (log scale)")
    ax.set_title("test reconstruction error · healthy vs anomalous")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, out_path)


# --------------------------------------------------------------------------- #
#  4. ROC curve
# --------------------------------------------------------------------------- #

def plot_roc(
    fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, out_path: str | Path,
    operating_point: tuple[float, float] | None = None,
) -> None:
    import matplotlib.pyplot as plt
    _style()

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot([0, 1], [0, 1], color=FG_1, lw=0.8, ls="--", alpha=0.6,
            label="random classifier")
    ax.plot(fpr, tpr, color=PURPLE, lw=2.0, label=f"autoencoder (AUC = {roc_auc:.3f})")
    if operating_point is not None:
        ax.plot([operating_point[0]], [operating_point[1]], marker="o",
                color=PINK, ms=9, label=f"q99 threshold")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC · anomaly score vs RUL≤30 label")
    ax.legend(loc="lower right")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    _save(fig, out_path)


# --------------------------------------------------------------------------- #
#  5. confusion matrix, normalised by row
# --------------------------------------------------------------------------- #

def plot_confusion(
    confusion: dict[str, int], out_path: str | Path,
) -> None:
    import matplotlib.pyplot as plt
    _style()

    matrix = np.array([
        [confusion["tn"], confusion["fp"]],
        [confusion["fn"], confusion["tp"]],
    ], dtype=np.float64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    rates = matrix / np.maximum(row_sums, 1)

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    im = ax.imshow(rates, cmap="magma", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i,
                    f"{int(matrix[i, j])}\n({rates[i, j] * 100:.1f}%)",
                    ha="center", va="center",
                    color="white" if rates[i, j] < 0.6 else "black",
                    fontsize=12, family=MONO[0])
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred healthy", "pred anomalous"])
    ax.set_yticklabels(["true healthy", "true anomalous"])
    ax.set_title("confusion matrix · row-normalised")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row fraction")
    fig.tight_layout()
    _save(fig, out_path)


# --------------------------------------------------------------------------- #
#  6. per-engine error trajectory (illustrative)
# --------------------------------------------------------------------------- #

def plot_engine_trajectory(
    err_test: np.ndarray, rul_test: np.ndarray, unit_test: np.ndarray,
    threshold: float, anomaly_cutoff: int, out_path: str | Path,
    units: tuple[int, ...] = (81, 95),
) -> None:
    """Two engines: reconstruction error vs RUL with the threshold drawn."""
    import matplotlib.pyplot as plt
    _style()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    palette = [PURPLE, AMBER]
    for ax, u, color in zip(axes, units, palette):
        mask = unit_test == u
        order = np.argsort(-rul_test[mask])  # left → right = early to late life
        rul_u = rul_test[mask][order]
        err_u = err_test[mask][order]
        cycle_idx = np.arange(len(rul_u))
        ax.plot(cycle_idx, err_u, color=color, lw=1.3,
                label=f"engine #{u}")
        ax.axhline(threshold, color=PINK, ls="--", lw=1.0, alpha=0.9,
                   label=f"threshold {threshold:.3f}")
        # Mark the cutoff between healthy and anomalous regions.
        cutoff_idx = np.argmin(np.abs(rul_u - anomaly_cutoff))
        ax.axvspan(cutoff_idx, len(rul_u) - 1, color=AMBER, alpha=0.10,
                   label=f"RUL ≤ {anomaly_cutoff}")
        ax.set_xlabel("window index along trajectory (early → late)")
        ax.legend(loc="upper left", fontsize=9)
    axes[0].set_ylabel("reconstruction MSE")
    fig.suptitle("per-engine reconstruction error grows as failure approaches",
                 color=FG_0, fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_path)


__all__ = [
    "plot_confusion",
    "plot_engine_trajectory",
    "plot_error_distributions",
    "plot_roc",
    "plot_sensor_overview",
    "plot_training_loss",
]
