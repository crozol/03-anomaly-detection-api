"""Streamlit demo for the CMAPSS anomaly-detection autoencoder.

Two ways to use it:

* upload a CSV with the 14 informative sensor columns and the demo will
  build sliding windows, score each one, and overlay the flagged regions
  on the input series;
* press the "Use a built-in CMAPSS engine" button to score a random
  engine from the test split — useful for portfolio demos when the user
  has no CSV at hand.

Launch with:

    streamlit run src/app_streamlit.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = os.environ.get("MODEL_PATH", str(ROOT / "checkpoints" / "autoencoder.pt"))
EVAL_PATH = os.environ.get("THRESHOLD_PATH", str(ROOT / "data" / "eval_report.json"))


# --------------------------------------------------------------------------- #
#  cached resources
# --------------------------------------------------------------------------- #

@st.cache_resource
def _load_model_and_threshold():
    from src.train import load_checkpoint
    import json

    if not Path(MODEL_PATH).exists():
        return None, None
    model, payload = load_checkpoint(MODEL_PATH)
    threshold = float("nan")
    if Path(EVAL_PATH).exists():
        with open(EVAL_PATH) as fh:
            threshold = float(json.load(fh)["threshold"])
    return (model, payload), threshold


# --------------------------------------------------------------------------- #
#  scoring
# --------------------------------------------------------------------------- #

def score_dataframe(
    df: pd.DataFrame, sensor_cols: list[str], mean: np.ndarray, std: np.ndarray,
    seq_len: int, model, stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Window the DataFrame and return ``(window_end_index, score)`` arrays."""
    arr = df[sensor_cols].to_numpy(dtype=np.float32)
    if arr.shape[0] < seq_len:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    arr_norm = (arr - mean) / std
    starts = np.arange(0, arr.shape[0] - seq_len + 1, stride, dtype=np.int64)
    xs = np.stack([arr_norm[s : s + seq_len] for s in starts], axis=0)

    with torch.no_grad():
        x = torch.from_numpy(xs)
        x_hat = model(x)
        err = ((x - x_hat) ** 2).mean(dim=(1, 2)).cpu().numpy()
    end_idx = starts + seq_len - 1
    return end_idx, err


def _builtin_engine(payload: dict) -> pd.DataFrame:
    """Pick a random test engine from FD001 raw data and return its DataFrame."""
    from src.data import load_split, INFORMATIVE_SENSORS_FD001
    df = load_split("FD001", "train", raw_dir=str(ROOT / "data" / "raw"))
    test_units = list(range(81, 101))
    unit = int(np.random.choice(test_units))
    sub = df[df["unit"] == unit].sort_values("cycle").reset_index(drop=True)
    sensor_cols = [f"sensor_{i}" for i in INFORMATIVE_SENSORS_FD001]
    return sub[["unit", "cycle"] + sensor_cols].rename(columns={"cycle": "time"})


# --------------------------------------------------------------------------- #
#  layout
# --------------------------------------------------------------------------- #

def main() -> None:
    st.set_page_config(
        page_title="CMAPSS Anomaly Detector",
        page_icon=":chart_with_downwards_trend:",
        layout="wide",
    )
    st.title("CMAPSS · Anomaly Detection Demo")
    st.caption(
        "LSTM autoencoder trained on healthy CMAPSS FD001 windows. "
        "Upload a CSV with the 14 informative sensor columns or use a "
        "built-in test engine."
    )

    bundle, threshold = _load_model_and_threshold()
    if bundle is None:
        st.error(
            "Trained checkpoint not found. Run `python main.py` to produce "
            "checkpoints/autoencoder.pt and data/eval_report.json first."
        )
        return
    model, payload = bundle
    sensor_cols = list(payload["sensor_cols"])
    mean = payload["mean"].astype(np.float32)
    std = payload["std"].astype(np.float32)
    seq_len = int(payload["seq_len"])

    with st.sidebar:
        st.markdown("### Model")
        st.write(f"**seq_len** = `{seq_len}`")
        st.write(f"**n_features** = `{len(sensor_cols)}`")
        st.write(f"**threshold** = `{threshold:.4f}`")
        st.markdown("**sensor columns**:")
        st.code("\n".join(sensor_cols))

    col1, col2 = st.columns([2, 1])
    with col2:
        use_builtin = st.button("Use a built-in CMAPSS engine", type="primary")
    with col1:
        uploaded = st.file_uploader("CSV with the sensor columns", type=["csv"])

    df: pd.DataFrame | None = None
    title_suffix = ""
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        title_suffix = f" · uploaded `{uploaded.name}`"
    elif use_builtin:
        df = _builtin_engine(payload)
        title_suffix = f" · built-in engine #{int(df['unit'].iloc[0])}"

    if df is None:
        st.info("Waiting for input… upload a CSV or click the built-in button above.")
        return

    missing = [c for c in sensor_cols if c not in df.columns]
    if missing:
        st.error(
            f"CSV is missing the following sensor columns: {missing[:5]}"
            + (" …" if len(missing) > 5 else "")
        )
        return

    end_idx, scores = score_dataframe(df, sensor_cols, mean, std, seq_len, model)
    if scores.size == 0:
        st.error(f"need at least {seq_len} rows, CSV has {len(df)}")
        return
    flagged = scores > threshold
    n_flagged = int(flagged.sum())

    metric_cols = st.columns(4)
    metric_cols[0].metric("rows", f"{len(df):,}")
    metric_cols[1].metric("windows scored", f"{len(scores):,}")
    metric_cols[2].metric("flagged", f"{n_flagged:,}",
                          f"{100 * n_flagged / max(len(scores), 1):.1f}%")
    metric_cols[3].metric("max score", f"{scores.max():.3f}")

    st.subheader("Reconstruction error along the trajectory" + title_suffix)
    score_df = pd.DataFrame({
        "window_end": end_idx,
        "score": scores,
        "threshold": threshold,
        "is_anomaly": flagged,
    })
    st.line_chart(
        score_df.set_index("window_end")[["score", "threshold"]],
        use_container_width=True,
    )

    st.subheader("Sensor traces with flagged windows highlighted")
    chosen = st.multiselect(
        "Sensors to plot", sensor_cols, default=sensor_cols[:3])
    if chosen:
        plot_df = df[chosen].reset_index().rename(columns={"index": "row"})
        st.line_chart(plot_df.set_index("row"), use_container_width=True)
        if n_flagged:
            spans = score_df[score_df["is_anomaly"]]["window_end"].tolist()
            st.warning(
                f"{n_flagged} windows flagged · first at row "
                f"{spans[0]}, last at row {spans[-1]}"
            )

    with st.expander("Raw scores (window_end, score, is_anomaly)"):
        st.dataframe(score_df, use_container_width=True)


if __name__ == "__main__":
    main()
