"""Training loop for the LSTM autoencoder on CMAPSS healthy windows.

The procedure is the standard reconstruction-only objective: minimise
MSE between input window and reconstruction on the *healthy* training
set. The validation loss is monitored on a disjoint set of healthy
windows held out engine-wise.

A checkpoint plus the per-epoch loss curve are written to disk so that
the threshold-calibration stage and the API can both load the same
weights without re-training.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .autoencoder import AEConfig, LSTMAutoencoder, count_parameters
from .data import Dataset, build_dataset


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    cosine_t_max: int | None = None        # defaults to epochs
    seed: int = 0
    device: str = "cpu"
    log_every: int = 1


@dataclass
class TrainResult:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    final_train_loss: float = float("nan")
    final_val_loss: float = float("nan")
    epochs: int = 0
    n_params: int = 0
    seconds: float = 0.0


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #

def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _to_loader(x: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor = torch.from_numpy(np.asarray(x, dtype=np.float32))
    return DataLoader(
        TensorDataset(tensor), batch_size=batch_size, shuffle=shuffle, drop_last=False,
    )


# --------------------------------------------------------------------------- #
#  training and evaluation
# --------------------------------------------------------------------------- #

def train(
    model: LSTMAutoencoder,
    ds: Dataset,
    cfg: TrainConfig = TrainConfig(),
) -> TrainResult:
    """Fit the autoencoder on healthy training windows, monitor val loss."""
    _seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    model.to(device)

    train_loader = _to_loader(ds.train.x, cfg.batch_size, shuffle=True)
    val_loader = _to_loader(ds.val.x, cfg.batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.cosine_t_max or cfg.epochs,
    )
    loss_fn = nn.MSELoss()

    out = TrainResult(n_params=count_parameters(model))
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss, n_seen = 0.0, 0
        for (xb,) in train_loader:
            xb = xb.to(device)
            opt.zero_grad(set_to_none=True)
            x_hat = model(xb)
            loss = loss_fn(x_hat, xb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            epoch_loss += loss.item() * xb.shape[0]
            n_seen += xb.shape[0]
        sched.step()
        train_loss = epoch_loss / max(n_seen, 1)

        model.eval()
        with torch.no_grad():
            v_loss, v_seen = 0.0, 0
            for (xb,) in val_loader:
                xb = xb.to(device)
                x_hat = model(xb)
                v_loss += loss_fn(x_hat, xb).item() * xb.shape[0]
                v_seen += xb.shape[0]
            val_loss = v_loss / max(v_seen, 1)

        out.train_loss.append(train_loss)
        out.val_loss.append(val_loss)
        if cfg.log_every and epoch % cfg.log_every == 0:
            print(
                f"  epoch {epoch:>3d}/{cfg.epochs} | "
                f"train {train_loss:.5f} | val {val_loss:.5f} | "
                f"lr {sched.get_last_lr()[0]:.2e}"
            )

    out.final_train_loss = out.train_loss[-1] if out.train_loss else float("nan")
    out.final_val_loss = out.val_loss[-1] if out.val_loss else float("nan")
    out.epochs = cfg.epochs
    out.seconds = time.time() - t0
    return out


def reconstruction_errors(
    model: LSTMAutoencoder, x: np.ndarray, batch_size: int = 256, device: str = "cpu",
) -> np.ndarray:
    """Per-window mean-squared reconstruction error on a stack of windows."""
    model.eval()
    out = []
    dev = torch.device(device)
    model.to(dev)
    loader = _to_loader(x, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(dev)
            err = model.reconstruction_error(xb)
            out.append(err.cpu().numpy())
    return np.concatenate(out, axis=0)


# --------------------------------------------------------------------------- #
#  checkpoint serialisation
# --------------------------------------------------------------------------- #

def save_checkpoint(
    path: str | Path,
    model: LSTMAutoencoder,
    ds: Dataset,
    train_result: TrainResult,
    train_cfg: TrainConfig,
) -> None:
    """Persist weights + dataset normalisation + training metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "model_cfg": asdict(model.cfg),
        "mean": ds.mean,
        "std": ds.std,
        "sensor_cols": ds.sensor_cols,
        "seq_len": ds.seq_len,
        "healthy_cutoff": ds.healthy_cutoff,
        "anomaly_cutoff": ds.anomaly_cutoff,
        "train_result": asdict(train_result),
        "train_cfg": asdict(train_cfg),
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[LSTMAutoencoder, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = AEConfig(**payload["model_cfg"])
    model = LSTMAutoencoder(cfg)
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, payload


# --------------------------------------------------------------------------- #
#  small CLI used by main.py
# --------------------------------------------------------------------------- #

def fit_pipeline(
    raw_dir: str = "data/raw",
    checkpoint_path: str = "checkpoints/autoencoder.pt",
    train_cfg: TrainConfig = TrainConfig(),
    ae_cfg: AEConfig | None = None,
    metrics_path: str | None = "data/train_metrics.json",
) -> tuple[LSTMAutoencoder, Dataset, TrainResult]:
    print("[train] building dataset …")
    ds = build_dataset(raw_dir=raw_dir)
    if ae_cfg is None:
        ae_cfg = AEConfig(n_features=ds.train.n_features, seq_len=ds.seq_len)
    model = LSTMAutoencoder(ae_cfg)
    print(f"[train] model has {count_parameters(model):,} trainable parameters")
    print(f"[train] training for {train_cfg.epochs} epochs on {train_cfg.device}")
    result = train(model, ds, train_cfg)
    save_checkpoint(checkpoint_path, model, ds, result, train_cfg)
    if metrics_path:
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as fh:
            json.dump({
                "epochs": result.epochs,
                "n_params": result.n_params,
                "final_train_loss": result.final_train_loss,
                "final_val_loss": result.final_val_loss,
                "seconds": result.seconds,
                "train_loss": result.train_loss,
                "val_loss": result.val_loss,
            }, fh, indent=2)
    return model, ds, result


__all__ = [
    "TrainConfig",
    "TrainResult",
    "fit_pipeline",
    "load_checkpoint",
    "reconstruction_errors",
    "save_checkpoint",
    "train",
]
