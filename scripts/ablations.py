"""Architecture ablation: the alternatives discussed in README section 7.

Every model is a reconstruction autoencoder with the same 16-dimensional
latent, trained on the same healthy windows with the same optimiser,
schedule and budget as the main LSTM (Adam, lr 2e-3, cosine, 80 epochs,
batch 256, grad-clip 1). Each one is trained with 3 seeds; the score is
the per-window reconstruction MSE, evaluated on the same test windows.

* ``lstm``        — the project's LSTMAutoencoder (retrained per seed).
* ``mlp``         — MLP on the flattened 30 x 14 window (420-128-16).
* ``cnn``         — two 1D convolutions over time, then a linear bottleneck.
* ``transformer`` — one self-attention encoder layer, mean-pooled to the
  latent; one attention layer decodes from the latent plus positions.
* ``vae``         — the LSTM autoencoder with a Gaussian latent; loss is
  the negative ELBO under a unit-variance Gaussian likelihood
  (MSE + 2 KL / (T F)); scored with the latent mean.

Usage:
    python -m scripts.ablations          # writes data/ablations.json (~20 min CPU)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.baselines import _ranking
from src.autoencoder import AEConfig, LSTMAutoencoder, count_parameters
from src.data import build_dataset
from src.train import _to_loader, reconstruction_errors

LATENT = 16


class _ReconMixin:
    aux = None   # extra loss term set by forward (VAE only)

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self(x)) ** 2).mean(dim=(1, 2))


class MLPAutoencoder(_ReconMixin, nn.Module):
    def __init__(self, seq_len: int, n_features: int, hidden: int = 128):
        super().__init__()
        d = seq_len * n_features
        self.shape = (seq_len, n_features)
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, LATENT),
            nn.Linear(LATENT, hidden), nn.ReLU(), nn.Linear(hidden, d),
        )

    def forward(self, x):
        return self.net(x).view(-1, *self.shape)


class CNNAutoencoder(_ReconMixin, nn.Module):
    def __init__(self, seq_len: int, n_features: int, ch: int = 32):
        super().__init__()
        self.seq_len, self.ch = seq_len, ch
        self.enc = nn.Sequential(
            nn.Conv1d(n_features, ch, 5, padding=2), nn.ReLU(),
            nn.Conv1d(ch, ch, 5, padding=2), nn.ReLU(),
            nn.Flatten(), nn.Linear(ch * seq_len, LATENT),
        )
        self.dec_lin = nn.Linear(LATENT, ch * seq_len)
        self.dec = nn.Sequential(
            nn.ReLU(), nn.Conv1d(ch, ch, 5, padding=2), nn.ReLU(),
            nn.Conv1d(ch, n_features, 5, padding=2),
        )

    def forward(self, x):                       # x: (B, T, F)
        z = self.enc(x.transpose(1, 2))
        h = self.dec_lin(z).view(-1, self.ch, self.seq_len)
        return self.dec(h).transpose(1, 2)


class TransformerAutoencoder(_ReconMixin, nn.Module):
    def __init__(self, seq_len: int, n_features: int, d: int = 64, heads: int = 4):
        super().__init__()
        self.inp = nn.Linear(n_features, d)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d) * 0.02)
        layer = lambda: nn.TransformerEncoderLayer(          # noqa: E731
            d, heads, dim_feedforward=2 * d, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer(), 1)
        self.to_latent = nn.Linear(d, LATENT)
        self.from_latent = nn.Linear(LATENT, d)
        self.dec = nn.TransformerEncoder(layer(), 1)
        self.out = nn.Linear(d, n_features)

    def forward(self, x):
        h = self.enc(self.inp(x) + self.pos)
        z = self.to_latent(h.mean(dim=1))
        h = self.from_latent(z).unsqueeze(1) + self.pos
        return self.out(self.dec(h))


class LSTMVAE(_ReconMixin, LSTMAutoencoder):
    def __init__(self, cfg: AEConfig):
        super().__init__(cfg)
        self.to_logvar = nn.Linear(cfg.hidden, cfg.latent_dim)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        mu, logvar = self.to_latent(h[-1]), self.to_logvar(h[-1])
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            kl = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1).mean()
            self.aux = 2.0 * kl / (x.shape[1] * x.shape[2])
        else:
            z = mu
        return self.decode(z)


def make_model(kind: str, seq_len: int, n_features: int) -> nn.Module:
    cfg = AEConfig(n_features=n_features, seq_len=seq_len)
    return {
        "lstm": lambda: LSTMAutoencoder(cfg),
        "mlp": lambda: MLPAutoencoder(seq_len, n_features),
        "cnn": lambda: CNNAutoencoder(seq_len, n_features),
        "transformer": lambda: TransformerAutoencoder(seq_len, n_features),
        "vae": lambda: LSTMVAE(cfg),
    }[kind]()


def fit(model: nn.Module, x_train: np.ndarray, *, epochs: int, seed: int,
        lr: float = 2e-3, batch_size: int = 256) -> float:
    """Same recipe as src.train.train (Adam + cosine + clip); returns seconds."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = _to_loader(x_train, batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for (xb,) in loader:
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(xb), xb)
            if getattr(model, "aux", None) is not None:
                loss = loss + model.aux
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    model.eval()
    return time.time() - t0


def main(raw_dir: str, out: str, kinds: list[str], seeds: list[int], epochs: int) -> None:
    torch.set_num_threads(max(1, torch.get_num_threads()))
    ds = build_dataset(raw_dir=raw_dir)
    y = ds.test_label
    results = {}
    for kind in kinds:
        runs = []
        for seed in seeds:
            torch.manual_seed(seed)
            model = make_model(kind, ds.seq_len, ds.train.n_features)
            secs = fit(model, ds.train.x, epochs=epochs, seed=seed)
            roc, ap = _ranking(reconstruction_errors(model, ds.test.x), y)
            runs.append({"seed": seed, "roc_auc": roc, "average_precision": ap,
                         "train_seconds": secs})
            print(f"  {kind:<12} seed {seed}  ROC-AUC {roc:.4f}  AP {ap:.4f}  "
                  f"{secs:5.0f} s", flush=True)
        r = np.array([x["roc_auc"] for x in runs])
        a = np.array([x["average_precision"] for x in runs])
        results[kind] = {
            "n_params": count_parameters(model),
            "roc_auc_mean": float(r.mean()), "roc_auc_std": float(r.std(ddof=1)),
            "ap_mean": float(a.mean()), "ap_std": float(a.std(ddof=1)),
            "train_seconds_mean": float(np.mean([x["train_seconds"] for x in runs])),
            "runs": runs,
        }
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:            # rewrite after each model
            json.dump({"epochs": epochs, "seeds": seeds, "results": results}, fh, indent=2)
    print(f"[ok] {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out", default="data/ablations.json")
    p.add_argument("--models", nargs="+",
                   default=["lstm", "mlp", "cnn", "transformer", "vae"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=80)
    a = p.parse_args()
    main(a.raw_dir, a.out, a.models, a.seeds, a.epochs)
