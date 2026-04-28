"""LSTM autoencoder for fixed-length multivariate sensor windows.

The encoder reads a `(seq_len, n_features)` window into a single hidden
state, projects it to a low-dimensional latent code, and the decoder
generates the sequence back step-by-step from that code. Trained on
healthy windows only, the reconstruction error becomes the anomaly
score at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class AEConfig:
    n_features: int
    seq_len: int
    hidden: int = 64
    latent_dim: int = 16
    num_layers: int = 1
    dropout: float = 0.0


class LSTMAutoencoder(nn.Module):
    """Sequence-to-sequence LSTM autoencoder.

    The encoder consumes the input window and the final hidden state of
    the top layer is projected through a Linear bottleneck to a latent
    code ``z``. The decoder uses ``z`` (broadcast across time) as input
    at every step, conditioned on a learnable initial hidden state seeded
    from ``z`` itself.
    """

    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.LSTM(
            input_size=cfg.n_features,
            hidden_size=cfg.hidden,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.to_latent = nn.Linear(cfg.hidden, cfg.latent_dim)
        self.from_latent = nn.Linear(cfg.latent_dim, cfg.hidden)
        self.decoder = nn.LSTM(
            input_size=cfg.latent_dim,
            hidden_size=cfg.hidden,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(cfg.hidden, cfg.n_features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        return self.to_latent(h[-1])

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        batch = z.shape[0]
        h0 = self.from_latent(z).unsqueeze(0).repeat(self.cfg.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        z_seq = z.unsqueeze(1).repeat(1, self.cfg.seq_len, 1)
        out, _ = self.decoder(z_seq, (h0.contiguous(), c0.contiguous()))
        return self.head(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-window MSE between input and reconstruction."""
        x_hat = self.forward(x)
        return ((x - x_hat) ** 2).mean(dim=(1, 2))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["AEConfig", "LSTMAutoencoder", "count_parameters"]
