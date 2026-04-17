"""Autoencoder para detección de anomalías en series multivariadas."""

from __future__ import annotations

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    """Autoencoder LSTM-based para secuencias de longitud fija."""

    def __init__(self, n_features: int, seq_len: int, latent_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        z = self.to_latent(h[-1])
        h0 = self.from_latent(z).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        repeat = h0.squeeze(0).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.decoder(repeat, (h0, c0))
        return self.head(out)


def reconstruction_error(model: LSTMAutoencoder, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        x_hat = model(x)
    return ((x - x_hat) ** 2).mean(dim=(1, 2))
