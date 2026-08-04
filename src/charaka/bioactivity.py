"""Bioactivity surrogate: deep-ensemble MLP over the latent space.

The surrogate predicts ``log10(IC50)`` (nanomolar) from the encoder
mean ``mu``. Five members trained with distinct random seeds form an
ensemble whose disagreement is used as an epistemic uncertainty
signal in :mod:`charaka.nsga`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn


POTENCY_CENTRE = 3.0
POTENCY_SLOPE = 1.0


class BioactivityMLP(nn.Module):
    """4-layer MLP: ``latent_dim -> hidden -> hidden -> 1``."""

    def __init__(
        self,
        latent_dim: int = 256,
        hidden: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        return self.net(mu).squeeze(-1)


class BioactivityEnsemble(nn.Module):
    """Deep ensemble of :class:`BioactivityMLP` members."""

    def __init__(
        self,
        n_members: int = 5,
        latent_dim: int = 256,
        hidden: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.members = nn.ModuleList(
            BioactivityMLP(latent_dim, hidden, dropout) for _ in range(n_members)
        )

    def forward(self, mu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack([m(mu) for m in self.members], dim=0)
        return stacked.mean(dim=0), stacked.std(dim=0)

    def load_members(self, paths: Iterable[Path], device: torch.device) -> None:
        for member, path in zip(self.members, paths):
            member.load_state_dict(torch.load(path, map_location=device))


def potency_norm(log_ic50: torch.Tensor) -> torch.Tensor:
    """Logistic squash centred at 1 uM (log10(IC50) = 3)."""
    return torch.sigmoid(-(log_ic50 - POTENCY_CENTRE) * POTENCY_SLOPE)


@torch.no_grad()
def predict_potency(
    ensemble: BioactivityEnsemble,
    mu: torch.Tensor,
) -> Tuple[float, float, float]:
    """Returns ``(predicted_log10_IC50, potency_norm, ensemble_std)``.

    ``mu`` is a 1-D latent vector of length ``latent_dim``.
    """
    ensemble.eval()
    pred, std = ensemble(mu.unsqueeze(0))
    norm = potency_norm(pred)
    return float(pred.item()), float(norm.item()), float(std.item())
