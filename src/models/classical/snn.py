"""
snn.py
======
Self-Normalizing Network (SNN) with SELU activation.

Uses SELU + AlphaDropout to maintain self-normalizing properties throughout
the network depth, providing the most stable gradient baseline for
high-dimensional tabular data (Klambauer et al., 2017).

Weights are initialized with Kaiming normal (fan_in) which is the correct
initialization scheme for SELU activations.
"""

from __future__ import annotations

import torch
from torch import nn

from src.config import SNNConfig


class SNN(nn.Module):
    """
    Self-Normalizing Neural Network for binary classification.

    Input:  (batch, input_dim)
    Output: (batch, 1) — probability of fraud
    """

    def __init__(self, input_dim: int, cfg: SNNConfig):
        super().__init__()
        self.cfg = cfg

        layers: list[nn.Module] = []
        in_dim = input_dim
        for out_dim in cfg.hidden_dims:
            linear = nn.Linear(in_dim, out_dim)
            # Kaiming normal init for SELU (fan_in, linear nonlinearity)
            nn.init.kaiming_normal_(linear.weight, mode="fan_in", nonlinearity="linear")
            nn.init.zeros_(linear.bias)
            layers.extend([
                linear,
                nn.SELU(),
                nn.AlphaDropout(cfg.alpha_dropout),
            ])
            in_dim = out_dim

        # Output layer
        out_linear = nn.Linear(in_dim, 1)
        nn.init.kaiming_normal_(out_linear.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(out_linear.bias)
        layers.append(out_linear)
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def param_count(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"classical": total, "quantum": 0, "total": total}
