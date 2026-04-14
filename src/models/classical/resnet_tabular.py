"""
resnet_tabular.py
=================
ResNet for tabular data (Gorishniy et al., 2021 — "Revisiting Deep Learning Models
for Tabular Data").

Architecture:
  Input → Linear(input_dim, d) →
  [ResBlock: LayerNorm → Linear(d, d_hidden) → GELU → Dropout
             → Linear(d_hidden, d) → Dropout → skip] × n_blocks →
  LayerNorm → Linear(d, 1) → Sigmoid

Key distinction from SNN: residual (skip) connections with LayerNorm replace
SELU self-normalization. Key distinction from FT-Transformer: a single joint
input projection replaces per-feature tokenization — all features are embedded
together rather than independently.
"""

from __future__ import annotations

import torch
from torch import nn

from src.config import ResNetConfig


class _ResBlock(nn.Module):
    """Pre-norm residual block: LayerNorm → Linear → GELU → Dropout → Linear → Dropout → skip."""

    def __init__(self, d: int, d_hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.linear1 = nn.Linear(d, d_hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_hidden, d)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.linear1(h)
        h = self.act(h)
        h = self.drop1(h)
        h = self.linear2(h)
        h = self.drop2(h)
        return x + h


class ResNet(nn.Module):
    """
    ResNet for tabular binary classification.

    Input:  (batch, input_dim)
    Output: (batch, 1) — probability of fraud
    """

    def __init__(self, input_dim: int, cfg: ResNetConfig) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, cfg.d)
        self.blocks = nn.ModuleList([
            _ResBlock(cfg.d, cfg.d_hidden, cfg.dropout)
            for _ in range(cfg.n_blocks)
        ])
        self.norm = nn.LayerNorm(cfg.d)
        self.head = nn.Linear(cfg.d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return torch.sigmoid(self.head(x))

    def param_count(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"classical": total, "quantum": 0, "total": total}
