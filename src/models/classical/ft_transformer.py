"""
ft_transformer.py
=================
Feature Tokenizer + Transformer (FT-Transformer) for tabular binary classification.

Architecture (Gorishniy et al., 2021 — "Revisiting Deep Learning Models for Tabular Data"):
  1. Feature Tokenizer: each numerical feature xi → Wi * xi + bi ∈ R^d_token
  2. Prepend a learnable [CLS] token
  3. n_blocks × Transformer block (pre-norm: LayerNorm before attention and FFN)
  4. Read out [CLS] token → LayerNorm → Linear(d_token, 1) → Sigmoid

Selected as a third classical baseline because:
- Self-attention over features is robust to small feature counts (unlike TabNet's
  sequential selection, which degrades on PCA-compressed input)
- Outperforms TabNet on most tabular benchmarks (Gorishniy et al., 2021)
- Inter-feature attention provides a classical analogue to quantum entanglement-based
  feature correlation, enabling a stronger theoretical comparison
"""

from __future__ import annotations

import math

import torch
from torch import nn

from src.config import FTTransformerConfig


class _FeatureTokenizer(nn.Module):
    """Map (batch, n_features) → (batch, n_features, d_token).

    Each feature gets an independent linear projection:
        token_i = weight_i * x_i + bias_i
    """

    def __init__(self, n_features: int, d_token: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(n_features) if n_features > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features) → (batch, n_features, d_token)
        return x.unsqueeze(-1) * self.weight + self.bias


class _TransformerBlock(nn.Module):
    """Single pre-norm Transformer block: attention + FFN with residual connections."""

    def __init__(
        self,
        d_token: int,
        n_heads: int,
        d_ffn: int,
        attn_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_ffn),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_ffn, d_token),
        )
        self.dropout = nn.Dropout(residual_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.dropout(attn_out)
        # Pre-norm FFN
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class FTTransformer(nn.Module):
    """
    Feature Tokenizer + Transformer for binary classification.

    Input:  (batch, input_dim)
    Output: (batch, 1) — probability of fraud
    """

    def __init__(self, input_dim: int, cfg: FTTransformerConfig) -> None:
        super().__init__()
        d = cfg.d_token
        d_ffn = max(1, int(d * cfg.ffn_d_hidden_factor))

        self.tokenizer = _FeatureTokenizer(input_dim, d)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d))
        nn.init.normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([
            _TransformerBlock(
                d_token=d,
                n_heads=cfg.n_heads,
                d_ffn=d_ffn,
                attn_dropout=cfg.attn_dropout,
                ffn_dropout=cfg.ffn_dropout,
                residual_dropout=cfg.residual_dropout,
            )
            for _ in range(cfg.n_blocks)
        ])

        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)                          # (batch, n_feat, d)
        cls = self.cls_token.expand(x.size(0), -1, -1)     # (batch, 1, d)
        tokens = torch.cat([cls, tokens], dim=1)            # (batch, n_feat+1, d)

        for block in self.blocks:
            tokens = block(tokens)

        cls_out = self.norm(tokens[:, 0])                   # (batch, d)
        return torch.sigmoid(self.head(cls_out))            # (batch, 1)

    def param_count(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"classical": total, "quantum": 0, "total": total}
