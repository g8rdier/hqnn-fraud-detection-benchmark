"""
saint.py
========
SAINT: Self-Attention and Intersample Attention Transformer for tabular data.
(Somepalli et al., 2021 — https://arxiv.org/abs/2106.01342)

Architecture
------------
Input (B, F)
  → _FeatureTokenizer  → (B, F, D)
  → prepend CLS token  → (B, F+1, D)
  → n × _SAINTBlock
      • column (feature) self-attention    — attends over F+1 tokens per sample
      • row (intersample) attention        — attends over B samples per token
      • FFN after each, all pre-norm + residual
  → LayerNorm → CLS readout (B, D) → Linear(D,1) → Sigmoid
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.config import SAINTConfig


# ── Shared building blocks ────────────────────────────────────────────────────


class _FeatureTokenizer(nn.Module):
    """Per-feature linear projection: x_i → W_i * x_i + b_i."""

    def __init__(self, input_dim: int, d: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, d))
        self.bias = nn.Parameter(torch.zeros(input_dim, d))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F) → (B, F, D)
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class _FFN(nn.Module):
    def __init__(self, d: int, d_hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── SAINT block ───────────────────────────────────────────────────────────────


class _SAINTBlock(nn.Module):
    """
    One SAINT block:
      1. Pre-norm column (feature) self-attention
      2. Pre-norm FFN
      3. Pre-norm row (intersample) attention
      4. Pre-norm FFN
    """

    def __init__(self, d: int, n_heads: int, d_hidden: int, dropout: float) -> None:
        super().__init__()

        # Column attention (standard: each sample attends over its own tokens)
        self.norm_col_attn = nn.LayerNorm(d)
        self.col_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm_col_ffn = nn.LayerNorm(d)
        self.col_ffn = _FFN(d, d_hidden, dropout)

        # Row / intersample attention
        # Trick: reshape (B, T, D) → treat T as batch and B as seq length
        # by passing with batch_first=False so PyTorch sees (B, T, D) as
        # (seq=B, batch=T, D). Each "position" in the sequence is a sample;
        # attention is computed across the batch dimension.
        self.norm_row_attn = nn.LayerNorm(d)
        self.row_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=False)
        self.norm_row_ffn = nn.LayerNorm(d)
        self.row_ffn = _FFN(d, d_hidden, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)  where T = n_features + 1 (CLS)

        # 1. Column self-attention
        residual = x
        x_norm = self.norm_col_attn(x)
        attn_out, _ = self.col_attn(x_norm, x_norm, x_norm)
        x = residual + attn_out

        # 2. Column FFN
        x = x + self.col_ffn(self.norm_col_ffn(x))

        # 3. Row (intersample) attention
        # Reshape: (B, T, D) → (B, T, D) but pass as batch_first=False
        # PyTorch MHA with batch_first=False expects (seq, batch, D)
        # We pass x as (B, T, D) which PyTorch interprets as (seq=B, batch=T, D)
        # → attention over samples (B dimension) for each token position
        residual = x
        x_norm = self.norm_row_attn(x)          # (B, T, D)
        # batch_first=False → expects (seq, batch, embed) → treat as (B, T, D) directly
        attn_out, _ = self.row_attn(x_norm, x_norm, x_norm)   # (B, T, D)
        x = residual + attn_out

        # 4. Row FFN
        x = x + self.row_ffn(self.norm_row_ffn(x))

        return x


# ── SAINT model ───────────────────────────────────────────────────────────────


class SAINT(nn.Module):
    """
    SAINT for binary tabular classification.

    Parameters
    ----------
    input_dim : int
        Number of input features (after PCA).
    cfg : SAINTConfig
        Model hyperparameters.
    """

    def __init__(self, input_dim: int, cfg: SAINTConfig) -> None:
        super().__init__()
        d = cfg.d_token
        d_hidden = max(1, round(d * cfg.ffn_d_hidden_factor))
        dropout = cfg.attn_dropout  # shared dropout for both attn and ffn

        self.tokenizer = _FeatureTokenizer(input_dim, d)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d))
        nn.init.normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList(
            [_SAINTBlock(d, cfg.n_heads, d_hidden, dropout) for _ in range(cfg.n_blocks)]
        )
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F)
        tokens = self.tokenizer(x)                              # (B, F, D)
        cls = self.cls_token.expand(x.size(0), -1, -1)         # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)                # (B, F+1, D)

        for block in self.blocks:
            tokens = block(tokens)

        cls_out = self.norm(tokens[:, 0])                       # (B, D)
        return torch.sigmoid(self.head(cls_out)).squeeze(-1)    # (B,)

    def param_count(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        return {"classical": total, "quantum": 0, "total": total}
