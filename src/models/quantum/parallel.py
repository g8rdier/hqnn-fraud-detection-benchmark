"""
parallel.py
===========
Parallel Hybrid Architecture.

Architecture:
    Input ──┬── MLP branch ──┐
            │                ├── concat → FC → Sigmoid
            └── VQC branch ──┘

Concatenates embeddings from a classical MLP and a VQC to leverage both
classical representation learning and quantum-enhanced feature mapping.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from src.config import NoiseConfig, ParallelHybridConfig
from src.models.quantum.vqc import build_vqc_layer


class _PiSigmoid(nn.Module):
    """Maps any real input to (0, π): x → π · sigmoid(x)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return math.pi * torch.sigmoid(x)


class ParallelHybrid(nn.Module):
    """
    Parallel Hybrid Neural Network.

    Parameters
    ----------
    input_dim : int
        Number of input features (= PCA components = n_qubits).
    cfg : ParallelHybridConfig
        Architecture configuration.
    noise_cfg : NoiseConfig, optional
        Noise model for the VQC branch.
    """

    def __init__(
        self,
        input_dim: int,
        cfg: ParallelHybridConfig,
        noise_cfg: NoiseConfig | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        n_qubits = cfg.vqc.n_qubits

        # ── Classical MLP branch ─────────────────────────────────────────
        mlp_layers: list[nn.Module] = []
        in_dim = input_dim
        for out_dim in cfg.mlp_dims:
            mlp_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
            ])
            in_dim = out_dim
        self.mlp_branch = nn.Sequential(*mlp_layers)
        mlp_out_dim = cfg.mlp_dims[-1] if cfg.mlp_dims else input_dim

        # ── Quantum VQC branch ───────────────────────────────────────────
        # Project input to qubit dimension, then scale to (0, π) for AngleEmbedding.
        # π*sigmoid maps any real value to (0, π) — the correct encoding range.
        self.vqc_proj = nn.Sequential(
            nn.Linear(input_dim, n_qubits),
            _PiSigmoid(),
        )
        self.vqc = build_vqc_layer(cfg.vqc, noise_cfg)
        vqc_out_dim = 1  # Single expectation value

        # ── Post-concat FC block ─────────────────────────────────────────
        concat_dim = mlp_out_dim + vqc_out_dim
        post_layers: list[nn.Module] = []
        in_dim = concat_dim
        for out_dim in cfg.post_fc_dims:
            post_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
            ])
            in_dim = out_dim
        post_layers.append(nn.Linear(in_dim, 1))
        post_layers.append(nn.Sigmoid())
        self.post_fc = nn.Sequential(*post_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device

        # Classical branch
        mlp_out = self.mlp_branch(x)           # (batch, mlp_out_dim)

        # Quantum branch (VQC runs on CPU via PennyLane)
        vqc_in = self.vqc_proj(x).cpu()        # (batch, n_qubits)
        vqc_out = self.vqc(vqc_in).unsqueeze(-1).to(device)  # (batch, 1)

        # Concatenate and classify
        combined = torch.cat([mlp_out, vqc_out], dim=-1)  # (batch, mlp+1)
        return self.post_fc(combined)  # (batch, 1)

    def param_count(self) -> dict[str, int]:
        """Count trainable parameters by component."""
        quantum_names = {"vqc", "vqc_proj"}
        classical = sum(
            p.numel() for n, p in self.named_parameters()
            if p.requires_grad and not any(qn in n for qn in quantum_names)
        )
        quantum = sum(
            p.numel() for n, p in self.named_parameters()
            if p.requires_grad and any(qn in n for qn in quantum_names)
        )
        return {"classical": classical, "quantum": quantum, "total": classical + quantum}
