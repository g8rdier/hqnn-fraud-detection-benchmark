"""
shnn.py
=======
Sequential Hybrid Neural Network.

Architecture:
    Input → FC layers → VQC → FC layers → Sigmoid

The quantum layer sits between classical fully connected layers, acting as a
high-dimensional feature transformer in the quantum-enhanced Hilbert space.
"""

from __future__ import annotations

import torch
from torch import nn

from src.config import NoiseConfig, SHNNConfig
from src.models.quantum.vqc import build_vqc_layer


class SHNN(nn.Module):
    """
    Sequential Hybrid Neural Network.

    Parameters
    ----------
    input_dim : int
        Number of input features (= PCA components = n_qubits).
    cfg : SHNNConfig
        Architecture configuration.
    noise_cfg : NoiseConfig, optional
        Noise model for the VQC layer.
    """

    def __init__(
        self,
        input_dim: int,
        cfg: SHNNConfig,
        noise_cfg: NoiseConfig | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        n_qubits = cfg.vqc.n_qubits

        # ── Pre-quantum FC block ─────────────────────────────────────────
        pre_layers: list[nn.Module] = []
        in_dim = input_dim
        for out_dim in cfg.pre_fc_dims:
            pre_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
            ])
            in_dim = out_dim
        # Project to qubit dimension
        pre_layers.append(nn.Linear(in_dim, n_qubits))
        pre_layers.append(nn.Tanh())  # Bound inputs for angle encoding
        self.pre_fc = nn.Sequential(*pre_layers)

        # ── VQC layer ────────────────────────────────────────────────────
        self.vqc = build_vqc_layer(cfg.vqc, noise_cfg)

        # ── Post-quantum FC block ────────────────────────────────────────
        post_layers: list[nn.Module] = []
        in_dim = 1  # VQC outputs single expectation value
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
        x = self.pre_fc(x)             # (batch, n_qubits)
        x = self.vqc(x).unsqueeze(-1)  # (batch, 1)
        x = self.post_fc(x)            # (batch, 1)
        return x

    def param_count(self) -> dict[str, int]:
        """Count trainable parameters by component."""
        classical = sum(
            p.numel() for n, p in self.named_parameters()
            if p.requires_grad and "vqc" not in n
        )
        quantum = sum(
            p.numel() for n, p in self.named_parameters()
            if p.requires_grad and "vqc" in n
        )
        return {"classical": classical, "quantum": quantum, "total": classical + quantum}
