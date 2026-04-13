"""
vqc.py
======
Variational Quantum Circuit as a PyTorch-compatible layer.

Uses PennyLane's TorchLayer to wrap a parameterized quantum circuit so it
participates natively in PyTorch autograd (gradients via parameter-shift rule
or adjoint differentiation).

Architecture:
    AngleEmbedding(x) → StronglyEntanglingLayers(weights) → ⟨Z₀⟩
"""

from __future__ import annotations

import pennylane as qml
import torch
from torch import nn

from src.config import NoiseConfig, VQCConfig


def build_vqc_layer(
    cfg: VQCConfig,
    noise_cfg: NoiseConfig | None = None,
) -> nn.Module:
    """
    Build a VQC as a PyTorch nn.Module via PennyLane TorchLayer.

    Parameters
    ----------
    cfg : VQCConfig
        Qubit count, layer count, backend, diff method.
    noise_cfg : NoiseConfig, optional
        If provided and enabled, uses a noisy backend with depolarizing channels.

    Returns
    -------
    nn.Module
        A TorchLayer with input dim = n_qubits, output dim = 1.
    """
    n_qubits = cfg.n_qubits
    n_layers = cfg.n_layers

    # Select backend: noisy (default.mixed) or ideal (lightning.qubit)
    if noise_cfg is not None and noise_cfg.enabled:
        dev = qml.device(noise_cfg.backend, wires=n_qubits)
        diff_method = "backprop"  # default.mixed supports backprop
        noise_p = noise_cfg.depolarizing_p
    else:
        dev = qml.device(cfg.backend, wires=n_qubits)
        diff_method = cfg.diff_method
        noise_p = 0.0

    @qml.qnode(dev, diff_method=diff_method, interface="torch")
    def circuit(inputs, weights):
        # Angle encoding: each feature → Ry rotation on corresponding qubit
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

        # Parameterized layers
        for layer_idx in range(n_layers):
            qml.StronglyEntanglingLayers(
                weights[layer_idx : layer_idx + 1], wires=range(n_qubits)
            )
            # Inject depolarizing noise after each layer if enabled
            if noise_p > 0:
                for wire in range(n_qubits):
                    qml.DepolarizingChannel(noise_p, wires=wire)

        return qml.expval(qml.PauliZ(0))

    # Weight shape: (n_layers, 1, n_qubits, 3) for StronglyEntanglingLayers
    # Total trainable params: n_layers × n_qubits × 3
    weight_shapes = {"weights": (n_layers, 1, n_qubits, 3)}

    # Initialize with restricted variance to mitigate barren plateaus
    init_method = {
        "weights": lambda size: torch.randn(size) * 0.1,
    }

    layer = qml.qnn.TorchLayer(circuit, weight_shapes, init_method=init_method)
    return layer


class VQCModule(nn.Module):
    """
    Standalone VQC module with sigmoid output for binary classification.

    Input:  (batch, n_qubits)
    Output: (batch, 1) — probability of fraud
    """

    def __init__(self, cfg: VQCConfig, noise_cfg: NoiseConfig | None = None):
        super().__init__()
        self.vqc = build_vqc_layer(cfg, noise_cfg)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TorchLayer processes one sample at a time internally but supports batches
        out = self.vqc(x)  # (batch,)
        return self.sigmoid(out).unsqueeze(-1)  # (batch, 1)
