#!/usr/bin/env python3
"""
ablation_noise.py
=================
Ablation: MCC degradation of the trained SHNN under per-gate depolarising noise.

Loads the best fold 0 SHNN weights and evaluates MCC at increasing
depolarising noise levels using an analytically derived per-gate noise model.

Noise model
-----------
Real NISQ hardware accumulates errors at every gate.  For independent
per-gate depolarising noise with probability p, the expectation value
of the measured observable degrades as:

    <Z0>_noisy = (1 - 4p/3)^N_gates × <Z0>_ideal

Circuit gate count (n_qubits=8, n_layers=2):
    AngleEmbedding : 8  Ry gates
    StronglyEntanglingLayers : 16 Rot + 16 CNOT = 32 gates
    Total N_gates  : 40

VQC outputs are computed once with the ideal lightning.qubit backend,
then scaled analytically per noise level — no density-matrix simulation.

Usage
-----
    pixi run python scripts/ablation_noise.py

Output
------
    results/ablation_noise.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from rich.logging import RichHandler
from sklearn.metrics import matthews_corrcoef

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger(__name__)

from src.config import load_config
from src.data.cv import create_folds
from src.data.loader import load_dataset
from src.models.quantum.shnn import SHNN
from src.training.trainer import find_optimal_threshold

NOISE_LEVELS: list[float] = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
# Gate count for 8-qubit, 2-layer VQC:
#   8 Ry (AngleEmbedding) + 16 Rot + 16 CNOT (StronglyEntanglingLayers) = 40
N_GATES = 40
CHECKPOINT_PATH = Path("results/models/shnn_fold0.pt")
OUT_PATH = Path("results/ablation_noise.json")


def load_best_weights(checkpoint_path: Path) -> dict:
    """Load best model state from a crash-recovery checkpoint or plain state dict."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "best_state" in ckpt:
        logger.info(
            "Loaded checkpoint: epoch=%d, best_val_mcc=%.4f",
            ckpt.get("epoch", -1),
            ckpt.get("best_val_mcc", float("nan")),
        )
        return ckpt["best_state"]
    logger.warning("No 'best_state' key — treating checkpoint as plain state dict.")
    return ckpt


def main() -> None:
    cfg = load_config(Path("configs/default.yaml"))

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Regenerate with: pixi run fold -- --model shnn --fold 0"
        )

    best_state = load_best_weights(CHECKPOINT_PATH)

    X, y = load_dataset(cfg.data)
    X_np = X.values if hasattr(X, "values") else np.asarray(X)
    y_np = y.values if hasattr(y, "values") else np.asarray(y)
    folds = create_folds(X_np, y_np, cfg)
    fold = folds[0]

    n_fraud = int(fold.y_test.sum())
    logger.info("Test set: %d samples (%d fraud)", len(fold.y_test), n_fraud)

    # Build ideal model (lightning.qubit) and load best weights
    model = SHNN(input_dim=fold.X_test.shape[1], cfg=cfg.shnn, noise_cfg=None)
    model.load_state_dict(best_state)
    model.eval()

    # ── Compute ideal VQC outputs once ────────────────────────────────────────
    # pre_fc maps input → (0, π) for AngleEmbedding; VQC returns <Z0> per sample.
    # These are fixed for all noise levels — only the scale factor changes.
    logger.info("Computing ideal VQC outputs on full test set …")
    X_t = torch.tensor(fold.X_test, dtype=torch.float32)
    with torch.no_grad():
        x = model.pre_fc(X_t)          # (n, n_qubits)
        vqc_out = model.vqc(x.cpu())   # (n,)  — ideal <Z0> expectation values

    logger.info("VQC output range: [%.4f, %.4f]", vqc_out.min().item(), vqc_out.max().item())

    # Baseline threshold from ideal model — fixed for the realistic deployment scenario
    with torch.no_grad():
        prob_ideal = model.post_fc(vqc_out.unsqueeze(-1)).numpy().flatten()
    fixed_threshold = find_optimal_threshold(fold.y_test, prob_ideal)
    logger.info("Baseline threshold (fixed for deployment scenario): %.2f", fixed_threshold)

    results = []

    for p in NOISE_LEVELS:
        # Per-gate depolarising: <Z0>_noisy = (1 - 4p/3)^N_gates × <Z0>_ideal
        noise_factor = (1.0 - 4.0 * p / 3.0) ** N_GATES
        vqc_noisy = vqc_out * noise_factor  # (n,)

        with torch.no_grad():
            prob = model.post_fc(vqc_noisy.unsqueeze(-1)).numpy().flatten()

        # Tuned: re-optimise threshold at each noise level (best possible performance)
        tuned_threshold = find_optimal_threshold(fold.y_test, prob)
        mcc_tuned = matthews_corrcoef(fold.y_test, (prob >= tuned_threshold).astype(int))

        # Fixed: keep baseline threshold (realistic NISQ deployment — calibrate once, deploy)
        mcc_fixed = matthews_corrcoef(fold.y_test, (prob >= fixed_threshold).astype(int))

        logger.info(
            "p=%.3f (scale=%.4f) → MCC tuned=%.4f (t=%.2f) | fixed=%.4f (t=%.2f)",
            p, noise_factor, mcc_tuned, tuned_threshold, mcc_fixed, fixed_threshold,
        )

        results.append({
            "depolarizing_p": p,
            "noise_factor": round(noise_factor, 6),
            "mcc_tuned_threshold": float(mcc_tuned),
            "tuned_threshold": float(tuned_threshold),
            "mcc_fixed_threshold": float(mcc_fixed),
            "fixed_threshold": float(fixed_threshold),
            "n_samples": int(len(fold.y_test)),
            "n_fraud": n_fraud,
        })

        # Save incrementally
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

    logger.info("Saved → %s", OUT_PATH)
    logger.info("── Summary ──")
    for r in results:
        logger.info(
            "p=%.3f → MCC tuned=%.4f | fixed=%.4f",
            r["depolarizing_p"], r["mcc_tuned_threshold"], r["mcc_fixed_threshold"],
        )


if __name__ == "__main__":
    main()
