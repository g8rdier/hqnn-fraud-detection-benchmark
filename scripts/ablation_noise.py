#!/usr/bin/env python3
"""
ablation_noise.py
=================
Ablation: MCC degradation of the trained SHNN under depolarizing noise.

Loads the best fold 0 SHNN weights and evaluates MCC at increasing
depolarizing noise levels using an analytically exact noise model.

Noise model
-----------
For a DepolarizingChannel(p) on the measured qubit (qubit 0):

    <Z0>_noisy = (1 - 4p/3) × <Z0>_ideal

Depolarising channels on qubits 1–7 do not affect <Z0> by the
trace-preserving property of local quantum channels.  This avoids
density-matrix simulation entirely: VQC outputs are computed once
with the ideal lightning.qubit backend, then scaled per noise level.

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

    results = []

    for p in NOISE_LEVELS:
        # Analytical depolarising: <Z0>_noisy = (1 - 4p/3) × <Z0>_ideal
        noise_factor = 1.0 - (4.0 * p / 3.0)
        vqc_noisy = vqc_out * noise_factor  # (n,)

        with torch.no_grad():
            prob = model.post_fc(vqc_noisy.unsqueeze(-1)).numpy().flatten()

        threshold = find_optimal_threshold(fold.y_test, prob)
        mcc = matthews_corrcoef(fold.y_test, (prob >= threshold).astype(int))

        logger.info(
            "p=%.3f (scale=%.4f) → MCC=%.4f (threshold=%.2f)",
            p, noise_factor, mcc, threshold,
        )

        results.append({
            "depolarizing_p": p,
            "noise_factor": round(noise_factor, 6),
            "mcc": float(mcc),
            "threshold": float(threshold),
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
        logger.info("p=%.3f → MCC=%.4f", r["depolarizing_p"], r["mcc"])


if __name__ == "__main__":
    main()
