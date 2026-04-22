#!/usr/bin/env python3
"""
ablation_noise.py
=================
Ablation: MCC degradation of the trained SHNN under depolarizing noise.

Loads the best fold 0 SHNN weights and evaluates on the test set at
increasing depolarizing noise levels to characterise NISQ-era robustness.
Noise is injected via DepolarizingChannel after the VQC (inference only —
no retraining). Uses default.mixed backend for p > 0.

Note: results/models/shnn_fold0.pt must contain the best trained weights.
      If in doubt, regenerate via:
          pixi run python scripts/run_benchmark.py --model shnn --fold 0

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

from src.config import NoiseConfig, load_config
from src.data.cv import create_folds
from src.data.loader import load_dataset
from src.models.quantum.shnn import SHNN
from src.training.trainer import find_optimal_threshold

NOISE_LEVELS: list[float] = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
CHECKPOINT_PATH = Path("results/models/shnn_fold0.pt")
OUT_PATH = Path("results/ablation_noise.json")
# Stratified subsample size for p > 0 (density matrix sim is ~10–20× slower)
N_SUBSAMPLE = 1_000
RANDOM_SEED = 42


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
    # Fallback: plain state dict saved directly
    logger.warning("No 'best_state' key — treating checkpoint as plain state dict.")
    return ckpt


def stratified_subsample(
    X: np.ndarray, y: np.ndarray, n: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified subsample: keep all fraud cases, fill remainder with legit."""
    rng = np.random.default_rng(seed)
    fraud_idx = np.where(y == 1)[0]
    legit_idx = np.where(y == 0)[0]
    n_fraud = len(fraud_idx)
    n_legit = min(len(legit_idx), n - n_fraud)
    sel_legit = rng.choice(legit_idx, size=n_legit, replace=False)
    idx = np.concatenate([fraud_idx, sel_legit])
    rng.shuffle(idx)
    return X[idx], y[idx]


def evaluate(
    model: SHNN, X: np.ndarray, y: np.ndarray
) -> tuple[float, float]:
    """Return (mcc, threshold) for the given model on X, y."""
    model.eval()
    device = next(model.parameters()).device
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        prob = model(X_t).cpu().numpy().flatten()
    threshold = find_optimal_threshold(y, prob)
    mcc = matthews_corrcoef(y, (prob >= threshold).astype(int))
    return float(mcc), float(threshold)


def main() -> None:
    cfg = load_config(Path("configs/default.yaml"))

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Regenerate with: pixi run python scripts/run_benchmark.py --model shnn --fold 0"
        )

    best_state = load_best_weights(CHECKPOINT_PATH)

    X, y = load_dataset(cfg.data)
    X_np = X.values if hasattr(X, "values") else np.asarray(X)
    y_np = y.values if hasattr(y, "values") else np.asarray(y)
    folds = create_folds(X_np, y_np, cfg)
    fold = folds[0]

    n_fraud = int(fold.y_test.sum())
    logger.info("Test set: %d samples (%d fraud)", len(fold.y_test), n_fraud)

    results = []

    for p in NOISE_LEVELS:
        logger.info("── Noise p=%.3f ──", p)

        if p == 0.0:
            # Ideal: fast lightning.qubit backend, full test set
            model = SHNN(input_dim=fold.X_test.shape[1], cfg=cfg.shnn, noise_cfg=None)
            X_eval, y_eval = fold.X_test, fold.y_test
            eval_label = f"ideal / full test set (n={len(y_eval)})"
        else:
            # Noisy: default.mixed backend + stratified subsample
            noise_cfg = NoiseConfig(
                enabled=True, backend="default.mixed", depolarizing_p=p
            )
            model = SHNN(
                input_dim=fold.X_test.shape[1], cfg=cfg.shnn, noise_cfg=noise_cfg
            )
            X_eval, y_eval = stratified_subsample(
                fold.X_test, fold.y_test, N_SUBSAMPLE, RANDOM_SEED
            )
            eval_label = f"noisy default.mixed (n={len(y_eval)}, fraud={int(y_eval.sum())})"

        model.load_state_dict(best_state)
        logger.info("Evaluating: %s …", eval_label)

        mcc, threshold = evaluate(model, X_eval, y_eval)
        logger.info("p=%.3f → MCC=%.4f (threshold=%.2f)", p, mcc, threshold)

        results.append(
            {
                "depolarizing_p": p,
                "mcc": mcc,
                "threshold": threshold,
                "n_samples": int(len(y_eval)),
                "n_fraud": int(y_eval.sum()),
            }
        )

        # Save incrementally so a crash doesn't lose completed levels
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved → %s", OUT_PATH)

    logger.info("── Summary ──")
    for r in results:
        logger.info("p=%.3f → MCC=%.4f", r["depolarizing_p"], r["mcc"])


if __name__ == "__main__":
    main()
