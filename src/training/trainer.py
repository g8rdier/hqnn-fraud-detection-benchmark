"""
trainer.py
==========
Unified PyTorch training loop for SHNN, ParallelHybrid, and SNN models.

Features:
    - Early stopping on validation MCC
    - Automatic device selection (MPS > CUDA > CPU)
    - Per-epoch logging with Rich progress bars
    - Threshold tuning on validation set
    - Crash-recovery checkpointing: saves state after every epoch,
      resumes automatically if a checkpoint file exists
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import TrainingConfig

logger = logging.getLogger(__name__)


def _resolve_device(device_str: str) -> torch.device:
    """Resolve 'auto' to the best available device."""
    if device_str != "auto":
        return torch.device(device_str)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainResult:
    """Training result container."""
    y_pred: np.ndarray
    y_prob: np.ndarray
    threshold: float
    fit_time: float
    best_epoch: int
    train_losses: list[float] = field(default_factory=list)
    val_mccs: list[float] = field(default_factory=list)


def find_optimal_threshold(y_true: np.ndarray, y_logits: np.ndarray) -> float:
    """Find the threshold that maximizes MCC on validation data.

    For tanh outputs in [-1, 1], search thresholds in that range.
    """
    best_mcc = -1.0
    best_t = 0.0
    for t in np.arange(-0.95, 1.0, 0.05):
        preds = (y_logits >= t).astype(int)
        mcc = matthews_corrcoef(y_true, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            best_t = t
    return float(best_t)


def train_pytorch_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    cfg: TrainingConfig,
    checkpoint_path: Path | None = None,
) -> TrainResult:
    """
    Train a PyTorch model with early stopping and crash-recovery checkpointing.

    Parameters
    ----------
    model : nn.Module
        SHNN, ParallelHybrid, or SNN.
    X_train, y_train : Training data (post-SMOTE, post-PCA).
    X_val, y_val : Validation data (real class distribution).
    X_test : Test data for final predictions.
    cfg : TrainingConfig
    checkpoint_path : Path | None
        If provided, training state is saved here after every epoch and
        resumed automatically on restart.

    Returns
    -------
    TrainResult with predictions, threshold, timing, and history.
    """
    device = _resolve_device(cfg.device)
    logger.info("Training on device: %s", device)
    model = model.to(device)

    # Build data loaders (convert labels from [0,1] to [-1,+1] for MSE loss)
    y_train_mse = 2 * y_train - 1  # [0,1] → [-1,+1]
    y_val_mse = 2 * y_val - 1
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_mse, dtype=torch.float32).unsqueeze(-1),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_np = y_val
    y_val_mse_t = torch.tensor(y_val_mse, dtype=torch.float32).to(device).unsqueeze(-1)

    # Optimizer
    if cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9
        )

    criterion = nn.MSELoss()

    # ── Resume from checkpoint if available ──────────────────────────────────
    best_val_mcc = -1.0
    best_epoch = 0
    best_state: dict | None = None
    patience_counter = 0
    train_losses: list[float] = []
    val_mccs: list[float] = []
    start_epoch = 0

    if checkpoint_path is not None and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        start_epoch = ckpt["epoch"]          # resume from next epoch
        best_val_mcc = ckpt["best_val_mcc"]
        best_epoch = ckpt["best_epoch"]
        patience_counter = ckpt["patience_counter"]
        train_losses = ckpt["train_losses"]
        val_mccs = ckpt["val_mccs"]
        best_state = ckpt["best_state"]
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        logger.info(
            "Resumed from checkpoint (epoch %d, best_mcc=%.4f)",
            start_epoch, best_val_mcc,
        )

    t0 = time.perf_counter()

    for epoch in range(start_epoch, cfg.epochs):
        # ── Train ─────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_hat = model(X_batch)
            loss = criterion(y_hat, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_loss)

        # ── Validate ──────────────────────────────────────────────────
        # Tanh outputs are in [-1, 1]. Use threshold 0 for early stopping signal,
        # then tune on validation set after training for final predictions.
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t).cpu().numpy().flatten()
        val_pred = (val_logits >= 0.0).astype(int)
        val_mcc = matthews_corrcoef(y_val_np, val_pred)
        val_mccs.append(val_mcc)

        logger.info(
            "Epoch %3d/%d | loss=%.4f | val_mcc=%.4f",
            epoch + 1, cfg.epochs, avg_loss, val_mcc,
        )

        # ── Early stopping ────────────────────────────────────────────
        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                logger.info(
                    "Early stopping at epoch %d (best: %d, MCC=%.4f)",
                    epoch + 1, best_epoch, best_val_mcc,
                )
                break

        # ── Checkpoint ───────────────────────────────────────────────
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "best_val_mcc": best_val_mcc,
                    "best_epoch": best_epoch,
                    "patience_counter": patience_counter,
                    "train_losses": train_losses,
                    "val_mccs": val_mccs,
                    "best_state": best_state,
                    "model_state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    "optimizer_state": optimizer.state_dict(),
                },
                checkpoint_path,
            )

    fit_time = time.perf_counter() - t0

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    # ── Threshold tuning on validation set ────────────────────────────
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t).cpu().numpy().flatten()
    threshold = find_optimal_threshold(y_val_np, val_logits)
    logger.info("Tuned threshold: %.4f (best_epoch=%d)", threshold, best_epoch)

    # ── Test predictions ──────────────────────────────────────────────
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        test_logits = model(X_test_t).cpu().numpy().flatten()
    test_pred = (test_logits >= threshold).astype(int)

    return TrainResult(
        y_pred=test_pred,
        y_prob=test_logits,
        threshold=threshold,
        fit_time=fit_time,
        best_epoch=best_epoch,
        train_losses=train_losses,
        val_mccs=val_mccs,
    )
