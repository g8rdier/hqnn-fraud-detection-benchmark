"""
trainer.py
==========
Unified PyTorch training loop for SHNN, ParallelHybrid, and SNN models.

Features:
    - Early stopping on validation MCC
    - Automatic device selection (MPS > CUDA > CPU)
    - Per-epoch logging with Rich progress bars
    - Threshold tuning on validation set
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

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


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find the threshold that maximizes MCC on validation data."""
    best_mcc = -1.0
    best_t = 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (y_prob >= t).astype(int)
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
) -> TrainResult:
    """
    Train a PyTorch model with early stopping.

    Parameters
    ----------
    model : nn.Module
        SHNN, ParallelHybrid, or SNN.
    X_train, y_train : Training data (post-SMOTE, post-PCA).
    X_val, y_val : Validation data (real class distribution).
    X_test : Test data for final predictions.
    cfg : TrainingConfig

    Returns
    -------
    TrainResult with predictions, threshold, timing, and history.
    """
    device = _resolve_device(cfg.device)
    logger.info("Training on device: %s", device)
    model = model.to(device)

    # Build data loaders
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_np = y_val

    # Optimizer
    if cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=0.9
        )

    criterion = nn.BCELoss()

    # Early stopping state
    best_val_mcc = -1.0
    best_epoch = 0
    best_state = None
    patience_counter = 0

    train_losses: list[float] = []
    val_mccs: list[float] = []

    t0 = time.perf_counter()

    for epoch in range(cfg.epochs):
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
        # Use fixed 0.5 threshold for early stopping signal. Although the
        # val set has real class distribution (0.17% fraud), the slowly-
        # climbing MCC at 0.5 provides a stable monotonic proxy for
        # representation quality. Threshold tuning happens *after* training
        # for final test predictions — tuning it per-epoch causes premature
        # early stopping because val_mcc peaks too early.
        model.eval()
        with torch.no_grad():
            val_prob = model(X_val_t).cpu().numpy().flatten()
        val_pred = (val_prob >= 0.5).astype(int)
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

    fit_time = time.perf_counter() - t0

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    # ── Threshold tuning on validation set ────────────────────────────
    model.eval()
    with torch.no_grad():
        val_prob = model(X_val_t).cpu().numpy().flatten()
    threshold = find_optimal_threshold(y_val_np, val_prob)
    logger.info("Tuned threshold: %.4f (best_epoch=%d)", threshold, best_epoch)

    # ── Test predictions ──────────────────────────────────────────────
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        test_prob = model(X_test_t).cpu().numpy().flatten()
    test_pred = (test_prob >= threshold).astype(int)

    return TrainResult(
        y_pred=test_pred,
        y_prob=test_prob,
        threshold=threshold,
        fit_time=fit_time,
        best_epoch=best_epoch,
        train_losses=train_losses,
        val_mccs=val_mccs,
    )
