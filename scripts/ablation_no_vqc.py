#!/usr/bin/env python3
"""
ablation_no_vqc.py
==================
Ablation: Run SHNN fold 0 with VQC replaced by constant zeros.

If val_mcc stays ~0.22 → VQC contributes nothing (classical wrapper is doing all the work).
If val_mcc improves significantly → VQC was actually hurting.
If val_mcc collapses → VQC was contributing something useful.
"""
import logging
import numpy as np
import torch
from torch import nn
from pathlib import Path
from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger(__name__)

from src.config import load_config
from src.data.loader import load_dataset
from src.data.cv import create_folds
from src.training.trainer import train_pytorch_model
from src.models.quantum.shnn import SHNN


class SHNNNoVQC(SHNN):
    """SHNN with VQC replaced by constant zero — ablation only."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        _ = self.pre_fc(x)                          # run pre-fc (ignore output)
        x = torch.zeros(x.shape[0], 1, device=device)  # zero instead of VQC
        x = self.post_fc(x)
        return x


def main():
    cfg = load_config(Path("configs/default.yaml"))
    cfg.training_quantum.epochs = 10  # quick test

    X, y = load_dataset(cfg.data)
    X_np = X.values if hasattr(X, "values") else np.asarray(X)
    y_np = y.values if hasattr(y, "values") else np.asarray(y)

    folds = create_folds(X_np, y_np, cfg)
    fold = folds[0]

    input_dim = fold.X_train.shape[1]
    model = SHNNNoVQC(input_dim=input_dim, cfg=cfg.shnn)

    logger.info("Running SHNN ablation (no VQC) — fold 0, 10 epochs")
    result = train_pytorch_model(
        model=model,
        X_train=fold.X_train,
        y_train=fold.y_train,
        X_val=fold.X_val,
        y_val=fold.y_val,
        X_test=fold.X_test,
        cfg=cfg.training_quantum,
    )
    logger.info("Best val_mcc from val_mccs: %.4f", max(result.val_mccs))
    logger.info("Val MCC history: %s", [f"{v:.4f}" for v in result.val_mccs])


if __name__ == "__main__":
    main()
