"""
tabnet_model.py
===============
TabNet wrapper for the benchmark.

Uses pytorch-tabnet which provides its own training loop, early stopping,
and attention-based feature selection. This wrapper adapts it to our unified
evaluation interface.

TabNet is selected as a baseline because its attention-based feature selection
provides a classical parallel to quantum amplitude amplification (Arik & Pfister, 2021).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

from src.config import TabNetConfig, TrainingConfigTabNet

logger = logging.getLogger(__name__)


@dataclass
class TabNetResult:
    """Result container matching our evaluation interface."""
    y_pred: np.ndarray
    y_prob: np.ndarray
    param_count: dict[str, int]


class TabNetWrapper:
    """
    Wrapper around pytorch-tabnet's TabNetClassifier.

    TabNet has its own training loop (not PyTorch Lightning / manual loop),
    so it's handled separately from the PyTorch-based models.
    """

    def __init__(self, cfg: TabNetConfig, training_cfg: TrainingConfigTabNet):
        self.cfg = cfg
        self.training_cfg = training_cfg
        self.model: TabNetClassifier | None = None

    def fit(  # pragma: no cover
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """Fit TabNet with its internal training loop and early stopping."""
        self.model = TabNetClassifier(
            n_d=self.cfg.n_d,
            n_a=self.cfg.n_a,
            n_steps=self.cfg.n_steps,
            gamma=self.cfg.gamma,
            lambda_sparse=self.cfg.lambda_sparse,
            optimizer_fn=torch.optim.Adam,
            verbose=0,
            seed=42,
        )

        logger.info(
            "Training TabNet (n_d=%d, n_a=%d, n_steps=%d, max_epochs=%d)",
            self.cfg.n_d, self.cfg.n_a, self.cfg.n_steps, self.training_cfg.max_epochs,
        )

        self.model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=["auc"],
            max_epochs=self.training_cfg.max_epochs,
            patience=self.training_cfg.patience,
            batch_size=self.training_cfg.batch_size,
        )

    def predict(
        self,
        X: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> TabNetResult:
        """Generate predictions with optional threshold tuning on validation set."""
        if self.model is None:
            raise RuntimeError("TabNet not fitted yet. Call fit() first.")

        y_prob = self.model.predict_proba(X)[:, 1]

        # Threshold tuning on real-distribution validation set
        threshold = 0.5
        if X_val is not None and y_val is not None:  # pragma: no cover
            from src.training.trainer import find_optimal_threshold
            val_prob = self.model.predict_proba(X_val)[:, 1]
            threshold = find_optimal_threshold(y_val, val_prob)
            logger.info("TabNet tuned threshold: %.4f", threshold)

        y_pred = (y_prob >= threshold).astype(int)

        total_params = sum(
            p.numel() for p in self.model.network.parameters() if p.requires_grad
        )

        return TabNetResult(
            y_pred=y_pred,
            y_prob=y_prob,
            param_count={"classical": total_params, "quantum": 0, "total": total_params},
        )
