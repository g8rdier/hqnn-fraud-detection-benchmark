"""
metrics.py
==========
Evaluation metrics optimized for extreme class imbalance.

Primary: MCC and PR-AUC (robust to 0.17% fraud rate).
Secondary: F1-Fraud and ROC-AUC (for reference).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)


@dataclass
class ModelMetrics:
    """All metrics for one model on one fold or aggregated."""

    model_name: str
    mcc: float
    pr_auc: float
    f1_fraud: float
    roc_auc: float
    threshold: float
    param_count: dict[str, int]
    confusion: np.ndarray | None = None

    def to_dict(self) -> dict:
        d = {
            "model_name": self.model_name,
            "mcc": self.mcc,
            "pr_auc": self.pr_auc,
            "f1_fraud": self.f1_fraud,
            "roc_auc": self.roc_auc,
            "threshold": self.threshold,
            "param_count": self.param_count,
        }
        if self.confusion is not None:
            d["confusion_matrix"] = self.confusion.tolist()
        return d


def compute_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    param_count: dict[str, int],
) -> ModelMetrics:
    """
    Compute all benchmark metrics.

    Parameters
    ----------
    y_true : ground truth labels
    y_pred : hard predictions (after threshold)
    y_prob : fraud probabilities (continuous)
    threshold : decision threshold used
    param_count : {"classical": int, "quantum": int, "total": int}
    """
    mcc = matthews_corrcoef(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0.0)
    roc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    return ModelMetrics(
        model_name=model_name,
        mcc=mcc,
        pr_auc=pr_auc,
        f1_fraud=f1,
        roc_auc=roc,
        threshold=threshold,
        param_count=param_count,
        confusion=cm,
    )


@dataclass
class AggregatedMetrics:
    """Cross-validated metrics: mean +/- std across folds."""

    model_name: str
    mcc_mean: float
    mcc_std: float
    pr_auc_mean: float
    pr_auc_std: float
    f1_fraud_mean: float
    f1_fraud_std: float
    roc_auc_mean: float
    roc_auc_std: float
    param_count: dict[str, int]
    fold_mccs: list[float]
    fold_pr_aucs: list[float]

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "mcc_mean": self.mcc_mean,
            "mcc_std": self.mcc_std,
            "pr_auc_mean": self.pr_auc_mean,
            "pr_auc_std": self.pr_auc_std,
            "f1_fraud_mean": self.f1_fraud_mean,
            "f1_fraud_std": self.f1_fraud_std,
            "roc_auc_mean": self.roc_auc_mean,
            "roc_auc_std": self.roc_auc_std,
            "param_count": self.param_count,
            "fold_mccs": self.fold_mccs,
            "fold_pr_aucs": self.fold_pr_aucs,
        }


def aggregate_fold_metrics(fold_metrics: list[ModelMetrics]) -> AggregatedMetrics:
    """Aggregate per-fold metrics into mean +/- std."""
    name = fold_metrics[0].model_name
    mccs = [m.mcc for m in fold_metrics]
    pr_aucs = [m.pr_auc for m in fold_metrics]
    f1s = [m.f1_fraud for m in fold_metrics]
    rocs = [m.roc_auc for m in fold_metrics]

    return AggregatedMetrics(
        model_name=name,
        mcc_mean=float(np.mean(mccs)),
        mcc_std=float(np.std(mccs)),
        pr_auc_mean=float(np.mean(pr_aucs)),
        pr_auc_std=float(np.std(pr_aucs)),
        f1_fraud_mean=float(np.mean(f1s)),
        f1_fraud_std=float(np.std(f1s)),
        roc_auc_mean=float(np.mean(rocs)),
        roc_auc_std=float(np.std(rocs)),
        param_count=fold_metrics[0].param_count,
        fold_mccs=mccs,
        fold_pr_aucs=pr_aucs,
    )
