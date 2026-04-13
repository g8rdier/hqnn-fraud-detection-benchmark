"""Tests for evaluation metrics and statistics."""

import numpy as np
import pytest

from src.evaluation.metrics import ModelMetrics, aggregate_fold_metrics, compute_metrics
from src.evaluation.statistics import compare_models, rank_biserial_correlation


def test_compute_metrics_perfect():
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.1, 0.2, 0.9, 0.95])

    m = compute_metrics("test", y_true, y_pred, y_prob, 0.5, {"total": 100})
    assert m.mcc == 1.0
    assert m.f1_fraud == 1.0


def test_compute_metrics_random():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=100)
    y_prob = rng.random(100)
    y_pred = (y_prob > 0.5).astype(int)

    m = compute_metrics("random", y_true, y_pred, y_prob, 0.5, {"total": 50})
    assert -1.0 <= m.mcc <= 1.0
    assert 0.0 <= m.pr_auc <= 1.0


def test_aggregate_fold_metrics():
    folds = [
        ModelMetrics("m", mcc=0.8, pr_auc=0.9, f1_fraud=0.7, roc_auc=0.95,
                     threshold=0.5, param_count={"total": 100}),
        ModelMetrics("m", mcc=0.85, pr_auc=0.92, f1_fraud=0.72, roc_auc=0.96,
                     threshold=0.5, param_count={"total": 100}),
    ]
    agg = aggregate_fold_metrics(folds)
    assert agg.mcc_mean == pytest.approx(0.825)
    assert len(agg.fold_mccs) == 2


def test_rank_biserial_all_positive():
    x = np.array([0.9, 0.8, 0.7])
    y = np.array([0.1, 0.2, 0.3])
    r = rank_biserial_correlation(x, y)
    assert r == 1.0


def test_rank_biserial_ties():
    x = np.array([0.5, 0.5, 0.5])
    y = np.array([0.5, 0.5, 0.5])
    r = rank_biserial_correlation(x, y)
    assert r == 0.0


def test_compare_models_identical():
    scores = [0.8, 0.82, 0.79, 0.81, 0.83]
    result = compare_models("a", scores, "b", scores, "MCC")
    assert result.rank_biserial == 0.0
