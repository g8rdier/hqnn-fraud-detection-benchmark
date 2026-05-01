"""Tests for evaluation metrics and statistics."""

import numpy as np
import pytest

from src.evaluation.metrics import AggregatedMetrics, ModelMetrics, aggregate_fold_metrics, compute_metrics
from src.evaluation.statistics import StatisticalResult, compare_models, rank_biserial_correlation


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


def test_compare_models_small_effect():
    # 3 positive diffs, 2 negative → r = (3-2)/5 = 0.2 → "small"
    a = [0.6, 0.7, 0.5, 0.6, 0.4]
    b = [0.5, 0.6, 0.6, 0.5, 0.5]
    result = compare_models("a", a, "b", b, "MCC")
    assert result.interpretation.startswith("small")


def test_compare_models_medium_effect():
    # 2 positive, 1 negative, 2 tied → n_nonzero=3, r = (2-1)/3 ≈ 0.333 → "medium"
    a = [0.7, 0.6, 0.5, 0.5, 0.5]
    b = [0.5, 0.5, 0.6, 0.5, 0.5]
    result = compare_models("a", a, "b", b, "MCC")
    assert result.interpretation.startswith("medium")


def test_compare_models_large_effect():
    # all diffs positive → r = 1.0 → "large"
    a = [0.8, 0.9, 0.7, 0.8, 0.85]
    b = [0.5, 0.4, 0.3, 0.4, 0.50]
    result = compare_models("a", a, "b", b, "MCC")
    assert result.interpretation.startswith("large")


def test_model_metrics_to_dict_no_confusion():
    m = ModelMetrics("shnn", mcc=0.58, pr_auc=0.59, f1_fraud=0.60,
                     roc_auc=0.95, threshold=0.5, param_count={"total": 122})
    d = m.to_dict()
    assert d["model_name"] == "shnn"
    assert "confusion_matrix" not in d


def test_model_metrics_to_dict_with_confusion():
    m = ModelMetrics("shnn", mcc=0.58, pr_auc=0.59, f1_fraud=0.60,
                     roc_auc=0.95, threshold=0.5, param_count={"total": 122},
                     confusion=np.array([[100, 2], [3, 10]]))
    d = m.to_dict()
    assert "confusion_matrix" in d
    assert d["confusion_matrix"] == [[100, 2], [3, 10]]


def test_aggregated_metrics_to_dict():
    agg = AggregatedMetrics(
        model_name="shnn", mcc_mean=0.58, mcc_std=0.04,
        pr_auc_mean=0.59, pr_auc_std=0.03, f1_fraud_mean=0.60,
        f1_fraud_std=0.02, roc_auc_mean=0.95, roc_auc_std=0.01,
        param_count={"total": 122}, fold_mccs=[0.54, 0.56, 0.58, 0.60, 0.62],
        fold_pr_aucs=[0.55, 0.57, 0.59, 0.61, 0.63],
    )
    d = agg.to_dict()
    assert d["model_name"] == "shnn"
    assert d["fold_mccs"] == [0.54, 0.56, 0.58, 0.60, 0.62]


def test_statistical_result_to_dict():
    r = StatisticalResult(
        model_a="shnn", model_b="snn", metric="MCC",
        wilcoxon_statistic=4.0, wilcoxon_p=0.5,
        rank_biserial=0.2, n_folds=5,
        interpretation="small effect (shnn > snn), r=0.200, W=4.0, p=0.5000 (n=5 folds, min p=0.0625)",
    )
    d = r.to_dict()
    assert d["model_a"] == "shnn"
    assert d["rank_biserial"] == 0.2
