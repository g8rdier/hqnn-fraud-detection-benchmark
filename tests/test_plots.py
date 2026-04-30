"""Smoke tests for src/evaluation/plots.py.

Each test calls one plot function with minimal mock data and asserts that
the output file is created and non-empty. Visual correctness is not checked.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import AggregatedMetrics
from src.evaluation.plots import (
    plot_ablation_noise,
    plot_ablation_vqc,
    plot_aggregated_confusion_matrices,
    plot_class_imbalance,
    plot_efficiency_comparison,
    plot_efficiency_frontier,
    plot_fold_consistency,
    plot_fold_trajectories,
    plot_hilbert_space,
    plot_mcc_vs_prauc,
    plot_metric_comparison,
    plot_parameter_breakdown,
    plot_parameter_efficiency,
    plot_pca_scree,
    plot_shnn_architecture,
    plot_smote_illustration,
    plot_statistical_heatmap,
    plot_vqc_circuit,
)


@pytest.fixture
def two_models() -> list[AggregatedMetrics]:
    return [
        AggregatedMetrics(
            model_name="shnn",
            mcc_mean=0.58, mcc_std=0.04,
            pr_auc_mean=0.59, pr_auc_std=0.03,
            f1_fraud_mean=0.60, f1_fraud_std=0.02,
            roc_auc_mean=0.95, roc_auc_std=0.01,
            param_count={"classical": 74, "quantum": 48, "total": 122},
            fold_mccs=[0.54, 0.56, 0.58, 0.60, 0.62],
            fold_pr_aucs=[0.55, 0.57, 0.59, 0.61, 0.63],
        ),
        AggregatedMetrics(
            model_name="snn",
            mcc_mean=0.56, mcc_std=0.02,
            pr_auc_mean=0.64, pr_auc_std=0.01,
            f1_fraud_mean=0.65, f1_fraud_std=0.01,
            roc_auc_mean=0.96, roc_auc_std=0.01,
            param_count={"classical": 3201, "quantum": 0, "total": 3201},
            fold_mccs=[0.54, 0.55, 0.56, 0.57, 0.58],
            fold_pr_aucs=[0.63, 0.64, 0.64, 0.65, 0.64],
        ),
    ]


def _assert_file(path) -> None:
    assert path.exists(), f"{path} was not created"
    assert path.stat().st_size > 0, f"{path} is empty"


def test_plot_metric_comparison(two_models, tmp_path):
    out = tmp_path / "metric_comparison.png"
    plot_metric_comparison(two_models, out)
    _assert_file(out)


def test_plot_parameter_efficiency(two_models, tmp_path):
    out = tmp_path / "parameter_efficiency.png"
    plot_parameter_efficiency(two_models, out)
    _assert_file(out)


def test_plot_efficiency_comparison(two_models, tmp_path):
    out = tmp_path / "efficiency_comparison.png"
    plot_efficiency_comparison(two_models, out)
    _assert_file(out)


def test_plot_fold_consistency(two_models, tmp_path):
    out = tmp_path / "fold_consistency.png"
    plot_fold_consistency(two_models, out)
    _assert_file(out)


def test_plot_statistical_heatmap(tmp_path):
    stat_results = [
        {"model_a": "shnn", "model_b": "snn", "metric": "MCC", "rank_biserial": 0.4},
        {"model_a": "shnn", "model_b": "snn", "metric": "PR-AUC", "rank_biserial": -0.2},
    ]
    out = tmp_path / "statistical_heatmap.png"
    plot_statistical_heatmap(stat_results, out)
    _assert_file(out)


def test_plot_ablation_vqc(tmp_path):
    out = tmp_path / "ablation_vqc.png"
    plot_ablation_vqc(shnn_mcc=0.58, shnn_std=0.04, save_path=out)
    _assert_file(out)


def test_plot_ablation_noise(tmp_path):
    noise_data = [
        {"depolarizing_p": 0.0, "mcc_tuned_threshold": 0.58, "mcc_fixed_threshold": 0.55},
        {"depolarizing_p": 0.01, "mcc_tuned_threshold": 0.52, "mcc_fixed_threshold": 0.50},
        {"depolarizing_p": 0.05, "mcc_tuned_threshold": 0.30, "mcc_fixed_threshold": 0.28},
    ]
    out = tmp_path / "ablation_noise.png"
    plot_ablation_noise(noise_data, out)
    _assert_file(out)


def test_plot_aggregated_confusion_matrices(tmp_path):
    fold_cms = {
        "shnn": [[[56000, 20], [10, 80]], [[55000, 25], [8, 82]]],
    }
    out = tmp_path / "confusion_matrices.png"
    plot_aggregated_confusion_matrices(fold_cms, out)
    _assert_file(out)


def test_plot_mcc_vs_prauc(two_models, tmp_path):
    out = tmp_path / "mcc_vs_prauc.png"
    plot_mcc_vs_prauc(two_models, out)
    _assert_file(out)


def test_plot_efficiency_frontier(two_models, tmp_path):
    out = tmp_path / "efficiency_frontier.png"
    plot_efficiency_frontier(two_models, out)
    _assert_file(out)


def test_plot_fold_trajectories(two_models, tmp_path):
    out = tmp_path / "fold_trajectories.png"
    plot_fold_trajectories(two_models, out)
    _assert_file(out)


def test_plot_vqc_circuit(tmp_path):
    out = tmp_path / "vqc_circuit.png"
    plot_vqc_circuit(n_qubits=4, n_layers=2, save_path=out)
    _assert_file(out)


def test_plot_class_imbalance(tmp_path):
    out = tmp_path / "class_imbalance.png"
    plot_class_imbalance(n_legit=284315, n_fraud=492, save_path=out)
    _assert_file(out)


def test_plot_parameter_breakdown(two_models, tmp_path):
    out = tmp_path / "parameter_breakdown.png"
    plot_parameter_breakdown(two_models, out)
    _assert_file(out)


def test_plot_hilbert_space(tmp_path):
    out = tmp_path / "hilbert_space.png"
    plot_hilbert_space(n_qubits_highlight=8, save_path=out)
    _assert_file(out)


def test_plot_shnn_architecture(tmp_path):
    out = tmp_path / "shnn_architecture.png"
    plot_shnn_architecture(save_path=out)
    _assert_file(out)


def test_plot_pca_scree(tmp_path):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 20))
    out = tmp_path / "pca_scree.png"
    plot_pca_scree(X, n_highlight=8, save_path=out)
    _assert_file(out)


def test_plot_smote_illustration(tmp_path):
    rng = np.random.default_rng(42)
    X_pre = rng.standard_normal((300, 8))
    y_pre = np.array([0] * 290 + [1] * 10)
    X_post = rng.standard_normal((580, 8))
    y_post = np.array([0] * 290 + [1] * 290)
    out = tmp_path / "smote_illustration.png"
    plot_smote_illustration(X_pre, y_pre, X_post, y_post, save_path=out)
    _assert_file(out)
