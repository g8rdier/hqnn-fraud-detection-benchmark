"""
plots.py
========
Visualization for the HQNN benchmark.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.evaluation.metrics import AggregatedMetrics

logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-whitegrid")

COLORS = {
    "shnn": "#6C5CE7",
    "parallel": "#0984E3",
    "snn": "#00B894",
    "tabnet": "#E17055",
    "resnet": "#636E72",
    "ftt": "#636E72",
    "saint": "#636E72",
}

MODEL_LABELS = {
    "shnn": "SHNN",
    "parallel": "Parallel",
    "snn": "SNN",
    "tabnet": "TabNet",
    "resnet": "ResNet",
    "ftt": "FT-T",
    "saint": "SAINT",
}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_metric_comparison(results: list[AggregatedMetrics], save_path: Path) -> None:
    """Bar chart of MCC and PR-AUC with std error bars. Labels sit above error bar caps."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    names = [MODEL_LABELS.get(r.model_name, r.model_name) for r in results]
    colors = [COLORS.get(r.model_name.lower(), "#636E72") for r in results]

    for ax, means, stds, title, ylabel in [
        (axes[0],
         [r.mcc_mean for r in results], [r.mcc_std for r in results],
         "MCC (primary)", "MCC"),
        (axes[1],
         [r.pr_auc_mean for r in results], [r.pr_auc_std for r in results],
         "PR-AUC (primary)", "PR-AUC"),
    ]:
        bars = ax.bar(names, means, yerr=stds, capsize=5,
                      color=colors, alpha=0.85, edgecolor="white", error_kw={"linewidth": 1.5})
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis="x", labelsize=10)

        for bar, val, std in zip(bars, means, stds):
            label_y = bar.get_height() + std + 0.025
            ax.text(
                bar.get_x() + bar.get_width() / 2, label_y,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9,
            )

    fig.suptitle("Model Comparison — 5-Fold CV", fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)


def plot_parameter_efficiency(results: list[AggregatedMetrics], save_path: Path) -> None:
    """Scatter: MCC vs total parameter count (log scale)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for r in results:
        total = r.param_count.get("total", 0)
        color = COLORS.get(r.model_name.lower(), "#636E72")
        label = MODEL_LABELS.get(r.model_name, r.model_name)
        ax.scatter(total, r.mcc_mean, s=120, c=color,
                   edgecolors="white", linewidth=1.5, zorder=5)
        ax.errorbar(total, r.mcc_mean, yerr=r.mcc_std,
                    fmt="none", ecolor=color, capsize=4, alpha=0.7)
        ax.annotate(label, (total, r.mcc_mean),
                    textcoords="offset points", xytext=(8, 6), fontsize=10)

    ax.set_xlabel("Trainable Parameters (log scale)", fontsize=12)
    ax.set_ylabel("MCC (mean ± std)", fontsize=12)
    ax.set_title("Parameter Efficiency: MCC vs Model Complexity", fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    fig.tight_layout()
    _save(fig, save_path)


def plot_efficiency_comparison(results: list[AggregatedMetrics], save_path: Path) -> None:
    """Bar chart of MCC/kParam and PR-AUC/kParam — the central thesis metric."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    names = [MODEL_LABELS.get(r.model_name, r.model_name) for r in results]
    colors = [COLORS.get(r.model_name.lower(), "#636E72") for r in results]

    for ax, efficiencies, title, ylabel in [
        (axes[0],
         [r.mcc_mean / (r.param_count["total"] / 1000) for r in results],
         "MCC / kParam", "MCC per 1,000 parameters"),
        (axes[1],
         [r.pr_auc_mean / (r.param_count["total"] / 1000) for r in results],
         "PR-AUC / kParam", "PR-AUC per 1,000 parameters"),
    ]:
        bars = ax.bar(names, efficiencies, color=colors, alpha=0.85, edgecolor="white")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(axis="x", labelsize=10)

        for bar, val in zip(bars, efficiencies):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(efficiencies) * 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9,
            )

    fig.suptitle("Parameter Efficiency Advantage — MCC and PR-AUC per kParam",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)


def plot_fold_consistency(results: list[AggregatedMetrics], save_path: Path) -> None:
    """Box plots of per-fold MCC per model, showing variance across 5 folds."""
    fig, ax = plt.subplots(figsize=(10, 5))

    names = [MODEL_LABELS.get(r.model_name, r.model_name) for r in results]
    fold_data = [r.fold_mccs for r in results]
    colors = [COLORS.get(r.model_name.lower(), "#636E72") for r in results]

    bp = ax.boxplot(fold_data, patch_artist=True, medianprops={"color": "white", "linewidth": 2},
                    whiskerprops={"linewidth": 1.2}, capprops={"linewidth": 1.2},
                    flierprops={"marker": "o", "markersize": 5, "alpha": 0.6})

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("MCC", fontsize=12)
    ax.set_title("Fold-Level MCC Consistency (5-Fold CV)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.0)

    fig.tight_layout()
    _save(fig, save_path)


def plot_statistical_heatmap(stat_results: list[dict], save_path: Path) -> None:
    """Heatmap of rank-biserial correlation r for each HQNN vs classical pair."""
    quantum = sorted({s["model_a"] for s in stat_results})
    classical = sorted({s["model_b"] for s in stat_results})
    metrics = ["MCC", "PR-AUC"]

    fig, axes = plt.subplots(1, 2, figsize=(11, max(3, len(quantum) * 1.4 + 1.5)))

    for ax, metric in zip(axes, metrics):
        matrix = np.zeros((len(quantum), len(classical)))
        for s in stat_results:
            if s["metric"] != metric:
                continue
            i = quantum.index(s["model_a"])
            j = classical.index(s["model_b"])
            matrix[i, j] = s["rank_biserial"]

        q_labels = [MODEL_LABELS.get(m, m) for m in quantum]
        c_labels = [MODEL_LABELS.get(m, m) for m in classical]

        sns.heatmap(
            matrix, ax=ax,
            xticklabels=c_labels, yticklabels=q_labels,
            annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=-1, vmax=1, center=0,
            linewidths=0.5, annot_kws={"size": 11},
        )
        ax.set_title(f"Rank-Biserial r — {metric}", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Classical model", fontsize=11)
        ax.set_ylabel("HQNN model", fontsize=11)
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=10, rotation=0)

    fig.suptitle(
        "Wilcoxon Signed-Rank Effect Size (r > 0: HQNN wins, r < 0: classical wins)\n"
        "n=5 folds, min p=0.0625",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, save_path)


def plot_ablation_vqc(shnn_mcc: float, shnn_std: float, save_path: Path) -> None:
    """Bar: SHNN full model vs VQC replaced by zeros (structural ablation)."""
    fig, ax = plt.subplots(figsize=(6, 5))

    labels = ["SHNN (full)", "SHNN (VQC → zeros)"]
    values = [shnn_mcc, 0.0]
    errors = [shnn_std, 0.0]
    colors = [COLORS["shnn"], "#B2BEC3"]

    bars = ax.bar(labels, values, yerr=errors, capsize=6,
                  color=colors, alpha=0.85, edgecolor="white",
                  error_kw={"linewidth": 1.5})
    ax.set_ylabel("MCC", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_title("Structural Ablation: Quantum Contribution to SHNN",
                 fontsize=13, fontweight="bold", pad=10)

    for bar, val, err in zip(bars, values, errors):
        label_y = bar.get_height() + err + 0.025
        ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, save_path)


def plot_ablation_noise(noise_data: list[dict], save_path: Path) -> None:
    """Line plot: MCC vs depolarizing noise probability."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ps = [d["depolarizing_p"] for d in noise_data]
    mccs_tuned = [d["mcc_tuned_threshold"] for d in noise_data]
    mccs_fixed = [d["mcc_fixed_threshold"] for d in noise_data]

    ax.plot(ps, mccs_tuned, marker="o", linewidth=2, color=COLORS["shnn"],
            label="Tuned threshold", markersize=7)
    ax.plot(ps, mccs_fixed, marker="s", linewidth=2, color=COLORS["parallel"],
            linestyle="--", label="Fixed threshold (0.94)", markersize=7)

    ax.set_xlabel("Depolarizing noise probability (p)", fontsize=12)
    ax.set_ylabel("MCC", fontsize=12)
    ax.set_title("NISQ Noise Robustness: SHNN MCC vs Depolarizing Noise",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(left=-0.001)

    fig.tight_layout()
    _save(fig, save_path)


def plot_confusion_matrices(results: list[dict], save_path: Path) -> None:
    """Grid of confusion matrices. results: list of dicts with name, y_true, y_pred."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(r["y_true"], r["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
        ax.set_title(r["name"], fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, save_path)


def plot_pr_curves(results: list[dict], save_path: Path) -> None:
    """Precision-Recall curves. results: list of dicts with name, y_true, y_prob."""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results:
        precision, recall, _ = precision_recall_curve(r["y_true"], r["y_prob"])
        ap = average_precision_score(r["y_true"], r["y_prob"])
        ax.plot(recall, precision, label=f"{r['name']} (AP={ap:.3f})",
                color=COLORS.get(r["name"].lower()), linewidth=2)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    _save(fig, save_path)
