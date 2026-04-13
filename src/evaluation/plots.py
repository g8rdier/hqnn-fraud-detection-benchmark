"""
plots.py
========
Visualization for the HQNN benchmark.

Generates publication-quality figures for:
  - Metric comparison bar charts (MCC, PR-AUC with error bars)
  - Parameter efficiency scatter (MCC vs param count)
  - PR curves per model
  - Confusion matrices
  - Training curves (loss + val MCC)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.evaluation.metrics import AggregatedMetrics

logger = logging.getLogger(__name__)

# Consistent styling
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "shnn": "#6C5CE7",
    "parallel": "#0984E3",
    "tabnet": "#E17055",
    "snn": "#00B894",
}


def plot_metric_comparison(
    results: list[AggregatedMetrics],
    save_path: Path,
) -> None:
    """Bar chart of MCC and PR-AUC with std error bars."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = [r.model_name for r in results]
    colors = [COLORS.get(n.lower(), "#636E72") for n in names]

    for ax, metric, means, stds, title in [
        (axes[0], "MCC",
         [r.mcc_mean for r in results], [r.mcc_std for r in results], "MCC (primary)"),
        (axes[1], "PR-AUC",
         [r.pr_auc_mean for r in results], [r.pr_auc_std for r in results], "PR-AUC (primary)"),
    ]:
        bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="white")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.05)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    fig.suptitle("Model Comparison (5-Fold CV)", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_parameter_efficiency(
    results: list[AggregatedMetrics],
    save_path: Path,
) -> None:
    """Scatter: MCC vs total parameter count. The core thesis metric."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for r in results:
        total = r.param_count.get("total", 0)
        color = COLORS.get(r.model_name.lower(), "#636E72")
        ax.scatter(total, r.mcc_mean, s=120, c=color, edgecolors="white", linewidth=1.5, zorder=5)
        ax.errorbar(total, r.mcc_mean, yerr=r.mcc_std, fmt="none", ecolor=color, capsize=4, alpha=0.7)
        ax.annotate(
            r.model_name, (total, r.mcc_mean),
            textcoords="offset points", xytext=(8, 8), fontsize=11,
        )

    ax.set_xlabel("Trainable Parameters", fontsize=12)
    ax.set_ylabel("MCC (mean ± std)", fontsize=12)
    ax.set_title("Parameter Efficiency: MCC vs Model Complexity", fontsize=14, fontweight="bold")
    ax.set_xscale("log")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_confusion_matrices(
    results: list[dict],
    save_path: Path,
) -> None:
    """
    Grid of confusion matrices.

    Parameters
    ----------
    results : list of dicts with keys "name", "y_true", "y_pred"
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(r["y_true"], r["y_pred"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"],
        )
        ax.set_title(r["name"], fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_pr_curves(
    results: list[dict],
    save_path: Path,
) -> None:
    """
    Precision-Recall curves.

    Parameters
    ----------
    results : list of dicts with keys "name", "y_true", "y_prob"
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score

    fig, ax = plt.subplots(figsize=(8, 6))

    for r in results:
        precision, recall, _ = precision_recall_curve(r["y_true"], r["y_prob"])
        ap = average_precision_score(r["y_true"], r["y_prob"])
        color = COLORS.get(r["name"].lower(), None)
        ax.plot(recall, precision, label=f"{r['name']} (AP={ap:.3f})", color=color, linewidth=2)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)
