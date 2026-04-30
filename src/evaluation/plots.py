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


def plot_aggregated_confusion_matrices(
    fold_data: dict[str, list[list[list[int]]]],
    save_path: Path,
) -> None:
    """Heatmap grid of confusion matrices aggregated across folds.

    Parameters
    ----------
    fold_data : {model_name: [cm_fold0, cm_fold1, ...]} where each cm is [[TN,FP],[FN,TP]]
    """
    models = list(fold_data.keys())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        cms = np.array(fold_data[model])       # (n_folds, 2, 2)
        cm_sum = cms.sum(axis=0).astype(int)   # aggregate across folds
        label = MODEL_LABELS.get(model, model)
        color = COLORS.get(model, "#636E72")

        # Custom colormap anchored to model colour
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("model", ["#ffffff", color])

        sns.heatmap(
            cm_sum, annot=True, fmt="d", cmap=cmap, ax=ax,
            xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"],
            linewidths=0.5, linecolor="white",
        )
        tn, fp, fn, tp = cm_sum.ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        ax.set_title(
            f"{label}\nRecall={recall:.2f}  Precision={precision:.2f}",
            fontsize=11, fontweight="bold", pad=8,
        )
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)

    fig.suptitle("Aggregated Confusion Matrices (5-Fold CV, summed)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)


def plot_mcc_vs_prauc(results: list[AggregatedMetrics], save_path: Path) -> None:
    """2D scatter: MCC vs PR-AUC with std error bars — shows both primary metrics at once."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for r in results:
        color = COLORS.get(r.model_name, "#636E72")
        label = MODEL_LABELS.get(r.model_name, r.model_name)
        ax.errorbar(
            r.pr_auc_mean, r.mcc_mean,
            xerr=r.pr_auc_std, yerr=r.mcc_std,
            fmt="o", color=color, markersize=10,
            capsize=4, elinewidth=1.2, alpha=0.85,
        )
        ax.annotate(label, (r.pr_auc_mean, r.mcc_mean),
                    textcoords="offset points", xytext=(8, 5), fontsize=10)

    ax.set_xlabel("PR-AUC (mean ± std)", fontsize=12)
    ax.set_ylabel("MCC (mean ± std)", fontsize=12)
    ax.set_title("Performance Space: MCC vs PR-AUC per Model",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0.5, 0.85)
    ax.set_ylim(0.4, 0.8)

    # Quadrant annotation
    ax.axvline(0.65, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.axhline(0.58, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)

    fig.tight_layout()
    _save(fig, save_path)


def plot_efficiency_frontier(results: list[AggregatedMetrics], save_path: Path) -> None:
    """MCC vs log(params) with Pareto efficiency frontier highlighted."""
    fig, ax = plt.subplots(figsize=(9, 6))

    # Compute Pareto frontier: non-dominated points (max MCC, min params)
    sorted_by_params = sorted(results, key=lambda r: r.param_count["total"])
    frontier = []
    best_mcc = -1.0
    for r in sorted_by_params:
        if r.mcc_mean > best_mcc:
            best_mcc = r.mcc_mean
            frontier.append(r)

    frontier_names = {r.model_name for r in frontier}

    for r in results:
        total = r.param_count["total"]
        color = COLORS.get(r.model_name, "#636E72")
        label = MODEL_LABELS.get(r.model_name, r.model_name)
        on_frontier = r.model_name in frontier_names
        ax.scatter(total, r.mcc_mean,
                   s=140 if on_frontier else 80,
                   c=color,
                   edgecolors="black" if on_frontier else "white",
                   linewidth=1.5 if on_frontier else 0.8,
                   zorder=5)
        ax.errorbar(total, r.mcc_mean, yerr=r.mcc_std,
                    fmt="none", ecolor=color, capsize=4, alpha=0.6)
        ax.annotate(label, (total, r.mcc_mean),
                    textcoords="offset points", xytext=(8, 6), fontsize=10)

    # Draw frontier as step line
    fx = [r.param_count["total"] for r in frontier]
    fy = [r.mcc_mean for r in frontier]
    ax.step(fx + [fx[-1] * 3], fy + [fy[-1]],
            where="post", color="#2D3436", linewidth=1.5,
            linestyle="--", alpha=0.6, label="Efficiency frontier")

    ax.set_xlabel("Trainable Parameters (log scale)", fontsize=12)
    ax.set_ylabel("MCC (mean ± std)", fontsize=12)
    ax.set_title("Efficiency Frontier: MCC vs Model Complexity",
                 fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, save_path)


def plot_fold_trajectories(results: list[AggregatedMetrics], save_path: Path) -> None:
    """Line plot of fold-by-fold MCC for each model — shows directional trends."""
    fig, ax = plt.subplots(figsize=(9, 5))

    folds = list(range(5))
    for r in results:
        color = COLORS.get(r.model_name, "#636E72")
        label = MODEL_LABELS.get(r.model_name, r.model_name)
        ax.plot(folds, r.fold_mccs, marker="o", linewidth=1.8,
                color=color, label=label, markersize=6, alpha=0.85)

    ax.set_xlabel("Fold", fontsize=12)
    ax.set_ylabel("MCC", fontsize=12)
    ax.set_xticks(folds)
    ax.set_xticklabels([f"Fold {i}" for i in folds])
    ax.set_title("Fold-by-Fold MCC Trajectories", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.set_ylim(0.3, 0.8)
    fig.tight_layout()
    _save(fig, save_path)


def plot_vqc_circuit(n_qubits: int, n_layers: int, save_path: Path) -> None:
    """Gate-level VQC circuit diagram via PennyLane draw_mpl."""
    import pennylane as qml
    import torch

    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    inputs = torch.zeros(n_qubits)
    weights = torch.zeros(n_layers, n_qubits, 3)

    fig, _ = qml.draw_mpl(circuit, level="device")(inputs, weights)
    fig.set_size_inches(16, 6)
    fig.suptitle(
        f"VQC Circuit: AngleEmbedding + StronglyEntanglingLayers "
        f"({n_qubits} qubits, {n_layers} layers, {n_layers * n_qubits * 3} parameters)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    _save(fig, save_path)


def plot_class_imbalance(n_legit: int, n_fraud: int, save_path: Path) -> None:
    """Side-by-side bar + inset pie showing the extreme class imbalance."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: absolute counts (log scale so fraud is visible)
    ax = axes[0]
    bars = ax.bar(["Legitimate", "Fraud"], [n_legit, n_fraud],
                  color=["#636E72", COLORS["shnn"]], alpha=0.85, edgecolor="white")
    ax.set_yscale("log")
    ax.set_ylabel("Number of transactions (log scale)", fontsize=11)
    ax.set_title("Absolute Class Counts", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, [n_legit, n_fraud]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.3,
                f"{val:,}", ha="center", va="bottom", fontsize=11)

    # Right: pie with exploded fraud slice
    ax2 = axes[1]
    total = n_legit + n_fraud
    fraud_pct = n_fraud / total * 100
    legit_pct = n_legit / total * 100
    wedges, texts, autotexts = ax2.pie(
        [n_legit, n_fraud],
        labels=[f"Legitimate\n({legit_pct:.2f}%)", f"Fraud\n({fraud_pct:.2f}%)"],
        colors=["#636E72", COLORS["shnn"]],
        explode=(0, 0.12),
        autopct="%1.2f%%",
        startangle=90,
        textprops={"fontsize": 10},
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax2.set_title("Class Distribution", fontsize=12, fontweight="bold")

    fig.suptitle(
        f"Credit Card Fraud Dataset — Extreme Class Imbalance "
        f"(n = {total:,})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, save_path)


def plot_parameter_breakdown(results: list[AggregatedMetrics], save_path: Path) -> None:
    """Stacked bar: quantum vs classical parameter count per model."""
    fig, ax = plt.subplots(figsize=(10, 5))

    names = [MODEL_LABELS.get(r.model_name, r.model_name) for r in results]
    classical_params = [r.param_count.get("classical", r.param_count.get("total", 0))
                        for r in results]
    quantum_params = [r.param_count.get("quantum", 0) for r in results]

    x = np.arange(len(names))
    b_classical = ax.bar(x, classical_params, label="Classical parameters",
                         color="#636E72", alpha=0.85, edgecolor="white")
    b_quantum = ax.bar(x, quantum_params, bottom=classical_params,
                       label="Quantum parameters", color=COLORS["shnn"],
                       alpha=0.85, edgecolor="white")

    ax.set_yscale("log")
    # Floor must be below the smallest classical count (SHNN=74) so that bar is visible
    all_nonzero = [v for v in classical_params + quantum_params if v > 0]
    ax.set_ylim(bottom=min(all_nonzero) * 0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Parameter count (log scale)", fontsize=12)
    ax.set_title("Trainable Parameter Breakdown: Classical vs Quantum",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    for xi, (c, q) in enumerate(zip(classical_params, quantum_params)):
        total = c + q
        ax.text(xi, total * 1.4, f"{total:,}", ha="center", va="bottom", fontsize=8)
        if q > 0:
            # For small bars place the split label above; for tall bars, inside
            y_lim_min = ax.get_ylim()[0]
            inside_height = total / y_lim_min  # ratio of bar height in log space
            if inside_height > 3:
                ax.text(xi, total * 0.4, f"{q}q / {c}c",
                        ha="center", va="center", fontsize=7.5,
                        color="white", fontweight="bold")
            else:
                ax.text(xi, total * 2.2, f"{q}q / {c}c",
                        ha="center", va="bottom", fontsize=7.5,
                        color="#2D3436", fontweight="bold")

    fig.tight_layout()
    _save(fig, save_path)


def plot_hilbert_space(n_qubits_highlight: int, save_path: Path) -> None:
    """2^n exponential growth curve with a marker at the thesis qubit count."""
    fig, ax = plt.subplots(figsize=(8, 5))

    n_range = np.arange(1, 21)
    dims = 2.0 ** n_range

    ax.plot(n_range, dims, linewidth=2.5, color=COLORS["shnn"], zorder=3)
    ax.fill_between(n_range, dims, alpha=0.08, color=COLORS["shnn"])

    # Highlight thesis qubit count
    highlight_dim = 2 ** n_qubits_highlight
    ax.scatter([n_qubits_highlight], [highlight_dim], s=150, zorder=5,
               color=COLORS["shnn"], edgecolors="black", linewidth=1.5)
    ax.annotate(
        f"  This thesis\n  n={n_qubits_highlight} qubits\n  2^{n_qubits_highlight} = {int(highlight_dim)} dims",
        (n_qubits_highlight, highlight_dim),
        textcoords="offset points", xytext=(10, -30),
        fontsize=10, color="#2D3436",
        arrowprops={"arrowstyle": "->", "color": "#2D3436", "lw": 1.2},
    )

    # Reference lines for classical bits
    for bits, label in [(8, "8-bit"), (16, "16-bit"), (32, "32-bit")]:
        if bits <= 20:
            ax.axhline(2 ** bits, color="gray", linewidth=0.8,
                       linestyle=":", alpha=0.5)
            ax.text(20.2, 2 ** bits, label, va="center", fontsize=8, color="gray")

    ax.set_xlabel("Number of qubits (n)", fontsize=12)
    ax.set_ylabel("Hilbert space dimension (2ⁿ)", fontsize=12)
    ax.set_title("Exponential Growth of Quantum State Space",
                 fontsize=14, fontweight="bold")
    ax.set_yscale("log", base=2)
    ax.set_xlim(1, 21)
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))

    fig.tight_layout()
    _save(fig, save_path)


def plot_shnn_architecture(save_path: Path) -> None:
    """Flow diagram of the full SHNN hybrid pipeline."""
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 1)
    ax.axis("off")

    blocks = [
        (0.4,  "Input\n(8 features)",     "#DFE6E9", "black"),
        (2.0,  "Linear\n8→8",             "#DFE6E9", "black"),
        (3.6,  "PiSigmoid\n×π",           "#DFE6E9", "black"),
        (5.5,  "VQC\n8 qubits · 2 layers\n48 params", "#D8CCF5", COLORS["shnn"]),
        (7.8,  "<Z0>\nMeasurement",         "#D8CCF5", COLORS["shnn"]),
        (9.4,  "Linear\n1→1",             "#DFE6E9", "black"),
        (11.0, "Sigmoid\n→ P(fraud)",      "#DFE6E9", "black"),
    ]

    box_w, box_h = 1.35, 0.52
    cy = 0.55

    for i, (cx, label, facecolor, edgecolor) in enumerate(blocks):
        fancy = plt.matplotlib.patches.FancyBboxPatch(
            (cx - box_w / 2, cy - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.04",
            facecolor=facecolor, edgecolor=edgecolor, linewidth=1.8, zorder=3,
        )
        ax.add_patch(fancy)
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=8.5, fontweight="bold", zorder=4, color="#2D3436")

        if i < len(blocks) - 1:
            next_cx = blocks[i + 1][0]
            ax.annotate(
                "", xy=(next_cx - box_w / 2 - 0.03, cy),
                xytext=(cx + box_w / 2 + 0.03, cy),
                arrowprops={"arrowstyle": "->", "color": "#636E72", "lw": 1.5},
                zorder=2,
            )

    # Bracket: quantum section
    q_start = blocks[3][0] - box_w / 2 - 0.05
    q_end = blocks[4][0] + box_w / 2 + 0.05
    bracket_y = cy - box_h / 2 - 0.12
    ax.annotate(
        "", xy=(q_end, bracket_y), xytext=(q_start, bracket_y),
        arrowprops={"arrowstyle": "<->", "color": COLORS["shnn"], "lw": 1.5},
    )
    ax.text((q_start + q_end) / 2, bracket_y - 0.08,
            "Quantum module (PennyLane · lightning.qubit)",
            ha="center", fontsize=8.5, color=COLORS["shnn"], style="italic")

    ax.set_title("SHNN — Sequential Hybrid Neural Network Architecture",
                 fontsize=13, fontweight="bold", pad=8)
    fig.tight_layout()
    _save(fig, save_path)


def plot_pca_scree(X: np.ndarray, n_highlight: int, save_path: Path) -> None:
    """Scree plot of cumulative explained variance, vertical line at n_highlight."""
    from sklearn.decomposition import PCA

    pca = PCA().fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100

    fig, ax = plt.subplots(figsize=(9, 5))

    components = np.arange(1, len(cumvar) + 1)
    ax.plot(components, cumvar, marker="o", linewidth=2,
            color=COLORS["shnn"], markersize=5)
    ax.fill_between(components, cumvar, alpha=0.08, color=COLORS["shnn"])

    ax.axvline(n_highlight, color=COLORS["shnn"], linewidth=1.8,
               linestyle="--", alpha=0.8)
    ax.axhline(cumvar[n_highlight - 1], color="#636E72", linewidth=1,
               linestyle=":", alpha=0.6)
    ax.scatter([n_highlight], [cumvar[n_highlight - 1]], s=120, zorder=5,
               color=COLORS["shnn"], edgecolors="black", linewidth=1.5)
    ax.annotate(
        f"  {n_highlight} components\n  {cumvar[n_highlight-1]:.1f}% variance",
        (n_highlight, cumvar[n_highlight - 1]),
        textcoords="offset points", xytext=(10, -25),
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "#2D3436", "lw": 1.2},
    )

    ax.set_xlabel("Number of principal components", fontsize=12)
    ax.set_ylabel("Cumulative explained variance (%)", fontsize=12)
    ax.set_title("PCA Scree Plot — Feature Dimensionality Reduction",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(1, len(cumvar))
    ax.set_ylim(0, 102)
    fig.tight_layout()
    _save(fig, save_path)


def plot_smote_illustration(
    X_before: np.ndarray, y_before: np.ndarray,
    X_after: np.ndarray, y_after: np.ndarray,
    save_path: Path,
) -> None:
    """2D PCA projection of fraud samples before/after SMOTE."""
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, X, y, title in [
        (axes[0], X_before, y_before, "Before SMOTE"),
        (axes[1], X_after, y_after, "After SMOTE"),
    ]:
        # Fit a fresh 2D PCA for each dataset independently (feature counts differ)
        pca = PCA(n_components=2).fit(X)
        X_2d = pca.transform(X)
        legit_mask = y == 0
        fraud_mask = y == 1

        # Sample legit points to avoid overplotting
        legit_idx = np.where(legit_mask)[0]
        sample_idx = np.random.default_rng(42).choice(
            legit_idx, size=min(500, len(legit_idx)), replace=False
        )

        ax.scatter(X_2d[sample_idx, 0], X_2d[sample_idx, 1],
                   c="#DFE6E9", s=12, alpha=0.5, label="Legitimate (sample)", zorder=2)
        fraud_idx = np.where(fraud_mask)[0]
        fraud_sample = np.random.default_rng(42).choice(
            fraud_idx, size=min(2000, len(fraud_idx)), replace=False
        )
        ax.scatter(X_2d[fraud_sample, 0], X_2d[fraud_sample, 1],
                   c=COLORS["shnn"], s=20, alpha=0.7,
                   label=f"Fraud (n={fraud_mask.sum():,})", zorder=3)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("PC 1", fontsize=10)
        ax.set_ylabel("PC 2", fontsize=10)
        ax.legend(fontsize=9)

    fig.suptitle("SMOTE: Synthetic Minority Oversampling in PCA Space (fold 0 train set)",
                 fontsize=12, fontweight="bold")
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
