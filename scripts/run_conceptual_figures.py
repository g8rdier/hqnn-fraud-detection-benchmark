#!/usr/bin/env python3
"""
run_conceptual_figures.py
=========================
Generate conceptual and data-pipeline figures for the thesis.

Produces 6 figures that visualise the dataset, architecture, and quantum
theory context — content that cannot be conveyed by text alone.

Usage
-----
    pixi run python scripts/run_conceptual_figures.py
    python scripts/run_conceptual_figures.py --data data/raw/creditcard.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.logging import RichHandler

from src.config import load_config
from src.data.loader import load_dataset
from src.data.cv import create_folds
from src.evaluation.metrics import AggregatedMetrics
from src.evaluation.plots import (
    plot_class_imbalance,
    plot_hilbert_space,
    plot_parameter_breakdown,
    plot_pca_scree,
    plot_shnn_architecture,
    plot_smote_illustration,
)

console = Console()

QUANTUM_MODELS = {"shnn", "parallel"}
CLASSICAL_MODELS = {"snn", "tabnet", "resnet", "ftt", "saint"}
MODEL_ORDER = ["shnn", "parallel", "snn", "tabnet", "resnet", "ftt", "saint"]


def _load_all_results(folds_dir: Path, metrics_dir: Path) -> list[AggregatedMetrics]:
    buckets: dict[str, list[dict]] = {}
    for path in sorted(folds_dir.glob("*.json")):
        d = json.loads(path.read_text())
        model = d["model_name"]
        if model in QUANTUM_MODELS:
            buckets.setdefault(model, []).append(d)

    results = []
    for model, folds in buckets.items():
        folds.sort(key=lambda x: x["fold_idx"])
        mccs = [f["mcc"] for f in folds]
        pr_aucs = [f["pr_auc"] for f in folds]
        f1s = [f["f1_fraud"] for f in folds]
        rocs = [f["roc_auc"] for f in folds]
        results.append(AggregatedMetrics(
            model_name=model,
            mcc_mean=statistics.mean(mccs), mcc_std=statistics.stdev(mccs),
            pr_auc_mean=statistics.mean(pr_aucs), pr_auc_std=statistics.stdev(pr_aucs),
            f1_fraud_mean=statistics.mean(f1s), f1_fraud_std=statistics.stdev(f1s),
            roc_auc_mean=statistics.mean(rocs), roc_auc_std=statistics.stdev(rocs),
            param_count=folds[0]["param_count"],
            fold_mccs=mccs, fold_pr_aucs=pr_aucs,
        ))

    seen: dict[str, AggregatedMetrics] = {}
    for path in sorted(metrics_dir.glob("benchmark_*.json")):
        d = json.loads(path.read_text())
        for entry in d.get("aggregated", []):
            if entry["model_name"] in CLASSICAL_MODELS:
                seen[entry["model_name"]] = AggregatedMetrics(**entry)
    results += list(seen.values())

    order = {name: i for i, name in enumerate(MODEL_ORDER)}
    return sorted(results, key=lambda r: order.get(r.model_name, 99))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console)],
    )
    logger = logging.getLogger(__name__)

    p = argparse.ArgumentParser(description="Generate conceptual thesis figures")
    p.add_argument("--data", type=Path, default=Path("data/raw/creditcard.csv"))
    p.add_argument("--folds-dir", type=Path, default=Path("results/folds"))
    p.add_argument("--metrics-dir", type=Path, default=Path("results/metrics"))
    p.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    args = p.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(Path("configs/default.yaml"))

    # ── 1. Class imbalance ────────────────────────────────────────────────
    logger.info("Loading dataset for class counts...")
    X, y = load_dataset(cfg.data)
    y_np = y.values if hasattr(y, "values") else np.asarray(y)
    n_fraud = int(y_np.sum())
    n_legit = len(y_np) - n_fraud
    logger.info("Dataset: %d legit, %d fraud", n_legit, n_fraud)
    plot_class_imbalance(n_legit, n_fraud, out / "class_imbalance.png")

    # ── 2. Parameter breakdown ────────────────────────────────────────────
    all_results = _load_all_results(args.folds_dir, args.metrics_dir)
    plot_parameter_breakdown(all_results, out / "parameter_breakdown.png")

    # ── 3. Hilbert space growth ───────────────────────────────────────────
    plot_hilbert_space(cfg.shnn.vqc.n_qubits, out / "hilbert_space.png")

    # ── 4. SHNN architecture diagram ─────────────────────────────────────
    plot_shnn_architecture(out / "shnn_architecture.png")

    # ── 5. PCA scree plot ─────────────────────────────────────────────────
    # Apply RobustScaler + MinMaxScaler first (matching the actual pipeline),
    # then show how much variance PCA captures at each component count.
    logger.info("Fitting PCA scree on preprocessed feature matrix...")
    from sklearn.preprocessing import RobustScaler, MinMaxScaler
    from sklearn.model_selection import StratifiedKFold

    X_np = X.values if hasattr(X, "values") else np.asarray(X)

    skf = StratifiedKFold(n_splits=cfg.cv.n_folds, shuffle=True,
                          random_state=cfg.seed)
    splits = list(skf.split(X_np, y_np))
    train_idx, _ = splits[0]
    X_train_raw = X_np[train_idx]
    y_train_raw = y_np[train_idx]

    X_scaled = RobustScaler().fit_transform(X_train_raw)
    X_scaled = MinMaxScaler().fit_transform(X_scaled)
    plot_pca_scree(X_scaled, n_highlight=cfg.shnn.vqc.n_qubits,
                   save_path=out / "pca_scree.png")

    # ── 6. SMOTE illustration (fold 0 train set) ──────────────────────────
    # Before SMOTE: preprocessed (RobustScaler + MinMaxScaler + PCA) but no SMOTE
    # After SMOTE: same preprocessing + SMOTE applied
    logger.info("Building fold 0 for SMOTE illustration...")
    from src.data.preprocessing import FoldPreprocessor

    pre = FoldPreprocessor(config=cfg.preprocessing)
    X_pre_smote = pre.fit_transform(X_train_raw)   # 8D, no SMOTE
    y_pre_smote = y_train_raw

    # Apply SMOTE to get the after state
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(k_neighbors=cfg.smote.k_neighbors,
                  random_state=cfg.smote.random_state)
    X_post_smote, y_post_smote = smote.fit_resample(X_pre_smote, y_pre_smote)

    plot_smote_illustration(
        X_pre_smote, y_pre_smote,
        X_post_smote, y_post_smote,
        save_path=out / "smote_illustration.png",
    )

    console.print(f"\n[bold green]All conceptual figures saved to {out}/[/]")


if __name__ == "__main__":
    main()
