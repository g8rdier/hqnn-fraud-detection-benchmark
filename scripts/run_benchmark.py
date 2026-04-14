#!/usr/bin/env python3
"""
run_benchmark.py
================
Main entry point for the HQNN Benchmark.

Runs all models through 5-fold CV, collects metrics, runs statistical tests,
and generates figures.

Usage
-----
    pixi run benchmark
    python scripts/run_benchmark.py --config configs/default.yaml
    python scripts/run_benchmark.py --models shnn parallel  # subset of models
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from src.config import BenchmarkConfig, load_config
from src.data.cv import FoldData, create_folds
from src.data.loader import load_dataset
from src.evaluation.metrics import (
    AggregatedMetrics,
    ModelMetrics,
    aggregate_fold_metrics,
    compute_metrics,
)
from src.evaluation.statistics import compare_models
from src.models.registry import build_model

console = Console()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HQNN vs Classical Deep Learning Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=None, help="YAML config file.")
    p.add_argument("--models", nargs="+", default=None, help="Subset of models to run.")
    p.add_argument("--no-plots", action="store_true", help="Skip figure generation.")
    p.add_argument("--parallel", action="store_true", help="Launch folds as subprocesses.")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    return p.parse_args()


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def _seed_everything(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_fold_pytorch(
    model_name: str,
    fold: FoldData,
    cfg: BenchmarkConfig,
) -> ModelMetrics:
    """Train a PyTorch-based model (SHNN, ParallelHybrid, SNN) on one fold."""
    from torch import nn

    from src.training.trainer import train_pytorch_model

    input_dim = fold.X_train.shape[1]
    model = build_model(model_name, input_dim, cfg)
    assert isinstance(model, nn.Module)

    checkpoint_path = (
        Path(cfg.paths.models_dir)
        / f"{model_name}_fold{fold.fold_idx}.pt"
    )

    result = train_pytorch_model(
        model=model,
        X_train=fold.X_train,
        y_train=fold.y_train,
        X_val=fold.X_val,
        y_val=fold.y_val,
        X_test=fold.X_test,
        cfg=cfg.training,
        checkpoint_path=checkpoint_path,
    )

    param_count = model.param_count() if hasattr(model, "param_count") else {
        "classical": sum(p.numel() for p in model.parameters()),
        "quantum": 0,
        "total": sum(p.numel() for p in model.parameters()),
    }

    return compute_metrics(
        model_name=model_name,
        y_true=fold.y_test,
        y_pred=result.y_pred,
        y_prob=result.y_prob,
        threshold=result.threshold,
        param_count=param_count,
    )


def _run_fold_tabnet(
    fold: FoldData,
    cfg: BenchmarkConfig,
) -> ModelMetrics:
    """Train TabNet on one fold (uses its own internal loop)."""
    from src.models.classical.tabnet_model import TabNetWrapper

    wrapper = TabNetWrapper(cfg=cfg.tabnet, training_cfg=cfg.training_tabnet)
    wrapper.fit(fold.X_train, fold.y_train, fold.X_val, fold.y_val)
    result = wrapper.predict(fold.X_test, X_val=fold.X_val, y_val=fold.y_val)

    return compute_metrics(
        model_name="tabnet",
        y_true=fold.y_test,
        y_pred=result.y_pred,
        y_prob=result.y_prob,
        threshold=0.5,
        param_count=result.param_count,
    )


def _print_results(aggregated: list[AggregatedMetrics]) -> None:
    """Print a Rich table of aggregated results."""
    table = Table(title="Benchmark Results (5-Fold CV)", show_lines=True)
    table.add_column("Model", style="bold")
    table.add_column("MCC", justify="right")
    table.add_column("PR-AUC", justify="right")
    table.add_column("F1-Fraud", justify="right")
    table.add_column("ROC-AUC", justify="right")
    table.add_column("Params", justify="right")

    for r in sorted(aggregated, key=lambda x: x.mcc_mean, reverse=True):
        table.add_row(
            r.model_name.upper(),
            f"{r.mcc_mean:.3f} ± {r.mcc_std:.3f}",
            f"{r.pr_auc_mean:.3f} ± {r.pr_auc_std:.3f}",
            f"{r.f1_fraud_mean:.3f} ± {r.f1_fraud_std:.3f}",
            f"{r.roc_auc_mean:.3f} ± {r.roc_auc_std:.3f}",
            f"{r.param_count.get('total', '?')}",
        )

    console.print(table)


def main() -> None:
    args = _parse_args()
    _setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    console.print("\n[bold magenta]╔══════════════════════════════════════════════════╗[/]")
    console.print("[bold magenta]║  HQNN Benchmark                                  ║[/]")
    console.print("[bold magenta]║  Hybrid Quantum vs Classical Deep Learning        ║[/]")
    console.print("[bold magenta]╚══════════════════════════════════════════════════╝[/]\n")

    cfg = load_config(args.config)
    _seed_everything(cfg.seed)

    models_to_run = args.models or cfg.models_to_run
    logger.info("Models: %s", models_to_run)

    # ── 1. Load and prepare data ──────────────────────────────────────────
    logger.info("Step 1/4 — Loading dataset")
    X, y = load_dataset(cfg.data)
    X_np = X.values if hasattr(X, "values") else np.asarray(X)
    y_np = y.values if hasattr(y, "values") else np.asarray(y)

    logger.info("Step 2/4 — Creating CV folds (SMOTE inside folds)")
    folds = create_folds(X_np, y_np, cfg)

    # ── 2. Train all models across all folds ──────────────────────────────
    logger.info("Step 3/4 — Training models")
    all_aggregated: list[AggregatedMetrics] = []

    for model_name in models_to_run:
        console.print(f"\n[bold cyan]{'=' * 60}[/]")
        console.print(f"[bold cyan]  Training: {model_name.upper()}[/]")
        console.print(f"[bold cyan]{'=' * 60}[/]")

        fold_metrics: list[ModelMetrics] = []

        for fold in folds:
            logger.info("── Fold %d/%d ──", fold.fold_idx + 1, cfg.cv.n_folds)
            t0 = time.perf_counter()

            if model_name == "tabnet":
                metrics = _run_fold_tabnet(fold, cfg)
            else:
                metrics = _run_fold_pytorch(model_name, fold, cfg)

            elapsed = time.perf_counter() - t0
            logger.info(
                "Fold %d: MCC=%.4f | PR-AUC=%.4f | %.1fs",
                fold.fold_idx, metrics.mcc, metrics.pr_auc, elapsed,
            )
            fold_metrics.append(metrics)

        agg = aggregate_fold_metrics(fold_metrics)
        all_aggregated.append(agg)
        logger.info(
            "%s → MCC: %.3f±%.3f | PR-AUC: %.3f±%.3f",
            model_name, agg.mcc_mean, agg.mcc_std, agg.pr_auc_mean, agg.pr_auc_std,
        )

    # ── 3. Results and statistical tests ──────────────────────────────────
    logger.info("Step 4/4 — Results and statistical analysis")
    _print_results(all_aggregated)

    # Pairwise comparisons: every HQNN vs every classical
    stat_results = []
    quantum_models = [r for r in all_aggregated if r.model_name in ("shnn", "parallel")]
    classical_models = [r for r in all_aggregated if r.model_name in ("tabnet", "snn", "ftt", "resnet")]

    for q in quantum_models:
        for c in classical_models:
            for metric, q_scores, c_scores in [
                ("MCC", q.fold_mccs, c.fold_mccs),
                ("PR-AUC", q.fold_pr_aucs, c.fold_pr_aucs),
            ]:
                sr = compare_models(q.model_name, q_scores, c.model_name, c_scores, metric)
                stat_results.append(sr)

    # Save results
    metrics_dir = Path(cfg.paths.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    results_path = metrics_dir / f"benchmark_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "aggregated": [r.to_dict() for r in all_aggregated],
                "statistics": [sr.to_dict() for sr in stat_results],
            },
            f,
            indent=2,
        )
    logger.info("Results saved to %s", results_path)

    # ── 4. Plots ──────────────────────────────────────────────────────────
    if not args.no_plots:
        from src.evaluation.plots import plot_metric_comparison, plot_parameter_efficiency

        figures_dir = Path(cfg.paths.figures_dir)
        plot_metric_comparison(all_aggregated, figures_dir / "metric_comparison.png")
        plot_parameter_efficiency(all_aggregated, figures_dir / "parameter_efficiency.png")

    console.print(f"\n[bold green]Benchmark complete. Results → {cfg.paths.results_dir}/[/]")


if __name__ == "__main__":
    main()
