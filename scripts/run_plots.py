#!/usr/bin/env python3
"""
run_plots.py
============
Generate all benchmark figures from saved results (no retraining needed).

Assembles data from the scattered result files produced by separate per-model
runs: quantum models from results/folds/*.json, classical from
results/metrics/benchmark_*.json, statistics from results/metrics/statistics_*.json,
and ablation data from results/ablation_noise.json.

Usage
-----
    pixi run plots
    python scripts/run_plots.py
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from src.evaluation.metrics import AggregatedMetrics
from src.evaluation.plots import (
    plot_ablation_noise,
    plot_ablation_vqc,
    plot_efficiency_comparison,
    plot_fold_consistency,
    plot_metric_comparison,
    plot_parameter_efficiency,
    plot_statistical_heatmap,
)

console = Console()

QUANTUM_MODELS = {"shnn", "parallel"}
CLASSICAL_MODELS = {"snn", "tabnet", "resnet", "ftt", "saint"}
MODEL_ORDER = ["shnn", "parallel", "snn", "tabnet", "resnet", "ftt", "saint"]


def _load_quantum(folds_dir: Path) -> list[AggregatedMetrics]:
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
            mcc_mean=statistics.mean(mccs),
            mcc_std=statistics.stdev(mccs),
            pr_auc_mean=statistics.mean(pr_aucs),
            pr_auc_std=statistics.stdev(pr_aucs),
            f1_fraud_mean=statistics.mean(f1s),
            f1_fraud_std=statistics.stdev(f1s),
            roc_auc_mean=statistics.mean(rocs),
            roc_auc_std=statistics.stdev(rocs),
            param_count=folds[0]["param_count"],
            fold_mccs=mccs,
            fold_pr_aucs=pr_aucs,
        ))
    return results


def _load_classical(metrics_dir: Path) -> list[AggregatedMetrics]:
    seen: dict[str, AggregatedMetrics] = {}
    for path in sorted(metrics_dir.glob("benchmark_*.json")):
        d = json.loads(path.read_text())
        for entry in d.get("aggregated", []):
            model = entry["model_name"]
            if model in CLASSICAL_MODELS:
                seen[model] = AggregatedMetrics(**entry)
    return list(seen.values())


def _ordered(results: list[AggregatedMetrics]) -> list[AggregatedMetrics]:
    order = {name: i for i, name in enumerate(MODEL_ORDER)}
    return sorted(results, key=lambda r: order.get(r.model_name, 99))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console)],
    )
    logger = logging.getLogger(__name__)

    p = argparse.ArgumentParser(description="Generate all benchmark figures")
    p.add_argument("--folds-dir", type=Path, default=Path("results/folds"))
    p.add_argument("--metrics-dir", type=Path, default=Path("results/metrics"))
    p.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    args = p.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # ── Load model results ─────────────────────────────────────────────────
    all_results = _ordered(_load_quantum(args.folds_dir) + _load_classical(args.metrics_dir))
    logger.info("Loaded %d models: %s", len(all_results), [r.model_name for r in all_results])

    shnn = next(r for r in all_results if r.model_name == "shnn")

    # ── 1. Metric comparison ───────────────────────────────────────────────
    plot_metric_comparison(all_results, out / "metric_comparison.png")

    # ── 2. Parameter efficiency scatter ───────────────────────────────────
    plot_parameter_efficiency(all_results, out / "parameter_efficiency.png")

    # ── 3. Efficiency bar (MCC/kParam + PR-AUC/kParam) ────────────────────
    plot_efficiency_comparison(all_results, out / "efficiency_comparison.png")

    # ── 4. Fold consistency box plots ─────────────────────────────────────
    plot_fold_consistency(all_results, out / "fold_consistency.png")

    # ── 5. Statistical heatmap ────────────────────────────────────────────
    stat_files = sorted(args.metrics_dir.glob("statistics_*.json"))
    if stat_files:
        stat_results = json.loads(stat_files[-1].read_text())
        plot_statistical_heatmap(stat_results, out / "statistical_heatmap.png")
    else:
        logger.warning("No statistics_*.json found — skipping heatmap")

    # ── 6. VQC ablation bar ───────────────────────────────────────────────
    plot_ablation_vqc(
        shnn_mcc=shnn.mcc_mean,
        shnn_std=shnn.mcc_std,
        save_path=out / "ablation_vqc.png",
    )

    # ── 7. Noise ablation curve ───────────────────────────────────────────
    noise_path = Path("results/ablation_noise.json")
    if noise_path.exists():
        noise_data = json.loads(noise_path.read_text())
        plot_ablation_noise(noise_data, out / "ablation_noise.png")
    else:
        logger.warning("results/ablation_noise.json not found — skipping noise plot")

    console.print(f"\n[bold green]All figures saved to {out}/[/]")


if __name__ == "__main__":
    main()
