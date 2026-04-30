#!/usr/bin/env python3
"""
run_statistics.py
=================
Run Wilcoxon signed-rank + rank-biserial on completed benchmark results.

Assembles fold-level MCC and PR-AUC from existing result files (quantum
models from results/folds/*.json, classical models from results/metrics/
benchmark_*.json) without retraining, then runs pairwise HQNN vs classical
comparisons and saves to results/metrics/statistics_<timestamp>.json.

Usage
-----
    pixi run python scripts/run_statistics.py
    python scripts/run_statistics.py --folds-dir results/folds --metrics-dir results/metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from src.evaluation.statistics import compare_models

console = Console()

QUANTUM_MODELS = {"shnn", "parallel"}
CLASSICAL_MODELS = {"snn", "tabnet", "resnet", "ftt", "saint"}


def _load_quantum_folds(folds_dir: Path) -> dict[str, dict[str, list[float]]]:
    """Load per-fold MCC and PR-AUC for quantum models from individual fold files."""
    results: dict[str, dict[str, list[float]]] = {}
    for path in sorted(folds_dir.glob("*.json")):
        d = json.loads(path.read_text())
        model = d["model_name"]
        if model not in QUANTUM_MODELS:
            continue
        if model not in results:
            results[model] = {"mcc": [], "pr_auc": [], "fold_idx": []}
        results[model]["fold_idx"].append(d["fold_idx"])
        results[model]["mcc"].append(d["mcc"])
        results[model]["pr_auc"].append(d["pr_auc"])

    # Sort by fold index so scores are in order 0..4
    for model, data in results.items():
        order = sorted(range(len(data["fold_idx"])), key=lambda i: data["fold_idx"][i])
        data["mcc"] = [data["mcc"][i] for i in order]
        data["pr_auc"] = [data["pr_auc"][i] for i in order]

    return results


def _load_classical_folds(metrics_dir: Path) -> dict[str, dict[str, list[float]]]:
    """Load fold-level MCC and PR-AUC for classical models from benchmark JSON files.

    Multiple benchmark files may exist (one per model run). The most recent
    file containing each model's data is used.
    """
    results: dict[str, dict[str, list[float]]] = {}
    for path in sorted(metrics_dir.glob("benchmark_*.json")):
        d = json.loads(path.read_text())
        for entry in d.get("aggregated", []):
            model = entry["model_name"]
            if model not in CLASSICAL_MODELS:
                continue
            # Later files overwrite earlier ones (sorted ascending → last wins)
            results[model] = {
                "mcc": entry["fold_mccs"],
                "pr_auc": entry["fold_pr_aucs"],
            }
    return results


def _print_results(stat_results: list[dict]) -> None:
    table = Table(title="Wilcoxon Signed-Rank + Rank-Biserial (HQNN vs Classical)", show_lines=True)
    table.add_column("HQNN", style="bold cyan")
    table.add_column("Classical", style="bold")
    table.add_column("Metric")
    table.add_column("r (rank-biserial)", justify="right")
    table.add_column("W", justify="right")
    table.add_column("p", justify="right")
    table.add_column("Effect")

    for r in stat_results:
        effect = r["interpretation"].split(" effect")[0]
        direction = "+" if r["rank_biserial"] > 0 else "-"
        table.add_row(
            r["model_a"].upper(),
            r["model_b"].upper(),
            r["metric"],
            f"{r['rank_biserial']:+.3f}",
            f"{r['wilcoxon_statistic']:.1f}",
            f"{r['wilcoxon_p']:.4f}",
            f"{effect} ({direction})",
        )

    console.print(table)
    console.print(
        "[dim]Note: n=5 folds → minimum achievable p=0.0625. "
        "Rank-biserial r is the primary effect size measure.[/dim]\n"
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console)],
    )
    logger = logging.getLogger(__name__)

    p = argparse.ArgumentParser(description="Run statistical tests on completed benchmark results")
    p.add_argument("--folds-dir", type=Path, default=Path("results/folds"))
    p.add_argument("--metrics-dir", type=Path, default=Path("results/metrics"))
    args = p.parse_args()

    logger.info("Loading quantum fold results from %s", args.folds_dir)
    quantum = _load_quantum_folds(args.folds_dir)

    logger.info("Loading classical fold results from %s", args.metrics_dir)
    classical = _load_classical_folds(args.metrics_dir)

    logger.info("Quantum models found: %s", sorted(quantum))
    logger.info("Classical models found: %s", sorted(classical))

    if not quantum:
        raise SystemExit("No quantum model fold files found — check --folds-dir")
    if not classical:
        raise SystemExit("No classical model metrics files found — check --metrics-dir")

    stat_results = []
    for q_name, q_data in sorted(quantum.items()):
        for c_name, c_data in sorted(classical.items()):
            for metric in ("MCC", "PR-AUC"):
                key = "mcc" if metric == "MCC" else "pr_auc"
                sr = compare_models(
                    q_name, q_data[key],
                    c_name, c_data[key],
                    metric=metric,
                )
                stat_results.append(sr.to_dict())
                logger.info("[%s vs %s | %s] %s", q_name, c_name, metric, sr.interpretation)

    _print_results(stat_results)

    out_path = args.metrics_dir / f"statistics_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(stat_results, indent=2))
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
