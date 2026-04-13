#!/usr/bin/env python3
"""
run_plots.py
============
Regenerate figures from saved benchmark results (no retraining needed).

Usage
-----
    pixi run plots
    python scripts/run_plots.py --results results/metrics/benchmark_*.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from src.evaluation.metrics import AggregatedMetrics
from src.evaluation.plots import plot_metric_comparison, plot_parameter_efficiency

console = Console()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerate benchmark figures")
    p.add_argument("--results", type=Path, required=True, help="Path to benchmark JSON")
    p.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console)],
    )
    args = _parse_args()

    with open(args.results) as f:
        data = json.load(f)

    aggregated = [AggregatedMetrics(**r) for r in data["aggregated"]]

    out = args.output_dir
    plot_metric_comparison(aggregated, out / "metric_comparison.png")
    plot_parameter_efficiency(aggregated, out / "parameter_efficiency.png")

    console.print(f"[bold green]Figures saved to {out}/[/]")


if __name__ == "__main__":
    main()
