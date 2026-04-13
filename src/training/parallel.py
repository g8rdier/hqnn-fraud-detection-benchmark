"""
parallel.py
===========
Parallel fold executor for concurrent cross-validation.

Each fold runs as an independent subprocess to maximize throughput on
multi-core machines. This is critical for feasibility: ~37h/fold × 5 folds
= ~185h sequential vs ~37h parallel.

Two execution modes:
  1. subprocess — launches `scripts/run_fold.py` per fold (recommended)
  2. in-process — runs folds sequentially (for debugging)
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

from src.config import BenchmarkConfig

logger = logging.getLogger(__name__)


def launch_parallel_folds(
    cfg: BenchmarkConfig,
    config_path: str | Path,
    model_name: str,
) -> list[subprocess.Popen]:
    """
    Launch one subprocess per fold.

    Each subprocess runs `scripts/run_fold.py --config <path> --model <name> --fold <i>`.
    Results are written to `results/folds/<model>_fold_<i>.json`.

    Parameters
    ----------
    cfg : BenchmarkConfig
    config_path : path to the YAML config file
    model_name : which model to train

    Returns
    -------
    List of Popen handles for monitoring.
    """
    processes: list[subprocess.Popen] = []

    for fold_idx in range(cfg.cv.n_folds):
        cmd = [
            sys.executable,
            "scripts/run_fold.py",
            "--config", str(config_path),
            "--model", model_name,
            "--fold", str(fold_idx),
        ]

        log_path = Path(cfg.paths.folds_dir) / f"{model_name}_fold_{fold_idx}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Launching fold %d/%d for %s → %s", fold_idx, cfg.cv.n_folds, model_name, log_path)

        log_fh = open(log_path, "w")
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
        processes.append(proc)

    return processes


def collect_fold_results(
    cfg: BenchmarkConfig,
    model_name: str,
) -> list[dict]:
    """
    Collect results from all completed fold JSON files.

    Returns
    -------
    List of fold result dicts, sorted by fold_idx.
    """
    results = []
    for fold_idx in range(cfg.cv.n_folds):
        path = Path(cfg.paths.folds_dir) / f"{model_name}_fold_{fold_idx}.json"
        if not path.exists():
            logger.warning("Missing result: %s", path)
            continue
        with open(path) as f:
            data = json.load(f)
        results.append(data)

    results.sort(key=lambda r: r.get("fold_idx", 0))
    return results
