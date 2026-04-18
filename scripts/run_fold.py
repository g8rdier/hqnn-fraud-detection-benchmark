#!/usr/bin/env python3
"""
run_fold.py
===========
Run a single CV fold for a single model. Designed to be launched as an
independent subprocess by the parallel executor.

Usage
-----
    python scripts/run_fold.py --config configs/default.yaml --model shnn --fold 0
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from src.config import load_config
from src.data.cv import create_folds
from src.data.loader import load_dataset
from src.evaluation.metrics import compute_metrics
from src.models.registry import build_model

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a single benchmark fold")
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)

    # Seed
    import random
    import torch

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Load data and create folds
    X, y = load_dataset(cfg.data)
    X_np = X.values if hasattr(X, "values") else np.asarray(X)
    y_np = y.values if hasattr(y, "values") else np.asarray(y)
    folds = create_folds(X_np, y_np, cfg)

    fold = folds[args.fold]
    model_name = args.model

    logger.info("=== Fold %d | Model: %s ===", args.fold, model_name)
    t0 = time.perf_counter()

    if model_name == "tabnet":
        from src.models.classical.tabnet_model import TabNetWrapper

        wrapper = TabNetWrapper(cfg=cfg.tabnet, training_cfg=cfg.training_tabnet)
        wrapper.fit(fold.X_train, fold.y_train, fold.X_val, fold.y_val)
        result = wrapper.predict(fold.X_test)
        metrics = compute_metrics(
            model_name=model_name,
            y_true=fold.y_test,
            y_pred=result.y_pred,
            y_prob=result.y_prob,
            threshold=0.5,
            param_count=result.param_count,
        )
    else:
        from torch import nn

        from src.training.trainer import train_pytorch_model

        input_dim = fold.X_train.shape[1]
        model = build_model(model_name, input_dim, cfg)
        assert isinstance(model, nn.Module)

        train_cfg = cfg.training_quantum if model_name in ("shnn", "parallel") else cfg.training
        train_result = train_pytorch_model(
            model=model,
            X_train=fold.X_train,
            y_train=fold.y_train,
            X_val=fold.X_val,
            y_val=fold.y_val,
            X_test=fold.X_test,
            cfg=train_cfg,
        )

        param_count = model.param_count() if hasattr(model, "param_count") else {
            "classical": sum(p.numel() for p in model.parameters()),
            "quantum": 0,
            "total": sum(p.numel() for p in model.parameters()),
        }

        metrics = compute_metrics(
            model_name=model_name,
            y_true=fold.y_test,
            y_pred=train_result.y_pred,
            y_prob=train_result.y_prob,
            threshold=train_result.threshold,
            param_count=param_count,
        )

    elapsed = time.perf_counter() - t0
    logger.info("Fold %d done: MCC=%.4f, PR-AUC=%.4f, time=%.1fs", args.fold, metrics.mcc, metrics.pr_auc, elapsed)

    # Save fold result
    out_path = Path(cfg.paths.folds_dir) / f"{model_name}_fold_{args.fold}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_dict = metrics.to_dict()
    result_dict["fold_idx"] = args.fold
    result_dict["elapsed_seconds"] = elapsed

    with open(out_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    logger.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
