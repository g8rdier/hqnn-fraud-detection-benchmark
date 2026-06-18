#!/usr/bin/env python3
"""
scratch/classical_operating_points.py
======================================
Reconstruct per-fold operating points for the five classical baselines.

PyTorch models (SNN, Tabular ResNet, FT-Transformer, SAINT)
------------------------------------------------------------
  Loaded from results/models/{model}_fold{n}.pt  (best_state key).
  No retraining — pure inference on reconstructed data splits.

TabNet
------
  No weights were persisted during the original run.  Retrained here using
  the identical pipeline: same seeds, same config, same TabNetWrapper.fit
  call.  TabNet's constructor uses seed=42 so results are deterministic.

Threshold for all models
------------------------
  find_optimal_threshold on the validation split only (sweep 0.05–0.95
  step 0.01, max MCC) — test set is never seen during selection.

Data
----
  configs/default.yaml  — CV params (seed 42, test_size 0.20)
  data/raw/creditcard.csv — original dataset

Usage
-----
    pixi run python scratch/classical_operating_points.py
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "results" / "models"
CONFIG_PATH = ROOT / "configs" / "default.yaml"
OUTPUT_CSV = ROOT / "appendix_classical_operating_points.csv"

# Internal imports (no source files modified)
sys.path.insert(0, str(ROOT))
from src.config import load_config
from src.data.cv import create_folds
from src.data.loader import load_dataset
from src.models.registry import build_model
from src.training.trainer import find_optimal_threshold

# ── Thesis display names ────────────────────────────────────────────────────
THESIS_NAMES = {
    "snn":    "SNN",
    "resnet": "Tabular ResNet",
    "ftt":    "FT-Transformer",
    "saint":  "SAINT",
}
CLASSICAL_PYTORCH = list(THESIS_NAMES.keys())   # order: snn, resnet, ftt, saint

TABNET_THESIS_NAME = "TabNet"


def load_best_model(model_name: str, fold_idx: int, input_dim: int, cfg):
    """Build model, load best_state from checkpoint, set eval mode."""
    ckpt_path = MODELS_DIR / f"{model_name}_fold{fold_idx}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(model_name, input_dim, cfg)
    model.load_state_dict(ckpt["best_state"])
    model.eval()
    return model, ckpt["best_epoch"]


def infer(model, X: np.ndarray) -> np.ndarray:
    """Return sigmoid probabilities for X."""
    with torch.no_grad():
        t = torch.tensor(X, dtype=torch.float32)
        return model(t).cpu().numpy().flatten()


def fold_metrics(model_name: str, folds, cfg):
    """
    Return list of per-fold dicts:
      threshold, TP, FP, FN, TN, recall, precision, mcc
    """
    results = []
    for fold in folds:
        model, best_epoch = load_best_model(
            model_name, fold.fold_idx, fold.X_val.shape[1], cfg
        )

        # ── Threshold: validation split only ──────────────────────────
        val_prob = infer(model, fold.X_val)
        threshold = find_optimal_threshold(fold.y_val, val_prob)

        # ── Test set predictions ───────────────────────────────────────
        test_prob = infer(model, fold.X_test)
        test_pred = (test_prob >= threshold).astype(int)

        y = fold.y_test
        tp = int(((test_pred == 1) & (y == 1)).sum())
        fp = int(((test_pred == 1) & (y == 0)).sum())
        fn = int(((test_pred == 0) & (y == 1)).sum())
        tn = int(((test_pred == 0) & (y == 0)).sum())
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        mcc       = matthews_corrcoef(y, test_pred)

        results.append({
            "fold": fold.fold_idx,
            "best_epoch": best_epoch,
            "threshold": threshold,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "recall": recall, "precision": precision, "mcc": mcc,
            "row_sum": tp + fp + fn + tn,
        })
    return results


def tabnet_fold_metrics(folds, cfg):
    """Retrain TabNet per fold (deterministic, seed=42) and compute metrics."""
    from src.models.classical.tabnet_model import TabNetWrapper

    results = []
    for fold in folds:
        wrapper = TabNetWrapper(cfg=cfg.tabnet, training_cfg=cfg.training_tabnet)
        wrapper.fit(fold.X_train, fold.y_train, fold.X_val, fold.y_val)

        # Threshold: validation split only (identical to original run)
        val_prob = wrapper.model.predict_proba(fold.X_val)[:, 1]
        threshold = find_optimal_threshold(fold.y_val, val_prob)

        test_prob = wrapper.model.predict_proba(fold.X_test)[:, 1]
        test_pred = (test_prob >= threshold).astype(int)

        y = fold.y_test
        tp = int(((test_pred == 1) & (y == 1)).sum())
        fp = int(((test_pred == 1) & (y == 0)).sum())
        fn = int(((test_pred == 0) & (y == 1)).sum())
        tn = int(((test_pred == 0) & (y == 0)).sum())
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        mcc       = matthews_corrcoef(y, test_pred)

        results.append({
            "fold": fold.fold_idx,
            "best_epoch": None,
            "threshold": threshold,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "recall": recall, "precision": precision, "mcc": mcc,
            "row_sum": tp + fp + fn + tn,
        })
    return results


def agg(values):
    return statistics.mean(values), statistics.stdev(values)


def main():
    cfg = load_config(CONFIG_PATH)

    # ── Announce data sources ───────────────────────────────────────────
    print("=" * 70)
    print("DATA SOURCES")
    print(f"  Config:       {CONFIG_PATH}")
    print(f"  Dataset:      {ROOT / cfg.data.raw_path}")
    print(f"  Model weights:{MODELS_DIR}/{{model}}_fold{{n}}.pt  (best_state key)")
    print(f"  Test size:    {cfg.data.test_size}  random_state={cfg.data.random_state}")
    print(f"  CV:           {cfg.cv.n_folds}-fold, shuffle={cfg.cv.shuffle}, "
          f"random_state={cfg.cv.random_state}")
    print()

    # ── Build folds (deterministic — same seeds as original runs) ──────
    print("Reconstructing CV splits (no retraining) …")
    X, y = load_dataset(cfg.data)
    X_np = X.values if hasattr(X, "values") else np.asarray(X)
    y_np = y.values if hasattr(y, "values") else np.asarray(y)
    folds = create_folds(X_np, y_np, cfg)

    # Sanity-check test set
    test_total = len(folds[0].y_test)
    test_fraud = int(folds[0].y_test.sum())
    print(f"\nHeld-out test set: {test_total:,} rows, {test_fraud} frauds")
    assert test_total == 56962, f"Unexpected test size: {test_total}"
    assert test_fraud == 98,    f"Unexpected fraud count: {test_fraud}"
    print("  ✓ matches §4.1 (56,962 rows, 98 frauds)\n")

    # ── Per-model computation ───────────────────────────────────────────
    summary_rows = []   # for CSV

    print("=" * 70)
    print("PER-FOLD RESULTS")
    print("=" * 70)

    for model_name in CLASSICAL_PYTORCH:
        thesis_name = THESIS_NAMES[model_name]
        print(f"\n{'─' * 60}")
        print(f"  {thesis_name}  ({model_name})")
        print(f"{'─' * 60}")
        print(f"  {'Fold':>4}  {'Thresh':>6}  {'TP':>4}  {'FP':>5}  {'FN':>4}  {'TN':>6}"
              f"  {'Recall':>7}  {'Prec':>7}  {'MCC':>7}  {'RowSum':>7}")

        per_fold = fold_metrics(model_name, folds, cfg)

        for r in per_fold:
            print(f"  {r['fold']:>4}  {r['threshold']:>6.2f}  {r['TP']:>4}  {r['FP']:>5}"
                  f"  {r['FN']:>4}  {r['TN']:>6}"
                  f"  {r['recall']:>7.4f}  {r['precision']:>7.4f}  {r['mcc']:>7.4f}"
                  f"  {r['row_sum']:>7}")

        # Aggregate
        m_thr,  s_thr  = agg([r["threshold"]  for r in per_fold])
        m_tp,   s_tp   = agg([r["TP"]         for r in per_fold])
        m_fp,   s_fp   = agg([r["FP"]         for r in per_fold])
        m_fn,   s_fn   = agg([r["FN"]         for r in per_fold])
        m_rec,  s_rec  = agg([r["recall"]     for r in per_fold])
        m_pre,  s_pre  = agg([r["precision"]  for r in per_fold])
        m_mcc,  s_mcc  = agg([r["mcc"]        for r in per_fold])

        print(f"\n  MEAN ± STD")
        print(f"    Threshold : {m_thr:.2f} ± {s_thr:.2f}")
        print(f"    TP        : {m_tp:.1f} ± {s_tp:.1f}")
        print(f"    FP        : {m_fp:.1f} ± {s_fp:.1f}")
        print(f"    FN        : {m_fn:.1f} ± {s_fn:.1f}")
        print(f"    Recall    : {m_rec:.4f} ± {s_rec:.4f}")
        print(f"    Precision : {m_pre:.4f} ± {s_pre:.4f}")
        print(f"    MCC       : {m_mcc:.4f} ± {s_mcc:.4f}")

        summary_rows.append({
            "model": thesis_name,
            "threshold_mean": round(m_thr, 4), "threshold_std": round(s_thr, 4),
            "TP_mean": round(m_tp, 1),         "TP_std": round(s_tp, 1),
            "FP_mean": round(m_fp, 1),         "FP_std": round(s_fp, 1),
            "FN_mean": round(m_fn, 1),         "FN_std": round(s_fn, 1),
            "recall_mean": round(m_rec, 4),    "recall_std": round(s_rec, 4),
            "precision_mean": round(m_pre, 4), "precision_std": round(s_pre, 4),
            "mcc_mean": round(m_mcc, 4),       "mcc_std": round(s_mcc, 4),
        })

    # ── TabNet (retrained — deterministic seed=42) ──────────────────────
    print(f"\n{'─' * 60}")
    print(f"  {TABNET_THESIS_NAME}  (tabnet) — retraining (no weights persisted)")
    print(f"{'─' * 60}")
    print(f"  {'Fold':>4}  {'Thresh':>6}  {'TP':>4}  {'FP':>5}  {'FN':>4}  {'TN':>6}"
          f"  {'Recall':>7}  {'Prec':>7}  {'MCC':>7}  {'RowSum':>7}")

    tabnet_per_fold = tabnet_fold_metrics(folds, cfg)

    for r in tabnet_per_fold:
        print(f"  {r['fold']:>4}  {r['threshold']:>6.2f}  {r['TP']:>4}  {r['FP']:>5}"
              f"  {r['FN']:>4}  {r['TN']:>6}"
              f"  {r['recall']:>7.4f}  {r['precision']:>7.4f}  {r['mcc']:>7.4f}"
              f"  {r['row_sum']:>7}")

    m_thr,  s_thr  = agg([r["threshold"]  for r in tabnet_per_fold])
    m_tp,   s_tp   = agg([r["TP"]         for r in tabnet_per_fold])
    m_fp,   s_fp   = agg([r["FP"]         for r in tabnet_per_fold])
    m_fn,   s_fn   = agg([r["FN"]         for r in tabnet_per_fold])
    m_rec,  s_rec  = agg([r["recall"]     for r in tabnet_per_fold])
    m_pre,  s_pre  = agg([r["precision"]  for r in tabnet_per_fold])
    m_mcc,  s_mcc  = agg([r["mcc"]        for r in tabnet_per_fold])

    print(f"\n  MEAN ± STD")
    print(f"    Threshold : {m_thr:.2f} ± {s_thr:.2f}")
    print(f"    TP        : {m_tp:.1f} ± {s_tp:.1f}")
    print(f"    FP        : {m_fp:.1f} ± {s_fp:.1f}")
    print(f"    FN        : {m_fn:.1f} ± {s_fn:.1f}")
    print(f"    Recall    : {m_rec:.4f} ± {s_rec:.4f}")
    print(f"    Precision : {m_pre:.4f} ± {s_pre:.4f}")
    print(f"    MCC       : {m_mcc:.4f} ± {s_mcc:.4f}")

    summary_rows.append({
        "model": TABNET_THESIS_NAME,
        "threshold_mean": round(m_thr, 4), "threshold_std": round(s_thr, 4),
        "TP_mean": round(m_tp, 1),         "TP_std": round(s_tp, 1),
        "FP_mean": round(m_fp, 1),         "FP_std": round(s_fp, 1),
        "FN_mean": round(m_fn, 1),         "FN_std": round(s_fn, 1),
        "recall_mean": round(m_rec, 4),    "recall_std": round(s_rec, 4),
        "precision_mean": round(m_pre, 4), "precision_std": round(s_pre, 4),
        "mcc_mean": round(m_mcc, 4),       "mcc_std": round(s_mcc, 4),
    })

    # ── Summary table ───────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SUMMARY TABLE  (mean ± std, 5 folds, held-out test set)")
    print(f"{'=' * 70}")
    hdr = f"  {'Model':<18}  {'Thresh':>12}  {'Recall':>12}  {'Prec':>12}  {'MCC':>12}  {'TP':>10}  {'FP':>10}  {'FN':>10}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for r in summary_rows:
        print(
            f"  {r['model']:<18}"
            f"  {r['threshold_mean']:.2f} ± {r['threshold_std']:.2f}"
            f"  {r['recall_mean']:.4f} ± {r['recall_std']:.4f}"
            f"  {r['precision_mean']:.4f} ± {r['precision_std']:.4f}"
            f"  {r['mcc_mean']:.4f} ± {r['mcc_std']:.4f}"
            f"  {r['TP_mean']:.1f} ± {r['TP_std']:.1f}"
            f"  {r['FP_mean']:.1f} ± {r['FP_std']:.1f}"
            f"  {r['FN_mean']:.1f} ± {r['FN_std']:.1f}"
        )

    # ── CSV ─────────────────────────────────────────────────────────────
    fields = [
        "model",
        "threshold_mean", "threshold_std",
        "TP_mean", "TP_std",
        "FP_mean", "FP_std",
        "FN_mean", "FN_std",
        "recall_mean", "recall_std",
        "precision_mean", "precision_std",
        "mcc_mean", "mcc_std",
    ]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nCSV written → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
