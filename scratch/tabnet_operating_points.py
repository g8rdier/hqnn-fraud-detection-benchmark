#!/usr/bin/env python3
"""
scratch/tabnet_operating_points.py
====================================
Recover TabNet's per-fold operating points for the appendix table.

Reproducing the original run
-----------------------------
The original TabNet benchmark was a standalone invocation of run_benchmark.py
with TabNet as the only model.  Critically, _seed_everything(42) was called
before anything else.  This script mirrors that exactly:

  1. _seed_everything(42)           ← must come first, before any torch/numpy op
  2. create_folds(...)              ← deterministic; SMOTE uses cfg.smote.random_state
  3. for each fold: TabNetWrapper.fit(seed=42 in constructor)

No other model is loaded or run before TabNet training — avoiding the random-
state drift that caused folds 2-4 to diverge in the previous attempt.

Verification gate
-----------------
After training all 5 folds, per-fold test MCCs are printed and compared
against the stored benchmark values.  If any fold diverges by more than
0.002, the script aborts without writing any output.

File writes (repo-touching)
----------------------------
If verification passes, best-epoch weights are saved to
  results/models/tabnet_fold{n}.pt
in the same checkpoint format used by PyTorch models (key: best_state).
This is the only write outside scratch/.

Usage
-----
    pixi run python scratch/tabnet_operating_points.py
"""

from __future__ import annotations

import csv
import random
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.cv import create_folds
from src.data.loader import load_dataset
from src.training.trainer import find_optimal_threshold

CONFIG_PATH  = ROOT / "configs" / "default.yaml"
MODELS_DIR   = ROOT / "results" / "models"
OUTPUT_CSV   = ROOT / "appendix_classical_operating_points.csv"

# Stored per-fold MCCs from results/metrics/benchmark_20260415_031627.json
STORED_FOLD_MCCS = [
    0.33883203318188126,
    0.49944967364678144,
    0.5337903777781711,
    0.534655261675904,
    0.5050356159820975,
]
STORED_MEAN = 0.4823525924529671
STORED_STD  = 0.07319343637548649
MCC_TOL     = 0.002


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def agg(values):
    return statistics.mean(values), statistics.stdev(values)


def main() -> None:
    # ── Step 1: seed everything — must come before any numpy/torch op ──
    seed_everything(42)
    print("Seeded (42) — matching original standalone benchmark run.\n")

    cfg = load_config(CONFIG_PATH)

    print(f"Config:    {CONFIG_PATH}")
    print(f"Dataset:   {ROOT / cfg.data.raw_path}")
    print(f"Test size: {cfg.data.test_size}  random_state={cfg.data.random_state}")
    print(f"CV:        {cfg.cv.n_folds}-fold  random_state={cfg.cv.random_state}\n")

    # ── Step 2: create folds ───────────────────────────────────────────
    X, y = load_dataset(cfg.data)
    X_np = X.values if hasattr(X, "values") else np.asarray(X)
    y_np = y.values if hasattr(y, "values") else np.asarray(y)
    folds = create_folds(X_np, y_np, cfg)

    test_total = len(folds[0].y_test)
    test_fraud = int(folds[0].y_test.sum())
    assert test_total == 56962, f"Unexpected test size: {test_total}"
    assert test_fraud == 98,    f"Unexpected fraud count: {test_fraud}"
    print(f"Held-out test set: {test_total:,} rows, {test_fraud} frauds — ✓\n")

    # ── Step 3: train TabNet per fold, collect per-fold results ────────
    from src.models.classical.tabnet_model import TabNetWrapper

    print("Training TabNet (5 folds) …\n")
    per_fold = []

    for fold in folds:
        wrapper = TabNetWrapper(cfg=cfg.tabnet, training_cfg=cfg.training_tabnet)
        wrapper.fit(fold.X_train, fold.y_train, fold.X_val, fold.y_val)

        # Threshold: validation split only
        val_prob  = wrapper.model.predict_proba(fold.X_val)[:, 1]
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

        per_fold.append({
            "fold": fold.fold_idx,
            "threshold": threshold,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "recall": recall, "precision": precision, "mcc": mcc,
            "row_sum": tp + fp + fn + tn,
            "best_state": {k: v.cpu().clone()
                           for k, v in wrapper.model.network.state_dict().items()},
        })

    # ── Step 4: MCC verification gate ─────────────────────────────────
    print("=" * 60)
    print("MCC VERIFICATION (must reproduce within ±0.002 per fold)")
    print("=" * 60)
    print(f"  {'Fold':>4}  {'Stored MCC':>12}  {'Retrained MCC':>14}  {'Δ':>8}  Status")

    all_pass = True
    for r, stored in zip(per_fold, STORED_FOLD_MCCS):
        delta = abs(r["mcc"] - stored)
        ok    = delta <= MCC_TOL
        status = "✓" if ok else "✗ FAIL"
        print(f"  {r['fold']:>4}  {stored:>12.5f}  {r['mcc']:>14.5f}  {delta:>8.5f}  {status}")
        if not ok:
            all_pass = False

    m_mcc, s_mcc = agg([r["mcc"] for r in per_fold])
    print(f"\n  Retrained mean ± std : {m_mcc:.4f} ± {s_mcc:.4f}")
    print(f"  Stored   mean ± std  : {STORED_MEAN:.4f} ± {STORED_STD:.4f}")

    if not all_pass:
        print("\nABORTED — per-fold MCCs do not reproduce within tolerance.")
        print("No output written.  Do not use these numbers.")
        sys.exit(1)

    print("\nAll folds within tolerance — proceeding.\n")

    # ── Step 5: persist best-epoch weights ────────────────────────────
    print("Writing model weights (repo-touching: results/models/tabnet_fold{n}.pt)")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for r in per_fold:
        ckpt_path = MODELS_DIR / f"tabnet_fold{r['fold']}.pt"
        torch.save({"best_state": r["best_state"]}, ckpt_path)
        print(f"  Saved: {ckpt_path}")

    # ── Step 6: print operating-point table ───────────────────────────
    print(f"\n{'─' * 62}")
    print(f"  TabNet  —  per-fold operating points")
    print(f"{'─' * 62}")
    print(f"  {'Fold':>4}  {'Thresh':>6}  {'TP':>4}  {'FP':>5}  {'FN':>4}  {'TN':>6}"
          f"  {'Recall':>7}  {'Prec':>7}  {'MCC':>7}  {'RowSum':>7}")
    for r in per_fold:
        print(f"  {r['fold']:>4}  {r['threshold']:>6.2f}  {r['TP']:>4}  {r['FP']:>5}"
              f"  {r['FN']:>4}  {r['TN']:>6}"
              f"  {r['recall']:>7.4f}  {r['precision']:>7.4f}  {r['mcc']:>7.4f}"
              f"  {r['row_sum']:>7}")

    m_thr, s_thr = agg([r["threshold"]  for r in per_fold])
    m_tp,  s_tp  = agg([r["TP"]         for r in per_fold])
    m_fp,  s_fp  = agg([r["FP"]         for r in per_fold])
    m_fn,  s_fn  = agg([r["FN"]         for r in per_fold])
    m_rec, s_rec = agg([r["recall"]     for r in per_fold])
    m_pre, s_pre = agg([r["precision"]  for r in per_fold])

    print(f"\n  MEAN ± STD")
    print(f"    Threshold : {m_thr:.2f} ± {s_thr:.2f}")
    print(f"    TP        : {m_tp:.1f} ± {s_tp:.1f}")
    print(f"    FP        : {m_fp:.1f} ± {s_fp:.1f}")
    print(f"    FN        : {m_fn:.1f} ± {s_fn:.1f}")
    print(f"    Recall    : {m_rec:.4f} ± {s_rec:.4f}")
    print(f"    Precision : {m_pre:.4f} ± {s_pre:.4f}")
    print(f"    MCC       : {m_mcc:.4f} ± {s_mcc:.4f}")

    # ── Step 7: append TabNet row to CSV ──────────────────────────────
    tabnet_row = {
        "model": "TabNet",
        "threshold_mean": round(m_thr, 4), "threshold_std": round(s_thr, 4),
        "TP_mean": round(m_tp, 1),         "TP_std": round(s_tp, 1),
        "FP_mean": round(m_fp, 1),         "FP_std": round(s_fp, 1),
        "FN_mean": round(m_fn, 1),         "FN_std": round(s_fn, 1),
        "recall_mean": round(m_rec, 4),    "recall_std": round(s_rec, 4),
        "precision_mean": round(m_pre, 4), "precision_std": round(s_pre, 4),
        "mcc_mean": round(m_mcc, 4),       "mcc_std": round(s_mcc, 4),
    }
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

    # Read existing rows, append TabNet, rewrite
    existing = []
    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, newline="") as f:
            existing = list(csv.DictReader(f))

    # Remove any stale TabNet row from a previous aborted run
    existing = [row for row in existing if row["model"] != "TabNet"]
    existing.append(tabnet_row)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing)

    print(f"\nCSV updated → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
