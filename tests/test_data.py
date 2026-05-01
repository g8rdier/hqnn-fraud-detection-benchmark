"""Tests for src/data/loader.py and src/data/cv.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import BenchmarkConfig, DataConfig


# ── loader ────────────────────────────────────────────────────────────────────

def test_load_dataset_file_not_found(tmp_path):
    from src.data.loader import load_dataset
    cfg = DataConfig(raw_path=str(tmp_path / "missing.csv"), target_column="Class")
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_dataset(cfg)


def test_load_dataset_missing_target(tmp_path):
    from src.data.loader import load_dataset
    csv = tmp_path / "data.csv"
    pd.DataFrame({"A": [1, 2], "B": [3, 4]}).to_csv(csv, index=False)
    cfg = DataConfig(raw_path=str(csv), target_column="Class")
    with pytest.raises(ValueError, match="Target column"):
        load_dataset(cfg)


def test_load_dataset_basic(tmp_path):
    from src.data.loader import load_dataset
    csv = tmp_path / "data.csv"
    pd.DataFrame({"V1": [1.0, 2.0, 3.0], "Amount": [10.0, 20.0, 30.0],
                  "Time": [0, 1, 2], "Class": [0, 1, 0]}).to_csv(csv, index=False)
    cfg = DataConfig(raw_path=str(csv), target_column="Class", drop_columns=["Time"])
    X, y = load_dataset(cfg)
    assert "Time" not in X.columns
    assert list(y) == [0, 1, 0]


# ── cv ────────────────────────────────────────────────────────────────────────

def test_create_folds_shape():
    from src.data.cv import create_folds
    rng = np.random.default_rng(42)
    n_legit, n_fraud = 300, 60
    X = rng.standard_normal((n_legit + n_fraud, 8))
    y = np.array([0] * n_legit + [1] * n_fraud)
    cfg = BenchmarkConfig(preprocessing={"n_components": 4})
    folds = create_folds(X, y, cfg)
    assert len(folds) == cfg.cv.n_folds
    assert all(f.X_train.shape[1] == 4 for f in folds)
    assert all(f.X_val.shape[1] == 4 for f in folds)
