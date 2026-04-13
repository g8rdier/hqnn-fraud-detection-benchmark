"""
loader.py
=========
Dataset loading and initial validation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import DataConfig

logger = logging.getLogger(__name__)


def load_dataset(cfg: DataConfig) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the raw CSV and return (features, target).

    Drops columns specified in config, validates the target column exists,
    and logs basic dataset statistics.
    """
    path = Path(cfg.raw_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Download from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place creditcard.csv in data/raw/"
        )

    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path)

    if cfg.target_column not in df.columns:
        raise ValueError(f"Target column '{cfg.target_column}' not found in dataset.")

    fraud_rate = df[cfg.target_column].mean() * 100
    logger.info("Shape: %s | Fraud rate: %.4f%%", df.shape, fraud_rate)

    cols_to_drop = [c for c in cfg.drop_columns if c in df.columns]
    if cols_to_drop:
        logger.info("Dropping columns: %s", cols_to_drop)
        df = df.drop(columns=cols_to_drop)

    X = df.drop(columns=[cfg.target_column])
    y = df[cfg.target_column]
    return X, y
