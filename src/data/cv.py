"""
cv.py
=====
Cross-validation splitter with SMOTE strictly inside each fold.

This is the critical design choice from the exposé: synthetic samples never
leak into validation sets, preserving benchmark integrity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import BenchmarkConfig
from src.data.preprocessing import FoldPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class FoldData:
    """All arrays for a single CV fold, ready for model consumption."""

    fold_idx: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    preprocessor: FoldPreprocessor


def create_folds(
    X: np.ndarray,
    y: np.ndarray,
    cfg: BenchmarkConfig,
) -> list[FoldData]:
    """
    Create all CV folds with preprocessing and SMOTE applied per-fold.

    Pipeline per fold:
      1. StratifiedKFold split → train_fold / val_fold
      2. FoldPreprocessor.fit_transform(train_fold), .transform(val_fold, test)
      3. SMOTE on preprocessed train_fold only
    """
    # Hold out a fixed test set first (never touched during CV)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y,
        test_size=cfg.data.test_size,
        stratify=y,
        random_state=cfg.data.random_state,
    )
    logger.info("Held-out test set: %d samples", len(y_test))

    skf = StratifiedKFold(
        n_splits=cfg.cv.n_folds,
        shuffle=cfg.cv.shuffle,
        random_state=cfg.cv.random_state,
    )

    folds: list[FoldData] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_dev, y_dev)):
        logger.info("── Fold %d/%d ──", fold_idx + 1, cfg.cv.n_folds)

        X_train_fold = X_dev[train_idx]
        y_train_fold = y_dev[train_idx]
        X_val_fold = X_dev[val_idx]
        y_val_fold = y_dev[val_idx]

        # Preprocessing: fit on train fold, transform val + test
        preprocessor = FoldPreprocessor(config=cfg.preprocessing)
        X_train_fold = preprocessor.fit_transform(X_train_fold)
        X_val_fold = preprocessor.transform(X_val_fold)
        X_test_fold = preprocessor.transform(X_test)

        # SMOTE: strictly inside the fold, on preprocessed train data only
        if cfg.smote.enabled:
            n_before = len(y_train_fold)
            smote = SMOTE(
                k_neighbors=cfg.smote.k_neighbors,
                random_state=cfg.smote.random_state,
            )
            X_train_fold, y_train_fold = smote.fit_resample(X_train_fold, y_train_fold)
            logger.info(
                "SMOTE: %d → %d samples (fold %d)", n_before, len(y_train_fold), fold_idx
            )

        folds.append(
            FoldData(
                fold_idx=fold_idx,
                X_train=X_train_fold,
                y_train=y_train_fold,
                X_val=X_val_fold,
                y_val=y_val_fold,
                X_test=X_test_fold,
                y_test=y_test,
                preprocessor=preprocessor,
            )
        )

    return folds
