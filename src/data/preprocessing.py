"""
preprocessing.py
================
Two-stage normalization + PCA pipeline.

Pipeline per fold:
  1. RobustScaler (fitted on train fold only)
  2. MinMaxScaler to [0, π] (fitted on train fold only)
  3. PCA to n_components (fitted on train fold only)

This module provides a composable FoldPreprocessor that is instantiated fresh
for each CV fold to prevent data leakage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from src.config import PreprocessingConfig

logger = logging.getLogger(__name__)


@dataclass
class FoldPreprocessor:
    """Stateful preprocessor fitted on one training fold."""

    config: PreprocessingConfig
    _robust: RobustScaler | None = None
    _minmax: MinMaxScaler | None = None
    _pca: PCA | None = None

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit all transformers on training data and return transformed array."""
        # Stage 1: RobustScaler — handles financial outliers
        if self.config.robust_scale:
            self._robust = RobustScaler()
            X_train = self._robust.fit_transform(X_train)

        # Stage 2: MinMax to [0, π] for angle encoding compatibility
        lo, hi = self.config.minmax_range
        self._minmax = MinMaxScaler(feature_range=(lo, hi))
        X_train = self._minmax.fit_transform(X_train)

        # Stage 3: PCA to qubit-compatible dimensionality
        n_comp = min(self.config.n_components, X_train.shape[1], X_train.shape[0])
        self._pca = PCA(n_components=n_comp, random_state=42)
        X_train = self._pca.fit_transform(X_train)

        explained = self._pca.explained_variance_ratio_.sum() * 100
        logger.info("PCA: %d components retain %.2f%% variance", n_comp, explained)
        return X_train

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform validation/test data using fitted transformers."""
        if self._robust is not None:
            X = self._robust.transform(X)
        if self._minmax is not None:
            X = self._minmax.transform(X)
        if self._pca is not None:
            X = self._pca.transform(X)
        return X

    @property
    def n_features_out(self) -> int:
        if self._pca is not None:
            return self._pca.n_components_
        return 0
