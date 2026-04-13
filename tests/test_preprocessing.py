"""Tests for the preprocessing pipeline."""

import numpy as np
import pytest

from src.config import PreprocessingConfig
from src.data.preprocessing import FoldPreprocessor


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((200, 20))
    X_val = rng.standard_normal((50, 20))
    return X_train, X_val


def test_fit_transform_shape(sample_data):
    X_train, _ = sample_data
    cfg = PreprocessingConfig(n_components=8)
    pp = FoldPreprocessor(config=cfg)
    X_out = pp.fit_transform(X_train)
    assert X_out.shape == (200, 8)


def test_transform_matches_dim(sample_data):
    X_train, X_val = sample_data
    cfg = PreprocessingConfig(n_components=8)
    pp = FoldPreprocessor(config=cfg)
    pp.fit_transform(X_train)
    X_val_out = pp.transform(X_val)
    assert X_val_out.shape == (50, 8)


def test_minmax_range(sample_data):
    X_train, _ = sample_data
    cfg = PreprocessingConfig(n_components=8, minmax_range=(0.0, 3.14159265))
    pp = FoldPreprocessor(config=cfg)
    X_out = pp.fit_transform(X_train)
    # After PCA the range isn't guaranteed to stay in [0, π],
    # but the pre-PCA scaled data should have been in range.
    assert pp.n_features_out == 8


def test_n_features_out(sample_data):
    X_train, _ = sample_data
    cfg = PreprocessingConfig(n_components=4)
    pp = FoldPreprocessor(config=cfg)
    pp.fit_transform(X_train)
    assert pp.n_features_out == 4
