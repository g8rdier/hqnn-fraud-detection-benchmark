"""Tests for model instantiation and forward pass."""

import numpy as np
import pytest
import torch

from src.config import BenchmarkConfig


@pytest.fixture
def cfg():
    return BenchmarkConfig(preprocessing={"n_components": 4})


@pytest.fixture
def sample_batch():
    return torch.randn(4, 4)  # batch=4, features=4


def test_snn_forward(cfg, sample_batch):
    from src.models.classical.snn import SNN
    model = SNN(input_dim=4, cfg=cfg.snn)
    out = model(sample_batch)
    assert out.shape == (4, 1)
    assert (out >= 0).all() and (out <= 1).all()


def test_snn_param_count(cfg):
    from src.models.classical.snn import SNN
    model = SNN(input_dim=4, cfg=cfg.snn)
    counts = model.param_count()
    assert counts["quantum"] == 0
    assert counts["total"] > 0


def test_shnn_forward(cfg, sample_batch):
    from src.models.quantum.shnn import SHNN
    model = SHNN(input_dim=4, cfg=cfg.shnn)
    out = model(sample_batch)
    assert out.shape == (4, 1)
    assert (out >= 0).all() and (out <= 1).all()


def test_shnn_param_count(cfg):
    from src.models.quantum.shnn import SHNN
    model = SHNN(input_dim=4, cfg=cfg.shnn)
    counts = model.param_count()
    assert counts["quantum"] > 0
    assert counts["classical"] > 0
    assert counts["total"] == counts["quantum"] + counts["classical"]


def test_parallel_forward(cfg, sample_batch):
    from src.models.quantum.parallel import ParallelHybrid
    model = ParallelHybrid(input_dim=4, cfg=cfg.parallel)
    out = model(sample_batch)
    assert out.shape == (4, 1)
    assert (out >= 0).all() and (out <= 1).all()


def test_parallel_param_count(cfg):
    from src.models.quantum.parallel import ParallelHybrid
    model = ParallelHybrid(input_dim=4, cfg=cfg.parallel)
    counts = model.param_count()
    assert counts["quantum"] > 0
    assert counts["total"] == counts["quantum"] + counts["classical"]


def test_registry_builds_all(cfg):
    from src.models.registry import build_model
    for name in ["shnn", "parallel", "snn", "tabnet"]:
        model = build_model(name, input_dim=4, cfg=cfg)
        assert model is not None
