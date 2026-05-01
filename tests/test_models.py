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


def test_ftt_forward(cfg, sample_batch):
    from src.models.classical.ft_transformer import FTTransformer
    model = FTTransformer(input_dim=4, cfg=cfg.ftt)
    out = model(sample_batch)
    assert out.shape == (4, 1)


def test_ftt_param_count(cfg):
    from src.models.classical.ft_transformer import FTTransformer
    model = FTTransformer(input_dim=4, cfg=cfg.ftt)
    counts = model.param_count()
    assert counts["quantum"] == 0
    assert counts["total"] > 0


def test_resnet_forward(cfg, sample_batch):
    from src.models.classical.resnet_tabular import ResNet
    model = ResNet(input_dim=4, cfg=cfg.resnet)
    out = model(sample_batch)
    assert out.shape == (4, 1)


def test_resnet_param_count(cfg):
    from src.models.classical.resnet_tabular import ResNet
    model = ResNet(input_dim=4, cfg=cfg.resnet)
    counts = model.param_count()
    assert counts["quantum"] == 0
    assert counts["total"] > 0


def test_saint_forward(cfg, sample_batch):
    from src.models.classical.saint import SAINT
    model = SAINT(input_dim=4, cfg=cfg.saint)
    out = model(sample_batch)
    assert out.shape == (4, 1)


def test_saint_param_count(cfg):
    from src.models.classical.saint import SAINT
    model = SAINT(input_dim=4, cfg=cfg.saint)
    counts = model.param_count()
    assert counts["quantum"] == 0
    assert counts["total"] > 0


def test_vqc_module_forward(cfg, sample_batch):
    from src.models.quantum.vqc import VQCModule
    model = VQCModule(cfg=cfg.shnn.vqc)
    out = model(sample_batch)
    assert out.shape == (4, 1)


def test_vqc_noise_forward(cfg, sample_batch):
    from src.models.quantum.vqc import build_vqc_layer
    from src.config import NoiseConfig
    noise_cfg = NoiseConfig(enabled=True, backend="default.mixed", depolarizing_p=0.01)
    layer = build_vqc_layer(cfg.shnn.vqc, noise_cfg)
    out = layer(sample_batch)
    assert out.shape == (4,)


def test_registry_builds_all(cfg):
    from src.models.registry import build_model
    for name in ["shnn", "parallel", "snn", "tabnet", "ftt", "resnet", "saint"]:
        model = build_model(name, input_dim=4, cfg=cfg)
        assert model is not None


def test_tabnet_predict_not_fitted(cfg):
    import numpy as np
    from src.models.classical.tabnet_model import TabNetWrapper
    wrapper = TabNetWrapper(cfg=cfg.tabnet, training_cfg=cfg.training_tabnet)
    with pytest.raises(RuntimeError, match="not fitted"):
        wrapper.predict(np.zeros((4, 4)))


def test_tabnet_predict_mocked(cfg):
    from unittest.mock import MagicMock
    import numpy as np
    from src.models.classical.tabnet_model import TabNetWrapper
    wrapper = TabNetWrapper(cfg=cfg.tabnet, training_cfg=cfg.training_tabnet)
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array(
        [[0.9, 0.1], [0.3, 0.7], [0.8, 0.2], [0.4, 0.6]]
    )
    mock_model.network.parameters.return_value = []
    wrapper.model = mock_model
    result = wrapper.predict(np.zeros((4, 4)))
    assert result.param_count["quantum"] == 0
