"""
registry.py
===========
Model registry for config-driven instantiation.

Maps model names (from config.models_to_run) to their constructors,
so the benchmark loop can build any model from a string key.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from src.config import BenchmarkConfig

if TYPE_CHECKING:
    from src.models.classical.tabnet_model import TabNetWrapper


def build_model(name: str, input_dim: int, cfg: BenchmarkConfig) -> nn.Module | "TabNetWrapper":
    """
    Instantiate a model by name from the config.

    Parameters
    ----------
    name : str
        One of "shnn", "parallel", "snn", "tabnet".
    input_dim : int
        Number of input features after PCA.
    cfg : BenchmarkConfig
        Full benchmark configuration.

    Returns
    -------
    nn.Module or TabNetWrapper
    """
    noise_cfg = cfg.noise if cfg.noise.enabled else None

    if name == "shnn":
        from src.models.quantum.shnn import SHNN
        return SHNN(input_dim=input_dim, cfg=cfg.shnn, noise_cfg=noise_cfg)

    if name == "parallel":
        from src.models.quantum.parallel import ParallelHybrid
        return ParallelHybrid(input_dim=input_dim, cfg=cfg.parallel, noise_cfg=noise_cfg)

    if name == "snn":
        from src.models.classical.snn import SNN
        return SNN(input_dim=input_dim, cfg=cfg.snn)

    if name == "tabnet":
        from src.models.classical.tabnet_model import TabNetWrapper
        return TabNetWrapper(cfg=cfg.tabnet, training_cfg=cfg.training_tabnet)

    if name == "ftt":
        from src.models.classical.ft_transformer import FTTransformer
        return FTTransformer(input_dim=input_dim, cfg=cfg.ftt)

    if name == "saint":
        from src.models.classical.saint import SAINT
        return SAINT(input_dim=input_dim, cfg=cfg.saint)

    raise ValueError(f"Unknown model: '{name}'. Choose from: shnn, parallel, snn, tabnet, ftt, saint")
