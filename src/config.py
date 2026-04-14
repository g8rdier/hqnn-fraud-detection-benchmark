"""
config.py
=========
Pydantic-validated configuration for the HQNN benchmark.

All hyperparameters live here. Override via YAML config files or CLI flags.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


# ── Sub-configs ──────────────────────────────────────────────────────────────


class DataConfig(BaseModel):
    raw_path: Path = Path("data/raw/creditcard.csv")
    target_column: str = "Class"
    drop_columns: list[str] = ["Time"]
    test_size: float = 0.20
    random_state: int = 42


class PreprocessingConfig(BaseModel):
    n_components: int = Field(8, description="PCA components = qubit count")
    robust_scale: bool = True
    minmax_range: tuple[float, float] = (0.0, 3.14159265)  # [0, π] for angle encoding


class SmoteConfig(BaseModel):
    enabled: bool = True
    k_neighbors: int = 5
    random_state: int = 42


class CVConfig(BaseModel):
    n_folds: int = 5
    shuffle: bool = True
    random_state: int = 42


class VQCConfig(BaseModel):
    n_qubits: int = 8
    n_layers: int = 2
    backend: str = "lightning.qubit"
    diff_method: str = "adjoint"


class NoiseConfig(BaseModel):
    enabled: bool = False
    backend: str = "default.mixed"
    depolarizing_p: float = 0.0


class SHNNConfig(BaseModel):
    """Sequential Hybrid Neural Network: FC → VQC → FC."""
    pre_fc_dims: list[int] = [16]
    post_fc_dims: list[int] = [8]
    dropout: float = 0.1
    vqc: VQCConfig = VQCConfig()


class ParallelHybridConfig(BaseModel):
    """Parallel Hybrid: [MLP(x) ∥ VQC(x)] → concat → FC."""
    mlp_dims: list[int] = [16, 8]
    post_fc_dims: list[int] = [8]
    dropout: float = 0.1
    vqc: VQCConfig = VQCConfig()


class SNNConfig(BaseModel):
    """Self-Normalizing Network with SELU + AlphaDropout."""
    hidden_dims: list[int] = [64, 32, 16]
    alpha_dropout: float = 0.05


class TabNetConfig(BaseModel):
    n_d: int = 8
    n_a: int = 8
    n_steps: int = 3
    gamma: float = 1.3
    lambda_sparse: float = 1e-3


class FTTransformerConfig(BaseModel):
    """Feature Tokenizer + Transformer (Gorishniy et al., 2021)."""
    d_token: int = 32
    n_blocks: int = 2
    n_heads: int = 4
    ffn_d_hidden_factor: float = 1.333
    attn_dropout: float = 0.0
    ffn_dropout: float = 0.0
    residual_dropout: float = 0.0


class ResNetConfig(BaseModel):
    """ResNet for tabular data (Gorishniy et al., 2021)."""
    d: int = 32
    d_hidden: int = 64
    n_blocks: int = 2
    dropout: float = 0.0


class TrainingConfig(BaseModel):
    epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 9999
    optimizer: Literal["adam", "sgd"] = "adam"
    device: str = "auto"  # "auto" → MPS if available, else CPU


class TrainingConfigTabNet(BaseModel):
    """TabNet uses its own internal training loop."""
    max_epochs: int = 100
    patience: int = 9999
    batch_size: int = 256


class EvaluationConfig(BaseModel):
    primary_metrics: list[str] = ["mcc", "pr_auc"]
    secondary_metrics: list[str] = ["f1_fraud", "roc_auc"]
    significance_test: Literal["wilcoxon"] = "wilcoxon"
    effect_size: Literal["rank_biserial"] = "rank_biserial"


class PathsConfig(BaseModel):
    results_dir: Path = Path("results")
    figures_dir: Path = Path("results/figures")
    metrics_dir: Path = Path("results/metrics")
    models_dir: Path = Path("results/models")
    folds_dir: Path = Path("results/folds")


# ── Root config ──────────────────────────────────────────────────────────────


class BenchmarkConfig(BaseModel):
    """Root configuration for the entire benchmark."""

    data: DataConfig = DataConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    smote: SmoteConfig = SmoteConfig()
    cv: CVConfig = CVConfig()
    noise: NoiseConfig = NoiseConfig()

    shnn: SHNNConfig = SHNNConfig()
    parallel: ParallelHybridConfig = ParallelHybridConfig()
    snn: SNNConfig = SNNConfig()
    tabnet: TabNetConfig = TabNetConfig()
    ftt: FTTransformerConfig = FTTransformerConfig()
    resnet: ResNetConfig = ResNetConfig()

    training: TrainingConfig = TrainingConfig()
    training_tabnet: TrainingConfigTabNet = TrainingConfigTabNet()
    evaluation: EvaluationConfig = EvaluationConfig()
    paths: PathsConfig = PathsConfig()

    seed: int = 42
    models_to_run: list[str] = ["shnn", "parallel", "tabnet", "snn"]

    @model_validator(mode="after")
    def sync_qubits_to_pca(self) -> "BenchmarkConfig":
        """Ensure PCA components match qubit count across all quantum sub-configs."""
        n_q = self.preprocessing.n_components
        self.shnn.vqc.n_qubits = n_q
        self.parallel.vqc.n_qubits = n_q
        return self


# ── I/O ──────────────────────────────────────────────────────────────────────


def load_config(path: str | Path | None = None) -> BenchmarkConfig:
    """Load config from YAML, falling back to defaults."""
    if path is None:
        return BenchmarkConfig()
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return BenchmarkConfig(**raw)
