"""Tests for the config system."""

from src.config import BenchmarkConfig, load_config


def test_default_config_valid():
    cfg = BenchmarkConfig()
    assert cfg.preprocessing.n_components == 8
    assert cfg.cv.n_folds == 5
    assert cfg.shnn.vqc.n_qubits == 8
    assert cfg.parallel.vqc.n_qubits == 8


def test_qubits_sync():
    """PCA components should sync to qubit count."""
    cfg = BenchmarkConfig(preprocessing={"n_components": 4})
    assert cfg.shnn.vqc.n_qubits == 4
    assert cfg.parallel.vqc.n_qubits == 4


def test_load_config_defaults():
    cfg = load_config(None)
    assert cfg.seed == 42
    assert "shnn" in cfg.models_to_run


def test_load_config_from_yaml(tmp_path):
    yaml_path = tmp_path / "test.yaml"
    yaml_path.write_text("seed: 123\ncv:\n  n_folds: 3\n")
    cfg = load_config(yaml_path)
    assert cfg.seed == 123
    assert cfg.cv.n_folds == 3
