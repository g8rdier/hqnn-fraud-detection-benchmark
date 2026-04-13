# HQNN Fraud Detection Benchmark

Empirical benchmark of Hybrid Quantum Neural Networks (SHNN, Parallel Hybrid) vs classical
deep learning (TabNet, SNN+SELU) on the Kaggle Credit Card Fraud Detection dataset.

## Stack

- **Pixi** for dependency management (`pixi install` / `pixi run <task>`)
- **PyTorch** for all models (classical + hybrid)
- **PennyLane** for quantum circuit simulation (TorchLayer integration)
- **Pydantic** for config validation, YAML for config files

## Key design decisions

- SMOTE is applied **strictly inside CV folds**, never globally
- Two-stage normalization: RobustScaler → MinMax to [0, π]
- PCA fitted on training fold only to prevent leakage
- Primary metrics: MCC and PR-AUC (robust to extreme class imbalance)
- Statistical validation: Wilcoxon signed-rank + rank-biserial correlation
- Parallel fold execution for feasibility (~37h/fold)

## Running

```bash
pixi install
pixi run benchmark          # full benchmark
pixi run fold -- --fold 0   # single fold (for parallel dispatch)
pixi run test               # tests
```

## Structure

- `configs/` — YAML config files
- `src/` — all library code (data, models, training, evaluation)
- `scripts/` — entry-point CLI scripts
- `tests/` — pytest suite
- `results/` — auto-generated outputs
