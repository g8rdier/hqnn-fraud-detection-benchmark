# HQNN Fraud Detection Benchmark

## Academic Context

| | |
|---|---|
| **Degree** | Bachelor of Science in Artificial Intelligence |
| **Institution** | IU International University of Applied Sciences, Munich |
| **Supervisor** | Prof. Dr. rer. nat. Michael Barth |
| **Student** | Gregor Kobilarov |
| **Dataset** | [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (n = 284,807) |

---

## Research Question

> "To what extent do Hybrid Quantum Neural Networks (HQNNs) demonstrate Parameter Efficiency Advantage — defined as achieving comparable or superior MCC and PR-AUC scores with significantly fewer trainable parameters — over state-of-the-art classical models for imbalanced financial data?"

---

## Results

5-fold stratified CV on the Kaggle Credit Card Fraud dataset (n = 284,807). Metrics are mean ± std across folds.

| Model | Type | Params | MCC | PR-AUC | MCC / kParam | PR-AUC / kParam |
|---|---|---|---|---|---|---|
| **SHNN** | Quantum hybrid | 122 | 0.5758 ± 0.0371 | 0.5910 ± 0.0323 | 4.720 | 4.844 |
| **Parallel Hybrid** | Quantum hybrid | 489 | 0.5688 ± 0.0371 | 0.6239 ± 0.0101 | 1.163 | 1.276 |
| SNN | Classical | 3,201 | 0.5633 ± 0.0139 | 0.6449 ± 0.0086 | 0.176 | 0.201 |
| TabNet | Classical | 6,176 | 0.4824 ± 0.0732 | 0.6551 ± 0.0399 | 0.078 | 0.106 |
| ResNet | Classical | 8,897 | 0.6933 ± 0.0329 | 0.7170 ± 0.0164 | 0.078 | 0.081 |
| FT-Transformer | Classical | 14,869 | 0.6934 ± 0.0164 | 0.7061 ± 0.0220 | 0.047 | 0.047 |
| SAINT | Classical | 29,357 | 0.6975 ± 0.0164 | 0.6570 ± 0.0505 | 0.024 | 0.022 |

**Key finding:** SHNN achieves comparable MCC to SNN (0.576 vs 0.563) with 26× fewer parameters, yielding a ~27× MCC/kParam advantage and ~24× PR-AUC/kParam advantage. Larger classical models (ResNet, FT-Transformer, SAINT) achieve higher absolute MCC but at 73–240× the parameter count, resulting in 60–197× lower efficiency. The central thesis claim — quantum advantage through parameter efficiency rather than raw performance — is supported.

---

## Evaluation Design

The benchmark measures two dimensions:

**1. Absolute performance** — MCC and PR-AUC across 5 stratified CV folds, reported as mean ± std. Statistical consistency is assessed via the Wilcoxon signed-rank test. Given n=5 folds, the minimum achievable p-value (0.0625) exceeds the standard significance threshold; rank-biserial correlation therefore serves as the primary effect size measure.

**2. Parameter efficiency** — MCC/kParam and PR-AUC/kParam (predictive performance per 1,000 trainable parameters) for each architecture. This operationalises the theoretical quantum expressivity advantage: a small quantum model should achieve disproportionately high performance relative to its parameter count compared to larger classical baselines.

---

## Dataset

| | |
|---|---|
| **Source** | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **Samples** | 284,807 transactions |
| **Features** | 30 (28 PCA-anonymised V1–V28 + Amount + Time) |
| **Target** | Binary: Fraud (492 cases, 0.17%) / Legitimate (284,315 cases) |
| **CV** | 5-fold stratified, SMOTE applied strictly inside each fold |

```bash
# Download via Kaggle CLI
kaggle datasets download -d mlg-ulb/creditcardfraud --path data/raw --unzip

# Or manually place creditcard.csv at:
# data/raw/creditcard.csv
```

---

## Project Structure

```
hqnn-fraud-detection-benchmark/
├── configs/
│   └── default.yaml            # All hyperparameters (single source of truth)
├── data/
│   └── raw/                    # Place creditcard.csv here
├── results/
│   ├── folds/                  # Per-fold JSON results
│   ├── figures/                # Generated plots
│   ├── metrics/                # Aggregated metrics
│   └── models/                 # Saved model states
├── scripts/
│   ├── run_benchmark.py        # Full benchmark entrypoint
│   ├── run_fold.py             # Single fold (for parallel dispatch)
│   └── run_plots.py            # Plot generation
├── src/
│   ├── config.py               # Pydantic config schema
│   ├── data/                   # Loader, CV splits, preprocessing
│   ├── models/                 # SHNN, Parallel Hybrid, SNN, TabNet, FT-T, ResNet, SAINT
│   ├── training/               # PyTorch training loop with early stopping
│   └── evaluation/             # Metrics, statistical tests, plots
└── tests/                      # pytest suite
```

---

## Setup

```bash
# Install pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Install all dependencies
pixi install
```

---

## Running

```bash
# Full benchmark (all models, 5 folds)
pixi run benchmark

# Single fold — for parallel dispatch across machines
pixi run fold -- --model shnn --fold 0
pixi run fold -- --model parallel --fold 0

# Tests
pixi run test
```

All hyperparameters are in `configs/default.yaml`. Pass `--config path/to/custom.yaml` to override.

---

## Preprocessing Pipeline

| Challenge | Solution |
|---|---|
| Class imbalance (~0.17% fraud) | SMOTE inside each CV fold (never globally) |
| Outliers in Amount | `RobustScaler` (median/IQR-based) |
| Qubit count constraint | PCA to 8 components (= qubit count), fitted on train fold only |
| Angle encoding range | MinMax to [0, π] after RobustScaler |

---

## HQNN Architectures

### SHNN — Sequential Hybrid Neural Network

```
Input (8) → Linear(8→8) + PiSigmoid → VQC(8 qubits, 2 layers) → Linear(1→1) + Sigmoid → Output
```

- VQC uses AngleEmbedding + BasicEntanglerLayers
- 48 quantum parameters, ~74 classical parameters, ~122 total

### Parallel Hybrid Neural Network

```
Input (8) ──┬─→ MLP [16, 8] ──────────────────┐
            └─→ VQC(8 qubits, 2 layers) ──┐   concat → FC [8] → Sigmoid → Output
                                           └───┘
```

- Quantum and classical streams process the same input independently
- Outputs are concatenated before the final classification head

---

## Evaluation

| Metric | Role |
|---|---|
| **MCC** | Primary — balanced, threshold-aware, robust to imbalance |
| **PR-AUC** | Primary — threshold-free, captures precision/recall trade-off |

Early stopping: patience 20 (quantum) / 15 (classical), monitored on validation MCC.
Final threshold: tuned on validation set post-training using `find_optimal_threshold`.
Statistical test: Wilcoxon signed-rank, effect size: rank-biserial correlation.

---

## Ablation

A structural ablation replaces the VQC output with a constant zero vector to isolate the quantum contribution:

| Condition | MCC | Loss |
|---|---|---|
| SHNN (full) | ~0.22 | ~0.08 |
| SHNN (VQC → zeros) | 0.000 | 0.6932 (random) |

The VQC provides 100% of the model's predictive signal. Without it, SHNN collapses to random prediction.

---

## Glossary

| Term | Definition |
|---|---|
| **Qubit** | Quantum bit — exists in superposition of 0 and 1 simultaneously until measured |
| **VQC** | Variational Quantum Circuit — a parameterised quantum circuit trained by gradient descent |
| **AngleEmbedding** | Encodes classical features as rotation angles on qubits |
| **Adjoint differentiation** | An efficient gradient method for quantum circuits; mathematically equivalent to the parameter-shift rule but faster in simulation |
| **NISQ** | Noisy Intermediate-Scale Quantum — current era of 50–1000 qubit hardware with non-negligible error rates |
| **Hilbert space** | The exponentially large mathematical space in which quantum states live (2ⁿ dimensions for n qubits) |
| **SMOTE** | Synthetic Minority Oversampling Technique — generates synthetic fraud examples to counteract class imbalance |
| **MCC** | Matthews Correlation Coefficient — single balanced metric for binary classification on imbalanced data (range: −1 to +1) |
| **PR-AUC** | Area under the Precision-Recall curve — more informative than ROC-AUC under heavy class imbalance |
| **Wilcoxon signed-rank** | Non-parametric paired statistical test used to compare fold-level metrics without normality assumption |
| **Barren plateau** | Phenomenon where gradients vanish exponentially with circuit depth, making VQC training increasingly difficult |
| **Parameter efficiency** | MCC per trainable parameter — the central thesis metric quantifying representational value per unit of model complexity |

---

## Author

[Gregor Kobilarov](https://github.com/g8rdier)

## License

[MIT](LICENSE)
