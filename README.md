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

> "To what extent do Hybrid Quantum Neural Networks achieve superior parameter efficiency compared to classical deep learning models for tabular binary classification under extreme class imbalance?"

---

## Hypotheses

### H1: Classification Performance

**H0₁:** HQNN architectures (SHNN, Parallel Hybrid) achieve no statistically significant difference in MCC compared to classical deep learning baselines (SNN, TabNet, FT-Transformer, ResNet, SAINT).

**H1₁:** HQNN architectures achieve a significantly different MCC compared to classical deep learning baselines.

*Rationale:* MCC is the primary metric — it is the most informative single scalar for binary classification under extreme class imbalance (0.17% fraud). Quantum circuits project input features into an exponentially large Hilbert space, which may yield superior model capacity relative to parameter count even on NISQ simulators.

### H2: Parameter Efficiency

**H0₂:** HQNN architectures achieve no statistically significant advantage in MCC per trainable parameter compared to size-matched classical baselines.

**H1₂:** HQNN architectures achieve a significantly higher MCC per trainable parameter than classical deep learning models of comparable size.

*Rationale:* Hybrid quantum models are intentionally small (~100–150 total parameters). The central thesis claim is not raw performance, but parameter efficiency — whether quantum circuits provide disproportionate representational power relative to their parameter count.

*Statistical validation:* Wilcoxon signed-rank test across 5 CV folds; effect size via rank-biserial correlation.

---

## Models

| Model | Type | Parameters (approx.) |
|---|---|---|
| **SHNN** (Sequential Hybrid NN) | Quantum hybrid | ~122 |
| **Parallel Hybrid NN** | Quantum hybrid | ~200 |
| **SNN** (Self-Normalizing Network) | Classical | ~3,000 |
| **TabNet** | Classical | ~7,000 |
| **FT-Transformer** | Classical | ~12,000 |
| **ResNet** (tabular) | Classical | ~6,000 |
| **SAINT** | Classical | ~18,000 |

Both HQNN architectures use 8 qubits, 2 VQC layers, and PennyLane's `lightning.qubit` backend with adjoint differentiation.

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
| F1-Fraud | Secondary reference |
| ROC-AUC | Secondary reference |

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

MIT
