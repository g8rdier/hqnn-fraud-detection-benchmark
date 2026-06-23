# HQNN Fraud Detection Benchmark

[![CI](https://github.com/g8rdier/hqnn-fraud-detection-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/g8rdier/hqnn-fraud-detection-benchmark/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/g8rdier/hqnn-fraud-detection-benchmark/branch/main/graph/badge.svg)](https://codecov.io/gh/g8rdier/hqnn-fraud-detection-benchmark)

## Academic Context

| | |
|---|---|
| **Thesis title** | Quantum Machine Learning: An Empirical Benchmark of Parameter Efficiency in Hybrid Quantum Neural Networks vs. Classical Deep Learning for Tabular Binary Classification |
| **Degree** | Bachelor of Science Business Informatics |
| **Institution** | IU International University of Applied Sciences, Munich (Dual Study Program) |
| **Supervisor** | Prof. Dr. rer. nat. Michael Barth |
| **Student** | Gregor Kobilarov |
| **Submission** | 24.06.2026 |
| **Dataset** | [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (n = 284,807) |

---

## Research Question

This study explicitly avoids framing HQNNs as absolute performance replacements for classical deep learning. Given the physical constraints of the NISQ era and the inherently parameter-hungry nature of classical architectures, the primary objective is to evaluate **parameter efficiency**: whether HQNNs can achieve a superior ratio of predictive performance to trainable parameter count relative to state-of-the-art classical models.

> "To what extent do Hybrid Quantum Neural Networks (HQNNs) demonstrate a **Parameter Efficiency Advantage** — defined by superior ratios of predictive performance to trainable parameter count (MCC/kParam and PR-AUC/kParam) — over state-of-the-art classical models when classifying highly imbalanced financial tabular data?"

---

## Hypotheses

**H1 — Parameter Efficiency Advantage**
The HQNNs achieve higher parameter efficiency ratios, measured as MCC/kParam and PR-AUC/kParam, than all classical state-of-the-art baselines.

**H2 — Competitive Absolute Performance**
The absolute MCC of the HQNNs is not significantly inferior to that of the Self-Normalizing Neural Network (SNN), the classical baseline closest in trainable parameter count, even though larger classical architectures are expected to retain a higher absolute predictive performance. PR-AUC is assessed as a secondary metric, where a modest disadvantage consistent with the parameter constraint may remain.

**H3 — Non-Trivial VQC Contribution**
The variational quantum circuit (VQC) contributes a non-trivial predictive signal to the hybrid architecture; removing this component causes the model's performance to collapse toward random prediction.

---

## Results

Stratified 80/20 hold-out split, then 5-fold stratified CV on the remaining 80% (development set). Metrics are mean ± std across folds evaluated on the shared hold-out test set.

| Model | Type | Params | MCC | PR-AUC | MCC / kParam | PR-AUC / kParam |
|---|---|---|---|---|---|---|
| **SHNN** | Quantum hybrid | 122 | 0.5758 ± 0.0371 | 0.5910 ± 0.0323 | 4.720 | 4.844 |
| **PHNN** | Quantum hybrid | 489 | 0.5688 ± 0.0371 | 0.6239 ± 0.0101 | 1.163 | 1.276 |
| SNN | Classical | 3,201 | 0.5633 ± 0.0139 | 0.6449 ± 0.0086 | 0.176 | 0.201 |
| TabNet | Classical | 6,176 | 0.4824 ± 0.0732 | 0.6551 ± 0.0399 | 0.078 | 0.106 |
| ResNet | Classical | 8,897 | 0.6933 ± 0.0329 | 0.7170 ± 0.0164 | 0.078 | 0.081 |
| FT-T | Classical | 14,869 | 0.6934 ± 0.0164 | 0.7061 ± 0.0220 | 0.047 | 0.047 |
| SAINT | Classical | 29,357 | 0.6975 ± 0.0164 | 0.6570 ± 0.0505 | 0.024 | 0.022 |

**Key finding:** SHNN achieves comparable MCC to SNN (0.576 vs 0.563) with 26× fewer parameters, yielding a ~27× MCC/kParam advantage. Larger classical models (ResNet, FT-T, SAINT) achieve higher absolute MCC but at 73–240× the parameter count, resulting in 60–197× lower efficiency. The central thesis claim — quantum advantage through parameter efficiency rather than raw performance — is supported.

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
| **Samples** | 284,807 transactions (September 2013) |
| **Features** | 30 (28 PCA-anonymised V1–V28 + Amount + Time) |
| **Target** | Binary: Fraud (492 cases, 0.172%) / Legitimate (284,315 cases) |
| **CV** | Stratified 80/20 hold-out split; 5-fold stratified CV on the 80% development set; SMOTE applied strictly inside each training fold |

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
│   ├── figures/                # Generated plots (PNG + SVG)
│   ├── metrics/                # Aggregated metrics
│   └── models/                 # Saved model states
├── scripts/
│   ├── run_benchmark.py        # Full benchmark entrypoint
│   ├── run_fold.py             # Single fold (for parallel dispatch)
│   ├── run_plots.py            # Benchmark figure generation
│   └── run_conceptual_figures.py  # Conceptual/architecture figures
├── src/
│   ├── config.py               # Pydantic config schema
│   ├── data/                   # Loader, CV splits, preprocessing
│   ├── models/                 # SHNN, PHNN, SNN, TabNet, FT-T, ResNet, SAINT
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

## Reproducing Figures

Fold results are committed, so all benchmark figures can be regenerated without re-running the benchmark (~185 h).

```bash
# 12 benchmark figures — no dataset needed, reads committed results/folds/*.json
pixi run plots

# 7 conceptual figures — requires creditcard.csv in data/raw/
pixi run python scripts/run_conceptual_figures.py

# Wilcoxon signed-rank statistics — no dataset needed
pixi run python scripts/run_statistics.py
```

All output goes to `results/figures/`.

---

## Preprocessing Pipeline

| Challenge | Solution |
|---|---|
| Class imbalance (~0.172% fraud) | SMOTE inside each CV fold (never globally) |
| Outliers in Amount | `RobustScaler` (median/IQR-based) |
| Qubit count constraint | PCA to 8 components (= qubit count), fitted on train fold only |
| Angle encoding range | MinMaxScaler to [0, π] after RobustScaler |

---

## HQNN Architectures

### SHNN — Serial Hybrid Quantum Neural Network

```
Input (8) → Linear(8→8) + PiSigmoid → VQC(8 qubits, 2 layers) → Linear(1→1) + Sigmoid → Output
```

- Strict bottleneck architecture: classical layers compress before passing sequentially to the VQC
- VQC uses AngleEmbedding + StronglyEntanglingLayers (adjoint differentiation via `lightning.qubit`)
- 48 quantum parameters, 74 classical parameters, 122 total

### PHNN — Parallel Hybrid Quantum Neural Network

```
Input (8) ──┬─→ Linear(8→8) + PiSigmoid → VQC(8 qubits, 2 layers) → <Z₀> ──┐
            └─→ MLP [8→16→8] ReLU ────────────────────────────────────────────┤
                                                                    Concat [8+1=9] → FC [9→8→1] + Sigmoid → Output
```

- Single VQC with the same 48-angle ansatz as the SHNN, plus a concurrent classical MLP branch
- Both streams are concatenated before the final classification head
- 48 quantum parameters, 441 classical parameters, 489 total
- Additional capacity over SHNN is entirely classical — controlled test of supplementary classical feature interaction

---

## Evaluation

| Metric | Role |
|---|---|
| **MCC** | Primary (threshold-dependent) — balanced, robust to imbalance; high score only when both classes are correctly identified |
| **PR-AUC** | Primary (threshold-independent) — captures precision/recall trade-off across all thresholds; more informative than ROC-AUC under heavy class imbalance |

Early stopping: patience 20 (quantum) / 15 (classical), monitored on validation MCC.
Final threshold: tuned via grid search on validation set post-training to maximise MCC.
Statistical test: Wilcoxon signed-rank; effect size: rank-biserial correlation.

---

## Ablation

A structural ablation replaces the VQC output with a constant zero vector to isolate the quantum contribution. Both conditions were trained from scratch for 10 epochs on fold 0 under identical hyperparameters:

| Condition | MCC | Loss |
|---|---|---|
| SHNN (full, 10 epochs) | ~0.22 | ~0.08 |
| SHNN (VQC → zeros, 10 epochs) | 0.000 | 0.6932 (random) |

The VQC provides 100% of the model's predictive signal. Without it, SHNN collapses to random prediction.

---

## Glossary

| Term | Definition |
|---|---|
| **Qubit** | The fundamental unit of quantum information that, unlike a classical bit, can exist in a superposition of 0 and 1 simultaneously until a measurement is performed |
| **VQC** | Variational Quantum Circuit — a parameterised quantum circuit trained by gradient descent |
| **SHNN** | Serial Hybrid Quantum Neural Network — the minimal-parameter HQNN variant (122 params); classical layers compress the input before passing it sequentially into a single VQC |
| **PHNN** | Parallel Hybrid Quantum Neural Network — the second HQNN variant (489 params); a single VQC and a classical MLP branch process the input in parallel, with outputs concatenated before the classification head |
| **AngleEmbedding** | Encodes classical features as rotation angles on qubits (Pauli-X rotations); requires O(n) qubits for n features but yields shallow circuits compatible with NISQ hardware |
| **StronglyEntanglingLayers** | Parameterised ansatz used in the VQC; interleaves single-qubit rotations with entangling gates to create expressive quantum feature maps |
| **Adjoint differentiation** | Gradient method that computes exact analytical gradients of a quantum circuit without additional circuit evaluations; used here via PennyLane's `lightning.qubit` backend to bypass shot-noise overhead |
| **NISQ** | Noisy Intermediate-Scale Quantum — current era of 50–1,000 qubit hardware characterised by high gate error rates, limited coherence times, and no fault tolerance |
| **Hilbert space** | The exponentially large (2ⁿ-dimensional) mathematical space in which n-qubit quantum states live; the source of QNN expressivity advantage |
| **Barren plateau** | Phenomenon where gradients vanish exponentially with circuit depth/width, making VQC training increasingly difficult at scale |
| **Parameter efficiency** | MCC (or PR-AUC) per 1,000 trainable parameters — the central thesis metric quantifying representational value per unit of model complexity |
| **MCC** | Matthews Correlation Coefficient — single balanced metric for binary classification on imbalanced data (range: −1 to +1); high only when both classes are correctly identified |
| **PR-AUC** | Area under the Precision-Recall curve — more informative than ROC-AUC under heavy class imbalance; evaluates global ordering of prediction scores across all thresholds |
| **SMOTE** | Synthetic Minority Over-Sampling Technique — generates synthetic fraud examples by interpolating between existing minority-class neighbours to counteract class imbalance |
| **SNN** | Self-Normalizing Neural Network — uses the SELU activation function to induce self-normalizing properties, enabling stable training of deep feed-forward networks |
| **FT-T** | Feature Tokenizer Transformer — adapts the transformer architecture to tabular data by tokenizing each numerical/categorical feature into a uniform embedding |
| **SAINT** | Self-Attention and Intersample Attention Transformer — extends tabular transformers by capturing both intra-instance feature correlations and inter-sample correlations within a batch |
| **Wilcoxon signed-rank** | Non-parametric paired statistical test used to compare fold-level metrics without normality assumption; with n=5 folds the minimum attainable p-value is 0.0625 |
| **Rank-biserial correlation** | Effect size measure for the Wilcoxon test; used as the primary inferential measure because p=0.0625 cannot reach the conventional 0.05 threshold with 5 folds |
| **Depolarizing noise** | Standard NISQ noise model in which each quantum gate introduces random errors with probability p, approximating physical gate error rates |

---

## Cite this repository

> Kobilarov, G. (2026). *HQNN Fraud Detection Benchmark* (v1.0.0) \[Software\]. GitHub.
> https://github.com/g8rdier/hqnn-fraud-detection-benchmark/releases/tag/v1.0.0

---

## Author

[Gregor Kobilarov](https://github.com/g8rdier)

## License

[MIT](LICENSE)
