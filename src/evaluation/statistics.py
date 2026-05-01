"""
statistics.py
=============
Statistical tests for the benchmark.

- Wilcoxon signed-rank test on paired MCC values across folds
- Rank-biserial correlation as effect size measure

With n=5 folds, the minimum achievable p-value is 0.0625 (exceeds α=0.05),
so the Wilcoxon test is used primarily for consistency analysis and trend
identification. The rank-biserial correlation provides the primary evidence
for effect magnitude.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class StatisticalResult:
    """Result of a pairwise statistical comparison."""

    model_a: str
    model_b: str
    metric: str
    wilcoxon_statistic: float
    wilcoxon_p: float
    rank_biserial: float
    n_folds: int
    interpretation: str

    def to_dict(self) -> dict:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "metric": self.metric,
            "wilcoxon_statistic": self.wilcoxon_statistic,
            "wilcoxon_p": self.wilcoxon_p,
            "rank_biserial": self.rank_biserial,
            "n_folds": self.n_folds,
            "interpretation": self.interpretation,
        }


def rank_biserial_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute rank-biserial correlation (Kerby, 2014) as effect size.

    For paired samples: r = (n_favorable - n_unfavorable) / n_total

    Returns
    -------
    float in [-1, 1]. Positive means x > y on average.
    """
    diffs = x - y
    diffs = diffs[diffs != 0]  # Remove ties
    n = len(diffs)
    if n == 0:
        return 0.0

    n_pos = np.sum(diffs > 0)
    n_neg = np.sum(diffs < 0)
    return (n_pos - n_neg) / n


def compare_models(
    name_a: str,
    scores_a: list[float],
    name_b: str,
    scores_b: list[float],
    metric: str = "MCC",
) -> StatisticalResult:
    """
    Compare two models using Wilcoxon signed-rank test + rank-biserial correlation.

    Parameters
    ----------
    name_a, name_b : Model names.
    scores_a, scores_b : Per-fold metric values (must be same length).
    metric : Name of the metric being compared.

    Returns
    -------
    StatisticalResult
    """
    a = np.array(scores_a)
    b = np.array(scores_b)
    n = len(a)

    if n != len(b):
        raise ValueError(f"Score arrays must match: got {n} vs {len(b)}")  # pragma: no cover

    # Wilcoxon signed-rank test (two-sided)
    diffs = a - b
    if np.all(diffs == 0):
        w_stat, w_p = 0.0, 1.0
    else:
        try:
            result = stats.wilcoxon(a, b, alternative="two-sided")
            w_stat = float(result.statistic)
            w_p = float(result.pvalue)
        except ValueError:  # pragma: no cover
            w_stat, w_p = 0.0, 1.0

    # Effect size
    r = rank_biserial_correlation(a, b)

    # Interpret
    if abs(r) < 0.1:
        effect = "negligible"
    elif abs(r) < 0.3:
        effect = "small"
    elif abs(r) < 0.5:
        effect = "medium"
    else:
        effect = "large"

    direction = f"{name_a} > {name_b}" if r > 0 else f"{name_b} > {name_a}"
    interp = (
        f"{effect} effect ({direction}), r={r:.3f}, "
        f"W={w_stat:.1f}, p={w_p:.4f} (n={n} folds, min p=0.0625)"
    )
    logger.info("Statistical comparison [%s]: %s", metric, interp)

    return StatisticalResult(
        model_a=name_a,
        model_b=name_b,
        metric=metric,
        wilcoxon_statistic=w_stat,
        wilcoxon_p=w_p,
        rank_biserial=r,
        n_folds=n,
        interpretation=interp,
    )
