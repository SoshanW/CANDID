"""Arm (a): noise-blind split conformal, label-conditional (roadmap §5 step 5).

The standard method that Sesia, Wang and Tong (2025) review in their §2.2, applied to
proxy labels as if they were clean. It is arm (a) of D-032 and the reference every
other arm is compared against.

Conventions, each fixed by a decision entry:

- **Prediction function (D-048).** Sesia et al.'s equation (3),
  ``C(x, tau) = {k : pi_hat(x, k) >= 1 - tau_k}``, with non-conformity score
  ``s_hat(x, k) = 1 - pi_hat(x, k)`` (their equation (2)). Membership is therefore
  ``s_hat(x, k) <= tau_k``, and ``tau_k = 1`` puts k in every set (Definition 1).
  Deterministic; empty sets are allowed and reported.
- **Mondrian calibration (D-029).** One threshold per condition, from the calibration
  points whose *proxy* label is that condition: equation (4),
  ``tau_k = ceil((1 + n_k)(1 - alpha))``-th smallest of ``{s_hat(X_i, k) : Ytilde_i = k}``,
  and ``tau_k = 1`` when that rank exceeds ``n_k``.
- **Coverage is measured against the clean label** ``Y``, which only the synthetic
  harness has. That is the quantity Theorem 1 bounds and the one C2 is about.

Light module: numpy only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "SetMetrics",
    "evaluate_sets",
    "nonconformity_scores",
    "predict_sets",
    "standard_rank",
    "standard_thresholds",
]

_RANK_GUARD = 1e-9
"""Floating-point guard on ceil(): (n + 1)(1 - alpha) can land a hair above an integer
(for example 0.9 * 10 = 9.000000000000002 in some orderings), which would push the rank
up by one. The guard is far below 1 / n for any n here."""


def nonconformity_scores(probs: np.ndarray) -> np.ndarray:
    """Equation (2) for the prediction function (3): ``s_hat(x, k) = 1 - pi_hat(x, k)``."""
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2:
        raise ValueError(f"probs must be (n, K), got shape {probs.shape}")
    return 1.0 - probs


def standard_rank(n_k: int, alpha: float) -> int:
    """The rank in equation (4): ``ceil((1 + n_k)(1 - alpha))``."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return math.ceil((1 + n_k) * (1 - alpha) - _RANK_GUARD)


def standard_thresholds(
    scores: np.ndarray, y_tilde: np.ndarray, alpha: float = 0.10
) -> np.ndarray:
    """Label-conditional thresholds of equation (4), one per condition.

    Args:
        scores: calibration non-conformity scores, shape ``(n, K)``.
        y_tilde: calibration proxy labels, shape ``(n,)``.
        alpha: target miscoverage.

    Returns:
        ``tau`` of shape ``(K,)``. ``tau[k] = 1`` when the rank exceeds ``n_k``.

    Raises:
        ValueError: if a condition has no calibration points.
    """
    scores = np.asarray(scores, dtype=float)
    y_tilde = np.asarray(y_tilde)
    k_classes = scores.shape[1]
    tau = np.ones(k_classes)
    for k in range(k_classes):
        class_scores = np.sort(scores[y_tilde == k, k])
        n_k = class_scores.size
        if n_k == 0:
            raise ValueError(f"condition {k} has no calibration points")
        rank = standard_rank(n_k, alpha)
        if rank <= n_k:
            tau[k] = class_scores[rank - 1]
    return tau


def predict_sets(scores: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Prediction sets of equation (3) as a boolean ``(n, K)`` membership matrix.

    ``k`` is in the set for ``x`` exactly when ``s_hat(x, k) <= tau_k``, and every
    condition with ``tau_k = 1`` is in every set.
    """
    scores = np.asarray(scores, dtype=float)
    tau = np.asarray(tau, dtype=float)
    return (scores <= tau[None, :]) | (tau[None, :] >= 1.0)


@dataclass(frozen=True)
class SetMetrics:
    """Per-condition behaviour of a set of predictions, conditioned on the clean label.

    Attributes:
        coverage: ``P[Y in C(X) | Y = k]`` per condition, shape ``(K,)``.
        mean_size: mean set size among points with ``Y = k``, shape ``(K,)``.
        empty_rate: share of empty sets among points with ``Y = k``, shape ``(K,)``.
        full_rate: share of full sets (all K conditions) among points with ``Y = k``.
        n: number of test points per clean condition, shape ``(K,)``.
    """

    coverage: np.ndarray
    mean_size: np.ndarray
    empty_rate: np.ndarray
    full_rate: np.ndarray
    n: np.ndarray


def evaluate_sets(sets: np.ndarray, y: np.ndarray) -> SetMetrics:
    """Label-conditional coverage and size against the clean label (D-048 reporting)."""
    sets = np.asarray(sets, dtype=bool)
    y = np.asarray(y)
    k_classes = sets.shape[1]
    sizes = sets.sum(axis=1)
    covered = sets[np.arange(len(y)), y]
    coverage, mean_size, empty, full, counts = (np.full(k_classes, np.nan) for _ in range(5))
    for k in range(k_classes):
        mask = y == k
        counts[k] = mask.sum()
        if counts[k] == 0:
            continue
        coverage[k] = covered[mask].mean()
        mean_size[k] = sizes[mask].mean()
        empty[k] = (sizes[mask] == 0).mean()
        full[k] = (sizes[mask] == k_classes).mean()
    return SetMetrics(coverage, mean_size, empty, full, counts.astype(int))
