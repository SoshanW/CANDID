"""Sesia et al.'s Algorithm 2: adaptive calibration under a bounded contamination model.

Arm (b) of D-032 (a point estimate, as a zero-width box) and, later, arm (c) (an
elicited region). Ported from the equations of Sesia, M., Wang, Y.X.R. and Tong, X.
(2025) 'Adaptive conformal classification with noisy labels', JRSSB 87(3), 796-815,
and checked term by term against the authors' implementation,
``cln/classification.py`` in github.com/msesia/conformal-label-noise (MIT licence,
Copyright (c) 2023 Matteo Sesia), class ``LabelNoiseConformal``. Not wrapped: ``cln``
imports ``arc``, ``statsmodels`` and torch, and ``src/conformal/`` is numpy-only.

The computation, per condition k, with scores ``s_hat(x, k) = 1 - pi_hat(x, k)``
(D-048):

- eq. (12): ``F_hat^k_l(t)``, the empirical CDF of ``s_hat(X_i, k)`` over calibration
  points with proxy label ``l``;
- eq. (19): ``delta_star_k = max_{l != k} (V_upp - V_low)[k, l]`` and the uniformity
  bound ``zeta_upp_k``;
- eq. (20): ``Delta_ci_k(t) = sum_{l != k} V_upp[k, l] (F^k_l - F^k_k)
  - delta_star_k (K - 1) |F^k_k - sum_{l != k} F^k_l / (K - 1)|
  - |zeta_upp_k| sum_{l != k} |F^k_l - F^k_k|``;
- eq. (21): ``delta_ci(n_k, n_star)``, computed by ``v_region.correction_term``;
- eq. (22), conservative: ``I_k = {i : i / n_k >= 1 - alpha - Delta_ci_k(S_(i)) + delta_ci}``;
- eq. (25), optimistic: ``I_k = {i : i / n_k >= 1 - alpha
  - max{Delta_ci_k(S_(i)) - delta_ci, -(1 - alpha) / n_k}}``;
- eq. (15): ``tau_k = S_(min I_k)``, or 1 when ``I_k`` is empty.

D-049 makes the conservative variant the arm of record and reports the optimistic one
alongside.

Two choices made here, both visible to an examiner:

1. **``zeta_upp`` defaults to ``delta_star`` per row.** Equation (19) defines
   ``zeta_k = max_l (V_upp - V)[k, l] - min_l (V_upp - V)[k, l]``, which needs the true V.
   When V lies in the box, every ``(V_upp - V)[k, l]`` lies in ``[0, width_kl]``, so
   ``zeta_k <= delta_star_k``: a valid upper bound "extracted directly from" the box, as
   the paper allows. For a zero-width box it is zero, and equation (20) then reduces to
   equation (13) with the point estimate (D-049).
2. **``delta_star`` is per row**, as equation (19) defines it. ``cln`` uses the maximum
   width over the whole matrix (``np.nanmax(self.V_upp - self.V_low)``), which is at
   least as large and so at least as conservative. The two coincide for a zero-width box.

Light module: numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.conformal.v_region import correction_term

__all__ = [
    "AdaptiveCalibration",
    "InflationReference",
    "PropositionThreeCheck",
    "adaptive_thresholds",
    "coverage_inflation",
    "proposition_three_check",
    "zero_width_box",
]

VARIANTS = ("conservative", "optimistic")
_RANK_GUARD = 1e-9
"""Floating-point slack on the membership test i / n_k >= bound, so that an exact
equality is not lost to rounding (see ``split_conformal._RANK_GUARD``)."""


def zero_width_box(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A point estimate as Algorithm 2's input: ``V_low = V_upp = V``, ``alpha_V = 0``."""
    v = np.asarray(v, dtype=float)
    return v.copy(), v.copy()


def _empirical_cdf_at(sorted_values: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Eq. (12) evaluated at ``t``: share of ``sorted_values`` that are ``<= t``."""
    if sorted_values.size == 0:
        raise ValueError("empty calibration class")
    return np.searchsorted(sorted_values, t, side="right") / sorted_values.size


@dataclass(frozen=True)
class AdaptiveCalibration:
    """Output of Algorithm 2 for every condition.

    Attributes:
        tau: thresholds ``tau_k``, shape ``(K,)``; 1 means condition k is in every set.
        delta_ci: eq. (21) correction per condition, shape ``(K,)``.
        index: the selected rank ``i_hat_k`` (1-based), or 0 when ``I_k`` was empty.
        n_k: calibration count per proxy-label class.
        variant: ``"conservative"`` (eq. 22) or ``"optimistic"`` (eq. 25).
    """

    tau: np.ndarray
    delta_ci: np.ndarray
    index: np.ndarray
    n_k: np.ndarray
    variant: str


def adaptive_thresholds(
    scores: np.ndarray,
    y_tilde: np.ndarray,
    v_low: np.ndarray,
    v_upp: np.ndarray,
    alpha: float = 0.10,
    variant: str = "conservative",
    alpha_v: float = 0.0,
    v_bar_upp: np.ndarray | None = None,
    zeta_upp: np.ndarray | None = None,
    c_n_k: dict[int, float] | None = None,
) -> AdaptiveCalibration:
    """Algorithm 2, steps 3 to 12, for every condition.

    Args:
        scores: calibration scores ``s_hat(X_i, k)``, shape ``(n, K)``.
        y_tilde: calibration proxy labels, shape ``(n,)``.
        v_low, v_upp: the box on the off-diagonals of ``V = M^-1`` (diagonals ignored).
        alpha: target miscoverage.
        variant: ``"conservative"`` (eq. 22, Theorem 4) or ``"optimistic"`` (eq. 25,
            Proposition 3).
        alpha_v, v_bar_upp: the box's level and a priori bound, passed to eq. (21).
        zeta_upp: per-condition bound for eq. (19)'s ``zeta_k``; defaults to
            ``delta_star_k`` (module docstring, choice 1).
        c_n_k: optional precomputed ``c(n_k)`` keyed by ``n_k``; the Monte Carlo of
            eq. (16) is slow at large n (``CLAUDE.md``).

    Returns:
        ``AdaptiveCalibration``.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    scores = np.asarray(scores, dtype=float)
    y_tilde = np.asarray(y_tilde)
    v_low = np.asarray(v_low, dtype=float)
    v_upp = np.asarray(v_upp, dtype=float)
    k_classes = scores.shape[1]
    if v_low.shape != (k_classes, k_classes) or v_upp.shape != v_low.shape:
        raise ValueError(f"V box must be {k_classes}x{k_classes}")
    off = ~np.eye(k_classes, dtype=bool)
    if (v_upp[off] < v_low[off]).any():
        raise ValueError("V_upp must be >= V_low on every off-diagonal entry")

    width = np.where(off, v_upp - v_low, 0.0)
    delta_star = width.max(axis=1)  # eq. (19), per row
    zeta = delta_star if zeta_upp is None else np.asarray(zeta_upp, dtype=float)

    # Per (l, k): sorted scores for class k among calibration points with proxy label l.
    by_label = [
        [np.sort(scores[y_tilde == l, k]) for k in range(k_classes)]
        for l in range(k_classes)
    ]
    n_k = np.array([by_label[k][k].size for k in range(k_classes)])
    if (n_k == 0).any():
        raise ValueError(f"every condition needs calibration points, got counts {n_k}")
    n_star = int(n_k.min())

    tau = np.ones(k_classes)
    delta_ci = np.zeros(k_classes)
    index = np.zeros(k_classes, dtype=int)
    for k in range(k_classes):
        s_sorted = by_label[k][k]  # S^k_(1..n_k), step 7
        others = [l for l in range(k_classes) if l != k]
        f_kk = _empirical_cdf_at(s_sorted, s_sorted)  # step 8
        f_lk = {l: _empirical_cdf_at(by_label[l][k], s_sorted) for l in others}

        # Step 9, eq. (20).
        term_a = sum(v_upp[k, l] * (f_lk[l] - f_kk) for l in others)
        term_b = (
            delta_star[k]
            * (k_classes - 1)
            * np.abs(f_kk - sum(f_lk[l] for l in others) / (k_classes - 1))
        )
        term_c = abs(zeta[k]) * sum(np.abs(f_lk[l] - f_kk) for l in others)
        delta_hat = term_a - term_b - term_c

        # Step 10, eq. (21).
        c = None if c_n_k is None else c_n_k.get(int(n_k[k]))
        delta_ci[k] = correction_term(
            k, v_low, v_upp, int(n_k[k]), n_star, alpha_v=alpha_v,
            v_bar_upp=v_bar_upp, c_n_k=c,
        )

        # Step 11, eq. (22) or (25).
        i = np.arange(1, n_k[k] + 1)
        if variant == "conservative":
            adjustment = delta_hat - delta_ci[k]
        else:
            adjustment = np.maximum(delta_hat - delta_ci[k], -(1 - alpha) / n_k[k])
        members = np.flatnonzero(i / n_k[k] + _RANK_GUARD >= 1 - alpha - adjustment)

        # Step 12, eq. (15).
        if members.size:
            index[k] = members[0] + 1
            tau[k] = s_sorted[members[0]]

    return AdaptiveCalibration(tau=tau, delta_ci=delta_ci, index=index, n_k=n_k, variant=variant)


def coverage_inflation(
    scores_k: np.ndarray, y: np.ndarray, y_tilde: np.ndarray, k: int, t: np.ndarray
) -> np.ndarray:
    """The true ``Delta_k(t) = F^k_k(t) - Ftilde^k_k(t)`` of eq. (7), on a labelled sample.

    ``F^k_k`` is the CDF of ``s_hat(X, k)`` given the *clean* label ``Y = k`` and
    ``Ftilde^k_k`` given the *proxy* label ``Ytilde = k`` (eq. (6)). Only the synthetic
    harness has ``Y``, so this is computable there and nowhere else.
    """
    scores_k = np.asarray(scores_k, dtype=float)
    clean = np.sort(scores_k[np.asarray(y) == k])
    noisy = np.sort(scores_k[np.asarray(y_tilde) == k])
    return _empirical_cdf_at(clean, t) - _empirical_cdf_at(noisy, t)


@dataclass(frozen=True)
class PropositionThreeCheck:
    """Proposition 3's condition, in the two forms D-049 records, for one condition.

    Attributes:
        inf_delta: ``inf_t Delta_k(t)`` over all real t, which is ``<= 0`` because both
            CDFs vanish below the score range.
        at_threshold: ``Delta_k(tau_k)`` at the optimistic variant's threshold.
        bound: ``delta_ci(n_k, n_star) - (1 - alpha) / n_k``.
        holds_as_stated: ``inf_delta >= bound`` (Proposition 3 as printed).
        holds_at_threshold: ``at_threshold >= bound`` (the form its proof uses; D-049).
    """

    inf_delta: float
    at_threshold: float
    bound: float

    @property
    def holds_as_stated(self) -> bool:
        return self.inf_delta >= self.bound

    @property
    def holds_at_threshold(self) -> bool:
        return self.at_threshold >= self.bound


class InflationReference:
    """A large labelled sample standing in for the population, presorted once.

    ``Delta_k`` (eq. (7)) needs the clean label, so it is computable only on the
    synthetic harness. Evaluating it at many thresholds would otherwise re-sort the
    reference sample on every call.
    """

    def __init__(self, scores: np.ndarray, y: np.ndarray, y_tilde: np.ndarray) -> None:
        scores = np.asarray(scores, dtype=float)
        y, y_tilde = np.asarray(y), np.asarray(y_tilde)
        self.k_classes = scores.shape[1]
        self._clean = [np.sort(scores[y == k, k]) for k in range(self.k_classes)]
        self._noisy = [np.sort(scores[y_tilde == k, k]) for k in range(self.k_classes)]
        self._grid = [np.sort(scores[:, k]) for k in range(self.k_classes)]
        self._inf = [
            float(min(0.0, self.delta(k, self._grid[k]).min())) for k in range(self.k_classes)
        ]

    def delta(self, k: int, t: np.ndarray) -> np.ndarray:
        """``Delta_k(t) = F^k_k(t) - Ftilde^k_k(t)``."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return _empirical_cdf_at(self._clean[k], t) - _empirical_cdf_at(self._noisy[k], t)

    def inf_delta(self, k: int) -> float:
        """``inf_t Delta_k(t)`` over all real t: the minimum over every reference score
        and 0, the value of ``Delta_k`` below the score range."""
        return self._inf[k]


def proposition_three_check(
    reference: InflationReference,
    optimistic: AdaptiveCalibration,
    alpha: float = 0.10,
) -> list[PropositionThreeCheck]:
    """Evaluate Proposition 3's condition per condition, in both forms (D-049)."""
    if optimistic.variant != "optimistic":
        raise ValueError("pass the optimistic calibration; its threshold is the one the proof uses")
    return [
        PropositionThreeCheck(
            inf_delta=reference.inf_delta(k),
            at_threshold=float(reference.delta(k, optimistic.tau[k])[0]),
            bound=float(optimistic.delta_ci[k] - (1 - alpha) / optimistic.n_k[k]),
        )
        for k in range(reference.k_classes)
    ]
