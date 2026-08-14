"""Convert C1 transition matrices into Sesia's V, and test Algorithm 2 for degeneracy.

Sesia, Wang and Tong (2025) define their contamination model in the *reverse*
direction from the transition matrix that HOC and cleanlab estimate:

    Sesia    M[k, l] = P[Y = l | Ytilde = k]     (their Proposition 1)
    C1       T[l, k] = P[Ytilde = k | Y = l]     (D-037, D-039, D-041)

Algorithm 2 consumes a ``1 - alpha_V`` simultaneous confidence region on the
*off-diagonal* entries of ``V = M^-1``. Only the off-diagonals are needed because M's
rows sum to one, hence so do V's, hence ``V[k, k] = 1 - sum_{l != k} V[k, l]``.

The degeneracy test is derived in ``Docs/c2-interface.md`` sec. 5. Evaluating the
membership test of Algorithm 2 at ``i = n_k``, where the class-k score CDF equals one
by construction, gives the sufficient condition

    tau_k = 1   whenever   delta_ci(n_k, n_star) > alpha + Dhat_ci_k(S_(n_k))

and ``tau_k = 1`` places class k in *every* prediction set, which is the always-abstain
failure mode named in D-032. Since ``0 <= Dhat_ci_k(S_(n_k)) <= sum_{l != k}
|V_upp[k, l]|``, testing ``delta_ci > alpha`` alone over-flags degeneracy by at most
that bound, so a verdict is robust only when the margin exceeds it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ConditionBudget",
    "DegeneracyVerdict",
    "correction_term",
    "degeneracy_test",
    "expected_sup_deviation",
    "max_admissible_offdiagonal_mass",
    "region_from_matrices",
    "transition_to_v",
]


def transition_to_v(
    transition: np.ndarray, noisy_prior: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a C1 transition matrix into Sesia's ``V``.

    Args:
        transition: ``T[l, k] = P[Ytilde = k | Y = l]``, rows summing to one.
        noisy_prior: observed proxy-label frequencies ``rho_tilde``.

    Returns:
        ``(V, M, clean_prior)``. The clean prior is *recovered* from the identity
        ``rho_tilde = T.T @ rho`` rather than assumed, so it inherits whatever error
        ``T`` carries and can in principle leave the simplex. Callers should check it.

    Raises:
        ValueError: if ``transition`` rows do not sum to one, or either matrix is
            singular.
    """
    transition = np.asarray(transition, dtype=float)
    noisy_prior = np.asarray(noisy_prior, dtype=float)
    noisy_prior = noisy_prior / noisy_prior.sum()

    # The matrices in DECISIONS.md are transcribed to 4 decimal places, so rows land
    # within 1e-4 of one rather than on it. Tolerate that, then renormalise.
    row_sums = transition.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        raise ValueError(f"transition rows must sum to 1, got {row_sums}")
    transition = transition / row_sums[:, None]

    try:
        clean_prior = np.linalg.solve(transition.T, noisy_prior)
        # M[k, l] = T[l, k] * rho[l] / rho_tilde[k]
        m = (transition.T * clean_prior[None, :]) / noisy_prior[:, None]
        v = np.linalg.inv(m)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - degenerate input
        raise ValueError(f"matrix inversion failed: {exc}") from exc

    return v, m, clean_prior


def region_from_matrices(
    transitions: list[np.ndarray], noisy_prior: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Elementwise min/max box over the ``V`` implied by several estimates.

    This is the deterministic-containment region of D-031 option (a): it contains
    every supplied estimate by construction, so ``alpha_V = 0``.
    """
    stack = np.stack([transition_to_v(t, noisy_prior)[0] for t in transitions])
    return stack.min(axis=0), stack.max(axis=0)


def expected_sup_deviation(n: int, reps: int = 20_000, seed: int = 0) -> float:
    """``c(n_k)`` of Sesia et al. eq. (16), by Monte Carlo as they prescribe.

    ``c(n) = E[ sup_i ( i/n - U_(i) ) ]`` for ``U`` iid uniform. Scales as
    ``1 / sqrt(n)``.
    """
    rng = np.random.default_rng(seed)
    index = np.arange(1, n + 1) / n
    draws = rng.random((reps, n))
    draws.sort(axis=1)
    return float(np.max(index[None, :] - draws, axis=1).mean())


def correction_term(
    condition: int,
    v_low: np.ndarray,
    v_upp: np.ndarray,
    n_k: int,
    n_star: int,
    alpha_v: float = 0.0,
    v_bar_upp: np.ndarray | None = None,
    c_n_k: float | None = None,
) -> float:
    """``delta_ci(n_k, n_star)`` of Sesia et al. eq. (21).

    Note that for negative off-diagonals (which is every matrix on record here)
    ``|V_upp| + width == |V_low|``, so the sum below is the largest-magnitude end of
    the region. The binding quantity is therefore the greatest total off-diagonal mass
    the region admits.
    """
    k_classes = v_low.shape[0]
    others = [l for l in range(k_classes) if l != condition]

    sum_upp = sum(abs(v_upp[condition, l]) for l in others)
    sum_width = sum(v_upp[condition, l] - v_low[condition, l] for l in others)

    if alpha_v > 0.0:
        if v_bar_upp is None:
            raise ValueError("v_bar_upp is required when alpha_v > 0")
        sum_bar = sum(abs(v_bar_upp[condition, l]) for l in others)
    else:
        sum_bar = 0.0

    multiplier = min(
        k_classes * np.sqrt(np.pi / 2),
        1 / np.sqrt(n_star)
        + np.sqrt((np.log(2 * k_classes) + np.log(n_star)) / 2),
    )
    c = expected_sup_deviation(n_k) if c_n_k is None else c_n_k

    return float(
        c
        + 2 * alpha_v * sum_bar
        + 2 * (sum_upp + sum_width) / np.sqrt(n_star) * multiplier
    )


@dataclass(frozen=True)
class DegeneracyVerdict:
    """Outcome of the closed-form test for one condition."""

    condition: int
    delta_ci: float
    alpha: float
    correction_bound: float
    """``Dhat_max``: the most the exact criterion can differ from ``delta_ci > alpha``."""

    @property
    def margin(self) -> float:
        return self.delta_ci - self.alpha

    @property
    def degenerate(self) -> bool:
        """True when the test fires. Class enters every prediction set."""
        return self.margin > 0

    @property
    def robust(self) -> bool:
        """True when the verdict survives the exactness correction."""
        return self.margin > self.correction_bound

    def label(self) -> str:
        if not self.degenerate:
            return "ok"
        return "DEGENERATE (robust)" if self.robust else "DEGENERATE (borderline)"


def degeneracy_test(
    v_low: np.ndarray,
    v_upp: np.ndarray,
    n_per_condition: list[int],
    alpha: float = 0.10,
    alpha_v: float = 0.0,
    v_bar_upp: np.ndarray | None = None,
) -> list[DegeneracyVerdict]:
    """Run the closed-form degeneracy test for every condition.

    ``n_star = min(n_per_condition)``, so the rarest condition's calibration count
    sets the correction term for *all* conditions, not only its own.
    """
    n_star = min(n_per_condition)
    verdicts = []
    for k, n_k in enumerate(n_per_condition):
        delta = correction_term(
            k, v_low, v_upp, n_k, n_star, alpha_v=alpha_v, v_bar_upp=v_bar_upp
        )
        bound = sum(
            abs(v_upp[k, l]) for l in range(v_low.shape[0]) if l != k
        )
        verdicts.append(
            DegeneracyVerdict(
                condition=k, delta_ci=delta, alpha=alpha, correction_bound=bound
            )
        )
    return verdicts


@dataclass(frozen=True)
class ConditionBudget:
    """The widest region a condition can tolerate before Algorithm 2 degenerates."""

    condition: int
    max_offdiagonal_mass: float
    """Largest admissible ``sum_{l != k} (|V_upp[k, l]| + width)``."""


def max_admissible_offdiagonal_mass(
    n_per_condition: list[int],
    k_classes: int = 4,
    alpha: float = 0.10,
) -> list[ConditionBudget]:
    """Invert the degeneracy test: how much region width fits inside ``alpha``.

    This is the feasibility frontier of the roadmap's step 3. Assumes
    ``alpha_v = 0``.
    """
    n_star = min(n_per_condition)
    multiplier = min(
        k_classes * np.sqrt(np.pi / 2),
        1 / np.sqrt(n_star)
        + np.sqrt((np.log(2 * k_classes) + np.log(n_star)) / 2),
    )
    budgets = []
    for k, n_k in enumerate(n_per_condition):
        c = expected_sup_deviation(n_k)
        budgets.append(
            ConditionBudget(
                condition=k,
                max_offdiagonal_mass=float(
                    (alpha - c) * np.sqrt(n_star) / (2 * multiplier)
                ),
            )
        )
    return budgets
