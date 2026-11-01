"""Scope a clinically elicited region against Algorithm 2's budget (roadmap step 3b).

D-028 elicits the contamination model in Penso's direction, ``T[l, k] = P[Ytilde = k |
Y = l]``, and converts it into Sesia's ``V = M^-1``. An elicited region is therefore a
set of transition matrices, not a single one: here, an interval ``[lower, upper]`` on
every off-diagonal ``T[l, k]``, with each diagonal implied by its row.

Deterministic containment (D-031 option (a), ``alpha_V = 0``) requires the V-box handed
to Algorithm 2 to contain ``V(T)`` for *every* T in the elicited set. ``v_box`` computes
that box as the per-entry minimum and maximum of V over the set, by enumerating the
vertices of the interval box. V is a rational function of T, so its extremes need not
sit at vertices in general; ``v_box`` therefore also samples the interior and reports
whether any interior point escapes the vertex box. For every region evaluated in
``Docs/c2-elicitation-scoping.md`` none did, and a bounded local optimiser run offline
(scipy L-BFGS-B from every vertex extreme) moved no extreme either.

The scenario regions at the bottom are CONSTRUCTED to map the feasibility frontier.
None of them is an elicitation, and none should be quoted as one.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from src.conformal.v_region import correction_term

__all__ = [
    "ElicitedRegion",
    "RegionVerdict",
    "SCOPING_REGIONS",
    "evaluate_region",
    "single_entry_breach_points",
    "v_box",
    "worst_case_label_ppv",
]

K = 4
OFF_DIAGONAL = [(i, j) for i in range(K) for j in range(K) if i != j]


@dataclass(frozen=True)
class ElicitedRegion:
    """Interval bounds on the off-diagonals of ``T``, in ``CONDITIONS`` order.

    ``upper`` and ``lower`` map ``(true, label)`` index pairs to probabilities. A pair
    missing from ``upper`` is fixed at zero; a pair missing from ``lower`` has lower
    bound zero.
    """

    upper: dict[tuple[int, int], float]
    lower: dict[tuple[int, int], float] = field(default_factory=dict)
    note: str = ""

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([float(self.lower.get(e, 0.0)) for e in OFF_DIAGONAL])
        hi = np.array([float(self.upper.get(e, 0.0)) for e in OFF_DIAGONAL])
        if (lo > hi).any():
            raise ValueError("a lower bound exceeds its upper bound")
        if (lo < 0).any() or (hi > 1).any():
            raise ValueError("bounds must lie in [0, 1]")
        return lo, hi


def _transition(offdiag: np.ndarray) -> np.ndarray:
    t = np.zeros((K, K))
    for (i, j), x in zip(OFF_DIAGONAL, offdiag):
        t[i, j] = x
    t[np.diag_indices(K)] = 1.0 - t.sum(axis=1)
    return t


def _v_and_m(offdiag: np.ndarray, noisy_prior: np.ndarray):
    t = _transition(offdiag)
    rho = np.linalg.solve(t.T, noisy_prior)
    m = (t.T * rho[None, :]) / noisy_prior[:, None]
    return np.linalg.inv(m), m, rho


def _vertices(region: ElicitedRegion):
    lo, hi = region.bounds()
    free = [n for n in range(len(OFF_DIAGONAL)) if hi[n] > lo[n]]
    for corner in itertools.product((0, 1), repeat=len(free)):
        vec = lo.copy()
        for n, c in zip(free, corner):
            vec[n] = hi[n] if c else lo[n]
        yield vec


def v_box(
    region: ElicitedRegion,
    noisy_prior: np.ndarray,
    interior_samples: int = 2_000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per-entry min and max of ``V`` over the elicited set.

    Returns ``(v_low, v_upp, info)``. Diagonals are set to NaN because Algorithm 2 takes
    only the off-diagonals. ``info`` records the vertex count, how many vertices
    recover a clean prior off the simplex (the D-031 pruning question), and whether any
    sampled interior point fell outside the vertex box.
    """
    prior = np.asarray(noisy_prior, dtype=float)
    prior = prior / prior.sum()
    v_low = np.full((K, K), np.inf)
    v_upp = np.full((K, K), -np.inf)
    n_vertices = off_simplex = 0
    for vec in _vertices(region):
        v, _, rho = _v_and_m(vec, prior)
        n_vertices += 1
        off_simplex += int((rho < 0).any())
        v_low = np.minimum(v_low, v)
        v_upp = np.maximum(v_upp, v)

    lo, hi = region.bounds()
    rng = np.random.default_rng(seed)
    escaped = 0.0
    for _ in range(interior_samples):
        v, _, _ = _v_and_m(lo + rng.random(lo.size) * (hi - lo), prior)
        mask = ~np.eye(K, dtype=bool)
        escaped = max(
            escaped,
            float(np.max((v_low - v)[mask])),
            float(np.max((v - v_upp)[mask])),
        )

    v_low[np.diag_indices(K)] = np.nan
    v_upp[np.diag_indices(K)] = np.nan
    info = {
        "vertices": n_vertices,
        "vertices_off_simplex": off_simplex,
        "max_interior_escape": escaped,
    }
    return v_low, v_upp, info


def worst_case_label_ppv(region: ElicitedRegion, noisy_prior: np.ndarray) -> np.ndarray:
    """Smallest ``M[k, k] = P[Y = k | Ytilde = k]`` over the elicited set's vertices.

    In plain terms: the pessimistic share of posts carrying label k that were written
    by someone who truly has condition k. Vertices whose recovered prior leaves the
    simplex are skipped, since M is not a probability matrix there.
    """
    prior = np.asarray(noisy_prior, dtype=float)
    prior = prior / prior.sum()
    best = np.full(K, np.inf)
    for vec in _vertices(region):
        _, m, rho = _v_and_m(vec, prior)
        if (rho < 0).any():
            continue
        best = np.minimum(best, np.diag(m))
    return best


@dataclass(frozen=True)
class RegionVerdict:
    condition: int
    offdiagonal_mass: float
    delta_ci: float
    alpha: float
    correction_bound: float

    @property
    def margin(self) -> float:
        return self.delta_ci - self.alpha

    def label(self) -> str:
        if self.margin <= 0:
            return "ok"
        return "DEGENERATE (robust)" if self.margin > self.correction_bound else "DEGENERATE (borderline)"


def evaluate_region(
    region: ElicitedRegion,
    noisy_prior: np.ndarray,
    n_per_condition: list[int],
    alpha: float = 0.10,
    c_n_k: list[float] | None = None,
) -> tuple[list[RegionVerdict], dict]:
    """Closed-form degeneracy test (D-042) on the V-box an elicited region induces.

    ``c_n_k`` lets callers pass precomputed ``c(n_k)`` values, since the Monte Carlo
    for the depression count is the slow part.
    """
    v_low, v_upp, info = v_box(region, noisy_prior)
    n_star = min(n_per_condition)
    verdicts = []
    for k, n_k in enumerate(n_per_condition):
        others = [l for l in range(K) if l != k]
        mass = sum(abs(v_upp[k, l]) + (v_upp[k, l] - v_low[k, l]) for l in others)
        delta = correction_term(
            k, v_low, v_upp, n_k, n_star, c_n_k=None if c_n_k is None else c_n_k[k]
        )
        bound = sum(abs(v_upp[k, l]) for l in others)
        verdicts.append(RegionVerdict(k, float(mass), delta, alpha, float(bound)))
    return verdicts, info


def single_entry_breach_points(
    noisy_prior: np.ndarray,
    budgets: list[float],
    step: float = 0.0005,
    ceiling: float = 0.95,
) -> dict[tuple[int, int], tuple[float, list[int]]]:
    """For each off-diagonal ``T[i, j]`` alone, the smallest value that breaks a budget.

    A point matrix with one non-zero off-diagonal, no interval width. Returns
    ``{(i, j): (value, [conditions over budget])}``. These are optimistic limits: a
    real region has many non-zero entries and width, and breaches earlier.
    """
    prior = np.asarray(noisy_prior, dtype=float)
    prior = prior / prior.sum()
    out = {}
    for n, (i, j) in enumerate(OFF_DIAGONAL):
        for x in np.arange(0.0, ceiling, step):
            vec = np.zeros(len(OFF_DIAGONAL))
            vec[n] = x
            v, _, rho = _v_and_m(vec, prior)
            masses = [sum(abs(v[k, l]) for l in range(K) if l != k) for k in range(K)]
            over = [k for k in range(K) if masses[k] > budgets[k]]
            if over or (rho < 0).any():
                out[(i, j)] = (round(float(x), 4), over)
                break
    return out


# --- scenario regions for the feasibility frontier (CONSTRUCTED, not elicited) ------

BIPOLAR, DEPRESSION, EATING_DISORDER, SCHIZOPHRENIA = 0, 1, 2, 3
_RARE = (BIPOLAR, EATING_DISORDER, SCHIZOPHRENIA)


def _scenario(
    leak_bipolar: float,
    leak_other: float,
    rare_to_rare: float,
    dep_to_bipolar: float,
    dep_to_other: float,
    note: str,
) -> ElicitedRegion:
    upper = {
        (BIPOLAR, DEPRESSION): leak_bipolar,
        (EATING_DISORDER, DEPRESSION): leak_other,
        (SCHIZOPHRENIA, DEPRESSION): leak_other,
        (DEPRESSION, BIPOLAR): dep_to_bipolar,
        (DEPRESSION, EATING_DISORDER): dep_to_other,
        (DEPRESSION, SCHIZOPHRENIA): dep_to_other,
    }
    for i in _RARE:
        for j in _RARE:
            if i != j:
                upper[(i, j)] = rare_to_rare
    return ElicitedRegion(upper=upper, note=note)


SCOPING_REGIONS: dict[str, ElicitedRegion] = {
    "F1": _scenario(0.40, 0.40, 0.005, 0.005, 0.005,
                    "fits: leakage into r/depression up to 40%, rare-to-rare and "
                    "depression-to-rare up to 0.5%"),
    "F2": _scenario(0.60, 0.30, 0.005, 0.005, 0.005,
                    "fits: bipolar leakage into r/depression up to 60%"),
    "F3": _scenario(0.40, 0.40, 0.005, 0.010, 0.005,
                    "edge: as F1 but depression-to-bipolar up to 1%"),
    "L1": _scenario(0.40, 0.40, 0.005, 0.024, 0.005,
                    "literature-leaning: as F1 but depression-to-bipolar up to 2.4%, "
                    "worst-case bipolar label PPV about 0.45 (Zimmerman et al. 2010 "
                    "report 63/145 = 0.43 in an outpatient clinic)"),
    "R0": ElicitedRegion(
        upper={
            (BIPOLAR, DEPRESSION): 0.60, (SCHIZOPHRENIA, DEPRESSION): 0.40,
            (EATING_DISORDER, DEPRESSION): 0.40,
            (BIPOLAR, SCHIZOPHRENIA): 0.10, (BIPOLAR, EATING_DISORDER): 0.05,
            (SCHIZOPHRENIA, BIPOLAR): 0.10, (SCHIZOPHRENIA, EATING_DISORDER): 0.02,
            (EATING_DISORDER, BIPOLAR): 0.05, (EATING_DISORDER, SCHIZOPHRENIA): 0.02,
        },
        lower={
            (BIPOLAR, DEPRESSION): 0.05, (SCHIZOPHRENIA, DEPRESSION): 0.02,
            (EATING_DISORDER, DEPRESSION): 0.02,
        },
        note="open item 6's three directions taken wide, rare-to-rare up to 10%, "
             "and NO depression-to-rare inflow at all",
    ),
}


def main() -> None:  # pragma: no cover - reporting
    from src.conformal.c1_matrices import CALIBRATION_COUNTS, CONDITIONS, NOISY_PRIOR
    from src.conformal.v_region import (
        expected_sup_deviation,
        max_admissible_offdiagonal_mass,
    )

    budgets = [b.max_offdiagonal_mass for b in max_admissible_offdiagonal_mass(CALIBRATION_COUNTS)]
    c_n_k = [expected_sup_deviation(n) for n in CALIBRATION_COUNTS]
    short = ["bip", "dep", "ed", "scz"]

    print("Single-entry breach points (point matrix, one non-zero off-diagonal):")
    for (i, j), (x, over) in single_entry_breach_points(NOISY_PRIOR, budgets).items():
        print(f"  T[{short[i]}->{short[j]}]  {x:.4f}  breaks: {[CONDITIONS[k] for k in over]}")

    for name, region in SCOPING_REGIONS.items():
        verdicts, info = evaluate_region(region, NOISY_PRIOR, CALIBRATION_COUNTS, c_n_k=c_n_k)
        ppv = worst_case_label_ppv(region, NOISY_PRIOR)
        print(f"\n{name}: {region.note}")
        print(f"  vertices {info['vertices']}, off simplex {info['vertices_off_simplex']}, "
              f"max interior escape {info['max_interior_escape']:.2e}")
        for v in verdicts:
            print(f"  {CONDITIONS[v.condition]:16s} mass {v.offdiagonal_mass:7.3f} "
                  f"(budget {budgets[v.condition]:.3f})  delta_ci {v.delta_ci:.4f}  "
                  f"margin {v.margin:+.4f}  {v.label()}")
        print("  worst-case label PPV:", np.round(ppv, 3))


if __name__ == "__main__":  # pragma: no cover
    main()
