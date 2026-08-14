"""Reproduce the tables in ``Docs/c2-interface.md`` section 6.

Usage::

    python -m src.conformal
    python -m src.conformal --alpha 0.05
"""

from __future__ import annotations

import argparse

import numpy as np

from src.conformal.c1_matrices import (
    CALIBRATION_COUNTS,
    CONDITIONS,
    NOISY_PRIOR,
    REGION_A_MEASURED,
    REGION_B_MEASURED,
    T_CLEANLAB,
    T_HOC_BASE,
    T_HOC_FINE_TUNED,
    T_HOC_MPNET,
)
from src.conformal.v_region import (
    degeneracy_test,
    max_admissible_offdiagonal_mass,
    region_from_matrices,
    transition_to_v,
)

REGIONS = {
    "A  HOC fine-tuned + cleanlab": REGION_A_MEASURED,
    "B  all four C1 estimates": REGION_B_MEASURED,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument(
        "--alpha-v",
        type=float,
        default=0.0,
        help="0 for deterministic containment, D-031 option (a)",
    )
    parser.add_argument(
        "--calibration-scale",
        type=float,
        default=1.0,
        help="multiply the C0 test-split counts, e.g. 2.0 for a 20%% split",
    )
    args = parser.parse_args()

    np.set_printoptions(precision=4, suppress=True)
    counts = [max(1, int(n * args.calibration_scale)) for n in CALIBRATION_COUNTS]

    print("Sanity check: the clean prior is recovered, not assumed.")
    estimates = [
        ("HOC fine-tuned", T_HOC_FINE_TUNED),
        ("HOC mpnet", T_HOC_MPNET),
        ("HOC base", T_HOC_BASE),
        ("cleanlab", T_CLEANLAB),
    ]
    for name, t in estimates:
        _, m, prior = transition_to_v(t, NOISY_PRIOR)
        on_simplex = bool((prior > 0).all() and abs(prior.sum() - 1) < 1e-9)
        print(f"  {name:<16} rho = {prior}  on simplex: {on_simplex}")
        assert np.allclose(m.sum(axis=1), 1.0), "M rows must sum to 1"

    for label, transitions in REGIONS.items():
        v_low, v_upp = region_from_matrices(transitions, NOISY_PRIOR)
        print(f"\n{'=' * 76}\nRegion {label}")
        print(f"  bipolar row  V_low = {v_low[0]}")
        print(f"               V_upp = {v_upp[0]}")
        print(
            f"\n  {'condition':<16} {'delta_ci':>9} {'margin':>9} "
            f"{'Dhat_max':>9}  verdict"
        )
        verdicts = degeneracy_test(
            v_low, v_upp, counts, alpha=args.alpha, alpha_v=args.alpha_v
        )
        for verdict in verdicts:
            print(
                f"  {CONDITIONS[verdict.condition]:<16} {verdict.delta_ci:>9.4f} "
                f"{verdict.margin:>+9.4f} {verdict.correction_bound:>9.4f}  "
                f"{verdict.label()}"
            )

    print(f"\n{'=' * 76}\nFeasibility frontier at alpha = {args.alpha}")
    print("  largest admissible sum_(l!=k) (|V_upp| + width) before degeneracy:")
    for budget in max_admissible_offdiagonal_mass(counts, alpha=args.alpha):
        print(
            f"  {CONDITIONS[budget.condition]:<16} "
            f"{budget.max_offdiagonal_mass:.4f}"
        )


if __name__ == "__main__":
    main()
