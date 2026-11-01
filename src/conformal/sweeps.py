"""Roadmap §5 step 9: the width sweep and D-032's noise-heterogeneity sweep.

**Exploratory.** D-043 pre-registered the alpha sweep (run in ``arm_c``). Neither sweep
here has pre-registered numbers; both designs were fixed in this docstring before
either sweep was run, and their results are reported as descriptive.

Width sweep
-----------
Truth ``T_HOC_FINE_TUNED`` (region A's truth), harness as in D-043 §1, primary zeta.
For each width w on ``WIDTHS``, the box puts the true V at the upper end of every
off-diagonal interval: ``V_low = V - w``, ``V_upp = V`` (every true off-diagonal is
negative, so ``V_upp`` stays non-positive and ``P_k = 0``). The measured degeneracy
point ``w*_k`` is the smallest w at which tau_k = 1 in at least 50 of 100 runs. It is
compared with three closed-form widths, per condition:

- D-042's frontier: the w at which ``delta_ci(w) = alpha``;
- the largest w that is certain-ok (``alpha - delta_ci >= 6 w``, zeta = w);
- the smallest w that is certain-degenerate (``delta_ci - alpha > D_max``).

Heterogeneity sweep (D-032)
---------------------------
How unequal the true noise rates are, against whether arm (c) overtakes arm (b).
Truth ``T_h``: each rare condition sends a share of its posts to depression,
``T[bipolar, depression] = 0.10 (1 + h)``, ``T[eating disorder, depression] = 0.10 (1 - h)``,
``T[schizophrenia, depression] = 0.10 (1 + h / 2)``, and depression sends 0.2% to each
rare label (a small fixed inflow). ``h`` runs over ``HETEROGENEITY``. Arm (b) is given
the homogeneous point ``T_0``, the estimate that ignores heterogeneity. Arm (c) is
given the deterministic-containment box over ``{T_0, T_h}``, so it contains the truth
and widens as h grows. Both conservative, primary zeta, scored on bipolar and
schizophrenia with D-043 §4(c)'s rules (WIN, WIN BY ABSTENTION, LOSS, TIE).
"""

from __future__ import annotations

import numpy as np

from src.conformal.arm_c import (
    ALPHA,
    K,
    SHORT,
    _replay,
    calibrations,
    draws_for,
)
from src.conformal.c1_matrices import CALIBRATION_COUNTS, NOISY_PRIOR, T_HOC_FINE_TUNED
from src.conformal.poc import c_for_counts, run_arm
from src.conformal.v_region import correction_term, region_from_matrices, transition_to_v

WIDTHS = (0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5)
HETEROGENEITY = (0.0, 0.25, 0.5, 0.75, 0.9)


def width_box(v: np.ndarray, w: float) -> tuple[np.ndarray, np.ndarray]:
    off = 1 - np.eye(K)
    return v - w * off, v.copy()


def closed_form_widths(v: np.ndarray, k: int, alpha: float = ALPHA) -> dict:
    """D-042 frontier, largest certain-ok and smallest certain-degenerate widths."""
    c = c_for_counts(tuple(CALIBRATION_COUNTS))
    n_star = min(CALIBRATION_COUNTS)
    grid = np.linspace(0, 0.5, 50_001)
    d042 = ok = deg = None
    for w in grid:
        low, upp = width_box(v, w)
        delta = correction_term(k, low, upp, CALIBRATION_COUNTS[k], n_star, c_n_k=c[CALIBRATION_COUNTS[k]])
        d_max = sum(abs(upp[k, l]) for l in range(K) if l != k)
        if d042 is None and delta > alpha:
            d042 = w
        if alpha - delta >= 2 * (K - 1) * w:
            ok = w
        if deg is None and delta - alpha > d_max:
            deg = w
    return {"d042_frontier": d042, "certain_ok_max": ok, "certain_degenerate_min": deg}


def width_sweep(reps: int = 100) -> dict:
    v = transition_to_v(T_HOC_FINE_TUNED, NOISY_PRIOR)[0]
    draws = draws_for(T_HOC_FINE_TUNED, reps=reps)
    degenerate = np.zeros((len(WIDTHS), K), dtype=int)
    for i, w in enumerate(WIDTHS):
        low, upp = width_box(v, w)
        taus = np.array([c.tau for c in calibrations(draws, low, upp, "conservative", None)])
        degenerate[i] = (taus >= 1.0).sum(axis=0)
    measured = [
        next((WIDTHS[i] for i in range(len(WIDTHS)) if degenerate[i, k] >= 50), None)
        for k in range(K)
    ]
    closed = [closed_form_widths(v, k) for k in range(K)]
    return {"degenerate": degenerate, "measured": measured, "closed": closed}


def heterogeneous_truth(h: float) -> np.ndarray:
    t = np.zeros((K, K))
    t[0, 1] = 0.10 * (1 + h)
    t[2, 1] = 0.10 * (1 - h)
    t[3, 1] = 0.10 * (1 + h / 2)
    t[1, [0, 2, 3]] = 0.002
    t[np.diag_indices(K)] = 1 - t.sum(axis=1)
    return t


def heterogeneity_sweep(reps: int = 100) -> list[dict]:
    t0 = heterogeneous_truth(0.0)
    v0 = transition_to_v(t0, NOISY_PRIOR)[0]
    rows = []
    for h in HETEROGENEITY:
        th = heterogeneous_truth(h)
        draws = draws_for(th, reps=reps)
        low_c, upp_c = region_from_matrices([t0, th], NOISY_PRIOR)
        arms = {}
        for label, (low, upp) in {"b": (v0.copy(), v0.copy()), "c": (low_c, upp_c)}.items():
            taus = np.array([c.tau for c in calibrations(draws, low, upp, "conservative", None)])
            arms[label] = (run_arm(label, _replay(taus), draws), (taus >= 1.0).sum(axis=0))
        row = {"h": h}
        for k in (0, 3):
            b, b_deg = arms["b"]
            c, c_deg = arms["c"]
            b_valid = b.coverage[k] >= 0.90 - 3 * b.coverage_se[k]
            c_valid = c.coverage[k] >= 0.90 - 3 * c.coverage_se[k]
            c_dg = c_deg[k] >= 50
            if c_valid and not c_dg and (not b_valid or c.mean_size[k] <= b.mean_size[k] - 0.05):
                verdict = "WIN"
            elif c_valid and c_dg and not b_valid:
                verdict = "WIN BY ABSTENTION"
            elif b_valid and (not c_valid or c.mean_size[k] >= b.mean_size[k] + 0.05):
                verdict = "LOSS"
            else:
                verdict = "TIE"
            row[SHORT[k]] = {
                "verdict": verdict,
                "b": (float(b.coverage[k]), float(b.coverage_se[k]), float(b.mean_size[k]), int(b_deg[k])),
                "c": (float(c.coverage[k]), float(c.coverage_se[k]), float(c.mean_size[k]), int(c_deg[k])),
            }
        rows.append(row)
    return rows


def main() -> int:  # pragma: no cover - reporting
    ws = width_sweep()
    print("=== Width sweep (exploratory), truth T_HOC_FINE_TUNED, conservative ===")
    print("degenerate runs per width:")
    for i, w in enumerate(WIDTHS):
        print(f"  w = {w:<6} {ws['degenerate'][i].tolist()}")
    for k in range(K):
        c = ws["closed"][k]
        print(f"{SHORT[k]:16} measured w* {ws['measured'][k]!s:6}  D-042 frontier {c['d042_frontier']}"
              f"  certain-ok up to {c['certain_ok_max']}  certain-degenerate from {c['certain_degenerate_min']}")
    print("\n=== Heterogeneity sweep (exploratory, D-032), conservative ===")
    for row in heterogeneity_sweep():
        for name in ("bipolar", "schizophrenia"):
            r = row[name]
            b, c = r["b"], r["c"]
            print(f"h = {row['h']:<5} {name:14} {r['verdict']:18} b cov {b[0]:.4f}±{b[1]:.4f} size {b[2]:.3f} deg {b[3]:3}"
                  f" | c cov {c[0]:.4f}±{c[1]:.4f} size {c[2]:.3f} deg {c[3]:3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
