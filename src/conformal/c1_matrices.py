"""C1's estimated transition matrices, transcribed from DECISIONS.md.

Every matrix here is ``T[l, k] = P[Ytilde = k | Y = l]`` with rows in
``CONDITIONS`` order and summing to one.

Three of the four are transcribed from measured run artifacts. The fourth (the base
MentalBERT arm of D-041) is **constructed** from its published diagonal and is marked
as such, because ``Models/embeddings/train__base/`` holds only ``features.npy`` and
``metadata.csv``: that arm's HOC output has not been brought down from Drive. Pulling
it is the remaining half of step 2 of the C2 proof-of-concept.

Provenance, all under ``Models/embeddings/`` (gitignored, so the values are transcribed
here rather than loaded):

===================  ==========================  =================================
arm                  source                      status
===================  ==========================  =================================
HOC fine-tuned       ``train/hoc_mean_T_rerun``  measured; the run D-037 reports
HOC mpnet            ``train__mpnet/hoc_mean_T`` measured; D-041 Arm B
HOC base             diagonal only, D-041        CONSTRUCTED, arm A output not local
cleanlab             D-039                       measured
===================  ==========================  =================================

Note on the fine-tuned arm: ``train/`` holds both ``hoc_mean_T.csv`` (bipolar diagonal
0.95393) and ``hoc_mean_T_rerun.csv`` (0.95339). D-037 quotes 0.9534 and per-seed
values matching ``hoc_per_seed.csv``, whose mean is 0.95339, so the rerun is the run of
record and is the one used here.
"""

from __future__ import annotations

import numpy as np

CONDITIONS = ["bipolar", "depression", "eating_disorder", "schizophrenia"]

NOISY_PRIOR = np.array([0.038, 0.812, 0.096, 0.054])
"""Observed proxy-label frequencies, D-037 ('bipolar 0.038, depression 0.812')."""

T_HOC_FINE_TUNED = np.array(
    [
        [0.9533942960, 0.0263194664, 0.0035295785, 0.0167566591],
        [0.0024830883, 0.9963807495, 0.0002981096, 0.0008380526],
        [0.0008415658, 0.0038677609, 0.9945811054, 0.0007095678],
        [0.0133223802, 0.0073718715, 0.0012478386, 0.9780579096],
    ]
)
"""D-037, mean over 5 seeds, HOC on fine-tuned MentalBERT embeddings. MEASURED.

Source: ``Models/embeddings/train/hoc_mean_T_rerun.csv``.
"""

T_HOC_MPNET = np.array(
    [
        [0.7070852129, 0.1760278135, 0.0262074374, 0.0906795361],
        [0.0120076226, 0.9729338478, 0.0058996388, 0.0091588908],
        [0.0092296435, 0.0588094787, 0.9209626288, 0.0109982489],
        [0.0619233630, 0.0954782081, 0.0237533392, 0.8188450896],
    ]
)
"""D-041 Arm B, HOC on ``all-mpnet-base-v2`` embeddings. MEASURED.

Source: ``Models/embeddings/train__mpnet/hoc_mean_T.csv``. Confirms the clinically
predicted direction: bipolar's dominant off-diagonal is depression (0.176), and
schizophrenia's is depression (0.095).
"""

T_CLEANLAB = np.array(
    [
        [0.8483, 0.1027, 0.0062, 0.0429],
        [0.0093, 0.9790, 0.0041, 0.0076],
        [0.0017, 0.0277, 0.9685, 0.0022],
        [0.0326, 0.0638, 0.0048, 0.8988],
    ]
)
"""D-039, cleanlab on author-grouped out-of-sample probabilities. MEASURED."""

T_HOC_BASE = np.array(
    [
        [0.5409376263, 0.2509011967, 0.0582794067, 0.1498817703],
        [0.0160702369, 0.9576853354, 0.0106762139, 0.0155682139],
        [0.0258352693, 0.1180396763, 0.8265194632, 0.0296055912],
        [0.1015493228, 0.1865242474, 0.0671284704, 0.6447979595],
    ]
)
"""D-041 Arm A, HOC on base MentalBERT embeddings. MEASURED.

Source: ``Models/embeddings/train__base/hoc_mean_T.csv``, brought down from Drive on
2026-08-14. No ``hoc_mean_T_rerun.csv`` accompanies it, so there is no ambiguity about
which run this is. Cross-checks against D-041: bipolar diagonal 0.5409, cross-seed
standard deviation 0.0084, mean |det| 0.2595, branch DEGENERATE.

On invertibility, which Sesia's Theorem 4 requires: ``hoc_d040_verdict.csv`` records
``nonsingular_all_seeds=False``, but that field reports failure of D-040's
pre-registered *quality* bar of |det| >= 0.5, not singularity. Every seed in
``hoc_per_seed.csv`` has ``nonsingular=True`` with |det| between 0.247 and 0.267, so
the matrix inverts and the conversion below is valid.
"""

DIAGONAL_HOC_BASE = np.diag(T_HOC_BASE)
"""Retained for the D-041 cross-check; superseded as an input by :data:`T_HOC_BASE`."""


def build_from_diagonal(
    diagonal: np.ndarray, shape_source: np.ndarray = T_CLEANLAB
) -> np.ndarray:
    """Distribute off-diagonal mass in ``shape_source``'s proportions.

    CONSTRUCTED, NOT MEASURED. Any number derived from the result carries that
    caveat and must state it wherever it is quoted.
    """
    n = len(diagonal)
    out = np.zeros((n, n))
    for row in range(n):
        others = [k for k in range(n) if k != row]
        mass = shape_source[row, others].astype(float)
        mass = mass / mass.sum() * (1.0 - diagonal[row])
        out[row, row] = diagonal[row]
        for position, col in enumerate(others):
            out[row, col] = mass[position]
    return out


REGION_A_MEASURED = [T_HOC_FINE_TUNED, T_CLEANLAB]
"""The two full matrices on record. Every entry measured."""

REGION_B_MEASURED = [
    T_HOC_FINE_TUNED,
    T_CLEANLAB,
    T_HOC_MPNET,
    T_HOC_BASE,
]
"""All four C1 estimates, every one measured. The full C1 region."""

REGION_B_APPROXIMATE = REGION_B_MEASURED
"""Deprecated alias kept so earlier references resolve. No longer approximate."""

CALIBRATION_COUNTS = [481, 11351, 1421, 786]
"""Per-condition counts in the C0 held-out test split, CONDITIONS order.

Note the ordering: baseline-results.md reports these by condition name, and bipolar
(481) is the minimum, so it sets ``n_star`` for every condition.
"""
