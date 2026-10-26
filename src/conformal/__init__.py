"""Noise-coupled calibrated abstention layer (C2).

Implements the input side of Sesia, Wang and Tong (2025), 'Adaptive conformal
classification with noisy labels', JRSSB 87(3): the conversion from C1's estimated
transition matrices into the confidence region on ``V = M^-1`` that their Algorithm 2
consumes, and the closed-form test for whether that region is wide enough to make the
algorithm degenerate. See ``Docs/c2-interface.md`` for the derivation and
``DECISIONS.md`` D-027, D-028, D-029, D-031, D-032 for the design decisions.

Light module: numpy only, no torch, no transformers.
"""

from src.conformal.v_region import (
    ConditionBudget,
    DegeneracyVerdict,
    correction_term,
    degeneracy_test,
    max_admissible_offdiagonal_mass,
    region_from_matrices,
    transition_to_v,
)
from src.conformal.split_conformal import (
    SetMetrics,
    evaluate_sets,
    nonconformity_scores,
    predict_sets,
    standard_thresholds,
)
from src.conformal.synthetic import (
    SyntheticConfig,
    SyntheticSample,
    generate,
    make_transition,
    recover_clean_prior,
)

__all__ = [
    "ConditionBudget",
    "DegeneracyVerdict",
    "correction_term",
    "degeneracy_test",
    "max_admissible_offdiagonal_mass",
    "region_from_matrices",
    "transition_to_v",
    "SyntheticConfig",
    "SyntheticSample",
    "generate",
    "make_transition",
    "recover_clean_prior",
    "SetMetrics",
    "evaluate_sets",
    "nonconformity_scores",
    "predict_sets",
    "standard_thresholds",
]
