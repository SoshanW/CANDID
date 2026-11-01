"""Tests for arm (a), noise-blind label-conditional split conformal (roadmap step 5).

Torch-free. ``test_gate_identity_transition_gives_nominal_coverage`` is the step 5
gate: with T = I, per-condition coverage against the clean label is nominal within a
stated tolerance.
"""

import numpy as np
import pytest

from src.conformal.poc import draw_calibration_and_test, run_arm
from src.conformal.split_conformal import (
    evaluate_sets,
    nonconformity_scores,
    predict_sets,
    standard_rank,
    standard_thresholds,
)

K = 4


def test_scores_follow_equation_two_for_equation_three():
    probs = np.array([[0.7, 0.1, 0.1, 0.1]])
    np.testing.assert_allclose(nonconformity_scores(probs), [[0.3, 0.9, 0.9, 0.9]])


@pytest.mark.parametrize("n_k, expected", [(9, 9), (10, 10), (99, 90), (481, 434), (1, 2)])
def test_standard_rank_is_equation_four(n_k, expected):
    assert standard_rank(n_k, 0.10) == expected


def test_thresholds_pick_the_rank_within_each_proxy_label():
    # Condition 0: 9 calibration points with scores 0.1..0.9 -> rank 9 -> 0.9.
    # Condition 1: a single point -> rank 2 exceeds n_k -> tau = 1.
    scores = np.full((10, 2), 0.5)
    scores[:9, 0] = np.arange(1, 10) / 10
    y_tilde = np.array([0] * 9 + [1])
    tau = standard_thresholds(scores, y_tilde, alpha=0.10)
    assert tau[0] == pytest.approx(0.9)
    assert tau[1] == 1.0


def test_thresholds_reject_a_condition_with_no_calibration_points():
    with pytest.raises(ValueError, match="condition 1"):
        standard_thresholds(np.zeros((3, 2)), np.array([0, 0, 0]))


def test_prediction_sets_follow_equation_three_and_definition_one():
    scores = np.array([[0.2, 0.95, 0.8], [0.9, 0.9, 0.9]])
    tau = np.array([0.5, 1.0, 0.85])
    sets = predict_sets(scores, tau)
    # tau = 1 puts condition 1 in every set; the second point clears nothing else.
    np.testing.assert_array_equal(sets, [[True, True, True], [False, True, False]])
    assert not predict_sets(scores, np.array([0.1, 0.1, 0.1])).any(axis=1).any()


def test_evaluate_sets_conditions_on_the_clean_label():
    sets = np.array(
        [[True, False], [True, True], [False, False], [False, True]], dtype=bool
    )
    y = np.array([0, 0, 1, 1])
    m = evaluate_sets(sets, y)
    np.testing.assert_allclose(m.coverage, [1.0, 0.5])
    np.testing.assert_allclose(m.mean_size, [1.5, 0.5])
    np.testing.assert_allclose(m.empty_rate, [0.0, 0.5])
    np.testing.assert_allclose(m.full_rate, [0.5, 0.0])
    np.testing.assert_array_equal(m.n, [2, 2])


def test_gate_identity_transition_gives_nominal_coverage():
    """Step 5 gate. With T = I, conformal theory puts mean coverage of each condition in
    [1 - alpha, 1 - alpha + 1 / (n_k + 1)]. Tolerance: three standard errors of the
    mean over repetitions on either side of that interval."""
    alpha, counts, reps = 0.10, [200, 2000, 400, 300], 200
    draws = [
        draw_calibration_and_test(np.eye(K), rep, counts=counts, n_test=5_000)
        for rep in range(reps)
    ]
    summary = run_arm("a", lambda s, y: standard_thresholds(s, y, alpha), draws)
    lower = 1 - alpha - 3 * summary.coverage_se
    upper = 1 - alpha + 1 / (np.array(counts) + 1) + 3 * summary.coverage_se
    assert ((summary.coverage >= lower) & (summary.coverage <= upper)).all(), (
        summary.coverage,
        summary.coverage_se,
    )
    # The tolerance is tight enough to mean something: every SE is under 0.005.
    assert (summary.coverage_se < 0.005).all()
