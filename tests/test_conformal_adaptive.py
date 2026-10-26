"""Tests for the port of Sesia et al.'s Algorithm 2 (roadmap step 6). Torch-free.

The two step 4 sanity-gate tests, as corrected by D-049:

- ``test_gate_optimistic_equals_arm_a_exactly_at_identity``
- ``test_gate_conservative_differs_by_exactly_the_c_shift_at_identity``

``test_zero_width_box_reduces_to_equation_thirteen`` checks D-049's claim that
Algorithm 2 with a point estimate is Algorithm 1 with that estimate, against an
independent implementation of equations (13) to (15) written in eq. (13)'s own form.
"""

import math

import numpy as np
import pytest

from src.conformal.adaptive import (
    InflationReference,
    adaptive_thresholds,
    coverage_inflation,
    proposition_three_check,
    zero_width_box,
)
from src.conformal.poc import draw_calibration_and_test
from src.conformal.split_conformal import nonconformity_scores, standard_thresholds
from src.conformal.synthetic import make_transition
from src.conformal.v_region import expected_sup_deviation, transition_to_v

K = 4
ALPHA = 0.10
COUNTS = [150, 1500, 300, 220]
C_N_K = {n: expected_sup_deviation(n) for n in COUNTS}


def _calibration(transition, rep=0, separability=2.0):
    draw = draw_calibration_and_test(
        transition, rep, counts=COUNTS, n_test=100, separability=separability
    )
    return nonconformity_scores(draw.calibration.probs), draw.calibration.y_tilde


@pytest.mark.parametrize("rep", range(5))
def test_gate_optimistic_equals_arm_a_exactly_at_identity(rep):
    scores, y_tilde = _calibration(np.eye(K), rep)
    low, upp = zero_width_box(np.eye(K))
    optimistic = adaptive_thresholds(
        scores, y_tilde, low, upp, ALPHA, variant="optimistic", c_n_k=C_N_K
    )
    np.testing.assert_array_equal(optimistic.tau, standard_thresholds(scores, y_tilde, ALPHA))


@pytest.mark.parametrize("rep", range(5))
def test_gate_conservative_differs_by_exactly_the_c_shift_at_identity(rep):
    scores, y_tilde = _calibration(np.eye(K), rep)
    low, upp = zero_width_box(np.eye(K))
    conservative = adaptive_thresholds(
        scores, y_tilde, low, upp, ALPHA, variant="conservative", c_n_k=C_N_K
    )
    for k, n_k in enumerate(COUNTS):
        expected_rank = math.ceil(n_k * (1 - ALPHA + C_N_K[n_k]) - 1e-9)
        assert conservative.index[k] == expected_rank
        assert conservative.delta_ci[k] == pytest.approx(C_N_K[n_k])
        s_sorted = np.sort(scores[y_tilde == k, k])
        assert conservative.tau[k] == s_sorted[expected_rank - 1]
        standard = math.ceil((n_k + 1) * (1 - ALPHA) - 1e-9)
        assert expected_rank > standard  # the shift is real, not zero


def _algorithm_one_reference(scores, y_tilde, v, alpha, c_n_k):
    """Equations (13) to (17), written independently in eq. (13)'s form."""
    n_k = np.array([(y_tilde == k).sum() for k in range(K)])
    n_star = n_k.min()
    tau = np.ones(K)
    for k in range(K):
        s = np.sort(scores[y_tilde == k, k])
        f = [np.array([(scores[y_tilde == l, k] <= t).mean() for t in s]) for l in range(K)]
        delta_hat = (v[k, k] - 1) * f[k] + sum(v[k, l] * f[l] for l in range(K) if l != k)
        mult = min(K * math.sqrt(math.pi / 2),
                   1 / math.sqrt(n_star) + math.sqrt((math.log(2 * K) + math.log(n_star)) / 2))
        delta = c_n_k[n_k[k]] + 2 * sum(abs(v[k, l]) for l in range(K) if l != k) / math.sqrt(n_star) * mult
        i = np.arange(1, n_k[k] + 1)
        members = np.flatnonzero(i / n_k[k] + 1e-9 >= 1 - alpha - delta_hat + delta)
        if members.size:
            tau[k] = s[members[0]]
    return tau


def test_zero_width_box_reduces_to_equation_thirteen():
    t = make_transition(0.15, 0.01)
    scores, y_tilde = _calibration(t, rep=3)
    v = transition_to_v(t, t.T @ np.array([0.038, 0.812, 0.096, 0.054]))[0]
    low, upp = zero_width_box(v)
    ported = adaptive_thresholds(scores, y_tilde, low, upp, ALPHA, c_n_k=C_N_K)
    np.testing.assert_allclose(ported.tau, _algorithm_one_reference(scores, y_tilde, v, ALPHA, C_N_K))


def test_optimistic_threshold_never_exceeds_standard_conformal():
    """D-049: eq. (25) caps the threshold at standard conformal's, for any box."""
    t = make_transition(0.30, 0.02)
    scores, y_tilde = _calibration(t, rep=1)
    v = transition_to_v(t, t.T @ np.array([0.038, 0.812, 0.096, 0.054]))[0]
    for widen in (0.0, 0.05, 0.5, 2.0):
        low, upp = v - widen, v.copy()
        optimistic = adaptive_thresholds(
            scores, y_tilde, low, upp, ALPHA, variant="optimistic", c_n_k=C_N_K
        )
        assert (optimistic.tau <= standard_thresholds(scores, y_tilde, ALPHA)).all()
        assert (optimistic.index > 0).all()  # never degenerates (n_k >= 9)


def test_conservative_degenerates_when_the_closed_form_says_so():
    """A box wide enough that delta_ci > alpha sets tau = 1, as D-042 derives."""
    scores, y_tilde = _calibration(np.eye(K), rep=2)
    low = np.eye(K) - 3.0 * (1 - np.eye(K))  # off-diagonal mass 9 per row
    upp = np.eye(K)
    conservative = adaptive_thresholds(scores, y_tilde, low, upp, ALPHA, c_n_k=C_N_K)
    assert (conservative.delta_ci > ALPHA).all()
    np.testing.assert_array_equal(conservative.tau, np.ones(K))
    np.testing.assert_array_equal(conservative.index, np.zeros(K, dtype=int))


def test_inputs_are_validated():
    scores, y_tilde = _calibration(np.eye(K))
    low, upp = zero_width_box(np.eye(K))
    with pytest.raises(ValueError, match="variant"):
        adaptive_thresholds(scores, y_tilde, low, upp, variant="greedy")
    with pytest.raises(ValueError, match="V_upp"):
        adaptive_thresholds(scores, y_tilde, upp + 0.1 * (1 - np.eye(K)), low)


def test_coverage_inflation_is_zero_without_noise_and_outside_the_score_range():
    rng = np.random.default_rng(0)
    scores_k = rng.random(10_000)
    y = rng.integers(0, K, 10_000)
    t = np.array([-1.0, 0.25, 0.5, 2.0])
    np.testing.assert_array_equal(coverage_inflation(scores_k, y, y, 0, t), np.zeros(4))
    y_tilde = rng.integers(0, K, 10_000)
    edges = coverage_inflation(scores_k, y, y_tilde, 0, np.array([-1.0, 2.0]))
    np.testing.assert_array_equal(edges, [0.0, 0.0])


def test_proposition_three_as_stated_cannot_hold_when_delta_exceeds_the_slack():
    """D-049: inf_t Delta_k(t) <= 0, so the printed condition needs delta_ci <= (1 - alpha) / n_k."""
    t = make_transition(0.15, 0.0)
    draw = draw_calibration_and_test(t, 0, counts=COUNTS, n_test=200_000)
    scores = nonconformity_scores(draw.calibration.probs)
    v = transition_to_v(t, t.T @ np.array([0.038, 0.812, 0.096, 0.054]))[0]
    low, upp = zero_width_box(v)
    optimistic = adaptive_thresholds(
        scores, draw.calibration.y_tilde, low, upp, ALPHA, variant="optimistic", c_n_k=C_N_K
    )
    reference = InflationReference(
        nonconformity_scores(draw.test.probs), draw.test.y, draw.test.y_tilde
    )
    checks = proposition_three_check(reference, optimistic, ALPHA)
    for check in checks:
        assert check.inf_delta <= 0.0
        assert check.bound > 0.0
        assert not check.holds_as_stated
