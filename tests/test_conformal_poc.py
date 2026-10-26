"""Tests for the C2 proof-of-concept harness (roadmap steps 5 and 6). Torch-free.

The step 6 tests pin the numbers quoted in ``Docs/progress-and-roadmap.md`` §5 step 6,
produced by ``python -m src.conformal.poc --step 6`` with the configuration fixed in
commit 951816d before the first run. They run the full 100-repetition harness once per
session (about ten seconds).
"""

import numpy as np
import pytest

from src.conformal.poc import draw_calibration_and_test, run_step6

K = 4
BIPOLAR, DEPRESSION, EATING_DISORDER, SCHIZOPHRENIA = range(K)


def test_fixed_count_calibration_has_exactly_the_requested_counts():
    counts = [50, 500, 120, 80]
    draw = draw_calibration_and_test(np.eye(K), rep=0, counts=counts, n_test=100)
    np.testing.assert_array_equal(np.bincount(draw.calibration.y_tilde, minlength=K), counts)


def test_calibration_and_test_draws_are_independent_streams():
    draw = draw_calibration_and_test(np.eye(K), rep=0, counts=[20, 200, 50, 30], n_test=500)
    assert not np.array_equal(draw.calibration.probs[:50], draw.test.probs[:50])
    again = draw_calibration_and_test(np.eye(K), rep=0, counts=[20, 200, 50, 30], n_test=500)
    np.testing.assert_array_equal(draw.test.probs, again.test.probs)


@pytest.fixture(scope="module")
def step6():
    result = run_step6()
    return {arm.name: arm for arm in result.arms}, result


# Coverage against the clean label, mean over 100 repetitions, CONDITIONS order.
PINNED_COVERAGE = {
    "arm (a)": [0.9005, 0.9059, 0.9004, 0.9016],
    "arm (b) oracle, conservative": [0.9269, 0.9075, 0.9171, 0.9217],
    "arm (b) oracle, optimistic": [0.9005, 0.9059, 0.9004, 0.9016],
    "arm (b) wrong point, conservative": [0.9269, 0.9106, 0.9171, 0.9217],
    "arm (b) wrong point, optimistic": [0.9005, 0.9059, 0.9004, 0.9016],
}
PINNED_SIZE = {
    "arm (a)": [1.250, 1.250, 1.251, 1.252],
    "arm (b) oracle, conservative": [1.325, 1.333, 1.331, 1.329],
    "arm (b) wrong point, conservative": [1.329, 1.336, 1.335, 1.332],
}


@pytest.mark.parametrize("name", list(PINNED_COVERAGE))
def test_step6_coverage_is_pinned(step6, name):
    arms, _ = step6
    np.testing.assert_allclose(arms[name].coverage, PINNED_COVERAGE[name], atol=5e-5)


@pytest.mark.parametrize("name", list(PINNED_SIZE))
def test_step6_mean_set_size_is_pinned(step6, name):
    arms, _ = step6
    np.testing.assert_allclose(arms[name].mean_size, PINNED_SIZE[name], atol=5e-4)


def test_step6_oracle_coverage_holds(step6):
    """Sub-run 1: with the true V, both variants cover at the nominal level or above."""
    arms, _ = step6
    for variant in ("conservative", "optimistic"):
        arm = arms[f"arm (b) oracle, {variant}"]
        assert (arm.coverage >= 0.90 - 3 * arm.coverage_se).all()


def test_step6_wrong_point_does_not_break_coverage(step6):
    """Sub-run 2, as expected before the run (951816d) and against the roadmap's
    "coverage should break": with no inflow into bipolar, under-estimating the noise
    leaves every condition covered, and bipolar's coverage is identical to the oracle's."""
    arms, _ = step6
    for variant in ("conservative", "optimistic"):
        wrong = arms[f"arm (b) wrong point, {variant}"]
        oracle = arms[f"arm (b) oracle, {variant}"]
        assert (wrong.coverage >= 0.90 - 3 * wrong.coverage_se).all()
        assert wrong.coverage[BIPOLAR] == oracle.coverage[BIPOLAR]
    # Under-estimating the noise corrects less, so the conservative variant covers
    # depression more, not less (Corollary 1's regime).
    assert (
        arms["arm (b) wrong point, conservative"].coverage[DEPRESSION]
        > arms["arm (b) oracle, conservative"].coverage[DEPRESSION]
    )


def test_step6_optimistic_variant_is_exactly_arm_a(step6):
    """In this configuration Delta_hat never exceeds delta_ci, so eq. (25) always picks
    -(1 - alpha) / n_k and returns standard conformal's threshold on every repetition."""
    arms, _ = step6
    for name in ("oracle", "wrong point"):
        np.testing.assert_array_equal(arms[f"arm (b) {name}, optimistic"].tau, arms["arm (a)"].tau)


def test_step6_conservative_variant_is_above_arm_a_on_every_repetition(step6):
    arms, _ = step6
    for name in ("oracle", "wrong point"):
        assert (arms[f"arm (b) {name}, conservative"].tau > arms["arm (a)"].tau).all()


def test_step6_proposition_three_fails_in_both_forms_without_under_coverage(step6):
    """D-049's diagnostic, measured without a prediction: the condition fails on every
    repetition in both forms, and the optimistic variant does not under-cover."""
    arms, result = step6
    for summary in result.proposition_three:
        np.testing.assert_array_equal(summary.holds_as_stated, np.zeros(K))
        np.testing.assert_array_equal(summary.holds_at_threshold, np.zeros(K))
        assert (summary.inf_delta <= 0).all()
        optimistic = arms[f"arm (b) {summary.name}, optimistic"]
        assert (optimistic.coverage >= 0.90 - 3 * optimistic.coverage_se).all()
