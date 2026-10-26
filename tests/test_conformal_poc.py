"""Tests for the C2 proof-of-concept harness (roadmap steps 5 and 6). Torch-free."""

import numpy as np

from src.conformal.poc import draw_calibration_and_test

K = 4


def test_fixed_count_calibration_has_exactly_the_requested_counts():
    counts = [50, 500, 120, 80]
    draw = draw_calibration_and_test(np.eye(K), rep=0, counts=counts, n_test=100)
    np.testing.assert_array_equal(np.bincount(draw.calibration.y_tilde, minlength=K), counts)


def test_calibration_and_test_draws_are_independent_streams():
    draw = draw_calibration_and_test(np.eye(K), rep=0, counts=[20, 200, 50, 30], n_test=500)
    assert not np.array_equal(draw.calibration.probs[:50], draw.test.probs[:50])
    again = draw_calibration_and_test(np.eye(K), rep=0, counts=[20, 200, 50, 30], n_test=500)
    np.testing.assert_array_equal(draw.test.probs, again.test.probs)
