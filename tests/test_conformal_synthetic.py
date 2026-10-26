"""Tests for the synthetic C2 problem with a known transition matrix (roadmap §5 step 4).

Torch-free. Two tests carry weight beyond software checking:

- ``test_identity_transition_leaves_labels_unchanged`` is the step 4 sanity gate on
  the generator side: with ``T = I`` the proxy label is the clean label.
- ``test_recovered_clean_prior_matches_truth_within_sampling_error`` is a numerical
  check of D-028's conversion ``rho = (T^T)^-1 rho_tilde``, recorded in DECISIONS.md
  as evidence bearing on open item 3. It does not close that item.
"""

import numpy as np
import pytest

from src.conformal.c1_matrices import NOISY_PRIOR
from src.conformal.synthetic import (
    BIPOLAR,
    DEPRESSION,
    K,
    SyntheticConfig,
    generate,
    make_transition,
    recover_clean_prior,
)


def test_make_transition_has_bipolar_to_depression_dominant():
    t = make_transition(bipolar_to_depression=0.15, background=0.01)
    np.testing.assert_allclose(t.sum(axis=1), 1.0)
    off = t.copy()
    np.fill_diagonal(off, 0.0)
    assert np.unravel_index(off.argmax(), off.shape) == (BIPOLAR, DEPRESSION)
    assert t[BIPOLAR, DEPRESSION] == pytest.approx(0.15)
    assert t[BIPOLAR, BIPOLAR] == pytest.approx(1 - 0.15 - 2 * 0.01)


def test_make_transition_rejects_a_background_that_would_dominate():
    with pytest.raises(ValueError, match="dominant"):
        make_transition(bipolar_to_depression=0.05, background=0.05)


def test_config_rejects_rows_that_are_not_distributions():
    bad = np.eye(K)
    bad[0, 1] = 0.2
    with pytest.raises(ValueError, match="rows"):
        SyntheticConfig(n=10, transition=bad)


def test_generation_is_reproducible_under_a_seed():
    config = SyntheticConfig(n=5_000, transition=make_transition(), seed=7)
    a, b = generate(config), generate(config)
    np.testing.assert_array_equal(a.y, b.y)
    np.testing.assert_array_equal(a.y_tilde, b.y_tilde)
    np.testing.assert_array_equal(a.probs, b.probs)
    c = generate(SyntheticConfig(n=5_000, transition=make_transition(), seed=8))
    assert not np.array_equal(a.y_tilde, c.y_tilde)


def test_scores_are_distributions_and_separability_controls_accuracy():
    accuracy = []
    for separability in (0.0, 1.0, 2.0, 4.0):
        sample = generate(
            SyntheticConfig(
                n=50_000, transition=np.eye(K), separability=separability, seed=1
            )
        )
        np.testing.assert_allclose(sample.probs.sum(axis=1), 1.0)
        assert (sample.probs > 0).all()
        accuracy.append((sample.probs.argmax(axis=1) == sample.y).mean())
    # Separability 0 is an uninformative scorer: argmax is uniform over K classes.
    assert accuracy[0] == pytest.approx(1 / K, abs=0.01)
    assert all(lo < hi for lo, hi in zip(accuracy, accuracy[1:]))


def test_identity_transition_leaves_labels_unchanged():
    """Step 4 sanity gate, generator side: T = I means no contamination."""
    sample = generate(SyntheticConfig(n=100_000, transition=np.eye(K), seed=3))
    np.testing.assert_array_equal(sample.y_tilde, sample.y)


def test_empirical_transition_matches_the_generator():
    t = make_transition(bipolar_to_depression=0.15, background=0.01)
    sample = generate(SyntheticConfig(n=400_000, transition=t, seed=4))
    counts = np.zeros((K, K))
    np.add.at(counts, (sample.y, sample.y_tilde), 1)
    row_n = counts.sum(axis=1, keepdims=True)
    empirical = counts / row_n
    se = np.sqrt(t * (1 - t) / row_n)
    assert (np.abs(empirical - t) <= 4 * se + 1e-12).all()


@pytest.mark.parametrize(
    "bipolar_to_depression, background", [(0.15, 0.0), (0.15, 0.01), (0.30, 0.02)]
)
def test_recovered_clean_prior_matches_truth_within_sampling_error(
    bipolar_to_depression, background
):
    """Numerical check of D-028's rho = (T^T)^-1 rho_tilde (bears on open item 3).

    ``Ytilde`` is iid categorical with parameter ``T^T rho``, so ``rho_tilde_hat`` is a
    multinomial proportion with covariance ``(diag(p) - p p^T) / n``, and the recovered
    prior has covariance ``A Sigma A^T`` with ``A = (T^T)^-1`` (exact, the map is
    linear). Every component must sit within four of its standard errors.
    """
    n = 1_000_000
    t = make_transition(bipolar_to_depression, background)
    sample = generate(SyntheticConfig(n=n, transition=t, seed=11))
    recovered = recover_clean_prior(sample.y_tilde, t)

    p = t.T @ NOISY_PRIOR
    a = np.linalg.inv(t.T)
    cov = a @ ((np.diag(p) - np.outer(p, p)) / n) @ a.T
    se = np.sqrt(np.diag(cov))
    assert (np.abs(recovered - NOISY_PRIOR) <= 4 * se).all()
    # The observed noisy bipolar share is visibly off the truth, so the check is not
    # passing merely because the noise is too small to matter. The sign varies: with a
    # background rate, inflow from depression (0.812 * background) can outweigh the
    # bipolar outflow and push the noisy share *above* the truth (D-047's inflow point).
    observed = np.bincount(sample.y_tilde, minlength=K) / n
    assert abs(NOISY_PRIOR[BIPOLAR] - observed[BIPOLAR]) > 10 * se[BIPOLAR]


def test_recovered_clean_prior_is_unbiased_and_its_error_is_calibrated():
    """Across 200 seeds, recovery z-scores have mean near 0 and spread near 1.

    A single seed can only show the error is small; this shows it is centred on the
    truth and that the delta-method standard error above is the right yardstick.
    """
    n, seeds = 200_000, 200
    t = make_transition(0.15, 0.01)
    p = t.T @ NOISY_PRIOR
    a = np.linalg.inv(t.T)
    se = np.sqrt(np.diag(a @ ((np.diag(p) - np.outer(p, p)) / n) @ a.T))
    z = np.array(
        [
            (recover_clean_prior(generate(SyntheticConfig(n=n, transition=t, seed=s)).y_tilde, t)
             - NOISY_PRIOR) / se
            for s in range(seeds)
        ]
    )
    # Mean of 200 standard normals has sd 0.07; spread of 200 has sd about 0.05.
    assert (np.abs(z.mean(axis=0)) < 0.25).all()
    assert ((z.std(axis=0) > 0.85) & (z.std(axis=0) < 1.15)).all()
