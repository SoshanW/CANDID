"""Tests for the C1 -> Sesia conversion and the closed-form degeneracy test.

The regression tests at the end pin the numbers quoted in ``Docs/c2-interface.md``
section 6, so that document cannot silently drift from the code that produced it.
"""

import numpy as np
import pytest

from src.conformal.c1_matrices import (
    CALIBRATION_COUNTS,
    CONDITIONS,
    NOISY_PRIOR,
    REGION_A_MEASURED,
    REGION_B_MEASURED,
    T_CLEANLAB,
    T_HOC_FINE_TUNED,
    T_HOC_BASE,
    T_HOC_MPNET,
    build_from_diagonal,
)
from src.conformal.v_region import (
    DegeneracyVerdict,
    correction_term,
    degeneracy_test,
    max_admissible_offdiagonal_mass,
    region_from_matrices,
    transition_to_v,
)

BIPOLAR = 0
SCHIZOPHRENIA = 3


# --- the conversion ---------------------------------------------------------

def test_identity_transition_gives_identity_v():
    """No noise means no correction: T = I implies M = V = I."""
    v, m, prior = transition_to_v(np.eye(4), NOISY_PRIOR)
    assert np.allclose(v, np.eye(4))
    assert np.allclose(m, np.eye(4))
    assert np.allclose(prior, NOISY_PRIOR / NOISY_PRIOR.sum())


@pytest.mark.parametrize("transition", [T_HOC_FINE_TUNED, T_CLEANLAB])
def test_rows_of_m_and_v_sum_to_one(transition):
    """M is a conditional distribution over true labels, so M 1 = 1, hence V 1 = 1.

    This is what makes a region on the off-diagonals alone sufficient.
    """
    v, m, _ = transition_to_v(transition, NOISY_PRIOR)
    assert np.allclose(m.sum(axis=1), 1.0)
    assert np.allclose(v.sum(axis=1), 1.0)


@pytest.mark.parametrize("transition", [T_HOC_FINE_TUNED, T_CLEANLAB])
def test_clean_prior_is_recovered_and_stays_on_the_simplex(transition):
    v, _, prior = transition_to_v(transition, NOISY_PRIOR)
    assert (prior > 0).all()
    assert prior.sum() == pytest.approx(1.0)
    # rho_tilde = T^T rho is the identity the recovery inverts. Compare against the
    # row-normalised T, since that is what transition_to_v solves against.
    normalised = transition / transition.sum(axis=1)[:, None]
    assert np.allclose(normalised.T @ prior, NOISY_PRIOR / NOISY_PRIOR.sum())
    assert np.isfinite(v).all()


@pytest.mark.parametrize("transition", [T_HOC_FINE_TUNED, T_CLEANLAB])
def test_off_diagonals_of_v_are_negative(transition):
    """Relied on twice: it makes Delta_k >= 0 under Assumption 3, and it makes
    |V_upp| + width equal |V_low| in the correction term."""
    v, _, _ = transition_to_v(transition, NOISY_PRIOR)
    off_diagonal = v[~np.eye(4, dtype=bool)]
    assert (off_diagonal < 0).all()


def test_rejects_transition_whose_rows_do_not_sum_to_one():
    bad = np.eye(4) * 0.5
    with pytest.raises(ValueError, match="rows must sum to 1"):
        transition_to_v(bad, NOISY_PRIOR)


def test_tolerates_four_decimal_place_transcription():
    """T_CLEANLAB as transcribed sums to 1.0001 on some rows."""
    assert not np.allclose(T_CLEANLAB.sum(axis=1), 1.0, atol=1e-6)
    transition_to_v(T_CLEANLAB, NOISY_PRIOR)  # must not raise


# --- the region -------------------------------------------------------------

def test_region_contains_every_supplied_estimate():
    """Deterministic containment is what licenses alpha_V = 0 (D-031 option a)."""
    v_low, v_upp = region_from_matrices(REGION_A_MEASURED, NOISY_PRIOR)
    for transition in REGION_A_MEASURED:
        v, _, _ = transition_to_v(transition, NOISY_PRIOR)
        assert (v >= v_low - 1e-12).all()
        assert (v <= v_upp + 1e-12).all()


def test_region_b_is_wider_than_region_a():
    a_low, a_upp = region_from_matrices(REGION_A_MEASURED, NOISY_PRIOR)
    b_low, b_upp = region_from_matrices(REGION_B_MEASURED, NOISY_PRIOR)
    assert (b_upp - b_low >= a_upp - a_low - 1e-12).all()


def test_build_from_diagonal_preserves_the_diagonal_and_normalises():
    diagonal = np.array([0.5409, 0.9577, 0.8265, 0.6448])
    built = build_from_diagonal(diagonal)
    assert np.allclose(np.diag(built), diagonal)
    assert np.allclose(built.sum(axis=1), 1.0)


def test_transcribed_diagonals_match_the_decision_log():
    """Guards against transcription drift from D-037 and D-041."""
    assert np.diag(T_HOC_FINE_TUNED) == pytest.approx(
        [0.9534, 0.9964, 0.9946, 0.9781], abs=5e-5
    )
    assert np.diag(T_HOC_MPNET) == pytest.approx(
        [0.7071, 0.9729, 0.9210, 0.8188], abs=5e-5
    )


def test_bipolar_leaks_into_depression_in_every_measured_matrix():
    """The clinically predicted direction, consistent across D-034, D-037, D-039, D-041."""
    for transition in (T_HOC_FINE_TUNED, T_CLEANLAB, T_HOC_MPNET):
        bipolar_row = transition[BIPOLAR]
        off_diagonal = {k: v for k, v in enumerate(bipolar_row) if k != BIPOLAR}
        assert max(off_diagonal, key=off_diagonal.get) == 1  # depression


def test_diagonal_reconstruction_errs_in_an_unstable_direction():
    """Records why no region here is built from a reconstructed matrix.

    Both non-fine-tuned arms are now measured, so ``build_from_diagonal`` is retained
    only for this check. Run against both arms it shows the error does not have a
    consistent sign: the construction understates bipolar's off-diagonal mass in V for
    mpnet (ratio 0.92) and overstates it for base (1.19). A one-arm validation would
    therefore have licensed a false claim about which way the approximation errs, and
    that is what happened before the base arm was retrieved.
    """
    def bipolar_mass(transition):
        v, _, _ = transition_to_v(transition, NOISY_PRIOR)
        return sum(abs(v[BIPOLAR, l]) for l in range(4) if l != BIPOLAR)

    mpnet_ratio = bipolar_mass(build_from_diagonal(np.diag(T_HOC_MPNET))) / bipolar_mass(
        T_HOC_MPNET
    )
    base_ratio = bipolar_mass(build_from_diagonal(np.diag(T_HOC_BASE))) / bipolar_mass(
        T_HOC_BASE
    )
    assert mpnet_ratio < 1.0 < base_ratio


# --- the degeneracy test ----------------------------------------------------

def test_wider_region_raises_the_correction_term():
    a_low, a_upp = region_from_matrices(REGION_A_MEASURED, NOISY_PRIOR)
    b_low, b_upp = region_from_matrices(REGION_B_MEASURED, NOISY_PRIOR)
    n_star = min(CALIBRATION_COUNTS)
    narrow = correction_term(BIPOLAR, a_low, a_upp, CALIBRATION_COUNTS[BIPOLAR], n_star)
    wide = correction_term(BIPOLAR, b_low, b_upp, CALIBRATION_COUNTS[BIPOLAR], n_star)
    assert wide > narrow


def test_more_calibration_data_lowers_the_correction_term():
    v_low, v_upp = region_from_matrices(REGION_B_MEASURED, NOISY_PRIOR)
    small = correction_term(BIPOLAR, v_low, v_upp, 481, 481)
    large = correction_term(BIPOLAR, v_low, v_upp, 1924, 1924)
    assert large < small


def test_n_star_is_set_by_the_rarest_condition():
    """Bipolar's calibration count drives the correction for depression too."""
    v_low, v_upp = region_from_matrices(REGION_B_MEASURED, NOISY_PRIOR)
    depression = 1
    with_rare = correction_term(depression, v_low, v_upp, 11351, n_star=481)
    without_rare = correction_term(depression, v_low, v_upp, 11351, n_star=11351)
    assert with_rare > without_rare


def test_zero_width_region_still_carries_a_correction():
    """A point estimate is not free: c(n_k) and the |V_upp| term remain."""
    v, _, _ = transition_to_v(T_CLEANLAB, NOISY_PRIOR)
    delta = correction_term(BIPOLAR, v, v, 481, 481)
    assert delta > 0


def test_verdict_labels_borderline_when_margin_is_below_the_correction_bound():
    """The label logic itself, on a constructed case.

    No condition is borderline on the measured regions, so this exercises the branch
    directly rather than through the data.
    """
    verdict = DegeneracyVerdict(
        condition=BIPOLAR, delta_ci=0.11, alpha=0.10, correction_bound=0.05
    )
    assert verdict.degenerate and not verdict.robust
    assert "borderline" in verdict.label()

    robust = DegeneracyVerdict(
        condition=BIPOLAR, delta_ci=0.30, alpha=0.10, correction_bound=0.05
    )
    assert robust.degenerate and robust.robust
    assert robust.label() == "DEGENERATE (robust)"


def test_both_rare_conditions_degenerate_robustly_at_the_full_region():
    v_low, v_upp = region_from_matrices(REGION_B_MEASURED, NOISY_PRIOR)
    verdicts = degeneracy_test(v_low, v_upp, CALIBRATION_COUNTS, alpha=0.10)
    for k in (BIPOLAR, SCHIZOPHRENIA):
        assert verdicts[k].degenerate, CONDITIONS[k]
        assert verdicts[k].robust, CONDITIONS[k]
    for k in (1, 2):  # depression, eating disorder
        assert not verdicts[k].degenerate, CONDITIONS[k]


# --- regression on the numbers quoted in Docs/c2-interface.md ----------------

def test_region_a_survives_and_region_b_degenerates_for_bipolar():
    a_low, a_upp = region_from_matrices(REGION_A_MEASURED, NOISY_PRIOR)
    b_low, b_upp = region_from_matrices(REGION_B_MEASURED, NOISY_PRIOR)

    a = degeneracy_test(a_low, a_upp, CALIBRATION_COUNTS, alpha=0.10)[BIPOLAR]
    b = degeneracy_test(b_low, b_upp, CALIBRATION_COUNTS, alpha=0.10)[BIPOLAR]

    assert a.delta_ci == pytest.approx(0.092, abs=0.002)
    assert not a.degenerate
    assert b.delta_ci == pytest.approx(0.285, abs=0.002)
    assert b.degenerate and b.robust


def test_bipolar_region_a_fails_at_the_tighter_alpha():
    """The 0.008 margin at alpha = 0.10 does not survive alpha = 0.05."""
    v_low, v_upp = region_from_matrices(REGION_A_MEASURED, NOISY_PRIOR)
    assert degeneracy_test(v_low, v_upp, CALIBRATION_COUNTS, alpha=0.05)[BIPOLAR].degenerate


def test_feasibility_frontier_matches_the_documented_budgets():
    budgets = max_admissible_offdiagonal_mass(CALIBRATION_COUNTS, alpha=0.10)
    expected = [0.3783, 0.4970, 0.4406, 0.4107]
    for budget, value in zip(budgets, expected):
        assert budget.max_offdiagonal_mass == pytest.approx(value, abs=0.002)
