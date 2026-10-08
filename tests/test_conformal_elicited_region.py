"""Tests for scoping an elicited region against Algorithm 2's budget (roadmap step 3b).

The regression tests pin the numbers quoted in ``Docs/c2-elicitation-scoping.md``, so
that document cannot drift from the code that produced it.
"""

import numpy as np
import pytest

from src.conformal.c1_matrices import CALIBRATION_COUNTS, NOISY_PRIOR, T_CLEANLAB
from src.conformal.elicited_region import (
    BIPOLAR,
    DEPRESSION,
    EATING_DISORDER,
    SCHIZOPHRENIA,
    SCOPING_REGIONS,
    ElicitedRegion,
    evaluate_region,
    single_entry_breach_points,
    v_box,
    worst_case_label_ppv,
)
from src.conformal.v_region import (
    degeneracy_test,
    expected_sup_deviation,
    max_admissible_offdiagonal_mass,
    region_from_matrices,
    transition_to_v,
)


@pytest.fixture(scope="module")
def c_n_k():
    return [expected_sup_deviation(n) for n in CALIBRATION_COUNTS]


@pytest.fixture(scope="module")
def budgets():
    return [b.max_offdiagonal_mass for b in max_admissible_offdiagonal_mass(CALIBRATION_COUNTS)]


def _verdicts(name, c_n_k):
    verdicts, info = evaluate_region(
        SCOPING_REGIONS[name], NOISY_PRIOR, CALIBRATION_COUNTS, c_n_k=c_n_k
    )
    return verdicts, info


# --- the propagation agrees with the existing single-matrix path ----------------

def test_point_region_reproduces_v_region(c_n_k):
    """A zero-width region at cleanlab's T must give exactly D-042's numbers."""
    t = T_CLEANLAB / T_CLEANLAB.sum(axis=1, keepdims=True)
    point = {(i, j): t[i, j] for i in range(4) for j in range(4) if i != j}
    verdicts, _ = evaluate_region(
        ElicitedRegion(upper=point, lower=point), NOISY_PRIOR, CALIBRATION_COUNTS, c_n_k=c_n_k
    )
    lo, hi = region_from_matrices([T_CLEANLAB], NOISY_PRIOR)
    reference = degeneracy_test(lo, hi, CALIBRATION_COUNTS)
    for mine, theirs in zip(verdicts, reference):
        assert mine.delta_ci == pytest.approx(theirs.delta_ci, abs=2e-3)


def test_box_contains_every_interior_point():
    for region in SCOPING_REGIONS.values():
        _, _, info = v_box(region, NOISY_PRIOR)
        assert info["max_interior_escape"] <= 1e-12


# --- the structural finding: what the bipolar budget does and does not see -------

def test_bipolar_row_of_v_ignores_leakage_out_of_bipolar():
    """True-bipolar posts landing in r/depression leave the bipolar label's row of M at
    exactly (1, 0, 0, 0), so bipolar's budget does not see that entry at all."""
    t = np.eye(4)
    t[BIPOLAR, DEPRESSION] = 0.6
    t[BIPOLAR, BIPOLAR] = 0.4
    v, m, _ = transition_to_v(t, NOISY_PRIOR)
    assert np.allclose(m[BIPOLAR], [1, 0, 0, 0])
    assert np.allclose(v[BIPOLAR], [1, 0, 0, 0])


def test_single_entry_breach_points(budgets):
    points = single_entry_breach_points(NOISY_PRIOR, budgets)
    assert points[(DEPRESSION, BIPOLAR)] == (0.0130, [BIPOLAR])
    assert points[(DEPRESSION, SCHIZOPHRENIA)] == (0.0190, [SCHIZOPHRENIA])
    assert points[(DEPRESSION, EATING_DISORDER)] == (0.0350, [EATING_DISORDER])
    assert points[(BIPOLAR, DEPRESSION)] == (0.8765, [DEPRESSION])


def test_clinic_rate_is_inconsistent_with_the_observed_bipolar_share():
    """An 11% depression-to-bipolar rate (the outpatient figure) recovers a negative
    clean bipolar prior on some vertices: the data refute it as a Reddit rate."""
    region = ElicitedRegion(upper={(DEPRESSION, BIPOLAR): 0.11})
    _, _, info = v_box(region, NOISY_PRIOR, interior_samples=0)
    assert info["vertices_off_simplex"] == 1


# --- regression: the scenario table in the scoping document ---------------------

@pytest.mark.parametrize(
    "name, bipolar_delta, bipolar_label",
    [
        ("F1", 0.0647, "ok"),
        ("F2", 0.0615, "ok"),
        ("F3", 0.1026, "DEGENERATE (borderline)"),
        ("L1", 0.3110, "DEGENERATE (robust)"),
        ("R0", 0.2457, "DEGENERATE (robust)"),
    ],
)
def test_scenario_bipolar_verdicts(name, bipolar_delta, bipolar_label, c_n_k):
    verdicts, _ = _verdicts(name, c_n_k)
    assert verdicts[BIPOLAR].delta_ci == pytest.approx(bipolar_delta, abs=5e-4)
    assert verdicts[BIPOLAR].label() == bipolar_label


def test_fitting_regions_pass_for_all_four(c_n_k):
    for name in ("F1", "F2"):
        verdicts, _ = _verdicts(name, c_n_k)
        assert all(v.label() == "ok" for v in verdicts), name


def test_worst_case_bipolar_ppv_brackets_the_threshold():
    assert worst_case_label_ppv(SCOPING_REGIONS["F1"], NOISY_PRIOR)[BIPOLAR] == pytest.approx(0.873, abs=1e-3)
    assert worst_case_label_ppv(SCOPING_REGIONS["F3"], NOISY_PRIOR)[BIPOLAR] == pytest.approx(0.763, abs=1e-3)
    assert worst_case_label_ppv(SCOPING_REGIONS["L1"], NOISY_PRIOR)[BIPOLAR] == pytest.approx(0.450, abs=1e-3)
