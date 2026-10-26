"""Tests pinning arm (c) and D-043's scoring (roadmap step 8). Torch-free.

``run_d043`` is run once per session (about 40 seconds) with D-043's committed run
specification (cf12627): 100 repetitions, base seed 0, reference samples at seed
10**6. The pinned numbers are the ones D-046 reports; a change to any of them means
the run no longer reproduces the record.
"""

import numpy as np
import pytest

from src.conformal.arm_c import (
    PRIMARY_CERTAIN,
    REGION_NAMES,
    closed_form,
    normalised,
    recovered_prior,
    region_box,
    region_truth,
    run_d043,
)
from src.conformal.c1_matrices import NOISY_PRIOR, T_CLEANLAB
from src.conformal.v_region import transition_to_v

K = 4

DEG_STAR = {'A': [77, 0, 1, 32], 'B': [100, 0, 50, 100], 'F1': [28, 0, 0, 4], 'F2': [20, 0, 0, 6], 'F3': [100, 0, 0, 5], 'L1': [100, 0, 0, 6], 'R0': [100, 0, 6, 100]}
DEG_ZERO = {'A': [62, 0, 0, 7], 'B': [100, 0, 17, 100], 'F1': [4, 0, 0, 0], 'F2': [2, 0, 0, 0], 'F3': [100, 0, 0, 0], 'L1': [100, 0, 0, 0], 'R0': [100, 0, 0, 100]}
A_COV = {'A': [0.9556, 0.9018, 0.9039, 0.9195], 'B': [0.9556, 0.9018, 0.9039, 0.9195], 'F1': [0.9467, 0.9433, 0.9184, 0.9345], 'F2': [0.946, 0.9392, 0.9185, 0.9347], 'F3': [0.976, 0.9429, 0.9184, 0.9345], 'L1': [0.9977, 0.9417, 0.9184, 0.9345], 'R0': [0.9919, 0.9534, 0.9169, 0.9538]}
A_SE = {'A': [0.0015, 0.0003, 0.001, 0.0014], 'B': [0.0015, 0.0003, 0.001, 0.0014], 'F1': [0.0015, 0.0003, 0.0009, 0.0012], 'F2': [0.0015, 0.0003, 0.0009, 0.0012], 'F3': [0.0011, 0.0003, 0.0009, 0.0012], 'L1': [0.0002, 0.0003, 0.0009, 0.0012], 'R0': [0.0004, 0.0003, 0.0009, 0.001]}
A_SIZE = {'A': [1.328, 1.374, 1.373, 1.369], 'B': [1.328, 1.374, 1.373, 1.369], 'F1': [1.428, 1.434, 1.452, 1.442], 'F2': [1.419, 1.429, 1.442, 1.432], 'F3': [1.456, 1.552, 1.57, 1.558], 'L1': [1.474, 1.86, 1.876, 1.866], 'R0': [1.544, 1.761, 1.795, 1.756]}
A_FULL = {'A': [0.0, 0.0, 0.0, 0.0], 'B': [0.0, 0.0, 0.0, 0.0], 'F1': [0.0, 0.0, 0.0, 0.0], 'F2': [0.0, 0.0, 0.0, 0.0], 'F3': [0.001, 0.001, 0.001, 0.001], 'L1': [0.001, 0.005, 0.007, 0.005], 'R0': [0.002, 0.006, 0.009, 0.006]}
C_COV = {'A': [1.0, 0.9283, 0.9983, 1.0], 'B': [1.0, 0.9828, 1.0, 1.0], 'F1': [1.0, 0.9999, 0.9995, 1.0], 'F2': [1.0, 0.9999, 0.9995, 1.0], 'F3': [1.0, 0.9999, 0.9995, 1.0], 'L1': [1.0, 0.9999, 0.9996, 1.0], 'R0': [1.0, 1.0, 1.0, 1.0]}
C_SE = {'A': [0.0, 0.0003, 0.0001, 0.0], 'B': [0.0, 0.0001, 0.0, 0.0], 'F1': [0.0, 0.0, 0.0, 0.0], 'F2': [0.0, 0.0, 0.0, 0.0], 'F3': [0.0, 0.0, 0.0, 0.0], 'L1': [0.0, 0.0, 0.0, 0.0], 'R0': [0.0, 0.0, 0.0, 0.0]}
C_SIZE = {'A': [2.789, 3.567, 3.122, 2.82], 'B': [3.312, 3.95, 3.344, 3.313], 'F1': [3.597, 3.678, 3.804, 3.652], 'F2': [3.597, 3.684, 3.789, 3.643], 'F3': [3.606, 3.709, 3.825, 3.678], 'L1': [3.625, 3.732, 3.828, 3.687], 'R0': [3.924, 3.947, 3.976, 3.924]}
C_FULL = {'A': [0.106, 0.594, 0.148, 0.109], 'B': [0.339, 0.95, 0.344, 0.339], 'F1': [0.682, 0.723, 0.832, 0.709], 'F2': [0.683, 0.731, 0.821, 0.706], 'F3': [0.688, 0.74, 0.844, 0.721], 'L1': [0.701, 0.759, 0.846, 0.73], 'R0': [0.929, 0.947, 0.976, 0.929]}
PROP3_AT = {'A': [100, 100, 100, 100], 'B': [100, 100, 100, 100], 'F1': [100, 0, 100, 100], 'F2': [100, 100, 100, 100], 'F3': [100, 0, 100, 100], 'L1': [100, 18, 100, 100], 'R0': [100, 100, 100, 100]}
FLIPS = {'A': [0.15, 0.02, 0.05, 0.1], 'B': [0.5, 0.02, 0.1, 0.3], 'F1': [0.1, 0.05, 0.05, 0.1], 'F2': [0.1, 0.05, 0.05, 0.1], 'F3': [0.15, 0.05, 0.05, 0.1], 'L1': [0.5, 0.05, 0.05, 0.1], 'R0': [0.3, 0.1, 0.1, 0.2]}
H2H = {'O1': {0: ('WIN BY ABSTENTION', 0.8095, 1.0, 1.102, 2.789, 77), 3: ('WIN', 0.8424, 1.0, 1.121, 2.82, 32)}, 'O2': {0: ('LOSS', 0.9991, 1.0, 1.773, 2.789, 37), 3: ('LOSS', 0.9912, 1.0, 2.003, 2.846, 3)}}
SIZE_BIP = 2.789



@pytest.fixture(scope="module")
def run():
    return run_d043(sweep=True)


def test_transcribed_certain_cells_match_the_closed_form():
    expected = {"degenerate": "certain-degenerate", "admissible": "certain-ok", None: "undetermined"}
    for mode in ("zeta_star", "zeta_zero"):
        for name in REGION_NAMES:
            for k in range(K):
                assert closed_form(name, k, mode).primary_class() == expected[
                    PRIMARY_CERTAIN[mode].get((name, k))
                ], (name, k, mode)


def test_every_synthetic_truth_lies_in_its_box_with_a_valid_prior():
    off = ~np.eye(K, dtype=bool)
    truths = {name: region_truth(name) for name in REGION_NAMES}
    for name, truth in truths.items():
        v = transition_to_v(truth, NOISY_PRIOR)[0]
        lo, hi = region_box(name)
        assert (recovered_prior(truth) > 0).all()
        assert np.all((v[off] >= lo[off] - 1e-12) & (v[off] <= hi[off] + 1e-12)), name
    v = transition_to_v(normalised(T_CLEANLAB), NOISY_PRIOR)[0]
    lo, hi = region_box("A")
    assert np.all((v[off] >= lo[off] - 1e-12) & (v[off] <= hi[off] + 1e-12))


def test_no_certain_cell_and_no_flip_breaks_the_closed_form(run):
    assert not [c for c in run["contradictions"] if c.startswith("BUG")]


@pytest.mark.parametrize("name", REGION_NAMES)
def test_degenerate_run_counts_are_pinned(run, name):
    res = run["results"]
    assert res[(name, "zeta_star")].degenerate_runs["conservative"].tolist() == DEG_STAR[name]
    assert res[(name, "zeta_zero")].degenerate_runs["conservative"].tolist() == DEG_ZERO[name]


@pytest.mark.parametrize("name", REGION_NAMES)
def test_coverage_size_and_full_rate_are_pinned(run, name):
    arms = run["results"][(name, "zeta_star")].arms
    for arm, cov, size, full in (
        ("arm (a)", A_COV, A_SIZE, A_FULL),
        ("arm (c) conservative", C_COV, C_SIZE, C_FULL),
    ):
        np.testing.assert_allclose(arms[arm].coverage, cov[name], atol=6e-5)
        np.testing.assert_allclose(arms[arm].mean_size, size[name], atol=6e-4)
        np.testing.assert_allclose(arms[arm].full_rate, full[name], atol=6e-4)
    # The optimistic variant is arm (a) on every run (D-043 §4(a), §5).
    np.testing.assert_array_equal(arms["arm (c) optimistic"].tau, arms["arm (a)"].tau)


def test_proposition_three_counts_are_pinned(run):
    for name in REGION_NAMES:
        r = run["results"][(name, "zeta_star")]
        assert r.prop3_as_stated_fail.tolist() == [100] * K
        assert r.prop3_at_threshold_fail.tolist() == PROP3_AT[name]


def test_flip_alphas_are_pinned(run):
    for name in REGION_NAMES:
        assert [s["measured"] for s in run["flips"][name]] == FLIPS[name]
        assert all(s["in_bracket"] for s in run["flips"][name])


def test_head_to_head_is_pinned_and_not_won(run):
    for o, rows in H2H.items():
        for k, (verdict, b_cov, c_cov, b_size, c_size, c_deg) in rows.items():
            got = run["h2h"][o][k]
            assert got["verdict"] == verdict
            assert got["b_coverage"] == pytest.approx(b_cov, abs=6e-5)
            assert got["c_coverage"] == pytest.approx(c_cov, abs=6e-5)
            assert got["b_size"] == pytest.approx(b_size, abs=6e-4)
            assert got["c_size"] == pytest.approx(c_size, abs=6e-4)
            assert got["c_degenerate_runs"] == c_deg
    assert run["h2h_won"] is False


def test_region_a_bipolar_set_size_is_pinned(run):
    assert run["set_size"]["bipolar"] == pytest.approx(SIZE_BIP, abs=6e-4)
    assert run["set_size"]["branch"] == "partly missed"
    assert run["set_size"]["optimistic_same_as_a"].tolist() == [100] * K


def test_assumptions_are_pinned(run):
    for name in REGION_NAMES:
        for k, row in enumerate(run["assumptions"][name]):
            assert row["a2_fmin_fails"]
            assert row["a3_holds"]
            assert row["a5_holds"] == ((name, k) == ("A", 1))


def test_contradiction_count_is_pinned(run):
    assert len(run["contradictions"]) == 52
