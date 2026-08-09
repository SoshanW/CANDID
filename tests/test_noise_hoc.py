"""Tests for the HOC estimator (Stage 2, C1).

The centrepiece is a synthetic recovery test: generate clusterable data from a known
transition matrix and confirm HOC recovers it (up to the true-class permutation the
method is identifiable to). Torch-free pieces (config, consensus counting, prediction
check, data generator) are tested without the deep-learning stack; the estimator
itself is guarded on torch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.noise.hoc_estimate import (
    HOCConfig,
    HOCResult,
    _count_consensus,
    aggregate_results,
    check_d040_rule,
    check_pre_registered_prediction,
    compare_to_reference,
    format_d040_verdict,
    make_clusterable_noisy_data,
    prior_divergence,
)


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        HOCConfig(sample_size=2)
    with pytest.raises(ValueError):
        HOCConfig(order_weights=(1.0, 1.0))  # wrong length
    with pytest.raises(ValueError):
        HOCConfig(order_weights=(1.0, -1.0, 1.0))  # negative


def test_consensus_counts_are_normalised_distributions() -> None:
    conditions = ["a", "b", "c"]
    t = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    p = np.array([0.4, 0.3, 0.3])
    feats, labels = make_clusterable_noisy_data(t, p, conditions, n=600, seed=0)
    x = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    idx = {c: i for i, c in enumerate(conditions)}
    labels_idx = np.array([idx[l] for l in labels])
    c1, c2, c3 = _count_consensus(x.astype(np.float32), labels_idx, 3,
                                  HOCConfig(n_rounds=4, sample_size=300), np.random.default_rng(0))
    assert c1.shape == (3,) and c2.shape == (3, 3) and c3.shape == (3, 3, 3)
    assert c1.sum() == pytest.approx(1.0)
    assert c2.sum() == pytest.approx(1.0)
    assert c3.sum() == pytest.approx(1.0)


def test_check_pre_registered_prediction_branches() -> None:
    conds = ["bipolar", "depression", "eating_disorder", "schizophrenia"]
    # All diagonals high -> prediction holds, falsifier not met.
    high = np.full((4, 4), 0.02)
    np.fill_diagonal(high, 0.9)
    out = check_pre_registered_prediction(high, np.zeros((4, 4)), conds)
    assert out["prediction_holds"] is True
    assert out["falsifier_met"] is False

    # Bipolar low with dominant off-diagonal = depression -> falsifier met.
    fals = np.array([
        [0.55, 0.35, 0.05, 0.05],  # true bipolar -> mostly depression off-diagonal
        [0.02, 0.94, 0.02, 0.02],
        [0.02, 0.06, 0.90, 0.02],
        [0.05, 0.20, 0.05, 0.70],
    ])
    out2 = check_pre_registered_prediction(fals, np.zeros((4, 4)), conds)
    assert out2["prediction_holds"] is False
    assert out2["falsifier_met"] is True
    assert out2["bipolar"]["dominant_offdiagonal"] == "depression"


def test_prior_divergence_zero_and_positive() -> None:
    conds = ["bipolar", "depression", "eating_disorder", "schizophrenia"]
    obs = np.array([0.038, 0.812, 0.096, 0.055])
    assert prior_divergence(obs, conds)["l1"] == pytest.approx(0.0, abs=1e-9)
    uni = np.full(4, 0.25)
    assert prior_divergence(uni, conds)["l1"] == pytest.approx(float(np.abs(uni - obs).sum()))


# --- D-040 pre-registered decision rule -----------------------------------------

_D040_CONDS = ["bipolar", "depression", "eating_disorder", "schizophrenia"]


def _fake_results(bipolar_diag: float, det: float, dominant: bool, n_seeds: int = 3) -> list[HOCResult]:
    """Build HOCResults with a chosen bipolar diagonal and assumption state.

    The rule reads the bipolar diagonal off the aggregated mean T and the assumption
    flags off every individual seed, so both have to be set consistently here.
    """
    t = np.full((4, 4), (1.0 - bipolar_diag) / 3.0)
    np.fill_diagonal(t, bipolar_diag)
    row_dd = {c: dominant for c in _D040_CONDS}
    return [
        HOCResult(
            conditions=list(_D040_CONDS),
            T=t,
            p=np.full(4, 0.25),
            noisy_marginal=np.full(4, 0.25),
            seed=s,
            diagonal=np.diag(t).copy(),
            row_diagonally_dominant=row_dd,
            is_nonsingular=abs(det) > 1e-8,
            det=det,
            final_loss=0.0,
        )
        for s in range(n_seeds)
    ]


@pytest.mark.parametrize(
    "bipolar_diag, det, dominant, expected",
    [
        # Failure survives the prescribed fix, assumptions intact.
        (0.95, 0.92, True, "STRENGTHENED_ARM"),
        # HOC recovered real noise, assumptions intact -> C1 narrows.
        (0.30, 0.80, True, "NARROWED"),
        # Same low diagonal, but the matrix collapsed: NOT a rescue.
        (0.30, 0.01, True, "DEGENERATE"),
        (0.30, 0.80, False, "DEGENERATE"),
        # The pre-registered no-claim band.
        (0.75, 0.92, True, "INDETERMINATE"),
        # Nothing may fall through the table.
        (0.95, 0.01, True, "ASSUMPTIONS_FAILED"),
    ],
)
def test_d040_rule_branches(bipolar_diag, det, dominant, expected) -> None:
    results = _fake_results(bipolar_diag, det, dominant)
    verdict = check_d040_rule(results, aggregate_results(results), extractor="base")
    assert verdict["branch"] == expected
    assert verdict["extractor"] == "base"


@pytest.mark.parametrize(
    "bipolar_diag, expected",
    [
        (0.85, "STRENGTHENED_ARM"),  # strengthen threshold is inclusive
        (0.70, "INDETERMINATE"),     # narrow threshold is exclusive: 0.70 is in the band
    ],
)
def test_d040_rule_threshold_boundaries_are_as_pre_registered(bipolar_diag, expected) -> None:
    # One seed, so the aggregated mean is bit-exact and the assertion is about the
    # comparison operators rather than about floating-point averaging.
    results = _fake_results(bipolar_diag, det=0.92, dominant=True, n_seeds=1)
    verdict = check_d040_rule(results, aggregate_results(results), extractor="base")
    assert verdict["bipolar_diagonal"] == bipolar_diag
    assert verdict["branch"] == expected


def test_d040_rule_requires_assumptions_on_every_seed() -> None:
    # A single collapsed seed must not be averaged away by four healthy ones: D-040
    # says the assumptions have to hold on every seed, not on the mean.
    results = _fake_results(0.30, det=0.80, dominant=True, n_seeds=5)
    results[2] = HOCResult(**{**results[2].__dict__, "det": 0.001})
    verdict = check_d040_rule(results, aggregate_results(results), extractor="mpnet")
    assert verdict["branch"] == "DEGENERATE"
    assert verdict["nonsingular_all_seeds"] is False


def test_d040_rule_without_bipolar_is_not_applicable() -> None:
    results = _fake_results(0.95, 0.92, True)
    conds = ["a", "b", "c", "d"]
    patched = [HOCResult(**{**r.__dict__, "conditions": conds}) for r in results]
    verdict = check_d040_rule(patched, aggregate_results(patched), extractor="base")
    assert verdict["branch"] == "NOT_APPLICABLE"
    assert "not applicable" in format_d040_verdict(verdict)


def test_d040_verdict_prints_thresholds_before_the_number() -> None:
    # The rendered verdict must show the pre-registered thresholds, so a reader can
    # see they were fixed in advance rather than fitted to the result.
    results = _fake_results(0.30, 0.80, True)
    text = format_d040_verdict(check_d040_rule(results, aggregate_results(results), "base"))
    assert "0.85" in text and "0.7" in text and "0.5" in text
    assert text.index("thresholds") < text.index("BRANCH")
    assert "NARROWED" in text


def test_compare_to_reference_pass_and_fail(tmp_path) -> None:
    conds = ["bipolar", "depression"]
    mean_t = np.array([[0.9, 0.1], [0.05, 0.95]])
    ref = pd.DataFrame(mean_t, index=["true_bipolar", "true_depression"],
                       columns=["noisy_bipolar", "noisy_depression"])
    path = tmp_path / "hoc_mean_T.csv"
    ref.to_csv(path)
    ok = compare_to_reference(mean_t, conds, path, tol=1e-6)
    assert ok["within_tol"] and ok["max_abs_diff"] == pytest.approx(0.0)
    bad = compare_to_reference(mean_t + 0.02, conds, path, tol=1e-3)
    assert not bad["within_tol"] and bad["max_abs_diff"] == pytest.approx(0.02)


# --- torch-dependent estimator (skips where torch is absent) --------------------

torch = pytest.importorskip("torch")

from src.noise.hoc_estimate import (  # noqa: E402
    estimate_hoc,
    per_seed_T_long,
    per_seed_frame,
    run_hoc_multiseed,
)


def _align_rows(t_hat: np.ndarray, t_true: np.ndarray) -> np.ndarray:
    """Reorder t_hat's rows to best match t_true (HOC identifies T up to a
    true-class permutation)."""
    from scipy.optimize import linear_sum_assignment

    cost = np.linalg.norm(t_hat[:, None, :] - t_true[None, :, :], axis=2)
    hat_rows, true_rows = linear_sum_assignment(cost)
    aligned = np.empty_like(t_hat)
    for h, tr in zip(hat_rows, true_rows):
        aligned[tr] = t_hat[h]
    return aligned


def test_hoc_recovers_known_transition_matrix() -> None:
    conditions = ["bipolar", "depression", "eating_disorder"]
    t_true = np.array([
        [0.70, 0.25, 0.05],   # true bipolar -> 25% mislabelled depression
        [0.03, 0.95, 0.02],
        [0.04, 0.06, 0.90],
    ])
    p_true = np.array([0.2, 0.6, 0.2])
    feats, labels = make_clusterable_noisy_data(t_true, p_true, conditions, n=3000, seed=0)

    res = estimate_hoc(
        feats, labels,
        HOCConfig(n_rounds=8, sample_size=1500, max_iter=1200, lr=0.1),
        seed=0,
    )
    assert res.conditions == conditions
    aligned = _align_rows(res.T, t_true)
    assert np.abs(aligned - t_true).max() < 0.12
    # The dominant-diagonal structure must be recovered.
    assert np.all(np.diag(aligned) > 0.5)
    assert res.is_nonsingular


def test_multiseed_reports_a_spread() -> None:
    conditions = ["bipolar", "depression", "eating_disorder"]
    t_true = np.array([[0.7, 0.25, 0.05], [0.03, 0.95, 0.02], [0.04, 0.06, 0.90]])
    p_true = np.array([0.2, 0.6, 0.2])
    feats, labels = make_clusterable_noisy_data(t_true, p_true, conditions, n=1500, seed=1)
    results = run_hoc_multiseed(
        feats, labels, seeds=[0, 1, 2],
        config=HOCConfig(n_rounds=5, sample_size=800, max_iter=600),
    )
    agg = aggregate_results(results)
    assert np.asarray(agg["mean_T"]).shape == (3, 3)
    assert np.asarray(agg["std_T"]).shape == (3, 3)
    assert agg["n_seeds"] == 3


def test_per_seed_outputs_and_reproducibility() -> None:
    conditions = ["bipolar", "depression", "eating_disorder"]
    t_true = np.array([[0.7, 0.25, 0.05], [0.03, 0.95, 0.02], [0.04, 0.06, 0.90]])
    p_true = np.array([0.2, 0.6, 0.2])
    feats, labels = make_clusterable_noisy_data(t_true, p_true, conditions, n=1200, seed=2)
    cfg = HOCConfig(n_rounds=4, sample_size=700, max_iter=500)

    r1 = run_hoc_multiseed(feats, labels, [0, 1, 2], cfg)
    # Reproducibility: identical seeds -> identical matrices (deterministic on CPU).
    r2 = run_hoc_multiseed(feats, labels, [0, 1, 2], cfg)
    for a, b in zip(r1, r2):
        assert np.allclose(a.T, b.T)

    ps = per_seed_frame(r1)
    assert len(ps) == 3
    for c in conditions:
        assert f"diag_{c}" in ps.columns
        assert f"p_{c}" in ps.columns
    assert "prior_l1_div" in ps.columns

    long = per_seed_T_long(r1)
    assert len(long) == 3 * 3 * 3  # seeds * K * K
    assert set(long.columns) == {"seed", "true", "noisy", "value"}
