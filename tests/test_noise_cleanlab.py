"""Tests for the cleanlab transition-matrix estimate (torch-free; uses cleanlab)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.noise.cleanlab_estimate import (
    CleanlabResult,
    compare_elementwise,
    estimate_transition_matrix,
    load_hoc_matrix,
    reduce_probs_to_classes,
)


def test_reduce_probs_selects_and_renormalises() -> None:
    probs = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.1, 0.1, 0.4]])
    reduced = reduce_probs_to_classes(probs, ["a", "b", "c", "d"], ["b", "d"])
    assert reduced.shape == (2, 2)
    np.testing.assert_allclose(reduced.sum(axis=1), 1.0)
    # Row 0: cols b=0.2, d=0.4 -> renormalised 1/3, 2/3.
    np.testing.assert_allclose(reduced[0], [0.2 / 0.6, 0.4 / 0.6])


def test_reduce_probs_missing_class_raises() -> None:
    with pytest.raises(ValueError, match="lack columns"):
        reduce_probs_to_classes(np.ones((2, 2)), ["a", "b"], ["a", "c"])


def test_estimate_recovers_known_noise_matrix_in_hoc_orientation() -> None:
    conditions = ["bipolar", "depression", "eating_disorder"]
    # Known transition in HOC orientation: T[true][noisy], rows sum to 1.
    t_true = np.array([[0.80, 0.15, 0.05],
                       [0.05, 0.90, 0.05],
                       [0.10, 0.10, 0.80]])
    prior = np.array([0.3, 0.5, 0.2])
    rng = np.random.default_rng(0)
    n = 5000
    true = rng.choice(3, size=n, p=prior)
    noisy = np.array([rng.choice(3, p=t_true[t]) for t in true])
    labels = np.array([conditions[j] for j in noisy])

    # A good classifier: predicted probs peaked at the TRUE class, columns in
    # `conditions` order.
    probs = np.full((n, 3), 0.05)
    probs[np.arange(n), true] = 0.90
    probs = probs / probs.sum(axis=1, keepdims=True)

    res = estimate_transition_matrix(probs, labels, conditions)
    assert res.conditions == conditions
    # Orientation: rows (true) sum to 1.
    np.testing.assert_allclose(res.T.sum(axis=1), 1.0, atol=1e-6)
    # Recovered close to the known matrix.
    assert np.abs(res.T - t_true).max() < 0.12
    # Each true class's largest mass is on its own diagonal.
    assert np.all(np.argmax(res.T, axis=1) == np.arange(3))
    # noise_matrix is the transpose of T.
    np.testing.assert_allclose(res.noise_matrix.T, res.T, atol=1e-9)


def test_load_hoc_matrix_and_compare_elementwise(tmp_path) -> None:
    conds = ["bipolar", "depression"]
    hoc = pd.DataFrame([[0.95, 0.05], [0.10, 0.90]],
                       index=["true_bipolar", "true_depression"],
                       columns=["noisy_bipolar", "noisy_depression"])
    path = tmp_path / "hoc_mean_T.csv"
    hoc.to_csv(path)
    hoc_conditions, hoc_T = load_hoc_matrix(path)
    assert hoc_conditions == conds

    res = CleanlabResult(
        conditions=conds,
        T=np.array([[0.80, 0.20], [0.15, 0.85]]),
        noise_matrix=np.array([[0.80, 0.15], [0.20, 0.85]]),
        py=np.array([0.5, 0.5]),
        confident_joint=np.array([[1, 1], [1, 1]]),
    )
    cmp = compare_elementwise(res, hoc_conditions, hoc_T)
    assert len(cmp) == 4
    bip = cmp[(cmp["true"] == "bipolar") & (cmp["noisy"] == "bipolar")].iloc[0]
    assert bip["cleanlab"] == pytest.approx(0.80)
    assert bip["hoc"] == pytest.approx(0.95)
    assert bip["diff"] == pytest.approx(0.80 - 0.95)
