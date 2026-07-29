"""Tests for author-grouped OOS CV (torch-free; a fake fit-predict is injected).

The load-bearing test is that no author appears on both sides of any fold: a random
KFold would silently leak writing style and contaminate the OOS probabilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modeling.labels import CONDITION_NAMES, CONDITION_TO_ID
from src.noise.oos_probabilities import (
    OOSConfig,
    assert_no_group_leakage,
    author_grouped_folds,
    generate_oos_probabilities,
    load_oos,
)


def _frame() -> pd.DataFrame:
    # 12 authors x 2 posts each; conditions drawn from the present POC classes.
    conds = ["bipolar", "depression", "eating_disorder", "schizophrenia"]
    rows = []
    for a in range(12):
        for _ in range(2):
            rows.append({"text": f"t{a}", "author_id": f"user_{a}", "condition": conds[a % 4]})
    return pd.DataFrame(rows)


def test_no_author_appears_on_both_sides_of_a_fold() -> None:
    df = _frame()
    folds = author_grouped_folds(df, OOSConfig(n_folds=3))
    groups = df["author_id"].to_numpy()
    for train_idx, val_idx in folds:
        assert not (set(groups[train_idx]) & set(groups[val_idx]))
    # And the dedicated guard does not raise on valid folds.
    assert_no_group_leakage(df, folds, "author_id")


def test_assert_no_group_leakage_catches_a_leak() -> None:
    df = _frame()
    groups = df["author_id"].to_numpy()
    a0 = np.where(groups == "user_0")[0]
    # Deliberately put user_0 on both sides.
    bad = [(np.array([a0[0]]), np.array([a0[1]]))]
    with pytest.raises(AssertionError, match="both sides"):
        assert_no_group_leakage(df, bad, "author_id")


def test_oos_config_validation() -> None:
    with pytest.raises(ValueError, match="n_folds"):
        OOSConfig(n_folds=1)


def _fake_fit_predict(train_df: pd.DataFrame, val_df: pd.DataFrame) -> np.ndarray:
    """Return a near-one-hot toward each val row's own condition, so assembled probs
    can be checked for correct row alignment."""
    n_classes = len(CONDITION_NAMES)
    out = np.full((len(val_df), n_classes), 0.01)
    for k, cond in enumerate(val_df["condition"].to_numpy()):
        out[k, CONDITION_TO_ID[cond]] = 0.9
    return out / out.sum(axis=1, keepdims=True)


def test_generate_oos_alignment_coverage_and_cache(tmp_path) -> None:
    df = _frame()
    probs, meta = generate_oos_probabilities(
        df, oos_config=OOSConfig(n_folds=3), cache_dir=tmp_path, fit_predict=_fake_fit_predict
    )
    # Every row filled, folds partition the data.
    assert not np.isnan(probs).any()
    assert set(meta["fold"]) == {0, 1, 2}
    assert (meta["condition"].to_numpy() == df["condition"].to_numpy()).all()
    # Alignment preserved: each row's argmax is its own condition id.
    for i in range(len(df)):
        assert probs[i].argmax() == CONDITION_TO_ID[df.iloc[i]["condition"]]
    # Cache round-trip.
    p2, m2, classes = load_oos(tmp_path)
    assert classes == CONDITION_NAMES
    np.testing.assert_allclose(p2, probs.astype(np.float32), rtol=1e-6)


def test_generate_oos_resumes_from_cache(tmp_path) -> None:
    df = _frame()
    calls = {"n": 0}

    def counting_fp(train_df, val_df):
        calls["n"] += 1
        return _fake_fit_predict(train_df, val_df)

    generate_oos_probabilities(df, oos_config=OOSConfig(n_folds=3),
                               cache_dir=tmp_path, fit_predict=counting_fp)
    after_first = calls["n"]
    assert after_first == 3  # one fit per fold
    # Second run finds every fold cached and does not re-train.
    generate_oos_probabilities(df, oos_config=OOSConfig(n_folds=3),
                               cache_dir=tmp_path, fit_predict=counting_fp)
    assert calls["n"] == after_first
