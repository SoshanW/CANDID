"""Out-of-sample predicted probabilities via author-grouped k-fold CV (C1).

cleanlab's confident-joint estimator (:mod:`src.noise.cleanlab_estimate`) requires
*out-of-sample* predicted probabilities: each post's probability must come from a
model that never trained on that post, or the confident joint overfits and the
transition matrix is meaningless. This module produces those probabilities for the
Reddit training split by k-fold cross-validation, then caches them aligned to
``author_id`` and ``condition`` so nothing can silently misalign downstream.

DECISION (author-grouped folds, NON-NEGOTIABLE): folds are formed with
``sklearn.model_selection.GroupKFold`` keyed on ``author_id``, exactly as the main
train/val/test split is (D-010). A random KFold would NOT error, but an author's
posts would then straddle the fit/predict boundary; the model would recognise that
author's writing style rather than the condition, the held-out probabilities would
be contaminated, and every downstream cleanlab number would be silently wrong. The
test ``tests/test_noise_oos.py`` asserts no ``author_id`` appears on both sides of
any fold.

DECISION (comparable to the baseline): each fold reuses ``build_trainer`` and the
Milestone 0 hyperparameters (:class:`TrainConfig`), the exact path the baseline of
record ran through. NOTE: D-011 documents a class-weighted ``WeightedLossTrainer``,
but no such class exists in the code; ``build_trainer`` uses a plain unweighted
``Trainer``. These folds therefore match the actual (unweighted) baseline, and
D-011 needs reconciling separately.

DECISION (no val selection leakage): each fold trains with best-model-on-val
selection disabled, so the held-out fold never influences the returned model; the
OOS probabilities come from the final-epoch weights of a model that only saw the
fold's training rows.

DECISION (quiet by default): k folds x 3 epochs over ~112k rows is ~40k training
steps, and tqdm redraws the progress bar on every one of them. Launched from Colab
as ``!python -m src.noise.oos_probabilities``, all of that streams into a single
notebook output cell, which grows until the browser tab freezes -- the run itself is
fine, the page is what dies. So this CLI disables tqdm and logs one line every
``--logging-steps`` steps (a few hundred lines for the whole job). Pass
``--progress-bars`` to get the bars back in a terminal, where they cost nothing.

Torch/transformers are imported lazily inside the default fit-predict, so the
config, the fold logic and the assembly/caching stay importable and unit-testable
without the deep-learning stack (a fake fit-predict is injected in tests). The real
run is a thin Colab/Kaggle job that clones the repo and calls
:func:`generate_oos_probabilities`.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.combine import combine_sources
from ..data.config import DataPaths
from ..modeling.config import ArtifactPaths, ModelConfig, TrainConfig
from ..modeling.labels import CONDITION_NAMES

#: A fit-predict step: train on ``train_df``, return ``(len(val_df), n_classes)``
#: predicted probabilities for ``val_df`` in the column order of ``class_names``.
FitPredict = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]

PROBS_FILENAME = "oos_probabilities.npy"
METADATA_FILENAME = "oos_metadata.csv"
CLASSES_FILENAME = "oos_classes.txt"


@dataclass(frozen=True)
class OOSConfig:
    """Knobs for the out-of-sample CV.

    Attributes:
        n_folds: number of GroupKFold folds (k). 3 to start; configurable.
        group_col: the column grouped on. ``author_id`` (D-010) is the only correct
            choice; exposed for tests, not for changing in production.
    """

    n_folds: int = 3
    group_col: str = "author_id"

    def __post_init__(self) -> None:
        if self.n_folds < 2:
            raise ValueError(f"n_folds must be >= 2; got {self.n_folds}.")


def author_grouped_folds(
    df: pd.DataFrame, config: OOSConfig = OOSConfig()
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return ``[(train_idx, val_idx), ...]`` with no group crossing a fold.

    Uses ``GroupKFold`` (deterministic, no shuffle) on ``config.group_col``. Each
    row appears in exactly one validation fold, so the folds partition the data and
    the assembled OOS array is fully populated.
    """
    from sklearn.model_selection import GroupKFold

    n_groups = df[config.group_col].nunique()
    if n_groups < config.n_folds:
        raise ValueError(
            f"Need >= n_folds={config.n_folds} distinct '{config.group_col}' values; "
            f"got {n_groups}."
        )
    gkf = GroupKFold(n_splits=config.n_folds)
    groups = df[config.group_col].to_numpy()
    return list(gkf.split(df, groups=groups))


def assert_no_group_leakage(
    df: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]], group_col: str
) -> None:
    """Raise if any group appears in both the train and val side of any fold."""
    groups = df[group_col].to_numpy()
    for i, (train_idx, val_idx) in enumerate(folds):
        overlap = set(groups[train_idx]) & set(groups[val_idx])
        if overlap:
            raise AssertionError(
                f"Fold {i}: {len(overlap)} '{group_col}' value(s) on both sides "
                f"(e.g. {sorted(overlap)[:3]}). Author leakage; folds are invalid."
            )


def quiet_hf_logging() -> None:
    """Silence transformers' info logs and the hub download bars.

    Complements ``TrainConfig.disable_tqdm``: without this, every fold still
    re-emits the model-loading banners and a download bar per file, which is the
    other half of what floods a Colab output cell. Best-effort -- import failures
    are ignored so a missing optional dependency never breaks the run.
    """
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:  # pragma: no cover - depends on transformers version
        pass
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:  # pragma: no cover - depends on hub version
        pass


def _default_fit_predict(
    model_config: ModelConfig, train_config: TrainConfig
) -> FitPredict:
    """Build the real (torch) fit-predict: fine-tune on the fold, predict on val.

    Reuses ``build_trainer`` (the baseline path) with best-model-on-val selection
    disabled (``save_steps`` large) so val never selects the model, then runs the
    baseline's ``predict_softmax`` on the held-out rows.
    """

    def fit_predict(train_df: pd.DataFrame, val_df: pd.DataFrame) -> np.ndarray:
        import tempfile

        from ..modeling.dataset import TextClassificationDataset
        from ..modeling.hf_model import load_model, load_tokenizer
        from ..modeling.predict import predict_softmax
        from ..modeling.train import build_trainer

        tokenizer = load_tokenizer(model_config.pretrained_name)
        model = load_model(model_config)
        with tempfile.TemporaryDirectory() as tmp:
            # save_steps huge -> resilience mode: best-model-on-val selection OFF, so
            # the held-out fold cannot influence the returned model (no leakage).
            trainer = build_trainer(
                train_df=train_df,
                val_df=val_df,
                model_config=model_config,
                train_config=train_config,
                output_dir=Path(tmp),
                save_steps=10**9,
                tokenizer=tokenizer,
                model=model,
            )
            trainer.train()
            val_ds = TextClassificationDataset(val_df, tokenizer, model_config.max_length)
            return predict_softmax(trainer.model, val_ds, batch_size=train_config.batch_size)

    return fit_predict


def generate_oos_probabilities(
    df: pd.DataFrame,
    model_config: ModelConfig = ModelConfig(),
    train_config: TrainConfig = TrainConfig(),
    oos_config: OOSConfig = OOSConfig(),
    cache_dir: Path | None = None,
    fit_predict: FitPredict | None = None,
    class_names: list[str] = CONDITION_NAMES,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Produce and cache out-of-sample probabilities for every row of ``df``.

    Args:
        df: prepared Reddit training frame (``text``, ``condition``, ``author_id``).
        model_config / train_config: baseline settings, reused per fold.
        oos_config: fold count and grouping column.
        cache_dir: where to write per-fold and final artifacts (created if given).
        fit_predict: injected training step; defaults to the real torch one.
        class_names: column order of the returned probabilities.

    Returns:
        ``(probs, metadata)`` where ``probs`` is ``(len(df), len(class_names))`` in
        ``df`` row order and ``metadata`` carries ``row_index``, ``author_id``,
        ``condition``, ``fold``.
    """
    df = df.reset_index(drop=True)
    folds = author_grouped_folds(df, oos_config)
    assert_no_group_leakage(df, folds, oos_config.group_col)

    if fit_predict is None:
        fit_predict = _default_fit_predict(model_config, train_config)

    n = len(df)
    n_classes = len(class_names)
    probs = np.full((n, n_classes), np.nan, dtype=np.float64)
    fold_assign = np.full(n, -1, dtype=np.int64)

    for i, (train_idx, val_idx) in enumerate(folds):
        val_probs = _run_or_resume_fold(
            i, df, train_idx, val_idx, fit_predict, cache_dir, n_classes
        )
        probs[val_idx] = val_probs
        fold_assign[val_idx] = i
        print(f"[fold {i}] filled {len(val_idx)} held-out rows")

    if np.isnan(probs).any():
        missing = int(np.isnan(probs).any(axis=1).sum())
        raise RuntimeError(f"{missing} rows never received OOS probabilities.")

    metadata = pd.DataFrame(
        {
            "row_index": np.arange(n, dtype="int64"),
            "author_id": df["author_id"].to_numpy() if "author_id" in df else "unknown",
            "condition": df["condition"].to_numpy(),
            "fold": fold_assign,
        }
    )
    if cache_dir is not None:
        _save_oos(Path(cache_dir), probs.astype(np.float32), metadata, class_names)
    return probs, metadata


def _run_or_resume_fold(
    fold: int,
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    fit_predict: FitPredict,
    cache_dir: Path | None,
    n_classes: int,
) -> np.ndarray:
    """Compute (or reload) one fold's held-out probabilities. Resumable: a completed
    fold's probs are cached so a dropped Colab session restarts mid-CV, not from 0."""
    if cache_dir is not None:
        probs_path = Path(cache_dir) / f"fold_{fold}_probs.npy"
        idx_path = Path(cache_dir) / f"fold_{fold}_val_idx.npy"
        if probs_path.exists() and idx_path.exists():
            cached_idx = np.load(idx_path)
            if np.array_equal(cached_idx, val_idx):
                print(f"[fold {fold}] resumed from cache")
                return np.load(probs_path)

    # One timestamped line per fold: with tqdm off this is the coarse heartbeat that
    # says the job is alive, and it survives in a log file after the session drops.
    start = time.time()
    print(
        f"[fold {fold}] start {time.strftime('%H:%M:%S')} -- "
        f"train {len(train_idx)} rows, predict {len(val_idx)} rows",
        flush=True,
    )
    val_probs = fit_predict(df.iloc[train_idx], df.iloc[val_idx])
    print(f"[fold {fold}] done in {(time.time() - start) / 60:.1f} min", flush=True)
    val_probs = np.asarray(val_probs, dtype=np.float64)
    if val_probs.shape != (len(val_idx), n_classes):
        raise ValueError(
            f"Fold {fold}: fit_predict returned {val_probs.shape}, expected "
            f"{(len(val_idx), n_classes)}."
        )
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.save(Path(cache_dir) / f"fold_{fold}_probs.npy", val_probs)
        np.save(Path(cache_dir) / f"fold_{fold}_val_idx.npy", val_idx)
    return val_probs


def _save_oos(
    cache_dir: Path, probs: np.ndarray, metadata: pd.DataFrame, class_names: list[str]
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / PROBS_FILENAME, probs)
    metadata.to_csv(cache_dir / METADATA_FILENAME, index=False)
    (cache_dir / CLASSES_FILENAME).write_text("\n".join(class_names) + "\n", encoding="utf-8")


def load_oos(cache_dir: Path | str) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Reload ``(probs, metadata, class_names)`` written by a previous run."""
    cache_dir = Path(cache_dir)
    probs_path = cache_dir / PROBS_FILENAME
    meta_path = cache_dir / METADATA_FILENAME
    classes_path = cache_dir / CLASSES_FILENAME
    if not (probs_path.exists() and meta_path.exists() and classes_path.exists()):
        raise FileNotFoundError(
            f"Expected {PROBS_FILENAME}, {METADATA_FILENAME} and {CLASSES_FILENAME} "
            f"under {cache_dir}; run `python -m src.noise.oos_probabilities` first."
        )
    probs = np.load(probs_path)
    metadata = pd.read_csv(meta_path)
    class_names = classes_path.read_text(encoding="utf-8").split()
    if len(probs) != len(metadata):
        raise ValueError("OOS probs/metadata length mismatch.")
    return probs, metadata, class_names


def _default_oos_dir(artifacts: ArtifactPaths, split: str = "train") -> Path:
    return artifacts.root / "oos" / split


def _load_training_frame(data_root: Path | None) -> pd.DataFrame:
    """Rebuild the exact POC training split the baseline used (author-grouped)."""
    from ..modeling.splits import build_poc_splits
    from ..modeling.train import _load_combined

    paths = DataPaths(data_root=data_root) if data_root else DataPaths.default()
    combined = _load_combined(paths, include_interviewer=False)
    return build_poc_splits(combined).train


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Author-grouped OOS probabilities for cleanlab (C1).")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--artifacts-root", type=Path, default=None)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--n-folds", type=int, default=OOSConfig().n_folds)
    p.add_argument("--num-epochs", type=int, default=TrainConfig().num_epochs)
    p.add_argument("--batch-size", type=int, default=TrainConfig().batch_size)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--model-name", type=str, default=ModelConfig().pretrained_name)
    p.add_argument("--logging-steps", type=int, default=200,
                   help="Steps between loss log lines (higher = quieter; default 200).")
    p.add_argument("--progress-bars", action="store_true",
                   help="Re-enable tqdm/HF bars. Terminal only -- they freeze Colab output cells.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    artifacts = ArtifactPaths(root=args.artifacts_root) if args.artifacts_root else ArtifactPaths.default()

    # Prefer the persisted split (matches training exactly); else rebuild it.
    split_csv = artifacts.splits_dir / f"{args.split}.csv"
    if split_csv.exists():
        df = pd.read_csv(split_csv)
        print(f"Loaded persisted split -> {split_csv}  ({len(df)} rows)")
    else:
        df = _load_training_frame(args.data_root)
        print(f"Rebuilt {args.split} split from data ({len(df)} rows)")

    if not args.progress_bars:
        quiet_hf_logging()

    model_config = ModelConfig(pretrained_name=args.model_name)
    train_config = TrainConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        disable_tqdm=not args.progress_bars,
    )
    oos_config = OOSConfig(n_folds=args.n_folds)
    cache_dir = _default_oos_dir(artifacts, args.split)

    probs, metadata = generate_oos_probabilities(
        df, model_config, train_config, oos_config, cache_dir=cache_dir
    )
    print(f"\nWrote OOS probabilities {probs.shape} -> {cache_dir}")
    print(metadata["fold"].value_counts().sort_index())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
