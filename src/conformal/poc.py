"""The C2 proof-of-concept harness: run the arms on synthetic data (roadmap §5 steps 5-6).

Each repetition draws, from ``synthetic.generate``:

- a **calibration set with fixed per-condition proxy-label counts**, by default
  ``c1_matrices.CALIBRATION_COUNTS`` (the C0 test split: bipolar 481, depression 11,351,
  eating disorder 1,421, schizophrenia 786). Fixing the counts puts every run at the
  operating point D-042's degeneracy frontier was computed for, and lets ``c(n_k)``
  (a slow Monte Carlo, Sesia et al. eq. (16)) be computed once per condition. Points
  are taken in draw order from a larger pool, so within a condition they are still an
  iid sample of ``X | Ytilde = k``;
- an independent **test set**, on which coverage is measured against the clean label.

Results are reported per condition as the mean over repetitions, with the standard
error of that mean, so a table entry can be compared against a nominal level with its
own uncertainty attached.

Run ``python -m src.conformal.poc`` for the step 5 report.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.conformal.c1_matrices import CALIBRATION_COUNTS, CONDITIONS
from src.conformal.split_conformal import (
    SetMetrics,
    evaluate_sets,
    nonconformity_scores,
    predict_sets,
    standard_thresholds,
)
from src.conformal.synthetic import SyntheticConfig, SyntheticSample, generate

__all__ = [
    "ArmSummary",
    "Draw",
    "draw_calibration_and_test",
    "run_arm",
]

K = len(CONDITIONS)

ThresholdFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
"""Maps calibration ``(scores, y_tilde)`` to thresholds ``tau`` of shape ``(K,)``."""


@dataclass(frozen=True)
class Draw:
    """One repetition's data. ``test.y`` is the clean label used to score coverage."""

    calibration: SyntheticSample
    test: SyntheticSample


def _fixed_count_sample(
    transition: np.ndarray,
    counts: list[int],
    separability: float,
    seed: int,
) -> SyntheticSample:
    """Draw until every proxy-label class has its count, then keep the first ``n_k``."""
    expected_share = np.asarray(transition, dtype=float).T @ SyntheticConfig(
        n=1, transition=transition
    ).clean_prior
    pool = int(1.5 * max(c / s for c, s in zip(counts, expected_share))) + 1_000
    sample = generate(
        SyntheticConfig(n=pool, transition=transition, separability=separability, seed=seed)
    )
    keep = []
    for k, n_k in enumerate(counts):
        idx = np.flatnonzero(sample.y_tilde == k)
        if idx.size < n_k:
            raise RuntimeError(
                f"pool of {pool} gave {idx.size} points with proxy label {k}, need {n_k}"
            )
        keep.append(idx[:n_k])
    keep = np.sort(np.concatenate(keep))
    return SyntheticSample(
        y=sample.y[keep], y_tilde=sample.y_tilde[keep], probs=sample.probs[keep]
    )


def draw_calibration_and_test(
    transition: np.ndarray,
    rep: int,
    counts: list[int] = CALIBRATION_COUNTS,
    n_test: int = 20_000,
    separability: float = 2.0,
    base_seed: int = 0,
) -> Draw:
    """Calibration and test sets for repetition ``rep``, on disjoint seed streams."""
    seed = base_seed + 2 * rep
    calibration = _fixed_count_sample(transition, counts, separability, seed)
    test = generate(
        SyntheticConfig(
            n=n_test, transition=transition, separability=separability, seed=seed + 1
        )
    )
    return Draw(calibration=calibration, test=test)


@dataclass(frozen=True)
class ArmSummary:
    """Mean and standard error over repetitions of each per-condition metric.

    Attributes:
        name: arm label for reports.
        coverage, coverage_se: label-conditional coverage against the clean label.
        mean_size, empty_rate, full_rate: means over repetitions, shape ``(K,)``.
        tau: thresholds of every repetition, shape ``(reps, K)``.
    """

    name: str
    coverage: np.ndarray
    coverage_se: np.ndarray
    mean_size: np.ndarray
    empty_rate: np.ndarray
    full_rate: np.ndarray
    tau: np.ndarray


def run_arm(name: str, thresholds: ThresholdFn, draws: list[Draw]) -> ArmSummary:
    """Calibrate on each draw's calibration set and score its test set."""
    metrics: list[SetMetrics] = []
    taus = []
    for draw in draws:
        tau = thresholds(
            nonconformity_scores(draw.calibration.probs), draw.calibration.y_tilde
        )
        sets = predict_sets(nonconformity_scores(draw.test.probs), tau)
        metrics.append(evaluate_sets(sets, draw.test.y))
        taus.append(tau)
    coverage = np.array([m.coverage for m in metrics])
    reps = len(metrics)
    return ArmSummary(
        name=name,
        coverage=coverage.mean(axis=0),
        coverage_se=coverage.std(axis=0, ddof=1) / np.sqrt(reps) if reps > 1 else np.zeros(K),
        mean_size=np.mean([m.mean_size for m in metrics], axis=0),
        empty_rate=np.mean([m.empty_rate for m in metrics], axis=0),
        full_rate=np.mean([m.full_rate for m in metrics], axis=0),
        tau=np.array(taus),
    )


def format_summary(summary: ArmSummary) -> str:
    lines = [
        f"{summary.name}",
        f"  {'condition':<16}{'coverage':>10}{'± se':>8}{'size':>7}{'empty':>8}{'full':>7}",
    ]
    for k, name in enumerate(CONDITIONS):
        lines.append(
            f"  {name:<16}{summary.coverage[k]:>10.4f}{summary.coverage_se[k]:>8.4f}"
            f"{summary.mean_size[k]:>7.3f}{summary.empty_rate[k]:>8.4f}"
            f"{summary.full_rate[k]:>7.4f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - reporting
    args = _parse_args(argv)
    transition = np.eye(K)
    draws = [
        draw_calibration_and_test(
            transition, rep, n_test=args.n_test, separability=args.separability,
            base_seed=args.seed,
        )
        for rep in range(args.reps)
    ]
    print(
        f"T = I, alpha = {args.alpha}, reps = {args.reps}, calibration counts "
        f"{CALIBRATION_COUNTS}, n_test = {args.n_test}, separability = {args.separability}"
    )
    print(
        format_summary(
            run_arm(
                "arm (a): noise-blind split conformal",
                lambda s, y: standard_thresholds(s, y, args.alpha),
                draws,
            )
        )
    )
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=20_000)
    parser.add_argument("--separability", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
