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

Run ``python -m src.conformal.poc`` for the step 5 report and
``python -m src.conformal.poc --step 6`` for step 6 (configuration fixed in the roadmap
before the first run, commit ``951816d``).
"""

from __future__ import annotations

import argparse
import functools
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.conformal.adaptive import (
    InflationReference,
    adaptive_thresholds,
    proposition_three_check,
    zero_width_box,
)
from src.conformal.c1_matrices import CALIBRATION_COUNTS, CONDITIONS, NOISY_PRIOR
from src.conformal.split_conformal import (
    SetMetrics,
    evaluate_sets,
    nonconformity_scores,
    predict_sets,
    standard_thresholds,
)
from src.conformal.synthetic import (
    SyntheticConfig,
    SyntheticSample,
    generate,
    make_transition,
)
from src.conformal.v_region import expected_sup_deviation, transition_to_v

__all__ = [
    "ArmSummary",
    "Draw",
    "PropositionThreeSummary",
    "Step6Result",
    "draw_calibration_and_test",
    "run_arm",
    "run_step6",
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
    clean_prior: np.ndarray = NOISY_PRIOR,
) -> SyntheticSample:
    """Draw until every proxy-label class has its count, then keep the first ``n_k``."""
    expected_share = np.asarray(transition, dtype=float).T @ np.asarray(clean_prior, dtype=float)
    pool = int(1.5 * max(c / s for c, s in zip(counts, expected_share))) + 1_000
    sample = generate(
        SyntheticConfig(
            n=pool, transition=transition, clean_prior=clean_prior,
            separability=separability, seed=seed,
        )
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
    clean_prior: np.ndarray = NOISY_PRIOR,
) -> Draw:
    """Calibration and test sets for repetition ``rep``, on disjoint seed streams.

    ``clean_prior`` defaults to ``NOISY_PRIOR`` (steps 5 and 6). D-043's arm (c) runs
    pass the prior recovered from each region's synthetic truth instead.
    """
    seed = base_seed + 2 * rep
    calibration = _fixed_count_sample(transition, counts, separability, seed, clean_prior)
    test = generate(
        SyntheticConfig(
            n=n_test, transition=transition, clean_prior=clean_prior,
            separability=separability, seed=seed + 1,
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


@functools.lru_cache(maxsize=None)
def c_for_counts(counts: tuple[int, ...]) -> dict[int, float]:
    """``c(n_k)`` of eq. (16) for each calibration count, computed once per process."""
    return {n: expected_sup_deviation(n) for n in counts}


STEP6_TRUTH = (0.15, 0.0)
"""Bipolar to depression 15%, no other off-diagonal (roadmap step 6, fixed in 951816d)."""

STEP6_WRONG_POINT = (0.046, 0.0)
"""The HOC-like wrong point estimate: bipolar to depression 4.6%."""


@dataclass(frozen=True)
class PropositionThreeSummary:
    """Proposition 3's condition across repetitions, per condition (D-049).

    Attributes:
        name: which point estimate the optimistic variant was given.
        holds_as_stated: share of repetitions where the printed condition holds.
        holds_at_threshold: share where the at-threshold form holds.
        inf_delta: ``inf_t Delta_k(t)``, shape ``(K,)``; does not vary by repetition.
        mean_at_threshold: mean ``Delta_k(tau_k)`` over repetitions.
        mean_bound: mean ``delta_ci - (1 - alpha) / n_k`` over repetitions.
    """

    name: str
    holds_as_stated: np.ndarray
    holds_at_threshold: np.ndarray
    inf_delta: np.ndarray
    mean_at_threshold: np.ndarray
    mean_bound: np.ndarray


@dataclass(frozen=True)
class Step6Result:
    """Every arm of step 6 and the Proposition 3 diagnostics."""

    arms: list[ArmSummary]
    proposition_three: list[PropositionThreeSummary]
    v_points: dict[str, np.ndarray]


def run_step6(
    reps: int = 100,
    alpha: float = 0.10,
    n_test: int = 20_000,
    separability: float = 2.0,
    base_seed: int = 0,
    counts: list[int] = CALIBRATION_COUNTS,
    reference_n: int = 2_000_000,
) -> Step6Result:
    """Arm (a), and arm (b) given the oracle and the wrong point, in both variants."""
    truth = make_transition(*STEP6_TRUTH)
    # Population proxy frequencies; the fixed-count calibration set's are not used.
    noisy_prior = truth.T @ NOISY_PRIOR
    points = {
        "oracle": transition_to_v(truth, noisy_prior)[0],
        "wrong point": transition_to_v(make_transition(*STEP6_WRONG_POINT), noisy_prior)[0],
    }
    c_n_k = c_for_counts(tuple(counts))
    draws = [
        draw_calibration_and_test(
            truth, rep, counts=counts, n_test=n_test, separability=separability,
            base_seed=base_seed,
        )
        for rep in range(reps)
    ]

    arms = [run_arm("arm (a)", lambda s, y: standard_thresholds(s, y, alpha), draws)]
    for name, v in points.items():
        low, upp = zero_width_box(v)
        for variant in ("conservative", "optimistic"):
            arms.append(
                run_arm(
                    f"arm (b) {name}, {variant}",
                    lambda s, y, low=low, upp=upp, variant=variant: adaptive_thresholds(
                        s, y, low, upp, alpha, variant=variant, c_n_k=c_n_k
                    ).tau,
                    draws,
                )
            )

    reference_sample = generate(
        SyntheticConfig(
            n=reference_n, transition=truth, separability=separability,
            seed=base_seed + 10**6,
        )
    )
    reference = InflationReference(
        nonconformity_scores(reference_sample.probs),
        reference_sample.y,
        reference_sample.y_tilde,
    )
    proposition = []
    for name, v in points.items():
        low, upp = zero_width_box(v)
        checks = [
            proposition_three_check(
                reference,
                adaptive_thresholds(
                    nonconformity_scores(d.calibration.probs), d.calibration.y_tilde,
                    low, upp, alpha, variant="optimistic", c_n_k=c_n_k,
                ),
                alpha,
            )
            for d in draws
        ]
        proposition.append(
            PropositionThreeSummary(
                name=name,
                holds_as_stated=np.mean([[c.holds_as_stated for c in r] for r in checks], axis=0),
                holds_at_threshold=np.mean(
                    [[c.holds_at_threshold for c in r] for r in checks], axis=0
                ),
                inf_delta=np.array([c.inf_delta for c in checks[0]]),
                mean_at_threshold=np.mean([[c.at_threshold for c in r] for r in checks], axis=0),
                mean_bound=np.mean([[c.bound for c in r] for r in checks], axis=0),
            )
        )
    return Step6Result(arms=arms, proposition_three=proposition, v_points=points)


def format_proposition_three(summary: PropositionThreeSummary) -> str:
    lines = [
        f"Proposition 3, optimistic variant given the {summary.name}",
        f"  {'condition':<16}{'inf Delta':>10}{'Delta(tau)':>11}{'bound':>9}"
        f"{'as stated':>11}{'at tau':>8}",
    ]
    for k, name in enumerate(CONDITIONS):
        lines.append(
            f"  {name:<16}{summary.inf_delta[k]:>10.4f}{summary.mean_at_threshold[k]:>11.4f}"
            f"{summary.mean_bound[k]:>9.4f}{summary.holds_as_stated[k]:>11.2f}"
            f"{summary.holds_at_threshold[k]:>8.2f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - reporting
    args = _parse_args(argv)
    if args.step == 6:
        result = run_step6(
            reps=args.reps, alpha=args.alpha, n_test=args.n_test,
            separability=args.separability, base_seed=args.seed,
        )
        print(
            f"Step 6: truth bipolar->depression {STEP6_TRUTH[0]}, wrong point "
            f"{STEP6_WRONG_POINT[0]}, alpha = {args.alpha}, reps = {args.reps}, "
            f"calibration counts {CALIBRATION_COUNTS}, n_test = {args.n_test}, "
            f"separability = {args.separability}"
        )
        for name, v in result.v_points.items():
            print(f"V ({name}):\n{np.array2string(v, precision=4, suppress_small=True)}")
        for summary in result.arms:
            print(format_summary(summary))
        for summary in result.proposition_three:
            print(format_proposition_three(summary))
        return 0
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
    parser.add_argument("--step", type=int, choices=(5, 6), default=5)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=20_000)
    parser.add_argument("--separability", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
