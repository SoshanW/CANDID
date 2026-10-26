"""Synthetic K = 4 screening problem with a known transition matrix (roadmap §5 step 4).

The C2 proof-of-concept tests the *mechanism* of Sesia, Wang and Tong (2025) on data
where the truth is known, which the real Reddit data can never supply. This module
draws that data:

1. a clean condition ``Y`` from the clean prior, by default ``c1_matrices.NOISY_PRIOR``
   (the observed proxy-label frequencies, used here as a realistic class balance:
   about 0.81 depression and 0.04 bipolar);
2. a softmax-like score vector ``pi_hat(x, .)`` whose informativeness is set by one
   ``separability`` parameter: logits are ``separability * onehot(Y)`` plus independent
   standard normal noise, passed through a softmax;
3. a proxy label ``Ytilde`` drawn from row ``Y`` of a known transition matrix
   ``T[l, k] = P[Ytilde = k | Y = l]``, independently of the scores given ``Y``.

Step 3 makes Sesia et al.'s Assumption 1 (``Ytilde`` independent of ``X`` given ``Y``)
hold by construction. The real data may violate it (``c2-interface.md`` §8); this
generator deliberately does not, so that a failure in the arms cannot be blamed on it.

``make_transition`` builds the controllable ``T`` the roadmap asks for: bipolar to
depression is the dominant off-diagonal, with an optional uniform background on every
other off-diagonal entry. Orientation follows the project convention (``src/CLAUDE.md``):
rows are the true condition, in ``CONDITIONS`` order, and sum to one.

Everything is seeded through ``numpy.random.default_rng``. Light module: numpy only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from src.conformal.c1_matrices import CONDITIONS, NOISY_PRIOR
from src.conformal.v_region import transition_to_v

__all__ = [
    "SyntheticConfig",
    "SyntheticSample",
    "generate",
    "make_transition",
    "recover_clean_prior",
]

K = len(CONDITIONS)
BIPOLAR = CONDITIONS.index("bipolar")
DEPRESSION = CONDITIONS.index("depression")


def make_transition(bipolar_to_depression: float = 0.15, background: float = 0.0) -> np.ndarray:
    """A transition matrix whose dominant off-diagonal is bipolar to depression.

    Args:
        bipolar_to_depression: ``T[bipolar, depression]``, the share of truly bipolar
            authors whose posts carry the depression label. The roadmap's "truth" for
            the wrong-point sub-run of step 6 is 0.15.
        background: value of every *other* off-diagonal entry. Must be strictly
            smaller than ``bipolar_to_depression`` (or both zero) so that bipolar to
            depression stays the dominant off-diagonal.

    Returns:
        ``T`` with rows in ``CONDITIONS`` order, each summing to one.

    Raises:
        ValueError: if an entry leaves [0, 1], a diagonal is not positive, or bipolar
            to depression would not be dominant.
    """
    if not 0.0 <= background <= 1.0 or not 0.0 <= bipolar_to_depression <= 1.0:
        raise ValueError(
            f"entries must lie in [0, 1], got bipolar_to_depression="
            f"{bipolar_to_depression}, background={background}"
        )
    if bipolar_to_depression > 0.0 and background >= bipolar_to_depression:
        raise ValueError(
            f"background={background} must be below bipolar_to_depression="
            f"{bipolar_to_depression} for bipolar to depression to be dominant"
        )
    transition = np.full((K, K), background)
    transition[BIPOLAR, DEPRESSION] = bipolar_to_depression
    np.fill_diagonal(transition, 0.0)
    diagonal = 1.0 - transition.sum(axis=1)
    if (diagonal <= 0.0).any():
        raise ValueError(f"off-diagonal mass leaves a non-positive diagonal: {diagonal}")
    transition[np.diag_indices(K)] = diagonal
    return transition


@dataclass(frozen=True)
class SyntheticConfig:
    """Parameters of one synthetic draw.

    Attributes:
        n: number of samples.
        transition: true ``T[l, k] = P[Ytilde = k | Y = l]``; rows sum to one.
        clean_prior: class frequencies of the clean label ``Y``, ``CONDITIONS`` order.
        separability: logit margin of the true class over the others. 0 gives an
            uninformative scorer; larger values give sharper, more accurate scores.
        seed: seed for ``numpy.random.default_rng``.
    """

    n: int
    transition: np.ndarray
    clean_prior: np.ndarray = field(default_factory=NOISY_PRIOR.copy)
    separability: float = 2.0
    seed: int = 0

    def __post_init__(self) -> None:
        transition = np.asarray(self.transition, dtype=float)
        prior = np.asarray(self.clean_prior, dtype=float)
        if self.n <= 0:
            raise ValueError(f"n must be positive, got {self.n}")
        if transition.shape != (K, K):
            raise ValueError(f"transition must be {K}x{K}, got {transition.shape}")
        if (transition < 0).any() or not np.allclose(transition.sum(axis=1), 1.0):
            raise ValueError(f"transition rows must be distributions, got {transition}")
        if prior.shape != (K,) or (prior < 0).any() or not np.isclose(prior.sum(), 1.0):
            raise ValueError(f"clean_prior must be a distribution over {K}, got {prior}")
        if self.separability < 0:
            raise ValueError(f"separability must be non-negative, got {self.separability}")


@dataclass(frozen=True)
class SyntheticSample:
    """One synthetic draw.

    Attributes:
        y: clean condition, shape ``(n,)``. Never visible to a method under test.
        y_tilde: proxy label after corruption through ``T``, shape ``(n,)``.
        probs: softmax-like scores ``pi_hat(x, k)``, shape ``(n, K)``, rows sum to one.
    """

    y: np.ndarray
    y_tilde: np.ndarray
    probs: np.ndarray


def generate(config: SyntheticConfig) -> SyntheticSample:
    """Draw clean labels, scores, and proxy labels corrupted through ``T``."""
    rng = np.random.default_rng(config.seed)
    transition = np.asarray(config.transition, dtype=float)
    prior = np.asarray(config.clean_prior, dtype=float)

    y = rng.choice(K, size=config.n, p=prior / prior.sum())

    logits = rng.standard_normal((config.n, K))
    logits[np.arange(config.n), y] += config.separability
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)

    # Inverse-CDF draw from row y of T, independent of the scores given y.
    cumulative = np.cumsum(transition, axis=1)
    cumulative[:, -1] = 1.0
    u = rng.random(config.n)
    y_tilde = (u[:, None] >= cumulative[y]).sum(axis=1)

    return SyntheticSample(y=y, y_tilde=y_tilde, probs=probs)


def recover_clean_prior(y_tilde: np.ndarray, transition: np.ndarray) -> np.ndarray:
    """D-028's conversion on observed labels: ``rho = (T^T)^-1 rho_tilde``.

    Delegates to ``v_region.transition_to_v`` so that the code under test is the code
    every C2 number already depends on, not a re-derivation.
    """
    rho_tilde = np.bincount(y_tilde, minlength=K) / len(y_tilde)
    return transition_to_v(transition, rho_tilde)[2]


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - reporting
    args = _parse_args(argv)
    transition = make_transition(args.bipolar_to_depression, args.background)
    sample = generate(
        SyntheticConfig(
            n=args.n, transition=transition, separability=args.separability, seed=args.seed
        )
    )
    observed = np.bincount(sample.y_tilde, minlength=K) / args.n
    recovered = recover_clean_prior(sample.y_tilde, transition)
    accuracy = float((sample.probs.argmax(axis=1) == sample.y).mean())
    print(f"n = {args.n}, separability = {args.separability}, seed = {args.seed}")
    print(f"argmax accuracy against the clean label: {accuracy:.4f}")
    print(f"{'condition':<16}{'true prior':>12}{'noisy':>10}{'recovered':>11}")
    for k, name in enumerate(CONDITIONS):
        print(f"{name:<16}{NOISY_PRIOR[k]:>12.4f}{observed[k]:>10.4f}{recovered[k]:>11.4f}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=200_000)
    parser.add_argument("--bipolar-to-depression", type=float, default=0.15)
    parser.add_argument("--background", type=float, default=0.0)
    parser.add_argument("--separability", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
