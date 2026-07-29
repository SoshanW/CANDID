"""cleanlab (Confident Learning) transition-matrix estimate (C1, second estimator).

The C1 diagnostic needs a second, independent no-clean-data estimator to sit beside
HOC (D-037). This runs cleanlab's confident-joint estimation on the author-grouped
out-of-sample probabilities from :mod:`src.noise.oos_probabilities` and the noisy
proxy labels, producing a 4x4 transition matrix in the same sorted condition order
as HOC (bipolar, depression, eating_disorder, schizophrenia).

The key question, stated explicitly: does cleanlab also return a near-identity
bipolar row (agreeing with HOC that there is almost no estimated noise), or does it
disagree? If two independent estimators fail identically on this data, that is the
strongest form of the C1 result (D-037). Both outcomes are informative; the output
is reported raw and is NOT smoothed or clipped.

Verified against the installed cleanlab (2.9.0, the v2 API):
``estimate_py_and_noise_matrices_from_probabilities(labels, pred_probs, *,
calibrate=True, ...)`` returns ``(py, noise_matrix, inverse_noise_matrix,
confident_joint)`` and assumes ``pred_probs`` are OUT-OF-SAMPLE (hence the CV).

DECISION (orientation transpose): cleanlab's ``noise_matrix`` is
``P(label=k_s | true_label=k_y)`` with entry ``[noisy][true]`` and columns (true)
summing to 1. HOC's ``T[i][j] = P(noisy=j | true=i)`` has rows (true) summing to 1.
So the HOC-comparable matrix is ``T = noise_matrix.T``. We assert its rows sum to 1.

DECISION (4 present classes, sorted): the baseline model has 6 output columns
(``POC_CONDITIONS``) but only 4 are populated. We select the 4 present classes in
sorted order, renormalise each row, and map labels to 0..3 in that order, so the
result lines up cell-for-cell with HOC's 4x4.

Torch-free: numpy + cleanlab + pandas only; runs locally on the cached OOS probs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..modeling.config import ArtifactPaths
from .oos_probabilities import _default_oos_dir, load_oos


@dataclass(frozen=True)
class CleanlabConfig:
    """Knobs for the cleanlab estimate.

    Attributes:
        calibrate: cleanlab's confident-joint calibration (its default is True).
    """

    calibrate: bool = True


@dataclass(frozen=True)
class CleanlabResult:
    """cleanlab estimate in HOC's orientation.

    Attributes:
        conditions: class labels in row/col order (sorted; row=true, col=noisy).
        T: ``(K, K)`` transition matrix ``T[i][j] = P(noisy=j | true=i)`` (rows sum
            to 1), i.e. ``noise_matrix.T``, directly comparable to HOC.
        noise_matrix: cleanlab's raw ``P(noisy | true)`` with entry ``[noisy][true]``.
        py: estimated true-class prior.
        confident_joint: ``(K, K)`` confident-joint counts.
    """

    conditions: list[str]
    T: np.ndarray
    noise_matrix: np.ndarray
    py: np.ndarray
    confident_joint: np.ndarray


def reduce_probs_to_classes(
    probs: np.ndarray, class_names_in_probs: list[str], target_conditions: list[str]
) -> np.ndarray:
    """Select ``target_conditions`` columns from ``probs`` and renormalise each row.

    ``probs`` columns are in ``class_names_in_probs`` order (the model's 6 outputs);
    we keep the target columns in target order and renormalise so each row is a
    distribution over just those classes.
    """
    col = {c: i for i, c in enumerate(class_names_in_probs)}
    missing = [c for c in target_conditions if c not in col]
    if missing:
        raise ValueError(f"probs lack columns for {missing}; have {class_names_in_probs}.")
    idx = [col[c] for c in target_conditions]
    reduced = np.asarray(probs, dtype=np.float64)[:, idx]
    row_sums = reduced.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # guard; softmax over the present classes is > 0
    return reduced / row_sums


def estimate_transition_matrix(
    probs: np.ndarray,
    labels: np.ndarray | pd.Series,
    class_names_in_probs: list[str],
    config: CleanlabConfig = CleanlabConfig(),
) -> CleanlabResult:
    """Run cleanlab's confident-joint estimate; return T in HOC's orientation."""
    from cleanlab.count import estimate_py_and_noise_matrices_from_probabilities

    labels = np.asarray(labels).astype(str)
    conditions = sorted(set(labels.tolist()))
    idx = {c: i for i, c in enumerate(conditions)}
    labels_int = np.array([idx[c] for c in labels], dtype=np.int64)

    pred_probs = reduce_probs_to_classes(probs, class_names_in_probs, conditions)

    py, noise_matrix, _inv, confident_joint = estimate_py_and_noise_matrices_from_probabilities(
        labels_int, pred_probs, calibrate=config.calibrate
    )
    # noise_matrix[noisy][true] -> transpose to T[true][noisy] (HOC orientation).
    t_mat = np.asarray(noise_matrix).T
    row_sums = t_mat.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise AssertionError(f"Transposed T rows should sum to 1; got {row_sums}.")

    return CleanlabResult(
        conditions=conditions,
        T=t_mat,
        noise_matrix=np.asarray(noise_matrix),
        py=np.asarray(py),
        confident_joint=np.asarray(confident_joint),
    )


def load_hoc_matrix(hoc_csv: Path | str) -> tuple[list[str], np.ndarray]:
    """Load a HOC ``hoc_mean_T.csv`` (index ``true_<c>``, cols ``noisy_<c>``)."""
    frame = pd.read_csv(hoc_csv, index_col=0)
    conditions = [i.replace("true_", "") for i in frame.index]
    return conditions, frame.to_numpy(dtype=float)


def compare_elementwise(
    result: CleanlabResult, hoc_conditions: list[str], hoc_T: np.ndarray
) -> pd.DataFrame:
    """Long-format cell-by-cell comparison of cleanlab vs HOC transition matrices."""
    hoc_idx = {c: i for i, c in enumerate(hoc_conditions)}
    rows = []
    for i, tc in enumerate(result.conditions):
        for j, nc in enumerate(result.conditions):
            hoc_val = (
                float(hoc_T[hoc_idx[tc], hoc_idx[nc]])
                if tc in hoc_idx and nc in hoc_idx
                else float("nan")
            )
            cl_val = float(result.T[i, j])
            rows.append(
                {
                    "true": tc,
                    "noisy": nc,
                    "cleanlab": cl_val,
                    "hoc": hoc_val,
                    "diff": cl_val - hoc_val,
                }
            )
    return pd.DataFrame(rows)


def format_comparison(
    result: CleanlabResult, hoc_conditions: list[str] | None, hoc_T: np.ndarray | None
) -> str:
    """Report cleanlab T, the bipolar-row question, and (if given) the HOC diff."""
    conds = result.conditions
    lines = ["=== cleanlab (Confident Learning) transition matrix ===",
             f"conditions (row=true, col=noisy): {conds}",
             ""]
    lines.append(pd.DataFrame(result.T, index=[f"true_{c}" for c in conds],
                              columns=[f"noisy_{c}" for c in conds]).to_string(
                              float_format=lambda v: f"{v:.4f}"))

    if "bipolar" in conds:
        b = conds.index("bipolar")
        cl_diag = float(result.T[b, b])
        lines.append("")
        lines.append("KEY QUESTION (D-037): does cleanlab also return a near-identity bipolar row?")
        lines.append(f"  cleanlab bipolar diagonal (P(noisy=bipolar | true=bipolar)) = {cl_diag:.4f}")
        off = {conds[j]: float(result.T[b, j]) for j in range(len(conds)) if j != b}
        dom = max(off, key=off.get) if off else None
        lines.append(f"  cleanlab bipolar off-diagonal: {off}  (dominant: {dom})")
        if hoc_T is not None and hoc_conditions is not None and "bipolar" in hoc_conditions:
            hoc_diag = float(hoc_T[hoc_conditions.index("bipolar"), hoc_conditions.index("bipolar")])
            verdict = ("AGREES with HOC (both near-identity: little estimated noise)"
                       if cl_diag >= 0.85 else
                       "DISAGREES with HOC (cleanlab estimates materially more bipolar noise)")
            lines.append(f"  HOC bipolar diagonal = {hoc_diag:.4f}  ->  cleanlab {verdict}.")
        lines.append("  (Reported raw; not smoothed or clipped. Both outcomes are informative.)")

    if hoc_T is not None and hoc_conditions is not None:
        lines.append("")
        lines.append("elementwise cleanlab vs HOC (diff = cleanlab - hoc):")
        lines.append(compare_elementwise(result, hoc_conditions, hoc_T).to_string(
            index=False, float_format=lambda v: f"{v:.4f}"))
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="cleanlab confident-joint transition matrix (C1).")
    p.add_argument("--artifacts-root", type=Path, default=None)
    p.add_argument("--oos-dir", type=Path, default=None,
                   help="Dir with oos_probabilities.npy + metadata (default <Models>/oos/<split>).")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--hoc-csv", type=Path, default=None,
                   help="HOC hoc_mean_T.csv for the side-by-side (default "
                        "<Models>/embeddings/<split>/hoc_mean_T.csv).")
    p.add_argument("--no-calibrate", action="store_true", help="Disable cleanlab calibration.")
    p.add_argument("--out-csv", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    artifacts = ArtifactPaths(root=args.artifacts_root) if args.artifacts_root else ArtifactPaths.default()
    oos_dir = Path(args.oos_dir) if args.oos_dir else _default_oos_dir(artifacts, args.split)

    probs, metadata, class_names = load_oos(oos_dir)
    result = estimate_transition_matrix(
        probs, metadata["condition"].to_numpy(), class_names,
        CleanlabConfig(calibrate=not args.no_calibrate),
    )

    hoc_csv = args.hoc_csv or (artifacts.root / "embeddings" / args.split / "hoc_mean_T.csv")
    hoc_conditions, hoc_T = (None, None)
    if Path(hoc_csv).exists():
        hoc_conditions, hoc_T = load_hoc_matrix(hoc_csv)
    else:
        print(f"[note] HOC matrix not found at {hoc_csv}; reporting cleanlab alone.")

    print(format_comparison(result, hoc_conditions, hoc_T))

    out_csv = args.out_csv or (oos_dir / "cleanlab_T.csv")
    conds = result.conditions
    pd.DataFrame(result.T, index=[f"true_{c}" for c in conds],
                 columns=[f"noisy_{c}" for c in conds]).to_csv(out_csv)
    print(f"\nWrote cleanlab T -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
