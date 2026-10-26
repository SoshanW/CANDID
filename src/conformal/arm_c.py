"""Arm (c): Algorithm 2 fed a region instead of a point (roadmap §5 steps 8 and 9).

Runs the pre-registration D-043 (commit cf12627) exactly as written and scores every
prediction against its rules. Nothing here chooses a threshold: the run
specification, the outcome definitions, the predicted classes and the scoring rules
are D-043's, transcribed into ``PRIMARY_CERTAIN``, ``SECONDARY`` and the scoring
functions, and cross-checked against the closed form by the tests.

Run specification (D-043 §1):

- regions A and B from ``region_from_matrices``; F1, F2, F3, L1 and R0 from
  ``elicited_region.v_box``; alpha_V = 0;
- primary ``zeta_upp_k = delta_star_k`` (the ``adaptive.py`` default), and a
  sensitivity with ``zeta_upp_k = 0`` that is reported but not scored;
- synthetic truth: ``T_HOC_FINE_TUNED`` for A and B, the midpoint of the T-interval
  box for the scenario regions, with clean prior ``(T^T)^-1 NOISY_PRIOR``;
- harness of steps 5 and 6: calibration counts 481 / 11,351 / 1,421 / 786, 100
  repetitions, 20,000 test points, separability 2.0, base seed 0; a 2 x 10^6-point
  reference sample per region at seed ``base + 10**6``;
- outcome per cell: degenerate if tau_k = 1 in at least 90 of 100 runs, admissible if
  tau_k < 1 in at least 90, split otherwise; valid if coverage >= 0.90 - 3 SE.

Run ``python -m src.conformal.arm_c`` for the full report.

Light module: numpy only.
"""

from __future__ import annotations

import argparse
import functools
import math
from dataclasses import dataclass, field

import numpy as np

from src.conformal.adaptive import (
    InflationReference,
    adaptive_thresholds,
    proposition_three_check,
    zero_width_box,
)
from src.conformal.c1_matrices import (
    CALIBRATION_COUNTS,
    CONDITIONS,
    NOISY_PRIOR,
    REGION_A_MEASURED,
    REGION_B_MEASURED,
    T_CLEANLAB,
    T_HOC_FINE_TUNED,
)
from src.conformal.elicited_region import SCOPING_REGIONS, _transition, v_box
from src.conformal.poc import Draw, c_for_counts, draw_calibration_and_test, run_arm
from src.conformal.split_conformal import (
    evaluate_sets,
    nonconformity_scores,
    predict_sets,
    standard_thresholds,
)
from src.conformal.synthetic import SyntheticConfig, generate
from src.conformal.v_region import correction_term, region_from_matrices, transition_to_v

K = len(CONDITIONS)
ALPHA = 0.10
REPS = 100
N_TEST = 20_000
SEPARABILITY = 2.0
BASE_SEED = 0
REFERENCE_N = 2_000_000
REGION_NAMES = ("A", "B", "F1", "F2", "F3", "L1", "R0")
ALPHA_GRID = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 0.95)
SHORT = ("bipolar", "depression", "eating disorder", "schizophrenia")

# --- D-043 §2, transcribed -----------------------------------------------------------

PRIMARY_CERTAIN = {
    "zeta_star": {
        ("B", 0): "degenerate", ("B", 3): "degenerate", ("L1", 0): "degenerate",
        ("R0", 0): "degenerate", ("A", 1): "admissible",
    },
    "zeta_zero": {
        ("B", 0): "degenerate", ("B", 3): "degenerate", ("L1", 0): "degenerate",
        ("R0", 0): "degenerate", ("A", 1): "admissible", ("B", 1): "admissible",
    },
}
"""Certain cells. Any exception is an implementation error (D-043 §2)."""

SECONDARY = {
    (r, k): "degenerate" for r in REGION_NAMES for k in range(K)
} | {("A", 1): "admissible", ("B", 1): "admissible"}
"""D-043's secondary (judgement) calls, primary zeta. Certain cells included."""

SECONDARY_LOW_CONFIDENCE = {("A", 2), ("B", 1)}

D042_NOT_OK = {("B", 0), ("B", 3), ("F3", 0), ("L1", 0), ("R0", 0), ("R0", 3)}
"""Cells D-042 / D-047 did not call "ok" (robust or borderline degenerate)."""


# --- regions and truths --------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def region_box(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name == "A":
        return region_from_matrices(REGION_A_MEASURED, NOISY_PRIOR)
    if name == "B":
        return region_from_matrices(REGION_B_MEASURED, NOISY_PRIOR)
    lo, hi, _ = v_box(SCOPING_REGIONS[name], NOISY_PRIOR)
    return lo, hi


def normalised(transition: np.ndarray) -> np.ndarray:
    """Rows renormalised to sum to one, as ``transition_to_v`` does internally.

    The C1 matrices are transcribed to four decimals, so their rows sum to one only to
    within 1e-4; the regions' V-boxes were built from the renormalised rows, so the
    synthetic truth uses the same ones.
    """
    transition = np.asarray(transition, dtype=float)
    return transition / transition.sum(axis=1, keepdims=True)


def region_truth(name: str) -> np.ndarray:
    if name in ("A", "B"):
        return normalised(T_HOC_FINE_TUNED)
    lo, hi = SCOPING_REGIONS[name].bounds()
    return _transition((lo + hi) / 2)


def recovered_prior(transition: np.ndarray) -> np.ndarray:
    return transition_to_v(transition, NOISY_PRIOR)[2]


def zeta_vector(mode: str) -> np.ndarray | None:
    """``None`` selects the adaptive.py default (zeta_upp = delta_star); else zeros."""
    return None if mode == "zeta_star" else np.zeros(K)


# --- closed form (D-043 §2 and §3) ---------------------------------------------------

@dataclass(frozen=True)
class ClosedForm:
    delta_ci: float
    d_max: float
    delta_min: float

    def primary_class(self, alpha: float = ALPHA) -> str:
        if self.delta_ci - alpha > self.d_max:
            return "certain-degenerate"
        if alpha - self.delta_ci >= -self.delta_min:
            return "certain-ok"
        return "undetermined"


def closed_form(name: str, k: int, zeta_mode: str = "zeta_star") -> ClosedForm:
    lo, hi = region_box(name)
    c = c_for_counts(tuple(CALIBRATION_COUNTS))
    n_star = min(CALIBRATION_COUNTS)
    others = [l for l in range(K) if l != k]
    delta = correction_term(
        k, lo, hi, CALIBRATION_COUNTS[k], n_star, c_n_k=c[CALIBRATION_COUNTS[k]]
    )
    d_max = sum(abs(hi[k, l]) for l in others)
    d_star = max(hi[k, l] - lo[k, l] for l in others)
    pos = sum(max(hi[k, l], 0.0) for l in others)
    zeta = d_star if zeta_mode == "zeta_star" else 0.0
    return ClosedForm(delta, d_max, -pos - (K - 1) * (d_star + zeta))


def secondary_flip_low_end(name: str, k: int) -> float:
    lo, hi = region_box(name)
    cf = closed_form(name, k)
    others = [l for l in range(K) if l != k]
    d_star = max(hi[k, l] - lo[k, l] for l in others)
    pos = sum(max(hi[k, l], 0.0) for l in others)
    return cf.delta_ci - cf.d_max + pos + 0.4 * (K - 1) * 2 * d_star


# --- runs ----------------------------------------------------------------------------

def draws_for(transition: np.ndarray, reps: int = REPS, n_test: int = N_TEST) -> list[Draw]:
    prior = recovered_prior(transition)
    return [
        draw_calibration_and_test(
            transition, rep, n_test=n_test, separability=SEPARABILITY,
            base_seed=BASE_SEED, clean_prior=prior,
        )
        for rep in range(reps)
    ]


def reference_for(transition: np.ndarray, n: int = REFERENCE_N) -> tuple[InflationReference, object]:
    sample = generate(
        SyntheticConfig(
            n=n, transition=transition, clean_prior=recovered_prior(transition),
            separability=SEPARABILITY, seed=BASE_SEED + 10**6,
        )
    )
    scores = nonconformity_scores(sample.probs)
    return InflationReference(scores, sample.y, sample.y_tilde), (scores, sample.y_tilde)


def calibrations(draws, low, upp, variant, zeta, alpha=ALPHA):
    c = c_for_counts(tuple(CALIBRATION_COUNTS))
    return [
        adaptive_thresholds(
            nonconformity_scores(d.calibration.probs), d.calibration.y_tilde, low, upp,
            alpha, variant=variant, zeta_upp=zeta, c_n_k=c,
        )
        for d in draws
    ]


def outcome(tau_runs: np.ndarray) -> str:
    degenerate = int((tau_runs >= 1.0).sum())
    if degenerate >= 90:
        return "degenerate"
    if len(tau_runs) - degenerate >= 90:
        return "admissible"
    return "split"


@dataclass
class RegionResult:
    name: str
    zeta_mode: str
    arms: dict = field(default_factory=dict)       # name -> ArmSummary
    degenerate_runs: dict = field(default_factory=dict)  # variant -> (K,) counts
    outcomes: dict = field(default_factory=dict)   # variant -> list[str]
    prop3_as_stated_fail: np.ndarray | None = None
    prop3_at_threshold_fail: np.ndarray | None = None


def run_region(name: str, zeta_mode: str, draws: list[Draw], reference=None) -> RegionResult:
    low, upp = region_box(name)
    zeta = zeta_vector(zeta_mode)
    result = RegionResult(name=name, zeta_mode=zeta_mode)
    result.arms["arm (a)"] = run_arm("arm (a)", lambda s, y: standard_thresholds(s, y, ALPHA), draws)
    for variant in ("conservative", "optimistic"):
        cals = calibrations(draws, low, upp, variant, zeta)
        taus = np.array([c.tau for c in cals])
        result.arms[f"arm (c) {variant}"] = run_arm(
            f"arm (c) {variant}", _replay(taus), draws
        )
        result.degenerate_runs[variant] = (taus >= 1.0).sum(axis=0)
        result.outcomes[variant] = [outcome(taus[:, k]) for k in range(K)]
        if variant == "optimistic" and reference is not None:
            checks = [proposition_three_check(reference, c, ALPHA) for c in cals]
            result.prop3_as_stated_fail = np.array(
                [[not ch.holds_as_stated for ch in run] for run in checks]
            ).sum(axis=0)
            result.prop3_at_threshold_fail = np.array(
                [[not ch.holds_at_threshold for ch in run] for run in checks]
            ).sum(axis=0)
    return result


def _replay(taus: np.ndarray):
    """A ThresholdFn that hands back precomputed thresholds in draw order."""
    it = iter(taus)
    return lambda scores, y: next(it)


# --- assumptions (D-043 §4(b)) -------------------------------------------------------

def assumption_checks(name: str, reference: InflationReference, ref_scores, ref_y_tilde):
    lo, hi = region_box(name)
    c = c_for_counts(tuple(CALIBRATION_COUNTS))
    n_star = min(CALIBRATION_COUNTS)
    mult = 1 / math.sqrt(n_star) + math.sqrt((math.log(2 * K) + math.log(n_star)) / 2)
    rows = []
    for k in range(K):
        own = ref_scores[ref_y_tilde == k, k]
        hist, _ = np.histogram(own, bins=200, range=(0.0, 1.0), density=True)
        a2_fmin_fails = hist.min() < 0.01 * np.median(hist)

        grid = np.sort(ref_scores[:, k])
        f_kk = np.searchsorted(np.sort(own), grid, side="right") / own.size
        worst = np.zeros_like(f_kk)
        for l in range(K):
            if l == k:
                continue
            other = np.sort(ref_scores[ref_y_tilde == l, k])
            worst = np.maximum(worst, np.searchsorted(other, grid, side="right") / other.size)
        a3_sup = float((worst - f_kk).max())

        others = [l for l in range(K) if l != k]
        d_star = max(hi[k, l] - lo[k, l] for l in others)
        rhs = (
            -ALPHA + c[CALIBRATION_COUNTS[k]]
            + 2 * sum(abs(hi[k, l]) + (hi[k, l] - lo[k, l]) for l in others) / math.sqrt(n_star) * mult
            + (K - 1) * (d_star + d_star)
        )
        rows.append(
            {
                "a2_fmin_fails": bool(a2_fmin_fails),
                "a2_min_over_median": float(hist.min() / np.median(hist)),
                "a3_sup": a3_sup,
                "a3_holds": a3_sup <= 0.005,
                "a5_rhs": float(rhs),
                "inf_delta": reference.inf_delta(k),
                "a5_holds": reference.inf_delta(k) >= rhs,
            }
        )
    return rows


# --- alpha sweep (D-043 §3) ----------------------------------------------------------

def flip_alpha(name: str, draws: list[Draw]) -> list[float | None]:
    low, upp = region_box(name)
    admissible = np.zeros((len(ALPHA_GRID), K), dtype=int)
    for g, alpha in enumerate(ALPHA_GRID):
        cals = calibrations(draws, low, upp, "conservative", None, alpha=alpha)
        admissible[g] = (np.array([c.tau for c in cals]) < 1.0).sum(axis=0)
    flips = []
    for k in range(K):
        hit = [ALPHA_GRID[g] for g in range(len(ALPHA_GRID)) if admissible[g, k] >= 50]
        flips.append(hit[0] if hit else None)
    return flips, admissible


def score_flip(name: str, k: int, measured: float | None) -> dict:
    cf = closed_form(name, k)
    lower, upper = cf.delta_ci - cf.d_max, cf.delta_ci - cf.delta_min
    first_at_least = lambda x: next((g for g in ALPHA_GRID if g >= x), None)
    last_at_most = lambda x: max((g for g in ALPHA_GRID if g <= x), default=None)
    m = math.inf if measured is None else measured
    g_low = first_at_least(lower)
    g_up = first_at_least(upper)
    in_bracket = (g_low is None or m >= g_low) and (g_up is None or m <= g_up)
    g_d042 = first_at_least(cf.delta_ci)
    d042_held = g_d042 is not None and m <= g_d042
    g_sec = last_at_most(secondary_flip_low_end(name, k))
    secondary_held = g_sec is None or m >= g_sec
    return {
        "measured": measured, "bracket": (lower, upper), "in_bracket": in_bracket,
        "d042_held": d042_held, "secondary_held": secondary_held,
    }


# --- head-to-head (D-043 §4(c)) ------------------------------------------------------

def head_to_head(truth, point, draws):
    low_c, upp_c = region_box("A")
    v_point = transition_to_v(point, NOISY_PRIOR)[0]
    low_b, upp_b = zero_width_box(v_point)
    out = {}
    for label, (low, upp) in {"b": (low_b, upp_b), "c": (low_c, upp_c)}.items():
        cals = calibrations(draws, low, upp, "conservative", None)
        taus = np.array([c.tau for c in cals])
        out[label] = (run_arm(label, _replay(taus), draws), (taus >= 1.0).sum(axis=0))
    arm_a = run_arm("a", lambda s, y: standard_thresholds(s, y, ALPHA), draws)
    verdicts = {}
    for k in (0, 3):
        b, b_deg = out["b"]
        c, c_deg = out["c"]
        b_valid = b.coverage[k] >= 0.90 - 3 * b.coverage_se[k]
        c_valid = c.coverage[k] >= 0.90 - 3 * c.coverage_se[k]
        c_degenerate = c_deg[k] >= 50
        if c_valid and not c_degenerate and (not b_valid or c.mean_size[k] <= b.mean_size[k] - 0.05):
            verdict = "WIN"
        elif c_valid and c_degenerate and not b_valid:
            verdict = "WIN BY ABSTENTION"
        elif b_valid and (not c_valid or c.mean_size[k] >= b.mean_size[k] + 0.05):
            verdict = "LOSS"
        else:
            verdict = "TIE"
        verdicts[k] = {
            "verdict": verdict, "b_coverage": float(b.coverage[k]), "b_se": float(b.coverage_se[k]),
            "b_size": float(b.mean_size[k]), "b_degenerate_runs": int(b_deg[k]),
            "c_coverage": float(c.coverage[k]), "c_se": float(c.coverage_se[k]),
            "c_size": float(c.mean_size[k]), "c_degenerate_runs": int(c_deg[k]),
            "b_full": float(b.full_rate[k]), "c_full": float(c.full_rate[k]),
            "a_coverage": float(arm_a.coverage[k]), "a_se": float(arm_a.coverage_se[k]),
            "a_size": float(arm_a.mean_size[k]), "a_full": float(arm_a.full_rate[k]),
            "a_valid": bool(arm_a.coverage[k] >= 0.90 - 3 * arm_a.coverage_se[k]),
        }
    return verdicts


# --- full D-043 run ------------------------------------------------------------------

def valid(summary, k: int) -> bool:
    """D-043 §1: coverage against the clean label >= 0.90 - 3 SE."""
    return bool(summary.coverage[k] >= 0.90 - 3 * summary.coverage_se[k])


def run_d043(sweep: bool = True) -> dict:
    """Every D-043 measurement, with each prediction scored by D-043's own rules.

    Returns a dict with keys ``results`` ({(region, zeta_mode): RegionResult}),
    ``assumptions`` ({region: rows}), ``set_size`` (§4(a)), ``h2h`` (§4(c)),
    ``flips`` ({region: list}) and ``contradictions`` (list of str).
    """
    contradictions: list[str] = []
    region_draws, assumptions, results = {}, {}, {}
    for name in REGION_NAMES:
        truth = region_truth(name)
        region_draws[name] = draws_for(truth)
        reference, (scores, y_tilde) = reference_for(truth)
        assumptions[name] = assumption_checks(name, reference, scores, y_tilde)
        for mode in ("zeta_star", "zeta_zero"):
            results[(name, mode)] = run_region(name, mode, region_draws[name], reference)
        del reference, scores, y_tilde

    scored = {}
    for name in REGION_NAMES:
        for mode in ("zeta_star", "zeta_zero"):
            r = results[(name, mode)]
            for k in range(K):
                deg = int(r.degenerate_runs["conservative"][k])
                out = r.outcomes["conservative"][k]
                certain = PRIMARY_CERTAIN[mode].get((name, k))
                if certain and deg != (100 if certain == "degenerate" else 0):
                    contradictions.append(f"BUG certain {name} {SHORT[k]} {mode}: {deg}/100")
                if mode != "zeta_star":
                    continue
                pred = SECONDARY[(name, k)]
                sec = "held" if out == pred else ("split" if out == "split" else "falsified")
                if sec != "held" and not certain:
                    contradictions.append(
                        f"secondary {name} {SHORT[k]}: predicted {pred}, got {out} ({deg}/100 degenerate)"
                    )
                d042 = None
                if (name, k) not in D042_NOT_OK:
                    d042 = {"admissible": "confirmed", "degenerate": "overturned"}.get(out, "split")
                scored[(name, k)] = {"outcome": out, "degenerate_runs": deg,
                                     "secondary": None if certain else sec, "d042": d042}

    ra = results[("A", "zeta_star")]
    size_bip = float(ra.arms["arm (c) conservative"].mean_size[0])
    branch = (
        "held" if 2.9 <= size_bip <= 3.3
        else "D-042 confirmed, prediction falsified" if size_bip <= 1.6
        else "partly missed"
    )
    if branch != "held":
        contradictions.append(f"§4(a) bipolar set size {size_bip:.3f}: {branch}")
    same_as_a = (ra.arms["arm (a)"].tau == ra.arms["arm (c) optimistic"].tau).sum(axis=0)
    for k in (0, 2, 3):
        if same_as_a[k] < 95:
            contradictions.append(f"§4(a) optimistic equals arm (a) for {SHORT[k]} in {same_as_a[k]}/100")

    for name in REGION_NAMES:
        for k, row in enumerate(assumptions[name]):
            if not row["a2_fmin_fails"]:
                contradictions.append(f"A2 {name} {SHORT[k]}: f_min part predicted to fail, held")
            if not row["a3_holds"]:
                contradictions.append(f"A3 {name} {SHORT[k]}: sup {row['a3_sup']:.4f}")
            if row["a5_holds"] != ((name, k) == ("A", 1)):
                contradictions.append(f"A5 {name} {SHORT[k]}: opposite of prediction")
        r = results[(name, "zeta_star")]
        opt = r.arms["arm (c) optimistic"]
        for k in range(K):
            if r.prop3_as_stated_fail[k] != 100:
                contradictions.append(f"Prop 3 as stated {name} {SHORT[k]}: {r.prop3_as_stated_fail[k]}/100")
            if r.prop3_at_threshold_fail[k] < 90:
                contradictions.append(
                    f"Prop 3 at threshold {name} {SHORT[k]}: failed {r.prop3_at_threshold_fail[k]}/100 (predicted >= 90)"
                )
            if not valid(opt, k):
                contradictions.append(f"optimistic invalid {name} {SHORT[k]}")

    h2h = {
        "O1": head_to_head(normalised(T_HOC_FINE_TUNED), normalised(T_CLEANLAB), region_draws["A"]),
        "O2": head_to_head(
            normalised(T_CLEANLAB), normalised(T_HOC_FINE_TUNED), draws_for(normalised(T_CLEANLAB))
        ),
    }
    won = all(
        any(h2h[o][k]["verdict"] == "WIN" for k in (0, 3))
        and not any(h2h[o][k]["verdict"] == "LOSS" for k in (0, 3))
        for o in h2h
    )
    for (o, k), p in {("O1", 0): "WIN BY ABSTENTION", ("O1", 3): "WIN BY ABSTENTION",
                      ("O2", 0): "LOSS", ("O2", 3): "LOSS"}.items():
        if h2h[o][k]["verdict"] != p:
            contradictions.append(f"head-to-head {o} {SHORT[k]}: predicted {p}, got {h2h[o][k]['verdict']}")
    if won:
        contradictions.append("head-to-head: predicted not won, got WON")

    flips = {}
    if sweep:
        for name in REGION_NAMES:
            measured, _ = flip_alpha(name, region_draws[name])
            flips[name] = [score_flip(name, k, measured[k]) for k in range(K)]
            for k, s in enumerate(flips[name]):
                if not s["in_bracket"]:
                    contradictions.append(f"BUG flip outside bracket {name} {SHORT[k]}")
                if not s["secondary_held"]:
                    contradictions.append(f"flip secondary {name} {SHORT[k]}: measured {s['measured']}")

    return {
        "results": results, "scored": scored, "assumptions": assumptions,
        "set_size": {"bipolar": size_bip, "branch": branch, "optimistic_same_as_a": same_as_a},
        "h2h": h2h, "h2h_won": won, "flips": flips, "contradictions": contradictions,
    }


# --- report --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:  # pragma: no cover - reporting
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-sweep", action="store_true")
    args = parser.parse_args(argv)
    run = run_d043(sweep=not args.skip_sweep)
    results = run["results"]

    print("=== D-043 §2: admissibility at alpha = 0.10 (conservative) ===")
    for mode in ("zeta_star", "zeta_zero"):
        print(f"\n-- {mode} ({'primary, scored' if mode == 'zeta_star' else 'sensitivity'})")
        for name in REGION_NAMES:
            r = results[(name, mode)]
            for k in range(K):
                s = run["scored"].get((name, k), {}) if mode == "zeta_star" else {}
                print(f"{name:3}{SHORT[k]:16} degenerate {int(r.degenerate_runs['conservative'][k]):3}/100"
                      f"  {r.outcomes['conservative'][k]:10}  secondary {s.get('secondary')!s:9}  D-042 {s.get('d042')}")

    print("\n=== every cell: clean-label coverage ± SE, mean set size, full-set rate (primary zeta) ===")
    for name in REGION_NAMES:
        r = results[(name, "zeta_star")]
        for arm_name, s in r.arms.items():
            for k in range(K):
                print(f"{name:3}{arm_name:24}{SHORT[k]:16} cov {s.coverage[k]:.4f}±{s.coverage_se[k]:.4f}"
                      f" valid {valid(s, k)!s:5} size {s.mean_size[k]:.3f} full {s.full_rate[k]:.3f}")

    ss = run["set_size"]
    print(f"\n=== §4(a) === region A bipolar size {ss['bipolar']:.3f} -> {ss['branch']};"
          f" optimistic equal to arm (a) in {ss['optimistic_same_as_a'].tolist()} runs")

    print("\n=== §4(b) assumptions ===")
    for name in REGION_NAMES:
        for k, row in enumerate(run["assumptions"][name]):
            print(f"{name:3}{SHORT[k]:16} A2 f_min fails {row['a2_fmin_fails']!s:5} ({row['a2_min_over_median']:.4f})"
                  f"  A3 sup {row['a3_sup']:+.4f}  A5 rhs {row['a5_rhs']:+.4f} inf Delta {row['inf_delta']:+.4f} holds {row['a5_holds']}")

    print("\n=== §5 Proposition 3 ===")
    for name in REGION_NAMES:
        r = results[(name, "zeta_star")]
        print(f"{name:3} as stated fails {r.prop3_as_stated_fail.tolist()}  at threshold fails {r.prop3_at_threshold_fail.tolist()}")

    print("\n=== §4(c) head-to-head (region A width) ===")
    for o, rows in run["h2h"].items():
        for k, row in rows.items():
            print(f"{o} {SHORT[k]:14} {row['verdict']:18}"
                  f" a cov {row['a_coverage']:.4f}±{row['a_se']:.4f} size {row['a_size']:.3f} full {row['a_full']:.3f}"
                  f" | b cov {row['b_coverage']:.4f}±{row['b_se']:.4f} size {row['b_size']:.3f} full {row['b_full']:.3f}"
                  f" | c cov {row['c_coverage']:.4f}±{row['c_se']:.4f} size {row['c_size']:.3f} full {row['c_full']:.3f} deg {row['c_degenerate_runs']}")
    print(f"head-to-head WON: {run['h2h_won']}")

    if run["flips"]:
        print("\n=== §3 alpha sweep ===")
        for name, rows in run["flips"].items():
            for k, s in enumerate(rows):
                print(f"{name:3}{SHORT[k]:16} measured {s['measured']!s:>5} bracket [{s['bracket'][0]:.3f}, {s['bracket'][1]:.3f}]"
                      f" in {s['in_bracket']!s:5} D-042 held {s['d042_held']!s:5} secondary held {s['secondary_held']}")

    print("\n=== results contradicting D-043 ===")
    for line in run["contradictions"] or ["none"]:
        print(" -", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
