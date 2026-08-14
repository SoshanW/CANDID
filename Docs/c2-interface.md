# C2 interface: what Sesia's Algorithm 2 consumes, and what C1 can supply

_Date: 2026-08-14. Source read in full: Sesia, M., Wang, Y.X.R. and Tong, X. (2025)
'Adaptive conformal classification with noisy labels', JRSSB 87(3), 796–815,
doi:10.1093/jrsssb/qkae114. Local copy: `D:\Uni\Papers\Top 15`._

_This document answers step 1 of the C2 proof-of-concept
(`progress-and-roadmap.md` §5): the exact input signature of Algorithm 2, the
conversion from C1's output into that signature, and the arithmetic that follows._

---

## 1. The input signature

Algorithm 2 takes:

| Input | Meaning |
|---|---|
| `{(X_i, Ỹ_i)}` | the contaminated data set; here, Reddit posts with proxy labels |
| `[V̂_low, V̂_upp]` | a 1 − α_V **simultaneous confidence region for the off-diagonal entries of V**, where V := M⁻¹ |
| `V̄_upp` | deterministic a priori bounds satisfying max{\|V̂_upp_kl\|, \|V_kl\|} ≤ \|V̄_upp_kl\| almost surely, for all l ≠ k |
| `X_{n+1}` | the test point |
| `A` | any black-box classifier; the fine-tuned MentalBERT of C0 qualifies |
| `C` | a prediction function per their Definition 1 |
| `α` | the target significance level |

Theorem 4 then gives label-conditional coverage: P[Y_{n+1} ∈ Ĉ_ci(X_{n+1}) | Y = k] ≥
1 − α for all k. That guarantee is unconditional on the classifier's accuracy, which
matters here because C0's bipolar F1 is 0.75.

**The region is an interval box on the off-diagonals of a matrix, not a set of
matrices and not an ε-contamination ball.** This is the answer to the open question
recorded in the roadmap.

## 2. The direction problem

Their matrix is defined in Proposition 1 as

> M_kl = P[Y = l | Ỹ = k]

This is the **reverse** of the transition matrix T that HOC and cleanlab estimate:

> T_lk = P[Ỹ = k | Y = l]

C1's outputs (D-037, D-039, D-041) are all in T's direction. They cannot be handed to
Algorithm 2 without conversion. This is the concrete content of D-028's instruction to
elicit in one direction and convert to Sesia's.

## 3. The conversion arithmetic

By Bayes, M_kl = T_lk · ρ_l / ρ̃_k, where ρ is the clean class prior and ρ̃ the noisy
one. In matrix form:

```
ρ    = (Tᵀ)⁻¹ ρ̃          (the clean prior is recovered, not assumed)
M    = diag(ρ̃)⁻¹ Tᵀ diag(ρ)
V    = M⁻¹
```

ρ̃ is observable (it is the proxy-label frequency). ρ is recovered from the identity
ρ̃ = Tᵀρ, so no external prior is required, though the recovered ρ inherits whatever
error T carries and can in principle leave the simplex. It does not here: both
matrices on record give strictly positive priors summing to one (HOC fine-tuned
0.0369 / 0.8132 / 0.0961 / 0.0538; cleanlab 0.0336 / 0.8199 / 0.0952 / 0.0513),
against observed noisy proportions of 0.038 / 0.812 / 0.096 / 0.054.

**Why only the off-diagonals need a region.** M's rows sum to one, so M·1 = 1 and
therefore V·1 = 1. Each diagonal entry is determined by its row: V_kk = 1 − Σ_{l≠k}
V_kl. Supplying the off-diagonals supplies V.

A useful consequence is that equation (11) collapses to

> Δ_k(t) = Σ_{l≠k} V_kl · (F̃^k_l(t) − F̃^k_k(t))

Under their Assumption 3 each bracket is non-positive, and the off-diagonals of V are
negative for every matrix on record, so Δ_k ≥ 0. Standard conformal calibration is
therefore conservative on this data, which is the gap Algorithm 2 recovers.

## 4. The α_V trade-off, and why D-031 option (a) matters

α_V enters the correction term (21) as `2 α_V Σ_{l≠k} |V̄_upp_kl|` and the coverage
upper bound (Theorem 5) as `(1 + 4 Σ |V̄_upp_kl|) α_V`. Sesia et al. report α_V = 0.01
as a workable default. A region elicited to contain V deterministically has α_V = 0,
which removes that term outright. This is the corollary D-031 option (a) preserves,
and reading the equations confirms it is real rather than notional.

## 5. The degeneracy criterion, derived

The set constructed at step 11 of Algorithm 2 is

> Î_ci_k := { i ∈ [n_k] : i/n_k ≥ 1 − α − Δ̂_ci_k(S_(i)) + δ_ci(n_k, n\*) }

and step 12 sets τ̂_k = 1 whenever Î_ci_k is empty. By Definition 1, τ_k = 1 puts class
k in **every** prediction set. That is the always-abstain failure mode named in D-032,
and it is now a checkable inequality rather than a simulation outcome.

Evaluating the membership test at i = n_k, where F̂^k_k = 1 by construction, gives the
sufficient condition

> **τ̂_k = 1 whenever δ_ci(n_k, n\*) > α + Δ̂_ci_k(S_(n_k))**

with 0 ≤ Δ̂_ci_k(S_(n_k)) ≤ Σ_{l≠k} |V̂_upp_kl| =: D̂_max. Testing `δ_ci > α` alone
therefore over-flags degeneracy by at most D̂_max, and a verdict is robust when the
margin exceeds D̂_max.

Expanding (21) with α_V = 0, and noting that for negative off-diagonals
|V̂_upp_kl| + δ̂^(V)_kl = |V̂_low_kl|:

> δ_ci(n_k, n\*) = c(n_k) + (2 / √n\*) · Σ_{l≠k} |V̂_low_kl| · min{ K√(π/2), 1/√n\* +
> √((log 2K + log n\*)/2) }

Two properties follow, and both are load-bearing for this project:

1. **The binding quantity is Σ_{l≠k} |V_kl| at the widest end of the region**, that is,
   the largest total off-diagonal mass the region admits. Region width and noise
   magnitude enter through the same channel.
2. **n\* = min_k n_k**, the calibration count of the *smallest* class. Bipolar's
   calibration count sets the correction term for every condition, not just its own.

## 6. Evaluated on this project's numbers

**What the two regions below are, and what they are not.** Neither is C2's input. Both
are built by stacking C1's estimated matrices into a containment box, so they express
**C1's disagreement in V coordinates**. Running them through δ_ci measures the
*consequence* of C1's finding for a downstream consumer, which strengthens C1; it does
not test C2. C2's input is the **elicited** region of D-028, which is a different object
derived from clinical sources, and it is not evaluated anywhere in this document.

- **Region A** stacks D-037 and D-039, the two estimates available before the D-040
  arms. It is retained as the narrow illustrative case.
- **Region B** stacks all four C1 estimates and is therefore the full spread reported
  in D-041.

α = 0.10, α_V = 0, K = 4, calibration counts as in the C0 test split (bipolar 481,
depression 11,351, eating disorder 1,421, schizophrenia 786), so n\* = 481.

**Region A.** The two full transition matrices on record, D-037 (HOC on
fine-tuned features) and D-039 (cleanlab). Bipolar row of V:

| | bipolar | depression | eating dis. | schizophrenia |
|---|---|---|---|---|
| V̂_low | 1.0807 | −0.2630 | −0.0048 | −0.0676 |
| V̂_upp | 1.3353 | −0.0577 | −0.0021 | −0.0209 |

| condition | δ_ci | margin vs α | verdict |
|---|---|---|---|
| bipolar | 0.0918 | −0.0082 | survives |
| depression | 0.0084 | −0.0916 | survives |
| eating_disorder | 0.0244 | −0.0756 | survives |
| schizophrenia | 0.0551 | −0.0449 | survives |

Bipolar clears the bar by 0.008 at α = 0.10 and fails it at α = 0.05.

**Region B, all four estimates, every one measured.** The mpnet and base arms were
transcribed from `Models/embeddings/train__mpnet/hoc_mean_T.csv` and
`Models/embeddings/train__base/hoc_mean_T.csv` on 2026-08-14. Bipolar row of V:

| | bipolar | depression | eating dis. | schizophrenia |
|---|---|---|---|---|
| V̂_low | 1.0804 | −0.7073 | −0.1418 | −0.5049 |
| V̂_upp | 2.3540 | −0.0573 | −0.0023 | −0.0209 |

| condition | δ_ci | margin vs α | D̂_max | verdict |
|---|---|---|---|---|
| bipolar | 0.2848 | +0.1848 | 0.0804 | degenerate, robust |
| schizophrenia | 0.1524 | +0.0524 | 0.0262 | degenerate, robust |
| eating_disorder | 0.0506 | −0.0494 | 0.0046 | survives |
| depression | 0.0152 | −0.0848 | 0.0022 | survives |

**Two of four conditions degenerate at the C0 defaults, and both verdicts are robust.**
Schizophrenia was borderline while the base arm was reconstructed; the measured matrix
moved it to robust. Bipolar's δ_ci fell from 0.3328 to 0.2848 and stayed robust.

### A correction to the earlier construction check

Before the base arm was retrieved, mpnet was the only arm on which the
diagonal-reconstruction could be checked. It understated bipolar's off-diagonal mass in
V by about 8% (ratio 0.92), and this document previously inferred from that single case
that the reconstruction errs towards narrowness, making the degeneracy verdict
conservative. **The base arm contradicts it:** the reconstruction *overstates* bipolar's
mass there (ratio 1.19) and understates schizophrenia's (0.79). The error has no stable
sign, across arms or across conditions, and the earlier inference was not supported by
one arm. Both regions above now use measured matrices only, and
`build_from_diagonal` is retained solely for this check
(`tests/test_conformal_v_region.py::test_diagonal_reconstruction_errs_in_an_unstable_direction`).

### Sensitivity across the two secondary axes

δ_ci depends on the region width, on α, and on the calibration count. The width is the
axis that matters (§7); the other two are recorded here for completeness. Calibration
shares are of the whole corpus, with real per-condition counts (bipolar totals 5,195
across train 4,221 / val 493 / test 481).

| calibration share | bipolar n | region A δ_ci | region B δ_ci | region B admissible at α = |
|---|---|---|---|---|
| 10% | 519 | 0.0887 | 0.2751 | none |
| 20% | 1,039 | 0.0642 | 0.2004 | none |
| 30% | 1,558 | 0.0531 | 0.1665 | 0.20 |
| 40% | 2,078 | 0.0465 | 0.1460 | 0.15, 0.20 |
| 50% | 2,597 | 0.0419 | 0.1318 | 0.15, 0.20 |

Two things this shows, and one it does not.

- **Region A is admissible at α = 0.10 with the calibration data already available**,
  at every share including the smallest.
- **Region B is inadmissible at α = 0.10 at any share.** Enlarging the calibration set
  buys width at a √n rate, so it cannot close a gap of this size: going from 10% to 50%
  removes barely half the excess.
- **It does not identify a usable configuration for region B.** The α = 0.20 cell is
  80% label-conditional coverage for a screening instrument, bought by moving 20% of the
  corpus out of training, which degrades the classifier and therefore the scores. Both
  costs are real and neither is recommended here.

## 7. What this establishes, and what it leaves untested

**What it establishes is a C1 result, not a C2 one.** Region B is D-041's spread in
different coordinates. Passing it through δ_ci converts C1's qualitative claim into a
quantitative one:

> C1 shows the estimators disagree. It cannot say *how much* disagreement is too much,
> because that has no meaning without a downstream consumer. Sesia supplies the
> consumer. **Bipolar's admissible off-diagonal mass in V is 0.378 at α = 0.10; region
> B sits at 1.354, which is 3.6 times over budget.**

That closes a gap in C1's argument. An examiner can ask C1 "the estimators disagree, so
what, perhaps it is close enough in practice"; the answer is now arithmetic rather than
assertion. It is a strengthening of C1 and should be presented as one.

**What it leaves untested is C2's entire claim.** C2 was never "use C1's spread as the
region". It is that clinical elicitation supplies a region the data cannot, and nothing
here evaluates an elicited region because none has been built yet.

What the arithmetic does give C2 is a **design budget**, which open item 6 previously
lacked. The elicitation is no longer an open-ended reading task; it has a numeric
success criterion:

| condition | admissible Σ_{l≠k}(\|V̂_upp_kl\| + δ̂^(V)_kl) at α = 0.10 |
|---|---|
| bipolar | 0.378 |
| schizophrenia | 0.411 |
| eating_disorder | 0.441 |
| depression | 0.497 |

Both outcomes remain reportable, as D-032 requires. If the elicited region fits inside
the budget, C2 is a method and the contrast against region B's 1.354 is the headline.
If it does not, the per-condition frontier is the characterisation result.

The simulation steps still need to run: the closed-form condition is sufficient for
degeneracy but says nothing about set sizes in the non-degenerate region, and Theorem
5's tightness bound rests on Assumptions 2, 3 and 5, which only a run can inspect.

## 8. Gaps in this reading

- **The supplementary material was not held when §1 to §7 were written, and has since
  been located.** Sections S3 (special cases including the randomised response model,
  and the form of V̄_upp), S4 (Algorithms S2 and S3) and S5 are referenced throughout
  and are absent from the JRSSB article PDF in `D:\Uni\Papers\Top 15`. They are
  included in the arXiv preprint, **arXiv:2309.05092**, which runs to 127 pages with
  all appendices (`https://arxiv.org/pdf/2309.05092`). Note that the arXiv v2 is dated
  February 2024 and the JRSSB version was accepted in November 2024, so section
  numbering and constants may differ; anything load-bearing should be checked against
  the OUP supplementary file, reachable from the article page already used for the main
  PDF. S3 carries the worked construction of V̂_low, V̂_upp and V̄_upp, which is the
  template the elicited region should follow. Nothing in §1 to §7 depends on the
  appendices, but §4's α_V handling and the form of V̄_upp should be re-checked against
  S3 before the elicited region is built.
- **Sesia's own route to V requires clean labels.** Section 3.3 and Proposition 4
  estimate V = Q Q̃⁻¹ from a clean set D₀ and a contaminated set D₁, with the region
  obtained by parametric multinomial bootstrap (Sison and Glaz, 1995). This project
  has no clean labels, which is what makes the elicited region the available route
  rather than a preferred one. It also means the head-to-head against Sesia's own
  estimation procedure cannot be run on this data.
- **The prediction function is not yet fixed.** Their equation (3) is the simple
  threshold on π̂; Section 4 uses the generalised inverse quantile scores of Romano,
  Sesia et al. (2020). Set sizes will differ between them and the choice needs
  recording before the arms are compared.
- **Assumption 1 (Ỹ ⊥ X | Y) is untested on this data** and is not obviously true of
  subreddit self-selection: which subreddit a person posts to plausibly depends on the
  text itself, not only on the underlying condition. This is a limitation to name in
  Chapter 4 before an examiner raises it (D-019).

## 9. Reproducing the arithmetic

```bash
python -m src.conformal                    # reproduces every table in §6
python -m src.conformal --alpha 0.05       # the tighter level
python -m src.conformal --calibration-scale 4.0   # a 40% calibration split
```

`src/conformal/v_region.py` holds the conversion and the degeneracy test;
`src/conformal/c1_matrices.py` holds the transcribed matrices, with the constructed
pair marked as such in their docstrings. `tests/test_conformal_v_region.py` pins the
numbers quoted in §6, so this document cannot drift from the code that produced it.

Step 2 of the proof-of-concept replaces `build_from_diagonal(...)` in
`REGION_B_APPROXIMATE` with the measured base and mpnet matrices from
`hoc_per_seed.csv`. Nothing else needs to change, and the regression tests will
report the movement.
