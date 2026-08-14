# Project Progress & Roadmap — resume-from-here reference

_Last updated: 2026-08-14. This is the single "where are we / what's next" document.
Read §0, then jump to §5 "Next action when you resume."_

_Authority note: `Docs/DECISIONS.md` is the authoritative record for every claim and
reversal. This file summarises status and sequencing only. Where the two disagree,
DECISIONS.md wins._

---

## 0. TL;DR (30-second version)

- **C0 (baseline) is DONE.** MentalBERT fine-tuned 2026-07-07 on Colab. Held-out
  macro-F1 **0.8782**, accuracy 0.9605. Full record in `Docs/baseline-results.md`.
- **C1 (noise-estimator diagnostic) is COMPLETE on three legs** — two estimators
  (HOC, cleanlab) and three representations (fine-tuned MentalBERT, base MentalBERT,
  `all-mpnet-base-v2`). D-041 closed it on 2026-08-09. **No further estimator or arm
  is needed to make the point.**
- **The C1 result, in one line:** the condition *ranking* is identified and the noise
  *magnitude* is not. Bipolar's implied noise rate spans **4.6% to 45.9%** across
  representations and estimators, each answer internally tight and mutually exclusive.
  No point estimate is available, and D-042 now quantifies by how far: that spread is
  **3.6 times wider** than Sesia's Algorithm 2 can absorb.
- **C2 (noise-coupled calibrated abstention) is BARELY STARTED.** `src/conformal/`
  holds the input side (the T → V conversion and the closed-form degeneracy test,
  D-042). There is still no conformal calibration, no abstention, and no elicited
  region. **C2's actual claim has not been tested.**
- **C3 (shift evaluation) has NOT STARTED.** DAIC-WoZ loads (188 participants) but is
  not wired into any evaluation.
- **Immediately next:** §5 step 3b, **scoping the clinical elicitation against a
  numeric budget**. The elicited region's off-diagonal mass must stay under 0.378 for
  bipolar at α = 0.10. That is where C2's contribution begins, and it is now the
  critical path. The simulation arms (steps 4 to 10) follow it, not the reverse.

---

## 1. What this project is

Multi-condition mental-health screening from text, studying **proxy-label noise**
(Reddit "subreddit = diagnosis" is a noisy label) and **calibrated abstention** (the
model should refuse to answer when unsure), evaluated under a **Reddit → clinical
distribution shift**. Downstream model: **MentalBERT**. Full technical framing lives
in `Docs/technical-contribution-in-depth.md`.

Three datasets:
- **Low et al.** (Reddit, Zenodo 3941387) — per-post, subreddit = proxy label.
- **SWMH** (Reddit) — excluded, D-005; also not on disk.
- **DAIC-WoZ** (clinical interviews) — per-participant, PHQ-8 depression labels; held
  out for the C3 shift evaluation.

---

## 2. Status by contribution

| Phase | What it is | Status | Evidence |
|---|---|---|---|
| **C0 — baseline** | Naive MentalBERT multi-class fine-tune on proxy labels; raw softmax confidence | **DONE** 2026-07-07 | `Docs/baseline-results.md` |
| **C1 — noise-estimator diagnostic** | Run HOC + cleanlab on the real data and measure whether the no-clean-data estimators are usable | **COMPLETE** 2026-08-09 | D-034, D-036, D-037, D-038, D-039, D-040, D-041 |
| **C2 — noise-coupled calibrated abstention** | Sesia's Algorithm 2 fed a clinically elicited *region* rather than a point estimate | **INPUT SIDE ONLY** (D-042); the claim itself untested | design in D-027, D-028, D-029, D-031, D-032 |
| **C3 — identifiability-under-shift eval** | Synthetic recovery test, per-condition degradation curves, Reddit → DAIC shift | **NOT STARTED** | design in D-032 |
| Optional — symptom-alignment signal | Per-post DSM-5/ICD-11 symptom-match score | dropped unless time permits | — |

**Where the POC sits.** C0 and C1 are past proof-of-concept: they are finished,
pre-registered, written-up results. The POC that still matters is C2's, and it is
described step by step in §5.

---

## 3. What exists in code

`src/data/` — ingestion + harmonization. Loads all sources, harmonizes labels, emits
one canonical table. `python -m src.data` → **140,086 rows**. Deep dive:
`Docs/data-harmonization.md`.

| File | Role |
|---|---|
| `schema.py` | `Condition` enum, `POC_CONDITIONS`, canonical columns, `validate_schema()` |
| `label_map.py` | the ONE `(source, raw_label) -> Condition` table |
| `text_normalization.py` | the ONE text-cleaning policy (preserves the register gap) |
| `config.py` / `base.py` | injectable `DataPaths`; `DatasetLoader` Protocol |
| `loaders/{swmh,low_et_al,daic_woz}.py` | one loader per dataset |
| `combine.py` / `__main__.py` | `combine_sources()`; sanity-check CLI |

`src/modeling/` — the C0 baseline. Path-injected and CLI-driven so the same code runs
locally and on Colab: `config.py`, `labels.py`, `splits.py`, `dataset.py`,
`hf_model.py`, `metrics.py`, `train.py`, `predict.py`.

`src/noise/` — the C1 diagnostic layer (D-030: a diagnostic, not an estimator).

| File | Role |
|---|---|
| `embeddings.py` | extract + cache pooled MentalBERT embeddings for the training split |
| `clusterability.py` | 2-NN noisy-label agreement per condition, with the one-sided caveat baked in |
| `hoc_estimate.py` | HOC (Zhu, Song & Liu 2021) reimplementation, run per D-035's protocol |
| `oos_probabilities.py` | author-grouped k-fold out-of-sample probabilities (cleanlab's precondition) |
| `cleanlab_estimate.py` | cleanlab confident-joint transition matrix, same condition ordering |

`src/conformal/` — C2's input side, started 2026-08-14 with roadmap step 1.

| File | Role |
|---|---|
| `v_region.py` | T → M → V conversion, the containment region, and the closed-form degeneracy test |
| `c1_matrices.py` | C1's transition matrices transcribed from DECISIONS.md; the two constructed ones flagged in their docstrings |
| `__main__.py` | `python -m src.conformal` reproduces every table in `Docs/c2-interface.md` §6 |

**Not built:** the conformal calibration itself (arms a, b, c), the synthetic
generator, and anything for C3. No shift-evaluation script.

`notebooks/train_colab.ipynb` — the GPU runner. Outputs land in `Models/`
(gitignored; on Drive when using Colab).

---

## 4. Results of record

### C0 baseline (test set, 14,039 rows)

Accuracy 0.9605 · **macro-F1 0.8782** · per-condition F1: depression 0.9787,
eating_disorder 0.9451, schizophrenia 0.8372, bipolar 0.7516. Splits are
author-grouped 0.80/0.10/0.10 (train 111,892 / val 13,967 / test 14,039).

Dominant error mode: minority classes misclassified as depression. Three epochs gave
no improvement over one, which is the evidence that performance is bounded by
proxy-label noise rather than model capacity — the motivation for C1.

### C1 diagnostic — implied noise rate (1 minus the diagonal)

| condition | HOC fine-tuned | HOC mpnet | HOC base | cleanlab | spread |
|---|---|---|---|---|---|
| bipolar | 4.6% | 29.3% | 45.9% | 15.2% | **10x** |
| schizophrenia | 2.2% | 18.1% | 35.5% | 10.1% | 16x |
| eating_disorder | 0.5% | 7.9% | 17.4% | 3.2% | 35x |
| depression | 0.4% | 2.7% | 4.2% | 2.1% | 11x |

**The rank order is identical in all four columns.** Two independent estimators and
three independent representations agree completely on which conditions are noisier
and in what order, and disagree by an order of magnitude on how noisy any of them is.
Each individual answer carries a tight cross-seed spread (sd 0.0008 to 0.0104) that
excludes the others, so the disagreement is confident mutual contradiction, not noise.
Bipolar's dominant off-diagonal leaks into depression in every run, the clinically
predicted direction.

Full detail and every caveat: D-037, D-039, D-041.

---

## 5. ▶ NEXT ACTION — the C2 proof-of-concept

**The kill question.** C1 hands C2 a bipolar region spanning 4.6% to 45.9%. Feeding a
region that wide into Sesia's Algorithm 2 may produce prediction sets so conservative
that bipolar always returns the full label set, which is abstention in every case and
no method at all. D-032 names this failure mode.

**Steps 1 and 2 answered it analytically on 2026-08-14, and reframed the question.**
Reading Algorithm 2 in full showed the failure mode is a closed-form inequality, not a
simulation outcome: τ̂_k = 1 (class k enters every prediction set) whenever
`δ_ci(n_k, n*) > α`. The binding quantity is the total off-diagonal mass of `V = M⁻¹`
that the region admits.

> **Bipolar's admissible mass is 0.378 at α = 0.10. C1's four-estimate spread sits at
> 1.354, which is 3.6 times over budget.** At that width bipolar and schizophrenia both
> degenerate robustly, and no calibration size fixes it.

**Read that as a C1 result, not a C2 one.** C1's spread is not C2's input; C2's input is
the *elicited* region (D-028), which does not exist yet. What the arithmetic does is
convert C1's "the estimators disagree" into "and by 3.6 times more than the downstream
method can absorb", which is the quantity C1 could not supply on its own. It also gives
the elicitation a **design budget** where open item 6 previously had none. Full
derivation and caveats: **`Docs/c2-interface.md`**; decision entry D-042.

**Scope discipline — what the POC deliberately excludes.** No clinician elicitation
(open item 6). No DAIC. No retraining. No MentalBERT in the loop. No thesis prose.
The POC tests the *mechanism*, not the content, so it runs on synthetic data with a
known `T` and needs only numpy/scikit-learn on CPU. No Colab.

### Step 1 — Pin Sesia's Algorithm 2 interface. ✅ DONE 2026-08-14

Full paper read; `Docs/c2-interface.md` written. What it settled:

- The region is an **interval box on the off-diagonal entries of V = M⁻¹**, at level
  1 − α_V, plus deterministic a priori bounds `V̄_upp`. Not a set of matrices, not an
  ε-contamination ball.
- `M_kl = P[Y = l | Ỹ = k]` is the **reverse** of the `T` that HOC and cleanlab
  estimate. Conversion is `ρ = (Tᵀ)⁻¹ρ̃`, `M = diag(ρ̃)⁻¹ Tᵀ diag(ρ)`, `V = M⁻¹`. The
  clean prior is recovered, not assumed, and stays on the simplex for both matrices on
  record.
- Only off-diagonals need a region because V's rows sum to one.
- α_V = 0 (D-031 option (a)) removes the `2α_V Σ|V̄_upp|` term from the correction
  outright. The corollary D-031 protects is real.
- **Degeneracy is closed-form:** `δ_ci(n_k, n*) > α` ⟹ τ̂_k = 1 ⟹ class k in every
  set. The binding quantity is Σ_{l≠k}|V_kl| at the region's widest end, and
  `n* = min_k n_k` means bipolar's calibration count sets the correction for *all*
  conditions.
- **Sesia's own estimation route (Proposition 4) needs clean labels.** This project has
  none, which is what makes the elicited region the available route. It also means the
  head-to-head against Sesia's own bootstrap cannot be run on this data.

Evaluated at α = 0.10, α_V = 0, n* = 481 (the C0 test split):

| region | bipolar δ_ci | verdict |
|---|---|---|
| A: HOC fine-tuned + cleanlab (measured) | 0.0918 | survives by 0.0082 |
| B: + base and mpnet arms (approximate) | 0.3327 | degenerate, robustly |

Schizophrenia also degenerates at region B, but borderline. Depression and eating
disorder survive everywhere tested.

### Step 2 — Transcribe the base and mpnet matrices. ✅ DONE 2026-08-14

Both retrieved from `Models/embeddings/train__{mpnet,base}/hoc_mean_T.csv`. Region B is
now measured throughout. It moved the result twice over:

- bipolar δ_ci 0.3328 → **0.2848**, still robustly degenerate
- schizophrenia 0.1251 → **0.1524**, and **borderline → robust**

So **two of four conditions are inadmissible at the C0 operating point**, not one.

It also corrected an inference this roadmap carried: the diagonal reconstruction was
validated on mpnet alone, where it understates bipolar's off-diagonal mass, and that was
read as meaning the verdict was conservative. The base arm overstates it. The error has
no stable sign, and reconstruction is now used only as a check, never as an input.

### Step 3 — Solve the feasibility frontier in closed form. ◐ MOSTLY DONE

δ_ci is closed-form in Σ|V̂_low|, so the region width at which each condition
degenerates is **solved for**, not searched. Maxima for
Σ_{l≠k}(|V̂_upp_kl| + δ̂^(V)_kl) at α = 0.10: bipolar 0.378, schizophrenia 0.411,
eating disorder 0.441, depression 0.497.

Remaining: plot it properly as a per-condition surface over (region width, α,
calibration fraction) for Chapter 4. The numbers exist; the figure does not.

This is D-032's sweep and it no longer needs a simulation to produce its main curve.

### Step 3b — Scope the elicitation against the budget (open item 6)

**This is where C2's actual contribution starts, and it is now the critical path.**
Nothing before this step tests C2's claim. The question is whether a defensible reading
of the clinical literature yields a region whose off-diagonal mass fits under 0.378 for
bipolar.

D-031 interacts directly: not pruning keeps deterministic containment and α_V = 0, but
it also keeps the region wide, and width is exactly what the budget constrains. That
trade is no longer abstract and should be settled with the supervisor with these
numbers in hand.

### Step 4 — Synthetic generator with a known `T`

New `src/conformal/synthetic.py`. Draw a K = 4 problem at this project's real
prevalences (roughly 0.81 / 0.10 / 0.06 / 0.04), a controllable true `T` with
bipolar → depression as the dominant off-diagonal, and a scorer that emits softmax-like
probabilities with a tunable separability parameter. Corrupt labels through `T`.

**Sanity gate:** with `T = I`, all three arms must coincide. That identity check is the
POC's own unit test — if it fails, nothing downstream means anything.

### Step 5 — Arm (a), noise-blind conformal

Standard split conformal, **Mondrian / label-conditional** per D-029, at α = 0.1.
Measure per-class coverage and mean set size.

**Gate:** on synthetic data with `T = I`, per-class coverage hits nominal. This
validates the harness before noise enters.

Fix and record the prediction function here: Sesia's equation (3) is the plain
threshold on π̂, their §4 uses the generalised inverse quantile scores of Romano,
Sesia et al. (2020). Set sizes differ between them, so the choice must be recorded
before the arms are compared.

### Step 6 — Arm (b), Algorithm 2 with a plug-in point estimate

Two sub-runs, both diagnostic:

1. **Oracle:** feed the true `T`. Coverage should be correct. Confirms the Algorithm 2
   implementation.
2. **Wrong point:** feed a deliberately wrong point estimate (HOC-like 4.6% when the
   truth is 15%). Coverage should break.

Sub-run 2 is the argument for a region, made on synthetic data where the truth is
known. It is worth having on its own.

### Step 7 — PRE-REGISTER before arm (c) runs

Write **D-042** in DECISIONS.md fixing branches and thresholds *before the numbers
exist*. The project's own scoreboard is the reason: two of four pre-registered
predictions did not hold, and in both cases the pre-registration is what stopped the
result being reinterpreted after the fact (D-034, D-041).

Note the change of role forced by Step 1. The closed-form analysis already predicts
degeneracy at region B, so D-042 is no longer predicting *whether* it degenerates. It
should pre-register the quantities the closed form cannot give:

| To pre-register | Why it is still open |
|---|---|
| mean set size for bipolar in the **non**-degenerate region A | δ_ci says nothing about set sizes short of degeneracy |
| whether Assumptions 2, 3 and 5 hold empirically | only a run can inspect them |
| whether arm (c) beats arm (b) at region A width | the D-032 head-to-head, still a coin flip |
| the α at which each condition flips, measured vs the closed-form prediction | a direct check on the Step 3 arithmetic |

That last row matters: it turns Step 3 into a falsifiable prediction rather than a
derivation nobody tested.

### Step 8 — Arm (c), region input at the real width

Feed the region A and region B boxes from `c2-interface.md` §6. Measure per-class
coverage, mean set size, and the full-set rate. Fire the D-042 branches.

### Step 9 — The sweep, now a check on Step 3

Sweep region width from 0 (a point estimate) upward and compare the measured
degeneracy point against the closed-form frontier from Step 3. Run D-032's
noise-heterogeneity sweep on the same harness.

**This is what makes every outcome a result.** "The coupling pays off below region
width w*, and this project's proxy noise sits above it" is a publishable finding even
if arm (c) loses, and Step 1 has already located w* to within a transcription.

### Step 10 — Write the result up as D-043

Branch fired, predictions held or missed, and the consequence for C2's scope. Update
the reversal scoreboard either way.

### What "POC passed" unlocks

Swap synthetic scores for the real held-out softmax already sitting in
`Models/predictions/test_predictions.csv`; swap the synthetic region for the elicited
one (open item 6, which is where the clinical citation work goes); then C3 adds DAIC
and the degradation curves.

---

## 6. Open items and caveats

- **`Docs/DECISIONS.md` has uncommitted changes** (the D-041 entry). Commit it.
- **Prevalence counts: the apparent disagreement is RESOLVED (2026-08-14), the
  arithmetic is still unverified.** Both totals are correct and count different
  things: the 15 Zenodo CSVs hold exactly 146,400 rows (the figure D-018 uses, sourced
  from the project proposal's June 2026 verification), and `combine_sources()` then
  drops 6,502 exact `(text, source, author_id)` duplicates, giving 139,898 Reddit rows
  plus 188 DAIC = 140,086. Nothing on disk changed. Full per-condition table and
  provenance are now in the **D-018 amendment**. Deduplication falls hardest on the
  rare conditions (3.2% of depression, 7.8% to 11.9% of the rest), so every rare class
  gets rarer and the "required diagonal above 1" conclusion moves further from the
  boundary, not closer. **Still open:** the Theorem 2 formula itself has not been
  checked independently, and HOC ran on the train split (111,892 rows), so the
  operative shares may be the train-split ones.
- **D-041 has measured-but-untranscribed content**: off-diagonal structure per arm,
  estimated prior `p` against observed noisy proportions, per-arm 2-NN agreement
  values, and the `SentenceTransformer.encode()` parity check on Arm B. None change
  any branch, but they belong in the entry.
- **D-031 is still OPEN** (prune or do not prune the elicited set). Needs a decision
  before Chapter 4, ideally with the supervisor. It does not block the POC.
- **Open item 6, elicitation sources, is the long pole after the POC.** The three
  directional claims (bipolar → depression dominates, schizophrenia's diagonal is
  lower, depression's is highest) are currently stated from general knowledge and each
  needs a citation to a specific study. Where the literature is thin the range goes
  wide, and the envelope propagates that width honestly.
- **4 of 6 POC conditions have data.** `anxiety` and `suicidality` are absent; the
  split code warns rather than fails. No code change needed to add them.
- **Class imbalance ~24:1** is why macro-F1 is the headline metric, and it is also why
  three of four conditions have no finite-sample guarantee under D-018.
- **The C0 checkpoint is final-step, not best-epoch** (a consequence of step-based
  saving). Three epochs matched one, so this is unlikely to matter.

---

## 7. Roadmap beyond the POC

| Phase | What it gives |
|---|---|
| **C2 full** | Elicited region from cited clinical sources → Algorithm 2 → per-condition abstention thresholds coupled to C1's ranking. Headline claim reduces to arm (c) beating arm (b) on risk-coverage for bipolar and schizophrenia. |
| **C3** | Synthetic recovery test, per-condition degradation curves, Reddit → DAIC shift evaluation. Evidence for *where and why* the method breaks, matched against C1's theory. |

**End state:** a screener that (a) knows its labels are noisy and has measured that the
noise magnitude is not identifiable from data, (b) translates that irreducible
label-uncertainty into honest prediction-time abstention, and (c) has a characterised
failure boundary under the Reddit → clinical shift. Evaluation is per-condition AND
pooled: accuracy/macro-F1, ECE, risk–coverage/AURC, and the degradation of each under
shift.

**The project is safe either way.** If C2 works, there is a method. If it degenerates,
there is the first characterisation of where this family of methods stops working on
real mental-health proxy labels. A project whose headline survives its own negative
result is well designed, and D-032 says that should be stated in the defence.

---

## 8. Document history

- **2026-08-14** — rewritten. The previous version was dated 2026-07-01 and stated
  "no MentalBERT fine-tune has run yet", five weeks and eight decision entries behind
  reality. C0 and C1 marked complete; §5 replaced with the C2 POC plan.
- **2026-07-24** — C1's framing reversed from estimator to diagnostic (D-030); the
  superseded C1 row is preserved in DECISIONS.md rather than here.
- **2026-07-01** — original, written when the data layer was done and the baseline was
  code-complete but untrained.
