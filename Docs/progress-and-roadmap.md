# Project Progress & Roadmap — resume-from-here reference

_Last updated: 2026-10-08. This is the single "where are we / what's next" document.
Read §0, then jump to §5 "Next action when you resume."_

_Authority note: `Docs/DECISIONS.md` is the authoritative record for every claim and
reversal. This file summarises status and sequencing only. Where the two disagree,
DECISIONS.md wins._

---

## 0. TL;DR (30-second version)

- **C0 (baseline) is DONE.** MentalBERT fine-tuned 2026-07-07 on Colab. Held-out
  macro-F1 **0.8782**, accuracy 0.9605. Full record in `Docs/baseline-results.md`.
  MentalRoBERTa replication (D-045, 2026-09-27): 0.8841 under the identical protocol,
  not separable on macro-F1; MentalBERT retained as the encoder of record.
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
- **Step 3b (elicitation scoping) is DONE as a derivation (D-047, 2026-10-08).** The
  budget binds on posts *entering* the rare labels, not on misdiagnosis: bipolar needs a
  worst-case PPV of about **0.78** for r/bipolarreddit (depression → bipolar near 1% or
  less). The closest measured analogues sit at 0.22 to 0.43, and no subreddit-level
  study exists. Projected: bipolar inadmissible at the C0 operating point, depression and
  eating disorder admissible, schizophrenia open. Full record:
  `Docs/c2-elicitation-scoping.md`.
- **Immediately next:** take D-047 to the supervisor with open items 3, 5 and 14 (verify
  D-028's conversion; prune or not; whether to elicit label PPVs directly). Then §5 step
  4, the synthetic generator, with D-043 pre-registering D-047's per-condition
  predictions.

---

## 1. What this project is

Multi-condition mental-health screening from text, studying **proxy-label noise**
(Reddit "subreddit = diagnosis" is a noisy label) and **calibrated abstention** (the
model should refuse to answer when unsure), evaluated under a **Reddit → clinical
distribution shift**. Downstream model: **MentalBERT**. The technical framing is
split by contribution: C1 in `Docs/contribution-1.md` (why C1 became a diagnostic,
D-030) and `Docs/c1-noise-design.md` (the pre-D-030 estimator design); C2 in
`Docs/c2-interface.md` and `Docs/c2-elicitation-scoping.md`; C3 and the evaluation
design in D-032 and §7 below. (`technical-contribution-in-depth.md` was removed in
94c031d.)

Three datasets:
- **Low et al.** (Reddit, Zenodo 3941387) — per-post, subreddit = proxy label.
- **SWMH** (Reddit) — excluded, D-005; also not on disk.
- **DAIC-WoZ** (clinical interviews) — per-participant, PHQ-8 depression labels; held
  out for the C3 shift evaluation.

---

## 2. Status by contribution

| Phase | What it is | Status | Evidence |
|---|---|---|---|
| **C0 — baseline** | Naive MentalBERT multi-class fine-tune on proxy labels; raw softmax confidence | **DONE** 2026-07-07; encoder replication 2026-09-27 | `Docs/baseline-results.md`, D-045 |
| **C1 — noise-estimator diagnostic** | Run HOC + cleanlab on the real data and measure whether the no-clean-data estimators are usable | **COMPLETE** 2026-08-09 | D-034, D-036, D-037, D-038, D-039, D-040, D-041 |
| **C2 — noise-coupled calibrated abstention** | Sesia's Algorithm 2 fed a clinically elicited *region* rather than a point estimate | **INPUT SIDE ONLY** (D-042); elicitation scoped against the budget (D-047); the claim itself untested | design in D-027, D-028, D-029, D-031, D-032, D-047 |
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
| `elicited_region.py` | an elicited T-box propagated to Sesia's V-box, the degeneracy test on it, worst-case label PPV, and the constructed scoping regions of D-047 |
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

**Encoder replication (D-045).** MentalRoBERTa under the identical protocol and
fingerprint-matched splits: accuracy 0.9634 · macro-F1 **0.8841** (Δ +0.0060,
test-set bootstrap 95% CI [−0.0021, +0.0139]) · per-condition F1: depression 0.9806,
eating_disorder 0.9557, schizophrenia 0.8368, bipolar 0.7635. Same condition ranking,
same depression confusion sink, ECE 0.030 vs 0.031. One seed per encoder. MentalBERT
stays the encoder of record. Full record in `Docs/roberta-replication-results.md`.

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

### Step 3b — Scope the elicitation against the budget (open item 6). ✅ SCOPED 2026-10-08

**This is where C2's actual contribution starts, and it is now the critical path.**
Nothing before this step tests C2's claim. The question is whether a defensible reading
of the clinical literature yields a region whose off-diagonal mass fits under 0.378 for
bipolar.

**Result (D-047, `Docs/c2-elicitation-scoping.md`).** A region can fit all four budgets,
but the binding entries are not the ones open item 6 listed. Bipolar's row of V is
built from posts carrying the bipolar label, so bipolar → depression leakage is
invisible to it and affordable up to about 60%. What exhausts it is inflow: depression →
bipolar near 1%, equivalently a worst-case PPV for r/bipolarreddit below about 0.78.

| Region (constructed) | Depression → bipolar | Bipolar mass (budget 0.378) | Verdict | Worst-case bipolar PPV |
|---|---|---|---|---|
| F1 | ≤ 0.5% | 0.192 | ok | 0.873 |
| F3 | ≤ 1% | 0.392 | degenerate, borderline | 0.763 |
| L1 | ≤ 2.4% | 1.492 | degenerate, robust | 0.450 |

The closest measured PPVs are 0.43 (a self-reported prior bipolar diagnosis in an
outpatient clinic) and 0.22 to 0.24 (schizophrenia self-disclosure on Twitter). No study
measures subreddit PPV. The literature-leaning region L1 fails by a factor similar to
C1's region B (1.49 against 1.354). **Projected:** bipolar inadmissible at the C0
operating point, depression and eating disorder admissible, schizophrenia open.

What can still move it, for the supervisor meeting: a structured elicitation of label
PPVs (a direction change from D-028, open item 14); a descriptive audit of carer and
"am I bipolar" posts and of author cross-posting; the α and calibration share (the 0.43
clinic figure is admissible only near α = 0.20 with a 20% share); or reporting
per-condition admissibility as the C2 result under D-032.

D-031 interacts directly: not pruning keeps deterministic containment and α_V = 0, but
it also keeps the region wide, and width is exactly what the budget constrains. That
trade is no longer abstract and should be settled with the supervisor with these
numbers in hand. **D-047 adds one fact:** pruning cannot rescue bipolar, because the
budget is exhausted near 1% depression → bipolar and the simplex ceiling sits near 4.7%,
so D-031 can be decided on its other merits.

### Step 4 — Synthetic generator with a known `T`. ✅ DONE 2026-10-26

New `src/conformal/synthetic.py`. Draw a K = 4 problem at this project's real
prevalences (roughly 0.81 / 0.10 / 0.06 / 0.04), a controllable true `T` with
bipolar → depression as the dominant off-diagonal, and a scorer that emits softmax-like
probabilities with a tunable separability parameter. Corrupt labels through `T`.

**Built.** Clean prior `NOISY_PRIOR`; `make_transition(bipolar_to_depression,
background)`; scores are a softmax of `separability · onehot(Y)` plus standard normal
noise; labels corrupted through `T` independently of the scores, so Sesia's Assumption 1
holds by construction. Seeded; `python -m src.conformal.synthetic` prints a summary.
`tests/test_conformal_synthetic.py`, 11 tests.

**Gates.** With `T = I` the proxy labels equal the clean labels on 100,000 draws. The
recovered clean prior `ρ = (Tᵀ)⁻¹ρ̃` lands within four standard errors of the truth for
three matrices at n = 10⁶, and across 200 seeds its z-scores have mean ≈ 0 and SD ≈ 1.
Recorded as a numerical check on D-028 (amendment) bearing on open item 3, which stays
open. The arm-side half of the sanity gate needs steps 5 and 6.

**Sanity gate (corrected 2026-10-26, D-049).** The original text read "with `T = I`, all
three arms must coincide". That is false for the conservative Algorithm 2: with V = I
the noise terms vanish but δ keeps the constant c(n_k), so its threshold sits above
standard conformal's (bipolar at n = 481: rank 447 against 434). The gate is now two
tests at `T = I`: the optimistic variant (equations (18) and (25)) equals arm (a)
exactly, and the conservative variant differs from arm (a) by exactly the c(n_k) shift,
rank ⌈n_k(1 − α + c(n_k))⌉ against ⌈(n_k + 1)(1 − α)⌉. Either failing still means
nothing downstream can be trusted.

### Step 5 — Arm (a), noise-blind conformal. ✅ DONE 2026-10-26

Standard split conformal, **Mondrian / label-conditional** per D-029, at α = 0.1.
Measure per-class coverage and mean set size.

**Gate:** on synthetic data with `T = I`, per-class coverage hits nominal. This
validates the harness before noise enters.

Fix and record the prediction function here: Sesia's equation (3) is the plain
threshold on π̂, their §4 uses the generalised inverse quantile scores of Romano,
Sesia et al. (2020). Set sizes differ between them, so the choice must be recorded
before the arms are compared.

**Built.** Prediction function fixed by **D-048** (equation (3), score 1 − π̂,
deterministic, empty sets reported). `src/conformal/split_conformal.py` (equation (4),
Mondrian) and the harness `src/conformal/poc.py`, whose calibration sets take exactly
the C0 test split's counts (481 / 11,351 / 1,421 / 786), the operating point of
D-042's frontier. `python -m src.conformal.poc` reproduces the table below.

**Gate: passed.** T = I, α = 0.10, 100 repetitions, 20,000 test points each,
separability 2.0. Coverage against the clean label (mean ± SE over repetitions):

| condition | coverage | mean set size | empty-set rate |
|---|---|---|---|
| bipolar | 0.8994 ± 0.0017 | 1.241 | 0.0011 |
| depression | 0.9002 ± 0.0003 | 1.243 | 0.0010 |
| eating disorder | 0.9003 ± 0.0010 | 1.242 | 0.0009 |
| schizophrenia | 0.9000 ± 0.0015 | 1.242 | 0.0011 |

Theory puts each mean in [0.900, 0.900 + 1/(n_k + 1)]; all four sit within one SE of
it. The pinned test (`tests/test_conformal_split_conformal.py`) uses smaller counts and
200 repetitions, with a tolerance of three SEs either side of that interval.

### Step 6 — Arm (b), Algorithm 2 with a plug-in point estimate

Two sub-runs, both diagnostic:

1. **Oracle:** feed the true `T`. Coverage should be correct. Confirms the Algorithm 2
   implementation.
2. **Wrong point:** feed a deliberately wrong point estimate (HOC-like 4.6% when the
   truth is 15%). Coverage should break.

Sub-run 2 is the argument for a region, made on synthetic data where the truth is
known. It is worth having on its own.

**Configuration and expectation, fixed 2026-10-26 before any arm (b) number exists.**
Truth `T = make_transition(0.15, 0.0)`: bipolar → depression 15%, no other
off-diagonal (the generator defaults committed at step 4, `db1a3c4`). Oracle point: the
true `T`. Wrong point: `make_transition(0.046, 0.0)`. Both are converted to V with the
population noisy prior `Tᵀρ`, which stands in for the corpus-wide proxy frequencies a
practitioner would observe; the fixed-count calibration set's own frequencies are not
used. α = 0.10, calibration counts as in step 5, 100 repetitions, separability 2.0.
Each arm is reported as conservative (equation (22)) and optimistic (equation (25)),
per D-049, with arm (a) alongside.

*Expectation, from D-047's inflow argument.* With bipolar → depression as the only
off-diagonal, every post labelled bipolar is truly bipolar: the bipolar row of M is
(1, 0, 0, 0) under both the truth and the wrong point, and bipolar's calibration set is
uncontaminated. Bipolar coverage should therefore be nominal in every arm (above it for
the conservative variant), and the wrong point should **not** break it. The only
contaminated calibration set is depression's, about 0.7% truly bipolar posts, so any
arm differences should be small and confined there. If this holds, sub-run 2 does not
make the argument for a region in this configuration, which contradicts the roadmap's
"coverage should break". No configuration with inflow into a rare label is run here;
choosing one is for D-043 to pre-register. Proposition 3's condition is measured
without a prediction (D-049).

### Step 7 — PRE-REGISTER before arm (c) runs

Write **D-043** in DECISIONS.md fixing branches and thresholds *before the numbers
exist*. (D-043 is the number D-042 reserved for this pre-registration; D-044 and D-045
were taken by other work in the meantime.) The project's own scoreboard is the reason: two of four pre-registered
predictions did not hold, and in both cases the pre-registration is what stopped the
result being reinterpreted after the fact (D-034, D-041).

Note the change of role forced by Step 1. The closed-form analysis already predicts
degeneracy at region B, so D-043 is no longer predicting *whether* it degenerates. It
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
coverage, mean set size, and the full-set rate. Fire the D-043 branches.

### Step 9 — The sweep, now a check on Step 3

Sweep region width from 0 (a point estimate) upward and compare the measured
degeneracy point against the closed-form frontier from Step 3. Run D-032's
noise-heterogeneity sweep on the same harness.

**This is what makes every outcome a result.** "The coupling pays off below region
width w*, and this project's proxy noise sits above it" is a publishable finding even
if arm (c) loses, and Step 1 has already located w* to within a transcription.

### Step 10 — Write the result up as D-046

Branch fired, predictions held or missed, and the consequence for C2's scope. Update
the reversal scoreboard either way.

### What "POC passed" unlocks

Swap synthetic scores for the real held-out softmax already sitting in
`Models/predictions/test_predictions.csv`; swap the synthetic region for the elicited
one (open item 6, which is where the clinical citation work goes); then C3 adds DAIC
and the degradation curves.

---

## 6. Open items and caveats

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
- **Open item 6, elicitation sources, re-scoped by D-047.** The three directional
  claims (bipolar → depression dominates, schizophrenia's diagonal is lower,
  depression's is highest) turn out to constrain entries the budget barely sees; they
  remain useful for setting wide ranges there. The binding quantity is the worst-case
  PPV of each rare label, for which no subreddit-level study exists. Open item 13 lists
  the PDFs still to read; open item 14 is the direction decision.
- **4 of 6 POC conditions have data.** `anxiety` and `suicidality` are absent; the
  split code warns rather than fails. No code change needed to add them.
- **Class imbalance ~24:1** is why macro-F1 is the headline metric, and it is also why
  three of four conditions have no finite-sample guarantee under D-018.
- **Encoder choice (D-011) is now evidenced, with one caveat.** D-045 closed open
  item 12. The "small gap" classification had no threshold fixed before the run, and
  each encoder has one training seed; both are stated in D-045.
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

- **2026-10-26** — Step 5 done: arm (a) built under D-048; T = I gate passes (coverage
  nominal for all four conditions). D-049 recorded; step 4's gate text corrected.
- **2026-10-26** — Step 4 done: synthetic generator with known `T`; T = I gate and the
  ρ-recovery check pass (D-028 amendment; open item 3 stays open).
- **2026-10-08** — Step 3b scoped (D-047): binding quantity identified as rare-label
  PPV; `src/conformal/elicited_region.py` and `Docs/c2-elicitation-scoping.md` added;
  §0, §2, §3, §5 and §6 updated.
- **2026-10-01** — Housekeeping: Step 7 pre-registration renumbered D-042 → D-043 (the
  number D-042 reserved), Step 10 write-up → D-046; stale "commit D-041" open item
  removed.
- **2026-09-27** — MentalRoBERTa replication recorded (D-045); C0 row and §4 updated,
  open item 12 closed.
- **2026-08-14** — rewritten. The previous version was dated 2026-07-01 and stated
  "no MentalBERT fine-tune has run yet", five weeks and eight decision entries behind
  reality. C0 and C1 marked complete; §5 replaced with the C2 POC plan.
- **2026-07-24** — C1's framing reversed from estimator to diagnostic (D-030); the
  superseded C1 row is preserved in DECISIONS.md rather than here.
- **2026-07-01** — original, written when the data layer was done and the baseline was
  code-complete but untrained.
